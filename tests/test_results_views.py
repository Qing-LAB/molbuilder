"""The views a folder is read with, served beside its own answer
(`execution/architecture.md` § 3.2, *the views a folder is read with*;
`web/results.md` § 0.2, § 2.4, § 3a; `web-api.md` `/api/results/dir`).

**Why this is needed.**  `/api/results/dir` serves the run door's answer for
the one folder and, beside it, floor 6's views: a calculation root's ladder
with each stage's every run, a run of points' view at its root and at each
of its points' folders, and the point a folder is.  Every path the page
opens is whole, so the page joins none.  Without these, the page composed
paths itself, a run of points' root opened nothing and hid its Run panel,
and the run door walked a run's points to answer one folder (plan F1).

**Category: API** -- through the road (`molbuilder describe`, `jobset
prep`), no engine run: the folders and records are prep's; each case is a
row of the table below, its expectations from the contract.  The page reads
each answer through its one reading (`file-picker.js` `readAnswer`, run
under node): the picker's scan and a run picked off the ladder alike.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static/lib"


#: ``(calculation, folder below the bundle, what the answer states)`` --
#: ``status``: the run door's state for the folder (``None`` for a
#: container); ``points``: the run of points' view's state (``None``: no
#: view); ``point``: the phase of the point the folder is (``None``: none);
#: ``ladder``: each stage's runs as ``(folder below the bundle, phase)``;
#: ``page_points``: whether the page's reading carries the run of points'
#: view -- at a run of points' root alone, never at a point's folder, which
#: shows its own record (§ 3a).
CASES = [
    ("vibration", "", {
        "status": None, "points": None, "point": None,
        "ladder": {"freq": [("01_freq/run-0", None),
                            ("01_freq/run-0/relax", "relax"),
                            ("01_freq/run-0/fc", "fc")]},
        "page_points": False}),
    ("vibration", "01_freq/run-0", {
        "status": None, "points": "pending", "point": None,
        "ladder": None, "page_points": True}),
    ("vibration", "01_freq/run-0/relax", {
        "status": "pending", "points": "pending", "point": "relax",
        "ladder": None, "page_points": False}),
    ("vibration", "01_freq/run-0/fc", {
        "status": "pending", "points": "pending", "point": "fc",
        "ladder": None, "page_points": False}),
    ("optimization", "01_coarse/run-0", {
        "status": "pending", "points": None, "point": None,
        "ladder": None, "page_points": False}),
]


def _page_reading(body: dict) -> dict:
    """``body`` read as the page reads a folder's answer -- the picker's
    one reading (`resultsFilePicker.readAnswer`), under node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = "\n".join([
        "global.window = global;",
        (STATIC / "time-ago.js").read_text(),
        (STATIC / "results/file-picker.js").read_text(),
        "const a = window.molbuilder.resultsFilePicker.readAnswer("
        + json.dumps(body) + ", null);",
        "console.log(JSON.stringify({opens: a.opens, points: a.points}));"])
    proc = subprocess.run([node, "--input-type=commonjs", "-e", script],
                          capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("calculation,where,want", CASES,
                         ids=[f"{c}:{w or 'root'}" for c, w, _ in CASES])
def test_a_folder_is_served_with_the_views_it_is_read_with(
        tmp_path, monkeypatch, calculation, where, want):
    """Each folder's answer, through the route: its own state (the run
    door's), the views it is read with -- by what it is -- and every path
    whole: each file's, each run's folder, what a ladder row opens."""
    pytest.importorskip("flask")
    from conftest import write_machine_record
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.web.app import create_app
    from support.road import describe_calculation, jobset
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "projects"))
    client = create_app(config={}).test_client()
    write_machine_record()
    bundle = describe_calculation(tmp_path, monkeypatch,
                                  calculation=calculation)
    got = jobset("prep", "task", "--group", "1", "--bundle", bundle,
                 "--target", "this")
    assert got.exit_code == 0, got.output
    whole = bundle.resolve()
    folder = whole / where if where else whole

    body = client.get("/api/results/dir?path=" + str(folder)).get_json()
    assert body["ok"] is True, body
    assert ((body["status"] or {}).get("state")) == want["status"], (
        body["status"])
    assert ((body["points"] or {}).get("state")) == want["points"], (
        body["points"])
    if body["points"] is not None:
        # NOTHING RAN: no point at work, nothing the run opens yet (§ 3a).
        assert (body["points"]["at_work"], body["points"]["opens"]) == (
            None, None), body["points"]
    assert ((body["point"] or {}).get("phase")) == want["point"], (
        body["point"])
    # THE PAGE'S ONE READING: the run of points' view at its root alone,
    # and what it opens -- the view's there, the run door's elsewhere.
    page = _page_reading(body)
    assert (page["points"] is not None) == want["page_points"], page
    assert page["opens"] == (body["points"]["opens"] if want["page_points"]
                             else body["openable"]), page
    # EVERY FILE'S PATH, WHOLE -- the page joins none (§ 0.3).
    assert body["files"] and all(f["path"] == str(folder / f["name"])
                                 for f in body["files"]), body["files"]
    if want["ladder"] is None:
        assert body["ladder"] is None, body["ladder"]
        return
    runs = {s["name"]: [(str(Path(r["dir"]).relative_to(whole)),
                         r.get("phase")) for r in s["runs"]]
            for s in body["ladder"]["stages"]}
    assert runs == want["ladder"], runs
    assert not any(r["opens_report"] for s in body["ladder"]["stages"]
                   for r in s["runs"]), body["ladder"]
