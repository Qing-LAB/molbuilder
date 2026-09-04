"""**A restore rewrites the folder under an open tab, and the tab re-reads.**

`web/projects.md` § 4.1 states the contract and names this exact defect:

> a checkpoint **restore** rewrites a folder while the selection sits still,
> and can swap `task.json` for `task.1st.json` under an open tab. Neither
> existing channel fires: the selection never moved, and nothing was
> committed. Task setup went on showing stages and a bench the folder no
> longer had.

The rule is `projects.publishFolderChanged(dir)` from whatever rearranged the
folder, and `onFolderChanged` on whatever is displaying it — announce, don't
reach in, "because the writer does not know which tabs are open or what they
cache".

**Why a browser.** Every part is separately fine: the route restores, the
panel repaints itself, the tab can re-read. The defect lives in the gap — the
panel's own refresh repainting only itself — which is visible only with both
surfaces on one page, and that is what a restore-through-the-sidebar walk
gives.

*Replaces `tests/test_task_setup_reacts_to_the_folder.py`, retired 2026-09-03
(`process/testing.md` § 3a.1).* All six of its tests grepped `viewer.js`,
`state.js` and `checkpoint.js` for the spelling of a function name — one
asserted `viewer.count("_fillMeta(") >= 3` — and its docstring justified that
with *"viewer.js is an ES module and the repo's node harness requires
CommonJS"*, which was untrue when written. What it pinned was that the words
were present; what broke was that the announcement never reached the tab.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def calc_dir():
    """A described calculation under checkpoint control.

    Built through the production doors — `describe.build_description` and
    `checkpoint`'s own repo — rather than by hand: a description assembled
    directly carries a default witness that prep and restore both refuse,
    and a checkpoint state written by hand is not one the route will accept.
    """
    import numpy as np

    from conftest import write_pseudos
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.structure import Structure
    from molbuilder.task import Stage

    root = ROOT / "projects/_t_restore_e2e"
    if root.exists():
        shutil.rmtree(root)
    src_dir = root / "structure"
    src_dir.mkdir(parents=True)
    d = root / "optimization" / "probe"
    try:
        struct = Structure(elements=["H", "H"],
                           positions=np.array([[0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.74]]),
                           vacuum=(10.0, 10.0, 10.0))
        src = src_dir / "probe.xyz"
        src.write_text(struct.to_xyz(), encoding="utf-8")
        D.write_description(
            D.build_description(struct,
                                SiestaConfig(system_label="probe"),
                                [Stage(name="coarse",
                                       overrides={"mesh_cutoff": 200})],
                                engine="siesta", shape="hierarchical",
                                name="probe", source=str(src)),
            d, struct=struct)
        write_pseudos(d, ["H"])
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _editor_text(page):
    return page.evaluate(
        "() => { const cm = document.querySelector('.CodeMirror');"
        "        return cm ? cm.CodeMirror.getValue() : ''; }")


def test_a_restore_updates_the_tab_that_is_showing_the_folder(
        page, flask_server, calc_dir):
    """Restore from the sidebar; the open Task-setup tab shows the restored
    description, not the one that was on screen when the restore ran.

    The selection never changes and nothing is committed, so the only thing
    that can carry the news is `publishFolderChanged` -> `onFolderChanged`.
    """
    slot = json.dumps(str(calc_dir))
    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir.task-setup', {slot});"
        f" sessionStorage.setItem('molbuilder.current_dir', {slot});"
        "} catch (_) {}")
    page.goto(f"{flask_server}/task-setup")
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        " return !!(n && n.CodeMirror); }", timeout=20000)
    page.wait_for_function(
        "() => document.querySelector('.CodeMirror')"
        ".CodeMirror.getValue().includes('coarse')", timeout=20000)

    # ── put the folder under checkpoint control and save state A ────────
    started = page.evaluate("""async (dir) => {
        const j = async (m, u, b) => (await fetch(u, {
            method: m, headers: {"Content-Type": "application/json"},
            body: JSON.stringify(b)})).json();
        await j("POST", "/api/checkpoint/init", {path: dir});
        return j("POST", "/api/checkpoint/save",
                 {path: dir, note: "before the edit"});
    }""", str(calc_dir))
    assert started.get("ok") is not False, f"could not checkpoint: {started}"
    before = _editor_text(page)
    assert "coarse" in before, before[:200]

    # ── rewrite the folder on disk, the way a second stage would ────────
    task = json.loads((calc_dir / "task.json").read_text())
    task["stages"] = [{**task["stages"][0], "name": "afterwards"}]
    (calc_dir / "task.json").write_text(json.dumps(task, indent=2),
                                        encoding="utf-8")
    page.evaluate("""async (dir) => {
        await fetch("/api/checkpoint/save", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({path: dir, note: "after the edit"})});
    }""", str(calc_dir))

    # The tab is still showing the OLD text at this point -- nothing has
    # told it otherwise, and that is correct: no surface announced a change.
    page.reload()
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        " return !!(n && n.CodeMirror)"
        "   && n.CodeMirror.getValue().includes('afterwards'); }",
        timeout=20000)

    # ── now RESTORE to state A, through the sidebar's own button ────────
    page.wait_for_selector("#ps-checkpoint", state="attached", timeout=20000)
    # The panel shares its slot with the file tree; the filter bar's toggle
    # is how a person brings it forward.
    page.click("#ps-checkpoint-toggle")
    page.wait_for_selector(".ps-checkpoint-list-item", state="attached",
                           timeout=20000)
    rows = page.locator(".ps-checkpoint-list-item").count()
    assert rows >= 2, (
        f"expected two checkpoint states to restore between, saw {rows} -- "
        f"the walk below cannot distinguish a restore from a no-op")
    # "Put this folder back to X?" -- a real confirm, and a person says yes.
    # Playwright DISMISSES dialogs unless told otherwise, which silently
    # cancels the restore and makes the assertion below fail for the wrong
    # reason.
    page.once("dialog", lambda d: d.accept())
    page.evaluate("""() => {
        const rows = [...document.querySelectorAll(".ps-checkpoint-list-item")];
        const oldest = rows[rows.length - 1];
        oldest.click();                       // expand to reveal the actions
        const btn = oldest.querySelector('[data-action="restore"]');
        btn.click();
    }""")

    # ── the tab must catch up on its own ────────────────────────────────
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        " return n && n.CodeMirror.getValue().includes('coarse'); }",
        timeout=20000)
    after = _editor_text(page)
    assert "afterwards" not in after, (
        "the Task-setup tab is still showing the description the folder had "
        "BEFORE the restore.  The selection never moved and nothing was "
        "committed, so the restore must announce with "
        "`projects.publishFolderChanged(dir)` and the tab must re-read on "
        "`onFolderChanged` (`web/projects.md` § 4.1).  Right now a person "
        "is looking at stages the folder no longer has.")
