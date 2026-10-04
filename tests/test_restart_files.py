"""The restart files in effect (`docs/execution/job-contracts.md` § 4.2a,
`architecture.md` § 3.2): a calculation's own `warm-files.toml` beside its
`task.json` is followed by every reader, molbuilder's list for its engine
otherwise.

THE CASES ARE DATA -- ``tests/data/restart_files.toml`` -- and each runs down
the road a person runs, through the one runner every contract table shares
(`support.road.run_road_case`).  The Task setup page's statement of which list
is in effect is asked of its route below: an answer the page lays out, which
no `jobset` verb prints.

PREVENTS (plan W36 ⑧): a calculation's own list followed by prep's carry and
ignored by the run script's restart check and `jobset status`, which read the
engine's file -- two doors onto one list.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import describe_h2, run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "restart_files.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_restart_files_in_effect(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)


@pytest.mark.parametrize("own", [False, True],
                         ids=["molbuilder's list", "its own list"])
def test_the_page_says_which_list_the_calculation_follows(own, tmp_path,
                                                          monkeypatch):
    """The Task setup files card's last group (`web/task-setup.md` § 7.2):
    which list is in effect -- the one door's answer, sent with the plan --
    and, while it is molbuilder's, where a custom copy goes.

    MUTATION THIS MUST FAIL AGAINST: the route asking the door without the
    calculation's folder (it would answer molbuilder's list for both)."""
    import json
    from molbuilder import diagnostics
    from molbuilder.warmfiles import FILENAME, warm_list
    from molbuilder.web.app import create_app
    bundle = describe_h2(tmp_path, monkeypatch)
    if own:
        (bundle / FILENAME).write_text(
            Path(warm_list("siesta").path).read_text())
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)
    task = json.loads((bundle / "task.json").read_text())
    got = create_app(config={}).test_client().post(
        "/api/task-setup/prep-plan",
        json={"task": task, "dest": str(bundle)}).get_json()
    assert got["ok"] is True, got
    assert got["warm"]["own"] is own, got["warm"]
    assert got["warm"]["path"] == (str(bundle / FILENAME) if own
                                   else warm_list("siesta").path)
    assert got["warm"]["copy_to"] == str(bundle / FILENAME)
