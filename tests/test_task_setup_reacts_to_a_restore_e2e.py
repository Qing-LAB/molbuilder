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
def calc_dir(isolated_projects_root):
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

    root = isolated_projects_root / "restore_e2e"
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
        pass    # tmp_path removes the tree


@pytest.fixture
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _editor_text(page):
    return page.evaluate(
        "() => { const cm = document.querySelector('.CodeMirror');"
        "        return cm ? cm.CodeMirror.getValue() : ''; }")


def _dialogs(page, answers):
    """Answer the panel's own prompts, the way a person at the keyboard does.

    Every step of this walk is a real control and two of them ask a
    question: Commit prompts for the note (`_onCommitClick`), Restore
    confirms before rewriting the folder (`_onRestoreClick`).  Playwright
    DISMISSES dialogs unless told otherwise, which silently cancels the
    action and leaves the assertion failing for a reason that has nothing
    to do with the contract.

    ``page.on`` rather than ``page.once``: `_restore` asks a SECOND time
    when the server refuses because of unsaved work, and a handler that has
    already fired would let that one be dismissed -- turning a restore that
    the code correctly guarded into a silent no-op the test blames on the
    announcement.
    """
    queue = list(answers)

    def _answer(d):
        if d.type == "prompt":
            d.accept(queue.pop(0) if queue else "")
        else:
            d.accept()

    page.on("dialog", _answer)


def _show_panel(page):
    """Bring the checkpoint panel forward, and be idempotent about it.

    The panel shares its slot with the file tree and the toggle REMEMBERS
    which one is showing, so a second unconditional click after a reload
    puts the tree back -- which is how this walk spent twenty seconds
    waiting for a row that was on screen and then wasn't.
    """
    page.wait_for_selector("#ps-checkpoint-toggle", timeout=20000)
    showing = page.evaluate(
        "() => { const p = document.getElementById('ps-checkpoint');"
        "        return !!(p && !p.hidden && p.offsetParent !== null); }")
    if not showing:
        page.click("#ps-checkpoint-toggle")
    page.wait_for_selector("#ps-checkpoint", state="visible", timeout=20000)


def _rows(page):
    """Every checkpoint state on the panel, as {note-ish text: handle}."""
    return page.evaluate(
        "() => [...document.querySelectorAll('.ps-checkpoint-list-item')]"
        "        .map(li => li.textContent)")


def test_a_restore_updates_the_tab_that_is_showing_the_folder(
        page, flask_server, calc_dir):
    """Restore from the sidebar; the open Task-setup tab shows the restored
    description, not the one that was on screen when the restore ran.

    The selection never changes and nothing is committed, so the only thing
    that can carry the news is `publishFolderChanged` -> `onFolderChanged`.

    **Every step is a control on the page** -- Set up, Save a state, expand
    a row, Restore -- because the defect was a gap BETWEEN two surfaces and
    only a walk that uses both can stand in it.  The one thing done behind
    the page's back is rewriting `task.json` on disk, and that is the
    premise rather than a shortcut: `projects.md` § 4.1 is about a folder
    "rearranged behind an open tab", which is by definition something the
    tab did not do.
    """
    _dialogs(page, ["after the edit"])

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

    # ── put the folder under checkpoint control, and save state A ───────
    # The panel shares its slot with the file tree; the filter bar's toggle
    # is how a person brings it forward.
    _show_panel(page)
    page.wait_for_selector("#ps-checkpoint-init", state="visible",
                           timeout=20000)
    page.click("#ps-checkpoint-init")
    # Set-up SAVES the first state itself, noted "set up" -- so this is
    # state A and there is nothing to commit yet.  Pressing Save-a-state
    # here would answer "Nothing changed since the state this folder stands
    # at" and record no row, which is how the first version of this walk
    # waited twenty seconds for a row that was never coming.
    page.wait_for_function(
        "() => [...document.querySelectorAll('.ps-checkpoint-list-item')]"
        "        .some(li => li.textContent.includes('set up'))",
        timeout=20000)
    assert "coarse" in _editor_text(page)

    # ── rewrite the folder behind the tab, then record that as state B ──
    # THROUGH THE PROJECT'S OWN DOOR.  `molbuilder.task` owns this format
    # and exposes `read_task` / `write_task`; hand-editing the JSON is a
    # second writer, and a second writer can produce a `task.json` the app
    # would refuse or read differently -- which would make this test's
    # premise ("the folder changed underneath the tab") a lie about a file
    # the tab could not have shown in the first place.
    import dataclasses

    from molbuilder.task import Stage, read_task, write_task

    before = read_task(calc_dir / "task.json")
    renamed = dataclasses.replace(before.stages[0], name="afterwards")
    write_task(calc_dir / "task.json",
               dataclasses.replace(before, stages=(renamed,)))
    page.click("#ps-checkpoint-commit-btn")
    page.wait_for_function(
        "() => [...document.querySelectorAll('.ps-checkpoint-list-item')]"
        "        .some(li => li.textContent.includes('after the edit'))",
        timeout=20000)

    # The tab is still showing the OLD text -- nothing announced the write,
    # and that is correct.  Reload so the tab is honestly showing B before
    # the restore, or "it changed" could just be the tab catching up.
    page.reload()
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        " return !!(n && n.CodeMirror)"
        "   && n.CodeMirror.getValue().includes('afterwards'); }",
        timeout=20000)
    _show_panel(page)
    page.wait_for_selector(".ps-checkpoint-list-item", state="visible",
                           timeout=20000)

    notes = _rows(page)
    assert sum("set up" in t for t in notes) == 1, (
        f"expected exactly one state named 'set up' to restore to; "
        f"the panel shows {notes}")
    assert any("after the edit" in t for t in notes), (
        f"the second state was not recorded, so a 'restore' would land on "
        f"the state the folder is already at and prove nothing: {notes}")

    # ── RESTORE to state A, picked BY NAME ──────────────────────────────
    # Not by position: the panel's order is its own decision, and a test
    # that says "the last row" silently changes meaning the day that flips.
    page.evaluate("""() => {
        const row = [...document.querySelectorAll(".ps-checkpoint-list-item")]
            .find(li => li.textContent.includes("set up"));
        row.click();                          // expand to reveal the actions
        row.querySelector('[data-action="restore"]').click();
    }""")

    # ── the tab must catch up on its own ────────────────────────────────
    # The wait is bounded and its failure is SWALLOWED, so the assertion
    # below is what a person reads.  Letting `wait_for_function` raise
    # reports "Timeout 20000ms exceeded" and nothing else -- which is the
    # correct verdict delivered in a form that teaches nobody what broke.
    try:
        page.wait_for_function(
            "() => { const n = document.querySelector('.CodeMirror');"
            " return n && n.CodeMirror.getValue().includes('coarse'); }",
            timeout=20000)
    except Exception:                      # noqa: BLE001 - see above
        pass
    after = _editor_text(page)
    assert "coarse" in after and "afterwards" not in after, (
        "the Task-setup tab is still showing the description the folder had "
        "BEFORE the restore.  The selection never moved and nothing was "
        "committed, so the restore must announce with "
        "`projects.publishFolderChanged(dir)` and the tab must re-read on "
        "`onFolderChanged` (`web/projects.md` § 4.1).  Right now a person "
        "is looking at stages the folder no longer has.")
