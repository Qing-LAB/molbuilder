"""**A restore rewrites the folder under an open tab, and the tab re-reads.**

`web/projects.md` § 2 states the contract and names this exact defect:

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
"""
from __future__ import annotations

import pytest
from support.road import ladder_edited
from support.sidebar import WAIT_MS, history_shown
from support.task_setup_page import file_text, task_setup_on, wait_for_file

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture
def calc_dir(tmp_path, monkeypatch):
    """A described calculation, as `molbuilder describe` leaves one
    (`support.road`), its one stage `coarse`; the test puts it under checkpoint
    control through the page."""
    from support.road import described, road_tree
    return described(road_tree(tmp_path, monkeypatch), name="probe",
                     stage_strategy="")


@pytest.fixture
def flask_server(isolated_projects_root):
    """A server for this test, on the test's own projects tree -- the one
    its calculation is described in (`support.live_server.serve`)."""
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _note(page, note):
    """Answer Save-a-state's question, the way a person does: the page's own
    dialog (`ui-contract.md` § 11), its note field filled and its answer
    clicked."""
    asked = page.locator("dialog.mb-dialog[open]")
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("input[data-role='note']").fill(note)
    asked.locator("[data-action='confirm']").click()


def _yes(page):
    """Answer the one confirm (`warningModal.confirm`) with its costly
    choice -- Restore asks before rewriting the folder."""
    asked = page.locator("dialog.molbuilder-warning-modal[open]")
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("[data-action='discard']").click()


def _rows(page):
    """Every checkpoint state on the panel, the text of its row."""
    return page.evaluate(
        "() => [...document.querySelectorAll('.ps-checkpoint-list-item')]"
        "        .map(li => li.textContent)")


def test_a_restore_updates_the_tab_that_is_showing_the_folder(
        page, flask_server, calc_dir):
    """Restore from the sidebar; the open Task-setup tab shows the restored
    description, not the one that was on screen when the restore ran.

    The selection never changes and nothing is committed, so the only thing
    that can carry the news is `publishFolderChanged` -> `onFolderChanged`.

    **Every step is a control on the page** -- the folder's rows, Set up,
    Save a state, expand a row, Restore -- because the defect was a gap
    BETWEEN two surfaces and only a walk that uses both can stand in it.
    The one thing done behind the page's back is another client's Save
    renaming the stage, and that is the premise rather than a shortcut:
    `projects.md` § 2 is about *"anything that rearranges a folder behind
    an open tab"*, which is by definition something the tab did not do.
    """
    native = []
    page.on("dialog", lambda d: (native.append(d.message), d.dismiss()))

    # The folder opened as a person opens it: its rows in the sidebar.
    task_setup_on(page, flask_server, calc_dir)
    wait_for_file(page, "task", "coarse")

    # ── put the folder under checkpoint control, and save state A ───────
    # The panel shares its slot with the file tree; the filter bar's toggle
    # is how a person brings it forward.
    history_shown(page)
    page.wait_for_selector("#ps-checkpoint-init", state="visible",
                           timeout=WAIT_MS)
    page.click("#ps-checkpoint-init")
    # Set-up SAVES the first state itself, noted "set up" -- so this is
    # state A and there is nothing to commit yet.  Pressing Save-a-state
    # here would answer "Nothing changed since the state this folder stands
    # at" and record no row.
    page.wait_for_function(
        "() => [...document.querySelectorAll('.ps-checkpoint-list-item')]"
        "        .some(li => li.textContent.includes('set up'))",
        timeout=WAIT_MS)
    assert "coarse" in file_text(page, "task")

    # ── rewrite the folder behind the tab, then record that as state B ──
    # BY ANOTHER WRITER THROUGH THE PROJECT'S OWN DOOR: Task setup's Save,
    # from another client, renames the one stage -- the description and the
    # job request beside it, which names the stages too
    # (`execution/job-request.md` § 3), as one ladder.  Nothing this tab
    # did, which is the premise: `projects.md` § 2 is about *"anything that
    # rearranges a folder behind an open tab"*.
    assert ladder_edited({"removed": ["coarse"],
                          "added": [{"name": "afterwards"}]}, calc_dir)
    page.click("#ps-checkpoint-commit-btn")
    _note(page, "after the edit")
    page.wait_for_function(
        "() => [...document.querySelectorAll('.ps-checkpoint-list-item')]"
        "        .some(li => li.textContent.includes('after the edit'))",
        timeout=WAIT_MS)

    # The tab is still showing the OLD text -- nothing announced the write,
    # and that is correct.  Reload so the tab is honestly showing B before
    # the restore, or "it changed" could just be the tab catching up.
    page.reload()
    wait_for_file(page, "task", "afterwards")
    history_shown(page)
    page.wait_for_selector(".ps-checkpoint-list-item", state="visible",
                           timeout=WAIT_MS)

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
    row = page.locator(".ps-checkpoint-list-item", has_text="set up")
    row.click()                               # expand to reveal the actions
    row.locator('[data-action="restore"]').click()
    _yes(page)

    # ── the tab must catch up on its own ────────────────────────────────
    # The wait is bounded and its failure is SWALLOWED, so the assertion
    # below is what a person reads.  Letting `wait_for_function` raise
    # reports only that the wait timed out -- which is the
    # correct verdict delivered in a form that teaches nobody what broke.
    try:
        wait_for_file(page, "task", "coarse")
    except Exception:                      # noqa: BLE001 - see above
        pass
    after = file_text(page, "task")
    assert "coarse" in after and "afterwards" not in after, (
        "the Task-setup tab is still showing the description the folder had "
        "BEFORE the restore.  The selection never moved and nothing was "
        "committed, so the restore must announce with "
        "`projects.publishFolderChanged(dir)` and the tab must re-read on "
        "`onFolderChanged` (`web/projects.md` § 2).  Right now a person "
        "is looking at stages the folder no longer has.")
    assert native == [], f"a native dialog was opened: {native}"
