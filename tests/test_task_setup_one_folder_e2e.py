"""The page is a function of the folder — checked by moving between two.

`web/task-setup.md` § 2.1: *"the page holds no state of its own… the folder
is the only link."*  A hand-written list of clears only ever covers what
someone remembered to add to it, so the rule needs a check that does not
depend on remembering: **open A, open B, and assert that nothing of A is on
the page.**

Measured on 2026-09-19, before the folder door: with the sidebar moved to a
brand-new SIESTA calculation, two cards still rendered the PySCF vibration
run that had been open before — "what this calculation writes" listed
`bridgespec_initial.xyz`, `bridgespec.spectra.json`, `bridgespec_01_freq.py`
under a heading naming the new one, and the per-stage card showed its `freq`
/ `01_freq`.  A reload cleared both, so it was the directory-change render
path rather than bad data.

This drives the DIRECTORY CHANGE, not a reload: the sidebar's rows clicked
down to the other folder (`support.sidebar.walk_to`), which the page hears
as the selection moving.  A reload would prove nothing — it is the one thing
that was already known to work.
"""
from __future__ import annotations

import pytest

from support.road import (described, ladder_edited, read_request, road_tree,
                          write_request)
from support.sidebar import WAIT_MS, committed, walk_to
from support.task_setup_page import (every_card_text, picked_by_the_page,
                                     task_setup_on, unfold, wait_for_file)


def _describe(root, name, label, stage):
    """A described PySCF calculation in the project ``root``: `molbuilder
    describe` (`support.road`), its one stage named ``stage`` through Task
    setup's Save.

    Two of these differ in every name a card can print: the label leads the
    filenames, and the stage leads the per-stage rows.
    """
    with pytest.MonkeyPatch.context() as local:
        dest = described(road_tree(root.parents[1], local, engine="pyscf"),
                         f"{root.name}/optimization/{name}",
                         name=label, engine="pyscf",
                         stage_strategy="")
    assert ladder_edited({"removed": ["coarse"], "added": [{"name": stage}]},
                         dest)
    return dest


def _hand_over(dest, label):
    """A hand-over as the Structure-optimization tab leaves one -- made by
    the hand-over route, its files written through the door the browser
    writes them by (`lib/task-handover.js`: `projects.writeFile`, `POST
    /api/files/write`, overwriting), the structure first.  The hand-over
    file's path."""
    from molbuilder.web.app import create_app
    from support.envelope import envelope
    dest.mkdir(parents=True)
    web = create_app(config={}).test_client()
    out = web.post(
        "/api/task-setup/handover", json=dict(
            {"structure": envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])},
            engine="pyscf", calculation="optimization", name=label,
            params={"job_name": label}, dest=str(dest))).get_json()
    assert out["ok"], out
    for name, text in [*((f["name"], f["text"])
                         for f in out["structure_files"]),
                       (out["template_name"], out["template_text"]),
                       (out["handover_name"], out["handover_text"])]:
        wrote = web.post("/api/files/write", json={
            "path": str(dest / name), "text": text, "overwrite": True})
        assert wrote.status_code == 200, wrote.get_json()
    return dest / out["handover_name"]


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.fixture
def two_calcs(isolated_projects_root):
    root = isolated_projects_root / "umbrella"
    a = _describe(root, "alpha-run", "alphalabel", "alphastage")
    b = _describe(root, "beta-run", "betalabel", "betastage")
    return a, b


def test_opening_another_folder_leaves_nothing_of_the_first(
        page, flask_server, two_calcs):
    """The rule, executed: after moving A -> B, no trace of A is on screen.

    Asserted on the WHOLE page text rather than on named cards, deliberately.
    A per-card assertion is the same hand-kept list that failed — it can only
    catch the cards someone thought of.  This catches any of them, including
    ones added later.
    """
    a, b = two_calcs
    task_setup_on(page, flask_server, a)
    page.wait_for_function(
        "() => document.body.innerText.includes('alphalabel')", timeout=WAIT_MS)

    # The sidebar moves -- its rows clicked, not a reload.
    walk_to(page, b)
    page.wait_for_function(
        "() => document.body.innerText.includes('betalabel')", timeout=WAIT_MS)
    # ...AND THE READ ENDED, not only its first card painted: the column
    # says so (`aria-busy`, § 9a).
    page.wait_for_selector("#ts-main:not([aria-busy])", state="attached",
                           timeout=WAIT_MS)

    # THE TASK SETUP SURFACE, not the whole document.  The projects sidebar
    # is a different component with its own contract (`web/projects.md`) and
    # its own idea of when to re-list; section 2.1 is about THIS page's
    # cards.  Still EVERY card and not named ones -- a per-card list is the
    # same hand-kept thing that failed -- each opened, since a folded card
    # still holds what it would show (`task-setup.md` § 9a.1).
    text = every_card_text(page)
    assert "betalabel" in text, "the second folder must actually be shown"
    assert "alphalabel" not in text, (
        "the first folder's label is still on the page after moving to the "
        "second — a card is rendering state the directory change did not "
        "replace")
    assert "alphastage" not in text, (
        "the first folder's stage is still on the page after moving to the "
        "second")


def test_moving_to_a_folder_with_LESS_to_show_clears_what_the_last_one_had(
        page, flask_server, two_calcs, isolated_projects_root):
    """The discriminating case, and the one actually measured.

    Two DESCRIBED folders both have stages and filenames, so every card has
    something new to paint and the old content is overwritten whether or not
    anything was cleared.  The 2026-09-19 observation was described -> HAND
    OVER: a brand-new SIESTA calculation with no `task.json`, no stages and
    no `<label>_*` files yet.  A card with nothing to render is the one that
    keeps what it had, which is why *that* is where `bridgespec_initial.xyz`
    and a `freq` / `01_freq` row were still on screen under a heading naming
    the new calculation.

    So: open the described one, then move to a folder that can say less.
    The described one names a queue, which its queue card says; the
    hand-over's proposal names none (the 2.27.0 review: the last folder's
    queue stayed on the card, and a click on it wrote into no description).

    THE MACHINE IS THE PAGE'S PICK: this machine is the one listed, so no
    choice is required and the page picks it (`task-setup.md` § 6.4) --
    which the queue card needs before it says the queue.

    MUTATION THIS MUST FAIL AGAINST: the read's reset not hiding *What
    this calculation writes* (`hideWritten` in `_resetPerFolderState`):
    nothing a hand-over draws repaints that card, so the described folder's
    file names -- `alphalabel_…` -- stay on the page.
    """
    a, _b = two_calcs
    req = read_request(a)
    req["allocation"] = {"domain": "alphaqueue"}
    write_request(a, req)
    bare = isolated_projects_root / "umbrella" / "optimization" / "bare-run"
    _hand_over(bare, "barelabel")

    task_setup_on(page, flask_server, a)
    page.wait_for_function(
        "() => document.body.innerText.includes('alphalabel')", timeout=WAIT_MS)
    picked_by_the_page(page)
    page.wait_for_function(
        "() => document.body.innerText.includes('alphaqueue')", timeout=WAIT_MS)
    # ...AND ITS FILES LISTED, so a card the move must clear has something
    # in it (`task-setup.md` § 7.2).
    page.wait_for_selector("#ts-written-card:not([hidden])",
                           state="attached", timeout=WAIT_MS)

    walk_to(page, bare)
    # WAIT FOR THE LOAD TO FINISH, not for the first card to paint: the
    # folder's read ends with the hand-over's proposal landed, and the
    # column says so (`aria-busy`, § 9a, which the editor's tab waits on)
    # -- the proposal's description names the new run.
    wait_for_file(page, "task", "barelabel")

    text = every_card_text(page)
    assert "alphalabel" not in text, (
        "the described folder's label survived a move to one that has no "
        "description -- a card with nothing new to paint kept what it had")
    assert "alphastage" not in text, (
        "the described folder's stage survived a move to one that has none")
    assert "alphaqueue" not in text, (
        "the described folder's queue card survived a move to a hand-over")



def test_a_hand_over_is_finished_on_the_page_and_saved_as_both_records(
        page, flask_server, isolated_projects_root):
    """`task-setup.md` § 4, § 7a, § 8, § 9a: a hand-over becomes a
    calculation on this page -- the proposal drawn into both records as it
    opens, a run card filled, Save -- and the folder then holds the
    description, stating the one layout (`engines/stages.md` § 6.7), and its
    job request, the run card's value in the job request, the save said
    beside the button after the folder is read again.  The page arrives from
    another folder, as a hand-over sent while Task setup is open does.

    THE SILENT FAILURES: a hand-over that proposes nothing -- the page
    editing the hand-over's own text as a description; the save's
    confirmation wiped by the folder's re-read -- a save that looks like
    nothing happened; the run card's value written into no record.

    MUTATION THIS MUST FAIL AGAINST: the hand-over's text taken for a
    description (no stages proposed): no run card is drawn, and the
    description written states no `shape`."""
    d = (isolated_projects_root / "handover" / "optimization"
         / "finished_here")
    _hand_over(d, "handed")
    came_from = _describe(isolated_projects_root / "handover", "came-from",
                          "camefromlabel", "camefromstage")

    task_setup_on(page, flask_server, came_from)
    page.wait_for_function(
        "() => document.body.innerText.includes('camefromlabel')",
        timeout=WAIT_MS)
    walk_to(page, d)
    # THE NEW FOLDER READ -- the read ended with the proposal landed
    # (`aria-busy`, § 9a, which the editor's tab waits on) -- its run named
    # in the editor.
    wait_for_file(page, "task", "handed")
    # THE PROPOSAL, DRAWN AS THE HAND-OVER OPENED (§ 9a): the description
    # it becomes, stating the one layout.
    wait_for_file(page, "task", '"shape": "hierarchical"')
    # THE GROUPS CARD APPEARS WITH THE PROPOSAL, folded as every card is:
    # opened as a person opens it.
    unfold(page, "ts-groups-card")
    field = (".ts-next-step:not([hidden]) .ts-runcard "
             "[aria-label='value for CPU threads']")
    page.wait_for_selector(field, timeout=WAIT_MS)
    committed(page, field, "2")
    wait_for_file(page, "request", '"threads": 2')

    page.click("#ts-save")
    page.wait_for_selector("#ts-main:not([aria-busy])", state="attached",
                           timeout=WAIT_MS)
    page.wait_for_function(
        "() => { const n = document.getElementById('ts-save-said');"
        " return n && !n.hidden && /Saved/.test(n.textContent); }",
        timeout=WAIT_MS)
    said = page.evaluate(
        "() => document.getElementById('ts-save-said').textContent")
    assert "task.json" in said and "job-request.json" in said, said
    # BOTH RECORDS THROUGH THEIR READERS -- the description's refuses any
    # layout but the one (`task.LAYOUT`), by name.
    from molbuilder import job_request
    from molbuilder.task import of as task_of
    described = task_of(d)
    assert [st.name for st in described.stages] == ["coarse"], described
    request = job_request.of(d, described).to_dict()
    assert request["stages"] == {
        "coarse": {"group": 1, "execution": {"threads": 2}}}, request

