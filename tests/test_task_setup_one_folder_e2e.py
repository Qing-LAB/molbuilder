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

This drives the DIRECTORY CHANGE, not a reload: `projects.publishCommit` is
the app's own "the sidebar selected this", which is how `test_build_e2e`
drives it too.  A reload would prove nothing — it is the one thing that was
already known to work.
"""
from __future__ import annotations

import json

import pytest

from support.road import read_request, write_request
from support.task_setup_page import (every_card_text, file_text, unfold,
                                     wait_for_file)

pytestmark = pytest.mark.e2e


def _describe(root, name, label, stage):
    """A described PySCF calculation, through the production door.

    Two of these differ in every name a card can print: the label leads the
    filenames, and the stage leads the per-stage rows.
    """
    import numpy as np

    from molbuilder import describe as D
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.structure import Structure
    from molbuilder.task import Stage

    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    src_dir = root / "structure"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / f"{name}.xyz"
    src.write_text(struct.to_xyz(), encoding="utf-8")

    dest = root / "optimization" / name
    D.write_description(D.build_description(
        struct, PySCFConfig(job_name=label),
        [Stage(name=stage, overrides={})],
        engine="pyscf",
        calculation="optimization", shape="hierarchical", name=label,
        source=str(src)), dest)
    return dest


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.fixture
def two_calcs(isolated_projects_root):
    root = isolated_projects_root / "umbrella"
    a = _describe(root, "alpha-run", "alphalabel", "alphastage")
    b = _describe(root, "beta-run", "betalabel", "betastage")
    return a, b


def _open(page, base, calc):
    slot = json.dumps(str(calc))
    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir.task-setup', {slot});"
        f" sessionStorage.setItem('molbuilder.current_dir', {slot});"
        "} catch (_) {}")
    page.goto(f"{base}/task-setup")
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        " return !!(n && n.CodeMirror); }", timeout=20000)
    unfold(page)


def test_opening_another_folder_leaves_nothing_of_the_first(
        page, flask_server, two_calcs):
    """The rule, executed: after moving A -> B, no trace of A is on screen.

    Asserted on the WHOLE page text rather than on named cards, deliberately.
    A per-card assertion is the same hand-kept list that failed — it can only
    catch the cards someone thought of.  This catches any of them, including
    ones added later.
    """
    a, b = two_calcs
    _open(page, flask_server, a)
    page.wait_for_function(
        "() => document.body.innerText.includes('alphalabel')", timeout=20000)

    # The sidebar moves — the app's own publish, not a reload.
    page.evaluate("(d) => window.molbuilder.projects.publishCommit(d, '')",
                  str(b))
    page.wait_for_function(
        "() => document.body.innerText.includes('betalabel')", timeout=20000)

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
    The described one names a queue, which its queue card says -- a card a
    hand-over does not show (the 2.27.0 review: it stayed, that queue on
    it, and a click on it wrote into no description).

    MUTATION THIS MUST FAIL AGAINST: the hand-over branch drawing no
    machine card (`applySetMachine`), so the queue card is never hidden.
    """
    a, _b = two_calcs
    req = read_request(a)
    req["allocation"] = {"domain": "alphaqueue"}
    write_request(a, req)
    bare = isolated_projects_root / "umbrella" / "optimization" / "bare-run"
    bare.mkdir(parents=True)
    (bare / "task.1st.json").write_text(json.dumps({
        "schema": "molbuilder/task-handover@1",
        "engine": {"name": "pyscf"},
        "run": {"name": "barelabel", "id": "barelabel_H2"},
        "structure": {"source": "bare.xyz", "formula": "H2", "atoms": 2},
        "awaiting": ["shape", "stages"],
        "_what": "a hand-over, as the Structure-optimization tab leaves one",
    }))

    _open(page, flask_server, a)
    page.wait_for_function(
        "() => document.body.innerText.includes('alphalabel')", timeout=20000)
    if not page.is_visible("#ts-ask-mem"):
        page.click("#ts-target-choice .opt")     # the machine, as a person picks it
    page.wait_for_function(
        "() => document.body.innerText.includes('alphaqueue')", timeout=20000)

    page.evaluate("(d) => window.molbuilder.projects.publishCommit(d, '')",
                  str(bare))
    # WAIT FOR THE LOAD TO FINISH, not for the first card to paint.  The
    # editor is filled LAST, after two awaits, while "what came over" paints
    # before them -- so waiting on the new run's name let the assertions run
    # mid-load and fail on a buffer that was about to be replaced anyway.
    # `awaiting` is the hand-over's own key and appears in no description,
    # so this marks completion without being the thing under test.
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        "  return !!(n && n.CodeMirror"
        "            && n.CodeMirror.getValue().includes('awaiting')); }",
        timeout=20000)

    text = every_card_text(page)
    assert "alphalabel" not in text, (
        "the described folder's label survived a move to one that has no "
        "description -- a card with nothing new to paint kept what it had")
    assert "alphastage" not in text, (
        "the described folder's stage survived a move to one that has none")
    assert "alphaqueue" not in text, (
        "the described folder's queue card survived a move to a hand-over")



def test_a_hand_over_of_a_kind_that_fixes_its_shape_starts_at_it(
        page, flask_server, isolated_projects_root):
    """`web/task-setup.md` § 4, `engines/stages.md` § 6.7: a SIESTA
    vibration's hand-over -- the road from the Spectrum page -- opens with
    its kind's shape on the card, pressed, the other not offered, the kind's
    one sentence beside it; the proposal is drawn at that shape, and Save
    writes it.

    THE SILENT FAILURE, shipped in 2.47.5: the card offered both shapes,
    flat among them, and Save refused it in a sentence the card never said.

    MUTATIONS THIS MUST FAIL AGAINST: the folder answer asking the shape
    only of a calculation that opens; the page's hand-over branch not
    reading the answer."""
    from molbuilder.kind_table import kind
    from molbuilder.task import of as task_of
    from molbuilder.web.app import create_app
    from support.envelope import envelope
    from conftest import write_pseudos
    d = isolated_projects_root / "handover" / "frequency" / "kind_shape"
    d.mkdir(parents=True)
    # SIESTA's pseudopotentials beside the calculation, as a person's lie
    # there (`engines/siesta.md`).
    write_pseudos(d, ["H"])
    out = create_app(config={}).test_client().post(
        "/api/task-setup/handover", json=dict(
            {"structure": envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])},
            engine="siesta", calculation="vibration", name="handed",
            params={"system_label": "handed"}, dest=str(d))).get_json()
    assert out["ok"], out
    for f in out["structure_files"]:
        (d / f["name"]).write_text(f["text"])
    (d / out["template_name"]).write_text(out["template_text"])
    (d / out["handover_name"]).write_text(out["handover_text"])
    fixed = kind("vibration").on("siesta").fixed_shape

    _open(page, flask_server, d)
    page.wait_for_selector("#ts-shape-fixed:not([hidden])", timeout=20000)
    assert page.inner_text("#ts-shape-fixed") == fixed["why"]
    page.wait_for_function(
        "() => document.getElementById('ts-shape-hier')"
        "  .getAttribute('aria-pressed') === 'true'", timeout=20000)
    assert page.locator("#ts-shape-flat").is_disabled(), (
        "a SIESTA vibration's hand-over was offered the flat shape")
    wait_for_file(page, "task", '"shape": "hierarchical"')

    page.click("#ts-save")
    page.wait_for_function(
        "() => { const n = document.getElementById('ts-save-said');"
        " return n && !n.hidden && /Saved/.test(n.textContent); }",
        timeout=20000)
    assert task_of(d).shape == fixed["shape"]


def test_a_hand_over_is_finished_on_the_page_and_saved_as_both_records(
        page, flask_server, isolated_projects_root):
    """`task-setup.md` § 4, § 7a, § 8: a hand-over becomes a calculation on
    this page -- a shape picked, the proposal drawn into both records, a run
    card filled, the shape changed once more, Save -- and the folder then
    holds the description and its job request, the run card's value in the
    job request, the save said beside the button after the folder is read
    again.  The page arrives from another folder, as a hand-over sent while
    Task setup is open does.

    THE SILENT FAILURES: a shape picked on a hand-over that proposes
    nothing -- the page editing the hand-over's own text as a description;
    the save's confirmation wiped by the folder's re-read -- a save that
    looks like nothing happened; a shape changed after the proposal
    rebuilding it, every edit made since lost; the run card's value written
    into no record.

    MUTATIONS THIS MUST FAIL AGAINST: the hand-over's text taken for a
    description (the shape written into it, no stages proposed); a shape
    change proposing the records again; the empty job-request text a
    hand-over loads, written into a document not on show, taken for a hand
    edit that does not parse -- the shape refused, the hand-over never
    finished."""
    from molbuilder.web.app import create_app
    from support.envelope import envelope
    d = (isolated_projects_root / "handover" / "optimization"
         / "finished_here")
    d.mkdir(parents=True)
    out = create_app(config={}).test_client().post(
        "/api/task-setup/handover", json=dict(
            {"structure": envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])},
            engine="pyscf", calculation="optimization", name="handed",
            params={"job_name": "handed"}, dest=str(d))).get_json()
    assert out["ok"], out
    for f in out["structure_files"]:
        (d / f["name"]).write_text(f["text"])
    (d / out["template_name"]).write_text(out["template_text"])
    (d / out["handover_name"]).write_text(out["handover_text"])
    came_from = _describe(isolated_projects_root / "handover", "came-from",
                          "camefromlabel", "camefromstage")

    _open(page, flask_server, came_from)
    page.wait_for_function(
        "() => document.body.innerText.includes('camefromlabel')",
        timeout=20000)
    page.evaluate("(d) => window.molbuilder.projects.publishCommit(d, '')",
                  str(d))
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        "  return !!(n && n.CodeMirror"
        "            && n.CodeMirror.getValue().includes('awaiting')); }",
        timeout=20000)
    page.wait_for_selector("#ts-main:not([aria-busy])", state="attached",
                           timeout=20000)
    page.click("#ts-shape-card .opt[data-shape='hierarchical']")
    # THE GROUPS CARD APPEARS WITH THE PROPOSAL, folded as every card is:
    # opened as a person opens it.
    unfold(page, "ts-groups-card")
    field = (".ts-next-step:not([hidden]) .ts-runcard "
             "[aria-label='value for threads']")
    page.wait_for_selector(field, timeout=20000)
    page.fill(field, "2")
    page.dispatch_event(field, "change")
    wait_for_file(page, "request", '"threads": 2')
    # ...THE SHAPE CHANGED ONCE MORE: one field of the description, the
    # edits made since kept.
    page.click("#ts-shape-card .opt[data-shape='flat']")
    wait_for_file(page, "task", '"shape": "flat"')
    assert '"threads": 2' in file_text(page, "request"), (
        "a shape changed after the proposal rebuilt the records, the run "
        "card's value lost")

    page.click("#ts-save")
    page.wait_for_selector("#ts-main:not([aria-busy])", state="attached",
                           timeout=20000)
    page.wait_for_function(
        "() => { const n = document.getElementById('ts-save-said');"
        " return n && !n.hidden && /Saved/.test(n.textContent); }",
        timeout=20000)
    said = page.evaluate(
        "() => document.getElementById('ts-save-said').textContent")
    assert "task.json" in said and "job-request.json" in said, said
    assert json.loads((d / "task.json").read_text())["shape"] == "flat"
    request = json.loads((d / "job-request.json").read_text())
    assert request["stages"] == {
        "coarse": {"group": 1, "execution": {"threads": 2}}}, request
