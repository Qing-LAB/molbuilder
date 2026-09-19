"""The page is a function of the folder — checked by moving between two.

`web/task-setup.md` § 2.1: *"the page holds no state of its own… the folder
is the only link."*  That sentence was written before it was kept.  The page
assembled itself from twelve endpoints, each painting its own card, and the
per-folder ones were cleared by `_resetPerFolderState()` — a hand-written
list of clears in a module with twenty-five variables.  A list only ever
covers what someone remembered to add to it, so the rule needed a check that
does not depend on remembering: **open A, open B, and assert that nothing of
A is on the page.**

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
        [Stage(name=stage, enabled=True, overrides={})],
        engine="pyscf", shape="hierarchical", name=label,
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
    # cards.  Asserting on `document.body` swept the sidebar's file list in
    # and failed on `alphalabel.template.toml` sitting there -- a true
    # observation about the wrong component.  Still the WHOLE panel and not
    # named cards: a per-card list is the same hand-kept thing that failed.
    text = page.evaluate(
        "() => (document.querySelector('main') || document.body).innerText")
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
    """
    a, _b = two_calcs
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

    text = page.evaluate(
        "() => (document.querySelector('main') || document.body).innerText")
    assert "alphalabel" not in text, (
        "the described folder's label survived a move to one that has no "
        "description -- a card with nothing new to paint kept what it had")
    assert "alphastage" not in text, (
        "the described folder's stage survived a move to one that has none")
