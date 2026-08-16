"""/molbuilder end-to-end — the contract's promise, driven the way a user drives it.

WHAT REPLACED WHAT, AND WHY IT MATTERS
--------------------------------------
The file that stood here had 139 tests and drove the page through
``window.molbuilder.molview.data``.  MolView publishes nothing to
``window.molbuilder`` (molview.md § 4) and has not since it was rebuilt, so on
2026-08-02 106 of those 139 failed with ``Cannot read properties of undefined``.
They were not testing the page; they were testing an architecture that no longer
exists.

**A test may not reach past the seal, and this is why.**  § 4 exports exactly
``mount`` and ``formula``; § 5.6 says a viewer belongs to whoever mounted it and
there is no registry.  A test that reaches the model is asserting on a thing the
page's own controls do not use — so it can pass while every control is dead.
That is not hypothetical: a browser walk of this tab on 2026-08-02 found SEVEN
defects the old suite was green on, including ``getState().indices`` — a key on
no snapshot, throwing on the first line of every UI refresh, swallowed by both
subscriber paths.  Every button, every readout and the whole state timeline sat
frozen at its template state, and 139 passing tests said nothing.

So every assertion below is something a user can see: DOM in, DOM out.  The
tests are the six steps of the browser walk in
``docs/web/molview.md`` § 6.5, and each one would have caught
at least one of those seven.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Coverage the retired file had that is NOT reproduced yet, recorded so its
absence is a known hole rather than a silent one: the electrode/junction ops,
the transform sub-tab (translate / rotate / centre), the by-residue and by-label
filters, the measurement readout, DNA/RNA/peptide generators, and the narrow-
viewport layout.  Each needs writing from the contract the same way; none of it
can come back by un-deleting, because all of it drove the dead global.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def flask_server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Pin ``tmp_path`` as the only Capabilities picker root, so the files
    blueprint will serve what the test writes there."""
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)


@pytest.fixture(autouse=True)
def _a_session_that_did_not_happen(tmp_path, monkeypatch):
    """Each test starts as a browser that has never opened this tab.

    MolView's sequence is PERSISTENT — it outlives the page (molview.md § 11.2)
    — so from the moment `load(0)` learned to adopt one, a test inherited
    whatever the test before it left on the canvas, unsaved badge and all.  The
    symptom was not a wrong assertion: the next test's Load hit the
    discard-unsaved gate and its click waited on a modal nobody answered.

    The isolation comes from giving the test server a projects root of its own.
    ``tmp_path`` is already per-test, so "no state from the last test" is not
    something this fixture has to arrange — it is true by construction, and the
    directory is thrown away with the test.

    This replaced a fixture that reached into the DEVELOPER'S real ``projects/``
    and unlinked ``ws-modify*.wc.json`` before and after every test.  It got the
    isolation by deleting files it did not own — including, if the developer had
    a live server on the same tree, their parked Modify-tab session.  The state
    landed there because ``workspace_storage`` resolves ``projects_root()`` =
    ``Path.cwd()/projects``, and pytest's cwd is the repo.  A test that wants a
    different root should say so, which is all this does.

    Patched on ``workspace_storage`` rather than on ``molbuilder.projects``
    because the name is bound at import (``from ... import projects_root``), so
    rebinding the source module would not reach the caller.
    """
    from molbuilder.web.blueprints import workspace_storage
    monkeypatch.setattr(workspace_storage, "projects_root", lambda: tmp_path)
    yield


@pytest.fixture
def labelled_xyz(tmp_path, monkeypatch):
    """A structure WITH its sidecar — the pair a project file really is.

    The label matters: it is carried only by the ``.molstruct.json``, so a load
    that shows it proves the server read the pair and the labels survived into
    the panel.  A bare .xyz would pass a weaker test.
    """
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    struct = Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.239, 0.927, 0.000],
        ]),
    )
    struct.regions = {"SOLVENT": [0, 1, 2]}
    xyz = tmp_path / "water.xyz"
    # WRITTEN BY THE CODEC THAT OWNS THE PAIR, not hand-authored JSON.  A sidecar
    # assembled by hand is missing the schema fields the load door checks (it
    # answers 400), and hand-authoring one here would also be the test asserting
    # against a schema it invented rather than the one the app writes.
    StructureCodec().write(struct, xyz)
    return xyz


# --------------------------------------------------------------------- #
#  Helpers — the page's OWN controls, nothing else                      #
# --------------------------------------------------------------------- #

_BOOT_MS = 15_000
_ACT_MS = 15_000

#: The card MolView builds into the tab's empty host (§ 8: one call builds it).
_CARD = "#molview-host .molviewer-card"
#: "N of M selected" — the panel's own line, and the only atom count on screen.
_COUNT = "#molview-host .molviewer-selection-count"
#: The unsaved-work badge, bottom-right of the 3D window (§ 11.2).
_BADGE = "#molview-host .molviewer-overlay--warn"


def _open(page, base_url):
    """Open /molbuilder and wait for the card MolView mounts.

    Waiting on the CARD, not on a global: the contract's promise is that one
    ``mount`` call builds the whole thing (§ 8), so the card appearing IS the
    page having a viewer.  There is nothing else to ask, by design.
    """
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{base_url}/molbuilder")
    page.wait_for_selector(_CARD, timeout=_BOOT_MS)
    page.wait_for_selector(f"{_CARD} canvas", timeout=_BOOT_MS)
    return errors


def _load(page, path: Path):
    """Pick the file in the sidebar, then press Load — the only supported route.

    Picking is browsing; loading is a separate intent that can discard unsaved
    work, which is why the tab makes it two acts (tabs.md).
    """
    page.evaluate(
        "(a) => window.molbuilder.projects.setShared(a.dir, a.file)",
        {"dir": str(path.parent.resolve()), "file": str(path.resolve())},
    )
    btn = page.locator("#load-candidate-btn")
    btn.wait_for(state="visible", timeout=_ACT_MS)
    page.wait_for_function(
        "() => !document.getElementById('load-candidate-btn').disabled",
        timeout=_ACT_MS)
    btn.click()
    page.wait_for_function(
        "() => /\\d+ of [1-9]\\d* selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=_ACT_MS)


def _atom_count(page) -> int:
    text = page.locator(_COUNT).inner_text()
    return int(text.split(" of ")[1].split()[0])


def _pick_atom(page, one_based: int):
    """Tick an atom's box in the panel — what a user does to select one."""
    page.locator(f"{_CARD} input[aria-label='Select atom #{one_based}']").click()


# --------------------------------------------------------------------- #
#  § 6.5 step 1 — the page mounts                                       #
# --------------------------------------------------------------------- #

def test_the_page_mounts_a_viewer(page, flask_server):
    """The card, its canvas and the op controls are on screen, with no error.

    This is step 1, and it is not trivial: this tab did not mount AT ALL for
    weeks.  ``selection-bootstrap.js`` tested for a viewer *before* calling
    ``mount`` — a name looked up in a global MolView had stopped publishing — so
    the guard failed every time and returned without ever mounting.
    """
    errors = _open(page, flask_server)
    assert page.locator("#save-state").count() == 1
    assert page.locator("#optab-btn-cell").count() == 1
    assert not errors, f"the page threw while starting up: {errors}"


# --------------------------------------------------------------------- #
#  § 6.5 step 2 — a file loads, and says so                             #
# --------------------------------------------------------------------- #

def test_loading_a_project_file_shows_its_atoms_and_its_labels(
        page, flask_server, labelled_xyz):
    """The atoms are drawn AND the sidecar's label reaches the panel.

    The label is the part worth asserting: it lives only in the
    ``.molstruct.json``, so seeing ``SOLVENT`` on the rows proves the server
    read the pair and that ``installMolecule`` installed the whole thing in one
    write.  Loading is what failed on every tab with *"installMolecule
    unavailable"* when the load door looked its viewer up by name.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    assert _atom_count(page) == 3
    assert "SOLVENT" in page.locator(f"{_CARD}").inner_text()


def test_the_status_line_says_what_landed(page, flask_server, labelled_xyz):
    """After a load the page says which file, and how many atoms.

    It used to speak only on FAILURE, so a successful load left the template's
    opening words — "No structure loaded." — sitting beside a drawn molecule.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    status = page.locator("#status").inner_text()
    assert "water.xyz" in status and "3" in status, status


def test_the_loader_readout_tells_picked_from_loaded(
        page, flask_server, labelled_xyz):
    """Picked (chosen in the sidebar) and Loaded (on the canvas) look different,
    and Load goes dead once there is nothing left to do.

    The page keeps this note itself: the viewer tracks contents, not files
    (§ 6.7).  It used to ask the selection snapshot for ``sourceFile`` — a key no
    snapshot has ever carried — so the readout said "Picked" with that very file
    on screen and the button never disabled.  A no-op click looked exactly like a
    real one.
    """
    _open(page, flask_server)
    page.evaluate(
        "(a) => window.molbuilder.projects.setShared(a.dir, a.file)",
        {"dir": str(labelled_xyz.parent.resolve()),
         "file": str(labelled_xyz.resolve())},
    )
    page.wait_for_function(
        "() => /^Picked:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=_ACT_MS)
    page.locator("#load-candidate-btn").click()
    page.wait_for_function(
        "() => /^Loaded:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=_ACT_MS)
    assert page.locator("#load-candidate-btn").is_disabled()


# --------------------------------------------------------------------- #
#  § 6.5 step 3 — the cell reads as the data says, on both surfaces      #
# --------------------------------------------------------------------- #

def test_the_cell_reads_the_same_on_both_surfaces(
        page, flask_server, labelled_xyz):
    """MolView's Cell page and the tab's Cell editor agree, and "(default)"
    marks a value the structure did not state.

    Derived-versus-explicit is a fact in the DATA — the server sends
    ``resolved_*`` beside the raw fields and neither surface computes it (§ 6.2).
    The editor used to call a ``{value, isDefault}`` family of which three names
    were on no MolView surface at all, each behind a feature guard, so every row
    rendered "(default)" whatever the structure said.  Two surfaces disagreeing
    about whether a structure has a cell is the bug the one server-side resolver
    exists to prevent.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    page.wait_for_selector("#pv-vac-a", state="visible", timeout=_ACT_MS)

    # This structure states no cell, so the tab's vacuum row is marked derived...
    assert "default" in page.locator("#pv-vac-tag").inner_text().lower()
    # ...and MolView's own readout says the same about it.
    readout = page.locator(f"{_CARD} .molviewer-cell-readout").inner_text()
    assert "default" in readout.lower()


def test_committing_a_vacuum_changes_the_box_and_drops_the_default_mark(
        page, flask_server, labelled_xyz):
    """A cell edit goes through the ONE cell door and the answer is displayed.

    ``commitPeriodicityOp`` is the only way the cell changes (§ 6.2): the server
    decides what the box becomes and MolView stores what comes back,
    interpreting none of it.  Once the vacuum is stated it is no longer derived,
    so the "(default)" mark must come OFF that row — that is the whole
    derived-vs-explicit contract, visible.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    page.wait_for_selector("#pv-vac-a", state="visible", timeout=_ACT_MS)
    before = page.locator(f"{_CARD} .molviewer-cell-readout").inner_text()

    for box in ("#pv-vac-a", "#pv-vac-b", "#pv-vac-c"):
        page.fill(box, "5")
    page.locator("#pv-vac-update").click()

    page.wait_for_function(
        "(prev) => (document.querySelector("
        "  '.molviewer-cell-readout')?.innerText || '') !== prev",
        arg=before, timeout=_ACT_MS)
    assert "default" not in page.locator("#pv-vac-tag").inner_text().lower(), \
        "a vacuum the user typed is explicit, and must stop reading as derived"


# --------------------------------------------------------------------- #
#  § 6.5 step 4 — an edit lands, and the page notices                   #
# --------------------------------------------------------------------- #

def test_selecting_an_atom_wakes_the_controls_that_need_one(
        page, flask_server, labelled_xyz):
    """Picking an atom enables Delete and names the anchor.

    THE REGRESSION THIS EXISTS FOR.  ``selectedIndices()`` read
    ``getState().indices`` — no snapshot has that key, so it was
    ``undefined.slice()``, a TypeError on the first line of every refresh, and
    both subscriber paths swallow what a subscriber throws.  Nothing reached the
    console.  Delete, Add, Orient, the anchor readouts, Save state, Retract and
    the timeline indicator never updated again after the page was built.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    assert page.locator("#delete-apply").is_disabled()

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=_ACT_MS)
    assert "#1" in page.locator("#add-anchor-readout").inner_text()


def test_the_edit_survives_a_page_reload(
        page, flask_server, labelled_xyz):
    """§ 11.2a: "the sequence outlives the page" — a fresh viewer ADOPTS the
    draft already in storage, and what comes back is the DRAFT, not the point.

    NOTHING IN THIS SUITE RELOADED A PAGE until 2026-08-04 — thirteen e2e files,
    zero ``page.reload()``.  So the whole of #44 was unguarded: the write half
    (history.js called a two-call door no workspace has ever had, so nothing was
    saved and every stub satisfied it perfectly) and the read half (``load(0)``
    refused on a fresh viewer, so the bytes were on disk and nothing could reach
    them).  A GENERATED structure — SMILES, DNA, peptide, no file behind it —
    was simply gone on leaving the tab.  It was verified once by hand and never
    since.

    Three things must come back, and they are the three a reopened page cannot
    infer: the atoms as EDITED (not as loaded), the unsaved badge, and the
    position in the sequence.  Asserting only the atom count would pass on a
    viewer that re-read the FILE and threw the edit away.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    before = _atom_count(page)

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=_ACT_MS)
    page.locator("#delete-apply").click()
    page.wait_for_function(
        f"() => /of {before - 1} selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=_ACT_MS)
    page.wait_for_selector(_BADGE, state="visible", timeout=_ACT_MS)

    page.reload()
    page.wait_for_selector(_CARD, timeout=_BOOT_MS)
    page.wait_for_selector(f"{_CARD} canvas", timeout=_BOOT_MS)
    page.wait_for_function(
        "() => /\\d+ of [1-9]\\d* selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=_BOOT_MS)

    assert _atom_count(page) == before - 1, (
        f"the reopened page shows {_atom_count(page)} atoms, not the {before - 1} "
        f"the edit left -- the draft was not adopted, or the FILE was re-read "
        f"and the edit thrown away")
    assert page.locator(_BADGE).is_visible(), (
        "the unsaved badge did not come back: `dirty` is one of the three "
        "fields that must travel WITH the draft, because a reopened page has "
        "no way to work it out")


def test_an_edit_changes_the_structure_and_raises_the_unsaved_badge(
        page, flask_server, labelled_xyz):
    """Delete removes the atom, the count follows, and the badge appears.

    The badge is raised INSIDE the viewer's gate when the change lands
    (§ 11.2) — not set by the page afterwards — so its appearing proves the edit
    reached the model rather than only the screen.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    assert page.locator(_BADGE).is_hidden()

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=_ACT_MS)
    page.locator("#delete-apply").click()

    page.wait_for_function(
        "() => /of 2 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=_ACT_MS)
    page.wait_for_selector(_BADGE, state="visible", timeout=_ACT_MS)
    assert "2 atoms" in page.locator("#edit-status").inner_text(), \
        "the op line reports the count off the structure the door handed back"


# --------------------------------------------------------------------- #
#  § 6.5 step 5 — the state timeline                                    #
# --------------------------------------------------------------------- #

def test_save_state_then_retract_puts_the_atom_back(
        page, flask_server, labelled_xyz):
    """A saved point is a place to come back to, and Retract comes back to it.

    Retract spends unsaved work first (§ 11.2): from a saved point with edits on
    top, the first press discards the edits and leaves you ON that point.  Here
    the edit sits on point 0, so one Retract restores the deleted atom.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => !document.getElementById('save-state').disabled",
        timeout=_ACT_MS)
    assert page.locator("#undo-op").is_disabled(), \
        "nothing has happened yet, so there is nothing to retract"

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=_ACT_MS)
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => /of 2 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=_ACT_MS)

    page.locator("#save-state").click()
    page.wait_for_function(
        "() => /saved #1/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=_ACT_MS)

    page.locator("#undo-op").click()
    page.wait_for_function(
        "() => /of 3 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=_ACT_MS)
    assert "#0" in page.locator("#timeline-status").inner_text()


def test_retract_says_so_when_the_point_it_wanted_is_gone(
        page, flask_server, labelled_xyz, tmp_path):
    """A retraction that could not happen must not be reported as one.

    A saved sequence is bounded at its last 30 saves (`workspace.md` § 9.1), so
    the oldest points are deleted as new ones arrive.  `load` answers that
    honestly -- it returns null and leaves `position` alone -- but the caller
    used to throw the answer away and print ``Retracted to state #N`` with "ok"
    styling, N being the position it was ALREADY at.  The user was told a
    retraction happened that did not.

    Reaching it for real would take 31 saves.  The state files are the test's
    own now (its projects root is `tmp_path`), so deleting point 0 reproduces
    exactly what the rolling window does, in one line and without the wait.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => !document.getElementById('save-state').disabled",
        timeout=_ACT_MS)

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=_ACT_MS)
    page.locator("#delete-apply").click()
    page.locator("#save-state").click()
    page.wait_for_function(
        "() => /saved #1/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=_ACT_MS)

    # Drop point 0 the way the rolling window would.
    states = tmp_path / ".molbuilder_workspace" / "states"
    gone = [p for p in states.glob("*.0.wc.json") if "-draft" not in p.name]
    assert gone, f"expected a point 0 to delete, found {list(states.iterdir())}"
    for p in gone:
        p.unlink()

    # Retract is offered -- we are at #1, so the button is enabled and the user
    # has every reason to expect it to work.
    assert not page.locator("#undo-op").is_disabled()
    page.locator("#undo-op").click()

    status = page.locator("#edit-status")
    page.wait_for_function(
        "() => /Nothing changed/.test("
        "  document.getElementById('edit-status').textContent)",
        timeout=_ACT_MS)
    text = status.inner_text()
    assert "Retracted to state" not in text, \
        f"retract claimed a move that did not happen: {text!r}"
    assert "modify-status--warn" in (status.get_attribute("class") or ""), \
        "a refusal styled as success reads as success"
    # And it really did not move.
    assert "#1" in page.locator("#timeline-status").inner_text()


def test_the_timeline_indicator_says_where_you_are(
        page, flask_server, labelled_xyz):
    """Unsaved work and the point Retract would restore are both on screen.

    A push-only timeline with no indicator is a promise the user cannot check:
    "Save state" only means something if you can see that you have not.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => /saved #0/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=_ACT_MS)

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=_ACT_MS)
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => /unsaved/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=_ACT_MS)
    assert "#0" in page.locator("#timeline-status").inner_text(), \
        "the indicator names the point Retract will return to"


# --------------------------------------------------------------------- #
#  § 6.5 step 6 — saving to the project                                 #
# --------------------------------------------------------------------- #

def test_saving_to_the_project_writes_the_pair_and_remembers_where(
        page, flask_server, labelled_xyz, tmp_path):
    """A save writes BOTH files, and the page records the target.

    The pair is the point: coordinates alone lose the labels, and a browser must
    never author the sidecar schema — the server writes both through
    ``StructureCodec.write`` (§ 11.3).  Where they went is the PAGE's note, not
    the viewer's (§ 6.7), and nothing was setting it: ``markSavedTo`` had no
    caller, so the readout still offered "Save as…" straight after a save.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)

    page.locator("#save-to-source-btn").click()
    name = page.locator(".molbuilder-save-name-modal input[data-role='name-input']")
    name.wait_for(state="visible", timeout=_ACT_MS)
    name.fill("saved_by_test")
    page.locator(
        ".molbuilder-save-name-modal button[data-action='save']").click()

    page.wait_for_function(
        "() => /Saved/.test(document.getElementById('save-status').textContent)",
        timeout=_ACT_MS)

    assert (tmp_path / "saved_by_test.xyz").exists()
    assert (tmp_path / "saved_by_test.molstruct.json").exists(), \
        "the sidecar carries the labels; a lone .xyz silently loses them"
    sidecar = json.loads((tmp_path / "saved_by_test.molstruct.json").read_text())
    assert sidecar["regions"] == {"SOLVENT": [0, 1, 2]}
    assert "saved_by_test.xyz" in page.locator("#save-readout").inner_text()


def test_the_page_remembers_which_file_it_is_showing_across_a_reload(
        page, flask_server, labelled_xyz):
    """The page's own note survives, so Load does not offer to discard your work.

    workspace.md § 4: a page may have several savers, kept apart by their tags,
    and it names this one — *"the Modify tab has a viewer holding a molecule AND
    its own panel state"*.  The viewer saves under `modify`; the page saves under
    `modify:panel`.  Two tags, two slots.

    THE BUG THIS CLOSES.  `loadedFrom` used to live in a closure variable, set
    only by the path that READS A FILE.  When a reload was served by the viewer's
    restore instead — now the normal case — it was empty while a structure was
    plainly on the canvas, so the readout fell back to "Picked:" and **the Load
    button re-enabled against the very file the work came from**.  One press
    discarded the restored work through the dirty gate.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => /^Loaded:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=_ACT_MS)

    # Come back to the tab.
    page.goto(f"{flask_server}/molbuilder")
    page.wait_for_selector(_CARD, timeout=_BOOT_MS)
    page.wait_for_function(
        "() => /\\d+ of [1-9]\\d* selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=_ACT_MS)

    page.wait_for_function(
        "() => /^Loaded:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=_ACT_MS)
    assert page.locator("#load-candidate-btn").is_disabled(), (
        "the Load button came back enabled against the file the structure is "
        "already showing — pressing it discards the restored work"
    )


def test_a_generated_structure_claims_no_file(page, flask_server, labelled_xyz):
    """A structure built from SMILES has no file behind it, and the page says so.

    The note is written at the ONE gate every generator comes through, which
    already knows whether a file was involved.  Before, `loadedFrom` was whatever
    the last file load had left there, so a generated molecule inherited a
    filename it had nothing to do with — and the loader readout claimed that file
    was on the canvas.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)          # a real file first, so the note is set
    page.wait_for_function(
        "() => /^Loaded:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=_ACT_MS)

    # Now generate, which replaces it with something that came from no file.
    page.evaluate(
        "() => { [...document.querySelectorAll('.modify-init-tab')]"
        "  .find(b => /SMILES/i.test(b.textContent)).click();"
        "  const i = document.getElementById('smiles-input');"
        "  i.value = 'CCO';"
        "  i.dispatchEvent(new Event('input', {bubbles:true})); }")
    page.locator("#smiles-generate-btn").click()
    page.wait_for_function(
        "() => /of 9 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=30_000)

    readout = page.locator("#load-candidate-readout").inner_text()
    assert not readout.startswith("Loaded:"), (
        f"a generated structure is claiming to be the file that was loaded "
        f"before it: {readout!r}"
    )
