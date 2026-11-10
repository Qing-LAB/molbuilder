"""/molbuilder end-to-end — the contract's promise, driven the way a user drives it.

**A test may not reach past the seal, and this is why.**  `web/molview.md` § 4
exports ``mount``, ``formula``, ``toDisplay`` and ``fromDisplay`` and nothing
else; § 5.6 says a viewer belongs to whoever mounted it and there is no
registry.  A test that reaches the model is asserting on a thing the page's own
controls do not use — so it can pass while every control is dead.

So every assertion below is on what a person sees on the page, or on what
the server answered the page for the person's act -- the structure an op's
door sent back, the lattice a measurement read: never on the model behind
the seal.  The walk is the tab's own: a file loads, and a second one is added
(`web/tabs.md` § 2; molview § 6.6, § 6.7); the cell reads alike on both
surfaces and changes through its one door (molview § 9.3;
`model/structure-periodicity.md` § 7); an edit lays down its own point and
Retract steps back (molview § 1.1, § 11.2, § 11.2a); the structure is saved
to the project (`tabs.md` § 6).

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Coverage NOT reproduced here, recorded so its
absence is a known hole rather than a silent one: the transform sub-tab's
translate and centre, the by-residue and by-label filters' selections,
the measurement readout, the DNA/RNA/peptide generators and the findings their
panels draw (`science/validation.md` § 4.1 R5), and the narrow-viewport
layout.  Each needs writing from the contract the same way.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from support.sidebar import WAIT_MS, walk_to  # noqa: E402


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.fixture
def labelled_xyz(isolated_projects_root):
    """A structure WITH its sidecar — the pair a project file really is.

    The label matters: it is carried only by the ``.molstruct.json``, so a load
    that shows it proves the server read the pair and the labels survived into
    the panel.  A bare .xyz would pass a weaker test.
    """
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    struct = Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.239, 0.927, 0.000],
        ]),
    )
    struct.regions = {"SOLVENT": [0, 1, 2]}
    xyz = isolated_projects_root.resolve() / "water.xyz"
    # WRITTEN BY THE CODEC THAT OWNS THE PAIR, not hand-authored JSON.  A sidecar
    # assembled by hand is missing the schema fields the load door checks (it
    # answers 400), and hand-authoring one here would also be the test asserting
    # against a schema it invented rather than the one the app writes.
    StructureCodec().write(struct, xyz)
    return xyz


# --------------------------------------------------------------------- #
#  Helpers — the page's OWN controls, nothing else                      #
# --------------------------------------------------------------------- #

#: The card MolView builds into the tab's empty host (§ 8: one call builds it).
_CARD = "#molview-host .molviewer-card"
#: "N of M selected" — the panel's own line, and the only atom count on screen.
_COUNT = "#molview-host .molviewer-selection-count"


def _open(page, base_url):
    """Open /molbuilder and wait for the card MolView mounts.

    Waiting on the CARD, not on a global: the contract's promise is that one
    ``mount`` call builds the whole thing (§ 8), so the card appearing IS the
    page having a viewer.  There is nothing else to ask, by design.
    """
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{base_url}/molbuilder")
    page.wait_for_selector(_CARD, timeout=WAIT_MS)
    page.wait_for_selector(f"{_CARD} canvas", timeout=WAIT_MS)
    return errors


def _load(page, path: Path, answer=None):
    """Pick the file in the sidebar, then press Load -- one of the two ways a
    person loads a file; a double-click on its row is the other, the same
    action (`tabs.md` § 2).

    Picking is browsing; loading is a separate intent that can discard unsaved
    work, which is why the tab makes it two acts (tabs.md).  With a structure
    open, Load asks whether to add the file to the view or clear the view
    (tabs.md § 2, user 2026-10-09): ``answer`` is ``"add"`` -- Enter, the
    focused default -- or ``"clear"``.
    """
    walk_to(page, path)
    btn = page.locator("#load-candidate-btn")
    btn.wait_for(state="visible", timeout=WAIT_MS)
    page.wait_for_function(
        "() => !document.getElementById('load-candidate-btn').disabled",
        timeout=WAIT_MS)
    before = _atom_count_or_zero(page)
    btn.click()
    if answer is not None:
        asked = page.locator("dialog.molbuilder-warning-modal[open]")
        asked.wait_for(state="visible", timeout=WAIT_MS)
        assert ("add the structure to the existing view or clear the current "
                "view") in asked.inner_text(), asked.inner_text()
        if answer == "add":
            page.keyboard.press("Enter")
        else:
            asked.locator("[data-action='discard']").click()
    # WAIT FOR THE CHANGE, not for a pattern the PREVIOUS state already
    # satisfies.
    page.wait_for_function(
        "(n) => { const t = document.querySelector("
        "  '.molviewer-selection-count')?.textContent || '';"
        "  const m = /\\d+ of (\\d+) selected/.exec(t);"
        "  return !!m && Number(m[1]) !== n; }",
        arg=before, timeout=WAIT_MS)


def _atom_count_or_zero(page) -> int:
    """The count, or 0 when nothing is open yet."""
    try:
        return _atom_count(page)
    except Exception:
        return 0


def _atom_count(page) -> int:
    text = page.locator(_COUNT).inner_text()
    return int(text.split(" of ")[1].split()[0])


def _pick_atom(page, one_based: int):
    """Tick an atom's box in the panel — what a user does to select one."""
    page.locator(f"{_CARD} input[aria-label='Select atom #{one_based}']").click()


def _clear_ruler(page):
    """The ruler's own Clear — *"the selection is not touched"*.

    A separate control from the selection's, because they are separate tracks
    (`molview.md` § 11.6).  The Cell page's gestures read the RULER, so a test
    re-picking a pair must clear THAT one; clearing the selection instead
    leaves the ruler holding the old picks and the second pair never forms.
    """
    page.locator(f"{_CARD} .molviewer-measure-clear").click()


@pytest.fixture
def two_files(isolated_projects_root):
    """Two DIFFERENT structures, each a real pair, SHARING one label name.

    The shared name is the point: `Structure.concat` unions labels by name, so
    two fragments both called ``FRAG`` would end up one region and stop being
    separately selectable.  The incoming one is numbered instead, and that is
    visible on the rows.
    """
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    made = []
    for name, elements, xs in (
        ("water", ["O", "H", "H"], [0.0, 0.957, -0.239]),
        ("pair",  ["N", "N"],      [10.0, 11.1]),
    ):
        struct = Structure(
            elements=elements,
            positions=np.array([[x, 0.0, 0.0] for x in xs]),
        )
        struct.regions = {"FRAG": list(range(len(elements)))}
        path = isolated_projects_root.resolve() / f"{name}.xyz"
        StructureCodec().write(struct, path)
        made.append(path)
    return made


# --------------------------------------------------------------------- #
#  Loading ADDS to what is open (user, 2026-09-07)                      #
# --------------------------------------------------------------------- #

def test_a_second_load_adds_to_the_first_instead_of_replacing_it(
        page, flask_server, two_files):
    """*"the load from project or other generators should by default ADD their
    results into the molview structure instead of clear the existing one ...
    such that we can keep adding content into the same editing session"*
    (user, 2026-09-07).

    Driven the way a person drives it: pick, Load, pick, Load.  `tabs.md` § 2:
    *"With a structure open, it is ADDED"*, and the merge numbers *"a label the
    canvas already has"* (`model/structure.md` § 2.2b).  So the rows, read
    whole, are the canvas's three atoms under FRAG and the incoming two under
    FRAG2 -- the sum, with the first fragment still identifiable and the
    second not merged into it.

    THE SILENT FAILURES: a replace (two rows), and the incoming label merged
    into the canvas's (five rows of FRAG, the fragments no longer separable).
    """
    water, pair = two_files
    errors = _open(page, flask_server)
    _load(page, water)
    _load(page, pair, answer="add")

    # WHICH ATOMS ARE THERE, not how many: each row's labels, as whole names,
    # so FRAG cannot pass as a part of FRAG2.
    rows = page.eval_on_selector_all(
        f"{_CARD} .molviewer-atoms-table tr",
        "rows => rows.map((r) => [...r.querySelectorAll("
        "  '.molviewer-atoms-column-labels .molviewer-selection-tag > span')]"
        "  .map((s) => s.textContent))")
    assert rows == [["FRAG"]] * 3 + [["FRAG2"]] * 2, (
        "the canvas's three atoms keep FRAG and the incoming two arrive as "
        f"FRAG2: {rows}")
    assert not errors, errors


def test_the_status_line_says_a_load_was_added(
        page, flask_server, two_files):
    """The line distinguishes the two things a Load can do, because they are
    different: the first put a structure on an empty canvas, the second added
    to one that was not.

    `web/molview.md` § 6.7: *"Loaded wire.xyz" ... are the tab's to say* --
    the tab did the operation, so the tab reports it; `tabs.md` § 2 makes the
    second Load an ADD.  THE SILENT FAILURE: an add reported as a load, which
    reads as the file having replaced what was open.
    """
    water, pair = two_files
    _open(page, flask_server)
    _load(page, water)
    assert "Loaded" in page.locator("#status").inner_text()

    _load(page, pair, answer="add")
    status = page.locator("#status").inner_text()
    assert "Added" in status and "5 in total" in status, status


def test_start_empty_then_load_is_how_you_replace(
        page, flask_server, two_files):
    """After *Clear structure*, a Load installs the file without asking, and
    the canvas holds the file alone.

    `web/molview.md` § 6.7a: *Clear* is the start-empty operation -- nothing
    is loaded, and the sequence starts again; `tabs.md` § 2: *"With nothing
    open the file is installed and nothing is asked."*

    THE SILENT FAILURES: Clear leaving atoms behind for the next Load to add
    to (5, not 2); and the add-or-clear question put on an empty canvas --
    ``_load`` with no answer then waits on a count that never changes.
    """
    water, pair = two_files
    _open(page, flask_server)
    _load(page, water)
    assert _atom_count(page) == 3

    # The confirm is the app's own modal, not window.confirm (viewer.js).
    page.locator("#clear-apply").click()
    page.locator("button:has-text('Clear structure')").last.click()
    page.wait_for_function(
        "() => (document.getElementById('edit-status')?.textContent || '')"
        ".includes('Cleared')", timeout=WAIT_MS)

    _load(page, pair)
    assert _atom_count(page) == 2, (
        "loading after Clear structure did not replace -- it added to a canvas "
        "that should have been empty")


def test_a_second_load_answered_clear_replaces_the_view(
        page, flask_server, two_files):
    """The Load button's question answered *Clear current view*: the file
    replaces what was open (user, 2026-10-09; tabs.md § 2)."""
    water, pair = two_files
    _open(page, flask_server)
    _load(page, water)
    _load(page, pair, answer="clear")
    assert _atom_count(page) == 2, (
        "answered clear, the load added to the open structure")


def test_each_append_is_a_point_the_timeline_can_come_back_to(
        page, flask_server, two_files):
    """*"all operation should automatically call state save timeline api, such
    that the user always can retract back"* (user, 2026-09-07).

    An append is *"an edit like any other, a point on the timeline"*
    (`tabs.md` § 2), and every edit lays down a point of its own
    (`web/molview.md` § 1.1, § 11.2).  Retract therefore steps back exactly ONE
    load -- with Save state never pressed.
    """
    water, pair = two_files
    _open(page, flask_server)
    _load(page, water)
    _load(page, pair, answer="add")
    assert _atom_count(page) == 5

    page.locator("#undo-op").click()
    page.wait_for_function(
        "() => /\\d+ of 3 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)
    assert _atom_count(page) == 3, (
        "Retract did not step back exactly one load")


# --------------------------------------------------------------------- #
#  A file loads, and says so (tabs.md § 2; molview § 6.6, § 6.7)         #
# --------------------------------------------------------------------- #

def _offered(page, select: str) -> list[str]:
    """The names a label chooser offers, in order: its options' values,
    without the chooser's own empty entry (*+ new label…* / *Choose a
    label…*, value "")."""
    return page.eval_on_selector_all(
        f"{select} option", "os => os.map((o) => o.value).filter((v) => v)")


def test_loading_a_project_file_shows_its_atoms_and_its_labels(
        page, flask_server, labelled_xyz):
    """The atoms are drawn AND the sidecar's label reaches the panel -- and
    both label choosers offer the one list.

    The label is the part worth asserting: it lives only in the
    ``.molstruct.json``, so seeing ``SOLVENT`` on the rows proves the server
    read the pair and that ``installMolecule`` installed the whole thing in one
    write.  Loading is what failed on every tab with *"installMolecule
    unavailable"* when the load door looked its viewer up by name.

    THE LABEL PALETTE (`web/molview.md` § 6.6, *Where a label can be chosen*;
    2.65.0): *"The offered names first — the kind's labels in its table's
    order, then MolView's own (`offeredLabels`); then whatever the loaded
    structure carries that is not already among them; and a name appears
    once"* -- the kind's arriving with the structure (the answer's
    ``labels``, `kind_table.labels`), MolView's own *"`interface` and
    `frozen_atoms` ... offered after the kind's"*.  The Assign chooser and a
    `by label` row *"both read the one list"*, and *"until a person picks a
    name, the Assign chooser shows the first one offered"*.

    THE SILENT FAILURES: the kind's labels not reaching the page (every chip
    in one neutral colour, the leads typed by hand), the two controls
    disagreeing about what exists, and a default picked for the person.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    assert _atom_count(page) == 3
    assert "SOLVENT" in page.locator(f"{_CARD}").inner_text()

    from molbuilder.kind_table import labels
    want: list[str] = []
    for name in ([name for name, _about in labels()]
                 + ["interface", "frozen_atoms", "SOLVENT"]):
        if name not in want:
            want.append(name)
    chooser = f"{_CARD} .molviewer-selection-assign-select"
    page.wait_for_function(
        "(sel) => [...document.querySelectorAll(sel + ' option')]"
        "  .some((o) => o.value === 'SOLVENT')",
        arg=chooser, timeout=WAIT_MS)
    assert _offered(page, chooser) == want, _offered(page, chooser)
    assert page.locator(chooser).input_value() == want[0], (
        "the Assign chooser does not show the head of the list while nobody "
        "has picked a name")

    page.locator(f"{_CARD} .molviewer-selection-mode-option",
                 has_text="Filter").click()
    page.locator(f"{_CARD} .molviewer-selection-add-filter-row").click()
    page.locator(f"{_CARD} .molviewer-filter-row .molviewer-filter-kind"
                 ).first.select_option("by_label")
    by_label = f"{_CARD} .molviewer-filter-row select.molviewer-filter-text"
    page.wait_for_selector(by_label, timeout=WAIT_MS)
    assert _offered(page, by_label) == want, _offered(page, by_label)


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
    """Picked (chosen in the sidebar) and Loaded (on the canvas) look
    different.

    The page keeps this note itself: the viewer tracks contents, not files
    (§ 6.7).  It used to ask the selection snapshot for ``sourceFile`` — a key
    no snapshot has ever carried — so the readout said "Picked" with that very
    file on screen.

    Loading the same file again is a real action — it is how you throw your
    edits away and go back to what is on disk (user: *"why do we have this
    guard of not allowing to pick the same file again?"*).
    """
    _open(page, flask_server)
    walk_to(page, labelled_xyz)
    page.wait_for_function(
        "() => /^Picked:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=WAIT_MS)
    page.locator("#load-candidate-btn").click()
    page.wait_for_function(
        "() => /^Loaded:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=WAIT_MS)
    assert not page.locator("#load-candidate-btn").is_disabled(), (
        "loading the same file again is how you discard your edits and go "
        "back to what is on disk -- the button must stay live")


# --------------------------------------------------------------------- #
#  The cell reads as the data says, on both surfaces                     #
#  (molview § 9.3; structure-periodicity.md § 7)                         #
# --------------------------------------------------------------------- #

def _cell_row(page, label: str) -> str:
    """MolView's Cell-page row named ``label``, as it reads."""
    return page.locator(
        f"{_CARD} .molviewer-cell-readout dt:text-is('{label}') + dd"
    ).inner_text()


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

    Read row by row: MolView's Vacuum row is *"a default when the structure
    states none"* (molview § 9.3's table) -- and its Origin and Axes rows are
    defaults for this water too, so the whole readout carrying "(default)"
    somewhere says nothing about the vacuum.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    page.wait_for_selector("#pv-vac-a", state="visible", timeout=WAIT_MS)

    # This structure states no vacuum, so the tab's vacuum row is marked
    # derived...
    assert "default" in page.locator("#pv-vac-tag").inner_text().lower()
    # ...and MolView's own Vacuum row says the same about it.
    row = _cell_row(page, "Vacuum")
    assert "(default)" in row, row


def test_committing_a_vacuum_changes_the_box_and_drops_the_default_mark(
        page, flask_server, labelled_xyz):
    """A cell edit goes through the ONE cell door and the answer is displayed.

    ``commitPeriodicityOp`` is the only way the cell changes (§ 6.2): the server
    decides what the box becomes and MolView stores what comes back,
    interpreting none of it.  Once the vacuum is stated it is no longer derived,
    so the "(default)" mark must come OFF that row on both surfaces — that is
    the whole derived-vs-explicit contract, visible — and MolView's Vacuum row
    reads the gap that was committed, per axis (molview § 9.3).
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    page.wait_for_selector("#pv-vac-a", state="visible", timeout=WAIT_MS)
    before = page.locator(f"{_CARD} .molviewer-cell-readout").inner_text()

    for box in ("#pv-vac-a", "#pv-vac-b", "#pv-vac-c"):
        page.fill(box, "5")
    # ONE COMMIT for the whole cell (§ 6.2).
    page.locator("#pv-apply").click()

    page.wait_for_function(
        "(prev) => (document.querySelector("
        "  '.molviewer-cell-readout')?.innerText || '') !== prev",
        arg=before, timeout=WAIT_MS)
    assert "default" not in page.locator("#pv-vac-tag").inner_text().lower(), \
        "a vacuum the user typed is explicit, and must stop reading as derived"
    row = _cell_row(page, "Vacuum")
    assert "(default)" not in row, (
        f"MolView's Vacuum row still reads as derived after the commit: {row}")
    assert [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", row)] \
        == [5.0, 5.0, 5.0], f"MolView's Vacuum row is not the gap committed: {row}"


# --------------------------------------------------------------------- #
#  structure-periodicity.md § 7 — a cell value taken off the structure  #
# --------------------------------------------------------------------- #

def _cell_explicit(page):
    """Choose the EXPLICIT regime, which is what shows the 3x3 and the origin.

    The panel asks WHICH BOX first and then shows that regime's fields (user,
    2026-09-07).
    """
    page.locator('input[name="pv-regime"][value="explicit"]').check()
    page.wait_for_selector("#pv-cell-grid input", state="visible",
                           timeout=WAIT_MS)


def _cell_boxes(page):
    return [page.locator("#pv-cell-grid input").nth(i).input_value()
            for i in range(9)]


def _clear_selection(page):
    page.locator(f"{_CARD} .molviewer-selection-clear-btn").click()


def test_two_picked_atoms_become_a_lattice_vector_and_commit_nothing(
        page, flask_server, labelled_xyz):
    """§ 7: *Use picked atoms* beside the axis chooser writes `second − first` into
    that row — and STAGES it.

    Both halves matter.  The vector is checkable (water's O→H is 0.957 Å along
    x), and the staging is what keeps the group's own Update the only commit:
    a gesture that committed would be a second door for the gate to stand in
    front of, and the user would never see what was about to be sent.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    _cell_explicit(page)
    mirror_before = page.locator(f"{_CARD} .molviewer-cell-readout").inner_text()

    # The button says what it needs while it cannot run.
    assert page.locator("#pv-cell-from-selection").is_disabled()
    assert "two atoms" in page.locator("#pv-cell-from-selection").get_attribute("title")

    _pick_atom(page, 1)                       # O at the origin
    _pick_atom(page, 2)                       # H at (0.957, 0, 0)
    page.wait_for_function(
        "() => !document.getElementById('pv-cell-from-selection').disabled",
        timeout=WAIT_MS)
    page.locator("#pv-cell-from-selection").click()

    row_a = [float(v) for v in _cell_boxes(page)[:3]]
    assert abs(row_a[0] - 0.957) < 1e-3, row_a
    assert abs(row_a[1]) < 1e-3 and abs(row_a[2]) < 1e-3, row_a
    # The chooser carries each axis's length, so all three read from one control.
    assert "0.957" in page.locator("#pv-cell-axis").inner_text()
    # NOTHING WAS COMMITTED: MolView's read-only Cell page still says what it did.
    assert page.locator(f"{_CARD} .molviewer-cell-readout").inner_text() \
        == mirror_before, "the gesture committed; it must only stage"


def test_the_click_order_is_the_axis_direction(page, flask_server, labelled_xyz):
    """§ 7: the axis runs from the atom clicked FIRST to the one clicked SECOND,
    so the same pair the other way round negates it.

    That is not a nicety — it is the stated way out of the left-handed refusal
    below, so it has to be true rather than approximately true.

    The gesture reads the RULER, whose list is ordered by construction and is
    a different track from the selection (`molview.md` §§ 9.5, 11.6).

    Opening the Cell tab turns the ruler on, which is why the picks below land
    in it: `pickAtom` routes by whether measuring is on, so the same click that
    would select an atom measures one instead.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    _cell_explicit(page)

    _pick_atom(page, 1)
    _pick_atom(page, 2)
    page.wait_for_function(
        "() => !document.getElementById('pv-cell-from-selection').disabled",
        timeout=WAIT_MS)
    page.locator("#pv-cell-from-selection").click()
    forward = [float(v) for v in _cell_boxes(page)[:3]]

    # THE RULER's Clear, not the selection's -- they are separate tracks, and
    # clearing the wrong one leaves the old pair in place so the second never
    # forms.
    _clear_ruler(page)
    _pick_atom(page, 2)
    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('pv-cell-from-selection').disabled",
        timeout=WAIT_MS)
    page.locator("#pv-cell-from-selection").click()
    back = [float(v) for v in _cell_boxes(page)[:3]]

    assert all(abs(f + b) < 1e-6 for f, b in zip(forward, back)), (
        f"picking the pair the other way round did not negate the axis: "
        f"{forward} vs {back}")


def test_setting_a_length_keeps_the_direction(page, flask_server, labelled_xyz):
    """§ 7: *Set length* rescales the row to the stated magnitude — the spacing
    between periodic images, set without touching where the axis points."""
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    _cell_explicit(page)

    _pick_atom(page, 1)
    _pick_atom(page, 3)                       # O -> H at (-0.239, 0.927, 0)
    page.locator("#pv-cell-from-selection").click()
    before = [float(v) for v in _cell_boxes(page)[:3]]

    page.fill("#pv-cell-len", "12")
    page.locator("#pv-cell-set-len").click()
    after = [float(v) for v in _cell_boxes(page)[:3]]

    length = sum(v * v for v in after) ** 0.5
    assert abs(length - 12.0) < 1e-3, f"length not set: {after} -> {length}"

    # SAME DIRECTION, to the precision the panel writes.  Every staged number is
    # rounded at the 6th decimal -- the page's one rounding, because the input
    # IS the value that will be sent -- so the scaled components' ratios agree
    # to ~1e-7, not to machine precision.  The honest assertion is therefore the
    # ANGLE between the two vectors, not the equality of their ratios.
    dot = sum(x * y for x, y in zip(before, after))
    mags = (sum(v * v for v in before) ** 0.5) * length
    assert dot / mags > 1 - 1e-6, (
        f"rescaling turned the axis: {before} -> {after}")
    assert all(a * b > 0 for a, b in zip(before, after) if abs(b) > 1e-6), (
        f"rescaling flipped a component: {before} -> {after}")


def test_one_picked_atom_becomes_the_box_origin(page, flask_server,
                                                 labelled_xyz):
    """§ 7: *Use picked atom* beside the origin boxes puts the corner the box is
    drawn from on the selected atom.  Staged, like the other one."""
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    _cell_explicit(page)

    assert page.locator("#pv-org-from-selection").is_disabled()
    _pick_atom(page, 3)                       # H at (-0.239, 0.927, 0.000)
    page.wait_for_function(
        "() => !document.getElementById('pv-org-from-selection').disabled",
        timeout=WAIT_MS)
    page.locator("#pv-org-from-selection").click()

    got = [float(page.locator(f"#pv-org-{ax}").input_value()) for ax in "abc"]
    assert abs(got[0] + 0.239) < 1e-3 and abs(got[1] - 0.927) < 1e-3 \
        and abs(got[2]) < 1e-3, got


def _apply_and_wait(page):
    """Press Apply and wait for MolView's readout to show the answer."""
    before = page.locator(f"{_CARD} .molviewer-cell-readout").inner_text()
    page.locator("#pv-apply").click()
    page.wait_for_function(
        "(prev) => (document.querySelector("
        "  '.molviewer-cell-readout')?.innerText || '') !== prev",
        arg=before, timeout=WAIT_MS)


def test_an_origin_is_assigned_on_a_typed_cell_and_automatic_returns(
        page, flask_server, labelled_xyz):
    """`model/structure-periodicity.md` § 6.0, *A stated offset*: on a typed
    cell a person may assign the box's origin -- three numbers -- and blank is
    Automatic, the atoms centred by the rule.  Both surfaces
    say which: the editor's origin boxes with their "(default)" tag, and
    MolView's Origin row (molview § 9.3, `getUnitCellOrigin`).  Automatic is a
    gesture too, since once something is typed there is otherwise no way back
    to nothing.

    And what an Apply re-sends is what was typed, to the page's one rounding
    at the 6th decimal (§ 7): a six-decimal lattice length and origin come
    back exactly, through a second Apply as well.  At three decimals they came
    back altered, which is how a re-sent corner once put an atom 1.6e-5 Å past
    a face (§ 6.0).
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    _cell_explicit(page)
    _apply_and_wait(page)                     # the molecule's box, now typed

    org = [page.locator(f"#pv-org-{ax}") for ax in "abc"]
    assert [b.input_value() for b in org] == ["", "", ""]
    assert "default" in page.locator("#pv-org-tag").inner_text().lower()
    assert "(default)" in _cell_row(page, "Origin")

    page.locator("#pv-cell-grid input").nth(0).fill("12.345678")
    for b in org:
        b.fill("-1.234567")
    _apply_and_wait(page)
    assert [float(b.input_value()) for b in org] == [-1.234567] * 3
    assert float(page.locator("#pv-cell-grid input").nth(0).input_value()) == 12.345678
    assert "default" not in page.locator("#pv-org-tag").inner_text().lower()
    row = _cell_row(page, "Origin")
    assert row.startswith("-1.235, -1.235, -1.235") and "(default)" not in row, row
    # A second Apply re-sends what the boxes now hold, and nothing drifts.
    # The edit it answers is c's length (the vacuum is hidden under a typed
    # cell); a and the origin ride along untouched.
    page.locator("#pv-cell-grid input").nth(8).fill("15.5")
    _apply_and_wait(page)
    assert [float(b.input_value()) for b in org] == [-1.234567] * 3
    assert float(page.locator("#pv-cell-grid input").nth(0).input_value()) == 12.345678

    page.locator("#pv-org-auto").click()
    _apply_and_wait(page)
    assert [b.input_value() for b in org] == ["", "", ""]
    assert "default" in page.locator("#pv-org-tag").inner_text().lower()
    assert "(default)" in _cell_row(page, "Origin")


def test_the_panel_says_left_handed_before_the_server_refuses_it(
        page, flask_server, labelled_xyz):
    """§ 7: the gate refuses `det <= 0` with a 400, and picking three atom pairs
    produces one about half the time — so the panel says it while the row is
    still being built, with the gesture's own way out.

    Advisory, not a second rule: the note appears and disappears with the sign,
    and the server remains the thing that refuses.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-cell").click()
    _cell_explicit(page)

    # A deliberately mirrored frame: x, y, and MINUS z.
    for i, v in enumerate([1, 0, 0, 0, 1, 0, 0, 0, -1]):
        page.locator("#pv-cell-grid input").nth(i).fill(str(v))
    note = page.locator("#pv-cell-hand")
    page.wait_for_function(
        "() => !document.getElementById('pv-cell-hand').hidden", timeout=WAIT_MS)
    said = note.inner_text()
    assert "left-handed" in said, said
    # It must carry the way OUT, not just the diagnosis.
    assert "other order" in said or "swap" in said.lower(), said

    # Flip that axis back and the note goes — the sign is read live.
    page.locator("#pv-cell-grid input").nth(8).fill("1")
    page.wait_for_function(
        "() => document.getElementById('pv-cell-hand').hidden", timeout=WAIT_MS)


# --------------------------------------------------------------------- #
#  An edit lands, and the page notices (molview § 1.1, § 8.4)           #
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
        timeout=WAIT_MS)
    assert "#1" in page.locator("#add-anchor-readout").inner_text()


def test_the_edit_survives_a_page_reload(
        page, flask_server, labelled_xyz):
    """molview § 11.2a, *Coming back to a session*: "the sequence outlives the
    page" — a fresh viewer ADOPTS the draft already in storage instead of
    re-reading the file, and takes its place on the sequence with it.

    Two things come back here that a reopened page cannot infer: the atoms as
    EDITED — a page that re-read the FILE shows 3, not 2 — and the position on
    the sequence, which a fresh viewer starting at 0 cannot work out.  (The
    delete laid its own point, § 1.1, so on this walk no work sits above #1:
    the draft and point #1 hold the same atoms, and the unsaved flag is down.)
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    before = _atom_count(page)

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=WAIT_MS)
    page.locator("#delete-apply").click()
    page.wait_for_function(
        f"() => /of {before - 1} selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)
    # The edit is ON the sequence now, so this is where it put us -- and it is
    # the fact the reopened page below has to bring back.
    page.wait_for_function(
        "() => /saved #1/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=WAIT_MS)

    page.reload()
    page.wait_for_selector(_CARD, timeout=WAIT_MS)
    page.wait_for_selector(f"{_CARD} canvas", timeout=WAIT_MS)
    page.wait_for_function(
        "() => /\\d+ of [1-9]\\d* selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)

    assert _atom_count(page) == before - 1, (
        f"the reopened page shows {_atom_count(page)} atoms, not the {before - 1} "
        f"the edit left -- the draft was not adopted, or the FILE was re-read "
        f"and the edit thrown away")
    assert "#1" in page.locator("#timeline-status").inner_text(), (
        "the reopened page does not know where on the sequence it is: the "
        "position is one of the three fields that must travel WITH the draft, "
        "because a fresh viewer starts at 0 and cannot work it out")


def test_an_edit_changes_the_structure_and_records_itself(
        page, flask_server, labelled_xyz):
    """Delete removes the atom, the count follows, and the edit puts ITSELF on
    the timeline.

    The point is laid down INSIDE the viewer's gate when the change lands
    (molview § 1.1, § 11.2: *every edit lays down a point of its own*) — not
    by the page afterwards — so the sequence advancing proves the
    edit reached the model rather than only the screen.

    The badge answers the same for a recorded edit and a refused one, so the
    position is what tells them apart.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => /saved #0/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=WAIT_MS)

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=WAIT_MS)
    page.locator("#delete-apply").click()

    page.wait_for_function(
        "() => /of 2 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)
    page.wait_for_function(
        "() => /saved #1/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=WAIT_MS)
    # THE OP LINE SAYS WHAT IS LEFT, off the structure the door handed back:
    # the measured defect U1-4 (closed 2.67.1,
    # `docs/archive/2026-10-31-finish-line-review.md`) -- one atom deleted,
    # and the line said "Deleted: 18 atoms.", the count it held after.
    assert ": 2 left." in page.locator("#edit-status").inner_text(), \
        "the op line reports the count off the structure the door handed back"


# --------------------------------------------------------------------- #
#  The edits that place atoms (molview § 11.1)                          #
#                                                                       #
#  THE SILENT FAILURE: a control renamed in the page, and `viewer.js`    #
#  throwing on it the first time a person uses it.  Each is pressed      #
#  here, and what the door answered read back.                           #
# --------------------------------------------------------------------- #

def _slide(page, slider: str, value) -> None:
    """A range control set to ``value`` as the browser sets one a person
    moves (Playwright's ``fill`` on a range: the value, then the `input` and
    `change` a drag ends with)."""
    page.locator(f"#{slider}").fill(str(value))


def _ready(page, button: str) -> None:
    page.wait_for_function(
        "(id) => !document.getElementById(id).disabled",
        arg=button, timeout=WAIT_MS)


def _applied(page, op: str, button: str) -> dict:
    """``button`` pressed: the structure its door answered the page."""
    with page.expect_response(
            lambda r: r.url.endswith(f"/api/modify/{op}"),
            timeout=WAIT_MS) as answered:
        page.locator(f"#{button}").click()
    answer = answered.value.json()
    assert answer["ok"] is True, answer
    return answer["structure"]


def _said(page, words: str) -> None:
    """The op line says ``words``."""
    page.wait_for_function(
        "(w) => document.getElementById('edit-status').textContent"
        "  .includes(w)", arg=words, timeout=WAIT_MS)


def _at_point(page, point: int) -> None:
    """The timeline line says the tab stands on saved point ``point``
    (molview § 11.2a: the tab shows where Save state and Retract stand)."""
    page.wait_for_function(
        "(n) => new RegExp('saved #' + n + '(?!\\d)')"
        "  .test(document.getElementById('timeline-status').textContent)",
        arg=point, timeout=WAIT_MS)


def test_add_atom_lands_where_its_sliders_say_from_the_picked_atom(
        page, flask_server, labelled_xyz):
    """The new atom is the element typed, at the offset the sliders show from
    the one atom selected (molview § 11.1), and the readouts say so before
    anything is sent."""
    _open(page, flask_server)
    _load(page, labelled_xyz)
    _pick_atom(page, 2)                       # H at (0.957, 0, 0)
    page.wait_for_function(
        "() => document.getElementById('add-anchor-readout').textContent"
        "  .includes('#2 H')", timeout=WAIT_MS)

    page.locator("#add-element").fill("F")
    _slide(page, "add-dx", 0.5)
    _slide(page, "add-dy", 0)
    _slide(page, "add-dz", 1.5)
    assert page.locator("#add-dx-val").inner_text() == "0.50"
    assert page.locator("#add-dy-val").inner_text() == "0.00"
    assert page.locator("#add-dz-val").inner_text() == "1.50"
    assert page.locator("#add-distance").inner_text().startswith("1.58")

    _ready(page, "add-apply")
    made = _applied(page, "add_atom", "add-apply")
    assert made["elements"] == ["O", "H", "H", "F"]
    assert made["positions"][3] == pytest.approx([1.457, 0.0, 1.5], abs=1e-3)
    _said(page, "4 now")
    assert _atom_count(page) == 4


def test_orient_lays_the_two_picked_atoms_at_the_tilt_its_controls_say(
        page, flask_server, labelled_xyz):
    """First → second at the tilt shown from the axis chosen, the first at the
    origin when Center says so (`modify.orient_along_axis`; molview § 11.1:
    the anchors are the ruler's, in click order).  O→H is 0.957 Å, tilted 30°
    from y in the yz-plane."""
    import math
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-transform").click()
    _pick_atom(page, 1)                       # O at the origin
    _pick_atom(page, 2)                       # H at (0.957, 0, 0)
    _ready(page, "orient-apply")
    anchors = page.locator("#orient-anchor-readout").inner_text()
    assert "#1 O" in anchors and "#2 H" in anchors, anchors

    page.locator("input[name='orient-axis'][value='y']").check()
    _slide(page, "orient-angle", 30)
    assert page.locator("#orient-angle-val").inner_text() == "30°"
    page.locator("#orient-center").select_option("first")

    made = _applied(page, "orient", "orient-apply")
    tilt = math.radians(30)
    assert made["positions"][0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert made["positions"][1] == pytest.approx(
        [0.0, 0.957 * math.cos(tilt), 0.957 * math.sin(tilt)], abs=1e-3)
    _said(page, "Oriented (y, tilt 30°)")


def test_rotate_turns_the_structure_by_the_angle_its_slider_says(
        page, flask_server, labelled_xyz):
    """About the axis chosen, around the centroid by default (molview § 11.1):
    +90° about z takes (x, y) to (-y, x), measured from the centroid."""
    import numpy as np
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.locator("#optab-btn-transform").click()
    _ready(page, "rotate-apply")

    page.locator("input[name='rotate-axis'][value='z']").check()
    _slide(page, "rotate-angle", 90)
    assert page.locator("#rotate-angle-val").inner_text() == "90°"

    made = _applied(page, "rotate", "rotate-apply")
    from molbuilder.workingcopy_structure import StructureCodec
    drawn = StructureCodec().load(labelled_xyz).positions
    centroid = drawn.mean(axis=0)
    away = drawn - centroid
    want = centroid + np.c_[-away[:, 1], away[:, 0], away[:, 2]]
    assert np.allclose(made["positions"], want, atol=1e-3), made["positions"]
    _said(page, "Rotated 90° around z")


# --------------------------------------------------------------------- #
#  The state timeline (molview § 1.1, § 11.2, § 11.2a)                  #
# --------------------------------------------------------------------- #

def test_retract_puts_the_atom_back_with_no_save_state_pressed(
        page, flask_server, labelled_xyz):
    """Retract steps back exactly one EDIT, and nobody had to press anything to
    make that true.

    *"all operation should automatically call state save timeline api, such
    that the user always can retract back"* (user, 2026-09-07; molview § 1.1).
    The delete lays down its own point, so one Retract -- § 11.2's `load(-1)`
    -- restores the atom.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => !document.getElementById('save-state').disabled",
        timeout=WAIT_MS)
    assert page.locator("#undo-op").is_disabled(), \
        "nothing has happened yet, so there is nothing to retract"

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=WAIT_MS)
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => /of 2 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)
    # THE EDIT ITSELF PUT THIS HERE.  No Save state was pressed.
    page.wait_for_function(
        "() => /saved #1/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=WAIT_MS)

    page.locator("#undo-op").click()
    page.wait_for_function(
        "() => /of 3 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)
    assert "#0" in page.locator("#timeline-status").inner_text()


def test_retract_says_so_when_the_point_it_wanted_is_gone(
        page, flask_server, labelled_xyz):
    """A retraction that could not happen must not be reported as one.

    A saved sequence is bounded at its last 30 saves (`workspace.md` § 9.1), so
    the oldest points are deleted as new ones arrive.  `load` answers that
    honestly -- it returns null and leaves `position` alone -- but the caller
    used to throw the answer away and print ``Retracted to state #N`` with "ok"
    styling, N being the position it was ALREADY at.  The user was told a
    retraction happened that did not.

    Reached as a person reaches it: an edit lays down #1 and Save state #2 to
    #30 -- 31 points, so the server's window drops #0 when #30 lands
    (`workspace_storage._STATE_WINDOW`) -- then Retract, back to #1, and once
    more, for the #0 that is gone (measured 2026-11-07: the test takes 11 s,
    8 s more than one that retracts once).
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => !document.getElementById('save-state').disabled",
        timeout=WAIT_MS)

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=WAIT_MS)
    page.locator("#delete-apply").click()
    # The EDIT lays down #1 (§ 11.2); Retract wants #0.
    _at_point(page, 1)

    for point in range(2, 30):
        page.locator("#save-state").click()
        _at_point(page, point)
    # The 31st point, #30: its write is the one the window drops #0 on.
    with page.expect_response(
            lambda r: r.url.endswith("/api/workspace-storage/write")
            and (r.request.post_data_json or {}).get("state_index") == 30,
            timeout=WAIT_MS):
        page.locator("#save-state").click()
    _at_point(page, 30)
    for point in range(29, 0, -1):
        page.locator("#undo-op").click()
        _at_point(page, point)

    # Retract is offered -- we are at #1, so the button is enabled and the user
    # has every reason to expect it to work.
    assert not page.locator("#undo-op").is_disabled()
    page.locator("#undo-op").click()

    status = page.locator("#edit-status")
    page.wait_for_function(
        "() => /Nothing changed/.test("
        "  document.getElementById('edit-status').textContent)",
        timeout=WAIT_MS)
    text = status.inner_text()
    assert "Retracted to state" not in text, \
        f"retract claimed a move that did not happen: {text!r}"
    # The shared status vocabulary (ui-contract.md § 5): `.status.warn`.
    assert {"status", "warn"} <= set((status.get_attribute("class") or "").split()), \
        "a refusal styled as success reads as success"
    # And it really did not move.
    assert "#1" in page.locator("#timeline-status").inner_text()


def test_the_timeline_indicator_says_where_you_are(
        page, flask_server, labelled_xyz):
    """The point Retract would restore is on screen.

    molview § 11.2: the position on the sequence, and the point `load(-1)` --
    Retract -- lands on; § 11.2a: Save state and Retract are the host's
    controls, so the tab shows where they stand.  A push-only timeline with no
    indicator is a promise the user cannot check:
    "Save state" only means something if you can see that you have not.
    """
    _open(page, flask_server)
    _load(page, labelled_xyz)
    page.wait_for_function(
        "() => /saved #0/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=WAIT_MS)

    _pick_atom(page, 1)
    page.wait_for_function(
        "() => !document.getElementById('delete-apply').disabled",
        timeout=WAIT_MS)
    page.locator("#delete-apply").click()
    # THE EDIT IS ON THE SEQUENCE, so the indicator says where that put you and
    # which point Retract goes back to.
    page.wait_for_function(
        "() => /saved #1/.test("
        "  document.getElementById('timeline-status').textContent)",
        timeout=WAIT_MS)
    assert "#0" in page.locator("#timeline-status").inner_text(), \
        "the indicator names the point Retract will return to"


# --------------------------------------------------------------------- #
#  Saving to the project (tabs.md § 6; molview § 11.3)                  #
# --------------------------------------------------------------------- #

def test_saving_to_the_project_writes_the_pair_and_remembers_where(
        page, flask_server, labelled_xyz, isolated_projects_root):
    """A save writes BOTH files, and the page records the target.

    The pair is the point: coordinates alone lose the labels, and a browser must
    never author the sidecar schema — the server writes both through
    ``StructureCodec.write`` (§ 11.3).  Where they went is the PAGE's note, not
    the viewer's (§ 6.7), and nothing was setting it: ``markSavedTo`` had no
    caller, so the readout still offered "Save as…" straight after a save.
    """
    from molbuilder.projects import populate_project_skeleton
    # A PROJECT TO SAVE INTO, made as New project makes one: a file is placed
    # in a folder below the projects root, never in the root itself
    # (`projects.md` § 4.1).
    project = isolated_projects_root.resolve() / "P"
    populate_project_skeleton(project, project_name="P")
    where = project / "structure"
    _open(page, flask_server)
    _load(page, labelled_xyz)

    # TWO QUESTIONS, in this order: WHERE, then what to call it
    # (`tabs.md` § 6 -- the door owns the destination).
    page.locator("#save-to-source-btn").click()

    # 1. WHERE.  The picker's Choose stays disabled until a folder is
    #    selected -- opening the project and clicking its structure folder's
    #    row is the selection, which is what a person does.
    dialog = page.locator("dialog.molbuilder-tree-picker-dialog")
    # The projects root is the tree, never an answer: a file is placed in a
    # folder below it, so its row cannot be chosen (`projects.md` § 4.1).
    root_row = dialog.locator(
        f'li.tp-node[data-path="{project.parent}"] > .tp-row')
    root_row.wait_for(state="visible", timeout=WAIT_MS)
    assert "tp-row--inert" in (root_row.get_attribute("class") or "")
    dialog.locator(f'li.tp-node[data-path="{project}"]'
                   " > .tp-row > .tp-twisty").click()
    row = dialog.locator(f'li.tp-node[data-path="{where}"] > .tp-row')
    row.wait_for(state="visible", timeout=WAIT_MS)
    row.click()
    page.locator("dialog [data-action='confirm']").first.click()

    # 2. WHAT TO CALL IT.
    name = page.locator("dialog input[data-role='name']")
    name.wait_for(state="visible", timeout=WAIT_MS)
    name.fill("saved_by_test")
    page.locator("dialog [data-action='confirm']").last.click()

    page.wait_for_function(
        "() => /Saved/.test(document.getElementById('save-status').textContent)",
        timeout=WAIT_MS)

    assert (where / "saved_by_test.xyz").exists()
    assert (where / "saved_by_test.molstruct.json").exists(), \
        "the sidecar carries the labels; a lone .xyz silently loses them"
    # The saved pair opens through the one read door, its labels with it.
    from molbuilder.workingcopy_structure import StructureCodec
    assert StructureCodec().load(where / "saved_by_test.xyz").regions == {
        "SOLVENT": [0, 1, 2]}
    assert "saved_by_test.xyz" in page.locator("#save-readout").inner_text()


def test_the_page_remembers_which_file_it_is_showing_across_a_reload(
        page, flask_server, labelled_xyz):
    """The page's own note survives, so Load does not offer to discard your work.

    workspace.md § 4: a page may have several savers, kept apart by their tags,
    and it names this one — *"the Modify tab has a viewer holding a molecule AND
    its own panel state"*.  The viewer saves under `modify`; the page saves under
    `modify:panel`.  Two tags, two slots.

    THE BUG THIS CLOSES.  `loadedFrom` used to live in a closure variable, set
    only by the path that READS A FILE.  When a reload was served by the
    viewer's restore instead — now the normal case — it was empty while a
    structure was plainly on the canvas, so the readout fell back to
    "Picked:" after a reload that had lost nothing.

    The reload waits for the note naming the file to be saved: a save is sent
    without waiting (`workspace.md` § 3), one after another, so a reload
    before it lands is a reload before the page had kept anything to come
    back to.
    """
    _open(page, flask_server)
    with page.expect_response(
            lambda r: (r.url.endswith("/api/workspace-storage/write")
                       and ((r.request.post_data_json or {}).get("data")
                            or {}).get("loadedFrom")),
            timeout=WAIT_MS):
        _load(page, labelled_xyz)
    page.wait_for_function(
        "() => /^Loaded:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=WAIT_MS)

    # Come back to the tab.
    page.goto(f"{flask_server}/molbuilder")
    page.wait_for_selector(_CARD, timeout=WAIT_MS)
    page.wait_for_function(
        "() => /\\d+ of [1-9]\\d* selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)

    page.wait_for_function(
        "() => /^Loaded:/.test("
        "  document.getElementById('load-candidate-readout').textContent)",
        timeout=WAIT_MS)


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
        timeout=WAIT_MS)

    # Now generate.  It ADDS to what is open (user, 2026-09-07) -- 3 + 9 = 12 --
    # and the note goes to null either way: the canvas is no longer that file.
    page.locator(".modify-init-tab", has_text=re.compile("SMILES", re.I)
                 ).click()
    page.fill("#smiles-input", "CCO")
    page.locator("#smiles-generate-btn").click()
    page.wait_for_function(
        "() => /of 12 selected/.test("
        "  document.querySelector('.molviewer-selection-count')?.textContent || '')",
        timeout=WAIT_MS)

    readout = page.locator("#load-candidate-readout").inner_text()
    assert not readout.startswith("Loaded:"), (
        f"a generated structure is claiming to be the file that was loaded "
        f"before it: {readout!r}"
    )


# --------------------------------------------------------------------- #
#  A checkbox row is laid out inline                                    #
#                                                                       #
#  Contract: docs/web/ui-contract.md § 1 (a page sheet arranges, it does #
#  not re-decide form-control layout).                                  #
#                                                                       #
#  THE DEFECT THIS EXISTS FOR.  The op-block's label rule was a blanket  #
#  `label` descendant selector.  It is there for m/n/layers -- label     #
#  ABOVE a number input -- and it also caught the checkbox row, where    #
#  `flex-direction: column` put the box above its own text and           #
#  `align-items: stretch` widened the <input> to the full row (measured  #
#  384.5px), painting a tick centred in empty space.                     #
#                                                                       #
#  The bug is in the cascade, so this test measures the painted result. #
# --------------------------------------------------------------------- #

def test_a_checkbox_sits_beside_its_own_text(page, flask_server):
    """The box is before the words and on the same line, at box width.

    Three questions, because the stacked treatment got all three wrong: is
    the label BESIDE the box (not under it), do the two share a row, and is
    the box still a box (not stretched across the panel)?
    """
    _open(page, flask_server)
    page.locator("#optab-btn-slab").click()
    page.wait_for_selector("#optab-panel-slab", state="visible",
                           timeout=WAIT_MS)

    box = page.locator("#slab-orthogonal").bounding_box()
    txt = page.locator('label[for="slab-orthogonal"]').bounding_box()
    row = page.locator("#slab-orthogonal").locator("xpath=..").bounding_box()
    assert box and txt and row, "the slab panel's checkbox row did not render"

    assert txt["x"] >= box["x"] + box["width"] - 1, (
        f"the label starts at x={txt['x']:.1f}, not clear of the box's right "
        f"edge at {box['x'] + box['width']:.1f} -- the box is above its own "
        f"text, not before it")

    box_mid = box["y"] + box["height"] / 2
    txt_mid = txt["y"] + txt["height"] / 2
    assert abs(box_mid - txt_mid) <= 4, (
        f"box centre y={box_mid:.1f} and text centre y={txt_mid:.1f} are "
        f"{abs(box_mid - txt_mid):.1f}px apart -- they are not on one row")

    assert box["width"] <= row["width"] / 2, (
        f"the checkbox is {box['width']:.1f}px wide inside a "
        f"{row['width']:.1f}px row -- it has been stretched by the row's "
        f"align-items, which paints the tick centred in empty space")


# --------------------------------------------------------------------- #
#  "From a bulk run…" -- the measurement's notes are findings           #
#                                                                       #
#  Contract: docs/science/validation.md § 4.1 R2a (a notice is a        #
#  finding, drawn by the one renderer).                                 #
# --------------------------------------------------------------------- #

@pytest.fixture
def stretched_gold(isolated_projects_root):
    """A perfect fcc gold crystal stretched to a = 4.35 Å -- 7% from the
    experimental constant and 5% from PBE's, which the route says at two
    severities."""
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    a = 4.35
    base = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]) * a
    pos = np.vstack([base + np.array([i, j, k]) * a
                     for i in range(2) for j in range(2) for k in range(2)])
    path = isolated_projects_root.resolve() / "Au-stretched.xyz"
    StructureCodec().write(
        Structure(elements=["Au"] * len(pos), positions=pos,
                  cell=np.diag([2 * a] * 3)), path)
    return path


def test_a_measured_lattices_notes_are_rows_each_at_its_own_severity(
        page, flask_server, stretched_gold):
    """Every note the route answers with is a row of its own, at its own
    severity and in the route's order; the status line says only what was
    measured; and a value typed into the box takes the notes away, since
    they describe the file the value came from.

    The picker is asked for while the page does not yet know its projects
    root: the sidebar's one request for it (`GET /api/files/roots`) is held
    at the browser until *From a bulk run…* is clicked, then let go, the
    server's own answer -- and the picker opens once the root lands
    (`web/projects.md` § 5).

    MUTATION THIS MUST FAIL AGAINST: the notes joined into the status line
    in the worst one's tone (`modify/slab-panel.js` before 2026-09-27); the
    picker answering null, as a cancel, when it is asked before the root is
    known (`lib/tree-picker.js`'s `pickPath` returning while
    `getProjectsRoot()` is empty).
    """
    held = []
    page.route("**/api/files/roots", lambda route: held.append(route))
    _open(page, flask_server)
    page.locator("#optab-btn-slab").click()
    page.wait_for_selector("#optab-panel-slab", state="visible",
                           timeout=WAIT_MS)
    # The panel's menu is in: the element the measurement names is Au.
    page.wait_for_function(
        "() => document.getElementById('slab-element').value === 'Au'",
        timeout=WAIT_MS)

    page.locator("#slab-pick-run").click()
    # THE ORDER IS HELD, not timed: the click loads the picker's module and
    # then asks it, so the root is let through only once that module has
    # loaded here too -- by then the click, first to ask for the module, has
    # asked the picker.
    page.evaluate("() => import('/static/lib/tree-picker.js')")
    assert len(held) == 1, f"the sidebar asked for its root {len(held)} times"
    held.pop().continue_()
    page.unroute("**/api/files/roots")
    row = page.locator(
        f"dialog li.tp-node[data-path$='/{stretched_gold.name}'] > .tp-row")
    row.wait_for(state="visible", timeout=WAIT_MS)
    row.click()
    with page.expect_response("**/api/modify/lattice-from-run") as answer:
        page.locator("dialog [data-action='confirm']").click()
    said = answer.value.json()
    assert said["ok"] is True, said
    notes = said["notes"]
    assert len({n["severity"] for n in notes}) >= 2, (
        f"the crystal should draw notes at two severities; it drew {notes}")

    findings = page.locator("#slab-lattice-findings")
    findings.wait_for(state="visible", timeout=WAIT_MS)
    rows = findings.locator("li.issue-item").all()
    assert [(r.get_attribute("data-severity"),
             r.locator(".issue-msg").inner_text()) for r in rows] == [
        (n["severity"], n["message"]) for n in notes]

    status = page.locator("#app-notifications .app-notification",
                          has_text=stretched_gold.name)
    assert "app-notification--info" in status.get_attribute("class")
    line = status.locator(".app-notification__msg").inner_text()
    assert f"a = {said['a']:.4f} Å" in line, line
    assert not [n for n in notes if n["message"] in line], line

    page.locator("#slab-a").fill("4.2")
    findings.wait_for(state="hidden", timeout=WAIT_MS)
    assert findings.locator("li.issue-item").count() == 0
