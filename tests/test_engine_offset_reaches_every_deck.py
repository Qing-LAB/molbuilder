"""Every engine is handed the design coordinates plus the engine offset, and
every deck says so -- `model/structure-periodicity.md` § 6.0.

Through the road: `molbuilder describe` of the structure's pair, then `jobset
prep task`, then the deck on disk.  Per engine:

* the invariant is coordinates + offset: the deck writes the design
  coordinates plus ``cell.engine_offset`` of the design structure -- one rigid
  translation, frozen atoms included -- and what it writes carries no offset
  of its own (zero at the hand-off: the correction applied);
* every atom is inside the cell, with equal margins along each lattice vector;
* the deck's ENGINE-OFFSET record states that same offset and that cell.
"""
from __future__ import annotations

from support.road import stage_file, water_with_an_assigned_origin

import ast
import re

import numpy as np
import pytest

from molbuilder import cell as cellmod
from molbuilder.deck_record import extract_engine_offset
from molbuilder.structure import Structure

#: A skewed cell, as the 2026-09-25 junction's was.
_HEX = np.array([[8.651, 0.0, 0.0], [4.326, 7.492, 0.0], [0.0, 0.0, 14.0]])


def _slab():
    """A periodic structure on the skewed cell, authored AROUND the world
    origin -- how the junction builder authors -- so the offset that places
    it is far from zero.  Two atoms are frozen.

    Its widest gap along c is INSIDE it (7 Å between the S layers, against a
    2 Å seam), as the junction's Au-S contacts were wider than its seam: a
    rule that cut the cell at the widest gap would split it, and the
    invariant below would say so."""
    frac_ab = np.array([[0.0, 0.0], [0.95, 0.3], [0.05, 0.95], [0.5, 0.5]])
    xy = frac_ab @ _HEX[:2, :2] - np.array([5.0, 3.0])
    z = np.array([-6.0, 6.0, -3.5, 3.5])
    return Structure(elements=["Au", "Au", "S", "S"],
                     positions=np.column_stack([xy, z]), cell=_HEX.copy(),
                     axis_kind=("periodic", "periodic", "periodic"),
                     frozen_atoms=[0, 1])


def _water():
    """A molecule that states no cell: its box is derived from its vacuum."""
    return Structure(elements=["O", "H", "H"],
                     positions=np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0],
                                         [-0.757, 0.586, 0.0]]),
                     vacuum=(5.0, 5.0, 5.0))


def _design(dest):
    """The ORIGINAL file -- the pair the description names, as prep reads it.
    The invariant is about files: its coordinates plus its offset."""
    from molbuilder.task import of as task_of
    from molbuilder.workingcopy_structure import StructureCodec
    return StructureCodec().load(dest / task_of(dest).structure.source)


def _assert_placed(design, written, text):
    """The invariant is coordinates + offset (user, 2026-09-25): the design
    coordinates plus their offset ARE what the deck writes, the deck's record
    states that correction, and what the deck writes carries no offset of its
    own -- zero at the hand-off, the correction applied."""
    offset = cellmod.engine_offset(design)
    assert np.linalg.norm(offset) > 1.0, "the fixture must be one the rule moves"
    np.testing.assert_allclose(written, design.positions + offset, atol=1e-7)
    record = extract_engine_offset(text)
    assert record is not None, "the deck carries no ENGINE-OFFSET record"
    np.testing.assert_allclose(record["applied_offset"], offset, atol=1e-7)
    # And the viewer draws what the deck says (plan § 5q.8 D4): the structure's
    # own view puts the box at minus the offset this deck applied.
    np.testing.assert_allclose(design.to_wire()["resolved_box"]["box_corner"],
                               -np.asarray(record["applied_offset"]), atol=1e-7)
    cell = np.asarray(record["cell"], dtype=float)
    np.testing.assert_allclose(cell, design.resolve_cell(), atol=1e-7)
    frac = np.linalg.solve(cell.T, written.T).T
    near, far = frac.min(axis=0), 1.0 - frac.max(axis=0)
    assert np.all(near > 0), near
    np.testing.assert_allclose(near, far, atol=1e-7)


def _rows(block):
    return [l.split() for l in block.strip().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def test_a_siesta_deck_writes_the_design_coordinates_plus_the_offset(
        prepared):
    """A periodic slab on a skewed cell, authored around the origin, reaches
    the SIESTA deck as its design coordinates plus the engine offset -- one
    rigid translation, frozen atoms included, centred in the cell -- and the
    run's trajectory starts where the deck does.

    CONTRACT: `model/structure-periodicity.md` § 6.0 (the rule; the
    invariant; frozen atoms move with the rest).
    """
    dest, stage, text = prepared(_slab(),
                                                "siesta")
    block = re.search(r"%block AtomicCoordinatesAndAtomicSpecies\n(.*?)%endblock",
                      text, re.S).group(1)
    written = np.array([[float(v) for v in r[:3]] for r in _rows(block)])
    design = _design(dest)
    np.testing.assert_allclose(design.resolve_cell(), _HEX, atol=1e-9)
    _assert_placed(design, written, text)
    # The run's trajectory starts where the deck does: step 0 used to be the
    # DESIGN coordinates while the deck beside it was placed -- a jump of the
    # whole offset between step 0 and step 1.
    # prep seeds it in the attempt it opens (`01_<stage>/run-0/`)
    log = stage_file(dest, stage, ".molwatch.log").read_text()
    step0 = log.split("coordinates (Ang):", 1)[1].split("energy", 1)[0]
    np.testing.assert_allclose(
        np.array([[float(v) for v in r[1:4]] for r in _rows(step0)]),
        written, atol=1e-6)


@pytest.mark.parametrize("calculation", ["optimization", "vibration"])
def test_a_pyscf_deck_writes_the_design_coordinates_plus_the_offset(
        prepared, calculation):
    """A molecule with no cell reaches the PySCF deck -- the optimisation
    deck and the vibration deck, two spec builders -- as its design
    coordinates plus the engine offset, centred in the box its vacuum sizes.

    CONTRACT: `model/structure-periodicity.md` § 6.0 (every engine is handed
    the design coordinates plus the offset).
    """
    dest, _stage, text = prepared(_water(), "pyscf", calculation=calculation,
        template={"already_relaxed": True} if calculation == "vibration"
        else None)
    written = _written("pyscf", text)
    design = _design(dest)
    assert tuple(design.vacuum) == (5.0, 5.0, 5.0), "the vacuum reached prep"
    _assert_placed(design, written, text)


def _written(engine, text):
    """The coordinates a deck writes, in Å."""
    if engine == "siesta":
        block = re.search(r"%block AtomicCoordinatesAndAtomicSpecies\n(.*?)%endblock",
                          text, re.S).group(1)
        return np.array([[float(v) for v in r[:3]] for r in _rows(block)])
    m = re.search(r"_atom_block = '''\n(.*?)'''", text, re.S)
    if m:                                      # the optimisation deck
        return np.array([[float(v) for v in r[1:4]] for r in _rows(m.group(1))])
    # the vibration deck: ATOMS = [(element, x, y, z), ...]
    block = re.search(r"^ATOMS = (\[.*?^\])", text, re.S | re.M).group(1)
    return np.array([row[1:4] for row in ast.literal_eval(block)], dtype=float)


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_an_assigned_origin_places_the_deck_at_design_minus_it(
        prepared, engine):
    """T1's assigned-origin clause (plan § 5q.4): on a typed cell with an origin
    the person assigned at P, the deck writes the design coordinates minus P --
    not the rule's centring -- and its record says the offset was stated.

    CONTRACT: `model/structure-periodicity.md` § 6.0, *A stated offset* (D1).
    """
    P = np.array([-3.0, -3.0, -3.0])
    typed, assigned = water_with_an_assigned_origin(P)
    assert not np.allclose(-P, cellmod.engine_offset(typed)), (
        "the fixture must be one the rule would place elsewhere")
    dest, _stage, text = prepared(assigned, engine)
    design = _design(dest)
    np.testing.assert_allclose(design.engine_offset, -P)    # the pair carried it
    np.testing.assert_allclose(_written(engine, text), design.positions - P,
                               atol=1e-7)
    record = extract_engine_offset(text)
    assert record["stated"] is True, record
    np.testing.assert_allclose(record["applied_offset"], -P, atol=1e-7)
    # The kinds as the structure had them (D5) -- what the Results tab reads.
    assert record["axis_kind"] == list(design.axis_kind), record
    np.testing.assert_allclose(design.to_wire()["resolved_box"]["box_corner"],
                               P, atol=1e-7)          # the viewer's box is there
    if engine == "pyscf":
        # The pair the run saves beside its geometry states the engine's
        # origin: the coordinates it lands beside are the engine's (§ 6.0,
        # every save of a run's output), and the design's -P would place them
        # a second time at the next prep.
        line = re.search(r"^_MB_SIDECAR = (.*)$", text, re.M).group(1)
        assert ast.literal_eval(line)["engine_offset"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_an_origin_that_leaves_an_atom_outside_is_refused_naming_it(
        prepared, engine):
    """The same box with its origin assigned at (-0.5, -1, -1): one hydrogen,
    atom 2, would land at x = -0.257 -- outside along a, every other atom
    inside and every other axis clear.  The deck is refused, and prep names
    that atom and that axis, in a sentence rather than a traceback.

    CONTRACT: `model/structure-periodicity.md` § 6.0, check 3 (an origin that
    leaves an atom outside is refused naming it); `workflow.md` § 9
    (a gate refuses with the reason, never a stack trace).
    """
    _typed, assigned = water_with_an_assigned_origin([-0.5, -1.0, -1.0])
    _dest, _stage, said = prepared(assigned, engine,
                                                  refused=True)
    assert "outside the cell" in said, said
    assert re.search(r"a \(isolated\): atom index 2(?!\d)", said), said
    assert "b (" not in said and "c (" not in said, said
    # Under the hand-off's own id: the Cell page warned of the same atom as
    # `cell.atoms_outside`, and one id carries one severity (plan § 5q D11).
    assert "[deck.atoms_outside]" in said, said


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_a_box_nothing_can_be_placed_in_is_refused_in_a_sentence(
        prepared, engine):
    """Flat water with no vacuum across its plane has a box with no volume:
    prep refuses it with the checker's own finding.

    GOAL: the PySCF spec builders place the atoms before the settings gate
    runs, and placing them in a singular box raised a bare ValueError that
    reached the person as a traceback (found by review, 2026-09-25).
    CONTRACT: `workflow.md` § 9 (a gate refuses with its reason);
    `model/structure-periodicity.md` § 6.1a (`cell.no_volume`, an error).
    """
    flat = Structure(elements=["O", "H", "H"], positions=_water().positions,
                     vacuum=(5.0, 5.0, 0.0))
    _dest, _stage, said = prepared(flat,
                                                  engine, refused=True)
    assert "[cell.no_volume]" in said, said
    assert "Traceback" not in said, said


def test_an_atom_past_a_periodic_face_is_a_warning_in_the_report(
        prepared):
    """Along a periodic vector an atom past a face is legal -- an image the
    engine wraps -- and said.

    GOAL: with the correction applied at every deck, an atom still outside
    the cell after it is something the person should see, not a silence
    (user, 2026-09-25: "we should now give warning/error when atoms are
    outside boundary for all cases").  CONTRACT: `model/structure-periodicity.md`
    § 6.0, checks 1 and 3: warned along a periodic vector, refused along any
    other.
    """
    # 1.2 cells long along a, so no translation fits it -- and the atom past
    # the face sits 3 Å off the others' line, so its image meets no atom.
    wide = Structure(elements=["C", "C", "C"],
                     positions=np.array([[0.0, 1.0, 1.0], [3.0, 1.0, 1.0],
                                         [6.0, 4.0, 1.0]]),
                     cell=np.diag([5.0, 8.0, 8.0]),
                     axis_kind=("periodic", "periodic", "periodic"))
    dest, stage, _text = prepared(wide,
                                                 "siesta")
    report = stage_file(dest, stage, ".validation.txt").read_text()
    assert "[cell.beyond_periodic_face]" in report, report
    assert "along a, which is periodic" in report, report


def test_a_crystal_that_fills_its_cell_is_not_called_tight(
        prepared):
    """A periodic structure fills its cell by construction, and its report
    does not call the cell tight.

    GOAL: every rung of the 2026-09-25 transport ladder was told "cell is
    suspiciously tight; expect image-image interactions" -- the bulk gold lead
    at 1.45, the junction at 1.03 -- because the volume ratio was taken over
    all three axes.  CONTRACT: `model/structure-periodicity.md` § 2 (images
    across a periodic or transport axis are intended; only an isolated axis is
    measured).
    """
    dest, stage, _text = prepared(_slab(), "siesta")
    report = stage_file(dest, stage, ".validation.txt").read_text()
    assert "[cell.volume]" not in report, report


def _engine_frame():
    """Water inside a typed 10 Å box, OFF its centre, stating the engine's
    origin -- coordinates read back from a run (`runs.Declared.frame`): every
    atom already inside, so the engine's frame is placeable as it stands, and
    the rule would centre it elsewhere."""
    return Structure(elements=["O", "H", "H"],
                     positions=_water().positions + np.array([3.0, 3.0, 3.0]),
                     cell=np.eye(3) * 10.0, axis_kind=("isolated",) * 3,
                     engine_offset=[0.0, 0.0, 0.0], offset_by="engine")


@pytest.mark.parametrize("calculation, said, kept", [
    # an engine's origin is not carried into a new calculation ...
    ("optimization", {}, False),
    ("vibration", {}, False),
    # ... save a SIESTA vibration's input stated relaxed: its grid minimum --
    # by the template, or by the force-constant stage's own value
    ("vibration", {"template": {"already_relaxed": True}}, True),
    ("vibration", {"overrides": {"already_relaxed": True}}, True),
])
def test_an_engines_origin_enters_a_new_calculation_by_the_rule(
        prepared, calculation, said, kept):
    """A structure whose stated origin is the engine's enters a new
    calculation stating none, so the deck is placed by the rule -- unless the
    calculation keeps the engine's frame: a vibration whose input is stated
    relaxed, as the stage being prepared resolves it.  Read from the deck's
    own ENGINE-OFFSET record.

    THE SILENT FAILURE: the engine's origin carried over keeps whatever
    placement the run had -- a layer flush against a face reached a
    transport prep through two hand-offs on 2026-10-11 -- while dropping it
    on a relaxed vibration moves the reference geometry off its force
    minimum; neither says so.

    CONTRACT: `model/structure-periodicity.md` § 6.0, *Entering a new
    calculation*.
    """
    struct = _engine_frame()
    _dest, _stage, text = prepared(struct, "siesta", calculation=calculation,
        **said)
    record = extract_engine_offset(text)
    rule = cellmod.engine_offset(struct.replace(engine_offset=None))
    assert np.linalg.norm(rule) > 1.0, "the fixture must be one the rule moves"
    assert record["stated"] is kept, record
    np.testing.assert_allclose(record["applied_offset"],
                               np.zeros(3) if kept else rule, atol=1e-7)
