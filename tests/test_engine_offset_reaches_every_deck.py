"""Every engine is handed the design coordinates plus the engine offset, and
every deck says so -- `model/structure-periodicity.md` § 6.0.

Through the road: a description (the door `jobset init` and the hand-over use),
then `jobset prep task`, then the deck on disk.  Per engine:

* the invariant is coordinates + offset: the deck writes the design
  coordinates plus ``cell.engine_offset`` of the design structure -- one rigid
  translation, frozen atoms included -- and what it writes carries no offset
  of its own (zero at the hand-off: the correction applied);
* every atom is inside the cell, with equal margins along each lattice vector;
* the deck's ENGINE-OFFSET record states that same offset and that cell.
"""
from __future__ import annotations

import ast
import json
import re

import numpy as np
import pytest
from click.testing import CliRunner

from molbuilder import cell as cellmod
from molbuilder import describe as D
from molbuilder.jobset._cli import jobset_group
from molbuilder.scheduler import Environment, Topology
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


def _prep(root, struct, cfg, stages, engine, *, stage=None,
          calculation="optimization", name="JOB", before_prep=None,
          refused=False):
    """Describe, then `jobset prep task` one rung (the first unless ``stage``
    names another); the deck's text.  ``root`` is the per-test projects tree
    (`isolated_projects_root`): `prep` reads a calculation only from inside
    the tree.  ``before_prep(dest)`` puts on disk what an earlier rung left --
    a finished attempt -- before the prep that reads it.  ``refused`` expects
    prep to refuse, and returns what it said in place of the deck."""
    # AS A PAIR, through the codec: the cell, the vacuum and the frozen set
    # live in the sidecar, and a bare `.xyz` would hand prep a molecule in the
    # default box -- a different structure from the one described.
    from molbuilder.workingcopy_structure import StructureCodec
    StructureCodec().write(struct, root / "in.xyz")
    # ONE CALCULATION PER FOLDER: each call describes a new one, so each has
    # a folder of its own -- a prepared stage is not prepared again
    # (`job-system.md` § 5.0).
    dest, n = root / "t" / "calc", 1
    while dest.exists():
        n += 1
        dest = root / "t" / f"calc{n}"
    D.write_description(
        D.build_description(struct, cfg, stages, engine=engine,
                            shape="hierarchical", name=name,
                            calculation=calculation,
                            source=str(root / "in.xyz")),
        dest, struct=struct)       # as `jobset init` does: the pair travels
    if engine == "siesta":
        from conftest import write_pseudos
        write_pseudos(dest, sorted(set(struct.elements)))
    # THE RUN CARD STATES THE LAUNCH SHAPE, as a described calculation does
    # (`architecture.md` § 5.2): four ranks of one thread for SIESTA -- this
    # record's four cores -- and one thread for PySCF.
    task = json.loads((dest / "task.json").read_text())
    task["execution"] = ({"mpi_np": 4, "omp_threads": 1}
                         if engine == "siesta" else {"threads": 1})
    (dest / "task.json").write_text(json.dumps(task, indent=2))
    # The machine the calculation is prepared for, with how a shell enters an
    # environment there -- the record carries it for the generator
    # (`configuration.md` § 4).
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4),
                    env_init={"activation": "conda activate",
                                       "preamble": "true"}
                    ).to_json() + "\n")
    if before_prep is not None:
        before_prep(dest)
    stage = stage or stages[0].name
    r = CliRunner().invoke(jobset_group, ["prep", "task", "--stage", stage, "--bundle",
                                          str(dest), "--no-sbatch"])
    if refused:
        assert r.exit_code != 0, r.output
        return dest, stage, r.output
    assert r.exit_code == 0, r.output
    suffix = ".fdf" if engine == "siesta" else ".py"
    # THE DECK BY ITS NAME, through the catalogue -- not the first file with
    # the suffix: a stage directory may hold other `.py` files than the deck
    # (a person's own scripts may).
    from molbuilder.runfiles import compose
    stage_dir = next(dest.glob(f"*_{stage}"))
    deck = stage_dir / compose(name, suffix, stage_dir.name)
    return dest, stage, deck.read_text()


def _design(dest):
    """The ORIGINAL file -- the pair the description names, as prep reads it.
    The invariant is about files: its coordinates plus its offset."""
    from molbuilder.workingcopy_structure import StructureCodec
    task = json.loads((dest / "task.json").read_text())
    return StructureCodec().load(dest / task["structure"]["source"])


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
    np.testing.assert_allclose(design.to_wire()["periodicity"]["box_corner"],
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
        isolated_projects_root):
    """A periodic slab on a skewed cell, authored around the origin, reaches
    the SIESTA deck as its design coordinates plus the engine offset -- one
    rigid translation, frozen atoms included, centred in the cell -- and the
    run's trajectory starts where the deck does.

    CONTRACT: `model/structure-periodicity.md` § 6.0 (the rule; the
    invariant; frozen atoms move with the rest).
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import default_siesta_stages
    struct = _slab()
    dest, stage, text = _prep(isolated_projects_root, struct,
                              SiestaConfig(system_label="JOB"),
                              default_siesta_stages("publishable"), "siesta")
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
    log = next((dest / f"01_{stage}").rglob("*.molwatch.log")).read_text()
    step0 = log.split("coordinates (Ang):", 1)[1].split("energy", 1)[0]
    np.testing.assert_allclose(
        np.array([[float(v) for v in r[1:4]] for r in _rows(step0)]),
        written, atol=1e-6)


@pytest.mark.parametrize("calculation", ["optimization", "vibration"])
def test_a_pyscf_deck_writes_the_design_coordinates_plus_the_offset(
        isolated_projects_root, calculation):
    """A molecule with no cell reaches the PySCF deck -- the optimisation
    deck and the vibration deck, two spec builders -- as its design
    coordinates plus the engine offset, centred in the box its vacuum sizes.

    CONTRACT: `model/structure-periodicity.md` § 6.0 (every engine is handed
    the design coordinates plus the offset).
    """
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.pyscf.stages import default_pyscf_stages, vibration_stages
    struct = _water()
    if calculation == "vibration":
        cfg = PySCFConfig(job_name="JOB", already_relaxed=True)
        stages = vibration_stages("pyscf", already_relaxed=True)
    else:
        cfg, stages = PySCFConfig(job_name="JOB"), default_pyscf_stages("publishable")
    dest, _stage, text = _prep(isolated_projects_root, struct, cfg, stages,
                               "pyscf", calculation=calculation)
    written = _written("pyscf", text)
    design = _design(dest)
    assert tuple(design.vacuum) == (5.0, 5.0, 5.0), "the vacuum reached prep"
    _assert_placed(design, written, text)


def _engine(engine):
    """A description's config and ladder, per engine."""
    if engine == "siesta":
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.siesta.stages import default_siesta_stages
        return SiestaConfig(system_label="JOB"), default_siesta_stages("publishable")
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.pyscf.stages import default_pyscf_stages
    return PySCFConfig(job_name="JOB"), default_pyscf_stages("publishable")


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


def _assigned(corner):
    """Water in a typed 10 Å box, its origin assigned at ``corner`` through the
    Cell page's own op (`periodicity_gate.apply_edit`, ``box_corner``)."""
    from molbuilder.periodicity_gate import apply_edit
    typed = Structure(elements=["O", "H", "H"], positions=_water().positions,
                      cell=np.eye(3) * 10.0, axis_kind=("isolated",) * 3)
    out, _ = apply_edit(typed, "box_corner", list(corner))
    return typed, out


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_an_assigned_origin_places_the_deck_at_design_minus_it(
        isolated_projects_root, engine):
    """T1's assigned-origin clause (plan § 5q.4): on a typed cell with an origin
    the person assigned at P, the deck writes the design coordinates minus P --
    not the rule's centring -- and its record says the offset was stated.

    CONTRACT: `model/structure-periodicity.md` § 6.0, *A stated offset* (D1).
    """
    P = np.array([-3.0, -3.0, -3.0])
    typed, assigned = _assigned(P)
    assert not np.allclose(-P, cellmod.engine_offset(typed)), (
        "the fixture must be one the rule would place elsewhere")
    cfg, stages = _engine(engine)
    dest, _stage, text = _prep(isolated_projects_root, assigned, cfg, stages,
                               engine)
    design = _design(dest)
    np.testing.assert_allclose(design.engine_offset, -P)    # the pair carried it
    np.testing.assert_allclose(_written(engine, text), design.positions - P,
                               atol=1e-7)
    record = extract_engine_offset(text)
    assert record["stated"] is True, record
    np.testing.assert_allclose(record["applied_offset"], -P, atol=1e-7)
    # The kinds as the structure had them (D5) -- what the Results tab reads.
    assert record["axis_kind"] == list(design.axis_kind), record
    np.testing.assert_allclose(design.to_wire()["periodicity"]["box_corner"],
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
        isolated_projects_root, engine):
    """The same box with its origin assigned at (-0.5, -1, -1): one hydrogen,
    atom 2, would land at x = -0.257 -- outside along a, every other atom
    inside and every other axis clear.  The deck is refused, and prep names
    that atom and that axis, in a sentence rather than a traceback.

    CONTRACT: `model/structure-periodicity.md` § 6.0, check 3 (an origin that
    leaves an atom outside is refused naming it); `workflow.md` § 9
    (a gate refuses with the reason, never a stack trace).
    """
    _typed, assigned = _assigned([-0.5, -1.0, -1.0])
    cfg, stages = _engine(engine)
    _dest, _stage, said = _prep(isolated_projects_root, assigned, cfg, stages,
                                engine, refused=True)
    assert "outside the cell" in said, said
    assert "a (isolated): atom(s) 3 " in said, said
    assert "b (" not in said and "c (" not in said, said
    # Under the hand-off's own id: the Cell page warned of the same atom as
    # `cell.atoms_outside`, and one id carries one severity (plan § 5q D11).
    assert "[deck.atoms_outside]" in said, said


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_a_box_nothing_can_be_placed_in_is_refused_in_a_sentence(
        isolated_projects_root, engine):
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
    cfg, stages = _engine(engine)
    _dest, _stage, said = _prep(isolated_projects_root, flat, cfg, stages,
                                engine, refused=True)
    assert "[cell.no_volume]" in said, said
    assert "Traceback" not in said, said


def test_an_atom_past_a_periodic_face_is_a_warning_in_the_report(
        isolated_projects_root):
    """Along a periodic vector an atom past a face is legal -- an image the
    engine wraps -- and said.

    GOAL: with the correction applied at every deck, an atom still outside
    the cell after it is something the person should see, not a silence
    (user, 2026-09-25: "we should now give warning/error when atoms are
    outside boundary for all cases").  CONTRACT: `model/structure-periodicity.md`
    § 6.0, checks 1 and 3: warned along a periodic vector, refused along any
    other.
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import default_siesta_stages
    # 1.2 cells long along a, so no translation fits it -- and the atom past
    # the face sits 3 Å off the others' line, so its image meets no atom.
    wide = Structure(elements=["C", "C", "C"],
                     positions=np.array([[0.0, 1.0, 1.0], [3.0, 1.0, 1.0],
                                         [6.0, 4.0, 1.0]]),
                     cell=np.diag([5.0, 8.0, 8.0]),
                     axis_kind=("periodic", "periodic", "periodic"))
    dest, stage, _text = _prep(isolated_projects_root, wide,
                               SiestaConfig(system_label="JOB"),
                               default_siesta_stages("publishable"), "siesta")
    report = next(next(dest.glob(f"*_{stage}")).glob("*.validation.txt")).read_text()
    assert "[cell.beyond_periodic_face]" in report, report
    assert "along a, which is periodic" in report, report


def test_a_crystal_that_fills_its_cell_is_not_called_tight(
        isolated_projects_root):
    """A periodic structure fills its cell by construction, and its report
    does not call the cell tight.

    GOAL: every rung of the 2026-09-25 transport ladder was told "cell is
    suspiciously tight; expect image-image interactions" -- the bulk gold lead
    at 1.45, the junction at 1.03 -- because the volume ratio was taken over
    all three axes.  CONTRACT: `model/structure-periodicity.md` § 2 (images
    across a periodic or transport axis are intended; only an isolated axis is
    measured).
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import default_siesta_stages
    dest, stage, _text = _prep(isolated_projects_root, _slab(),
                               SiestaConfig(system_label="JOB"),
                               default_siesta_stages("publishable"), "siesta")
    report = next(next(dest.glob(f"*_{stage}")).glob("*.validation.txt")).read_text()
    assert "[cell.volume]" not in report, report
