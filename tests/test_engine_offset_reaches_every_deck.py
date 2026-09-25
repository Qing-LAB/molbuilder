"""Every engine is handed the design coordinates plus the engine offset, and
every deck says so -- `model/structure-periodicity.md` § 6.0.

Through the road: a description (the door `jobset init` and the hand-over use),
then `jobset prep run`, then the deck on disk.  Per engine:

* the invariant is coordinates + offset: the deck writes the design
  coordinates plus ``cell.engine_offset`` of the design structure -- one rigid
  translation, frozen atoms included -- and what it writes carries no offset
  of its own (zero at the hand-off: the correction applied);
* every atom is inside the cell, with equal margins along each lattice vector;
* the deck's ENGINE-OFFSET record states that same offset and that cell.

The transport rungs are asserted beside their own fixtures, in
``test_transport_prep.py``.
"""
from __future__ import annotations

import json
import re

import numpy as np
from click.testing import CliRunner

from molbuilder import cell as cellmod
from molbuilder import describe as D
from molbuilder.jobset._cli import jobset_group
from molbuilder.scheduler import Environment, Topology
from molbuilder.script_emit import extract_engine_offset
from molbuilder.structure import Structure

#: A skewed cell, as the 2026-09-25 junction's was.
_HEX = np.array([[8.651, 0.0, 0.0], [4.326, 7.492, 0.0], [0.0, 0.0, 20.0]])


def _slab():
    """A periodic structure on the skewed cell, authored AROUND the world
    origin -- how the junction builder authors -- so the offset that places
    it is far from zero.  Two atoms are frozen."""
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


def _prep(root, struct, cfg, stages, engine, before_prep=None):
    """Describe, then `jobset prep run` the first rung; the deck's text.
    ``root`` is the per-test projects tree (`isolated_projects_root`): `prep`
    reads a calculation only from inside the tree.  ``before_prep(dest)``
    edits the described calculation first, as a file on disk may differ."""
    # AS A PAIR, through the codec: the cell, the vacuum and the frozen set
    # live in the sidecar, and a bare `.xyz` would hand prep a molecule in the
    # default box -- a different structure from the one described.
    from molbuilder.workingcopy_structure import StructureCodec
    StructureCodec().write(struct, root / "in.xyz")
    dest = root / "t" / "calc"
    D.write_description(
        D.build_description(struct, cfg, stages, engine=engine,
                            shape="hierarchical", name="JOB",
                            source=str(root / "in.xyz")),
        dest, struct=struct)       # as `jobset init` does: the pair travels
    if engine == "siesta":
        from conftest import write_pseudos
        write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4)
                    ).to_json() + "\n")
    if before_prep is not None:
        before_prep(dest)
    stage = stages[0].name
    r = CliRunner().invoke(jobset_group, ["prep", "run", stage, "--bundle",
                                          str(dest), "--no-sbatch"])
    assert r.exit_code == 0, r.output
    suffix = ".fdf" if engine == "siesta" else ".py"
    deck = next((dest / f"01_{stage}").glob(f"*{suffix}"))
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
    cell = np.asarray(record["cell"], dtype=float)
    np.testing.assert_allclose(cell, design.resolve_cell(), atol=1e-7)
    cellmod.require_placed(cellmod.engine_frame(cell, written))
    frac = np.linalg.solve(cell.T, written.T).T
    near, far = frac.min(axis=0), 1.0 - frac.max(axis=0)
    assert np.all(near > 0), near
    np.testing.assert_allclose(near, far, atol=1e-7)


def _rows(block):
    return [l.split() for l in block.strip().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def test_a_siesta_deck_writes_the_design_coordinates_plus_the_offset(
        isolated_projects_root):
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


def test_a_pyscf_deck_writes_the_design_coordinates_plus_the_offset(
        isolated_projects_root):
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.pyscf.stages import default_pyscf_stages
    struct = _water()
    dest, _stage, text = _prep(isolated_projects_root, struct,
                               PySCFConfig(job_name="JOB"),
                                default_pyscf_stages("publishable"), "pyscf")
    block = re.search(r"_atom_block = '''\n(.*?)'''", text, re.S).group(1)
    written = np.array([[float(v) for v in r[1:4]] for r in _rows(block)])
    design = _design(dest)
    assert tuple(design.vacuum) == (5.0, 5.0, 5.0), "the vacuum reached prep"
    _assert_placed(design, written, text)


def test_the_renderer_refuses_coordinates_that_still_carry_an_offset():
    """Zero at the hand-off (user, 2026-09-25): *"offset should be zero at that
    point, meaning the correction should have been applied"* -- and the deck
    renderer checks it before it writes a line.

    API-level, because the road cannot reach it: every emitter places through
    ``cell.to_engine``, so no prep produces uncorrected coordinates.  What this
    pins is that one that did would be refused rather than written.
    """
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.script_emit import render_deck
    from molbuilder.siesta.input import spec_for
    struct, cfg = _slab(), SiestaConfig(system_label="JOB")
    spec = spec_for(struct, cfg)
    unplaced = cellmod.EngineFrame(cell=spec.engine_frame.cell,
                                   positions=struct.positions,
                                   applied_offset=np.zeros(3))
    import pytest
    with pytest.raises(ValueError, match="still carry an offset"):
        render_deck(dataclasses.replace(spec, engine_frame=unplaced),
                    struct, cfg)


#: `[item.wrap_into_cell]` as the writer emitted it before 2026-09-25 --
#: copied from `projects/claude-vib-ui/optimization/au333bdt-loose`'s template.
_RETIRED_WRAP_ITEM = '''
[item.wrap_into_cell]
kind = "produce"
category = ["procedure"]
engine_key = "(molbuilder: pre-emission positioning)"
type = "bool"
value = true
default = true
role = ["transport"]
group = "profile"
label = "Wrap atoms into cell"
help = """
Move any atom that sits outside the cell box back inside it.
"""
'''


def test_a_template_written_before_wrap_into_cell_retired_still_preps(
        isolated_projects_root):
    """A calculation described before `wrap_into_cell` retired still preps.

    GOAL: every SIESTA template written before 2026-09-25 carries that item
    with a value -- 17 on the development machine that day, four under
    `projects/Au-BDT-Au` -- and the reader refused it as a name the schema
    does not declare, so each of those calculations stopped prepping (found by
    an independent review of the retirement).  CONTRACT: `engines/template.md`
    § 7 -- a RETIRED item is accepted and ignored; an unknown one is refused.

    The item block is a measured fixture, copied from a real template.
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import default_siesta_stages

    def add_the_retired_item(dest):
        tpl = next(dest.glob("*.template.toml"))
        tpl.write_text(tpl.read_text() + _RETIRED_WRAP_ITEM)

    _dest, _stage, text = _prep(isolated_projects_root, _slab(),
                                SiestaConfig(system_label="JOB"),
                                default_siesta_stages("publishable"), "siesta",
                                before_prep=add_the_retired_item)
    assert "%block AtomicCoordinatesAndAtomicSpecies" in text

