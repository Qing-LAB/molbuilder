"""The author's regions, frozen atoms and cell reach the deck -- on the road.

WHAT THIS IS FOR, in one sentence: a `.xyz` and the `.molstruct.json` beside
it are one file (`model/structure-molstruct.md` § 6), so what a person set in
the viewer -- which atoms are held, what the regions are called, the cell --
must arrive in the input the engine runs.

**It did not, once.** On 2026-09-07 four readers of that pair existed. The one
a person was most likely to reach for -- `molbuilder.load()`, the obvious name,
in `__all__`, docstring accurate for what it did -- read the geometry and not
the sidecar. It predated the sidecar by two months and nothing swept it. Its
one production caller was `jobset init`, the verb that turns a structure into a
calculation (`job-system.md` § 5.1), so a description was born with the
author's regions, frozen atoms and cell missing, and everything downstream was
then faithfully correct about the wrong thing. It is the same failure
`siesta/input.py` records having fixed at its own door -- *"the script relaxed
every atom of a structure whose author had frozen two."*

So the road is driven: the pair is saved through `StructureCodec` (the one
door, `structure.md` § 2.4), `jobset init` describes it, `jobset prep` renders
the deck -- no engine runs -- and the deck is read back.  A second reader
anywhere on that road that takes the geometry and drops the sidecar shows up
here as a deck that relaxes every atom in a box nobody chose.  *(This file read
the package's source for direct calls to the low-level readers until
2026-09-26, and wrote a probe file into the package to test itself.)*
"""
from __future__ import annotations

import numpy as np

from conftest import write_pseudos


#: Two H2 molecules: the first carries a region, the second is held.  An
#: explicit cell that no bounding box would produce -- 12.5 x 13.5 x 14.5 A --
#: so a deck written from a structure that lost its sidecar says so in its
#: lattice as well as in its constraints.
POSITIONS = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.74],
             [8.0, 6.0, 5.0], [8.0, 6.0, 5.74]]
CELL = np.diag([12.5, 13.5, 14.5])
FROZEN = [2, 3]
REGIONS = {"anchor": [0, 1]}


def _jobset(*args):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _block(deck: str, name: str) -> list:
    """The lines between ``%block <name>`` and ``%endblock <name>`` -- none
    when the deck has no such block, which is itself the answer."""
    lines = deck.splitlines()
    if f"%block {name}" not in lines:
        return []
    start = lines.index(f"%block {name}")
    return lines[start + 1:lines.index(f"%endblock {name}", start)]


def test_what_the_author_set_on_the_structure_reaches_the_deck(tmp_path,
                                                               monkeypatch):
    """**Held atoms held, the stated cell the lattice, the regions carried.**

    The failure: a reader on the road from a saved structure to a deck that
    takes the `.xyz` and drops the `.molstruct.json`.  The deck then relaxes
    every atom -- including the ones the author froze -- in a box derived from
    the bounding box, and loses the region names the Results tab shows.
    Nothing raises: every step is correct about the structure it was handed.

    Contract: `model/structure-molstruct.md` § 6 (the pair is one unit) and
    § 7 (SIESTA `frozen_atoms` -> `Geometry.Constraints`); `structure.md`
    § 2.4 (`StructureCodec` is the paired-file door); `job-system.md` § 5.1.

    MUTATION THIS MUST FAIL AGAINST: `jobset init` -- or `prep`'s
    `_structure_for` -- reading the structure with
    ``Structure.from_xyz(path.read_text())`` instead of
    ``StructureCodec().load(path)``.
    """
    from molbuilder.runs import declared, run_of
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    write_pseudos(tree / "pseudopotential", ["H"])
    s = Structure(elements=["H"] * 4, positions=np.asarray(POSITIONS),
                  cell=CELL, axis_kind=("isolated",) * 3)
    s.frozen_atoms = list(FROZEN)
    s.regions = dict(s.regions or {}, **REGIONS)
    StructureCodec().write(s, tree / "P" / "structure" / "h4.xyz")
    sidecar = tree / "P" / "structure" / "h4.molstruct.json"
    assert sidecar.is_file(), "the codec wrote no sidecar -- nothing to lose"
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)

    r = _jobset("init", "--structure", "P/structure/h4.xyz",
                "--bundle", "P/optimization/H4", "--engine", "siesta",
                "--shape", "hierarchical", "--name", "H4",
                "--psml-lib", "pseudopotential")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H4"
    # THE LAUNCH SHAPE STATED on the prep (`architecture.md` § 5.2); how a
    # shell enters an environment is the machine record's.
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this", "--np", "2", "--cpus-per-task", "1")
    assert r.exit_code == 0, r.output

    attempt = bundle / "01_coarse" / "run-0"
    decks = sorted(attempt.glob("*.fdf"))
    assert len(decks) == 1, [p.name for p in attempt.iterdir()]
    deck = decks[0].read_text()

    held = {int(t) for line in _block(deck, "Geometry.Constraints")
            for t in line.split()[1:]}
    assert held == {i + 1 for i in FROZEN}, (
        f"the deck holds atoms {sorted(held)} (1-based); the author froze "
        f"{[i + 1 for i in FROZEN]} -- the rest of the structure would relax "
        f"with them")

    lattice = np.array([[float(x) for x in line.split()]
                        for line in _block(deck, "LatticeVectors")])
    assert np.allclose(lattice, CELL, atol=1e-9), (
        f"the deck's lattice is not the cell the author stated:\n{lattice}")

    # THE RUN'S OWN DECK says what it carries (`runs.declared`).
    carried = declared(run_of(attempt)).atom_metadata_for(4) or {}
    assert carried.get("regions", {}).get("anchor") == REGIONS["anchor"], (
        f"the region the author named did not reach the deck: {carried}")
