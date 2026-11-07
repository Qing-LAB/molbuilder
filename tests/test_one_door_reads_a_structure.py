"""The author's regions, frozen atoms and cell reach the deck -- on the road.

WHAT THIS IS FOR, in one sentence: a `.xyz` and the `.molstruct.json` beside
it are one file (`model/structure-molstruct.md` § 5), so what a person set in
the viewer -- which atoms are held, what the regions are called, the cell --
must arrive in the input the engine runs.

**It did not, once.** On 2026-09-07 four readers of that pair existed. The one
a person was most likely to reach for -- `molbuilder.load()`, the obvious name,
in `__all__`, docstring accurate for what it did -- read the geometry and not
the sidecar. It predated the sidecar by two months and nothing swept it. Its
one production caller was `molbuilder describe`, the verb that turns a structure into a
calculation (`job-system.md` § 5.1), so a description was born with the
author's regions, frozen atoms and cell missing, and everything downstream was
then faithfully correct about the wrong thing. It is the same failure
`siesta/input.py` records having fixed at its own door -- *"the script relaxed
every atom of a structure whose author had frozen two."*

So the road is driven: the pair is saved through `StructureCodec` (the one
door, `structure.md` § 2.4), `molbuilder describe` describes it, `jobset prep` renders
the deck -- no engine runs -- and the deck is read back.  A second reader
anywhere on that road that takes the geometry and drops the sidecar shows up
here as a deck that relaxes every atom in a box nobody chose.
"""
from __future__ import annotations

import numpy as np

from support.road import describe_calculation, jobset, stage_file


#: Two H2 molecules: the first carries a region, the second is held.  An
#: explicit cell that no bounding box would produce -- 12.5 x 13.5 x 14.5 A --
#: so a deck written from a structure that lost its sidecar says so in its
#: lattice as well as in its constraints.
FROZEN = [2, 3]
REGIONS = {"anchor": [0, 1]}
EDGES = [12.5, 13.5, 14.5]
H4 = {"elements": ["H"] * 4,
      "positions": [[5.0, 5.0, 5.0], [5.0, 5.0, 5.74],
                    [8.0, 6.0, 5.0], [8.0, 6.0, 5.74]],
      "regions": {"frozen_atoms": FROZEN, **REGIONS},
      "cell": EDGES}


def test_what_the_author_set_on_the_structure_reaches_the_deck(tmp_path,
                                                               monkeypatch):
    """**Held atoms held, the stated cell the lattice, the regions carried.**

    The failure: a reader on the road from a saved structure to a deck that
    takes the `.xyz` and drops the `.molstruct.json`.  The deck then relaxes
    every atom -- including the ones the author froze -- in a box derived from
    the bounding box, and loses the region names the Results tab shows.
    Nothing raises: every step is correct about the structure it was handed.

    Contract: `model/structure-molstruct.md` § 5 (the pair is one unit) and
    § 6 (SIESTA `frozen_atoms` -> `Geometry.Constraints`); `structure.md`
    § 2.4 (`StructureCodec` is the paired-file door); `job-system.md` § 5.1.

    MUTATION THIS MUST FAIL AGAINST: `molbuilder describe` -- or `prep`'s
    `_structure_for` -- reading the structure with
    ``Structure.from_xyz(path.read_text())`` instead of
    ``StructureCodec().load(path)``.
    """
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.parse.fdf import fdf_key, read_deck, read_fdf
    from molbuilder.runs import declared, run_of
    from molbuilder.task import of as task_of

    bundle = describe_calculation(tmp_path, monkeypatch, name="H4",
                                  structure=H4)
    # The launch shape is its run card's (`architecture.md` § 5.2); how a
    # shell enters an environment is the machine record's.
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output

    # THE DECK AS THE ONE FDF READER READS IT: no such block is itself the
    # answer.
    blocks = read_fdf(read_deck(stage_file(bundle, "coarse", ".fdf")))[1]

    held = {int(t) for row in blocks.get(fdf_key("Geometry.Constraints"), [])
            for t in row[1:]}
    assert held == {i + 1 for i in FROZEN}, (
        f"the deck holds atoms {sorted(held)} (1-based); the author froze "
        f"{[i + 1 for i in FROZEN]} -- the rest of the structure would relax "
        f"with them")

    lattice = np.array([[float(x) for x in row]
                        for row in blocks.get(fdf_key("LatticeVectors"), [])])
    assert np.allclose(lattice, np.diag(EDGES), atol=1e-9), (
        f"the deck's lattice is not the cell the author stated:\n{lattice}")

    # THE RUN'S OWN DECK says what it carries (`runs.declared`).
    run = latest_attempt(stage_home(bundle, task_of(bundle), "coarse").dir)
    carried = declared(run_of(run)).atom_metadata_for(4) or {}
    assert carried.get("regions", {}).get("anchor") == REGIONS["anchor"], (
        f"the region the author named did not reach the deck: {carried}")
