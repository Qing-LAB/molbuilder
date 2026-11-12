"""The progress log prep seeds, read as a viewer opens it.

THE SILENT FAILURE: a seeded log its own parser cannot read -- the Results
tab shows nothing until the engine writes -- or its preview read as a step
the engine computed, an energy, forces or an SCF history where none was.

Contract: `engines/pyscf.md` § 4 (the progress log): step 0 is the
initial-state preview -- coordinates only, no energy -- written by the one
writer of the header and step 0, which prep's seed asks before a run starts;
`model/parse.md` § 5.5 (the progress log, SEEDED at prep).  Where the
preview's atoms stand is the engine frame's tests'
(`tests/test_engine_offset_reaches_every_deck.py`), in whose order the
held-atoms table's (`tests/data/held_atoms.toml`, its `steps_read` row).
"""

from __future__ import annotations


def test_the_log_prep_seeds_reads_as_its_atoms_and_nothing_computed(prepared):
    """The H2 described and prepared for SIESTA: its seeded progress log
    reads, through the registry, as one frame of its two atoms -- engine
    `siesta`, as the log's header declares -- read as the preview, with no
    energy, no forces and no SCF history."""
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.parse import parse
    from molbuilder.runfiles import ROLE_MOLWATCH
    from molbuilder.runs import run_of
    from molbuilder.task import of as task_of
    from support.road import H2, road_structure
    calc, stage, _deck = prepared(road_structure(H2), "siesta")
    run = run_of(latest_attempt(stage_home(calc, task_of(calc), stage).dir))
    read = parse(run.file(ROLE_MOLWATCH))
    assert read.source_format == "siesta", read.source_format
    (frame,) = read.frames
    assert list(frame.structure.elements) == list(H2["elements"])
    assert frame.preview is True
    assert (frame.energy, frame.forces, frame.max_force, frame.scf_history) \
        == (None, None, None, [])
