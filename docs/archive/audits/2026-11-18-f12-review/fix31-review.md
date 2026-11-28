# F12 fix 31 (the example notebook, and molbuilder's examples in your tree, 2.88.0) — the review

**Verdict: closed with 2.88.0.** R3d2b2c, the plan's example notebook — the user, 2026-11-23: *"an example
jupyternotebook that can open a calculation result folder and generate some mixing/plots as the
demonstration/example/tests to validate these calls"*. molbuilder ships its example notebooks and writes them into the
projects tree's `notebooks/examples/` at every installation and by `molbuilder examples`; the vibration notebook runs
the toolkit on a carried PySCF methanol vibration, every number computed and every check against the science; a frame
set read back carries its rows; a result records no machine's name.

The lessons: **a check of the code against its model is not a check of the physics** (the user, 2026-11-27: *"why did
your last try end up with the photon tried to push the vibration (soft mode) to such an extend that it is physically
impossible?"*; *"is it that the harmonic assumption is completely broken or something else? the photon certainly
should not be strong enough to break a molecule"*): the plan put one photon on methanol's 338 cm⁻¹ torsion because it
is warm at 300 K, a vivid demonstration; every assert held — they hold the toolkit to the harmonic formulas — while
the frames moved the hydroxyl H 0.85 Å and stretched O–H 0.97 → 1.33 Å: a straight-line path where the real motion is
a turn, above a ~370 cm⁻¹ barrier; the 1 eV mark never fired (0.06 eV). The science reader caught it by reading a size
against the molecule. **Recall the person's stance before offering a choice**: the user had said *"we never, almost
never, work with soft modes"*, and the question asked only which excitation the torsion should carry. **An example the
installation writes into every tree must run in every tree**: transport's notebook read a run in one developer's tree.

## The user's decisions (2026-11-27)

- The examples folder: *"how about having a "example" subdir under notebooks? all example post-processing, api
  demonstration, framework support examples can be put there and all installation will cleanly put the correct version
  there."*; an edited example: *"Replace it"*; *"shall we have a verb/script option just to reinstall/refresh the
  example in the project folder?"* — `molbuilder examples`, installation calling the same door.
- The excitation: first *"One photon on the warm torsion"*; then, shown the frames' bond stretch, *"A stiff mode"*;
  the torsion in the snapshot sets: *"Leave it out"*.
- Results' host: *"Keep host names out"*. The carried run: *"Allow the exception"* (`process/testing.md` § 6).
- Transport's notebook: *"Read your own run"*.

## What was built

`molbuilder/data/examples/` (package data): `vibration-frames.ipynb`, `transport-junction-by-hand.ipynb` (moved from
`docs/engines/`), `vibration-frames-run/methanol/` (the three files the toolkit reads); `projects.write_examples`,
`examples_dir`, `EXAMPLES_FOLDER`, `_tree_own_folders`; `molbuilder examples`; `envs init-config` writing the examples
last, a tree that cannot be written said and the seeding finished; `scripts/install-env.sh examples`;
`tools/make_example_runs.py` (the methanol run down the road, the notebooks executed and saved, in scratch config and
projects); `nbclient`/`nbformat` in the jupyternb recipe; `read_frame_set` filling `rows` on `Frames` and `Snapshots`;
`postprocess.openable`; `runtime_info` without `hostname`; tests: the notebook e2e over every shipped notebook, the
shipped examples, the seeding, the projects tree's own folders, rows read back, a held atom in snapshots, run facts
naming no machine. Documents: `engines/vibration-frames.md` § 1, 4, 5, 5.1, 6.1, 7, 8, 9.1; science § 2d.7, § 2e, § 6;
`engines/vibration-postprocessing.md`, `engines/transport.md`, `engines/transport-postprocessing.md`,
`engines/vibration.md` § 6.2, `engines/pyscf.md`, `model/datafiles.md`, `model/parse.md` § 3, § 6;
`execution/project-layout.md` § 2.6 (the examples rule's home), `execution/job-contracts.md` § 2.5;
`ops/installation.md`, `ops/deployment.md`, `configuration.md` § 3.2, `web/jupyter.md`, `process/conventions.md` § 3,
`process/testing.md` § 6.

## Round 1 — three readers (code and API 4, science and documents 15, tests 5): 24

All applied, the user's approvals and decisions above. **The code**: installation died on a projects tree that cannot
be written yet (a cluster's, not mounted) before the machine record was written; the example tool could write another
checkout. **The science**: the photon on the soft torsion outside the harmonic picture; transport's notebook reading
one developer's run; "run it on your own vibration" not working as written; the photon checks unable to tell the
photon from its stand-in; the read-back plot comparing the in-memory numbers with themselves; a map hiding the move its
title quoted; science § 6's *only* against the contract's *at least*; "every piece"; an assert that could not fail;
level(2)'s shape unchecked; tolerances not from the science; imports past the notebook's one layer; the examples rule
written five times. **The tests**: the notebook e2e passing with a cell's checks skipped, blind to a write into the
person's tree, running one notebook of two; the default-root seeding test narrowed.

## The reader of round 1's fixes: 8 findings

Applied, the user's *"Leave it out"* and *"Yes, apply them"*: the torsion still moved in every snapshot set (now left
out, found from the run); the response's coefficient explained as a tunnelling rate 7× off (said as a model, its decay
printed); the 1 eV mark called a check; a tolerance's hidden floor; a settings page contradicting the examples step; the
seeding test blind to the examples step's order; the transport slice naming the chain's rows on a person's run; the
examples' edit instructions not saying an edited example is replaced.

## The e2e run

After the last fixes: the fast suite, 7069 passed; end to end, 198 passed —
`test_vibration_frames_notebook_e2e.py` (both shipped notebooks, each run twice in one kernel from a tree laid out as
installation writes it), `test_vibration_e2e.py`, `test_vibration_mechanics_e2e.py`, `test_spectrum_results_e2e.py`,
`test_siesta_vibration_e2e.py`, `test_the_viewers_follow_a_real_run_e2e.py` (the spectra and trajectory pages, their
result Host row gone, in a browser), `test_a_person_reads_a_pyscf_run_e2e.py`, `test_notebook_through_one_port_e2e.py`.
None failed.

## Topic → test

| topic | test |
|---|---|
| every shipped notebook run twice as installation writes it, every cell run, no error, every figure drawn, nothing else written | `tests/test_vibration_frames_notebook_e2e.py` |
| `molbuilder examples`: written, replaced, kept; a person's file left alone; a tree that cannot be written refused in words | `tests/test_the_shipped_examples.py` |
| installation writes the examples last, a tree that cannot be written said and the seeding finished | `tests/test_seeding_a_fresh_machine.py` |
| the projects tree's own folders are no project | `tests/test_projects.py::test_the_trees_own_folders_are_no_project` |
| a set read back carries its rows, which `weights_for_set` takes | `tests/test_vibration_frames.py::test_a_set_read_back_is_the_set_written` |
| snapshots keep a held atom still | `tests/test_vibration_mechanics.py::test_snapshots_keep_a_held_atom_still` |
| a run's facts name no machine | `tests/test_pyscf_thread_sizing.py::test_what_a_run_records_of_itself_names_no_machine` |
| end to end | `tests/test_vibration_frames_notebook_e2e.py`, `test_vibration_e2e.py`, `test_vibration_mechanics_e2e.py`, `test_spectrum_results_e2e.py`, `test_siesta_vibration_e2e.py`, `test_the_viewers_follow_a_real_run_e2e.py`, `test_a_person_reads_a_pyscf_run_e2e.py`, `test_notebook_through_one_port_e2e.py` |
