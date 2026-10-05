# A flat calculation of ours: H2 relaxed at `coarse`, `medium` prepped beside it

Measured 2026-10-04 through the jobset road, on the workstation, with the
packaged SIESTA 5.4.2 (`molbuilder-siesta`).  The structure pair was written
by `StructureCodec` -- H2 in a 10 Å box, isolated on every axis, the first atom
held (`frozen_atoms = [0]`), H at z = 5.0 Å and 5.741 Å -- and the run made
from it:

    jobset init --structure P/structure/h2.xyz --bundle P/opt/H2flat \
        --engine siesta --shape flat --calculation optimization --name H2 \
        --psml-lib pseudopotential --stage-strategy publishable
    jobset prep run coarse --bundle P/opt/H2flat --target this \
        --np 1 --cpus-per-task 1
    jobset launch run coarse --bundle P/opt/H2flat --mode direct --yes
    jobset prep run medium --bundle P/opt/H2flat --target this \
        --np 1 --cpus-per-task 1

The run concluded `rc=0`, converged: seven CG moves to a largest force of
0.0094 eV/Å on the moved atom, 42 SCF rows, 60.5 s by SIESTA's own timer
(`timer: Elapsed wall time`).

**Why it exists: a run of ours names its files two ways.**  The output
carries the stage and the run index, `H2_01_coarse-run0.out`, while SIESTA
names what it writes by `SystemLabel` -- `H2.MD.nc`, `H2.XV`, `H2.xyz` -- and
the output prints that label (`reinit: System Label: H2`).  In the flat layout
every stage's deck lies in the one folder, `H2_01_coarse.fdf` beside
`H2_02_medium.fdf`, and prep's progress log is there before the run's output.
A reader that pairs files meets all of that here, as it does on every run of
ours.

Every file is the run's or the prep's own, unchanged; the folder holds only
what the tests read (the run's `.DM`, basis files, run scripts and wrapper
logs, and the medium stage's progress log, are left out):

* `H2_01_coarse-run0.out` -- the run's output (its `Directory` line names the
  scratch folder it ran in; `siesta: Constraint (1): pos` is the held atom);
* `H2.MD.nc` -- the run's MD history (`WriteMDhistory`, on by default);
* `H2.XV` -- the run's last geometry;
* `H2.xyz` -- SIESTA's own structure file, the last geometry with no box and
  no sidecar (atom 0 at z = 4.6295 Å, the moved atom at 5.404389 Å);
* `fdf.20261004T220618.285.log` -- SIESTA's echo of the deck it read;
* `H2_01_coarse-run0.concluded`, `H2_01_coarse.run.json` -- the wrapper's
  conclusion and the launch record;
* `H2_01_coarse.fdf`, `H2_02_medium.fdf` -- the two stages' decks, each with
  `%block Geometry.Constraints / position 1`;
* `H2_01_coarse.molwatch.log` -- the progress log prep writes (626 bytes, one
  `initial_preview` block, `# frozen_atoms: 0`); SIESTA never writes into it;
* `H2.source.xyz`, `H2.source.molstruct.json` -- the structure pair the
  calculation was prepped from, named on the run's name;
* `task.json` -- the description.

Read in place by the tests, never copied under another name
(`process/testing.md` § 6): `tests/parse/test_siesta_mdnc.py` (the history
found by the label the output prints), `tests/test_results_blueprint.py`
(what the folder answers and opens), `tests/test_structure_info_bridge.py`
(each run's own deck, and SIESTA's `H2.xyz` opened with its run's box) and
`tests/test_xv2xyz.py` (`--from-run` on `H2.XV`).
