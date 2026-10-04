# A flat calculation of ours: H2 relaxed at `coarse`, `medium` prepped beside it

Measured 2026-10-04 through the jobset road, on the workstation, with the
packaged SIESTA 5.4.2 (`molbuilder-siesta`):

    jobset init --structure P/structure/h2.xyz --bundle P/opt/H2flat \
        --engine siesta --shape flat --calculation optimization --name H2 \
        --psml-lib pseudopotential --stage-strategy publishable
    # task.json: "execution": {"mpi_np": 1, "omp_threads": 1}
    jobset prep run coarse --bundle P/opt/H2flat --target this
    jobset launch run coarse --bundle P/opt/H2flat --mode direct --yes
    jobset prep run medium --bundle P/opt/H2flat --target this

The structure is H2 in a 10 Å box, isolated on every axis, the first atom
held (`frozen_atoms = [0]`): H at z = 5.0 Å and 5.741 Å.  The run concluded
`rc=0`, converged, in 69.5 s of SIESTA wall time (42 SCF rows).

**Why it exists: a run of ours names its files two ways.**  The output
carries the stage and the run index, `H2_01_coarse-run0.out`, while SIESTA
names what it writes by `SystemLabel` -- `H2.MD.nc`, `H2.XV` -- and in the
flat layout every stage's deck lies in the one folder, `H2_01_coarse.fdf`
beside `H2_02_medium.fdf`.  A reader that pairs files by name meets all of
that here, as it does on every run of ours.

Every file is the run's or the prep's own, unchanged; the folder holds only
what the tests read (the run's `.DM`, basis files and logs are left out):

* `H2_01_coarse-run0.out` -- the run's output (its `Directory` line names the
  scratch folder it ran in);
* `H2.MD.nc` -- the run's MD history (`WriteMDhistory`, on by default);
* `H2.XV` -- the run's last geometry;
* `fdf.20261004T082401.341.log` -- SIESTA's echo of the deck it read;
* `H2_01_coarse-run0.concluded`, `H2_01_coarse.run.json` -- the wrapper's
  conclusion and the launch record;
* `H2_01_coarse.fdf`, `H2_02_medium.fdf` -- the two stages' decks, each with
  `%block Geometry.Constraints / position 1`;
* `h2.source.xyz`, `h2.source.molstruct.json` -- the structure pair the
  calculation was prepped from;
* `task.json` -- the description.

Read in place by the tests, never copied under another name
(`process/testing.md` § 6).
