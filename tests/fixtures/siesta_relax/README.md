# A two-atom relaxation, one atom held -- the `relax` stage of a vibration

SIESTA 5.4.2 (the packaged `molbuilder-siesta` binary), H2 in a 10 Å box,
PBE/DZP (`PAO.BasisSize DZP`, `PAO.EnergyShift 0.01 Ry`), `MeshCutoff 300 Ry`,
`ElectronicTemperature 300 K`, Broyden to `MD.MaxForceTol 0.01 eV/Ang`
(`MD.Steps 100`, `MD.MaxDispl 0.02 Ang`), atom 1 held
(`%block Geometry.Constraints / position 1`).  Measured 2026-09-24 through the
jobset road: `jobset init --engine siesta --calculation vibration` on the
experimental bond (H at z = 5.741 Å above the held H at z = 5.0), the
`relax` stage prepped and launched with `--mode direct` and `-np 2`
(`run.json`).  The atom order is the sorted copy's (held atom first), which is
why the record's held set is `[0]`.

The tests read this directory the way the Results tab does -- through
`parse.dirs.openable_in` and the registry -- so it keeps the layout a run
has: `task.json` at the calculation root, `01_relax/run-0/` with
`calcdir.json` pointing at it.

Files, and which of them the RUN wrote:

* `01_relax/run-0/H2_01_relax-run0.out` -- the run's output.  What the tests
  pin, with the line in the file:
  `redata: Force tolerance = 0.0100 eV/Ang` (the run's own criterion);
  `siesta: Constraint (1): pos / [ 1 ]` (the held atom);
  four `Begin Broyden opt. move = 0 .. 3` blocks, so the parser's last frame
  (the `outcoor: Relaxed atomic coordinates` block) is step 4;
  the last force block `1  ... -0.004887` / `2  ... 0.001042` with
  `Max 0.004887` and `Max 0.001042 constrained` (the largest force over all
  atoms, and over the moved atom);
  `outcoor: Relaxed atomic coordinates (Ang)`: the moved atom at
  z = 5.77458276, which is 5.774583 at the fingerprint's millionth of an
  ångström.
* `01_relax/run-0/H2.XV`, `H2_01_relax-run0.concluded` -- the run's.
* `01_relax/run-0/H2_01_relax.fdf`, `H2_01_relax.molwatch.log` (607 bytes,
  the seed prep writes and SIESTA never writes into -- the shape that made
  the door offer a stub before 2026-09-24), `calcdir.json`, `run.json`,
  `../../task.json` -- prep's and launch's.

Read by `tests/parse/test_contract.py` (the relaxation record, § 5b.1 of
`model/parse.md`), `tests/parse/dirs/test_rundir.py` and
`tests/test_path_framework_doors.py` (what opens), `tests/test_results_blueprint.py`
and `tests/test_structure_info_bridge.py` (the composer and its door), and
`tests/test_siesta_vibration_deck.py` (the record table of
`engines/vibration.md` § 2.2).
