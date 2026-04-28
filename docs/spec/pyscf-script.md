# Spec — generated PySCF script

**File**: `molbuilder/pyscf_input.py` &nbsp;·&nbsp; **Module under test**: `tests/test_pyscf.py`

This document is the contract for the runnable Python script
`render_script(struct, cfg)` produces.  Tests must verify the spec;
spec changes must come with both code and test updates.

When in doubt, prefer behavioural assertions over structural ones —
a structural test ("the string `mol = mol_pre` appears") accepts any
implementation that writes that line; a behavioural test ("after
running, `<job>.log` contains both pre-opt and production SCF entries
in order") catches subtler regressions.

---

## Output files

A successful run of the generated script produces, in the working
directory it was launched from, exactly the files in the table below.
File presence depends on the cfg flags listed in the second column.

| file | enabled when | contents |
| --- | --- | --- |
| `<job>.log` | `cfg.log_file` (default `True`) | the verbose PySCF log for **all stages** (pre-opt + production); appended, never truncated mid-run |
| `<job>.chk` | `cfg.chkfile` (default `True`) | PySCF checkpoint (DM, mol, energies) |
| `<job>_initial.xyz` | `cfg.save_initial_xyz` (default `True`) | the user's actual input geometry, snapshotted **immediately after** `gto.M(...)` builds the original molecule — *before* any optimization runs. |
| `<job>_optimized.xyz` | `cfg.save_optimized_xyz` AND `cfg.optimize` | final relaxed geometry, written at end of script |
| `<job>_geom_optim.xyz` | `cfg.optimize` AND `cfg.write_trajectory` AND `cfg.optimizer == "geometric"` | streaming production-stage trajectory, multi-frame XYZ with `Iteration K Energy E` comment lines, **one frame per accepted geometry step** |
| `<job>_geom.log` | same as above | geomeTRIC's own log for the production stage |
| `<job>_preopt_optim.xyz` | `cfg.preopt` AND `cfg.write_trajectory` AND `cfg.optimizer == "geometric"` | streaming pre-opt-stage trajectory, same format |
| `<job>_preopt.log` | same as above | geomeTRIC's own log for the pre-opt stage |
| `<job>.molwatch.log` | `cfg.molwatch_log` (default `True`) AND `cfg.optimize` AND `cfg.optimizer == "geometric"` | unified per-step trajectory log written **alongside** the standard outputs (additive). One marker-delimited block per accepted opt step containing coordinates, total energy (eV), per-atom forces (eV/Å), max force (eV/Å), and the SCF cycle history for that step. Single-file input for molwatch. |

The script's header docstring "Outputs:" block must list **exactly**
the set of files this table promises for the active config — no
under- or over-promising.

---

## Logging contract

All PySCF runtime output (SCF iteration tables, gradient banners,
geometry-step summaries) goes to `<job>.log`.  Specifically:

- The original `gto.M(..., output=JOB+".log", ...)` call opens
  `<job>.log` in append-friendly mode.
- The pre-opt stage runs on `mol_pre = mol.copy()`, which inherits
  the open file handle.  Pre-opt SCFs and gradients land in `<job>.log`.
- After pre-opt completes, the script transitions to the production
  stage.  **It must NOT call `gto.M(output=JOB+".log", ...)` a second
  time** — that call opens the file in `'w'` mode and truncates the
  pre-opt log entries.
- The transition must reuse `mol_pre` (which has the right `stdout`)
  and only call `mol.build()` if the production basis differs from
  the pre-opt basis.  This keeps the same open file handle alive.
- The production stage's `mf = dft.RKS(mol)` therefore inherits the
  same `stdout`, and its SCFs land in the same `<job>.log`.

The `print(...)` statements that announce stage banners (`"=== Stage:
pre-optimization ==="`, etc.) go to the user's terminal (stdout) by
design — they're for the user watching the run, not for the log.

### Forbidden patterns in the generated script

The following code shapes break the logging contract and MUST NOT
appear in any generated script:

1. A `gto.M(...)` call between pre-opt's `optimize(mf1, ...)` and
   the production `mf = ` line.  Reason: truncates `<job>.log`.
2. A bare `mol.basis = "..."; mol.build()` without `dump_input=False`
   — it would re-echo the input file into `<job>.log` redundantly.
3. Any explicit reassignment of `mol_pre.stdout` or `mol.stdout`.
   Reason: PySCF's internal logging routes through that handle and
   relies on it staying open.

---

## Trajectory contract

When `cfg.write_trajectory=True` and `cfg.optimizer=="geometric"`,
**every** call to geomeTRIC's `optimize(...)` in the generated script
must include a `prefix=` kwarg, so geomeTRIC streams a trajectory
file the user can tail in molwatch.

The naming convention is per-stage:

- pre-opt: `prefix=JOB + "_preopt"` → streams `<job>_preopt_optim.xyz`
- production: `prefix=JOB + "_geom"` → streams `<job>_geom_optim.xyz`

This guarantees each stage's trajectory is separately watchable, and
the user can pick which stage to follow by which file they point
molwatch at.

---

## Optimizer contract

- `cfg.optimizer="geometric"` (default) requires the `geometric` PyPI
  package.  The generated script imports it inside a `try/except
  ImportError` that raises `SystemExit` with an actionable message
  ("`pip install geometric`"), not a 6-frame traceback.
- `cfg.optimizer="berny"` requires `pyberny`.  Same `try/except`
  contract.
- `cfg.optimize=False` produces a single-point script: `mf.kernel()`
  is called, no `optimize(...)`, no trajectory files.

### Pre-opt failure must not kill the production run

The pre-opt stage exists to clean up obvious geometry sins before
the expensive functional starts.  By design it doesn't have to
fully converge — its convergence threshold is intentionally looser
than the production stage.

The pre-opt `optimize(mf1, ...)` call therefore **MUST** pass
`assert_convergence=False`.  Without it, PySCF raises
`RuntimeError` if pre-opt hits its `maxsteps` without converging,
and the production stage never runs.

The production-stage `optimize(mf, ...)` call **MUST NOT** set
`assert_convergence` (so the default True applies).  We DO want to
hear about real production-run failures — those affect the data the
user is trusting.

## Spin / method compatibility

PySCF's `RKS` and `RHF` are restricted methods that assume
`mol.spin == 0` (closed-shell).  Setting `cfg.spin != 0` with a
restricted method is a physics error: PySCF will raise at SCF setup,
but only after the user has invoked Python.

The generator **MUST** raise `ValueError` at script-generation time
if `cfg.method` is `RKS` or `RHF` AND `cfg.spin != 0`, with a
message that points the user at `UKS` / `UHF`.

For `UKS` / `UHF`, any `cfg.spin >= 0` is accepted.  The user is
responsible for matching `cfg.spin` to the actual multiplicity of
the system.

---

## Unified molwatch log contract

`<job>.molwatch.log` is an **additive** file: emission of every
other standard output file (`.log`, `.chk`, `_geom_optim.xyz`,
`_geom.qdata.txt`, ...) is unaffected by `cfg.molwatch_log`.
Setting `cfg.molwatch_log = False` simply suppresses the additional
file; nothing else changes.

### Format

Each step is one marker-delimited block.  Markers are literal
strings the parser locates by prefix match — no positional
fragility, no dependency on column widths.

```
# molwatch trajectory log v1
# generator: molbuilder/pyscf_input
# engine: pyscf
# job: <job_name>
# units: energy=eV, force=eV/Ang, coords=Ang
# created: <ISO8601 local timestamp>

==== molwatch step 0 begin ====
step_index: 0
n_atoms:    <K>
coordinates (Ang):
   <element>   <x>   <y>   <z>
   ...
energy (eV): <E>
forces (eV/Ang):
   <element>  <fx>  <fy>  <fz>
   ...
max_force (eV/Ang): <Fmax>
scf_history begin
#  cycle    energy(eV)    delta_E(eV)    gnorm(eV/Ang)    ddm
       <c>     <e>           <de>           <g>            <d>
   ...
scf_history end
==== molwatch step 0 end ====
```

### Unit conventions

The emitter performs all unit conversions at write time so the file
is unit-self-consistent and the molwatch parser does **zero**
conversion:

- coordinates: Angstrom
- energy: eV (Hartree → eV via 27.211386245988)
- forces, gradient norm: eV/Å (Hartree/Bohr → eV/Å via 51.42208619)
- ddm: dimensionless

The header `# units:` line documents this.

### Live-tail safety

The `==== molwatch step <N> begin ====` / `==== ... end ====`
bracketing makes torn-EOF detection trivial: a block with `begin`
but no matching `end` is the in-flight current step and must be
dropped on parse.  This guarantees molwatch never displays a
half-written final step while a run is still going.

The emitter `flush()`es after each end-marker so the file's last
complete byte is always at a step boundary.

### Initial-state preview block (step 0)

The emitter writes **step 0 immediately at instantiation**, before
the first SCF runs.  Step 0 is the *initial-state preview*: it
carries the molecule's coordinates, but `energy (eV): None`,
`max_force (eV/Ang): None`, an empty `forces` section, and an
empty `scf_history`.  A `kind: initial_preview` line distinguishes
it from real opt-step blocks.

Why: molwatch must be able to render the molecular structure the
moment a user loads the log -- they should not have to wait for the
first SCF to finish.  Without this preview block, a user pointing
molwatch at a freshly-started run sees nothing for tens of seconds
or longer (the first SCF is the slowest, by far).

Real opt-step blocks therefore start at step 1.  The first opt
step's geometry (in PySCF/geomeTRIC, `calc_new` is called with the
initial coordinates first) coincides with step 0's geometry; this
is intentional duplication, not a bug, and lets molwatch's plots
show their first energy/force data point at index 1 while index 0
just renders the molecule.

### Hook wiring

The emitter is driven by two hooks on existing PySCF / geomeTRIC
extension points — no monkey-patching:

- `mf.callback = _molwatch.scf_cycle_hook` — fires per SCF cycle
  with `cycle`, `e_tot`, `last_hf_e`, `norm_gorb`, `norm_ddm` in
  the `envs` dict.  The hook accumulates a per-cycle list,
  resetting on `cycle == 0` (which marks a fresh SCF run).
- `optimize(..., callback=_molwatch.opt_step_hook)` — fires per
  accepted opt step with `mol`, `energy`, `gradients` in `envs`.
  The hook flushes one block to the file using the SCF cycles
  accumulated since the previous opt step.

These extension points are documented in PySCF's geometric_solver
(`callback=` kwarg on `optimize()`) and in the SCF kernel
(`mf.callback`).  No fragile internals are touched.

---

## Charge contract

The atom count, charge, and spin in the generated `gto.M(...)` call
must match what `_resolve_charge(struct, cfg)` returns:

- If `cfg.charge` is not None, it wins (including `cfg.charge=0`).
- Otherwise, fall back to `chemistry.formal_charge_from_phosphates(struct)`,
  which counts deprotonated phosphate non-bridging oxygens and adds
  -1 per bare oxygen.

Charged side chains (Asp, Glu, Lys, Arg, His) are NOT counted by the
heuristic and the user must override via `cfg.charge`.

---

## Versioning rules

A change to this spec is a minor-version bump (0.x → 0.x+1) at
minimum, unless it's purely additive (new optional config field,
new optional output file).  Removing a previously-promised output
file or changing its filename is a **major-version** bump.
