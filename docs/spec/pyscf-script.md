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
| `<job>_initial.xyz` | `cfg.save_initial_xyz` (default `True`) | input geometry, written at end of script |
| `<job>_optimized.xyz` | `cfg.save_optimized_xyz` AND `cfg.optimize` | final relaxed geometry, written at end of script |
| `<job>_geom_optim.xyz` | `cfg.optimize` AND `cfg.write_trajectory` AND `cfg.optimizer == "geometric"` | streaming production-stage trajectory, multi-frame XYZ with `Iteration K Energy E` comment lines, **one frame per accepted geometry step** |
| `<job>_geom.log` | same as above | geomeTRIC's own log for the production stage |
| `<job>_preopt_optim.xyz` | `cfg.preopt` AND `cfg.write_trajectory` AND `cfg.optimizer == "geometric"` | streaming pre-opt-stage trajectory, same format |
| `<job>_preopt.log` | same as above | geomeTRIC's own log for the pre-opt stage |

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
