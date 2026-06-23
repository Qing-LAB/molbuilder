# Script-execution contract — warm-restart, cold-restart, project-id consistency

**Audience**: anyone implementing or modifying a script generator
(`molbuilder/siesta/input.py`, `molbuilder/pyscf/input.py`,
`molbuilder/transport/*.py`, future engines) and the corresponding
runwrap (`molbuilder/runwrap.py`).

**Captured 2026-06-20** from the PDT-incident chain of questions about
how SIESTA `--continue` / `--cold` behave vs. PySCF.  The contract
below is what we want to be **consistently true across all engines**;
when a generator violates it, that's a bug, not a feature.

---

## The four behaviors that must be consistent across engines

For every engine molbuilder generates a script + runwrap for:

| Behavior | Contract |
|---|---|
| **Project ID** | Every generated script declares its project ID in a single literal field.  This ID keys all warm-restart files: `<ID>.<ext>`.  SIESTA's `SystemLabel` directive in `.fdf` and PySCF's `JOB = "..."` literal in `.py` are the two forms. |
| **Warm-restart, auto** | When the script runs and a warm-restart file (named by the project ID) exists in the working directory, the engine MUST pick it up and resume.  No user-visible flag needed.  Absent files = clean cold start. |
| **`--continue` flag** | Identical to "auto" except it asserts the warm files MUST be present (errors loudly if they aren't, instead of silently cold-starting).  Used when the user knows a prior run produced warm state and doesn't want a surprise cold restart. |
| **`--cold` flag** | Forces a clean start regardless of what warm files are on disk.  Moves them aside (NOT deletes — never auto-delete the user's prior results) into `<basename>-restart-aside-<UTC-timestamp>/`.  The MUST cover ALL files that the warm-start branch reads, or `--cold` silently leaks prior state. |

These four behaviors are the user-visible contract.  Internal
implementation differs per engine (SIESTA pushes the warm-restart
into the SIESTA binary via `MD.UseSaveDM T` etc.; PySCF puts the
warm-restart into the generated Python script via an `if exists →
init_guess = "chkfile"` block).  Different machinery, same surface.

---

## Per-engine warm-start file inventory

These tables enumerate the files each engine's warm-restart branch
reads.  Every file in the right column MUST be in the `--cold`
move-aside glob for that engine (`runwrap.py:_cold_restart_aside_block`).

### SIESTA

| Warm-restart file | What it carries | Required by |
|---|---|---|
| `<SystemLabel>.DM` | density matrix | `DM.UseSaveDM T` (default on warm-restart) |
| `<SystemLabel>.CG` | CG optimizer state | `MD.UseSaveCG T` |
| `<SystemLabel>.XV` | coords + velocities | `MD.UseSaveXV T` (default on for relaxations) |
| `<SystemLabel>.LWF` | Wannier functions | when `Wannier90` post-proc is in use |
| `<SystemLabel>.ZM` | Z-matrix | rarely used |
| `<SystemLabel>.Bonds` / `.PARTIAL` / `.EIG` | bond cache / partial sums / eigenvalues | various |
| `<SystemLabel>.HSX` | Hamiltonian + overlap | TranSIESTA NEGF restart |
| `<SystemLabel>.WFSX` | saved wavefunctions | `SaveWaveFunctions T` + post-proc |
| `<SystemLabel>.STRUCT_NEXT_ITER` | next-iter geometry | `MD.UseStructFile T` (default true for relaxations) |
| `<SystemLabel>.TSHS` | TranSIESTA self-energy H | TranSIESTA electrode reuse |
| `<SystemLabel>.TSDE` | TranSIESTA density matrix (NEGF) | TranSIESTA |

Pre-2026-06-14 the `--cold` glob missed HSX / WFSX / STRUCT_NEXT_ITER
/ TSHS / TSDE — silent state-leak bug fixed via task #438 / commit
G3.  Lesson: **anytime a generator adds a new warm-restart hook,
its `--cold` glob entry MUST land in the same commit.**

### PySCF

| Warm-restart file | What it carries | Read by |
|---|---|---|
| `<JOB>.chk` | SCF DM (init guess) | `if exists → mf.init_guess = "chkfile"` in script |
| `<JOB>_optimized.xyz` | latest converged geometry | `_atom_block` override before `gto.M(atom=_atom_block, ...)` in the generated script (landed #539, 2026-06-23) |
| `<JOB>_geom_optim.xyz` | geomeTRIC trajectory (frames) | geomeTRIC append-mode hazard if present |
| `<JOB>_geom_optim.tmp` | geomeTRIC temp | geomeTRIC may resume from temp on certain failures |
| `<JOB>_geom.tmp` | geomeTRIC temp | same |

**Landed 2026-06-23 (#539):** the PySCF `--cold` glob now covers
all 5 files in the inventory above (suffix-keyed, with braced
`${_warm_label}` expansion to handle underscore-prefixed
suffixes safely).  Pinned by
`tests/test_runwrap.py::test_pyscf_cold_aside_block_covers_full_warm_restart_inventory`
+ an end-to-end bash test that plants all 5 files and confirms
`--cold` moves every one of them into the dated aside dir.  Going
forward, **any new PySCF warm-restart hook MUST add its suffix to
the inventory table above AND to `_PYSCF_WARM_RESTART_INVENTORY`
in the test file** — the parity test catches a hook that lands
the read-side but forgets the `--cold` glob entry.

### TranSIESTA / transport engines

Inherits the SIESTA inventory PLUS `.TSHS` + `.TSDE` electrode/NEGF
files.  Already covered in SIESTA glob (commit G3, 2026-06-14).

---

## Project ID extraction (the runwrap-side ID lookup)

For `--cold` to find the right files, the runwrap needs to know the
project ID — which is INSIDE the script, not the basename of the
wrapper file.  `<basename>-stage2.fdf` may carry `SystemLabel
foo` (not `foo-stage2`).  Same for PySCF.

The runwrap reads the ID from inside the script at runtime:

* **SIESTA**: `awk` for the `SystemLabel` line in `<basename>.fdf`.
  Pre-2026-06-14 the runwrap globbed against the wrapper basename
  only, missing every staged-relaxation project; fixed via the
  SystemLabel-extract block in `_cold_restart_aside_block`.

* **PySCF**: `awk` for the `JOB = "..."` line in `<basename>.py`.
  The char class must match both `"` and `'`; the canonical
  bash-side escape pattern uses awk's octal `\047` for `'` (post
  2026-06-20 fix in commit 5a8fed1, after the bash SQ-escape
  malformed `-F'["\\'"]'` shipped → unterminated DQ).

Both extractors fall back to the wrapper basename when the project
ID can't be parsed, AND sanitize the ID to `[A-Za-z0-9._-]` before
interpolation to block shell-injection via a hostile script value.

The `--cold` glob uses BOTH the ID-derived name and the wrapper
basename so a project with `SystemLabel == basename` is also covered.

---

## Generator-side warm-restart contract

Every script generator must implement the four behaviors above
**in the script it generates** (not in the runwrap).  The runwrap
only handles `--cold` move-aside + the `--continue` insistence
checks; the actual "read warm-state if present" lives in the
script body so the script remains self-contained at runtime.

### Required code in every generated script

1. **Project ID declaration** at the top, as a single literal:

   ```fdf
   SystemLabel  pdt-mol      # SIESTA
   ```
   ```python
   JOB = "pdt-mol"           # PySCF
   ```

2. **Warm-restart for SCF/electronic state** — auto, no flag:

   SIESTA: pass through (SIESTA binary reads `.DM` etc. on its own
   when the file is present).

   PySCF: explicit `if exists → init_guess = "chkfile"` block in
   the generated script.  Example (from existing PDT script):

   ```python
   mf.init_guess = "minao"
   mf.chkfile = _mb_outfile(JOB + ".chk")
   _chk_path = _mb_outfile(JOB + ".chk")
   if _os.path.exists(_chk_path) and _os.path.getsize(_chk_path) > 0:
       mf.init_guess = "chkfile"
       print(f"[molbuilder] continuation: loading SCF init guess from {_chk_path}")
   ```

3. **Warm-restart for geometry** — auto, no flag:

   SIESTA: pass through (SIESTA reads `.XV` automatically when
   `MD.UseSaveXV T` is set, which is the default for relaxations).

   PySCF: explicit XYZ-parse-and-override block in the generated
   script.  Example (from PDT script, will be auto-emitted after
   #539):

   ```python
   _atom_block = '''  ...literal coordinates... '''
   _opt_path = _mb_outfile(JOB + "_optimized.xyz")
   if _os.path.exists(_opt_path) and _os.path.getsize(_opt_path) > 0:
       # parse N atoms from <JOB>_optimized.xyz, format as atom-block lines,
       # override _atom_block; print "[molbuilder] continuation: loaded
       # geometry from <path> (N atoms)" so user sees what happened
       ...
   mol = gto.M(atom=_atom_block, ...)
   ```

4. **Fall-through to cold start** when warm files absent — both
   engines do this by definition (the warm-restart branches are
   guarded by `if exists`).

### Required code in the runwrap

The runwrap's responsibilities:

* `--continue` / `-f` / `--force` flag parsing (bash arg loop).
* `--cold` / `--from-scratch` flag parsing + the move-aside block
  in `_cold_restart_aside_block(basename, engine=...)`.
* Project-ID extraction (`awk` for `SystemLabel` / `JOB`) with the
  sanitizer guard.
* Status banner that reports MODE (`COLD` / `WARM-RESUME` / etc.)
  before the engine starts so the user sees what's about to happen.

### Required tests

For each engine, the test suite must cover:

| Test | What it pins |
|---|---|
| Render a wrapper + run `bash -n` on it | template syntax (added 2026-06-20 after PDT incident; gates the whole render path) |
| Render a wrapper + check `--cold` aside-glob covers EVERY file in the warm-restart inventory | catches the 2026-06-14 (SIESTA) and 2026-06-20-pending (PySCF) state-leak class |
| Render a script + check that the warm-restart `if exists → ...` block is present and references the project-ID-derived file path | catches generator-template regressions where a warm-restart branch goes missing |
| End-to-end: write a fake warm file → render + run → assert "loaded from <path>" appears in stdout | catches the "branch present but never fires" class |

---

## Cross-engine equivalence table (user-visible)

When a user reads the runwrap status banner, the wording should be
engine-agnostic.  Same labels mean the same thing:

| Banner mode | Meaning (same for both engines) |
|---|---|
| `initial-run (clean state)` | No warm files present; running cold from script literal |
| `WARM-RESTART (silent; engine will load existing <files>)` | Warm files present, no flag passed; auto-warm-resume |
| `WARM-RESUME (--continue; engine will load <files>)` | `--continue` passed + warm files present |
| `WARM-RESUME REQUESTED but no prior state found -- starting cold by necessity` | `--continue` passed but no warm files; degraded to cold |
| `COLD (--cold; warm-start files moved aside)` | `--cold` passed; warm files moved to `-restart-aside-<UTC>/` |

The user shouldn't need to know whether they're running SIESTA or
PySCF to understand the banner.

---

## Open implementation gaps (as of 2026-06-23)

| Gap | Task | Severity |
|---|---|---|
| PySCF generator doesn't include basis-tier / functional-tier / convergence-tier guidance comments | _spinoff (not yet assigned)_ | Discoverability — users default to def2-SVP and don't realize it's screening-only.  The convergence-tier guidance now lives in [`engines/optimization-tuning.md`](../engines/optimization-tuning.md); the gap is wiring it into the generator's verbose-comment block. |

**Closed 2026-06-23**:

* **#539 (PySCF `--cold` glob + geometry warm-restart hook).**  Two
  pieces shipped in one commit per the rule above: the generator
  emits the `_atom_block` literal-override block before `gto.M()`
  (auto-resumes from `<JOB>_optimized.xyz` when present), and the
  runwrap `_cold_restart_aside_block` covers the full 5-file
  inventory.  Pinned by 6 new tests across `test_pyscf.py`
  (generator) and `test_runwrap.py` (runwrap) including an
  end-to-end bash test that plants all 5 files and verifies the
  move-aside.  See `_PYSCF_WARM_RESTART_INVENTORY` in
  `tests/test_runwrap.py` for the authoritative inventory tuple.

* **#534 (PySCF staged auto-optimization).**  Per-stage convergence
  ladder shipped via the `cfg.stages: List[StageSpec]` data model,
  emitted as `STAGES = [...]` + `for STAGE in STAGES:` in the
  generated script with inter-stage warm-start via `mf.reset(mol_eq);
  mf.kernel(dm0=dm_prev)`.  See decision-log 2026-06-22 in
  [`design.md`](../design.md) for the full design rationale.

---

## Sibling docs

* `docs/engines/siesta.md` — SIESTA script-output spec
* `docs/engines/pyscf.md` — PySCF script-output spec
* `docs/engines/transport.md` — TranSIESTA-specific guidance
* `docs/engines/pyscf-publication-guide.md` — parameter tiers + staged-PySCF design + methods-section template
* `docs/protocols/script-contract.md` — the script-contract reserved blocks (HEADER / PROVENANCE / BENCH-MARKS / ATOM-METADATA / USER-CUSTOM)
* `molbuilder/runwrap.py:_cold_restart_aside_block` — the canonical implementation of the `--cold` move-aside logic
