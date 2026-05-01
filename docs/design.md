# molbuilder — design and roadmap

This document is the durable design reference for molbuilder. It captures
mission, architectural principles, decisions made, and active roadmap items
that span multiple sessions of work. Per-component test contracts live under
[`docs/spec/`](spec/README.md); this document sits above them.

When in doubt about whether to do something, read this file first. When the
file is wrong (the decision changed, the constraint shifted), update it in
the same PR as the code change.

---

## Mission

molbuilder builds 3-D molecular structures from sequence / SMILES / name
input, generates SIESTA and PySCF input files for those structures, and
provides a live trajectory viewer that monitors the resulting calculations.

The package is a single, internally-coherent toolkit covering the full
pipeline:

```
sequence ──► Structure ──► SIESTA .fdf  ──► siesta ──┐
                       └─► PySCF .py    ──► python  ─┴──► .molwatch.log
                                                                 │
                                              ◄──── live watch ──┘
```

Both halves were initially separate repos (`Qing-LAB/molbuilder` for the
build side, `Qing-LAB/molwatch` for the watch side). They are being merged
into `molbuilder` because they share a single file-format contract
(`.molwatch.log v1`), a single core dataclass (`Structure`), and the same
Flask + 3Dmol.js stack. See "Merge plan" below.

---

## Architecture (target post-merge)

Three layers, with the dataclass as the single source of truth at the
bottom and the web UI as a thin convenience portal at the top.

```
┌────────────────────────────────────────────────────────────┐
│  Web UI (Flask + 3Dmol.js)            ← portal              │
│  Build tab + Watch tab                                       │
│  Calls the same Python API the CLI calls. No own logic.      │
├────────────────────────────────────────────────────────────┤
│  CLI scripts                          ← composable Unix tools│
│  molbuilder peptide│dna│rna│smiles│name → Structure file     │
│  molbuilder fdf    [in.xyz/-]            → SIESTA .fdf       │
│  molbuilder pyscf  [in.xyz/-]            → PySCF .py         │
│  molbuilder watch parse <traj>           → JSON frames stdout│
│  molbuilder watch tail  <traj>           → streaming JSON    │
│  molbuilder serve                        → starts the portal │
├────────────────────────────────────────────────────────────┤
│  Python API + Structure dataclass     ← single source of truth│
│  build_*, render_fdf, render_script, parsers (yield Structure)│
└────────────────────────────────────────────────────────────┘
```

---

## Design principles

These are load-bearing. Don't violate without updating this document.

### 1. The `Structure` dataclass is the lingua franca

Every builder yields one. Every output method consumes one. Every parser,
post-merge, yields one (or a sequence of them) per frame. Field metadata
— label, type, default, validation rule, UI hint — lives on the dataclass
field, **not** in parallel registries in the CLI or web layers.

**Why:** A previous custom registry framework was deleted because click
(and dataclass-driven introspection) is the right tool. Three places
declaring the same field metadata (dataclass + argparse + HTML form) is
how silent drift happens. CLI and HTML form should be *generated* from
the dataclass, not maintained in lockstep with it.

### 2. CLI scripts are small, focused, and composable

Each subcommand does one job. They chain through files / stdin / stdout
in classic Unix style. Treat `-` as stdin where it makes sense:

```bash
molbuilder dna ATGC | molbuilder fdf - out.fdf
molbuilder watch tail run.molwatch.log | jq '.energy_eV'
```

Machine-consumable subcommands (`watch parse`, `watch tail`) emit JSON
or NDJSON on stdout. Human subcommands emit text. Status / progress /
warnings always go to stderr so they don't pollute the pipe.

### 3. The web UI is a portal, not a separate product

The UI calls the same Python API the CLI calls. It contains no logic
that isn't trivially also exposed elsewhere. Tabs (Build / Watch)
share the 3Dmol viewer, style controls, atom rendering, and CSS. The
Build tab's "Generate FDF / script" flow drops a "Watch this run"
affordance that pre-fills the Watch tab with the predicted output
path so the user moves naturally from one phase to the next.

UI redesign mandate: concise, easy, visually fluent. Single layout
shell, two views, no duplicated chrome.

### 4. Generated outputs must be both syntactically correct AND scientifically defensible

An FDF that SIESTA accepts but silently produces wrong physics is a
bug, not a feature. A PySCF script that runs but converges to a
broken-symmetry saddle for an open-shell system is a bug.

Code review for this project must include target-platform correctness
checks: are the keywords real? Are the values in scientifically
defensible ranges? Are open-shell / charged / periodic special cases
handled? See "Scientific correctness" below for the validation
requirements and the known gap list.

### 5. Generated outputs are tunable by manual editing

Generated scripts use plain object APIs (no convenience wrappers
that hide what's happening), keep all SIESTA / PySCF configuration
in scope at the natural location, and provide post-processing hook
placeholders for common follow-ups (Mulliken population, dipole
moment, BandLines, PDOS).

Verbose-comments mode (default ON) inlines tuning hints next to
every parameter — the generated FDF / .py is meant to be readable
as a tutorial. Section headers are mandatory; they make `Ctrl-F`
in the file work for someone unfamiliar with the platform.

### 6. Pre-emission geometry validation

Before any FDF or PySCF script is written, run a scientific sanity
pass on the structure + cell. Errors stop emission; warnings print
to stderr but proceed. See "Validation pass" below for the check
list.

### 7. Generated artifacts are self-contained

The generated PySCF script does **not** import molbuilder at runtime.
A user can `scp` the .py to a cluster that has only `pyscf +
geometric` installed and run it. Helper classes (the molwatch
emitter) live as real Python under `molbuilder/pyscf/_runtime/` for
IDE-checkability and unit tests, but their source is pasted
verbatim into the generated script via `inspect.getsource()` so the
generated artifact has no extra imports.

---

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-30 | Merge `Qing-LAB/molwatch` into `Qing-LAB/molbuilder`. molwatch repo archived after merge stabilizes. | Already coupled by file format spec, web stack, and author. Single repo removes drift surface. |
| 2026-05-01 | Top-level package name remains `molbuilder`. | Established name; "watch" is a verb on it. |
| 2026-05-01 | Keep a `molwatch` console-script shim in `pyproject.toml` post-merge. | Zero cost, real friction saved for existing users / scripts. |
| 2026-05-01 | argparse → click conversion deferred to a follow-up PR. | Touching CLI plumbing while moving files makes diffs harder to review. |
| 2026-05-01 | History preservation via `git subtree add`. | Preserves molwatch's commit history with `git log --follow` working. |
| 2026-05-01 | `_MolwatchEmitter` extracted to a real Python file, pasted via `inspect.getsource()`. NOT runtime-imported from generated script. | Keeps generated script self-contained for cluster use; emitter still IDE-checkable and unit-testable. |
| 2026-05-01 | `molbuilder watch parse` / `molbuilder watch tail` are the resolution of issue #81. | Same JSON-over-stdout shape the original handoff was gesturing at, under the unified CLI. |
| 2026-05-01 | 3DNA backend added as `molbuilder/backends/_threedna.py`; auto-detect order becomes `threedna > amber > rdkit`. | True canonical helix; only thing the existing backends do not provide. |

---

## Merge plan

| Phase | Outcome | Status |
|---|---|---|
| 1 | molwatch source pulled in via subtree-merge; final layout in place; tests green; no logic change | not started |
| 2 | Parsers refactored to yield `Structure` per frame; molwatch's ad-hoc atom rep removed | not started |
| 2.5 | 3DNA backend added (`backends/_threedna.py`); registered in dispatch; CLI / web `--backend` extended | not started |
| 2.6 | Pre-emission geometry validation added (`validation.py`); wired into `render_fdf` and `render_script` | not started |
| 3 | Web UI redesigned: Build tab + Watch tab; shared 3Dmol viewer, style controls; clean styling pass; "Watch this run" handoff | not started |
| 4 | `_MolwatchEmitter` extracted to `molbuilder/pyscf/_runtime/molwatch_emitter.py`; pasted into generated script via `inspect.getsource()`; round-trip emitter→parser unit test added | not started |
| 5 | Issue #81 — `molbuilder watch parse` and `molbuilder watch tail` shell-friendly JSON subcommands | not started |
| 6 | v0.4 scientific polish — known-gaps list below | not started |

### Post-merge package layout

```
molbuilder/
  structure.py                # Structure dataclass (parser output type too)
  residues.py
  peptide.py
  nucleic.py
  backends/
    __init__.py
    _amber.py                 # tleap-driven extended chain
    _rdkit.py                 # ETKDG embedded conformer (folded for >6mers)
    _threedna.py              # NEW: canonical B/A/Z-form helix via fiber
    _common.py
  smiles.py, pubchem.py, chemistry.py
  siesta/
    __init__.py
    input.py                  # was siesta.py (render_fdf, convert)
    parser.py                 # was molwatch/parsers/siesta.py
  pyscf/
    __init__.py
    input.py                  # was pyscf_input.py
    parser.py                 # was molwatch/parsers/pyscf.py
    _runtime/
      molwatch_emitter.py     # extracted; pasted into generated script
  molwatch_log/
    __init__.py
    format.py                 # was _molwatch_log.py + shared block writer
    parser.py                 # was molwatch/parsers/molwatch_log.py
  parsers/
    base.py                   # TrajectoryParser ABC (shared)
  validation.py               # NEW: pre-emission geometry checks
  cli.py                      # adds `watch` subcommand group
  web/
    app.py                    # / (build) + /watch routes; tabbed UI
    templates/index.html
    static/{viewer.js, style.css}
tests/
  ...
```

---

## Scientific correctness

### Validation pass (pre-emission)

Runs before `render_fdf` / `render_script` writes any output. Implemented
in `molbuilder/validation.py:validate_geometry(struct, cell, cfg) -> List[Issue]`.
Errors stop emission; warnings print to stderr.

| Check | Severity | Rationale |
|---|---|---|
| min atom-atom distance < 0.3 Å | error | Atoms on top of each other; SCF will diverge |
| min atom-atom distance 0.3 – 0.7 Å | warn | Likely broken structure (failed protonation, bad backend output) |
| atom-to-nearest-image distance < 2 × cell_padding (vacuum case) | warn | Image-image interaction; suggest larger padding |
| cell volume / atom-bounding-volume < 3 | warn | Cell suspiciously tight |
| cell determinant ≤ 0 | error | Left-handed or degenerate cell |
| `kgrid != 1` along axis with extent < 10 Å | warn | k-points along a vacuum direction is wasted |
| `kgrid == 1` along axis with extent > 10 Å (periodic system) | warn | Likely under-converged k-grid |
| net dipole > 1 D in vacuum (no dipole correction) | warn | Image-image dipole; suggest dipole correction or bigger cell |
| atom outside [0, 1) fractional with `wrap_into_cell=False` | warn | Atom in neighbor cell; visualisations will look broken |
| explicit `Spin.Total` set but `spin_polarized=False` | warn | Total-spin pin will be silently ignored |

Reused by both SIESTA and PySCF generators. Unit-tested against fixtures
in `tests/conftest.py`.

### Known SIESTA / PySCF science gaps

Identified during the 2026-05-01 design review. Each should become a
tracked issue and land before v0.4 "scientific polish":

1. **`SpinTotal` keyword in FDF (`siesta.py:587`) is probably not a real
   SIESTA keyword.** SIESTA uses `Spin.Fix true` + `Spin.Total <v>`.
   Verify against the SIESTA manual for the targeted version range and
   fix the emission. (Currently silently ignored by SIESTA's fdf parser
   on a value mismatch.)
2. **`SpinPolarized true` (`siesta.py:579`) is the v4-era keyword.**
   SIESTA v5 prefers single-line `Spin polarized`. Either feature-detect
   or document the targeted SIESTA version range.
3. **No SIESTA dispersion-correction emission.** For organic / biomolecule
   work without a vdW-aware functional, plain PBE / B3LYP underbinds.
   Add a commented-out `%block MM.Potentials` (D2/D3 empirical) template
   when the chosen XC is non-dispersive.
4. **`mf.stability_analysis()` is not auto-emitted for UKS / UHF.**
   Open-shell SCFs can converge to broken-symmetry saddles; without a
   stability check the user gets a non-variational answer with no
   warning. Auto-emit when `method` starts with `U`.
5. **`PAO.EnergyShift 0.02 Ry` default is loose.** Production SIESTA
   work typically uses 0.005 – 0.01 Ry. Tighten the default to 0.01 Ry.
6. **No post-processing block in either generator.** Add a commented-out
   `# --- Post-processing hook ---` placeholder to both with 2-3 common
   follow-ups (SIESTA: `BandLines` / `PDOS`; PySCF: `analyze()` /
   `mulliken_pop()` / `dip_moment()`).
7. **No SIESTA minimum version pinned.** `requirements-runtime.txt`
   doesn't declare a minimum; emitted keywords like `DM.Energy.Tolerance`
   may be silently ignored on old builds. Document the targeted range.
8. **No ECP support for non-def2 bases.** A user with a Pt/Pd structure
   on cc-pVDZ needs a manual `ecp = {...}` block. Lower priority.
9. **`save_optimized_xyz` writes from `mol_eq` (correct), but `mf.e_tot`
   may not match `mol_eq`'s geometry for non-converged opts.** Probably
   a non-issue in practice; flag for awareness.
10. **No `mf.diis_space` / `mf.damp` in `PySCFConfig`.** Hard-SCF
    troubleshooting requires editing the generated script. Mentioned in
    the troubleshooting block; could be exposed as config fields.

### Generated-output style requirements

- **Verbose-comments mode** (default ON) emits inline tuning hints next
  to each parameter plus a troubleshooting block at end of file. Both
  must remain feature-complete through the merge.
- **Section headers** (`# --- Lattice ---`, `#  1. Build the molecule`,
  etc.) are mandatory.
- **Every tunable parameter** appears with its default value visible
  and a comment range (e.g. `# Range 0.001 - 0.5`) rather than hidden
  behind a function call.
- **Post-processing hook placeholders** (commented-out, ready to
  uncomment) belong at the end of every generated script / FDF.

---

## Backend roadmap

### 3DNA (canonical helix builder)

3DNA's `fiber` command produces true B-form / A-form / Z-form helical
geometry — the only thing the existing `rdkit` (folded conformer) and
`amber` (extended chain) backends do not provide.

#### Licensing and distribution constraints

**3DNA is not auto-installable, and molbuilder must not attempt to fetch it.**

- 3DNA is distributed by the Olson lab (Columbia University) through
  http://x3dna.org/ behind a **registration form** that requires the
  user to accept the license. The archive is not on a public mirror and
  cannot be obtained via `pip`, `conda`, `wget`, or any automated
  fetcher driven by molbuilder. Users **must** download it themselves
  by following the instructions on x3dna.org.
- The 3DNA license is **non-commercial-use only**. molbuilder itself is
  MIT-licensed; bundling, redistributing, or auto-mirroring 3DNA would
  drag the molbuilder distribution under 3DNA's restricted terms. We
  do neither, and shipped CI / docs / examples never invoke a fetch.
- The `x3dna-*.tar.gz` and `x3dna-*.zip` patterns in `.gitignore` exist
  for both reasons (a) keep the binary archive out of git on developer
  machines and (b) make it structurally hard for someone to accidentally
  commit a 3DNA archive into a public-facing molbuilder release.
- Documentation (this file, READMEs, error messages) must always tell
  users to **download from x3dna.org per their instructions and accept
  the license** rather than implying any automated install path exists.

#### Backend implementation

**Backend file:** `molbuilder/backends/_threedna.py`, mirroring the shape of
`_amber.py`: shell out to `fiber`, parse the output PDB into `Structure`,
run the backbone-connectivity self-check (`_common.verify_backbone_connectivity`).

**Detection (`is_available()`):** must return True only when **all** of:

1. `fiber` is on `PATH` (`shutil.which("fiber") is not None`);
2. `X3DNA` environment variable is set (`os.environ.get("X3DNA")`);
3. the directory `$X3DNA/config/atomic_*.par` exists (3DNA's geometry
   parameter files; without them `fiber` fails at runtime with cryptic
   errors). A fast `os.path.isdir(os.path.join(os.environ["X3DNA"], "config"))`
   check is sufficient — the absence of the config directory is the
   clearest signal that `X3DNA` points at the wrong place.

If any of those fails, `is_available()` returns False **and**
`BackendUnavailable` is raised with the canonical error message below.

**Required error message contract.** When the user explicitly requests
`--backend threedna` (or any equivalent in the web UI / Python API)
and the backend is unavailable, the raised `BackendUnavailable`
message must include all of:

- which precondition failed (PATH / env var / config dir);
- the URL `http://x3dna.org/` and an explicit "register and accept the
  license to download — molbuilder cannot fetch this for you";
- a one-line reminder that 3DNA is non-commercial-use only;
- the names of the two fallback backends (`amber`, `rdkit`).

Example of the required shape (final wording lives in the implementation,
keep this contract in sync):

```
3DNA is not available: $X3DNA is set but $X3DNA/config/ does not exist
(extraction is incomplete or X3DNA points at the wrong directory).

3DNA must be downloaded directly from http://x3dna.org/ after registering
and accepting the license — molbuilder cannot fetch it for you. The
license is non-commercial-use only; do not redistribute the archive.

If you don't need a canonical helix, the `amber` (extended chain) and
`rdkit` (folded conformer) backends remain available.
```

**Runtime errors during `fiber` execution** (timeout, non-zero exit,
empty PDB, malformed PDB, missing parameter files at runtime even
though config/ existed at detection time) are caught and re-raised as
`RuntimeError` with the captured stdout/stderr included verbatim.
Mirrors `_amber.py:96-108` in spirit. Do not silently swallow.

**Auto-detect order** in `backends/__init__.py:dispatch` becomes
`threedna > amber > rdkit` (best geometry first). When 3DNA isn't
available the auto path falls through cleanly with no error — only
explicit `--backend threedna` raises.

**CLI / web surface:** existing `--backend` choices (`auto / rdkit / amber`)
extend to include `threedna`. The CLI's argparse `choices=` list and the
web UI's `<select>` options must include the new value. The web UI's
"backend not available" feedback for `threedna` must surface the same
"download from x3dna.org / non-commercial" guidance — not a bare
HTTP 500.

**Tests must cover:** `is_available()` returns False with each of the
three precondition failures (PATH missing, env var missing, config
dir missing) without raising; explicit `--backend threedna` request
produces a `BackendUnavailable` containing both the URL and the
non-commercial license note; `auto` falls through silently when 3DNA
is unavailable.

#### 3DNA installation

3DNA is distributed by the Olson lab (Columbia, x3dna.org). The canonical
install on Linux / macOS:

```bash
tar -xzf x3dna-v2.4-<platform>.tar.gz -C ~/opt
export X3DNA=$HOME/opt/x3dna-v2.4
export PATH=$X3DNA/bin:$PATH
fiber -h
fiber -seq=ATCG /tmp/probe.pdb && head /tmp/probe.pdb
```

The `X3DNA` environment variable is required by 3DNA's auxiliary scripts.

##### Windows install (project-specific)

3DNA's official binary distribution does **not** include a native-Windows
build. The Linux tarball runs only inside WSL or Cygwin. **Recommended
path: WSL2 (Ubuntu).**

The archive on this machine is `x3dna-v2.4-linux-64bit.tar.gz` at the
molbuilder repo root (gitignored — see `.gitignore`). Concrete install
inside WSL2:

```bash
# 1. From a WSL2 (Ubuntu) shell.  The Windows path Y:\GitHub\quantum_simulation\molbuilder
#    is reachable from WSL as /mnt/y/GitHub/quantum_simulation/molbuilder.
mkdir -p ~/opt
tar -xzf /mnt/y/GitHub/quantum_simulation/molbuilder/x3dna-v2.4-linux-64bit.tar.gz \
        -C ~/opt
ls ~/opt/x3dna-v2.4/bin/fiber          # smoke check that extraction worked

# 2. Persist the env vars (append to ~/.bashrc):
echo 'export X3DNA=$HOME/opt/x3dna-v2.4'    >> ~/.bashrc
echo 'export PATH=$X3DNA/bin:$PATH'         >> ~/.bashrc
source ~/.bashrc

# 3. Verify fiber works
fiber -h                                      # prints usage
fiber -seq=ATCGATCG /tmp/probe.pdb && \
  head -5 /tmp/probe.pdb                      # prints REMARK lines

# 4. Verify molbuilder picks it up (run from inside WSL)
cd /mnt/y/GitHub/quantum_simulation/molbuilder
python -c "from molbuilder.backends import available_backends; print(available_backends())"
# expected (after _threedna.py lands): {'rdkit': True, 'amber': ..., 'threedna': True}
```

Notes specific to running molbuilder from WSL on this host:

- **Run molbuilder from inside WSL,** not from Windows Python — only the
  WSL Python sees `fiber` on PATH and the `X3DNA` env var.
- File paths are interchangeable: WSL sees Windows drives at `/mnt/<letter>/`,
  Windows sees WSL files at `\\wsl$\Ubuntu\home\<user>\...`. Generated
  `.fdf` and `.py` files written from WSL are immediately editable from
  Windows tools.
- If you also want molbuilder's CLI from PowerShell, that's fine for
  build subcommands that don't need 3DNA (`peptide`, `smiles`, `fdf`,
  etc.); just don't pass `--backend threedna` from the Windows side —
  it'll fail `is_available()` and the user gets a clear
  `BackendUnavailable` error.

##### Alternative: Cygwin / MSYS2

The Linux tarball usually extracts and runs under Cygwin. Set the same
env vars in `~/.bashrc` inside the Cygwin shell. Path translation is
handled by Cygwin automatically. Less common than WSL2 these days.

##### Backend behavior when 3DNA isn't installed

`backends/_threedna.py:is_available()` returns False when `fiber` isn't on
PATH or `X3DNA` isn't set. With `--backend auto` (default), molbuilder
falls through to `amber > rdkit` cleanly. With `--backend threedna`
explicit, the user gets a `BackendUnavailable` error citing the missing
PATH / env-var so they know exactly what to fix.

---

## File format spec

`.molwatch.log v1` — single source of truth post-merge:
`molbuilder/molwatch_log/format.py`. Both the writer (in PySCF input
generation + the standalone preview helper) and the reader (the parser)
read field names from the same place. The format is marker-delimited and
tolerant of truncation (a torn final block on a still-running job is
dropped on parse).

---

## Open questions

- Whether to rename `molbuilder/molwatch_log/` to something more neutral
  (`trajectory_log/`?) post-merge since the format isn't "molwatch's"
  anymore. Defer until after the merge stabilizes.
- Frequency / thermochemistry support in the PySCF script (post-relax
  Hessian + RRHO). Lower priority than the science gap list above.

---

## Process rules

- Any change to the principles or decisions in this document requires
  updating it in the same PR as the code change. A drift between this
  doc and the code is a bug.
- Test contracts (the per-component specs) live under `docs/spec/`. Tests
  must be derivable from those specs without reading the implementation.
  See [`docs/spec/README.md`](spec/README.md) for the rule.
- Code review must explicitly check (a) target-tool correctness for
  generated SIESTA / PySCF outputs and (b) scientific defensibility of
  defaults — not just code quality and tests.
