# Roadmap

> **This document is the sole source of truth for molbuilder's
> backend / feature roadmap.**  Pointer in `design.md` § 0.
>
> Closed items move into the `design.md` Decisions log (cross-
> cutting) or the relevant per-tab / per-protocol decisions log
> (subsystem-scoped).  Open items live here until they ship.
>
> Tab UI reorganization (Phase A/B/C/D, 2026-06-06) is tracked in
> `docs/tabs/architecture.md`, not here.

---

## 1. 3DNA (canonical helix builder)

3DNA's `fiber` command produces true B-form / A-form / Z-form
helical geometry — the only thing the existing `rdkit` (folded
conformer) and `amber` (extended chain) backends do not provide.

### 1.1 Licensing and distribution constraints

**3DNA is not auto-installable, and molbuilder must not attempt
to fetch it.**

- 3DNA is distributed by the Olson lab (Columbia University)
  through `http://x3dna.org/` behind a **registration form**
  that requires the user to accept the license.  The archive is
  not on a public mirror and cannot be obtained via `pip`,
  `conda`, `wget`, or any automated fetcher driven by
  molbuilder.  Users **must** download it themselves by
  following the instructions on x3dna.org.
- The 3DNA license is **non-commercial-use only**.  molbuilder
  itself is MIT-licensed; bundling, redistributing, or
  auto-mirroring 3DNA would drag the molbuilder distribution
  under 3DNA's restricted terms.  We do neither, and shipped
  CI / docs / examples never invoke a fetch.
- The `x3dna-*.tar.gz` and `x3dna-*.zip` patterns in
  `.gitignore` exist for both reasons (a) keep the binary
  archive out of git on developer machines and (b) make it
  structurally hard for someone to accidentally commit a 3DNA
  archive into a public-facing molbuilder release.
- Documentation (this file, READMEs, error messages) must
  always tell users to **download from x3dna.org per their
  instructions and accept the license** rather than implying
  any automated install path exists.

### 1.2 Backend implementation

**Backend file**: `molbuilder/builders/backends/_threedna.py`,
mirroring the shape of `_amber.py`: shell out to `fiber`, parse
the output PDB into `Structure`, run the backbone-connectivity
self-check (`_common.verify_backbone_connectivity`).

### 1.3 Detection chain

First hit wins; `is_available()` returns True iff any source
resolves to a *complete* install (i.e. both `bin/fiber` executable
AND `config/` directory present).  The chain is:

1. **In-tree** — glob `<repo_root>/x3dna*/`, where
   `<repo_root>` is one level above the molbuilder package.  The
   user can simply unpack the 3DNA tarball at the repo root
   (gitignored — see `.gitignore`) and the backend lights up
   automatically.  Easiest path for a dev install; useless for a
   wheel install (site-packages has no meaningful
   "next-to-package" location), so the env-var fallback exists.
2. **`$X3DNA` env var** — the canonical 3DNA install
   convention.  Set `export X3DNA=$HOME/opt/x3dna-v2.4` and we
   use it.
3. **`fiber` on `PATH`** — last resort; we derive `X3DNA` root
   from `shutil.which("fiber")` (assumes the standard
   `$X3DNA/bin/` layout).  Useful when the user has a system
   install that doesn't bother with the env var.

For **each** candidate root the backend verifies *completeness*:
`bin/fiber` is a regular file with the executable bit set, AND
`config/` exists as a directory (it holds 3DNA's atomic-parameter
PDB templates; without them `fiber` fails at runtime with cryptic
errors).  The completeness check filters out the easy foot-gun
where `$X3DNA` points at a half-extracted tarball or a sibling
directory.

When `fiber` is shelled out, the resolved root is injected into
the subprocess environment as `X3DNA` (and prepended to `PATH`)
regardless of the calling shell's setup, so 3DNA's auxiliary
scripts resolve their config files correctly even when the user
found the install via the in-tree or PATH path rather than the
env var.

If the entire chain fails, `is_available()` returns False **and**
`BackendUnavailable` is raised on explicit `--backend threedna`
requests, with the canonical error message below.

### 1.4 Required error message contract

When the user explicitly requests `--backend threedna` (or any
equivalent in the web UI / Python API) and the backend is
unavailable, the raised `BackendUnavailable` message must include
all of:

- which sources were checked (the three resolution strategies
  above — in-tree glob, `$X3DNA` env var, `fiber` on PATH — and
  their current values, so the user can see exactly what fell
  through);
- the URL `http://x3dna.org/` and an explicit "register and
  accept the license to download — molbuilder cannot fetch this
  for you";
- a one-line reminder that 3DNA is non-commercial-use only;
- the names of the two fallback backends (`amber`, `rdkit`).

Example shape (final wording lives in the implementation, keep
this contract in sync):

```
3DNA is not available.  Tried, in order:
  1. in-tree   : no match for /path/to/repo/x3dna*
                 (unpack the 3DNA tarball at the repo root and this
                 lights up automatically)
  2. $X3DNA    : (unset)
                 (must point at a directory containing bin/fiber + config/)
  3. fiber on PATH: (not on PATH)

3DNA must be downloaded directly from http://x3dna.org/ after
registering and accepting the license — molbuilder cannot fetch it
for you.  The license is non-commercial-use only; do not redistribute
the archive.

If you don't need a canonical helix, the `amber` (extended chain) and
`rdkit` (folded conformer) backends remain available.
```

**Runtime errors during `fiber` execution** (timeout, non-zero
exit, empty PDB, malformed PDB, missing parameter files at
runtime even though `config/` existed at detection time) are
caught and re-raised as `RuntimeError` with the captured
stdout/stderr included verbatim.  Mirrors `_amber.py:96-108` in
spirit.  Do not silently swallow.

**Auto-detect order** in `builders/backends/__init__.py::dispatch`
becomes `threedna > amber > rdkit` (best geometry first).  When
3DNA isn't available the auto path falls through cleanly with no
error — only explicit `--backend threedna` raises.

**CLI / web surface**: existing `--backend` choices (`auto / rdkit
/ amber`) extend to include `threedna`.  The CLI's click
`Choice(...)` and the web UI's `<select>` options must include the
new value.  The web UI's "backend not available" feedback for
`threedna` must surface the same "download from x3dna.org /
non-commercial" guidance — not a bare HTTP 500.

### 1.5 Test coverage required

- `is_available()` returns False with each detection-chain step
  missing (no in-tree dir, env unset, fiber off PATH) without
  raising.
- An env-var path that points at an incomplete install (no
  `config/`) is rejected.
- Explicit `--backend threedna` request when nothing is reachable
  produces a `BackendUnavailable` containing the URL, the
  non-commercial license note, and the named fallback backends.
- `auto` falls through silently when 3DNA is unavailable.
- When an install IS reachable the build produces a chemically
  plausible Structure (P present, expected base residues,
  backbone connectivity passing).
- A-form and B-form coordinates differ (the form flag actually
  plumbs through to fiber).
- RNA build uses U not T (the `-rna` flag is set).

### 1.6 3DNA installation guide

Two install shapes work; pick whichever matches how you use
molbuilder.

**Option A — in-tree (recommended for dev / editable installs).**
Unpack the tarball at the molbuilder repo root.  The detection
chain's first step globs `<repo_root>/x3dna*/` and verifies
completeness, so no shell config or env var is needed:

```bash
cd /path/to/molbuilder              # the repo root, alongside pyproject.toml
tar -xzf x3dna-v2.4-<platform>.tar.gz
ls x3dna-v2.4/bin/fiber             # smoke check
python -c "from molbuilder.backends import available_backends; \
           print(available_backends())"
# expected: {'threedna': True, 'amber': ..., 'rdkit': ...}
```

The `x3dna*/` directory is gitignored (see `.gitignore`) — both
for hygiene and to make it structurally hard for someone to
accidentally commit the 3DNA archive into a public-facing
molbuilder release.

**Option B — system install with `$X3DNA` env var (canonical).**
This is the install path the 3DNA upstream documents; the second
step in the detection chain picks it up:

```bash
tar -xzf x3dna-v2.4-<platform>.tar.gz -C ~/opt
export X3DNA=$HOME/opt/x3dna-v2.4
export PATH=$X3DNA/bin:$PATH
fiber -h
fiber -seq=ATCG /tmp/probe.pdb && head /tmp/probe.pdb
```

The `X3DNA` environment variable is required by 3DNA's auxiliary
scripts; molbuilder's `_threedna.py` injects it into the
subprocess environment automatically when shelling out, so the
env var only needs to be set in the user's shell when they want
to invoke 3DNA tools directly outside molbuilder.

#### Windows install (project-specific)

3DNA's official binary distribution does **not** include a
native-Windows build.  The Linux tarball runs only inside WSL or
Cygwin.  **Recommended path: WSL2 (Ubuntu).**

Concrete install inside WSL2:

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
python -c "from molbuilder.builders.backends import available_backends; print(available_backends())"
# expected (after _threedna.py lands): {'rdkit': True, 'amber': ..., 'threedna': True}
```

Notes specific to running molbuilder from WSL on this host:

- **Run molbuilder from inside WSL,** not from Windows Python —
  only the WSL Python sees `fiber` on PATH and the `X3DNA` env
  var.
- File paths are interchangeable: WSL sees Windows drives at
  `/mnt/<letter>/`, Windows sees WSL files at
  `\\wsl$\Ubuntu\home\<user>\...`.  Generated `.fdf` and `.py`
  files written from WSL are immediately editable from Windows
  tools.
- If you also want molbuilder's CLI from PowerShell, that's fine
  for build subcommands that don't need 3DNA (`peptide`,
  `smiles`, `fdf`, etc.); just don't pass `--backend threedna`
  from the Windows side — it'll fail `is_available()` and the
  user gets a clear `BackendUnavailable` error.

#### Alternative: Cygwin / MSYS2

The Linux tarball usually extracts and runs under Cygwin.  Set
the same env vars in `~/.bashrc` inside the Cygwin shell.  Path
translation is handled by Cygwin automatically.  Less common than
WSL2 these days.

#### Backend behavior when 3DNA isn't installed

`builders/backends/_threedna.py::is_available()` returns False
when `fiber` isn't on PATH or `X3DNA` isn't set.  With
`--backend auto` (default), molbuilder falls through to `amber >
rdkit` cleanly.  With `--backend threedna` explicit, the user
gets a `BackendUnavailable` error citing the missing PATH / env-var
so they know exactly what to fix.

---

## 2. Transport calculation backends (Phase B.3)

Phase B.2 landed the engine abstraction
(`molbuilder/transport/engine_base.py` Protocol + registry +
`TransportResults` dataclass + `TransportConfig` field-metadata
dataclass).  Phase B.3 implements the first concrete engines.

### 2.1 Status

- **transiesta** — TranSIESTA from the SIESTA suite, NEGF +
  LDA/GGA pseudopotentials.  Handles realistic electrode + bridge
  sizes (≤ ~few hundred atoms in the device region).
  **Status: ZERO-BIAS DEVICE `.fdf` SHIPPED 2026-06-10**
  (commits 78547c4 + efb01cd).  See § 2.2 below for what's still
  deferred.
- **pyscf-negf** — PySCF + a custom NEGF self-energy driver,
  Gaussian-basis with hybrid functionals available.  Smaller
  systems (~ tens of atoms in the device region) but
  higher-level XC.  **Status: planned.**  The registry pattern
  proved on transiesta means this is mechanical to add — drop
  `molbuilder/transport/pyscf_negf.py` mirroring transiesta's
  shape, decorate with `@register_engine`, import in
  `transport/__init__.py`.  Endpoint code stays unchanged.

Both engines consume the SAME relaxed geometry + the SAME
`.molstruct.json` sidecar (region labels assigned in the
Molbuilder tab — see `molbuilder/sidecars/molstruct.py` (write) / `molbuilder/parse/sidecars/molstruct.py` (read)).
The two engines' `render_script` methods emit different inputs
from the same TransportConfig + Structure pair.

Future: inelastica integration (electron-phonon-resolved
transmission, IETS).  That's a SEPARATE engine that consumes a
TransportConfig + the `.spectra.json` from the Spectrum-calculation tab; not
in scope for B.3.

### 2.1.1 Bundle-handoff consumption (after the 2026-06-17 Step 3 ship)

- **TranSIESTA generator consumes the bundle's regions for
  electrode/buffer assignment** (task #487).  Once a project tree
  carries a `<stem>.xyz` + `<stem>.molstruct.json` pair from the
  Step-3 PR-E Bundle button, the Transport tab's `.fdf` emitter
  should pull `Structure.regions["L-electrode"]` /
  `Structure.regions["R-electrode"]` / `Structure.regions["bridge"]`
  for `TS.Atoms.Buffer` / `TS.Elecs.*` lines instead of requiring
  the user to retype them.  Contract:
  [`bundle-contract.md`](protocols/bundle-contract.md) § 6
  ("Continuation Optimization tab consumes the bundle as starting
  structure" + "Transport tab consumes the bundle for
  electrode/buffer assignment").  Sequence-wise, this depends on
  Step 3 PR-E (shipped) but not on the screenshot-capture pass
  (which is separable).

### 2.2 What's deferred for transiesta (after the 2026-06-10 ship)

Documented in-tree at
`molbuilder/transport/transiesta.py` module docstring +
memories `project_transport_electrode_bias_workflow.md` and
`project_transport_results_tab_framework.md`:

1. **Electrode `.fdf` generation.**  TranSIESTA needs separate
   `.TSHS` Hamiltonian files per electrode.  Today users
   generate them manually (build the lead geometry on
   `/molbuilder`, emit a SIESTA `.fdf` from
   `/structure-optimization` with `TS.HSFileEnable = T` added by
   hand, run SIESTA, copy the resulting `.TSHS` files into the
   device run dir).  Planned: "electrode wizard" that extracts
   the `L-electrode` / `R-electrode` regions from the labeled
   device structure + emits the matching SIESTA `.fdf`
   automatically.
2. **Bias-scan loop.**  `TransportConfig.bias_voltages_v` is a
   `List[float]` but today's engine emits only the first value
   (with a preflight WARN if `len > 1`).  Planned: for `len > 1`,
   emit one `.fdf` per bias point + a driver shell script that
   loops over them.
3. **`parse_output` + the `<job>.transport.json` schema.**
   `TransportEngine.parse_output` raises `NotImplementedError`
   on transiesta today.  Needs a JSON schema designed first
   (mirror of `.spectra.json`).
4. **`/results` Transport inspector.**  No way to view the
   transmission data in molbuilder today — users open the
   `.TBT.AVTRANS_*` files in an external plotter.  Planned: new
   `lib/inspectors/transport.js` + `_transport_inspector.html`
   partial + Plotly T(E) chart + IV chart (when multi-bias
   data lands).  Tracked in memory
   `project_transport_results_tab_framework.md`.
5. **Full Methods-paragraph generator.**  Today's
   `methods_fragment` is a placeholder; full version lands with
   `parse_output` so it can interpolate actual run parameters.

### 2.3 Wire shape (post-2026-06-10)

The web blueprint adds `POST /api/transport/render` dispatching
via the registry:

```text
POST /api/transport/render
body:   { params: {<TransportConfig field values>},
          structure_path: "/abs/path/to/relaxed.xyz" }
=>      { ok, engine, script, filename, issues, errors }
```

The Transport tab's Generate button reads the structure file
from the sidebar commit channel, POSTs the rendered script back,
and shows it inline with copy + download affordances + a
preflight issues panel.

---

## 3. Closed items (historical)

Roadmap items completed before this doc was created — reconstruct
from git history.  Each was tracked in `design.md`'s decisions
log when it shipped:

- molbuilder + molwatch merge (2026-04-30 → 2026-05-01).
- Frame / Trajectory promotion (Phase 2).
- 3DNA backend land + auto-detect chain (2026-05).
- argparse → click conversion (Phase 5).
- Embed module ship + 5 site migrations (Phase 5a–5g).
- Transport engine abstraction (Phase B.2, 2026-06-05).
- Transport B.3 step 1+2: transiesta engine (zero-bias) + web
  blueprint render endpoint + Generate UI wiring (2026-06-10,
  commits 78547c4 + efb01cd).  Deferred items tracked in § 2.2
  above.
- Makov-Payne charge-correction emit (2026-06-05).
- Structure-inspector hand-off to /molbuilder (2026-06-05; route was /modify).
- Animation/snapshot save-to-project fix + export modal (Phase
  6e, 2026-06-05 + cleanup commits through 2026-06-06).
- **Step 3 PR-A through PR-E: run-bundle handoff** (2026-06-17,
  tasks #485 + #488–#492).  Closes the workflow-continuation gap:
  `POST /api/results/bundle` reads a finished SIESTA/PySCF run dir,
  fuses final coords with the labels the originating script
  carried in its in-body ATOM-METADATA block, and writes a portable
  `.xyz` + `.molstruct.json` pair the next workflow tab loads with
  no copy/paste.  New protocol `bundle-contract.md` (13 sections),
  `web-api.md` § 13, design.md 2026-06-17 entry.  Five audit
  rounds closed ~20 verified findings; 253 targeted-sweep tests
  pass.  Follow-on #487 (TranSIESTA consumes the bundle's regions
  for electrode/buffer assignment) is the trivial next step.

---

## 4. Maintenance protocol

When adding a roadmap item to this doc:

1. State the goal in one sentence.
2. Identify dependencies (what must ship first).
3. Identify the test pin shape (what test fails if the work is
   incomplete).
4. Don't list code-review polish or stylistic cleanup — those
   live in commit messages and PRs, not the roadmap.

When closing a roadmap item:

1. Move the closed item to § 3 with a one-line summary.
2. Add a decisions-log entry to the appropriate doc:
   - Cross-cutting → `design.md` decisions log.
   - Subsystem-scoped → that subsystem's decisions log
     (e.g. `protocols/projects-sidebar.md`,
     `tabs/architecture.md`).
3. Update test pins; remove `xfail` markers if any.
