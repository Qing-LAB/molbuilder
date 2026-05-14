# Migration Guide — continuing molbuilder on a new machine

State as of merge `79dd3e1` on `main` (post-Spectra-tab v1 landing).
This document is the one place a new machine needs to read to pick
up exactly where the previous machine left off, including the
context an AI assistant needs to be useful.

---

## 1. What's on disk that matters

Two stores must travel together:

1. **The git repo** — code, tests, design docs, references.bib.
   Everything is on `origin/main` after the v1 merge.
2. **Claude Code's auto-memory** — non-code project context (author
   conventions, architectural principles, dependency licensing
   constraints, "design.md is source of truth" rule).  This lives
   *outside* the repo and is the thing that won't transfer
   automatically.

If you only carry the repo, the new Claude session will be capable
but will not know the project conventions until you teach it again.
**Carry both.**

### 1.1 Repo state

* Default branch: `main` (currently at `79dd3e1`, the no-ff merge of
  `feature/raman-spectra` which delivered Spectra v1).
* Preserved branch ref: `origin/feature/raman-spectra` (kept after
  merge for traceability of the 45-commit spectra implementation
  history; safe to delete after the migration if not needed).
* Other branches on origin (all merged or stale; safe to ignore on
  the new machine unless you're auditing): `feature/modify-tab`,
  `merge-molwatch`.

### 1.2 Claude auto-memory location

```
/home/quan_qing/.claude/projects/-mnt-y-GitHub-quantum-simulation-molbuilder/memory/
```

Contents (10 files, ~17 KB total):

* `MEMORY.md` — the index; every persistent feedback / project /
  reference fact has a one-line pointer here
* `feedback_commits.md` — author is `Quan <qqing@asu.edu>`; never
  add Co-Authored-By: Claude trailers
* `feedback_workflow_scope.md` — the GitHub PAT lacks `workflow`
  scope; do not push changes under `.github/workflows/*`
* `feedback_cli_click.md` — click is the canonical CLI library
  (argparse → click conversion already landed in Phase 5)
* `feedback_dataclass_source_of_truth.md` — `molbuilder.Structure`
  and peers are the lingua franca; no parallel metadata in
  CLI/web layers
* `feedback_architecture_layers.md` — small Unix-pipeable
  subcommands; web UI is a thin wrapper over the same Python API
* `feedback_scientific_review.md` — code review must check
  target-platform correctness (SIESTA / PySCF keyword spellings) +
  scientific defensibility of defaults, not just tests-pass
* `feedback_design_doc_first.md` — `docs/design.md` is the source
  of truth; read it before acting, update it in the same commit
  as code that changes a principle / decision
* `feedback_3dna_license.md` — 3DNA is non-commercial, no auto-
  download; missing-tool errors must point at `x3dna.org`
* `project_repos.md` — molbuilder + molwatch are merged; the
  6-phase merge plan is done; design.md is the post-merge source
  of truth

---

## 2. New-machine setup checklist

Run these in order on the destination machine.

### 2.1 Clone the repo

```bash
git clone https://github.com/Qing-LAB/molbuilder.git
cd molbuilder
git log -1   # should land on the v1 Spectra merge commit
```

### 2.2 Copy the auto-memory directory

The memory path is hashed from the project's absolute path.  If
your new machine puts the repo at the same absolute path
(`/mnt/y/GitHub/quantum_simulation/molbuilder` — likely if Windows
+ WSL2 with drive Y: mounted the same way), the hash is identical
and you can drop the directory in unchanged.  If the path differs,
Claude Code will compute a different hash on first run — copy the
contents into the new hash path, **not** the old one.

The old machine ships the memory directory as `claude-memory.tgz`
at the repo root.  It's gitignored — the tarball is a transfer
artifact, not committed history.  Carry it along with the working
tree to the new machine (Y: drive mount, scp, USB — your call).

```bash
# On the old machine: archive the memory dir.  Already done; the
# tarball is at ./claude-memory.tgz at the repo root.
tar czf claude-memory.tgz -C /home/<user>/.claude/projects \
    -mnt-y-GitHub-quantum-simulation-molbuilder/memory

# On the new machine, after Claude Code has been launched once
# (so the projects dir exists):
ls ~/.claude/projects/    # find the auto-created hash dir
# Extract into that hash dir.  If the new machine uses the same
# repo path, the tarball already has the expected directory name:
tar xzf claude-memory.tgz -C ~/.claude/projects/
```

If the hash differs and you'd rather not move files, paste the
contents of `MEMORY.md` into the bootstrap prompt below — Claude
will recreate the memory entries from the indexed summaries.

### 2.3 Conda environment

The transport env is what `pytest` and PySCF runs need.
`environment.yml` doesn't ship with the repo; recreate from these
known-good packages:

```bash
conda create -n transport python=3.12
conda activate transport
conda install -c conda-forge ambertools     # provides tleap (DNA backend)
pip install -e .                            # editable molbuilder install
pip install pyscf pyscf-properties pytest playwright plotly
pip install gpu4pyscf-cuda12x cupy-cuda12x  # optional, GPU acceleration
playwright install chromium                 # for /modify E2E tests
```

Versions known to work:

* pyscf 2.13.0
* pytest 9.0.3
* Python 3.12

### 2.4 Optional dependencies

* **3DNA `fiber`** — canonical helix builder.  Restricted licence
  (non-commercial); molbuilder does not auto-download.  See
  `docs/design.md` § "Backend roadmap → 3DNA installation" for the
  Windows / WSL2 install steps (matches your previous machine's
  setup).
* **Plotly** — needed for the spectrum chart on `/spectra`.
  Bundled via `pip install plotly`; the web app serves it locally
  from `/vendor/plotly.min.js` so the tab works offline.

### 2.5 Verify

```bash
conda activate transport
python -c "from molbuilder.backends import available_backends; print(available_backends())"
# Expect: {'threedna': True/False, 'amber': True, 'rdkit': True}

pytest tests/ --ignore=tests/spectra/test_smoke.py \
              --ignore=tests/test_modify_e2e.py -q
# Expect: ~1220 passed in ~3 min

pytest tests/spectra/test_smoke.py -m smoke -q
# Expect: 6 passed in ~4 min (runs PySCF on water + HCl)

pytest tests/test_modify_e2e.py -q
# Expect: all pass (Playwright/Chromium E2E)
```

---

## 3. Where the project is right now

Read `docs/tabs/spectra/spec.md` for the v1 spec and audit
checklist.  Short version of what's done vs. deferred:

### 3.1 Spectra tab v1 — what shipped

Complete and covered by tests:

* L1 dataclasses: `SpectraConfig` (with `__post_init__` choices
  validation), `SpectraResults`, `ModeData`, JSON wire format with
  atomic-replace writes + NaN/Inf rejection + complex-number
  rejection.
* L2 algorithms: `select_modes` (5 selectors + freq-range filter),
  `validate_selection`, `render_methods_md` (citation-aware Methods
  prose with section-suffix stripping + comma-list expansion).
* L2 engine: `PySCFSpectraEngine` with preflight advisories
  (scientific caveats from spec § 9.5), GPU compute-capability
  check (≥ 7 = Volta+), methods fragment hook.
* L2 script template: emits a runnable `<job>.spectra.py` with
  inline scientific commentary + verified citation keys + atomic
  JSON checkpoint writes + GPU runtime probe with CPU fallback.
* L3 web: `/spectra` page, schema-driven form, spectrum chart
  (Lorentzian envelope + sticks), mode list (sort/filter/CSV
  export, accessibility), 3Dmol mode-animation viewer (with
  verified eigenvector amplitudes), ES panel, live-watch poller,
  plain-language section descriptions.
* Tests: 369 unit + 6 smoke + targeted web-blueprint tests under
  `tests/spectra/` (one file per topic).
* Physics: mass-weighting bug (caught + fixed in
  `2837631`); polarizability API misuse (fixed); SCHEMA_VERSION /
  MOLBUILDER_VERSION drift (fixed); `_build_mf_at` scoping bug
  (fixed).

### 3.2 Spec gaps to address next

In rough priority order (see also `docs/tabs/spectra/spec.md` § 13):

1. **Playwright `/spectra` tests** (spec § 12.3) — zero E2E
   coverage today; the tab has ~1800 lines of JS, biggest risk.
2. **Verify `freq_min_cm1` / `freq_max_cm1` form widgets work in
   the UI** (spec § 8.1) — fields are in the dataclass and the
   selector honours them, but UI reachability hasn't been
   confirmed end-to-end.
3. **Stepper UI** (spec § 2.5.5) — currently 3 passive phase-status
   dots; spec calls for a clickable 4-step navigator that opens
   per-layer form subsets with a "discard downstream" confirm.
4. **"Columns…" menu** (spec § 9.2.2.1) — mode-list column
   visibility toggle so advanced columns (ΔHOMO/ΔLUMO) don't
   bloat the default view.
5. **Live Methods-preview modal** (spec § 9.4) — currently
   post-generate only; spec says it should update live as form
   values change.

### 3.3 Deliberately out of scope for v1

Per spec § 13 and the user's explicit decisions:

* **IR intensities** (`compute_ir`) — schema reserved, no code.
* **SIESTA engine** — slot reserved in `SpectraConfig.engine`
  enum, no implementation.
* **`molbuilder spectra methods <rundir>` extractor CLI**.
* **Cost preview** (spec § 9.3) — user declined.
* **Stepper UI v2 / 3Dmol bundling** — declined as
  hyperengineering this session.

### 3.4 Cross-tab follow-ups (not Spectra-specific)

Tracked in `docs/design.md` but not blocking:

* SIESTA help-string sweep (same plain-language pass that landed
  on Spectra in `30c9194` hasn't run on `SiestaConfig`).
* `_form_section_descriptions` on `PySCFConfig` + `SiestaConfig`
  (Spectra has them; older configs don't).
* **Transport tab** — v1 scope locked in `docs/design.md`
  decisions log (2026-05-11 entry); no code yet.

---

## 4. Bootstrap prompt for the new machine's Claude session

Paste the block below verbatim as the **first message** in a new
Claude Code conversation on the new machine.  It is an active
migration runner — Claude will execute the six steps in order,
asking for confirmation before any destructive or long-running
action, and report status at the end.  No feature work happens in
this session; this is migration verification only.

```
You are continuing the molbuilder project on a new machine.  The previous
machine prepared a complete handoff.  Your job in this conversation is to
execute the migration steps below in order and report the new environment's
state, NOT to start any feature work.

Rules for this session:
- At every step: say what you're about to do, then do it, then state what
  you found.  One short status line per step is enough.
- Read-only operations (reading files, `git log`, `conda env list`, `pytest`)
  proceed without asking.
- Anything that installs software, creates a conda env, or downloads
  packages: STOP first and ask me to confirm.
- If a step fails, report the failure and stop — do not improvise fixes.

STEP 1 — Locate the repo and load context

- Run `pwd` and `git log -1 --oneline` to confirm you're in the molbuilder
  repo at commit 79dd3e1 (the Spectra v1 merge) or later on main.
- Read `docs/MIGRATION.md` — the single source of truth for this handoff.
- Read `docs/design.md` § "Mission" and § "Design principles" (the rest
  is reference material; don't read end-to-end yet).
- Read `docs/tabs/spectra/spec.md` § 13 "Future extensions" so you know
  the v1 / v1.1+ boundary.

STEP 2 — Restore Claude auto-memory

Claude Code stores per-project memory under a hash of the project's
absolute path:
    ~/.claude/projects/<hash>/memory/

The user manually copied ./claude-memory.tgz to the repo root on this
machine.  The tarball is gitignored (so it never enters the GitHub
remote) but is reachable from the working tree.

- Run `ls ~/.claude/projects/` to find the hash directory for this
  repo (Claude Code created it on first launch).
- Check whether `<that-hash-dir>/memory/MEMORY.md` already exists.
  - If yes: read it, list the indexed entries, report.
  - If no: extract ./claude-memory.tgz:
        tar xzf ./claude-memory.tgz -C ~/.claude/projects/
    The tarball's embedded directory name assumes the old machine's
    path; if the new machine's hash differs, extract into a temp dir
    instead and move the `memory/` subdirectory into the new-machine
    hash dir manually.  Then verify by reading the relocated
    MEMORY.md.
- Report which hash path the memory landed at.

STEP 3 — Verify the conda env (`transport`)

- Run `conda env list`.
- If `transport` exists:
  - Activate it.  Verify: `python -c "import pyscf, pytest, flask; print(pyscf.__version__)"`.
  - Verify `which tleap` finds the env's tleap (Amber backend).
  - Verify `python -c "from molbuilder.backends import available_backends; print(available_backends())"`
    shows `'amber': True`.
- If `transport` does NOT exist:
  - Print the setup commands from MIGRATION.md § 2.3.
  - STOP.  Ask me whether to run them.  Don't install anything without
    confirmation.

STEP 4 — Sanity-check the test suite

Run with the transport env active:

    pytest tests/ --ignore=tests/spectra/test_smoke.py \
                  --ignore=tests/test_modify_e2e.py -q

Expected: ~1220 passed in ~3 min, 0 failures.

If any fail, report the failure list verbatim and stop.  Do not try to
fix them — failures here mean the migration wasn't clean and we need to
diagnose together.

STEP 5 — Sanity-check the auto-memory transfer

Answer these four questions WITHOUT using grep / Read / web search.  Just
recall.  These prove the memory transfer worked end-to-end.

1. Should we add a `Co-Authored-By: Claude` trailer to commits on this repo?
2. Can we modify files under `.github/workflows/` and push them?
3. Should I use argparse for a new CLI in molbuilder?
4. What's the single source-of-truth document I should read before acting
   on this project, per the user's standing convention?

Expected answers: (1) No.  (2) No, the PAT lacks workflow scope.  (3) No,
click is canonical.  (4) docs/design.md.

If any answer is wrong, the auto-memory didn't transfer.  Report which
ones, then read `docs/MIGRATION.md` § 1.2 to restore them by hand.

STEP 6 — Status report

In one short block, tell me:
- Branch + HEAD commit
- Whether auto-memory is loaded
- Conda env status
- Test count + pass/fail
- Memory sanity-check results (4/4 correct, or which failed)
- Anything I need to address before we continue feature work

Then STOP and wait.  The next session decides what we work on; this
session's job is done once you've reported status.
```

If you'd rather skip the active runner and just preload context for an
interactive session, use this minimal version instead — same conventions,
no execution steps:

```
I'm continuing the molbuilder project on a new machine.  Read
docs/MIGRATION.md, docs/design.md (Mission + Design principles
sections), and docs/tabs/spectra/spec.md § 13 before doing anything
else.  Apply the conventions listed in MIGRATION.md § 4 throughout
this session.  Spectra v1 just merged to main; MIGRATION.md § 3.2
lists the next priorities — wait for me to pick one.
```

---

## 5. Sanity-check tags

When you land on the new machine and run the bootstrap, the
assistant should be able to demonstrate it has the context by
answering these without grep:

* "Why don't we have a Co-Authored-By trailer?" → because author
  convention says no
* "Should we modify `.github/workflows/ci.yml`?" → no, PAT lacks
  workflow scope
* "Should I just use argparse for this new CLI?" → no, click is
  canonical
* "What's the next priority on Spectra?" → Playwright /spectra
  tests, then `freq_min/max` UI verification

If any of those answers comes back wrong, the memory transfer
didn't take — paste the relevant `feedback_*.md` content into the
conversation and continue.
