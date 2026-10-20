# Full code audit — main at 2.24.0, three rounds consolidated

**Role:** audit report
**Domain:** process
**Companions:** [`code-audit.md`](?doc=process/code-audit.md) (the playbook this audit follows: § 1 principles, § 1b review rules R1–R4, § 1c.2 how to report); [`plans/plan.md`](?doc=plans/plan.md) (where open work is tracked; row **AU1** points here); [`testing.md`](?doc=process/testing.md) (the tier-1 lane run below)
**Date:** 2026-10-20 (first round 2026-10-18)
**Audited commit:** `origin/main` **`d77bb564`**, molbuilder **2.24.0**
**Baseline:** the full review of 2026-10-14 at **`83e72b2`** (2.9.4)

This document replaces the 2026-10-18 audit, `code-audit-2026-10-18.md` (renamed to this file; the earlier text is in git history). That audit was at 2.10.2 and re-validated at 2.11.0. This one carries every one of its findings forward to 2.24.0. It also reviews the 43 commits since 2.11.0.

---

## 0. What this is, and how it was done

### 0.1 Lineage

| Round | Commit | Version | What was done | New findings numbered |
|---|---|---|---|---|
| Baseline | `83e72b2` | 2.9.4 | The 2026-10-14 full review: 315 findings | `A<area>-<nn>` |
| 1 | `f3d5a4e4` | 2.10.2 | All 315 re-checked on main; 110 commits reviewed | `A<area>-N<k>` (67) |
| 2 | `0ff77d99` | 2.11.0 | Re-validated against 4 new commits (84 files) | `B<k>-N<j>` (15) |
| **3** | **`d77bb564`** | **2.24.0** | **Every finding still open re-checked; 43 new commits (313 files) reviewed** | **`C<area>-N<j>`** |

Each round validates against the actual code at its commit.

**Status codes:**

| Status | Meaning |
|---|---|
| **FIXED** | the commit that fixed it is named |
| **PARTLY FIXED** | part of the finding is gone; what remains is stated |
| **STILL PRESENT** | still there at the cited place, or where the code moved |
| **MOVED/CHANGED** | still a problem, in a different place or shape |
| **NOT A DEFECT** | the original claim was wrong on re-check |
| **CANNOT VERIFY** | what is missing is named |

### 0.2 Method of round 3

The work was split into the same ten areas as before. One reviewer per area worked on a read-only checkout of `d77bb564`, following the playbook:

- every finding is a candidate until verified (§ 1);
- the silent consequence is stated (§ 1c.2);
- each reviewer names the technique used and what it cannot see (§ 1b R3).

Every reviewer did two things:

1. **Validate.** Every open finding of the area was checked at its place on HEAD. Files that moved were followed.
   - If the cited file is byte-identical to `0ff77d99`, its status was carried over. Each reviewer spot-checked three of these.
   - High items were re-run wherever the code could be run.
2. **Review the new components.** Every file added since `0ff77d99`, and every substantially changed file, was read in full for defects.

**Re-run independently while writing this document** (on `d77bb564`):

- **A7-N1:** a cross-site multipart upload writes `<calc>/.git/config` with `core.fsmonitor`. A cross-site `GET /api/checkpoint/state` then executed the command (a marker file appeared).
- **C4-N1:** `transport/record.py:1161` rebinds `base`.
- **C6-N1:** `notify_setup.py:553-572` builds the reports path from `?user=`.
- **C6-N3:** `monitor.py:1974-1979` re-raises a refused card layout.
- **A5-07:** `canonical_time('30:00')` is `'1-06:00:00'`.
- **C9-N3:** `derived.py:81` computes `1.0 / math.tanh(x)` with x = 0 at T = ∞.

**Labels used below:**

- *(verified)* — reproduced by running code on `d77bb564`.
- *(read)* — confirmed by reading the code at the cited line.
- *(unverified)* — plausible, not demonstrated.

Line numbers are those of `d77bb564` unless stated otherwise.

**Areas:**

| Area | Scope |
|---|---|
| A1 | Spine docs and global architecture (layers, floors, Makefile, tooling) |
| A2 | Data model: `Structure`, cell, chemistry, sidecars, parse stack, frame sets |
| A3 | Engines: SIESTA, PySCF, catalogue, stages, pseudopotentials, validation |
| A4 | Transport (TranSIESTA/TBtrans) and vibrations/spectra, Python side |
| A5 | Execution: prep, run wrappers, launch, checkpoints, JobSets, scheduler, `jobs` |
| A6 | Configuration, environments, serving daemon, auth, rate limiting, notify, report cards |
| A7 | Web backend and HTTP API |
| A8 | Front end 1: MolView, Task Setup, tabs, workspace, projects, form schema, This machine |
| A9 | Front end 2: Results, inspectors, spectra, spectrum chart, VibrationView, trajectory |
| A10 | Test suite, plan, duplicated/dead code, repository hygiene |

### 0.3 What the 43 commits added

| Theme | Commits (examples) | Areas |
|---|---|---|
| **One home per fact:** catalogue as the single home of parameter facts; one writer (`persist.write_bytes`); records through their owner's "door" (`calculation.open`, `task.of`) | `df200c2b`, `1a0ae147`, `9cf38d5d`, `c7fb10c2` | A1, A3, A7 |
| **Multi-frame transport:** frame sets, frame averages, `.nc` data files (`ncdata.py`, `frameset.py`, `frame_average.py`) | `4c17ae7e`, `5867b2db`, `07e03f23` | A2, A4 |
| **Leads' surface Green's functions:** one pair per voltage, reused across frames; `jobset scratch` | `b0e5ee86`, `556cd3c9`, `d77bb564` | A4, A5 |
| **Vibration results:** schema 9, `<label>.spectra.nc`, thermal spread served by the server | `ae4101a5`, `befe9338`, `e5b60a59` | A4, A9 |
| **Scheduler ending:** a run SLURM stopped, past its wall, or forgotten | `5e188b0e`, `00cd5162`, `735d2c27`, `642d8410` | A5 |
| **Notify:** listener reports, report cards as data, the This machine Reports card | `8d15021c`, `aff58812` | A6, A8 |
| **Hand-over and Task setup:** the server proposes the ladder; the page derives nothing | `f40f377d`, `47675bb7` | A7, A8 |
| **Migration:** `jobset migrate` reviewed whole | `b7875bcf`, `abbb2092` | A5 |
| **Review of the project's own lanes:** "every finding validated, the errors fixed" | `825287c2` (2.16.3), `e5b60a59` (2.17.0), `c7353447` | all |

---

## 1. Results at a glance

### 1.1 Status at 2.24.0, by area

These are the findings that were still open at 2.11.0, re-checked on `d77bb564`. A finding that sits in two areas (for example `A6-01 / A7-03`) is counted in both, and B-findings are counted in the area that checked them. So the column sums are a little higher than the totals in § 1.2.

| Area | Checked | Fixed | Partly | Still present | Moved | Not a defect | Cannot verify | **New (C)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 44 | 1 | 3 | 39 | 1 | 0 | 0 | **11** |
| A2 | 43 | 0 | 4 | 38 | 1 | 0 | 0 | **9** |
| A3 | 41 | 1 | 3 | 36 | 0 | 0 | 1 | **6** |
| A4 | 31 | 6 | 1 | 23 | 0 | 0 | 1 | **10** |
| A5 | 38 | 2 | 7 | 29 | 0 | 0 | 0 | **9** |
| A6 | 40 | 1 | 6 | 32 | 1 | 0 | 0 | **7** |
| A7 | 37 | 1 | 3 | 33 | 0 | 0 | 0 | **8** |
| A8 | 38 | 1 | 2 | 34 | 1 | 0 | 0 | **9** |
| A9 | 33 | 3 | 4 | 26 | 0 | 0 | 0 | **8** |
| A10 | 43 | 4 | 5 | 33 | 0 | 0 | 1 | **6** |

### 1.2 Totals

| | Count |
|---|---:|
| Open at 2.11.0 (still present + partly fixed + moved) | 373 |
| Fixed in round 3 (§ 3.3) | 20 |
| Not a defect in round 3 | 0 |
| New in round 3, after merging duplicates (§ 5c) | 71 |
| **Open at 2.24.0** | **424** |

Several reviewers found the same new defect from different sides. For example, the reports path traversal was found from the server (A6), the route table (A7) and the page (A8). Each such defect is counted once, under one ID, with the other IDs listed beside it.

**Reading the numbers:**

- **About 5 % of the open findings were fixed in 43 commits.** Most fixes came from the project's own lane reviews (`825287c2`, `e5b60a59`). Those reviews worked from the project's lane findings, not from this audit; only B3-N1 reached main, as "a user's bug report".
- **Nothing in the priority security list was fixed.** `rate_limit.py`, `auth.py`, `jupyter.py`, `serve_daemon.py` and the notebook blueprint are byte-identical to `0ff77d99`.
- **One security item is worse than reported:** A7-N1 (writes into `.git/`) now gives **command execution**, and it works cross-site (verified).
- **One new High:** a frame set run under the low-bias approximation cannot build its record (C4-N1).
- **Four regressions** among the new findings. In each, an earlier refusal or catch was lost in a rewrite:
  - per-species pseudopotentials (C3-N1);
  - the save door's guards (C7-N1);
  - `<job>_initial.xyz` (C9-N2);
  - the bench `commands` route (C7-N4).

### 1.3 The test suite on main

| Item | Value |
|---|---|
| Command | `pytest tests --ignore-glob='*_e2e.py' -n 4 --dist loadgroup -p no:cacheprovider` |
| Environment | Python 3.13 venv with netCDF4 1.7.4, all state directories in scratch |
| **Result at 2.24.0** | **6,613 passed · 12 failed · 4 errors · 55 skipped · 6 xfailed** (6,690 collected, 250 s) |
| At 2.11.0 | 6,581 passed · 11 failed · 4 errors · 55 skipped · 6 xfailed |
| At 2.10.2 | 6,566 passed · 13 failed · 4 errors · 54 skipped · 6 xfailed |

A second run with `-n 6` gave identical failure, error and skip sets (166 s). **The result no longer depends on worker count:** B3-N1 and A10-N1 are fixed by `7ef8167d`.

| Failure(s) | Cause | Finding |
|---|---|---|
| 4 × `test_auth_config.py::TestARotatedSecretIsTakenUp` | authlib not installed and not gated | A10-02 |
| 5 permission tests | run as root, no euid guard | A10-05 |
| `test_smiles_benzene_planarity` | the test measures orientation, not planarity | A10-03 |
| `test_web.py::test_preflight_returns_issues_for_siesta` | **product defect:** peptide hydrogens dropped without openbabel | A10-N2 |
| **new:** `test_machine_record.py::test_what_the_probe_records[manager's run]` | **the test runs the host's real conda/mamba** (`<manager> run --prefix … true`, 120 s); it passes with a stub manager | **C10-N1** |
| ERROR `test_where_an_item_binds.py` | e2e test in a non-`_e2e` file | A10-01, A10-N3 |
| 3 × ERROR `test_wheel_ships_the_front_end.py` | setuptools missing; they pass with it | A10-N11 |

**netCDF4 is now a hidden tier-1 requirement.** Without it, 691 tests fail and 728 error. None of them skip cleanly, because the import fallbacks turn a missing netCDF4 into "No module named 'structure'" (C10-N2 / C1-N7).

`tools/testrun.py` was not used, because it writes `.test-progress/` into the repository. Both checkouts were confirmed unchanged before and after.

---

## 2. Priority findings open at 2.24.0

These are ordered by consequence. "Round" says when the finding was first reported; the last column is its status at 2.24.0.

### 2.1 Security

| ID | Sev. | Finding | Location (HEAD) | Check | 2.24.0 |
|---|---|---|---|---|---|
| **A7-N1** | **H** | **Writes into a calculation's `.git/` lead to command execution as the server's account.** No path check refuses `.git`; `checkpoint.NEVER_STORED` is used only to filter the zip export (`files.py:589`). Steps: `POST /api/files/write`, or a **cross-site multipart** `/api/files/upload` with `overwrite=1`, writes `<calc>/.git/config` with `core.fsmonitor = <cmd>`. Any later `git` call in that folder then runs `<cmd>`, for example a plain `GET /api/checkpoint/state`. `DELETE` of `.git` also returns 200 | `web/blueprints/files.py` (write, upload, rename, delete); `checkpoint.py` | **verified** (re-run: cross-site upload 200, cross-site GET 200, marker file created) | **still present, worse than reported** (round 1 had "effect not demonstrated") |
| A6-01 / A7-03 | H | **No-sign-in server: "loopback = same account".** Any local account can open `/jupyter/`, where the pass-through adds the token | `web/blueprints/jupyter.py` (unchanged since 2.11.0) | verified (loopback, no Origin → 202 and SIGUSR1) | partly fixed (2.11.0); unchanged since |
| A6-N1 / A7-02 | H | **The documented CSRF defence does not exist outside the notebook.** A cross-site `notify/listener/keys` returns 200 and issues a key. A cross-site multipart upload returns 200 and writes the file, which is the delivery route for A7-N1 | `web/app.py:229-230`; `web-api.md:445` | verified | partly fixed (notebook only) |
| A7-01 | H | **No Host-header check** outside the notebook, so DNS rebinding reads the API | `web/app.py` | verified (`/api/files/roots` with a foreign Host → 200) | partly fixed (notebook only) |
| **C6-N1** / C7-N2 / C8-N9 | **M–H** | **New: `GET /api/notify/reports?user=` is a path traversal with no per-user check.** The path is `reports_dir() / f"{user}.jsonl"` with `user` taken from the query. `?user=../../../../tmp/private/notes` and absolute paths return 200, with the file's lines as cards and the absolute path in `file`. Any signed-in user reads another's reports. It also works under a foreign Host on a no-sign-in loopback server (DNS rebinding). `tests/data/path_fence.toml:402` records the route as "takes_no_path" | `web/blueprints/notify_setup.py:553-572` | verified (three reviewers; read again) | new (2.18.0) |
| B1-N1 | M (H on a shared host with sign-in) | **`/jupyter/` relays whatever listens on 127.0.0.1:serve+1** onto molbuilder's origin, including upstream `Set-Cookie`, with only `frame-ancestors` CSP | `web/blueprints/jupyter.py` (unchanged) | verified (stand-in upstream, re-run) | still present |
| A6-02 / A7-04 | H | **Rate limiter takes the client-supplied leftmost `X-Forwarded-For`** | `web/rate_limit.py:219-222` | verified (`'127.0.0.1, x'` allow-listed) | still present |
| A6-03 / A6-N2 | M–H | **Behind the documented same-host proxy, every request is loopback** | `rate_limit.py:131`; `deployment.md` § 2 | verified (limiter) | still present / partly fixed |
| A6-04 / A7-05 | M–H | **Notify routes have no admin gate:** `/test` is an SSRF probe; any user can rotate another's key. The new `/reports` route is ungated too | `notify_setup.py:302,414,587` | read | still present (grew) |
| C7-N3 | L–M | **New: the one writer ignores umask.** Under umask 077, `/api/files/write` and `/upload` create 0644 files (0600 before `1a0ae147`), so files written from the browser become world-readable on a shared host | `persist.py:115-119` | verified | new (regression) |
| B1-N2 / B1-N3 | L–M | Proxy example omits `Host` and WebSocket; Origin-only gate allows a cross-site existence probe | `deployment.md` § 2; jupyter gate | read | still present |

### 2.2 Data loss and data integrity

| ID | Sev. | Finding | Location (HEAD) | Check | 2.24.0 |
|---|---|---|---|---|---|
| A7-06 | H | **Deleting a symlinked file or folder deletes its target;** rename and move act on the target | `files.py:2310` `shutil.rmtree(resolved)`; `projects.py:290` | verified (re-run: both link kinds lose their targets) | still present |
| A5-02 | H | **A flat warm retry overwrites another launch's run record** (no exclusive create); `job-system.md:738` allows "both run" | `runrecord.py:427` | verified (re-run: job 200 becomes 100, `retry_of` 0) | still present |
| A5-N1 / A1-N9 | M | **Job address files are keyed by job id alone,** so two clusters sharing a home folder overwrite each other's entries | `joblist.py:137-138` | verified (re-run: job 4242 from sol and phx leaves only phx's) | still present |
| **C5-N1** | M | **New: `jobset scratch --empty` deletes the scratch of a run that is still running.** It decides "going" per stage from the stage's newest run, so an older run still running loses its `tsgf/*.TSGFL`/`.TSGFR`. A swept rung between two points reads failed, so its round's pair is emptied mid-walk (by reading). Outputs are never touched (deleted by family name, symlinks skipped). `job-system.md:1756` and the `--help` text promise "every run that is not queued or running". No test covers the command | `jobset/_cli.py:2458-2467` | verified (run-0 running, run-1 failed → run-0's file removed); read again | new (`d77bb564`) |
| **C5-N2** | M | **New: `jobset migrate` renames run files with no check that a run is still going.** Status refuses unnumbered files and names `migrate`, which pushes a person to migrate while a pre-upgrade job runs; the engine then reopens `.XV`, `.DM` and `.TSDE` by their old names and the folder holds both spellings | `jobset/migrate.py:827-834`; `runrecord.py:465-480` | read | new |
| **C7-N1** | M | **New, regression: when a calculation fails to open, the save door skips its guards.** `except Exception: prior = None` at `build.py:1538-1548`, and `except (NotACalculation, Exception)` at `:2276-2279`. With a `job-set.json` that does not read, saving a different calculation over it, or flipping hierarchical→flat, returns 200 (409 at 2.11.0). The folder answer shows no stages and no reason. A folder that needs `jobset migrate` can therefore be overwritten from the page | `web/blueprints/build.py:1538-1548,2276-2279` | verified (fabricated old schema) | new (`a062b82b`) |
| **C2-N1** / C9-N4 | M | **New: a frame set's row units are checked only on the page path.** `fill_data_file` and `average_block` do not check them, so a `displacement_ang` row in bohr is stored as ångström in the `.nc`. The check that does run raises out of `answer()`, which is uncaught → HTTP 500 on `/api/transport/describe` and `/api/task-setup/folder` | `frame_average.py:343-348,482-485`; `blueprints/transport.py:266`; `build.py:2362,2499` | verified (function level) | new |
| **C2-N2** / C4-N9 | L–M | **New: frame weights are not validated.** Negative weights are averaged silently (weights (−1, 2) give ⟨G⟩ = −0.3 G₀). Weights that do not sum to 1 scale the average, with only a warning. The doc calls sum-to-one "an explicit rule" | `frameset.py:111-116`; `frame_average.py:236,249`; `vibrational-averaging.md:366-367` | verified | new |
| **C8-N2** | M | **New: typed booleans lose their meaning.** `cell-readers.js` reads a bool as `t === "true"`, so `"True"`, `"yes"`, `"1"` and `"on"` become `false`. A `use_gpu` typed that way is written `false`, and trials run on the CPU. The removed `coercePoint` mapped yes/on to true | `lib/task-setup/cell-readers.js:62`; `viewer.js:3381-3388,4523-4531` | verified (node) | new |
| A2-02 / A2-N2 | M | Sidecar v7–v11 migration never runs, yet docs say it does | `parse/sidecars/molstruct.py:112`; `sidecars/molstruct.py:798-800` | verified | still present |
| A2-10 / A2-11 | M | PDB round trip drops atoms; `HG21` read as mercury | `structure.py:2141-2145,2199,2308,2311` | verified | still present |
| A2-N1 / A2-28 | M | A typed title such as `Properties=…` makes the saved pair unopenable; `structure.md:738-744` now claims no key becomes metadata | `structure.py:2269-2273` | verified | still present |
| A2-N3 | M | The save door replaces another file's sidecar (`x.pdb` beside `x.xyz`), and accepts `.txt`, `.cif` and no-suffix paths that the loader refuses | `build.py:510-512` | verified (Flask) | still present |
| A2-06 / A2-07 | M | `frozen_atoms` setter and `vacuum` are unvalidated | `structure.py:1620,889`; `periodicity_gate.py:397-401` | verified | still present |
| A2-08 | M | A failed `apply_to_structure` leaves a half-applied structure | `sidecars/molstruct.py:774-793` | verified | still present |
| A5-09 / A5-10 | M | An interrupted checkpoint restore leaves a folder no save ever held; `migrate`'s writes are not atomic as a set, and a re-run is refused | `checkpoint.py:1833`; `migrate.py:817,827-834` | read | still present |
| C4-N3 / C2-N6 / C4-N4 | L | **New: the spectra pair is not both-or-neither.** The JSON is replaced before the `.nc` is written, so a refused `.nc` leaves a new JSON beside an old `.nc` whose sha256 no longer matches. A null text field comes back as `""`. The module header says "standard library only" but loads numpy and netCDF4 | `sidecars/spectra.py:20-26`; `spectra/datafile.py` | verified (probe) | new |
| C1-N6 / C7-N6 | L | **New: fsync dropped from page saves** when the files API moved to `persist.write_bytes` | `persist.py:120-132` | read | new (regression, `1a0ae147`) |
| C4-N8 | L | **New: a point killed while writing a Green's function file ends the sweep.** Every later frame at that voltage reads the truncated file; `tsgf/` is emptied only at round starts | `submit.py:2513-2520` | unverified (no TranSIESTA) | new |

### 2.3 Scientific correctness

| ID | Sev. | Finding | Location (HEAD) | Check | 2.24.0 |
|---|---|---|---|---|---|
| A4-01 / C4-N10 | H | **Linear-response I(V) has the wrong sign for V < 0.** For V = [−0.5, 0, +0.5] it returns `[3.874e-06, 0.0, 3.874e-06]`. The per-spin halves of a polarized run are unsigned too. This path now runs only when the description sets `low_bias_approximation: true`; it is no longer the default | `transport/record.py:316` `half = abs(float(v)) / 2.0`; `:1150-1156` | verified (re-run) | still present (narrower reach) |
| A3-01…04, A3-N1, **C3-N1** | H | **Pseudopotential coverage.** A WARN masks a wrong-XC ERROR; every vdW-DF run is blocked; `Au.psml` is "misnamed". **New regression:** a labelled species such as `Au1` now *raises* `RunFileError "element='Au1' does not match the declared shape"` from `validate(dest_dir=…)` and from the copier, so per-species pseudopotentials cannot be used at all. Prep, hand-over and the settings gate give a traceback or a 500. It returned an Issue at 2.11.0 | `pseudos.py:109,613,656-660,716,738,814`; `siesta/input.py:237`; `validation/siesta.py:228`; `runfiles.py:150-153` | verified | still present; **C3-N1 new** (`c7fb10c2`, 2.13.3) |
| **C4-N2** | M | **New: `eigenvector_canonical` in `<label>.spectra.nc` is labelled `units "1"`.** Its unit is amu⁻¹ᐟ² (`vibration.md:2080` records "dimensionless" as an earlier mistake). A CF/xarray reader gets wrong units for any displacement built from it | `spectra/datafile.py:216-218` | verified (read back) | new |
| A5-07 / A5-N3 | M | **`"30:00"` is read as 30 hours** (SLURM: 30 minutes); `launch --time 30:00` sends `-t 1-06:00:00` | `scheduler/quantities.py` | verified (`canonical_time('30:00')` → `'1-06:00:00'`) | still present (re-run through the stub `sbatch`: a prep for a 15-minute queue sends `-t 1-06:00:00` and `MB_WALL_S=108000`; `parse_duration('45:00')` is 162,000 s; the refusal text now invites "SLURM's own" spelling, C5-N9) |
| A10-N2 | M | **Without openbabel, a peptide build drops real hydrogens:** AG gives C₅H₇N₂O₂, 67 electrons | `chemistry.py:1185-1211` | verified (re-run) | still present |
| A3-08 | M | **Impossible values only warn:** `MeshCutoff` 0 or −5, `DM.Tolerance` ≤ 0, `max_scf_iter` 0, `mixing_weight` ≤ 0, `electronic_temperature` < 0, `conv_tol` ≤ 0 … The catalogue's `above` limit is on 7 items, none of these | `validation/metadata.py:87-110` | verified | still present |
| A3-05/06/07/16 | M | No XC functional/authors pairing; `d3bj` with `wb97m-v`; symmetry with geomeTRIC; spin–orbit with scalar-relativistic pseudos warns only | `config/siesta.py`; `pyscf/layout.py:204`; `pyscf/input.py:480`; `pseudos.py:646` | verified | still present |
| A4-03 | M | Transport assumes c ∥ z and a, b ⊥ z, unchecked | `transiesta.py:192`; `wizard.py:205,386-389` | read | still present |
| C3-N2 | L–M | **New: a PySCF rung stopped by its wall still runs a full cold SCF after the deadline,** so it can overrun the 600 s `wall_slack` and be killed instead of reporting "relaxed: no" | `pyscf/input.py:1133-1187`; `relax_policy.py:132-141` | unverified (no PySCF) | new |
| C4-N6 | L | **New: the GF-file estimate ignores units** (`ElectronicTemperature 0.0019 Ry` → 38,882,470 energy points). It affects only the status line on a hand-edited deck | `transport/gf_files.py:77-91` | verified | new |

**Checked and found sound:** the reuse of the leads' Green's function files across frames (C4 § C). A file is keyed by run folder, SystemLabel, signed voltage and lead. One prep fixes everything else the file depends on: the bulk file, cell, k-grid, spin, temperature, contour and η. Frames must keep the electrode atoms, and each launch opens a new `run-<n>/` with an empty `tsgf/`. The Gauss–Hermite frame weights and the zero-point amplitude constant also check out.

### 2.4 Functional defects

| ID | Sev. | Finding | Location (HEAD) | Check | 2.24.0 |
|---|---|---|---|---|---|
| **C4-N1** | **H** | **New: a frame set run under the low-bias approximation cannot build its transport record.** `base = len(out["voltages_v"])` rebinds the `base: Path` parameter. The next frame's `base / zero["attempt"]` then raises `TypeError`. Summarize prints a traceback and the Results tab returns 500 | `transport/record.py:1161` (used at `:1144`) | verified (two frames); read again | new (`825287c2`) |
| A9-01 | H | Spectrum chart paints the wrong sticks while Raman values are pending | `lib/spectrumchart/_seal.js:339,557-561` | verified (node, re-run) | still present |
| **C9-N1** | M | **New: every plain `.xyz` opens in the trajectory viewer** as an "Optimization" movie, because `TRAJECTORY_PARSERS` includes `pyscf` and the trajectory viewer is registered before structure. The structure preview (sidecar labels, cell, "Open in Molbuilder") is bypassed, against `presenters.md:42,131-133` | `lib/inspectors/trajectory.js:62,71-72`; `results.html:280,285` | verified (node, real registry) | new |
| **C9-N2** | M | **New, regression: `<job>_initial.xyz` is unreadable.** Two parsers claim it (`AmbiguousFormatError`), so it is dropped from the Results menu. A test hand-spells the role and misses it | `parse/engines/pyscf.py:519`; `runs._file_row`; `tests/test_results_file_picker_js.py:498-502` | verified | new (`825287c2`) |
| **C6-N3** | M | **New: a user copy of `report-cards.json` that omits a newer field stops every job's monitor** after two log lines. Lost with it: utilisation, the wall-near warning, listener reports, and the `[MONITOR] job ended` line that `parse/dirs/job.py` and `materialize.written_as_ended` read | `monitor.py:1974-1979` (re-raises `LayoutError`) | verified (read again) | new (`aff58812`) |
| **C6-N2** / C2-N7 | M | **New: one malformed report line makes the reports route return 500 until the file is rotated** (`sent_at: "yesterday"`, `elapsed_s: "abc"`, `Infinity`). The listener stores any string | `parse/instruments/reports.py:46-52`; `report_fields.py:100-109` | verified | new |
| **C8-N1** | M | **New: the leads' GF file size is never shown.** `viewer.js` reads `r.tsgf` from ladder rows, but the server never sends it. `transport.md:2053-2058` promises the 49–116 GiB figure in Task setup | `viewer.js:2703`; `build.py:2443-2445` | read | new |
| **C8-N3** | M–L | **New: a hand-over written before 2.16.3** gives an empty stages card, loses its message, and leaves an unhandled rejection. Its schema is still `task-handover@1` | `viewer.js:3442-3453,4161-4178`; `build.py:971` | read | new |
| A8-N1 | M | The atom-order line deep-copies the trajectory on every model change | `lib/molview/ui.js:1363`; `model.js:766` | verified (timing, 2.10.2) | still present |
| A4-N1 | M | `ComposeError` uncaught: HTTP 500 and CLI traceback on a corrupt junction sidecar | `compose.py:958`; `record.py:758,1061`; `blueprints/transport.py:313,358`; `jobset/_cli.py:1549-1550` | verified | still present |
| A8-02 | M | A stale filter answer overwrites a newer selection | `lib/molview/stores.js:278-291` | verified (node) | still present |
| A7-10 / A7-33 | M | HTML 404/405/500 pages instead of the JSON envelope; a non-object JSON body gives 500 on **40** mutating routes | `web/app.py`; `request.get_json(silent=True) or {}` | verified (sweep) | still present |
| A1-09 | M | `make web` / `make web-bg` call a command that does not exist | `Makefile:38,46` | verified | still present |
| A5-01 | M | Partly fixed: answering "no" and a partly refused benchmark still exit 0 | `jobset/_cli.py:2160-2163`; `submit.py:1830-1833` | verified (exit codes re-run) | partly fixed |
| C5-N3 | L | **New: one unreadable wall in a launch record makes all of `jobset status` exit 1** instead of reading that row `unreadable` | `runrecord.py:528-553`; `parse/dirs/job.py:277` | verified | new |
| C5-N4 | L | **New: the scheduler's "ended" line outranks a requeue,** so a requeued run reads failed while it runs again (and `jobset scratch` treats it as not live) | `parse/dirs/job.py:587,648`; `runs.concluded:685` | unverified (no SLURM) | new |
| C5-N8 | L | **New: launch no longer calls `jobset.validate()`,** so a hand-edited plan with duplicate job names or an unknown `kind` is not refused at launch | `submit._planned` | read | new |
| C6-N4 | L–M | **New: the Reports card is inert without sign-in.** The page never sends `?user=`, so it shows 0 | `notify_setup.py:553-559`; `this-machine/page.js:376,453` | verified | new |
| C7-N4 | L | **New, regression:** `/api/task-setup/commands?kind=bench` on a non-calculation folder → HTML 500 | `build.py:2600-2603` | verified | new |
| C9-N3 / C7-N8 | L | **New:** `temperature_k=inf` or `1e400` passes the check; `1/tanh(0)` then raises `ZeroDivisionError` → HTML 500 | `blueprints/spectra.py:103`; `spectra/derived.py:81` | verified | new |

### 2.5 Review gates that are red

These matter because they are the mechanisms meant to catch the rest.

| ID | Finding | Location (HEAD) | Check | 2.24.0 |
|---|---|---|---|---|
| A1-33 | The pyflakes pre-commit hook cannot install (no `.pre-commit-hooks.yaml` at rev 3.2.0), so no hook runs | `.pre-commit-config.yaml` | verified (HTTP 404) | still present |
| A1-N7 → **C10-N6 / C1-N10** | `submit.py`'s undefined `Environment` was fixed (`825287c2`). **The same commit added a new pyflakes hit.** `plan.md:385-386` makes a clean pyflakes run a milestone gate | `transport/compose.py:587`; three hits in `tools/` | verified | A1-N7 fixed; new hit |
| A1-N3 / B3-N3 | `classify_path_finders --check` exits 1: one unclassified finder and two stale overrides. `project-layout.md:2566` and `plan.md:386` still call it clean | `joblist.py:197`; `tools/classify_path_finders.py:99,105` | verified | still present / partly fixed |
| A1-03 / A1-06 / **C1-N3** | **Layering erodes further.** L1→L2 import sites went from 6 to 10. Upward floor imports went from 21 to 24, with migrate's eight made legal. Seven new upward imports sit inside functions. `identity` ships in the monitor and vibration bundles, where `continues()` without a restart would raise | `identity.py:115`; `parse/contract.py:94`; `task.py:338,724,741`; `structure.py:1201` | read (AST graph) | worse |
| **C1-N1** | **The A16 "doors" sit on two floors, and `calculation` on none.** `calculation.py` imports `jobset.model` (floor 3) at module level | `execution/architecture.md:184-191,219,475-478`; `calculation.py:26-28` | read | new |
| A1-13 | 37 test files cited in docs do not exist (41 at 2.10.2) | e.g. `molview.md`, `form-schema.md` | verified (sweep) | still present |
| **C10-N1** | **New: a tier-1 test runs the host's real environment manager,** so its result depends on the machine | `tests/data/machine_record.toml:51-54`; `diagnostics.py:552-560` | verified | new |
| **C10-N2** / C1-N7 | **New: 39 `except ImportError: # beside a job` fallbacks mask real import errors.** A missing netCDF4 becomes "No module named 'structure'", so 1,419 tier-1 tests fail or error and none skips. A same-named file in the current directory can shadow a module | `spectra/datafile.py:42-53` and 38 others | verified | new |
| **C1-N2** / C2-N5 | **New: a guard that is never called.** `_every_cited_item_is_read_back` is defined but has no caller | `parse/contract.py:86-103` | verified | new |
| B3-N1, A10-N1 | Tier-1 results depended on worker count | `tests/conftest.py:433-455` | verified (`-n 4` and `-n 6` identical) | **fixed** (`7ef8167d`) |

---

## 3. What was fixed

### 3.1 Round 1 (2.9.4 → 2.10.2)

| ID | Finding fixed | Commit |
|---|---|---|
| A2-01 | A sidecar is refused when its geometry changed | badc9178 |
| A2-03 | The PySCF `_optimized.xyz` is read with its sidecar | badc9178 |
| A3-11 | `stages.md` §1.3 group counts | cd7deb94 |
| A5-08 | A warm retry's output no longer lands in the previous run's log | 5c63bf95 |
| A7-12 | Path-fence table names real helpers; two fence holes closed | 62fb65b0 |
| A8-01 | Read-only trajectory viewer rebuilds again | 8597632f |
| A8-23 | Drag-select listener removed on dispose | ee68556e |
| A9-03 | VibrationView no longer accumulates WebGL contexts | 691e7f0a |
| A9-11 | `presenters.md` §7 module table | a10c49db |
| A1-31 | `composed_junction` defined once | 74916c45 |
| A10-11 | Source-read classifier retired | 769f8f4c |
| A10-16, A10-17 | Plan Q13 and §5aa archived | — |

### 3.2 Round 2 (2.11.0)

| ID | Finding fixed | Commit |
|---|---|---|
| A6-05 | Jupyter always binds 127.0.0.1 | ebaaebb4 |
| A6-12 | No http/https split between server and notebook | ebaaebb4 |
| A7-30 | The per-port wildcard `frame-src` CSP is gone | ebaaebb4 |

### 3.3 Round 3 (2.11.0 → 2.24.0)

| ID | Finding fixed | Commit |
|---|---|---|
| A1-N7 | pyflakes: undefined `Environment` in `submit.py` (but see C10-N6) | 825287c2 |
| A3-15 | `kind` disagreements between config and catalogue: the catalogue is the one home | df200c2b |
| A4-07, A4-N5 | `TRANSPORT_AXIS` defined once (`kmesh.py:36`); `compose.py` imports it | 825287c2 |
| A4-10, A9-12 | The thermal spread is computed once, by the server (`derived.py:69-95`); the page reads it | e5b60a59 |
| A4-12 | Rotation-origin wording in `normal-modes.md` | 825287c2 |
| A4-17 | The stale 12 Å note in `wizard.py` | 825287c2 |
| A4-N4 | Vibration results are schema 9 in code and doc | ae4101a5 |
| A6-28 | `MOLBUILDER_FORCE` is gone | 5d0a33de |
| A7-07 | A page save keeps an existing file's mode (0755 stays 0755); but see C7-N3 | 1a0ae147 |
| A8-20 | Doc drift at `task-setup.md:1073` | 825287c2 |
| A9-07 | `spectra.md:65` | 825287c2 |
| A9-30 | `trajectory.js:83-88` | 825287c2 |
| A10-27 | The config PySCF/SIESTA duplicate | df200c2b |
| A10-N1, B3-N1 | Tier-1 results no longer depend on worker count (session-level `delenv` in `conftest.py:433-455`) | 7ef8167d |
| A10-N8 | `plan.md` 284,974 → 204,772 bytes (35 rows archived) | c7353447 |
| A5-04 | The abbreviated job-set.json example is gone from `job-system.md` (its key list omits `scratch_dir`, C5-N5) | 825287c2 |
| A5-11 | The web attempt count counts launches via `newest_run` | 50b8ef68 |

**Partly fixed in round 3:**

- **A1-21:** the files API and docs now write through `persist` (`1a0ae147`).
- **A2-21:** the spectra `structure_hash` is retired (`ae4101a5`).
- **A4-09:** help text and ranges live only in the catalogue (`df200c2b`). Default and type are still declared twice; a test guards them, with gaps (C3-N6).
- **A4-16:** the `wizard.py:427` citation is gone; `deck.py:527,711` still cite a § 4.2 that does not exist.
- **B2-N5:** `gpu.md` and `generator.md` now match the code (`e5b60a59`). Two GPU-per-node rules still exist.
- **A8-12:** `molview.md`'s "sixteen names" is right (`043309ca`); "four things" is still wrong.
- **A9-20 to A9-23:** docs reconciled (`e5b60a59`); the `viewer.js:383` heuristic and the old registry signature remain.
- **A10-08, A10-10, A10-13, A10-18:** counts and drift partly refreshed.
- **B3-N3:** the live plan no longer calls the classifier clean, but `project-layout.md:2566` and the archive do.

**Sound in round 3** (reviewed in full, no defect found):

- `ncdata.py` (round trip verified) and `transport/datafile.py`, `stages.py`, `tbtnc.py`, `deck.py`;
- `persist.py` as a module (but see C7-N3 and C1-N6);
- `structure.py`'s diff, `frame_choice`, the `molstruct` bundle fallbacks;
- `parse/dirs/job.py` (scheduler ended, past wall, quiet for), by reading;
- the report-card rules and the monitor's webhook formatting: `json.dumps`, size caps, no untrusted third-party text forwarded;
- the This machine page's XSS handling;
- the rung surface filter, which matches the old filter on all 5 rungs (run);
- the hand-over `proposed_stages` refusal;
- `bench/result.py`;
- `spectra/results.py` schema 9;
- the server-served thermal spread formula, σ = Q_zp·√coth(ħω/2kT);
- `jobset migrate`'s decision ordering (every write decided before any is made, target collisions checked), the scheduler-ended reading for the documented `*** JOB <id> … ***` frame, `submit._sbatch_request` (a hand-edited header is refused at launch), and `calculation.py`;
- the monitor bundle on bare Python 3.11–3.13; on 3.6 it fails to load and says so, as documented.

---

## 4. Corrections

| ID | Earlier claim | Verdict |
|---|---|---|
| A4-02 | The Raman chart is "not labelled as activity" | **Not a defect** (round 1): it is labelled "Raman activity, Å⁴/amu" |
| A4-28 | TBtrans current doubling is suspect | **Not a defect** (round 1): it matches sisl |
| A7-32 | Modify spacings accept NaN/inf/negative | **Not a defect** (round 1): `cell.py` refuses them |
| A6-15, A6-24 | — | **Narrowed** (round 1) |
| A5-18 | "31 rules" was a miscount | **Not a defect** (round 1) |
| A10-04 | Peptide preflight failure is an rdkit-version issue | **Re-diagnosed** as product defect A10-N2 (round 1) |
| A7-N1 | "Effect not demonstrated" (rounds 1–2) | **Raised:** command execution demonstrated, cross-site (round 3) |
| A4-01 | "This is the default path" (rounds 1–2) | **Narrowed:** since 2.12–2.13 the linear-response path runs only when `low_bias_approximation: true` is set (`task.py:492`) |
| A4-09 | Round-3 reviewers disagreed: STILL (A4) vs PARTLY (A3) | **Partly fixed:** `df200c2b` removed the help-text and range copies; default and type are still in both places (see § 3.3) |
| C6-N3 | One round-3 reviewer read the refused layout as "caught" at `monitor.py:1976-1977` | **The monitor stops:** the `except` logs the error and re-raises (`monitor.py:1979`). The catch that does exist is in the notify route (`test_channel`) |

---

## 5. New findings

### 5a. Round 1 (A*-N*, 67) — status at 2.24.0

| ID | Sev. | Finding | 2.24.0 |
|---|---|---|---|
| A1-N1 | M | `task` (L1) imports `template` (L2) | still present, worse (`task.py:338,724,741`) |
| A1-N2 | M | Upward floor imports | still present (`placement.py:146`; `transport_rung.py:316`) |
| A1-N3 | M | Classifier `--check` fails; plan said clean | still present |
| A1-N4 | M | `<name>.tmp` temp-name trap in the pair writers | still present (`workingcopy_structure.py:560`; `molstruct.py:658`) |
| A1-N5 | M | `conventions.md` command catalogue | still present, worse (20 commands; `jobset` lists 5 of 9 verbs) |
| A1-N6 | L | New modules absent from spine maps | still present, grew |
| A1-N7 | L | pyflakes undefined `Environment` | **fixed** 825287c2 |
| A1-N8 | L | `code-audit.md` contradicts itself on lints vs outcome tests | still present |
| A1-N9 | L | Job address keyed by bare scheduler id | still present (`joblist.py:137-138`) |
| A1-N10 | L | Makefile header names `tools/testrun.py` | still present |
| A1-N11 | L | Identical table-format closures | still present (`runstatus.py:820`; `plan.py:93`) |
| A2-N1 | M | Typed title read as extxyz metadata | still present (`structure.py:2269`) |
| A2-N2 | M | Docs say v7–v11 migration works | still present |
| A2-N3 | M | Save door ignores sidecar and suffix | still present (re-run) |
| A2-N4 | L | `_write_pair` not both-or-neither | still present (`workingcopy_structure.py:570-571`) |
| A2-N5 | L | Long PDB title word broken | still present (`:604`) |
| A2-N6 | L | Newline in PDB title | still present (`:601`) |
| A2-N7 | L | `resolve_element("x")` suggests `X`, then refuses it | still present (`chemistry.py:299-301`) |
| A2-N8 | L | Two XYZ frame parsers | still present (`molstruct.py:289`) |
| A2-N9 | L | `atom_z_span_ang` without consumer | still present (carried) |
| A3-N1 | M | Coverage check vs copier answer differently | still present; worsened by C3-N1 |
| A3-N2 | L | `architecture.md` lists deleted `find_psml` | still present (`:257`) |
| A3-N3 | L | `siesta.md` MPS rule incomplete | still present (`:564-573`) |
| A3-N4 | L | BlockSize/ParallelOverK explained twice | still present |
| A3-N5 | L | "Same engine input" folds case on `SystemLabel` | still present (re-run: `BDT` vs `bdt` → `[]`) |
| A4-N1 | M | `ComposeError` uncaught | still present (re-run) |
| A4-N2 | L | `selection_pdos` AttributeError | still present (`record.py:1061,1084`) |
| A4-N3 | L | `point_currents` can mix runs | still present (`record.py:235-247`) |
| A4-N4 | L | Schema 9 described before built | **fixed** ae4101a5 |
| A4-N5 | L | Second transport-axis constant | **fixed** 825287c2 |
| A5-N1 | M | Job address files collide across clusters | still present (re-run) |
| A5-N2 | L–M | Unknown-cluster job checked locally and forgotten | still present (unverified); the same reading now also feeds `standing` |
| A5-N3 | M | `launch --time` unchecked: `30:00` → 30 h | still present (re-run) |
| A5-N4 | L | Run script and monitor read only `MB_*` | still present (`monitor.py:639,748,1782`) |
| A5-N5 | L | A benchmark shelf's wall is one trial's | still present, worse: later trials are told almost no wall (`MB_WALL_S` counts from the job's start) |
| A5-N6 | L | `scontrol show config` once per submission | still present (`submit.py:1967-1969`) |
| A6-N1 | M–H | CSRF defence does not exist | partly fixed (notebook only) |
| A6-N2 | M | Same-host proxy: notebook gate admits all | partly fixed |
| A6-N3 | L–M | Two "is this pid ours" checks | still present (`serve_daemon.py:200-226`) |
| A6-N4 | L | Admin list gates three subsystems, not two | still present (`access-control.md:314`) |
| A7-N1 | H | File API writes into `.git/` | **still present; command execution verified** |
| A7-N2 | M | Transport routes use two relative-path bases | still present (re-run) |
| A7-N3 | M | Route count and its "pin" | still present, worse (103 routes; doc says 99; no test pins it) |
| A7-N4 | L | Notify key URL from the request Host | still present (`notify_setup.py:610`) |
| A7-N5 | L | Upload branch of `/api/watch/load` lacks the catch | still present (`watch.py:628`) |
| A7-N6 | L | `err()` used only in `modify.py` | still present (294 inline refusals) |
| A8-N1 | M | Atom-order line deep-copies the trajectory | still present |
| A8-N2 | L | Docs name guard tests that do not exist | still present (`molview.md:3511`; `form-schema.md:275`) |
| A8-N3 | L | Two 3Dmol window pools | still present (= A8-30, A10-N4) |
| A8-N4 | L | `parser.js` "here and nowhere else" | still present (`:88`) |
| A8-N5 | L | Two `openMolecule` signatures | still present (carried) |
| A8-N6 | L | `SHAPED_METADATA_KEYS` read by one translator | still present (`model-jobs.js:211`) |
| A8-N7 | L | `mount` can reject | still present (unverified) |
| A9-N1 | L | Docs still describe "Refresh" | still present |
| A9-N2 | L | Disposed spectra tab still exports | still present (`core.js:2898`) |
| A9-N3 | L | Dead `const sign = 1` | still present (`_seal.js:316`) |
| A10-N1 | M | Seeding tests and `$MOLBUILDER_PROJECTS` | **fixed** 7ef8167d |
| A10-N2 | M | Peptide hydrogens dropped without openbabel | still present (re-run) |
| A10-N3 | M | Playwright skip keyed on file name | still present (`conftest.py:1027-1034`) |
| A10-N4 | M | 3Dmol pool twice (= A8-N3) | still present |
| A10-N5 | L | "By any spelling" overstates the tree guard | still present (`testing.md:207`) |
| A10-N6 | L | Residual source pin | still present (`test_web_files.py:771-777`) |
| A10-N7 | L | Q13 cited as open | still present (`plan.md:31,375`) |
| A10-N8 | L | Plan size and the 26 KB cell | **fixed** c7353447 |
| A10-N9 | L | Dead tools | still present |
| A10-N10 | L | Stale `testing.md` § 4 numbers | still present (`:628`) |
| A10-N11 | L | Tier-1 requirements unstated | still present, worse (netCDF4 now also needed) |

### 5b. Round 2 (B*-N*, 15) — status at 2.24.0

| ID | Sev. | Finding | 2.24.0 |
|---|---|---|---|
| B1-N1 | M (H) | Pass-through relays any listener on serve+1 | still present (re-run) |
| B1-N2 | L–M | Proxy example omits Host and WebSocket | still present |
| B1-N3 | L | Cross-site existence probe of `/jupyter/files/` | still present |
| B1-N4 | L | Two false token claims | still present (`blueprints/jupyter.py:181`; `jupyter.py:339`; `cli.py:2725`) |
| B1-N5 | L | 50 MB cap on notebook saves | still present (carried) |
| B2-N1 | L | `fit` reads an unstated GPU or node count as zero | still present (carried) |
| B2-N2 | L | Fit message truncates the asked wall | still present (carried) |
| B2-N3 | L | `molbuilder.scheduler.fit` is a function | still present (re-run) |
| B2-N4 | L | `a100 xNone` | still present (moved: `record.py:1166`) |
| B2-N6 | L | Comments cite deleted `max_cpus` | still present (`probe.py:74,159`) |
| B2-N7 | L | Comment says the tab fills time and memory | still present (`record.py:1140-1145`) |
| B2-N5 | L | `gpu.md` vs code on the GPU count | partly fixed (e5b60a59); two rules remain (`prep_inputs.py:126-138` vs `fit.py:181-187`) |
| B3-N1 | L–M | Results depend on scheduling | **fixed** 7ef8167d |
| B3-N2 | L | Grouping uses pytest-private API | still present (`conftest.py:958-959`) |
| B3-N3 | L | Plan certifies red classifier | partly fixed |

### 5c. Round 3 (C*-N*) — new in the 43 commits

The severity is the reviewer's, after verification. Where several reviewers found one defect, the first ID listed is the one counted.

<details open><summary><b>A1 — spine and architecture (11)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C1-N1 | M | The A16 doors sit on two floors and `calculation` on none; `calculation.py` imports floor 3 at module level | `execution/architecture.md:184-191,219,475-478`; `calculation.py:26-28` | read |
| C1-N2 | M | `_every_cited_item_is_read_back` is never called (= C2-N5) | `parse/contract.py:86-103` | verified |
| C1-N3 | M | Seven new upward imports at function level; `identity` in the monitor and vibration bundles would raise in `continues()` without a restart (latent) | `identity.py:115`; `contract.py:94`; `task.py:338,741`; `structure.py:1201` | read |
| C1-N4 | L | `ncdata.py`, `frameset.py` and `spectra/derived.py` call themselves L1 but are not in the L1 index | `ncdata.py:5`; `frameset.py:4`; `derived.py:4` | read |
| C1-N5 | L | Two docs list the deleted `JobSet.load` | `architecture.md:220`; `execution/architecture.md:220` | read |
| C1-N6 | L | `persist.write_bytes` has no fsync (= C7-N6) | `persist.py:120-132` | read |
| C1-N7 | L | 39 import fallbacks mask real `ImportError`s (= C10-N2) | e.g. `spectra/datafile.py:42-53` | verified |
| C1-N8 | L | A16's "path formed only by its owner" is broken | `runs.py:900,941`; `parse/sidecars/task.py:41-45` | read |
| C1-N9 | L | Duplicate `_number` | `frame_average.py:123`; `frameset.py:64` | read |
| C1-N10 | L | New pyflakes hit (= C10-N6) | `transport/compose.py:587` | verified |
| C1-N11 | L | A template-only folder gets the empty-folder refusal | `calculation.py:109-127` | verified |

</details>

<details open><summary><b>A2 — data model (9)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C2-N1 | M | Frame-set row units checked only on the page path; bohr stored as Å; uncaught → 500 (= C9-N4) | `frame_average.py:343-348,482-485` | verified |
| C2-N2 | L–M | Negative frame weights averaged silently (with C4-N9) | `frameset.py:111-116`; `frame_average.py:236,249` | verified |
| C2-N3 | L | A unitless row refuses the unit `"1"` | `frame_average.py:343` | verified |
| C2-N4 | L | `structure-periodicity.md:43` says null; `frame_at(1).offset_by` is `"engine"` | `structure.py:1697-1700` | read |
| C2-N5 | L | = C1-N2 | `parse/contract.py:86-103` | verified |
| C2-N6 | L | `sidecars/spectra.py` header says stdlib-only; JSON and `.nc` not written as a pair (= C4-N4, C4-N3) | `sidecars/spectra.py:21-26` | read |
| C2-N7 | L | `float(sent_at)` on `"soon"` → `read_reports` raises (= C6-N2) | `parse/instruments/reports.py:50-52` | verified |
| C2-N8 | L | NetCDF unit `"amu^1/2 angstrom"` is not UDUNITS | `spectra/datafile.py` | unverified |
| C2-N9 | L | Stale citation | `structure.md:743` | read |

</details>

<details open><summary><b>A3 — engines (6)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C3-N1 | M (H with A3-0x) | **Regression:** a labelled species `Au1` → `RunFileError` from `validate(dest_dir=…)` and the copier; per-species pseudos unusable since 2.13.3 | `pseudos.py:109`; `siesta/input.py:237`; `validation/siesta.py:228`; `runfiles.py:150-153` | verified |
| C3-N2 | L–M | PySCF rung stopped by its wall runs a full cold SCF after the deadline | `pyscf/input.py:1133-1187`; `relax_policy.py:132-141` | unverified |
| C3-N3 | L | Docs and docstrings still say parameters are "authored in two places" | `template.md:1005-1072`; `science/validation.md:447`; `config/siesta.py:3,224`; `config/pyscf.py:3,120`; `validation/metadata.py:1-6`; `validation/task.py:361-362`; `_shared.py:63` | read |
| C3-N4 | L | Dead code | `template.py:659-669,718` | read |
| C3-N5 | L | Orphan comment | `config/siesta.py:545-551` | read |
| C3-N6 | L | The catalogue-agreement test skips `None` defaults (`ecp_atoms`) and tuples, lists and enums | `tests/test_catalogue_agreement.py:82-141` | read |

</details>

<details open><summary><b>A4 — transport and vibrations (10)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C4-N1 | H | Frame set under the low-bias approximation: `base` rebound → `TypeError` | `transport/record.py:1161,1144` | verified |
| C4-N2 | M | `eigenvector_canonical` labelled `units "1"` (is amu⁻¹ᐟ²) | `spectra/datafile.py:216-218` | verified |
| C4-N3 | L | Null text → `""`; an int set to `None` raises after the JSON is replaced | `spectra/datafile.py`; `sidecars/spectra.py` | verified (probe) |
| C4-N4 | L | Header says stdlib-only (= C2-N6) | `sidecars/spectra.py:20-25` | read |
| C4-N5 | L | `vibration.md` § 6.10.5 layout and attribute names differ from the code (`temperature_K`, `pressure_atm`) | `vibration.md` § 6.10.5 | verified |
| C4-N6 | L | GF-file estimate ignores units | `transport/gf_files.py:77-91` | verified |
| C4-N7 | L | "116.0 GiB" should be 115.6 GiB | `transport.md:2063,2073` | verified |
| C4-N8 | L | A truncated GF file ends the sweep | `submit.py:2513-2520` | unverified |
| C4-N9 | L | Doc calls sum-to-one weights "an explicit rule"; code only warns (with C2-N2) | `vibrational-averaging.md:366-367`; `frameset.py:111-113` | read |
| C4-N10 | L | Per-spin currents unsigned too (part of A4-01) | `record.py:1150-1156` | read |

</details>

<details open><summary><b>A5 — execution (9)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C5-N1 | M | `jobset scratch --empty` deletes a still-running older run's scratch (liveness by the stage's newest run) | `jobset/_cli.py:2458-2467` | verified |
| C5-N2 | M | `jobset migrate` renames files with no check that a run is still going | `migrate.py:827-834` | read |
| C5-N3 | L | One unreadable launch-record wall makes `jobset status` exit 1 | `runrecord.py:528-553`; `parse/dirs/job.py:277` | verified |
| C5-N4 | L | The scheduler's "ended" line outranks a requeue | `parse/dirs/job.py:587,648` | unverified |
| C5-N5 | L | `job-system.md:1250`'s key list omits `scratch_dir` | `model.py:538-540` | read |
| C5-N6 | L | The walk script spells the scratch glob a second time, and wider (`*.TSGF*`) | `submit.py:2185-2189` vs `warmfiles.scratch_families` | read |
| C5-N7 | L | No test covers `jobset scratch` or `checkpoint.prune_scratch` | — | read |
| C5-N8 | L | Launch no longer calls `jobset.validate()` | `submit._planned` | read |
| C5-N9 | L | The `--time` refusal text invites SLURM's spelling, which A5-07 misreads | `jobset/_cli.py` | verified |

</details>

<details open><summary><b>A6 — configuration, ops, notify, report cards (7)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C6-N1 | M–H | `/api/notify/reports?user=` path traversal, no per-user check (= C7-N2, C8-N9) | `notify_setup.py:553-572`; `tests/data/path_fence.toml:402` | verified |
| C6-N2 | M | One malformed report line → reports route 500 until rotated (= C2-N7) | `parse/instruments/reports.py:46`; `report_fields.py:100-109` | verified |
| C6-N3 | M | A user `report-cards.json` missing a newer field stops every job's monitor | `monitor.py:1974-1979` | verified (read) |
| C6-N4 | L–M | Reports card inert without sign-in | `notify_setup.py:553-559`; `page.js:376,453` | verified |
| C6-N5 | L–M | `remember_manager_run` round-trips `environment.json` and drops unknown keys, incomplete rows and extras; `configuration.md` § 5 says it only adds a fact | `envs/builds.py` | verified |
| C6-N6 | L | `notify.py:13-14` "No tab reads that file" is false since 2.18.0 | `notify.py:13-14` | read |
| C6-N7 | L | `installation.md:111-115` quotes a stub note the code does not print | `builds.py:781-791` | read |

</details>

<details open><summary><b>A7 — web backend (8)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C7-N1 | M | **Regression:** a calculation that fails to open skips the save door's guards; the folder answer hides the reason | `build.py:1538-1548,2276-2279` | verified |
| C7-N2 | M | = C6-N1 | `notify_setup.py:560,572` | verified |
| C7-N3 | L–M | The one writer ignores umask (0644 under umask 077) | `persist.py:115-119` | verified |
| C7-N4 | L | `commands?kind=bench` on a non-calculation → HTML 500 | `build.py:2600-2603` | verified |
| C7-N5 | L | The "under a topic" depth rule is enforced only by the hand-over; save writes at depth 1; the same refusal is 400 in one place and 409 in another | `build.py:1075-1081`; `task-setup.md:80` | verified |
| C7-N6 | L | = C1-N6 (no fsync) | `persist.py:120-132` | read |
| C7-N7 | L | Export's 400 names an envelope shape that does not exist | `build.py:562` | verified |
| C7-N8 | L | `temperature_k` accepts infinity (= C9-N3) | `blueprints/spectra.py:103` | verified |

</details>

<details open><summary><b>A8 — front end 1 (9)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C8-N1 | M | Ladder rows never carry `tsgf`, so the GF file size is never shown | `viewer.js:2703`; `build.py:2443-2445` | read |
| C8-N2 | M | `"True"`, `"yes"`, `"1"` and `"on"` read as `false`; `use_gpu` written false | `cell-readers.js:62`; `viewer.js:3381-3388,4523-4531` | verified (node) |
| C8-N3 | M–L | Pre-2.16.3 hand-over: empty stages, message lost, unhandled rejection | `viewer.js:3442-3453,4161-4178` | read |
| C8-N4 | L | `_rowKinds` never reset per folder → stale class badges | `viewer.js:62,532-563,1007` | read |
| C8-N5 | L | Reports list repaints every 30 s, collapsing an opened report and resetting "since you last looked" | `this-machine/page.js:438-478` | read |
| C8-N6 | L | The previous folder's structure names are kept when `structure_pair` is null | `viewer.js:1431-1432` | read |
| C8-N7 | L | Spells `"molbuilder/task@1"` though the hand-over carries `task_schema` | `viewer.js:4134` | read |
| C8-N8 | L | `handover-procedure.md:169` names `task.task.from_text` | — | read |
| C8-N9 | — | = C6-N1 | — | — |

</details>

<details open><summary><b>A9 — front end 2 (8)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C9-N1 | M | Every plain `.xyz` opens in the trajectory viewer | `inspectors/trajectory.js:62,71-72`; `results.html:280,285` | verified (node) |
| C9-N2 | M | **Regression:** `<job>_initial.xyz` unreadable (`AmbiguousFormatError`) and dropped from the menu | `parse/engines/pyscf.py:519` | verified |
| C9-N3 | L | `temperature_k=inf` → `ZeroDivisionError` → 500 (= C7-N8) | `blueprints/spectra.py:103`; `derived.py:81` | verified |
| C9-N4 | L | = C2-N1 (500 path) | `blueprints/transport.py:266`; `build.py:2499` | verified |
| C9-N5 | L | Suffix fallback and registry signatures undocumented or stale | `viewer.js:580-583`; `registry.js:18,94-97`; `presenters.md:43,129-130` | read |
| C9-N6 | L | `bench-summary.md:221` names deleted `bundle_for_sweep_file` | — | read |
| C9-N7 | L | Stale comment | `spectra/core.js:2499` | read |
| C9-N8 | L | Possible race in `_refreshThermal` | `core.js:3116-3132` | unverified |

</details>

<details open><summary><b>A10 — tests, plan, hygiene (6)</b></summary>

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| C10-N1 | M | A tier-1 test runs the host's real environment manager | `tests/data/machine_record.toml:51-54`; `diagnostics.py:552-560` | verified |
| C10-N2 | L | netCDF4 missing → masked import error, 1,419 fail or error, none skips (= C1-N7) | `spectra/datafile.py:42-53` | verified |
| C10-N3 | L | W5 archived as done, but dead CSS rules remain | `lib/inspectors/inspector-card.css:113-137,176-177` | read |
| C10-N4 | L | W44's closure left two docs contradicting `template.md:2874` | `worked-example.md:532`; `run-identity.md:573-575` | read |
| C10-N5 | L | `web-api.md:633` names nonexistent `compose.frame_set_answer` | `frame_average.answer:318` | read |
| C10-N6 | L | Gates not clean despite `plan.md:385-386`: pyflakes `compose.py:587`; classifier exit 1 | `tools/classify_path_finders.py:99,105` | verified |

</details>

---

## 6. Status of every 2026-10-14 finding at 2.24.0

`ID` numbers follow the order of the 2026-10-14 review's appendix (`A<area>-<nn>`). Status codes:

| Code | Meaning |
|---|---|
| S | still present |
| F | fixed |
| P | partly fixed |
| M | moved/changed |
| ND | not a defect |
| CV | cannot verify |

"(c)" marks a finding whose file is byte-identical to `0ff77d99`, so its status was carried over.

### A1 — spine and architecture

| ID | Finding (short) | Status | Where at 2.24.0 |
|---|---|---|---|
| A1-01 | Design principle "metadata on the dataclass" | S | `design.md:182-184`; now also contradicted by `backend-architecture.md:77` |
| A1-02 | `trajectory_log.emitter` imports L2 | S | `emitter.py:45` (22 modules) |
| A1-03 | L1 modules import L2 in functions | S (worse) | `identity.py:115`; `structure.py:1201`; `task.py:338,724,741,1408`; `electronic_state.py:227`; `runfiles.py:2075`; `warmfiles.py:186` |
| A1-04 | Layer tables disagree | S | `backend-architecture.md:55,220` |
| A1-05 | Scheduler L1 rationale | S | `design.md:116-121`; `scheduler/record.py:41,46`; `queue.py:132-178` |
| A1-06 | Upward floor imports | S (24) | — |
| A1-07 | runwrap floor 5 vs 3 | S | `runwrap.py:46` vs `:3871` |
| A1-08 | Counts of jobset floors | M | `README.md:34` now right; `execution/architecture.md:151` "all seven" now wrong; `architecture.md:302` four |
| A1-09 | `make web` broken | S | `Makefile:38,46` (re-run) |
| A1-10 | Makefile ignore/help | S | `Makefile:28` |
| A1-11 | Root README stale | S | — |
| A1-12 | conventions.md command lists | S (worse) | `jobset scratch` missing too |
| A1-13 | Missing test files cited | S (37) | — |
| A1-14 | Index not "the ONE index" | S | — |
| A1-15 | GET writes `toc.json` | S | `docs.py:329,356-360` |
| A1-16 | Broken plans/ links | S | `installation.md:158`; `env-framework.md:147,633` |
| A1-17 | backend-architecture symbols | S | `backend-architecture.md:80,213,309` |
| A1-18 | `sidecars.{to_dict,…}` | S | `architecture.md:239` |
| A1-19 | "only via `from molbuilder import`" | S | `cli.py` has 97 submodule imports |
| A1-20 | package-layout behind | S (worse) | — |
| A1-21 | Hand-rolled atomic writes | P | `1a0ae147` moved files and docs to `persist`; others remain |
| A1-22 | Timestamp helpers | S | 6 helpers, 4 formats |
| A1-23 | runwrap god module | S | 4,559 lines; `render_run_wrapper` 1,912 lines |
| A1-24 | cli.py domain logic | S | 3,586 lines |
| A1-25 | Lazy imports | S | 2,092 in functions vs 1,741 at module level |
| A1-26 | Contract restated | S | — |
| A1-27 | Spine docs mostly history | S | `docs/README.md` 102,672 bytes |
| A1-28 | `scripts/` not inventoried | S | — |
| A1-29 | Dead `capture_screenshots.py` | S | — |
| A1-30 | Small drift | P | — |
| A1-31 | `composed_junction` twice | F | 74916c45 |
| A1-32 | Prep-import rule understated | S | `submit.py:888,1605,2788` |
| A1-33 | pyflakes hook | S | 404 (re-run) |

### A2 — data model

| ID | Finding (short) | Status | Where at 2.24.0 |
|---|---|---|---|
| A2-01 | Sidecar hash never compared | F | badc9178 |
| A2-02 | Migration never fires | S | `parse/sidecars/molstruct.py:112`; `sidecars/molstruct.py:798-800` |
| A2-03 | pyscf-geom without sidecar | F | badc9178 |
| A2-04 | One sidecar for two geometries | P | loud since 2.10 |
| A2-05 | `write(fmt)` vs suffix | S | `workingcopy_structure.py:507` |
| A2-06 | `frozen_atoms` setter | S | `structure.py:1620` |
| A2-07 | `vacuum` NaN/inf/negative | S | `structure.py:889`; `periodicity_gate.py:397-401` |
| A2-08 | Half-applied sidecar | S | `sidecars/molstruct.py:774-793` |
| A2-09 | Sidecar identity overrides PDB | P | — |
| A2-10 | PDB drops atoms | S | `structure.py:2141-2145,2308,2311` |
| A2-11 | HG21 → Hg | S | `structure.py:2199` |
| A2-12 | Second PDB reader | S (c) | `builders/backends/_common.py` |
| A2-13 | Bond detection ×3 | S (c) | — |
| A2-14 | Chemistry helpers drop metadata | S (c) | — |
| A2-15 | `.XV` metadata rule | M | `cli.py:1071,1110` |
| A2-16 | `<target>.tmp` writers | S | `workingcopy_structure.py:560`; `molstruct.py:658` |
| A2-17 | Two element tables | S (c) | — |
| A2-18 | Label/channel merge crash | S | `structure.py:1507` |
| A2-19 | `ByElement` literal | S (c) | — |
| A2-20 | `parse/` scope claim | S | `parse.md:88` |
| A2-21 | Structure identities | P | spectra `structure_hash` retired (ae4101a5) |
| A2-22 | Two `ParseWarning` | S (c) | — |
| A2-23 | Remap ×5 | S (c) | — |
| A2-24 | `to_wire` swallows | S | `structure.py:1324,1340` |
| A2-25 | Channel keys/aliasing | S | `structure.py:483,510,517` |
| A2-26 | "pairwise-disjoint" docstring | S | `structure.py:38` |
| A2-27 | fdf z-span units | P (c) | — |
| A2-28 | XYZ title newline | S | `structure.py:2269-2273` |
| A2-29 | Drifted citations | S | `structure.md:372,743` |
| A2-30 | "ten sites" list | S | — |
| A2-31 | Schema/API drift | S | — |
| A2-32 | parse.md structure claims | S | `parse.md:28,169,530` |
| A2-33 | Docstrings | S (c) | — |
| A2-34 | `to_pdb` own `+1` | S | `structure.py:2305` |
| A2-35 | Axis-kind/vacuum parsing copies | S | — |

### A3 — engines

| ID | Finding (short) | Status | Where at 2.24.0 |
|---|---|---|---|
| A3-01 | WARN masks wrong-XC ERROR | S | `pseudos.py:814` (re-run) |
| A3-02 | vdW-DF blocked | S | `pseudos.py:656-660` (re-run) |
| A3-03 | Au "misnamed" | S | `pseudos.py:613,738` (re-run) |
| A3-04 | Au1 "missing" | S (worse) | `pseudos.py:716`; now raises, see C3-N1 |
| A3-05 | No XC pairing check | S | re-run (GGA+CA, LDA+PBE, VDW+PBE) |
| A3-06 | Dispersion double counting | S | `pyscf/layout.py:204` |
| A3-07 | symmetry + geomeTRIC | S | `pyscf/input.py:480` |
| A3-08 | Impossible values warn only | S | `validation/metadata.py:87-110` (re-run) |
| A3-09 | Geometry blocks ×2 | S | `siesta/input.py:1144,1163` vs `transiesta.py:330,431` |
| A3-10 | tuning.md eigensolver default | S | `tuning.md:983-984` |
| A3-11 | stages.md group counts | F | cd7deb94 |
| A3-12 | template.md anchor | S | `template.md:1010` |
| A3-13 | `_GEOM_KWARG` literal | S | `pyscf/layout.py:66-73` |
| A3-14 | overview.md retired names | S | `engines/overview.md:359,367-369` |
| A3-15 | `kind` disagreements unguarded | F | df200c2b |
| A3-16 | SO + SR pseudos warn only | S | `pseudos.py:646` (re-run) |
| A3-17 | Status counts in docs | S | `pseudopotentials.md:52,90,280` |
| A3-18 | siesta.md k=1 | S | `siesta.md:498` |
| A3-19 | siesta.md block table | S | `siesta.md:240,250` |
| A3-20 | Ladder builders ×2 | S | `siesta/stages.py:29` vs `pyscf/stages.py:56` |
| A3-21 | `gto.M` ×2 | S | `pyscf/input.py:469`; `vibration_emitters.py:424` |
| A3-22 | PSML parsed repeatedly | S | `validation/siesta.py:139,179,262,324` |
| A3-23 | Makov–Payne ×2 | S | `makov_payne.py:129,364` |
| A3-24 | tuning.md stale | P | — |
| A3-25 | "tight" 0.01 vs 0.001 | S | catalogue `:556` vs `tuning.md:151,162` |
| A3-26 | Level shift wording | S | `pyscf.md:770-772` |
| A3-27 | SaveHS default | CV | no SIESTA manual offline |
| A3-28 | LDA authors mismatch | S | `pseudos.py:303` (re-run) |
| A3-29 | Dispersion template authors | S | `siesta/input.py:1644` |
| A3-30 | science/overview.md stale | S | — |
| A3-31 | `ecp_atoms` default | S | catalogue `:1444-1455` |
| A3-32 | GPU BlockSize cap | S | `siesta/input.py:166-168` (re-run) |
| A3-33 | Default relaxation CG 0.02 | S | `config/siesta.py:261,263` |
| A3-34 | overview.md two GPU items | S | `engines/overview.md:182,189` |

### A4 — transport and vibrations

| ID | Finding (short) | Status | Where at 2.24.0 |
|---|---|---|---|
| A4-01 | I(V) sign | S | `record.py:316` (re-run); opt-in path now |
| A4-02 | Raman "not labelled" | ND | — |
| A4-03 | c ∥ z unchecked | S | `transiesta.py:192`; `wizard.py:205,386-389` |
| A4-04 | kz 40 vs 100 | S | `transport.md:1265,1274` |
| A4-05 | §2a.6 vs §0.3 | S | `transport.md:344` vs `:1176` |
| A4-06 | Bias Class C diagram | S | `transport.md:1129,1132` vs `:1069` |
| A4-07 | `TRANSPORT_AXIS` ×3 | F | 825287c2 |
| A4-08 | Two lower-lead decisions | S | `sort.py:105-107`; `transiesta.py:191-194` |
| A4-09 | Items declared twice | P | default and type still twice: `config/siesta.py:219,459,477,479` |
| A4-10 | Thermal spread Py + JS | F | e5b60a59 |
| A4-11 | `harmonic_analysis` claim | S | `normal-modes.md:93-94` |
| A4-12 | Rotation origin wording | F | 825287c2 |
| A4-13 | `elec-pos` always written | S | `transport.md:3991` vs `transiesta.py:572-587` |
| A4-14 | Bias sign wording | S | `transport.md:4091-4092` |
| A4-15 | Caveat "still landing" | S | `transport.md:987` |
| A4-16 | Code cites missing sections | P | `deck.py:527,711`; `wizard.py:9` |
| A4-17 | 12 Å note | F | 825287c2 |
| A4-18 | "Wizard deleted" | S | `transport.md:4118-4119` |
| A4-19 | Dead cell-less fallbacks | S | `wizard.py:381-383`; `transiesta.py:359-360` |
| A4-20 | §7 baseline conflicts | S | `transport.md:4991,4993` |
| A4-21 | Frame generator present tense | S | `transport.md:1427` |
| A4-22 | Held-atom IR caveat | S | `vibration.md:864-865` |
| A4-23 | Coupling per display vector | S (deliberate) | `pyscf/vibration_emitters.py:1312-1318` |
| A4-24 | Two unit conventions | S (documented) | `constants.py:101-106` |
| A4-25 | "isotropic" label | S | `pyscf/vibration_emitters.py:1206` |
| A4-26 | transport.md history | S (grew to 5,217 lines) | — |
| A4-27 | PySCF Hessian shapes | CV | PySCF not installed |
| A4-28 | TBtrans current doubling | ND | — |

### A5 — execution

| ID | Finding (short) | Status | Where at 2.24.0 |
|---|---|---|---|
| A5-01 | Launch exit 0 | P | "no" and a partly refused bench still exit 0: `_cli.py:2160`; `submit.py:1830` |
| A5-02 | Retry record collision | S | `runrecord.py:427` (re-run) |
| A5-03 | `jobset prep run` in docs | P (c) | `worked-example.md:488` |
| A5-04 | job-set.json example | F | 825287c2 |
| A5-05 | Hierarchical tree names | S (c) | `overview.md:251` |
| A5-06 | §2.6 names, by-hand line | P | `job-contracts.md:1075` |
| A5-07 | `30:00` → 30 h | S | `quantities.py:144` (re-run) |
| A5-08 | Retry log leakage | F | 5c63bf95 |
| A5-09 | Restore atomicity | S | `checkpoint.py:1833` |
| A5-10 | migrate non-atomic | S | `migrate.py:817,827-834` |
| A5-11 | Web attempt count | F | 50b8ef68 |
| A5-12 | God functions | S (grew) | `render_run_wrapper` 1,912 lines; `_plan_calculation` 487 |
| A5-13 | PySCF arg loop | S | `runwrap.py:2844` |
| A5-14 | Floor-1 upward imports | S | `calcdirs.py:129`; `errors.py:33`; `machine.py:130` |
| A5-15 | `ask.canonical_*` | S | `job-contracts.md:2373-2374` |
| A5-16 | job-set.json docstring | S | `model.py:591` |
| A5-17 | "stdout teed" | S | `runwrap.py:2819` vs `running-a-job.md:119,640` |
| A5-18 | Doc contradictions | P | `job-system.md:42`; `running-a-job.md:15`; `architecture.md:959` |
| A5-19 | `_DONE`; "no ladder" | S | `runstatus.py:58`; `pipeline_log.py:125` |
| A5-20 | Writes skip persist | P | `summarize.py:342` fixed (1a0ae147); `script_emit.py:1542,1990` remain |
| A5-21 | Walk quoting; exit codes | S | `submit.py:2156,2174,2226` |
| A5-22 | Clocks | S | `runwrap.py:242,436,3090,3428`; `submit.py:2145` |
| A5-23 | Parsing | P (c) | `parse_walltime('1:2:3:4')` = 3723 |
| A5-24 | Restore CLI; save message | S | `cli.py:3310` |

### A6 — configuration and ops

| ID | Status | Where at 2.24.0 |
|---|---|---|
| A6-01 | P | as at 2.11.0 (re-run) |
| A6-02 | S | `rate_limit.py:219-222` (re-run) |
| A6-03 | S | `rate_limit.py:131` |
| A6-04 | S | `notify_setup.py:302,414,587` |
| A6-05 | F | ebaaebb4 |
| A6-06 | S | `cli.py:1313` |
| A6-07 | S | `cli.py:2366` |
| A6-08 | S | `serve_daemon.py:224` |
| A6-09 | S | `install-env.sh:593-600`; `diagnostics.py:287-299` |
| A6-10, A6-11 | S | — |
| A6-12 | F | ebaaebb4 |
| A6-13 | S | `cli.py:1426-1440` |
| A6-14 | S | `recipes.py:76-79` |
| A6-15 | S (narrowed) | — |
| A6-16 | S | — |
| A6-17 | S | `cli.py:2517` |
| A6-18 | S | `cli.py:1287` |
| A6-19 | S | `deployment.md:549` |
| A6-20, A6-21 | S | — |
| A6-22 | S | `access-control.md:78` vs `:97` |
| A6-23 | M | `configuration.md:1127` |
| A6-24 | S (narrowed) | `install-env.sh:289` |
| A6-25 | S | `deployment.md:623` |
| A6-26 | S | `runtime_config.py:537-563` |
| A6-27 | S | — |
| A6-28 | F | 5d0a33de |
| A6-29 | S | — |
| A6-30 | S | `builds.py:1881-1895` |

### A7 — web backend

| ID | Status | Where at 2.24.0 | Note |
|---|---|---|---|
| A7-01 | P | `/api/files/roots` with foreign Host → 200 | Host checked for the notebook only |
| A7-02 | P | cross-site multipart upload → 200 | Origin check notebook-only |
| A7-03 | P (c) | — | = A6-01 |
| A7-04 | S (c) | `rate_limit.py:220-222` | = A6-02 |
| A7-05 | S | `notify_setup.py:415,588` | `/reports` ungated too |
| A7-06 | S | `files.py:2310`; `projects.py:290` | re-run |
| A7-07 | F | 1a0ae147 | see C7-N3 |
| A7-08 | S | `watch.py:59` global `_state` | |
| A7-09 | S (c) | `pubchem.py:41` `socket.setdefaulttimeout` | |
| A7-10 | S | `/api/nope` 404 text/html | re-run |
| A7-11 | S | `projects.py:290` | re-run |
| A7-12 | F | 62fb65b0 | |
| A7-13 | S | `web-api.md:718-722` vs `access-control.md:358` | |
| A7-14 | S | `_shared.py:475` `err` used only by `modify.py:60` | = A7-N6 |
| A7-15 | S | `watch.py:535,628`; `transport.py:259` | |
| A7-16 | S | `web-api.md:382` | |
| A7-17 | S | `watch.py:308` → 500 (re-run); `build.py:323-328` | |
| A7-18 | S | `files.py:711` | |
| A7-19 | S | `build.py`: 19 routes, 2,700 lines | |
| A7-20 | S | `build.py:1223` vs `:1594-1597` | |
| A7-21 | S | `structure/save` at `projects/top.xyz` → 200 | |
| A7-22 | S | `files.py:1787,2015` | |
| A7-23 | S | `files.py:1582` | re-run |
| A7-24 | S | `files.py:1591-1604` | |
| A7-25 | S | 33 `except _PickerError` | |
| A7-26 | S | `web-api.md:768` | re-run: multipart → 400 |
| A7-27 | S (c) | `overview.md:8,136` | |
| A7-28 | S (c) | `auth.py:85` | |
| A7-29 | S (c) | `cas.py:147-150` | |
| A7-30 | F | ebaaebb4 | |
| A7-31 | S | `docs.py:358`; `notify.py:283` | |
| A7-32 | ND | — | |
| A7-33 | S | 40 routes | re-run sweep |

### A8 — front end 1

| ID | Status | Where at 2.24.0 |
|---|---|---|
| A8-01 | F | 8597632f |
| A8-02 | S | `stores.js:278-291` (re-run) |
| A8-03 | S | `projects.md:398` |
| A8-04 | S | `molview.md:4899` vs `:3759`; `model.js:409-411` |
| A8-05, 07, 08, 10 | S (c) | — |
| A8-06 | S | `workspace.md:248` |
| A8-09 | S | `molview.md:4335`; `ui.js:1104` |
| A8-11 | S (c) | — |
| A8-12 | P | `molview.md:4409` "four things" vs 6 at `mount.js:344-351` |
| A8-13 | S | `tabs.md:118,139` |
| A8-14 | P | `molview.md:4275` |
| A8-15 | S | `molview.md:747` vs `3dmol-embed.js:75` |
| A8-16 to A8-19 | S (c) | — |
| A8-20 | F | 825287c2 |
| A8-21 | S | `viewer.js:1780,1797` (now also drops served lane asks) |
| A8-22 | S | `documents/page.js:100-112` |
| A8-23 | F | ee68556e |
| A8-24 | S | `modify/viewer.js:409-413` |
| A8-25 | S | `form-schema.md:356` |
| A8-26, 27, 29, 32, 33 | S (c) | — |
| A8-28 | S | 10 `$` / 9 `_basename` |
| A8-30 | M | = A8-N3 |
| A8-31 | S | `viewer.js` 4,892 lines, 38 top-level `let`/`var`; `ui.js` 3,014 |

### A9 — front end 2

| ID | Status | Where at 2.24.0 |
|---|---|---|
| A9-01 | S | `_seal.js:339,557-561` (re-run) |
| A9-02 | S | `core.js:3218`; `index.js:356-358` |
| A9-03 | F | 691e7f0a |
| A9-04 | S | `core.js:3390-3404,3664-3666` |
| A9-05 | S | `trajectory.md:301-305` |
| A9-06 | S | `watch.py:55` |
| A9-07 | F | 825287c2 |
| A9-08 | S | `spectra.md:87-91` vs `index.js:340-345` |
| A9-09, A9-10 | S | `spectrumchart.md:590,601` |
| A9-11 | F | a10c49db |
| A9-12 | F | e5b60a59 |
| A9-13 | S | `spectra/core.js:442`; `trajectory/core.js:336` |
| A9-14 | S | `viewer.js:8` (784 lines) |
| A9-15 | S | `bench-summary.js:634,645` |
| A9-16, A9-17 | S | `results.md:1016,677-683` |
| A9-18 | S | `vibrationview.md:355` |
| A9-19 | S | — |
| A9-20 | P | `web-api.md:630`; constants and `motion_share` undocumented |
| A9-21 | P | `spectra.md:657-658` |
| A9-22 | P | `registry.js:18,94-97` |
| A9-23 | P | `viewer.js:383` heuristic remains |
| A9-24 | S | `core.js:2877` |
| A9-25 | S | `viewer.js:663` |
| A9-26 | S | 7 dispatches |
| A9-27 | S | `core.js:2150-2153,2924`; `trajectory/core.js:2429` |
| A9-28 | S | — |
| A9-29 | S | `core.js:3423-3429` |
| A9-30 | F | 825287c2 |
| A9-31 | S | `_maths.js:61-69`; `core.js:2123-2127` |
| A9-32 | S | `viewer.js:51` |

### A10 — tests, plan, hygiene

| ID | Status | Where at 2.24.0 |
|---|---|---|
| A10-01 | S | `test_where_an_item_binds.py:286-288` |
| A10-02 | S | `oauth.py:59` |
| A10-03 | S | — |
| A10-04 | ND (re-diagnosed as A10-N2) | — |
| A10-05 | S | 5 tests |
| A10-06 | S | `Makefile:26-28` |
| A10-07 | S | unit 1, module 6, interface 0, smoke 0 |
| A10-08 | P | `testing.md:628-635` |
| A10-09 | S | 15 `run_node` |
| A10-10 | P | — |
| A10-11 | F | 769f8f4c |
| A10-12 | S | `test_electrode_pair_seam.py:64` |
| A10-13 | P | — |
| A10-14 | S | 18 waits |
| A10-15 | S | 3 × `_allowed_root` |
| A10-16, A10-17 | F | archived |
| A10-18 | P | drift 0–20 lines |
| A10-19 | S | `plan.md:12` dated 2026-10-09 vs hand-over 10-18 |
| A10-20 | S | ZIP pair |
| A10-21 | S | — |
| A10-22 | S | `el()` in `viewer.js:127`, `run-panel.js:245`, `form-schema.js:62`, `ui.js:227`, `mount.js:450` |
| A10-23 | CV | not re-traced |
| A10-24, A10-25 | S | — |
| A10-26 | S | `files.py` 1726↔2247, 2018↔2154 |
| A10-27 | F | df200c2b |
| A10-28 | S | `plan.md:485` |
| A10-29 | S | `molwatch.py:125`; `state.js:532`; `_atom.js:164` |
| A10-30 | S (worse) | archive 6.7 MB, 78 files, 14 unindexed; `README:23` broken |
| A10-31 | S | — |
| A10-32 | S | mermaid 3,337,508 B |
| A10-33 | S | — |

---

## 7. Cross-cutting observations

1. **The consolidation pattern keeps dropping guards.** The 43 commits moved many things to "one home" or "one door". Examples: one writer (`persist.write_bytes`), records opened through `calculation.open`, names spelled once through `runfiles.fixed_name`, file roles through the parser registry. This is good design, and most of it is sound. But several regressions come from exactly these moves, where the old path carried a check the new one did not:
   - **one writer:** lost the file mode under umask (C7-N3) and lost fsync (C1-N6);
   - **one name rule:** a labelled species now raises (C3-N1);
   - **one door:** a failed open is swallowed, so the save guards are skipped (C7-N1); the bench `commands` route now 500s (C7-N4);
   - **parser registry:** `_initial.xyz` is now claimed twice (C9-N2).

   When a rule moves to one home, list what each old copy refused, and add one test per refusal at the new home.
2. **Security has not moved since 2.11.0.** Every priority security item sits on files byte-identical to `0ff77d99`. Meanwhile new routes added new holes: the reports traversal (C6-N1), and an ungated `/reports` route. A7-N1 also turned out to be command execution. The fix is still one small, separate change:
   - one `before_request` check of Origin/`Sec-Fetch-Site` and Host for every state-changing route;
   - refuse `.git`/`NEVER_STORED` path components in `_resolve_within_roots`;
   - validate `user` as a plain name equal to `g.user`;
   - take the rightmost XFF hop;
   - add an admin gate on `notify_setup`.
3. **The review gates are still red, and new code adds to them.**
   - The pyflakes hook still cannot install, and the commit that fixed one pyflakes hit added another.
   - The classifier still exits 1.
   - A dead guard function was found (C1-N2).
   - A tier-1 test depends on the host (C10-N1).
   - A missing netCDF4 is masked as a missing internal module (C10-N2).

   `plan.md:385-386` now makes clean gates a milestone condition, and they are not clean.
4. **"Two implementations of one rule" is shrinking, but not gone.** Fixed:
   - the transport axis (A4-07);
   - the thermal spread, Python plus JS (A4-10);
   - the config/catalogue copy (A10-27).

   New or remaining:
   - two GPU-per-node rules (B2-N5);
   - two `_number` helpers (C1-N9);
   - a deck reader in `gf_files` beside `parse.fdf`, which ignores units (C4-N6);
   - default and type still in config and catalogue, guarded with gaps (A4-09 / C3-N6).
5. **Layering keeps eroding.**
   - L1→L2 import sites went from 6 to 10.
   - Upward floor imports went from 21 to 24.
   - The new A16 doors are not placed on any floor (C1-N1).

   An AST check over all imports, function-level ones included, would make this visible at each commit.
6. **Doc size is moving the right way in the plan only:**
   - `plan.md`: 285 → 205 KB;
   - `transport.md`: 4,969 → 5,217 lines;
   - `docs/README.md`: still 102 KB;
   - the archive: 6.7 MB, 14 files unindexed.
7. **What improved** in the 43 commits:
   - the test suite is stable across worker counts;
   - `TRANSPORT_AXIS` is one constant;
   - the thermal spread is served by the server;
   - the catalogue is the one home of parameter help;
   - page saves keep the file mode;
   - the leads' GF file reuse is scientifically sound as built;
   - the new `ncdata` record round-trips;
   - the This machine page is XSS-safe.

---

## 8. Recommended order of work

1. **Security, one change (≈ a day):**
   - **refuse `.git` and other `NEVER_STORED` components in every file route** (A7-N1, which gives command execution);
   - an Origin/`Sec-Fetch-Site` and Host check in one `before_request` for every state-changing route (A6-N1/A7-02, A7-01);
   - `user` validation and per-user authorization on `/api/notify/reports` (C6-N1);
   - an admin gate on `notify_setup` (A6-04);
   - the rightmost XFF hop (A6-02);
   - harden the notebook pass-through (B1-N1) and gate (A6-01);
   - `persist.write_bytes`: honour umask for new files (C7-N3).
2. **Regressions from the consolidation (small, local):**
   - rename the loop variable in `transport/record.py:1161` and add a two-frame low-bias test (C4-N1);
   - accept labelled species in `fixed_name(PSEUDO_FILE, …)` (C3-N1, then A3-N1 for the coverage check);
   - catch only `NotACalculation` in the save door and refuse on a reader error (C7-N1);
   - drop `_initial.xyz` from `PySCFOutFileParser.can_parse` (C9-N2);
   - wrap the bench `commands` branch (C7-N4);
   - fsync in `write_bytes` (C1-N6).
3. **Data loss and integrity:**
   - symlink-safe delete, rename and move (A7-06);
   - an exclusive retry record (A5-02);
   - job address keyed by cluster (A5-N1);
   - decide `jobset scratch` liveness per run, and add a test (C5-N1, C5-N7);
   - refuse `jobset migrate` while a run is going (C5-N2);
   - frame-set unit and weight checks in one place, used by every path (C2-N1, C2-N2);
   - boolean typing in the Task setup cells (C8-N2);
   - carry the sidecar `schema_version` (A2-02);
   - neutralise XYZ titles (A2-N1, A2-28);
   - validate `frozen_atoms` and `vacuum` (A2-06/07);
   - a sidecar-aware save door (A2-N3);
   - write the spectra `.nc` before the JSON (C4-N3).
4. **Science:**
   - signed I(V), including the spin halves (A4-01, C4-N10);
   - `check_coverage` by the exact `<label>.psml` (A3-N1), which fixes A3-03/04;
   - most-severe status plus a GGA family for vdW-DF (A3-01/02);
   - the `.nc` unit of `eigenvector_canonical` (C4-N2);
   - SLURM `MM:SS` (A5-07/N3);
   - peptide hydrogens (A10-N2);
   - hard limits for impossible values (A3-08).
5. **Functional:**
   - the spectrum stick colours (A9-01);
   - plain `.xyz` to the structure preview (C9-N1);
   - the monitor survives a refused card layout, falling back to the shipped file (C6-N3);
   - the reports reader skips a malformed line (C6-N2);
   - send `tsgf` on ladder rows (C8-N1);
   - catch `ComposeError` (A4-N1);
   - a JSON error handler on `/api/*` (A7-10/33);
   - `make web` (A1-09);
   - a narrow `orderSaid` getter (A8-N1).
6. **Restore the gates:**
   - a working pyflakes hook;
   - fix `compose.py:587`;
   - classify `joblist.py:197` and drop the two stale overrides;
   - call or delete `_every_cited_item_is_read_back`;
   - stub the environment manager in the machine-record test (C10-N1);
   - choose bundle fallbacks by `__package__`, so a real `ImportError` surfaces (C10-N2);
   - state netCDF4, setuptools and authlib as tier-1 requirements, or gate them;
   - remove or write the 37 cited-but-missing tests.
7. **Docs:** fix the user-facing commands and examples, then the contradictions in §§ 5–6. Move dated narrative out of contract documents.

---

## 9. Method and blind spots (R3)

**Round 3, what was done:**

- Each area's diff `0ff77d99..d77bb564` was read in full, along with every cited prior location.
- **Run on `d77bb564`** (Python 3.13 venv with netCDF4, all state in scratch; both checkouts confirmed unchanged):
  - the tier-1 suite at `-n 4` and `-n 6`, plus a run with netCDF4 hidden;
  - Flask test-client probes:
    - route enumeration (103 rules);
    - symlinks;
    - file modes under umask 022/077;
    - `.git` writes leading to `fsmonitor` execution, same-site and cross-site;
    - error content types;
    - a non-object-body sweep over every mutating route;
    - relative bases;
    - save-door guards;
    - the depth rule;
    - reports traversal;
  - pseudopotential coverage and the copier on synthetic PSML, including labelled species;
  - `validate()` on edge-case configs;
  - `linear_response_iv` with negative bias, and `_iv` with two frames;
  - `.nc` write/read edge probes;
  - the GF k-point and size estimator against the doc's numbers;
  - the zero-point constant against scipy;
  - structure round trips and the save door;
  - node reproductions:
    - A9-01;
    - A8-02;
    - C8-N2 (cell readers);
    - C9-N1 (the real inspector registry);
  - an AST import graph against the L1 index and floor table;
  - duplicate-window scans for Python and JS;
  - pyflakes and `classify_path_finders --check`;
  - the A5 reproductions in a scratch pytest harness built on the repository's road fixtures and stub `sbatch`: `--time 30:00`, launch exit codes, `record_retry`, the job address, `jobset scratch` with a live older run, a hand-edited header, a bad wall;
  - the monitor bundle imported on bare Python 3.11, 3.12 and 3.13 (`-I -S`).

**Rounds 1 and 2:** as recorded in the 2026-10-18 version of this audit (in git history under its old name, `code-audit-2026-10-18.md`; this document absorbs it).

- Round 1: each area's diff since `83e72b2` and every baseline location.
- Round 2: the four 2.11.0 commits in full, including a stand-in Jupyter upstream and the jupyter_server and JupyterLab source.

**What could not be seen:**

- **No real engines:** no SIESTA, TranSIESTA, TBtrans or PySCF runs. PySCF and openbabel are not installed (A3-27 and A4-27 are CV). So the claims about TranSIESTA's `check_Green`, `GF.ReUse` and the GF energy reference rest on the docs. C3-N2 and C4-N8 are unverified.
- **No real SLURM:** requeue and error-file append behaviour (C5-N4) and A5-N2 are unverified. Whether a running TranSIESTA survives its GF files being deleted (the real consequence of C5-N1) is unknown. `jobset scratch` was simulated on a SIESTA H2 ladder with hand-written launch records. migrate's mid-write crash and its race with a live run were not induced.
- **No real browser (R4):** front-end items rest on node reproductions and reading. HTTP 500s were shown with the Flask test client, not in a page.
- **No real supervisor, Jupyter, nginx, Slack or Discord.** Auth-enabled mode was not run, so the cross-user leg of C6-N1 is by reading.
- **No e2e or tier-2/3 lanes.**
- **Unchanged code was carried over, not re-read end to end.** About 30 % of statuses carry over on byte-identical files, each area spot-checked three.
- **Plan sampling:** about 14 of some 160 `file:line` citations in `plan.md`, and about 7 of 90+ closed lane findings, were spot-checked.
- **Duplicate scans** cannot see renamed or reshaped copies, or copies shorter than 8 lines.
