# Full code audit — main at 2.11.0, validated against the 2026-10-14 review

**Role:** audit report
**Domain:** process
**Companions:** [`code-audit.md`](?doc=process/code-audit.md) (the playbook this audit follows: § 1 principles, § 1b review rules R1–R4, § 1c.2 how to report); [`plans/plan.md`](?doc=plans/plan.md) (where open work is tracked; row **AU1** points here); [`testing.md`](?doc=process/testing.md) (the tier-1 lane run below)
**Date:** 2026-10-18
**Audited commit:** `origin/main` **`0ff77d99`**, molbuilder **2.11.0** (first round at `f3d5a4e4`, 2.10.2; re-validated at `0ff77d99`, see § 0.1)
**Baseline:** the full review of 2026-10-14 at **`83e72b2`** (2.9.4); 110 commits lie between them

---

## 0. What this is, and how it was done

This audit does two things:

1. **It validates the 2026-10-14 review against current main.** Each of that review's 315 findings was re-checked at its place in the code on `f3d5a4e4` and given one status:
   - **FIXED**;
   - **PARTLY FIXED**;
   - **STILL PRESENT**;
   - **MOVED/CHANGED** (still a problem, different place or shape);
   - **NOT A DEFECT** (the original claim was wrong on re-check);
   - **CANNOT VERIFY** (what is missing is named).
2. **It reviews what changed** since the baseline in each area, looking for new defects. New findings are numbered `A<area>-N<k>`.

**Method.** The work was split into ten areas, the same split as the baseline review. One reviewer per area worked on a read-only checkout of main. Each followed this playbook:

- every finding is a candidate until verified (§ 1);
- the silent consequence is stated (§ 1c.2);
- each reviewer names the technique used and what it cannot see (§ 1b R3).

High-severity items were re-run where the code could be run. After the area reports came in, these key items were re-run independently on `f3d5a4e4` when writing this document:

- the I–V sign (A4-01);
- the `30:00` wall time (A5-07);
- the symlink-delete code path (A7-06);
- the `.git` exclusion being applied only to archive export (A7-N1).

**Labels used below:**

- *(verified)* — reproduced by running code on main.
- *(read)* — confirmed by reading the code at the cited line.
- *(unverified)* — plausible, not demonstrated.

Line numbers are those of `f3d5a4e4` unless a row says *HEAD*, which means `0ff77d99`. For a file the four 2.11.0 commits did not touch, the two are the same.

**Areas:**

| Area | Scope |
|---|---|
| A1 | Spine docs and global architecture (layers, floors, Makefile, tooling) |
| A2 | Data model: `Structure`, cell, chemistry, sidecars, parse stack |
| A3 | Engines: SIESTA, PySCF, catalogue, stages, pseudopotentials, validation |
| A4 | Transport (TranSIESTA/TBtrans) and vibrations/spectra, Python side |
| A5 | Execution: prep, run wrappers, launch, checkpoints, JobSets, scheduler, `jobs` |
| A6 | Configuration, environments, serving daemon, auth, rate limiting |
| A7 | Web backend and HTTP API |
| A8 | Front end 1: MolView, Task Setup, tabs, workspace, projects, form schema |
| A9 | Front end 2: Results, inspectors, spectra, spectrum chart, VibrationView, trajectory |
| A10 | Test suite, plan, duplicated/dead code, repository hygiene |

### 0.1 Re-validation at 2.11.0 (`0ff77d99`)

While this audit was being written, main moved four commits:

| Commit | What it changed |
|---|---|
| `393c70a8` | serve start refuses a pidfile directory it cannot make, before it detaches |
| `ebaaebb4` | **The notebook goes through molbuilder's one port** (a pass-through at `/jupyter/`); a server restart stops its notebook |
| `e2fdb2d4` | plan update (2.10.3) |
| `0ff77d99` | **The machine's record is shown, never a veto** (2.11.0): `scheduler/admit.py` replaced by `scheduler/fit.py` |

Together they touch 84 files. Every finding was then validated against the actual code at `0ff77d99`, using this rule:

- **The cited file is unchanged** (`git diff f3d5a4e4 0ff77d99 -- <file>` is empty): the code is byte-identical, so the status carries over. Each of the three reviewers spot-checked three carried-over findings on HEAD, and all held.
- **The cited file changed:** the finding was re-checked on HEAD by one of three reviewers, split by cluster:
  - security, notebook and serving;
  - scheduler, jobsets and Task Setup;
  - docs, tests and the review gates.

  High items were re-run.

In total 111 findings were re-checked; the rest carried over. The four 2.11.0 commits were also reviewed in full for new defects; those are numbered `B<k>-N<j>` (§ 5b). After the cluster reports, the I–V sign, the `30:00` wall time and the unchanged status of the files behind the other priority findings were re-checked independently on `0ff77d99`.

---

## 1. Results at a glance

### 1.1 Status of the 2026-10-14 findings on main (first round, `f3d5a4e4`)

| Area | Prior | Fixed | Partly | Still present | Moved | Not a defect | Cannot verify | **New** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 33 | 1 | 1 | 29 | 2 | 0 | 0 | **11** |
| A2 | 35 | 2 | 3 | 28 | 2 | 0 | 0 | **9** |
| A3 | 34 | 1 | 1 | 31 | 0 | 0 | 1 | **5** |
| A4 | 28 | 0 | 0 | 25 | 0 | 2 | 1 | **5** |
| A5 | 24 | 1 | 5 | 17 | 0 | 0 | 1 | **6** |
| A6 | 30 | 0 | 0 | 30 | 0 | 0 | 0 | **4** |
| A7 | 33 | 1 | 0 | 31 | 0 | 1 | 0 | **6** |
| A8 | 33 | 2 | 1 | 29 | 1 | 0 | 0 | **7** |
| A9 | 32 | 2 | 3 | 27 | 0 | 0 | 0 | **3** |
| A10 | 33 | 4 | 6 | 22 | 0 | 0 | 1 | **11** |
| **Total** | **315** | **14** | **20** | **269** | **5** | **3** | **4** | **67** |

**Reading the table:**

- **About 85 % of the baseline findings are still present.** Most of the 110 commits since 2.9.4 went to new work and test rework (the "TST" series), not to the findings.
- **Security and ops (A6) had no fixes at all at `f3d5a4e4`.** `jupyter.py`, `serve_daemon.py`, `rate_limit.py`, `auth.py`, `notify_setup.py` and `install-env.sh` have no commits since the baseline.
- **Three baseline findings were wrong** (§ 4); two more were narrower than stated.
- **67 new findings**, most of them Low. Four are security- or data-relevant: A7-N1, A6-N1, A6-N2 and A5-N1/A1-N9.

### 1.1b Status at 2.11.0 (`0ff77d99`)

Changes from the re-validation:

- **The 315 baseline findings:**
  - Three moved to FIXED, all in `ebaaebb4`: A6-05 (Jupyter now always binds 127.0.0.1), A6-12 (no more http/https split under `--no-auth` + TLS) and A7-30 (the wildcard `frame-src` is gone).
  - Four moved to PARTLY FIXED: A6-01, A7-01, A7-02 and A7-03. The notebook gate now checks Origin and a loopback Host, but every other route still does not, and "loopback = same account" is unchanged.
  - The first-round findings A6-N1 and A6-N2 are also partly fixed, for the same reason.
- **Nothing else re-checked changed status,** including everything in the scheduler/jobset cluster and the docs/tests cluster. A5-N3 (`--time 30:00`) is now *shown* in the launch question, but still sends 30 hours.

| | Fixed | Partly | Still present | Moved | Not a defect | Cannot verify | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-14 findings at `0ff77d99` | **17** | **24** | **262** | 5 | 3 | 4 | 315 |
| First-round new findings (A*-N*) at `0ff77d99` | 0 | 2 | 65 | — | — | — | 67 |
| New in 2.11.0 (B*-N*, § 5b) | — | — | 15 | — | — | — | 15 |

**Open at 2.11.0:** 262 + 24 + 5 + 67 + 15 = **373 findings** (still present, partly fixed or moved). Most are Low.

### 1.2 The test suite on main

| Item | Value |
|---|---|
| Command | `pytest tests --ignore-glob='*_e2e.py' -n 4 --dist loadgroup -p no:cacheprovider` |
| Environment | Python 3.13 venv, all state directories in scratch, 194 s |
| **Result** | **6,566 passed · 13 failed · 4 errors · 54 skipped · 6 xfailed** (6,643 collected) |
| Baseline | 7,105 passed · 11 failed · 1 error · 53 skipped · 6 xfailed |

About 530 fewer tests are collected, because the TST work retired tests that read source files as text.

**No failure is a regression from the last 110 commits.** By cause:

| Failure(s) | Cause | Finding |
|---|---|---|
| 4 × `test_auth_config.py::TestARotatedSecretIsTakenUp` | authlib not installed and not gated | A10-02 |
| 5 permission tests | run as root, no euid guard | A10-05 |
| 2 × `test_seeding_a_fresh_machine.py:378,391` | fail whenever `$MOLBUILDER_PROJECTS` is set | A10-N1 |
| `test_smiles_benzene_planarity` | the test measures orientation, not planarity | A10-03 |
| `test_web.py::test_preflight_returns_issues_for_siesta` | **a product defect**: peptide hydrogens dropped without openbabel | A10-N2 |
| ERROR `test_where_an_item_binds.py` | e2e test in a non-`_e2e` file | A10-01, A10-N3 |
| 3 × ERROR `test_wheel_ships_the_front_end.py` | setuptools missing; they pass with it | A10-N11 |

`tools/testrun.py` was not used because it writes `.test-progress/` into the repository. The checkout was confirmed unchanged before and after the run.

**At 2.11.0 (`0ff77d99`), same command:** 6,581 passed · 11 failed · 4 errors · 55 skipped · 6 xfailed (6,657 collected, 198 s).

- **No new failure:** every failure and error is in the list above.
- The two seeding tests (A10-N1) passed in this run, but they are **not fixed**. A second run with `-n 6` failed one of them again, so the suite's result now depends on scheduling (B3-N1).
- One new skip: `test_serve_verbs.py:277`, a new test that skips under root.

---

## 2. Priority findings — open on main

Ordered by consequence. Every item here was verified on `f3d5a4e4`, by running code or by reading the line; the label says which. The **2.11.0** column gives the status at `0ff77d99`.

### 2.1 Security

| ID | Sev. | Finding | Location | Check | 2.11.0 |
|---|---|---|---|---|---|
| A6-01 / A7-03 | H | **No-sign-in server: "loopback = same account".** Any local account, or a cross-site page, can start the notebook (code execution as the server's account) and read its token | `web/blueprints/jupyter.py:111-118`; `docs/ops/access-control.md:121` | verified (cross-site POST → 202, signal sent) | **partly fixed** (`ebaaebb4`): a cross-site POST now gets 403. A loopback request with `Host: localhost` and no Origin still gets 202, and any local account can open `/jupyter/`, where the pass-through adds the token for it. The premise is still stated at HEAD `jupyter.py:84-85`, `:126` |
| A6-N1 / A7-02 | H | **The documented CSRF defence does not exist.** `connect-src 'self'` limits molbuilder's own pages, not requests other sites send. There is no Origin or `Sec-Fetch-Site` check anywhere | `web/app.py:231-232`; `docs/web/web-api.md:442` | verified (cross-site `jupyter/start` 202, `notify/listener/keys` 200, multipart upload 200) | **partly fixed**: the Origin check exists only for the notebook (HEAD `jupyter.py:285-295`). Re-run: a cross-site `notify/listener/keys` still returns 200 and issues a key; a cross-site multipart upload still returns 200 and writes the file. `app.py:229-230` and `web-api.md:445` still call CSP the CSRF defence |
| A7-01 | H | **No Host-header check.** A DNS-rebinding page can read the API | `web/app.py` (no host validation) | verified (`Host: rebind.evil.example` → 200 with the root path) | **partly fixed**: Host is checked for the notebook only. Re-run: `/api/files/roots` with a foreign Host → 200 |
| **B1-N1** | M (H on a shared host with sign-in) | **New in 2.11.0: `/jupyter/` relays whatever listens on 127.0.0.1:serve+1 onto molbuilder's own origin.** It never checks the listener is our notebook; it relays upstream `Set-Cookie`; and with no upstream CSP it sends only `frame-ancestors`. Another account can bind that port and run script as molbuilder's origin in a signed-in admin's browser | HEAD `web/blueprints/jupyter.py:357-381,437-444,473-476` | verified (stand-in upstream: page, `Set-Cookie` and the token header relayed); signed-in leg by reading | new |
| A7-N1 | H | **The file API writes inside a calculation's `.git/`.** `checkpoint.NEVER_STORED` is applied only to archive export; the server later runs `git` in that folder | `web/blueprints/files.py` write/upload/rename; `checkpoint.py:77,345` | verified write (`<calc>/.git/config` → 200); effect not demonstrated | unchanged files: still present |
| A6-02 | H | **Rate limiter takes the client-supplied leftmost `X-Forwarded-For`** | `web/rate_limit.py:219-222` | verified | unchanged: still present |
| A6-03 / A6-N2 | M–H | **Behind the documented same-host proxy, every request is loopback.** The limiter is inert, and the notebook gate gives kernels to everyone the proxy admits | `rate_limit.py:130-131`; `jupyter.py:111-118`; `docs/ops/deployment.md:296-303` | verified (limiter); read (gate) | limiter still present. Gate **partly fixed**: it now also needs a loopback Host, but nginx's default `Host: $proxy_host` still passes for proxied clients that send no Origin (by reading) |
| A6-04 / A7-05 | M–H | **Notify routes have no admin gate.** `/test` is an SSRF probe; any user can rotate another user's key | `web/blueprints/notify_setup.py:302,414,531` | read | unchanged: still present |
| A6-05 | M | **Jupyter binds to `serve`'s interface,** outside SSO and the limiter | `jupyter.py:793`; `cli.py:2450` | read | **fixed** (`ebaaebb4`): always 127.0.0.1 (HEAD `jupyter.py:109,755`) |
| B1-N2 / B1-N3 | L–M | **New in 2.11.0:** `deployment.md` § 2 does not say the proxy must keep `Host` and forward WebSocket upgrades, so the notebook breaks behind nginx defaults (unverified). The gate checks only Origin, so a cross-site page can probe `/jupyter/files/<path>` as a subresource (by reading) | `docs/ops/deployment.md` § 2; HEAD `jupyter.py` gate | read | new |

### 2.2 Data loss and data integrity

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| A7-06 | H | **Deleting a symlinked folder deletes its target and everything in it;** rename and move also act on the target | `files.py:176` (`_resolve_within_roots` resolves), `files.py:2336` `shutil.rmtree(resolved)` | verified (agent run) and read |
| A5-02 | H | **A flat warm retry overwrites another launch's run record;** two processes then share run files. job-system.md:733 now allows "both run" | `runrecord.py:377` | verified |
| A5-N1 / A1-N9 | M | **Job address files are keyed by job id alone,** so two clusters sharing a home folder overwrite each other's entries | `joblist.py:160,196` | verified |
| A7-07 | M | **`/api/files/write` makes files 0600;** run scripts lose their execute bit | `files.py:1614-1631` | verified |
| A2-02 / A2-N2 | M | **Sidecar v7–v11 migration never runs** (the reader stamps v14), yet two docs say it does | `parse/sidecars/molstruct.py:112`; `sidecars/molstruct.py:788-790` | verified |
| A2-10 / A2-11 | M | **PDB round trip drops atoms** (residue > 9999, x ≤ −1000 Å); `HG21` read as mercury | `structure.py:2082-2088,2144-2147,2258-2260` | verified |
| A2-N1 | M | **A typed title such as `Properties=…` makes the just-saved pair unopenable** (read as extxyz metadata) | `structure.py:2219-2229` | verified |
| A2-N3 | M | **The save door replaces another file's sidecar without a prompt** (`x.pdb` beside `x.xyz`), and accepts suffixes the loader refuses | `web/blueprints/build.py:508-511` | verified |
| A2-06 / A2-07 | M | **`frozen_atoms` setter and `vacuum` are unvalidated** (`[5, True]` → `[1, 5]`; NaN, inf and negative accepted) | `structure.py:1570,888-891`; `periodicity_gate.py:397-401` | verified |
| A2-08 | M | **A failed `apply_to_structure` leaves a half-applied structure** | `sidecars/molstruct.py:764-784` | verified |
| A5-09 | M | **An interrupted checkpoint restore leaves a folder no save ever held** | `checkpoint.py:1717-1804` | read |

### 2.3 Scientific correctness

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| A4-01 | H | **Linear-response I(V) has the wrong sign for V < 0:** I(−0.5 V) = I(+0.5 V) = +3.874 × 10⁻⁵ A. This is the default path | `transport/record.py:310` `half = abs(float(v)) / 2.0` | verified (twice) |
| A3-01…04, A3-N1 | H | **Pseudopotential coverage check:** a WARN masks a wrong-XC ERROR; every vdW-DF run is blocked; `Au.psml` is "misnamed" beside `Au-fr.psml`; label `Au1` is "missing". Root cause: `check_coverage` matches by declared element, while the copier uses the exact `<label>.psml` | `pseudos.py:603-826`; `validation/siesta.py:135` | verified |
| A5-07 / A5-N3 | M | **`"30:00"` is read as 30 hours** (SLURM: 30 minutes). `launch --time 30:00` sends `-t 1-06:00:00`, not re-admitted | `scheduler/quantities.py:144`; `jobset/_cli.py:1979` | verified (twice; again at 2.11.0, where the launch question now *shows* `-t 1-06:00:00` but still sends it) |
| A10-N2 | M | **Without openbabel, a peptide build drops 5 real hydrogens** and returns a radical. The later error blames charge and spin | `chemistry.py:1185-1211` | verified |
| A3-08 | M | **Impossible values only warn** (`MeshCutoff = 0`, `DM.Tolerance < 0`, `conv_tol = 0`, `relax_force_tol < 0`); the catalogue's `above` limit exists but is not applied to them | `validation/metadata.py:97`; catalogue | verified |
| A3-05/06/07/16 | M | **Missing checks:** no XC functional/authors pairing; `d3bj` with `wb97m-v`; symmetry with geomeTRIC optimisation; spin–orbit with scalar-relativistic pseudos only warns | `config/siesta.py:263-280`; `pyscf/layout.py:203-206`; `pyscf/input.py:472`; `pseudos.py:644` | verified (rendered decks) |
| A4-03 | M | **Transport assumes c ∥ z and a, b ⊥ z, unchecked** | `transiesta.py:169`, `sort.py`, `wizard.py`, `cell.py:1125,1233` | read |

### 2.4 Functional defects

| ID | Sev. | Finding | Location | Check |
|---|---|---|---|---|
| A9-01 | H | **Spectrum chart paints the wrong sticks while Raman values are pending** | `lib/spectrumchart/_seal.js:339,557-561` | verified (node) |
| A8-N1 | M | **The atom-order line deep-copies the whole trajectory on every model change:** 1.4 s per update at 2,000 frames × 300 atoms | `lib/molview/ui.js:1362-1370`; `model.js:761-772` | verified (timing) |
| A4-N1 | M | **A corrupt junction sidecar now gives HTTP 500 and CLI tracebacks:** `ComposeError` is not caught (regression since 2.9.4) | `transport/compose.py:40`; `web/blueprints/transport.py:308,348`; `jobset/_cli.py:1565,1574` | verified (exception type) |
| A8-02 | M | **A stale filter answer overwrites a newer selection** | `lib/molview/stores.js:278-291` | verified (node) |
| A7-10 / A7-33 | M | **HTML 404/405/500 pages instead of the JSON envelope;** a non-object JSON body gives 500 | `web/app.py`; `files.py:1464` | verified |
| A1-09 | M | **`make web` / `make web-bg` call a command that does not exist** | `Makefile:43,52` | verified |
| A5-01 | M | **Partly fixed:** a refused benchmark shelf, and answering "no", still exit 0 | `submit.py:1727`; `jobset/_cli.py:2140` | read |

### 2.5 Review gates that have gone red unnoticed

These matter because they are the mechanisms meant to catch the rest.

| ID | Finding | Location | Check |
|---|---|---|---|
| A1-33 | **The pyflakes pre-commit hook cannot install** (rev 3.2.0 has no `.pre-commit-hooks.yaml`), so `make precommit` aborts and no hook runs | `.pre-commit-config.yaml` | verified (HTTP 404) |
| A1-N7 | **The package is not pyflakes-clean,** contrary to `conventions.md:34`: undefined name `Environment` | `jobset/submit.py:202` | verified |
| A1-N3 | **`tools/classify_path_finders.py --check` exits 1 on main.** It is the review tool that replaced a retired test, and `plan.md` records it as clean | `joblist.py:197` | verified |
| A1-N1, A1-N2 | **New layering breaks in the last commits:** `task` (L1) → `template` (L2) in the HEAD commit; two new upward floor imports | `task.py:684`; `jobset/placement.py:138`; `jobset/transport_rung.py:295` | read (AST graph) |
| A1-13 / A8-N2 / A8-09 | **41 test files and functions cited in docs do not exist,** among them three that are named as the guard of a duplicated rule | e.g. `web/molview.md:4322,3500`; `web/form-schema.md:291` | verified (sweep) |
| **B3-N1** | **New in 2.11.0: tier-1 results depend on scheduling.** `0ff77d99` lets files without shared fixtures run on any worker, and `tests/test_describe.py:146,154,330` pops the session's `MOLBUILDER_PROJECTS` instead of restoring it. A10-N1 now fails at `-n 6` and passes at `-n 4`, and `testing.md` § 6.1a says "where it runs changes nothing" | HEAD `tests/conftest.py:925-956,1000-1016` | verified (two runs) |
| B3-N3 | **The 2.11.0 docs still certify red gates:** `plan.md:364` calls `classify_path_finders --check` "clean"; it exits 1 at HEAD. The pyflakes hook still cannot install, and `submit.py:203` is still flagged | `docs/plans/plan.md:364` | verified |

---

## 3. What was fixed since 2.9.4

| ID | Finding fixed | Commit |
|---|---|---|
| A2-01 | A sidecar is now refused when its geometry changed (`geometry_hash`, schema v13+) | badc9178 |
| A2-03 | The PySCF `_optimized.xyz` is read with its sidecar | badc9178 |
| A3-11 | `stages.md` §1.3 group counts | cd7deb94 |
| A5-08 | A warm retry's output no longer lands in the previous run's log | 5c63bf95 |
| A7-12 | The path-fence table names real helpers; two fence holes closed (relative `psml_lib` with `..`, Prep's `from`) | 62fb65b0 |
| A8-01 | The read-only trajectory viewer rebuilds again (`enforce: true`, failure shown) | 8597632f |
| A8-23 | The drag-select listener is removed on dispose | ee68556e |
| A9-03 | VibrationView no longer accumulates WebGL contexts (window pool) | 691e7f0a |
| A9-11 | `presenters.md` §7 module table | a10c49db |
| A1-31 | `composed_junction` defined once | 74916c45 |
| A10-11 | Source-read classifier retired | 769f8f4c |
| A10-16, A10-17 | Plan Q13 and §5aa archived | — |
| A6-05 | Jupyter always binds 127.0.0.1 (*2.11.0*) | ebaaebb4 |
| A6-12 | No http/https split between server and notebook under `--no-auth` + TLS (*2.11.0*) | ebaaebb4 |
| A7-30 | The per-port wildcard `frame-src` CSP is gone (*2.11.0*) | ebaaebb4 |

**Partly fixed (24 at 2.11.0; 20 in the first round):**

- A2-04/09/27
- A3-24
- **A5-01** (exit 1 now for a failed run and a single refused submission, 9fc066b9)
- A5-03, A5-06, A5-18
- **A5-23** (job-id parsing now uses the scheduler kind's own pattern, b74276bd)
- A8-14
- A9-20/21/22
- A10-08/09/10/13/19
- A1-30
- *2.11.0, `ebaaebb4`:* A6-01, A7-01, A7-02, A7-03 — the notebook gate checks Origin and a loopback Host; nothing else does

**Also reviewed sound in 2.11.0:**
- the fit/place rewrite (`scheduler/fit.py`, 37 new tests pass);
- `ensure_runtime_dir`;
- the restart-stops-notebook loop;
- the pass-through's header handling (session cookie stripped, token added, `X-Forwarded-For` and `Access-Control-*` dropped, absolute targets refused).

The details are in § 6.

---

## 4. Corrections to the 2026-10-14 review

| ID | Original claim | Verdict |
|---|---|---|
| A4-02 | The Raman chart is "not labelled as activity" | **Not a defect.** It is labelled "Raman activity, Å⁴/amu" (`lib/spectra/core.js:3176`), a label older than the baseline. Plotting activity without (ν₀−ν)⁴ and Bose factors is a choice the label states |
| A4-28 | Doubling of TBtrans's printed current is suspect | **Not a defect.** sisl computes (e/h)∫T for one spin channel (`io/tbtrans/tbt.py:881`), which matches `record.py:254` |
| A7-32 | Modify spacings accept NaN/inf/negative | **Not a defect.** `cell.py:1335` refuses them (re-run: 400) |
| A6-15 | A `_comment` key in `tls` is refused | **Narrowed:** a string comment is accepted; a list or object comment is refused |
| A6-24 | The shim's help omits `--python` | **Narrowed:** `--python` is documented (`install-env.sh:415`); only the micromamba omission is real |
| A5-18 | "31 rules" was a miscount | **Not a defect:** 31 rule rows counted |
| A10-04 | The peptide preflight failure is an rdkit-version issue | **Re-diagnosed:** it is product defect A10-N2 |

---

## 5. New findings (67)

<details open><summary><b>A1 — spine and architecture (11)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A1-N1 | M | HEAD commit adds an L1→L2 import: `Task._rung_reads` imports `template`, against `task.py`'s own header | `task.py:684,653,26-30` |
| A1-N2 | M | Two new upward floor imports | `jobset/placement.py:138` → `.ask`; `jobset/transport_rung.py:295` → `.ready` |
| A1-N3 | M | Review tool `classify_path_finders --check` fails on main; plan records it as clean | `tools/classify_path_finders.py`; `joblist.py:197`; `plan.md:364` |
| A1-N4 | M | Rewritten pair writer brings back the fixed `<name>.tmp` temp-name trap, inside jobs too | `workingcopy_structure.py:566`; `sidecars/molstruct.py:652` |
| A1-N5 | M | `conventions.md` command catalogue: "19 top-level commands" (20), `jobs` group missing, `jobset` lists 5 of 8 verbs | `conventions.md:157,162-170,204-212` |
| A1-N6 | L | New modules absent from the spine maps (`joblist`, `jobs_cli`, `jobset/shelves`, `scheduler/kinds`, `scheduler/queue`, `siesta/engine_input`) | `architecture.md` L1 index; `code-audit.md:267-269` |
| A1-N7 | L | pyflakes: undefined name `Environment` | `jobset/submit.py:202` |
| A1-N8 | L | `code-audit.md` contradicts itself on artifact lints vs outcome tests | `code-audit.md:6,46-66` vs `:230`, § 6 |
| A1-N9 | L | Job address keyed by bare scheduler id (same defect as A5-N1) | `joblist.py:136` |
| A1-N10 | L | Makefile header says `tools/testrun.py`; the target calls pytest | `Makefile:4-5,28-31` |
| A1-N11 | L | Identical table-format closures | `jobset/runstatus.py:752`; `jobset/plan.py:92` |

</details>

<details open><summary><b>A2 — data model (9)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A2-N1 | M | A typed title read as extxyz metadata makes the saved pair unopenable (verified) | `structure.py:2219-2229`, `from_xyz` |
| A2-N2 | M | Docs say the v7–v11 offset migration works; it never runs | `structure-periodicity.md:43`; `structure-molstruct.md:170`; `cell.py:634` |
| A2-N3 | M | Save door's overwrite gate ignores the sidecar and the suffix | `web/blueprints/build.py:508-511`; `workingcopy_structure.py:495-498` |
| A2-N4 | L | `_write_pair` not both-or-neither at the replace step (verified) | `workingcopy_structure.py:577-578` |
| A2-N5 | L | Long PDB title word broken on wrap; doc says "never cut" | `structure.py:604`; `structure.md:372` |
| A2-N6 | L | A newline in a PDB title breaks the TITLE record (verified) | `structure.py:601-602` |
| A2-N7 | L | `resolve_element("x")` suggests `X`, then refuses it | `chemistry.py:299-301` |
| A2-N8 | L | Two XYZ frame parsers with different failure modes (partly unverified) | `sidecars/molstruct.py:289-306` vs ASE |
| A2-N9 | L | `FdfParams.atom_z_span_ang` has no production consumer | `parse/fdf.py:390-404` |

</details>

<details open><summary><b>A3 — engines (5)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A3-N1 | M | Root cause of A3-03/04: the coverage check and the copier answer "which file SIESTA opens" differently, though 9e393a69 claims one rule | `pseudos.py:703-744` vs `:74-110`; `siesta/input.py:222-240` |
| A3-N2 | L | `architecture.md` still lists the deleted `find_psml` | `architecture.md:257` |
| A3-N3 | L | `siesta.md`'s MPS rule omits `--no-mps` / `MOLBUILDER_USE_MPS` and the visible-GPU condition | `siesta.md:568-571` vs `running-a-job.md:420-426` |
| A3-N4 | L | Deck explains BlockSize/ParallelOverK twice, with different remedies | `siesta/layout.py:126-148` vs catalogue `:872,900-916` |
| A3-N5 | L | "Same engine input" comparison folds case on every word, `SystemLabel` included (verified) | `siesta/engine_input.py:69-86`; `jobset/gather.py:184` |

</details>

<details open><summary><b>A4 — transport and vibrations (5)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A4-N1 | M | `ComposeError` uncaught: HTTP 500 and CLI traceback on a corrupt junction sidecar (regression) | `transport/compose.py:40,979-984`; `web/blueprints/transport.py:308,348`; `jobset/_cli.py:1565,1574` |
| A4-N2 | L | `selection_pdos` AttributeError when no composed junction exists (unverified reachability) | `transport/record.py:1055,1067,1074` |
| A4-N3 | L | `point_currents` can mix k-points, loop time and current from an older run (unverified reachability) | `transport/record.py:228-246` |
| A4-N4 | L | `vibration.md` describes unbuilt schema 9 in the present tense (code is schema 8) | `vibration.md:345`; `spectra/results.py:79` |
| A4-N5 | L | `compose.py` has its own transport-axis constant beside the imported one | `compose.py:663,683` vs `:408` |

</details>

<details open><summary><b>A5 — execution (6)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A5-N1 | M | Job address files collide across clusters (verified) | `joblist.py:160,196` |
| A5-N2 | L–M | A job whose cluster was unknown is checked against the local cluster and forgotten as ended (unverified) | `jobs_cli.py:77,152-154`; `scheduler/queue.py:49` |
| A5-N3 | M | `launch --time` is passed through the human parser unchecked: `30:00` → 30 h | `jobset/_cli.py:1979`; `submit.py:328-337` |
| A5-N4 | L | Run script and monitor read only `MB_*`, so a `--mode direct` run inside `salloc`/`srun` is not seen as in a job; undocumented | `monitor.py:609,719`; `kinds.Kind.job_lines` |
| A5-N5 | L | A benchmark shelf's wall time is one trial's, while groups' are summed (intent unverified) | `jobset/shelves.py:122-127` vs `group.py:221-223` |
| A5-N6 | L | `scontrol show config` (60 s timeout) runs once per submission | `submit.py:1849-1851` |

</details>

<details open><summary><b>A6 — configuration and ops (4)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A6-N1 | M–H | The stated CSRF defence does not exist (verified; see § 2.1) | `web/app.py:231-232`; `web-api.md:442` |
| A6-N2 | M | Behind a same-host proxy with no sign-in, the notebook gate admits every proxied user | `jupyter.py:111-118`; `deployment.md:296-303` |
| A6-N3 | L–M | Two "is this pid ours" implementations: the new one (boot id + start ticks) is sound, `serve`'s substring match is not | `joblist.py:83-112` vs `serve_daemon.py:200-226` |
| A6-N4 | L | `access-control.md` says the admin list gates two subsystems; Jupyter is a third | `access-control.md:310` |

</details>

<details open><summary><b>A7 — web backend (6)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A7-N1 | H | File API writes into a calculation's `.git/` (verified write) | `files.py`; `checkpoint.py:77,345` |
| A7-N2 | M | Transport routes resolve relative paths against two bases; the doc says tree-relative | `web/blueprints/transport.py:68` vs `:467` |
| A7-N3 | M | `web-api.md` says a test pins the route count; the pin was deleted (703bf43c), and the count description is wrong | `web-api.md` § 3 |
| A7-N4 | L | The URL shown for a new notify key is built from the request's Host header | `notify_setup.py:554` |
| A7-N5 | L | The upload branch of `/api/watch/load` lacks the ValueError catch the path branch has (unverified) | `watch.py:605` vs `:490-496` |
| A7-N6 | L | `err()` is documented as the one error builder but used only in `modify.py` (276 inline refusals) | `_shared.py:477`; `web-api.md:27-30` |

</details>

<details open><summary><b>A8 — front end 1 (7)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A8-N1 | M | Atom-order line deep-copies the trajectory on every update (1.4 s measured) | `lib/molview/ui.js:1362-1370,2995-3000` |
| A8-N2 | L | Docs name guard tests that do not exist | `molview.md:3500`; `form-schema.md:291` |
| A8-N3 | L | Two independent 3Dmol window pools, already diverging | `lib/molview/3dmol-embed.js:200-246`; `lib/vibrationview/_seal.js:117-156` |
| A8-N4 | L | `parser.js` says enforcing is "here and nowhere else"; two other hosts enforce | `lib/projects/parser.js:88-93` |
| A8-N5 | L | Two signatures for `openMolecule` in `projects.md` | `projects.md:138` vs `:401` |
| A8-N6 | L | `SHAPED_METADATA_KEYS` is read by one translator only; the other hand-writes the keys | `model-jobs.js:211` vs `:477-485`; `molview.md` § 6.2 |
| A8-N7 | L | `mount` can reject after `createEmbed`, against "MOUNT ALWAYS RESOLVES" (unverified) | `lib/molview/mount.js:53,160-404` |

</details>

<details open><summary><b>A9 — front end 2 (3)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A9-N1 | L | Docs and comments still describe "Refresh" after its listeners were deleted (91f26e3b) | `spectra.md:548`; `trajectory.md:251,298`; `file-picker.js:309,430,584,634,1012-1016`; `trajectory/core.js:176,191,254-265` |
| A9-N2 | L | A disposed spectra tab still finishes and downloads an animation export, or throws a raw TypeError | `lib/spectra/core.js:2863-2890,3603-3634`; `vibrationview/index.js:322` |
| A9-N3 | L | Dead `const sign = 1` multiplied into every curve | `lib/spectrumchart/_seal.js:316,324,341` |

</details>

<details open><summary><b>A10 — tests, plan, hygiene (11)</b></summary>

| ID | Sev. | Finding | Location |
|---|---|---|---|
| A10-N1 | M | Two seeding tests fail whenever `$MOLBUILDER_PROJECTS` is set | `tests/test_seeding_a_fresh_machine.py:378,391` |
| A10-N2 | M | Product defect: peptide build drops real hydrogens without openbabel (verified) | `chemistry.py:1185-1211` |
| A10-N3 | M | The no-playwright skip is keyed on the file name, not the `e2e` marker; three tier-1 files drive Playwright | `tests/conftest.py:983-986`; `test_form_schema_setvalues_js.py:32`, `test_form_schema_diff_js.py:31`, `test_pages_no_js_errors.py:33` |
| A10-N4 | M | 3Dmol window pool implemented twice (same as A8-N3) | see A8-N3 |
| A10-N5 | L | `testing.md` § 2a's "by any spelling" overstates the projects-tree guard (relative `open` and subprocesses not watched) | `tests/conftest.py:786-787` |
| A10-N6 | L | Residual source pin against § 3a | `tests/test_web_files.py:771-777` |
| A10-N7 | L | `plan.md` cites Q13 as open, but it is archived and missing from the resolvable-ids list | `plan.md:31,298,334` |
| A10-N8 | L | The TST row is a single 26 KB table cell; `plan.md` grew from 253 to 282 KB | `plan.md:364` |
| A10-N9 | L | Dead tools: `capture_screenshots.py` (drives 404 pages), `css_rename.py`, `css_control_audit.py` | `tools/` |
| A10-N10 | L | `testing.md` § 4's "measured 2026-09-29" numbers are stale | `testing.md:617-626` |
| A10-N11 | L | Tier-1 requirements (setuptools, authlib) unstated; missing tools ERROR instead of skip | `testing.md` tier 1 |

</details>

---

## 5b. New findings in the 2.11.0 commits (15)

Found by reviewing `393c70a8`, `ebaaebb4` and `0ff77d99` in full.

| ID | Sev. | Finding | Location (HEAD) | Check |
|---|---|---|---|---|
| B1-N1 | M (H on a shared host with sign-in) | `/jupyter/` relays whatever listens on 127.0.0.1:serve+1, without checking it is our notebook, onto molbuilder's own origin. It relays upstream `Set-Cookie` (session fixation), and with no upstream CSP sends only `frame-ancestors`, so the page's scripts run. `jupyter.md:509` says it answers 502 "when no notebook is running", but it does so only when nothing listens. Fix: Jupyter on a Unix socket in the 0700 runtime directory, or forward only while the shepherd pidfile is ours; drop molbuilder's cookie name from upstream `Set-Cookie`; always add `script-src` | `web/blueprints/jupyter.py:357-381,437-444,473-476` | verified (stand-in upstream); signed-in leg by reading |
| B1-N2 | L–M | `deployment.md` § 2's proxy example does not say the proxy must keep `Host` (or send `X-Forwarded-Host` with `trust_proxy`) and forward WebSocket upgrades. With nginx defaults, kernels never connect and a signed-in admin's Lab POSTs get 403 | `docs/ops/deployment.md` § 2; `jupyter.md` § 2 | unverified (no nginx) |
| B1-N3 | L | The gate checks only Origin, and Jupyter's own Referer check is skipped for token-authenticated requests. A cross-site page can load `/jupyter/files/<path>` as an image or script to learn whether a file exists and its size | `web/blueprints/jupyter.py` gate; `jupyter.md:513-519` | read (jupyter_server source) |
| B1-N4 | L | Two new claims are false: "molbuilder never hands it [the token] to the browser" (JupyterLab's page carries it, and token requests set Jupyter's login cookie), and "nothing but the pass-through reaches it" (any local process reaches 127.0.0.1:serve+1) | `web/blueprints/jupyter.py:181`; `jupyter.py:339`; `cli.py:2725`; `access-control.md:132`; `data/jupyter.toml:44-58` | read |
| B1-N5 | L | molbuilder's 50 MB `MAX_CONTENT_LENGTH` caps notebook saves through the pass-through (Jupyter allows 512 MB), with an unrelated error message; bodies are buffered in memory | pass-through; `app.py` | verified (51 MB PUT → 413) |
| B2-N1 | L | `fit` reads an unstated GPU or node count as zero, giving a false "fits no node" (contrary to `scheduler.md` R2a). It can leave out every GPU benchmark cell; reachable with hand-edited or older records | `scheduler/fit.py:181-187,279-282` | verified |
| B2-N2 | L | The fit message truncates the asked wall: 4 h 00 min 30 s on a 4-hour queue reads "4h is over htc's wall (0-04:00:00)"; 59 s reads "0m" | `scheduler/fit.py:84-85,241` | verified |
| B2-N3 | L | The package re-exports the function `fit` under the submodule's name, so `molbuilder.scheduler.fit` is a function, not the module | `scheduler/__init__.py:53-55` | verified |
| B2-N4 | L | An unread `devices` field renders an unreadable count as "a100 xNone" | `scheduler/record.py:1129-1130` | verified |
| B2-N5 | L | `gpu.md` contradicts the code (and `generator.md`) on which GPU count bounds the benchmark proposal. Two rules exist for "GPUs on a node": the per-type maximum vs the sum over cards | `docs/execution/gpu.md:104,111`; `jobset/prep_inputs.py:126-138`; `fit._gpus_on` | read |
| B2-N6 | L | Comments cite `max_cpus`, which this commit deleted | `scheduler/probe.py:73-74,159` | verified |
| B2-N7 | L | A comment says the tab fills time and memory asks from the queue's limits; `task-setup.md` now says it "fills neither" | `scheduler/record.py:1103-1108` | read |
| B3-N1 | L–M | Tier-1 results depend on scheduling: new worker grouping plus `test_describe.py`'s `os.environ.pop` of `MOLBUILDER_PROJECTS` | `tests/conftest.py:925-956,1000-1016`; `tests/test_describe.py:146,154,330`; `testing.md` § 6.1a | verified |
| B3-N2 | L | Worker grouping depends on pytest-private `item._fixtureinfo.name2fixturedefs`; if it disappears, grouping silently stops and module fixtures are rebuilt per worker | `tests/conftest.py:930-936` | unverified risk |
| B3-N3 | L | `plan.md:364` still certifies the red `classify_path_finders` gate as clean | `docs/plans/plan.md:364` | verified |

---

## 6. Status of every 2026-10-14 finding on main

`ID` numbers follow the order of the 2026-10-14 review's appendix (`A<area>-<nn>`).

**Status codes:**

| Code | Meaning |
|---|---|
| S | still present |
| F | fixed |
| P | partly fixed |
| M | moved/changed |
| ND | not a defect |
| CV | cannot verify |

| ID | Finding (short) | Status | Where on main |
|---|---|---|---|
| A1-01 | Design principle "metadata on the dataclass" false | S | `design.md:181-187`; `conventions.md:282` |
| A1-02 | L1 `trajectory_log.emitter` imports L2 (26 modules) | S | `trajectory_log/emitter.py:41` |
| A1-03 | L1 modules import L2 in functions | M (worse) | `electronic_state.py:227`; `runfiles.py:1910`; `task.py:1323,684`; `warmfiles.py:173` |
| A1-04 | Layer tables disagree | S | `backend-architecture.md:54-55,220` vs `architecture.md:246-250` |
| A1-05 | Scheduler L1 rationale false | S (worse) | `design.md:117-121`; `scheduler/queue.py:70,85,99` |
| A1-06 | Upward floor imports | S (29) | e.g. `jobset/errors.py:33`, `plan.py:24`, `migrate.py:262-494` |
| A1-07 | runwrap floor 5 vs 3 | S | `runwrap.py:39` vs `:3829` |
| A1-08 | Three counts of jobset floors | S | `docs/README.md:34`; `execution/architecture.md:151`; `architecture.md:302` |
| A1-09 | `make web` broken | S | `Makefile:43,52` |
| A1-10 | Makefile ignore/help | S | `Makefile:30` |
| A1-11 | Root README stale | S | `README.md:143,170` |
| A1-12 | conventions.md command lists | S (worse) | `conventions.md:157,204,261,317` |
| A1-13 | Missing test files cited | S (41) | e.g. `conventions.md:296`; `package-layout.md:76` |
| A1-14 | Index not "the ONE index" | S | 5 docs unindexed; `docs/README.md:76` |
| A1-15 | GET writes `toc.json` | S | `web/blueprints/docs.py:355-361` |
| A1-16 | Broken plans/ links | S | `ops/installation.md:142`; `ops/env-framework.md:147,633` |
| A1-17 | backend-architecture symbols | S | `backend-architecture.md:80,81,186,213,309` |
| A1-18 | `sidecars.{to_dict,…}` | S | `architecture.md:239` |
| A1-19 | "only via `from molbuilder import`" | S | `design.md:94-95` |
| A1-20 | package-layout behind | S (worse) | 34 modules unmentioned |
| A1-21 | Hand-rolled atomic writes | M | 5 sites; see A1-N4 |
| A1-22 | Timestamp helpers | S (+1) | `joblist.py:128` new |
| A1-23 | runwrap god module | S (grew) | `render_run_wrapper` 1,842 lines |
| A1-24 | cli.py domain logic | S (grew) | 3,575 lines |
| A1-25 | Lazy imports | S | 1,720 in functions vs 561 at module level |
| A1-26 | Contract restated in many places | S | 6 docs |
| A1-27 | Spine docs mostly history | S | `docs/README.md` 102,723 bytes |
| A1-28 | `scripts/` not inventoried | S | `scripts/bench-siesta-blocksize.sh` |
| A1-29 | Dead `tools/capture_screenshots.py` | S | — |
| A1-30 | Small drift | P | 3 items remain |
| A1-31 | `composed_junction` twice | F | 74916c45 |
| A1-32 | Prep-import rule understated | S | `submit.py:791,1449,2642` |
| A1-33 | pyflakes hook | S (raised to M) | `.pre-commit-config.yaml` |
| A2-01 | Sidecar hash never compared | F | badc9178 |
| A2-02 | Migration never fires | S | `parse/sidecars/molstruct.py:112` |
| A2-03 | pyscf-geom without sidecar | F | badc9178 |
| A2-04 | One sidecar for two geometries | P | now loud (`GeometryChangedError`) |
| A2-05 | `write(fmt)` vs suffix | S | `workingcopy_structure.py:513` |
| A2-06 | `frozen_atoms` setter | S | `structure.py:1570` |
| A2-07 | `vacuum` NaN/inf/negative | S | `structure.py:888-891` |
| A2-08 | Half-applied sidecar | S | `sidecars/molstruct.py:764-784` |
| A2-09 | Sidecar identity overrides PDB | P | hand edits now refused |
| A2-10 | PDB drops atoms | S | `structure.py:2082-2088` |
| A2-11 | HG21 → Hg | S | `structure.py:2144-2147` |
| A2-12 | Second PDB reader | S | `builders/backends/_common.py:57` |
| A2-13 | Bond detection ×3 | S | `chemistry.py:870-875,1452`; `_amber.py:223` |
| A2-14 | Chemistry helpers drop metadata | S | `chemistry.py:1031-1039,1266-1278` |
| A2-15 | `.XV` metadata rule | M | flag vs sniff (`cli.py:1098,1110`) |
| A2-16 | `<target>.tmp` writers | S | see A1-N4 |
| A2-17 | Two element tables | S | `chemistry.py:89` |
| A2-18 | Label/channel name merge crash | S | `structure.py:1403-1406` |
| A2-19 | `ByElement` literal | S | `selection.py:228-231` |
| A2-20 | `parse/` scope claim | S | `parse.md:88` |
| A2-21 | Structure identities | M | five identities now |
| A2-22 | Two `ParseWarning` | S | `frame.py:35`; `parse/types.py:33` |
| A2-23 | Remap ×5 | S | `modify.py:63-85` |
| A2-24 | `to_wire` swallows | S | `structure.py:1276,1292` |
| A2-25 | Channel keys/aliasing | S | `structure.py:~477-483` |
| A2-26 | "pairwise-disjoint" docstring | S | `structure.py:38` |
| A2-27 | fdf z-span units | P | fallback `parse/fdf.py:397-401` |
| A2-28 | XYZ title newline | S (worse) | `structure.py:2219` |
| A2-29 | Drifted citations | S | `structure.md:372,382` |
| A2-30 | "ten sites" list | S | `structure.md:380-382` |
| A2-31 | Schema/API drift | S | `structure-annotations.md:370`; `parse.md:462` |
| A2-32 | parse.md structure claims | S | `parse.md:28,169,524` |
| A2-33 | Docstrings: JS files, `measure_fcc` | S | `selection.py:18-20`; `cell.py:894` |
| A2-34 | `to_pdb` own `+1` | S | `structure.py:2253` |
| A2-35 | Axis-kind/vacuum parsing copies | S | `periodicity_gate.py:271,309-319,390-422` |
| A3-01 | WARN masks wrong-XC ERROR | S | `pseudos.py:807-826` |
| A3-02 | vdW-DF blocked | S | `pseudos.py:656` |
| A3-03 | Au "misnamed" | S | `pseudos.py:603,613,735` |
| A3-04 | Au1 "missing" | S | `pseudos.py:703,711` |
| A3-05 | No XC pairing check | S | `config/siesta.py:263-280` |
| A3-06 | Dispersion double counting | S | `pyscf/layout.py:203-206` (plan K18) |
| A3-07 | symmetry + geomeTRIC | S | `pyscf/input.py:472` |
| A3-08 | Impossible values warn only | S | `validation/metadata.py:97` |
| A3-09 | Geometry blocks ×2 | S | `siesta/input.py:1126-1148`; `transiesta.py:393-410` |
| A3-10 | tuning.md eigensolver default | S | `tuning.md:973-974` |
| A3-11 | stages.md group counts | F | cd7deb94 |
| A3-12 | template.md anchor derived | S | `template.md:1127` vs `:1168` |
| A3-13 | `_GEOM_KWARG` literal | S | `pyscf/layout.py:63-73` |
| A3-14 | overview.md §5 retired names | S | `engines/overview.md:358,366` |
| A3-15 | `kind` disagreements unguarded | S | `config/siesta.py:487,503` |
| A3-16 | SO + SR pseudos warn only | S | `pseudos.py:644` |
| A3-17 | Status counts in docs | S | `pseudopotentials.md:52,101` |
| A3-18 | siesta.md k=1 "never checked" | S | `siesta.md:497` |
| A3-19 | siesta.md block table | S | `siesta.md:240,250` |
| A3-20 | Ladder builders ×2 | S | `siesta/stages.py:76,90` |
| A3-21 | `gto.M` ×2 | S | `pyscf/input.py:461`; `vibration_emitters.py:424` |
| A3-22 | PSML parsed repeatedly | S (changed shape) | `pseudos.py:608`; `validation/siesta.py:178,260,322` |
| A3-23 | Makov–Payne ×2 | S | `makov_payne.py:129,359` |
| A3-24 | tuning.md stale | P | `PAO.Basis` at `:284,847` |
| A3-25 | "tight" 0.01 vs 0.001 | S | catalogue `:539` |
| A3-26 | Level shift wording | S | `pyscf.md:750-751` |
| A3-27 | SaveHS default | CV | no SIESTA manual offline |
| A3-28 | LDA authors mismatch | S | `pseudos.py:297-300` |
| A3-29 | Dispersion template authors | S | `siesta/input.py:1628` |
| A3-30 | science/overview.md stale | S | `:232,297,309,311` |
| A3-31 | `ecp_atoms` default | S | catalogue `:1392` |
| A3-32 | GPU BlockSize cap | S | `siesta/input.py:166` |
| A3-33 | Default relaxation CG 0.02 | S | `config/siesta.py:437,462` |
| A3-34 | overview.md two GPU items | S | `engines/overview.md:182,190` |
| A4-01 | I(V) sign | S | `transport/record.py:310` |
| A4-02 | Raman "not labelled" | ND | `core.js:3176` |
| A4-03 | c ∥ z unchecked | S | `transiesta.py:169` etc. |
| A4-04 | kz 40 vs 100 | S | `transport.md:1262,1271` |
| A4-05 | §2a.6 vs §0.3 | S | `transport.md:1168-1172` vs `:340` |
| A4-06 | Bias Class C diagram | S | `transport.md:1124,1129` |
| A4-07 | `TRANSPORT_AXIS` ×3 | S | `kmesh.py:36`; `sort.py:73`; `compose.py:663` |
| A4-08 | Two lower-lead decisions | S | `sort.py:106-108`; `transiesta.py:163-172` |
| A4-09 | Items declared twice | S (grew) | `tbt_progress`, `h_tolerance` added |
| A4-10 | Thermal spread Py + JS | S | `derived.py:68-80`; `core.js:2536-2543` |
| A4-11 | normal-modes.md `harmonic_analysis` | S | `normal-modes.md:93-94` |
| A4-12 | Rotation origin wording | S | `normal-modes.md:326` |
| A4-13 | `elec-pos` always written | S | `transport.md:3766-3768` |
| A4-14 | Bias sign wording | S | `transport.md:3869` |
| A4-15 | Caveat "still landing" | S | `transport.md:984` |
| A4-16 | Code cites missing sections | S | `deck.py:468,641`; `wizard.py:9,427` |
| A4-17 | 12 Å note | S | `wizard.py:424-430` |
| A4-18 | "Wizard deleted" | S | `transport.md:3894-3895` |
| A4-19 | Dead cell-less fallbacks | S | `wizard.py:389-391`; `transiesta.py:336-337` |
| A4-20 | §7 baseline conflicts | S | `transport.md:4741-4744` |
| A4-21 | Frame generator present tense | S | `transport.md:1422` |
| A4-22 | Held-atom IR caveat | S | `vibration.md:868` |
| A4-23 | Coupling per display vector | S (deliberate) | `vibration_emitters.py:1315-1321` |
| A4-24 | Two unit conventions | S (documented) | `constants.py:101-106` |
| A4-25 | "isotropic" label | S | `vibration_emitters.py:1209,1234` |
| A4-26 | transport.md history | S (grew to 4,969 lines) | — |
| A4-27 | PySCF Hessian shapes | CV | PySCF not installed |
| A4-28 | TBtrans current doubling | ND | sisl `tbt.py:881` |
| A5-01 | Launch exit 0 | P | `jobset/_cli.py:2238` (fixed 9fc066b9); shelves and "no" remain |
| A5-02 | Retry record collision | S | `runrecord.py:377` |
| A5-03 | `jobset prep run` in docs | P | `worked-example.md:488` |
| A5-04 | job-set.json example | S | `job-system.md:1237-1254` |
| A5-05 | Hierarchical tree names | S | `execution/overview.md:247-254` |
| A5-06 | §2.6 names, by-hand line | P | `job-contracts.md:1078` |
| A5-07 | `30:00` → 30 h | S | `scheduler/quantities.py:144` |
| A5-08 | Retry log leakage | F | 5c63bf95 |
| A5-09 | Restore atomicity | S | `checkpoint.py:1717-1804` |
| A5-10 | migrate non-atomic | S | `migrate.py:631-637` |
| A5-11 | Web attempt count | S | `web/blueprints/build.py:2085` |
| A5-12 | God functions | S (grew) | — |
| A5-13 | PySCF arg loop | S | `runwrap.py:2855` |
| A5-14 | Floor-1 upward imports | S | `calcdirs.py:125` etc. |
| A5-15 | `ask.canonical_*` | S | `job-contracts.md:2392-2393` |
| A5-16 | job-set.json docstring | S | `model.py:575` |
| A5-17 | "stdout teed" | S | `running-a-job.md:118` |
| A5-18 | Doc contradictions | P | `job-system.md:42` vs `:1741`; `running-a-job.md:15`; `architecture.md:946` |
| A5-19 | `_DONE`; "no ladder" | S | `runstatus.py:58`; `pipeline_log.py:126` |
| A5-20 | Writes skip persist | S | `summarize.py:338`; `script_emit.py:1988` |
| A5-21 | Walk quoting; exit codes | S | `submit.py:2032,2059` |
| A5-22 | Clocks | S | `runwrap.py:223,3100,3417,3447` |
| A5-23 | Parsing | P | job id fixed (b74276bd); `quantities.py:70-82` remains. *2.11.0:* `admit.py` deleted; the minutes message moved to `fit.py:241` (see B2-N2) |
| A5-24 | Restore CLI; save message | S / CV | `cli.py:3282,3304` |
| A6-01…30 | Configuration/ops findings | S, except at 2.11.0: A6-05 F, A6-12 F, A6-01 P | Files mostly unchanged. A6-15 and A6-24 narrowed (§ 4). A6-23 moved to `configuration.md:1107` |
| A7-01…33 | Web backend findings | S, except A7-12 F, A7-32 ND; at 2.11.0 A7-30 F, A7-01/02/03 P | See § 2 and the A7 notes below. At HEAD, A7-N3's route count is 102 (the pass-through added 3 rules), not 99 |
| A8-01 | Trajectory rebuild | F | 8597632f |
| A8-02…22, 24…29, 31…33 | MolView/Task Setup findings | S | files largely unchanged since baseline |
| A8-14 | Ruler-on logic | P | `molview.md:4262` stale |
| A8-23 | Drag-select leak | F | ee68556e |
| A8-30 | Second 3Dmol seal | M | see A8-N3 |
| A9-03 | VibrationView WebGL | F | 691e7f0a |
| A9-11 | presenters.md table | F | a10c49db |
| A9-20/21/22 | spectra reply, classic-script label, registry signatures | P | `spectra.md:656`; `registry.js:18-60` |
| A9-others | Results/spectra findings | S | `spectrumchart` unchanged since baseline |
| A10-11, 16, 17 | classifier, Q13, §5aa | F | 769f8f4c; archived |
| A10-08/09/10/13/19 | test docs, node helpers, pins, counts, plan status | P | see § 5 A10 |
| A10-23 | Inspector lifecycle duplication | CV | not re-traced |
| A10-others | Tests and hygiene findings | S | Makefile, `.gitignore` byte-identical to baseline |

**Selected A7 locations on main:**

| ID | Location | Note |
|---|---|---|
| A7-06 | `files.py:176,2336` | |
| A7-07 | `files.py:1614,1631` | |
| A7-10 | `app.py` | handlers only for 413, PeriodicityRefused, DeckRecordUnreadable |
| A7-11 | `projects.py:290` vs `transport.py:68` | |
| A7-13 | `web-api.md:716` vs `app.py:672` | |
| A7-14 | `web-api.md:27` | |
| A7-16 | `web-api.md:383` | |
| A7-17 | `watch.py:308`; `build.py:325` | |
| A7-18 | `files.py:737` | |
| A7-20 | `build.py:1183` | |
| A7-22 | `files.py:1813,2041` | |
| A7-23 | `files.py:1581` | |
| A7-26 | `web-api.md:761` | |
| A7-33 | `files.py:1464` | |

---

## 7. Cross-cutting observations

1. **Review-only gates are failing quietly.** The project moved several invariants from tests to "review" (`code-audit.md` § 1c), and on main three of those mechanisms are red without anyone noticing:
   - the path-finder classifier (A1-N3);
   - pyflakes, which is not clean (A1-N7);
   - the pre-commit hook, which cannot run (A1-33).

   Separately, 41 cited guard tests do not exist. A review step that is never executed is weaker than the test it replaced.
2. **"Two implementations of one rule" keeps coming back in new code:**
   - pseudopotential coverage vs the copier (A3-N1);
   - two "is this pid ours" checks (A6-N3);
   - two 3Dmol window pools (A8-N3 / A10-N4);
   - two "same engine input" comparers (A3-N5);
   - a second `TRANSPORT_AXIS` inside one module (A4-N5).

   Each pair already disagrees somewhere. The project's own D4 rule asks for a parity test when two implementations must agree.
3. **Security moved only around the notebook.** At `f3d5a4e4`, all eight priority security items sat on unchanged files. In 2.11.0, `ebaaebb4` added an Origin and loopback-Host check, but only to the notebook's own gate, and its new pass-through (B1-N1) opens a fresh path. The fix is still one small, separate change: apply the same Origin/Host check in one `before_request` for every state-changing route, and harden the notebook gate, the XFF hop, the admin gate and the `.git` refusal in the file fence.
4. **Layering keeps eroding with each feature.**
   - Upward floor imports went from about 22 to 29.
   - The HEAD commit added an L1→L2 import.
   - Imports hidden inside functions went from 1,677 to 1,720.

   The import-direction rule is held only by review. A cheap AST check over all imports, function-level ones included, would make it visible again.
5. **Documents keep growing with history:**
   - `transport.md`: 4,768 → 4,969 lines;
   - `plan.md`: 253 → 282 KB;
   - `docs/README.md` is still 102 KB;
   - one plan row is a 26 KB cell.

   Most doc–code contradictions in this audit sit in dated narrative inside contract documents.
6. **What did improve:**
   - the sidecar geometry pin (A2-01);
   - the warm-retry log (A5-08);
   - the trajectory rebuild (A8-01);
   - VibrationView's WebGL contexts (A9-03);
   - the path fence (A7-12);
   - removal of most source-text tests;
   - producer/consumer JSON field names across the Results tab now match exactly (A9).

---

## 8. Recommended order of work

1. **Security, one change (≈ a day):**
   - an Origin/`Sec-Fetch-Site` and Host check for state-changing routes;
   - the notebook gate: not loopback-implies-account; Jupyter bound to loopback;
   - the rightmost XFF hop;
   - an admin gate on `notify_setup`;
   - refuse `.git`/`NEVER_STORED` components in `_resolve_within_roots`;
   - harden the 2.11.0 notebook pass-through (B1-N1): forward only to our live notebook, preferably over a Unix socket in the 0700 runtime directory; drop molbuilder's cookie name from upstream `Set-Cookie`; always send `script-src`;
   - correct the CSRF text in `web-api.md` and `app.py`.
2. **Data loss:**
   - symlink-safe delete, rename and move (A7-06);
   - keep the file mode on write (A7-07);
   - an exclusive retry record (A5-02);
   - job address keyed by cluster (A5-N1);
   - carry the sidecar `schema_version` (A2-02);
   - neutralise XYZ titles (A2-N1, A2-28);
   - validate `frozen_atoms` and `vacuum` (A2-06/07);
   - make the save door sidecar-aware (A2-N3).
3. **Science:**
   - signed I(V) (A4-01);
   - `check_coverage` by exact `<label>.psml` (A3-N1, which fixes A3-03/04);
   - most-severe status plus a GGA family for vdW-DF (A3-01/02);
   - SLURM `MM:SS` on `--time` and in records (A5-07/N3);
   - stop dropping peptide hydrogens (A10-N2);
   - hard limits for impossible values (A3-08).
4. **Functional:**
   - the spectrum stick colours (A9-01);
   - a narrow `orderSaid` getter (A8-N1);
   - catch `ComposeError` (A4-N1);
   - a JSON error handler on `/api/*` (A7-10/33);
   - `make web` (A1-09).
5. **Restore the gates:**
   - a working pyflakes hook (flake8 `--select=F` or a local hook);
   - fix `submit.py:202`;
   - classify `joblist.py:197`;
   - run `classify_path_finders --check` at each milestone;
   - remove or write the 41 cited-but-missing tests.
6. **Tests:**
   - gate authlib, setuptools and euid;
   - unset `$MOLBUILDER_PROJECTS` in the seeding tests;
   - key the Playwright skip on the `e2e` marker;
   - fix the benzene planarity test;
   - restore the session's environment in `test_describe.py` (B3-N1), so results stop depending on worker count.
7. **Docs:** fix the user-facing commands and examples, then the contradictions listed in §§ 5–6. Move dated narrative out of contract documents.

---

## 9. Method and blind spots (R3)

**What was done:**

- Each area's diff since `83e72b2` was read in full, along with every cited prior location.
- **Run on main** (Python 3.13 venv, all state in scratch; the checkout confirmed unchanged):
  - the tier-1 suite;
  - Flask test-client probes: route enumeration, symlink operations, file modes, error pages, Host/Origin, cross-site POSTs, relative paths;
  - pseudopotential coverage on synthetic PSML;
  - `validate()` on edge-case configs;
  - rendered SIESTA and PySCF decks;
  - `linear_response_iv`, `canonical_time` and `RunNames`;
  - the retry-record and job-address collisions;
  - structure round trips (PDB, XYZ titles, sidecars);
  - node reproductions of A9-01 and A8-02, and a timing for A8-N1;
  - an AST import graph against the L1 index and floor table;
  - duplicate-window scans for Python and JS;
  - pyflakes and `classify_path_finders --check`.

**2.11.0 re-validation:**
- the tier-1 suite twice (`-n 4` and `-n 6`);
- a Flask test client with a stand-in Jupyter (a ~30-line `http.server`) and a threaded Werkzeug server for the WebSocket join;
- cross-site POSTs, Host checks and the 51 MB body;
- `fit()` scenarios;
- the `--time 30:00` launch through the repository's stub `sbatch`;
- the retry and job-address collisions;
- pyflakes, the classifier, and the AST layering graph diffed against `f3d5a4e4`;
- jupyter_server and JupyterLab source, read for B1-N3 and B1-N4.

**What could not be seen:**

- **No real engines:** no SIESTA, TBtrans or PySCF runs. PySCF and openbabel are not installed (A3-27 and A4-27 are CV).
- **No real SLURM:** A5-N2 is unverified.
- **No real browser (R4):** the visual and race items in the front end rest on node reproductions and reading. The reach of A6-N1 under browser private-network protections is unverified.
- **No real supervisor or Jupyter:** the pidfile race and token exposure were checked by reading. At 2.11.0, no real JupyterLab, nginx or signed-in browser, so B1-N1's admin leg, B1-N2 and the nginx case of A6-N2 are by reading.
- **No e2e or tier-2/3 lanes.**
- **Unchanged code was not re-read end to end:** `task-setup/viewer.js`, the spectra core's unchanged lines, builders, `jupyter.py`, `serve_daemon.py`.
- **Plan sampling:** only about 10 of 162 `file:line` citations and about 15 "done" claims in `plan.md` were spot-checked.
- **Duplicate scans** cannot see renamed or reshaped copies, or copies shorter than 8 lines.
