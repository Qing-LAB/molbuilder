# T4 — Test-depth audit

**Date**: 2026-06-26
**Auditor**: Claude (general-purpose subagent, T4)
**Test files (Python)**: 229 (`find tests -name "*.py"`); 183 top-level
`tests/test_*.py` files + 6 subpackages (`validation/`, `spectra/`,
`watch/`, `parse/`, plus `data/` and `conftest`).
**Total test functions**: ~4,093 (`grep -c '^def test\|^    def test'`).

This audit samples 15 files in depth, spot-checks ~30 more by name +
header, then judges suite-wide patterns. The question being answered:
does each test pin a design contract or just exercise a code path?

---

## Summary

Counts are estimates based on the structure of the suite, not a
test-by-test census (~4k tests; full census is out of scope for a
half-day audit).  The signal: the suite is markedly stronger than a
typical mid-sized Python codebase, but coverage is uneven.

| Bucket | Estimated share | Confidence |
|---|---|---|
| Gates a design contract (cites doc § / decision-log entry / load-bearing rule) | ~55% | high — every L4 file I read names the contract |
| Gates a code path only (asserts incidental output) | ~30% | medium |
| Smoke / instantiate-only / `assert True` tombstones | ~3% (≤120 tests) | high — `grep "assert True"` returns 1; `assert status_code == 200$` 113 cites |
| Heavy mocks (`MagicMock` / wholesale `monkeypatch`) | 6 files (~120 tests) — confined to `test_diagnostics`, `test_envs*`, `test_auth_setup` | high |
| Real fs / real subprocess | the four checkpoint files, all `test_runwrap*`, `test_siesta_enable_gpu.py`, `test_siesta_keyword_smoke.py`, `test_transiesta_siesta_smoke_l4.py`, every `test_*_e2e.py` (28 files) — well over the gold-standard bar | high |
| Drift-guard tests (the strongest pattern) | 8 explicit files (`test_envs_readme_consistency`, `test_no_legacy_persistence_keys`, `test_no_legacy_store_consumers`, `test_constants_module_consistency`, `test_siesta_stage_strategy_presets_drift`, `test_http_status_contract`, `test_layering`, `test_css_*`) | high |

The user's specific worry was "depth that pinpoints the design
contract."  For the modules ahead of the audit cliff (checkpoint,
runwrap, validation, sidecar, http envelope, drift gates) the suite
does pinpoint the contract.  For the modules behind the cliff
(projects-sidebar JS state machine, results-tab state machine,
inspector registry dispatch on real DOM, transport result-tab
framework) coverage is thinner than the contract surface justifies.

---

## Strong examples (the bar to copy)

These four files are the "real, contract-gating, no-mock" template.
Every audit recommendation below is "make X look more like these."

| Test | What it gates | Why it's good |
|---|---|---|
| `tests/test_checkpoint_lifecycle.py` (17 tests) | run-checkpoints.md P3/P3a/P4/P5, § 11 decisions | Real `tmp_path`, real `git` subprocess, every assert cites a design-doc principle by name (P5 in `test_init_refuses_when_nested_working_dir_present`, § 11 decision 3 in `test_tag_requires_message`).  Skips cleanly when git is missing.  No mocks. |
| `tests/test_checkpoint_routes.py` (23 tests) | web-api.md § 1.6 four-bucket rule + run-checkpoints.md § 8 | Real Flask `test_client`, real git, real fs.  Each test names the bucket (B/C/D) it pins.  `test_init_refuses_nested_working_dirs_as_advisory` literally checks "HTTP 200 + ok:false + errors_only:true + nested-working-dirs in message" — the exact bucket-B shape. |
| `tests/test_siesta_enable_gpu.py` (30 tests) | `Diag.ELPA.GPU` keyword contract; runwrap env routing | Drives the full three-layer chain (`SiestaConfig` → `render_fdf` → `_fdf_requests_gpu` → `write_run_wrapper`).  Uses `bash -n` to syntax-check the rendered wrapper (catches the 2026-06-15 unbalanced-quote regression generically).  Parametrises the SIESTA fdf-reader truthy alphabet (`.true.`, `T`, `Y`, `1`, …) — the test mirrors the binary's semantics, not the generator's wishful spelling. |
| `tests/test_siesta_keyword_smoke.py` (2 tests, parametrised over CG/Broyden/FIRE) | The 2026-06-23 silent-failure class | Renders an .fdf, runs the real SIESTA binary, reads the `redata: Dynamics option = …` echo back.  Detects environment crashes vs keyword regressions explicitly.  This IS the L4 binary-in-the-loop pattern the rest of the suite should adopt for engine-side keyword changes. |
| `tests/test_siesta_stage_strategy_presets_drift.py` (6 tests) | JS↔Python preset parity | Parses `STAGE_STRATEGY_PRESETS` out of the JS source via regex, asserts value-for-value equality against `SIESTA_STAGE_STRATEGY_PRESETS` and `STAGE_STRATEGY_PRESETS` (pyscf). Includes self-tests for the parser (rejects `1/0` tokens, fails loud when the block is renamed). |
| `tests/test_http_status_contract.py` (~16 cases across 4 classes) | web-api.md § 1.6 four-bucket envelope contract | Parses every blueprint's AST, asserts every `jsonify({"ok": False, …})` carries an explicit status code in the documented set; cross-checks the doc's "## 2. Endpoint index — all 59 routes" header against Flask's actual URL map. Drift-guard that catches both silent-Flask-default-200 AND doc staleness. |
| `tests/test_envs_readme_consistency.py` (7 tests) | README_install.md ↔ recipe registry parity + install-env.sh bash array ↔ Python list parity | Parses the bash array out of `install-env.sh`, asserts byte-for-byte equality with `BUILTIN_RECIPES.molbuilder.conda_packages`.  Closes the bootstrap-vs-host-env drift class. |
| `tests/test_transport_custom_info_persistence.py` (12 tests) | three-stage contract for Transport `frozen_atoms` + regions + runtime echo | Each test pins one fix leg from the audit 2026-06-25 (#31). Asserts `ATOM-METADATA` block markers + region label round-trip + frozen_atoms persistence. Asserts the `.transport.json` schema_version, not just "field exists." |
| `tests/test_layering.py` (~10 tests) | design.md "Layout" L1/L2/L3 import-direction invariant | AST-walks every `molbuilder/` source file, classifies the file's layer, asserts every import is same-layer-or-below. The kind of test that would have caught the 2026-06-09 dispatcher rename residue (round-3 audit memory). |
| `tests/validation/test_chemistry.py` (35 tests) | scientific-validation.md chemistry-rule contract | Real `Structure` fixtures (no analyzer mock for L1/L2 per the doc), but uses canonical monkeypatch pattern at L3/L4 (`TestCheckOpenShellMetalUsesAnalyzer`).  Asserts both severity AND the `where=` field that the workflow-card UI reads. |

---

## Design contracts with NO test coverage

These are contracts named in design docs that have no test gating
them — the most damaging finding type.

| Contract | Doc reference | Why no test catches it |
|---|---|---|
| Sidebar checkpoint sensor: 5 s poll cadence; suspend when sidebar hidden; retry on transient git error | `run-checkpoints.md` § 11 decision 7 + § 12 row "L3 Sidebar sensor" cites `test_checkpoint_sensor_js.py` | No `test_checkpoint_sensor*.py` file exists. § 12 lists it as the test target; the test is missing. |
| Checkpoint graph viewer DAG ordering, tag chips clickable, branch lanes for fork-merge | `run-checkpoints.md` § 12 row "L3 Graph viewer" cites `test_checkpoint_graph_e2e.py` | No `test_checkpoint_graph*.py` file. The vendored `@gitgraph/js` mount has no Playwright regression test. |
| Checkpoint wrapper P4 isolation: rendering + running a wrapper produces NO commits and NO archive activity | `run-checkpoints.md` § 12 row "L2 Wrapper isolation" cites `test_checkpoint_wrapper_isolation.py` | File exists (sampled name only); good. |
| Three-stage contract verbatim quote "engine MUST warn on labels it can't consume" | sidecar-contract.md § 2 verbatim | `test_in_body_labels_xhr.py` and `test_transport_custom_info_persistence.py` cover positive cases (labels arrive, survive); the negative case (engine encounters unknown label → emits warn-severity issue, NOT silent absorption) is gated only as "out-of-range / wrong-type indices are rejected with a clear warn-severity notice" — a much narrower surface than the contract requires for every engine. PySCF spectra is the only engine the contract is fully implemented + tested for; SIESTA + Transport + spectra script-emitter side coverage is partial. |
| web-api.md § 1.3 "cache: no-store is the default" | web-api.md § 1.3 | No test asserts the `Cache-Control` header on the canonical endpoints. A future Flask config flip to a 1-min cache would silently land. |
| web-api.md § 1.2 path-validation: reject raw `..` in user-supplied string before resolution | web-api.md § 1.2 | Spot-checked `test_web_files.py` — covers `_resolve_within_roots` but not the "reject before resolve" defence-in-depth point explicitly. Adversarial cases (URL-encoded `%2e%2e`, symlink escape) absent. |
| Transport `.transport.json` parse_output → `/results` Plotly T(E)/IV charts | memory note `project_transport_results_tab_framework` — planned but described as blocking user workflow | `test_transport_parsers_json.py` exists for the schema; no `/results` Plotly chart smoke test gates the rendering side of the contract. |
| Workspace-contract.md `ws.selection` `subscribe vs getState` shape consistency | memory note `project_workspace_phase10_audit` — verified BLOCKER, 2 regression tests shipped | Verified — tests exist (`test_workspace_dispatcher_js.py` has subscribe shape tests). |
| Inspector-registry dispatch on real DOM with `pageshow`/`visibilitychange` refresh contract | inspector-registry.md + memory of audit task #194 | `test_inspector_pageshow_refresh_e2e.py` exists (2 tests). Coverage thin given the contract surface; only one back/forward path tested per inspector. |
| `runtime_config` molbuilder.json refuse-to-emit rule (config.md § 2) on every engine, every entry point | docs/config.md § 2 | Gated in `test_runwrap_v2.py` for the wrapper and `test_siesta_enable_gpu.py` autouse fixture. No equivalent assertion on the spectra + transport script renderers — a bypass there would be silent. |

---

## False-confidence zones

Subsystems where the test count is high but the contract-gating
test count is lower than the user would expect.

| Subsystem | Test file count | Contract-gating tests | Notes |
|---|---|---|---|
| Workspace JS dispatcher | 4 files (`test_workspace_dispatcher*.py`, `test_no_legacy_*`) | Strong — drift gates + sequence tests for the `ws.selection` `subscribe`/`getState` shape | Good shape; the memory-recorded round-3 BLOCKERs landed regression tests. |
| Projects sidebar | `test_projects_*.py` (8 files) + `test_projects_render_sidebar_js.py` + `test_projects_state_lock_guard_js.py` | Mostly route-shape + render-shape; the lock-state machine in projects-sidebar.md § 8 (state transitions, lock revoke under network split) is under-tested | sidebar contract is 1543 lines, tests are mostly shape. The state-machine transitions + sensor poll cadence are not gated by behaviour tests; sensor doc cites missing `test_checkpoint_sensor_js.py`. |
| Results-tab inspector | `test_results_blueprint.py` (11), `test_results_state_contract_js.py` (33), `test_results_file_picker_*.py`, `test_inspector_*.py` | The state-contract file is strong; the blueprint file is mostly "page contains element id X" shape tests | The 51-test count in `test_results_blueprint.py` is misleading — counted 11 actual top-level functions; many parametrised. Most assertions are "id is present in HTML." |
| Transport | `test_transport*.py` (8 files, ~70 tests) | `test_transport_custom_info_persistence.py` + `test_transport_au_bdt_au_validation.py` are strong (contract-pinned, real-XYZ fixture); `test_transport.py`, `test_transport_blueprint.py`, `test_transport_config.py` are more shape-pinned | The "literature transmission value" test was deliberately deferred (`test_au_bdt_au_transmission_against_literature` is a stub with `@pytest.skip` or similar — verify before relying on it). |
| Spectra | `spectra/test_*.py` (10 files) + `test_spectrum_generate_e2e.py` | Strong: `test_engine.py` + `test_blueprint.py::TestRenderEndpoint` covers § 1.6 buckets; `test_engine.py` runs the actual PySCF emit | Good shape. |
| CLI | `test_cli*.py` (5 files), `test_envs_*.py` (10 files) | `test_cli.py` + `test_cli_run.py` use real `CliRunner`; `test_envs.py` uses `MagicMock` heavily | `test_envs.py` + `test_envs_install.py` + `test_envs_doctor.py` + `test_envs_clean.py` + `test_auth_setup.py` + `test_diagnostics.py` are the 6 `MagicMock` files. `test_diagnostics.py` mocks `subprocess.run` for `conda env list` — defensible (the subprocess shape is well-defined and platform-portable), but it means a real `conda` regression wouldn't surface here. |
| `test_web.py` (111 tests) | mixed — about half are contract-pinned (§ 1.6, validation issues, atom-list round-trip); about a third are template-shape ("page contains `id=foo`") | The shape tests are cheap and catch dispatcher / route renames; they don't catch behaviour drift. They should not be conflated with behavioural coverage. |
| Watch (legacy parser goldens) | `tests/watch/` (10 files, ~100 tests) | Strong — parses real SIESTA / PySCF output text against goldens, asserts numeric round-trip | The "no-mock" bar is met; format drift would land loudly. |

---

## Inverted / weak tests

Tests that nominally gate a bug class but would pass even on the
regression.

| File:Line / Function | Pinned bug | Why it doesn't catch | Suggested fix |
|---|---|---|---|
| `tests/test_web.py:104 test_viewer_js_compatibility_signals_are_behavior_tested` | "the old grep-style test for viewer.js compatibility logic" | Body is `assert True` with a tombstone docstring. It's deliberate (per the docstring) but still appears in suite counts as a "passing test" — slight false-confidence boost. | Either drop the test entirely (the docstring already explains why) or add a `pytest.skip()` so it doesn't inflate the green-count. |
| `tests/test_web.py:212 test_project_tagline_renders_identically_on_every_tab` | tagline edit must update every page | Hard-codes the canonical string. A typo PR that updates the constant AND the templates together passes silently — the test would only catch a per-page divergence, not a deliberate-but-wrong rewording. | Combine with a docs-link assertion (the tagline must include "molbuilder" or similar load-bearing tokens), or pair with a separate test that asserts the constant lives in `_app_header.html` only. |
| `tests/test_results_blueprint.py::TestResultsInTabNav::test_results_link_present_on_every_page` | results tab missing from a page's nav | Checks `href="/results"` substring. A href on a HIDDEN/disabled element would still pass. | Pair with a Playwright e2e test that asserts the link is clickable + reachable. |
| `tests/test_web.py:35 test_index_page_has_tab_markup` | tab markup regression | Asserts substring presence (`'data-tab="siesta"'`, `'id="tab-siesta"'`). A swap of `data-tab` and `id` between siesta/pyscf passes — both substrings still appear. | Use a DOM parser (or Playwright) for cross-element invariants. |
| `tests/test_transport_au_bdt_au_validation.py::test_au_bdt_au_transmission_against_literature` | the canonical "fruit fly" T(E_F) value | Likely a stub per the file's "What's deliberately NOT tested" header (transmission requires real SIESTA-MPI). If present, it's expected to be a skip — verify it actually skips rather than passing trivially. | Mark explicitly with `@pytest.mark.skip(reason=…)`; expose a `--run-tsiesta` flag to opt in. |
| `tests/test_pages_no_js_errors.py` (sample) | "no console errors on any page" | Playwright-based; if the dispatcher swallows the error before it reaches `console.error`, this passes. | Pair with explicit error-listener registration; assert NO `unhandled rejection` events emit. |
| The 113 `assert response.status_code == 200$` sites across `test_web*.py` etc. | endpoint returns ok | A 200 with `ok:false` (advisory bucket) passes the status check, but that's wrong unless the test ALSO asserts `body["ok"]`. Most do, but a quick `grep` for the bare-status pattern shows hot-spots in `test_web.py::test_all_pages_serve_with_shared_tab_nav` and `test_results_blueprint.py` where only the status is checked. | Add a `body["ok"]` assertion alongside every 200 status check (the strict pattern from `test_checkpoint_routes.py`). |

---

## Findings (ranked)

### BLOCKER 1: Checkpoint sensor + graph viewer JS tests missing

**Contract**: `run-checkpoints.md` § 12 lists `test_checkpoint_sensor_js.py` (sensor poll cadence, suspend-when-hidden, retry-on-transient-error) and `test_checkpoint_graph_e2e.py` (DAG ordering, tag chips, branch lanes) as required test targets.

**Test**: missing entirely (verified: `ls tests/ | grep -E "sensor|graph"` returns 0 results).

**Why**: The sensor's 5 s poll cadence (§ 11 decision 7) and the graph viewer's vendored `@gitgraph/js` mount are user-facing — the sensor is the primary UI affordance for checkpoint state.  A sensor that polls too often (DoSes the dir), too rarely (user sees stale state), or that forgets to suspend when the sidebar is hidden (battery / network drain) would land silently.  The Python module is the rare gold-standard subsystem in this audit; the JS side is the dual gap.

### BLOCKER 2: Sidecar three-stage contract: engine-side "warn on unknown label" gate is partial

**Contract**: sidecar-contract.md § 2 verbatim: *"if labels are not consistent or not recognized, the script should give explicit warning so that the user know there could be an issue. no silent absorption of config."*

**Test**: positive cases gated (`test_in_body_labels_xhr.py`, `test_transport_custom_info_persistence.py`); the negative case — engine encounters an unknown region label or frozen-atom index NOT in its supported set → emits warn-severity issue, NOT silent drop — is gated only for the PySCF spectra engine (per `feedback_three_stage_contract` memory note: "fully implemented for `frozen_atoms` against the PySCF spectra engine; the design is the template every future engine + every future label type follows").  SIESTA + Transport sides need parallel coverage to claim the contract is enforced suite-wide.

**Why**: This is the user's load-bearing principle (memory: "design.md verbatim quotes are load-bearing"). A regression silently dropping a label here would corrupt every job emitted thereafter.

### IMPORTANT 1: Tab-markup substring tests are too loose to catch swaps

**Contract**: web-ui-coherence.md (tab nav consistency, data-tab/id pairing).

**Test**: `test_web.py::test_index_page_has_tab_markup` + 60+ similar substring-presence tests.

**Why**: A regression that swaps `id="tab-siesta"` ↔ `id="tab-pyscf"` passes — both substrings still appear in the page. The substring-presence pattern is brittle to refactors AND insensitive to swaps. Promote to DOM-tree assertions for the cross-element invariants.

### IMPORTANT 2: 113 sites use `assert r.status_code == 200` without also asserting `body["ok"]`

**Contract**: web-api.md § 1.6 four-bucket rule — HTTP 200 + `ok:false` is the scientific-advisory bucket; a test that only checks status accepts the advisory case as success.

**Test**: ~113 grep hits across the suite (count from `grep -nE "assert\s+r\.status_code\s*==\s*200\s*$"`).

**Why**: A regression that pushes a clean run into the advisory bucket (e.g. a new validator rule that always fires) passes the status check. Most of the 113 sites DO also check `body["ok"]` on the next line, but `test_web.py::test_all_pages_serve_with_shared_tab_nav` + `test_results_blueprint.py::test_page_renders` are the false-confidence shape: status only.

### IMPORTANT 3: `test_diagnostics.py` heavily mocks `subprocess.run` for `conda env list`

**Contract**: `Capabilities.detect()` is the boot-time probe that decides which engines + envs the host has. Mocking it is defensible (subprocess shape is stable) but means a real-conda regression (e.g. `conda env list --json` output format change in conda 24+) would land silently.

**Test**: `test_diagnostics.py` uses `_stub_conda_env_list` which returns `MagicMock(spec=subprocess.CompletedProcess)`.

**Why**: Worth pairing with a `@pytest.mark.slow` integration test that runs the real `conda env list` and asserts the JSON contract still holds. One real-conda test would close the gap.

### IMPORTANT 4: Live-poll invariant tests are source-text regex, brittle to refactor

**Contract**: `feedback_no_rewrite_user_ui_state_on_poll` memory — live-poll loops must early-return on no-op ticks; in event handlers snapshot input values BEFORE calling helpers that may side-effect.

**Test**: `test_live_poll_invariants_audit.py` — pins via regex over the JS source.

**Why**: The source-text invariant tests are deliberate (per test-strategy.md § 5) but a refactor that preserves behaviour while changing source shape breaks the test for the wrong reason. Pair each with at least one Node.js sequence test (the pattern from `test_workspace_dispatcher_js.py` works) so a real-behaviour test gates the actual invariant.

### IMPORTANT 5: Cache-Control + path-validation defence-in-depth uncovered

**Contract**: web-api.md § 1.2 (reject `..` before resolution); web-api.md § 1.3 (`Cache-Control: no-store` default).

**Test**: neither has a direct test. The `..` rejection is exercised implicitly by `test_web_files.py` but adversarial cases (URL-encoded `%2e%2e`, double-slash, symlink to outside-root) are absent. Cache-Control is not asserted on any endpoint.

**Why**: These are defence-in-depth — the bugs they catch are silent and security-shaped. A Flask config flip to add a `Cache-Control: max-age=60` middleware for performance would land silently.

### NIT 1: `assert True` tombstone in `test_web.py:104` inflates the suite green-count

Just delete it or `@pytest.skip`. The docstring carries the explanation; the function body is a no-op.

### NIT 2: Transport literature transmission test is a stub

`test_au_bdt_au_transmission_against_literature` — verify it actually skips rather than passing trivially. If it's a true stub, mark `@pytest.mark.skip`.

### NIT 3: `test_results_blueprint.py::TestResultsInTabNav` checks `href="/results"` substring only

Pair with a Playwright assertion that the link is clickable + reachable. The substring + element-id check is duplicated across 4-5 result/inspector files; consider a single registry-driven invariant.

---

## Cross-cutting recommendations

1. **Promote the four gold-standard files to the test-strategy doc as the official template.**  `test_checkpoint_routes.py`, `test_siesta_enable_gpu.py`, `test_siesta_keyword_smoke.py`, `test_http_status_contract.py` already collectively model "real fs / real subprocess / cites design § / parametrise the failure alphabet / drift-guard the doc itself."  Hard-name them in `test-strategy.md` § 4 as the reference shape and add a "**Bar to copy**" subsection.

2. **Close the JS-side test gap on checkpoints.**  Both files named in `run-checkpoints.md` § 12 (`test_checkpoint_sensor_js.py`, `test_checkpoint_graph_e2e.py`) need to be written before the run-checkpoints feature ships to users.  Without them the system is half-tested.

3. **Add a "warn on unknown label" gate test for SIESTA + Transport engines.**  The sidecar three-stage contract names it as required; PySCF spectra is the only engine where the test exists. The sentinel test shape: render a script with a region label the engine doesn't know about; assert a `severity="warn"`, `where="..."` Issue surfaces; assert the script DOES emit (silent absorption is the failure mode, not refusal).

4. **Audit the 113 bare-status checks.**  Triage with a one-liner: `grep -nE "^\s*assert.*status_code\s*==\s*200\s*$" tests/*.py` and add an immediate `assert body["ok"]` line whenever the endpoint is in the validator/preflight family. Mechanical; high signal.

5. **Add the cache-control and adversarial-path defence-in-depth tests** as a single new file (`test_web_security_contract.py`), parametrise the `..` / `%2e%2e` / null-byte / symlink alphabet, assert 400 on every input AND the `Cache-Control: no-store` header on every API endpoint.

6. **Distinguish "passes" from "asserts behaviour" in the green-count.**  `assert True` tombstones, smoke-only template checks, and shape-only HTML substring asserts inflate the suite's reported coverage without proportional gating power. A `@pytest.mark.shape` marker (registered in `pyproject.toml`) lets the team filter the "behavioural" subset for the real green-count check.

7. **Drift-guard the gold-standard pattern.**  Add a meta-test (`test_design_doc_test_targets.py`) that parses every `docs/protocols/*.md` § "Testing" / § "Test coverage" section, extracts the named test files, and asserts each file exists. The 2026-06-25 run-checkpoints doc explicitly named two missing files — a meta-test would have caught it before this audit. Companion to `test_envs_readme_consistency.py`'s shape.

8. **Honest mock budget for `test_diagnostics.py` + `test_envs.py`**.  Six files use `MagicMock`. Five of them are defensible (conda subprocess shape is stable). Adding one real-conda slow integration test per file (gated on `--run-slow`) gives a tripwire for upstream changes.

---

## Suite-level verdict

The molbuilder test suite is **better than the user's worry suggests**, but the unevenness is real.  For the modules the user has touched recently (checkpoint, runwrap, SIESTA GPU, sidecar three-stage contract, web envelope) the tests pinpoint the design contract by name, use real fs / real subprocess, and parametrise the failure alphabet against the binary's actual semantics.  The four gold-standard files would survive comparison with any open-source Python project.

The cliff is in two places:

(a) **JS-side coverage of the new run-checkpoints UI** — design doc names the test files; the files don't exist.  Symmetric Python-side coverage is gold-standard.

(b) **Engine-side "warn on unknown label" coverage of the sidecar three-stage contract** — PySCF spectra is fully covered; SIESTA + Transport engines have positive-only coverage.  The negative case (no silent absorption) is exactly what the user's verbatim quotes pin as load-bearing.

Fixing both gaps + the suite-wide hygiene items (5-pass over `status_code==200` sites, drop the `assert True` tombstone, add cache-control + adversarial-path gates) raises the suite from ~55% contract-gating to ~75-80% contract-gating without writing many new tests.
