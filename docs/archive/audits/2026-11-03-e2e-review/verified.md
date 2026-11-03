# Findings verified by hand (2026-11-02)

## W4
- W4-1 ✔ presenters.toml heading rows (257-290) give meta {engine, role, parser}; trajectory.js:85-88 reads only
  engine_words + noun (presenters.md § 2 l.136-142, 2.65.0) → 5 rows answer "Trajectory". Fix the rows (engine_words,
  noun from the tables) — the fixture's meta must be what /api/results/dir answers.
- W4-2 ✔ results.md § 6.3 (l.1343) "the driver is checked once, when the server starts … Restart it" and the notice
  (lib/system-load-monitor.js:178-181) — but blueprints/system_load.py:82 `_SMI_TTL_S = 3.0`, :142-146 asks
  nvidia-smi again after 3 s. A PRODUCT defect (the notice tells the person to restart for nothing) + contract text;
  the test pins the wrong words. Fix contract § 6.3, the notice, the test.

## W2
- W2-stale ✔ "Loading from schema" and "cell was checked" occur nowhere in molbuilder/ — the absence asserts at
  test_build_e2e.py:185, :888 are vacuous.
- W2-prune ✔ (partly) the tab's panel saves under `workspaceId("structure-opt:panel")` = `ws-structure-opt-panel`
  (structure-optimization/viewer.js:108-109; dispatcher.js:77); the fixture prunes `ws-structure-opt`, the bare tag
  MolView's key — the panel's form state is not cleared by it.
- W2-method ✔ the card's "DFT — stated" for a form whose Method nobody chose: form-schema.md § 1.1 l.103 "A value the
  person did not choose is never presented as if they had"; chemistry-correctness.md l.409 "method is never blank: the
  catalogue's DFT is written into a template like any other value". Not a conflict: the VALUE is never blank (DFT), but
  its WORD is "stated" while nobody stated it — the card's provenance word is the defect (a measured defect against
  form-schema § 1.1), unless the form starts with DFT chosen. To check at fix time: what the PySCF form's Method shows
  on a new form, and the card's provenance words (electronic_state.py:292).

## W1
- W1-1 ✔ task-setup.md § 6.2b l.698 "The launch rows stand on every rung's card from the start"; the test (prep_e2e
  :186-198) adds mpi_np by "+ Add setting" because "the card starts with no rows" and cites the removed stages § 6.8d.
- W1-2 ✔ bench sweep fixture puts use_gpu/diag_algorithm in SiestaConfig (template), writes the calculation's
  environment.json by hand (:73-79).

## W3
- W3-picker: the picker's option value is the server's entry path (file-picker.js:396 `opt.value = r.path`); whether
  that equals the absolute run-door path the test waits for is UNVERIFIED by reading — the e2e run shows it.
- W3-doc ✔ molview.md § 1.1 "Every edit lays down a point of its own" (user, 2026-09-07; model.js lays a point per
  edit) vs § 11.2 "An edit … does not record a state", the test table row l.4947 "an edit records nothing". The
  contract contradicts itself; § 1.1 is the user's and the code's. Fix § 11.2/11.2a/11.3/13.3 + workspace.md § 3,
  and the tests' citations → § 1.1.

## P
- P-1 ✔ test_vibration_e2e.py:648 expects "frozen_atoms: 1 listed indices"; the banner now says
  `_held_words` (runwrap.py:689-705): "1 atom held (the deck's atom record)". Will fail.
- P-2 ✔ test_notebook :234-259 connects to 192.0.2.1 (UDP) and the outward address; conftest's guard
  (conftest.py:105, 116, 134-145) raises ReachedOutOfTheSandbox, a BaseException — `except OSError` /
  `pytest.raises(ConnectionRefusedError)` cannot catch it. Will fail. Revision: assert the notebook's listening socket's
  bound address is loopback (psutil connections of the port) — the outcome, with no connection off loopback.
- P-3 plausible: 2.63.0's deck read (runwrap.py:1479-1483, stderr into the session log) adds a third
  "did not load" line with a broken bundle. The two broken-bundle tests stand on a hand-edited product (faked input →
  retired, gate 5); test 1 (monitor start logged) stays.

## T2
- No wrong detail / stale road claimed. CHANGES WITH 2.65.4: :5051 (junction.xyz, slot-provenance.json, H.psml right
  after the tab's Describe, :5078-5079) and :4648's describe note (:4674-4676, composed in write_reusing).

## S1
- S1-1 ✔ "launch this stage again to continue" occurs only in pyscf/relax_policy.py:163; SIESTA's walled member
  says the run-of-points words (runwrap.py:1329-1335) — the absence assert at siesta_vibration :1718-1720 is vacuous.
- S1-2 ✔ `ir-atoms` occurs nowhere under molbuilder/web — :1068's count-0 is vacuous (removed in 2.49.0).

## S2
- S2-1 ✔ (by reading) runwrap.py:2569-2581 pads every banner label to the longest one printed (with a program, its
  "<words> version" label); the wall test (:117) and running-a-job § 3.6 l.643/656 + job-contracts § 4.4 copy a fixed
  10-space padding. The padding is presentation: the test reads the line by its label and value, whitespace-tolerant;
  the docs show the line without claiming a column.
