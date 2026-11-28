# F12 fix 32 (the probe at the nodes of each mode's state, 2.89.0) — the review

**Verdict: closed with 2.89.0.** R3d2c: PySCF's orbital probe at the Gauss–Hermite nodes of each probed mode's thermal
state (D7, D8 — the user, 2026-09-28; § 6.11 § 9's decisions of 2026-11-22), its record what the weights alone give and
its derived numbers views; the PySCF decks' change said as its visible change.

The lessons: **a catch-all around a step catches the stops it was never meant to** — the round-1 fix that recorded an
SCF raising at a node also swallowed the wall's stop, so a run past its time limit read *finished* with blank nodes;
the reader of the fixes caught it on a real run with a one-minute wall. **A number keyed by position is not the level's
where levels cross** — energy order labels the orbitals, and at a crossing (water's bend: two virtuals of different
symmetry) or a degenerate pair split by a mode the per-label "shift" belongs to no level (−162 meV against +50/−188
followed by overlap; 31.9 against 44.1 meV): the slope was already blanked there, the shift now too, in the views
*(the user: "Blank in views, keep record")*. **Test data from an older release is not carried** *(the user: "Follow
the rule")* — the version rules are held by today's real runs stating the older version.

## The user's decisions

- 2026-11-27: schema-15 results — *"Read those without an old probe"*; the bond check — *"Numbers + a sentence, next
  release"*; the carried 2.88.0 files — *"Follow the rule"*; a marked level's shift — first *"need more context to
  discuss this"*, then, told the two cases and the options, *"(a) Blank in views, keep record"*; the last fixes —
  *"Yes, apply them"*.
- The user's soft-mode stance (*"we never, almost never, work with soft modes"*): the probe is for stiff modes; a soft
  one is said by its harmonicity, with the run's own numbers (methanol's 338 cm⁻¹ torsion), never refused.
- **Mine, said to the user**: the even share's noise floor (a `noise` mark); a failed node's explicit nulls; "reordered"
  by orbital overlap; the node counts' one home; launch saying *changed since prepared* before it runs; the progress
  log's wording a set-aside word, so a PySCF optimization launched under 2.88.0 is prepared again unrefused.

## What was built

`vibration` (`OFFERED_NODES`, `probe_state`, `probe_rule`, `probe_average`, `probe_marks`, `probe_derivations`, the
`Probe` view, `Mode.probe`); `spectra/results.py` (schema 16, `translated`, `ModeElectronicStructure`);
`spectra/derived.py`; `spectra/datafile.py` (`@12`, the `node` axis); `spectra/pyscf_vibration.orbital_match`; the PySCF
deck (the nodes along the canonical eigenvector; a failed node recorded; the wall's stop passing through; the
tolerances read back; `runrecord.stamp`); `pyscf/input.py`, `trajectory_log/emitter.py`, `script_input` (the log's
words set aside); the catalogue, `config/pyscf.py`, `kind_table.toml`, `validation/spectra.py`, `spectra/methods.py`,
`template.py`, `jobset/migrate.py`; `jobset/_cli.py` (launch's line before it runs); the load door and the page
(`web/blueprints/spectra.py`, `lib/spectra/core.js`, `_spectra_inspector.html`, `chart_table.toml`). Documents:
`engines/vibration.md`, `engines/pyscf.md`, `engines/template.md`, `execution/job-contracts.md` § 6.1,
`execution/job-request.md` § 6, `execution/job-system.md`, `model/datafiles.md`, `web/spectra.md`, `web/results.md`,
`web/web-api.md`, `science/normal-modes.md`, `science/references.bib`, `engines/vibration-frames.md`,
`engines/vibration-postprocessing.md`, `architecture.md`, `persist.py`'s docstring.

## Round 1 — four readers (code and API 5, science 6, documents 15, tests 5): 31

All applied, the user's decisions above. **The code**: a PySCF probe prepared under 2.88.0 and launched after the
upgrade ran whole and wrote a refused result, said only after the run; the way back undid migrate; the log's words
literal in a compared line refused a launched optimization; the node counts' second home; the API rows and the module
index. **The science**: a marked level's shift not the level's; the diagram joining a reordered orbital by its place;
the polaron term's sign; the even share's example; the deck's comments of the old push; a digit. **The documents**: the
gap column's old meaning; "one schema alone" in five places; the served probe missing from the API pages; old-push words
in four chapters; nodes numbered from 0 against the messages' 1; the deck's imports; the person's manual. **The tests**:
the carried 2.88.0 files against the testing rules; `orbital_match` untested; the probe panel unread; a noise test that
could not fail; stale docstrings.

## The reader of round 1's fixes: 7 findings

Applied, the user's *"Yes, apply them"*: the raising-SCF catch swallowing the wall's stop (a run past its limit read
*finished*; now *failed — launch it again*, held by a real run with a one-second wall); the set-aside rule's contract
sentence and two test rows; `orbital_match`'s sign; the diagram's mark names tested; a failed node's words; a docstring.
**Untested, said**: an SCF that raises at a node for a reason other than the wall — no road causes one.

## The e2e run

After the last fixes: the fast suite, 7090 passed; end to end, 223 — 222 passed and one failed,
`test_the_viewers_follow_a_real_run_e2e.py::test_a_followed_runs_news_keeps_the_box_and_the_zoom`: the wheel over the
level diagram scrolled the page. A diagnostic on the real followed run showed the page right — the wheel zooms the energy
axis (−548…52 eV to −519…24 eV) — and the test turning the wheel where the plot was before the panel's numbers, drawn
under the diagram as the tab opens, moved its box: the test now waits for the plot's box to stop moving, as a person's
hand does, and passes (judged against `web/spectra.md` § 3.1, never repaired to the code's output). The other files —
`test_spectrum_results_e2e.py`, `test_vibration_e2e.py`, `test_vibration_mechanics_e2e.py`,
`test_siesta_vibration_e2e.py`, `test_pyscf_relaxation_outcome_e2e.py` (the wall stopping a probe),
`test_a_person_reads_a_pyscf_run_e2e.py`, `test_vibration_frames_notebook_e2e.py`, `test_a_stage_that_has_run_e2e.py`,
`test_spectra_no_spectrum_sentence_e2e.py` — passed.

## Topic → test

| topic | test |
|---|---|
| the probe's nodes, weights, average, slope's error, harmonicity, marks, noise floor, imaginary mode, re-weighting, largest move, failed node; a marked level's shift blank | `tests/test_vibration_mechanics.py` (the probe rows) |
| `orbital_match` (a swapped pair, the HOMO moved, a sign turned) | `tests/spectra/test_methods.py` |
| the node counts' one home | `tests/test_catalogue_agreement.py::test_the_probes_node_counts_are_the_rules` |
| migrate takes the old push out and writes the probe's temperature | `tests/test_prep_calculation.py::test_a_pyscf_vibration_stating_the_old_push_is_migrated_to_the_probes_nodes` |
| the deck's items and cost line | `tests/data/the_deck.toml` |
| the log's words set aside | `tests/test_pyscf_stages.py` (`_BESIDE_THE_CALCULATION`) |
| launch says a change before it runs | `tests/data/launch_protocol.toml` |
| the probe on a real run: nodes, the bend's harmonicity ≈ 1, the noise mark, the data file @12, the load door's view, a stored average refused; schema 15 / @11 read and refused on today's runs; the panel and the diagram's marks | `tests/test_spectrum_results_e2e.py` |
| the wall stops a probe, never a failed node | `tests/test_pyscf_relaxation_outcome_e2e.py::test_the_wall_stops_a_probe_and_its_node_is_not_a_failed_scf` |
| end to end | `tests/test_spectrum_results_e2e.py`, `test_vibration_e2e.py`, `test_vibration_mechanics_e2e.py`, `test_siesta_vibration_e2e.py`, `test_pyscf_relaxation_outcome_e2e.py`, `test_a_person_reads_a_pyscf_run_e2e.py`, `test_the_viewers_follow_a_real_run_e2e.py`, `test_vibration_frames_notebook_e2e.py`, `test_a_stage_that_has_run_e2e.py`, `test_spectra_no_spectrum_sentence_e2e.py` |
