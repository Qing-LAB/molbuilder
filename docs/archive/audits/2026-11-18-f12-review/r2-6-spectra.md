# F12 review, round 2 — R2-6: the vibration and spectra side

Checkout `the checkout` at f578482db (2.77.2). Read only. Nothing in
the checkout was written; the only Python run was in-memory reads (noted where
a finding rests on one).

**Count: 26 findings** — 15 in the package, 11 in the tests. None changes a
number a run computes today. Several are two sources of one fact that already
disagree (F1, F4, F7, F11); the rest will drift the first time one side changes.

## What was read

- **Package, every module whole:** `spectra/` (all 22 modules: `__init__`,
  `activity`, `datafile`, `derived`, `displacement_sweep`, `export`,
  `infrared`, `methods`, `mode_frames`, `normal_modes`, `optical_data`,
  `pyscf_vibration`, `raman`, `results`, `selection`, `siesta_rung`,
  `siesta_vibration`, `vibrational_analysis`, `vibration_points`,
  `vibration_sum`), `sidecars/` (`__init__`, `molstruct`, `spectra`),
  `trajectory_log/` (`__init__`, `format`, `keys`, `emitter`).
- **Owner doors, the parts needed to compare:** `persist.py` 1-180;
  `structure.py` 100-116, 1167-1201, 1375-1400, 1680-1715; `runfiles.py` 55-75,
  1108-1120, 1200-1240; `runrecord.py` 240-262; `runpoints.py` 385-410;
  `parse/engines/_run_ending.py` 360-385; `parse/engines/siesta.py` 95-175;
  `parse/engines/siesta_fc.py` 1-30, 161-188; `parse/engines/molwatch_grammar.py`
  95-140, 330-370; `parse/sidecars/molstruct.py` and `parse/sidecars/spectra.py`
  whole; `parse/contract.py` 180-280; `ncdata.state`; `transport/datafile.py`
  930-940; `siesta/vibration_phases.py` 136-139, 285-325; `data/engine_table.toml`.
- **Tests read whole:** `tests/spectra/` `_helpers`, `test_activity`,
  `test_blueprint`, `test_config`, `test_derived`, `test_methods`,
  `test_motion_share`, `test_normal_modes`, `test_raman`, `test_selection`;
  `test_molwatch_emitter`, `test_runtime_info_persistence`,
  `test_cli_runtime_info`, `test_convergence_targets_card`, `parse/test_sidecars`,
  `parse/test_round2_fixes`, `test_sidecar_annotations`, `test_molstruct_json`,
  `test_spectrum_results_e2e`, `test_vibration_e2e`.
- **Tests read in part:** `test_siesta_vibration_e2e` (lines 1-440 whole, the
  rest by its function list and greps); `tests/spectra/test_spectrumchart_handle`
  (1-120), `_seal` (1-160), `_maths` (1-40), `_box_e2e` (1-50);
  `test_pyscf_relaxation_outcome_e2e` (395-412); `watch/test_api_load` (150-233);
  `test_pyscf_bundle_runs_alone` (40-105); `test_pyscf_stages` (170-235);
  `test_reserved_label_one_store` (1-40, 210-232); `test_junction_sidecar_roundtrip`
  (150-170); `test_pdb_workflow_integration` (155-170); `test_chemistry`
  (220-250, 395-440), for comparison only.
- **Tests not read beyond a grep of their slice imports:** `test_geometry_pin`,
  `test_structure_pair_one_generator`, `test_structure_envelope_protocol`,
  `test_structure_authority_roundtrip`, `test_frame_sets`, `test_transport_cell`,
  `test_structure_periodicity`, `test_script_emit`, `test_web_files`,
  `test_results_blueprint`, `test_transport_on_a_real_junction_e2e`. These are
  the structure codec's and transport's tests; they touch this slice only through
  `molstruct.sidecar_path_for` / `load` / `frozen_atoms` and
  `parse_spectra_json`.

Round 1's slice F (`r1-F-vibration.md`) covered correctness here. Nothing below
repeats it.

---

## Package findings

### F1 — Three JSON record readers, each written by hand, with three rules for NaN

**Places:**
- `sidecars/spectra.py:256-282` (`parse_spectra_json`):
  `open(p, "r", encoding="utf-8-sig")` … `json.loads(raw, parse_constant=_reject_nonfinite_constant, parse_float=_strict_finite_float)` … `if not isinstance(d, dict): raise SpectraJsonMalformedError(...)`
- `sidecars/molstruct.py:795` (`load`): `text = sidecar_path.read_text(encoding="utf-8-sig")`, then `parse/sidecars/molstruct.py:72-83`: `data = _json.loads(text)` … `top-level value must be an object`
- **Owner:** `persist.py:130-176` — `text_of` ("a byte-order mark is read past"), `json_of`, `object_of` ("A RECORD's bytes: one JSON object … so a record reads one way from the disk and through a plan"), `read_object`. `architecture.md` § 2, the "Persist a versioned JSON artifact" row: use `persist.read_object`; do NOT "hand-roll the schema check + IO".

**What already disagrees (measured in memory):**
- A `.molstruct.json` stating `"info": {"x": NaN}` reads fine through `load_text`.
- That sidecar's own writer, `molstruct.dumps`, then refuses it: `ValueError: Out of range float values are not JSON compliant: nan`.
- `persist.object_of(b'{"x": NaN}')` also accepts it.
- `parse_spectra_json` refuses the same token (`_reject_nonfinite_constant`).

So the rule "a number JSON cannot hold is not a value" is applied on read by one
reader in three. A sidecar a person edited opens, and then cannot be saved again.
Each reader also words "not UTF-8" and "not an object" in its own sentences,
where `persist.unread_words` is the one sentence.

**Fix:** both sidecar readers decode through `persist.object_of` (they wrap its
`ValueError` in their own error class). If refusing NaN on read is the rule, it
goes into `persist`, beside `json_text(refuse_non_finite=True)`, so every record
reads alike.

**Confidence:** all three read; the NaN behaviour measured.

### F2 — The `.spectra.json` version gate is coded twice, with different type rules

**Places:**
- `sidecars/spectra.py:207-218` `_validate_schema_version`: `if not isinstance(actual, int) or isinstance(actual, bool): raise …` / `if actual not in READABLE_SCHEMA_VERSIONS: raise …`
- `spectra/results.py:1361-1371` `SpectraResults.from_dict`: `sv = d.get("schema_version")` / `if sv is None or int(sv) not in READABLE_SCHEMA_VERSIONS: raise ValueError(...)`

**Owner:** `results.py` holds `READABLE_SCHEMA_VERSIONS` ("ONE home for it", in the sidecar's own docstring).

**What could drift:** `from_dict` has one caller, `_validate_dict_to_results`,
which runs right after the sidecar gate. The two gates already differ:
`from_dict` takes `"14"` or `14.0` through `int(sv)`, and the sidecar gate
refuses them. A second caller of `from_dict` (a web route, a script) gets the
looser gate and the other message. The `where` parameter of
`_validate_dict_to_results` is only ever given `repr(p)`.

**Fix:** keep one gate. Either `from_dict` applies the strict rule and the reader
maps its error to `SpectraJsonSchemaError`, or `from_dict` drops its copy.

**Confidence:** both read; the only caller found by grep.

### F3 — The sidecar's file suffix has two homes: the pairing rule and the file catalogue

**Places:**
- `sidecars/molstruct.py:203` `SUFFIX = ".molstruct.json"`, used by `sidecar_path_for` (`:230-248`). This is what `StructureCodec` writes beside every `.xyz`.
- `runfiles.py:64` `ROLE_SOURCE_SIDECAR = ".source.molstruct.json"`, `:67` `ROLE_INITIAL_SIDECAR = "_initial.molstruct.json"`, `:68` `ROLE_OPTIMIZED_SIDECAR = "_optimized.molstruct.json"`, `:1115` `JUNCTION_SIDECAR_FILE = "junction.molstruct.json"`.

**Owner:** `runfiles` is the file catalogue ("what molbuilder writes … a name composed by `runfiles.compose`", `execution/architecture.md` § 3.2). It is L1, so it cannot import `sidecars/`. The suffix belongs in `runfiles`, and `molstruct` should import it.

**What could drift:** change the pairing rule's suffix in `molstruct` and the codec
writes `<stem>.<new>` beside each `_initial.xyz`. The catalogue rows still name
`_initial.molstruct.json`, so `runs.about` and the Results tab's listing no longer
recognise the sidecar the PySCF deck wrote. Nothing fails; the file just stops
being "ours". The four catalogue names are also composed by hand from their
geometry rows (`ROLE_INITIAL` + suffix) rather than derived by the pairing rule.

**Fix:** `SUFFIX` (and `PAIRED_SUFFIX`) move to `runfiles`; `molstruct` imports
them both ways (both modules already travel). The catalogue's sidecar rows are
derived from their geometry rows by one rule, not typed.

**Confidence:** both sides read; users found by grep.

### F4 — A point's state is put into words three times, two of them divergent

**Places:**
- **Owner:** `runpoints.py:390-402` `state_word` — "A POINT'S OWN WORD … the one every printer of a point says it by": done → `done`; `NOT_RUN` → `not run`; `failed`; finished-not-done → `not done`; else the state itself (queued, running). `NOT_RUN` lives in `_run_ending.py:372`.
- `spectra/siesta_vibration.py:1029-1034`:
  `states = {r["name"]: ("finished" if r["done"] else "not done" if r["state"] == "finished" else r["state"] if r["state"] in ("failed", _rp.NOT_RUN) else "pending") …}`
  This is the same rule with two words swapped: done → `finished`, and any live state → `pending`. Its vocabulary is `results.py:252` `OPTICAL_STATES = ("finished", "not done", "failed", "not run", "pending")`.
- `spectra/raman.py:62-65` `NOT_ENDED = "not ended"` and `:406-417`, a third translation: `said in (None, "pending")` → `not ended`, then `next((w for w in ("failed", NOT_RUN, "not done") if w in said), NOT_ENDED)`. `transport/record.py:1778` holds a fourth (`("pending", "not ended")`), outside this slice.

**What could drift:** `runs-of-points.md` § 1.3 gains a state (for example
`cancelled`). `state_word` passes it through. `_optical_payload` maps it to
`pending`, so the data file says the frame is still to run, and the Raman view
says "not ended" forever about a call that ended.

**Fix:** frame and mode states are `state_word`'s words, plus at most one named
table for the data file's two extras (done → finished, live → pending).
`OPTICAL_STATES` is derived from that table, and the Raman view asks the same
table for its "worst side" order.

**Confidence:** all three read.

### F5 — The charge partitions are listed four times

**Places:**
- `spectra/results.py:263` `PARTITIONS = ("hirshfeld", "voronoi")`, which `checked_partial_charges` refuses others by.
- `siesta/vibration_phases.py:136-139` `IR_CHARGES = (("Charge.Hirshfeld", "geometry", "hirshfeld"), ("Charge.Voronoi", "geometry", "voronoi"))`: what the deck asks SIESTA, read back by `siesta_vibration.py:367-369`.
- `spectra/methods.py:310-311` `CHARGE_PARTITION_WORDS = {"hirshfeld": "Hirshfeld [Hirshfeld1977]", "voronoi": "Voronoi [FonsecaGuerra2004]"}`.
- `spectra/infrared.py:60` `DEFAULT_PARTITION = "hirshfeld"`, and the grammar's `(Hirshfeld|Voronoi)` (`siesta_grammar.py:546`).

**Owner:** one declaration. `IR_CHARGES` is the deck's own statement of what it
asks; a row there can carry its words and citation too.

**What could drift:** add a third partition (SIESTA's Mulliken) to `IR_CHARGES`.
The deck asks for it and the run prints it. Then, at the job's finish,
`checked_partial_charges` refuses the result ("a partition is one of
['hirshfeld', 'voronoi']") and `CHARGE_PARTITION_WORDS[p]` raises `KeyError`.
The finish fails on the cluster after the force constants are paid for.

**Fix:** one table of rows `(keyword, value, partition, words, citation)`.
`PARTITIONS`, the Methods words and the default are read from it.

**Confidence:** all read.

### F6 — The route words (`analytic`, `finite-difference`, `charges`, `optical`, `none`) have no home and no check

**Places:**
- Written: `pyscf_vibration.py:164, 218, 232` (`"finite-difference"`, `"analytic"`, `"none"`); `siesta_vibration.py:545` (`"charges" if charges else "none"`) and `:559` (`"optical" if … else "none"`); `vibrational_analysis.py:226-227` (`ir_route="none"`, `raman_route="none"`).
- Read: `methods.py:159, 163` (`if ir_route == "analytic"`, `== "finite-difference"`), `:278` (`raman_route == "optical"`); `raman.py:331` (`result.raman_route == "optical"`).
- Declared only as a comment: `results.py:853-884`. `__post_init__` checks every phase word against `_VALID_PHASE_STATES` (`:1092-1105`) and no route word.

**Owner:** `results.py`, the result record. Routes are a vocabulary like the
phases, which have constants and a check.

**What could drift:** respell `"finite-difference"` in the PySCF deck's module.
`ir_route_sentence` silently returns `""`, and the Methods paragraph a person
pastes into a paper loses its route sentence with no error. Respell `"optical"`
in the finish, and `has_optical_route` is false: every laser request is refused
as "not SIESTA's optical route".

**Fix:** route constants in `results.py`, checked in `__post_init__` like the
phases, imported by the writers and the two readers.

**Confidence:** all read.

### F7 — The engine's name in the data files comes from three sources, and they already disagree

**Places:**
- `spectra/datafile.py:549` `engine=str(payload.get("engine") or "")`: the result's machine name, so `<label>.spectra.nc` says "computed by siesta …" or "computed by pyscf …".
- `spectra/optical_data.py:238` and `spectra/raman.py:642`: `engine="SIESTA"`, typed.
- **Owner door:** `engine_table.engine(name).words` (`data/engine_table.toml:19` `words = "SIESTA"`, `:77` `"PySCF"`), as `transport/datafile.py:938` uses it (`engine=engine_row(ENGINE).words`). `execution/architecture.md` § 3.3: an engine's "name as a person reads it — SIESTA, PySCF (`words`)".

**What already disagrees:** one SIESTA vibration's NetCDF export holds
`spectra.nc` with source "computed by siesta 5.4.2 …" beside `optical.nc` and
`raman.nc` with "computed by SIESTA 5.4.2 …". `test_siesta_vibration_e2e.py:276`
pins the capitalised spelling for `optical.nc` only.

**Fix:** `raman.py` does not travel and can ask `engine_table`. The two travelling
writers state the result's own `engine` value the way the engine table words it.
The finish and the PySCF payload carry the words, or the writer is handed them;
neither types them.

**Confidence:** all four read.

### F8 — A SIESTA frame is built from the reading pass twice: by the parser, and by hand in the finish

**Places:**
- **Owner:** `parse/engines/siesta.py:126-143`: `Structure(elements=[row[0] for row in coords], positions=np.array([row[1:4] …]))` → `Frame(structure, step_index=len(frames), forces, max_force, max_force_constrained, …)`.
- `spectra/siesta_vibration.py:726-749` `_relaxation_record`: the same conversion with `SimpleNamespace(structure=Structure(elements=[str(r[0]) …], positions=np.array([r[1:4] …])), forces=…, max_force=…, max_force_constrained=…, step_index=i)` and a `SimpleNamespace` standing in for the parser's `Trajectory`, handed to `parse.contract.relaxation_of_output`. The same row decode appears again at `:337`, `:352`, `:701-702`, `:990`.

**Why it is there:** `siesta_fc.py:166-168` says it — "the same records the
parser's frames are built from, read beside the job where the parser is not".

**What could drift:** the two builders already differ. The finish drops steps
with no coordinates and numbers the rest by their place among all steps. The
parser keeps every step. So the relaxation record the record step writes into the
run's index, and the one `relaxation_of` makes in the viewer from the same
`.out`, can count `n_steps` differently. Any later change to the parser's frame
(units, live frames) reaches one and not the other.

**Fix:** one stdlib-plus-numpy function, step → frame, beside `siesta_reader`
(which already travels), asked by both the parser and the finish. The
`SimpleNamespace` stand-ins go.

**Confidence:** both read; the step-count difference read, not run.

### F9 — "The clock" has two identical copies and a third spelling

**Places:**
- **Owner:** `runrecord.py:245-253` `stamp()` — "THE CLOCK every record of a run is stamped with … One spelling, so two records of one run compare": `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")`.
- `sidecars/molstruct.py:333-340` `stamp()`: the same body, a second home.
- `vibrational_analysis.py:191-192` and `displacement_sweep.py:383-384`: `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")`, which carries microseconds. This is a third spelling for a vibration result's `timestamp` and the sweep's `generated_at`.

**What could drift:** a SIESTA result's `timestamp` and its run's index `planned`
or `ended` stamps are spelled differently, so `runrecord.stamp_time` must accept
both to compare them. A change to the clock's precision changes one home and not
the other.

**Fix:** `molstruct` imports `runrecord.stamp` (two ways), or both import one L1
`stamp`. The result's and the sweep's timestamps use it.

**Confidence:** all read.

### F10 — The result reads and writes the held atoms past the structure's one accessor

**Places:**
- `spectra/results.py:1117-1118`: `held = sorted(int(i) for i in (s.regions or {}).get(FROZEN_LABEL, ()))`.
- `:1260-1263`: `regions=({FROZEN_LABEL: held} if held else {})`.
- The same file at `:1393` uses the door: `_Structure.from_dict(structure).frozen_atoms`.
- **Owner:** `structure.py:1683-1699` `frozen_atoms` — "Callers use this rather than reaching into `regions` for the name themselves"; the constructor takes `frozen_atoms=`.

**What could drift:** the accessor normalises (sorted, deduped, empty removes the
label). A change there reaches one of the result's two reads and not the other.

**Fix:** `s.frozen_atoms` at `:1118`; `Structure(..., frozen_atoms=held)` at `:1260`.

**Confidence:** read both. Small.

### F11 — A result's `structure` is written in two shapes

**Places:**
- `spectra/results.py:1264` (`structure_stated`): `return s.to_dict()`, with `view` at its default, `True`.
- `spectra/siesta_vibration.py:398`: `struct.restored().to_dict(view=False)`.
- `spectra/pyscf_vibration.py:37`: `struct.to_dict(view=False)`.
- **Owner:** `structure.py:1375-1384` — "Without `view`, the view's fields are left out — a result file's embedded structure (F4: a file states the view's mark only as its map)".

**Measured:** for the same structure, `to_dict()` adds `metadata.order_unstated: False`,
which `to_dict(view=False)` leaves out. A result built with no `structure`
(`vibrational_analysis(structure=None)`) is written with the view's field. So is
a pre-v9 result read and served through `to_dict()`. A result from either
writer's road is not.

**Fix:** `structure_stated` calls `to_dict(view=False)`.

**Confidence:** both read; the key difference measured.

### F12 — The result's key sets live in `results.py`, and the data file keeps its own copy, checked in some blocks and not others

**Places:**
- `spectra/results.py:220-248`: `_ES_KEYS`, `_MODE_KEYS`, `_EQUILIBRIUM_KEYS`, `_RESULTS_KEYS`.
- `spectra/datafile.py:521-524`: the top-level list typed again (`*_ROOT, "schema_version", "free_atom_idxs", … MAP_KEY, "partial_charges"`).
- `datafile.py:588`: `block = {k: eq.get(k) for k in _EQUILIBRIUM}`.
- `datafile.py:611-632`: the modes loop, and `:656-663`, the probe loop, which read named keys and check none.

**What could drift:** the data file's own docstring says "A key of a block this
file does not know is refused by name". That holds for the root, `relaxation`,
`thermo` and the grid. A new key in `_MODE_KEYS`, `_ES_KEYS` or
`_EQUILIBRIUM_KEYS` would be written to the JSON and silently left out of the
`.nc`. Only the e2e round trip (`test_spectrum_results_e2e.py:291-316`) would
see it, and only on a run that fills it.

**Fix:** the writer checks each block's keys against `results.py`'s sets, the
way it already checks the root. Its own list of top-level keys becomes "every key
of `_RESULTS_KEYS` but the read-only `structure_hash`".

**Confidence:** read both. Low impact today; every current key is covered.

### F13 — SIESTA's force-constant step order is stated in two places

**Places:**
- `spectra/results.py:266-267` `FC_STEP_PATTERN = (("x", -1), ("x", 1), …)` (the output's steps), and `:287-290`, which counts `1 + 6 * len(free)` with a bare `6`. The same module's caller uses `len(FC_STEP_PATTERN)` (`siesta_vibration.py:421`).
- `parse/engines/siesta_fc.py:94` `rows.reshape(n_displaced, 3, 2, n_atoms, 3)` with its docstring (`:5-6`: "each direction, each side (minus, then plus)") — the `.FC` rows.

**Owner:** the SIESTA reader (`siesta_fc`). The order is SIESTA's, not the result's.

**What could drift:** both describe one SIESTA rule (`Src/fc`). A correction to
one, for a SIESTA version that orders otherwise, leaves the charges' steps and
the Hessian's rows disagreeing.

**Fix:** `siesta_fc` names the pattern; `results.py` imports it and uses its length.

**Confidence:** plausible. Both read; whether SIESTA ever reorders is not known.

### F14 — The point entry is unpacked key by key

**Places:**
- `siesta/vibration_phases.py:285-294` `point_entry` returns exactly `runpoints.add_point`'s keywords (`rel, words, phase, round, must_hold, verdict_counts, facts`).
- `spectra/siesta_vibration.py:891-906` unpacks them by hand: `_rp.add_point(index, rel=e["rel"], words=e["words"], phase=e["phase"], round=e["round"], must_hold=e["must_hold"], verdict_counts=e["verdict_counts"], …, facts={**e["facts"], …})`.
- `spectra/vibration_points.py:245-246` picks four of them again: `{k: e[k] for k in ("words", "phase", "round", "facts")}`.

**What could drift:** a key added to the entry (for example `needs` by phase) is
dropped at both call sites.

**Fix:** `add_point(index, **e, needs=…, planned_by=…, facts=…)`. `_entry`
states which keys the declaration keeps, by `point_entry`'s own key list.

**Confidence:** read. Low.

### F15 — The log's writer and reader share the SCF row by position, outside the log's vocabulary module

**Places:**
- `trajectory_log/emitter.py:224`: `fh.write("#  cycle      energy(eV)         delta_E(eV)        gnorm(eV)                ddm        wall_time(s)\n")`, and `:235-238`, the row in that order.
- `parse/engines/molwatch_grammar.py:343-366` `scf_history_row`: `parts[0]` cycle, `[1]` energy, `[2]` delta_E, `[3]` gnorm, `[4]` ddm, `[5]` wall clock. Plus `SCF_COLUMNS` (`:120-133`) and `OLD_GNORM_HEADER = "gnorm(eV/Ang)"` (`:339`).
- **Owner by its own words:** `trajectory_log/keys.py:163-166` — "THE LOG'S LINES, each word as it is written: the writer renders these … and the reader's patterns are built on them". It holds every other line and not the SCF table.
- Inside `emitter.py`, the atom row format `"   {el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}"` is written three times (`:109-110`, `:216`, `:221`).
- `MolwatchEmitter.__init__` (`:139`) types `engine="pyscf"` in a shared L1 module, where the PySCF deck that constructs it could pass its own engine.

**What could drift:** insert a column in the writer (for example an SCF
criterion) and the reader reads `ddm` as `gnorm`, with no error. Change the
preview's precision and the steps' stays.

**Fix:** the SCF columns' order and header words become a row in `keys.py`.
`SCF_COLUMNS` keys are already the same names; the writer and the reader both
take the order from there. One atom-row formatter in `emitter.py`.

**Confidence:** read both.

---

## Test findings

### T1 — The progress log's writer is tested by string-matching its text, not through its reader

**Places:**
- `tests/test_molwatch_emitter.py:70-95`: `"# molwatch trajectory log v1" in txt`, `"# engine: pyscf"`, `"kind: initial_preview"`, `"energy (eV): None"`, `"scf_history begin\nscf_history end"`.
- The same file `:177`, `:212-222`, `:258`.
- `tests/test_runtime_info_persistence.py:47-52` `_header_keys`, which splits `"# runtime."` by hand. `:70, :84, :94` match lines by text too.
- The same files test private state: `em._scf_buf` (`test_molwatch_emitter.py:111-134, 235-240`) and `em._step` (`:95, :179, :195`).

**Owners:** `trajectory_log/keys.py` (every one of those words) and the log's
reader, `parse.engines.molwatch` / `molwatch_reader` (testing.md: an artifact is
read through its parser).

**What goes wrong:** the tests are a second copy of the log's vocabulary. Respell
`ENERGY` in `keys.py` and the reader follows, but these tests fail on text the
format no longer uses. Break the reader's pairing with the writer (F15) and these
tests still pass.

**Fix:** write the log with the emitter, read it back through the molwatch parser,
and assert on the frames' energies, forces, SCF rows and `runtime_info`. Drop
the `_scf_buf`/`_step` asserts. `test_emitted_lines_exact_for_zero_force_step`
and `test_constructor_writes_header_and_preview_block` then pin the same
property and merge.

**Confidence:** read.

### T2 — The PySCF-molecule stand-in is copied into three files

**Places:** `test_molwatch_emitter.py:34-52` `_FakeMol`;
`test_runtime_info_persistence.py:36-44` `_Mol`; `test_cli_runtime_info.py:26-34`
`_Mol`. All three answer `natm`, `atom_coords(unit=)` and `atom_symbol(i)`.

**Owner:** `tests/support/`, the shared door for test preludes.

**What goes wrong:** the emitter starts asking the molecule something else (for
example `atom_charges`), and three copies need the same edit. The fake has a named
need (PySCF is not in the suite's env), so it stays, once.

**Fix:** one stand-in in `tests/support/`, imported by the three.

**Confidence:** read.

### T3 — The SpectrumChart page tests each carry their own copy of the browser and Plotly stand-in

**Places:** `tests/spectra/test_spectrumchart_handle.py:21-87` and
`test_spectrumchart_seal.py:32-88`. Each has: `BROWSER = DOM + """…"""`, the
head that loads what is appended, `globalThis.Plotly = { react, restyle, purge,
Plots.resize }`, and `globalThis.__host` with the same nine `--spectrumchart-*`
custom properties. They have already diverged: `relayout` and `ResizeObserver` in
one; a different drawing rectangle and `config` capture in the other.

**Owner:** `tests/support/` (where `molview_dom_standin.js` already lives).
Memory rule: "build the one door BEFORE moving the first file of a group, never
copy a fixture/prelude per file".

**Fix:** one `spectrumchart_standin.js` in `tests/support/`, parameterised by the
drawing rectangle.

**Confidence:** read the two preludes.

### T4 — `test_spectrumchart_box_e2e` starts its own server fixture, a copy of `app_at`

**Places:**
- `tests/spectra/test_spectrumchart_box_e2e.py:23-27`: `def flask_server(isolated_projects_root_module): from support.live_server import serve; with serve() as base_url: yield base_url`.
- **Owner:** `tests/conftest.py:287-295` `app_at` — the same body, "ONE fixture for every file (`process/testing.md` § 2a)".

**Fix:** use `app_at`. A grep finds `def flask_server` in 27 test files, so this is
a suite-wide pattern; only this one is in this slice.

**Confidence:** read both.

### T5 — `chemistry.atomic_mass`'s label rule is pinned in the spectra tests and again in the chemistry tests

**Places:**
- `tests/spectra/test_motion_share.py:232-265` `TestAtomicMass`: `atomic_mass("Au1") == atomic_mass("Au")`, and `"au"`/`"Xx"` refused, "Contract: `chemistry.resolve_element`".
- `tests/test_chemistry.py:399-440` `TestSpeciesLabel`: `resolve_element("Au1") == "Au"`, and `"au"`, `"FE"`, `"CA"` refused.

**Owner:** `test_chemistry.py`; the contract is `chemistry.resolve_element`'s.

**Fix:** retire `TestAtomicMass`. The share tests below it already go through
`atomic_mass` (`test_a_labelled_atom_s_share_is_its_element_s`).

**Confidence:** read both.

### T6 — `test_methods.py` pins three things twice, one of them the rank rule's

**Places:** `tests/spectra/test_methods.py`.
- `test_struct_provided_emits_atom_clause` (`:291`) and `test_real_structure_dataclass_works` (`:334`): both check "N atoms" and the 3N-6 count on a real `Structure`. The second's premise ("elements as List[str], not list-of-atom-objects") is no longer a case.
- `test_struct_with_frozen_atoms_counts_correctly` (`:311`) and `test_real_structure_with_frozen_atoms` (`:399`): both check that held atoms leave the free count.
- `test_a_linear_molecule_gets_3N_MINUS_5` (`:350`) asserts CO₂ → 4 and N₂ → 1. That is the rank rule, `tests/spectra/test_normal_modes.py:43-57` (row `co2-straight`, n_rigid 5). Methods only prints `modes_of`'s count.

**Fix:**
- Keep one atom-clause test and one held-atom test.
- The CO₂/N₂ cases go to `test_normal_modes`'s table if not already there (CO₂ is). Methods keeps one row showing the paragraph prints the rule's count.

**Confidence:** read.

### T7 — "No top-level `frozen_atoms` key in a sidecar" is pinned in six tests

**Places:** `test_molstruct_json.py:118` and `:454`; `test_sidecar_annotations.py:28`;
`test_reserved_label_one_store.py:229` (the designated home: "The saved file
keeps one key for labels"); `test_junction_sidecar_roundtrip.py:166`;
`test_pdb_workflow_integration.py:166`. Plus `test_structure_envelope_protocol.py:245`
on the handed-over sidecar.

**Fix:** keep `test_reserved_label_one_store.py`'s. The others assert what their
own road adds (a junction's labels, a PDB's residue identity), not this rule
again.

**Confidence:** read the lines.

### T8 — `test_molstruct_json` writes a geometry and a sidecar by hand, with a made-up hash, and reads them past the codec

**Places:**
- `tests/test_molstruct_json.py:39-56` `_toy_xyz_text` / `_write_xyz`: an XYZ typed as text.
- `:31` `_HASH = "b" * 64`.
- `:48-50` `_on_disk`: the sidecar written by `msj.dumps`.
- `:230-252` `test_round_trip_identity`: writes that pair and reads it with `msj.load`, never `StructureCodec.load`, which would refuse it because the hash is not the file's. Same `_hash()` in `test_sidecar_annotations.py:147-150`.

**Rules:** testing.md § 6 / memory: "a test's input is what the road writes".
The pair has one writer, `StructureCodec.write`.

**What goes wrong:** the round trip holds for a pair no writer makes. The same
property on a real pair is already pinned through the codec
(`test_sidecar_annotations.py::test_the_pair_roundtrips_annotations`,
`test_molstruct_json.py::test_info_rides_the_pair_whole`).

**Fix:** retire `test_round_trip_identity`; the codec round trips stand.
`to_dict`'s refusals keep a form-valid hash only where the builder alone is
tested.

**Confidence:** read.

### T9 — The e2e modules pin the same facts of the same runs twice

**Places:**
- The canonical normalisation `Σ m|L|² = 1` in the stated masses:
  - `test_spectrum_results_e2e.py:266-289`, on every state of every run (`real_co2_vibration` and `real_water_held` included);
  - `test_siesta_vibration_e2e.py:426-430` (`_states_its_masses_and_derived_values`), on `real_co2_vibration`;
  - `test_vibration_e2e.py:618-624`, on `real_water_held`.
- "Held atoms stated once" — `"frozen_atom_idxs" not in d`:
  - `test_spectrum_results_e2e.py:77`, every state;
  - `test_siesta_vibration_e2e.py:296`;
  - `test_vibration_e2e.py:555`.

**Fix:** leave both rules to `test_spectrum_results_e2e`'s every-state table.
The engine modules keep what only they can say (science windows, the orthogonality
to removed motions).

**Confidence:** read.

### T10 — `tests/spectra/_helpers.py` builds water and a held set by hand

**Places:**
- `tests/spectra/_helpers.py:20-28` `_struct_water`, a fourth hand-typed water. `support/road.py:165-174` `water_in_vacuum` and `road.WATER` exist. `test_methods.py:339-344` types the same water again, beside the helper.
- `_helpers.py:51-60` puts a held set into a `Structure` by writing `regions[FROZEN_LABEL]` and rebuilding the object. The structure's door is `Structure(..., frozen_atoms=…)` (see F10's owner).

**Fix:** take the road's water. Build the held structure with `frozen_atoms=`.

**Confidence:** read.

### T11 — Tests spell catalogue names and vocabulary words the package exports

**Places:**
- `test_pyscf_relaxation_outcome_e2e.py:405` `run.on_label(".spectra.json")`, where `runfiles.ROLE_SPECTRA` exists.
- `test_siesta_vibration_e2e.py:494` `on_label(".FC")`, where `runfiles.SIESTA_FC` exists.
- `test_siesta_vibration_e2e.py:332` `{"partial"}`, where `activity.CLASS_PARTIAL` exists.
- `test_siesta_vibration_e2e.py:295` `d["structure"]["metadata"]["regions"]["frozen_atoms"]`, where `Structure.from_dict(...).frozen_atoms` is the door, as `test_spectrum_results_e2e.py:78` uses it.
- `parse/test_round2_fixes.py:46`: the greedy parser's `path.name.endswith(".molstruct.json")`, where `molstruct.is_sidecar` exists.

**What goes wrong:** a rename in the catalogue fails these tests on the spelling,
not on the behaviour.

**Fix:** import the constant or ask the door. A contract word kept on purpose
(the phase words in `test_spectrum_results_e2e.py:114`) stays.

**Confidence:** read. Low.

---

## Checked, no finding

- `spectra/` choosing by engine (`"siesta"` in `siesta_vibration`,
  `displacement_sweep`, `mode_frames`, `methods`): allowed. "A kind's own modules
  choose by engine" (`execution/architecture.md` § 3.3).
- Constants: every unit factor comes from `constants` (`HARTREE_*`, `BOHR_*`,
  `DEBYE_E_ANGSTROM`, `ZERO_POINT_Q2_AMU_ANG2_CM1`, `CM1_KELVIN`, `HC_EV_NM`,
  `IR_KM_MOL_PER_D2_A2_AMU`). `pyscf_vibration._AU_BOHR_TO_DEBYE_ANG` is an alias
  of the constant, not a value.
- One derivation each, asked by every caller:
  - derived mode values (`derived`, written by `write_spectra_payload` and `to_dict`, checked on read);
  - the activity cut (`activity`);
  - the rank rule (`normal_modes`, asked by prep through `validation.spectra.modes_of`);
  - the thermo block (`normal_modes.vibrational_only_thermo`);
  - the displacement rule (`normal_modes.displaced`);
  - the selection (`selection.select_modes`);
  - the infrared strength formula (`activity.ir_strength_km_mol`);
  - the remedy text (`vibrational_analysis.nonstationary_remedy`).
- Masses: from `chemistry.atomic_mass` everywhere (`siesta_rung`, `results.mass_of`).
- Engine numbering: through `engine_atom_index`.
- Order records: through `atom_permutation`.
- File names: composed by `runfiles.compose` everywhere in the package except
  the export-only `.raman.nc` and `.steps.nc`, which are not files of a
  calculation (`raman.RAMAN_FILE`, `export.STEPS_FILE`).
- `trajectory_log/keys.py`'s other words are read by the grammar
  (`molwatch_grammar.py:24-78`) from the one module.
- The parse-side FileParsers for both sidecars ask the owner's reader.
- `vibration_points`' unused `read=`, `deck=` and `folder=` parameters are
  `PointsOf`'s protocol, passed by `jobset` (`prep.py:1238`,
  `materialize.py:410`).
