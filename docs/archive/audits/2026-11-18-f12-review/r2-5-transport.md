# F12 round 2 — R2-5, the transport package

**What was read.** Every module of `molbuilder/transport/` whole (all 22 files,
12,719 lines). Owner sides read to check each finding: `kind_table.py`
(the transport record and `COMPUTES`, `lead_of`, `rung_label`),
`data/kind_table.toml` § transport, `identity.electrode_hs_stem`,
`task.bias_token` / `Task.treatment`, `runfiles.TBT_CHANNEL_TAGS`,
`siesta_grammar.TBT_CHANNELS`, `parse/dirs/job.py` (the state tuples),
`calcdirs` roles, `siesta/engine_input.py` (`engine_defaults`, `NEQ_LINE`),
`siesta/warm-files.toml` § transport, the catalogue rows `psml_lib`,
`electronic_temperature` and the TS items, `kmesh.TRANSPORT_AXIS`,
`structure.AXIS_KINDS`, `tbtrans_nc.point_averages`, and the web route that
serves `CitationAnswers.source` (`web/blueprints/transport.py:880-900`) with
the two pages that read it.

**Tests.** Read whole: `test_transiesta_pole_energy.py`, `test_transport_record.py`,
`test_electrode_labels_are_frozen.py`, `test_transport_compose.py`,
`test_transport_cell.py`, `test_transport_wizard.py`, `test_transport_sort.py`,
`test_transport_task.py`, `test_transport_blueprint.py`,
`test_frame_set_meanings.py`, `test_transport_decks_e2e.py`,
`tests/support/junction.py`. Read in part:
`test_transport_on_a_real_junction_e2e.py` (6143 lines: 1–300, 1739–1870,
5667–5800, plus a scan of every line for text greps, hand-written files and
its transport imports); `tests/support/road.py` (193–360, 1590–1710);
`tests/_deck.py` (45–70); `test_cell.py` (195–300, for comparison);
`test_chemistry.py` (440–500); `test_siesta_relax_run_e2e.py` (535–612);
`test_siesta_vibration_e2e.py` (730–775);
`test_an_edit_on_a_recorded_structure_e2e.py` (125–181);
`test_frame_sets.py` (200–230). Grep only: `test_geometry_pin.py`,
`test_kind_table.py` and `test_docs_tab.py`. Their transport mentions are a
docstring, kind-table rows and a docs-index row.

**Totals: 23 findings.** 16 are in the code, 7 in the tests. None of them
repeats round 1. Finding 3 is the cause shared by round 1's E-D1, which is
why it is reported here.

---

## Code

### 1. Whether the citation is a run or a pair is decided in five places, and spelled two ways  *(medium; read both sides)*

- `transport/citation_defaults.py:217`: `return ("psml_lib",) if isinstance(cited, CitedPair) else ()` (`asked_of_the_person`).
- `transport/citation_defaults.py:290-291`: `if cited.source == "deck":` / `sources["psml_lib"] = "cited"`, where `source` is `"deck"` (line 121), `"pair"` (117) or `"none"` (127, a record that states nothing).
- `transport/rung.py:508`: `if composed.provenance["kind"] == "pair":` decides where prep takes the pseudopotentials from.
- `transport/deck.py:696`: `lib = getattr(cfg, "psml_lib", None)` decides which of the two `_pseudopotential_note` texts the seed deck prints.
- `transport/describe.py:423`: `pair = isinstance(cited, CitedPair)`, for the headline.
- The spellings: `compose.py:57` `CITATION_KINDS = ("run", "pair")` and `compose.py:761` `"kind": "pair" if isinstance(cited, CitedPair) else "run"` give the provenance's kind. `CitationAnswers.source` uses `"deck"` for the same kind. Both reach the browser: `web/blueprints/transport.py:900` serves `source = {"kind": answers.source, ...}`, read at `static/lib/transport/core.js:1794` `src.kind === "deck"`. The record's provenance is read at `static/lib/inspectors/transport.js:862` `prov.kind === "pair"`.

**Owner:** the classes `compose.CitedRun` / `CitedPair`, with the provenance's `kind` written from them. `asked_of_the_person` is the rule's own door; its docstring says so.

**What goes wrong:**
- `CitationAnswers.source` mixes two facts: which kind was cited, and whether its record stated anything. So a cited run whose record states no contract (`source = "none"`) does not get `psml_lib` marked `cited`, although its pseudopotentials travel with it (line 290).
- A template edited by hand to name a `psml_lib` for a cited run makes the seed deck's note say "prep copies this calculation's .psml files from {lib}". Prep still takes them from the run (`rung.py:508`), so the note is untrue.

**Smallest fix:** keep one word for the kind, "run" or "pair", taken from the class. Carry it on `CitationAnswers` as `kind`, separate from "states nothing". Have the template text, `provide_data` and the seed note each ask `asked_of_the_person` / the provenance kind, never `source` or `cfg.psml_lib`.

### 2. Which lead is the lower one is decided twice, and the transport axis is typed as `2` in two modules  *(medium; read both sides)*

- `transport/sort.py:156-158` decides it once: `lo, hi = ((left, right) if float(np.mean(z[left])) <= float(np.mean(z[right])) ...`, with `z` taken along `TRANSPORT_AXIS` (`sort.py:148`). Its own comment at `sort.py:301-304` says "a second z comparison here would be a second place that decides which block is lower, free to drift from the one every other surface asks."
- `transport/transiesta.py:163-165`: `z_centroid = float(np.mean(struct.positions[indices, 2]))` / `out.sort(key=lambda e: e[3])`. That order sets `semi-inf-direction -A3` and `elec-pos begin` (`transiesta.py:320`, `:428-433`).
- `transport/wizard.py:206` and `:386` take `pos[:, 2]` as well. `kmesh.TRANSPORT_AXIS` (`kmesh.py:39`) is the one home; `sort.py:121` records that a copy of it was already removed once.

**Owner:** `sort.electrode_orientation`, the door every other surface asks, and `kmesh.TRANSPORT_AXIS`.

**What goes wrong:** today both rules break a tie toward L-electrode, so they agree. Change the tie rule or the measure in `sort.py`, and the sort puts one block first in the atom list while the deck anchors `elec-pos begin` and `-A3` on the other. TranSIESTA then reads a lead that is not the first atoms.

**Smallest fix:** have `_find_electrode_regions` order the two blocks by `electrode_orientation`'s answer (canonical or inverted) instead of measuring z again. Use `TRANSPORT_AXIS` in `wizard.py`.

### 3. What reading a composed junction can raise has a door, and most of its readers do not ask it  *(medium; read; this is the cause of round 1's E-D1)*

- The door: `transport/rung.py:250-260`, `compose_refusals()` returns `(ComposeError, SortError, PermutationError, DeckRecordUnreadable)`. It is asked by `rung.junction_record`, `gather.composed_for_prep` and three places in `reuse`.
- Readers with their own list: `export.py:88` `except (RecordError, ComposeError, DataFileError, OSError, ValueError)`; `gather.py:318` `except ComposeError`; `deck.py:878` `except (ComposeError, AttributeError)`.
- Readers with no guard: `record.py:1007` and `:1492`, `datafile.py:662`, and `points.py:86` (`recorded = recorded_junction(Path(calc.base))`, called at launch).

**What goes wrong:** round 1's E-D1 is this. `ComposeError` is caught by `export` but escapes `collect_record` and `selection_pdos`, which gives an HTML 500 and a traceback from `summarize`. `points._run_facts` is a fourth reader with the same gap.

**Smallest fix:** every reader of `recorded_junction` catches `compose_refusals()` and turns it into its own refusal (`RecordError`, `DataFileError`, `PrepError`). No reader keeps its own list.

### 4. SIESTA's default electronic temperature is typed in the record, and the catalogue states it too  *(low–medium; read both sides)*

- `transport/record.py:253`: `return 300.0 if said is None else float(said)` (`temperature_k_of`). Its kT is used for the low-bias I–V and for a sharing point's current.
- `transport/gf_files.py:114-118`: `defaults = engine_defaults()` … `scalars.get(fdf_key(key)) or defaults.get(fdf_key(key))`. That reads the catalogue row `electronic_temperature`, whose `engine_default` is 300.0 and cites `read_options.F90:1020`.

**Owner:** the catalogue's `engine_default`, read through `siesta.engine_input.engine_defaults`.

**What goes wrong:** change or correct the engine default in the catalogue, and the file estimate follows it while the record's I–V keeps 300 K.

**Smallest fix:** `temperature_k_of` asks `engine_defaults()` for `ElectronicTemperature`, as `gf_files._Deck` does.

### 5. The transmission window's block is read by token position with its unit ignored, and its spacing is written three times  *(low–medium; read both sides)*

- Writer, `transport/deck.py:445`: `f"   from {float(lo.value)!r} {unit_word('transmission_emin_ev')}"` … `to … {unit_word(...)}`.
- Reader, `transport/energies.py:113-115`: `return ((float(rows["from"][1]), float(rows["from"][4]), int(rows["points"][1]))`. It takes positions 1 and 4 and never reads the unit word at 2 and 5.
- Another reader of this file, `gf_files._Deck`, reads its energies through `units.energy_ev`. The tests' own reader, `tests/_deck.py:45-70` `fdf_energy_window`, converts the unit "so a deck that switched unit while keeping the same figures is caught instead of silently reinterpreted". The product's reader does not.
- The same arithmetic in three places: `window.py:59-60` `return (self.emax - self.emin) / self.points` and `window.py:100` `h = (b - a) / n / finer` re-implement `energies.spacing` (`energies.py:45`).

**Owner:** `units` for a unit word (`docs/architecture.md`, the constants and units rule: "No reader keeps its own word list"); `energies.spacing` for a window's spacing.

**What goes wrong:** this reader feeds `reuse.finished_window`, the run index's `window` (`points._entry`) and the low-bias I–V spacing (`record._iv` via `window_of`). If the catalogue's unit for these items changed, or `unit_word` returned an empty word, the reader would return another number with no error, or fail with an `IndexError`.

**Smallest fix:** `deck_window` splits the row at `to` and converts each bound through `units.energy_ev`, with eV as the default, as `tests/_deck.py` does. `Fitted.spacing` and `fitted` call `energies.spacing`.

### 6. Vocabularies typed again in transport modules  *(low–medium; read both sides)*

- Spin channels: `transport/conductance.py:25-26` `SPIN_CHANNELS = {"polarized": ("up", "down"), "non-polarized": ("unpolarized",)}`. The home is `runfiles.py:154` `TBT_CHANNEL_TAGS`, which `siesta_grammar.py:1395` derives `TBT_CHANNELS` from, importing both ways for the travelling bundle. More literals: `record.py:338, 1602, 1625`, `tbtnc.py:225, 250`, `datafile.py:857-860`.
- What a rung computes: `transport/run_words.py:44` `SCF, PRODUCT = "scf", "product"` against `kind_table.py:57` `COMPUTES = ("scf", "product")`. `points._run_facts` writes the kind table's word into the run index; `run_words._SAID[...]` reads it back by its own copy.
- Axis kinds: `compose.py:460` `k in ("periodic", "isolated")` against `structure.py:117` `AXIS_KINDS`.
- Treatments: `record.py:1306-1307, 1566, 1721-1730` and `reuse.py:383-388` compare with `"low-bias-approximation"` / `"self-consistent"`, words that `Task.treatment` (`task.py:300-302`) spells inline.

**What goes wrong:** rename a channel in `runfiles`, for example `down` to `dn`, and `per_spin_channel` says "the run wrote no down channel". Rename a `computes` word in the kind table, and the job's milestone fails with a `KeyError` beside the job.

**Smallest fix:** derive `SPIN_CHANNELS` from `TBT_CHANNEL_TAGS` and `SCF`/`PRODUCT` from the kind table's words, each imported both ways as `siesta_grammar` does. Give the treatment words constants beside `Task.treatment`. Use `AXIS_KINDS` in `compose`.

### 7. How many frames the junction holds has three answerers and a stored copy  *(low–medium; read; plausible drift)*

- `stages.py:222-230` `frames_of`: `int((read_provenance(base, read=read) or {}).get("frames") or 1)`. It reads a copy of the count kept in `slot-provenance.json`, written at `compose.py:799` `{"frames": cited.n_frames}`.
- `rung.py:295-318` `frames_had`: the record if there is one, else the reuse junction, else `cited_frames`.
- `gather.py:50-54` `ReadOnce.frames`: `return self.junction().junction.n_frames`. That is the junction file's own count.
- Mixed in one decision: `reuse.py:909, 938, 1274, 1333, 1350` use `once.frames()`, while `reuse.py:791, 805, 1275, 1386` use `frames_of(...)`. `gather.py:118` uses `frames_of(base)`; `points.py:32` uses `frames_had`.

**Owner:** the composed junction's own `n_frames`. The provenance's `frames` is a second record of the same fact.

**What goes wrong:** before a calculation's first prep, `frames_of` answers 1 while `ReadOnce.frames` answers what the cited set holds. `_decided` and `_forecast` then work from two frame counts in one manifest. On the roads read here the second count meets no run yet, so nothing visible differs today.

**Smallest fix:** one door, `frames_had`, used by the gather, reuse and record alike. Drop the `frames` key from the provenance, or read it nowhere.

### 8. The rung names are typed across the package, though the kind table holds the roles  *(low–medium; read)*

- The kind table's roles give `stages.py:53` `TRANSPORT_STAGES`; `window.py:25` `window_rung()` finds the transmission from the catalogue.
- Typed again:
  - `deck.py:72-90` `SHAPE_OF_RUNG` / `SHAPE_RUN` list all five rungs. The "electrode" shape is the kind table's `lead` field, and its refusal at `deck.py:787` calls this table "the ladder".
  - `stages.py:299, 301, 353, 409` (`if stage == "device"`, `"transmission"`, `("seed", "device")`) and `stages.py:441` (`rung_notes` keyed by name).
  - `record.py:439, 627-628, 1015, 1033, 1654`.
  - `datafile.py:670, 747`; `tbtnc.py:434`.
  - `gf_files.py:275, 278` `stage_home(base, task, "device")`, although which rung keeps the scratch folder is `SHAPE_RUN[...]["keeps_scratch"]`.
- Meanwhile `reuse` and `gather` name the transmission through `window_rung()`.

**What goes wrong:** rename a role in `kind_table.toml`, and `transport_spec` refuses the rung. The schema route's `notes[n]` fails with a `KeyError`. The status line, the record and the data file look for a stage that does not exist.

**Smallest fix:** derive the shape from the role (`lead` gives "electrode"; `computes == product` gives the transmission). Name the transmission in one way, `window_rung()` or a role lookup. Have `gf_files` take the stage it is called for instead of typing "device".

### 9. Run states typed in `reuse._member`  *(low; read both sides)*

- `reuse.py:1293`: `if got.state in ("queued", "running", "unreadable"):` while `parse/dirs/job.py:398` holds `LIVE_STATES = ("queued", "running")`. `gather.py:170` and `reuse._unfinished` ask the tuples.

**What goes wrong:** add a live state to `LIVE_STATES`, and the reuse would copy in a run that is still going.

**Smallest fix:** `got.state in LIVE_STATES or got.state == "unreadable"`.

### 10. The device deck's keywords are read by typed names in the file estimate  *(low; read both sides)*

- `gf_files.py:127-138`: `said("TS.Elecs.Out-of-core")`, `said("TS.Voltage")`, `said("TS.Contours.Eq.Pole")`, `said("TS.Contours.nEq.Fermi.Cutoff")`. The catalogue rows `ts_elecs_out_of_core`, `bias_voltage_v`, `negf_eq_pole_ev` and `negf_neq_fermi_cutoff_kt` state these anchors, and the deck writer writes them by anchor.

**What goes wrong:** change an anchor in the catalogue, and the deck writes the new keyword. `_Deck` finds neither the old one nor a default (`engine_defaults()` is keyed by the new anchor), and the status line says "the size cannot be said".

**Smallest fix:** look each keyword up as the item's catalogue anchor.

### 11. A bias token is read back by hand, and a voltage is put in words three ways  *(low; read both sides)*

- `gf_files.py:80`: `return f"{float(token[1:]):g} V" if token.startswith("v") else token` inverts `task.py:123` `return f"v{v:g}"`, the "ONE spelling", outside its codec.
- `stages.py:174` `Point.round` gives `f"{self.bias_v:g} V"`, while `reuse.py:381` gives `f"{float(v)} V"`. The second prints "0.0 V" where the first prints "0 V".

**Smallest fix:** a voltage in words comes from `Point(volts=v).round`. The token's inverse sits beside `bias_token`.

### 12. "What an upstream run gives a point" is written three times  *(low; read)*

- `gather.py:209`: `where = (RunPlace.of(newest).point(point.rel) if up_points else newest)`, matched by the point's folder.
- `reuse.py:784-796` `_at` and `reuse.py:799-810` `_given_at` repeat the same body: `rung_points(... frames_of(calc.base))`, match by `point_key`, `RunPlace.of(run).point(match.rel)`. They differ only at lines 793 and 807 (`run if point_key(pt) == (0, 0.0) else None` against `run`).

**Smallest fix:** one function with a flag for the run-once case, used by the gather too, or the gather's rule taken by `reuse`.

### 13. The reuse's copy filter and its set of excluded records are each written more than once  *(low; read)*

- `reuse.py:1602-1603` and `:1742-1743` both build `records = {CALCDIR_FILE, *(r.name for r in RUN_RECORDS if r.copy != COPY_CARRIED)}`.
- `reuse.py:1612, 1620, 1748` repeat `if (f.is_file() and f.name not in records and not never_carried(f.name))`.
- `_copy_of` (1503-1521) and `_relaid_copy` (1458-1474) each loop to sum size and collect scratch left behind.

**Smallest fix:** one helper for "a folder's own files a copy takes", and one for the size and what is left behind.

### 14. Two transport readers parse a point's TBtrans files two ways  *(low; read; the split comes from what can travel beside a job)*

- `run_words.py:179-202` and `:240-246` (`_on_grid`) put each printed energy on the deck's grid by bisecting within `PRINTED_HALF_DECIMAL_EV`.
- `record.point_transmission` uses `tbtrans_nc.point_averages` (`tbtrans_nc.py:787-827`), which puts them on the data file's energies by order.

These are two implementations of "a point's T(E) at full precision". `test_transport_on_a_real_junction_e2e.py:5721-5745` pins that their G/G0 agree to 1e-9. This is noted rather than proposed for merging, because the module travels beside the job without netCDF. If kept, the one rule should be stated where both cite it.

### 15. A parameter nobody passes, and a product function only a test calls  *(low; read)*

- `wizard.py:330`: `layer_tol_ang: float = LAYER_TOL_ANG` is never passed by any caller in `molbuilder/` or `tests/`.
- `gf_files.py:227` `estimate_from` is called only by `test_transport_on_a_real_junction_e2e.py:1852`. The product path builds `_Deck` and calls `file_bytes` itself.

**Smallest fix:** drop the parameter. Make the status line go through `estimate_from`, or have the test use what the status line uses.

### 16. The composed junction carries its two leads as named fields, and a rung picks one by an if/else on the region  *(low; read)*

- `rung.py:160`: `model = (composed.electrode_left if region == REGION_LEFT_ELECTRODE else composed.electrode_right)`.
- `compose.ComposedJunction` has `electrode_left` / `electrode_right`, while the kind table keys leads by region (`kind_table.lead_of`), and `ElectrodeModel.label` already is the region.

**Smallest fix:** carry the leads as a mapping from region to model, read by `lead_of(stage)`.

---

## Tests

### T1. The seam verdict and the unfrozen-lead refusal are each pinned twice, once on a geometry the user ruled against  *(medium; read both)*

- `test_transport_compose.py:56-97` builds leads with `ase.build.fcc111` by hand and asserts 4 layers give "eclipsed" and 6 give "continues".
- `test_transport_wizard.py:171-229` asserts the whole table (3, 4, 5, 6 layers) through `modify.add_slab`. Its docstring at 177-182 records the user's ruling: "THROUGH THE REAL SLAB API, not `ase.build.fcc111` by hand".
- `test_transport_compose.py:44-53` (refuses "NOT FROZEN … freeze them") repeats `test_transport_wizard.py:73-82`. Both call `extract_electrode_model` on unfrozen leads.

**Fix:** retire the two compose-file tests. The wizard file covers both rules on the road's geometry.

### T2. A toy junction is hand-built in each file, though `tests/support/junction.py` is the shared door  *(low–medium; read)*

- `test_transport_wizard.py:14-45` `_au_device` and `test_transport_cell.py:21-34` `_hex_device` are near copies: 4 Au layers, an S–S bridge at +1.5 and +1.8 Å, 4 Au layers.
- Also `test_transport_sort.py:28-55` `_junction`, `test_transport_compose.py:56-80` `_au_lead_junction`, and `test_transport_on_a_real_junction_e2e.py:218-241` `_junction`.
- `support/junction.py` already holds `bridged_chain`, `uneven_chain` and the sandwich layers.

**Fix:** move the shapes several files need into `support/junction.py` (or `road_structure` rows). Keep a local builder only where its order or geometry is the test's subject, as in the sort and the e2e.

### T3. `test_transport_cell.py` is mostly Structure and codec tests, two of them repeats  *(low; read both)*

Six of its seven tests do not touch transport (lines 42-108). `test_a_degenerate_cell_is_reported_rather_than_unopenable` (59) repeats `test_cell.py:206` and `:269`. `test_sidecar_cell_round_trip` (86) repeats `test_structure_periodicity.py:114`.

**Fix:** retire the two repeats and move the other Structure tests to the structure suite. Keep only `test_emitter_preserves_hex_cell_verbatim`.

### T4. A test-side parser of the window block sits beside the product's  *(low; read)*

`tests/_deck.py:45-70` `fdf_energy_window` parses `TBT.Contour.window`. The product's door for that block is `energies.deck_window`. `test_transport_decks_e2e.py:245-270` asserts the window three ways: `read_fdf` rows, `fdf_energy_window` and `deck_window`. `_deck.py` also lies outside `tests/support/`.

**Fix:** once finding 5 makes `deck_window` read units, assert through it alone and retire the helper.

### T5. `test_transport_decks_e2e.py` re-implements a road helper, and regex-reads a deck it also parses  *(low; read)*

- `_deck` (46-56) finds a rung's deck with `latest_attempt` and a typed `".fdf"`, beside `support/road.py:1696` `stage_file`, which the same file also uses (lines 286, 404). The deck's suffix has its door, `runfiles.deck_role`.
- Line 119, `re.findall(r"%block TS\.Elec\.(\w+)(.*?)%endblock", ...)`, reads the electrode blocks by regex. The same test reads them through `read_fdf` lines later.

**Fix:** give `stage_file` a point argument, and read the blocks through `read_fdf` only.

### T6. The species and lattice blocks are tested through a private transport function  *(low; read)*

`test_chemistry.py:459, 474` and `test_transport_cell.py:119` call `transport.transiesta._emit_geometry`. The blocks they check are written by `siesta.deck_lines` (`species_block`, `lattice_block`), the shared writer.

**Fix:** test the shared writer at its own door, or through a deck prep wrote.

### T7. The five-rung ladder is typed in a test  *(low; read)*

`test_transport_task.py:31`: `_STAGES = ("seed", "electrode_L", "electrode_R", "device", "transmission")` builds descriptions. The assertion at 78-79 states the contract's words, which is fine. The fixture's tuple is `stages.TRANSPORT_STAGES`.

---

## The five that matter most

1. **Finding 3:** what reading a junction raises has a door (`rung.compose_refusals`), but the record, the data file, `points` and `export` each keep their own list or none. This is the cause of round 1's E-D1.
2. **Finding 2:** which lead is the lower one is decided again in `transiesta._find_electrode_regions`, with the axis typed as `2`, beside `sort.electrode_orientation`. The deck's `elec-pos` and `-A3` depend on it.
3. **Finding 1:** run or pair is decided in five places and spelled "run" in the provenance but "deck" in `CitationAnswers.source`, which also mixes in "states nothing".
4. **Finding 5:** `energies.deck_window` reads the window by token position and ignores its unit word, while the tests' own parser reads the unit.
5. **T1:** the seam and unfrozen-lead tests are repeated in `test_transport_compose.py`, on an ASE-by-hand geometry that the wizard test's docstring records the user ruling against.
