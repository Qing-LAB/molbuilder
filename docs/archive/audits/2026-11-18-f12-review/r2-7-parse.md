# F12 review, round 2 — R2-7: the readers of engine output and files (`molbuilder/parse/`)

Checkout `the checkout` at 2.77.2. Read only. Python was run only with `-B`, from the
checkout, to check three claims; it wrote nothing.

## Coverage

- **Read whole:** every `.py` under `molbuilder/parse/` — 52 modules, 14,069 lines. That includes
  `siesta_grammar`, `siesta_reader`, `tbtrans_nc`, `_run_ending`, `dirs/job` and `dirs/record`,
  each read in chunks from start to end.
- **Rules read:** `docs/architecture.md` §§ 1–3; `docs/model/parse.md` lines 1–175 (§§ 1–2) and
  404–823 (§§ 3–4a).
- **Tests read whole:** all 10 files in `tests/parse/`, `tests/watch/test_pyscf_parser.py`,
  `tests/test_parse_log_sidecar.py`, `test_ending_door.py`, `test_util_csv_is_time_weighted.py`,
  `test_molwatch_preview.py`, `test_siesta_fortran_overflow.py`, `test_held_atoms.py`,
  `test_presenters_rows.py`, `test_chart_table.py` and `tests/_deck.py`.
- **Tests read in part:** only the parts that use `parse`:
  - `test_template_citation_marker.py:130-189`
  - `test_one_door_reads_a_structure.py:55-98`
  - `test_structure_authority_roundtrip.py:405-440`
  - `test_a_sent_run_the_scheduler_forgot.py:60-110`
  - `watch/test_app_concurrency.py:1-80`
  - `test_siesta_vibration_e2e.py:682-706`
- **Tests not read:** about 30 other files import `parse` only in passing, among them the large
  e2e files, `support/road.py` and `support/real_runs.py`. I searched them for engine output typed
  by hand (`outcoor`, `scf:`, `E_KS`, the molwatch title, `>> End of run`) and for reads of source
  or doc text. Neither turned up.
- **Round 1:** P-D1, P-D2, P-D3 and P-U1 are not repeated here.

---

## The findings that matter most

### 1 · One run reading is held in two classes, and its parse warning in four shapes

- **Where:**
  - `frame.py:192-253`: `class Trajectory` (mutable). Its fields are `source_format, frames,
    lattice, run_state = "running", scf_converged, error_message, cause, runtime_info,
    parse_warnings`.
  - `parse/types.py:100-127`: `class TrajectoryResult(ParseResult)` (frozen), which says of
    itself *"Mirrors the `Trajectory` dataclass in `frame.py`"*. It has the same fields, but
    `run_state: str = "unknown"`.
  - `frame.py:34-60`: `class ParseWarning` (`line_no, snippet, error, category="scf"`).
  - `parse/types.py:32-39`: a second `class ParseWarning` (`source, line_no, snippet, error,
    category`).
  - `parse/engines/_helpers.py:29-69`: `wrap_trajectory` copies one shape into the other field by
    field, through `getattr(traj, "cause", None)`, `getattr(traj, "scf_converged", None)` and
    `getattr(w, "line_no", None)`.
- **Third and fourth shapes of a warning:** `siesta_reader.py:233-235` keeps a dict
  (`{"line_no", "snippet": line.rstrip()[:120], ...}`). `siesta.py:109-113` makes a
  `frame.ParseWarning` with the same `[:120]` cut. `_helpers.py:318-326` turns them back into
  dicts for the page.
- **Who builds which:**
  - Every text parser (`siesta.py:170`, `molwatch.py:117`, `pyscf.py:243`, `xyz.py:178`) builds a
    `frame.Trajectory` and wraps it at once.
  - `siesta_mdnc.py:146` builds the `TrajectoryResult` directly.
  - Only the parse package uses `frame.Trajectory` (grep), apart from one test fake (T3).
- **Owner:** `parse/types.py`. `parse.md` § 2 says *"one frozen-dataclass result hierarchy"*,
  and the result is the record every consumer reads.
- **What drifts:**
  - Add a field to `Trajectory` and set it in a parser. Unless `wrap_trajectory` copies it, it
    never reaches the result.
  - The `getattr(..., None)` copies hide the miss: they answer `None` instead of failing. `cause`
    went through exactly this road.
  - The two defaults already disagree (`"running"` and `"unknown"`).
- **Smallest fix:**
  - Have the four text parsers build `TrajectoryResult` directly, as `siesta_mdnc` does.
    `siesta.py:298-303` is the one place that changes a `Trajectory` after it is built (the MD.nc
    upgrade); it becomes a `dataclasses.replace`.
  - Have `siesta.py`'s `_warn` build `types.ParseWarning`.
  - Delete `frame.Trajectory`, `frame.ParseWarning` and `wrap_trajectory`'s field copy.
- **How sure:** read on both sides and every builder.

### 2 · Which settings a run's record carries is listed in three places, and two tests keep them in step

- **Where:**
  - **The catalogue's `citation = ["transport"]` tags** (`data/catalogue.template.toml`, e.g.
    lines 35, 62, 90). `tests/test_template_citation_marker.py:135-146` calls them *"the one
    source of what a citation takes"*.
  - **`parse/contract.py:68-90`**, three hand-kept tuples:
    `RECORDED_FIELDS = ("basis_size", "mesh_cutoff", "pao_energy_shift", "electronic_temperature",
    "xc_functional", "xc_authors", "net_charge", "spin_treatment", "unpaired_electrons")`,
    `K_MESH_RECORD_KEYS` and `SCF_RECORD_KEYS`.
  - **`parse/engines/siesta_contract.py:36-62`**, a dict literal that types the same 17 names
    again, each mapped by hand to its `FdfParams` attribute (`"pao_energy_shift":
    p.energy_shift_ry`, ...). Its comment says *"The catalogue's names (`RECORDED_FIELDS`)"*, but
    it does not read them.
- **Measured:** the catalogue's tagged set equals the three tuples less one name, `net_charge`.
  There is no name the other way.
- **The tests:** `tests/parse/test_contract.py:49-75` and `test_template_citation_marker.py:135`
  exist only to keep these lists aligned. Each guards one direction only.
- **What drifts:**
  - A setting added to `siesta_contract` that is a `SiestaConfig` name passes both tests.
  - `contract_of` then counts it as a level of theory (`contract.py:49`, *"nothing but state and
    SCF"*).
  - `contract_fields_of` never compares it.
- **Owner:** the catalogue (`engines/template.md` § 5, as the test quotes it).
- **Smallest fix:**
  - Take the set from the catalogue's `citation` tag, adding the state's keys.
  - Keep only the grouping in `contract.py` (state, SCF or k-mesh). That still needs a hand list
    unless the catalogue states a group.
  - Let `siesta_contract` walk that set through one table of catalogue name → `FdfParams`
    attribute.
  - Retire the two alignment tests.
- **How sure:** read on all three sides; the set difference was run.

### 3 · `FdfParams` carries four fields nothing reads, one of them restating SIESTA's XC default differently from the catalogue

- **Where:** `parse/fdf.py:212-217` declares `xc`, `z_len_ang`, `solution_method` and `n_atoms`.
  They are set at `:416-418`, `:448`, `:424-425` and `:454`.
- **Measured:** a grep over `molbuilder/` and `tests/` finds no reader of any of the four.
- **The XC default:** `fdf.py:416-418` types SIESTA's default itself:
  ```python
  func = (sc.get("xcfunctional") or ["LDA"])[0]
  auth = (sc.get("xcauthors") or ["CA"])[0]
  ```
  The catalogue's `engine_default` for `xc_authors` (`catalogue.template.toml:158`) is `"PZ"`,
  cited to `read_xc_info.F:105`. So the two sources give different words for the same engine
  default (CA and PZ name one LDA in SIESTA). Run: `parse_fdf_params("SystemLabel x\n").xc` →
  `'LDA/CA'`.
- **Owner:** the catalogue's `engine_default`, read through `template.default_of`.
- **What drifts:** the first reader to take up `p.xc` gets a default that disagrees with the
  catalogue's. The other three fields can drift from their meaning unseen.
- **Smallest fix:** delete the four fields and the lines that fill them.
- **How sure:** read and grepped; the default was run.

### 4 · Two rules for "which convergence targets are this run's"

- **Where:**
  - `molwatch_reader.py:306-313`, `MolwatchReader.targets()`: the stage's nested block if the
    reader was given a stage, else only the flat keys.
  - `trajectory_log/keys.py:55-74`, `targets_of`: *"The one rule, asked by the citation reader
    (`parse/contract.py`) and by the page's rows"*. One nested stage is the run's; several give
    `None`; a flat block is the run's.
- **Measured:**
  - A log whose header is `# convergence.01_coarse.max_force_tol_eV_per_A: 0.04`, read with no
    stage, gives `targets() == {}`, while `targets_of` gives `{'max_force_tol_eV_per_A': 0.04}`.
  - With stage `02_fine`, `targets()` gives `{}`.
- **Today:** the monitor always passes its stage, the same token the writer nests by
  (`emitter._convergence_lines`), so the two agree on the live road.
- **What drifts:** the monitor's report (`now()["targets"]`) and the page and record
  (`targets_of`) answer the same question with two codes. A header shape changed for one is not
  changed for the other.
- **Smallest fix:** `targets()` asks `targets_of` with its stage. That is either a `stage=`
  argument on `targets_of`, or the stage's block picked first and handed to it.
- **How sure:** read on both sides; run.

### 5 · Three readers of a Fortran number in the SIESTA family, each with its own rule

- **Where:**
  - `siesta_grammar.py:222-235`, `fortran_float`: asterisks become NaN; a `D` exponent is not
    handled. Its comment calls it *"The row's ONE value reader"*.
  - `siesta_fc.py:74,79`: `float(fields[1].replace("D", "E"))`. A capital `D` only, and no
    asterisks.
  - `siesta_epsimg_reader.py:94-98`, `_number`: `token.replace("D", "E").replace("d", "e")`. No
    asterisks.
- **Measured:** `fortran_float("1.0D-03")` and `("1.0d-03")` both raise `ValueError`.
- **Owner:** `siesta_grammar`, the family's one table. `epsimg_reader` already imports it, and
  `siesta_fc` travels in the same bundle (`mb_vibration.pyz`).
- **What drifts:**
  - An `.FC` written with a lower-case `d` (some compilers) is refused while an `.EPSIMG` reads.
  - A field-overflow `****` in an `.FC` row is refused while the `.out` reads it as NaN.
  - I did not check whether SIESTA ever prints `D` in its `.out`.
- **Smallest fix:** one `fortran_float` in the grammar that also takes `D`/`d`, used by all
  three.
- **How sure:** read on all three sides; the grammar's answer was run.

### 6 · One SCF row reads with two key sets, depending on whether its names row was seen

- **Where:**
  - `siesta_grammar.py:467-489`: `cycle_from_header` maps columns by the names row, through
    `SCF_COLUMN_KEYS` (`:422-447`). That keeps `ef`, `ef_up` and `ef_dn`.
  - `siesta_grammar.py:492-520`: `cycle_positional` hard-codes the positions again (`vals[1]`,
    `vals[3]`, `vals[5]` or `vals[6]`) and returns `cycle`, `energy`, `dDmax` and `dHmax` only.
- **Measured:** the same 6 values give `{..., 'ef': -4.5, ...}` through the header path and no
  `ef` through the positional path.
- **What drifts:**
  - An output with no `iscf` names row before its first rows (cut, or a build that prints none)
    loses the Fermi level.
  - `fermi_levels` (`:1429`) then answers `None`; for a transport lead, *"the one number that
    matters"* (`:428-436`).
  - The column order lives in two places: the key map, and the hand-typed positions.
- **Owner:** `SCF_COLUMN_KEYS`, which already holds the column names in `write_subs.F` order.
- **Smallest fix:** let the fallback build a default names row (the 6- and 7-value layouts as
  name lists) and map it through `cycle_from_header`.
- **How sure:** read and run.

### 7 · The series the `.MD.nc` upgrade carries are listed twice

- **Where:**
  - `siesta_mdnc.py:280-281` names them: `("eks_eV", eks), ("temperature_K", temp),
    ("pressure_kBar", psol), ("volume_A3", volume)`. `parse` adds the `mdnc_` prefix (`:162`).
  - `siesta_mdnc.py:430-431`, in `upgrade_frames`, lists them again by hand:
    `("mdnc_eks_eV", "mdnc_temperature_K", "mdnc_pressure_kBar", "mdnc_volume_A3")`.
- **What drifts:** a series added to `_frames_from` is in the `.MD.nc` result but silently
  missing from the `.out`'s `runtime_info`. That is the only place a viewer sees it.
- **Smallest fix:** one module tuple of series names that both read, or copy every `mdnc_` key
  except the reader's own bookkeeping.
- **How sure:** read.

### 8 · The run record lists its session-log and launch keys twice

- **Where (session log):**
  - `parse/dirs/record.py:280-282` (`_wrapper`) picks `("hostname", "user", "cwd", "conda_env",
    "python", "node_phys_cores", "node_sockets", "node_cores_per_socket")`, and `:283` picks
    `("ranks_asked", "threads")`.
  - The `wrapper-log` contributor at `:447-456` declares the same keys again, each with
    `computation.host.` or `computation.launch.` in front.
- **Where (launch):** the same split between `:360-361` (`keys = ["mode", "command", "job_id",
  "launched_at", "placed_on", "sequence"]`) and the `run-json` contributor at `:460-467`.
- **What drifts:** a key added to one list and not the other is cut without a word by `_declared`
  (`:510-524`), or declared and never filled. `wrapper_log.read_wrapper_log` already defines
  which keys exist.
- **Smallest fix:** keep one tuple per row, and build the declared fields from it, as `_TIMING`
  (`:428-432`) already does for the timing row.
- **How sure:** read.

### 9 · The sidecar's text reader lists the envelope keys its owner already declares

- **Where:**
  - `parse/sidecars/molstruct.py:130-143`, in `load_text`, passes each envelope key by hand into
    `to_dict` (`n_atoms_total=..., n_frames_total=..., geometry_hash=..., created_by=...,
    created_at=..., atom_index_map=..., schema_version=sv`).
  - `sidecars/molstruct.py:157-158` declares `ENVELOPE_KEYS` with exactly these.
  - `unread_keys` (`:173-195`) accepts a key because it is in `ENVELOPE_KEYS`.
- **What drifts:** a new envelope key added to `ENVELOPE_KEYS` and to `to_dict` passes the
  stray-key gate, but `load_text` drops it. The file reads back without it and nothing says so:
  the silent loss the gate exists to prevent.
- **Owner:** `sidecars/molstruct.py` (`ENVELOPE_KEYS`).
- **Smallest fix:** spread the envelope keys from `ENVELOPE_KEYS` as the fields are spread from
  `FILE_FIELDS`. That needs `to_dict` to take them by those names, which it nearly does.
- **How sure:** read on both sides. `to_dict`'s signature was not read in full.

### 10 · The "ended badly" pair of states is typed four times

- **Where:**
  - `siesta_reader.py:394`: `if self.run_state not in ("stopped", "out_of_memory")`
  - `siesta_reader.py:1538`: `if (self.run_state in ("stopped", "out_of_memory")`
  - `parse/dirs/job.py:664`: `if active_state in ("out_of_memory", "stopped")`
  - `web/blueprints/watch.py:342`, the same, outside this slice.
- **What is declared:** only `_run_ending.CONCLUDED = ("ended", "stopped", "out_of_memory")`
  (`:68`), which says *"P-S4: one door, or the copies drift silently"*.
- **What drifts:** a new fatal state added to `FATAL_MARKERS` (`siesta_grammar.py:609`), say
  `"killed"`, is taken as concluded by one copy, not a failure by another, and a reading the abort
  outranks by the third.
- **Owner:** the § 2b vocabulary. It has to sit below `siesta_reader`, since `_run_ending` imports
  the reader; `siesta_grammar` holds the states already, in `FATAL_MARKERS`.
- **Smallest fix:** derive a `FAILED` tuple once from `FATAL_MARKERS`' states in the grammar, and
  have all four ask it, with `CONCLUDED` built from it.
- **How sure:** read.

### 11 · TBtrans's channel names and their grouping are typed apart from their table

- **Owner:** `runfiles.py:154-155`, `TBT_CHANNEL_TAGS = {"unpolarized": ..., "up": ...,
  "down": ...}`, which the grammar re-exports as `TBT_CHANNELS`.
- **Where (names typed again):**
  - `siesta_grammar.py:1406-1410`, `tbt_channels`, returns the literals `("up", "down")`,
    `("up",)`, `("down",)` and `("unpolarized",)`.
  - `tbtrans_nc.py:67` and `:146` look up `["unpolarized"]` by name.
- **Where (meaning taken from the table's order):**
  - `siesta_grammar.py:1425-1426` builds `_OWN_FERMI` from `TBT_CHANNELS[1:]`.
  - `siesta_epsimg_reader.py:64-65` builds `CHANNELS = {1: _NAMES[:1], 2: _NAMES[1:]}`.
- **Where (the grouping again, plausible):** `transport/conductance.py:25-26`, `SPIN_CHANNELS`,
  outside this slice and read only by grep.
- **What is guarded:** only `tbtrans.CHANNEL_WORDS` has a load-time guard (`tbtrans.py:48-52`).
- **What drifts:** reorder or rename one entry of the table and:
  - `tbt_channels` names passes that `tbt_channel_of` never matches;
  - an unpolarized `.EPSIMG` is called `up`;
  - `fermi_levels` reads the wrong column.
- **Smallest fix:** let the table state the grouping, `{"unpolarized": ("unpolarized",),
  "polarized": ("up", "down")}`. Then `tbt_channels`, `_OWN_FERMI`, `CHANNELS` and conductance
  read it instead of slicing or spelling.
- **How sure:** read in this slice; the conductance side only grepped.

### 12 · The listener's reports file is recognised by a folder name and suffix spelled in the parser

- **Where:** `parse/instruments/reports.py:68`:
  `return (p.name.endswith(".jsonl") and p.parent.name == "reports" ...)`
- **Owner:** the path's home is `config_dir.py:297` (`return state_dir() / "reports"`) and
  `web/blueprints/notify.py:211` (`reports_dir() / f"{user}.jsonl"`).
- **What drifts:** move or rename the folder in `config_dir`, and the registry stops claiming the
  file. The listener still writes it and the tab still reads it, through `read_reports` directly,
  so nothing shows the break.
- **Smallest fix:** `can_parse` asks `config_dir.reports_dir()` (path equality), or a suffix
  constant both sides import.
- **How sure:** read.

### 13 · The progress log's SCF row: the writer's column order and the reader's positions are two sources

- **Where:**
  - `trajectory_log/emitter.py:224-238` writes the names row
    (`#  cycle energy(eV) delta_E(eV) gnorm(eV) ddm wall_time(s)`) and the row by position.
  - `parse/engines/molwatch_grammar.py:343-364`, `scf_history_row`, reads `parts[0]` to
    `parts[5]` by position.
  - `molwatch_grammar.SCF_COLUMNS` (`:120-134`) lists the columns a third time, for the data file.
  - `OLD_GNORM_HEADER = "gnorm(eV/Ang)"` (`:339`) is a fragment of the writer's names row, spelled
    on the reader side.
- **What the rule says:** `trajectory_log/keys.py` says *"the writer renders these and the
  reader's patterns are built on them, so a word reworded is reworded for its reader too"*. It
  holds every other word of the log, but not this column order.
- **What drifts:** a column inserted before `wall_time` is read as the cycle's epoch, and the
  timing figures (`scf_timing_rows.progress_log_timing_metrics`) go wrong with no error.
- **Smallest fix:** a column tuple in `keys`, which the emitter's names row, its row writer and
  `scf_history_row` all build from.
- **How sure:** read on both sides.

### 14 · Two rules written twice inside the SIESTA reading pass

- **The step's energy (NEGF first, else the last finite SCF energy):**
  - `siesta_reader.py:342-355` (`_commit`)
  - `siesta_reader.py:1522-1532` (`finish`, `live_scf`, whose comment says *"as `_commit`
    rules"*)
- **"The row in flight" (the current SCF's last row, else the last committed step's):**
  - `:917-919` (`_where_words`, *"by `now()`'s own rule"*)
  - `:1394-1396` (`now`)
- **What drifts:**
  - A change to which row is "finite" or "the device's" changes one copy only. The live chart's
    energy and the committed step's energy then disagree for the same rows.
  - The words and the numbers the monitor sends can name different rows.
- **Smallest fix:** two small methods on the reader, `_energy_of(rows)` and `_row_in_flight()`,
  each called from both places.
- **How sure:** read.

---

## Lower — real, small

- **15 · Registry leftovers.**
  - `registry.py:228-231`: `_detect_one(path, pool, kind_label)`. Its one caller always passes
    `_FILE_PARSERS, "file"`, a remnant of the retired directory parsers.
  - `:266`: `getattr(c, "footgun_hint_for", None)` is a dead branch, since every `FileParser` has
    it.
  - `register`'s docstring (`:190-193`) says *"Order matters: the dispatcher tries registered
    parsers in insertion order"*. `detect` (`:207-211`) and `engines/__init__.py:16-17` say order
    confers nothing. The cost note at `engines/__init__.py:32-35` is untrue too, because `detect`
    asks every parser.
  - Fix: drop the two parameters and the `getattr`, and correct the two comments.
- **16 · A record's schema lives in its reader.**
  - `parse/sidecars/transport.py:29` (`TRANSPORT_RESULT_SCHEMA`) and `parse/sidecars/fc_sweep.py:33`
    (`SWEEP_SCHEMA`) are imported by their writers (`transport/record.py:987`,
    `spectra/displacement_sweep.py:132`).
  - Their siblings keep the schema with the writer: `runpoints.SCHEMA`,
    `runrecord.NOTIFIED_SCHEMA`.
  - There is one source today, so nothing drifts. The writer owns the shape, though, so a version
    bump means editing the parse package.
- **17 · A derived constant outside `constants`.**
  - `siesta_fc.py:41`: `EV_PER_ANG2_TO_HARTREE_PER_BOHR2 = (BOHR_ANGSTROM ** 2) / HARTREE_EV`.
  - `spectra/displacement_sweep.py:141` imports it from the reader. A named value two modules use
    belongs in `constants` (architecture § 3, *A value*).
- **18 · Small helpers written twice.**
  - The `isoformat(timespec="milliseconds").replace("+00:00", "Z")` formatter is written at
    `types.py:87-89` and again as `job._iso_z` (`job.py:80-83`). `tbtrans.py:249` imports the
    private one.
  - `tbtrans_nc._version` (`:176-178`, `(size, mtime_ns)`) beside `_run_ending.version_of`
    (`:473-480`, `(path, size, mtime_ns)`).
- **19 · A second fail-soft rule.**
  - `parse/dirs/setup.py:161-166` wraps the fdf log's parse in `except Exception: pass`.
  - The record's one rule (`record.py:575-580`, and every docstring around it) is *"a reporter
    degrades on `OSError`/`ValueError`/`ParseError`; a defect in a reader is not caught"*.
    `setup.py` hides a reader defect the rest of the record would surface.
- **20 · The SIESTA sniffer's vocabulary is split.**
  - `siesta.py:218`: `_PREFIX_MARKERS = ("siesta:", "redata:")` is typed in the parser.
  - The strong markers are the grammar's (`SNIFF_MARKERS`, `siesta_grammar.py:555-562`), which
    says it holds *"Lines only a SIESTA-family output prints … for the format sniffer"*.

## Tests (point 5)

- **T1 · Two tests pin what a table row already pins, on files typed by hand.**
  - `tests/parse/test_coords.py:122-133` (`test_pyscf_geom_claims_optimized_xyz`,
    `..._doesnt_claim_plain_xyz`) asks `PySCFGeomFileParser.can_parse` directly. The input is
    `"0\n\n"`, a zero-atom XYZ that no writer makes.
  - `tests/parse/test_registry.py:99-136` (`_XYZ_READERS`, the rows `H2_optimized.xyz` and
    `H2.xyz`) already pins both, through `detect` and the codec's writer.
  - Fix: retire the two.
- **T2 · `tests/watch/test_pyscf_parser.py` repeats the registry's tests with XYZ typed by hand.**
  - `:19` (garbage text refused) repeats `test_registry.py:25`.
  - `:26-33` (a plain XYZ goes to `xyz`) repeats the `H2.xyz` row. Its XYZ is typed by hand where
    `Structure.to_xyz()` is the writer.
  - Its subject is `parse/engines/xyz.py`, yet it sits under `tests/watch/` and is named after
    PySCF.
  - Only `:36` (a count header over CSV) and `:48` (zero atoms) pin something new. They could be
    rows of one table in `tests/parse/`.
- **T3 · A fake parser returns the shape no real parser returns.**
  - `tests/watch/test_app_concurrency.py:46`: `SlowParser.parse` returns
    `frame.Trajectory(...)`.
  - Every registered `FileParser` returns a `TrajectoryResult`. It passes only because the two
    shapes duck-type alike, which is finding 1.
  - Build the fake's result through `ParseResult.envelope` and `TrajectoryResult`. With finding 1
    fixed, this is forced anyway.
- **T4 · Two tests exist only to keep hand lists aligned.**
  - `tests/parse/test_contract.py:49-75` and `tests/test_template_citation_marker.py:135-146`.
  - Each guards one direction of finding 2's lists. They retire with the derivation.
- **Also noted (placement, not a duplicate):** `tests/test_siesta_fortran_overflow.py:103-179`
  (`TestJsonSafe`) tests `persist.json_safe` inside a parse regression file.

---

## Totals

There are **24 findings**: 14 main ones (1–14), 6 smaller ones (15–20) and 4 in tests (T1–T4).

**The five that matter most:**
1. **Two classes for one run reading.** `frame.Trajectory` and `parse.types.TrajectoryResult`,
   with two `ParseWarning` classes, are joined by a `getattr` copy that silently drops any field it
   was not taught.
2. **The cited settings in three lists.** The catalogue's tags, `contract.py`'s three tuples and
   `siesta_contract`'s dict all list them, and two tests exist only to keep them aligned.
3. **Four dead fields in `FdfParams`.** One of them restates SIESTA's XC default as `LDA/CA`, where
   the catalogue's cited default is `PZ`.
4. **Two "targets of this run" rules.** `MolwatchReader.targets()` and `keys.targets_of` answer one
   question in two ways. They disagree on a staged header read without a stage (measured).
5. **Three readers of a Fortran number.** The grammar's `fortran_float`, the `.FC` reader and the
   `.EPSIMG` reader each handle `D` exponents and `****` differently.
