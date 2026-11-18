# F12 review, round 2 — slice R2-4: the template and the engines

Checkout `the checkout` at 2.77.2. Read-only throughout: nothing under
the checkout, `~/.config/molbuilder`, `~/.local/state/molbuilder` or `projects/`
was written. Three read-only measurements were run with
`python -B` from `/tmp` (no bytecode, no files); each is quoted where it is used.

**Result: 24 product findings (P1–P24) and 7 test findings (T1–T7) — 31 in all.**
Round 1's D1, D3, D8 and D9 are not repeated. Four findings here name a
duplicate that causes one of them (P1 → D9, P5 → D3, P8 → D1, P11 → the
PySCF twin of D8).

## What was read

**Source, read whole:** `template.py`, `template_file.py`, `kind_table.py`,
`engine_table.py`, `resolve.py`, `pseudos.py`, `electronic_state.py`,
`kmesh.py`, `units.py`, `constants.py`; `siesta/` (`__init__`, `stages`,
`deck_lines`, `seam`, `engine_input`, `vibration_deck`, `vibration_phases`,
`makov_payne`, `layout`, `input`); `pyscf/` (`__init__`, `end_lines`,
`stages`, `seam`, `ecp`, `scf_setup`, `script_input`, `relax_policy`,
`layout`, `input`, `vibration_deck`, `vibration_emitters`); `optimization/`;
`bench/`; `validation/` (`__init__`, `metadata`, `stages`, `task`,
`chemistry`, `geometry`, `sidecar`, `builders`, `siesta`, `pyscf`,
`spectra`, `transport`).

**Read outside the slice, only to confirm the other side of a finding:**
`docs/architecture.md`, `docs/execution/architecture.md` § 3.2–3.3,
`docs/process/conventions.md`, `docs/process/testing.md` §§ 2–3a,
`scheduler/quantities.py` 280–322, `jobset/model.py` 62–67 and 168–177,
`jobset/prep_inputs.py` 170–215, 500–530 and 670–705,
`spectra/siesta_rung.py` 125–150, `spectra/siesta_vibration.py` 845–880,
`spectra/vibrational_analysis.py` 330–370, `spectra/datafile.py` 150–175,
`transport/energies.py` 100–135, `parse/fdf.py` 35–60 and 285–305,
`identity.py` 90–111, `runpoints.py` 28–62, `config/pyscf.py` 248–272,
`cell.py` 20–45 and 900–925, `script_emit.py` 1105–1160, and the matching
rows of `data/catalogue.template.toml`, `data/kind_table.toml` and
`siesta/warm-files.toml`.

**Tests.** About 110 test files import or drive these modules (47,672 lines),
which is too much to read whole. Here is what was read:
- **Whole:** `test_doc_claims.py`, `test_catalogue_refs.py`,
  `test_pyscf_reads_the_catalogue.py`, `test_catalogue_agreement.py`,
  `test_template_roundtrip.py`, `test_makov_payne.py`, `test_restart_group.py`,
  `tests/_deck.py`, `test_deck_helpers.py`.
- **In part:**
  - `test_template_citation_marker.py` 1–130
  - `test_template_role_and_stages.py` 1–60
  - `test_the_setting_door.py` 1–120
  - `test_stage_resolution.py` 1–130
  - `test_resolve.py` 1–100, plus its list of tests
  - `test_kind_table.py` 136–165, 355–385, 455–482
  - `test_template_declarations.py` 30–110, plus its list of tests
  - `test_bench_result.py` (its list of tests)
  - `test_pyscf_periodicity_check.py` 120–205
  - `test_units.py` 1–80
  - `test_transport_decks_e2e.py` 255–275
- **Searched with grep across every slice test, not read:** the remaining
  ~90 files, mostly e2e and web. They were searched for source or doc reads,
  constants typed into tests, per-file copies of fixtures, and the table-swap
  prelude.

---

## A. One door per fact — a fact computed here that a door already answers

### P1. The solvent's name is normalised in seven places, and the gate and the writer disagree (the cause of round 1's D9)

- **The places:**
  - `pyscf/scf_setup.py:176` (the writer; `SOLVENTS` is at :150–160):
    `solvent = str(getattr(cfg, "solvent", "") or "").lower()`
  - `pyscf/input.py:269–271` (the optimization header, raw):
    `eps={_SOLVENTS.get(cfg.solvent, '?')}`
  - `pyscf/vibration_emitters.py:157`, the vibration header:
    `.lower()`
  - `pyscf/vibration_emitters.py:1483`, the Methods text:
    `.lower()`
  - `validation/pyscf.py:376`:
    `_solv = str(getattr(cfg, "solvent", "") or "").strip().lower()`
  - `validation/pyscf.py:303`: `.strip()`; `validation/spectra.py:357`:
    `.lower()`
- **Owner:** `scf_setup`, whose docstring says "one home both decks read" for
  the table and "ONE spelling for both decks" for the lines.
- **What goes wrong — measured:** with `PySCFConfig(solvent=' water')`,
  `validate(...)` reports **no** solvent finding, because it strips the name.
  `emit_solvent_lines(cfg)` then raises
  `ValueError unknown solvent ' water'`, because it does not strip. That is
  exactly the "stack trace at prep" that `validation/pyscf.py:385` says the
  gate exists to pre-empt. Round 1's D9 (the header printing `eps=?`) is the
  same split seen from another place.
- **Smallest fix:** one door in `scf_setup`, for example
  `solvent_of(cfg) -> (name, eps) | None`, normalised once. All seven readers
  ask it.
- **Certainty:** read on every side, and measured.

### P2. The vibration's displacement-amplitude range is typed into the kind's check, and fires twice

- **The two places:**
  - `validation/spectra.py:63–88` holds its own window:
    `if amp < 0.02: … "smaller than the accepted range (0.02-0.20 Å)"` and
    `elif amp > 0.20:`.
  - The catalogue states the same window, `displacement_amplitude_ang`
    `range = [0.02, 0.2]`. `validation/metadata.validate_config_metadata`
    already warns from that range, on the same `where`.
- **Owner:** the catalogue's `range` (`template.md` § 5; "one answer for every
  place a range is warned", `metadata.outside_range`).
- **What goes wrong — measured:** with `es_mode_selection='all'` and
  `displacement_amplitude_ang=0.01`, `validate(...)` returns **two** warnings
  on `config.displacement_amplitude_ang`:
  1. "Displacement amplitude = 0.01 Å is outside the recommended range
     [0.02, 0.2] Å"
  2. "Displacement amplitude 0.01 Å is smaller than the accepted range
     (0.02-0.20 Å)…"

  `validation/siesta.py:337–344` explicitly gates its own check to avoid this
  double fire for `mesh_cutoff`; this check does not. If the catalogue range
  changes, the two warnings disagree.
- **Smallest fix:** drop the two range branches and keep the
  amplitude-against-tolerance advisory (lines 90–108). Where the advice needs
  the bounds, read them from `item.range`. The same message also types
  "default scf_conv_tol=1e-9", which is the catalogue's default.
- **Certainty:** read both sides, and measured.

### P3. SIESTA's BENCH-MARKS fields are a hand-kept copy of catalogue facts

- **The copy:** `siesta/input.py:374–390`:
  ```
  _sc.BenchField("BlockSize", "BlockSize", "pow2"),
  _sc.BenchField("MaxSCFIterations", "MaxSCFIterations", "int"),
  _sc.BenchField("MD.Steps", "MD.Steps", "int"),
  _sc.BenchField("MeshCutoff", "MeshCutoff", "float", None, "Ry"),
  _sc.BenchField("Diag.Algorithm", "Diag.Algorithm", "enum"),
  ```
  The defaults are keyed by the same typed anchors at `:326–339`.
- **The original:** the catalogue states each of these facts for
  `block_size`, `max_scf_iter`, `relax_steps` (`expands = ["MD.Steps"]`),
  `mesh_cutoff` (`unit = "Ry"`) and `diag_algorithm`
  (`choices = ["ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE"]`).
- **Owner:** the catalogue. `job-contracts.md` § 3.3, quoted by the test:
  "BENCH-MARKS and the template are emitted from ONE source".
- **What goes wrong:**
  - The `Diag.Algorithm` field carries no `choices`, though `BenchField` has
    the slot. A tool reading the record cannot validate a value — the choices
    are already lost.
  - Rename an anchor or a unit in the catalogue, and BENCH-MARKS keeps the old
    word.
  - The only guard, `tests/test_template_declarations.py:47–95`, checks `type`
    alone. It never checks the unit or the choices, and its docstring admits
    "SIESTA's BENCH_FIELDS **is** hand-maintained, so the rule is an intention
    there rather than a mechanism".
- **Smallest fix:** build `BENCH_FIELDS` from the five item names through the
  parameter door (anchor or `expands`, type, unit, choices). Keep only the
  `pow2` narrowing as data, then retire the agreement test.
- **Certainty:** read both sides.

### P4. PySCF's line door hand-types the attribute paths that the catalogue's anchors hold, while the record reads the anchors

- **The hand-typed paths:** `pyscf/layout.py:212–238` (`_scf`) writes
  `f"mf.conv_tol  = …"`, `"mf.conv_tol_grad = …"`, `"mf.max_cycle = …"`,
  `'mf.init_guess = …'`, `"mf.level_shift = …"`, `"mf.diis_space = …"` and
  `"mf.damp = …"`. `:171–198` (`_theory`) writes `mf.xc`, `mf.grids.level`
  and `mf.disp`.
- **The original:** each one is the item's catalogue `anchor` (checked:
  `scf_conv_tol → "mf.conv_tol"`, … `damp → "mf.damp"`, `functional →
  "mf.xc"`).
- **The second reader:** `readback`, at `:328–347`, reads the effective value
  back from `param.writes`, the anchor.
- **What the module claims:** its own header (`:34–36`) says "the engine
  keyword each one writes is the catalogue's `anchor` and is never spelled
  twice".
- **What goes wrong:** change one anchor in the catalogue. The record's third
  column then reads an attribute the deck no longer sets, and shows
  `(absent)` or another value. The comparison that exists to show "the
  setting did not take" then blames a setting that did take.
- **Smallest fix:** write `f"{param.writes[0]} = {formatted}"` and keep only
  the per-item value format (`.1e`, quotes) in the door. The geometry half
  already reads its keyword from the anchor (`geom_keyword`).
- **Certainty:** read both sides.

### P5. The calculation's identity is a SIESTA-only list in shared code, beside the engine record that names it (the cause of round 1's D3)

- **The list:** `template.py:2190`:
  `IDENTITY_ITEMS = ("system_label", "species_order", "psml_lib")`, read by
  `why_shared` (`:2222–2227`, whose sentence names SIESTA's `.XV` and species
  table) and by the catalogue's `shared` check (`:1206–1210`).
- **The second home:** `engine_table.Engine.identity_field` (`system_label`,
  `job_name`) is the record that says which setting names a calculation's
  files.
- **The same pattern next to it:** `template.py:210–237` (`SETUP_WORDS`) keys
  per-engine sentences on `{"system_label", "psml_lib"}` and `{"job_name"}`.
  It writes SIESTA's and PySCF's file names into shared code
  (`deck_role('siesta')`, `SIESTA_XV`, `ROLE_PYSCF_LOG`).
- **Owner:** the engine table. `execution/architecture.md` § 3.3 says "a new
  engine is a record …"; § 3.3 A18 says shared code names no engine.
- **What goes wrong:**
  - PySCF's identity field is not shared, so a per-stage `job_name` override
    passes (round 1, D3).
  - A third engine needs `template.py` edited: `setup_words` refuses, saying
    "add one to template.SETUP_WORDS".
- **Smallest fix:**
  - Take each engine's identity from its record (`identity_field`, plus the
    rows that declare `shared` for it).
  - Compose the Setup line from the record's words and `deck_role(engine)`,
    not from a table keyed on field names.
- **Certainty:** read both sides.

### P6. The gate repeats the catalogue's range floor and default as literals

- **The two literals:**
  - `validation/siesta.py:345`: `if 100.0 <= mc_val < 150.0:`. Its comment
    (`:337–344`) says the 100 is "the dataclass metadata-range check (lower
    bound 100)". The catalogue holds it: `mesh_cutoff`
    `range = [100.0, 1000.0]`.
  - `validation/siesta.py:621`: `_tol_default = 1e-4`, the default the warning
    judges "away from its default". The catalogue holds it:
    `dm_energy_tolerance` `default = 0.0001`, plus its `engine_default`.
- **Owner:** the catalogue (`item.range`, `template.default_of`).
- **What goes wrong:**
  - Lower the catalogue's floor to 50, and 50–99 Ry draws neither warning — the
    double-fire the comment guards against turns into a gap.
  - Change the default, or ship a preset with another tolerance, and the
    warning fires at the shipped value.
  - Measured today: no preset sets `dm_energy_tolerance`, so the second has no
    live harm yet.
- **Smallest fix:** read `one(catalogue(), "mesh_cutoff").range[0]` and
  `default_of(item, kind)`.
- **Certainty:** read both sides.

### P7. The `gres` dialect is spelled by hand beside its door

- **Where:** `resolve.py:1139–1140`:
  `out["gres"] = f"gpu:{val}"  # the count, as the scheduler spells it`.
- **The door:** `scheduler/quantities.canonical_gres` (`:309–319`) is "a
  stated GPU ask -> what reaches sbatch: gpu:<count>". The architecture puts
  every dialect of a quantity in `quantities.py`.
- **Why there is no harm today:** `Resources.__post_init__`
  (`jobset/model.py:177`) runs the value through `canonical_gres` again, and
  `gpu_count`'s `above = 0` keeps "gpu:0" out.
- **The same spelling outside the slice:** `jobset/prep_inputs.py:203` and
  `jobset/model.py:268`.
- **Smallest fix:** call `canonical_gres(str(val))`.
- **Certainty:** read both sides.

### P8. The XC family is stated twice, and the author vocabulary is copied (the duplicate behind round 1's D1)

- **The copy:** `pseudos.py:608–612`:
  ```
  _XC_FAMILIES = {"GGA": ("pbe", …), "LDA": ("ca", "pz", "pw"),
                  "VDW": ("drsll", "lmkll")}
  ```
  `validation/siesta.py:131–132` and `:175–177` take the run's family from
  it: `expected_family = expected_xc_family(xc_authors)`.
- **The original:**
  - The config states the family itself: catalogue `xc_functional`,
    `choices = ["LDA", "GGA", "VDW"]`, which the deck writes as
    `XC.functional`.
  - The author list is the catalogue's `xc_authors` choices, spelled again
    here in lower case.
- **Owner:** the catalogue.
- **What goes wrong:**
  - With `xc_functional = "LDA"` and `xc_authors = "PBE"`, the pseudopotential
    check judges against GGA, not against the family the deck writes.
  - An author added to the catalogue's choices gets `None` from
    `expected_xc_family`, so its family is silently never checked.
- **Smallest fix:**
  - The family the check expects is `cfg.xc_functional`.
  - Where an author → family pairing is needed, declare it once, on the
    catalogue's `xc_authors` choices.
- **Certainty:** read both sides. D1's VDW row is round 1's finding and is not
  restated here.

### P9. The benchmark grid's axes are listed by hand again in prep, and paired by hand with their letters

- **The owner:** `template.py:2672`:
  `GRID_AXES = ("mpi_np", "omp_threads", "gpu_count")`.
- **The hand copy:** `jobset/prep_inputs.py:524–525`:
  `unstated = [name for name in ("mpi_np", "omp_threads", *(("gpu_count",) …`.
- **The second vocabulary:** `resolve.py:992`:
  `MACHINE_AXES = ("G", "N", "C")`. Its pairing with the catalogue names is
  hand-written in `prep_inputs._cell_resources` (`:196–203`:
  `{"mpi_np": point["N"], "cpus_per_task": point["C"]}`, `gres` from `G`).
- **What goes wrong:** add a grid axis to `GRID_AXES` (a memory cap, say). The
  setting door then lets a benchmark row name it, but prep never requires it to
  be stated and no letter carries it into a trial.
- **Smallest fix:** prep asks `GRID_AXES` (filtered by the GPU family), and the
  letter of each axis is declared beside it.
- **Certainty:** read both sides.

### P10. The SIESTA vibration's phase roles are a second, unchecked list of the kind table's roles

- **The two lists:**
  - `siesta/vibration_phases.py:113–128` hands each phase
    `role="relaxation"`, `"force_constants"` and `"optical"`.
  - `data/kind_table.toml:126–130` declares the same three as `[vibration]`
    `roles`.
- **The reader:** `spectra/siesta_rung.py:135–141` (`role_at`) picks the items
  each phase's deck renders by that word, through `template.reads(role=…)`.
- **What is and is not checked:** the catalogue's `stages` are checked against
  the kind table when it loads. `PHASES` is checked by nothing.
- **What goes wrong:** rename a role in the kind table (and in the catalogue's
  `stages`), and the relax phase's role no longer meets
  `restart`'s `stages = ["relaxation"]`. Its deck then silently writes no
  restart group.
- **Smallest fix:** check `PHASES`' roles against
  `kind_table.kind("vibration").role_names` where the role is asked (prep),
  or take each phase's role from the kind's role order.
- **Certainty:** read both sides. The impact is traced, not run.

### P11. PySCF's troubleshooting block states catalogue defaults as facts about this deck (the PySCF twin of round 1's D8)

- **Where:** `pyscf/input.py:1680`:
  `"#   * mf.diis_space = 12            (default 8)"`; `:1693`:
  `"#   * cfg.basis = 'def2-SVP' (already)"`; `:1694`:
  `"#   * cfg.density_fit = True (already)"`.
- **The source:** the catalogue holds `diis_space` default 8, `basis` default
  def2-SVP and `density_fit` default true.
- **What goes wrong:** with `basis = "def2-TZVP"` or density fitting off, the
  deck still says "(already)". The block is written whatever the config says.
- **Smallest fix:** compose those lines from `cfg` and
  `default_of(item, kind)`, or say them without the claim.
- **Certainty:** read both sides.

### P12. Smaller copies of a door's answer

- **The four state items, listed by hand:** `pyscf/input.py:833–836`:
  `_resolved = {"net_charge": …, "spin_treatment": …,
  "unpaired_electrons": …, "method": …}`. This is `template.STATE_ITEMS`
  (`template.py:2145`) listed again. A fifth state item would be recorded
  unresolved.
- **`is_member` written again inline:** `electronic_state.py:253–254`:
  `any(type(c) is type(m) and c == m for m in …choices)` is
  `template.is_member` (`template.py:1921–1929`), the one typed-membership
  test.
- **The catalogue's items by name, built by hand:**
  - `{it.name: it for it in select(catalogue(), engine=…)}` appears at
    `resolve.py:525`, `resolve.py:1081`, `template.py:3228` and
    `validation/task.py:411`, plus `validation/task.py:213–217`
    (`_items_of`).
  - `template.items_of` is "the catalogue's items for a config … the one
    home".
  - `kmesh.py:69–71` and `template.py:2205/2214/2618` each scan for one item
    instead of asking `one(...)`.
- **The set of launch-shape fields, worked out twice:** `resolve.py:510–511`
  (`machine_facts = all fields - template_fields`) and `resolve.py:95–108`
  (`_emitter_fields`) compute the same set from the same flag in two ways.
- **Smallest fix:** each asks the door it copies. For the third, add an
  `items_of(engine=...)` form.
- **Certainty:** read every side. Low severity.

### P13. The reading of `restart`, repeated

- **Where:** `validation/stages.py:193` reads the raw value:
  `if getattr(b_cfg, _RESTART, None) != "clean":`.
- **The door:** `identity.continues` (`identity.py:90–111`) is "the reading of
  the field … a copy per engine would be two spellings of one rule". Both deck
  writers call it.
- **Why the two agree today:** `None` reads as continuing in both.
- **Smallest fix:** `if continues(b_cfg): continue`.
- **Certainty:** read both sides. Low severity.

### P14. Values from a door, typed into a sentence the deck or a finding writes

- **The IR prefactor:** `pyscf/vibration_emitters.py:186` ("times the
  Gaussian/ORCA 42.2561 km/mol …") and `:1057` ("the … literature constant
  42.2561"). This is `constants.IR_KM_MOL_PER_D2_A2_AMU`.
- **Catalogue defaults in messages:** `validation/transport.py:79` ("300 K is
  the default") and `:99` ("The default is 10 eV") restate the defaults of
  `electronic_temperature` and `negf_eq_pole_ev`.
- **The charged-vacuum floor:** `siesta/input.py:1303` ("vacuum >= 25 A per
  side … molbuilder warns if it's thinner") restates
  `validation/siesta._VACUUM_MIN_CHARGED` (`validation/siesta.py:445`).
- **Units in deck headers:** see P15.
- **What goes wrong:** change the source, and the deck or message keeps the old
  number.
- **Smallest fix:** interpolate from the source, as `makov_payne` does
  (`architecture.md` § 3, "a second spelling is legal in exactly two places").
- **Certainty:** read every side. Low severity.

---

## B. One data shape

### P15. The relaxation record is built in two shapes by two writers of one result

- **The two writers:**
  - `pyscf/vibration_emitters.py:586–590` types the dict into the deck:
    `relaxation={'enabled': …, 'already_relaxed': …, 'n_steps': 0,
    'max_force_eh_bohr': None, 'max_force_all_atoms_eh_bohr': None,
    'converged': None, 'warning': None}`. The vibration deck then mutates it
    (`pyscf/vibration_deck.py:257–341`).
  - `spectra/vibrational_analysis.py:357–366` builds the SIESTA side's dict,
    which also carries `"record"`.
- **The keys:** declared once more in `spectra/datafile.py:154–170`.
- **What the other blocks do:** the PySCF deck already imports
  `equilibrium_block`, `thermo_block` and `ModeData` from `spectra.results` —
  "a mode's record is the one record's" (`vibration_emitters.py:925`).
- **What goes wrong:** a key added to one writer — `record` is the one already
  missing on PySCF — leaves the other engine's result without it.
- **Smallest fix:** a `relaxation_block(...)` builder in `spectra.results`,
  called by both writers.
- **Certainty:** read both sides.

### P16. The SIESTA spin words: the writer's map and the reader's table, unguarded

- **The two tables:**
  - `siesta/layout.py:101–106`: `SPIN_SPELLING = {"restricted":
    "non-polarized", "unrestricted": "polarized", …}`.
  - `parse/fdf.py:39–46`: `SPIN_WORDS`, the reader's table, which every deck
    reader resolves `Spin` through.
- **What goes wrong:** the comment at `layout.py:108–111` names the relation,
  but nothing holds it. Edit a writer word to one the reader's table lacks, and
  every reader (`deck_record`, `runs.declared`) reads that deck's spin as
  `None`.
- **Smallest fix:** declare the written spelling in the reader's table (which
  travels anyway), or check `SPIN_SPELLING[t] in SPIN_WORDS[t]` where the
  writer loads.
- **Certainty:** read both sides; the impact is plausible.

### P17. The words a run of points' state is said in, in two homes

- **The two homes:** `kind_table.py:133`: `FINISHED = ("done", "ran")`, with
  the reason at `:127–132` ("this table loads before the package can import
  that module"). `runpoints.py:60`: `DONE, RAN = "done", "ran"`.
- **What goes wrong:** nothing checks that the two agree.
- **Smallest fix:** import `runpoints` inside `_run_of_points`, as `_part`
  imports `atom_permutation`, or assert equality there.
- **Certainty:** read both sides. Whether that import cycles was not measured
  — plausible.

### P18. Two names for the pseudopotential folder, and its shared-package list composed twice

- **Two names for one constant:** `pseudos.py:51`:
  `from .runfiles import PSEUDO_DIR as PSEUDO_DIRNAME`. Readers import one
  (`siesta/seam.py:15`, `transport/rung.py:23`) or the other
  (`spectra/siesta_vibration.py:97`).
- **The shared package's file list, composed twice:**
  - `siesta/seam.py:162–174` (`_siesta_shared_package`, the seam's answer):
    `plan.glob` and `fixed_fields`.
  - `spectra/siesta_vibration.py:865–866` composes it again with
    `find_fixed`. Its comment says "the engine seam's rule".
- **What goes wrong:** if the seam's shared package gains a data file, the
  optical calls' folders do not get it.
- **Smallest fix:** one name for the folder. For the list, a function both
  modules ask. The seam imports `jobset.planned`, which does not travel with
  the job, so the function must be one that does.
- **Certainty:** read both sides. Moving a function might be held back by the
  travel constraint, so the second half is plausible.

---

## C. Duplicated, redundant or dead code

### P19. The template writer decides which keys are catalogue-only in two places, and its branches for them run only under tests

- **First place:** `template.py:1249–1296`. `template_with_values` clears
  `recommended`, `role_values`, `offered`, `result`, `active_when`,
  `choice_labels`, `why_fixed`, `why_shared` and `per_point_from` by
  `dataclasses.replace`.
- **Second place:** `template.py:952–1051`. `_item_payload` writes every one of
  those keys, but silently never writes `engine_default`, `engine_unfixed` or
  `not_offered`, although `_ITEM_KEY_ORDER` (`:926–934`) lists them and
  `_item_from` reads them.
- **The dead branches:** in the product, `_emit` is called only by
  `template_with_values` (`:1303`). The branches for the cleared keys are
  reached only by tests that call the private `T._emit` with catalogue rows
  (`test_template_roundtrip.py:243–256`;
  `test_template_citation_marker.py:79, 96, 126`).
- **What goes wrong:** that test's claim, "the writer writes every key the
  reader reads", is false for `engine_default` and `not_offered`. A new
  catalogue-only key has to be handled in both places.
- **Smallest fix:**
  - Declare one set of catalogue-only keys and have the writer skip it.
  - Drop the payload branches nobody reaches, and the tests that call the
    private writer.
- **Certainty:** read both sides.

### P20. Two SIESTA "stage science" lines, with the deck's units typed into them

- **The two functions:**
  - `siesta/input.py:262–291` (`_stage_science`):
    `f"MeshCutoff {cfg.mesh_cutoff} Ry"` and
    `f"force tol {cfg.relax_force_tol} eV/Ang"`, filtered by
    `inactive_settings`.
  - `siesta/vibration_deck.py:236–258`, whose relaxation branch writes the same
    line again — `"MeshCutoff … Ry, force tol … eV/Ang, …, max … steps"`
    without the filter — plus `"… Bohr"` and `"… Ry"`.
- **The units' owner:** `siesta/engine_input.unit_word` (`:286–293`), from the
  catalogue's `unit`, as the deck's own keyword lines use it.
- **What goes wrong:** a relaxation whose step count enters nothing still
  prints "max N steps" in the vibration header. A unit changed in the catalogue
  changes the keyword line but not the header.
- **Smallest fix:** one function, asked by both with the phase, with units from
  `unit_word`.
- **Certainty:** read both sides.

### P21. PySCF's geometry convergence targets: two constant spellings, two kwargs composers, and `maxsteps` typed by hand

- **The optimization deck:** `pyscf/layout.py:69–74` (`geom_kwargs`) and
  `:200–210` (`_geom`) name the targets `_GEOM_{name[5:].upper()}` and take
  every keyword, `maxsteps` included, from the anchors.
- **The vibration deck:**
  - `pyscf/vibration_deck.py:156–157` names them `GEOM_{…}`.
  - `:274–276` walks the section but skips `geom_max_steps`.
  - `:280` types `maxsteps=GEOM_MAX_STEPS`, while `geom_max_steps`'s catalogue
    anchor is `"maxsteps"`.
- **What goes wrong:** the `name[5:].upper()` naming rule is written in four
  places. If the anchor or a constant's name changes, one deck follows and the
  other does not.
- **Smallest fix:** one emitter in `layout`, returning the constants and the
  kwargs, called by both decks.
- **Certainty:** read both sides.

### P22. Dead Makov–Payne helpers re-derive a cell's volume, and the duplicated formula nothing compares

- **The dead helpers:** `siesta/makov_payne.py:133–156` (`_cell_volume_angstrom3`,
  `effective_L`) have no production caller — only
  `tests/test_makov_payne.py:100`.
- **What they copy:** a cell's volume, which `cell.resolve(...).volume` owns.
- **The unguarded copy:** the module deliberately keeps two copies of the
  formula — `compute_correction` at `:84–130`, and the script's at `:318–321`,
  "measured to agree 2026-09-09". The test compares the dead `effective_L`
  with the script, never the two `compute_correction`s.
- **Smallest fix:** delete the two helpers. Test the script's length against
  `cell.resolve`, and its `compute_correction` against the module's.
- **Certainty:** read both sides; dead per grep.

### P23. `resolved_ladder` lays resolve's layers again

- **The two compositions:**
  - `resolve.py:796–825` composes template ⊕ overrides ⊕ the rung's answers by
    its own code.
  - `resolve()` (`:442–793`) does the same with its override door, its
    refusals of a stated answer and of a pre-2.1.0 zero, and its run settings.
- **One template parsed three times:** inside `resolve()`, at `:499`
  (`config_from_template`), `:529` (`_refuse_a_stated_answer`) and `:535`.
- **What goes wrong:** a layer added to `resolve()` must be mirrored by hand,
  or the sequence checks compare decks that are not the ones prep writes. The
  docstring promises "resolved by the SAME primitives".
- **Smallest fix:** a shared helper laying the computed layers, which
  `resolve()` then extends; parse once.
- **Certainty:** read both sides; plausible impact.

### P24. Smaller duplicates

- **Two checks of the SIESTA vibration, two copies of the same text:**
  `validation/spectra.py:226–243` and `:499–512` carry the same "Holding N
  atom(s) leaves R whole-body motion(s)…" sentence. The docstring at `:469–472`
  promises one wording, yet it is written twice. The frozen-index range check
  is written at `:281–298` and again at `:482–490`.
- **Two near-identical PySCF emitters:** `pyscf/scf_setup.py:23–63` and
  `:66–122` run the same loop over two sections.
- **The run-settings block, scanned twice:** `pyscf/script_input.py:100–108`
  and `:196–207` each scan for the block's markers.
- **The same two-line pair, written twice:** `siesta/layout.py:346–349`
  (`_pair`) and `:397–399` (`_spin`, pinned) both write
  `f"{keys[0]:<18}.true.\n{keys[1]:<18}…"`.
- **Raw keys beside the key function:** `siesta/layout.check_rules`
  (`:497–507`) reads `"systemlabel"` and `"numberofatoms"` raw, beside
  `fdf_key(...)` in the same function.
- **The orbital estimate, three times:** `siesta/input.py` types `10 * n_atoms`
  at `:159`, `:181` and `:317`, which "the SAME estimate the deck's
  BENCH-MARKS block records" promises is one number.
- **A table name hand-copied in a shared module:** `bench/result.py:217–219`
  pairs `("cpus_per_task", "omp_threads")`, which is
  `jobset.model.AS_RESOURCE` ("the one map"). The same shared module reads
  SIESTA's grammar by module name (`:169–173`), where an engine record should
  name the trial reader. Only SIESTA has a benchmark lane today, so this is
  plausible.
- **Certainty:** read every side. Low severity.

---

## D. Tests

### T1. A second reader of `TBT.Contour.window`, written in the tests, disagreeing with the product's about units

- **The two readers:**
  - `tests/_deck.py:45–70` (`fdf_energy_window`) parses the window and
    converts units through `molbuilder.units`.
  - The product's reader is `transport/energies.deck_window` (`:103–116`). It
    reads tokens by position (`rows["from"][1]`, `[4]`) and ignores the unit
    word.
- **What the tests do with them:**
  - `test_transport_decks_e2e.py:269–274` asserts both readers, one after the
    other.
  - `test_deck_helpers.py` (66 lines) exists only to test the test helper. It
    uses hand-typed decks and typed constants (`13.605693122994`,
    `27.211386245988`, which are `constants.RYDBERG_EV` and `HARTREE_EV`).
  - `_deck.py` lives outside `tests/support/`.
- **What goes wrong:** the unit handling the tests verify is the helper's, not
  the product's. A window written in Ry would be read 13.6× wrong by
  `deck_window`, while the tests pass. This breaks `architecture.md` § 3, "a
  unit word lives in `units` … a reader passes its default".
- **Smallest fix:**
  - Assert through `deck_window`, made unit-aware through `units`.
  - Delete `fdf_energy_window` and `test_deck_helpers.py`.
- **Certainty:** read both sides.

### T2. A hand-written stand-in for a config class, rebuilding catalogue items from field metadata

- **Where:** `test_template_citation_marker.py:24–68`. A synthetic `_Cited`
  dataclass carries `citation`, `kind`, `engine_key` and the other markers in
  `field(metadata=…)`. `_decl` rebuilds `T.Item`s from that metadata.
- **What it re-implements:** this is the "`declaration_for`" path the test's
  own comment says "went with the class copies in 2.15.4". The tests also call
  the private `T._emit` and pass `template_with_values` a class that no engine
  record names.
- **What is already there:** the real catalogue carries `citation =
  ["transport"]` rows (`basis_size`, `mesh_cutoff`, …) on `SiestaConfig`.
- **Smallest fix:** use the real catalogue and `SiestaConfig` through
  `template_with_values(..., calculation="transport")` (`testing.md` § 3b: "a
  fake … built through the entry the tested code uses").
- **Certainty:** read.

### T3. Dead per-file copies of the road's H2

- **Where:** `test_template_roundtrip.py:23–27` and
  `test_stage_resolution.py:23–29` each define an `h2()` fixture: H2 in a
  12 Å vacuum.
- **What is wrong:** neither file's tests ask for it (grep finds no use). Both
  copy `support.road.H2` / `road_structure`.
- **Smallest fix:** delete both.
- **Certainty:** read and grepped.

### T4. One rule pinned three times, and API-shape assertions

- **"A stage's override wins, and a field it does not name keeps the
  template's value"** is pinned three times: `test_stage_resolution.py:100–110`
  (two tests), `test_template_roundtrip.py:102–108`
  (`test_a_stage_resolves_against_the_template_read_back`) and
  `test_resolve.py:76–81`.
- **"A required item key is refused"** is pinned twice:
  `test_template_roundtrip.py:139–151` (kind, type, help) and `:336–345`
  (category) — one rule in two tests.
- **Signatures restated, per `testing.md` § 3b:**
  - `test_stage_resolution.py:112–118`: the result's type is `SiestaConfig`.
  - `:120` onward: `replace` does not mutate.
  - `test_template_roundtrip.py:395–403`: `one` raises `KeyError`.
  - `:385–392`: `select` refuses an engine.
- **Smallest fix:** keep the `resolve` row as the one door's test, fold the two
  required-key tests into one parameter list, and retire the signature tests.
- **Certainty:** read.

### T5. Hand-typed tables and templates, and a six-line prelude copied per test

- **Hand-typed templates for the reader's refusals:**
  `test_template_roundtrip.py:139–180`, `:336–345` and `:440–454`;
  `test_template_declarations.py:151–156` and `:253–262`. Several are many
  wrong examples of one reader.
- **Hand-typed tables, each with the same prelude:** `test_kind_table.py`
  writes a table into `tmp_path`, points `TABLE_FILE` at it, clears
  `_table`, and clears it again in `try/finally`. The prelude is copied at
  `:147–159`, `:370–380` and `:424–434`, and once more for `engine_table` at
  `:473–482`.
- **Smallest fix:**
  - One support fixture: "the kind/engine table holding these rows".
  - A reader's refusal asked of an edit of a template the writer wrote, as
    `test_an_unknown_item_name_is_refused_not_dropped` (`:258–273`) already
    does.
- **Certainty:** read.

### T6. One engine's one section is checked for unknown item names that the framework drops silently everywhere

- **The test:** `test_pyscf_reads_the_catalogue.py:15–19` checks only
  `pyscf.layout.SCF_SECTION` against the catalogue.
- **The gap:** the framework's walk (`script_emit.py:1111–1145`) puts a name
  the catalogue lacks into `silent` and writes nothing. Its `line` doors return
  `None` for it: SIESTA's `layout._line` has `if not param.known: return None`.
  So a typo in any other PySCF section, or in any SIESTA section, drops a
  setting with nothing going red.
- **Smallest fix:** the render refuses a layout name the catalogue does not
  hold. Then this test retires.
- **Certainty:** read both sides.

### T7. A data pin where an outcome belongs

- **Where:** `test_doc_claims.py:22–41` asserts the catalogue's
  `type == "int"` for `block_size`.
- **What should be tested instead:** the failure it names — a benchmarked 24
  snapped to 16 — is an outcome. Test it as one: a run card's `block_size: 24`
  reaches the deck as `BlockSize 24`.
- **A stale header:** the file's comment (`:11–19`, "Gate A -- a worked example
  must be a VALID example") describes a gate the file no longer holds.
- **The units tables, which fall under the science exception:**
  `test_units.py:29–41` and `:62–78` pin the same factors twice (Ry, Ha, K,
  cm⁻¹, kcal/mol). Fold them into one table; nothing is to be deleted.
- **Certainty:** read. Low severity.

---

## The five that matter most

1. **P1:** the solvent's name is normalised in seven places. Measured: the gate
   accepts `" water"`, then the PySCF writer raises at prep. This is the cause
   of round 1's D9.
2. **P2:** the vibration's amplitude range is typed into its check. Measured:
   two warnings on one field for one value.
3. **P5:** `IDENTITY_ITEMS` and `SETUP_WORDS` hold per-engine identity in shared
   `template.py`, beside the engine record's `identity_field`. This is the cause
   of round 1's D3, and a new engine would need shared code edited.
4. **P3:** SIESTA's `BENCH_FIELDS` hand-copy catalogue anchors, types and units.
   `Diag.Algorithm`'s choices are already lost, and the only guard checks type.
5. **T1:** a test-written second reader of the transmission window disagrees
   with the product's `deck_window` about units, and is itself the subject of a
   test.
