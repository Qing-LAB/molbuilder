# F12 fix 16 (templates and engines: each fact in one home, 2.77.24, R3c3) — the review

**Verdict: closed with 2.77.24.** The `r2-4` audit's one-door rows (P3, P4, P6, P7, P10–P15, P17–P24),
`r2-3` F3, `r2-2` C19 and R3a's leftovers (the PySCF generated constants, the template's refusal kit),
each fact given one home; and the user's answer of 2026-11-22, *"Launcher by full path"*: every env
command molbuilder prints names the install script by the checkout's full path. Asked during the
review, the user decided two more, 2026-11-22: *"Ship it, say it"* (a PySCF vibration launched before
2.77.24 is refused at prepare-again, said in the changelog with its way back), and *"Full path here,
words there"* (a file prepared for another machine names the env command as typed in molbuilder's
checkout there).

The lessons, in the user's words where they gave them: a spelling that holds a path is data, and every
place it lands — a Python literal, a shell's double quotes, a file for another machine — must be found
before it ships (round 1 found four landings, round 2 a fifth, the full-text readers the sixth); a door
made strict (`one()`) is checked against every value its callers pass first (`kmesh` was handed `open`).

## What was built

- **One home each**: `siesta.input.bench_fields` from the catalogue (P3); `pyscf.layout.keyword`,
  `geom_constant`, `POLICY_CONSTANTS` (P4, P21); catalogue ranges and defaults in validation (P6, P14);
  `canonical_gres` (P7); the phase roles held to the kind table (P10); the PySCF hints from the deck's
  own values (P11); `items_of(engine=)`, `one()`, `STATE_ITEMS`, `is_member` (P12); `identity.continues`
  (P13); `spectra.results.relaxation_block` (P15); `runpoints.DONE/RAN` (P17); `runfiles.PSEUDO_DIR`,
  `pseudo_package`, `files_in` (P18); the catalogue's own keys stated once (P19); one stage line (P20);
  dead Makov–Payne helpers gone (P22); `resolve._overridden`, the template read once (P23); one sentence,
  one range check, one dresser, one orbital estimate, `AS_RESOURCE` (P24); `template.SAID` in decision
  order (F3); `deck_record.fence` / `in_blocks` (C19); constants interpolated; `template._refuse` on
  `persist.RecordRefusals`.
- **The env command, wherever it lands**: `envs.hints.launcher_of` and `LAUNCHER` (full path, shlex
  quoting); `identity.in_double_quotes` inside a shell's double quotes; `repr` inside Python; the
  install script's `LAUNCHER_TYPED` by the same quoting rule; `envs.hints.prepared_for` for a file
  prepared for another machine.
- **Tests' own door**: `tests/support/job_python.py` (molbuilder's absence enforced).

## Round 1 — seven readers (correctness ×3 parts, one-home ×3 parts, notes and tests)

| # | finding (reader) | verdict |
|---|---|---|
| 1 | the launcher's quoted full path closes a string in generated code: the Raman deck and PySCF's geomeTRIC stop (Python), PySCF's env check (`python -c`), the GPU env's check (`bash -c`) — and, found by the fix's own test, the run script's env stops (`echo "…"`) (corr-C 1, corr-A 1) | fixed — a Python literal by `repr`; in a shell's double quotes by `identity.in_double_quotes`, which `runwrap.shell_literal` now asks too; rows: a PySCF optimization and a Raman vibration prepared from a checkout path with a space and an apostrophe parse, both env checks parse |
| 2 | the install script printed `bash scripts/install-env.sh` (its usage, its no-subcommand line, its host-env stop) (home-A 2, tests 3) | fixed — `LAUNCHER_TYPED`, by the checkout's full path; the usage substituted after the quoted heredoc, its backquotes never run; rows on the help and the bare call |
| 3 | four documents and two notes quoted the old printed line; tooling and GPU rows expected the relative text (home-A 2, tests 3–4) | fixed — `<checkout>` in the documents; the rows expect `hints.opt_in_cmd`/`install_cmd`, or halves that hold for any path |
| 4 | the monitor e2e's control assumed molbuilder absent; two notify-policy probes too (tests 1) | fixed — `support.job_python` |
| 5 | F3's test passed on 2.77.23 (no run-card row) (tests 2) | fixed — prep with the rank count from the run card; fails on 2.77.23 at its order |
| 6 | P12 not closed: `resolve._card_read`, `template.why_role` (home-A 1) | fixed — `items_of(engine=)`, `one()` |
| 7 | P18 closed on prep's side only: the job's main road chose by its own rule (home-B B2, corr-B 3) | fixed — `pseudo_package(names, folder)` on both of the job's listings; the sentence and docstring true |
| 8 | `CATALOGUE_ONLY` read by no code (home-A 4, tests 7) | fixed — `_item_payload` drops it |
| 9 | `mf.conv_tol` lost its second space in every PySCF deck (corr-C 3, home-C 1) | fixed — restored |
| 10 | the vibration deck's constants renamed `_GEOM_*`, its policy still `ON_NONCONVERGENCE`, retries typed by hand in both decks (corr-C 2, home-C 2, 6, 8) | fixed — `layout.POLICY_CONSTANTS`/`policy_constants` for both decks; accepted as visible deck text (`_GEOM_*`, `_ON_NONCONVERGENCE`); vibration.md § 4.1 |
| 11 | P11's sentence promised more hints than carried the value (corr-C 4, home-C 3, tests 5) | fixed — every hint naming a deck setting says this deck's value, its attribute `layout.keyword` |
| 12 | `layout._kw` beside `geom_keyword`, two routes to one fact (home-C 4) | fixed — `layout.keyword`, the one; `geom_keyword` gone |
| 13 | the Raman factor typed in comments (home-C 5) | fixed — interpolated from `BOHR_ANGSTROM`, text byte-identical |
| 14 | the fence scan four ways, two whitespace rules (home-B B3) | fixed — `deck_record.fence`, asked by every block reader and the gate |
| 15 | a stray marker mid-line was refused and now passes (home-B B3) | accepted — no reader reads it as a fence; the gate agrees with them |
| 16 | the whole-body note's wording changed (corr-B 2, home-B B4) | accepted — visible, in the changelog |
| 17 | every PySCF deck names the checkout's path (corr-C 1) | accepted — the user's decision, in the changelog |
| 18 | notes and documents: the vacuum names, `stage_science`'s annotation, the BENCH-MARKS quote, the mesh floor's "100", the "10 * n_atoms" notes, `results` notes, `hints`/`stages` module notes, template.md § 2.1a/§ 2.2/§ 5/§ 5.4, `AS_RESOURCE`'s note, the deck table's comment order, three stale test references, `job_python` in testing.md § 8, the road's `part` key (corr-A 2, corr-B 1, 4, home-A 5–6, home-B B1, B5, home-C 7, tests 6, 8–14) | fixed |
| 19 | small copies of this release's class: `pseudos/` in the transport deck's note, the Madelung constant, the mesh phrase twice (home-B B6) | fixed — `PSEUDO_DIR`, `MADELUNG_CUBIC`, `siesta.input.mesh_words` |
| 20 | the rung's answers laid by both `resolve` and the ladder (home-A 3) | left as is — one door (`template.role_answers`) and one primitive; P23's composition (the stage layer, the template read thrice) is closed |
| 21 | the PySCF half of C19 has no test (tests 15) | left as is — it needs a deck the road never writes; the grammar's door is pinned by the gate rows |
| 22 | `geom_constants` has one caller; its name beside `geom_constant` (home-C 2) | left as is — each says what it returns |
| 23 | `kind_table` (L1) imports `runpoints` (home-A 7) | left as is — the edge existed before (`kind_table`, `runfiles`); the L1 index is R5's |
| 24 | outside this release: PySCF's optimization header types `maxsteps=` and `Ha/Bohr` (home-C); `cell.py`'s "8 Å"; the path tool's stale `launch` note and its pre-existing `--check` failure (home-B B6); a stage override `mesh_cutoff: "a"` crashes the loosening check (corr-A) | later — plan |

## Round 2 — five new readers (correctness ×3 parts, one-home and notes-and-tests across the whole change)

| # | finding (reader) | verdict |
|---|---|---|
| 1 | a PySCF vibration stage launched on 2.77.23 is refused at prepare-again on 2.77.24: the vibration deck's constants renamed (round 1 #10) and its relaxation block filled through `relaxation_block` (P15) are compared statements (corr-C 1) | the user's call, 2026-11-22: *"Ship it, say it"* — accepted, in the changelog with its way back |
| 2 | a sequence's walk named the repair unescaped inside `_log ERROR "…"`: from a checkout path with an apostrophe it never parses, no group launched (corr-B 1, home 1) | fixed — `identity.in_double_quotes`; row: the walk the launch wrote, from such a path, parses (`bash -n`) |
| 3 | the install script's help under bash 4.2 printed the replacement's quotes (corr-A 1) | fixed — the substitution assigned, not expanded inside double quotes; measured under `BASH_COMPAT=4.2` and 5.1 |
| 4 | six more of the script's own lines printed itself unquoted or pathless (corr-A 2, tests 4, home 2) | fixed — `${LAUNCHER_TYPED}` at each |
| 5 | the script quoted its path by `printf %q`, molbuilder by `shlex.quote`: two spellings (corr-A 4, tests 4, home 3) | fixed — the script quotes by `shlex.quote`'s rule (`_sh_quote`); its help from a path with a space and an apostrophe prints molbuilder's line exactly, and splits back to the path |
| 6 | the env-check rows never saw the quoted path: the recipe registry is cached per process (corr-A 3, tests 1) | fixed — the row builds the registry afresh (`builtin_recipes.__wrapped__`) and finds the path in the check |
| 7 | SIESTA's run-script env stops under a quoted path reached by no row (tests 2) | fixed — a SIESTA optimization row beside the PySCF ones; prep's own `bash -n` |
| 8 | "never wider" (BENCH-MARKS' `BlockSize` as `pow2`) unpinned since the deleted test (tests 3) | fixed — a deck row |
| 9 | round 1's accepted #15 (a marker's text within a line passes the gate) untested (tests 6) | fixed — a gate row |
| 10 | the P11 hints stated values an HF deck does not hold (its functional, its grid) and a tolerance unrounded; three hints name a setting with no value (corr-C 2, tests 5, home 8) | fixed — a setting that enters nothing here offers no hint (`template.inactive_settings` on the deck's own values, the electronic state's among them); values as the deck writes them (`layout.as_written`); the sentence: each hint *that suggests a value*; a deck row |
| 11 | vibration.md § 6.2 did not say PySCF's `record` is null (corr-C 3) | fixed |
| 12 | P12-class copies outside the claimed list: `template.waits_on_words`, `transport.window` (home 6) | fixed — `one()` |
| 13 | `_source_from` refused by its own words, beside the kit (home 6) | fixed — `_refuse`; the words gain the reader's `template: ` |
| 14 | `_names_in` repeated `find_fixed`'s listing (home 6) | fixed — `runfiles.files_in`, the one listing both ask |
| 15 | `mesh_words` in the writer, the vibration module importing back into it (home 7) | fixed — beside `unit_word` in `siesta.engine_input` |
| 16 | `layout.hard_scf_hint` typed `mf.level_shift`; `_kw` a second name for `keyword` (home 5, 8) | fixed |
| 17 | `validation/stages`' note said L3 (home 8); the F3 row cited § 6.6 (tests 7); tests stripped bundle members by hand and imported `job_python` twice (home 10, tests 7) | fixed |
| 18 | SIESTA's out-of-range sentence changed wording (home 9) | accepted — visible, in the changelog |
| 19 | P24(d) the SIESTA two-line pair and P24(e) its raw keys untouched (home 4) | left as is — the R3 triage verdict (LEAVE), named in the plan |
| 20 | P4-class anchors typed elsewhere in the PySCF decks: `mf.chkfile`, `mf.init_guess = "chkfile"`, `mf.newton()`, `mf.with_solvent.method`, two SOSCF notes (home 5) | later — plan (not catalogue items one-for-one; each needs its item named first) |
| 21 | `cell.py`'s "8 Å" and `structure.py`'s notes restate the vacuum floors (home 6) | later — plan (the floor must move below `validation` first) |
| 22 | `tools/testrun.py`'s xdist refusal imports molbuilder (corr-A note) | left as is — the documented invocation is the host env's Python |
| 23 | a backslash sequence or a brace in the checkout path still breaks the PySCF header docstrings and the env-ABI remedy template (corr-C, own sweep) | left as is — no ordinary path holds one; said in the changelog's Untested |

## The two independent full-text readers, and the discipline reader

| # | finding (reader) | verdict |
|---|---|---|
| 1 | prepared for another machine, the run script's env stops, the PySCF decks' remedies and prep's GPU note named this machine's checkout path — against *"nothing prep writes names the machine it was prepared on"* (`preparing-for-another-machine.md` § 2) (B) | the user's call, 2026-11-22: *"Full path here, words there"* — fixed: prep plans a stage for another machine under `envs.hints.prepared_for`, its env commands `bash scripts/install-env.sh … in molbuilder's checkout`; for this machine, the full path; both chapters say so; rows for Sol and for this machine |
| 2 | the quoted-checkout rows typed the launcher's quoting by hand, so the real rule went untested (discipline) | fixed — `envs.hints.launcher_of(checkout)`, the door `LAUNCHER` and the rows' stand-in both ask |
| 3 | review-row citations and old-state clauses in code, notes and three chapters (discipline) | fixed — removed; each positive sentence kept |
| 4 | tombstones: `kind_table`'s note where `FINISHED` stood, `pseudos`' pointer, the path tool's reworded history, `script_emit`'s "not checked here" note whose reason had gone (discipline) | fixed — removed |
| 5 | the catalogue-only set policed keys no branch writes (discipline; against round 1 #8) | fixed — the constant and its filter removed (*a home that must stay empty is removed, not policed*); template.md § 5 states the set and the writer has no line for it; the row pins it from the chapter's list |
| 6 | defensive `None` arms around catalogue defaults that always exist (discipline) | fixed — asked directly |
| 7 | the BENCH-MARKS rows read the deck's text where the block has a reader (discipline) | fixed — a road key, `bench_marks`, read through `deck_record.read_bench_marks_block` |
| 8 | the vibration's role check named code and was in no chapter (discipline) | fixed — a shipped-table defect raised as one, in words; vibration.md § 5.11 says prep holds the two tables to each other |
| 9 | `runfiles.find` and `find_by_role` listed a folder their own way beside `files_in` (discipline) | fixed — both ask `files_in` |
| 10 | tests built their own job's-python call for want of an `env` argument; testing.md § 8 left out `lines_without_molbuilder` (discipline, B) | fixed — `job_python.run(env=)`; § 8 names both |
| 11 | the P23 guard added a test where a unification should reduce them; the P18 guard said neither its silent failure nor why the code is not enough (discipline) | fixed — the P23 guard retired; the P18 row says both |
| 12 | the spin hint suggested values with no deck value (A, B) | fixed — through the one hint writer |
| 13 | the P20 row cited § 5.11 § 4 where the sentence is § 5.3 (B) | fixed |
| 14 | `config_from_template`'s parameter named `text`; `_attr`/`_item` annotated `str`; `deck_record`'s header (discipline) | fixed |
| 15 | the deck's PROVENANCE table also lists a prep flag before the run card (A) | accepted — in the changelog |
| 16 | the changelog draft's citations and missing rows (A, B) | fixed in the row |
| 17 | `_ITEM_KEY_ORDER` still orders the catalogue's own keys (B) | left as is — the reader's list of every key it reads |
| 18 | installation.md's bare-gcc remedy relative (B) | left as is — an instruction typed at the checkout root |

The user's call during this round, 2026-11-22, when the rule below collided with *"Full path here, words
there"*: *"Run script: words everywhere"* — a run script is one file for every machine
(`execution/architecture.md` § 9), so its env stops name the command as typed in molbuilder's checkout on
every machine; decks and prep's printed lines follow the first call (a fast-suite failure,
`test_the_inner_wrapper_is_byte_identical_on_both`, found the collision).

## The reader of the fixes

| # | finding | verdict |
|---|---|---|
| 1 | a later group prepared with no `--target` on a calculation set to Sol (Task setup's every prep after the first) named this checkout: the switch read the flag alone | fixed as the reader proposed and ran in scratch — decided by the machine the plan is for (`target`, else the calculation's own copy of its record); a row for the later group |
| 2 | `STAGE-PLAN.md`'s `warm-files:` line names this checkout's path for a calculation prepared for Sol (2.77.23 the same) | later — plan |

## Evidence

- The whole fast suite: 6463 passed and 23 failed before round 1 (one cause, `kmesh` handed `open`, fixed);
  6490 passed after round 1's fixes; 6494 after round 2's; 6494 and 1 failed after the full-text and
  discipline fixes (the run scripts' sameness, the user's call, fixed); 6494 passed after that, before the
  reader of the fixes' one change, whose file's tests (73 passed) and the end-to-end run came after it.
  pyflakes clean on every changed file; `bash -n` on the install script.
- Every behaviour-change row seen failing on 2.77.23 at its assertion: the ENV any-folder row, the P11,
  P3, P20 deck rows, the C19 spaced-marker gate rows, the F3 order row (with a run-card row), the
  quoted-checkout deck and env-check rows, the walk row, the gate's mention row, the HF hints row. Guards,
  passing on both (no behaviour change): P17's kind-table row, P18's package row, P19's catalogue-keys
  row, P22's script rows, the BlockSize row.
- The install script's help and stops measured by hand from a copy at a path with a space and an
  apostrophe, under bash 5.1 and `BASH_COMPAT=4.2`: each printed line is `shlex.quote`'s spelling and
  splits back to the path.
- End to end, after the last fixes: the PySCF water vibration (P15's eight keys, the renamed constants), the SIESTA vibration (the job's shared package, the relaxation call's line, the role check), a PySCF relaxation's outcomes (the policy constants), the monitor on a live run (its probe on a job's python), a sequence of groups (the walk), the real junction (one pseudopotential folder): 297 passed, 0 failed.
