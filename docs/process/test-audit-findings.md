# The 2026-09-08 test audit — what it found that is not yet acted on

**Role:** record — an investigation queue, not a contract
**Domain:** process
**Companions:** [`testing.md § 3b`](?doc=process/testing.md) — the standard the
audit ran under; [`science/test-design-findings.md`](?doc=science/test-design-findings.md)
— the audit's **scientific-validation** half, recorded separately because that
class was protected.

## 0. Why this file exists, and what it is not

On 2026-09-08 an agent audit gated roughly 3,000 of the suite's 4,448 test
functions against `testing.md` § 3b. It produced **three** outputs. Only two of
them had a home:

| output | where it lives | state |
|---|---|---|
| the cuts that were APPLIED | the commit messages of `67b46128`, `4ab23917`, `85275ecb`, `09c05c51`, `f5e16f1b` | durable |
| the SCIENCE design findings | `science/test-design-findings.md` | recorded, unactioned |
| **everything else** | **nowhere — this file** | **recorded 2026-09-09** |

That third bucket is real code defects the audit found by reading *tests*, a
coverage gap a deletion exposed, four tests owed a redesign, and a set of
subsumption verdicts that were reasoned but never verified. It sat in
conversation context only, which meant one compaction from being lost.

**Nothing in this file has been fixed.** Every item was re-verified against the
working tree on **2026-09-09** and carries the line numbers that verification
used; that is what makes it fact-checkable rather than remembered.

> **Read § 4 before acting on § 3.** The audit's own subsumption reasoning was
> mutation-tested on seven pairs and came back **1 wrong in 5 decided**. A
> verdict in § 3 is a lead, not a licence.

---

## 0a. The defect ledger — every numbered finding, and where it lives

**Nothing here is fixed.** Each row was reproduced against the working tree on
the date given, and the detail — the reproduction, the line numbers, and what
the fix costs — is in the section named. This table exists so there is ONE place
to look; the sections are where the argument is.

| # | what | live? | detail |
|---|---|---|---|
| **#64** | two task-setup routes drop `cfg`, so findings carry no `workflow_group` and the page cannot place them on a card | page defect | § 2.1 |
| **#65** | the XSS allowlist matches files by SUFFIX, so a bare `viewer.js` entry exempts five files; two of its three patterns are dead and the live one lands in a viewer the comment never names | security lint hole | § 2.2 |
| **#66** | four tests owed a REDESIGN, not a deletion — still in the tree, still weak — plus the untested BROKEN env-state gate in `run_install` | weak guards | § 3 |
| **#67** | `POST /api/selection/eval` answers **HTML 500 with a stack trace** where its contract says JSON 400, for three malformed-rule shapes | **live, on a per-keystroke path** | § 6.1 |
| **#68** | the topic refusal says *"canonical six"* for a set of **nine** — and `test_web_files.py:1077` asserts the stale wording, so the test protects the drift | user-visible | § 6.1 |
| **#69** | `files._validate_op_target` is a guard that **cannot fire**, while its docstring claims a protection that never runs | dead guard | § 6.1 |
| **#70** | six file routes have **no outside-root test at all** — three of them mutations — and `/upload`'s is #74 | coverage | § 6.1 |
| **#71** | `web-api.md § 1` says the fence answers **403**; the fence answers **400**, and one test hedges `in (400, 403)`, which is how it stayed invisible | doc-vs-code | § 6.1 |
| **#72** | `TestTheDownloadButtonSaysWhatItIsDoing` carries a user-dated contract and **zero tests** | § 3a.1 misinformation | § 6.1 |
| **#73** | a dead local at `test_periodicity_gate.py:775` marking a dropped assertion | cosmetic | § 6.1 |
| **#74** | `test_upload_outside_root_rejected` **passes on pytest's own temp-directory name** — a test named for the fence that never reaches it | **cannot fail** | § 6.2 |
| **#75** | `test_too_small_cell_is_a_hard_error` asserts `"a" in str(exc.value)` — true of every English sentence | **cannot fail** | § 6.2 |
| **#76** | an **inversion** (det −1, chirality-flipping) in place of `orient`'s rotation passes **all 320 tests** that touch the operation | **science coverage** | `science/test-design-findings.md` § 7a |

**Sequenced, judged and tracked in `plans/plan.md` § 5m.** This file is the
evidence; that section is the work.

---

## 1. What the audit covered, and what was lost

Seven auditors ran: four gating partitions over the main tree, then three over
the third it had not reached (`envs`/`checkpoint`/`chemistry`,
`validation`/`transport`/`parsers`, `envs`/`template`/`task`).

Two facts about that run are worth keeping, because they are the evidence that
the standard was applied rather than performed:

- **The densest science ground returned the SHORTEST cut list.** The
  `validation`/`transport`/`parsers` slice gated 297 functions and cut 11, while
  `envs`/`template`/`task` gated 321 and cut 43. That is the correct shape.
- **Inverting the default did not inflate the list.** The third auditor
  re-derived the `envs_*` files under the removal default and reached **fewer**
  cuts than the earlier pass — 8 against 19 — so the tests it kept were kept on
  their merits, not on inertia.

**What was lost, stated plainly.** Most auditors terminated on a session rate
limit before delivering, and their scratch output files were cleared with the
session directory. Of the audit's full reports, exactly one survives — the
partition recovered in § 3 below. For the rest, the surviving record is the
*applied* cut with its reasoning in the commit message; the *unapplied* verdicts
from those partitions are gone and would have to be re-derived. Roughly 110
subsumption verdicts were outstanding when the campaign stopped; 18 are recovered
here in full, and the remainder are not recoverable from this session.

---

## 2. Real code defects the audit found

These were found by auditing tests and are **not test hygiene**. Each is a defect
in shipped code or in a guard that is not guarding.

### 2.1 `#64` — two routes drop the config, so findings cannot be placed

**Verified 2026-09-09.**

`web/blueprints/_shared.py:81` — `issues_to_json(issues, cfg=None)`. Its docstring
is explicit about the consequence:

> *"``cfg`` is the engine config dataclass — when provided, each issue's
> ``workflow_group`` is resolved … so the frontend can attach findings to their
> workflow-group card per web-ui-coherence Rule 2. When ``cfg`` is None, the
> ``workflow_group`` key is **omitted** from the dict."*

Three call sites in `web/blueprints/build.py`:

| line | call | `cfg` passed |
|---|---|---|
| 1060 | `_issues_to_json(validate(...), cfg=cfg)` | yes |
| **1760** | `_issues_to_json(_pf)` | **no** |
| **1805** | `_issues_to_json(_pf)` | **no** |

Both bare sites are inside `api_task_setup_save` (the route opens at
`build.py:1683`). 1760 is the refusal path — the description failed its own
preflight — and 1805 is the success path's advisory findings. In both, the page
receives findings with no `workflow_group` and cannot attach them to a card.

**The guard that should have caught it does not cover it.**
`tests/test_workflow_group_wire_contract.py` exists for exactly this failure —
its own docstring names *"``_issues_to_json(issues)`` without the ``cfg``
kwarg"* as the defect it prevents, and its failure message tells the reader to
add `cfg=cfg`. Its route-level tests cover the SIESTA, PySCF and transport
preflight endpoints. **Neither task-setup site is among them.**

**The fix is not one line, and that is probably why it was left.** There is no
`cfg` in scope at either site: `api_task_setup_save` works from a `task` object,
not from an engine config dataclass, and `resolve_workflow_group`
(`_shared.py`) needs a real dataclass — it returns `None` for anything else.
Closing this means deciding where the task-setup route gets the config for the
task's engine. **That decision is the work; do not treat this as a typo.**

### 2.2 `#65` — the XSS allowlist's file match is a suffix, and it over-matches

**Verified 2026-09-09.**

`tests/test_xss_audit.py:322` matches an allowlist entry with
`rel_name.endswith(af)`. Three entries are the bare string `"viewer.js"`:

```
("viewer.js", "hint.innerHTML")
("viewer.js", "if (c) c.innerHTML =")
("viewer.js", "ul.innerHTML")
```

Under `endswith`, each matches **five** shipped files: `modify/viewer.js`,
`task-setup/viewer.js`, `spectra/viewer.js`, `structure-optimization/viewer.js`,
`results/viewer.js`. An exemption written for one tab silently exempts all five.

Re-verification found the situation is worse than "over-broad", in two ways:

1. **The comment above the entries names a file that does not exist.** It reads
   *"static/viewer.js, static/modify/viewer.js, static/spectra/viewer.js"*.
   There is no `molbuilder/web/static/viewer.js`.
2. **Of the three patterns, two match nothing at all and the third lands in a
   file the comment never mentions.** `hint.innerHTML` and `ul.innerHTML` appear
   in no shipped JS. `if (c) c.innerHTML =` appears in exactly one file:
   `static/structure-optimization/viewer.js`.

So the only live exemption of the three is doing its work in a viewer the
justification was not written for, and two dead entries are standing by to
exempt that pattern in any of five files the moment someone writes it.

**The fix has two halves:** match the allowlist on the exact relative path rather
than a suffix, and re-derive the three bare entries against what the tree
actually contains. The second half is the one that needs a security judgement,
not a mechanical edit.

---

## 3. Four tests owed a REDESIGN, and one gap a deletion exposed — `#66`

**Recorded 2026-09-08, re-verified 2026-09-09: all four are still in the tree and
still weak.** The auditors classified them CUT but said *redesign rather than
delete* — the mechanism is a § 3a source pin, but the property underneath is real
and otherwise unguarded. They were deliberately NOT deleted, which means the
suite currently carries four tests that look like coverage and are not.

### 3.1 `test_checkpoint_sensor_js.py::test_checkpoint_js_has_no_polling_timer` (L245)

- **Mechanism:** reads `checkpoint.js` as text and asserts `"setInterval"` is
  absent from the non-comment lines.
- **Property (real):** the explicit-refresh model in `docs/web/projects.md`
  forbids a background poll loop.
- **Why nothing else guards it:** the node tests below it wait only 20–40 ms, so
  a re-added 5-second poll would never be observed.
- **Redesign:** the file already has the harness. Stub `globalThis.setInterval`
  in `_DOM` to record registrations, then assert none after
  `onDirectoryChange`. That observes the behaviour instead of the spelling.

### 3.2 `test_science_gaps.py::test_gap_7_installation_documents_siesta_version` (L272)

- **Mechanism:** greps `docs/ops/installation.md` for `siesta.*5\.4\.2`.
- **Property (real):** the installation guide must name the SIESTA release the
  recipe pins.
- **The defect:** the version actually lives in `molbuilder/envs/recipes.py` —
  at `:160` (the pinned release tag) and `:253` (the gcc-compatibility key) —
  and **the test never compares the two.** The exact drift it exists to catch —
  the recipe moves to 5.5 and the doc does not — leaves it green, because the
  doc still says 5.4.2 and that is all the test reads.
- **Redesign:** assert the doc names the version `recipes.py` pins, read from
  `recipes.py`.

### 3.3 `test_task_preflight.py::test_the_codec_still_does_not_import_an_engine` (L237)

- **Mechanism:** reads `task.py` as text and asserts four literal spellings are
  absent (`config.siesta`, `config.pyscf`, `SiestaConfig`, `PySCFConfig`).
- **Property (real):** `task.py` stays engine-agnostic.
- **The defect, both directions:** it fires on a **comment** that mentions
  `SiestaConfig`, and it misses `from molbuilder.config import siesta`.
- **Why nothing else guards it:** `tests/test_layering.py` does not. Its
  `_L1_MODULES` set holds both `config` (L82) and `task` (L153), so an
  L1→L1 import is legal there and the layering check has no opinion.
- **Redesign:** an AST import check in `test_layering.py`, where the layer rule
  already lives.

### 3.4 `tests/watch/test_siesta_parser.py::test_scf_history_default_empty` (L241)

- **Mechanism:** parses a fixture of two header lines and asserts
  `result["scf_history"] == []`.
- **The defect:** the fixture has no `outcoor`, so there are no frames, so
  `out_scf` is `[]` before any SCF logic runs. The branch the test names —
  `parse/engines/_helpers.py:240`, `if result.frames and all(f.scf_history is
  None for f in result.frames)` — **is never reached.**
- **Redesign:** one `outcoor` frame and no `scf:` lines. That makes the
  assertion mean what its name says, and would be the **only** coverage of that
  branch.

### 3.5 The coverage gap a deletion exposed

`test_envs_install.py::test_run_install_blocks_when_env_state_is_broken` **was**
deleted, correctly — its own body comment admitted the mismatch:

> *"the fake_run above returns no real dir, so probe_env_state sees FRESH … this
> test as written confirms that the FRESH path still works"*

But its docstring promised a gate the suite does not otherwise have. **The BROKEN
env-state gate in `run_install` is untested.** The gate is the `probe_env_state`
branch at `molbuilder/envs/install.py:610-640`, whose `--force-resume` escape
hatch explicitly names `GHOST/ORPHAN/BROKEN` (`:624`); `BROKEN` itself is
produced at `:401` and explained at `:440`. **This is a test to write, not a test
to fix** — and by § 3b it earns its place, because the failure it catches (an
install proceeding into a directory with no `conda-meta/`) is silent.

---

## 4. The mutation-verification campaign, and the error rate it measured

The audit's `CUT-SUBSUMED` verdicts are claims about *which inputs reach which
code*. **Coverage cannot settle them** — proven on the control pair, where both
tests execute the same lines and the real gap was that no frozen fixture carries
both `SCF_NOT_CONV` and `>> End of run`. So `tools/verify_subsumption.py`
(committed `02ea6b46`, stdlib-only, `sys.settrace`) breaks a line the candidate
covers and requires **both** tests to go red.

**Final tally over seven pairs:**

| verdict | n |
|---|---|
| CONFIRMED | 4 |
| **NOT-SUBSUMED — test restored** | **1** |
| undecidable by design | 1 |
| addressing error of mine | 1 |

**One wrong in five decided.** The wrong one was
`test_parse_is_deterministic` (restored in `09c05c51`): it compares the whole
legacy dict for **one file parsed twice**, reaching the wall-clock→elapsed
derivation at `parse/engines/_helpers.py:303`. The test claimed to cover it
compares two **different** files and never reaches that line. Its docstring now
carries the reason so nobody re-cuts it on the same argument.

**The undecidable one generalises into a rule worth keeping.**
`test_carbon_is_twelve_times_hydrogen` cannot be settled by mutation because
**the value asserted is not ours**: `chemistry.atomic_mass` is a lookup into
ASE's table (its docstring: *"ASE ships the IUPAC standard atomic weights, so
this is a name for a table rather than a copy of one"*), and the tool mutates
only `molbuilder/` on purpose. For that class the answer comes from reading, not
from a mutant — here, the sibling pins C to 12.011±1e-3 and H to 1.008±1e-3,
confining the ratio to [11.903, 11.928], strictly inside the candidate's
11.9±0.2. The tool now says so rather than shrugging.

**The consequence, and it is the reason § 3 and § 5 are queues rather than
work:** at 1-in-5, the outstanding subsumption verdicts are unsafe to apply
unverified, and roughly 40% may come back inconclusive even when verified
(~5 min per pair). The ~160 cuts already applied do **not** rest on this — they
are source pins and cannot-fails, where the evidence is a line of shipped code,
not an argument about reachability.

---

## 5. The one surviving partition report — 18 unapplied verdicts, with reasoning

This is the only audit report recoverable in full. It gated **110 functions
across 13 files: 92 keep, 18 cut (16.4%), 0 unsure** — 39 of the keeps were
KEEP-SCIENCE and were never weighed against the removal default.

**Re-verified 2026-09-09: all 18 are still in the tree.** None was applied. Each
carries the auditor's own *what is lost* line, which is what § 3b requires of a
cut and what makes these re-judgeable.

**The auditor executed nothing and mutated nothing.** Every claim below is from
reading the code path named in it. Per § 4, treat each as a lead.

Re-verification on 2026-09-09 also confirmed that **every test named below as the
coverer still exists** — a subsumption verdict whose coverer was itself retired in
a later batch would be void, and none is.

### Genuine duplicates — lost: nothing

- `test_admin_reload.py::test_naming_nobody_means_anyone_who_signed_in` (L81) —
  byte-identical to `test_availability_answers_honestly[no-section-named]`: same
  app config, same GET, same `available is True`.
- `test_admin_reload.py::test_serve_supervises_by_default` (L319) — identical
  probe call to `test_the_parent_forks_before_importing_the_application` (L330),
  which asserts `forked` on its first line.
- `test_ask_parsers_agree_across_surfaces.py::test_nothing_that_should_be_refused_became_acceptable`
  (L240) — same three refusal classes on the same two functions as
  `test_ask.py::test_an_answer_that_is_not_a_number_is_refused_with_the_shape`,
  which additionally checks the message. One of the two was cut, not both.

### Subsumed with a stated, bounded loss

- `test_admin_reload.py::test_an_anonymous_request_is_refused` (L131) — anonymity
  returns at `web/admin.py:107` (`if not email: return False`) before the list is
  consulted. *Lost:* anonymous refusal specifically when admins ARE named —
  acceptable because the refusal is two lines above any list lookup, and
  `test_naming_nobody_still_refuses_a_stranger` (L108) runs the same line in the
  configuration where a mistake would be dangerous.
- `test_admin_reload.py::test_the_supervisor_does_not_import_the_app_it_restarts`
  (L253) — a top-level app import in `cli.py` would put `molbuilder.web.app` in
  `sys.modules` at fork time, which L330 asserts directly on the real `serve`
  path. *Lost:* a top-level **flask** import in `cli.py`, which L330's own probe
  cannot see because it imports flask itself.
- `test_ask.py::test_the_request_is_shown_even_when_it_is_accepted_unasked`
  (L90) — `confirm` echoes the text unconditionally at `jobset/ask.py:339` and
  only then appends `"  (--yes)"`, so a sibling asserting `said[1]` already
  requires `said[0]` to be the request. *Lost:* the request text being mangled
  while still occupying `said[0]` — `test_without_yes_the_answer_is_the_persons`
  asserts `said[0] == text` through the same single echo line.
- `test_app_notifications_e2e.py::test_bar_present_and_hidden_until_first_notification`
  (L41) — `test_stack_of_two_and_individual_clearance` (L77) ends with
  `clearAll()` and asserts `#app-notifications` hidden. *Lost:* hidden-on-first-load
  before any notification has been shown; the same `hidden` toggle is driven by
  list length in both cases, and this is an e2e second saved on every run.

### Thin wrappers over a tested door

- `test_api_bench_summary.py::test_a_traversal_is_refused_by_its_own_name` (L82)
  — door is `web/blueprints/files.py:163::_resolve_within_roots`; `bench.py:55-57`
  returns the door's own `exc.message`/`exc.status` unchanged. Door test:
  `test_web_files.py::TestPathTraversalDefense::test_dot_dot_in_raw_path_rejected`.
  *Lost:* nothing — `test_a_readable_sweep_outside_the_roots_is_still_refused`
  still proves this route reaches the fence at all.
- `test_api_bench_summary.py::test_a_missing_path_argument_is_refused` (L88) —
  same door; the route has no missing-arg branch (`request.args.get("path","")`
  straight in, `files.py:173` raises).
- `test_ask.py::test_the_listing_and_the_submission_cannot_disagree` (L143) —
  door is `scheduler/admit.py::admits`; `ask._why_not` (`ask.py:250-261`) is one
  `return list(admits(row, Request(...)))` and the test retypes that same
  `Request` on the right-hand side. *Lost:* someone reimplementing `_why_not`
  without the door — `test_a_queue_that_cannot_take_the_job_is_listed_WITH_THE_REASON`
  asserts the rendered reason text, which only the scheduler's admission produces.

### Shape assertions — the signature restated

- `test_api_bench_summary.py::test_a_file_that_is_not_there_is_a_404` (L98) —
  one `if not path.is_file()` line restated as a status code. *Lost:* 404 vs 400
  for a file that vanished between listing and click; the page renders both
  identically, and "never a 500" is held by the retained non-job-set test.
- `test_ask.py::test_it_says_how_to_choose_and_that_nothing_has_happened_yet`
  (L156) — static footer copy restated; no computation between the f-string and
  the assertion. *Lost:* the footer disappearing — acceptable because it is the
  CLI's most visible line, so its absence is loud rather than silent, and the
  test's only real effect is to make copy edits red.
- `test_atom_annotations.py::test_channel_kind_validation_and_defaults` (L18) —
  dataclass defaults plus a bad-kind raise. *Lost:* `AtomChannel("bogus")` being
  accepted — the kind is consumed immediately by `_validate_annotations` and the
  emitters, which branch on `kind == "value"`, so a bogus kind is not silent.
- `test_atom_annotations.py::test_annotations_default_empty_and_backcompat`
  (L36) — the empty-field default on a fresh Structure. *Lost:* nothing
  measurable; every other test in the file builds on an empty start.
- `test_atom_annotations.py::test_atom_metadata_none_when_empty` (L163) —
  "nothing in, None out" for the comment-block emitter. **The auditor flagged
  this one itself as its lowest-confidence cut and the only cut inside a
  protected file:** it touches the annotation persistence format but no atom
  mapping. **Reinstate it if the persistence group is protected wholesale.**

### Cannot fail

- `test_api_bench_summary.py::test_the_composed_sweep_survives_json` (L142) —
  `sweep_view` and `bundle_for_sweep_file` are both monkeypatched to return a
  plain dict, so the assertions read back the literal the test just injected and
  `json.dumps` cannot fail on it. **The docstring's claim — that a REAL composed
  sweep survives JSON — is unheld either way:** with the composer stubbed, no
  dataclass or `Path` ever reaches `jsonify` here. **Worth reopening as a real
  test against a prepped sweep**, which makes this the one entry in § 5 that is
  a gap and not only a cut.
- `test_a_validator_validates_the_whole_value.py::test_the_line_scanners_are_deliberately_left_alone`
  (L87) — asserts Python's own `$`/`fullmatch` semantics in a scenario
  production never produces: `parse/engines/molwatch.py:561` does
  `line = raw.rstrip("\n")` before every `_ERROR_RE.match` (lines 210, 278).
  *Lost:* the rule it claims to guard — that the ~23 line-scanner `.match()`
  calls must not be converted — but it never checked a call site, so nothing
  protected is lost; the rule stays in the module docstring, where review can
  enforce it.
- `test_admin_reload.py::test_both_sides_read_the_exit_code_from_one_place`
  (L174) — `inspect.getsource` plus a substring over two shipped modules; a § 3a
  source pin. *Lost:* someone re-spelling the sentinel locally instead of
  importing it — acceptable because a mismatched value makes reload stop
  respawning, which `test_the_supervisor_respawns_only_on_the_sentinel` fails on
  directly, driving the loop with `cli.RELOAD_EXIT_CODE`.

### A boundary the auditor flagged as a question, not a verdict

The SLURM time and memory parsers in `test_ask.py` and
`test_ask_parsers_agree_across_surfaces.py` are unit conversions — but of
**scheduler resources**, not physical quantities. The auditor read the science
exception conservatively and reinstated the two absolute-conversion tests
(`7-00:00:00`, `80GB`) it had provisionally cut, leaving one exact-duplicate
refusal test as the only cut in either file. **If "unit conversions" is meant to
cover scheduler resources, no revision is needed** — nothing conversion-related
was cut. Recorded because it is the kind of boundary that should be decided once
rather than re-argued per auditor.

---

## 6. The header pass, 2026-09-09 — what writing the goal down uncovered

Three agents wrote the § 3b header (*the failure it catches* + *the contract that
owns the rule*) into the ten files with the most undocumented tests: **436
headers over 897 functions in ten files.** Every edit was verified to be
**docstrings only** — the AST of each file with docstrings stripped is
byte-identical before and after — and all three suites pass.

**The method is the finding.** Writing the header honestly forces the § 3b
question, because you cannot state the failure a test catches when it catches
nothing. Every item below came out of trying to write one sentence.

### 6.1 Code defects, all reproduced before being recorded here

**`#67` — `POST /api/selection/eval` answers HTML 500 where its contract says
JSON 400.** `web/blueprints/selection.py:127` catches `SelectionError` **and
only** `SelectionError`. Three payload shapes escape it, measured through the
Flask test client on 2026-09-09 against a control that behaves correctly:

| `rule` | raises at | status |
|---|---|---|
| `{"op":"or","operands":5}` | `selection.py:461` `tuple(from_json(r) for r in raw)` | **500** |
| `{"op":"by_element","elements":5}` | `selection.py:228` `set(rule.elements)` | **500** |
| `{"op":"first_n","rule":{"op":"all"},"n":"x"}` | `selection.py:306` `rule.n < 0` | **500** |
| `{"op":"not_a_real_op"}` *(control)* | refused by the codec | 400, `{"ok":false,…}` |

The three 500s return Flask's HTML error page with a stack trace, not the
`{ok,error}` envelope `web-api.md` § 1 requires. **The filter panel round-trips a
rule on every keystroke in the index box**, so this is on a live path. The fix
has two halves and the second is a design question, which is why it is recorded
rather than patched: validate that a sub-rule LIST field is a list at
`selection.py:461`, and validate leaf field types at construction — the
`evaluate`-time `TypeError`s are not reachable from `from_json` alone.

**`#68` — the topic refusal says "canonical six" for a set of NINE, and a test
pins the drift.** `projects.py:70-80` holds nine entries (verified: `len == 9`);
`job-contracts.md` § 2.5 states nine and carries an explicit *"Drift corrected
(2026-07-27): the source doc said six."* The message an operator actually sees
still says six (`projects.py:115`), `projects.py:493`'s docstring repeats it, and
**`tests/test_web_files.py:1077` asserts the stale wording** — so the test is
currently protecting the drift rather than catching it. Fix the message, then
the test.

**`#69` — `files._validate_op_target` is a guard that cannot fire.** Defined at
`files.py:1903`, docstring: *"Centralises the 'would orphan a canonical project
layout' check so move / copy stay in lockstep with rename + delete."* It has
exactly one call site (`:2062`, in `api_files_move`) — and `:2053` returns 400
for `src.is_dir()` **first**, while both of the helper's branches require a
directory. **Copy never calls it at all.** Behaviour today is unaffected because
directories are refused wholesale; what is wrong is that a docstring claims a
protection that does not run, and `test_move_refuses_canonical_topic_dir` looks
like its coverage.

**`#70` — seven file routes have no working fence test.** Verified by name
search: genuine outside-root tests exist for `list` (at the door), `mkdir`,
`write`, `delete` and `zip_prepare`. **Zero** for `/api/files/stat`, `/read`,
`/read_range`, `/rename`, `/move`, `/copy` — and `/upload`'s is `#72` below, so
it does not count. Three of those six are MUTATIONS, and `web-api.md` § 2.1 says
the fence at the route is the only thing standing there.

**`#71` — the document and the fence disagree on the fence's status code.**
`web/web-api.md` § 1's table assigns **403** to *"a path escaping the allowed
roots"*; `files._resolve_within_roots` (`:203`, `:205`) raises **400** for both
`..` and outside-root, and every test asserts 400. The only 403s in `files.py`
are OS `PermissionError`. `test_web_files.py:3189` hedges with `in (400, 403)`,
which is how the disagreement stayed invisible. **The fix belongs in the
document or the fence, never in the test.**

**`#72` — `TestTheDownloadButtonSaysWhatItIsDoing` (`test_web_files.py:3085`)
carries a user-dated contract and ZERO tests.** Its docstring records the
2026-08-29 ruling (*Zipping…* / unclickable-until-save) and the class holds only
a `_src()` helper that nothing calls — residue of the eleven source pins removed
from this file. `testing.md § 3a.1`'s load-bearing misinformation: a reader sees
coverage that is not there. Write the behaviour test or delete the class.

**`#73` — a dead local marking a dropped assertion.**
`test_periodicity_gate.py:775` binds `said = " ".join(...)` and never uses it —
residue of the 2026-08-03 prose→id migration. Harmless, except that a reader
takes it as evidence a message check exists.

### 6.2 Tests that cannot fail — two measured, not argued

**The sharpest thing this pass found: `test_upload_outside_root_rejected`
(`test_web_files.py:2517`) passes on its own temp-directory name.** It asserts
`"outside" in err or "root" in err` after posting to `tmp_path / "elsewhere"` —
but the `picker_root` fixture wires *that same* `tmp_path` as the root, so the
target is INSIDE the fence and the real error is *"target_dir does not exist"*.
The assertion passes because pytest's temp directory is named after the test:

    /tmp/pytest-of-qqing/pytest-36/test_upload_outside_root_rejec0/elsewhere
                                        ^^^^^^^      ^^^^                 

Both words are in the path the message echoes. **A test named for the fence,
which never reaches the fence, and cannot fail.** `test_delete_outside_root_rejected`
(`:2235`) shows how to build a genuinely outside path.

**`test_too_small_cell_is_a_hard_error` (`test_periodicity_gate.py:219`) asserts
`"a" in str(exc.value)`.** The letter `a` occurs in "than", "cannot", and every
English sentence — verified to pass for a message naming the WRONG axis and for
one naming no axis at all. The comment above it records that the 2026-08-03
change existed specifically to stop pinning prose and start pinning *which
axis*; the replacement asserts nothing. Fix: `assert "along a" in msg`, or key on
the finding id as the rest of that file does.

### 6.3 Cut candidates — 22, and none applied

| slice | CUT | UNSURE | MOVE / redesign |
|---|---|---|---|
| A — `test_web_files`, `test_auth_config` | 16 | 4 | 1 (write the e2e first) |
| B — periodicity, spectra-json, modify, transport-prep | 5 | 5 | — |
| C — selection, jobset, rate-limit, projects-js, molstruct-json | 6 | 1 | 2 |

**Only one of the twenty-two is measured** (the tautological upload test above,
replayed verbatim). Every other verdict is read from the code path, and § 4
measures that reasoning at **1 wrong in 5** — so they are leads. The three
highest-confidence ones share a payload and a call rather than merely a code
path, and `tools/verify_subsumption.py` would settle them in about fifteen
minutes each.

**One verdict is worth reading for the reasoning rather than the outcome.**
Slice C had `test_specific_errors_inherit_base` on its cut list as an API-shape
assertion, then read the route: `/api/spectra/load` has exactly ONE
`except SpectraJsonError` and maps by type inside it, so re-parenting any of the
four classes turns a typed 404 into a 500 HTML page and nothing else in the suite
goes red. It stays, and its header now says why. That is the § 3b question
answered properly in the keep direction.

### 6.4 A file-level verdict worth keeping

`test_projects_public_surface_js.py` — 63 tests — reads from its name like the
API-shape file § 3b forbids. It is not: **54 of the 63 drive real behaviour** of
decisions the wrapper layer makes that `api.js` cannot (the refresh-on-success
policy and its parent derivation, the three-state `null`/`ReadOk`/`ReadErr`
contract, `navigateTo`'s pre-init fail-safe, `safeSave`'s abort→cancelled fold,
and the subscribe/publish semantics). **1** is the signature restated, **8** are
thin wrappers over a door tested elsewhere, and **3 are genuine cross-language
contracts that could break silently** — `relPath` computed on the JS side from a
route that does not return it, `actual_mtime` (a Python field name, dropped
once already), and the 409 status. That is the distinction § 3b asks for, made
per test rather than per file. **Every one of its 63 tests SKIPS** on this
machine (`node` not available), so those verdicts rest on reading.

---

## 7. What is deliberately NOT in this file

- **The applied cuts.** They are in the commit messages listed in § 0, each with
  the failure it would have caught and what now goes undetected. Restating them
  here would create a second copy to drift.
- **The science design findings.** `science/test-design-findings.md` owns them.
- **The rule itself.** `testing.md § 3b` owns it. This file is a queue against
  that rule, not a second statement of it.
