# Plan consolidation — 2026-10-08

**Role:** archive — the sections of `plans/plan.md` whose work was done, superseded or measured untrue, moved here on 2026-10-08 after nine read-only validation passes (one agent per section group; every claim checked against the code and docs text at the cited place, then re-read by the owner). Each section is kept verbatim under its original heading, with the validation verdict in front of it. What stayed open was carried back into the plan as one line under the section's pointer.

*(user, 2026-10-08: "clean up the plan to remove obsolete/done items so they can be archived correctly. we don't need shit piling up in the plan. validate those done items with agent and check for certainty")*

---

## 5a. A row is evidence of when it was written — re-derive before acting

> **Verdict (2026-10-08 validation):** the re-derive rule; its tool exists (`tools/classify_source_reads.py`). The rule lives in memory.


Twice now, this file's rows have been checked against the tree and most of
them had not survived. **The number is the point, not the rows.**

**2026-09-01, the roadmap's ~40 items.** Of everything it presented as open,
about a quarter survived contact with the code; the rest were closed and never
struck. It said so about itself twice before anyone acted on it — § 7.4's
*"carried as open for a day after they shipped"*, § 7.5's *"re-read against the
tree and most of it was already gone."*

**2026-09-06, nine of this file's own rows. Four held, five did not.** W11 was
done (both halves verified); W6 and W13 were overstated (two loaders not three;
160 px/rem in the page sheets, not 384 — and the bulk, `lib/`'s 740, was in
neither count). W2/W5/W8 are accurate exactly as
written, and two findings withdrew entirely: a deck assertion that never
executes (not reproducible under two mechanical searches) and a dropped
`max_memory_mb` unit (both halves already shipped).

**The rule.** Re-derive before acting, and **write the measurement into the
row** — every row corrected on 2026-09-06 now carries its own `file:line`, so
the next check is a check rather than a survey. A count with no stated
definition is not a measurement: 777 → 384 → 160 and 233 → 256 → 173 → 49 were
each three or four *scopes*, which is why the second one now has a tool
(`tools/classify_source_reads.py`) instead of a number.

---


---

## 5b. Open — found 2026-09-02, the run-decision round

> **Verdict (2026-10-08 validation):** every row done or history; R1 superseded by § 0c unit 8; T3 pinned (`tests/parse/test_round2_fixes.py:106-119`); B3.1/B3.2/B5 confirmed.


Two code-review agents and six test-audit agents, every acted-on claim verified
by hand. **Priority is the first column and means: P0 misleads a person right
now; P1 is a check that silently is not checking; P2 is a gap in what shipped
today; P3 is cleanup, large and mechanical.**

### The front end — what a person is told to run

| P | # | item | evidence |
|---|---|---|---|
| ~~P0~~ | ~~**R1**~~ | **DONE 2026-09-02.** `--from` is composed in the browser and was wrong for `flat`. | **this is V2 of the 2026-08-13 program, resurfaced.** Closed on the CLI, never on the browser |
| ~~P0~~ | ~~**R2**~~ | **DONE 2026-09-02, and UNVERIFIED** — no harness drives the stage panels, so the fix is read-checked only; a test needs **B3** and is tracked with **M2**. | verified |
| ~~P2~~ | ~~**R3**~~ | **DONE and PROVEN 2026-09-02.** Page state survived a folder change, against `task-setup.md` § 2.1's *"the page holds no state of its own"*: `_extraRunRows`, `_pendingDrop`, `_queue`. | verified |
| ~~P2~~ | ~~**D8**~~ | **DONE 2026-09-02.** | verified against all six surfaces |
| ~~P0~~ | ~~**R6**~~ | **DONE 2026-09-02.** | found by `test_task_setup_prep_e2e.py`; user: *"all environments have to be explicitly probed and stored. no environment json, error"* |
| ~~P1~~ | ~~**R7**~~ | **DONE 2026-09-02** *(user: "safety first … compatibility is not an issue. | pinned + mutation-tested |
| ~~P1~~ | ~~**T4b**~~ | **DONE 2026-09-02.** | pinned |

### Tests that are not testing

| P | # | item | evidence |
|---|---|---|---|
| ~~P1~~ | ~~**T1**~~ | **DONE 2026-09-02 — by making the claim TRUE, not by editing it away.** | verified |
| ~~P1~~ | ~~**T2**~~ | **DONE 2026-09-02.** Resolves from `__file__` like every other path in the file, and asserts the source list is non-empty so a future blind run says so. | verified |
| ~~P1~~ | ~~**T3**~~ | **DONE 2026-09-02.** Names `dataclasses.FrozenInstanceError`; mutation-tested by un-freezing the dataclass. | reported, spot-checked |

### Test bloat — measured, not estimated

**163 test files added, 36 deleted since 2026-08-01.** Whole files *do* get
retired; what never happens is pruning inside a surviving file — one commit
added 19 test definitions to `test_checkpoint_states.py` and removed none,
several of them written to *replace* pieces left standing. **That single habit
is 47 of 77 findings in one partition.**

**The rule this earns, and the only one that prevents the next round:**

> **A consolidating test deletes the pieces it consolidates, in the same
> commit.**

| P | # | item | size |
|---|---|---|---|
| ~~P3~~ | ~~**B1**~~ | **WRONG, and corrected 2026-09-02 — the tests stay.** | investigated in code + contract, not grepped |
| ↳ | **B3.1** | **DONE 2026-09-03 — the self-confessing subset.** A test whose own docstring says the real check lives elsewhere is retired, and the test it names gets written (`process/testing.md` § 3a.1, and `code-audit.md` § 5 rule 5 which was still *instructing* auditors to write these). Eleven confessed; **three were contrastive, not confessional** (`test_page_ids_unique.py`, `test_pages_no_js_errors.py`, `test_pdb_workflow_integration.py` all drive the artifact and cite string-pins only as what failed before) and one more survived reading (`TestSpectraIssuesPanelSeverityCoverage` is a CSS one-home lint whose confession describes the greps it replaced). **40 test functions removed** (38 deleted, 2 rehomed) **and 12 written** — 10 new plus the 2 rehomed. Every replacement mutation-verified. Every replacement mutation-verified. Two were *not* written and the reason is on record where each belongs: the second-load widget defect is unreachable by a single break (two publishers each rescue the other), and "a row is born at its value in force" states no rule any document carries | 40 |
| ↳ | **B3.2** | **DONE 2026-09-03 — reviewing the replacements found eight defects in them**, three substantive: a Slack channel was tested carrying a signing key (there is no such control — only a listener has one), an assertion demanded that NO part of a webhook appear when `MASK_TAIL = 4` shows the last four on purpose, and an `ok` check passed on an absent key. Two were vacuous (a vocabulary check that passes on a class with no tags; a chooser asserted to exist but not to offer anything) and three fragile (both timer tests counted the page's own intervals; a checkpoint picked by position not name; one dialog accepted where `_restore` asks twice). A green run plus one mutation had hidden all of it. Asserting the page reports no JS errors also found a **live product bug** — `pattern="[A-Za-z0-9_-]{1,64}"` never compiled under the `v` flag, so the channel-name rule was stated and never enforced (`d243e852`) | 8 |
| ~~P3~~ | ~~**B5**~~ | **DONE.** The shadowed definition is gone, and `tests/test_no_test_is_shadowed.py` makes the next one fail instead of disappearing. | 46 lines |

### Gaps in what shipped today — mine

| P | # | item |
|---|---|---|
| ~~P2~~ | ~~**M1**~~ | **DONE.**
| ~~P2~~ | ~~**M2**~~ | **DONE 2026-09-02** , and NOT the way this row expected.
| ~~P2~~ | ~~**M3**~~ | **DONE — and the fix was the OTHER direction.**

### Documents that state something false

| P | # | item |
|---|---|---|
| ~~P0~~ | ~~**D4**~~ | **DONE 2026-09-02** — and the function's OWN docstring was stale the same way, still describing the both-keys rule for a second shape that no longer exists.
| ~~P1~~ | ~~**D5**~~ | **DONE 2026-09-02**
| ~~P1~~ | ~~**D6**~~ | **DONE 2026-09-02.** Six labels, `buffer` included, in § 6.6 and § 13.3; no test hard-coded the count, so only the document was wrong.
| ~~P2~~ | ~~**D7**~~ | **DONE — the TEST was wrong, and its exemption was protecting something else.**

### The 2026-08-13 V-program — closed, with one survivor

Task #11 carried V1–V24 from `memory/project_final_fix_program.md`. Spot-checked
against the tree on 2026-09-02: **V1, V3–V7, V19, V22, V24 verified closed**
(one `_stage_bench_dir` owner; `continue_retries` on `Resources`;
`validate_ladder` gone; the config split resolved; floor-2 gates filter on field
metadata; the named dead symbols absent; `bench/` is the unified jobset stack).
**V2 is open and is R1 above** — closed on the CLI, never on the browser, and
found again by a fresh agent three weeks later. The tier-2 and tier-3 items
(test-logic, doc sweeps) are superseded where they overlap by §§ 5b's test and
document rows, which were measured against the current tree.

---


---

## 5c. The directory door — `JobDirParser`, and its migration

> **Verdict (2026-10-08 validation):** wholly superseded — `JobDirParser`, `RunDirResult`, `parse_dir` have no hit; the directory door is `runs.folder_answer` (`molbuilder/runs.py:429`); `parse/registry.py:61-66`; `docs/model/parse.md:69-74`. 28 confirmed, 16 obsolete, 11 partly, 4 refuted (fixed since), 2 unverified.


*(History since 2026-10-04: the directory door is the run door's
`runs.folder_answer`, and `JobDirParser`, `parse_dir` and `RunDirResult` are
gone -- plan B11, B14, W56 unit 3a.)*

*(Agreed 2026-09-04. Contract: [`model/parse.md` § 5](?doc=model/parse.md).
This section is the plan, and the caller list below is the completeness check
— the requirement is that nobody is left behind.)*

**TWO DIFFERENT THINGS SHARE THIS NAME, so read the state carefully.** The
`JobDirParser` that *existed* — an eleven-field `JobResult`, ten of whose
fields had no reader anywhere and whose eleventh was reached by parsing every
result file to build plots and then discarding them — **was DELETED
2026-09-04**, replaced by `job.run_status`. That work is done and is not what
follows.

**What follows is a NEW consolidation. Steps 1 AND 2 are DONE (2026-09-18);
§ 5c.2 reopened the section the same day.**
`parse/types.py` carries `RunDirResult`, `parse/dirs/rundir.py` carries
`JobDirParser` + `openable_in`, the parser is registered, and
`parse_dir(<a run directory>)` answers. `web/blueprints/watch.py` calls
`openable_in` and its own copy of the chain is deleted.

> **Step 2 moved ONE of the six sites, and withdrew the other five with
> measurements.** The caller map below was written from the names — six
> functions that all take a run directory — and five of them turned out not to
> be asking a run-directory question. The withdrawals are in the table itself,
> each with the measurement that settles it. **This is the third time this
> section's table has lost rows to re-reading the code** (two went on
> 2026-09-04), and the pattern is the same each time: a function whose
> ARGUMENT is a directory is not thereby answering *a question about the
> directory*.

**The shape.** One DirParser answers everything asked *about a run
directory*; the four fields each have a named reader before a line is
written (§ 5.0's table). `active` is picked **stage, then mtime** (user
ruling) — `summarize`'s highest-`-runN` rule loses, because a run index says
nothing about which stage a file belongs to.

### The complete caller map, measured

Only **three** modules consume any of this, which is what makes the migration
checkable rather than hopeful:

> **⚠ THE MAP WAS BUILT FROM NAMES, AND THAT IS ITS ONE WEAKNESS** *(found
> 2026-09-18, by a full-text audit of the four caller files rather than by
> grep)*. Every row below was found by searching for `run_status`,
> `_resolve_run_directory`, `engine_of`, `atom_metadata_json_for_run_dir` —
> so a module that answers the same QUESTION under its own names was never a
> candidate to appear, let alone to withdraw. `transport/record.py` is
> exactly that, and it is code written for this very milestone two days
> before the audit. **A completeness check over function names cannot see a
> re-implementation**; the row below was added by reading the files.

| what it does today | where | outcome, measured 2026-09-18 |
|---|---|---|
| `_resolve_run_directory(dir)` — the 4-rung chain | `web/blueprints/watch.py` ×1 | ✅ **MOVED** to `.openable` + `.attempts`. The only true duplicate in the list: 132 lines of second copy, plus five helpers serving only it — **166 lines deleted from the web layer**. |
| `run_status(dir)` | defined `parse/dirs/job.py`; called from `jobset/runstatus.py` ×1 | ❌ **WITHDRAWN.** It passes `out_glob = sh.stage_glob(token, label)` — `"*"` in the hierarchy but **`<label>_<token>*` in flat**, where one directory holds every rung. `parse_dir` has no narrowing, so flat would go back to *every stage row showing the newest stage's state* — the exact defect fixed 2026-09-08. And there is nothing to de-duplicate: `run_status` already has one home in the parse layer, which this section itself noted on 2026-09-07. **Asking a directory door a per-RUNG question is § 5c.1's problem wearing a different hat.** |
| `_engine_of(...)` | `web/blueprints/watch.py` ×3 | ❌ **WITHDRAWN.** Not a reader — a three-source fallback (`engine_of` → `payload["source_format"]` → `parser_cls.name`) whose first source *is* the one implementation. **Two of the three sites pass `search_dir=None`**: an upload has no run directory at all. Routing it through the door means parsing a whole directory — status over every `.out`, the four-rung chain — to obtain one string. That is precisely what got the predecessor deleted. |
| `engine_of(dir)` | `web/blueprints/watch.py` ×1 | ❌ **WITHDRAWN** — it IS the one implementation (`running-a-job.md` § 4.2 owns the rule). `.engine` calls it; a caller that wants only the engine should keep calling it too. |
| `atom_metadata_json_for_run_dir(dir)` | `web/blueprints/watch.py` ×1 | ❌ **WITHDRAWN** — already in `parse/dirs/`, already one home, its own verb. Same boundary as `contract_of` below. |
| `summarize`'s per-trial reads | `jobset/summarize.py` ×4 | ❌ **WITHDRAWN**, and the prose two sub-sections below already said why without the table catching up: `_latest_run_file` goes **through `runfiles.find`**, the stage is already chosen by the basename, and the only remaining choice is the RUN INDEX. Not `active`'s question. |
| `contract_of(dir)` | `parse/dirs/run_info.py` ×1 | unchanged — its own verb, over `.files["fdf"]` if the door is handy |
| **`transport/record.py`** — `_stage_facts` ×1 + `collect_record` ×1 | **WAS IN NO ROW** *(added 2026-09-18)* | ⬜ **the map's own blind spot.** Both pick the newest `.out` by **mtime alone** — a third rule beside the door's (stage, mtime) and `summarize`'s run index. **Not a defect**: `task.py` refuses a transport calculation whose shape is not hierarchical, so each rung has its own directory and there is only one stage to order — the rules coincide. But it is a third place the question is answered, and § 5.1's ruling names only two. **Nor is it a drop-in move**: `RunStatus` carries `state / detail / last_change_at / active_source` and `_stage_facts` needs `scf_converged`, the final energy and the lead `ef` — so the door would pick the file and the caller would still parse it. |
| ~~**the Results file picker** — `lib/results/file-picker.js`~~ | **the BROWSER** | ❌ **STRUCK 2026-09-18 — NOT A CALLER OF THIS DOOR, and it never was.** Its question is *"these five directories are one run"*, which is the **ladder's**, not a directory's; `openable` answers about one directory by construction. `stages.md` § 6.7 puts the layout in `task.json` and forbids inferring it from data, so `JobDirParser` — handed a bare path — **cannot** answer per-rung and must not be extended to try. The ladder door already exists and already reads the declared shape: `jobset/runstatus.py::jobset_status`. **What was PREDICTED here did not happen, and the prediction is the interesting part.** This row said the picker needed *"an HTTP surface over that"* — over `jobset_status`. What shipped on 2026-09-18 is `GET /api/results/dir` over **`parse.dirs`**: the blueprint imports `openable_in`, `run_status`, `labels_in`, `read_back` and `engine_of`, and imports nothing from `jobset` at all (verified 2026-09-20 — its one `jobset` mention is a comment). The per-directory door turned out to be exactly what the picker needed, because the picker lists ONE directory at a time. **The same wrong sentence had to be corrected in `web/results.md` § 2.3 and `web/presenters.md` § 1 on 2026-09-19/20**, where it was actively telling a reader to go build a route that already existed — this is its third home. *(Listed here as "the seventh consumer" from 2026-09-18 until later the same day, which made a finished migration read as unfinished every time its status was asked.)* |

> **What step 2 therefore proves about step 1.** The door earned its keep on
> exactly one site, and that site is the one that could not be fixed any other
> way: a browser cannot import Python, so the chain had to leave the web layer
> before the Results tab could ever ask it. Every other row was a name
> collision. Had the map been executed as written, five call sites would have
> been made slower and one of them wrong.

> **The seventh consumer is a BROWSER, and every other row here is
> server-side** *(added 2026-09-18, owed by § 5p.3p step 7 since 2026-09-17 and
> deferred twice while other work went in)*.
>
> The picker enumerates a folder and decides what to list **from filenames, in
> JavaScript** — it asks no server door, because none answers *what is in this
> directory*. `/api/results/contract` answers `info.calculation` and five other
> fields, none of which is *what should a viewer open here*.
>
> **That guess is why a finished transport run is invisible.**
> `<label>.transport.json` matches no presenter, so the picker drops it; the
> five rungs' `.out` files are each claimed by the trajectory viewer, so the
> ladder lists as five unrelated optimizations (§ 5p.3p.3, items 1–2).
> Transport is not the cause — it is the first result that is neither a
> trajectory nor a spectrum, so the guess stops producing a plausible answer.
>
> **What this row costs § 5c:** the door needs an HTTP surface, not just a
> Python one. Every other consumer imports it; this one must fetch it.

#### 5c.1 `openable` is one answer per DIRECTORY, and a ladder is five

*Owed by § 5p.3p step 7, same date, same deferral.*

`RunDirResult.openable` answers *what should a viewer load here*, once, for one
directory. **A transport run is five directories that are one result** — seed,
both leads, device, transmission — each a real SIESTA run with its own `.out`.

Neither mechanism in play can say so. `absorbs` (`web/presenters.md` § 2)
collapses siblings **within one directory**; it is asked *"does this master
subsume that sibling?"* and has no way to express *"these five folders are one
run."* `RunDirResult` as § 5.0 specifies it is per-directory by construction.

**This is not transport's question.** It reaches every multi-rung calculation —
a staged optimization is the same shape and only looks fine today because each
rung's log is independently interesting. It needs a home **before** code moves,
because the answer decides whether the door returns one result per directory
(and something above composes them) or learns about runs that span several.

**It is genuinely open.** No recommendation is recorded here on purpose: the two
shapes differ in what the door IS, not in how it is written.

**Two rows were in this table and are removed, both invented by me and
both caught by re-reading the code** *(2026-09-04, second pass)*:

* `_read_system(bundle)` takes the **bundle root** — the calculation
  folder holding `task.json` — reads the DESCRIPTION first, and falls back
  to root decks. It is not a run-directory question at all.
* `_latest_run_file(d, base, suffix)` is called four times for four
  artifact KINDS (`scf-timing.log`, `monitor.log`, `util.csv`, `out`), and
  its `basename` is `Path(j.script).stem` — for a staged deck
  `<label>_<NN>_<stage>`, so **the stage is already chosen** and the only
  remaining choice is the RUN INDEX. That is not `active`'s question
  (*which result file across stages carries the status*), and mapping them
  together would have applied a stage rule where stage is not a variable.

Internals absorbed rather than migrated: `_enumerate_files`, `_build_status`,
and `contract.py`'s three declaration rungs.

### Two boundaries, decided

**`attempt_concluded` stays out** — `jobset/prep.py`, `jobset/submit.py`,
`transport/compose.py`. It answers *"may I launch here"*, its consumers are
WRITERS, and folding it in would make the launch path depend on the parse
layer for a decision that is not a reading.

**Sibling lookups stay out** — `_sidecar._siesta_fdf_path_for`,
`pyscf._resolve_job_token`, `siesta_mdnc.sibling_md_nc`. § 5a's sibling
upgrade: a parser locating one file from another *of its own format*.
Absorbing them inverts § 5's rule that a DirParser composes FileParsers.

### The behaviour changes, named in advance

**There are none to the numbers.** An earlier draft of this section said
`summarize`'s per-trial pick would change from highest-`-runN` to
stage-then-mtime; that was the invented mapping above, and the claim is
withdrawn. `active` is stage-then-mtime (user ruling) and it governs the
STATUS only, which has always used that rule.

The one visible change: **`detect()` on a directory starts resolving
again.** It had no DirParser between 2026-09-04 and 2026-09-18 and could only
refuse. *(**That measurement was WRONG, and it cost three CLI verbs.** It said "no
call site in the tree passes a directory to `detect()`" and named three that
pass a concrete file. There was a FOURTH: `cli.py`'s `runtime-info`,
`watch parse` and `watch tail` hand `detect()` whatever path they are given
and then use the result as a trajectory. Registering `JobDirParser` turned
their clean `UnknownFormatError` refusal into an `AttributeError` traceback —
and the first guard written for it raised inside a poll loop that retries
`ParseError`, turning the traceback into a silent INFINITE HANG. Both fixed
2026-09-18. **Registering a parser is not additive: it changes what
`detect()` answers for every caller that does not narrow.**)*

### What `summarize` actually gets, which is less than first claimed

`_enumerate_files` buckets ENGINE artifacts only — `.fdf`, `.out`, `.XV`,
`.STRUCT_OUT`, `.molstruct.json`, `.ANI`, `.molwatch.log`. Three of the four
files `summarize` reads per trial are the WRAPPER's instrumentation
(`scf-timing.log`, `monitor.log`, `util.csv`) and are in no bucket.

**DECIDED AND DONE 2026-09-04** (user: *"yes give them parsers"*). The
three wrapper files are registered parsers now — `scf-timing`,
`monitor-log`, `util-csv`, returning `InstrumentResult`
([`parse.md` § 5c](?doc=model/parse.md)) — and `bench/result.py` no longer
opens a file. So the door CAN learn them: `_enumerate_files` gains three
buckets and `summarize` reads through `.files` like everyone else.

That still leaves the migration at TWO consumers and eleven sites for the
DIRECTORY door itself, because `summarize` finds its files by
`_latest_run_file` (run index, for an already stage-scoped basename) and
that is not `active`'s question — see the withdrawn row above:

* `web/blueprints/watch.py` — 9 sites
* `jobset/runstatus.py` — 1 site
* (`contract_of` unchanged)

### Order of work

1. ~~Write `RunDirResult` + `JobDirParser`, absorbing the chain verbatim.
   Prove equivalence on the real tree BEFORE any caller moves — the
   `run_status` split did this (113/113 identical) and it is the reason that
   deletion was safe.~~ **DONE 2026-09-18.**

   The proof ran over every run directory in `projects/`: `JobDirParser.parse`
   against `_resolve_run_directory`, `run_status` and `engine_of` —
   **identical 141/141**. It found two real absorption bugs first, which is
   what a verbatim gate is for: rung 4's attempt messages had been
   paraphrased, and rung 4's `*_geom_optim.xyz` had been rewritten to
   `find_by_role`, which *raises* on an underscore role with no label — the
   exact case rung 4 exists for. Both fixed, then 141/141.

   Registered in `parse/dirs/__init__.py` (the registry's own rule: *"per-package
   `__init__.py` files own the registration order"*). Three tests in
   `tests/parse/dirs/test_rundir.py` hold what step 1 ADDS — that the registry
   answers, that a prepped-but-unrun stage is still a run directory, and § 5.1's
   `active` ≠ `openable`; the chain's own rungs stay held by the three tests in
   `test_path_framework_doors.py`, which move onto `openable_in` at step 3.
2. ~~Move `runstatus` (1 site), then `watch` (**6**, not 9), then
   `summarize` (5).~~ **DONE 2026-09-18 — one site moved, five rows
   withdrawn with measurements** (the table above carries each one).

   `watch`'s `_resolve_run_directory` call became `openable_in`. The other
   five asked questions this door does not answer, and the measurements are
   in the table rather than here so the map and its verdict cannot drift
   apart.
3. ~~Delete the absorbed functions and sweep their documents.~~ **DONE with
   step 2**, since only one function was absorbed: `_resolve_run_directory`
   and its five private helpers are gone, and `job-contracts.md` § 2.4,
   `model/parse.md` § 5.2, `process/testing.md`'s door map and the three
   tests in `test_path_framework_doors.py` all name `openable_in` now.
4. ~~**Re-run the caller map and require it empty.**~~ **DONE 2026-09-18 —
   it comes back empty.** One row moved, five withdrawn with measurements,
   and the seventh (the Results picker) struck as never having been a caller
   of this door. ~~**§ 5c IS CLOSED.**~~ **REOPENED 2026-09-18 by § 5c.2 (open list: N8)** —
   the map asked *who calls this door* and never *what the door looks for*,
   and the door could not see PySCF's output at all.

   The picker's work is real and is `§ 5p.3p`'s: an HTTP surface over
   `jobset_status`, the ladder door that already exists.

   *(Two method failures are recorded above rather than quietly fixed,
   because both would repeat: the map was built from FUNCTION NAMES, so it
   could not see `transport/record.py` answering the same question under its
   own names; and § 5c.1 was posed as "should `RunDirResult` learn about
   runs spanning several directories" when the answer was already on the
   books — something above it composes them, and that something is
   `jobset_status`.)*

### 5c.2 Migrate onto `model/parse.md` § 5.5

*Reopens § 5c, whose completeness check asked WHO CALLS the door and never
WHAT THE DOOR LOOKS FOR. Open list: N8 (this) and N9 (the bundle).* *(N8 closed
2026-09-18; N9 partly — its row.)*

**The contract is [`model/parse.md` § 5.5](?doc=model/parse.md).** It settles
the design; this section is the migration and the evidence.

#### Why — measured

PySCF writes its stdout to `.pyscf.log`. Nothing collects it and nothing
reads it.

| | |
|---|---|
| PySCF run directories in the tree | 12, **all carrying an end marker** |
| of those, reporting `running` | **3** — oldest **97.6 days** |
| of those, `openable_in` returns `None` | **3** |
| `.pyscf.log` files any registered parser claims | **0 of 13** |

**The 9 that answer correctly do so from the molwatch footer, never from the
engine's own stdout.** The 3 that fail are **spectrum decks, which write no
molwatch log at all** — so the only evidence of how they ended is the file
nothing reads. That is why it is exactly those three.

#### Two questions, not one — and this corrects an earlier draft of this row

*What is a run's output* is the catalogue's. *What can a person open* is the
registry's. `.pyscf.log` is output and no parser claims it; `.spectra.json`
is what a person should see in 2 of the 3 broken directories and is **not**
output. An earlier version of this section told the discovery chain to "ask
the catalogue", which would have handed the viewer a file `detect()` refuses
and still answered `None` for both spectrum directories.

**And the third directory's `None` is not a role-list bug at all.** Rung 3
composes `<stem><role>` **with no attempt counter**, so it looks for
`water_01_coarse.pyscf.log` while the file is
`water_01_coarse-run0.pyscf.log`. `runfiles.find()` already returns every
attempt sorted by run index; the chain hand-rolls `compose` + `isfile`, which
is the door `find`'s own docstring says it replaces.

#### Order of work

Tests before code at every step: the mutation that restores the original bug
currently leaves 417 tests green.

| # | do | proves |
|---|---|---|
| **0** | revert the uncommitted attempt — it is a SECOND catalogue, stating the vocabulary in `_run_ending` that `runfiles.WRITTEN` already holds. Keep the `test_siesta_use_gpu.py` first-vs-last correction; re-land its `_pyscf_ending` and marker constants in step 2 | tree is HEAD + one correct test |
| **a** | `runfiles`: the `output` column, 3 rows, `.out` gains `engine="siesta"`; `run_output_roles()` / `stdout_roles(engine)` / `engines()` | a declaration. Gate: `manifest(…, engine="pyscf")` stops naming `.out` |
| **b** | the emitters name their end lines as constants; `_run_ending` grows the PySCF reader **importing them**, and `ending_of(path)` dispatching on `runfiles.canonical_role`; assert `set(READERS) == set(run_output_roles())` | the end text exists once. **Mutation gate:** change the emitter's constant → the reader's test fails |
| **c** | `run_status` collects by `run_output_roles()`, reads by `ending_of`; the two conclusion readers collapse to one loop | **the 3 directories answer `finished`.** Gate: re-measure all 141 — the only differences are those 3 |
| **d** | liveness → freshest run-output mtime; the speaker rule unchanged | a killed run reads `running` while the progress log is fresh, `stale` once nothing grows |
| **e** | derive the engine→role map: `contract._STDOUT_SUFFIX`, `contract._ENGINES`, `_from_cluster`'s `.fdf→siesta`, `runwrap`'s `_ext`, **and `pyscf/input.py:261`'s banner** | one spelling of PySCF's stdout, for readers **and writers** |
| **f** | `runstatus._OUTPUT_ROLES` → `run_output_roles()` | it expands to **11** roles today, including `.parse.log` — so a queued rung flips to `running` the moment molbuilder reads its directory. Latent (0 in `projects/`), one Watch poll away |
| **g** | LANDED 2026-09-18 — `openable_in` answers by DELEGATION, not a ladder: the calculation (task.json) picks the role, the catalogue names it (`result_roles`), the registry vets it (`detect()`). Fixed on the way: `runfiles.find` instead of a hand-rolled `compose` with no attempt counter, and gather across every deck before offering (role decides WHAT, mtime decides WHICH) | **sweep: 5 of 109 change, all 5 intended** — both found spectrum dirs `None`→`.spectra.json`, the generated CO2 spectrum stub→its spectrum, and the 2 predicted `slurm.*.out` false positives→`None` (no file in those dirs is claimed by any parser) |
| **h** | **REWRITTEN 2026-09-18 — the previous row said "delete `parse_dir`, `RunDirResult`, `JobDirParser`, the `DirParser` ABC", which contradicts N9 and § 5.5.** The door STAYS. What this step does: **wire the consumers to it** (`web/blueprints/watch.py`, the spectrum surface — which today takes a file path and cannot open a DIRECTORY at all), and decide the route: `detect()`/`parse_dir`, or a direct call. `cli._trajectory_parser_for`'s directory guard stays either way — those verbs ask `answers_a_trajectory()` now, which is the correct fix | the door has callers; no consumer names a reader by hand |
| **i** | DONE 2026-09-18 — swept `model/parse.md` § 5.2 (the ladder was replaced, and the section still described four rungs under the title "unchanged in behaviour"), § 5.4 + `base.py` + § 7's restatement ("dispatch each file through the registry" — which the code stopped obeying for a measured reason, so the RULE was the wrong one), § 5.0's no-consumer note, and `process/testing.md`'s door map (`openable_in` is no longer a THIN caller, by that section's own rule) | rule first, restatements second — and where code and contract disagreed, the contract was re-derived rather than the code bent to it |

#### Validation — against runs we WATCH FINISH, not found directories

The 141 directories in `projects/` are a **regression sweep only**: nobody
established what state they are in, so agreeing with them proves inertness.
Correctness is checked against jobs driven through the real UI to a known
ending — **CO2 optimization and spectrum, in both flat and hierarchical
layout** — so every expected answer is one we watched happen.

| # | check | today |
|---|---|---|
| V1 | a finished run → `finished`, from its own stdout, with no molwatch footer | 3 say `running` |
| V2 | the same directory → `openable_in` returns the file a person should see | `None` |
| V3 | a killed run → `running` while the progress log is fresh; `stale` once nothing grows | cannot be asked |
| V4 | a run whose optimizer package is missing (`SystemExit`, **no traceback**) → `failed` | cannot be asked |
| V5 | deleting `.pyscf.log` from the catalogue makes a test FAIL | **417 stay green** |
| V6 | `set(READERS) == set(run_output_roles())` | neither exists yet |
| V7 | `run_status` median on the largest real PySCF directory | 21.6 ms — target < 3 ms |
| V8 | a third engine is two edits | fourteen sites |

#### Needs a ruling

**`openable`'s preference order below rung 1.** 9 of 141 directories change;
2 are today's false positives. The other 7 need a rule: does a directory whose
only parseable file is a `.XV` or a hand-made `.xyz` offer it, or answer
`None`? The generated CO2 jobs above are what this should be decided against.

*(Settled, previously listed here in error: `.pyscf.log` does **not** get a
FileParser. The frames live in `_geom_optim.xyz`, which already has one;
`.pyscf.log` is the stdout capture, and stdout capture answers how a run
ended — `ending_of`'s question, not the registry's.)*

---

### 5c.3 A CALCULATION ROOT IS NOT A RUN DIRECTORY — the transport ladder has no view *(2026-09-18)*

> **The order of work is § 5u's since 2026-09-29** (transport consolidated, W43 / M5). This section keeps the *why* and the evidence; its own order-of-work table is history.

*Found by driving the real Results tab in a browser after § 5c.2 step (h)
landed. No test asserts any of it, because every test asks about a RUN
directory and this is about the folder above one.*

#### The measurement

`projects/Au-BDT-Au/transport/AuBDTAu-CT` is a transport calculation root: a
`task.json`, a `job-set.json`, the composed `junction.xyz`, and five stage
directories. Asked about that one directory, the two readers disagree
completely:

```
jobset_status  -- the LADDER reader
  complete=False  first_incomplete=seed
    seed          pending   prepped, not launched (no run.json)
    electrode_L   pending   prepped, not launched (no run.json)
    electrode_R   pending   prepped, not launched (no run.json)
    device        pending   prepped, not launched (no run.json)
    transmission  pending   prepped, not launched (no run.json)

run_status     -- the RUN reader, same directory
    running   (no result file yet)
```

`jobset_status` is right. `run_status` is answering a question the directory
was never asked — **and it is what `/api/results/dir` reports**, because that
route (§ 5c.2 step h) calls `run_status` unconditionally. So the Results tab
tells a person a calculation that has never been launched is *running*.

What the tab offers there, from the door:

```
openable : None                                  <- correct: no record exists
offerable: junction.molstruct.json, junction.xyz  <- the INPUT structure
           ...and 8 files it cannot read
```

The five stage directories are in the sidebar and **the Results tab has
nothing to say about them.** It offers the structure the calculation started
from as its result.

#### What the design already requires, and has since before this was built

`engines/transport.md` § 2a.12 names three things the Results surface reads,
none of which exists:

| | |
|---|---|
| **the curve, with its treatment NAMED beside it** | linear-response and finite-bias I–V "are different claims and look identical on a plot" (§ 2a.10). The record carries `treatment` for exactly this |
| **the provenance chain** | which device run, which lead runs, which relaxation the junction came from — "a transmission curve without its chain cannot be interpreted, reproduced, or compared" |
| **the ladder's state** | "a transmission that has not run yet is *pending*, never a failure of the calculation, and a reader needs to see which of five runs is the one still outstanding" |

The reader for the first two now exists — `parse/sidecars/transport.py`, added
2026-09-18, and `result_roles("transport")` names the record so the door
offers it. **The third has no surface at all.**

#### The one thing to decide, and it is not a layout question

**A calculation root and a run directory are different kinds of thing, and
the Results tab has only one notion.** Everything else follows from how that
is answered:

* a RUN directory holds one attempt's output → `run_status`, `openable_in`,
  one file mounted in an inspector;
* a CALCULATION root holds a *plan* and N stage directories → `jobset_status`,
  a ladder, and a deliverable that is written once at the root.

**The question already has an implementation**, and it is not in the parse
layer: `checkpoint.py::_is_bundle_root` — *"does `path` declare itself the
root of one multi-directory unit of work?"* — testing for `task.json` /
`job-set.json`, with both names imported from the modules that write them.
It is private to `checkpoint`. **Giving it a public home is the fix; a second
copy in `parse/dirs` would be instance 14 of § 8**, in the week that section
was written.

#### Order of work

Tests before code at every step, and **the validation is a browser walk, not
a directory sweep** — this defect was invisible to 2 800 passing tests.

| # | do | proves |
|---|---|---|
| **a** | give `_is_bundle_root` a public home and one owner; `checkpoint` asks it | one answer to *"is this a calculation root"*, not two |
| **b** | `/api/results/dir` asks that first: a calculation root answers with `jobset_status`'s ladder, a run directory with `run_status` as now | the transport root stops reporting `running` for a calculation nobody launched |
| **c** | **✅ done 2026-09-24** as `ladder: {complete, first_incomplete, stages: [{name, seq, state, detail, dir, attempt}]}` — the payload gains the ladder for a root, `null` for a run directory | the browser can render a ladder without a second route or a second scan |
| **d** | **✅ done 2026-09-24** — the empty-state card's ladder table: one row per rung, the bench summary's state chip, the rung to resume from in the title (`results.md` § 2.4) | § 2a.12's third requirement, and the one a person needs while a five-stage run is in flight |
| **e** | the transport inspector renders the record through `transport-json` rather than `JSON.parse`, with `treatment` NAMED beside the curve | § 2a.10 / § 2a.12's first requirement. The parser landed 2026-09-18; the inspector still parses in the browser |
| **f** | the provenance chain from `record.provenance` | § 2a.12's second requirement |

#### Deliberately NOT in this section

* **The frame axis** (§ 2a.9). The record's shape is already additive and
  § 2a.11's axis rule keeps the directory tree additive; a family of curves is
  a later row, not a rewrite, and pretending to design it now would be
  designing against no data.
* **Any change to the transport TAB** (W24–W27). This is the Results surface
  reading a finished calculation; that is the parameter surface setting one up.
* **A second status vocabulary.** `jobset_status` already answers in
  `_stage_state`'s eight words and `bench-summary.js` already tones seven of
  them. The ladder view reuses both or it is a third copy.

#### What this rests on, so it can be re-derived

Every number above was produced by running the code on 2026-09-18, not read
off a comment: the two readers' disagreement, the door's `openable: None` and
its two offerable files, and the absence of any `.transport.json` in
`projects/` (0 of 110 run directories, and none at any calculation root).

---



---

## 5f. Architecture seams — dropped by the consolidation, recovered 2026-09-06

> **Verdict (2026-10-08 validation):** its rows live in § 2 (S1, S13) or are closed; warm-file rules built as `warmfiles.warm_list` (not `rules_for`/`inventory`); S14 evolved (`parse/dirs/job.py:287`); S10-M2a built (`scheduler/emit.py:42`).


**How this section came to be missing.** The 2026-09-01 merge fact-checked the
roadmap's §§ 3, 4, 7.4c, 7.5, 7.8, 7.10 and 7.11 and the two audits — and
**never reached § 6, *Architecture seams***. So § 6's bullets were neither
verified nor carried, and § 5a's *"of roughly forty items… the ones that
survive are the twenty-odd rows above"* was written without them in view.
Nothing pointed this out for five days because **R3's own sentence still sent
readers to the archive**: when `roadmap.md` was archived, the line in
`docs/README.md` was re-pointed at its new path rather than at the file that
replaced it, and **53 pointers across 31 documents followed it in**. Fixed
2026-09-06; the pointers now name this file.

**ALL FOURTEEN RE-DERIVED 2026-09-06** *(user: "verify the 14 seams in
5f")*, and the ratio is the finding again.

> **⚠ THIS PARAGRAPH CONTRADICTED ITS OWN TABLE UNTIL 2026-09-07**, on four
> rows of fourteen. It counted **S8** and **S11** among "confirmed open" —
> the table marks both **WITHDRAWN**, and I wrote those cells. It counted
> **S14** among "already closed"; the table marks it **OPEN** and says the
> 2026-09-05 ruling made it sharper. And it listed **S12** as open where the
> cell reasons it closed. A summary that disagrees with the evidence
> directly beneath it is worse than no summary: this is the shape of the row
> that survived a whole day in § 4 because nobody re-derived it. Corrected
> below from the cells, not from memory.

**Three are CLOSED** (S2, S5, S12), **two are WITHDRAWN** as not-work-items
(S8, S11), **one is mostly wrong** (S4), **four are confirmed OPEN** (S1, S3,
S6, S13), **S9 is now DONE** — the two documents
stopped disagreeing on 2026-09-06 and `backend-architecture.md` § 2
retracts the claim in writing — and **two state cells describe the wrong
seam**: S10's answers S9's question, and S14's describes S12's GPU guard and
asserts it "does not exist" when S12's own cell shows it does.

> **Both unrecorded states are now measured — 2026-09-10**, and both are
> closed; the rows are in
> [`archive/2026-09-10-plan-consolidation.md` § 6](?doc=archive/2026-09-10-plan-consolidation.md).
> **S14** (a flat stage's verdict read folder-wide) was fixed 2026-09-08:
> `parse/dirs/job.py::run_status(run_dir, match)` narrows the bucket and
> `jobset/runstatus.py:267` passes the same glob the existence check used,
> with the before/after in the docstring. **S10-M2a** (the record naming one
> partition while the header submits to another) cannot happen through the
> placement door: `scheduler/emit.py::Directives.of` takes the partition and
> QoS **from a `Placement`**, and `place(..., named=…)` refuses a domain the
> machine record does not offer — checked against real Sol output, where
> `bench-group-gpu-G2K24C1.sbatch` carries `-p htc -q public -t 0-04:00:00`
> and the record's `htc` row reads `partition: htc, qos: public,
> max_time: 4:00:00`. M3's `account` half has **no door to declare one**:
> `runtime_config` has no `scheduler.account` key, nothing emits `#SBATCH
> -A`, and `configuration.md` § M-1 already records `Site.account` as a
> field *"nothing has ever written"* — so the missing record is the missing
> feature, and there is one fact, not two disagreeing.
>
> **The lesson is the same one § 5a states.** The two rows nobody could close
> were exactly the two whose state cell had been misfiled onto a neighbour —
> a row with no evidence of its own is a row that stays open forever. The line counts S11 quoted — `render_run_wrapper` grew to 2,134 lines and 499 f-strings
while carried as a stable ~1,780 / ~295. Three were measured when the section
was written (S1, S6, S13) and stand. Two halves are still unchecked and say so
(S7's P1/P5, S10's M2a). *(2026-09-29: S6 and S7 re-measured closed and archived —
`archive/2026-09-10-plan-consolidation.md`, the last section.)*

**Read the state column literally.** *measured* / *verified* means re-derived
against the tree and the evidence is in the row — § 5a's lesson was that ~85% of
what a roadmap carried as open had already shipped, so assume the same here
until each is re-derived. Numbering is `S`, not `W`, because
`backend-architecture.md § 5` already has a **W1–W5** of its own and this file
already has a different **W1**; the two collided in every conversation about
them.

*The rows that were here are in [§ 2](#2-open--the-one-list) — one list since 2026-09-10. The design above them is the substance and stays here.*

**Closed on the way in, 2026-09-06.** Roadmap § 6's *warm-file rules file*
bullet is **built** and its pointer is retired: `molbuilder/warmfiles.py` is the
one reader (`rules_for` type-scoped, `inventory` type-blind), and both engines
ship `warm-files.toml`. `job-contracts.md` § 4.2a's heading said
*"implementation tracked in [the roadmap]"* until today.

---


---

## 5h. The source-reading assertions — the remaining work list

> **Verdict (2026-10-08 validation):** 0 to convert; items 1–7 all confirmed in the code (`29b655ec`; `runwrap.py:567,620`; `core.js:1739,2022,110`; `tests/_node_esm.py:37`).


*(Supersedes **B3**'s estimate. The method and the lint-vs-pin rule live in
[`process/testing.md` § 6](?doc=process/testing.md), which owns the boundary;
the instrument is `tools/classify_source_reads.py`. **Run it — do not quote a
number from here.**)*

**Measured 2026-09-08 after the sweep: 0 to convert, 57 to keep** (was 64 to
keep, 0 to convert, on 2026-09-07 — the "3 to convert" that stood here was a
mis-read of a run whose overrides had come unanchored; see below).

**The sweep, 2026-09-08.** 59 source pins retired across the task-setup lane
and the rest of the suite, 7 restored after `testing.md` § 3a was re-read —
it blesses artifact-property checks by name ("a stylesheet declares no raw hex
colours, or no duplicate selector"), and the first pass had cut on the surface
feature *reads a shipped file* rather than on what the assertion ASKS of it.
Of the 10 assertions that left this population, 7 were in the KEEP bucket by
the SYNTACTIC pass with no override recorded — which this tool's own preamble
says is a first pass and not a verdict ("Every site was then READ, and the ones
the rules got wrong are named in `_OVERRIDES`"). What replaces them is
[`process/code-audit.md` § 1c](?doc=process/code-audit.md), four classes of
silent failure rather than 59 spellings.

**AND THE OVERRIDES CAME UNANCHORED, which is the finding worth keeping.**
`_OVERRIDES` is keyed by `(file, line number)`, and each entry is a reason
someone wrote after READING that site. Deleting tests moved the lines, so two
override entries silently stopped matching — one displaced by two lines, one
orphaned when its test went — and the sites fell back to the syntactic pass.
The tool then reported "3 to convert" for assertions nobody had reclassified.
Re-anchored 2026-09-08. **A verdict anchored to a line number is a verdict
that edits can move without saying so**, and this population is edited by
definition. Re-keying it — by test function name plus the asserted text, or
any anchor that survives an edit — is open work, and until it is done, editing
a file in this population means re-running the tool and checking its overrides
still point where their reasons say. Of 1,220 assertions over
a file's text, 1,153 read **generated output** — a deck, a wrapper, an
`.sbatch`, a log — which is a real property of a real product and correct as
text. 67 read a file a person wrote, and 64 of those are lints. Three earlier
counts said 233, 256 and 173 because each used a different definition and none
wrote it down.

**The verdicts moved as much as the code did, and both directions matter.**
The browser bucket GREW twice when reading a site showed the extension had
routed it wrong; it then SHRANK by six when reading showed the opposite —
*which sheet defines a vocabulary* is not a question a browser can answer,
because the cascade yields a computed value and never the file it came from.
Every reclassification carries its reason in the tool's override table, so a
verdict can be argued with instead of taken on trust.

| | | |
|---|---|---|
| **KEEP — lint** | 56 | quantifies over a class; text is the only instrument that proves absence |
| **KEEP — not ours** | 8 | vendored bundles, licences, the contact-distance data file |
| **convert — browser** | 3 | all three in `test_structure_info_bridge.py` — see the entry below |

**Order, cheapest first — each cluster mutation-tested on its own**, because
**B3.2** found eight defects in the *previous* round's replacements after a
green run had hidden all of them.

1. ~~`test_run_index_covers_every_artifact.py`~~ — **DONE 2026-09-06**
   (`29b655ec`). The directory is built by the real `prep_jobset`, so the
   shipped `mb_monitor.py` is the one under test. Four mutations caught; two
   left the retired pin strings byte-identical, which is the proof they were
   blind. The round found two defects in the replacements themselves: a
   **second consumer** of `OUR_FILE_PATTERNS` the new tests did not reach
   (`runwrap.py:490` substitutes one stem and does not expand, so `--cold`
   would have stopped protecting indexed artifacts silently), and a row-count
   assertion that **flaked 50%** because the sampler is change-gated.
2. ~~**`test_trajectory_clocks_js.py`**~~ — **DONE 2026-09-06.** The badge's
   clock choice was extracted into `badgeClocks(state)` (user-approved), beside
   `cumulativeElapsed`, which exists for the same reason; six node tests run it.
   The swap that passed **233 tests** now fails five of them.
3. ~~**`test_structure_info_bridge.py`**~~ — **DONE 2026-09-06.** Three of the
   five now run `model-jobs.js` under node with a stubbed `fetch`, so the
   REQUEST the door posts is what is asserted. Mutation-tested by **putting the
   original bug back** — reading a flat `payload.info`, the key no route sends
   — which is the mutation the retired pin passed through. **Four assertions
   are deliberately NOT converted and are reclassified browser, not node**: the
   aliasing runs inside `mountInspector` and the resets sit in `transition()`,
   a reducer that exists only once a viewer is mounted, and nothing mounts one
   headless. Their two whitespace-measuring assertions (nine embedded spaces,
   counted twice) now match on structure instead — same coverage, one less way
   to be wrong for no reason.
6. ~~**`test_trajectory_csv_redaction_js.py`**~~ (2, node) — **DONE
   2026-09-06, and it turned up a circular one.** The ten redaction cases
   EXTRACTED the function's source by anchored slicing (`marker_end_token`
   was `"        return p;\n    }"` — eight spaces and a newline) and ran that
   text; a separate assertion pinned that `core.js` exports
   `_redactSourcePath` *"so this test can drive the function"*. **No test drove
   it that way** — removing the export failed only the string pin. Now the
   module is loaded through `tests/_node_esm.run_node` with `static_root`, so
   its browser-absolute `/static/…` import resolves, and the function is
   called on the namespace production uses. The export pin is deleted as
   genuinely redundant: **verified** — removing the export now fails the real
   cases. Gutting the redaction fails six of ten.

   *`core.js` loads under the shared harness*, which the file had assumed
   impossible (*"the module's IIFE captures `window`/`this` which Node doesn't
   have"*). That is what `globals_js` is for, and it opens the same route for
   the remaining `core.js` pins.

5. ~~**`test_launch_ask_mode.py`**~~ (4, python) — **DONE 2026-09-06.** The
   query cap now ASKS about 30 trials against a cap of 24 and reads the answer;
   the `--test-only` check compares what `ask` reports against what `submit`
   plans, using `JobResult.command` — no spying, and that is the surface a
   person sees. Mutation-verified: dropping the `"not asked"` result instead of
   naming it, and appending the flag instead of inserting it, each fail.
   **The first draft SKIPPED** — the deck lacked the restart group the
   submission door verifies — which is the read-green-never-ran shape; fixed
   with a real deck rather than left as a skip.

4. ~~**`test_task_setup_tab.py`**~~ — **MOSTLY DONE 2026-09-06.** Twelve
   assertions became one e2e reading the rendered card: every enabled stage
   offers both commands naming its own stage, the bench order is shown in
   order, and the hints say what each half is for. Mutation-tested — hardwiring
   bench to `enabled[0]` and swapping the order each fail it.

   **Two are BLOCKED and stay pins, with the blocker measured.** `_targetArg()`
   returns `""` unless a NAMED machine is chosen, and the page can only be
   driven to *(this machine)* without a named record in the live server's
   config root — so an e2e check for `--target` passes whatever the code does.
   I wrote that assertion, then measured it: adding `_targetArg()` to the launch
   line left the e2e **green**. It is deleted. A vacuous assertion is the thing
   this whole cluster exists to remove, so the pin stays until a fixture can
   supply a named target — recorded in the tool's overrides, not just here.

   **Three CSS assertions remain, and are browser work** (`.ts-state[hidden]`,
   `.ts-facts[hidden]`): cascade questions jsdom cannot answer.

7. **2026-09-06/07 — the rest, and one that did not land.** Twelve more
   converted, each mutation-tested. Two were more than test work:

   * **A DEFECT THE PIN WAS HIDING.** `setMachine()` never re-ran
     `renderNext()`, so choosing a remote machine left the copied command
     without `--target` — it preps for THIS machine while the card names
     another, invariant **C1** in `preparing-for-another-machine.md`. The pin
     asserted `"_targetArg()" in src`, true throughout. Fixed.
   * **A CHECK GREEN BY COINCIDENCE.** `#slab-orthogonal` has no wrapping
     `<label>`; `html.rfind("<label", 0, i)` landed on an unrelated field's
     label fifteen lines up, and the failure message described a DOM that does
     not exist.

   **The "BLOCKED" verdict on `--target` in entry 4 above was wrong**, and the
   reason it was wrong is worth more than the fix: `support.live_server` runs
   the app on a THREAD in the test process, so `$MOLBUILDER_CONFIG_DIR` is
   shared and a record written by a test is a record the route serves. The
   fixture is six lines. **A blocker is a measurement, not a memory** — this
   one was recorded confidently and held for a day.

   **`test_structure_info_bridge.py`'s three stay pins**, and this is the
   second time that entry has been re-examined. A Playwright test for APPLY's
   keep-on-`undefined` was written, passed, and was **vacuous**: `rebuildModel`
   runs at LOAD as well as after a poll, so a mutation that empties the store
   fails on the *before* assertion while proving nothing about any poll. On
   that reading it sat green through the guard being deleted three ways.

   **It stalled on something that is probably not a test problem — see T1
   below.** The facts the next attempt needs are recorded at the foot of
   `tests/test_inspector_registry_e2e.py`: how to build a run that states a
   contract without SIESTA (one `.fdf` beside a generic `*_geom_optim.xyz`),
   why an appended frame tests nothing (a strict tail extension takes
   `addFrames` and never rebuilds), and why "wait until a poll has happened"
   is already true before the file is touched (polling starts at mount).

   **Also converted:** the CSV export's redaction (downloaded and read, not
   grepped for an indentation), both inspector cores' listener teardown
   (measured at the lifecycle scope, because a global add/remove spy counts
   Plotly, 3Dmol and the projects sidebar), the recommended-value panel on
   both engines, the bench grid's out-of-order reply, the notify one-door
   (issued on the tab, then on the CLI, against one file), prep's trial path
   (checked on disk), and all three spectrumchart box traps.



---

## 5m. The test screen — what the audit found, sequenced *(2026-09-09)*

> **Verdict (2026-10-08 validation):** TS15 confirmed; TS10/TS8/§ 5m.3 obsolete (node is installed, `envs/recipes.py:1240`; § 11 replaced the records; no TS7 row exists).


**§ 5k is the paths framework (§ 5l's standard on top of it was retired
2026-09-17); this is the suite.** Two audits ran under
`process/testing.md` § 3b — a four-partition gate over ~3,000 test functions
(2026-09-08) and a header pass over the ten most-undocumented files
(2026-09-09). Their EVIDENCE lives in two records and stays there:

- [`archive/2026-09-08-test-audit-findings.md`](?doc=archive/2026-09-08-test-audit-findings.md) — the
  numbered defect ledger (§ 0a), the reproductions, and the unapplied verdicts.
- [`archive/2026-09-08-test-design-findings.md`](?doc=archive/2026-09-08-test-design-findings.md) —
  the protected class: where a science test would pass for a physically wrong
  reason.

**This section is the WORK.** It exists because a record with no plan row is a
record nobody executes — which is how `archive/2026-09-08-test-design-findings.md` shipped
on 2026-09-08 linked from nothing, failing two of the repo's own doc lints, and
was one compaction from being lost.

### 5m.1 The rule this section is under

> **The default is REMOVAL, and the burden of proof is on keeping** (user,
> 2026-09-08). The one exception is scientific validation: protected, and only
> its DESIGN is open.

And the number that governs everything below: **subsumption reasoning was
measured wrong on 9 of 19 candidates -- nearly half** (`test-audit-findings.md` § 4;
the 1-in-5 and 1-in-3 figures came from a harness blind to the commonest
validator defect). A verdict is a
lead. `tools/verify_subsumption.py` is what turns one into a decision.

### 5m.2 The rows

**They are `TS`, not `T`, and that is not cosmetic.** The plan already had a
`T1` (a live-poll defect, § 2) and a `T4` (under `P1`), and this table was
written as `T1`–`T9` on 2026-09-09 — so an edit keyed on `| **T1** |` hit the
wrong row and destroyed it. Caught and restored the same hour. The plan has had
this collision before (`N3`/`N4`/`N5` mean one thing in § 2 and another in
§ 5l.6 before it was retired); a section that numbers its own rows must
prefix them.

| step | what | done when |
|---|---|---|
| **TS6** | **`#66`'s four redesigns — THREE DONE 2026-09-09, one BLOCKED.** The SIESTA-version test now reads the pin from `recipes.py` instead of retyping it (the recipe moved to 5.5.0 in a mutant and it failed — the exact drift the old one could not see). `task.py` stays engine-agnostic is an IMPORT check, not four literal spellings (all four spellings caught; a comment correctly does not fire — **and my first version had the same bug it was fixing**, caught by mutation). The SCF-history branch is now actually reached. **Blocked: the `setInterval` pin.** Its correct replacement drives the module through `tests/_node_esm.py`, and that harness skips — **#79**, 717 tests never execute. Replacing it would swap a weak test that RUNS for a correct test that SKIPS, which is an environment decision, not a test one |
| **TS15** | **`#80` — the X3DNA probe answers *installed* for a pack that cannot run.** **The instance is FIXED 2026-09-09 by removing the dependency**: `x3dna_utils` is 3DNA's only interpreted tool and the one sub-command we used from it (`cp_std BDNA`) is a pure file copy, so `_copy_standard_bases` does it directly and no environment needs ruby. It was USER-FACING — the Modify tab's comma-separated two-strand input is the only path through `rebuild`, and it died with `exit 127` while `ds,` kept working via `fiber`. **Still open: the probe itself**, which answers from file existence. Same question as `TS10`, opposite symptom — node's absence is honest, this one was not | the instance fixed 2026-09-09; **the probe itself CLOSED, won't do** — § 11.5a's ruling (availability stays a file check; the failure must say what failed) *(2026-09-29: this cell said "still open")* |
| **TS10** | **`#79` — 717 tests never execute.** 43 files gate on `node`, which is not installed, not in the env and not declared. Measured 2026-09-09: 67 passed, 717 skipped. **Needs a decision, not code**: node in the env, node in CI, or a documented developer prerequisite — and until one is taken, every § 3a source pin those files could replace stays put | the harness runs, and `TS6`'s last redesign is unblocked |
| **TS8** | **The science redesigns — FIVE closed 2026-09-09 under one question: WHO OWNS THE NUMBER, and what does this layer DECIDE?** Two slab tolerances re-aimed (they asserted ASE's crystallography; `add_slab` only cuts a window, so both take their expectation from ASE now and the mechanism is covered structurally over 144 cases). The `homo_idx` rule moved the other way — the record asked the READ side, which carries the index but not the occupations, so it went to the EMITTER where it is ours, and became callable because it has a branch. Then two from § 3: a rotation is now asserted RIGID and PROPER over 3 axes × 5 angles (`R(0)=I` held for a wrong axis, a flipped sign and a radians error alike — 4 mutants killed), and the junction fixture's frozen set is DERIVED FROM z rather than counted, so the audit's own scenario (freeze the interior, same count) now fails. **Remaining: the rest of §§ 3–6** | each fails for the reason it names, and asserts nothing we did not write |
| **TS9** | **The remaining 1,627 undocumented tests** (of 6,411 — the pass took the suite from 67.3% to 74.6%). Writing the header IS the audit; it is what made T1–T6 findable | every test states its goal and its contract, and the obsolete ones have surfaced |

### 5m.3 What is deliberately NOT a row

**Re-deriving the ~90 lost verdicts.** Most auditors terminated on a session
rate limit and their output was cleared with the session directory. Re-deriving
a subsumption verdict that then cannot be applied without a mutant costs more
than running the mutant on a verdict already in hand — so TS7 spends the budget
on the 40 that survive, and the rest are declared lost rather than quietly
carried as if they were pending.

**And the cuts already applied are not revisited.** They are source pins and
cannot-fails, where the evidence is a line of shipped code rather than an
argument about reachability, and their reasoning is in the commit messages
`test-audit-findings.md` § 0 lists.

---


---

## 5n. JupyterNB — settings as DATA, and the hand-built residue *(2026-09-15)*

> **Verdict (2026-10-08 validation):** every DONE claim true in the code (23 confirmed; 3 partly: the token rides `JUPYTER_TOKEN` since 2026-09-20 `jupyter.py:757-763`, counts stale; 1 refuted: an empty table). **The id § 5n stays citable**: ~15 sites name it (`jupyter.py`, `blueprints/jupyter.py`, `cli.py`, `jupyternb/index.js`, `data/jupyter.toml:3`, `tests/test_jupyter_contract.py:3`, `docs/web/jupyter.md`).


**Origin.** The user, reading `molbuilder/jupyter.py`: *"why is jupyter.py not
following a data-driven design but rather handcrafted jibberish of code?"* —
then *"use .json or .jsonl or .toml to help clean this up. this is a
systematic design, not some hacking"* — then *"make sure that you don't have
other hackish code in the design, put this all in a consolidated plan."*

**The judgement is right and it is narrower than the words.** Parts of this
feature are already declarative (`_LAB_OVERRIDES`, `prepare_lab_home`'s
option→path map, `envs/recipes.py::_JUPYTER`). What earns the word is that
**the same kind of fact is expressed three different ways inside one
function**, that **40 lines of real Python live inside a string literal**, and
that **nothing has ever forced a seam on any of it** — the feature has no
test. § 5n.5 lists what is NOT residue, so the sweep can be checked rather
than believed.

**Row: W23.** This section is the contract; no code until it is agreed.

---

### 5n.1 The data file — format, location, and the admission rule

**TOML.** The project already answers this. Authored **rule tables** are TOML,
read with stdlib `tomllib` and gated by `persist.check_schema`:
`siesta/warm-files.toml`, `pyscf/warm-files.toml`,
`data/catalogue.template.toml`. JSON here is reserved for **fundamental
numeric tables** nobody annotates (`data/contact_distance.json`,
`data/fcc_lattice.json`). JSONL is a record stream, and this is a table. And
every row in this table exists for a reason that has to travel with the row —
TOML takes comments; JSON does not.

**`molbuilder/data/jupyter.toml`**, covered by the existing `data/*.toml`
package-data pattern. No packaging change, and this repository already
carries the incident where a `.toml` shipped in no wheel (`pyproject.toml`,
the comment above `data/*.toml`).

**Schema-stamped**, exactly as `warm-files.toml` is:
`schema = "molbuilder/jupyter@1"`, gated by `persist.check_schema`. An unknown
top-level section is refused **by naming the sections that exist** —
`warmfiles`' own refusal style, and the reason it is worth copying is that a
typo in a settings file otherwise disables a setting silently.

**THE ADMISSION RULE — what stops the file becoming a dumping ground.**

> A row belongs in `jupyter.toml` **if and only if its value is knowable
> before the process starts.** Anything that needs a runtime fact stays in
> code, and is emitted through the same writer.

That is a testable line, not taste. It cuts the current settings as follows:

| in the file | stays in code, and why |
|---|---|
| `ServerApp.port_retries = 0` | `ServerApp.ip` — a CLI argument |
| `MappingKernelManager.cull_idle_timeout = 1800` | `ServerApp.port` — derived, `jupyter_port(serve_port)` |
| `MappingKernelManager.cull_connected = false` | `ServerApp.token` — generated per start; a credential gets no home in the repo |
| `ServerApp.shutdown_no_activity_timeout = 3600` | `ServerApp.root_dir` — the projects root |
| `[lab.home]` — the three `LabApp.*_dir` options → their subdirectory names | `ServerApp.tornado_settings` — the `frame-ancestors` grant, **derived** from `serve_port_of(port)`. A security header stays a computation with its reasoning beside it |
| `[lab.overrides]` — all four plugin sections, verbatim from `_LAB_OVERRIDES` | `certfile` / `keyfile` — present only when `serve` has TLS |

**Lifecycle constants do NOT move.** `_PR_SET_PDEATHSIG = 1` is a kernel ABI
number; `_MARKER` is an identity string; `_STOP_GRACE_S = 4.0` is coupled to
`serve_daemon.stop_by_pidfile`'s `grace_s = 5.0` and splitting a coupled pair
across a data file and a module is worse than leaving both in code. *(That
coupling is asserted by a comment and by nothing else — see J12.)*

### 5n.2 The generated Python is a FILE, not a string

`_SERVER_CONFIG` becomes **`molbuilder/data/jupyter_server_config.py`**, a real
`.py` file copied verbatim at every start. It does **not** become TOML:
putting Python in a data file is the same defect in a worse place.

**Why not `inspect.getsource`**, which is this project's pattern for generated
code (`trajectory_log/emitter.py`, `pyscf/input.py`, `pyscf/vibration_emitters.py`
×2): that needs the class importable, and `NoCheckpoints` must subclass
`jupyter_server`'s `AsyncCheckpoints` **at class-definition time**. The
shepherd runs in the HOST env, which does not have `jupyter_server` and must
not need it. A file under `data/` is never imported, so it may name
`jupyter_server` freely, while pyflakes, an editor and `ast.parse` all still
read it as Python. One packaging addition: `data/*.py`.

### 5n.3 One emitter

`notebook_argv` stops holding a list literal. It builds a single
`{trait: value}` map — the file's rows merged with the runtime rows — and
emits `--{k}={v}`. The three mechanisms collapse to one, and a dropped row
like `port_retries` becomes **visibly missing from a table** instead of
invisibly absent from a list.

### 5n.4 The sweep — the other hand-built residue

Twelve found 2026-09-15 by reading the whole feature; **J13, J14 and J15 are the user's own, added the same day**.  Read: `jupyter.py`,
`web/blueprints/jupyter.py`, `web/static/jupyternb/`, `web/templates/jupyternb.html`,
the notebook half of `serve_daemon.py`, the `jupyter` CLI group, `config_dir`'s
four doors, and `envs/recipes.py::_JUPYTER`.

| # | what | why it is residue |
|---|---|---|
| **J1** | **Three mechanisms for one traitlets setting**, inside `notebook_argv`: 11 hand-written f-strings in a list literal, 2 of them interpolating module constants, and 4 more passed as a dict through a `--{k}={v}` loop **twelve lines below**. **15 settings, one shape** -- `--<Class>.<trait>=<value>` -- said three ways | § 5n.1 + § 5n.3 |
| **J2** | **`_SERVER_CONFIG` — 40 lines of real Python in a string literal.** No linter, no import, no test reaches it. **The cost is measured:** it was deleted by accident on 2026-09-14 by an edit that removed everything between two dead functions, and that shipped a `NameError` on every notebook start. `pyflakes` caught the sibling (`_LAB_OVERRIDES`, a name that IS referenced) and could say nothing about this one | § 5n.2 |
| **J3** | **The control routes are gated twice, on two different facts.** `@bp.post("/api/jupyter/start")` is registered inside `if os.environ.get(SUPERVISED_ENV) == "1":` **at import**, and the module's own docstring then explains that this flag is *not* the real precondition — `serve foreground` sets it while writing no pidfile — so a second, per-request gate `_supervised()` was added on 2026-09-14. Two gates, one fact. The import-time one also makes the app's **URL map depend on the environment `create_app()` happened to run in**, which is why no test can reach these routes | **Fix:** register always; the handler answers 404 when `_supervised()` is false. That is the behaviour § 6 of `jupyter.md` already specifies (*"404 rather than 403"*), with one gate, evaluated when the answer is knowable |
| **J4** | **Two hand-written 403 messages for one refusal.** `start` returns a three-line sentence naming `molbuilder.json`'s `admin` section; `stop` returns `"admin auth required"`. Same gate, same condition, two answers — and the short one tells a person nothing about what to do | One `_refuse()`, one sentence |
| **J5** | **The status payload has no author.** `jupyter.status()` returns 8 keys, the blueprint bolts on 5 more (`may_control`, `supervised`, `env_name`, `env_installed`, `install_command`), and `index.js` reads all 13 plus `ok`. Three files define one shape and no document states it — the two-authored-homes smell `engines/template.md` § 6.0 has a rule for | State the response shape once, in `jupyter.md` § 5; assemble it in one place |
| **J6** | **The tab's waiting is five ad-hoc timers, not a state machine.** `pollTimer`; `START_POLLS`/`startPollsLeft` at 1200 ms; `WEDGE_POLLS`/`wedgePolls` at 1500 ms; `ROOT_WAIT_MS`/`rootGaveUp` on a raw `setTimeout`; and three bare `pollSoon(600 \| 1500 \| 30000)`. Each arrived after its own incident — the comments are the record — and **each has its own clearing rule scattered through `render`**: `startPollsLeft` is cleared in one branch, `wedgePolls` at the top of the function, `rootGaveUp` never | **Fix:** one table of states, each row `{when, message, framed, pollMs, budget, onExhausted}`; `render` selects a row and one function applies it. **A JS object literal, not a data file** — the strings are UI copy and belong beside the page (`ui-contract.md`), and shipping TOML to a browser buys nothing |
| **J7** | **`jupyter.md` § 4 says "exactly four states". `render` carries **13** `return` statements and `refresh`'s `catch` adds a fourteenth outcome.** Four of the undocumented ones carry their own time budget AND their own button — wedged, *"asked for a notebook and none started"*, waiting for the projects root, and the fetch failure. **The states with the most behaviour are the ones the contract does not name** | Fixed with J6: the table in the code and the table in § 4 become the same list |
| **J8** | **A third hand-built unverified TLS context.** `jupyter.py` collapsed two copies into `_unverified_ctx()` on 2026-09-14, with a docstring saying *"One home, because this is a security knob"*. `cli.py:2525-2527` still builds the same three lines for `serve status` | One home means one home. `serve`'s, not the notebook's — but the same knob, and the claim is already written |
| **J9** | **`--port` hand-written six times across the `jupyter` verbs, while `_serve_flags` sits ~350 lines above** saying *"one decorator, so the two verbs cannot drift apart about what a server accepts"*. **The six have already drifted:** two carry help text that disagrees (*"the SERVE port -- the notebook's own is that plus one"* vs *"the SERVE port this notebook belongs to"*) and three carry none | A `_port_flag` the six share |
| **J10** | **THE FEATURE HAS NO TEST.** 1,610 lines across its own five files, plus the notebook half of `serve_daemon` and six CLI verbs. Nothing under `tests/` imports `molbuilder.jupyter` or touches `/api/jupyter/*`; the only mentions are an env name in `test_diagnostics` / `test_envs_install` and a layer entry in `test_layering`. **Every one of the nine defects fixed on 2026-09-14 was found by eye or by running the live server, and J2 shipped.** This is the root of the whole row: nothing has ever forced a seam on this code | § 5n.7 |
| **J11** | **`config_dir.jupyter_lab_home`'s docstring says "Two directories under it"**; `prepare_lab_home` creates **three** (`settings`, `user-settings`, `workspaces`) and writes a config file beside them. The door's own contract is behind its caller | One sentence |
| **J12** | **`_STOP_GRACE_S < serve_daemon`'s `grace_s` is asserted by a comment and by nothing else.** The two were both `5.0`, and reconciliation then reported *"(forced)"* for every perfectly polite stop. It was fixed by changing one number; nothing stops the other moving back | One assertion, in the test set below |
| **J13** | **SWITCHING TABS LOSES THE OPEN NOTEBOOK, and the kernel it belongs to is still running.** *(user, 2026-09-15: "when we switch tabs (without quitting server), currently the jupyternb tab reinit and does not open the original tab before the switching even though the kernel etc is still running … we would like to have this persistent when switching tab, but do not need this when server get shutdown and restarted.")*  Leaving `/jupyternb` destroys the iframe, so coming back always re-points it — that part is a page navigation and cannot be avoided in the tab. What decides whether Lab comes back where you left it is its **workspace** (which documents were open), and `labUrl` sends `?reset` on **every** load (`index.js:137`), which throws it away. **This supersedes a decision taken with the user on 2026-09-14** (`jupyter.md` § 4.1, *"it remembers no LAYOUT"*), whose reason was that a restore argued with the folder the projects sidebar had selected — a reason that has since weakened on its own, because the sidebar is no longer SHOWN on this tab and cannot be changed from it | **Fix — server-side, one home (preferred): `prepare_lab_home()` DELETES `workspaces/` at every shepherd start, and `?reset` is dropped from the URL entirely.** A new server then has no workspace to restore (clean, as asked); a tab switch or a page reload restores the one Lab has been saving (persistent, as asked); and no browser state is involved, so there is nothing to keep in sync. **Two things to MEASURE before the shape is fixed:** whether `/lab/tree/<path>` fights a restored workspace or merely moves the file browser, and whether Lab writes the workspace eagerly enough that a switch seconds after opening a notebook is caught. **Fallback if the measurement says no:** compare `st.token` to one remembered in `localStorage` and send `?reset` only when it differs — the token is regenerated by `run_shepherd` at every start, so it already IS the server-session identity. `localStorage`, not `sessionStorage`: closing a browser tab is not a server restart |
| **J14** | **`serve status` answers only about the port you guessed.** *(user, 2026-09-15: "the serve status should report all running server on different ports - should not only report only when the correct port is provided. if there are server running the pid and port is available in the log/state dir.")*  `--port` defaults to 8000, so a server on 8888 makes `molbuilder serve status` print *"not running"* and exit 3 — the one answer that is worse than no answer, because it is confidently wrong. The information is already on disk and already keyed by port: `runtime_dir()` holds `serve-<port>.pid`, one per server, and `pid_state` already says whether each is alive and really ours | **Fix:** `--port` on the two STATUS verbs becomes optional. Given, the verb behaves exactly as today (`serve status`: exit 0 answering / **4** up-not-answering / 3 absent — the codes it already returns, re-derived from the code on 2026-09-15 after this row guessed 1; `jupyter status` has only 0 and 1). Omitted, each **surveys**: every `serve-*.pid` (or `jupyter-*.pid`) in the runtime dir, verified with `pid_state`, probed for an answer, one line per live server — and, for `serve`, whether a notebook is held beside it. A miss on a named port now also names the servers that ARE up, which is the whole complaint in one line.  **Two things it must say rather than imply:** a `serve foreground` writes NO pidfile and therefore cannot appear, so an empty survey says that instead of *"nothing is running"*; and a stale file is reported stale, never signalled, which is the rule this module already keeps.  The probe (https then http, `/api/health`) is inline in `cmd_serve_status` today and becomes one helper, because the survey and the single-port path must not answer *"is it up"* two ways |
| **J15** | **TWO MOLBUILDERS ON ADJACENT PORTS COLLIDE BY CONSTRUCTION, and nothing says so.** *(user, 2026-09-15: "how about conflicts in jupyter port?")*  `jupyter_port(p) = p + 1`, so a molbuilder on 6006 wants 6007 for its notebook and a molbuilder on 6007 wants 6007 for its WEB server. Whichever starts second loses, and which one that is decides which symptom you get: the notebook refuses to start, or `serve start` fails to bind. **`--ServerApp.port_retries=0` already makes this loud rather than silent** — without it jupyter-server picks a random free port out of 50 while the runtime file, the `frame-ancestors` grant and `answering()` all keep naming `p+1`, and the tab reports "wedged" over a perfectly healthy notebook. So the remaining defect is not the crash; it is that **nothing warns when the clash is knowable in advance, and the report points at the wrong log** | **Fix, three parts.** ① **MEASURE FIRST** what jupyter-server actually does and prints at `port_retries=0` against a taken port — this row assumes the setting's documented meaning and nothing has run it. ② **Warn where it is computable**: `serve start` knows its own port and `ports_with_pidfile()` now lists every other live server, so `jupyter_port(port)` landing on one of them is a sentence printed before anything starts, and the `serve status` survey carries the same line. Not a refusal — the web server is perfectly startable, only its notebook is doomed. ③ **Name the right log**: `index.js`'s *"asked for a notebook and none started"* sends a person to the SERVE log, and a port clash is written in the NOTEBOOK log.  **NOT a change to the derivation.** `p+1` is one number to remember and the pairing has one home in both directions; moving to `p+1000` relocates the collision without removing it, and letting Jupyter choose freely makes the runtime file the authority for a port that four other places derive — which is the design `port_retries=0` was chosen against |

### 5n.5 What is NOT residue — so the sweep can be checked

* **The three lifecycle layers and their constants.** Order is the content —
  *"everything that can raise happens before the pidfile exists"* is not
  something a table can say — and each layer is measured, not argued.
* **`jupyter_port` / `serve_port_of`.** One derivation, both directions, with
  a stated security reason for the inverse existing at all.
* **`_LAB_OVERRIDES` and `prepare_lab_home`'s option→path map.** Already
  tables. They move into the file; their shape does not change.
* **`envs/recipes.py::_JUPYTER`.** Already declarative, and its packages carry
  their reasons.
* **`config_dir`'s four doors.** Consistent and reasoned — including
  `jupyter_lab_home` **not** being port-keyed, which the docstring states and
  justifies (*"the defaults do not differ between servers"*).
* **`serve_daemon`'s stringly-keyed `state` dict** (`"child"`, `"hup"`,
  `"term"`, `"jupyter"`, `"nb_busy"`, `"nb_pending"`). It is L1, it predates
  this feature, and reshaping it is not this row. **Out of scope, on purpose.**

### 5n.6 Order of work

Each step is separately green and separately revertible.

1. `molbuilder/data/jupyter.toml` + the loader (schema gate, refusal by name).
2. `notebook_argv` → one emitter; `_LAB_OVERRIDES` and the lab-home map read
   from the file. *(**Done 2026-09-15, and the line-count claim here was
   wrong**: `jupyter.py` went 681 → 744. The two literals were 78 lines and
   the loader with its refusals is ~110. The win is not fewer lines — it is
   130 lines of data in two files that a linter, an editor and `tomllib` can
   all read, and one emitter instead of three mechanisms.)*
3. `molbuilder/data/jupyter_server_config.py` + `data/*.py` in `pyproject.toml`.
4. **J3 + J4** — one gate, one refusal. (This is what makes the blueprint
   testable, so it comes before the tests.)
5. **J9 + J8** — the shared port flag; the third TLS context.
6. **J6 + J7** — the tab's state table, and § 4 rewritten from it.
7. **J13** — the workspace survives a tab switch and dies with the server.
   *(Done 2026-09-15. Measurement (b) answered — the workspace file carries a
   live mtime, so Lab writes it continuously and `?reset` was what discarded
   it. Measurement (a) was **not** made and is no longer needed: the shape
   chosen never sends a tree path and a restore together, so Lab's precedence
   between them decides nothing. The payload is 16 keys with
   `workspace_saved`, which step 10 documents.)*
   **Measure first** (does `/lab/tree/<path>` fight a restore; is the
   workspace written eagerly enough), then `prepare_lab_home()` wipes
   `workspaces/`, `?reset` comes out of `labUrl`, and `jupyter.md` § 4.1's
   *"it remembers no LAYOUT"* paragraph is rewritten to say what replaced it
   and why — a decision taken with the user is superseded in writing, never
   silently flipped.
8. **J14** — the two status verbs survey every server when no port is named.
   *(Done 2026-09-15. The backticks in this line were eaten by an unquoted
   shell heredoc when § 5n was written; restored with the J15 entry.)*
9. **J15** — measure the port clash, then warn where it is knowable.
   *(Done 2026-09-15. Measured: jupyter-server exits 1 with "the Jupyter
   server could not be started because port 6007 is not available", so the
   assumption held. `jupyter.port_clash()` is the one home; `serve start`
   warns before detaching, the survey warns per server, and the status
   payload carries it so the tab can name the cause. The payload is now 15
   keys, which step 10 documents.)*
10. **J5 + J11** — the response shape stated once; the stale docstring.
    *(Done 2026-09-15. `jupyter.md` § 5.1 is the 16-key table and § 5.2 the
    two control routes' four status codes; the blueprint builds the payload
    in one expression instead of six `st[...] =` mutations.)*
11. § 5n.7's tests, and **J12**. *(Done 2026-09-15 —
    `tests/test_jupyter_contract.py`, 7 tests / 11 cases, all seven
    mutation-tested. **The suite grows by seven**, and § 5n.7's "does not
    grow" line was wrong: it assumed hand-written assertions to remove, and
    there were none — the coverage being replaced is zero, which is J10.)*

### 5n.8 The fresh-eyes review of the review *(2026-09-15, same day)*

Three reviewers over the eleven commits, every claim re-verified here before
acting. **Nine real defects in work that was hours old**, which is the
argument for the pass rather than against it. All fixed in one commit; the
numbers below were re-derived, not taken.

| what | why it mattered |
|---|---|
| **`_forget_workspace`'s guard let `..` through** | `home / ".."` is a real directory whose `.parent` IS `home`, so the check passed and the loop would have emptied the state directory — logs, reports, run/. `../../../escape` was refused and `..` was not, and the comment claimed both. **Destructive.** Now compares resolved paths |
| **the workspace was shared by every server** | `jupyter_lab_home()` is not port-keyed, and J13 put session state in it. Starting B's notebook emptied A's layout → A's next tab switch lost the notebook whose kernel was still running: **J13's own bug, hours after J13**. Now `workspaces/<serve port>/`; `settings/` and `user-settings/` stay shared |
| **the survey stopped recording a wedge** | before J14 the no-argument `serve status` took the single-port path and appended the detection to the log. Making it a survey dropped that for the **common** invocation, against the rule `_note_wedge`'s own docstring cites |
| **`port_clash` asked one of the two directions it names** | `serve start --port 6007` beside a live notebook on 6007 warned nothing and failed to bind after `daemonize()`, with the terminal gone. Both directions now, and the first no longer globs the runtime directory to answer a single-port question |
| **`run_shepherd`'s stated invariant was false** | *"everything that can raise happens before the pidfile exists"* — `projects_root()` and `load_rules()` both sat below the write. Hoisting `prepare_lab_home` had fixed the measured case and left the claim wrong |
| **state 3 blocked on a folder it was about to discard** | with a workspace to restore the tree path is unused, yet `opening` still hid the frame for its full 8 s budget — on the headline J13 case |
| **`_serve_flags` still hand-wrote `--port`** | the eighth copy, in the decorator whose docstring forbids drift, and the only one with no help text. **The step-5 commit message claimed this was covered.** It was not |
| **`web/app.py`'s comment described the design J3 deleted** | and cited the reference the blueprint corrected — the restatement a reader of `create_app` actually sees |
| **test 4 asserted the class, not the wiring** | deleting one of the two `checkpoints_class` assignments — the plausible "looks redundant" edit — left it green while `.ipynb_checkpoints/` came back |

Smaller, same pass: three pure forwarders inlined (`_unverified_ctx`,
`_port_clash`, `_notebook_is_up`); `whenRootKnown`'s unreachable door check
removed; the last two bare `pollSoon` intervals replaced by the table's own;
the dead `"port"` key dropped from the runtime file and its docstring
corrected (it named the dead key and hid `"base"`, which IS read);
`_notebook_action`'s docstring stopped claiming a queue that cannot lose an
action, because a one-bytecode window exists and no arrangement of Python
statements closes it. Documents: nine corrections in `jupyter.md`, the 409 in
`web-api.md`, and `config_dir`'s not-port-keyed reasoning, which J13 outgrew.

**Left on the table, deliberately** — both real, both predating this
programme, both in code it did not open:

| # | area | item | state |
|---|---|---|---|

### 5n.7 Tests — the set, deliberately small

`feedback_tests_earn_their_place` applies: this must not become a test per
setting. Eight, and each one fails on a real mutation:

1. The schema gate refuses a wrong stamp.
2. An unknown top-level section is refused **by naming the sections that exist**.
3. Every `[server]` row reaches the argv — the assertion a dropped
   `port_retries` trips.
4. `data/jupyter_server_config.py` parses (`ast.parse`) and defines
   `NoCheckpoints` — the assertion J2's accident would have tripped.
5. `/api/jupyter/start` answers **404** with no supervisor and **403** to a
   non-admin with one — J3 and J4 in one test, and impossible to write today.
6. `_STOP_GRACE_S < serve_daemon`'s grace — J12.
7. `prepare_lab_home()` leaves no workspace behind — write a file into
   `workspaces/`, call it, assert the file is gone while `user-settings/`
   is untouched. That is J13's whole contract on the server side, and the
   mutation it catches (wiping the wrong directory) would silently discard
   a person's Lab settings.
8. **one server's start keeps another server's layout** -- added by the
   2026-09-15 review, which found the workspace sharing a directory across
   servers and re-creating J13's bug.

*(Written 2026-09-15 and the sentence here was wrong: there were no
hand-written assertions to remove, because there was no coverage at all.
**The suite grows by seven.** That is J10's whole point, and the usual rule —
unifying an API must REDUCE the count — does not apply to a feature whose
count is zero.)*

---


---

## 5o. Transport — the parameter surface is inert, measured against the binary *(2026-09-15)*

> **Verdict (2026-10-08 validation):** the findings and the W25 fix true; four items superseded by TD4 / § 5u (`TransportConfig` retired `325cb1d4`; `wizard.py:287`/`preflight.py` gone — now the role item `ts_hs_save`); the order of work is § 5u's. `tests/test_transport_keywords_exist_in_the_binary.py:3,37` and `transport/record.py:115` cite § 5o.


> **The order of work is § 5u's since 2026-09-29** (transport consolidated, W43 / M5). This section keeps the *why* and the evidence; its own order-of-work table is history.

*(User: "check vigorously against actual transiesta manual and scientifically
how this computational process should be conducted. find the missing gap in UI
and parameter setting … and any inconsistency or potential issues. we need a
full solution for this particular task.")*

### 5o.0 How this was checked — the binary, not a memory of the manual

`molbuilder-siesta` ships **SIESTA 5.4.2** (`strings` on the binary). SIESTA's
fdf keywords are compiled into `siesta` and `tbtrans` as literal strings by
the `fdf_get` call sites, so **whether this installation can read a keyword is
measurable**, not a matter of recollection. Every claim below is a count taken
from `~/miniconda3/envs/molbuilder-siesta/bin/{siesta,tbtrans}`.

The method's one limit, stated: a label assembled at runtime from a prefix
does not appear whole — which is why `TS.ChemPots` counts 0 while `ChemPots`
and `.ChemPot.` are present, and the chempot blocks are *fine*. Each negative
below was therefore re-checked for the bare stem in **any** spelling.

### 5o.1 What is SOUND — so this review can be audited, not just believed

* **The five-stage ladder is the standard TranSIESTA recipe.** seed →
  electrode_L → electrode_R → device → transmission, with the electrodes run
  as bulk single-points and the device solved by NEGF. That is the workflow
  the method requires; nothing about the shape is wrong.
* **The electrode → device hand-off is correct and was measured live.** The
  electrode deck writes `TS.HS.Save true` **and** `SaveHS true`
  (`wizard.py:287`), a preflight refuses an electrode deck that would not
  write its `.TSHS` (`preflight.py:315`), and the device deck points tbtrans
  at the converged Hamiltonian with `TBT.HS <label>.TS.HSX` — that line
  carries a comment recording a live 5.4.2 measurement, and `TBT.HS` is in
  the binary.
* **Every per-electrode block key is the right 4.1+/5.x spelling:** `HS`,
  `chem-pot`, `used-atoms`, `elec-pos begin|end`, `bloch`,
  `semi-inf-direction`, inside `%block TS.Elec.<name>`, with
  `%block TS.Elecs`, `%block TS.ChemPots` / `TS.ChemPot.<name>`,
  `TS.Atoms.Buffer` and `TS.Voltage` — all present in `siesta`.
* **The consistency contract is the right physics.** Electrodes and device
  must share basis, XC and mesh or the lead self-energy cannot attach
  seamlessly; § 5 makes that an invariant and the contract fields are sealed
  at both doors. That is the single most important scientific rule in the
  workflow and it is correctly enforced.

### 5o.2 THE FINDING: ten of the twelve fields the tab renders cannot affect the run

| field | writes | in SIESTA 5.4.2? | effect |
|---|---|---|---|
| `transmission_emin_ev` | `TS.TBT.Emin` | **`Emin`: 0 occurrences in `tbtrans`, any spelling** | none |
| `transmission_emax_ev` | `TS.TBT.Emax` | **`Emax`: 0** | none |
| `transmission_n_points` | `TS.TBT.NumE` | **`NumE`: 0** (only the words "numeric") | none |
| `transmission_relative_to_ef` | `TS.TBT.Erange.RelToEF` | **`Erange`: 0 · `RelToEF`: 0** | none |
| `contour_n_circle` | *nothing* | — | **nothing** — it reached only the Methods paragraph (`transiesta.py:1015`), deleted 2026-09-17 with `methods_fragment`; see E12 |
| `contour_n_real` | *nothing* | — | **no consumer anywhere** |
| `contour_e_bottom_ev` | *nothing* | — | **no consumer anywhere** |
| `pyscf_functional` | *nothing* | engine not registered | none |
| `pyscf_basis` | *nothing* | engine not registered | none |
| `log_level` | `WriteVerbosity` claimed | **`WriteVerbosity`: 0 in `siesta`** | **no consumer anywhere** |
| `max_memory_mb` | runner `ulimit -v` | n/a (not fdf) | real |
| `num_threads` | `OMP_NUM_THREADS` | n/a (not fdf) | real |

**Two of twelve do anything, and both are shell knobs.** Every field that is
supposed to shape the *physics* is inert.

**Three distinct failures, not one:**

1. **Pre-4.1 keywords on a 5.4.2 binary.** `TS.TBT.*` and
   `TS.ComplexContour.NumCircle` / `NumLine` / `Emin` are SIESTA-3.x-era spellings.
   5.4.2 keeps exactly one legacy alias, `ComplexContour.NPoles`, which
   molbuilder does not use. The emitter's own header says "modern SIESTA
   4.1+/5.x NEGF syntax" — the device blocks are, the transmission window is
   not. **fdf ignores a label nobody queries**, so this is silent: the run
   completes and T(E) comes out on tbtrans's *default* energy grid, not the
   one the person asked for. A wrong answer that looks right.
2. **Fields wired to nothing.** Four have no consumer in the tree at all.
3. **A Methods paragraph that reports a setting the calculation never
   received.** `contour_n_circle` reaches only the manuscript sentence —
   *"NEGF density integration used a complex contour with N imaginary-axis
   points"* — for a deck that carries no contour keyword. That is the one
   finding here with a **publication** consequence.

*(No surviving artifact in `projects/` evidences the carbon-chain live walk
§ 8 cites, so I cannot say what that run's T(E) grid actually was — only that
these four keywords could not have set it.)*

### 5o.3 What a correct 5.4.2 run needs and the UI cannot say

Each verified PRESENT in the installed binary, so each is a real control
being left on the table:

| what | keyword | why it matters scientifically |
|---|---|---|
| the transmission energy window | `TBT.Contours` + `%block TBT.Contour.<name>` (`from … to`, `points`/`delta`, `method`) | **the actual replacement for the four dead fields** — the modern mechanism is a contour BLOCK, not scalars |
| transverse k for T(E) | `TBT.k` / `TBT.kgrid.MonkhorstPack` | T(E) needs a **denser** transverse grid than the SCF; convergence in it is the standard transport convergence study, and the tab cannot express it at all |
| broadening η | `TBT.Elecs.Eta`, `TBT.Contours.Eta` | sets the T(E) lineshape; too large smears resonances, too small makes them noise |
| tbtrans's own temperature | `TBT.ElectronicTemperature` | separate from the device SCF's; it is what enters the I–V Fermi functions |
| the SIESTA-side NEGF contour | `TS.Contours.nEq.Eta`, `TS.Contours.Eq.Pole`, `TS.Contours.nEq.Fermi.Cutoff` | the real controls the three dead contour fields were reaching for |
| what gets written | `TBT.DOS.Gf`, `TBT.DOS.A`, `TBT.DOS.Elecs`, `TBT.T.Gf`, `TBT.T.Bulk`, `TBT.T.All` | without these there is no PDOS or per-lead data on disk — which is what **W10**'s transmission inspector would have to read |
| lead treatment | `TS.Elecs.Bulk`, `bloch` (hardcoded `1 1 1`) | Bloch expansion of a minimal electrode cell is the standard cost saving; bulk-H in the lead region is a physics choice |
| spin | `TBT.Spin` | no spin-polarised transport path exists, and the junction chemistry check already detects open-shell metals |

### 5o.4 The rule that stops this recurring — and it is TESTABLE

`engines/transport.md` § 3.8.8's per-engine panels are the right frame, and the frame must carry this:

> **A field's `engine_key` names a keyword its engine's own binary accepts,
> and that is checked against the installed binary rather than asserted.**

The binaries are on disk in the env this project installs. A test can walk
every `engine_key` in a config dataclass, extract the fdf label, and grep the
engine's binary for its stem — skipping when the env is absent, the way
env-dependent tests already do. That converts *"the manual says this exists"*
— which is how seven dead keywords shipped — into a machine check, and it is
the only finding here that prevents the next one.

### 5o.5 Order of work

1. **W25 first, because it is a correctness bug, not a UI one.** Replace the
   four transmission scalars with the `%block TBT.Contour` the binary reads,
   keeping the person's window/points as the block's `from…to`/`points`.
   ✅ **Done 2026-09-15.** Three things came out of doing it that the
   investigation had not seen:
   * **Three tests were pinning the dead keywords**, including one whose own
     docstring describes the exact failure it was satisfied by — *"tbtrans
     falls back to its own defaults and the transmission curve … is reported
     over an interval nobody chose"*. A test written to catch this bug
     passed **because of** it.
   * **The fix's own comment resurrected the tokens.** Two of those tests
     assert a keyword by plain substring over the whole deck, so naming
     `TS.TBT.*` in an explanatory comment made them pass on prose. Caught
     within the hour; the emitter no longer spells them, and the rule is
     worth keeping: **a comment must not be able to satisfy an assertion.**
   * **`transport/record.py` holds the best evidence on the open question.**
     Its docstring pins the `AVTRANS` format *live on 5.4.2* and says E is
     relative to E_F *"when the deck said `TS.TBT.Erange.RelToEF T`"* — a
     keyword that can never have done anything, so what was observed was
     tbtrans's **default**. That points at *relative by default*, i.e. the
     retired switch was never needed rather than mis-spelled. Evidence, not
     settled: closed by reading one real `AVTRANS` beside its device `.fdf`.
2. The three contour fields become the real `TS.Contours.*` keywords, or go.
   **`contour_n_circle` leaves the Methods paragraph the same day**, whichever
   way it is settled — a sentence reporting an unset parameter is the one
   thing here that must not survive another commit.
3. `log_level` gets a real keyword or goes.
4. § 3.2's panel split (W24), with `TransportConfig` becoming TranSIESTA's.
5. The engine_key-versus-binary test, and the missing controls above added
   panel by panel with the test green at each step.

### 5o.6 The manual pass — the parameter inventory, and what it changed

*(User: "do thorough scientific investigation including transiesta's own
manual … a full list of parameter and settings … identify sources of data
user need to provide. check our framework and see where things are missing.")*

**Source.** The SIESTA **5.4.0** manual — the release note for the 5.4.2 this
project installs — pulled from the project's own GitLab and extracted
locally, plus the TBtrans reference page, cross-checked against the binaries'
compiled fdf labels. The inventory lived in `engines/transport.md` § 3.3 until
that section was rewritten the same day (c3d8b454); each parameter's decision
is now § 2a.13's map, and which program reads it § 6.1b (2026-09-29, from the
engine's source). This section records what the pass CHANGED and what it
leaves open.

**It answered § 5o's open question.** The manual states a TBtrans contour's
energy reference is **the equilibrium Fermi level by default**. So
`transport/record.py`'s live observation was right and only its attribution
was wrong: the retired `relative_to_ef` switch was **never needed**, not
mis-spelled. Closed.

**It explained the whole history.** tbtrans's own default contour is
`from -2. eV to 2. eV, delta 0.01 eV, mid-rule` — 401 points. molbuilder's
defaults are −2 eV, +2 eV, 401 points: **the same grid to the point.** The
four dead keywords therefore produced exactly the intended result for anyone
who changed nothing, and bit only somebody who changed a value. Latent, not
active — and latent by coincidence, not design. That is why a live walk
passed and why three tests could pin them unnoticed.

**One conformance defect, fixed here.** The manual: a `%block TS.Elec.<name>`
must carry `HS`, `semi-inf-dir`, `electrode-pos`, `chem-pot`. `elec-pos` sat
inside `if buffer_idx:`, so an ORDINARY junction got two electrode blocks
without it. Probably harmless — the junction is sorted so the electrodes are
the first and last atoms, which is where an omitted anchor would land — which
is precisely the problem: an undocumented default agreeing with the truth. Now
emitted unconditionally, with a test.

> **And mutation-testing that test caught a weak test of mine.** Checking
> that `elec-pos` is PRESENT passed the regression, because putting the line
> back inside the buffer branch made the `else` fire and gave the LEFT
> electrode `elec-pos end` — a wrong anchor a presence-check cannot see. The
> test now pins the PAIRING (z-min anchored by its first atom, z-max by its
> last) and catches all three mutations: the line moved, the line deleted,
> the anchors swapped.

#### What is missing, in the order it should be built

> ### ✅ **SEVEN OF THE NINE SHIPPED 2026-09-15.**
> The form went 12 fields → **21**, in seven sections ordered by the decision
> a person makes rather than by history: *Transmission* · *Transmission
> k-sampling* · *Broadening* · *Outputs* · *NEGF density contour* · *Leads* ·
> *Runtime*. **`TBT.k` is reachable**, so the T(E_F)-against-transverse-k
> study can be run from the tab. Every output flag is offered, so **W10** has
> data to read when asked for. The two PySCF fields are **gone** — the form
> contains the string "pyscf" nowhere — and the five electronic-contract
> fields have a section that names them correctly.
>
> **Emission policy, stated once and applied throughout:** where the manual's
> default is a NUMBER it is the field's default and is always emitted, so the
> deck is self-documenting and behaviour is unchanged; where the default is a
> FORMULA (`min[η_e]/10`, `5 k_B T`, *inherit the SCF grid*) the field
> defaults to 0/`0 0 0` meaning *leave it to the engine* and nothing is
> written — because a number there would silently replace a formula.
>
> **`log_level` got a real keyword rather than deletion:** `TBT.Verbosity`,
> an integer 0–10 defaulting to 5, is in the binary, and the three words map
> onto it (warning 2 · info 5 · debug 8). **The mapping is sourced from the
> TBtrans reference, not invented** — the same pass that retired
> `transmission_relative_to_ef` for want of a sourced mapping could not then
> invent one here.
>
> **Two are deliberately NOT knobs**, and that is the finding rather than a
> gap: `TS.Hartree.Fix` is derivable from the transport axis exactly as
> `semi-inf-direction` is, and the manual calls the boundary *"an intricate
> and important"* matter — it should be derived, never typed. And
> `TBT.ChemPot.<>.ElectronicTemperature` is per-chemical-potential, so it
> belongs with the bias scan, not the override lane.
>
> **Still open:** `TS.Elecs.Eta` (the TranSIESTA-side twin of a broadening
> that IS exposed), `TBT.T.Out` (meaningless below three terminals), and a
> spin-polarised device run wired end to end — `TBT.Spin` selects a channel,
> which is not the same as the SCF producing two.
>
> **Four tests failed on this change and every one was right to.** The
> Methods paragraph still interpolated the deleted `contour_n_circle`; two
> new sections had no description; the served section list is pinned; and a
> bias assertion searched the WHOLE deck for `"1.5000 eV"`, which
> `TS.Contours.Eq.Pole 1.5000 eV` now also matches — a loose assertion
> testing a number rather than a keyword, scoped to `TS.Voltage` lines now.
> The Methods sentence reports only emitted values, which closes the one
> defect here with a publication consequence.


Each verified present in the installed binary, with the manual's default.
**The framework for all of them is `engines/transport.md` § 3.8.8's per-engine panel** — these are
TranSIESTA's panel, and none of them belongs in a shared dataclass.

| # | what | why it is next |
|---|---|---|
| **1** | **`TBT.k` — the transverse grid for T(E)** | It *inherits the SCF's*, and an SCF grid is routinely far too coarse for transmission. The standard convergence study is **T(E_F) against transverse k** — and it cannot be run from this tab at all. The single largest scientific gap |
| **2** | the `TBT.DOS.*` / `TBT.T.Eig` / `TBT.T.All` output flags (all default `false`) | a run writes transmission and nothing else, so **W10**'s inspector has no DOS or eigenchannel data to read *even in principle* |
| **3** | `TBT.Elecs.Eta` (1 meV) · `TBT.Contours.Eta` (min η_e/10) | broadening sets the T(E) lineshape: too large smears resonances, too small makes them noise |
| **4** | the real `TS.Contours.*` — `Eq.Pole` (1.5 eV), `nEq.Eta`, `nEq.Fermi.Cutoff` (5 k_B T) | what the three dead `TS.ComplexContour.*` fields were reaching for. **Retire or rewire those three in the same commit**, and `contour_n_circle` leaves the Methods paragraph with them |
| **5** | `TBT.ChemPot.<>.ElectronicTemperature` | tbtrans's own, inheriting `TS.ElectronicTemperature`; it is what enters the I–V Fermi functions |
| **6** | `TS.Elecs.Bulk` (true) · `bloch` (hardcoded `1 1 1`) · `TS.Hartree.Fix` | lead treatment and the boundary condition the manual calls *"an intricate and important"* matter |
| **7** | `TBT.Spin` | no spin-polarised transport path, while the junction chemistry check already detects open-shell metals |
| **8** | re-section the five contract fields out of **NEGF** | `basis_size`, `energy_shift_ry`, `xc_functional`, `xc_authors`, `siesta_mesh_cutoff_ry` declare `section: "NEGF"` and are the *electronic contract*. Hidden today, but a form-B citation renders them under a heading that misdescribes them |
| **9** | `log_level` → a real keyword or gone | it claims `WriteVerbosity`, which is **zero** in `siesta` |

**Two things the pass did NOT settle**, recorded rather than guessed:

* **What TranSIESTA does when a required electrode line is absent** — error,
  or silent default. It could not be determined without a live run, and the
  fix made the question moot rather than answering it.
* **Whether the live carbon-chain walk § 8 cites ever exercised a
  non-default window.** No artifact survives in `projects/`. Given the
  default coincidence above, a default-valued walk would have looked correct
  either way.


---

## 5p. Transport — the implementation, step by step *(2026-09-16)*

> **Verdict (2026-10-08 validation):** TR1–TR8 built, several in a later shape (role items written by the walk; kz refused by `kmesh.fixed`; the bias from the list only; overrides refused, not routed; the shared panel; the gather at prep; the sweep as one run — § 5x); the deletion programme § 5p.3p.2 complete; § 5p.3p.5's own done-condition met. 94 confirmed, 30 obsolete, 29 partly, 3 refuted, 10 unverified. The order of work is § 5x's.


> **The order of work is § 5u's since 2026-09-29** (transport consolidated, W43 / M5). This section keeps the *why* and the evidence; its own order-of-work table is history.

**The design is [`engines/transport.md` § 2a](?doc=engines/transport.md)**, agreed
2026-09-16. This section is the distance from here to there, in dependency
order. It does not restate the design; read § 2a first.

### 5p.0 THE RULE — this section is updated as the work happens

> **Every step updates this section when it lands, and every problem found
> while doing a step is recorded here before it is worked around.**
>
> Not afterwards, not in a commit message alone. A step that changed shape, a
> step that turned out to depend on something not listed, a defect discovered
> mid-step: each goes in, with what was measured. A plan that is only written
> once describes an intention; a plan kept current describes the work.

*(User instruction, 2026-09-16: "make it a rule that update of this plan is
mandatory after each step of implementation and/or modification when new
problems discovered in that process.")*

This is R3 applied to one programme — the open rows still live in § 2, and this
section keeps the *why* and the order (§ 0).

### 5p.1 The wheels that already exist — reuse, do not rebuild

**Measured 2026-09-16 by reading the framework, not from memory.** Nearly
everything § 2a describes is already built for the optimization kind. The gap is
not that the machinery is missing; it is that **transport does not use it.**

| what § 2a needs | what already exists | where |
|---|---|---|
| a shared baseline of values (Class A) | **the template** — `<label>.template.toml`, written from the catalogue narrowed to engine **and calculation**, carrying the config's values | `describe.build_description` → `template_with_values(cfg, engine=, calculation=)` |
| per-stage values (Class C) | **`varies` + `Stage.overrides`**, resolved as template ⊕ this stage's overrides | `resolve.effective_config`; `engines/stages.md` § 6.2 |
| a per-stage editing surface | **the stage table** — rows are stages, columns are `varies`, cells are `overrides`, and *an empty cell means "the template's value" and shows it greyed* | `web/task-setup.md` § 5, § 5.1 |
| a kind-aware parameter list | **already kind-aware**: the columns endpoint skips an item whose `calculations` excludes this folder's kind | `GET /api/task-setup/columns?calculation=` |
| the shared baseline read back | `GET /api/task-setup/template-values?dir=` — parses with `read_template`, the same reader `prep` uses | — |
| machine facts per stage (Class E) | `allocation` items + `Stage.execution` | `task-setup.md` § 6 |
| rendering from a description | `spec_for(struct, cfg, names=, calculation=)` → `DeckSpec` → `prepare_deck` | `script_emit` |
| keeping another kind's rows out of a deck | **the kind gate** in the section walk | `script_emit._render_sections` (2026-09-15) |
| the attempt ladder | a launched attempt is never rewritten; re-prep opens the next | `materialize.resolve_attempt` |
| results moving between stages | the DAG copy with three gates + `.gathered-from` | `prep.gather_transport_inputs` |
| one submission walking an axis | the bias chain — `cd` per point, stop-or-continue by whether points depend on each other | `submit._plan_chain` |

**The precedent to follow is the vibration kind**, which is a second calculation
on one engine's seam: one dispatch arm in `spec_for`, the kind's own deck module,
and its rows tagged in the one catalogue.

### 5p.2 The gap, measured

| # | what transport does instead | consequence |
|---|---|---|
| **G1** | **writes no template at all** — a transport folder holds `task.json`, the composed junction and pseudos, and no `<label>.template.toml` | Class A has **no home**. The stage table has nothing to read; `template-values` would find nothing; there is no shared baseline for a cell to inherit |
| **G2** | `config/transport.py` — a second vocabulary, with four parameters renamed from the catalogue's spelling | two names for one quantity; a projection is needed wherever they meet |
| **G3** | the tab builds its form from the dataclass, sectioned by **topic** | a section called "Electrodes" holds the device's bias; nothing says which run a control changes |
| **G4** | the hand-over puts **every** override on the `device` bag | a parameter the transmission owns never reaches the transmission deck — the live defect |
| **G5** | `config_for` fills from the cited `.fdf`, not from a template | the values are inherited rather than single; the first ruling in § 2a.7 cannot be honoured |
| **G6** | `_prep_transport` is a parallel arm that returns before the shared path | no `resolve`, so no `ParameterSet` and no provenance; `--pipeline-log` is a documented no-op |
| **G7** | four of five rungs still render from hand-written emitters | no read-back check, no `.validation.txt`, no check gate on those four |
| **G8** | three keywords the design needs have **no catalogue row**: `TS.HS.Save`, the equilibrium pole **count**, and the bias point `TS.Voltage` | they cannot be declared, defaulted, or reached from a description |
| **G9** | `kgrid` is one row holding three numbers whose components fall in **three** classes | no interface can explain it; § 2a.13 flags it |

### 5p.3 The one design addition, and it needs a ruling

Everything above is reuse. **Two declarations do not exist yet**, and both are
small siblings of markers the catalogue already has (`allocation` — the
scheduler answers this; `citation` — a cited run answers this):

| proposed | means | serves |
|---|---|---|
| **`role`** | *the stage's role answers this; nobody chooses it* | Class D. A `role` item is never a form field and never a stage-table column |
| **`stages = [...]`** | *which rungs may carry their own value for this* | Class C. It is `calculations` one level down, and it is what routes an override to the rung that owns it — the proper fix for **G4** |

**Not yet approved.** Everything else in this section proceeds without them;
the steps that need them say so.

### 5p.3a FOUND WHILE IMPLEMENTING — `citation`'s semantics contradict the ruling

*Recorded 2026-09-16 under R3a, before working around it.*

`template.py::template_with_values` writes a `citation` item into the template
**valueless**:

```python
value=(None if (it.allocation or calculation in it.citation)
       else getattr(config, it.name, it.value))
```

and `config_from_template` refuses one that carries a value. That is the
**sealed** reading — the item is declared but never answered in floor 2, and
`prep` fills it from the cited deck, exactly as it fills an allocation item from
what the scheduler granted.

**§ 2a.7's first ruling reversed that.** The cited relaxation *defaults* these
values; the person may change them, and a change applies to every stage at once.
Under that ruling the citation is **a source of defaults at `init`**, not an
answerer at `prep` — so the template must carry the value, and it must be
editable.

**What this changes, and where it belongs:** the marker survives and keeps its
name, because *"which items are harvested from the cited run"* is still exactly
the question it answers — but its behaviour moves from *valueless, filled at
prep* to *defaulted at init, thereafter an ordinary template value*. That is
**TR1's** substance (it is what makes the template carry a shared baseline at
all), so it is scheduled there rather than done here.

Not a defect in what shipped: the seed migration relies on the current
behaviour and is consistent with it. It is a consequence of the ruling that had
not yet been traced into the code, and tracing it is what R3a is for.

### 5p.3b LANDED — `role` and `stages` *(2026-09-16)*

*Approved by the user and built. Recorded here under R3a.*

Both are siblings of `allocation` and neither is a new mechanism:

| | means | what it buys |
|---|---|---|
| **`role`** — a list of kinds | *the stage's own role answers this* — the third answerer, after the scheduler (`allocation`) and a cited run (`citation`) | a `role` item is not a form field, not a stage-table column, and carries **no value** in a template of that kind, so no description can claim to have set it |
| **`stages`** — a list of rung names | *which rungs may carry their own value*; `calculations` one level down. **Absent means any rung may**, which keeps the optimization ladder exactly as it was | the basis on which an override reaches the rung that owns it — TR8's proper fix |

**Declared on the items § 2a.13 already classified**, transcribing the agreed
map rather than re-deciding it: `solution_method` and `wrap_into_cell` as
`role = ["transport"]`; the thirteen transmission/`TBT.*` items to the
transmission; `electrode_kz` to the two electrodes; the three NEGF items to the
device. The seed owns nothing of its own — it only obeys, which is the formal
statement of why it is skippable.

**They have a reader, deliberately.** `GET /api/task-setup/columns` now skips a
`role` item for that kind and publishes each item's `stages`. A declaration
with no reader is a control that does nothing — the defect `electrode_kz`
shipped with, and not one to repeat while fixing it.

**Verified:** six behavioural tests, the two load-bearing ones mutation-proven
(remove the valueless rule and the template test fails; remove the endpoint
filter and the column test fails). The column test asserts the other half too —
that `solution_method` is *still* offered for an optimization — because without
it the test would pass on an empty column list. Contract in
[`engines/template.md` § 6.4](?doc=engines/template.md). Full sweep of the
affected areas: 1348 passed.

**What did NOT change:** nothing reads `stages` to route an override yet. That
is TR8, and it is the point of the declaration rather than a follow-up to it.

### 5p.3c LANDED — TR1, transport has a template *(2026-09-16)*

**Three edits, and the third is what makes the other two mean anything.**

1. **`citation` flipped from sealed to defaulted** — the finding recorded at
   § 5p.3a, done here because it *is* TR1's substance. The template now carries
   a cited item's value and `config_from_template` no longer refuses one. The
   marker keeps its name: *which items are harvested from the cited run* is
   still exactly what it answers — only the harvest moved to `init`.
2. **`jobset init` writes `<label>.template.toml`**, defaulted from the cited
   deck (`transport/citation_defaults.py`, one function, called once). The k-grid
   is the one value not copied verbatim: the transverse pair carries over and
   the transport axis is forced to 1, at the point the value is born rather
   than corrected by each consumer downstream.
3. **The template WINS at prep.** `siesta_config_for` reads it as the baseline
   and drops from the projection exactly the items the `citation` marker names
   — asked of the catalogue, not listed in code. Without this the citation
   would overwrite the person's edit a line later and the ruling would be a
   fiction; the template would be a file nothing reads, which is the
   `electrode_kz` defect one directory up.

**Verified:** the test that can fail is *edit the template's basis to a value
the citation does not hold, re-prep, the deck must follow* — mutation-proven by
removing the drop. 1672 passed across the affected areas.

**Three tests failed and every one was informative:**

| | |
|---|---|
| I hand-formed the template path | `test_doc_claims` walks the AST to keep `<label>.template.toml` formed in **one** place — six call sites spelled it two incompatible ways until 2026-08-17, so a folder with two templates had the web tab and `prep` reading different files. Mine was offence seven, caught the day it was written. Now uses `template_path` |
| `MeshCutoff 250.0 Ry` vs `250 Ry` | The framework's syntax door formats from the item's **declared type** (float); the hand-emitter it replaces formatted the Python value it held (int). During the migration one rung says each, for a value they agree on. The deck assertion now compares numbers **by value** — asserting the spelling fails a test for a reason that is not the science |
| `assert written == ["task.json"]`, *"floor 2 is task.json ALONE"* | The sealed design encoded as a test, and the same phrase flagged earlier as never having been a ruling. A shared baseline needs a file to live in: transport's floor 2 is `task.json` **and** a template, like every other kind |

**What TR1 did NOT do:** the other four rungs still render from the citation's
values through the old emitters, because they do not go through
`siesta_config_for`. Only the seed reads the template today. That is TR5, and it
is why TR5 follows rather than being optional.

### 5p.3d LANDED — TR2 and TR3 *(2026-09-16)*

**TR2 — the three missing rows, with every spelling verified against the
binary rather than recalled.** `caps.env_prefix` answered this time (it
returned `None` earlier in this programme, which is why the check had been
deferred), so the keywords were read out of the installed SIESTA 5.4.2 through
the manager's own door:

| keyword | in `siesta` | in `tbtrans` | row |
|---|---|---|---|
| `TS.Voltage` | 4 | 2 | `bias_voltage_v` — Class C, the **device**'s, binding its own transmission point |
| ~~`TS.Contours.Eq.Pole.N`~~ | — | — | **Withdrawn 2026-09-16: there is no such keyword.** SIESTA has the string and never queries it; the pole count is derived as `N = E / (pi kT)`. *(Corrected by § 5p.3o the same day: it IS queried, `m_ts_chem_pot.F90:113`, and overwritten from the energy on this deck shape — so it cannot act here.)* The abort came from `negf_eq_pole_ev = 1.5` eV shipping as the default — 18 poles at 300 K. See `engines/transport.md` § 3.6 |
| `TS.HS.Save` | 1 | 0 | `ts_hs_save` — the **electrodes**', and marked `role`: a lead that omits it converges happily and produces nothing the device can attach to, so nobody is offered a switch |

**TR3 — smaller than this plan predicted, and the difference is the finding.**

§ 2a.13 said the `kgrid` row *"needs to become the transverse grid for
transport, with the transport axis owned separately."* Building it showed two
things that change the answer:

* **a separate `kgrid_transverse` row would be a second name for one physical
  quantity** — precisely the `TransportConfig` problem TR6 exists to delete. The
  cure would have been an instance of the disease;
* **the lead's transport axis already IS owned separately**, by `electrode_kz`.

So nothing was missing from the *rows*. What was missing was a **check**: a
person could set the third component in the template and the renderer would
quietly write 1 anyway — a control that appears to do something and does not,
which is the defect this programme keeps finding. Tier 2 in § 2a.4's terms, so
it is a refusal that names the reason, not a silent correction.

**It is registered as a KIND validator, and that distinction matters.** The
existing `_validate_transport` is keyed on the `TransportConfig` *class*, so it
stopped firing for any rung that moved onto the seam (the seed, 2026-09-15). A
rule that runs for one of two config classes is not a gate. This also closes
§ 2a.7's note that passing `calculation="transport"` read as though a gate
existed when `_KIND_VALIDATORS` held only `vibration`.

**Verified:** both tests mutation-proven; the refusal test has a second half
asserting the transverse pair still reaches the deck, without which deleting the
k-grid control entirely would pass. 1752 passed across the affected areas.

**Housekeeping noticed, not a defect:** two doc counts were back at their
pre-4b values because the per-stage-presets revert restored those files
wholesale, taking two unrelated count edits with it. The revert was slightly
wider than the change it undid. All three counts are now the measured numbers.

### 5p.3e LANDED — TR4, the transport arm resolves *(2026-09-16)*

`engines/transport.md` § 3.2 measured `_prep_transport` as **a second
conductor that decides** — it composed, gated, extracted and rendered, so a
transport run had no `ParameterSet`, no provenance, and `--pipeline-log`
printed *"not wired for the transport arm yet"*.

The seed rung now goes through `resolve`, floor 3's own step 2. The log is
real, and every value names its source:

```
STEP 2 · RESOLVE — the description becomes a ParameterSet
  in   template               T.template.toml
  out  elements               1 (a run, not a sweep)
  ⊕    basis_size             TZP          <- template
  ⊕    dm_tolerance           1e-05        <- template
  ⊕    kgrid                  (4, 4, 1)    <- template
```

**TR1 did not just make this reachable — it made it sufficient.** `resolve`
reads a template, and a transport calculation did not have one; that was the
whole blockage. Now that it does, and the template carries **every** item the
kind has (the transport-only rows included), there was nothing left for a
hand-built projection to add.

**So `siesta_config_for` is dead — 143 lines deleted**, `_TO_SIESTA_NAME` with
it. That two-vocabulary mapping was written three hours earlier for TR1 and was
scheduled to die in **TR6**; it fell out here for free, because the thing it
bridged no longer has two sides for this rung. A step that deletes a later
step's work is the shape to look for.

**Verified:** 1914 passed. Three tests, including the honest failure — a
description written before TR1 has no template and is **refused by name with
what to do**, rather than falling back to re-reading the citation, which would
be the sealed behaviour returning silently by the back door.

### 5p.3f THE SEAM QUESTION — WITHDRAWN, it was mis-posed *(2026-09-16)*

I asked whether *"a derived bulk lead is legitimately a `Structure`"* and put
TR5 behind the answer. **The question does not arise, and the design already
said so** *(user, correcting me)*:

> **One structure carries everything.** It holds the frozen electrodes at both
> ends and the relaxed bridge between them. Transport takes that one file,
> checks the labeled electrodes are the fixed atoms, **takes those atoms out**
> to build each lead's single-point run, and combines the bridge with the two
> self-energies. One file, so *"all the structure and facts involved in the
> calculation are consistently constructed."*

Nothing is derived from elsewhere. `compose._extract_and_gate_electrodes` calls
`extract_electrode_model(dev, REGION_LEFT_ELECTRODE)` — **a subset of the cited
structure, selected by label**. Same atoms, same relaxation. Asking whether it
is legitimately a structure was asking whether a selection of a person's own
relaxed atoms is a structure.

**So TR5 is UNBLOCKED and the answer is option (b):** each rung renders the
structure it describes — the extracted lead for an electrode rung, the junction
for the device — and the seam stays `(struct, cfg)`. `ElectrodeModel` already
holds everything a `Structure` needs (elements, positions, the lateral cell,
the bulk z-period); yielding one is mechanical, not a design question.

*The lesson is the one this programme keeps relearning: I had read § 4 (*"region
labels drive everything"*) and the extraction code earlier in this work, and
still posed a question the design had answered. Consulting the contract is not
the same as remembering that I did.*

### 5p.3g LANDED — the first validation *(W29, 2026-09-16)*

*Found and built the same day, under R3a.*

The design's **first** check — *"look for the labeled electrodes and make sure
they are the fixed atoms"* — did not exist. What existed was a different and
**later** one, and the distinction is the whole finding:

| | when | asks |
|---|---|---|
| already there — the frozen-unmoved gate (`compose.py`) | at compose, **after the relaxation has run** | did the electrode atoms *move*? Compares the cited deck's coordinates against the `.XV` |
| **added** — `check_electrode_labels_are_frozen` | at setup, **before anything runs** | are the electrode-labeled atoms *declared frozen*? |

Neither substitutes for the other. Label the leads, forget to freeze them, and
nothing objected: the relaxation ran, the leads relaxed, and the junction was
found unusable at compose — **a wasted relaxation, on a metal junction where
that is not cheap.** The closest thing was a hint inside an unrelated warning.

It runs in **both engines' settings gates**, beside
`check_unconsumed_region_labels`, because it is the same question one step
further: the labels say which atoms are leads, and a lead must survive the
relaxation untouched.

**A warning, not a refusal — and the line is deliberate.** A structure carrying
electrode labels is *heading* for transport but has not committed: a person may
relax the whole junction once before freezing the leads for the run that
counts, and refusing here would block that. The refusal belongs at compose,
where transport IS the intent, and it is already there. *Open for the user to
overturn: making it an error is a one-word change.*

**Verified:** six tests, and two earn their place specifically. *The bridge is
not required to be frozen* is the discriminating case — a junction's purpose is
that the bridge relaxes while the leads do not, so a check demanding every
LABELED atom be frozen would refuse every correct junction. *It reaches a real
preflight* is mutation-proven by unwiring it: a check nothing calls is a check
that does not exist, which is the `electrode_kz` lesson. 1913 passed.

### 5p.3h LANDED — TR5a, the electrode rungs on the seam *(2026-09-16)*

**The seam did not have to change, and the reason is the one-file design.**
`ElectrodeModel.as_structure()` restates the extracted lead as an ordinary
`Structure` carrying its own cell — the device's lateral vectors verbatim and
the bulk repeat along transport. It is a restatement, not a conversion: those
atoms came out of the cited junction by their region label, same atoms, same
relaxation. So a deck still describes a structure; there are simply **two
structures in one file**, and each rung renders the one it describes.

Measured: the lead deck went from a hand-written f-string to **423 lines** with
the engine's full section set, a `.validation.txt`, the check gate, and:

```
    4    0    0      0.0      <- transverse, shared with the device
    0    4    0      0.0
    0    0   40      0.0      <- the transport axis, DENSE: a lead is
                                 periodic there and its Fermi level is the
                                 reference energy everything is measured against
```

**A test predicted its own obsolescence and was right.**
`test_the_un_migrated_rungs_still_lack_them` asserted the electrode deck
LACKED the template's items, and carried an instruction: *"delete this when the
electrode rung is tabled — at which point it should fail, and that failure is
the migration being done."* It failed, and was deleted. Its job — keeping
*"this rung is on the seam"* from degrading into *"SIESTA decks tend to have
SCF settings"* — passes to `TestTheElectrodeRungIsOnTheSeam`, which asserts the
contrast the other way round: the one thing the lead must have that the seed
must not, the dense transport axis.

**Verified:** five tests, including *the lead deck is the LEAD not the
junction* (it must describe the extracted subset's atom count, or the others
would pass on a deck that quietly rendered the whole junction). 144 passed.

**Remaining in TR5:** the `negf` shape — device and transmission. Those
describe the junction, which the seed already proves the seam can carry; what
they additionally need is the region partition for `%block TS.Elecs`, and it
travels on the structure's own `regions`.

### 5p.3i LANDED — TR5b, every rung on the seam; and the contract consolidated

**TR5 is complete.** All five rungs render through `spec_for` → `DeckSpec` →
`prepare_deck`. Measured on the repository's own fixture: seed 488 lines, a
lead 419, device and transmission 584, each with a `.validation.txt` and the
engine's check gate, against the **13 keywords** § 3.2 measured.

The device and transmission share ONE layout but resolve their OWN configs, so
they share their shape and differ in exactly what a person tuned for the
transmission — which is what § 2a.7's ruling asked for, reached **without
anyone deciding which keywords `tbtrans` requires**. That question would have
needed verification against the binary; the per-rung config made it moot.

**Deliberately NOT tabled: the NEGF electrode block.** `%block TS.Elecs` and the
per-electrode blocks stay the pre-seam emitter's, wrapped in one `Block` with a
small projection at the boundary. TranSIESTA identifies each electrode by a
**contiguous atom range**, so an off-by-one computes transmission through a
region that is not the molecule *and converges while doing it*. That emitter has
been measured against a live 5.4.2 binary; a rewrite would have to earn that
again for no gain.

#### Two defects this step found in itself

**The check gate caught a duplicated keyword the day it was written.** The
device deck said `SolutionMethod diagon` from a section and `transiesta` from
the NEGF block — libfdf takes the first. The cause is general: **a `role` item
must not be written by a section**, because a section resolves a value from the
config and a role item rightly has none there. `_render_sections` skips them
now and each rung's block writes its own. The same lift-boundary rule that kept
`_emit_basis_and_xc` out of the seed layout, one level deeper — I had applied
it once and still missed it.

**A test fixture was geometrically invalid and nothing had ever looked.** Its
buffer variant put 44.5 Å of atoms in a 40 Å cell. It survived for as long as it
existed because **the device deck had no settings gate until it joined the
seam**: the first time anything validated a device geometry was the moment this
work put a gate in front of it. The gate is right; the fixture is sized from its
own geometry now.

#### The contract, consolidated *(user: "fully updated, consolidated, and checked")*

`engines/transport.md` no longer describes a design and a different
implementation:

* the § 2a banner says **built**, not *"none of it is implemented"*;
* § 3.2's floor-3 row is **closed**, with the measurement;
* the *"same bytes"* note is **resolved**, with why the per-rung config settled
  it rather than a keyword audit;
* § 3.6's outcome list is reconciled item by item — and item 5's own wording
  corrected: the values arrive from the **description**, not from the citation,
  which supplies only the defaults;
* **§ 2a.14** is new: what landed, a diagram of template ⊕ overrides → resolve
  → spec_for → prepare_deck, the two-structures-one-file table, the measured
  deck sizes, the lead's k-mesh as the physics made visible, and what did NOT
  land;
* § 2a.8's worked example was written before any of this existed, so it was
  **re-run against real decks** and now carries the result: one shared edit,
  three rungs follow.

**Still open and named there:** the bias has two homes (`task.bias` the axis,
`bias_voltage_v` the template row), `TransportConfig` survives only to feed the
lifted block, and net charge / gating are deferred by ruling.

### 5p.3j FIXED — the stage deck ignored the bias axis *(2026-09-16)*

**I raised this as an open design question and it was not one.** Adding a
`bias_voltage_v` catalogue row in TR2 looked like it had given the bias two
homes — the row, and `task.bias`'s list — so I asked which should win.
§ 2a.10 had already answered:

> *"single bias is the degenerate case of the bias axis — one point, normally
> at zero."*

**One mechanism.** The template declares the parameter (its range, unit and
help) and answers the one-point case; a scan is that same parameter taking
several values, which is the framework's ordinary precedence — *the template's
value ⊕ this point's*. Nothing to rule on. That is the **second** time in this
programme that a question I posed as open had been settled in a section I had
already read; the first was the seam (§ 5p.3f).

**The defect underneath it was real.** Measured with the template set to 0.5 V
and the axis at `[0.0, 0.2]`:

| | before | after |
|---|---|---|
| `04_device/T_04_device.fdf` | **0.5 V** — the template's own value | 0.0 V |
| `04_device/v0/…` | 0.0 V | 0.0 V |

§ 2a.11 describes the stage-directory deck as *"the same deck `v0/` holds"* —
it exists so the job row's script sits where every generic reader looks. It was
not that deck, and whichever of the two a person opened they would have
believed it.

Three tests pin it, including the degenerate case: with no axis the template
answers, and there is no `v*` level at all — § 2a.11's own rule, *a stage
carries a sub-level for each axis it varies over and none for an axis it does
not*.

**Recorded while fixing it: the row says volts, the deck says eV.** Not a
mismatch. fdf treats `TS.Voltage` as an energy, and the energy an electron
gains across V volts is exactly V electron-volts — same number, different unit
word. Written into the row's help, because the next person to notice will
otherwise "fix" one side to match the other and break it.

### 5p.3k LANDED — TR8, an override reaches the rung that owns it *(2026-09-16)*

**The defect this programme started from is closed.**

Every override went onto the `device` rung, whatever it was. So a person who set
the transmission's energy window had it written into the deck `siesta` runs —
where `TBT.*` keywords are inert — and **not** into the deck `tbtrans` runs,
which is the one that computes T(E). No error, no warning: you asked for ±3 eV
and got the default.

`route_overrides` asks the catalogue which rungs may own each item (the `stages`
declaration, `engines/template.md` § 6.4) and puts the value there:

| set | lands on |
|---|---|
| `transmission_emin_ev` | the **transmission** |
| `electrode_kz` | **both** electrodes — a junction has two leads, and routing to one would leave the self-energies built on different Fermi-level resolutions |

**Verified:** three tests, all mutation-proven — revert the routing and all
three fail, reproducing the original defect. One exists to make this a
*routing* test rather than a *delivery* test: it asserts the device does NOT
also get the transmission's parameter, since being inert there is precisely why
nobody noticed it was the only place it landed.

#### Found while doing it

**`config_for` was refusing a person's own lead k-density.** It validated
overrides against `TransportConfig`'s vocabulary, but an override now names a
**catalogue row** — what the template declares and `resolve` resolves — so
`electrode_kz` came back as *"not a transport parameter"*, which it plainly is.
The vocabulary is the engine's now, with `TransportConfig`'s accepted beside it
only while that class survives.

**And that function is close to vestigial.** It is called once, and its result
is handed to a pseudo-provisioning step that reads **no field from it**. What it
still uniquely guards is **identity** — `job_name`, the bias axis — which
`resolve` would let a stage override, because to `resolve` those are ordinary
schema fields. *That is what TR6 must preserve when `TransportConfig` goes*, and
it is written at the decision point rather than left to be rediscovered.

#### A holding position, stated as one

An override for a **shared** SCF control (no `stages` declaration) still goes to
the device. That is not an answer: those are items any rung may legitimately
own, and a flat form cannot say which was meant. **TR7's per-stage surface is
where the question becomes askable** — and the stage table already exists for
optimization, which is the whole point of § 5p.1.

### 5p.3l PART OF TR7 — the seal's REASON corrected; the interface half not built

**What was wrong.** Two doors refused a Class A field as a per-stage override
with the message *"the citation's to say — cite a relaxation that ran with the
values you want."* § 2a.7 reversed that rule, so by 2026-09-16 the advice was
**actively misleading**: it sent a person to redo a relaxation when they could
edit one line of the template.

**The refusal is still right, for a different reason**, and that is the whole
correction: a value shared by every rung cannot be a per-stage override,
because that is exactly how the device would come to disagree with its own
leads. So the guard stays and the reason changes — and the message now names
**where the value is actually changed**.

**And the form-A/form-B distinction went with it**, which is more than a message
fix and is recorded as such. The old rule OPENED these fields for a form-B
citation (a labeled pair, no deck) because the override lane was then the only
way to state a basis at all. Every transport calculation carries a template now
— form A filled from the cited deck, form B from the catalogue's defaults — so
that hole no longer needs to exist, and leaving it would mean **a form-B
junction could give its device a different basis from its leads**. Verified
rather than assumed: a form-B calculation does get a template, and stating the
basis there reaches the deck.

**Four tests changed, and how matters.** Three asserted the *reason*
(`"citation's to say"`); they now demand the corrected reason **and** that the
message names where to change the value. The fourth set a stage override to
prove a pair's basis was stateable — its concern was exact, *"there would be no
way to state the basis at all"* — so it proves the same thing through the
template. The concern survived; the mechanism moved.

#### NOT built, and it is the part that needs a decision

The tab still builds its form from `dataclass_to_form_schema(TransportConfig)`
and still **hides** the Class A fields. Hiding them was correct under the old
rule — *"a form field the door is guaranteed to refuse is a trap, not a
control"* — but under § 2a.7 they are the person's to change, so hiding them now
means **there is no way to change a transport calculation's basis or XC except
by hand-editing the template file.**

That is § 2a.6's Panel 0, and it carries a real question rather than just work:
the tab writes **stage overrides** today, and Panel 0 would have to write the
**template** — a different file, written by `jobset init` from the citation.
Either the tab edits the template directly, or the Class A values are edited
through **Task Setup**, where template values already have a home
(`/api/task-setup/template-values`) and the stage table already exists. The
second reuses more and touches the Transport tab less.

**Left for a ruling** rather than decided — it is the one remaining piece that
designs what a person sees rather than what runs.

### 5p.3m FIXED — a description made in the browser had no template

**A regression of mine, from TR1, live for several steps.**

TR1 made `jobset init` write `<label>.template.toml`. The **web describe door
does not write files — it RETURNS them** for the browser to write, and it
returned only `task.json`. So a transport calculation created in the browser
had no shared electronic description, and `prep` refused it by name: *"this
transport calculation has no template, so there is nothing to resolve."*

Fixed: the door returns both, the template's values defaulted from the cited
run.

#### Why four sweeps missed it, which is the part worth keeping

The test that would have caught it **immediately** is
`test_task_setup_tab.py::test_transport_describe_answers_the_finished_description`
— it posts to `/api/transport/describe` and asserted the response's file list.

**My sweeps selected files by NAME**: `-k transport` never matched
`test_task_setup_tab.py`. The tests that exercise a surface are not always in
the file named for it, and a filter chosen from names measures the names.
Corrected by running every non-browser file that *mentions* transport
(`grep -rln transport tests/*.py`), which is 12 files beyond the ones named for
it, and all pass.

#### Three tests in that unswept file, each saying something

| | |
|---|---|
| `== ["task.json"]`, *"floor 2 is task.json ALONE"* | the sealed design encoded as an assertion — **the same phrase flagged twice already in this programme**. Now asserts both files, and that the template is real: the reader `prep` uses accepts it and it carries the cited values |
| `"citation's to say"` in the refusal | the message § 2a.7 reversed. Now demands the corrected reason **and** that the message names where the value is changed — a refusal with nowhere to go is not a refusal, it is a dead end |
| the override lands on `device` | TR8 working: `transmission_n_points` goes to the **transmission** now. Asserts that, and that the device does *not* also get it |

### 5p.3n FULL-TEXT REVIEW — what four agents found, verified, and fixed *(2026-09-16)*

Asked for after *"I don't trust your work at all"*: read every document, every
backend file and the UI **in full text**, find redundant code, duplicates,
magic numbers, hacks, and doc-vs-code disagreement. Four agents read
~12,000 lines. **Every claim below I re-derived against the code myself before
acting, and two agent verdicts were wrong in the direction that matters.**

#### The live defects, each measured

| | what was measured | fixed by |
|---|---|---|
| `_prep_transport` opened **no attempt** | it ended at `prep_jobset`; the CLI opened one itself. `web/blueprints/build.py:1656` calls `prep_calculation` directly, so the browser's Prep button reported success and handed back a folder `submit._launch_dir` refuses — *naming the command that had just run*. The shared arm's own 27-line note calls this out: *"A verb that is only finished by one of its callers is not a verb."* It reintroduced exactly that | one `_open_attempts(js, base, stage, containers=…)` that **both arms** call; the scan's per-point ladders are DATA on the call |
| the resolved **resources were thrown away** | `_resolve_transport` returned `element.values` and the arm rebuilt the allocation by hand. `resolve` folds two riders onto `element.resources` — `continue_retries` and `use_gpu`. Measured: `SiestaConfig` defaults them `1` / `False`, `Resources` defaults both `None`, so **every transport wrapper rendered with no warm-retry loop** and `use_gpu` fell back to grepping the deck for `Diag.ELPA.GPU` — the re-derivation `gpu.md` G7 deleted. `resolve.py`'s A-5 finding, on a third road | return the element; use `element.resources` and `element.render_config()`; fold the allocation **before** the resolve, as the shared arm does |
| wrappers were written **past** the remote-activation guard | `prep run device --target sol` against a record stating no activation wrote one wrapper per bias point carrying *this* machine's activation, then refused. The refusal's whole premise is that those files must not exist | the guard moved above the write |
| `TS.Contours.Eq.Pole.N` **reached no deck** | a catalogue row with an anchor and a default that no section named. *(The fix recorded here — a `CONTOUR_SECTION` naming the row — was **wrong**, and a real run disproved it the next day: there is no such keyword. See § 5p.3o)* | superseded |
| the high-bias advisory **stopped firing** | `TransiestaEngine.preflight` is registered in `_ENGINE_VALIDATORS` keyed on `TransportConfig`; every rung now resolves a `SiestaConfig`, so it dispatches for nothing. Of what it carried, the region partition and atom order are `sort`'s own refusals and structural here, and open-shell runs from the siesta validator against the run's REAL spin treatment. The remainder was the \|V\| > 2 V advisory — and bias is the one axis transport exists to sweep | re-homed into `_validate_transport_kind`, beside the kz≠1 refusal. **And the dead checker itself is now GONE — 2026-09-17.** This row said it "dispatches for nothing" and left it standing, because ONE surface still built and validated a `TransportConfig`: `POST /api/transport/render`. Deleting that route (step 2c) removed the last reason, and the holistic review found two documents still calling the checker *"defense in depth"* for an ordering that is held by construction. `TransiestaEngine` is deleted; its tombstone names a live holder for every check it carried, verified one by one |
| the Methods paragraph named a basis nobody ran | under a docstring reading **"EVERY NUMBER HERE IS NOW ONE THE DECK CARRIES"**, the next line wrote `"a DZP basis, the PBE functional"` as literals. TZP/revPBE produced a paragraph claiming DZP/PBE — the same class the docstring calls *"the one defect with a PUBLICATION consequence"* | read off the config; verified both ways |
| `_sh` meant two things in one function | `_sh = shape_of(...)`, then `import shutil as _sh` **rebinds that function-local for the whole body**, so a later iteration hands the shutil module to `trial_work_dir` as a shape | `from shutil import copy2 as _copy2` |
| the electrode deck contradicted itself | a comment saying *"(Not `SaveHS`…)"* two lines above a layout that includes `OUTPUT_SECTION`, whose `write_hs` row has `anchor = "SaveHS"`, `value = true`. Rendered the lead: `SaveHS .true.` is there | the comment says what is true |
| `?contract=open` was a trap | the schema endpoint un-hid the seven contract fields for that lane while the describe door refuses them **unconditionally** — seven controls guaranteed to 400. `?contract=open` now selects nothing | the filter hides them in both lanes; the test that asserted the old rule rewritten to the shipped one, and mutation-tested |
| form persistence matched **zero elements** | `_restoreFormValues` queried `[name=…]`; `form-schema.js` sets `id` only, on all four builders | calls `formSchema.setValues`, the renderer's own door |
| the web door refused what the CLI accepts | `_known` checked `TransportConfig` only; `electrode_kz` is a `SiestaConfig` field and a catalogue row | widened to the union, the same vocabulary `prep` uses |
| dead code | `stages.render_stage_deck`'s unreachable refusal in `transport_spec` (9 lines), `if True:` wrapping 90 lines in `submit.py`, `contract_sealed`, six dead imports, two placeholder-less f-strings in emitted deck text | deleted |
| a **parameter that did nothing** | `config_from_template(..., calculation=)` — threaded by two callers, read by the body **never**, while `template.md` § 6.4 pointed at it as *the* enforcement site | deleted; its two tests became one |

#### Two agent verdicts I did not take

* *"`select(role=)` / `select(stages=)` have zero callers — dead."* True of
  production and the **wrong verdict**: `tests/test_template_role_and_stages.py`
  uses both as the reading API, and the alternative is every reader
  hand-filtering the catalogue. Kept. Six lines that earn themselves.
* *"Fold the DAG gather into `prep`."* I did, and **19 tests failed** — correctly.
  `gather_transport_inputs` refuses an upstream that has not CONCLUDED, so
  calling it at prep means you can no longer render the device deck to READ it
  before spending the queue. That is the split `prepare_attempt` names in its own
  words. Backed out; the browser-road gap is recorded below as a question about
  what `prep` MEANS, not a defect to close by widening it.

#### The documents

`template.md` § 6.4 is the `citation` marker's ONE HOME and still stated the
**sealed** reading § 2a.7 reversed — in three sentences, pointing at the dead
parameter above as the enforcement. Under this project's own convention every
other restatement points there, so the one home being wrong is what made the
other sites unfixable one at a time. Corrected first, then the restatements:
`transport.md` § 0.4 rows 1–2, § 0.5 (*"and no validation file… writes with
`write_text`"* — measured false: every rung goes through `prepare_deck` and
writes its `.validation.txt`), § 3.6a's three residue rows, § 6.1's diagram and
§ 6.1a's table (which named three renderers, **none** on the composite path, so
a reader fixing a keyword would have edited dead code), § 7's *"the basis / XC
block is hardcoded… edit the emitted `.fdf`"* — advice the read-back gate would
now refuse — and the module docstrings of `transport/stages.py`,
`transport/__init__.py` and `transport/deck.py`.

#### Open, and NOT closed by me

| | |
|---|---|
| the DAG gather runs on the CLI road only | a browser prep opens the attempt but never carries the upstream `.TSHS`/`.DM` in. Closing it means deciding what `prep` means — see above |
| `stages.render_stage_deck` (52 lines) and `config_for` (140) are production-dead | `config_for` still has three tests. Deleting it retires them; that is a call about what those tests are for |
| ~20 doc counts have drifted | "13 keywords / 45 items", "32 fields", "49 siesta items", "the 21 keywords". Historical measurements stated in the present tense. `test_doc_claims.py` pins some and not these |
| `wrap_into_cell` declares `role = ["transport"]` and nothing answers it | inert today because `_emit_geometry` never wraps. A decorative declaration. **Resolved 2026-09-25**: retired with the knob (W33) |

### 5p.3o MEASURED ON THE ENGINE — the pole-count keyword cannot act on our deck, and our default aborted every device run *(2026-09-16; heading corrected 2026-09-29 to its own table: the keyword is real)*

Asked to verify one thing (does the electrode rung need `TS.DE.Save`?), which
needed a real TranSIESTA run. The run answered that question and two others
nobody had asked.

**The method, because it is the point.** A 12-atom Au wire junction, decks
rendered through molbuilder's own emitters — not hand-written — then the lead
run on SIESTA 5.4.2, its `.TSHS` handed to the device, and the device run.

| question | answer | how |
|---|---|---|
| does the lead need `TS.DE.Save`? | **No** — and the reason given here first was the wrong one | the run showed it (device completed, exit 0, no electrode `.TSDE` present, `L/R principal cell is perfect!`), but a run is one configuration. **Settled from `m_ts_options.F90`**: the demand is guarded by `DM_init > 0` (`:1571`), which needs `TS.Elecs.DM.Init` = `bulk`/`force-bulk` (`:450-459`; the default chain lands on `diagon`) AND zero bias (`:460-468`; finite bias forces it to 0). This ladder writes none of those keywords. `TS.Elecs.Bulk` — what I first credited — is a different option, selecting whether the SELF-ENERGY uses the bulk Hamiltonian (`m_ts_elec_se.F90:49`). See `engines/transport.md` § 2a.15 |
| is `TS.Contours.Eq.Pole.N` a keyword? | **Yes — and it cannot act here.** | it is a real `fdf_get` (`m_ts_chem_pot.F90:113`). Our deck shape takes the continued-fraction branch (`:299`), where the count is overwritten from the ENERGY at `:319` and the branch default is non-zero, so the override always fires. Only the block-interior `contour.eq.pole.n` (`:263`) can set it. *(I first reported this as "not a keyword" on the strength of the run alone — see below)* |
| why did the device abort? | **our own default** | `negf_eq_pole_ev = 1.5` eV is 18 poles at 300 K, and TranSIESTA refuses fewer than 20. Measured: 1.5 → abort, 1.7 → 20, 2.0 → 24, 4.0 → 49, nothing written → 42 |

**And then the METHOD was wrong too, which is the lesson worth keeping.** I
established all of the above by poking the binary — running decks and reading
what came back — when the source was one `curl` away and the project's own rule
says *"let me try X and see about code I can open IS the violation"*. The user
said so, and reading `Src/m_ts_chem_pot.F90` changed two of the three answers:
the keyword is real (it simply cannot act on this deck shape), and the rule is
`N = int(E / (pi * kT))` with a `< 20` die at `:324` — which is what the
validator now cites, rather than the curve I had fitted to five data points.
The fitted curve happened to be right. It was right by luck, and a fitted curve
that is right by luck is indistinguishable from one that is not.

A third fact only the source gives: an energy written INSIDE the chempot block
means something different from the same energy written at the top level —
`:312` applies a 0.7 factor *"only up to 70% as they become sparse"*. No
amount of running our own decks would have surfaced that, because our decks
never write it.

**So the fix shipped the day before was wrong.** § 5p.3n added a
`CONTOUR_SECTION` emitting `TS.Contours.Eq.Pole.N`, on the strength of a
catalogue row whose help text asserted the diagnosis and a binary-strings check
that passed the label. Both were wrong in the same direction, and neither could
have been right: **a string in a binary is not proof that fdf queries it.**
That is now written into the guard's own docstring, because the guard passed
this keyword and will pass the next one of its kind.

The real defect was one line away from the one I fixed, and worse: a shipped
default that stops every device run after the queue wait, having read the
electrodes.

**What landed.** `negf_eq_pole_n` deleted — the field, the catalogue row and
the section. `negf_eq_pole_ev` defaults to **0 = let the engine choose**, the
established "0 means the engine's formula" pattern this very block already uses
for `TS.Contours.nEq.Eta`, and the emitter writes the line only when a person
names a value. The engine's choice scales with the temperature; a fixed number
cannot, which is the argument against simply raising 1.5 to 2.0.

And because the threshold is a RELATION, `_validate_transport_kind` refuses a
stated energy that gives fewer than 20 poles **at the run's own temperature**,
with the arithmetic in the message — 1.7 eV passes at 300 K and is refused at
1000 K. A deck rendered with the new defaults now runs to completion: 42 poles,
`Job completed`.

### 5p.3p TWO ERAS IN ONE PACKAGE — the composite's residue, and the layers below it *(2026-09-17)*

**The nature, in one sentence.** Transport was built twice. The FIRST build
(2026-06-10 to 06-27) assembled a junction **by hand** — you wrote a device
deck, derived an electrode deck, and a preflight compared the two because
*"humans break exactly these couplings"* (`transport/preflight.py`). The SECOND
— the composite, migrated 2026-08-29 (`workflow.md` § 7), finished by TR5b
(§ 5p.3i) — **derives everything from one cited junction**. The second shipped;
**the first was never removed.**

This section tracks that removal AND the layers underneath it that the removal
exposed. It is re-consolidated at every milestone (§ 5p.3p.5), because two
passes over it already changed the answer rather than refining it.

#### 5p.3p.1 The layer map, top down — contract, and state

Read this before touching anything here. Two of these layers were missed on the
first two passes precisely because the work started in the middle.

| layer | contract | state |
|---|---|---|
| the whole | `workflow.md` § 7 | ✅ records transport as the composite |
| tab + describe | `web/tabs.md`, `web/form-schema.md` | ✅ four routes, citation-driven |
| **CLI surface** | **`process/conventions.md` § 3** | ✅ **closed 2026-09-17.** Decision 7 + 34 (*"everything is a job set"*; `run` and `fdf` deleted as *"obsolete residue from the flat-dir design"*) now holds without exception: `transport` (steps 2c/9) and `pyscf` (step 8) were the two survivors of that shape and both are gone. **19** top-level commands, re-derived from `cli.commands`; no calculation KIND and no engine has a verb, and § 3 states that as a rule rather than leaving it to be inferred from the roster — see § 5p.3p.6 |
| **presenters** | **`web/presenters.md`** | ⚠ § 1's count CORRECTED 2026-09-17 (six register, four are results — `bench-summary` had been omitted). **Step 5 adds the composite row**; what it cannot fix is the LADDER (five directories, one run), which is § 5c.1's open question, not a presenter's. A module rename to `presenters` is declared pending (W15) and not started — the code is still `inspectors` |
| results tab | `web/results.md` | ⚠ § 2.3 now states the LADDER case and § 3 the real viewer count (2026-09-17). Still ⚠ because it describes a gap rather than a behaviour — closes with step 5 + § 5c |
| HTTP | `web/web-api.md` | ✅ four live transport routes; `/api/transport/render` deleted 2026-09-17 with its orphaned config builder |
| description | `engines/stages.md`, `execution/job-contracts.md` | ✅ five rungs, hierarchical shape |
| template | `engines/template.md` | ⚠ facts in two homes — `template.md` § 2.1a counts them (W48); the mirrored guard restored 2026-09-17 |
| engine / deck | `engines/transport.md` | ✅ § 6 rewritten 2026-09-17 to name ROLES not function lists (§ 6a says why); § 5's holders named at 2c. The unbuilt transmission inspector is now stated as unbuilt, in `results.md` § 2.3 as well |
| **parse / directory** | **`model/parse.md` § 5** | ✅ `JobDirParser` → `RunDirResult` **BUILT + MIGRATED 2026-09-18** (§ 5c steps 1–4, **CLOSED**; proved 141/141 on the real tree before any caller moved). The discovery chain is single-homed in `parse/dirs/rundir.py`; `web/blueprints/watch.py` lost 166 lines. **Five of the six caller-map rows were withdrawn with measurements, not deferred** — they were name collisions, not duplicate readers. The Results picker was struck from the map, not left open: it is not a caller of this door — its question is the LADDER's, answered by `jobset_status`, and an HTTP surface over that belongs to § 5p.3p. § 7.9's absorption-site count corrected 2026-09-17 (four → two; the other two were the deleted transport readers) |
| **engine registry** | **`engines/overview.md` § 5** | ✅ *there is no registry* — spectra's went at P3 (2026-08-21), transport's 2026-09-17. § 5 told a new engine to `@register_engine` against it until corrected 2026-09-17; `spectra/methods.py` and `transiesta.py` carried the same claim in docstrings |
| **what the wrapper is HANDED vs re-reads** | **`execution/gpu.md` G7 + `execution/architecture.md` A8** | ⚠ **added 2026-09-17, and it is what stopped step N5d being trivial.** G7: *"the value travels; the deck is not re-read for it"* — reached 2026-08-23 **by carrying the answer on `Resources`**, not by a declaration alone. A8: the allocation *"arrives whole"* — `render_wrappers` was cut from eleven loose kwargs to one record on 2026-08-17. So handing the wrapper a value means a `Resources` field or nothing; there is no third door |
| **validation dispatch** | **`science/validation.md`** | ✅ **added 2026-09-17, and it is where the day's biggest defect lived.** Two registries, and WHICH one a science belongs in is the whole question: `_ENGINE_VALIDATORS` keys on a **config class**, `_KIND_VALIDATORS` on `task.calculation`. A row in the first is only as live as the callers that CONSTRUCT that class — and transport's keyed on `TransportConfig` while every rung resolves a `SiestaConfig`, so it dispatched for nothing. Now two engine rows (SIESTA / PySCF) and two kind rows (transport / vibration), asserted by **equality** |
| execution | `execution/script-preparation.md`, `architecture.md` | ✅ |

#### 5p.3p.2 Redundant — tracked, with status

| the June piece | replaced by | evidence | status |
|---|---|---|---|
| `render_script` + `_emit_header` + `_emit_k_mesh` | `deck.py` (negf shape) | only caller was `/api/transport/render`; no browser POSTed there since 2026-08-29 | **DONE** |
| `render_electrode_fdf` + `electrode_wizard` + `format_models` | `_electrode_layout` via `as_structure()` | `prep.py:1528` is `as_structure`'s only caller | **DONE** |
| `molbuilder transport electrode` | `jobset prep` | a deck from flags — what `molbuilder fdf` was deleted for | **DONE** |
| `engine_base.py` Protocol + registry | a direct call | 4 declared members, 1 implemented, 1 engine, 1 caller | **DONE** |
| `results.py` + `sidecars/transport.py` + `parse/sidecars/transport.py` (697 lines) | `record.py` | `dump_transport_json`: **zero production callers in every revision**; the reader claimed only `schema_version`, the writer emits `schema` | **DONE** |
| `preflight_files` + `format_report` + `transport preflight` | `compose` + one resolved contract | gates compare two decks for drift two decks from one config cannot have | **DONE** (2c) |
| **`TransiestaEngine` + `TransiestaEngine.preflight`** | `_KIND_VALIDATORS["transport"]` + `sort`'s refusals + construction | **registered in `_ENGINE_VALIDATORS` under `TransportConfig`, and nothing validates a `TransportConfig`** — every rung resolves a `SiestaConfig`, and the two sites that build one build it as a projection for the lifted NEGF emitter. It stayed live only through `POST /api/transport/render`, deleted in 2c; every check it carried has a named holder, verified one by one | **DONE 2026-09-17** |
| `TransportConfig` behind `_legacy_view` | — | § 5p.3i: *"survives only to feed the lifted block"*; 3 orphan fields, all already in `UNRESOLVED_FIELDS` | **DONE 2026-10-02** — `_legacy_view` went 2026-09-29, the class with M5 step 3 |

**NOT redundant, so not deleted by association:** `extract_electrode_model` +
`ElectrodeModel` (`compose.py:570`); `_emit_transiesta_block` + `_emit_geometry`
(`deck.py` reuses both; § 5p.3i ruled the NEGF block stays — an off-by-one in an
electrode range converges while computing the wrong thing);
`_compute_cell_from_extents`, `_find_electrode_regions`,
`electrode_hs_stem`; `preflight.parse_fdf_params` (**six production import
sites**, re-measured 2026-09-17 — `citation_defaults`, `compose` ×2 symbols,
`parse/contract`, `web/blueprints/transport`; it said four);
`cell.detect_layers` / `bulk_z_period` — this same pattern **done correctly**,
moved out of the wizard so `add_slab` and the extraction share one copy.

#### 5p.3p.3 The Results gap is a MISSING LAYER, not three defects

The tab is a dispatch shell; all dispatch is client-side. `file-picker.js`
enumerates a directory, keeps what an `isResult` presenter matches, applies
`absorbs`, groups by `resultCategory`. Point it at a finished five-rung junction:

1. **The result is invisible** — `<label>.transport.json` matches no `isResult`
   presenter, so `pickResult` returns null and the picker drops it.
2. ~~**The ladder lists as a pile of optimizations**~~ — **MEASURED WRONG
   2026-09-18, and it was repeated all day before anyone checked.** The claim
   was that every rung's `.out` and `.molwatch.log` are claimed by
   `trajectory.js` under *"SIESTA optimization"*, beside the result.
   `file-picker.js` calls `projects.listDir(dir)` — **ONE directory, not a
   walk** — and transport runs `--shape hierarchical` (§ 3, the worked CLI),
   so each rung is its own folder. The calculation root holds the record and
   the rungs are one level down; they never appear beside it. *The claim is
   true of the FLAT shape, where stages share a directory, and was written
   without asking which shape transport uses.*
3. **`absorbs` cannot express a ladder** — it collapses siblings in ONE
   directory; a ladder is five directories that are one run.
4. **From the sidebar it is a text pager** — and by `presenters.md` § 1's own
   table, *a `.json` gets a plain text pane*. **The code conforms; the contract
   has no row for a composite result.**

**The cause is one door that was never built.** `model/parse.md` § 5 specifies
`RunDirResult` with `engine` (which engine ran), `files` (what is here),
`openable` (**what a VIEWER should load**) and `active` — § 5.1 draws exactly
the distinction this menu needs and calls conflating them *"the trap this
section exists to mark"*. The picker guesses all of it from filenames in JS
because **there is no server door to ask**: `/api/results/contract` answers
`info.calculation`, one of six fields.

**The Results picker is a SEVENTH consumer and § 5c's caller map does not list
it** — every row there is server-side. Transport is not the cause; it is the
first result that is neither a trajectory nor a spectrum, so the guess stops
producing a plausible answer.

**One question is genuinely new**: `openable` is one answer per DIRECTORY; a
ladder is five. Neither `absorbs` nor `RunDirResult` can say so. It reaches every
multi-rung calculation and needs a home before code moves.

#### 5p.3p.4 The steps — each one validates against its contract before moving

**Every step has the same three parts: the change, the contract it is checked
against, and what to do when the check disagrees.** A disagreement is not a
blocker to route around — it sends the step back to § 5p.3p.5.

**Step 1 — delete the results chain.** *(DONE 2026-09-17)*
*Contract:* `model/parse.md` § 4 (the registry), `job-contracts.md` § 6.1 (the
record's owner). *Check:* the parse registry builds and no longer lists
`transport-json`; `record.py` still writes. *Result:* passed — 163 tests in the
parse layer green; 697 product lines and their tests gone.

**Step 2 — delete the cross-deck comparison.** `preflight_files`,
`format_report`, the `transport preflight` verb; keep `parse_fdf_params` and
`_BOHR_ANG`. *Contract:* `engines/transport.md` § 5 (the invariant set) — read it
first and record, invariant by invariant, **which are now held by construction
by `compose` + one resolved config and which are not**. *Check:* every invariant
either has a construction-time guarantee named, or a reason it still needs a
runtime gate. *If the check disagrees* — an invariant with neither — the verb
does not go; the invariant gets re-homed first and § 5p.3p.2 is corrected.

**Step 3 — settle `TransportConfig`.** **DONE 2026-10-02 — M5 step 3: the class retired (TD4).** *Contract:* `engines/template.md` § 2.1a
(the two-homes debt) and § 5p.3i's open item. *Check:* re-measure the orphan
fields. *Measured 2026-09-17:* three (`log_level`, `num_threads`,
`transmission_relative_to_ef`), all already in `stages.UNRESOLVED_FIELDS` or
documented at `stages.py:110` — the shim leaks nothing. *So this is bookkeeping,
not a defect*, and the step is to record that, not to act.

**Re-measured after the `TransiestaEngine` deletion, same day — and the surface
got smaller.** `TransportConfig` now has **no validation role whatsoever**: the
`_ENGINE_VALIDATORS` row keyed on it is gone, so nothing anywhere validates one.
What remains is exactly two jobs, and both are real:

1. **the lifted NEGF emitter's input vocabulary** — built as a projection at
   `deck.py::_legacy_view` and `stages.py::config_for`, consumed by
   `_emit_transiesta_block`. This is § 2a.14's *"survives only to feed the lifted
   block"*, unchanged;
2. **the transport tab's form shape** — `web/blueprints/transport.py` renders
   `GET /api/transport/schema` from the dataclass and its
   `_form_section_order`.

So the retirement condition (TR6) is now stateable precisely: **`TransportConfig`
goes when the NEGF block is tabled AND the tab's schema comes from the
catalogue** — two things, both already named elsewhere, neither blocked on this
section.

**Step 4 — correct the engine contract** — **DONE 2026-09-17, and it was not
two lines.** *Contract:* itself, plus every document its § 6 is restated in.
*Check:* every symbol the Results section names resolves in the tree.
**The check found nine documents, not one**, and a shape problem under them:
§ 6's pieces table enumerated modules AND their function lists, so six of nine
rows named code deleted that week. Rewritten to name **roles**, with the
reasoning in the new § 6a — the rule being that a contract enumerates only where
the list itself is the guarantee (§ 5's thirteen invariants are; a pieces map is
not). Also corrected: `overview.md` § 5 (no engine registry exists — a new
engine is described and rendered through `spec_for`, it does not register),
`script-preparation.md` (its *"what this costs today"* block was entirely about
deleted code), `parse.md` § 7.9 (four absorption sites → two), `tabs.md` (both
"still open" items closed by deletion), `presenters.md` § 1 and `results.md`
§§ 2.3/3 (the six-vs-five count, and the ladder case), and the six documents
still naming `molbuilder pyscf`. `transiesta.py`'s class docstring and
`spectra/methods.py`'s module header both described registries that no longer
exist and were corrected in product code.

**Step 5 — the Results placeholder, and only that.** Register a transport
presenter matching `*.transport.json`, `isResult: true`, category `"Transport"`,
that reads the record through `ctx.readFile`, names it, shows the I–V table the
record already carries, and says what is **not** drawn.

> **Corrected at the step's own review, 2026-09-18.** This said *"states that
> the transmission surface is not built"*. Measured: every point in the record
> carries `energy_ev[]` and `transmission[]` beside `conductance_g0` and
> `current_a` — **the curve's DATA is there and only the CHART is missing.**
> Telling a person the surface "is not built" when the numbers are in the file
> they are looking at is the kind of claim this section keeps having to
> retract. Say the narrower true thing, and say how many points are in hand.
*Contract:* `web/presenters.md` § 2 (the presenter contract) — and **§ 1's table
gains a row in the same change**, because that table is the declaration of what
is presented. *Check:* the table's count matches the registry's; a finished
junction appears in its own menu. *If the check disagrees* — e.g. the six-vs-five
drift means the table has other gaps — fix the table wholly, not this row alone.

**Step 6 — kill the tests pinning the retired design**, with the code and not
before it. *Contract:* the admission question (default ZERO). *Check:* every
deletion names the product symbol it pinned. *Already done for steps 1 and the
June writers;* the audit of 2026-09-17 lists the rest, of which the ones that
matter are the tests that **cannot fail** (`test_siesta.py:559`/`:571`,
`test_pyscf.py:225`/`:235`/`:253`). **DONE 2026-10-02 (M5 step 3):** those five
went with M6 (`c71ba618`, 2026-09-29), and the step's own retirements went with
the code — `test_transport_config.py`, three `config_for` tests, eight builder
tests, `TestEveryExposedFieldIsTagged`, and a `pyscf` CLI test that passed on
click's *"No such command"*.

**Step 7 — `JobDirParser`, which is NOT this section's to do** — **the two
additions LANDED 2026-09-18**, having been written as *"made now and not
deferred"* on 2026-09-17 and then deferred twice while six other commits went
in. They are § 5c's caller-map row for the Results picker and § 5c.1 for the
ladder question. *The instruction to do it now was in this sentence the whole
time; I re-derived the same finding twice instead of reading it.* It belongs to
**§ 5c**. What this section owed § 5c: the **Results picker as a seventh consumer** with a server door to ask,
and the **ladder question** (five directories, one run), which § 5c does not
cover. Items 1–3 of § 5p.3p.3 close when § 5c lands and not before.

#### 5p.3p.6 The CLI surface — two survivors of a ruling already made

*(Added 2026-09-17 after the layer map was re-read top-down; the CLI was not on
the first three passes, which is why this is a new sub-section and the map in
§ 5p.3p.1 gained a row rather than this being a bullet at the bottom.)*

**The ruling exists and is general.** `conventions.md` § 3, decision 7 (user,
2026-08-11): *"everything is a job set. There is no `molbuilder run` — it is
deleted, not deprecated, because a second way in is a second way to lose your
results."* And decision 34, the same day: *"there is no `molbuilder fdf`"* —
user's words, *"obsolete residue from the flat-dir design"*.

**The verbs sort into four kinds**, measured 2026-09-17: structure construction
(`peptide` `dna` `rna` `smiles` `name` `modify`), the job pipeline (`jobset`,
`validate`), machine/ops (`envs` `serve` `jupyter` `checkpoint` `auth-setup`
`notify-token` `monitor`), and format utilities (`xv2xyz` `runtime-info`
`watch parse` `pseudo check`). **No calculation KIND has a verb** — there is no
`spectra`, no `optimization`, no `vibration`; a kind is described in `task.json`
and run through `jobset`. Two break the pattern.

**`transport` — a calculation-kind group.** `conventions.md` already names this
as a closed design problem: *"`jobset`, `bench` and `transport` each ran
calculations their own way"* (2026-08-11). `bench` got the full closure —
folded into `jobset`, **its four verbs deleted, the group removed**. `transport`
got half: the `bundle` verb went 2026-08-29, the group stayed. Its two remaining
verbs are the June hand-assembly workflow's UI: `electrode` *made* an electrode
deck from flags, `preflight` *checked* it against a device deck. **Both are that
era.** With `electrode` gone (§ 5p.3p.2) and `preflight` going (step 2), the
group is empty and follows `bench`'s precedent. *(User ruled 2026-09-17:
removing the transport verb is correct.)*

**`pyscf` — an engine verb, and `fdf`'s surviving twin.** Same shape: a
structure plus every engine field as a flag → a finished deck, skipping the
description. `cli.py` states the exemption itself — *"`pyscf` keeps its own only
because its ladder runs inside one emitted script"* — and **the same file
refutes it two comments later**: *"THIS COMMAND WRITES ONE DECK, AND A LADDER IS
N DECKS … there is no `--stages-json` / `--stage-strategy` here … `jobset init
--engine pyscf --stage-strategy …` is the one door that writes one."* The
exemption describes PySCF's *script shape*, not a property of the verb — and
`--stage-strategy` was taken off this command on 2026-08-18 for exactly that
reason.

The framework already covers it: `prep.py:651` builds a full `EngineSeam` for
PySCF, so `jobset prep` renders it through `spec_for` → `prepare_deck` like any
other engine. `render_script` is *"a thin call over `spec_for`"* and `convert`
is *"STEP 3, WHOLE, IN ONE CALL — the same call `prep` makes."* The verb
duplicates no mechanism; it is a second **way in**, which is the thing decision 7
names.

**Step 8 — delete `molbuilder pyscf`** — **DONE 2026-09-17, and it grew.** *Contract:* `conventions.md` § 3
(decisions 7 and 34). *What goes with it, measured:* `cmd_pyscf` +
`_make_pyscf_options_decorator` (~75 lines); **`add_dataclass_options`
(~168 lines), whose only production consumer is this command**; `pyscf.convert`
(no other caller); `pyscf.render_script` if nothing else points at it; and the
`test_cli.py` group that tests the BRIDGE rather than any behaviour. *Check:*
`conventions.md` § 3's roster and its count are corrected in the same change —
the table currently lists 13, names `serve` (now a group) and omits
`notify-token`, so the count is re-derived, not decremented. *The check found no consumer outside
`cmd_pyscf`, so the bridge went too.*

**The review found a TWIN nobody had noticed.** `siesta.input.convert` — the
same single-shot "read a file, write a deck" worker — **has had no production
caller since `molbuilder fdf` was deleted on 2026-08-11**. One month dead, in
the engine whose command went first. Both `convert`s are deleted here; the
symmetry is the finding, and it is why this step is filed as one change rather
than two.

**`render_fdf` / `render_script` STAY**, and the distinction matters: 43 test
files call them and *that is not the reason*. They are thin calls over
`spec_for` and `engines/siesta.md` / `engines/pyscf.md` name them as each
emitter's public surface. A test never justifies code; a CONTRACT does.
*(Reversed 2026-10-07, W57: deleted, their tests rows down the road -- user:
"retire render_fdf and render_script, move their tests onto the road".)*

*Rosters re-derived, not decremented:* `conventions.md` § 3 said **13**
top-level commands, listed `pyscf`, and omitted seven others — it is 19 now,
measured from `cli.commands`. Both engine contracts and the index carry the
`convert()` tombstone.

**Step 9 — remove the `transport` group** — **DONE 2026-09-17 with 2c.**
`molbuilder --help` lists **19** top-level commands and no calculation-KIND
verb; `conventions.md` § 3 records the rule explicitly rather than leaving it
to be inferred from the roster. *(This line said 20. Re-derived from
`cli.commands` during step 4's review: 19, the same number step 8 recorded two
sub-sections above — a count written from memory one step after it had been
measured correctly.)*

#### 5p.3p.8 Steps 10 and 11 — the two fixes the 2026-09-17 sweep OWES *(added 2026-09-17)*

That sweep found nine documents wrong and corrected all nine. **Correcting them
is not a fix** — they were correct when written and decayed the same way the
next nine will. These two steps are the mechanism, and both are small because
the repo already proved the pattern.

**Step 10 — put the three surviving enumerations under a MEMBERSHIP assert.**

*Why these three and no others:* an enumeration earns an assert when a reader
ACTS on it — picks a route, adds a presenter, runs a command. Prose that merely
counts something does not (`test_doc_claims.MEASURED` already covers the counts
that matter).

| # | the list | assert it against | today |
|---|---|---|---|
| 10a | `web-api.md` §§ 4–5's route catalogue | `app.url_map`, **set equality**, `static` excluded | ⛔ **BLOCKED — measured 2026-09-18, and the measurement is the finding.** See below |
| 10b | `presenters.md` § 1's viewer table | the `register()` calls in `lib/inspectors/`, **set equality** on presenter name + `isResult` | ✅ **DONE 2026-09-18** — `test_inspector_registry_dispatch_js.py::test_the_documented_presenters_are_the_registered_ones`, which since 2026-09-26 runs the registry in node and compares its `list()` with the table (it read the `register()` calls by regex before). The table gained a `Presenter` column, because it was keyed by file pattern and had no name to assert on. Mutation-tested four ways, including the `makePartialInspector` default-`true` case that produced the original wrong count |
| 10c | `conventions.md` § 3's command roster | `cli.commands`, **set equality** | ✅ **DONE 2026-09-18** — `test_the_documented_command_roster_is_the_shipped_one`. Mutation-tested both directions |

**10a — why it is blocked, and what the measurement says.** The set equality is
written and runs; the document cannot pass it yet, and the fix is **not** to
relax the test.

*Parsing it is solved.* Three things had to be handled and all three are:
§ 4's tables use brace shorthand (`/api/files/{roots,list,stat,…}`), which
expands; § 5 documents the un-owned routes in prose, so both sections count;
and **blockquote lines are excluded, because that is where this document keeps
its history** — without that, every tombstone reads as a live claim.

*What is left is a content problem, in both directions:*

* **11 routes are claimed in ordinary prose and do not exist** —
  `GET /api/checkpoint/config`, `GET /api/docs/list`, `GET /login`,
  `POST /api/admin/reload`, `POST /api/build/fdf`, `POST /api/build/pyscf`,
  `POST /api/results/bundle`, `POST /api/run/install-wrapper`,
  `POST /api/selection/atoms`, `POST /api/siesta/install-pseudos`, and
  `GET /api/docs/img/<path>` (a placeholder-spelling difference from the live
  `<path:img_path>`). **At least two are conditional, not dead**: `/login`
  registers only with an `auth` config, and the document already records that
  *"a production config with rate limiting on registers a few additional
  admin/auth routes"* — so the assert needs a declared conditional set, not a
  deletion.
* **9 live routes are claimed nowhere** — every tab page: `/documents`,
  `/jupyternb`, `/molview-demo`, `/results`, `/spectrum-calculation`,
  `/structure-optimization`, `/task-setup`, `/this-machine`, plus
  `/api/docs/img/<path:img_path>`.

***`web-api.md` itself forbids the silent fix***, and it is right: *"Each needs
checking for **retired** versus **renamed** before its row is deleted, which is
the route-catalogue sweep, not a silent edit here."* Deleting ten rows I have not
individually traced would turn a stale index into a **wrong** one, and a renamed
route that quietly loses its row is exactly the drift step 10 exists to stop.

**So 10a lands with the route-catalogue sweep** (§ 0a's *Unscheduled*), and this measurement is its input: the parser is
written, the two directions are separated, and the conditional-route case is
named. *(The count test stays until then — it is weak, but it is not nothing.)*

*Model to copy:* `test_doc_claims.py::test_the_documented_L1_index_is_the_enforced_one`
— it read the documented set out of the table, read the enforced set out of
the code, and asserted **both directions** with a failure naming each side's
extras. It caught `ref`'s deletion on the first run after it. *(Retired
2026-09-27 with the layer scan whose copy it compared against; the pattern
stands for a set the RUNNING product states -- the routes the app serves, the
commands the CLI accepts.)*

*Done when:* deleting a route, a presenter or a command **fails a named test
that quotes the document and the line**, and each test is mutation-tested by
doing exactly that.

*The rule each one encodes:* **membership, never a count.** A count is satisfied
by any two errors that cancel, which is not a hypothetical — it is what 97 was.

**Step 11 — the deletion protocol, because no assert can catch what this
missed.**

Step 2c deleted `POST /api/transport/render`. That route was the **last live
caller** of `TransiestaEngine`, so the deletion silently made a whole class
unreachable — and for a day two contracts went on describing it as a live gate,
one of them calling it *"defense in depth"* for an ordering held by
construction. No membership assert would have caught that: the class was still
imported, still registered, still green.

**DONE 2026-09-18 — written into
[`process/code-audit.md`](?doc=process/code-audit.md) § 1d**, beside § 1c, which
is the other thing review must carry because a test cannot. The three actions
are unchanged; **point 2's stated reason was wrong and is corrected there**, by
measurement.

1. **Name what the deletion makes UNREACHABLE one layer away.** *What was this
   the last caller of?* `/api/transport/render` was the last thing that built a
   `TransportConfig` **and validated it** — the only reason
   `_ENGINE_VALIDATORS[TransportConfig]` still dispatched. The question was
   never asked and the answer was a 316-line class. *(A second case landed the
   same day: `render_electrode_fdf` was the last caller of
   `_emit_basis_and_xc`, which then outlived it by a day in code and longer in
   prose — four separate statements asserted callers it did not have.)*
2. **Sweep for the identifier — and READ every hit, including the ones in
   docstrings.** This said *"a symbol grep for `electrode_wizard` does not match
   'the electrode wizard' … **every miss in the 2026-09-17 sweep was of this
   shape**"*. **Re-derived 2026-09-18 against the actual misses: that is the
   MINORITY shape.** There are three, and a protocol written against the prose
   one catches almost none of them:
   **(a)** the prose name carries no symbol — nothing to match;
   **(b)** the grep *hits*, in a docstring, and the hit is dismissed as "just
   prose" — `transport/__init__.py` named three **deleted modules** and a
   nonexistent decorator, and a grep for any of the four **would have hit it at
   five lines**, every time, for a day;
   **(c)** the symbol never existed, so the grep returns one hit — the prose
   inventing it — and *absence* reads as *nothing to do*: `JobMonitor`,
   `transport.transiesta.validate`, `deck.py::_transport_view`.
   The rule is therefore **not "grep harder"**: a hit inside a docstring is a
   CLAIM, and a claim gets read. Cheapest reliable form — open the package
   `__init__.py` and the module header of every file the deletion touched.
3. **Re-run step 10's membership asserts** — one command, and 10b/10c exist
   now. **Necessary, not sufficient**, which is why § 1d sits beside them: no
   membership assert would have caught `TransiestaEngine`. It was still
   imported, still registered, still green.

*Done when:* the protocol is written into `process/code-audit.md` as a numbered
rule beside D1–D4 — it is a code-audit discipline, not a transport one, and it
is the third time this exact failure has been paid for (`molbuilder fdf` 2026-08-11,
spectra's registry 2026-08-21, transport's 2026-09-17).

#### 5p.3p.7 Step 2's review — the invariant set, one by one *(2026-09-17)*

`transport.md` § 5 declares thirteen invariants and says `transport preflight`
*"turns the prose Golden Rule into automated gates — the single biggest
correctness lever."* Step 2 proposed deleting that verb. **The review says
not yet**, and found a defect while checking.

| | invariant | held today by |
|---|---|---|
| I1 | XC functional + authors | **construction** — every rung resolves from one template |
| I2 | pseudopotentials | **construction** — the lead's atoms ARE the device's, extracted |
| I3 | MeshCutoff | **construction** — one template |
| I4 | PAO.EnergyShift | **construction** — one template |
| I5 | basis tier | **construction** — one template |
| I6 | lateral cell (a, b) | **construction** — `extract_electrode_model` takes the device's `lat_a`/`lat_b` verbatim |
| I7 | transverse k commensurate | **construction** — `citation_defaults` carries the transverse pair to both |
| I8 | device kz = 1 | **live gate** — `validation/__init__.py:528`, error |
| I9 | electrode kz dense | ⚠ **only `preflight.py:357-361`** |
| I10 | electrode geom = device frozen layers | **construction** — the extraction clones them |
| I11 | thickness ≥ principal layer | **`compose.py:596-640`, and BETTER** — it reads real orbital ranges from the citation's `.ion` files and refuses with the numbers. Its own comment retires the preflight's 12 Å floor as *"a GUESS made before anything was read … wrong about exactly the leads this measurement exists to judge"* |
| I12 | z-vacuum ≈ 0 at the leads | ⚠ **only `preflight.py:404-408`** |
| I13 | electrode writes its HS | **construction** — `TS.HS.Save` is a role item on the electrode rung |

**Eleven of thirteen are held without the verb, and one is held better.** Two
are not held at all once it goes.

**~~And I12 exposed a live defect~~ — RETRACTED the same day, 2026-09-17.**

I reported that `cell.vacuum_thin` fires on a transport device and advises
≥ 8 Å per side on the transport axis — the edit I12 says severs the lead. That
was **wrong**, and it is recorded here rather than deleted because the way it
was wrong is the point.

The check already filters on `axis_kind == "isolated"`, and `compose.py:777` /
`sort.py:326` both carry `axis_kind` through. Measured: the same atoms declared
as a junction (`periodic, periodic, transport`) produce **no vacuum finding at
all**. It fired only on a bare test fixture that carries no cell and no
`axis_kind` — a structure that declares itself an isolated molecule, which the
check then correctly described.

**I measured a fixture and reported it as the product.** Had the "fix" gone in,
it would have broken correct advice for every isolated molecule — nearly every
other calculation — to repair nothing. *(User caught the cross-task risk before
the edit: "make sure this is not conflicting with other tasks — these
suggestion is correct in those context.")*

What survives is smaller and real: **nothing requires a transport structure to
declare a transport axis.** `_validate_transport_kind` gated bias, net charge,
the pole energy, `tbt_k_grid` and `kgrid` when this was written (the two grids
are the k-point mesh's since 2026-09-30, `engines/siesta.md` § 6.1), and not
this. A junction with no
declared axes still renders — `_lattice_block` fabricates a vacuum box and says
so loudly IN THE DECK, but no validation Issue is raised, so nothing reaches the
form. That is a candidate gate, not a defect, and it is I12-adjacent rather than
I12 itself.

**Revised step 2, in three parts:**

2a. ~~Fix the vacuum advice~~ — **VOID**, the finding was retracted above. The
    check is correctly gated on `axis_kind` and needs no change. *Optional, and
    separable:* a gate requiring a transport structure to declare a transport
    axis, beside I8 in `_validate_transport_kind`. It is not a prerequisite for
    2b or 2c.
2b. **Re-home I9 and I12** — **DONE 2026-09-17.** Both now live in
    `_validate_transport_kind`, beside I8, keyed on the calculation KIND so they
    fire on every prep rather than on a command somebody remembers to run.
    I9 → `config.electrode_kz` (error at 1, warn below 20); I12 →
    `cell.transport_vacuum` (warn above 3 Å). Both thresholds are
    `preflight.py`'s own numbers, kept so the re-homing changed no verdict.
    *Check, passed:* seven tests in `tests/validation/test_transport_kind.py`,
    each carrying its discriminating half — the shipped `electrode_kz` default
    says nothing, a seamless cell says nothing, and **neither rule reaches a
    non-transport calculation**, which matters because I12 is the REVERSE of
    what `cell.vacuum_thin` tells an isolated molecule. Mutation-tested:
    disabling either gate fails its test. `transport.md` § 5's rows now name
    the live checks, and its "single biggest correctness lever" claim is
    corrected in the same change.
2c. **Delete the comparison and the verb** — **DONE 2026-09-17.** Gone:
    `Check`, `PreflightReport`, `preflight`, `preflight_files`,
    `format_report`, `molbuilder/transport/_cli.py`, and the `transport` group
    itself. Kept: `parse_fdf_params` (four production callers), `_parse_fdf`,
    `FdfParams`, `_BOHR_ANG` — reading an fdf and COMPARING two of them are
    different jobs, and only the second lost its subject.
    *Check, and it FAILED first:* § 5's table named `preflight.py`'s check-ids
    for eleven of thirteen rows, so "every invariant has a named holder" was
    false the moment the module went. Every row now names its real holder —
    seven **construction**, I8/I9/I12 `_validate_transport_kind`, I11
    `compose.py`. **Step 9 lands with this**: the group is empty, so it goes on
    the `bench` precedent, and `conventions.md` § 3's roster is re-derived
    (it said seven sub-groups and named `bench`, deleted 2026-08-17).

#### 5p.3p.5 The standing rule — consolidate, do not patch

**Four passes over this section have each CHANGED the answer rather than
refining it**, and every correction came from opening a contract the previous
pass had not:

| pass | what it said | what the next contract showed |
|---|---|---|
| 1 | *"no transport inspector exists"* | true, and the least of it |
| 2 | the picker drops the record and mis-files the ladder | `presenters.md` — the code CONFORMS; the contract has no row |
| 3 | three Results defects | `model/parse.md` § 5 — one unbuilt door underneath all three |
| 4 | the transport package | `conventions.md` § 3 — the CLI layer was never on the map |

That is the argument for the protocol below. It is not process for its own
sake; it is what four wrong answers cost.

##### The milestone review — run it BEFORE each step, and record the result

A step does not start until this has been done and its outcome written into
this section. It is five questions, top down:

1. **Re-read the contract that owns the layer being touched** — the row in
   § 5p.3p.1. Not the code first. The code is what the contract is checked
   against, never the source of the answer.
2. **Re-measure every claim § 5p.3p.2 makes about that layer.** A count in this
   plan is a hypothesis until re-derived. Anything that moved is corrected in
   place, with the date.
3. **Ask whether a LAYER is missing from § 5p.3p.1** — the failure that
   produced passes 3 and 4. The test: name the document that owns each layer
   the change touches; if a layer has no row, add it before proceeding.
4. **Check for drift** — does the code still do what the contract says, and does
   the contract still describe what shipped? Both directions. A disagreement is
   recorded as a finding here, not fixed silently in passing.
5. **If anything from 2–4 changed the picture, STOP and re-consolidate** —
   rewrite § 5p.3p.1 and § 5p.3p.2, renumber the steps if their order no longer
   holds. **Never append a bullet to the bottom.** A plan that grows by
   accretion is the thing this section exists to end.

##### The log — every review, recorded

| date | before step | what was re-read | what changed |
|---|---|---|---|
| 2026-09-17 | 2 | `presenters.md`, `model/parse.md` § 5, `conventions.md` § 3, `workflow.md` § 7, the contract index | CLI layer added to the map; Results gap re-framed as one unbuilt door; `pyscf` verb found obsolete by decision 34 (§ 5p.3p.6) |
| 2026-09-17 | **1, 2b, 2c — RUN LATE** | `model/parse.md` § 4, `job-contracts.md` § 6.1, `science/overview.md` § 4, `conventions.md` § 3 | **These three steps SHIPPED WITHOUT THIS REVIEW**, against the rule two sub-sections above. Run retroactively when asked whether the rule had been followed; the honest answer was no. Step 1: clean, nothing to correct. **Step 2b: the check catalogue (`science/overview.md` § 4) had NO transport section at all** — seven shipped checks uncatalogued, including the two that step added. **Step 2c: five documents still named the deleted verb, and one of them was a claim step 2b itself had falsified** — `transport.md` § 2a.13 row 2 said the warn "lives only in the standalone verb, so a composite run never sees it", hours after the re-homing that moved it. All corrected. *The reviews that ran on time (2 and 8) each changed the step; the three that ran late each found drift the step had introduced and left. That is the cost, measured.* |
| 2026-09-17 | **2 — STOPPED** | `engines/transport.md` § 5 (the 13 invariants), `compose.py`'s gates, `validation/__init__.py` | **The check disagreed and step 2 does not proceed.** 11 of 13 invariants are held by construction or by a live gate, and I11 is held BETTER (`compose` reads real orbital ranges); **I9 and I12 are held ONLY by the verb step 2 would delete**. I also reported a vacuum defect here and **RETRACTED it the same day** — I had measured a bare test fixture and reported it as the product; the check is correctly gated on `axis_kind`. See § 5p.3p.7 |
| 2026-09-17 | 8 | `conventions.md` § 3 (decisions 7 + 34), both engine contracts, `cli.py`'s own comments | **Found a twin.** `siesta.input.convert` has had no production caller since 2026-08-11 — the same shape as the PySCF one, dead a month, in the engine whose verb went first. Both deleted together. `render_fdf`/`render_script` kept: 43 test files call them, but the reason they stay is that the engine contracts name them, not the tests |
| 2026-09-17 | **4 — RUN ON TIME** | `engines/transport.md` § 6, `web/results.md`, `web/presenters.md`, `engines/overview.md` § 5, `execution/script-preparation.md`, `model/parse.md`, `web/tabs.md`, `process/conventions.md` § 3 | **The drift is an ENUMERATION problem, and it is not transport's.** Four hand-maintained lists were wrong the same week, each missed by the sweep that correctly fixed the *rules* beside it: § 6's pieces table (6 of 9 rows named deleted code), `presenters.md` § 1 (**5 viewers / 3 results declared, 6 / 4 measured** — `bench-summary` omitted entirely), `results.md` § 3 (*"the three viewers"*), `overview.md` § 5 (told a new engine to `@register_engine` against a Protocol deleted that week). Recorded as `transport.md` § 6a, which also states the rule that follows: **a contract enumerates only where the list IS the guarantee.** Two claims I had restated without resolving turned out never to have existed — `render_checks` on transport's engine base (its Protocol declared `render_script`/`parse_output`/`preflight`/`methods_fragment`) and `molbuilder siesta`. Step 8's sweep was **incomplete**: `engines/pyscf.md` carried a `convert()` tombstone AND, eight lines below it, the live `convert()` docs plus a runnable `molbuilder pyscf` example — six documents still named the deleted verb. One product-code orphan found and removed: `_transport_config_from_params`, zero callers since the route went in 2c |
| 2026-09-18 | **5 (the Results presenter) — RUN BEFORE** | `web/presenters.md` §§ 1–2 whole, `transport/record.py`'s record shape, the six `register()` calls, `web/results.md` § 2.3 | **The record already carries the curve.** Each point holds `energy_ev[]` and `transmission[]` beside `conductance_g0` and `current_a` — so step 5's instruction to *"state that the transmission surface is not built"* is too broad: the DATA is present and only the CHART is missing, which is a smaller and more honest thing to say. Step 5's text corrected. Also re-checked: `presenters.md` § 1 says six viewers / four results and the registry has six and four (corrected earlier today, still true), so the step's *"the table's count matches the registry's"* check is already satisfied and the row is a pure addition. No layer missing — § 5c's seventh-consumer row, which WAS the gap, landed 2026-09-18 |
| 2026-09-17 | **the fdf consolidation (N5a/b/d/f) — RUN BEFORE, on demand** *(user: "you are ignoring the rule we set for every step again")* | `engines/template.md` § 6.1 (`read_by`), `execution/gpu.md` G7, `execution/architecture.md` A8, `model/parse.md` §§ 1a + 4, `tests/test_layering.py`'s enforced sets | **It changed the step, which is the whole argument for the rule.** I had written N5d as *"two declarations on existing catalogue rows"* and was about to build it. § 6.1 says `read_by` **documents** a dependency and does not carry a value — G7 landed *"by carrying the answer"*, `resolve` putting `use_gpu` on `Resources`. Measured: neither `render_wrappers` nor `write_run_wrapper` receives the unsuffixed label, and **A8 forbids the loose kwarg** (that signature exists because eleven were removed 2026-08-17). So the label reaches the wrapper only via a new `Resources` field — identity riding a record that carries allocation/retry/notify — which is a **design decision awaiting a yes**, not a mechanical change. Also: § 5p.3p.2 said `parse_fdf_params` had *four* importers; it has **six**. Also: a LAYER was missing from § 5p.3p.1 — *what the wrapper is handed vs what it re-reads* — and it is precisely the layer that made the step non-trivial. **N5a (the move) is unaffected and stays first**, and it is what makes the cheap alternative to N5d possible |
| 2026-09-17 | **the § 5l retirement + a whole-document sweep — REVIEW RUN AFTER, and the user had to ask for it** | `plan.md` § 5l end to end, `architecture.md`, `science/validation.md`, `web/presenters.md` + `web/results.md` whole, `engines/transport.md` whole, `engines/pyscf.md` whole, `engines/overview.md` § 5, `execution/script-preparation.md` | **The finding is that SYMBOL-grep cannot see stale PROSE, and four passes of it had left the documents contradicting themselves.** § 8 of `transport.md` still shipped *"the electrode wizard"*, *"the `electrode`/`preflight` helper CLI"*, *"the render endpoint **remains**"* and *"arrives as a **registered engine**"* — every one deleted, none matchable by a symbol search. `results.md` § 7 told a reader to click a **Bundle** button § 5 of the same document records as deleted three weeks earlier. My own `presenters.md` edit that morning fixed ONE count and left **four** other places in that file saying five, and the presenter contract omitted `absorbs` entirely while claiming "two optional". **And the claim I had made that morning — that `ref.py` was named by no document — was FALSE**: `architecture.md`'s L1 index and `test_layering.py` both carried it as a bare `` `ref` `` in a list, which my grep for `molbuilder/ref.py` could not match. **The substantive find: `TransiestaEngine` was entirely dead** — registered under `TransportConfig`, which nothing validates; it had stayed live only through `/api/transport/render`, a route step 2c deleted without anyone noticing the consequence, and two documents were calling it *"defense in depth"* for an ordering held by construction. Deleted after verifying a live holder for every check. Two tests were holding the dead registration up; one pinned a `validate()` call **no production caller makes** and is gone, the other used `<=` so a dead row could hide and is now equality |
| 2026-09-17 | 6 (tests) | the repointed tests themselves, against the live deck | **A repoint is not a free pass.** Four checks were carried over from the deleted writer; three pinned deck TEXT with the emitter's column spacing baked in, and one -- `used-atoms` -- claimed to prove a count was "derived, NOT hardcoded" while asserting the literal `3`. Rewritten to parse the deck and compare against the structure's own regions; **mutation-testing then showed the rewrite STILL passed** against a hardcoded emitter, because the fixture's leads are 3 and 3. Now driven by an asymmetric junction (2 and 4), which is the only shape that can tell a derived count from a constant |

##### What "done" means for this section

**Five of the eight reviews logged above ran LATE**, and the pattern is now
measurable rather than anecdotal: *every review that ran on time changed the
step; every review that ran late found drift the step itself had introduced and
left behind.* 2c deleted `/api/transport/render` and thereby killed
`TransiestaEngine` without noticing — a day later the class was still documented
as a live gate in two contracts.

**The answer is NOT "run the review harder", and this section should stop
implying it is.** Five failures out of eight is not a discipline problem to
exhort away; it is a protocol asking for something at the wrong moment. A review
that runs *between* steps can only find what the last step broke, after it is
broken — and the reviewer is the person who just broke it, which is the worst
moment to ask.

**So the check moves INTO the step: § 5p.3p.8 step 11, the deletion protocol.**
Three questions asked while the deletion is being written, when the answers are
in hand — *what was this the last caller of?*, *what is this thing's PROSE name
in the documents?*, *do the membership asserts still pass?* Each of the three
corresponds to a specific thing this section paid for on 2026-09-17, and none of
them needs a review to be scheduled. **The review then does what a review is
for** — checking the plan against the contracts — instead of standing in for a
check the step should have made itself.

It is not "the steps are ticked". It is: every row in § 5p.3p.1 reads ✅, every
row in § 5p.3p.2 is DONE or has a recorded reason to stay, and the three
documents named in § 5p.3p.3 agree with the tree. Until then this section is
open, whatever the step list says.

### 5p.4 The steps

Ordered so each one is verifiable on its own and nothing depends on a later
step. **Every step ends with a check that can fail**, not with a declaration.

| # | step | reuses | done when |
|---|---|---|---|
| **TR1** | **Transport writes a template.** `jobset init --calculation transport` produces `<label>.template.toml` alongside `task.json`, its Class A values **defaulted from the cited relaxation** (§ 2a.7, ruling 1). The one adaptation needed: the description builder assumes a `StructureRef`, and a transport description has `slots` instead | `build_description` · `template_with_values(…, calculation="transport")` — already narrows the catalogue by kind | **✅ DONE 2026-09-16 — § 5p.3c.** the folder holds a template; `read_template` opens it; its values equal the citation's; changing one and re-prepping changes every stage's deck |
| **TR2** | **The three missing rows** (G8) — `TS.HS.Save`, the equilibrium pole count, `TS.Voltage` — added to the catalogue with their `SiestaConfig` fields | the one catalogue; `test_catalogue_agreement` | **✅ DONE 2026-09-16 — § 5p.3d.** the rows exist and agree with the fields; a device deck can state a bias from its description |
| **TR3** | **`kgrid` resolved for transport** (G9). The transverse grid and the transport axis stop being one row | § 2a.13's statement of the three classes | **✅ DONE 2026-09-16 — § 5p.3d**, as a CHECK rather than a row split. a description can set the transverse grid and the lead's axis separately, and the device's axis is not settable at all |
| **TR4** | **`prep` resolves transport through the one path** (G6). `_prep_transport` keeps the compose — genuinely new input — and hands to `resolve`, so a transport run has a `ParameterSet` with provenance | `resolve` · `effective_config` | **✅ DONE 2026-09-16 — § 5p.3e.** `--pipeline-log` produces a log for a transport prep; every value in a deck has a recorded source |
| **TR5** | **The remaining four rungs onto `spec_for`** (G7). **UNBLOCKED 2026-09-16** (§ 5p.3f): each rung renders the structure it describes — the lead EXTRACTED from the one cited structure by its label, the junction for the device — so the seam stays `(struct, cfg)` and `ElectrodeModel` need only yield a `Structure` | `transport/deck.py` (the seed arm, landed 2026-09-15) · `siesta/layout.py` sections | all five decks have a `.validation.txt`; all five pass the engine check gate; none is written by an f-string |
| **TR6** | **`TransportConfig` retires** (G2, G5). The shape is `SiestaConfig`; the projection introduced for the seed goes with it | — | `config/transport.py` does not exist; nothing maps one vocabulary to another — **✅ DONE 2026-10-02 (M5 step 3)**; the one table left mapped the RECORDED contract's names to the catalogue's (`parse/contract.RECORD_TO_SIESTA_FIELD`), D2's question — answered "unify", and gone with `e1d88b22` |
| **TR7** | **The transport tab reads the catalogue** (G3). Its bespoke form schema is deleted; the parameter surface is the kind-aware catalogue route, and per-stage values are the **stage table** rather than a second invention | `GET /api/build/schema/siesta?calculation=transport` · the task-setup stage table | `dataclass_to_form_schema` has no callers; the tab shows one shared panel and a per-stage table — **✅ DONE**: the tab reads the catalogue since 2026-09-24, and the builder was deleted 2026-10-02 (M5 step 3) |
| **TR8** | **Overrides route to the rung that owns them** (G4 — the live defect). The `stages` declaration landed 2026-09-16 (§ 5p.3b); the routing that reads it is what remains | the declaration, once approved | **✅ DONE 2026-09-16 — § 5p.3k.** setting the T(E) window reaches the transmission deck and nothing else |
| **TR9** | **Grouping** — the preparatory block as one submission (§ 2a.7). ⚠️ **Reconcile first** with `task-setup.md` § 1 (below) | `_plan_chain`'s shape — one submission walking a list | one command prepares and launches seed + both leads; the device and transmission stay separate |
| **TR10** | **Results reads the transmission** (§ 2a.12), with the treatment label and the provenance chain | `transport/record.py` | a curve carries how it was computed; a linear-response I–V says so |

### 5p.5 A contract tension TR9 must resolve, not ignore

[`web/task-setup.md`](?doc=web/task-setup.md) § 1 states, as contract:

> *"There is no 'run all stages' button, and that is deliberate. A stage is a
> long job, and a chain that continues by itself can spend a week refining a
> geometry you would have rejected in a minute."*

§ 2a.7's grouping ruling proposes seed + both leads as **one submission**. The
two are reconcilable and the argument must be written down rather than assumed:

* that rule is about a **ladder**, where each rung refines the last, and its
  harm is *continuing past a result you would have rejected*. Transport's three
  preparatory rungs are **independent** — none refines another, and there is no
  intermediate geometry to reject;
* it is one submission of three runs, **not** an auto-continuing chain. The
  precedent already exists and was accepted: the bias chain is one submission
  walking several runs;
* the thing the rule actually guards — *nothing starts the expensive stage for
  you* — is preserved exactly. The device still waits for a person.

**If that argument does not hold, TR9 is withdrawn rather than the rule bent.**

### 5p.6 Where each step's rows live

Per § 0, the open rows belong in § 2. This section holds the order and the
reasoning; § 2 holds what is outstanding.

---


---

## 10. Consolidated status — the 2026-09-19/20 session

> **Verdict (2026-10-08 validation):** a 2026-09-20 snapshot: 10.1 confirmed (`calcdirs.py`; `results.py:163-194`; `file-picker.js:242-282`; `viewer.js:914-940`); 10.2a confirmed (`results.md:140-206`); the other tabs CLOSED (`file-picker.js:450,630`).


One arc, started by an end-to-end workflow test and ended by the page
refactor that test's findings argued for, then a review round that found six
defects in the refactor itself. **All pushed** (`e06d345c..139b11b1`); both
gates green — `none2e` 9286/0 and `e2e` 164/0.

### 10.1 What is DONE

**A directory says what it is** — `calcdir.json`, § 1.4a's mechanism.
Written by the creator, read by one module, and the tuned walk it replaces
(`_CALC_SEARCH_DEPTH = 4`) deleted. Every consumer that used to guess now
asks: the parse door, the Results route, the checkpoint panel.

| | |
|---|---|
| `dc29f3df` `5334b76d` `76282e71` | the record, its contract, and the walk's deletion |
| `6ddc551a` `e808ce68` | the Results door asks what a directory IS; the page says so |
| `421d79f7` | the checkpoint panel asks instead of counting path segments |
| `75b71bb6` | an unmarked directory is not asked to invent a run state |
| `84553f7a` | a container has no run state but may have a PRODUCT |

**The render consumes the API.** The picker and the presenters take the
server's per-file answer instead of re-deriving it from filenames.

| | |
|---|---|
| `46fdfffa` `b5a7b858` | `resultCategory` takes `meta`; the contract says so; it reaches the heading and the mount |
| `06229ad8` | which files are one run is READ BACK (`runfiles.parse`), not cut out of a name by regex |

**One fact, one home.** `c286af2d` (a trial's warm files under the trial's
own label), `0ab07a14` (the browser prep writes the ledger it advertises),
`6916b4fc` (the progress channel lives in the run).

**The pseudopotential is explicit at configuration time** (user ruling).
`fcfde4c7` — the file SIESTA opens must exist and be named for its element;
`57c709d7` — the catalogue says *one source, one set, no conflicts*.
`49a3aa90` was reverted by `cd2a0b25`: two of its three guards had no
observed failure behind them, and the user's design argument against the
third was right.

**Task setup is a function of the folder** — `web/task-setup.md` § 2.1,
written long before it was kept. Four groups, and the measure of progress
was `_resetPerFolderState()` shrinking to nothing:

| | | reset list |
|---|---|---|
| `d2daa765` | the folder door, and the answer NAMES the folder | 8 |
| `97a25580` | `loadFolder` takes one answer, not four files | 8 |
| `4217f80e` | attempts and tokens come from it | 6 |
| `2f993971` | the hand-over branch paints every card the description one does | 6 |
| `bfb38c6e` | the buffers are one object, replaced whole | **0** |

`f6d9011d` — the hand-over writes its files and stops; the tab-to-tab jump
is deleted rather than fixed (user: *"skip the fancy tab to tab jump to
avoid implicit coupling"*).

### 10.2 The rule this session actually established

**An answer names its subject, and the reader checks.** `calcdir.json`'s
`of` on disk, `run_dir` / `dir` on the wire, `said.dir !== _dir` in the
page. It is the half that clearing cannot cover: there are two ways a
surface shows the wrong folder — state that LINGERS and an answer that
LANDS LATE — and Task setup had no `AbortController` among its twelve
calls, so the second was unguarded entirely.

### 10.2a The second day — the Results panel owns its folder *(2026-09-20)*

The workflow test's W1/W2/W3 are closed and the note is archived
([`archive/2026-09-19-workflow-test-findings.md`](?doc=archive/2026-09-19-workflow-test-findings.md)).
What followed was a user ruling on the tab's interaction model and a review
of the work that implemented it.

**The panel owns a directory.** Browsing the sidebar no longer moves it;
**Reload from current project dir** is the one gesture that re-points it; tab
re-entry re-reads the folder already bound; the dropdown alone decides what is
shown. `results.md` § 2.1 is the contract.

This had been decided twice before in opposite directions — retired 2026-06-09
(#301, single clicks hijacked the inspector mid-read), restored 2026-08-04
(a live BDT-Au111 job rendered as a finished run from another folder, *"every
number plausible and every number wrong"*). **The third answer works because
the header now names the bound folder and says when the sidebar has left it.**
Read the 2026-08-04 note again and the fault is its last clause — *with
nothing on screen saying so*. Following was one way to keep the panel honest;
naming the folder is the other, and it survives you scrolling around. **If
that readout goes away the subscription has to come back**, and the CSS, the
template, the contract and a test all say so.

**Three viewers were unreachable, all the same shape:** a reader that exists,
a presenter that claims the format, and no row in the parse registry —
`.transport.json` (2026-09-17), a sweep's `job-set.json`, and `.pdb`. Each was
found by asking the door what it answers for a REAL directory, never by
reading the presenter. `.pdb` then had to go through `StructureCodec`: a
`.pdb` here is a person's structure, so reading it with `from_pdb` dropped the
regions and frozen atoms its `.molstruct.json` carries.

**Six defects in the same session's own work**, found by read-only review and
each re-measured before acting: the divergence header composing a folder with
a file from a different one (the 2026-08-04 defect wearing its own
mitigation); a dropdown pick writing the panel's folder back to the sidebar
and silencing the warning; Reload taking its FILE from the sidebar; Reload no
longer re-reading for viewers that do not poll; a warning colour token that
does not exist; and a `.pdb` sniff window that refused ten real RCSB entries.

### 10.2b What this arc cost, and the two rules that came out of it

**Doctrine is not the contract.** Three failing tests were treated as
authorities and the code bent to satisfy them; checking each against the
document it cites changed the answer every time, and one cited a plan section
deleted three days earlier. Code pins to the design; a lint, ledger or
exemption that conflicts with it is the thing to fix.

**History does not belong in the file.** A comment describing what the code
used to do rots by construction, and a rotted comment is worse than none
because someone believes it — which is what `results.md` § 2.3 did on
2026-09-19, telling a reader to build a route that had shipped the same day.
One false claim was found in three separate homes. The fix is deletion; the
commit message is where history goes, being immutable and already dated.

### 10.3 What is OPEN

* **§ 9.2 — ATOM** as an optional pseudopotential validator, on the 3DNA
  model. Planned, not started; the open question is that ATOM writes PSML
  and cannot read one.
* **The `generator_mismatch` severity.** `projects/pseudopotential` mixes
  two ONCVPSP releases, and all ten files from the 4.0.1 batch are flagged
  by the physics checks (four block as `semilocal_only`, six warn as
  `partial_projectors`) while all sixty-two from 3.3.0 are clean. The
  version is a perfect predictor here, which is the evidence for making
  C4 an ERROR — but only after the library is unified, or every calculation
  touching those ten elements refuses.
* **`projects/BDT/`'s results.** Every SIESTA run under it used the
  defective v0.5 sulfur, in a thiol-gold junction where S is the binding
  atom. `Au-BDT-Au/`, `BDT-Au/`, `gasrun2` and `gasrun4` used the good one.
* ~~**The other tabs** — the Results picker never checked that the answer was
  about the folder it asked for.~~ **CLOSED 2026-09-20.** The picker now drops
  a reply whose scan is no longer the bound one (`if (boundDir !== dir) return`),
  which is the rule the rest of the session established and the one surface
  still missing it — on the tab the rule was learned on. Compared against OUR
  OWN request rather than `body.run_dir`, because the server resolves symlinks
  and a string compare there would refuse every correct answer under a
  symlinked root. Closed with it: `_metaFor` handed presenters three of the
  five fields `presenters.md` § 2 documents, so a `match()` reading
  `meta.label` or `meta.stage` got `undefined` — it returns the whole record
  now. § 2.1's rule still applies to any other page that follows the sidebar.
* **`detect()` cannot explain a refusal** *(2026-09-20)*. It fans a boolean
  `can_parse` over every registered parser, so a damaged file is
  indistinguishable from an absent one: a sweep plan truncated by a killed
  write lists as *"no result files yet"*. The parsers now say why when asked
  directly; carrying a reason through the fan-out is a registry change.
* **`task-setup` acts on `onChange`** (`viewer.js:3665-3668` subscribes both
  `onChange` and `onCommit` to the same `loadFolder`). The Results tab stopped
  doing this on 2026-09-20; this tab still does.


---

## 11. The test-audit consolidation — one list, re-derived 2026-09-20

> **Verdict (2026-10-08 validation):** P1–P5, P9, P13 confirmed; P8 and P10 are BUILT though marked open (`tests/test_structure_save_endpoint.py:65-72`; `tests/test_transport_au_bdt_au_validation.py:330-345`); P14 and § 11.4a obsolete (`/api/backends` route gone).


**This section replaces two records.** `archive/2026-09-08-test-audit-findings.md` and
`archive/2026-09-08-test-design-findings.md` were the 2026-09-08/09 audit's two halves.
Both are archived; everything still live is here, and nothing below is
inherited on trust — each row was re-measured against the tree, and the rows
where the audits were **wrong** are marked, because that is the part a reader
would otherwise re-derive.

**How to read the ranking.** By what a wrong answer costs the SCIENCE, not by
effort. The first three produce a run that converges on the wrong thing.

### 11.1 A wrong molecule reaches the engines — DECISION NEEDED

| | |
|---|---|
| **P1** | **DECIDED AND DONE 2026-09-20.** `build_peptide` returned a peptide ALDEHYDE and said nothing. Measured: `build_peptide("ARNDC")` → `C20 H35 N9 O8 S`, 38 heavy; the C-terminal carbonyl carbon's neighbours are **H (1.032 Å)**, O (1.229 Å), C (1.52 Å) — `-C(=O)H`, not `-C(=O)OH`. A free acid is `C20 H35 N9 O9 S`, 39 heavy. One oxygen short: the OXT, which the structure kit never writes and which occurred **zero times in the repository**. Silent because the H count is 35 either way — the aldehyde H stands in for the hydroxyl H — so every atom-count check is blind by construction. **THE RULING: molbuilder does not add the oxygen.** Adding hydrogens is a deliberate exception and stays — the geometry is predictable and two independent kits cross-check each other — but placing an OXT is a structural guess the tool has no business making for the person; *"we just need the student to understand what's the limit"*. So `build_peptide` now warns, naming the compound it produced and what that costs a calculation. It **detects** rather than assumes (`"OXT" in atom_names`), so it stops the day the kit writes one. The warning rides the channel the web page, CLI and console already display. `test_the_c_terminus_limit_is_stated` pins it and is the only check in that file that can see the defect at all |

### 11.2 Silent identity corruption — the top of the old list

| | |
|---|---|
| **P2** | **CLOSED 2026-09-20 — re-measured, and it was never a gap.** The complaint was that `test_delete_preserves_metadata_in_lockstep` asserts four `len(col) == n_atoms` and nothing about contents. True, and harmless: **the risk it implies is covered by siblings.** (a) *Per-atom columns ride the same slice* — `elements`, `positions`, `atom_names`, `residue_ids`, `residue_names`, `chain_ids` are each `[struct.X[i] for i in keep]`, so a label travels with its atom by construction (user, 2026-09-20: *"internally structure label goes with per atom"*). (b) `elements` and both survivor coordinates ARE content-asserted by `test_delete_drops_listed_indices` (`test_modify.py:212`); an `elements` mutant measurably dies there. (c) The one part that is genuine arithmetic rather than a slice — remapping `regions` and `frozen_atoms`, where indices shift and entries pointing at deleted atoms must be dropped — is asserted in `tests/test_reserved_label_one_store.py`: `delete_atoms(s, [0])`, then `regions["keep"] == [2]` and `frozen_atoms == [0, 2]`. `_reindex_transport_metadata`'s docstring records why that test exists: *"a remap that forgot it silently constrained the wrong atoms."* **This row is the FOURTH instance of the error § 11.6 records** — a suite-wide absence asserted from one file's contents. It was ranked first and was not a defect at all |
| **P3** | **CLOSED 2026-09-20 — the second half was a test of `mv`.** The finding asked the file-move tests to read `regions` and `frozen_atoms` back out of a moved sidecar. User: *"the so-called moving file problem is just hallucinating — why would you not trust the file system"*. Right: a move does not alter bytes, so asserting they survived is a test of the operating system, not of molbuilder. **What CAN break is the pairing**, because a sidecar's name is derived from the structure's stem (`files.py:246`: renaming to `bridge.xyz` *"orphans `water.molstruct.json`"*) — and that IS asserted: the rename test checks the old pair is gone and `picker_root / "bridge.molstruct.json"` exists, i.e. the sidecar followed to its derived name. The remaining half — `_seed_paired` hand-writing a v7-stamped payload carrying the top-level `frozen_atoms` key that v7 removed, which `molstruct.load_text` genuinely refuses — is real but inert: the tests using it only move bytes, and the read-back that would have exposed it is the thing that should not be added. Fixture hygiene at most. **FIFTH instance of § 11.6's error** |

### 11.3 A gate that does not run, and formulas tested where they cannot fail

| | |
|---|---|
| **P4** | **CLOSED 2026-09-20 — the premise was false, and it was mine.** The row said `test_pyscf_smoke.py` is *"the only test that executes a generated script"* and that *"a rendered deck that no longer runs ships silently"*. **Neither is true.** Measured across the suite: 133 test files touch the PySCF generator, and **four execute a real generated deck in the PySCF environment, all through the production door** — `test_spectra_from_a_real_run_e2e` (`prepare_deck` + `conda run -n <env> python <deck>`, asserting `returncode == 0`), `test_trajectory_from_a_real_run_e2e` (the OPTIMIZATION path, same pattern, line 124), `test_vibration_e2e` (whole bundles), and `test_molwatch_preview` (via `run_in_env`). SIESTA is covered too (`test_siesta_keyword_smoke`, and a keywords-exist-in-the-binary check). **What is actually left is one dead file**: `test_pyscf_smoke.py` collects zero tests anywhere — no pyscf in `molbuilder`, no pytest in `molbuilder-pySCF` — and the work it was written for is done four times over by neighbours that use `conda run` correctly. Retire it or point it at the same door; either way it is housekeeping, not a gap. *(User: "all the script generators have their PySCF component as well — I really don't know where you get those claims." The claim came from the inherited audit's § 1, which said only that THIS FILE never runs, and which I generalised into a suite-wide absence. Sixth instance of § 11.6's error.)* |
| **P5** | **WITHDRAWN 2026-09-20 — trivial, and the row was the pattern not the problem.** The claim: `bulk_z_period`'s period is asserted only on uniform layers, where a wrong formula coincides, while the one non-uniform fixture discards it (`_zper, d, _n = ...`). The mechanism is true and it does not matter. **The period is `z_span + d` — one line, no branch, no edge case** — so once `d` is right the period is right by arithmetic, and `test_bulk_z_period_uses_the_median_not_the_mean` already pins the only decision in the function: median, not mean, so one relaxed surface layer cannot drag the spacing. Asserting the period would pin a typo. *(I then proposed a "better" test on the physical invariant — after tiling, the seam equals one interlayer spacing. That is worse: seam = `(min + span + d) − max` ≡ `d` identically, for any input. A tautology of the implementation, offered as physics.)* **The production chain is correct and was never in doubt**: the lead's third lattice vector is `span + d`, so the next image's first layer lands one spacing above the top — setting it to the bare span would overlap the surfaces, which is what the `+ d` prevents (user raised exactly this; measured seam +2.50 Å on the compose fixture). Nothing owed. Filed because a value was computed and not asserted, without asking whether the value could be wrong |
| **P6** | **A-DNA and B-DNA cannot be told apart if they are SWAPPED.** `test_backends.py:307` asserts `diff.max() > 0.1` Å. *The audit's stated failure mode is measurably false* — two independent B builds differ by **0.000000** (fiber is deterministic) and would fail, and A vs B moves even the least-moved atom by 3.168 Å. But both directions of a swap give ≈7.9 and pass, and a swap is exactly the plumbing bug the test exists to catch. **Fix, derived and measured:** assert rise per base pair from the C1′ z-centroids — A gives 2.548 Å/step, B gives 3.375, against Arnott canonical 2.56 / 3.38. External anchor, kills the swap, subsumes the current assertion |

### 11.4 Real, cheaper, no scientific cost

| | |
|---|---|
| **P7** | `test_not_is_complement` has a **live mutant**: `Not` complementing within the operand's own span survives all 56 tests in the file. *The audit blamed the relational assertion form; measured, that is not the cause* — its premised mutant IS caught, by `test_minus:340`. The cause is the fixture: `ByElement(Au)` reaches the last atom, so `max(operand)+1 == n` and the two readings coincide. **Fix: one line** — use an operand that stops short of the last atom |
| **P8** | `test_structure_save_endpoint.py`'s docstring promises the written pair "reads back through the load door with metadata preserved". Seven tests, **none imports `StructureCodec`**; `_browser_blob()`'s `frozen_atoms` and `cell_origin` are never read back. That is the regression the file exists for (task #75, *"every save produced a pair the app could no longer open"*) |
| **P9** | **DONE 2026-09-20.** A backend the person did not install answered **HTTP 500** — reproduced with 3DNA stubbed absent: `{"backend":"threedna"}` and `{"input":"ds,..."}` both 500, via the route's generic `except Exception`. `BackendUnavailable` existed as a dedicated type and **nothing in the repo caught it**. Now the contract's own ADVISORY bucket — `web-api.md` § 1 reserves 5xx for *"an I/O error, an engine that fell over, a bug"*, and a tool you chose not to install is none of those — so **200 with `ok: false`**, plus `reason: "backend_unavailable"` and `backend`, machine-readable so a page can disable what the box cannot do. `nucleic.py`'s duplex gate raised a bare `ValueError` for the same condition and now raises `BackendUnavailable` too: one condition, one type, one handler. The message is unchanged — `design.md` requires it to name the preconditions tried, the download URL, the licence terms and the fallbacks, and it does. Three tests, mutation-checked (removing the handler fails both shapes with *"reserves 5xx for a fault on our side"*); 106 pass across backends, status contract and rate limit. **Still open, the other half:** `/api/backends` exists and no page reads it, so Modify still offers all four |
| **P10** | The TBT window's `from`/`to` **numbers** are asserted nowhere. *The audit filed this against a dead line* — it quotes `TS.TBT.NumE`/`Emin`, keywords removed 2026-09-15 because the installed tbtrans 5.4.2 cannot read them. **And do NOT attempt its proposed fix:** whether `%block TBT.Contour`'s bounds are absolute or E_F-relative is a recorded OPEN QUESTION in four places (`transport/stages.py:94`, `config/transport.py:340`, `record.py:70`), so asserting a sign relation with E_F would assert a rule no document states. Bracketing is already structurally forced by the `range:` on the two fields. **Fix:** add `from`/`to` against `cfg` to `test_transport_au_bdt_au_validation.py`, which already parses the block properly |
| **P11** | `spectra/test_selection.py:191` promises two cases and tests one; the two take **different branches** at `selection.py:119`, and no fixture in the tree can produce the second (`_prior_with_es_on_mode` always populates it). One extra assert |
| **P12** | The no-background-poll property is guarded by **nothing**. *Both audits are wrong about why*: `test_checkpoint_js_has_no_polling_timer` does not exist — deleted 2026-09-10 in `082ba979`, one day AFTER the doc's "re-verified 2026-09-09: still in the tree". The property holds (`checkpoint.js`'s only `setInterval` is prose at line 33) and `test_checkpoint_sensor_js.py:21` still advertises it as guarantee 3 over an empty region. **Fix:** stub `setInterval` in the existing `_DOM` harness and assert no timer after `onDirectoryChange`. The blocker that deferred this is gone — node is installed and 538 JS tests now pass where 67 of 717 used to |
| ~~**P13**~~ | **CLOSED by § 11.5a's ruling — won't do** *(noted 2026-09-29)*. `_threedna.py:126-134` decides availability from `isfile + X_OK + isdir` and never runs the tool. A design decision about how eagerly to probe at import |
| **P14** | Two small ones: `TestOpsPreservePeriodicity`'s class docstring still says the ops must preserve "k-grid", which moved to `SiestaConfig`, contradicting `_assert_lattice_preserved`'s own docstring five lines away; and `test_the_composed_sweep_survives_json` cannot test what it names, because both producers are stubbed — *though the audit's "every assertion cannot fail" is too strong: the 200 and the `ok` stamp are real* |

### 11.5 Closed by re-measurement — do not re-derive these

* **`test_add_atom_zero_offset_is_advisory_not_blocked`** — NOT REAL. `tests/validation/test_geometry.py` pins all three distance bands directly, including water at 0.957 Å producing no issue, which kills the mutant the finding posits.
* **`test_isolated_axis_keeps_pbc_false_despite_cell`** — NOT REAL. Wrong layer (the test is about the codec deriving `pbc` from `axis_kind`); the physics IS checked, by `validation/geometry.py:153` `cell.image_distance`; and the fixture's nearest image is 9.26 Å, above molbuilder's own 6.0 Å criterion, so there is nothing to catch.
* **`TestOpsPreservePeriodicity`'s rotation claim** — NOT REAL. A rotation returning `cell=None` IS caught, by `TestRigidTransformMovesTheBox` in the same file (4 failures). The audit names that class and files the gap anyway.
* **`detect_layers`' `abs=1e-3`** and **`bulk_z_period`'s default approx** — the filed complaints are unfounded (both are effectively exact). What is real is P5, and separately that `LAYER_TOL_ANG = 0.5` — the decision of what counts as one layer — has no test at all.
* **The junction `2.355`** — the stated failure mode is false: the Au planes are placed by absolute `start_z` at -6.855/-4.500/+4.500/+6.855 and cannot overlap. Only the retyped constant is real, and it is molbuilder's own `fcc_lattice.json` value, so deriving it is legitimate.

### 11.4a Decisions recorded, to implement together

Taken during the one-at-a-time review so the work can be batched rather
than paid for in separate test cycles.

* **The Modify tab asks what this machine can do** *(decided 2026-09-20,
  not yet built)*. `/api/backends` exists, reports which builders resolve,
  and **no page reads it** — its own handler comment says so. Meanwhile
  `modify.html:145-197` offers all four backends and the duplex input
  unconditionally, with static prose *"(Requires 3DNA…)"*. The server half
  is now in place: a missing backend answers `200 / ok:false` with
  `reason: "backend_unavailable"` and the backend name, so the page has
  something to branch on. **The decision:** fetch `/api/backends` on Modify
  mount, disable the options that do not resolve, and hide the two-strand
  hint when no duplex-capable backend is present. Not "grey it out and hope"
  — the refusal path stays as the backstop, because a backend can vanish
  between page load and submit.

### 11.5a Rulings that close items — do not reopen without asking

* **Hydrogens are added on purpose, and that stays.** `chemistry.add_hydrogens`
  saturating a dangling valence is a deliberate design decision, not drift: the
  geometry is predictable and molbuilder runs two independent kits that
  cross-check each other's result. It is the one place the tool completes the
  structure kit's work. Adding the C-terminal **oxygen** is explicitly NOT the
  same call — harder to predict and harder to justify — which is why P1 warns
  instead of fixing. *(I had filed the silent capping as a defect of the same
  class as the peptide; it is not. Recorded so the next reader does not refile
  it.)*
* **3DNA is not probed by running it — `#80` is CLOSED, won't do.** Availability
  stays a file check. The person installs 3DNA themselves under its licence and
  is responsible for it; going a mile to verify someone else's installation is
  not this tool's job. **What must improve instead is the failure**: when it does
  not work, say exactly what failed.
* **What replaces both:** an error surface that says exactly what failed — not yet written; it is § 0a's *Unscheduled*.

### 11.6 The pattern both audits kept repeating

`test-design-findings.md` § 4's preamble names its own recurring error —
*"a suite-wide absence asserted from one file's contents"* — and records seven
withdrawals for it. **Two more of exactly that were still in its open list**
(the two NOT REALs above), and the fresh-eyes pass found two further rows whose
stated failure mode was measurably false. Of the still-open science section,
roughly 4 in 7 rows were wrong in some load-bearing way while the underlying
concern was often real. The appendix's own warning — re-derive before acting —
held up completely, and is why every row above carries its measurement.


---

## 1. The 2026-09-01 fact-check — ARCHIVED

> *(a pointer section; its body follows as it stood)*


*Nine plan documents read against the code; three headers were flatly false. The detail is history now and lives in [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md). The one live fact it found — bench § 2.2's `parse_util_bound` reads the VERDICT and not the numbers — became row E2, done 2026-09-03 and archived.*

---


---

## 3. Configuration and ops — merged

> *(a pointer section; its body follows as it stood)*


*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---


---

## 4. The front end — merged

> *(a pointer section; its body follows as it stood)*


*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---


---

## 4a. Needs a decision from you — merged

> *(a pointer section; its body follows as it stood)*


*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---


---

## 5. Documentation drift — merged

> *(a pointer section; its body follows as it stood)*


*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---


---

## 5d / 5i / 5j — CLOSED, archived 2026-09-07

> *(a pointer section; its body follows as it stood)*


*In [`archive/2026-09-07-plan-closed-sections.md`](?doc=archive/2026-09-07-plan-closed-sections.md); the 2026-09-10 consolidation moved this pointer's body to [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md).*

---


---

## 5k — CLOSED 2026-09-08

> *(a pointer section; its body follows as it stood)*


*The paths framework's first program, M1–M8. **Its rule is the part that survived**: [`execution/project-layout.md` § 4.5](?doc=execution/project-layout.md), enforced by `tools/classify_path_finders.py --check` and `tests/test_path_framework.py`. The follow-on STANDARD that was to replace the API (§ 5l) is **retired 2026-09-17**; this section's rule and guards are untouched by that. Body archived 2026-09-10.*

---


---

## 5l — RETIRED 2026-09-17 *(user ruling)*

> *(a pointer section; its body follows as it stood)*


**The paths STANDARD — one address, three verbs — is retired**, and
`molbuilder/ref.py` + `tests/test_ref_address.py` are deleted with it. § 5k is
the paths framework and stands; this was a standard proposed on top of it.

The full record — what it was, the four APIs it meant to collapse, and the
lesson the user drew (*a module with no caller is not a green step, it is an
unmerged branch living in `main`*) — is
[`archive/2026-09-17-paths-standard-retired.md`](?doc=archive/2026-09-17-paths-standard-retired.md).


---

## 6. Closed by consolidation — archived

> *(a pointer section; its body follows as it stood)*


*The provenance map of the nine archived plan documents and where each one's open items went. Several of the rows it points at have since been killed as untrue, so it is history: archived 2026-09-10.*

---


---


# Sections trimmed on 2026-10-08 — the parts moved out

## 0a. THE WORK ORDER — milestones, each closed by a full code-text review *(2026-09-27)* (archived 2026-10-08)

*(The blocks below are the text removed from `docs/plans/plan.md` § 0a on 2026-10-08, verbatim, each under its verdict from the validation of that day. Table headers are repeated so the rows render.)*

### THE QUEUE — done rows

> **Verdict (2026-10-08 validation):** queue rows Q1, Q2, Q2c, Q2e, Q2h, Q3 CONFIRMED done — Q1 `scheduler/record.py:264-279`, `envs/initconfig.py:461`; Q2e `codemirror-load.js`, `ui.js:1940,2129,2192-2197`, `sidecar.py:272-276`, `molview.md:1579`; Q2h `identity.py:277`, `commands.py:118`, `runtime_info.py:96`, `runfiles.py:1386/1470`, `runrecord.py:218-259,373`, `_cli.py:489,504`, `task.py:979-989`, `tests/data/the_deck.toml`, `tests/field/`.

| # | item | state | next |
|---|---|---|---|
| **Q1** | **W53's follow-up — `env_init` and one config file** *(user, 2026-10-02)*: this machine's activation and preamble in `molbuilder.json` (`env_init`, asked by `envs init-config`), copied by `jobset probe --write` into every record it writes, the record field `env_init`; the per-calculation `.molbuilder.json` removed — code, contract, tests | **done `05c6bb99`** — 99 targeted files (3870 tests) and 20 browser files (167) green; 10 mutations, 9 red (the 10th found prep's own config read redundant, removed); your config files moved to `env_init` (backed up); the dev server restarted | — |
| **Q2** | **W54 — the config review** *(user: "do a agent scan focusing on config file including molbuilder.json, environment, secret etc")*: four reviewers in parallel — `molbuilder.json` and secrets code (C), the record and probe code (R), the documents (D), the tests (T) | **Q2a done `92e84cc8`; Q2b part 1 done `7dd8940f`, part 2 `0a7ea0e4`, part 3's first half `6401abfb`** — every finding's state is **§ 0b's ledger**, each re-read in the code before it is acted on *(user, 2026-10-02: "make sure ... all the review results is validated and fix of those are also considered - the review was done in parallel i don't want you to miss those")* | after Q2c: the ledger's open rows, then T15 as its own unit |
| **Q2c** | **The ten decisions of 2026-10-02** — W54's open words and step 3's D2–D4, answered one by one *(user: "go ahead")* | **done 2026-10-02** — items 1–3, 5–8 and 10a built (`e1d88b22` … `31ca9957`), 4 and 9 settled, 10b done | § 0b |
| **Q2e** | **MolView's Metadata pane — read through, and edited** *(user, 2026-10-03: "the meta data tab in the molview does not correctly scroll when meta data is a lot to be fully shown in that panel. we need a good way to go through them, - and we need to allow user to edit it too")* | **the scroll fixed `a7ca51af`** (the page took the Cell page's rule; measured on the AuBDTAu relaxation's 5749 px store). Navigation and editing reverse `web/molview.md` § 8.4a's *display, never a mutator* (user, 2026-08-29), so they are designed first: each key a collapsible section with a one-line summary, long lists collapsed to their count, a filter, the raw JSON a toggle; editing where the structure is saved, through the `data.info` door | **done** *(user: "a: yes, b: yes, c: before is fine, but it can also be done after"; "we have codeview integration ... so don't reinvent the wheels")* — the page is a tree under a pinned bar: each entry a fold with a one-line summary, a field a row, a nested block a fold, a long list folded to its count and opened as one wrapped line, a long string cut with its whole text on hover; a filter, expand / collapse all, and the raw JSON in the app's CodeMirror (`lib/codemirror-load.js`, read-only). On a viewer that saves its structure a field is changed in its row, an entry added, removed (asked in place) or edited as JSON in the same CodeMirror — refused with the error when it does not parse; a read-only viewer reads only and says where to edit. The person's doors `info.edit` / `info.drop` are gated and recorded (a point each, undone by Retract); the host's `set` stays silent; a block naming its `source` is stamped `edited_by_hand`, and the relaxation check says so first. `web/molview.md` § 8.4a / § 9.3, `model/parse.md` § 5b.1, `engines/vibration.md` § 2.2. A model test (the doors), a browser test (the page, on the demo), the record table's matching row extended; six mutations, each red; 375 targeted green; seen on the dev server — the AuBDTAu record 678 px instead of 5749, and an edit typed in the demo |
| **Q2h** | **W57 — the explicitness review** *(user, 2026-10-06: "review your fucking code again holistically ... overly simplify and being implicit rather than explicit")* | **built** -- `7d52b551`, `36c545c5`, `a1a74166`, `43b43fe7`, `58634eaa`, `bf2f68e1`, `422f365c`, `ec7823d9`, `5637cdb3`, `3b80b52c`, `adeb6c6d`; every decision answered | § 0c, W57 -- unit 12 |
| **Q3** | **M5 step 3 — `TransportConfig` retires** (TD4) | **done `325cb1d4`** — the class and its module gone, the region labels in `transport/sort.py`; `config_for`, the sealed sets and the dataclass form builder deleted, and with the builder the field keys nothing read once it was gone (`section`, `id_suffix`, `step`; `skip_cli` and `optional` unread before — 136 declarations, `web/form-schema.md` § 1a); compose writes the permutation through `write_permutation`, its key stamped; `spec_for`'s transport arm refuses the render arguments it does not read; the kind-narrowing found built by K4 (2026-09-30) and § 8 row 4 fixed since `6822f639`. Targeted files green, 4 mutations red; the documents swept (`engines/transport.md` § 3.6–§ 3.8, § 4; `web/form-schema.md` § 1a; `template.md` § 2.1a; `structure-annotations.md` § 5). **Reviewed** by a fresh agent, full code text: every done-when clause met but D2; its residue and false documents fixed (a dead coercion branch, compose's snapshot lines, four module headers, the blueprint's route list, § 1a's readers) | — (D2, D3 and D4 decided 2026-10-02: Q2c's items 1–3) |

### THE QUEUE — rows rewritten in the trim, as they were

> **Verdict (2026-10-08 validation):** rows rewritten in the trimmed § 0a — their text as it was. Q2d PARTLY stale (9a–11 done; 12a, 12b-1 done; 12b-2–12e open); Q2g PARTLY stale (W56 units 1–4 done, D29's list open); Q4 PARTLY → done (M2f → 9b, M2g → 9b 3, M2i → 9b 1 + 12b-1, M2k → 11a, step 5 built `0e11a018`); Q5 PARTLY stale (5, 6 built; the sweep `ddf12a43`; `launch --cold` `_cli.py:1708`; `--skip` / `--unskip` 0 hits; 7, 10, K21 open); Q6 PARTLY (step 9 building; 8, 11 open); Q10 done cell stale (`jobset/__init__.py` re-exports nothing; `planned.py:102` `Plan`); Q11 PARTLY (one open: the named-queue check, `placement.py:28` `launch_refusal` + `scheduler/place.py`); Q13 CONFIRMED built, the ready door is `ready.py:60` `readiness`, not `continuation.ready`; Q14 REFUTED stale → B0–B4 committed (`d398ae86`, `7351a3ff`, `ddf12a43`), B5–B9 open; Q12 OPEN, its list cut to the rows still open.

| # | item | state | next |
|---|---|---|---|
| **Q2d** | **W55 — the prep and jobset review** *(user, 2026-10-02: "use agent to read the full code and contract text of prep and other jobset, to make sure a logical workflow is designed with no redundancy, duplicated work, no handcraft, but with a good organized data structure, unified api design and good connection between operations")*: four reviewers in parallel — prep; launch, attempts and status; the two doors and the contract; data structures and APIs | **reviewed 2026-10-03; every finding re-read in the code** — § 0c's ledger: the defects fixed first, the structure proposed for the user's word; units 1–8 done (`34febeaa` the last) | **§ 0c's units 9–12 next, ahead of Q4–Q6** *(user, 2026-10-03: "agree, do framework units 9-12 first")* — designed into the contracts first, the four together, then built one unit a commit; what they take in is below |
| **Q2g** | **The file manifest** *(user, 2026-10-04: "make a manifest of all the output file and files your code generates in the hierarchical and flat directory ... integrated in a clear section in contract so that you ... stop keep re-inventing new ways to ... interpret the ... data"; "use another agent after this report is genrated, to review and validate, and also check overlap/redundancy/duplicated components")* | **the manifest written 2026-10-04** -- `project-layout.md` § 5: every file of both shapes with its writer and its one door, built by three agents from the code and two real H₂ calculations, checked by a fourth, every finding re-read in the code; the documents' other file lists point to it (D29). Found: D20–D29, B11–B14, Q6 (§ 0c); D19 rides with B12 | **ruled 2026-10-04** -- B11–B14 approved, Q6 (a), D19 with B12, the Results file card and the generated manifest; designed into the contracts the same day; **W56: units 1, 2 and 3a done, 3b in steps (3b.1-3b.3 done, 3b.4 next), then unit 4** (§ 0c), ahead of 9b part 5; the same day the env registry was reviewed whole and the host env's list became one file (D30) |
| **Q4** | **TD9 — M5 step 5 against M2** | **decided 2026-10-02** *(user: "#9 ok")*: M2f, M2g, M2i, M2k, then step 5 — **taken into Q2d's units 9–12 (2026-10-03)**: M2f, M2g and M2i's stage-number door → unit 9; M2k and step 5 → unit 11; M2i's disabled folder → unit 12 | with Q2d |
| **Q5** | **M5 steps 6, 7 and 10, and K21** — the transport work that does not wait on TD9 | after Q3 | **after Q2d** — built on unit 10's prep framework; each step's *done when*; K21 is § 5w's; **carried from unit 10's second review (2026-10-05)**: `status` asks the gather's gates (it says *prep it* where prep refuses -- an upstream rung running, a template changed since the leads ran), and the gather renders each upstream deck once, not once per bias point, with its warnings in the answer; **and the bias sweep as one run** (`engines/transport.md` § 2a.11, designed 2026-10-05 with the user): the run's record (`points.json`: each point done, skipped or neither, what it started from, how it ended), a point not done continued from its own last density in place, `launch --skip` / `--unskip` and `launch --cold`, the transmission taking one device run whole, prepped once it is complete -- with 11e's grouping, C8's hand-over fact and C1's road row, on the minimal junction -- and with it goes 11c's carry of a rung's gather into its next attempt, which copies a single-bias device's `.DM` twice and records the device's own density as the seed's (unit 11's first review) |
| **Q6** | **M5 step 8** (one panel per engine, after step 3), **step 9** (after M2j and M2l), **step 11** (after steps 4 and 5) | after Q5 | step 9 → Q2d's unit 12; steps 8 and 11 after Q5 — § 5u.1 |
| **Q10** | **W52's open**: (11) prep's transactional produce; (8) the conductor imported below the surfaces | open | taken into Q2d: (8) → unit 9 (B7), (11) → unit 10 (B1) |
| **Q11** | **W53's parked**: the next-step lines offering `--mode submit` for a target with no queue; the named-queue check in two doors (`launch_refusal`, `place`); `gpu_partition` and `~/.config/molbuilder.backup/` decided 2026-10-02 (§ 0b, item 10) | open | W53's row |
| **Q13** | **THE TASK — `prep task` / `launch task`, one protocol** *(user, 2026-10-07/08: "prep the whole task and ask the user which stage it intend to do"; "one unified framework and protocol and verb design and api ... systematic and holistic"; D1–D5 agreed 2026-10-08)* | **built 2026-10-08** — T1–T8 committed and two review rounds (code and docs, fresh agents) fixed (`214cc19d`, `91629b67`). **Left**: Task setup's Prep card seen in a real browser (the extension was down); `tests/test_task_setup_prep_e2e.py` drives the per-tab Prep buttons that are gone (e2e on the user's word); a group's header under a real scheduler (Sol); `prep_group` re-plans each member after the save (no refusal found reachable on the road). Contract written 2026-10-08 (`job-system.md` *The task*, § 5.0, § 5.3; `project-layout.md` § 1.6.6; `architecture.md` § 3.2); the review (three agents on conflicts, two on the doc sweep and the code inventory) found ~335 doc passages and ~30 code sites; its open points settled by the agreed rules: a kind with stage roles pre-selects its ready stages that build on nothing (transport's seed and leads); stages named together at prep OR launch go as one job when they pass the group checks (none builds on another, one allocation) -- no "never alone"; a stage not prepared reads `ready` / `waiting`; the ledger line `prepared`; `summarize task` takes no stage. **Milestones, each checked on the road:** T1 the low-bias gather fix and step 6's core (the treatment in `task.json`, the device without an axis); T2 the vocabulary -- kind `task`, `--stage`, `summarize task`, the `prepared` door, every printed command, the road and its rows; T3 the `ready` door (`continuation.ready`, the gather as an answer or a refusal, D5 newest-only); T4 the `prep task` entry -- the ladder offer, `ask.choose`, the no-terminal refusal, groups by pick, `prep_group` held to rule 3; T5 `launch task` -- the waiting units, the question, relaunch by name, the group checks at launch; T6 status -- ready / waiting rows, next lines, the wire form, the chips; T7 the web -- Task setup's one Prep with the ready stages, routes, handover text; T8 the doc sweep (~335 passages, the review's list); then the two review rounds |
| **Q14** | **Transport, made whole** — the sweep as one run, the low-bias switch, composition, the record, then the milestone review of backend, CLI, UI, checkpoint, stage/run configs and CSS (§ 5x) *(user, 2026-10-08: "You need a clear picture, a contract, how transport is done ... see where it's missing and what is wrong"; "stop fucking hand bake these fucking things")* | **approved 2026-10-08** — B0–B1 committed; B2–B4 written, uncommitted, verified only on a hand-written junction (not evidence); the road junction being built on the page; B5–B9 open; the review (B8) waits for B2–B7 | § 5x |
| **Q12** | **the rest of the work order below**: M2e–M2n, M3, M6 P5, M8, M10 | open | the table below — M2e, M2h, M2j and M2l's doors are the same code as Q2d's units 10–12 and are designed with them; M2m, M2n, M3, M6 P5, M8, M10 after |

### The milestone table — done rows

> **Verdict (2026-10-08 validation):** milestone rows CONFIRMED done — M1 (`atom_metadata.engine_frame_for_run_dir` gone → `runs.Declared.frame`, `runs.py:287`); M2a (`tests/test_monitor_start_is_logged_e2e.py`; `runwrap.py:1481-1506`); M2b (the peptide-warning web path still open); M2b′; M2b″ (the verb renamed `summarize task`); M2b‴; M2b⁗; M2b⁵; M2c (`create_app(config={})` noted-open stays); M2d (details obsolete: `PrepError` is `jobset/errors.py:10`; `Answer.evidence` / `SaveOffer` gone); M2e; M2h (`runrecord.py:64/81`; `continuation.py:501`); M2i (`materialize.py:49/97`; `task.py:1072-1081`); M2l (`template.py:930`; `paths.py:201`); M7 done; M9 folded.

| # | milestone | items | done when | review | status |
|---|---|---|---|---|---|
| **M1** | W33 P3's last check | T3: export from Results → reload (§ 5q.4) | T3 passes, mutation-checked; W33's status line corrected — its dev-server check is done (2026-09-25) | **done 2026-09-27** -- read by an agent in full (verdict *yes with notes*), every finding re-read against the code: ① a second export route, the structure preview of SIESTA's own `<label>.xyz`, stated no frame, against § 6.0 -- **fixed at the framework**: the run's engine frame has ONE composer (`parse/dirs/atom_metadata.engine_frame_for_run_dir`, moved from `watch.py`), and the codec gives it to an engine's own file read where its run is recorded (`structure.md` § 2.4's invariant refined); ② the test pressed Export before MolView held its frames -- it waits for the viewer now (`support/results_export.py`); ③ a PySCF run's cell (the deck's record) was pinned by no test -- a case of its own; ④ the export offered " .log_frame6" and saved a hidden file -- the viewer names its install after the file it loaded (`web/molview.md`'s own rule); ⑤ this plan's stale rows -- fixed; ⑥ the reference now parses as the door does. Checked in a browser on the dev server on a project made fresh on both roads (`projects/claude-validate`: the web UI's New project, the CLI's `smiles`, `jobset init/prep/launch`) | **done** -- `45ff5089`, `164549ec` |
| **M2a** | the monitor says it started | W36 ① | the session log holds `starting` / `started`, and for a broken member file the error and no `started`; the test breaks a member, not the entry | **done 2026-09-28** -- read by an agent in full (verdict *yes with notes*), every finding re-read against the text: the start pair cited as run-reports § 2.3 lives in § 2.6 -- fixed in five places; the contract's reading rule had a false positive -- a run that ends inside the monitor's ~0.2 s load leaves `starting` with no `started` and NO error -- so the `ERROR` line is now the evidence of a failed load and the stopped-while-loading case is named (§ 2.6); the ending door prints the load error once per question -- job-contracts § 2.6 now has the row; the test pins the contract's line format, the `ERROR` line and the traceback (red with the traceback dropped); stale text in the launch block, `monitor.py`, `wrapper_log.py`, three test docstrings and run-reports § 2.5 / running-a-job's session-log row -- fixed; the test file renamed `test_monitor_start_is_logged_e2e.py`, since it holds both cases. **Then built, on the user's word** *(2026-09-28: "ask the bundle once")*: the wrapper asks the bundle once whether it loads (`mb_monitor.pyz loads`, remembered by `_mb_ending_able`), so a bundle that cannot load prints its error once and the wrapper says the ending cannot be read; a SIESTA road case holds it (red, 3 errors, against the per-question check). Noted, not acted on: an unreadable zip, or an exception escaping `main`, reports in Python's own words, and a failing `ending` then exits 1, which the wrapper treats as 2 | **done** -- `e31e56f7`, `f3ae6e6b`, `5da4f85b` |
| **M2b** | the small code fixes | W36 ② constants in the zip · ④ the ASE probe · ③ the GPU warning · ⑨ three silent fallbacks · ⑩ one `_mb_outfile` | each as W36 records it | **Built:** ② `constants.py` travels and the grammar imports it both ways (the bundle test goes red without it); ④ validation takes its classes from `config/`, and importing it loads neither `ase.io` nor SciPy; ③ the advisory, its call and its test are gone; ⑨ "add hydrogens" with no engine raises `BackendUnavailable` -- which found the Build route's handler crashing on an unbound `requested` for any kind but DNA/RNA, fixed -- one API-level case (a refusal the road cannot reach), red against the old warning; the PDB reader imports ASE's table with no fallback; the guard is gone; ⑩ `pyscf/input.emit_outfile_helper` is the one definition, `job-contracts.md` § 4.2 says so. **Done 2026-09-28** -- read by an agent in full (verdict *not as it stands, the gap small*), every finding re-read against the text: ① the refusal's type sat in `builders/` (L2) and `chemistry` (L1) imported it -- an upward import, written on the false reading "chemistry is L2"; the type moved down into `chemistry` and carries what is `missing`, every raiser and importer switched, no shim; ② that false reading had rewritten ⑨'s agreed wording -- restored, and the PDB reader's real reason recorded in W36 ⑨ as a deviation for the user; ③ the CLI met the refusal as a traceback, with advice ("turn adding them off") no surface offers -- the CLI answers it `Error: ...`, the advice is gone; ④ the route's `backend` named the request, not what is missing (a peptide said `auto`) -- it reports `missing`; ⑤ the refusal was in the code, not in the document that owns `add_hydrogens` -- `model/chemistry.md` states it, and `chemistry-correctness.md`, `peptide.py`, `engines/builders.md` and the backends' docstring say it; ⑥ W36's status line was stale -- fixed; ⑦ the grammar's header said it imports "nothing of ours", and the companions list, `constants.py` and run-reports § 2.3 did not say `constants` travels -- each says it, and `constants.py` sits in the residual row's travels-as cell; ⑧ where a PySCF deck's outputs land was stated in two places, and the wrapper's header gave both rules for the same files -- job-contracts § 4.2 is the home, § 2.1 and the header point to it; ⑨ the test pinned a `reason` no document stated -- `web-api.md` states the route's answer. **Noted, open (outside ⑨'s three):** the Build route's peptide branch collects no warnings, so `peptide.py`'s C-terminus warning (the chain ends as an aldehyde) reaches no web user (R5, `science/validation.md`); the vibration deck spells `_initial.xyz`, `_optimized.xyz` and `.chk` by hand beside `pyscf/input`'s `ROLE_*` names, and `engines/overview.md` names the optimization deck's GPU switch as PySCF's only one | **done** -- `09de947a`, `bb46e09e` |
| **M2b′** | the vibration finishes itself | W41, W36 ⑤ (its recipe half) | a SIESTA vibration's launch ends with `<label>.spectra.json`, derived by the job through one engine-neutral analysis; `summarize` derives nothing; the modules the job imports are carried and importable in its env; the contract says where the result is and how to read it | **Reviewed 2026-09-28** -- read by an agent in full against `5e79002d` (the 22 bundle members byte-identical to the commit's, each imported alone by a python that cannot import molbuilder, the finish run from the bundle), every finding re-read against the text: ① a failed finish read *finished* on every surface -- `run_status` took the output's ending over the marker -- the marker names the failed finish (`parse/dirs/job.FINISH_FAILED`), `run_status` reads it as failed, `project-layout.md` § 1.6.3–1.6.4 say so, and the road test breaks a finish (red against rc=0 and against the ending read first); ② nothing asked before the run whether the finish could load, and a missing ASE blamed the output -- the wrapper asks the bundle `loads` after activation and stops before SIESTA, ASE is imported when the bundle loads, `ops/installation.md` names numpy and ASE in the job envs; ③ an attempt prepped before the finish has no door, and a failed finish's retry was unwritten -- § 5.5 states the command that finishes an attempt by hand; a cheaper door, and whether older attempts get one, is the user's call (open); ④ benchmark trials ran the finish over 3-iteration SCFs and wrote meaningless spectra -- a trial gets neither the block nor the finish; ⑤ bare `i + 1` / `n − 1` in the finish -- through `engine_atom_index`, which travels; ⑥ four contract sentences did not match the code (the command, `Geometry.Constraints` as read, the argument, the record's five keys) -- corrected; ⑦ stale text (`vibration.md` § 2 and I1, `normal-modes.md`, the README, `project-layout.md`'s marker and attempt rows, `job-system.md`'s `Job` and `shared`, two code comments) -- corrected; ⑧ tests: I22's failure half, the stop before the engine and the bundle run where molbuilder is not are pinned; the tautological criterion test reads the deck's criterion; the mass test goes through `result_of`; `identity` left the bundle (nothing there imports it); I23 says which functions may import the host's packages; **not pinned:** `read_force_constant_run`'s refusals, the force-constant seed without targets, PySCF's held-atom Methods sentence; ⑨ the behaviour against `from_siesta`, measured identical but `thermo.note` and the timestamp -- no change; ⑩ V1.11, A1.16 U10 and W41's list -- corrected; ⑪ `PermutationError` escaped prep's transport door as a traceback -- caught; ⑫ the travel notes written, `PERMUTATION_FILE` used, the rigid-motion count taken from the analysis. **Open:** the finish parses the whole output where FC step 0 suffices; numpy and ASE were installed here on the user's W41 words, which the review asked to confirm -- said to the user | **done** -- `5e79002d`, the review's fixes `51590fa6`: the finish (`spectra/siesta_vibration.py`) over the engine-neutral analysis (`spectra/vibrational_analysis.py`), shipped as `mb_vibration.pyz` and run by the wrapper after SIESTA (`Job.finish` from `DeckSpec.finish`); the deck's `vibration` block; the block grammar moved to `deck_record.py` and the permutation record to `atom_permutation.py` so a job can read both; `relaxation_of_output`; `summarize`'s vibration branch and Task setup's summarize line gone; the catalogue names both writers; numpy + ASE in the SIESTA recipes and ASE in PySCF's, installed here (dry runs: additions only); the SIESTA road test green, red with the finish removed; the fresh H₂O run on `claude-validate` gives the host path's 1509.2 / 3569.7 / 3667.4 cm⁻¹, and the Results tab opens its spectrum |
| **M2b″** | the displacement sweep | V1.23; W41's second force-constant stage at the input geometry | every force-constant stage takes the relaxed geometry whatever its name; `summarize run` compares the stages into one record that names each stage's files and copies none; the contract says what each file of a vibration holds; the Results tab opens the record | **Reviewed 2026-09-28** -- read by an agent in full against `51590fa6` (the bundle built and run where molbuilder is not; the sweep's overlaps and force-constant change recomputed from the raw `.FC`, 0.044374 eV/Å², equal), fifteen findings, each re-read against the text: ① the finish's failure changed the state rule in the code, not in the section that owns it -- `running-a-job.md` § 4.2 (the order of evidence, the states, the words) and `project-layout.md` § 1.6.3–1.6.4 state it; ② an ended output with no marker read *finished* while the finish ran, and for ever after a kill inside it -- the session log's `finish started:` line (`wrapper_log.FINISH_STARTED`) makes it *running*, and *failed* once the monitor saw the process go; ③ the pre-launch check stopped the job before the run index existed, so it wrote no marker and the attempt read *queued* for ever -- it runs after the monitor starts, on the job's own python, and concludes `finish cannot load`; ④ the wrapper table had no rows for the finish's two blocks and its guard rendered neither -- both rows, and a finish fixture in the guard (red without the row); ⑤ the sweep compared stages measured at different geometries -- refused by the results' structure hashes, naming how far an atom moved (red on the road with a relaxation re-run between the stages); ⑥ a failed stage read pending and a reader error was a traceback -- `pending[]` and `failed[]` carry `run_status`'s state and detail, reader errors caught per stage; ⑦ a bare `n − 1` on the FC range -- `from_engine_index`; ⑧ tests: a measured free-water fixture pins matching by shape (a rank swap, the mass-weighted cosine of a mixed pair), the PySCF refusal on the road, a pending stage, the geometry refusal, copies-nothing, the stop's marker, a finish stopped inside and one still running, the flat refusal counting a disabled stage -- each red against its mutation; ⑨ the record held what nothing drew -- the presenter and the printed table show every key, a varied value with its unit; ⑩ contract statements the code contradicted (the presenter counts, § 6.9's names, § 8.1's file map, *summarize derives nothing*, `relaxation_of`) -- corrected; ⑪ the marker's note went unparsed and a second reader matched it by substring -- `read_concluded` parses it, the one reader; ⑫ the check ran bare `python` -- `$_mb_py`, its words the real error; ⑬ *refused before anything is written* -- *before any sort, permutation record or deck*; ⑭ stale text (prep and `pyscf/stages` comments, § 5.9, `match_modes`, `summarize`'s docstring, § 9's test row, the road test's header, `web/trajectory.md` § 3, `running-a-job.md`'s node decisions, § 2.1's one-stage sentence, `web/results.md`, the warm-file note) -- corrected; ⑮ Task setup's card promised the sweep record as the launch's -- a `summarize` moment on the catalogue (the transport record's too), a disabled second stage counted by the flat refusal, the near-degenerate caution stated. The measured fixture is a fresh road run (`claude-validate/spectrum/h2o-siesta-free`, free water: 1574.7 / 3634.0 / 3807.0 cm⁻¹), whose launch also ran the moved check and the finish; 2593 tests of what the fixes touched and the six SIESTA road tests green | **done** -- `51590fa6`, the review's fixes `9ea467cf`: I24: which geometry a stage takes is asked of `vibration_render_kind`, not of the name `freq`, so every stage after `relax` measures at the relaxed geometry; the flat layout refuses a second force-constant stage before writing anything (its stages share one `<label>.FC` and one `<label>.spectra.json`); `jobset summarize run [--tolerance-cm1 X]` writes `<label>.fc-sweep.json` at the calculation root (`spectra/displacement_sweep.py`, `molbuilder/fc-displacement-sweep@1`): each stage with the displacement it used and its own overrides, every mode matched by shape (mass-weighted overlap, one-to-one), the largest force-constant change read from each stage's raw `.FC`, the stages still pending; I25: every stage's files are named by path, none copied; `engines/vibration.md` § 5.9 and § 6.9 say what each file holds; the parse registry reads the record (`parse/sidecars/fc_sweep.py` -- without it the Results door never offered the file, measured) and the `fc-sweep` presenter draws it; the road test runs relax → freq (0.04 Bohr) → freq_half (0.02) on H₂, red with only `freq` taking the relaxed geometry and red with the reader unregistered; on `claude-validate`, H₂O with O held, δ = 0.0212 / 0.0106 Å: 1509.2 / 1513.4, 3569.7 / 3568.4, 3667.4 / 3665.7 cm⁻¹, overlaps ≥ 0.99999, the force constants' largest change 0.0444 eV/Å² (0.13 %) |
| **M2b‴** | the SIESTA vibration shown as what it is | the user's rulings of 2026-09-28 (no spectrum and no width for a SIESTA result, its modes and their animation; the thermochemistry said as what it is; engine-neutral, engine-own and one-route parameters kept apart); V1.6; V1.7; W42 D5 (sisl) and E2 (no pressure where none enters) | a SIESTA result shows its modes and their animation -- no chart, no width control, no column or dot for a channel its route has not -- and its electronic-structure tab names the planned projected DOS; the thermochemistry is labelled by its regime and read from the file, not derived; `temperature_K` reaches SIESTA; the retired selectors are gone; sisl is in both SIESTA envs | **Reviewed 2026-09-28** -- read by an agent in full against `4cb650c2` (536 tests of what it touched run beside), every finding re-read against the code: ① an IR-only PySCF run said *asked and recorded none* mid-sweep and the viewer stopped watching -- infrared had no phase flag: `phase_ir`, on both writers, read by the viewer's done-rule, its sentence and its fingerprint (`vibration.md` § 4.9; red against three mutations); ② a file written before 2026-09-24 carries the old grid construction, so read against its own reference its bars no longer add up -- **ruled by the user, 2026-09-28**: *"don't worry about obsolete results. correctness of code base and UI design is the most important thing. dead runs will be removed and updated"* -- nothing marks or refuses them; ③ a mode the selector left out was told *not reached yet* while the probe ran -- read from `selected_mode_idxs_1based`; ④ 0 K gave a zero ZPE -- `vibrational_thermo` returns the ZPE at `T ≤ 0`, and the item starts at 1 K, the free molecule's entropy having no 0 K value; ⑤ PySCF's thermochemistry notes said nothing of what the numbers are good for, and the fallback's note blamed held atoms for a free molecule -- both say it, with the true reason; ⑥ an old template's `es_top_n` / `es_threshold` was refused as a misspelling -- named as retired (`template.RETIRED_ITEMS`); ⑦ the `vibration` block's reader requires `temperature_K` under the old tag -- `molbuilder-vibration/v2`, the three descriptions say it; ⑧ a served constant nothing read (`boltzmann_hartree_k`) -- gone, `architecture.md` says three; ⑨ about 25 stale passages across `vibration.md`, `web/spectra.md`, `spectrumchart.md`, `presenters.md`, `results.md`, `template.md`, `backend-architecture.md`, `installation.md` and the code's comments -- corrected; ⑩ tests: the thermochemistry's numbers were pinned nowhere -- the CO₂ page test reads the bars (they add up, the last is the file's `G − E_elec`, and CO₂'s translation + rotation + pV is 3.5 k_B T: red against the old derivation, which gives k_B T), the SIESTA page test reads them summing to `F_vib`, the held PySCF road run records no pressure, the four no-spectrum cases and the done-rule are a node test, the gate test goes through the kind's own door. **Found on the way and fixed:** the Raman and probe dots hidden by an attribute a `display` rule overrode (`.phase[hidden]`); the chart drew a curve in an infrared lane titled *not computed* (the sum fell back to unit heights); the SIESTA deck's Mulliken hint spelled the retired `WriteMullikenPop`. **Open, the user's calls:** `pressure_atm` offered for a held structure and the frequency window left editable under `skip` / `explicit`, where each enters nothing (§ 3.2's rule would lock or warn) -- *answered 2026-09-28 ("your recommendation is ok"), built with M2b⁗*; E3. 1907 targeted tests, the seven SIESTA and eleven PySCF road and page tests green; checked in a browser on `claude-validate`'s SIESTA and PySCF water results | **done** -- `e519e3cd` |
| **M2b⁗** | the mode positions, and the review's fixes | the user's rulings of 2026-09-28 — a result with no strength still draws where its modes are (*"we can still have the modes ploted as one straight bar in a wave-number plot ... just bar/line to show where the mode is"*); a setting that enters nothing warned about or locked (*"your recommendation is ok"*); the Methods text wraps on the Results tab; the Hartree–Fock dispersion question (*"correct comments and hint would be needed and scientifically correct decision applied"*); V1.8; V1.18 | a result with no strength draws one line of one height at each mode, picked like a stick, under *Mode positions*, with no width or floor control and the sentence saying why; `pressure_atm` with atoms held and the window outside `all` are warned about at prep, and the window is locked on the form; the level of theory is one answer, `PySCFConfig.is_dft`, and the dispersion correction applies on either method; the Methods rules live with the panel and wrap | **Reviewed 2026-09-28** -- read by an agent in full against `3f1b5575`, every finding re-read against the code: **D1** the explicit mode list, text by contract (`"3, 7"`, `"3-7, 12"`), was read character by character -- the deck wrote `['3', ',', ' ', '7']` and Phase 4 stopped on `int(',')`, the Methods paragraph counted characters, and the tests passed lists, a shape production never builds. One reader, `PySCFConfig.explicit_modes` -- the atom index list's grammar, made public as `selection.parse_index_list` rather than written a third time -- for the deck's constant, the Methods count, the reference selector and a refusal at prep; the parity test reads the deck's own constants (red against the old line), and a road test runs the probe on the modes `"1, 3"` names after `"0, 2"` is refused -- no road test had ever selected a mode. **D4** (older than this work) *dispersion "none" chosen on either form ran D3BJ*: the forms turned "none" into `None`, which a template writes valueless and `prep` fills with the default -- "none" is now the value itself (`dispersion: str`), the two mappings are gone, and a test follows it hand-over → template → prep → deck (red with the mapping back). **D2** the heading and the chart counted strengths by two rules -- the viewer takes the chart's. **D3 and the Hartree–Fock ruling:** the warning fired on "none", and its premise was false -- Hartree–Fock has no correlation, misses dispersion entirely, and D3/D4 carry parameters fitted for it; PySCF (and gpu4pyscf) apply `mf.disp` to HF through the energy, gradient and Hessian, reading the method as `hf`. Both decks had dropped it under HF, silently at its default, while the result recorded the value as run. **Decided and built:** the dispersion item applies as written on either method (RHF with `d3bj` is HF-D3(BJ)); the dresser is `_mb_configure_theory`, on every deck (`THEORY_SECTION`); only a changed functional is warned about under HF; the Methods paragraph names each correction by its own papers -- D3(BJ) [Grimme2010, Grimme2011], D3(0) [Grimme2010], D4 [Caldeweyher2019], the first and last added to `references.bib` unverified -- where every version had cited the damping paper alone; the method and dispersion hints say so (`pyscf.md` § 7a, `vibration.md` § 4.10). **Q1** a window set outside `all` was silent at prep, though a locked field keeps its value and the hand-over carries it -- warned. **Q3** `GRID_LEVEL` recorded as None under HF, like the functional. **S1–S10** stale text corrected (§ 6.5's SIESTA row, § 9's test map, I13, § 10; `web/spectra.md` § 9a.2's deleted *after* form; `presenters.md`; `spectrumchart.md` § 8.4's `positions` field; the lock comments; the modal the Methods composer no longer ships in; the chart's label, now its section's heading). **Tests (T1–T5):** the grid's silence under HF, the dispersion on HF decks (both decks, and the Methods paragraph), each correction's papers resolving in the bibliography, the positions picture with an imaginary mode and its switch to heights, the lock at page load, text-shaped fixtures -- each red against its mutation. **Q2, kept as designed:** the spectrum section appears with the first mode; before it the phase line is the run's status. **The batch found two more:** the optional-item count the documents state (17 → 16, `dispersion` no longer optional; the code comment's own numbers replaced by a pointer); and `_el` written twice (`fc-sweep.js`, `transport.js`, since M2b″) -- fixed with the Results-tab milestone, whose files it shares. 1043 + 57 targeted tests green, the prep-only road tests green; the PySCF and SIESTA launch tests wait for the Au–BDT–Au relaxation (one job at a time) | **done** -- `40720388` |
| **M2b⁵** | the Results tab, reviewed as a design | the user's asks of 2026-09-28 — *"the methods text in th resul tab is not wrapped ... check other css issues by a full text review of the rsult tab design"* — and the four recommendations that review produced (*"follow your recommendations on the four, go ahead"*) | a file reaches a viewer by the dropdown alone and a run still going is followed until it finishes; every viewer says why it could not load and lifts the cover at once when it cannot; muted text meets 4.5:1 on every ground; one surface, one inset; the documents say what the tab does | **Reviewed 2026-09-28** -- read by an agent in full against `9a89f731` (926 targeted tests beside it, no engine), every finding re-read against the code. **Built:** ① the spectra panel's path box and its *Load once* / *Start watching* / *Stop* went -- a second way to name a file, and a live run picked from the dropdown sat as a snapshot until someone pressed a button; `loadByPath(path)` follows any unfinished run and lets it go when its last phase lands (`_settlePostLoad()`, one rule for a load and a tick), and the status line says *following* only while the poll runs; ② the trajectory panel's own status line (`#trajectory-status`, namespaced) -- it carries a refusal, a failed mount, an export and the cell's provenance, and no longer the file's mtime or a frame count (the badge's time and the frame bar's frames; the count could disagree with the movie); ③ `--text-muted` #959ba7 -- 5.8 on a card, 4.6 on a focused input where a placeholder shows (#8e94a0, tried first, left that at 4.2) -- and the chart's own *not computed* colour with it (3.3:1 before, text in the plot); ④ the double-click opens the sidebar's file viewer (results.md § 2.1) and the documents say markdown and text mount nowhere now -- kept registered by decision. **From the CSS read and the review:** the banner scoped to `.app-header`; every viewer and the empty state on page-shell's `.card` -- the spectra partial has one card root, the record viewers and the fallback lost the insets and surfaces that restated it; the spectrum chart's host stopped drawing a second frame with its own height (420 px over a 460 px frame clipped the x-axis on a tall screen; `spectrumchart.md` § 5.4 now agrees with § 11 and the code -- the host gives the width, the chart's frame its height); an *Infrared* phase dot; the thermochemistry note in the panel's own sheet; the Methods summary's doubled margin; the Copy button's undefined classes; font shorthands and the sidebar's stacks on the font tokens; status lines, error boxes and the failed-mount card wrap a long path; the sweep's tables scroll in their own box (a scrolling table drops its semantics for some readers); the content-links rule for bare links only (it underlined *Open in Molbuilder*); 55 comment lines in the spectra sheet rejoined. **Defects the review found, fixed:** the picker started *Parsing…* after announcing, so a viewer answering inside the announcement -- the file already on screen, after a tab return -- left it up for 8 s (fixed, and a page test red against the old order); `status.set` replaced a line's whole class list, stripping `trajectory-status` on the first write (it now replaces the severity alone, and leaves an unchanged line unwritten so a live region is not re-announced every poll -- the spectra viewer's own writer now goes through it); a failed load in spectra, trajectory's network error, the structure viewer's six failure exits, the error card and `showError` never signalled, so the cover sat 15 s over the error (each signals now, through one `announceReady` in `lib/inspectors/lifecycle.js` that also fires by timer where a background tab runs no frames); dead rules and ~30 stale sentences (the markdown/text claims, the Spectrum tab "loading" results, the retired audit tests named as enforcers -- two Task-setup `[hidden]` guards untested since 2026-09-10, said so). **One element builder**, `lib/dom.js`, for the three record viewers and the Run panel (four copies). **Tests:** a followed run let go when it finishes, a failed load that says why and lets go (both red against their mutations), the rescan's *Parsing…*, the Infrared dot on a SIESTA result (runs with the engine tests); four source pins retired for them. **Left for M2f:** a run killed mid-phase leaves its flag *running*, so the viewers follow it until the page closes -- the stop belongs to M2f's one "finished" door, not a second reader. **For the user:** ~~Auto-detect suggests a total spin of 0 with a non-polarized SIESTA spin, which prep then warns is ignored~~ -- **closed by M6** (2026-09-28): the button and its suggestion are gone, and a closed shell writes `Spin non-polarized` with no pin. 1168 + 46 + 33 targeted tests green, the engine road tests after the relaxation | **done** -- `54af590c` |
| **M2c** | recipes and start-up | W36 ⑥ (⑤ built with M2b′) | the conda listing and the CUDA probe at their use; the two comments | **Built 2026-09-29:** `cli.main` takes no snapshot -- `diagnostics.get_capabilities` takes it on first use, and a `RuntimeConfigError` from any command is answered `Error: ...`, exit 2 (`molbuilder --help` 1.4 s -> 0.5 s); the registry is `envs/recipes.builtin_recipes()`, built on its first ask, with `_pyscf` / `_siesta_gpu` functions of the CUDA version (all six recipes repr-equal to HEAD's on this host), `BUILTIN_RECIPES` gone everywhere; the two comments corrected (`jobset prep --help` DOES print the header). **Reviewed 2026-09-29** -- read by an agent in full (the diff, `git diff -w` for the re-indented recipes, the touched functions and sections; logging stubs, a mutant package), every finding re-read against the code: ① **a regression**: the eager snapshot had been an accidental pre-flight -- with it gone, `serve restart` over a broken `molbuilder.json` SIGHUPs the supervisor, the fresh child exits 2, and the supervisor does not respawn (`deployment.md` § 1.0c), so no server; `jupyter restart` likewise stops a working notebook for a start that cannot happen -- **fixed**: `serve restart`, `jupyter start`/`restart`, the Reload route and the notebook tab's Start route (the two routes predate M2c: the same fact through its other doors, found by widening the review) read the file first and refuse in its words (a 409 on each route, which each page shows), the rule in `deployment.md` § 1.0b, `jupyter.md` § 5, `configuration.md` § 2.2; ② the probe test missed an eager probe in a group callback -- `envs list --help` and `jobset machines` added to the no-probe runs, `envs list` asserted 0; ③ "`--help` among them" was false for a `jobset` verb, whose header reads the file -- said; ④ the server's first notebook-status request paid `nvidia-smi` -- `create_app` builds the registry at start, beside the snapshot, for the snapshot's reason; the registry is built WHOLE on its first ask (the notebook's recipe asks it too), stated in `env-framework.md` § 3 -- **kept so, 2026-09-29** *(user: "go with your recommendations on all three")*: a per-recipe build would spare the notebook's start 0.08 s for more machinery; ⑤ the CUDA test was API-level while `envs install --dry-run` shows every fact -- the three tiers (the driver's 12, the variable's 11.8, the default 13) are read off the command line, the driver tier tested for the first time; three API tests and the cache-clearing fixture retired; ⑥ `env-framework.md` § 3 and `installation.md` (twice) restated the pin without the driver tier -- swept; ⑦ "startup snapshot" prose in `diagnostics`, `runtime_config`, `envs/install`, `recipes`, `_amber` -- reworded; ⑧ nits: one spelling of the CUDA default, the registry's order claim and the stale block header, the handler comment's "malformed", a test's docstring, this plan's W36 row, two test maps. Every new test red against its mutation (the eager snapshot in `main` and in the root callback, the import-time probe, each CUDA tier, each of the five doors). **Noted, open (older than M2c):** `create_app(config={})` still reads the disk config through the snapshot, so a broken file stops a `--no-auth` server, which `cli.py` says ignores the file | **done** -- `e5e162ad` |
| **M2d** | one prep entry | W38 F7 | the page and the CLI call one prep entry returning its findings and decisions as data; the CLI prints and asks, the page shows and confirms | **Built 2026-09-29:** `jobset/prep.py::prep_stage` → `PrepAnswer` is the verb, called by `molbuilder jobset prep` and by the Task setup route: refusals, the stage grammar, the preflight (findings), the inputs (their notes), the *already under way* question (returned with nothing rendered; answered by calling again with an `Answer`, recorded in the ledger in its words), the five steps, the attempt, the transport carry, the resources, the deck and its launch agreement, the ledger lines. The command line prints the answer and asks at the terminal; the route returns it whole (`PrepAnswer.as_dict`) and the tab shows it (`task-setup.md` § 11.1), with a Confirm for the question. **The assembly moved down with it** -- `_declared_execution_pins` through `bench_inputs` (was `_bench_inputs`), 1,140 lines, from `jobset/_cli.py` (floor 7) to `jobset/prep_inputs.py`, the conductor's own assembly beside it: refusals are `PrepError`s and what a person is told comes back as notes, so the `redirect_stdout` the route and the grid card wrapped them in is gone, and `web/` no longer reaches across to the command line (A12's recorded debt, A7). An axis-less bench preps the machine's proposal on both doors. Contract: `job-system.md` § 5.3 (one prep, two doors; § 9's map), `architecture.md` A12, `task-setup.md` §§ 1, 10, 11, 11.1, `web-api.md`. Tests: one prep through each door compared (findings, attempt, agreement, ledger), the question on both doors (nothing rendered until answered; the tab's Confirm and the terminal's *no* in the ledger), the axis-less bench on both, and the tab's question and Confirm in a browser -- each red against its mutation (the old second door; the entry never asking; the route ignoring `confirm`; the command line not asking; the page not sending `confirm`); 1943 + 14 targeted tests green. **Reviewed 2026-09-29** -- read by an agent in full (the diff, the moved block against HEAD line by line, every changed section against the code), sixteen findings, each re-read against the code and fixed: ① a refusal dropped what the entry had found -- a bench refused with *"see the crossed-out list above"* showed no list on either door -- so `PrepError` carries `findings`, `notes` and the `partial` answer, the terminal prints them before the refusal and the route returns them beside it, and the preflight's ledger line is written, first, on the pass that acts or refuses; ② the tab offered no Prep bench without a declared axis -- the measure block is on every stage now, saying the target proposes the grid; ③ Confirm skipped the Save-first check -- it asks what Prep asks; ④ an answer was not tied to what the person saw -- `Answer.evidence` names it, and the question comes back when the folder shows something else; ⑥ the page never saw a deck's own checks -- `prepare_deck` hands them to a sink and the answer carries `deck_findings`; ⑩ a user-fixable `ValueError`/`KeyError` was a 400 on the page and a traceback in the terminal -- the entry translates both, the route catches `PrepError` only; ⑪ `prep_inputs` is the conductor's own assembly, not floor 3, and the ledger a floor-1 writer the one entry appends through (`architecture.md` § 2.1, `ledger.py`); ⑤ ⑦ ⑧ ⑨ ⑫ ⑬ the documents and residue (the machine refusal before the question is a bench's only; *"prints nothing"*; the old *browser describes, terminal acts* in `job-system.md` § 1, `execution/overview.md`, `running-a-job.md`; § 11.1, § 7a, `web-api.md`; the answer table; `_bench_inputs`, `_ask_if_underway` and the stale *grid enumerator* claims in `prep.py`); ⑭ `Issue.to_json` for the wire form; ⑮ the *silent* deck untested -- a PySCF road test on both doors; ⑯ nits. Every new test red against its mutation (the refusal dropping its notes, any answer accepted, a silent deck given an agreement, the deck checks not collected, Confirm skipping the Save check, the bench offered only with axes); 2521 targeted tests green. **Second review 2026-09-29** -- a fresh agent over the whole uncommitted work (M2d with its fixes, V1.36, Fe), thirteen findings, each re-read against the code: ① the terminal took a question that came back (the folder changed while the prompt waited) for a finished prep -- it asks again now, saying so; ② the page offered a bench on every transport and PySCF stage, which the entry always refuses -- one function says why a description takes no bench (`prep_inputs.bench_refusal`), asked by the entry's gate, by the bench's assembly before it reads any machine, by the plan preview and by the folder answer, whose `bench_refusal` hides the Measure step; ③ a preflight error dropped the preflight's notes and their ledger line, and ④ a folder holding two templates escaped as a traceback and a 500 -- steps 1–3 moved inside the entry's refusal handling; ⑤ the preview called a one-point declaration *the target's proposal* -- it names the declared cell; ⑥ a bench trial of a force-constant stage was still judged against the input's record -- the relax record travels to every deck at that geometry as `spec_for`'s `relaxed_by`, which the deck no longer reads out of the finish's block; ⑦ the held set and the judged force spelled twice in `sidecar.py` -- one helper each; ⑧ a run's device type is read from the target before the question too -- the sentence says so; ⑨ residue: *CLI-shipped; the web describes and observes*, *five floors*, A7's *floors 3–7*, *the act half stays on the terminal*, *why this page cannot do it*, the retired § 1 rule quoted in two tests, *floor 3* in a comment and in this row, `_bench_inputs` in the present tense, a validator comment naming a deleted engine registry, `preparing-for-another-machine.md`'s quote of the old § 4; ⑩ the Fe comment and the chemistry document still said *usual guess* -- they quote the starting-guess text the code shows; ⑪ the terminal printed the preflight's notes after the decks' own warnings -- the entry hands them over before rendering (`on_found`); ⑫ the tests above, and `vibration.md` § 9's table; ⑬ nits (the remedy names the launch, previews hand in `Resources()`, a new preview retires the last answer, blank lines, one history name). Every new test red against its mutation (the prompt asked once; the notes after the error; the template gate outside the handling; the folder answer and the preview without the refusal; the page ignoring it; the record withheld from trials; the remedy without the launch; the held set never compared; the notes printed late); one old test moved to a SIESTA description, since a PySCF bench now stops before the machine question it was about; 1110 targeted tests and 41 browser tests green, the six engine e2e files not run while the Au–BDT–Au frequency job holds the machine | **done -- 379a1426** (V1.36 73374112, Fe 5a643599 closed with it) |
| **M2e** | the pipeline log, always | W20 | every prep writes it, from both doors; the flag is gone | | **done 2026-10-05** (unit 10a) -- every prep opens its log, both arms; `--pipeline-log` and the parameter it set are gone; the catalogue's row says every prep writes it, and the card's check looks for a calculation-level file at the root, where the log lands (`manifest` rows carry their `level`); the terminal prints the log's line on its own, where it had read as one more trial |
| **M2h** | the stage hand-over | W37, W38 F9 | the one contract section; the default, the warning, the explicit choice, one record read by all; *Continue from* on the page; `summarize` reads the record | | **done 2026-10-05** -- W37 (2026-10-01), and unit 10d: F9's frequency stage, the record's one writer |
| **M2i** | stage numbers, no on/off | W38 F4, F5 | numbers come from disk; no `enabled` (old files read by the agreed rule); a removed stage's files marked `.disabled` | | numbers from disk done (9b); **no `enabled` done 2026-10-07** (unit 12b-1); **no `.disabled` mark** -- dropped 2026-10-07 (user: "we practically can always use checkpoint"): a removed stage's files stay untouched |
| **M2l** | the small two-door cases | W38 M1–M5 *(transport `restart` refused: done by K4's one door, `template.unread_overrides`, 2026-09-30 — found built when M5 step 3 took it up)* | as W38 records them | | **done 2026-10-05** (9b part 5; user: "go ahead with 1") -- M1: every reader asks `template.find_template(base, label)`, the folder's one template, named for the label, refused by name (two, or another name); the writers form the name with `template_path`; M5: a benchmark trial's run script is told the trial's own label (`paths.trial_label`), the label its deck carries; M2 (the vibration summary asks `Shape.stage_dir`) and M3 (the preview reads the machine record, never probes) found built; M4 with 9b part 2c |
| **M7** | the spectrum view's API | W21 step 4 | W21's own | none owed — no new code; each part was reviewed in the milestone that built it | **done 2026-09-29** — nothing was left to build: every part of W21's step 4 was built by an earlier milestone, each re-read against the code (W21 archived with the evidence) |
| **M9** | two science features | V1.26, V1.27 | each needs its decision first | | **folded into M10 2026-09-29** — W42 folds both in; V1.27's decision is W42's D8, V1.26's is still owed |

### The milestone table — rows rewritten in the trim, as they were

> **Verdict (2026-10-08 validation):** rows rewritten in the trimmed § 0a — their text as it was. M2f REFUTED stale → done 9b (`runrecord.py:139,309,44`; `runs.py:343`); M2g REFUTED stale → done 9b part 3 (`warmfiles.py:122/156`; `tests/data/restart_files.toml`); M2k PARTLY (`placement.py:182,236,130`; `model.py:267`; `running-a-job.md:430`; the claim / match half unverified); M5 PARTLY stale → § 5u.1 holds each step's state.

| # | milestone | items | done when | review | status |
|---|---|---|---|---|---|
| **M2f** | launched and finished | W38 F2, F3 | one door each, the same for flat and hierarchical, asked by every caller -- the Results viewers among them, which follow a run until its phases finish and so, for a run killed mid-phase, until the page closes (found in M2b⁵'s review) | | open |
| **M2g** | one restart-file list | W36 ⑧, the bias chain's list (W38) | every reader asks the calculation's list; Task setup states which is in effect; the shipped lists say how to customize | | open |
| **M2k** | what a job runs with | W36 ⑦, W38 F1 + the one queue record, F6 | one placement and one record; GPU request / claim / match; the precedence table in the contract | | open |
| **M5** | transport | **W43 — § 5u, the one list** *(consolidated 2026-09-29: W27 floor 3 · W30 · W25 · W24 · W10 · W32 ②–⑤ · § 5c.3 · § 5o · § 5p)* | § 5u.1's eleven steps, each closed as a milestone |  | **Started 2026-09-29, ahead of M2e–M4** *(user: "after all work on spectrum is done and review is green, continue to work on transport as the next item on plan")*. Consolidated the same day (§ 5u) from an inventory that read every transport row against the code, and **thirteen decisions ruled, one open** (§ 5u.2: TD9, the order of step 5 against M2). **Step 1 built and reviewed (`4c612262`)** — read by a fresh agent in full, every finding re-read against the code and fixed; its two design questions are TD11 and TD12 — its rulings in `engines/transport.md` § 6.1b, from the engine's own source (SIESTA 5.4.2 and its `libfdf`). **Step 2 built and reviewed (`b63f219d`). Step 4 done 2026-09-29**: the Au–BDT–Au ladder ran seed to transmission on step 2's code — T(E_F) = 0.161, G = 0.161 G₀ at 0 V — and its record shows on the Results tab |

> **Verdict (2026-10-08 validation):** § 0b — items 1–3, 5–8, 10a, 11 CONFIRMED; 10b UNVERIFIED (the user's home); 12 OBSOLETE (the save is always, `checkpoint.py:1877`, `job-system.md:1017`; residue `job-system.md:960` *offering to save*); C1–C25 CONFIRMED; R1–R24 CONFIRMED (R11 parked, open); D1–D30 UNVERIFIED (commit `0a7ea0e4`); T1–T30 CONFIRMED; S12 UNVERIFIED; *your word* 1–5 CONFIRMED (`_cli.py:2357-2365,2073`; `diagnostics.py:497`; `record.py:44-65`, schema @3). The one line kept in the trim: *W54's ledger archived; R11 parked, open*.

### 0b. The ten decisions of 2026-10-02 — and every review finding's state

> *(user, 2026-10-02: "consolidate and give me a fucking list of things and
> details i need to read fucking through"; then "go ahead, make sure your plan
> is consolidated and all the review results is validated and fix of those are
> also considered - the review was done in parallel i don't want you to miss
> those")*

**The decisions, in the user's words.** Built in this order, each with its
contract text first and its own commit; a row moves to *done* with its commit.

| # | what | the user's words | what changes | state |
|---|---|---|---|---|
| 1 | the recorded contract's field names (D2) | *"if only three readers for D2, i'd rather unify the names, rather than having a drift from contract"*; *"leave the project alone"*; *"don't care about old projects"* | the record writes the catalogue's names — `siesta_mesh_cutoff_ry` → `mesh_cutoff`, `energy_shift_ry` → `pao_energy_shift`, `electronic_temperature_k` → `electronic_temperature`, `k_mesh_transverse` → `kgrid`; the translation table goes; `projects/` is not touched and nothing reads the old names | **done `e1d88b22`** — 503 targeted tests green, 2 mutations red |
| 2 | page code only the deleted form builder fed (D3) | *"agree to D3"* | the section paragraphs and the `comma-floats` control go: `form-schema.js`, 5 CSS rules, `test_form_schema_section_description_js.py` | **done `2ec16acd`** — `form-schema.md` § 4's kinds table now the page's eight; 615 targeted tests green, the Build and transport browser tests among them |
| 3 | which labels transport reads (D4) | *"why the fuck can you not make the molbuilder just fucking read L-electrode and R-electrode? … why the fuck do we need *-electrode matching"*; *"labels come by user's setup and they come with reasons you don't have any fucking business with"* | transport reads `L-electrode` and `R-electrode` by exact name; the `*-electrode` pattern goes — **nothing else** | **done `f5619181`** — `sort.ELECTRODE_LABELS` is the one list (the emitter, the lead extraction, the vacuum / tiling / frozen-lead checks); the emitter's third-lead branches went with the pattern; the device deck byte-identical but its comment; a `tip-electrode` is now named by the unread-label warning (its advice lost the *"plus any \*-electrode name"* clause), and the device deck declares `L` and `R` only — mutation-pinned in `test_transport_prep.py`, D4 closed; 1041 targeted tests green |
| 4 | who may reload the server and clear the block list | *"stop bothering me with #4. you fucking know it works and i don't give a fuck"* | no change | settled |
| 5 | secrets in `molbuilder.json` (C6, C7) | *"no fucking secret in molbuilder.json except the cert files"* — `configuration.md` § 3.1 had settled Google's on 2026-09-20 | a provider's secret is read from its fixed home in `secrets/`; `client_secret_file` leaves the file, refused by name; § 3.1's operator-named row keeps the TLS files only; your `molbuilder.json` loses its Google path (backed up) | **done `749716f4`** — each OAuth kind's secret has a fixed home, `config_dir.client_secret(kind)` → `secrets/<kind>_client_secret` (Google's file unchanged; GitHub, Microsoft and ORCID take the same pattern), one `placement` row each; `client_secret` and `client_secret_file` refused by name, naming the kind's home; C7 folded in (oauth re-reads the secret through the door — the mtime cache and its second resolver gone); your `molbuilder.json` lost its one line through `write_config_scope` (backup with the session), the server restarted on it and Google's sign-in redirect reads the secret; 812 targeted tests green, four mutations red |
| 6 | the host env's name (D21, C23) | *"i agree with (a). let's enforce one name"* | always `molbuilder`: `envs.host` and `MOLBUILDER_HOST_ENV` removed, the installer included | **done `311ab27e`** — `effective_name` answers the host recipe's own name; `diagnostics.HOST_CATEGORY` and `recipes.HOST_ENV_ENV` deleted; `envs.host` refused by name (a `molbuilder_json.toml` row, mutation-pinned); `install-env.sh` sets `HOST_ENV="molbuilder"` and its help and missing-env error lose the override (`MOLBUILDER_HOST_ENV_CHANNELS` stays); `configuration.md` § 2.1c / § 4, `running-a-job.md`; your `molbuilder.json` has no `envs.host`; 803 targeted tests green |
| 7 | `jobset probe --out` (R23) | *"i thought we agreed probe --out goes away"*; *"retire the test with that retired --out flag"* | the flag and its `job-contracts.md` row go; its three tests retire | **done `71e70dcd`** — the option, its branch and the import it alone used are gone; the row is a note saying why the record's place is its resolver's; the three tests retired (their `--set`, `--scheduler` and unreadable-record behaviours are T29's to give road rows); 361 targeted tests green |
| 8 | the file-mode warning said twice (T24, C8) | *"#8 ok"* | one sentence and one rule — the one with the fix, the terminal's: the placement table's `molbuilder.json` row (`configuration.md` § 2.1b). *(This cell named `machine_config_mode_warning` as the survivor until it was built — the reverse of the answer given.)* | **done `615858f6`** — `placement.machine_config_finding()` hands the card that row's sentence, `runtime_config.machine_config_mode_warning` deleted, § 2.1b says so; a `0700` file is now a finding on both surfaces (the table's rule); the two tests of the deleted sentence's rule and wording retired, a third (provenance has its own sentence) with its premise; 480 targeted tests green, the card's sentence re-worded goes red |
| 9 | M5 step 5's order (TD9) | *"#9 ok"* | M2f, M2g, M2i, M2k, then step 5 — Q4 | settled |
| 10 | `gpu_partition`, and the old config backup (Q11) | *"ok to let gpu_partition to go since gpu detection is done independently from this"*; *"for b, remove that old file/dir"* | the field goes from the record and placement; `~/.config/molbuilder.backup/` | **10a done `31ca9957`** — `Domain.gpu_partition` and its `_KNOWN` column gone (a record still carrying one keeps it in `extra`, and `jobset machines` says `?? not understood`; none of yours does); placement binds every job to the queue's own partition (`_bind` inlined); `domain_serves_gpu` reads the `gpu` column alone; `gpu.md`, `scheduler.md`, `generator.md`, `running-a-job.md`; the GPU case table's `gpu` queue is partition `gpu`, the `terse` row retired; the misspelling example is `max_tme` now; 674 targeted tests green. **10b done** 2026-10-02 (the live config holds its own copy of each file) |
| 11 | a prepped stage, prepped again | *"we never fucking prep something that has been prepped"*; *"1 refuse it, redo via rollback"* (2026-10-02) | `prep` refuses a stage its plan already holds, and a redo is a rollback — `molbuilder checkpoint restore` to a state saved before the stage's prep, then a prep anew; every remedy that said to prep again, or to delete the calculation's copy, says that instead | **done 2026-10-03** — `job-system.md` § 5.0 (checkpoint 2a, the agreement), `run-identity.md` § 6 (its 2026-08-08 *warn, do not refuse* superseded), `project-layout.md` § 1.6 (the next attempt is `launch`'s), `engines/transport.md`, `engines/vibration.md`; *prepped* is what `status` reads (the plan), a refusal after the five steps began puts the plan back; the *already under way* question is gone; the stage-less refusal offers only stages not prepped; the Task setup preview says a stage is prepped and offers no Prep; `tests/data/prep_protocol.toml`, mutations red |
| 12 | the save before prep writes | *"2 yes, offer save"* (2026-10-02) | `prep` offers to save the folder's state before it writes, so a redo has a state to go back to — both doors | **done 2026-10-03** — offered whenever the folder's state is not saved (not only once it holds results: the first stage's redo needs one too), the note drafted, *no* by default at the terminal and an unticked box on the tab, *no answer* preps without saving and says so (`checkpointing.md` § 9); the save is the checkpoint door's (`Repo.init` / `Repo.save`); the sweep's two-geometries refusal keeps its check but no road reaches it now, so it is not driven |

**The order of the queue after it**: Q2c, then the ledger's open rows below,
then Q5 (transport steps 6, 7 and 10, and K21), then Q4 (M2f, M2g, M2i, M2k,
then step 5), then Q6 onward.

#### The ledger — every finding of the parallel reviews

The four W54 reviewers (C1–C25, R1–R24, D1–D30, T1–T30; reports saved
verbatim with the session) and M5 step 3's milestone review. **A row is
re-read in the code before it is acted on** — a reviewer's line is a lead.

Every row not fixed by then was re-validated against the tree on 2026-10-02
(a read-only agent, 183 reads; its account spot-checked in the code, and the
four rows it called workflow-breaking — R3–R6 — re-read whole). **State
words:** *fixed* (with its commit); *to do* (a validated defect against a
written rule — fixed without asking); *your word* (a decision); *own unit*.

| row | finding | state |
|---|---|---|
| C1 | a misspelled `admin` key made every signed-in user an admin | fixed `7dd8940f` |
| C2 | `rate_limit` read unvalidated, the limiter coerced types | fixed `7dd8940f` |
| C3 | the empty-`admin` rule stated backwards in six texts | fixed `7dd8940f` (texts to the code); who may: decision 4, no change |
| C4 | unknown keys inside sections accepted | fixed `7dd8940f` |
| C5 | `molbuilder.json` could hold a secret's bytes | fixed `7dd8940f`; its path too, Q2c 5 `749716f4` |
| C6 | the Google secret's home had two answers | Q2c 5, `749716f4` |
| C7 | `oauth.py` worked out the secret's path itself | Q2c 5, `749716f4` |
| C8 | two mode checks, two rules, two sentences | Q2c 8, `615858f6` |
| C9 | `placement`'s `*` row polices operator-named files in `secrets/`; its README says `chmod 600 secrets/*` | **fixed `b31f2068`** — the row is the README's (each fixed home has its own); the README says a file you name is yours, and points at `envs doctor` |
| C10 | `config_provenance` missed the error that happens (the card vanished with a named record) | fixed `92e84cc8` |
| C11 | the registry did not mark retired rows | fixed `7dd8940f` |
| C12 | a misplaced-credential warning said twice | fixed `92e84cc8` |
| C13 | README seeding written in place | § 2.3's row and `--help` fixed `0a7ea0e4`; the four docstrings **fixed `b31f2068`** |
| C14 | multi-scope config leftovers taught | fixed `0a7ea0e4` |
| C15 | retired `scheduler` / `script_generation` leftovers in code | fixed `0a7ea0e4` |
| C16 | texts a person reads stating false facts | fixed `0a7ea0e4` |
| C17 | getters in two shapes; `create_app(config=)` neither validates nor isolates | **fixed `90ff8120`** — a given config is `_normalise`d; `create_app`'s docstring and `serve --no-auth`'s comment say what comes from it; `test_rate_limit`'s fixture is a config a server starts on; `get_launch()` → `get_launch_mode()`, the mode itself |
| C18 | one notify-user rule written as two regexes | **fixed `96f71a7c`** — `monitor.is_notify_user`, beside `is_route_segment`; the issuer and the listener ask it |
| C19 | `molbuilder.json` parsed three times, three error policies | **fixed `fca87a39`** — `_load_raw(path)`, the one parse: a broken file is refused in the path's words by every reader (the provenance display showed it found with no values); the writer adds that nothing was written |
| C20 | constants spelled again outside their owner | partly fixed `7dd8940f`; **fixed `eb212aea`** the rest — the machine config's constant `via` gone (the banner, its source row, the card, which shows a `via` only where it varies); `CONFIG_FILENAME` and `PRIVATE_FILE_MODE` at their sites |
| C21 | redundant validation and normalisation | **fixed `8f3d450c`** — the admin set normalised once (`get_admin_emails`, trusting `_read_admin`'s shape); `auth-setup` validates once, through `write_config_scope`, and writes Google's secret after it; the Microsoft tenant's default is the validator's alone |
| C22 | stranded comments | fixed `0a7ea0e4` |
| C23 | the envs report named the wrong source for the host env | Q2c 6, `311ab27e` |
| C24 | conftest re-derives the config-directory rule by hand | **fixed `5ba5ac5f`** — `_REAL_CONFIG_DIR` asks `config_dir()` at import |
| C25 | a second reader of the notify file's format | `_document` not real (§ 2.3's management carve-out); **fixed `5ba5ac5f`** — `_write` writes `persist.json_text`'s bytes |
| R1 | the activation refusal named the wrong record | fixed `92e84cc8` |
| R2 | an unreadable snapshot skipped silently | fixed `92e84cc8` |
| R3 | `--target this` takes another machine's snapshot unasked — and the tab sends `this` on every prep once "(this machine)" is picked | **your word** — below |
| R4 | `envs init-config` is a second, partial prober: on a cluster it seeds `slurm` with no queues and no `detected_at` | **fixed `5157ae90`** — `record.probe_queues`, run by both (and the submit-cap note it carried, dropped since it was written, now shown) |
| R5 | "lists no queues" prints the bare probe command, whatever the target or snapshot | **fixed `4d5b28f8`** — `record.record_and_renewal(base, target)`, the one spelling, shared with the GPU-bench refusal; two `launch_values.toml` rows |
| R6 | the launch re-check's remedy (re-run `prep`) cannot work — prep keeps the snapshot — and names a queue as the machine | **fixed `d04d830c`, `2979e2c1`** — the ask, another queue, or the snapshot deleted; a `launch_values.toml` row |
| R7 | the preamble guard's remedy | fixed `92e84cc8` |
| R8 | the provenance card vanished with a named record | fixed `92e84cc8` (C10) |
| R9 | the web prep route's own copy of the target checks | fixed `92e84cc8` |
| R10 | the record read 3–4 times per prep | the reads go through one door and agree — one source, read again, is not a defect (`One source, not one place`), left; **fixed `1b9ceeea`** — `_no_record`'s unreachable named-target branch |
| R11 | a second road into the wrapper that only tests use | launch's `machine_for` fallback is used, kept; the test-only fallbacks **parked** — every production caller passes the record, so D1's risk (a production default choosing the machine) does not arise; what they serve is 57 direct renderer calls in 13 test files, which move to the road with T15 / T29, and requiring the record first would edit those calls twice |
| R12 | paths and probe commands written by hand | **fixed `4d5b28f8`** (the rest, with R5) — `calculation_record` at prep and summarize; the probe commands through `probe_command` / `record_and_renewal` (the retired-`scheduler` message also said "copied here" for a named target's record, which would have overwritten this machine's) |
| R13 | every record says prep wrote it | **fixed `5157ae90`** — the field's default is `jobset-probe@1` |
| R14 | a value the probe keeps loses its `source` | **your word** — below |
| R15 | `env_init` carried key by key | fixed `92e84cc8` |
| R16 | `UnknownTarget` invited a hand-written record | fixed `0a7ea0e4` |
| R17 | the no-activation notes sent to a command that would not ask | fixed `92e84cc8` |
| R18 | unused parameters | **fixed `1fc16ce4`** — `n_atoms` of `_render_sbatch_for`; `target` of `declared_run_shape`, and of the two that only passed it down |
| R19 | dead browser code | fixed `0a7ea0e4` |
| R20 | comments describing the removed `scheduler` block | fixed `0a7ea0e4` |
| R21 | record docs that were false | fixed `0a7ea0e4` |
| R22 | fields with no writer or no reader | `gpu_partition` Q2c 10a, `31ca9957`; `Site.qos` kept by the contract; **fixed `39ae603a`** — the hand-declared device spelling and `Device.mem_gb` (no writer; a record is a measurement — the environments README), `Site.account`; the arch fields kept (asked for, 2026-08-26) and their notes no longer describe a check that does not exist; routing's getters stay in `runtime_config`, which `architecture.md` names as the door to a record's queues |
| R23 | `probe --out` | Q2c 7, `71e70dcd` |
| R24 | contract restatements against the design | fixed `0a7ea0e4` |
| D1–D20, D22–D30 | documents teaching removed designs | fixed `0a7ea0e4` (D24: `envs.manager` is a preference) |
| D21 | `envs.host` | Q2c 6, `311ab27e` |
| T1, T2 | two tests that could not fail | fixed `6401abfb` |
| T3–T6, T9–T12, T17, T18, T20, T23, T26 | tests of refused behaviour, or repeats | retired `6401abfb` (T4 written through its door) |
| T8 | retired keys only partly rows | fixed `6401abfb` |
| T7 | `test_machine_config_file` cites a precedence no document states; one test cannot fail | **fixed `54156262`** — the two retired, the write test's docstring in its place and true, the sandbox no longer a working directory |
| T13 | test texts still placing the activation in `script_generation` or the record alone | **fixed `876f96e4`** — 22 comments and docstrings: the record carries the probe's copy of `env_init`, which the generator reads (`configuration.md` § 4) |
| T14 | test docstrings teaching a cwd or cascade read | **fixed `876f96e4`** — nine files say what is read; their cwd changes stay as plain isolation (a relative write must not land in the checkout) |
| T15 | per-file isolation duplicating conftest's | **fixed** (`e223d0eb`, and the per-file blocks after it) — one fixture decides a test's config root; 35 files lost a HOME / XDG move that only repeated it, and the record rewrite it forced; kept where the move does work: the files that test those variables (`test_config_dir_has_one_home`, `test_machine_config_file`'s XDG branch, `test_seeding_a_fresh_machine`, `test_config_warnings`), the e2e files that hand a child process its environment, and four whose sandbox is load-bearing (`test_auth_setup`, `test_envs_install`, `test_routing_domains`, `test_task_setup_tab` — red without it); `test_target_machine_choice` now removes this machine's record by name, which is what its HOME move had been doing |
| T16 | "a cwd molbuilder.json is not read" asserted five times | partly fixed `6401abfb`; **fixed `54156262`** the rest — said once, by `test_and_its_contents_do_not_leak_in`, with a section a server starts on |
| T19 | tests that monkeypatch a door to prove a caller asks it | **fixed `02f67712`** — the two retired (review holds that class); the remedy test says why it stays API-level |
| T21 | probable subsumptions | **verified `1f115845`** — `test_the_config_molbuilder_SEEDS_reads` CONFIRMED subsumed (12/12 mutants), retired; the two exact-mode seeding tests NOT subsumed (the placement test survived 4/12, `config_dir.py:125`), kept |
| T22 | `_the_gate()` pins a refusal the road never gives | **fixed `02f67712`** — the baseline is the road's own fact (a fresh machine has no record); the call-site test retired; a docstring's unasserted promise trimmed |
| T24 | the mode warning asserted twice | Q2c 8, `615858f6` |
| T25 | `env_arch` is omitted when unknown; M-2 says null, never omitted | **your word** — below |
| T27 | four tests asserting strings no code has | **fixed `02f67712`** — retired, with the estimator test's helper |
| T28 | hand-built records and paths where a door exists | **fixed `70fa5038`** — the fixtures ask `environments_dir`, `named_environment_path`, `machine_scope_path` and the record's `FILENAME`, and create through `ensure_private_dir` |
| T29 | five tests drive the private `_probe_consent_merge` | **fixed `5d0253b1`** — `tests/data/machine_record.toml`, eight rows down the road (the road runs a probe that answers its questions, checks what it said and the record it wrote, and can end there); four tests retired, the domains-set one kept API-level with its reason (needs a cluster's `sinfo`); the rows also give back `--set`, `--scheduler` and the unreadable-file line, and M-6 now states that line |
| T30 | table hygiene | partly fixed `6401abfb`; **your word** — below |
| S12 | the form's `engine_key` test covers optimization only | **fixed `1f115845`** — one test over engine × calculation (five forms); red on a transport-only gap the old two could not see |
| M5 step 3 | its milestone review | every finding fixed `325cb1d4`; its three questions were Q2c 1–3 |
| Q2c + W54 | their review — a fresh agent over `e1d88b22..d470bff2` (31 commits), the full code text | **fixed 2026-10-02** — `envs init-config` says what its queue probe could not measure, as `jobset probe` does (pinned for both writers); the re-probe remedy for a calculation's own copy says where a named record's file goes; T29's rows give back what the retired tests held — `--set` and `--scheduler` mark their facts `flag` (M-1), an `env_init` corrected by hand survives an unanswered probe and `--yes` takes `molbuilder.json`'s copy, the stamp follows the probe (M-6) — through two runner keys, `record_declared` and `record_stamped`; six mutations, each red. Stale text fixed in five documents, five code comments and seven test files; two fixtures moved to the doors; four leftover home directories gone; this ledger's Q2c, A1.20 and TR6 rows brought up to date. **Dropped:** two providers of one kind sharing one secret file — only a hand-written config makes two (`auth-setup` writes one of each kind) |

**Your word — four questions open, one settled** *(asked 2026-10-02,
re-asked in plain words the same day with the options below; the fifth found
while answering the first four)*:

1. **R3 — settled by the user's rule, built 2026-10-02.** *"when a machine
   is set for a job, it is set, no changing. we have ... persistence designed
   to roll back"*; *"we never ... prep something that has been prepped"*;
   *"consistency check refuse[s] the ... mental problem"*. The calculation's
   copy names its machine (`machine`, written at its first prep); a later
   `--target` is checked by that name — `this` like any other, a re-probe no
   conflict; every remedy that said "delete the copy" says a new prep from a
   saved state (`molbuilder checkpoint restore`). `configuration.md` M-3; two
   `launch_values.toml` rows, three mutations red, 54 targeted files green.
2. **R14 — a kept declared value loses its `flag`.** M-5: `source: flag` admits
   a declared fact; M-6: `source` follows the new probe, so after No keeps a
   `--set` value the note no longer says `flag`. Nothing reads `source`. (a)
   Keep `flag` when a declared value is kept. (b) The note is the latest
   probe's; M-5 reworded. (c) Remove the note. **Proposed (a).**
3. **T25 — `null` or left out.** M-2: an undetected field is `null`, never
   omitted; `to_dict` omits `env_init`, `conda_envs` and `env_arch` when empty
   (the reader takes both alike). (a) Write `null`. (b) Change M-2.
   **Proposed (a).**
4. **T30 — "with nothing to copy, the record keeps its own `env_init`".** Code
   only (`jobset/_cli.py`, `carried`), pinned by a `launch_values.toml` row; it
   arises only when `molbuilder.json` has no `env_init`, which `envs
   init-config` never leaves (`seed_machine_config` adds one). (a) State it in
   `configuration.md` § 4. (b) Drop it: the record's `env_init` is always the
   copy (*"simply a copy"*), and with none to copy the difference is asked
   like any other. **Proposed (b).**
5. **The I–V current of an unpolarized junction.** TBtrans prints its current
   per spin channel (*"no spin degeneracy"*, `m_tbt_save.F90`: I = (e/h)∫T),
   so an unpolarized run's figure is half the physical current — while the
   conductance column is in G₀ = 2e²/h (`engines/transport.md`'s opening).
   The record keeps the printed figure, *"parsed, never recomputed"*
   (`job-contracts.md` § 6.1). (a) The total — ×2 unpolarized, the channels'
   sum polarized — with the printed figure beside it and the factor stated.
   (b) The printed figure, labelled per spin channel. **Proposed (a).** K21
   waits on it — both change that column.

**Answered 2026-10-03** (§ 0c, *the user's word*): 2 — (a); 3 — (b), and the
probe says what it leaves out and why; 4 — neither: the probe refuses without
`env_init`; 5 — (a), every surface saying what is what. Built as § 0c's work
order units 1–3 and 7.

### 0c. W55 — the ledger's done rows

> **Verdict (2026-10-08 validation):** § 0c rows — D1 CONFIRMED superseded (`_as_it_was` gone; road keys `reprobed`, `ledger_lacks`); D2 (`commands.py:191`; `submit.py:193`; `runtime_config.py:663`; `running-a-job.md:904-908`), D3 (`prep.py:2394`), D4 (name superseded), D5 (`runfiles.py:383`), D6 (`ask.py:302/316`; `ledger.py:48`), D8 (`validation/__init__.py:305`), D10, D11–D12 (`build.py:2076,2150`), D13, D14, D16, D18 (`materialize.py:552/571`), D19 (pins retired), D20 (`workingcopy_structure.py:316`), D21 (`prep.py:115-118`), D22–D23 (`runfiles.py:846,1650,1711/1738`; `tools/manifest.py`), D24, D25 (`runs.py:213`), D26 (`siesta/warm-files.toml:92`), D27, D28 (`jobset/plan.py:50-58`), D30 CONFIRMED; D7, D15 UNVERIFIED; D17 CONFIRMED, name stale (`placement.admitted`, `placement.py:182-236`; `admission_refusal` 0 hits); B1, B3, B6, B12, B13, B14 CONFIRMED done; B2, B4, B5, B7, B8, B9, B10 REFUTED stale *proposed* → all done; B11 PARTLY stale → 3b done; Q1 (0c) OBSOLETE (unit 5 → 12b-1). The *checked and not kept / part of B1* paragraph goes with B1.

| # | what | found by | state |
|---|---|---|---|
| D1 | a prep refused after it began writing leaves the calculation set to its machine (`environment.json` is not put back), `STAGE-PLAN.md` describing a stage the plan does not hold, and a `prepped` ledger line before the refusal — so a retry naming another machine is refused as *set to* the first | prep #1, #2 | **done 2026-10-03** — a refused prep puts back the plan, its `STAGE-PLAN.md` and the calculation's copy of the record (`prep._as_it_was`), and records *prepped* only when it finished; § 5.0 rule 3 says so. The GPU-build refusal's remedy (install, re-probe, prep again) is taken now — before, the stale copy refused it again; it no longer sends a first prep to a rollback. Row in `prep_protocol.toml` (`reprobed`, `ledger_lacks`); two mutations, each red |
| D2 | texts sending a person to steps now refused or retired: the re-launch note's `prep run … first`, *re-prep the bench* / *prep the stage again* (`agreement.py`), `_NO_SBATCH`'s *prep it for the machine with the queue*, a flat trial's *move the directory aside*, the deck header's `execution.mode` (renamed `launch.mode`), and the page's and the CLI's *sized from the target's own width* / *the scheduler's own default decides* / `auto` / *inherit defaults* | doors #2, launch #1, structure #11 | **done 2026-10-03** — each remedy names the rollback (`commands.rollback` where the folder is known; the same words where it is not): the re-launch note, the deck/launch disagreement, the cold trial (both), a flat trial measured again, and the missing scheduler header (`_no_sbatch`, now naming that a calculation is set to its first prep's machine); the deck header's note says `launch.mode`; the page and the CLI say a launch value stated nowhere is refused, and print only what is stated (`resources:` — `mpi_np auto` claimed a rank count a PySCF run does not have); launch's two *not stated* warnings deleted — its request refuses both values, and the memory one fired, falsely, for every `--exclusive` job; `Resources`, `job-system.md` (`Job.resources`, the `notes` row) and the summarize text with its copy in `project-layout.md` brought to § 5.2. The resources test rewritten (one mutation, red); 504 targeted green |
| D3 | the Task setup editor can change the shape of a calculation that has prepped stages; `task-setup.md` § 4 and `job-system.md` § 5.0 say it is fixed once produced | doors #4 | **done 2026-10-03** — Save refuses another shape once a stage is prepped, as a run or a benchmark (`prep.prepped_stages`, the prep entry's own `prepped_already`), naming the way back; a refused prep counts for none. `task-setup.md` § 4 and `web-api.md` say so. Three rows in `prep_protocol.toml` (road key `saved` / `save_refused`); three mutations, each red; 220 targeted green |
| D4 | `launch run <stage>` on a stage described and not prepped says *no stage named … in this job-set*; `status` says to prep it | doors #7 | **done 2026-10-03** — launch resolves the stage by the description, then refuses one not prepped by name, with its prep and its launch (`_refuse_unprepped`, the prep entry's `prepped_already`); § 5.0 *After prep* says so. Row in `prep_protocol.toml` (road key `launch_stage`; `launch_refused` takes a list like `refused`); one mutation, red; 563 targeted green |
| D5 | a flat stage launched again records the stage before it as what it continued from (prep's marker, not its own run) | launch #2 | **done 2026-10-03** — launched again, a flat stage's marker names its own latest run (`runfiles.run_name`), written where the hierarchy's next attempt writes its own: at the launch. The flat record test extended; one mutation, red; 166 targeted green |
| D6 | launch's question and its answer are not in the ledger — a *no* leaves no line (§ 5.0 agreement 6) | launch #3 | **done 2026-10-03** — `ask.confirm` returns the answer with its words (`Said`), and launch writes each question it asks and its answer (`question`, a *no* too) at all three of its questions; `ledger.py` names it. The launch-door test reads both answers; one mutation, red; 247 targeted green |
| D7 | launch's queue table is built from the launch flags alone, while the door reads the wall and memory prep baked too — a queue shown as fitting is refused | launch #5 | **done 2026-10-03** — the table asks the wall and memory the door admits: launch's flags, else what prep baked, the most any job being sent asks. Row in `launch_values.toml`; one mutation, red; 156 targeted green |
| D8 | prep swaps the process-wide `sys.stderr` while it renders decks; two Task setup preps at once leave the server's stderr de-duplicating | prep #9 | **done 2026-10-03** — the gate's report reads its stream from a context variable (`validation.REPORT_STREAM`), which prep sets for its rendering loop; `sys.stderr` is never swapped. A two-thread test holds both preps inside the loop; one mutation (the swap back), red; 405 targeted green |
| D11 | the Task setup page's launch line carries no `--mode`: on a machine whose `molbuilder.json` sets no `launch.mode` the terminal prints one line per mode, while the page's one line is refused when typed (`job-system.md` § 5.3, *what molbuilder prints, you can type*) -- the page composes it (`viewer.js`) beside the one composer (`commands.launch_lines`) | the explanation round, 2026-10-03 | **done 2026-10-03** with B4 (unit 8): the page's lines are the composer's |
| D12 | the Task setup page offers a machine for a calculation already set to one -- `/api/task-setup/machines` takes no folder -- and Prep refuses any other (`configuration.md` M-3) | the explanation round, 2026-10-03 | **done 2026-10-03** with B4 (unit 8): the folder answer's `set_to`, the card shows it fixed |
| D13 | Task setup's queue card writes the queue's ceiling as the wall and 95 % of its memory into the description when a queue is chosen (`viewer.js` `setQueue`) — saved, they read as stated: S1/S2 (*"unanswered is refused, never a default wearing a number's clothes"*) | the framework inventory, 2026-10-03 | **done 2026-10-05** (unit 11a): choosing a queue names it and fills nothing; its limits are shown under each field |
| D14 | launch: `--dry-run` writes ledger lines though it says it writes nothing; the refusals raised before the send are not in the ledger (§ 5.0 agreement 6); a direct run's `launched` line is written after the run ends | the framework inventory | **done 2026-10-05** (unit 11b): a dry run writes nothing; every refusal is written down, the entry's by the entry and the verb's own by the verb; each submission as it goes, a run here when it starts |
| D15 | the Task setup prep door refuses `#N` and a stage name in another case, which the entry takes (one prep, two doors) | the framework inventory | **done 2026-10-05** (unit 10e) -- the door hands the stage to the entry |
| D16 | a refused prep leaves decks, wrappers, `pseudos/`, `atom-permutation.json`, the compose record and an opened attempt behind; `engines/stages.md` § 7.2 says the produce is all-or-nothing, `job-system.md` § 5.0 that only the plan is put back, § 5.4 relies on the deck being left | the framework inventory | **done 2026-10-05** (unit 10b) -- a refused prep writes nothing but its ledger line; `stages.md` § 7.2 says how |
| D17 | a GPU count asked of a CPU run (`--gpus N`, `use_gpu` false) reaches the header and the queue as a GPU job while the deck and the run script run on the CPU; a GPU run naming a queue with no GPUs gets a header and is refused only at launch (`gpu.md` G4) | the framework inventory | **first half done** with unit 9b part 4 (2026-10-03): a count for a run that does not use the GPU is refused at prep, by the run card or `--gpus`; the second half -- the queue's GPUs checked before the header is written -- **done 2026-10-05** (unit 10's first review): prep admits the run's whole request on the queue it names at checkpoint 4 (`placement.admission_refusal`) |
| D18 | trial folders, bench folders, `launch/` folders and a stage that only has a benchmark carry no `calcdir.json` (invariant 6b); an attempt opened at launch and a transport attempt get no progress seed (`project-layout.md` § 1.6.3) | the framework inventory | **done 2026-10-07** (unit 12a): trial and bench folders were stamped since 3b.4a; `launch/` now by `open_container`; a transport attempt is seeded at prep (unit 10c); an attempt a launch opens is not seeded -- its progress is its engine's own (§ 1.6.2) |
| D19 | `xv2xyz --from-run` on a FLAT calculation's `.XV` keeps none of what the run declared -- measured 2026-10-04 on a real run of ours (`tests/fixtures/siesta_flat_h2`): it reports *"no sidecar and no constraints declared"*, drops the held atom both decks state (`position 1`) and writes the axes periodic for an isolated molecule.  The `.XV` is named by `SystemLabel` (`H2.XV`); the structure pair the run was prepped from is named by its file (`h2.source.molstruct.json`); and the deck lookup meets two decks and, rightly, guesses neither | the coverage measurement after the 2026-10-03 retirement | **awaiting a ruling** -- proposed: an `.XV` of ours takes its declarations from the run that wrote it, through the run record (the deck it read, the structure it was prepped from), never from names; **ruled with B12, 2026-10-04**: `xv2xyz --from-run` takes what the run declared from its own deck, through the run door -- the composer `model/structure-periodicity.md` § 6.0 already names for every read of an engine's own structure file *(user: "we will rule D19 after your report and consolidated analysis. they are related"; then B12 approved)*; **done 2026-10-04** (W56 4a; user: "go ahead") -- `runs.declared(run)` reads the run's own deck; `xv2xyz --from-run` finds the run holding the `.XV` (`run_of`) and takes its labels, held atoms and axis kinds from it, on the `.XV`'s cell and the engine's origin; a `.XV` no run of ours holds is refused. Pinned on the measured flat H2 (held atom 0, isolated axes, 10 Å cell, the deck named) and by the refusal -- three mutations red; the three hand-laid `--from-run` tests retired |
| D20 | `jobset init` names the structure pair after the structure FILE -- `describe.py:246`, `f"{src.stem}.source{src.suffix}"`: `h2.source.xyz` beside the label `H2`, measured -- where job-contracts § 6.3 and the hand-over (`build.py:1174`, `f"{label}.source"`) name it after the label; the catalogue and the Task setup card name the label's spelling, a file that is not there; a `.pdb` source travels as `<stem>.source.pdb.xyz` while `task.json` records `<stem>.source.pdb` (code text) | the file manifest, 2026-10-04 | **done 2026-10-04** (W56 unit 1) -- `StructureCodec.source_files(struct, label)`, the one call the hand-over and init make; init writes through the codec always (the raw copy, a second way to write one file, gone), and `task.json` records the name written. The CLI road test reads `JOB.source.xyz` beside a structure file `h2.xyz`, red with the file's stem back; its two API halves retired |
| D21 | a later prep renders every earlier stage's run script and monitor again -- prep renders the MERGED plan (`prep.py:1062` into `prep_jobset`, which loops every job, `:186-276`). Measured on both H₂ calculations: medium's prep rewrote `01_coarse/H2_01_coarse.run.sh` and `mb_monitor.pyz`, so the attempt's copy, which ran, differs from it; in the flat shape it rewrote the launched stage's own run script, in its run folder, after it ran -- against `project-layout.md` § 1.5 (*an attempt is immutable*), § 2.6 (*nothing ever writes there again*), § 1.6.2 (*a stage is prepared once*) | the file manifest | **done 2026-10-04** (W56 unit 1) -- `prep_jobset(..., render=)` renders the run scripts of the stage it preps (a benchmark's plan, every trial); two `prep_protocol.toml` rows, both shapes (road keys `kept`, `shape`), red against the old render |
| D22 | the Task setup card's file list (`runfiles.manifest`, which takes no shape) shows, for a hierarchical SIESTA calculation on a workstation, seven names no run writes: `<base>.run.json` and `<base>.continued-from` (the flat spellings -- the hierarchy writes `run-<n>/run.json`, `run-<n>/.continued-from`), `<base>.log` (row 14, PySCF's logger, names no engine), `<base>.parse.log` (really `<base>-run<N>.parse.log`, and only with `MOLBUILDER_PARSE_LOG`), `<base>.molwatch.parse.log` (the same switch), `<base>.transport.parse.log` (no writer), `<base>.sbatch` (a machine with a queue only); `WRITTEN` says *run* for `.molwatch.log` and `.continued-from`, which prep writes -- against job-contracts § 2.2a (*a card cannot show a spelling the writers do not use*) | the file manifest | **done 2026-10-04** with B13 (W56 unit 2) |
| D23 | `WRITTEN` -- *"the list that can be complete"* (job-contracts § 2.2) -- has no row for files molbuilder writes into a run folder: PySCF's `<label>_initial.molstruct.json` and `<label>_optimized.molstruct.json` (`pyscf/input.py:1629-1631`), `molbuilder runtime-info`'s `<stem>.runtime_info.json` (`cli.py:1812`), the hierarchy's `run.json` and `.continued-from`, `.gathered-from`, `calcdir.json`, `mb_monitor.pyz` / `mb_vibration.pyz`, `makov_payne_correction.py`, `atom-permutation.json`; so `--cold`'s exceptions (`patterns()`) name the PySCF pair and the runtime-info file as engine state it would overwrite (code text), and the card keeps a second, hand-made list of the calculation's own files (`build.py:2169`) | the file manifest | **done 2026-10-04** with B13 (W56 unit 2) |
| D24 | a name read without its label is taken for one of ours: the Results listing gives SIESTA's `fdf.<stamp>.log` and the ledger the role `.log` -- PySCF's logger's row -- because it falls back to `runfiles.role_of` (`results.py:280`), which reads any name ending in a declared dotted role (`slurm.<id>.out` is `.out`, `junction.cited.fdf` `.fdf`, `bench-group.run.sh` `.run.sh`), measured; `runfiles.find` returns the engine's `H2.XV` though its docstring leaves out what is not ours -- against job-contracts § 2.2a (*a name is read back with its label*) | the file manifest | **the listing's half done 2026-10-04** (W56 unit 3a) -- `runs.folder_answer` reads each name back with the run's label, to a declared role or to none (no `role_of`); pinned on the measured flat H2 (`siesta_flat_h2`: SIESTA's `fdf.<stamp>.log`, `H2.XV`, `H2.MD.nc` listed with no role, not ours), both mutations red. **`runfiles.find`'s half done 2026-10-04** (W56 3b.6; user: "go ahead") -- read against the contract, its answer is right: the label's files, the engine's own included, found by name as § 4.2 asks (§ 2.2a's carried files); the carry composes its names and never asks it, and every caller narrows by role, stage or run index, so none takes an engine file for one of ours. Its docstring said "our files": corrected, with job-contracts § 2.2a's doors table |
| D25 | the run record's *is this still the stage's deck* looks for a flat stage's deck outside the calculation (`record.py:486`, `f.directory.parent / f.deck.name` is the topic folder), so a flat run never states it | the file manifest | **done 2026-10-04** (W56 unit 3a) -- the run door places the stage's deck (`runs.Run.stage_deck`): the stage's folder in the hierarchy; a flat run's deck IS its stage's, one file, so a flat record states no `current` (`model/parse.md` § 5d.4) |
| D26 | `siesta/warm-files.toml` spells `.Bonds`; SIESTA writes `<label>.BONDS` and `.BONDS_FINAL` (measured), so the run script's banner and `--cold` never see them | the file manifest | **done 2026-10-04** (W56 unit 1) |
| D27 | `--cold`'s exceptions still name `*-restart-aside-*` (`runwrap.py:715`), a folder nothing has written since 2026-08-18, kept for *"folders written before the change"* (job-contracts § 6.3's Directories row) -- old runs are not a design input (user, 2026-10-03) | the file manifest | **done 2026-10-04** (W56 unit 1) -- the exception and § 6.3's reservation gone |
| D28 | `STAGE-PLAN.md`'s restart-file column prints what a stage declares, not what the carry took: medium lists `H2.CG`, which the carry withheld (a CG stage before a Broyden one; measured, the ledger's `copied` and `02_medium/run-0/`) | the file manifest | **done 2026-10-05** (unit 10d) -- the column headed as declared, this prep's carry under the table |
| D30 | I4's check of what travels to the compute node (`test_checkpoint_wrapper_isolation.py::test_no_emitted_wrapper_invokes_git`) refuses a module of the monitor's bundle that mentions git as a WORD; the catalogue -- `runfiles.py`, which the bundle ships because the monitor reads run names through it -- gained the checkpoint store's rows in W56 unit 2 (`.git/`, and `.gitignore`'s *rather than in git*), so the check is red on all four wrapper shapes since `6ee9d1b2`.  The contract's I4 forbids a git COMMAND in a wrapper (`checkpointing.md` I4), and `running-a-job.md` § 2 a node that needs anything but the folder's files; a row naming the `.git/` folder calls nothing | the unit 3a check (the test was outside unit 2's targeted set) | **retired 2026-10-04, with I4** *(user: "we have ... included git in env ... long time ago why do you still allow this obsolete ... test exist?")* -- I4's premise, a compute node without git, is false: every env molbuilder installs has declared `git` since 2026-06-25 (`envs/recipes.py`, `ops/env-framework.md`). The test file went whole: its two git checks with the rule, its render-equals-write check (`write_run_wrapper` writes `render_wrappers`' text) and its `bash -n` check (four other files hold one) with nothing left to guard. `checkpointing.md` (§ 9's line, I4, both tables), `env-framework.md`, `engines/stages.md` § 7, `checkpoint.py` and the README's count made to agree. **The miss, in its own words**: the test was trusted and the rule never read -- I offered to narrow it, where checking its premise against the code took minutes. **`envs/recipes.py` reviewed whole the same day** against `ops/env-framework.md` and the code (user: "holistically review recipies ... fully in compliance with design/contract after the edit"): its I4 comment and every stale or false comment fixed, `_RPATH_ELPA` (referenced nowhere) and `numactl` in the host and PySCF envs (nothing in either uses it) removed, and `tests/test_envs_readme_consistency.py` retired (user: "your useless test wastes both cpu time and my ... time. retire it") -- and then the duplication itself went: the host env's package list is one file, `molbuilder/envs/host-env.txt`, read by the shim and by `recipes.py` (`env-framework.md` § 3.4; user: "both bash and recipes.py can read one ... well formatted file that has one ... list"); `engines/stages.md` § 7.3 restated § 9 from before its 2026-10-03 ruling -- a checkpoint asked at an interactive prep and never taken, a second one tagged when a stage's run finished, a note never generated, a drafted note at a stage's end; the code (`checkpoint.save_before`, the sidebar's typed note) follows § 9, so § 7.3 now points to § 9 and L3/L4 instead (2026-10-04) |
| D10 | refusals that need no write come after the save offer: the activation, the structure witness, the flat sweep, the vibration rung's relax checks, transport's disabled rung, `--from` on a stage with nothing to carry — against § 5.0 rule 3 | prep #1, #6 | **done 2026-10-05** with B1 (unit 10b) -- every refusal comes from the plan, before the save |
| B1 | **prep in two halves**: checkpoints 1–5 build one plan that writes nothing — the description, the stage, the shape, the machine record, the resolved ladder, the structure, the continuation and what it carries, the folded allocation — and the writing half takes it and decides nothing; each file read once, the allocation folded once, one `Continuation` from 4a to the attempt | prep #1, #4–#7, structure #2–#5 | designed 2026-10-03 (unit 10) -- **the two halves done 2026-10-05** (10b), each file read once and the stage resolved once the same day; one `Continuation` done the same day (10d); the card's own fold the same day (10e) |
| B2 | **one transport arm**: `_prep_transport` -- one function for the five rungs -- is a second copy of the five steps' prologue and epilogue (the machine record, the log, the allocation, the job, the merge, the wrappers, the attempts; parts verbatim, comments included), and has drifted from the first: no progress seed, no once-per-line filter, no config provenance in its log, its job built without `_job_for`. The steps once; transport supplies only compose and its per-rung structure *(re-read 2026-10-03)* | prep #3 | proposed |
| B3 | **the preview is the entry**: `prep_stage(…, preview=True)` answers at checkpoint 5, and the route's own assembly (`_plan_chosen`, `_emitted_launch`, `_plan_continuation`, `_plan_prepped`) and the run card's fit check through the bench grid go | doors #1, #10, structure #8, prep #8 | **done 2026-10-05** (unit 10e) -- Prep names the preview's plan (`Plan.identity`) |
| B4 | **the page composes nothing**: the folder answer carries the calculation's machine, its prepped stages and the commands; the prep answer carries the machine, what the stage starts from and the next lines | doors #3, #5, #8 | proposed |
| B5 | **one launch entry like prep's**: a typed launch plan, shown then sent; one `sbatch` call and one sequencer; a re-launch's continuation as a `Continuation`, ledgered | launch #3, #6, #7, structure #6 | proposed |
| B6 | **one opener for attempts**, trials included, stamping `calcdir.json` and seeding the progress channel | launch #4 | **done 2026-10-07** (unit 12a) -- `materialize.open_run` makes and stamps every run folder (a stage's attempt, a bias point's, a trial's) with the containers above it; `open_container` a submission's `launch/` and `pseudos/`; prep seeds a stage's first run, a later run's progress is its engine's own |
| B7 | **layering**: an empty `jobset/__init__`; `token_for` to the description; the engine seam, `PrepError` and the record read to modules of their own — nothing below the conductor imports it | prep #10, launch #9, structure #1 | proposed |
| B8 | **one door per fact**: prepped, launched, concluded, the warm files, the GPU side, `job-set.json`'s reader, `run.json` through `persist`, a stage's home | structure #7, #10, launch #8, doors #8a | proposed |
| B9 | **dead code**: routes no page calls (`/resolved`, `/attempts`, `/template-values` -- only tests call them, which move to `/folder` or retire), what only re-prep reached (root-deck adoption, the merge's same-name replacement), `launched_trials` (no caller), `materialize`'s re-exports, a prep comment naming a `prep_calculation` call `build.py` no longer makes. **Attempt reuse is not dead** *(re-read 2026-10-03)*: a hierarchical benchmark reaches it within one prep, and a retry after a refused transport prep reuses the attempt the refusal left | doors #11, prep, structure #12 | proposed |
| B10 | **one save offer** for Task setup's Save and for prep | doors #6 | proposed |
| B11 | **one door for the run a file belongs to, and its files** -- `run_of(path)`: its calculation (`calcdirs.root_of`), its description (`read_task`), the label the description gives it (a trial's own, `<label>-<point>`), its stage, its run index, its folder; and the run's files by role -- the deck, the stdout, the marker, the launch record, the session log, the monitor's, the timing, the progress log. It retires the second readers the manifest marks: labels guessed from decks (`rundir.labels_in`, `read_back`), `record.run_files`' own search, `materialize.stage_stdout`, `summarize._latest_run_file`, `_wrapper_log` and `_trial_deck`, `transport.record._outs_newest_first`, `runstatus._label_of`, `runfiles.label_of_run_file`, the second stage grammar (`identity.parse_stage_token`, `job._detect_stage`, the PySCF parser's `_STAGE_TOKEN_RE`, `transport/deck.py`'s split), `_sidecar._siesta_fdf_path_for`, the Results listing's `role_of` (D24) -- and gives *which run a folder speaks for* one rule, `run_status`'s (the stage, then time: `parse.md` § 5.1), where `openable_in` takes the newest file and four readers the highest run index | the file manifest, 2026-10-04 | **approved 2026-10-04** *(user: "agree to your B11, B12 and B13, and B14 should be addressed by unified api call")* -- designed into the contracts (`execution/architecture.md` § 3, § 3.2; `job-contracts.md` § 2.2a; `model/parse.md` § 5.1–§ 5.3; `web/results.md` § 2.3); W56 unit 3 -- **3a done 2026-10-04**: the run door, `molbuilder/runs.py`, and the directory door moved up into it; the deck-label search retired (§ W56); **3b open**: the second readers it names from `materialize.stage_stdout` on |
| B12 | **one door for what a run of ours declared about its atoms** -- the held atoms, the regions, the axis kinds, the cell: the run's own deck (B11) and its `atom-metadata` and `engine-offset` records; the calculation's structure pair only through `task.json`; never a sidecar looked for beside an output. It retires `_sidecar.read_frozen_atoms`' name candidates and lone-sidecar guess (both shapes miss `h2.source.molstruct.json`, measured), `read_frozen_atoms_from_siesta_fdf`'s deck guess, the first-deck-wins of `atom_metadata_json_for_run_dir` and `engine_offset_record_for_run_dir`, `engine_frame_for_run_dir`'s `.source.xyz` search, and `xv2xyz --from-run`'s sidecar beside the `.XV`. **D19 is its first reader**: `model/structure-periodicity.md` § 6.0 already says every read of an engine's own structure file takes its run's frame from that composer, and `xv2xyz` is the one door that does not | the file manifest | **approved 2026-10-04**, D19 with it -- designed (`architecture.md` § 3.2, `model/parse.md` § 5.3, `structure-periodicity.md` § 6.0); W56 unit 4 |
| B13 | **the catalogue complete, and shaped** -- `WRITTEN` declares every file molbuilder writes into a calculation folder (D23), each row with its spelling in each shape, its level and its moment; `manifest(..., shape)`; the Task setup card's own list asks it; `patterns()` and `--cold` follow; `project-layout.md` § 5 is its reading | the file manifest | **approved 2026-10-04**; the catalogue is the ONE source of each file's row and `project-layout.md` § 5 is rendered from it, a check failing when they differ *(user: "Generate from catalogue")*; its rows also answer the Results tab's file card *(user, 2026-10-04: "the result tab could also benefit from this ... if it is generated by the engine, then the UI would just say this is not part of the file generated by molbuilder"; the card follows the dropdown and a sidebar click inside the panel's folder)* -- designed (`job-contracts.md` § 2.2, `project-layout.md` § 5, `web/task-setup.md` § 7.2, `web/results.md` § 3b, `web/web-api.md`); **built 2026-10-04**, W56 unit 2 |
| B14 | **the description through its door** -- `calcdirs.container_or_run` (the shape), `rundir._calculation_of` (the kind) and Task setup's folder answer read `task.json` raw beside `read_task` | the file manifest | **approved 2026-10-04** -- through one API call, `read_task` *(user: "B14 should be addressed by unified api call")*; designed (`architecture.md` § 3.2, `project-layout.md` § 1.4a); **done 2026-10-04** (W56 unit 3a) -- `calcdirs` reads no description (`container_or_run` gone); `runs.place_of` answers a root's place from the `Task` `read_task` returns; the Results folder answer and Task setup's folder answer read the `Task` |
| Q1 | a disabled stage named at prep: the ladder's arm preps it, transport's refuses it | prep #13 | the user's word |

*Checked and not kept:* a flat stage's `run.json` missed by transport's
compose (structure #7) — a transport calculation is hierarchical by rule.
*Checked, and part of B1:* the vibration rung takes a failed `relax` that
concluded and left coordinates — `engines/vibration.md` § 5.2a asks only
*concluded*, while an independent ladder refuses a failed run (`job-system.md`
§ 5.4); one `Continuation` gives both kinds one rule for a usable run.

### 0c. W55 — rows kept in the trim, as they were

> **Verdict (2026-10-08 validation):** rows kept in the trimmed § 0a with an annotated or rewritten state — their text as it was. D9 PARTLY (no document named by the validation); Q6 OPEN (`compose.py:618-660`'s swap still rewrites the cited files = § 5x C1). D29 PARTLY, kept verbatim in the trim (its open document list).

| # | what | found by | state |
|---|---|---|---|
| D9 | documents stating retired rules: `workflow.md` § 5 (step 5 *links*), `project-layout.md` (*delete it and re-prep*; re-prep as the way to regenerate), `preparing-for-another-machine.md` (*delete the environment.json*; *no machine name anywhere in the record*), `running-a-job.md` (the web generating decks), `job-system.md` § 5.3's stale lines, `docs/architecture.md`'s jobset row (`plan`, no `prep_stage`), `execution/architecture.md` § 2.1 (`relink`) | doors #12, prep, structure #12 | **done 2026-10-03** — each brought to the code: step 5 opens the attempt and copies (`workflow.md` § 5, and `script-preparation.md` § 5 and § 6, whose worked example also had step 1 probing and the decks at the root); a redo is a rollback (`project-layout.md` § 5, § 5.1; `template.md`'s two lines resting on it); a re-probe reaches a prepped calculation through a new prep, and the copy names its machine (`preparing-for-another-machine.md` § 3, § 4); the route is prep for its machine, copy, launch (`running-a-job.md`); the jobset row names `prep_stage` and the verbs that exist; `relink` gone. § 5.3's stale lines went with D2. 147 document tests green |
| Q6 | the Transport tab's electrode swap rewrites a cited relaxation's own deck, or its sidecar, inside that run's finished attempt, and leaves `<sidecar>.lock` there (`transport/compose.py:682-790`, `sidecars/molstruct.py:487-509`) -- `project-layout.md` § 1.5 says an attempt is never modified; `job-contracts.md` § 6.1 (slot provenance) names *"the file the label rename rewrites"* | the file manifest | **ruled 2026-10-04: (a)** *(user: "for Q6, i would agree to a")* -- the swap is the transport calculation's own statement (the junction slot's `swap_electrodes: true`), applied by compose to its own copy of the junction; a cited run is never written. Designed (`engines/transport.md` § 4, `web/web-api.md`); built with the transport units (Q5) |

> **Verdict (2026-10-08 validation):** W56 units 1, 2, 3a, 3b, 4 CONFIRMED done; names since: `Shape.run_basename` gone → `runfiles.RunNames`; `materialize.mark_run` gone → `open_run` (12a); `tests/fixtures/siesta_flat_h2` retired. Q6's row is ruled (its open half in § 0c's Q6); D28 done with unit 10d.

#### W56 -- the file manifest's doors, in order *(2026-10-04)*

Every row above from D20 on, as units -- one commit each, the contract already
written (2026-10-04), then the code, then rows of the case tables on the road,
each broken once; an agent reads the full code text when unit 5 closes.

| # | unit | what it is | state |
|---|---|---|---|
| 1 | D20 · D21 · D26 · D27 | `jobset init` writes the structure pair through the hand-over's own call, named for the label; prep renders the run scripts of the stage it preps, never an earlier stage's; `.BONDS`; `--cold`'s retired aside exception and § 6.3's reservation go | **done 2026-10-04** -- 2275 targeted green; two mutations red |
| 2 | B13 · D22 · D23 | the catalogue declares every file molbuilder writes into a calculation folder -- fixed names, each shape's spelling, its level, writer, door and kind; owners take fixed names from it; `manifest(..., shape)`; the Task setup card asks it alone; `tools/manifest.py` renders `project-layout.md` § 5 and a check fails when they differ | **done 2026-10-04** -- 63 rows (32 on the label), `fixed`, `row_for`, `ON_THE_LABEL`; the fixed names' owners import them (one spelling each; `atom-permutation.json`, `bench-result.json`, the Makov–Payne script and the launch folder were spelled twice); the pipeline log composed through its row; the card's per-rung names in the calculation's shape, its whole-run list the catalogue's, each conditional file with its condition; `tools/manifest.py` renders § 5, a door that does not exist refused. **Road check** `the_catalogue.toml` (road key `card_written`): every file the card names for a launched stage and the next prepped one is on disk, both shapes -- it found the SCF timing file conditional on the engine's first SCF row; two mutations red. 2889 targeted green. Reordered: B14 rides with B11, whose door answers a root's place from the description |
| 3a | B11 (the door) · B14 · D24 (the listing) · D25 · the Results file card | `run_of` -> `Run` and `about`: the label from the description, files read back with it; one rule for the run a folder speaks for; `task.json` read only by `read_task` -- `calcdirs` reads no description, the run door answers a root's place from the `Task`, the Results folder answer and Task setup's folder answer read the `Task`; `files[].about` and the card (`web/results.md` § 3b) | **done 2026-10-04** -- `molbuilder/runs.py` (floor 2): `place_of`, `speaking`, `run_of` -> `Run`, `about`, `run_answer`, `openable`, `folder_answer`; the directory door moved up into it (`JobDirParser`, `parse_dir`, the `DirParser` ABC, `RunDirResult` gone; `parse.dirs.rundir` keeps the floor-1 `openable_in` and `run_state_of`), and the deck-label search with it (`labels_in`, `read_back`); the watch, spectrum and Results loads ask the run door; the file card (`lib/results/file-card.js`). **Tests**: the road test of a calculation's folders (a root is a container with its ladder, a prepped attempt pending, each file's `about`); the measured flat H2 (the folder speaks for the stage that ran beside a second deck, its `-run0.out` opens, the engine's files listed as not ours); the flat launch-record test preps a second stage beside a queued one -- the speaking rule's launch-record clause was unpinned until then (mutation green, now red). Retired: hand-laid run folders (four `JobDirParser` tests, the viewer-load metadata classes and two structure-info tests -- their B12 half is unit 4's road rows -- and the two-jobs picker test: two jobs never share a folder). 5502 targeted green, four red -- unit 2's, D30 -- three mutations red. **Not pinned**: for a run of ours, the engine's output opening BEFORE the seeded progress log -- the measured fixture holds no seeded log (and its structure pair is pre-D20, `h2.source.*`): unit 4 re-measures it. **e2e, not run**: `test_trajectory_from_a_real_run_e2e.py::test_the_scf_line_states_the_runs_own_rate` runs its deck in a bare folder, which since B11 holds no run of ours -- its fixture needs the road. **The card, on the dev server** (a real SIESTA relaxation, `projects/PDT/.../01_coarse/run-0`): the dropdown's file and a sidebar click inside the panel's folder each show the catalogue's line or *not written by molbuilder*; a click in another folder leaves it; the mounted viewer stays. The sidebar half followed nothing until the card subscribed at DOMContentLoaded -- the sidebar's module loads after it; and the SCF timing row's line, which named only a TranSIESTA device, was corrected (`35102629`) |
| 3b | B11 (the second readers, the engine's label) · D24 (`find`) | **in steps, one commit each** (read in full 2026-10-04): **3b.1 one stage grammar** (**done 2026-10-04**, `302008cc`) -- `identity.parse_stage_token` retired: `paths.stages_in` and `materialize.stage_refs` / `_trial_stage_token` read a name back with its label (`runfiles.parse`), the run record's setup takes the stage the run door gave it (`RunFiles.stage`), `transport.deck.rung_of` asks `identity.parse_token`; **3b.2 the speaking file by its run index** (`model/parse.md` § 5.1, already written) -- **done 2026-10-04**: road case `the_speaking_run.toml` (keys `touched`, `speaks`; red with file time back), `Shape.run_basename`, `identity.parse_stage_token`, `runfiles.label_of_run_file` gone, an attempt holding several run indexes by our own warm retry (`runwrap`'s `--continue`) -- `run_status` orders by run index, never file time, asked with its run's stem by every caller (`Shape.run_basename` is `None` in the hierarchy, where an attempt can hold `-run0` and `-run1`); `runfiles.label_of_run_file`, the stem guessed off a name, retired from `at_latest_run`, `ending`, `_stderr_of`; `runs.run_answer` answers `null` for a file of no run of ours (`web/results.md` § 4.1 -- 3a answered for the folder); **3b.3 one trial label** (**done 2026-10-04**; user: "unification is the right direction. make sure your code is a unified entry for such task for all related work") -- `paths.trial_label(label, point)`, beside `trial_name` / `trial_point`, is the one composer: prep's element label (`resolve._label_for`), the run door's reading of a trial's folder (`runs._position`), a trial's deck read back for its token (`materialize._trial_stage_token`) and a status row's job (`runstatus`, whose `_label_of` -- the label cut off the deck's name -- is gone) all call it; inventoried whole, nothing else composes one (`summarize` reads the script path the job set records; `submit`'s tokens name a shelf, not a label); `project-layout.md` § 2.3.2 and § 4.5, `architecture.md` § 3.2 name it. 6203 targeted green. **Not pinned by a test, before or after**: a trial's status read under its own label (a status reading the calculation's label for a sweep stays green) -- the road preps a benchmark but does not launch one; the one composer keeps writer and readers in line by construction; **3b.4a a benchmark trial's folders say what they are** (**done 2026-10-04**, found reading 3b.4's readers) -- measured: after `prep bench`, every trial folder of both shapes carried no `calcdir.json` (only a stage's attempt was marked, by `prepare_attempt`), against `project-layout.md` § 1.4a, so since 3a the run door read a trial as a folder nothing marks and the Results tab showed it no state; `materialize.mark_run(base, run_dir)` is the one marker -- containers down to the run, the run a run -- asked by `prepare_attempt` and by both places prep makes a trial's folder; the road test of a calculation's folders preps a benchmark and reads a trial as a run, pending (red unmarked); **3b.4b the jobset and transport readers** (**done 2026-10-04**) -- each asks the run door: `run_of(path, stage)` takes the stage a reader asks about in a folder several stages share (a flat calculation's), and `Run` answers `stdout` (the engine's output at the run's index), `outputs` (its stage's, newest run first) and `session_log` (the log whose first section is the run, `wrapper_log.log_of_run`); `materialize.stage_stdout` (prep's relaxed geometry, continuation's flat verdict, whose progress log is the run's too) and `summarize._latest_run_file` / `_wrapper_log` / `_trial_deck` are gone -- a trial's figures are taken at its run's one index, its session log no longer the newest by stamp -- and the transport record asks `Run.outputs`; three tests whose subject went are retired. 1788 targeted green. **The mutation check found three holes, none new, recorded and not filled here**: (a) `Run.outputs` oldest-first stays green -- the transport record's folder walk (`collect_record`, `_stage_facts`) has no test, only its parsers do: the fake-junction ladder (§ 5q P5) is where it gets one; (b) no session log stays green -- a trial's ranks × threads from its session log was pinned only by the retired unit test, and the bench-summary tests write a trial's `.out` by hand (`test_value_axes.py`): a benchmark launched on the road (stand-in engine, our wrapper) and summarized is the test, carried to unit 4 -- dropped there (4f, 2026-10-05), every launch now handing each run its own `-np` / `-omp`; (c) ignoring the reader's stage stays green -- in a flat folder it matters only once a later stage has launched too (status's continuation verdict, the flat geometry handover), which only a real engine's output shows (the vibration e2e); **3b.5 the relaxation record is of the file on screen** (**done 2026-10-04**; user: "go ahead") -- a gap since 3a: `contract.relaxation_of(directory)` found its own file with the floor-1 search (`openable_in`), so the record on the Metadata page could describe another file than the one the viewer shows (and a loaded `-run0.out` got run 1's record); now `relaxation_of(output, traj=)` reads the file its caller hands it and searches for nothing, `run_info_for_dir(directory, output=, traj=)` passes it through, the watch load hands the file it opened with its parse, `/api/results/contract` the run door's choice for the structure's folder (`runs.openable`), and continuation reads a run's verdict one way in both shapes -- its progress log, then its engine output, through `run_of` (the hierarchy's folder search gone); `model/parse.md` § 5b.1 says so, `web-api.md`, the fixture's README and a CLI refusal stopped naming `openable_in` for a run of ours. The bridge's composer test (the composer compared with the function it calls) became one asking BOTH doors of the measured relaxation for one record of the file on screen -- red with either route handing no file. 416 targeted green. **Hole, not new**: no test reads a continuation verdict -- switching it off stays green, as before, since the road's stand-in writes no relaxation; old and new code read the measured relaxation alike (`finished`, converged); **3b.6 `runfiles.find` and a label-named engine file** (**done 2026-10-04**, D24) -- the rule read first: § 2.2a's grammar covers the engine's carried files and § 4.2 finds the engine's files by name, so `find` returning `H2.XV` is the contract's answer; only its docstring ("our files") was wrong -- corrected, no behaviour change; **3b.7 the run's ENGINE label -- moved to unit 4** (2026-10-04; user: "continue"): read whole, nothing in 3b reads it wrongly -- the carry and the device's electrode inputs name an electrode's files with their one composer (`electrode_hs_stem`), the run script's `--cold` sweep baked with the calculation's label also sweeps `<label>_*`, the transport record's transmission files are under the label the transmission rung runs on, and the run door places an electrode's files by their folder -- while the readers that WILL ask a run's engine label are B12's (the trajectory load and `xv2xyz --from-run` finding the run's `.XV` / `.MD.nc` and its deck by name). So § 3.2's *its deck's `SystemLabel` / `JOB`* (an electrode rung's `<label>_L-electrode`), through `parse.fdf.system_label` and `pyscf.input.job_name` (no caller yet), is built in unit 4 with the first reader that asks it -- never a door with no caller. **Moved to unit 4 too**: the PySCF parser's `_resolve_job_token` -- a trajectory paired with its molwatch log by name, B12's question | **done 2026-10-04** -- 3b.1–3b.6; 3b.7 in unit 4 |
| 4 | B12 · D19 · B11's engine label (3b.7) | **in steps, one commit each** (read in full 2026-10-04): **4a** `declared(run)` and its first reader, `xv2xyz --from-run` (D19) -- **done 2026-10-04**; **4b** the Results trajectory load and the codec ask it -- **done 2026-10-04** (user: "you can do it. just make sure that it does make each run sees its own .fdf and take information from there rather than mixing"): no visible change for a run of ours, whose decks all carry the same labels and box; `Declared.atom_metadata_for` and `Declared.frame` read both from the run's ONE deck, the Results load asks them for the file it opened, the codec for an engine's own `<label>.xyz` in a run of ours (`about`, not `role_of`); `parse/dirs/atom_metadata.py` deleted (first deck wins, and the `.source.xyz` fallback for runs made before the record). Pinned on the measured flat H2: the Results load shows atom 0 held in the isolated 10 Å box, and each of its two stages reads its own `.fdf` -- red when a run reads the folder's first deck; the six hand-laid bridge tests and the built-folder round trip retired. **Hole, not new**: the codec's branch for an engine's own `.xyz` is pinned only by the export e2e -- no measured fixture holds a `<label>.xyz`; 4e's re-measured run can -- **pinned by 4e**; **4c the atoms a run holds, stated by its own output** -- **done 2026-10-04** (user: "the structure information is equally exposed to siesta engine and pyscf, there is no reason why siesta can manage to store the information while pyscf fails to do that"; "go ahead"): a SIESTA run's `.out` echoes them, and now a run's progress log carries `# frozen_atoms: <i> ...` -- prep's preview writes it for either engine, a PySCF run's script writes it when it starts -- and each reader reads them as its own content (the PySCF trajectory reader from the run's progress log it already reads); `_sidecar.py` deleted (the sidecar beside the output, the deck guess and their precedence), its `.out` echo reader moved into the SIESTA parser; `siesta_mdnc.sibling_md_nc` finds `<label>.MD.nc` by the label the `.out` prints (`reinit: System Label:`), never the lone `.MD.nc` of a folder. **Found building it**: the echo reader waited for a `Constraints applied` header SIESTA 5.4.2 prints only above several constraint blocks, so a run holding one -- our H2 runs -- had its held atoms from the deck guess, and a PySCF run's came from nothing (the lookup's names missed the `_initial` / `_optimized` pairs a PySCF run writes, and two of them made the lone-sidecar fallback decline); it now starts at the first `Constraint (N): pos` line. Pinned: `tests/data/held_atoms.toml` (PySCF and SIESTA preps state atom 0 in their progress logs, the PySCF script hands it to its writer; road key `progress_log_holds`), the measured relaxation's record holds atom 0, the measured flat H2's load reports `H2.MD.nc` and atom 0 from its `.out` -- five mutations red; 3339 targeted green; the guess tests retired (`test_siesta_fdf_constraints.py`, three classes of the echo file). **Not pinned, needing a real PySCF run**: the script's writer writing the line, and a PySCF trajectory file read with its held atoms; **4d the PySCF trajectory names its run's other files itself** -- **done 2026-10-04** (user: "go ahead"): geomeTRIC's trajectory is named on its run's stem (`<label>_<token>_geom_optim.xyz`), so the reader takes the stem off its own name for the progress log, PySCF's log and the wrapper's output; any other XYZ names no run and reads only itself. `_resolve_job_token` gone -- its own stage pattern (`_STAGE_TOKEN_RE`, the last second stage grammar, B11) and its newest-file-time pick for a stageless file -- and the pairing test rewritten on the writer's own names (it pinned `w_geom_01_coarse_optim.xyz`, a spelling no writer of ours has produced since 2026-09-07); with it the `_initial.xyz` energy fallback, which read a lone initial geometry's energy from the newest stageless `<JOB>-run<N>.pyscf.log` -- a file no staged run writes (theirs is `<JOB>_<token>-run<N>.pyscf.log`), its rule in no document -- and its seven tests; the four progress-log enrichment tests moved beside the pairing test. 1081 targeted green; the stem pairing broken, seven red. **The engine label (3b.7) dropped**: every reader of an engine's file finds it from the output's own name or line (`<label>.MD.nc` by the `.out`'s `reinit: System Label:`, a trajectory's companions by its stem) or by its folder (`run_of`), so § 3.2 no longer promises it and `pyscf.input.job_name`, kept for it, is deleted; **4e** the fixtures on today's road -- **`siesta_flat_h2` re-measured, done 2026-10-04** (user: "go ahead"): the same road, prep's `--np 1 --cpus-per-task 1` in place of a hand-edited `task.json`, 60.5 s of SIESTA wall; the engine's files came back byte-identical (`H2.MD.nc`, `H2.XV` -- the run is deterministic), and the folder now holds what today's prep writes: the structure pair named on the run (`H2.source.*`, D20), the coarse stage's seeded progress log (`# frozen_atoms: 0`), and SIESTA's own `H2.xyz`. Pinned on it: the folder opens the run's `.out`, not the seed beside it (`model/parse.md` § 5.5), and `H2.xyz` opens with its run's 10 Å box -- the codec's branch 4b left to the export e2e; both mutations red, 172 targeted green. **Found doing it**: the fixture as committed (1390b3a7) held 6 of its 12 files -- `.gitignore` drops `*.fdf`, `*.out`, `*.log`, `*.xyz`, `*.XV` and `*.psml`, and `tests/fixtures/` has no `!` line as `tests/watch/fixtures/` and `tests/parse/fixtures/` have -- so a fresh clone lacked the deck, the output and the `.XV` the tests read; `tests/fixtures/psml/H.psml`, which its README says is checked in, never was (42151fa5). Both now committed with `git add -f`; a `!tests/fixtures/**` line awaits a yes. **The trajectory e2e's fixture** -- retired with its tests 2026-10-04: it ran its deck by hand, outside the road (below, *the faking tests retired*); **4f** a benchmark launched and summarized on the road -- **dropped 2026-10-05** (user: "This is a stupid corner case and assumes so much. Make all calling consistent with —omp and settle it"): the case it guarded was one trial re-run alone in the queue from a shell exporting `OMP_NUM_THREADS`, because a stage sent to the queue alone was the one launch passing no `-np` / `-omp`; it now passes them like every other launch (`job-system.md` § 6.1), so a launched run's counts are its own, and nothing is tested for the corner. The scope as written: `declared(run)` from the run's deck; the run's engine label (`Run.engine_label`, 3b.7) with its first reader; the trajectory load, the codec and `xv2xyz --from-run` ask it; the sidecar and deck guesses retired ; and from 3a/3b: the PySCF parser's `_resolve_job_token` (a trajectory paired with its molwatch log by name, B12's question); `tests/fixtures/siesta_flat_h2` re-measured through today's road -- its structure pair is pre-D20 (`h2.source.*`) and it holds no seeded progress log, so *the engine's output opens before the seeded log* has no test for a run of ours; the e2e `test_trajectory_from_a_real_run_e2e.py::test_the_scf_line_states_the_runs_own_rate` given a fixture on the road (its deck runs in a bare folder, which since B11 holds no run of ours); a benchmark launched on the road and summarized -- a trial's figures from its own run, its ranks × threads from its session log (3b.4b's hole (b)) -- dropped (4f) | **done 2026-10-05** -- 4e committed, 4f dropped for the `-np` / `-omp` consistency |
| -- | Q6 | with the transport units (Q5) | ruled |
| -- | D28 | with unit 10's one `Continuation` | done 2026-10-05 |

> **Verdict (2026-10-08 validation):** reviewed-whole before 3b.3 CONFIRMED (its findings fixed; 3b.5's gap and unit 4's items closed by 3b.5 / 4a–4e).

**Reviewed whole before 3b.3** *(2026-10-04, user: "review your work holistically before moving on, update plan")* -- every commit from `50937399` to `3ee47a7d` read again against the contracts: a folder inside a calculation that nothing marks was told it is a container (`runs.folder_answer`; it is now said to be unmarked, nothing asked of it); `runstatus._job_status` kept a parameter 3b.2 left unused; three imports 3a left unused; `results.md`, `running-a-job.md`, `testing.md`, `transport.md` § 3.8 and `web-api.md` still named the moved doors (`openable_in`, `container_or_run`) or omitted `null` for a file of no run of ours -- all fixed. Measured clean: pyflakes on every changed file, the monitor bundle's calls, both readers of `host-env.txt`. Known and planned, not fixed: 3b.5's gap above; unit 4's three items.

> **Verdict (2026-10-08 validation):** reviewed-whole after 4c CONFIRMED; (1) ruled and built 2026-10-04; **(2) the transport citation reading a lone `.fdf` and lone sidecar stays OPEN** — carried as a line in the trim.

**Reviewed whole after 4c** *(2026-10-04, user: "review your work so far, and then continue")* -- `0fb23be9` to `00a7ef7e` (3b.3 to 4c) read by three fresh readers, every finding re-read in the code. **Found, and fixed**: every targeted set since 3b.4b had been built from `tests/*.py`, skipping `tests/parse`, `tests/watch`, `tests/spectra` and `tests/validation` -- three of their tests were red (`relaxation_of` still handed a folder twice, a PySCF test still laying a sidecar) and a fourth passed for any folder: the first two now hand the run's output, the third states the held atom in the run's progress log, written by its own writer; the PySCF vibration script built its progress-log writer without its held atoms, so its first line wiped prep's (now `view.frozen_indices`, and a `held_atoms.toml` row for it, red before the fix); `runfiles.find`'s new docstring claimed no engine file carries a stage (geomeTRIC's trajectory does); the SIESTA header claim was wrong -- 5.4.2 heads several constraint blocks and leaves a single one bare, measured on 31 recorded outputs, so what the reader never read was a run holding one block; `runfiles.is_stage_token` dead since 3b.3; stale texts in `job-contracts.md` § 2.3, `architecture.md` § 3.2's readers (status and launch compose their stem, they do not ask `run_of`), `project-layout.md` § 1.5a, `structure-periodicity.md`, `web-api.md`, `web/results.md`, `structure-annotations.md`, `parse.md` (its tree, § 5.1's note, § 5a's table), three comments and four test docstrings; three hand-laid tests retired. 3037 targeted green, the four subfolders included. **Open, for a ruling**: (1) a run's electronic contract (`info.calculation`) is still read by a folder search -- `contract_of(directory)`, one `.fdf` or nothing -- so a flat calculation with two stages answers none, while § 3.2 gives `declared(run)` "its stated parameters" -- **ruled and built 2026-10-04** (user: "go ahead"): `contract_of(deck)` reads the deck its caller hands, the composer is `run_info(deck=, output=, traj=)`, the Results load and the structure inspector's door hand the run's own deck (`runs.declared`), and the measured flat H2 now records `H2_01_coarse.fdf`'s contract (red with either route handing no deck); `parse.md` § 5b states it; two bare-folder tests retired; (2) the transport citation reads a cited run's labels, held atoms and axes from the cited folder's lone `.fdf` and lone sidecar, and refuses a flat citation holding two decks (`transport-design.md` § 4.1b's one-deck rule). Noted: a run's held atoms now reach the Results load twice -- the deck's block on the structure, the output's statement in `runtime_info` -- one asked, one applied; they agree for every run of ours.

> **Verdict (2026-10-08 validation):** the faking tests retired CONFIRMED (446 tests in 72 files; `process/testing.md` § 6); `projects.find_geom_candidates` deleted 2026-10-05; `spectra.select_modes` imported by the deck.

**The faking tests retired** *(2026-10-04, user: "any fucking faking tests should be retired"; "i don't see any fucking point of faking output test"; "if you find any such tests, need to carefuly justify why we need them")* -- the 2026-10-03 sweep (`process/testing.md` § 6) had missed three bench-summary tests that laid an invented `.out` and SCF-timing log in trial folders, so the suite was read again: ten read-only agents classified every test of 120 candidate files against § 6's rule, and each one flagged was checked against its cited line before it went. **446 tests in 72 files retired, 16 files whole** -- parser tests on made-up output text (the SIESTA, progress-log and PySCF parsers' synthetic samples), refusals of hand-edited `.spectra.json`, restart files planted for the cold start, the carry and the identity check, earlier runs' outputs typed for the run index, monitor and timing lines typed for the instruments, the CO2 e2e decks run by hand, records of calculations nothing ran carried in structures -- with the two fixtures cut from their runs (`device-diverging.out`, `chain-tbtrans-v0.4.out`), `support/junction.xv_file` (a hand-rolled `.XV` writer) and the empty `tests/parse/dirs`; § 6 now says output handed to a parser counts, and that a faking test is kept only with its reason stated to the user. 688 + 189 targeted green; the touched e2e files checked statically only (pyflakes, fixture names). **Kept, reasons stated to the user**: the `siesta_frozen` outputs, `device-converging-live.out` and `chain.TBT.AVTRANS_L-R` -- real outputs of real runs, renamed or path-scrubbed, read as themselves (the only real outputs of runs that ended badly); the monitor replaying the measured relaxation under its own names (§ 6 allows it); our generated run script or deck run directly on a stand-in engine; three e2e runs with a file prep ships broken on purpose, to reach the explicit error; three PySCF cross-checks of our vibration math; pure functions on in-memory objects; checkpoint and zip mechanics on bytes; the suite's stand-in engine. **Found**: `projects.find_geom_candidates` had no caller -- deleted 2026-10-05 (user: "go ahead with 1, then 2") with its pattern helper, its last test, the path-finder survey's exemption and its line in `job-contracts.md`; `spectra.select_modes` had no production caller beside the PySCF deck's hand-written copy of it -- the user's call: the deck now imports it (*The PySCF script imports molbuilder's code*, below); two emitter tests always skip here (`pyscf.gto` lives only in the PySCF env).

> **Verdict (2026-10-08 validation):** the PySCF script imports molbuilder's code CONFIRMED (`mb_pyscf.pyz`, `runwrap.PYSCF_COMPANIONS`); of its *Left*, **the open-shell `mo_energy` read stays OPEN** — carried as a line in the trim; the thread/GPU set-up, the recipe comment and the SIESTA force-constant e2e were done the same day.

**The PySCF script imports molbuilder's code** *(2026-10-05, user: "can pyscf use something imported from code base just like the other facilities we exposed to run script through the pyz package? ... we could use one code base and maintain it rather than through \"generated python code\" which is another layer of indirect testing/validation"; "all three, go ahead")* -- a third bundle, `mb_pyscf.pyz` (`runwrap.PYSCF_COMPANIONS`, the one builder), travels beside every PySCF script: prep writes it (`render_wrappers`), `materialize` brings it into every attempt, the catalogue lists it (the manifest regenerated). The script puts it on its import path on its first lines -- a missing file stops it there, with a sentence -- and imports, as `_mb_<name>` (molbuilder's prefix in a script, now stated in `engines/pyscf.md` § 3): the progress-log writer (`MolwatchEmitter`, no longer pasted by `inspect.getsource`; its two literal unit factors are `constants`' now, and the test pinning them went, as it asked), the structure codec (`StructureCodec.write_moved`, through `write`'s own path, replacing the script's fifteen-line pair writer with its own number format and JSON settings -- so a PySCF run's `.xyz` now carries the codec's six decimals, not eight), and in a vibration script the mode selector (`select_modes`, on plain inputs; its hand-written copy, the test holding the two equal and the `prior` argument no caller passed went with their tests). No separate load check in the run script: the script's own first lines import the bundle before PySCF computes anything. Proved by the PySCF engine test (the real optimization script run under `molbuilder-pySCF`, molbuilder absent) and `test_pyscf_bundle_runs_alone.py` (the bundle alone writes a pair the package's codec reads back; mutation-checked). **The rest moved the same day** *(user: "run the e2e, then move the rest into the bundle" -- the fifteen PySCF end-to-end runs were green first)*: the relaxation (`relax_policy.relax`); the HOMO rule, the Hessian with dμ/dR and the GPU array bridge, now `spectra/pyscf_vibration.py`; the harmonic path and thermochemistry (`spectra/normal_modes.py`); the hash, the result's writer -- `sidecars.spectra.write_spectra_payload`, which `dump_spectra_json` now calls -- and its non-finite scrub (`finite_or_none`); the restart's geometry, read with the one XYZ reader (`Structure.from_xyz`), so a file it cannot read stops the run where a hand-written parser warned and fell back to the input geometry; and one copy the first list missed, the node's core count (`runtime_info.physical_core_count`), imported at the script's head because the threading setup must run before numpy loads. **What stays in the scripts:** their anchor (`_MB_SCRIPT_DIR`, `_mb_outfile`), which finds the bundle and so cannot come from it, and the run's own composition -- the SCF dressers, the GPU and threading set-up, the log's end hooks, the vibration's per-run helpers and its branchless IR and Raman formulas (`engines/vibration.md` § 6.4). **Found on the way:** the catalogue's template row named `template.template_path` as its door while 9b part 5 had hand-edited the generated table to `template.find_template` -- the catalogue says `find_template` now, so regenerating keeps it. **Reviewed with fresh eyes the same day** *(user: "use agent to review your work on the pyscf script with fresh eye and with full text/code review holistically to detect any overlapping, duplicated, redundant parts or inconsistency"; "fix the review findings, then commit and start unit 10")* -- one fresh reader went through both scripts, the bundle's members and their docs whole; what it found was fixed. **Copies the two passes left:** the orbital window (`pyscf_vibration.mo_window`); the wavenumbers and the display form, written a second time beside the SIESTA route's -- `normal_modes.signed_omega`, `frequencies_cm1`, `display_form` and `thermo_temperatures` now serve both routes, and the thermochemistry reads its constants from `constants`; the log's end lines (`MolwatchEmitter.conclude_at_exit`). **One writer where there were two:** a progress log's header and step 0, for prep's seed and the script alike (`trajectory_log.emitter.header_and_preview` -- the seed's `# stage:` line, which no reader read, went with its argument); and, for both decks, the held-atom file (`emit_constraints_file`), the header's Outputs list (`emit_outputs_block`) and the geometry saves (`emit_save_call` -- a vibration run's `.xyz` comment lines now read as an optimization's). **Spelled once:** five roles the composers wrote by hand now read from `input`'s role constants (`ROLE_LOG` and `ROLE_SPECTRA` added). **Tests:** § 3's import rule, *each import from the bundle is bound as `_mb_<its name>`*, is checked as stated, over every optimization and vibration deck against `PYSCF_COMPANIONS`, where a hand-kept allow-list had gone stale (mutation-checked); the end-to-end probes import from the bundle as a script does, no longer pasting source or putting the repository on the PySCF env's path; two duplicate tests, four tests of the seed writer's removed arguments and one vacuous assertion retired; an explicit mode list naming no mode is dropped, and the script says so (`engines/vibration.md` § 4.8). Docstrings naming a `convert` route that does not exist, or `inspect.getsource` as the project's pattern, corrected. 1,414 targeted tests and the fifteen PySCF end-to-end runs green. **Left, reported to the user:** the threading and GPU set-up, branching logic still written as text beside `runtime_info`, the member that could hold it -- § 3 keeps them as the run's own, so moving them is a design question; the comment in `envs/recipes.py` on why the PySCF env carries ASE (an env file); an open-shell `mo_energy` read that takes the first spin channel without saying so, older than this work; the SIESTA force-constant end-to-end run, the real proof of `vibrational_analysis` reading the shared helpers, not run -- its API-level test is green and every member of `mb_vibration.pyz` imports alone. **Then all three, at the user's word** *("yes to #1, ok for #2, and #3 can be done if it does not take up too much time - run it in parallel to other work")*: the thread and GPU set-up moved into `runtime_info` -- `cap_threads` (the run's count, else the allocation's, the node's last; BLAS capped before numpy loads), `runtime_facts`, `probe_gpu`, and `to_gpu` for the optimization script's promotion; a vibration script keeps its choice of gpu4pyscf's classes, and states `USE_GPU` once where it stated it twice -- and the run script's chain is built from the same list, `runtime_info.THREAD_SOURCES`, where a comment had held two spellings in step (`running-a-job.md` § 3.2's diagram named two of the four variables; corrected); the scripts are 120-136 lines shorter; the two tests holding the chains' texts in the same order retired, one test of the caps added; one mutation, red; 3,332 targeted tests and the fifteen PySCF end-to-end runs green, the GPU run among them. The PySCF env recipe's comment says what its ASE is for -- the script reads the rung before's geometry through `Structure.from_xyz`, whose parse is ASE's (a comment; the recipe's code unchanged). The SIESTA force-constant end-to-end run green, 5 of 5, run from a frozen copy of `c9906723` beside the work.

> **Verdict (2026-10-08 validation):** W57 *Fixed* CONFIRMED (`7d52b551`, `36c545c5`, `a1a74166`, `43b43fe7`).

#### W57 -- the explicitness review *(2026-10-06)*

*(user, 2026-10-06: "review your fucking code again holistically scoping focused on this fucking pattern of fucking overly simplify and being implicit rather than explicit for no fucking good reason"; "fix every defect the review finds")* -- four read-only reviewers on the prep, launch and run path: the records a run and a machine write (R1–R11), the printed commands (P1–P7), names and tokens (N1–N9), values code fills in (D1–D7). Each finding was read again in the code: a defect against a written rule was fixed; a deliberate choice, or two documents that disagree, is a decision below.

**Fixed.** `7d52b551` -- D1 (a run's stated cores per rank and GPUs reach both engines' decks, `model.AS_RESOURCE`), D2 (a half-declared grid refused, the constant grids gone: the machine proposes from what its record says, or not at all -- Q8's ruling, no grid the machine proposes, is still to build), D4 (help texts and docstrings promising removed defaults), D7 (the second default tables). `36c545c5` -- R1 (`run.json` writes `continued_from` and `placed_on` always, null an answer; a `.continued-from` that does not read stops the launch), R2 (`Gres=(null)` is 0 GPUs; the refusal tells unknown from none), R5 (the folder answer keeps its shape on failure), R7 (every run prep writes `continues` or `starts-cold`), R8 and D5 (the record states its scheduler or does not read; the per-core memory default R13's three states), R11 (a launch record that does not read is the run state `unreadable`); no SLURM text in the basic suite, and the field tier (`tests/field/`, `testing.md` § 0). `a1a74166` -- P1–P4 (`--bundle` and `checkpoint -p` printed every time; the way back in one spelling, `identity.checkpoint_words`; SIESTA's thread chain from `THREAD_SOURCES`), and, found reading the diff, the vibration finish's remedy importing `identity` package-only: a relax → vibration run whose reference geometry was not stationary crashed at the finish (since `a8f05977`); `identity` ships in `mb_vibration.pyz`. `43b43fe7` -- P6 (a launch offered in another mode is the whole command, `commands.launch_with`; probe's dry run prints its `--write` line), N3 (the trajectory seed named for its deck), N4 (the cell label by `point_token`), N7's stale comment, R3's asked slot (no `.out` count among the knobs) and the CLI's `no gpu`.

> **Verdict (2026-10-08 validation):** W57's decisions table OBSOLETE — every row resolved (ruled 1–7, the re-screen of 2026-10-06, the builds `58634eaa` … `adeb6c6d`); the last row (the e2e tests, the browser walk, the Sol field test) is Q13's *Left*.

**Open -- the user's decisions**, each with a proposal:

| # | finding | proposal |
|---|---|---|
| N1 | flat: `.molwatch.log`, `<stem>.run.json` and `<stem>.continued-from` carry no `-run<N>`, so a re-launch truncates run 0's PySCF trajectory and replaces its launch record; the documents disagree (`project-layout.md` § 1.5a: every per-run artifact carries the index; § 1.6.3, `job-contracts.md` § 6.1 and § 6.3 name these unindexed) | the index on all three in flat, the PySCF log's path from the wrapper's run index; one rule in the documents |
| N2 | `-J` is `<calculation>/<job>`: two stages' benchmarks queued together read alike in `squeue`, and `scancel -n` takes both | `<calculation>/<stage>/<job>` for a benchmark's jobs, § 6.3's examples with it |
| N5 | a shelf's name adds qualifiers only when needed (`bench-group`, `-cpu`, `-G0K4C1`) -- deliberate, `generator.md` § 4.3a | keep, or one spelling always |
| N6 | a single-bias rung's attempts are `04_device/run-0`, a scan's `04_device/v0/run-0`; `v-0.5` puts `.` and `-` in a folder name; § 6.3's Directories table has no v-folder | the table's row now; the single-bias shape with the transport work |
| N7 | `point_token` drops `-`, so a value `-2` is named `2` (latent: no swept value is negative) | spell a number's sign `m`, as its point is `p` |
| N8 | `.monitor.log` and `.util.csv` keep an unindexed spelling nothing writes; the deck banner's by-hand `.pyscf.log` is unindexed and ranks below run 0 | drop the unindexed rows -- old runs are not a design input |
| N9 | an empty stage token still names things -- `bench_container` → `bench`, `stage_home(None)`, `pipeline_log.log_name`, `stage_token or 'run'`, `_shelf_token`'s `N` spelling, `job_dir_names`' three tokenless rows, `_load_bench_set`'s hand-built set -- though every description has a stage since 2026-08-16 | refuse an empty token everywhere; tests build their sets with tokens |
| P5 | "launched here" is decided by the word `this` in the calculation's copy: prepped on Sol without `--target` and copied back, it reads as this machine's | the copy names the machine it was made on, never `this` |
| P7 | `jobset init --engine` defaults to SIESTA, silently | required |
| R3 | a trial's knobs and the summary's columns change with the sweep (`gres` on GPU trials only; the `machine` and `gpu` columns by content) | every trial carries all three knobs; the columns fixed |
| R4 | `choice: {}` stands for two situations, each reader guessing which | `choice` states why it is empty |
| R6 | a transport rung's state in two vocabularies (`state: "ran"` meaning two things); the status door not asked for an SCF rung | with the transport work: every rung's state from `run_status` |
| R9 | `job-set.json`: `point`, `finish`, `placement` absent when empty and `resumes` written only when false, beside `resources` writing every field | every key always, null when none |
| R10 | `task.json`: an absent `calculation` means optimization (restated in seven JS sites); `notify: {"every_hours": 0}` accepted and read as off | `calculation` always written and read through one door; `every_hours: 0` refused |
| D3 | the PySCF vibration deck writes `max_memory = 4000` for a blank; the optimization deck writes nothing; the catalogue calls the blank "the machine's maximum", which nothing resolves | **ruled and fixed 2026-10-06** *(user: "there is no fucking memor limit if we did not specify in the first place" -- ruled 2026-08-13 and 2026-08-14 already, as the field's own comment says; the proposal here was wrong)*: no cap unless stated -- the vibration deck's 4000 gone, both PySCF decks write `max_memory = None` (PySCF's `Mole.build` reads it as not given) and record none, the catalogue's help says so; `template.md` § 2; two rows in `launch_values.toml`, each turned red by its mutation; the test pinning the line's absence retired (its two reasons false against PySCF's source) |
| D6 | a notify block with no `channels` sends to every channel, and Task setup writes that same absent key for "every channel" | `"*"` for every channel; an absent key refused |
| T1 | the road's stand-in `sbatch` answers in SLURM's words (`Submitted batch job 4242`, a `--test-only` prediction), which the basic suite parses | the basic suite stops at the line molbuilder sends; reading the scheduler's answer is the field tier's |
| T2 | tests make a spectra result from invented numbers and write it with our writer (`tests/spectra/test_blueprint.py`'s load route, `test_parsers_json.py`, `tests/spectra/_helpers.py`) | the measured fixture (`tests/fixtures/siesta_h2o_modes`) where one serves; the type's own tests kept |
| T3 | `conftest`'s default machine record (a workstation, 2 × 8 cores, no GPU) is written by hand, in molbuilder's own format | keep: it is the road's stated machine, as rows state their queues |
| T4 | no basic-tier test loads the vibration bundle beside a job, as the monitor's and PySCF's do; `a1a74166`'s slip was caught only by reading | one, as the PySCF bundle's |
| -- | the Task setup page's e2e tests ((c)); the browser walk (manual, or Wednesday 2026-10-07); the field test on Sol (`jobset probe --write --name sol` there, the record copied here, then `MOLBUILDER_FIELD_RECORD=<it> python tools/testrun.py run field`) | the user's to run, or to say when |

> **Verdict (2026-10-08 validation):** ruled 1–7 CONFIRMED (decision 1 is Q2f's session; 2 and 6 built 2026-10-07 as `runfiles.RunNames` / `launch --run N`; 3 withdrawn; 4, 5, 7 built); the re-screen's *dropped* and *settled* lists built 2026-10-06 (`bf2f68e1`, `422f365c`, `ec7823d9`, `5637cdb3`, `3b80b52c`, `adeb6c6d`); *with the transport work* (N6, R6, the electrode deck header) is Q5–Q7's.

**Ruled, one at a time** *(user, 2026-10-06: "bring the decisions to me one at a time, after confirming answer then move to the next. need simple language, example and context for each decision discussion")*:
1. **How a file that runs beside a job imports** (raised by the vibration finish's crash, `a1a74166`): one spelling -- the absolute import, Q2f's form (`from molbuilder.identity import launch_as_typed`), the zips keeping molbuilder's folder layout -- *"B, with Q2f as one session"*; *"why can't we use absolute path?"* -- we can, and do (the Q2f row).
2. **Run numbers on a flat stage's trajectory log, launch record and `.continued-from`** (N1): **A** -- each carries `-run<N>` like every other per-run file of the flat layout (`H2_01_coarse-run0.molwatch.log`, `-run0.run.json`, `-run1.continued-from`); the hierarchical layout unchanged; the PySCF script takes its run number from the run script. *(user: "A. ... unified api let the result to be able to identify which is which?")* -- yes: the catalogue rows carry the run number, the writers name the files through `runfiles.compose(..., run=n)`, every reader asks `runs.Run.file(role, run)` -- the door that already tells `-run0.out` from `-run1.out`; `runrecord.launch_record_path`'s own flat spelling goes.
3. **N8 -- withdrawn.** Its example was a person re-running a stage with the engine command a deck's header prints, around `launch`: not molbuilder's road *(user: "why would you care if people do not honor jobset commands and fill their shit hole with more shit?")*. The catalogue's pre-2026-08-27 unnumbered monitor names go under "old runs are not a design input".

**The rest, re-screened 2026-10-06 against the user's rules** (MEMORY gate 6: only a failure on the jobset road is a decision):
- **dropped -- not on the jobset road:** N2 (two stages' benchmarks alike in `squeue`; the name follows § 6.3's `<calculation>/<job>`, and molbuilder tracks a job by its id), N7 (a negative swept value -- nothing produces one), P5 (a calculation prepped on one machine and launched on another -- a calculation's machine is set at its first prep).
- **settled by a standing rule -- fixed, no question:** N5 (shelf names in their full form always), N9 (an empty stage token refused; the tokenless rows go with the hand-built job sets that used them), R3 and R4 (the summary's knobs and columns one shape; an empty `choice` says why), R9, R10 and D6 (`job-set.json`, `task.json`'s `calculation` and the notify channels written whole -- "records keep their full form"; the documents' reasons were old files), R10's `every_hours: 0` refused, P7 (`init --engine` required -- "explicit job config is the only way allowed"), T1 (no invented SLURM answer in the basic suite -- the stand-in `sbatch`'s replies go; reading a scheduler's answer is the field tier's).
- **with the transport work (Q5–Q7):** N6, R6, the electrode deck header.
- **left for the user:** T3 (the road's own machine record, written in molbuilder's format), T2 (spectra results our writer makes from invented numbers); T4 with the Q2f session.
4. **The road's machine record** (T3): **A** -- a row states the machine it needs in molbuilder's own format (`tests/conftest.py`'s default record, the tables' `[[queues]]`), never a scheduler's text; reading a real machine stays the field tier's *(user: "A")*.
5. **T2 -- settled by the 2026-10-04 rule, not a decision:** the spectra tests that exist for "numeric-literal flavors that engines or hand-edited files might produce" (`test_parsers_json.py`, the scientific-notation pair) retire -- handcrafted input; the rest write through the one writer, `dump_spectra_json`, instead of `json.dumps(to_dict())` (`_write_json`) -- "our own file through our own writer".

6. **One launch, several runs** (found building decision 2: a SIESTA warm retry re-starts the run script, and the retry takes the next run number): **A** -- every run number has its own launch record; `launch` decides the first number and hands it to the run script, so the two cannot disagree; a retry writes its own record with `"retry_of": <n>` *(user: "A")*.

7. **Calculations prepped before the record fixes** (R9, R10, D6 make `job-set.json`, `task.json`'s `calculation` and the notify channels written whole; the reader is strict): **A** -- such a file is refused naming `molbuilder jobset migrate --bundle <calc>`, which rewrites it, every value kept, each change printed, the old file kept beside the new -- the verb's existing pattern for older templates *(user: "A. go ahead")*.

> **Verdict (2026-10-08 validation):** *Every decision answered* and the two *Built* paragraphs CONFIRMED (W57 built; no saved run in the tests, 2026-10-06; `runfiles.RunNames`, the run number handed by `launch`, 2026-10-07); unit 12's two review rounds done (`b1a1d4e3` and the commit after it).

**Every decision answered** (2026-10-06). To build, one commit each with its rows and mutations: P7, R10's `every_hours: 0`, D6; R9, R10's `calculation`, R4, R3; N5, N9, N8's old monitor names; decision 2 (the flat per-run files' run numbers); T1 and T2. Then the milestone's two automated review rounds, then unit 12. Decision 1 is the Q2f session's.

**Built** (2026-10-06): D3 `58634eaa`; P7, R3, R4 `bf2f68e1`; N5, N9 and the full names `422f365c`; T2 `ec7823d9`; T1 `5637cdb3`; R9, R10, D6 and decision 7 `3b80b52c` -- `job-set.json` and `task.json` written whole, a notify block's four keys every time (`every_hours: 0` refused: a block states every key, and `null` is the one spelling of never), an older file refused naming `jobset migrate`, which decides its records and its template before it writes either; R10 at every door `adeb6c6d` -- `init --calculation` required as `--engine` is (the workflow guide's own frequency example described an optimization), `Task.calculation` and `build_description` take the kind, the analyze route and Task setup's loaders state engine and kind or ask nothing. **No saved run in the tests** (2026-10-06; user: *"when a test need siesta's output why is it not part of a e2e test? ... what's different about the one that remains"*, *"yes, retire them and do the e2e move, deletions approved"*): every test that read a saved run or a frozen output moved to the e2e tier, rewritten against the contract on runs made in the test's own module through the road -- a flat H2 relaxation and a capped one (`test_siesta_flat_run_e2e.py`: the folder answer, a run reading its own deck, the `.XV`, the `.MD.nc`, the echo, the build line, the timing log, the fdf log, how a run ended), a vibration's relax stage (`test_siesta_relax_run_e2e.py`: the record, both doors, `freq` building on it, the gate's seven verdicts; `test_monitor_watches_a_live_run_e2e.py`: the monitor and its policy replaying the run's own output, the shipped bundle reading it), the stopped run's SCF stop as the retry asks it, free water's mode matching (`test_vibration_e2e.py`), the pageshow refresh on a live output.  Retired with no replacement: the parser's own output saved as its expectation (`test_combined_dispatch.py`), the synthetic header lines, a force-constant output cut to its first steps, the migration tests that froze or hand-edited an older record, and the TranSIESTA/TBtrans outputs (read again on the transport road's minimal junction, Q5-Q7).  `tests/fixtures/` holds the H pseudopotential alone; `testing.md` § 6 says why a saved run drifts (54 run-script changes since the relaxation was saved).  Found on the way: the thread-chain pin `a1a74166` broke (fixed), and the road runner's run-script section starts only for rows naming `given_gpus`, whose stand-in `nvidia-smi` answers invented text -- the T1 class, to retire with a row key of its own.
**Built (2026-10-07), decisions 2 and 6:** one naming object, `runfiles.RunNames` -- every writer and reader names a stage's files through it, and `runfiles.GroupNames` a launch group's; it refuses a run with no stage (`stages.md` § 6.5). `jobset launch` decides the run number and hands it on as `--run N`; the run script and the PySCF deck refuse to start without one; a flat stage's per-run files carry `-run<N>`; a warm retry writes its own launch record, `retry_of` its run. Found by reading, fixed: SCF tolerances to two significant figures in both engines; the deck check missed a keyword written twice; `MD.UseSaveCG` written for a CG relaxation alone (SIESTA reads it in its cg branch only); names spelled by hand in the code; the `propor` hint (restore the saved state, prep with fewer ranks). **The tests, judged** *(user: "go"; "approve the 199"; "approve A and B, drop the version banner, § 6 rules"; "retire render_fdf and render_script, move their tests onto the road"; "retire those tests that ping ... old obsolete things")*: three retirement rounds, the last a script the user runs (`retire_round3.py`); no retirement notes in tests; a test pinning an obsolete form retires. `render_fdf`, `render_script` and `apply_siesta_stage` are deleted -- nothing in molbuilder called them -- and their cases are rows down the road: `tests/data/the_deck.toml` (new, 28 rows), `held_atoms.toml`, `restart_files.toml`, `gpu_contract.toml`, each row turned red by a mutation of its input or of the code. The road runner takes a row's own `structure`, `stage_strategy`, `deck_order` and `deck_once`, and reads the row's stage's own files by their names; `testing.md` § 3a agrees with § 6 (a direct render only for a refusal the road cannot reach); the stand-in engine answers `siesta --version` with nothing. Obsolete names (`MD.NumCGsteps`, the phantom step keywords) are gone from the catalogue's help, the code and the contracts.

> **Verdict (2026-10-08 validation):** the user's word on § 0c (B1–B10, Q1–Q5) — every ruling built (units 1–12; Q1 superseded by 12b-1); kept here verbatim as history.

#### The user's word on § 0c *(2026-10-03, verbatim)*

> **B1**: always save through checkpoint, notify user, and make sure name of
> the checkpoint clearly shows timestamp. the checkpoint save is notified after
> the build is decided and checked. **B2** - we need a framework and template
> (or config data/instruction) driven approach such that the current prep can
> be used for the transport preparation too. the added transport steps can be
> designed as an optional step for transport operation but it would be in the
> same framework of prep. **B3**. agree, same suggestion as B2. **B4**: agree
> to your suggestion. **B5**. agree. **B6**: i like that idea, this gives a
> bird's eye of one parent task dir that can have different run results in one
> place such that result tab would be able to show them (don't have to get
> into individual run dir to probe, and rather result can display the result
> by simply select which run to pick. **B7**: agree, but make sure this is a
> framework level unification and contract/design document is updated.
> **B8**: agree. unification at api and framework is the goal. **B9**: agree.
> dead code will confuse review and distract/drift code from correct
> direction. **B10**: make this consistent with B1,B2 and B3. for **Q1**: (a),
> i don't quite understand why we allow off instead of delete, but i figure it
> could be some changing design and left some dead dir. in that case, mark the
> dir as disabled and never allow use would be the correct way. **Q2**: a,
> **Q3**: b, and write into log what is dropped for what reason, **Q4**: error
> when no env_init is present. this is required explicitly. **Q5**: a, but
> make sure the result presentation, data record and the summary/comments
> clearly explain what is what.

> **Verdict (2026-10-08 validation):** the user's word on the framework design — built (unit 10d's hand-over door, `job-system.md` § 5.4; ruled again 2026-10-06, the status door); kept here verbatim as history.

#### The user's word on the framework design *(2026-10-03, verbatim)*

> *(on "a run that ended with an error is never built on")* yes, for #1, but
> user can force a structure still - confirm that a structure with the correct
> json info can be used and clarify what information is needed. *(after the
> answer — `--from` a run is taken as said; `already_relaxed` takes a structure,
> its `info.relaxation` evidence, never required)* … nothing is stopping us to
> work from a run. the structure is just a plus that can be validated as a
> second option. for #2 yes, for #3: ok

> **Verdict (2026-10-08 validation):** *Settled* CONFIRMED — the default builds on a finished run (unit 10d, the 2026-10-06 ruling); units 9a → 12 approved and built but 12b-2–12e; Q2e done.

**Settled:** the default builds only on a finished run — the status door's
answer, from every fact of the run (ruled 2026-10-06, (b)) — a named run and a
structure stated relaxed are the person's to force (`job-system.md` § 5.4); a refused transport prep leaves no deck — the preview
shows it; units 9a → 9b → 10 → 11 → 12 are approved as designed. Q2e (the
Metadata pane) goes first.

> **Verdict (2026-10-08 validation):** the work order's units 1–4, 6–11 CONFIRMED done (unit 4's doc residue `job-system.md:960`; unit 7's schema @3; unit 11's stale `_bench_walk` comments `submit.py:22,525,1758`); unit 5 OBSOLETE (`task.stage_disabled` went with 12b-1); unit 12 PARTLY (12b-2–12e open) — its row stays in the trim, with one clause amended (§ 2a.11's sweep built since, `ddf12a43`); its text as it was is below.

**The work order** — one unit per commit, each with its contract text, rows and
mutations; the framework rounds' design written into the contracts first:

| # | unit | what it is | state |
|---|---|---|---|
| 1 | Q4 | `jobset probe --write` refuses when this machine's `molbuilder.json` states no `env_init` — the record's copy is required, never kept from before | **done** — refused before anything is probed, after what was typed (`--set`, the reserved name); the keep-the-old-copy case is gone; `configuration.md` § 4 and its table row say so; probe rows of the road stand on the `molbuilder.json` `init-config` leaves; one row (`probe_refused`), the retired keep-row, one test retired (the probe can no longer be the config directory's first writer); one mutation, red; 195 targeted green |
| 2 | Q2 | a declared value kept at a re-probe keeps `flag` in its `source` | **done** — `source` is noted per section, and a section whose kept value was declared keeps `flag` (`_probe_consent_merge`); M-6 says so; one row; one mutation, red |
| 3 | Q3 | M-2: `env_init`, `conda_envs`, `env_arch` are left out when empty, and the probe says which it left out and why | **done** — M-2 reworded; `diagnostics.local_facts` returns its notes, one saying `conda_envs`/`env_arch` were left out and why (no manager answered, or it listed none), shown by `jobset probe` and `envs init-config` (the probe keeps no log file: its notes are its log); one row (`record_lacks`); one mutation, red; 90 targeted green |
| 4 | B1's save · B10 | the folder's state is saved before every prep and every Task setup Save — always, through checkpoint, the note led by its timestamp, the person told; one server function for both; the offer, its question and the page's box retire | **done** — `checkpoint.save_before` (and `Kept`): its first state, a new one when anything changed, the one it stands at named when nothing did; the note `2026-10-03 14:05:12 · before prep run coarse`; a save that fails refuses (a molbuilder.json that does not read, in its own words). Prep calls it at checkpoint 5 and ledgers `saved`; the save route calls it before writing; the CLI's question, the page's offer box, Save's tick and note, `Answer`/`SaveOffer`/`save_offer` gone; the history panel re-reads on a folder change. `checkpointing.md` § 9, `job-system.md` § 5.0/§ 5.3, `task-setup.md` § 7/§ 8/§ 11.1, `web-api.md`. Three protocol rows (`saved_states` takes `{stamp}`), the three-door API test, two browser tests; three mutations, each red; 2189 targeted + 18 browser green |
| 5 | Q1 | a stage turned off is refused everywhere — prep, launch, what continues from it — and its folder, where one was left, is shown disabled and never used | **done** — one answer, `task.stage_disabled`, asked by the prep entry (both arms; transport's own refusal gone), launch (`run` and `bench`) and a `--from` naming its run; `status` shows a disabled stage's folder as kept and never used; the flat force-constant count is over enabled stages. The folder is marked through the description — its one source — on every surface; a mark inside the folder can ride B6's `calcdir.json`. `stages.md` § 6.2, `job-system.md` § 5.0 / § 5.4, `vibration.md` § 5.9. Three protocol rows (road keys `stage`, `disabled`); a transport test retired, two tests to the new rule; three mutations, each red; 3680 + 109 targeted green |
| 6 | B9 | dead code deleted (§ 0c's row) | **done** — the three routes no page called (`/template-values`, `/resolved`, `/attempts`; their tests read the folder answer's parts, the door-equality test retired, `web-api.md`'s rows and count, two docs' mentions); root-deck adoption (two API tests' fixtures write the deck where prep writes it); the merge's three replace-outright arms (its same-name filter kept: two preps racing past the gate would otherwise double a row); `launched_trials`; stale route lists in `build.py` and `viewer.js` (a `launcher` route that does not exist). `materialize`'s re-exports go with B7 — they move import paths. 2295 + 10 targeted green |
| 7 | Q5 | a junction's current is the total — ×2 unpolarized, the channels' sum polarized — with TBtrans's printed figure and the factor beside it, said plainly in the record, the Results tab and the summary | **done** — `current_a` is the total (twice the printed figure for a non-polarized run, by the spin its own deck states: `record.deck_spin`), `current_a_printed` beside it, `current_means` the words for each; the table `summarize` prints and the Results tab's view show both columns and the words. A polarized point's total — its two channels' sum — stays empty until K21 reads both channels. Schema `transport-result@2` (an `@1` record's `current_a` was the printed figure; refused by its version, written again). `transport.md` § 2a.4, `job-contracts.md`. The record test extended (both spins), the view rendered in node on a real record; four mutations, each red; 287 targeted green |
| 8 | B4 · D11 · D12 | the folder answer carries the machine the calculation is set to, its prepped stages and the command lines; the page shows them and composes none | **done** — the folder answer carries `set_to` (its first prep's machine, in the tab's word) and `prepped` (each prepped stage with the prep entry's own sentence); each stage's lines come from one door, `/api/task-setup/commands`, composed by the terminal's composer (`commands.stage_lines`, `target_flags`): a launch line per mode where the config sets none (D11), no `--target` once the calculation is set, no prep line for a stage prepped — as `status` prints them. The lines depend on the person's choices, so they are their own door rather than part of the folder answer. The page composes none (`_targetArg`, `_bundleArg`, `continueFlags` gone, and with them `path-utils`' `relativeFromDir`, whose one caller was `_bundleArg`, and an unread `_choiceRequired`), shows the machine fixed once set — the others disabled, a line saying why (D12) — and a prepped stage's sentence in place of its two buttons. Found on the way: the page re-read its own Save twice, once outside the Save's fence (unit 4's announcement), and a late read undid the next edit — the page now re-reads its own write once, where it is made; the GPU-binding browser test now waits for the busy cover, as a person does (reproduced red by slowing the re-read); the preview's *the scheduler's own default decides* (D2's last); a fixture's machine record moved out of the calculation to where the probe writes it. `task-setup.md` § 2.1 / § 10 / § 11 / § 11.1, `preparing-for-another-machine.md` § 5, `web-api.md` (98 routes). The commands test extended, a browser test for the fixed machine, the saved-first browser test reads the prepped stage, the page-verb test retired; six mutations, each red; 1119 targeted green, then the 148 tests that open or read the page |
| 9a | B7 · W52 (8) | **every module on one floor, nothing importing upward** — an empty `jobset/__init__`; `PrepError` to `jobset/errors.py`; the engine seam, its hooks and the one engine→config map to `jobset/engines.py`; `launch_refusal` to `jobset/placement.py`; `continuation` on floor 4 and free of the conductor; the run's own records (`run.json`, the markers) to floor 1; `materialize`'s pass-throughs gone; `commands`, `plan`, `ask`, `migrate` given floors. Behaviour unchanged — `execution/architecture.md` § 2.1, § 3 | **done** — `jobset/errors.py` (`PrepError`), `jobset/engines.py` (the seam, its SIESTA hooks, `engine_seam`), `jobset/machine.py` (the record read `machine_record`, `require_activation`, the first prep's `set_machine` — was `resolve_target`), `jobset/placement.py` (`launch_refusal`, `AS_RESOURCE`), `molbuilder/runrecord.py` (a run's `run.json`, its conclusion marker, `.continued-from`, `.gathered-from`; floor 1, so `parse/` no longer reaches the layout); the stage namer is `materialize.stage_home` — today's rule, the door 9b turns to the disk; `materialize`'s pass-throughs to `paths` (`bench_container`, `job_dir_name`, `attempts`, `TRIAL_PREFIX`) gone, every caller asking `paths`; `jobset/__init__` empty. Nothing below the two entries imports the conductor: `continuation`, `prep_inputs`, a spectra reader and launch's one request reached it before. Found on the way: unit 7's `deck_spin` globbed `*.fdf` — it asks `runfiles.find_by_role`. The 114 non-engine files touching a moved name (2158 tests) and the 7 engine end-to-end files (37) green |
| 9b | B8 · M2f · M2g · F4 · M1 · M2 · M4 | **one door per fact** (`execution/architecture.md` § 3.2, A15): a stage's number and folder, read from the disk (F4); prepped; launched; how a run ended (F3) and the one rule for a run to build on; the restart-file list in effect for the calculation (M2g); the GPU request; `job-set.json` and `run.json` through `persist`; the template (M1); the shape asked (M2); the machine record read once; the resolved ladder composed once; the transport record through the doors (M4). Where two answerers disagreed, the door's answer is the one kept — each a case-table row | **in progress** — **part 1 done: a stage's number is its files'** (W38 F4): `materialize.ladder_homes` / `stage_home` read the numbers off the disk (`paths.stages_in`, `identity.parse_token`) — a stage with files keeps its number, one without takes its place when free, else the next after every one in use; `#N`, status's rows, the stages a verb offers (`described_refs`, was `commands.enabled_refs`), the transport rungs, the bias chain's folder and the Task setup plan (now sent the folder) all ask it; the plan's merge matches rows by `stage_key`. Two protocol rows (road keys `removed`, `added`, `made`); one mutation, both red; 2324 targeted green. **Part 2a done: how a run ended, and the run to build on** (W38 F3): one door, `runrecord.ending` -- the run's own marker at its newest run index, else SIESTA's `0_NORMAL_EXIT` in a folder no wrapper of ours ran in, holding one deck (SIESTA deletes it as it starts, `siesta_init.F`) -- asked by status, the default continuation, the frequency stage, the transport gather and citation, launch's re-launch question and the monitor; `continuation.usable` (ended on its own with exit code 0) the default of every hand-over -- a frequency stage and a transport rung took a run that failed until now; status's state is built on it: finished is exit code 0, failed any other or an output's stop, and an output that ended is `running` until the job concludes, `failed` once its monitor saw the process go. `run_status` and `Shape.run_basename` (was `stage_glob`) narrow by the run's name; `runrecord` travels in the monitor bundle. Case table `how_a_run_ended.toml` -- `run_records.toml` since 2b -- (7 rows, road keys `ran`, `calculation`, the table's `machine`); four mutations, each red; the private-probe tests retired; 3385 targeted + 81 engine end-to-end green. **Measured on `projects/`:** 13 of 76 run folders now read `running`/`failed` where they read `finished` -- all from before the wrapper wrote its marker (2026-08-28), 10 in `Au-BDT-Au.old` -- — a consequence, not a question (user, 2026-10-03: *"I don't care about old runs"*). **Part 2b done: launched** (W38 F2): one door, `runrecord.launch_record` -- the attempt's `run.json`, a flat stage's own, the newest stage record for a flat folder asked whole -- read through `persist` with the schema checked; one that does not read is `LaunchRecordError` naming the file, never launched or not (it read `{}`, *launched, the details lost*, while launch's gates asked `was_launched`, the file's existence): status says *unknown* on the row, prep's attempt reuse and every launch gate refuse in their own errors, the citation refuses, the Run panel degrades. `write_launch` and `.gathered-from` write through `persist`; `was_launched`, `read_run_launch` gone. `job-set.json`'s last raw reader -- the sweep-plan parser the Results picker asks -- reads through `JobSet.load`. The table is `run_records.toml` now (+1 row, road key `launch_record`); two mutations, each red; one test of the retired rule retired, six fixtures write the record launch writes; 4012 targeted + 105 engine end-to-end green. `.continued-from`'s three hand-written writers go with unit 10's one `Continuation` (M2h). **Part 2c done: the transport record through the doors** (W38 M4): a rung's outputs are read newest RUN first, by the number each carries (`runfiles.find`), never by file time -- the rung facts and each point's current; the product rung asks the run-state door by its run's name with its launch record (one that does not read is said); a scan's rung speaks from its first point not finished, in the scan's order, else its last -- status's rule, asked of `ending` -- where the newest attempt by file time spoke. One case beside the record's measured fixtures; one mutation, red; 3921 targeted + 5 Transport-tab end-to-end green. **Part 2d done: the Results viewers follow the run** (M2f, `web/results.md` § 4.1): the server sends `run: {state, detail, live}` -- the one door's answer for the run the file belongs to (`parse.dirs.run_answer`: its folder and the run its name reads back to) -- with the watch load, every quiet watch poll and the spectra load, and when the run is no longer live it reads the file again if it changed, so a viewer stops on the file's last state; both viewers follow while the run is live (queued or running) -- the trajectory's two-finished-ticks buffer and the spectra's every-phase-complete rule gone -- and the trajectory badge reads the run, from one renderer drawn by every answer; each output's ending is kept by file version (`_run_ending.ending_of`), so a quiet poll costs a look at the folder, and an output that states how it ended takes its own parse's reading -- the parser and the ending read an output one way (`siesta_reader.read_output`) -- so a load reads its file once (the run's state had made it twice). Contracts: `results.md` § 4.1, `trajectory.md` §§ 4-5, `spectra.md` § 7, `web-api.md`, `model/parse.md`. Tests: the trajectory settle as a 12-row case table, its poll loop (4), the server's answers (6, the race among them), the spectra follow in the browser (a run concluding, and one killed between phases); six mutations, each red; the copied-alone viewer test retired (an output without its run's records is a run that never concluded), two browser fixtures model the live run (a launch record); 6412 targeted, 224 engine and browser end-to-end green -- the one red, `test_spectrum_form_locks_e2e`, stale since K7 and fixed on its own. Checked on the dev server: a concluded run's badge reads Finished from the run, not followed. **Part 3 done: one restart-file list** (M2g, W36 ⑧): `warmfiles.warm_list(engine, kind, base)` -> `WarmList` -- the calculation's own `warm-files.toml` first, else molbuilder's: its restart files and whether the kind resumes -- answers every reader: a job's declaration (the three `stages.py`), status's warm-file column, the run script's check of what it was handed and the bias chain's hand-forward -- both written at prep from it, where the run script knew only molbuilder's list and the chain spelled `{label}.TSDE` by hand -- and the parser's and the identity check's suffixes; a kind the file has no section for is refused, naming the ones it has; `runwrap`'s module tuples gone. `STAGE-PLAN.md` says whose list it is; the Task setup plan shows the list in effect, its file and how to change it (`task-setup.md` § 7.2). Case table `restart_files.toml` (4 rows, road keys `own_warm_files`, `status_lacks`); four mutations, each red; 5173 targeted + 233 engine and browser end-to-end green. **Part 4 done: the GPU request** (W38 F6's request half, D17's first half): `jobset.model.gpu_request(resources)` -> `GpuRequest(uses, count)`, consistent by construction -- a GPU run with no count and a count for a run that does not use the GPU are refused -- asked by every reader: the header and the run script (the deck scan `_fdf_requests_gpu` and its reader `_wants_gpu` gone), launch -- its sides, shelves, request and queue table (`_job_wants_gpu`, `_gres_count` gone) -- the bench report, and the Task setup card, which shows a refused request on its devices row; prep asks it before anything is written (`prep_inputs.run_gpu_request`, of the job `resolve` makes -- `run_uses_device`, a second reading of the template, gone), by the run card or `--gpus`, either engine; `resolve` always carries the job's own `use_gpu`; a bench group's envelope carries both halves; `canonical_gres` reads a count only (a stored card was read as its count). Contracts: `gpu.md` G5, G7, § 6, § 7; `architecture.md` § 3.2; `template.md`; `generator.md`; `job-system.md`. Tests: three rows of the GPU table (the card's count, `--gpus`, PySCF -- one also that nothing was written) and the card's four devices rows; the deck reader's tests retired with it; six fixtures that made a GPU job by its deck alone state the request; three mutations, each red; 6912 targeted green. **Retired 2026-10-03** *(user: "retire all 245")*: every test that laid a run that never happened -- 282 test functions and 10 case-table rows: `support.road.a_finished_run` and the road's `ran` key, `run_records.toml` whole, two of `restart_files.toml`'s rows, the viewers' server half (`test_viewers_follow_the_run.py`), `test_stage_continuation.py`, `test_run_status.py`, the transport suite's forged junction citation and every test on it (the Transport tab's end-to-end tests among them), hand-written markers, launch and monitor records, invented outputs and fabricated contract records elsewhere -- `process/testing.md` § 6, *a run is made on the road, or not at all*. **Coverage measured after it (2026-10-04):** each retired behaviour broken once and the remaining suite run, 4,688 targeted and 60 real-engine tests: a failed run read as finished caught by one real run only; a launch unseen, a calculation's own restart list ignored and a relaxation that never ran built on still caught; six uncaught -- a run that ended with an error built on, a live run not followed, a stage continuing from its oldest run, a transport rung taking an unconcluded rung's files, our runs losing their `.MD.nc`, a deck guessed among several.  **Rebuilt as framework tests** (user: *"rebuild 1-3, 5, 6 now; transport with Q4-Q7"*; *"rely on more api and framework test rather than ... e2e"*): `tests/data/hand_overs.toml` -- three rows on the road (a fourth, a run waiting in the queue, removed the same day: user, *"no need to test running a sbatch ... that's my job"*), our wrapper running the suite's stand-in engine, which ends as a row says (road keys `stand_in`, `run_answer`; `--target` passed to `prep` alone) -- and the `.MD.nc` pairing on a real flat H2 run of ours kept as a measured fixture (`tests/fixtures/siesta_flat_h2`); five breaks, each red against its own row.  The sixth, measured on that run, is a defect, not a gap -- D19.  Transport's own goes with Q4-Q7, on a minimal junction (user, 2026-10-04). **Part 5 done 2026-10-05** (user: "go ahead with 1"): the template read through one door, `find_template(base, label)` -- the folder's one template, named for the label, or refused by name -- by prep (its described route, the transport resolve, the root gate, the preflight), `jobset migrate`, the run inputs, continuation, the save route's preflight and the Task setup folder answer (its label the description's, else the hand-over's through the normaliser the hand-over named it with); a benchmark trial's run script told the trial's own label (W38 M5); M2 and M3 found built. *The machine record read once* and *the ladder composed once* are unit 10's one plan (B1). The bare-function test of the two-template refusal retired, the refusal pinned at the prep entry and the tab; two tab tests now write the hand-over the browser writes. 1346 + 44 targeted green |
| 10 | B1 · B2 · B3 · D10 · M2e · M2h · W52 (11) | **prep is plan → save → write → record** (`job-system.md` § 5.0, `script-preparation.md` § 3): every check and every decision — decks, wrappers, the plan merge, the attempt and what it receives — made with nothing written, from one table of steps, transport's and a vibration's own as optional steps declared by the kind; then the save; then the writing, which decides and refuses nothing; a refusal writes nothing but its ledger line, so nothing is put back. The preview is the same plan, stopping before the save; the route composes nothing. The pipeline log always (M2e); one `Continuation` for every hand-over (M2h, the freq geometry, the transport gather) | designed 2026-10-03 — the user's word; **in progress 2026-10-05, in five parts, each a commit with its rows and mutations** -- 10c is taken before 10b, so the plan-then-write split is made once, on one path (told to the user): **10a · the pipeline log, always** (M2e) -- every prep writes it, from both doors; `--pipeline-log` and the parameter it set go; the catalogue's row says prep writes it. *Done 2026-10-05*: the card's rows now hold it (`the_catalogue.toml`, both shapes; one mutation, both red); the two tests of the switch retired; 2584 targeted green. **10b · the run arm planned, then written** (B1, D10, D16) -- each step returns what it would write instead of writing it: the calculation's copy of the machine record, the permutation, the data files, each deck's text with the reader's section merged and both gates passed on that text in memory (`script_emit`'s check reads the text the file will hold), the progress seeds, the wrappers and `STAGE-PLAN.md`, the merged plan, the attempt and what it receives, the pipeline log; one plan, which the writing half carries out in order, `job-set.json` last, deciding nothing; the machine record read once and the ladder composed once; every refusal before the save (D10), so a refused prep writes only its ledger line and `_as_it_was` / `_put_back` go (D16) -- `script-preparation.md` § 4.5's *a refusal ends the file at the step that refused* is a restatement § 5.0 rule 3 retired, corrected with it. *Done 2026-10-05, but for the reads*: one `jobset.planned.Plan` -- each file's text or bytes, each copy, move and removal, in the order the steps decided them -- which the steps read for what an earlier one will write (the wrapper its deck's text, the attempt the stage's files, the shared package the data files), and which every reader of the calculation's pseudopotentials reads through `pseudos.PLANNED` (the data-files screening, the settings gate); `prep_stage` plans, saves, then carries the plan out, `job-set.json` last; a refusal writes only its ledger line -- `_as_it_was`, `_put_back` and `PrepError.partial` gone, and the protocol's refusal row names what a refused first prep does not leave (`prep_protocol.toml`; one mutation, red). The pipeline log is held and written with the plan, so its `!!` column went: a hook that raised ends the prep, which writes nothing, and says whose it was in its note (three tests of the old log retired, one narrowed). Checked by prepping a SIESTA ladder (two stages, the second cold), a flat stage, a PySCF stage, a vibration's relax and the five transport rungs with the code before and after: every deck, wrapper, attempt and marker the same; the log says each deck is checked *as it is written*; on a first prep the provenance block lists the calculation's copy of its machine's record as absent -- the machine's own record answered, and this prep writes the copy (`configuration.md` § 2.2: the first row found is the record that won). Found by that comparison and by no test: the shared package was named from the disk, so a first prep's run folders got no pseudopotentials, and the wrapper read its deck from the disk, so a first prep's run script said the deck's restart keywords could not be read -- both fixed, each with its row (`the_catalogue.toml`, `restart_files.toml`; each mutation red). `script-preparation.md` § 3, § 4.3, § 4.5, `stages.md` § 7.2, `job-system.md` § 5.3 and `web-api.md` corrected; 5,148 targeted tests green. *The reads, done 2026-10-05*: steps 1 and 2 are answered once and handed on (`prep.Resolved`) -- the entry reads the description and its template at checkpoint 1 and the machine's record at checkpoint 4, where its activation is now checked (§ 5.0's row; it was the body's, after 4a), and the stage is resolved once there (`_resolve_stage`, the allocation folded in it), so the GPU request (G5) and the placement are asked of the job the steps write; the body reads none of the three files again, the snapshot is the record read, the bench's grid takes the record and the template in hand, the description's shape is handed to the layers below (`materialize.shape_of`'s own rule), and the Task setup card's GPU answer goes through the same resolve. Until then a run's prep read the description three times, the template four, the record three, and resolved the stage twice. Checked by prepping the same calculations before and after: every file the same; 5,265 targeted tests green; the fold's move broken once, red. `prep_run_inputs` still folds the description's queue, wall and memory for the card's rows -- it goes with 10e, when the card asks the entry. **10c · transport folded in** (B2) -- `_prep_transport` goes; compose, the electronic state, the points and the gather are optional steps its kind declares, in the one table (`script-preparation.md` § 3.0). *Done 2026-10-05*: one conductor; what a kind adds is one record, its `Rung` (`RUNGS`, keyed by engine and kind) -- the transport rung's compose, electronic state and points, its data files from the citation, its job's restart files and program -- and the SIESTA vibration's held-first copy and relaxed geometry moved off the conductor's `if` into the same table; the gather stays the entry's until 10d. Transport takes the shared steps' answers to its drift: a progress seed beside each attempt's deck, each warning once, the config provenance in its log, its job from `_job_for`. Checked by prepping all five rungs of the minimal junction, a two-point scan, with the code before and after: every deck, wrapper and attempt file the same, the seeds and the log's provenance the additions, the jobs now carrying SIESTA's optimizer trait (read only by `requires_same` rules, which transport's have none of). Found: nothing short of an end-to-end run reaches a scan's points (the points dropped, every transport test stayed green) -- its tests are Q5–Q7's, on the minimal junction; `job-contracts.md` § 3.5 and `template.md` § 9.2 said transport decks skip `prepare_deck` and carry no reader's section, false since 2026-09-16 -- corrected; the GPU contract's PySCF row still read the deck text #1 moved (`to_gpu()`), outside that commit's targeted set -- corrected, with `gpu.md`'s spelling. 4,542 targeted tests green. **10d · one `Continuation` for every hand-over** (M2h, D28) -- the frequency stage's relaxed geometry and a rung's gather are continuations, planned at 4a; `STAGE-PLAN.md` lists the files the carry takes; one writer of `.continued-from`. *Done 2026-10-05*: a vibration's force-constant stage builds on its `relax` through the one hand-over door (`continuation`: the newest attempt that ended on its own with exit code 0, or a `relax` run named with `--from`, taken as said; no other run, and no `--cold`, while the ladder holds a `relax`; every situation in `vibration.md` § 5.2a's new table -- *user: "copy like any hand-over, but keep in mind we also have a structure-already-optimized option. you need to guarantee the logic is complete for all situations"*), and its hand-over copies like any other: the relaxed density warm-starts the reference SCF, as on the flat layout; its `.continued-from` names the run; the rung reads the geometry from that run and picks none of its own; Task setup offers it *Continue from*, without the structure. A transport rung's gather is decided at 4a (`gather_sources`, `transport_inputs`) and copied into the attempts the steps open. The files a hand-over carries are counted where the plan's row is merged, and `STAGE-PLAN.md` says them under its table, its column headed as each stage's declaration (D28; *user: "Declared column + a carry line"*). `.continued-from` has one writer and one reader (`runrecord.write_continued_from`, `read_continued_from`), prep's and launch's re-launch alike; the body takes one `Continuation` where three loose arguments stood. F9's *summarize reads the record*: nothing re-picks a relax run -- the force-constant deck carries the relaxation's record to the finish (§ 5.3), and a sweep compares each stage's own result. Checked: the same calculations before and after, only `STAGE-PLAN.md` changed (the column's heading, the line under it); four rows in `hand_overs.toml` (the carry line; a force-constant stage refused while `relax` is unlaunched, and while it failed; `--cold` refused) and one API-level check of the default on the measured relaxation (`tests/fixtures/siesta_relax`, read in place); three mutations, each red; 5,308 targeted green. The real-engine vibration end-to-end, which measures the new carry, waits for the user's word. **10e · the preview is the entry** (B3, D15) -- the entry stops before the save and answers the plan; the route's own assembly (`_plan_chosen`, `_emitted_launch`, `_plan_continuation`, `_plan_prepped`) and the run card's fit check through the bench grid go; Prep names the preview's plan and is refused when the plan differs; the Task setup door takes what the entry takes, `#N` and a stage name in any case. *Done 2026-10-05*: `prep_stage(…, preview=True)` stops before the save and answers the plan -- what it would write, what the stage builds on, A13's end point line for line as the header and the run script the plan holds carry it (`scheduler.emit.Directives.lines_of`, `runwrap.stated_counts`: each writer's own reader), the queue, wall and memory asked -- or the refusal prep would give, with nothing saved, written or recorded; Prep names the preview's plan (`Plan.identity`: every file it would write with its clock readings masked -- measured, two plans of one stage a moment apart differ in nothing else -- and each copy's source) and is refused when the plan it makes now differs. One answer, built from the plan before the save, serves both (the launch agreement read from the planned deck). The route composes nothing: `_plan_chosen`, `_emitted_launch`, `_plan_continuation`, `_plan_prepped` and its own stage and machine checks went, so the door takes `#N` and a stage name in any case (D15); `prep_inputs.run_gpu_request`, with no caller left, and `prep_run_inputs`' own fold of the description's queue, wall and memory went -- the allocation is folded once, at step 2; the run card's fit panel, which asked the bench grid with one-point axes, went (B3). Tests: one through the browser's door (the preview writes nothing; a stale plan refused; the previewed plan taken; `#1` and `COARSE`), one mutation red; the bench card's preview tests read the preview's lines; two machine tests made to reach the machine (one had passed on a refusal for a missing template); the card's four devices rows retired, the GPU contract's rows holding them; two browser end-to-end tests rewritten for the page, unrun -- the end-to-end batch waits for the user's word. 6,051 targeted green. **Then the unit's review** *(user, 2026-10-05: "a focused agent review with fresh eyes with full text and full code to cross check and validate findings")*: reviewers who have not worked on unit 10 read the whole contract text and the whole code of prep, each finding re-read in the code before it is acted on. **Review round 1, 2026-10-05** -- four reviewers with fresh eyes, each on one part (the plan/write split; reads and checkpoints; the hand-overs; the preview and the Task setup door), full contract text and full code; 61 findings, each read again in the code before it was fixed. **Fixed in one revision:** the provenance is built once, at checkpoint 4, from the record in hand and the scopes it was looked for in (a named target's included), and `STAGE-PLAN.md`, the pipeline log and the ledger's *prepped* line carry that one table (a first `prep --target sol` named this machine's record, never sol's); **a run is admitted on the queue it names at checkpoint 4** -- its cores, GPUs, memory and wall, by the binding launch asks (`placement.admission_refusal`, one request builder `placement.request_of`) -- the half of § 6.0's placement the preview's retirement of the card's fit panel had assumed (D17's second half, done); a bench's GPU ask bounds its trials (`resolve._check_fits`) and a trial on the CPU asks none; no folder made while planning (`trial_work_dir`); the preflight's ledger line after the save; `unread` refused first; a template refused at checkpoint 1 for every kind (transport's exemption and its dead branch gone); a `molbuilder.json` that does not read refused, not a 500; the template read once on the gather's path and by the hand-over door; the run card folded once (`_under_description`); `Resolved` carries the stage's token, the target, the sweep and the provenance; the hand-over door's catch-all narrowed; one stamp mask for a plan's identity and for `same_calculation` (`deck_record.without_stamps`, the generator's version included); a copy's or a move's file stamped when planned; the stale-plan refusal says what it knows; `writes` lists what the plan leaves, and the preview shows it; the answer's `continuation.carries` are the attempt's; a PySCF run script states its threads by the name A13 reads (`_omp_threads_default`); a bias scan's preview has its launch; the page shows a refused preview's findings, retires Prep while a new preview runs, and drops an answer that lands after another folder was opened; a Prep naming no plan is refused at the route; **transport `--from` / `--cold` refused** (C2); **no `relax` and not stated relaxed decided at 4a**, so `status` says it before the prep (C3); "takes nothing from another run" said on both doors (C4); the unconverged-relax remedy in an order that can be followed (C6); the gather recorded in the ledger (`gathers`); the newest attempt passing all three gather gates; status's row says what a stage declares (C16); the flat refusal and STAGE-PLAN's order line worded by kind; dead code (`calcdirs.write`, `set_machine`'s and `mark_run`'s no-plan branches, a trial's markers written twice); the docs (`job-system.md` § 5.0, § 5.4, § 6.0; `configuration.md` § 2.2; `generator.md` § 4.3; `task-setup.md` § 6.2b, § 11, § 11.1; `web-api.md`; `architecture.md` A12, A13; `script-preparation.md`; `stages.md` § 7.2; `vibration.md` § 5.2a, § 5.8; `transport.md`; `project-layout.md`). Six rows of the GPU and launch tables moved their fit refusal from launch to prep (the rule changed, the rows with it), and two others now state a wall their queue admits; rows added: the bench's GPU bound, a CPU trial's no GPU, the unrelaxed structure at 4a; an API-level check of the transport refusal (the road describes no transport calculation); six mutations, each red; 4,679 targeted tests green, the page's end-to-end tests unrun (their batch waits for the user's word). **Carried to unit 11** (launch's): a single-bias device launched again gets no electrode `.TSHS` (C1); a flat re-launch of a stage set `restart: clean` records a continuation (C5); the bias walker copies the device's whole declaration point to point, past `.gathered-from` (C8); a stopped force-constant stage is told "prep anew" by status while launch continues it (C10); launch's carrying note lists names that may not exist (C15); launch's own request builder (`submit._place`) -- one with prep's `request_of`, which counts a PySCF run's cores; the placement's record on the job (§ 6.0). **Approved 2026-10-06** *(user: "ok")*: the direct doors below the entry -- `prep_calculation` and `prep_jobset` called without `Resolved`, by tests alone, reading and deciding on their own (B10, C19) -- retired, their test calls moved onto the entry. *Done 2026-10-06*: `prep_calculation` takes the entry's `Resolved` and plan, and nothing else decides (`_read_and_resolve` gone); `prep_jobset` takes the plan, the shape, the provenance and the record (its own `set_machine`, plan and readings gone, and the shared-script copy only a hand-built set reached); `set_machine` takes the record the entry read. Of the 44 test calls: the described ones go through the entry (`prep_stage`) or the road verbs; the rules the hand-built ones held are rows -- `prep_protocol.toml` (the stage named is the stage written, a template value, the retry budget and the run card's over it, the reporting policy, a stage the ladder does not hold), `launch_values.toml` (a machine never probed); the road takes a description's `notify` and a machine never probed (`support/road.py`); 29 test functions retired for 11 rows, each pointing at the row or test that holds its rule; mutations red (the notify fold, the stage resolved, the run card's pins, a queue in the inner run script, the run index). **Open:** which file a SIESTA relax run's verdict is read from -- the hand-over line reads the progress log first (`continuation.read_run`), the force-constant deck's record the `.out` it reads the geometry from; `parse.md` § 5b.1 says the run door's choice (C17). *(C17 settled in round 2: the run door's order.)* **Review round 2, 2026-10-05** -- three fresh reviewers (the entry and the plan; the hand-overs; the preview, its door and the page) on the revised code; ~40 findings, each read again in the code. **Fixed:** a bench cell past this prep's own ask (`--np`, `--cpus-per-task`, `--gpus`) is crossed out by name beside the cells no queue holds, never the whole prep refused over a cell the machine proposed (round 1's bound had done that; `generator.md` § 4.1, § 4.3a); a prep's `dirs` and its ledger line name only its own jobs' folders (`materialize(only=)`; every prepped stage's were listed, and missing shared files copied into them); the provenance rides every answer, a preview's too, and the page shows it -- its folder card shows `molbuilder.json` alone, since which machine record answers turns on the machine named; a preview is rendered as Prep's answer is, worded as a preview (the deck's own checks, the launch disagreement, the attempt and gather it would make, which files answered), and the page's own summary of the run card is gone, with the answer's `allocation`, `chosen` and `bench_axes`; Preview and Prep hold the page's fence; a Continue-from choice the folder no longer offers is dropped; a calculation set to its machine is prepped with no machine named (its copy answers), the answer naming the machine (`machine`); the terminal says the resources and the agreement for flat and bias-scan preps too; a rank count for PySCF refused at prep; a benchmark of a force-constant stage says, and records, the relax run its trials are written at; every refusal that waits on another stage prints a command that can be typed -- launch it, let it finish, or the rollback, never the prep of a prepped stage (`continuation.state_remedy`, shared with the default's refusal; the gather's gates, the relax geometry's, the transport record's); Task setup's Continue from says why a force-constant stage with no `relax` is refused; the relaxation remedies lead with the restore, said conditionally; a run's verdict read in the run door's order, its output first (C17: the line and the force-constant deck read one file); the flat line names its run, and `.continued-from` takes that one answer; `--from` naming another stage's run at a vibration's `relax` refused; the junction's refusal names the rollback; R7's premise (prep knows no wall) corrected with its restatements; stale texts and a dead branch. Rows: the bench GPU rows (a declared cell, a proposed cell, crossed out), PySCF's rank count; six mutations, each red; 4,681 targeted tests green, the page's end-to-end tests unrun; the browser door's tests now change what the plan is made from between preview and Prep, read the preview's provenance and a PySCF run's A13, and compare a run's previewed header with the one written. **Carried to unit 11**: a benchmark's placement -- its trials' headers name the prep's queue unadmitted, and its sides are named again at launch (§ 6.0 says prep decides it; unit 11 records it); launch's request for PySCF. **Carried to Q5–Q7 (transport)**: `status` does not ask the gather's gates (it says "prep it" where prep refuses); the gather re-renders every upstream deck per bias point, printing their warnings outside the answer. **Ruled 2026-10-06** *(user: "the handover should call a unified api that tells the true status of previous stage that views all facts from exit code, log etc. thats the consistent effort and reason can be detailed for user in message")*: the hand-over's default asks the one run-status door `status` asks -- the exit code, the engine's output, the monitor -- and takes the previous stage's newest run only when it says *finished*; otherwise it refuses, the reason from the run's own files; `--from` still takes any run named. |
| 11 | B5 · M2k (F1, F6) · M5 step 5 | **launch is plan → show → ask → send → record** (`job-system.md` § 6, `submission.md`): one launch entry deciding everything once — the work, its submissions, each member's attempt and continuation, the placement, the exact lines; the send checks the folder is still the one shown, and carries the plan out through one sender -- a benchmark's trials walked by the benchmark's own script; a bias scan and step 5's groups transport's own mechanism, with the transport work (user, 2026-10-05: "bias scan is its own mechanism"); the placement decided at prep, admitted there with the real request (queue, wall, memory, GPUs) and recorded with each value's source, read by the header, the card and launch | designed 2026-10-03 — the user's word; **in parts (2026-10-05), each a commit with its rows and mutations, then the unit's two review rounds** -- **11a · the placement, recorded at prep** (§ 6.0's placement paragraph, M2k F1/F6, D13): each job carries, in `job-set.json`, the placement prep admitted -- the queue (name, partition, qos), wall, memory, ranks, cores per rank, GPUs, each with its source (flag, run card, description) -- read by the header (no second binding), by launch (a launch flag changes one value, admitted again, recorded in `run.json` and the ledger) and by the Task setup card; one request builder for prep, launch and the queue table (`placement.request_of`; launch's `_place` counts no cores for a PySCF run, and the table asks a third way); the card's queue choice fills neither wall nor memory (D13, `submission.md` S5); a benchmark's queue stays its launch's -- its shelves' headers are written there, its sides named and admitted there (closed 2026-10-06: nothing was wrong; the promise to move it to prep is withdrawn). *11a done for a run, 2026-10-05*: the admission at checkpoint 4 returns what it admitted (`placement.admitted`) and the job records it -- `placement`: the queue's name, partition and qos, and where each value came from, read off the one fold that fills them (`prep._fold_allocation`); the header renders that queue (`domain_pq`), binding no name again; launch takes the recorded queue and asks a queue prep's one request (`placement.request_of`, `one_process`), the queue table too -- a PySCF run's cores counted at launch for the first time; both doors say the placement in one line (`placement_line`), and the card shows a prepped run's from its job's record; the queue card fills nothing (D13). Rows: the placement said, a PySCF run's cores at launch; the card's line read from the folder; the Job's field set (`placement` decided); three mutations, each red; 4,598 targeted tests green. **11b · one launch entry: plan, show, ask, send, record** (§ 6.0's table, D14): the plan made once -- the work, its submissions, each member's attempt and continuation, its placement, the exact lines, every script and header rendered in it -- shown, asked, then sent as shown: the send checks the folder is the one planned (its identity, as prep's), and refuses otherwise; one sender for the four send sites; `--dry-run` writes nothing, the ledger included; every refusal ledgered; a direct run's `launched` written at its start. *Done 2026-10-05*: one entry, `submit.plan_launch`, makes the `LaunchPlan` once -- its submissions and their members, every gate, the queue, the exact lines, what the send writes (a re-launch's attempt through the one opener, planned; a flat re-launch's marker; a shelf's and a chain's scripts) and every file it read where it lies, by size and write time; the terminal shows it and asks once, and `send_launch` sends that same object: it makes the plan again and compares line by line -- a file written since, the stage launched since or its run ended since is refused, naming it, nothing sent -- then writes, then sends each submission through one function (`_go`), written down as it goes, a run here when it starts. The entry writes its own lines (`refused`, `question`, `asked`, `launched`, with what the verb was told on each, `architecture.md` § 2.1), the verb the refusals it says before it calls the entry; a dry run writes nothing, a refused one too; `trial-picked`, `bench-grouped`, `bias-chain` and the dry run's `planned` went. A dry run is the launch's plan, so a launch to a queue on a machine with none is refused at its dry run too -- which found the printed launch lines offering `--mode submit` on a workstation, a line no launch could send (only the old dry run, planning less than a launch, took it): the printer offers the queue's line only where the calculation's machine names one (`commands.takes_a_queue`). C15 went with the plan: the note says what the opener copies. The one-job-at-a-time refusal went with the stage door's direct callers -- through the entry a sweep sent to a scheduler goes by shelf. Rows: `launch_protocol.toml` (a re-launch's dry run writes nothing; the verb's own refusal recorded; a refused dry run writes nothing; a run here recorded as it starts, the stand-in engine waiting for the line); the launch-values row moved to a benchmark's shelf, whose header launch writes (a stage prepped with no header is refused for the header first); the folder check API-level (one command cannot change the folder between its plan and its send); eight hand-built door tests retired, each rule held on the road or by construction; the benchmark tests that sent to a queue from a workstation given one; six mutations, each red; 2,701 targeted tests green. **11c · members and continuations** (C1, C5, C10, C15): a re-launch's hand-over is the one door's `Continuation`, recorded as prep's is (`continues`); a gathered rung launched again carries its gather and its record; a flat re-launch of a stage set clean records no continuation; status and launch give one answer for a stopped stage that does not resume; the note lists what is copied. *Done 2026-10-05*: whether a stage launched again continues from its own latest run is one fact of its job, `Job.relaunch_continues` -- its kind resumes and it takes something from a run -- which status words a stopped stage's next step by and launch refuses a second launch by, in one sentence (`continuation.no_relaunch`): a stage that does not continue from a run of its own is not launched again, its redo the rollback and a prep anew (`job-system.md` § 5.3's rule; C10: a stopped force-constant stage was told to prep anew while launch continued it; C5: a flat stage set clean was launched again and recorded as continuing). One that does continues through the one door (`continuation.relaunch`): its own latest run, whatever its end, one `Continuation` as prep's -- the same line, the next attempt opened through the one opener, recorded (`continues`) -- and a transport rung's next attempt takes the inputs gathered for its run, copied with `.gathered-from` (C1; its row waits for the transport road, Q5-Q7, on the minimal junction). The status wire form says `relaunch_continues` where it said `resumes`. `job-system.md` § 5.4 (*A stage launched again*), § 5.3, § 6.0; `architecture.md` § 3.2 (a re-launch read `usable`'s rule there, which it never kept); `project-layout.md` § 2.3.4. Rows (`hand_overs.toml`): a re-launch continues from its own run, recorded as prep's; a stage whose kind's rerun starts over refused, and status saying the same; a flat stage set clean refused, recording nothing; the road takes a calculation's own list's `resumes` and a verb's decision in the ledger; three mutations, each red; 2,775 targeted tests green. **11d · a benchmark's walk** (§ 6.0, *A benchmark's walk*): planned as one sequencer for a benchmark's shelf, the bias chain and the direct benchmark loop (§ 6.0's text of 2026-10-03) -- **ruled otherwise 2026-10-05** *(user: "bias scan is its own mechanism - this is parameter sweep, not some ... computation resource experiment, why ... do you think benchmark should share anything with it?")*: the bias scan's walk is transport's own and stays as it was; this part is the benchmark's. *Done 2026-10-05*: one writer for a benchmark's walk, `submit._bench_walk` -- its shelf sent to a queue, and its unlaunched trials run here, now one walk (`_plan_bench_here`; they ran as one process after another from Python), `--trial-timeout` taken here as on a queue. Rows (`launch_protocol.toml`): a benchmark run here walks its trials in one walk; its bound reaches the walk (the road reads the walk a launch wrote, `walk`); three mutations, each red; 3,035 targeted tests green. **Review round 1, 2026-10-05** -- four fresh reviewers (the placement; the launch entry; a stage launched again; the benchmark's walk and the tests), full contract text and code; ~35 findings, each read again in the code, then triaged by the user's rule *(2026-10-05: an accident -- a failed, interrupted or still-running execution -- earns an honest message pointing at the checkpoint, never a mechanism)*. **Fixed:** the queue decided by the entry, not the verb (`submit._the_queue`; floor 7 never works out a launch); a run here shown and asked as a submission is *(user: "show and ask every time")*, and a launch with nobody to ask refused, not exit 0 (`ask.Said.asked`); the honest refusals -- prep's fit refusal names where each value was stated, launch's names the rollback for ranks and cores, a missing header names its cause (`--no-sbatch` or a workstation record), a named launched trial refused when sent or run here (a question to the scheduler says it already ran), a refused shelf's hint no longer says to re-send it by hand (its dead GPU branch gone), the hand-over's remedy no longer says "launch it again" of a stage launch refuses (`continuation.not_launched_again`), `restart: clean` no longer named for a rung that cannot have it; the PySCF vibration's restart files say `resumes = false` (its deck reads no checkpoint back -- a stopped one was "continued" and recomputed from scratch); Ctrl-C, a lost terminal or a cancel stops a benchmark's walk (no further trial); the card shows a queue's limit under an empty field; dead code (`prep._under_description`) and its four tests; tests of states only a hand-edited file makes retired (a deck rendered for another launch, a trial stripped of its shape, an unknown mode); the Job field-set test (an API shape); the docs (`job-system.md` § 5.4, § 6.0, § 7; `submission.md` S4; `project-layout.md` § 1.5, § 1.5b -- no cold `run-x`, the redo is the rollback -- § 1.6.4, § 1.6.5; `generator.md` § 4.3a; `scheduler.md` R5, § 6, § 7; `job-contracts.md` § 4.2, § 6.1, § 6.2). **Dropped** (accidents, the user's rule): a launch over a run still writing, a run that died at startup leaving nothing, a walk's unreached trials. **Parked, one line each** -- *all five fixed 2026-10-06 (user: "fix all five"), below*: the printed launch line for a calculation set to another machine collapses on this machine's `launch.mode` (typed there, it is refused for want of `--mode`); the walk's trials are planned twice -- `_plan_member` here, `_plan_shelves`' own `_launched` filter on a queue -- agreeing today; three shelf tests in `test_prep_bench_fold.py` stand off the road (a stand-in `sbatch` that refuses one shelf would let them on); a launch flag's wall and memory are recorded in `run.json`'s line only, not as values; rows for a walk's passed-over trial, its bound on a queue and a failed trial. *Fixed 2026-10-06*: a printed launch states its mode unless the calculation is launched on this machine and its config names one -- the machine it is set to read through one door, `scheduler.record.calculation_machine` (three readers read the copy themselves); a benchmark's walk takes its trials from one planner, `submit._bench_trials`, here and on a queue's shelves alike, each line built once (`_walk_of`); the three shelf tests are rows, a stand-in `sbatch` refusing one shelf (`sbatch_refuses`); `run.json`'s `placed_on` keeps the wall and memory sent, and the ledger's lines the flags as typed; rows for a trial passed over (here and on a queue), the bound on a queue, a failed trial (`walk_log`). **And the rows found a naming defect**: from 2026-08-21 (e9cae2bf) a benchmark on a machine with no GPU named its trials without G (`K1C1`) while `job-contracts.md` § 6.3 names every trial `bench-G<gpus>K<ranks-per-gpu>C<cores>` and its shelves said `G0` -- a naming change nobody asked for *(user, 2026-10-06: "we asked for procedure unification, never asked for any naming changes")*; every trial carries G again, and the tests that pinned the G-less spelling now read the contract's. Mutations, each red: the run.json values, the ledger flag, the printed mode, the launched check (now caught by the passed-over rows, which nothing caught before), the shelves' tolerance. **The browser e2e walk** waits for the user's manual run, or Wednesday 2026-10-07 (user). **Review round 2, 2026-10-06** -- three fresh reviewers on the revised code (the launch entry and the placement; a stage launched again and every message about a stopped stage; the benchmark's walk and the tests), the user's triage rule in their brief; ~20 findings, each read again in the code. **The one decision** *(user: "save the geometry, such that it matches the siesta behavior")*: **a PySCF relaxation keeps the geometry each geomeTRIC step reached** in `_optimized.xyz`, as SIESTA its `.XV` (`relax_policy.relax(keep=, resumable=)`), so a rung stopped at its step limit or its wall, launched again, continues from where it got to -- it restarted from its input until then while status and launch said it continued; a stopped rung still hands nothing on (a hand-over takes only a run that ended with exit code 0). **Fixed:** the ledger's question said as asked (a run here read "follow a run that never concluded"), the launch with nobody to ask refused by the entry, written down; launch's "states no wall" names the launch's flags, not task.json; `--mode ask` on a benchmark prepped with `--no-sbatch` refused for the header it lacks; the last two "ranks and cores at prep" messages; a benchmark's queue recorded as named, not admitted; the halt message and the wrapper's after-budget lines no longer advise a re-prep or a `--continue` inside a finished attempt; `vibration.md` § 5.5, `project-layout.md` § 2.3, `job-contracts.md` § 4.2a; the walk's log one path (`Submission.log`); `--only`'s walk words only for a walk. Rows: a benchmark keeps the calculation's wall and queue (`launch_values.toml`); the halt and continue e2e tests now read the kept geometry -- run on the user's word (2026-10-06): 5 passed, and the kept geometry's mutation turns 2 red. **11e · M5 step 5's groups** (transport.md § 2a.7's default grouping: the seed and both leads as one submission, TD2) -- **moved to the transport work (Q5-Q7)** with the bias-scan design, 2026-10-05: it is transport's own mechanism, built on the transport road (the minimal junction). **carried here from unit 10's first review (2026-10-05)**: a single-bias device launched again opens an attempt without its electrode `.TSHS` (C1 -- carry the source attempt's gather, recorded, or refuse with the rollback); a flat re-launch of a stage set `restart: clean` records a continuation (C5); the bias walker hands the device's whole declaration point to point, past `.gathered-from` (C8 -- moved to the transport work's bias-scan design, 2026-10-05); a stopped force-constant stage is told "prep anew" by status while launch continues it (C10); launch's *carrying* note lists names that may not exist (C15); launch's own request builder (`submit._place`) made one with prep's `placement.request_of` (it counts no cores for a PySCF run); the placement admitted at prep (done, unit 10) written on the job with its sources; **from the second review**: a benchmark's placement -- closed 2026-10-06, the bench's queue is launch's and § 6.0 says so. **The items carried from unit 10's first review, closed -- read again in the code 2026-10-06**: C1 (`submit._carry_the_gather`), C5 and C10 (one fact, `Job.relaunch_continues`, which `status` and `launch` both read), C15 (the opener lists only what it copied), launch's request builder (11a); C8 with the transport work |
| 12 | B6 · M2i (F5) · M2j · M5 step 9 · M2l M5 | **one opener, every folder marked, the calculation's runs in one place** (`project-layout.md` § 1.4a, § 1.6.2, § 4.2; `web/results.md` § 2.4): one function opens every run folder — a stage's attempt, a trial's, a bias point's — stamping each folder and the containers above it, seeding the progress channel, telling the wrapper the run's own label; no on/off: a stage removed from the description leaves its folder marked `NN_name.disabled/` (F5); the Results tab lists every run of a calculation at its root — every attempt, never hidden — and shows the one picked in place | designed 2026-10-03 — the user's word; **12a done 2026-10-07** (one opener, `open_run` / `open_container`); **12b-1 done** (no `enabled`); 12b-2-12e next, 12e designed with the user first. **Settled 2026-10-07, the user's words:** "let's just let all run continue warm or cold, the user knows the consequence and we just manage the flow ... run continue warm or cold is user's decision, and error or not, that's user's responsibility"; "checkpoint is used for the rollover and branching. that's it" -- **built 2026-10-07**: any stage launched again, however it ended, warm (its own latest run) or `launch --cold` (nothing of its own; flat: the run script's `--cold --force`, which now REMOVES the files it names -- before, it left them and SIESTA read them); no refusal, no never-concluded question; a launched bias scan opens each point's next attempt the same way, until § 2a.11's sweep-as-one-run is built; `.disabled` dropped; 12c/12d (status and the Results root listing every run, so a run can be picked) go with step 9 of § 5u.1 |

### Unscheduled — rows gone, and W45 as it was

> **Verdict (2026-10-08 validation):** Unscheduled — W45 PARTLY (the offer half superseded, the save is always; `checkpoint verify` is not a CLI verb) → the trim keeps the `checkpoint verify` half, its row as it was is below; W46 OBSOLETE (`running-a-job.md:442-453` documents `--force`; unit 12 built on it); the `execution` config-key row OBSOLETE (the key is `launch.mode`).

| row | what it says | note |
|---|---|---|
| W45 | checkpointing's owed half (the save prep offers, `checkpoint verify`, the unwritten invariants) | the offer is a second question on M2d's prep entry |
| W46 | `--force` retired in the contract, alive in the wrapper | natural home: M2l |
| *(no row)* | the `execution` config key has no live contract; `running-a-job.md` says `mode` is `direct` or `launch`, the code says `direct` or `submit` | natural home: M2k (W36 ⑦'s precedence table) |

> **Verdict (2026-10-08 validation):** M7 closed, M9 folded into M10 — CONFIRMED history (M7's row done 2026-09-29; V1.26's decision still owed, M10's row).

**M7 closed and M9 folded into M10** *(2026-09-29, the consolidation)*: every
part of M7 was built by an earlier milestone — the derived rules as spliced
callables, the optional MO block, `raman_route`, the reader's unknown-key gate,
the Raman-ranked selectors retired (V1.6) — each re-read against the code (W21
archived with the evidence); M9's two rows are W42's, so M10 holds them —
V1.27's decision is W42's D8, V1.26's is still owed.

---

## 2. OPEN — the one list (archived 2026-10-08)

*The rows § 2 shed on 2026-10-08 — done, superseded or measured untrue — verbatim and in their table order, each under the validation's verdict. The pointer rows left in § 2 resolve their ids.*

> **Verdict (2026-10-08 validation):** CONFIRMED — both decisions taken and built (`engines/transport.md` § 3.8.6; `engines/vibration.md` § 6.6); archive.

> **The two decisions this box held are taken.** **D-1** — what declares that
> a value binds every rung: decided 2026-09-24, the sibling marker `shared`,
> and built (`engines/transport.md` § 3.8.6). **D-2** — where a mode's activity
> is classified: decided 2026-09-23, the classifier as built is the rule and
> `top_n` / `threshold` retire under V1.6 (`engines/vibration.md` § 6.6).

> **Verdict (2026-10-08 validation):** OBSOLETE — `scan_ending` is gone; `parse/engines/_run_ending.py` dispatches by role; `monitor.py:170-211` reports converged per phase (W35 decision 10); archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **E10** | engine / science | **NEEDS A RE-SCOPE, not implementation — re-derived 2026-09-07.** The detection is **built**: `parse/engines/_run_ending.py` is the one marker table (`abnormal_termination` → `stopped`, OOM markers, `SCF_NOT_CONV`), `scan_ending()` returns `(run_state, scf_converged, error_message)`, and `summarize.py:217` + `cli.py:2771` consume it — the zero-exit case being the whole point. What is NOT built is the monitor half, and **deliberately**: `monitor.py:138` says *"NO COMPLETION MARKERS HERE"* and `job-contracts.md:214` now rules the monitor follows the launcher's PID *"rather than guessing from output markers"* — which contradicts this row's own "belongs in `mb_monitor.py`". The real gap is narrower: **the monitor's finish report carries no convergence fact.** Decide whether it should before writing anything | roadmap § 4 | open (re-scope) |

> **Verdict (2026-10-08 validation):** CONFIRMED closed — `parse/contract.py:118`; `pyscf/input.py:1325-1334`; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| ↳ | **V1.31** | *(**Closed 2026-09-29 by the one-source rule** — user: "agree with your recommendation on V1.31 and R6". The relaxation record is one reader's, `parse.contract.relaxation_of` over the run's own output, and the Results tab's export writes it; the deck writes no copy. The level-of-theory half the review found beside it, R6, is § 5w K18.)* ~~The PySCF deck's own `_optimized.xyz` pair records `info.relaxation`~~ | `vibration.md` § 2.2, § 10 | closed |

> **Verdict (2026-10-08 validation):** CONFIRMED done — `continuation.py:224,326,408,654`; `run-panel.js`; the parked flat-layout question answered by W52; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **W37** | execution / front end | **A LADDER'S NEXT STAGE CAN BE PREPPED WITH NOTHING HANDED OVER, AND NOTHING SAYS SO.** Found 2026-09-27 on `projects/PDT/optimization/PDT_FIX_moleculeonly`: all three stages prepped within 10 s, before coarse ran; fine started from the input geometry (SIESTA's `XV file not found`, the wrapper's `initial-run (clean state)`) and converged; final is prepped the same way. The rule (`execution/job-system.md` § 1, *How a ladder advances*): prep one stage, look, prep the next *from a run you name*. What lets it slip: the Task setup page prints `--from` only for a stage whose row sets `restart: continue` by hand (`task-setup/viewer.js:2222`), while `continue` is the default since 2026-08-18 (`engines/stages.md` § 1.3), so a default ladder's Prep button writes a continuing stage with nothing carried in; `prep` accepts a continuing stage with no `--from` and says only "nothing carried in"; `--from` copies whatever the named attempt holds, without the *(finished, converged)* check § 5.3's example prints; the hand-over is recorded in `.continued-from` and `run.json` only -- not the decision ledger, not the Run panel; the deck header says SIESTA reads what the previous run left "in this directory" and prints `launch run 02_medium`, which the resolver refuses *(that half closed by K12, 2026-10-01: the header prints the name)* | `execution/job-system.md` § 1, § 5.3 · `engines/stages.md` § 1.3 · `project-layout.md` § 1.6 | **Done 2026-10-01** — part 1 (user: "build W37 right after W51"): **the hand-over is `prep`'s, for both doors** (`jobset/continuation.py`, `job-system.md` § 5.4, now its one home): a continuing independent stage — a kind without rung roles — takes the newest attempt of the enabled stage before it, read before anything is written; refused, naming the commands (run it first, `--from` the newest earlier attempt that concluded, `--cold`), while that attempt has not concluded or has failed; taken with *NOT converged* in the line when it concluded without converging; a run named by `--from` taken as said, with what it is; the flat layout by the same rule, nothing copied. Prep prints the hand-over (*continues from 01_coarse/run-0 (the stage before it; concluded rc=0 at …; converged): copied …*), the decision ledger records it (`continues`), `status` names the run beside the bare command, the deck's start-state text says where the files come from. Tests through the road: the default end to end (refusals before anything is written, the line, the copy, the ledger, the status), the explicit choices and a failed run, the flat layout — six mutations, each red; three prep tests that prepped `medium` before `coarse` ran say `--cold` now. *The review* (an independent read, each claim verified in the code): on the flat layout the refusal offered `--cold`, which prep refuses there — a flat stage starts clean by its run card's `restart: clean`, and the refusal says so; a stage prepped and never launched was told it was "still running" — the refusal is worded by the run's state and always names a command (launch it, let it finish, run it again); the browser's Prep handed over silently and printed a `--from` of its own that skipped the conclusion check — it shows the hand-over's line and composes no `--from`; `status` answered by a second rule, telling a stage set to start clean that it would continue — the hand-over moved into one module, `continuation.continuation_answer`, which prep and `status` both ask; a flat PySCF run had no verdict (its progress log is read now); a `--cold` start is logged; a default-path refusal no longer quotes a `--from` nobody typed; the older texts that said a stage starts from the structure unless named were brought to the rule. Six more mutations, each red; the page's last-attempt `--from` test retired with what it pinned. **Part 2 done 2026-10-01**: each continuing rung's Task setup tab offers **Continue from** — the stage before it's newest run (the default, with its line, or why prep refuses it), each run of that stage with what it was, or the structure (`--cold`, where the layout has one) — from the folder's answer (`continue_from`, `continuation.continue_from_choices`, the one prep acts on); the command line follows the choice, Preview says what it will continue from (or why prep would refuse, and Write is not offered), and Prep sends it to the same prep (the route takes `from` / `cold`); the Results tab's Run panel shows *Continued from* (the run record's `computation.launch.continued_from`, from `run.json`). The stage concept was renamed `continuation` (module, class, JSON key) — the folder's `handover` is the Build tab's task file, and the same page read both. Tests: the browser's doors (the folder's choices, the preview, `from`, `cold`, a path out of the calculation refused), the Run panel's record, the tab end to end in a browser (a refused default, a chosen run on the command line, the preview, the prepared attempt) — six mutations, each red. *Part 2's review* (an independent read, each finding verified in the code): a refusal was cut to its first line on the page and in `status` — shown whole, with the commands it names; a refused preview left Write enabled from an earlier preview, beside that preview's end-point box — Write is held back and the box goes; the folder's answer parsed each stage's output for a verdict it never prints, and resolved the ladder twice — the verdict is read at preview, and the default has one core (`continuation._by_default`); the page's own Prep was read back twice (its announcement, then the sidebar re-publishing the same selection), wiping the answer just given and the open tab — a re-read of the same folder keeps the tab, the choices and the answers (a Save, or another writer's restore, retires the answers), and a publish naming the folder already shown is not read at all (`task-setup.md` § 2.1: the selection moving replaces the folder), which also ended the page's two reads at start, the second able to replace a first unsaved edit; the stage concept's leftover *hand-over* wording renamed. Eight mutations, each red; the targeted set green (468). **Parked for the user's word:** on the flat layout the Run panel shows no *Continued from* — the record is the attempt's `.continued-from` — so either a flat stage's `<stem>.run.json` records it too, or § 5.4 says the panel shows it on the hierarchy only. **Agreed 2026-09-27** *(user: "newest finished by default, warn on unconverged, explicit choice will override - user dictate when explicit")*. One section of `execution/job-system.md` owns the hand-over and the other documents point to it. A continuing stage is prepped after the stage before it has finished, from that stage's NEWEST attempt, which must have concluded (the vibration stage's rule, `prep._vibration_stage_geometry`) -- refused otherwise, with the command to run first; an unconverged one is taken with a warning. **An explicit choice is taken as said**: `--from <attempt>` or `--cold`; prep states what it sees about a named run (still running, stopped, unconverged) and refuses only what cannot be done (no such attempt, no restart files in it). Prep prints the hand-over (`02_fine/run-0 continues from 01_coarse/run-0 (finished 14:39, converged): copied <label>.XV, <label>.DM`); the decision ledger logs it; `.continued-from` and `run.json` stay; the Run panel shows *Continued from*; Task setup shows the same answer before Prep; the deck header's text is corrected. The flat shape's line is settled when the section is written. **Added 2026-09-27** *(user: "yes, add it to W37")*: the hand-over moves into the ONE prep both doors use -- today it is a second `prepare_attempt` call on the CLI's path only (`_cli.py:2589`), so the page's Prep door can carry nothing -- and each continuing stage's Task setup card gets a **Continue from** choice: the previous stage's attempts with their state (default: the newest finished), or *start from the input geometry* (`--cold`); the printed command follows the choice and the page sends it to the same prep |

> **Verdict (2026-10-08 validation):** OBSOLETE — `project-layout.md:431,606,619` define `--force`; `runwrap.py:384-399,461`; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **W46** | execution | **`--force` is retired in the contract and alive in the wrapper.** `project-layout.md` (*"`--force` is retired"*, invariant 4b) and `job-contracts.md` say so; `runwrap.py` still parses `--force|-f` and resets `_run_n=0`, so a redo can overwrite an attempt the contract calls immutable | `project-layout.md` · `runwrap.py` | open — a defect against a written rule; natural home M2l |

> **Verdict (2026-10-08 validation):** CONFIRMED done — no `plan` command; `runstatus.py:380`; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **W51** | execution / front end | **THE LADDER, LISTED AND EXPLAINED** *(user, 2026-10-01: "shall we have a list command such that people would be able to list the stages at cli?"; "yes, do both after K12"; "better add these clarification into the document such that the two different types of tasks (simple stages vs linked stages) are explained")*. **The two kinds of ladder** are written down once (`job-system.md` § 5.4, pointed to from § 1 and `project-layout.md` § 2.3.4): independent stages — one calculation tuned several ways, each continuing from the run you name — and linked stages — a vibration's `relax → freq`, transport's five — whose input `prep` takes from the stage before, refusing until it has finished; W37's agreed default for independent stages is marked there as not built. **`status` lists the description's ladder**: every stage with its number from the moment `init` writes it, the ones not prepped yet as not-started, a disabled one never the stage to resume from — composed in `jobset_status`, so the Results tab's ladder is the same answer; the route composed its own, and the CLI listed the prepped stages alone and refused a calculation before its first prep. **`plan` folded into `status <stage>`**: the deck, what it carries and the resources lead the per-stage form; the verb is gone, no alias, and `STAGE-PLAN.md` stays as prep's written record. Tests through the road: `init` → `status` before any prep → a stage disabled through the Task-setup save → `prep` → `status`, `status <stage>`, `status '#N'`, the Results ladder — six mutations, each red. *The review* (an independent read, each claim verified in the code): the status's next command for an independent stage was a bare `prep run medium`, which starts from the input geometry — it names the run to continue from now (`--from 01_coarse/run-0`, until W37 makes that the default); a disabled stage was offered a prep its transport ladder refuses (it is told to enable it); a stage prepped and then disabled could still be the one to resume from, and `complete` read "all stages finished" beside it (every enabled stage); § 5.4's linked column said no other run can be named — `--from`/`--cold` still name an earlier attempt of the same stage — and said finished for concluded; § 5's diagram, the verb count, `job-contracts.md`'s `--bundle` row and three docs still named `plan`, `describe` or `submit` | `execution/job-system.md` § 5.3, § 5.4 · `execution/project-layout.md` § 2.3.4 · `web/results.md` § 2.4 | **done 2026-10-01** |

> **Verdict (2026-10-08 validation):** CONFIRMED done — `runtime_config.py:94-97,680,960`; `configuration.md:847`; `launch_refusal` is `placement.py:28`; archive, the parked list kept in § 2 as the pointer row's remainder.

| # | area | item | from | state |
|---|---|---|---|---|
| **W53** | execution | **EXPLICIT JOB CONFIG ONLY** *(user, 2026-10-02: "explicit job config is the only way allowed"; "make sure molbuilder.json are cleaned up with the obsolete and conflicting parameters removed. contract and document need to be very clear what should be in molbuilder.json and what they are used for")*. Answers W52 pass 3's R1 and three of its questions (the `scheduler` block, an unnamed queue, an unstated rank count). **The six fills became refusals** naming where to state each value: the queue (the menu's first row, `execution.domain`), its partition/QoS (`scheduler.directives`), the ranks (the target's width, one per GPU), the cores per rank (`scheduler.defaults.cpus_per_task`, SIESTA's 1, the GPU policy, PySCF's node cores), the wall (the queue's ceiling, `defaults.time`), the memory (`defaults.mem`). ONE answer, `prep_inputs.launch_refusal`, asked by prep before anything is written and by launch's one request (`submit._sbatch_request`) of what it sends; the renderers take what is stated and ask nothing again (their own three copies, found by a mutation, deleted); `place` binds only a named queue; the header carries no mail or export line. **`molbuilder.json` holds preferences only** — `launch.mode`, `envs`, `paths`, the server's settings; `configuration.md` § 4 is the one table of its keys. `scheduler` (all of it) and `execution` (renamed `launch`) are refused by name, each saying what to write instead; the user's own file cleaned, every kept value unchanged. **Follow-up the same day (Q1)** *(user: "the activation/preamble for the current machine (where molbuilder is running) should be provided in molbuilder.json ... when jobset probe fills the environment either for the local machine or save as a named machine ... the activation/preamble would be simply a copy")*: this machine's activation and preamble are `molbuilder.json`'s **`env_init`** (`script_generation` refused by name, renamed), asked by `envs init-config`; `jobset probe --write` copies it into every record it writes (its `--activation`/`--preamble` flags gone), the record's field renamed `env_init` with no version change; prep reads the target's record, this machine included; a copy wrong for its machine is edited by hand. **The per-calculation `.molbuilder.json` removed** — nothing wrote one (user: *"why would anyone read a file that is not written"*): its reader, merge, scope and refusals, the contract's rows, and the tests and rows built on it (16 tests). **Tests as data:** three tables, one runner — `launch_values` (26 rows), `gpu_contract` (38), `molbuilder_json` (13); the runner gained `machine_config`, `named_record`, `run_unset`/`allocation_unset` and `probe` (this machine's record made by `jobset probe --write`); retired `test_scheduler_config.py`, `test_cheapest_ceiling_that_fits.py`, `test_bench_execution_mode.py` and the renderer's own refusal test | `configuration.md` § 2–5 · `execution/architecture.md` § 5.2, § 8, § 9 · `execution/running-a-job.md` § 3, § 5 · `execution/scheduler.md` · `execution/gpu.md` · `execution/job-contracts.md` § 2.6 | **Done 2026-10-02.** 22 mutations red (one survived until the re-probe row was added); 93 targeted files (3523 tests) and the 58 files the last fix touched (1232) green; 99 browser tests green. **Ruled 2026-10-02 (Q8):** one `proxy.trust` setting for the proxy (option A); a benchmark's grid stated, never proposed (option B). **Parked:** `probe --name` stamps THIS machine's envs and arch into another machine's record; `environments/README`'s `--set` route makes a cluster record with no queues; `gpu_partition` is never written by the probe, and R9's re-check misses a GPU job placed on one; the PySCF deck run by hand takes the node's cores; next-step lines offer `--mode submit` on a workstation; `~/.config/molbuilder.backup/` on disk; the named-queue check lives in two doors (`launch_refusal` up front, `place` when binding) |

> **Verdict (2026-10-08 validation):** REFUTED as open — BUILT: the pipeline log always (`_cli.py:1414-1421`; `prep.py:2765`); archived as done (M2e, 2026-10-05).

| # | area | item | from | state |
|---|---|---|---|---|
| **W20** | front end | **A prep from the browser can never produce a pipeline log.** `pipeline_log` is off by default and only `jobset prep --pipeline-log` sets it (`_cli.py:2252`; re-measured 2026-09-23); the web door builds its kwargs at `build.py:1639/1663/1682` and calls `prep_calculation` at `:1685` without it (re-measured 2026-09-23). Measured: 2 `*.pipeline.log` in the tree, both from e2e fixtures, **none** under any real project. Not a defect — the flag is deliberately opt-in (`pipeline_log.py`: *"the log observes the pipeline, it is not a step in it"*) — but if a person prepping from the UI should be able to ask for one, the door needs a way to say so | found 2026-09-11 | **agreed 2026-09-27, not started** *(user: "always write it, drop the flag")*: every prep writes its pipeline log, from every door (the command line and the Task setup page); `--pipeline-log` and the `pipeline_log=` switch go -- the log is small (18-32 KB measured) and never fatal, and a record asked for in advance is missing when it is needed |

> **Verdict (2026-10-08 validation):** CONFIRMED — ① moved to X4 ③; ②–⑤ deleted 2026-10-02; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **X1** | transport / cleanup | **Transport historical residue — LEADS, NOT A DELETION PLAN.** Five candidates, found 2026-09-22 by reference scan plus spot reads. **① `transiesta._compute_cell_from_extents`** (with `_TRANSVERSE_PAD_ANG` and its two callers, `_lattice_block`'s `cell is None` arm and `wizard.extract_electrode_model`'s lateral fallback) — fabricates a 30 Å transverse / `int(bbox_z+2)+1` box, which is the rule `Structure.resolve_cell()` explicitly forbids (*"a periodic axis needs a commensurate lattice … never a bounding box"*); unreachable because `compose` refuses a cell-less citation on **both** forms (`:814`, `:869`) and `ElectrodeModel.as_structure()` always states one. From `8c70e721`, i.e. it predates `cell.py`. **② `wizard.DEFAULT_ELECTRODE_KZ`** — the third home of `40`; measured equal to the catalogue row and the `SiestaConfig` default, and `engines/transport.md`'s own table says it became a catalogue row. **③ `stages.SEALED_TRANSPORT_FIELDS`** — a union production never reads; `stages.py:390/396` and `transport.py:415/420` branch on the two sets separately. **④ `stages.config_for`** (~90 lines) — the pre-TR4 projection, superseded by `jobset/prep.py::_resolve_transport` (`d6f0218d` *"the transport arm resolves, and 143 lines of projection die with it"*). **⑤ `TransportConfig.num_threads`** — no reader, hidden from the served form by a predicate filter at `transport.py:543`, superseded by `SiestaConfig.omp_threads`; **not a defect**, a dead field behind a working guard. **Every one needs the `process/code-audit.md` § 1d **step 0** pass before it is touched** *(user, 2026-09-22: "a full code review before you decide if your decision about a piece of function is correct")* — a reference scan cannot tell residue from a lost caller, and ④ was one read away from being the second  **① HAS NOW HAD ITS STEP-0 PASS (2026-09-23) and moved to X4 ③.** What it found: the reachability claim in this row is CORRECT (measured — the only renderer is `prep.py` and both structures it can hand over state a cell), but the reason this row gives is not the whole one, and the other machine's audit reached the opposite verdict off `wizard.py`'s stale I6 header. The contract settles it: § 7 calls a padded extent box wrong rather than approximate, and § 5 holds I6 by copying the device's vectors. **②–⑤ are still unexamined** and keep the warning below — and now a second one: the other machine published verdicts on all five, and its verdict on ① was wrong, so those are leads too, not answers. **The audit's own step-0 reads (2026-09-23), for the record:** ② `DEFAULT_ELECTRODE_KZ` residue — the commit that deleted its last readers edited `__all__` to keep it; ③ `SEALED_TRANSPORT_FIELDS` residue — production builds a different union inline, and its only reader is a test; ④ `config_for` part residue, part lost caller — the lost rule, filling the config from a form-B pair's recorded contract, was taken over by `764addd3` in `citation_defaults.py`, so the residue half remains; ⑤ `num_threads` not a defect, but `log_level` is the same shape and does reach the deck (`TBT.Verbosity`, W25). The two documents disagreed about ①, so these too are leads until re-read | found 2026-09-22 | **① passed, moved to X4; ②–⑤ done 2026-10-02** (M5 step 3): ② `DEFAULT_ELECTRODE_KZ`, ③ `SEALED_TRANSPORT_FIELDS` and ④ `config_for` deleted, ⑤ gone with `TransportConfig` |

> **Verdict (2026-10-08 validation):** OBSOLETE — `science/validation.md:226`; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| ↳ | **A1.7** | **The engine registry fails open.** With `molbuilder.siesta` unimportable the registry lists only `PySCFConfig`, and `validate(Au2, SiestaConfig())` runs no `config.*` check and says nothing. § 1.11b is done (`3382b851`); its sentence for `science/validation.md` § 7 is still owed | audit § 1.11 | open |

> **Verdict (2026-10-08 validation):** CONFIRMED — every item done or moved (U7 → V1.14; U10 → V1.11); archive the sub-row.

| # | area | item | from | state |
|---|---|---|---|---|
| ↳ | **A1.16** | **The UI walk of 2026-09-23, not yet rows elsewhere** (U3/U4 are V1.14; U6, `removed_motions` shown beside the result, built 2026-09-24 — `engines/vibration.md`'s built table). **U2** the printed prep command lacks `--target`, which `task-setup.md` § 10 says the page puts in; **U5** `max_force_eh_a` holds Eh/Bohr and the viewer prints Å; **U7** `[cell.vacuum_defaulted]` advises vacuum for a boxless PySCF run; **U8** the hand-over gate admits PySCF only for a vibration while the CLI road opened SIESTA vibrations on 2026-09-23; **U9** a SIESTA `.spectra.json`'s nulls must draw *not computed*; **U10** `compose.py:1082,1175` writes and reads the permutation record by hand beside `write_permutation`/`read_permutation`, and its record carries no `key` *(the READ half fixed 2026-09-28, M2b′: `compose` reads through `atom_permutation.read_permutation` and names the file by its one constant; it still writes by hand and stamps no `key`)* | audit § 1.18 | U2, U5, U8, U9 **done** — as V1.2, V1.4, V1.1, V1.3 (2026-09-24); U7 is V1.14; **U10's write half built 2026-10-02 (M5 step 3)** (V1.11) |

> **Verdict (2026-10-08 validation):** OBSOLETE; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **B2** | run-decision round | *(priority P3)*  | **PARTLY DONE 2026-09-03 — and the number was the wrong instrument.** Of the four shapes named here, only one is mechanically decidable: a `Test*` class whose body is a docstring collects nothing. Five existed. **Three were empty promises** — `TestBuildSiestaHonorsSidecarFrozenAtoms`, `TestWorkspacePayloadRegionsAndFrozen`, `TestGenerateWritesToWorkspace`, each stating in the present tense that it pins something (*"Tests pin both layers"*) while holding no test, so a reader scanning for coverage reads yes. Each is replaced by a pointer at the file that DOES cover it. **Two are deliberate retirement markers** that say so and name their successor — the same call D2 already made for two zero-test files. The other three shapes do not survive measurement: `assert len(X) == 5` where the test BUILT X is a real check, and `m = re.search(...); assert m` is a precondition with the real assertions after it. **A list of ~45 that cannot be re-derived is not a finding anyone can act on** — what is left needs the file-by-file read, not a regex | 5 measured |

> **Verdict (2026-10-08 validation):** OBSOLETE — § 5h: 0 to convert; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **B3** | run-decision round | *(priority P3)*  | **CLASSIFIED 2026-09-06 — and the population is a fifth of what three earlier counts claimed.** 233, then 256, then 173 were three definitions, none written down. Measured now by `tools/classify_source_reads.py`, which states its definition and can be re-run: of **1,255** assertions over a file's text, **1,147 read GENERATED output** and are correct as text — a property of a real product, never a defect. **108 read hand-written source**, in 31 files. Of those, **59 stay** (51 lints, where text is the only instrument that can prove absence, and 8 vendored/data files) and **49 convert**. Full method, per-bucket file list and the mutation proof are **§ 5h** | 49, not 233 |

> **Verdict (2026-10-08 validation):** OBSOLETE — `runtime_config.py:960` `_read_scheduler_retired`; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **S3** | architecture seams | **`runtime_config`'s untyped scheduler dicts + mixed concerns** | `backend-architecture.md` § 5 (**W3**) | **OPEN — verified 2026-09-06.** `runtime_config._validate_scheduler` still returns `Dict[str, Any]`; overlaps **S6** |

> **Verdict (2026-10-08 validation):** OBSOLETE — § 2a: 86 of ~95 done, the rest closed 2026-09-21; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **S18** | ops / envs / config | **The 2026-09-12 env-installer and config-and-secrets session has its own hand-over file: [`2026-09-12-env-config-handover.md`](?doc=archive/2026-09-12-env-config-handover.md).**  80 items, each marked by how it was checked (RAN / READ).  It exists because that session's own commit messages are not reliable: three independent audits were told to FALSIFY them, 14 of 16 behavioural claims held, and the failures were overstatements of scope -- four documentation statements written that day are false, two of them in text `envs init-config` ships into a user's config directory.  Its § 1 is the verified DONE list and exists to stop the next session re-deriving settled work; § 2 is the work, grouped as defects introduced (A), false docs (B), an instruction not implemented (C), sweeps that stopped at the first instance (D), pre-existing finds (E) and decisions for the user (F).  **§ 0 states the TARGET first** -- the installer's two state machines and one runner, and config's one resolver / one name / one writer, as 13 checkable invariants T1-T13 -- so every item reads as a named deviation rather than a patch.  **§ H is the residue of the pre-state-machine design**, swept against those invariants rather than against any diff: one question answered in two or more places four times over, a string where the design says state three times, dead parameters, and a door that never sanitises the environment it dispatches into.  **§ I is the config/secret residue**, and its first lesson is that **A11's own text in `architecture.md` still names a pre-consolidation owner**, so the rule as written licenses the three `.parent` climbs it forbids -- fix the rule before the sites.  § G records what was checked and found clean.  **§ 3 is the MIGRATION PLAN** -- eight phases scoped to `install-env.sh`, the `envs` verbs, deployment and how config is placed and validated, each stating what it closes and which end-state row (Z1-Z9) it realises; § 3.0 states that end state so it can be checked, and § 5 records what is deliberately out of scope.  **Read § 1 before touching § 2, and A0 before anything** -- `envs install molbuilder --clean` currently deletes the env the process is running from, and `envs doctor` prints that command as its remedy for a failed host verify.  No clean full-suite result exists for the work yet (F3) | audited 2026-09-12 | open -- nothing in § 2 started |

> **Verdict (2026-10-08 validation):** OBSOLETE — superseded by `runs.folder_answer` (W56 unit 3a, 2026-10-04); archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **N9** | parse / run files | **The front door had no consumers — wire them** *(one since 2026-09-26; the status says which are left)*. `JobDirParser` composes `run_status`, `_enumerate_files`, `engine_of` and the discovery chain into one answer, and `RunDirResult` is its shape. **Both STAY**: zero callers is a migration that has not happened, not evidence the door is unwanted. *(This row said "delete the bundle" until 2026-09-18 — four unlike things under one word, which read as "delete the framework". User: "job dir parser is actually the framework we're developing".)* What IS a defect is narrower: a **directory** routes through the same `detect()` the file verbs call, which cost three CLI verbs their clean refusal — one a silent hang. That is the ROUTE, and those verbs already ask `answers_a_trajectory()` instead. **Open: wire the consumers; decide whether the route is `detect()`/`parse_dir` or a direct call** | § 5c.2 · `model/parse.md` § 5.0, § 5.5 | **partly** *(re-measured 2026-09-29)* — `/api/results/dir` consumes the door through `parse_dir` since W35 P2 (2026-09-26), so the route question is answered; § 5c.2 h's two named consumers — `web/blueprints/watch.py` and the spectrum surface — are not wired, and whether each asks a whole-directory question is `model/parse.md` § 5's rule to apply (*one that does not, does not*) · **superseded 2026-10-04** (W56 unit 3a): the door moved up to the run door, `runs.folder_answer`, and the watch, spectrum and Results loads all ask it (plan B11, B14) |

> **Verdict (2026-10-08 validation):** CONFIRMED closed — `parse_stage_token` is gone; `job.py:82,88,217` and `materialize.py:63` read `runfiles.parse`; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **N5** | parse / run files | **The three defects § 5l measured, which outlived it** (`archive/2026-09-17-paths-standard-retired.md` § 5l.a). **① LIVE:** every staged run loses its frozen atoms — `_sidecar.read_frozen_atoms` needs a label to strip the rung and three of four callers omit it, so *"Hide frozen atoms"* and `runtime_info["frozen_atoms"]` are empty for every laddered calculation. **② duplication, NOT a latent defect — re-measured 2026-09-18:** `identity.parse_stage_token` is a second reader of the stage token alongside `runfiles.parse`. They differ on `_geom_optim.xyz` (a declared underscore ROLE, which the second swallows into the stage name) and on `.runwrap-*.log`. **Neither shape is ever passed to it.** Its three callers feed decks (`materialize` ×2, `job.script`) and `.out` / concluded `.molwatch.log` (`parse/dirs/job.py::_detect_stage`); measured on all five real shapes, the two readers AGREE every time. *This row said "latent" and was reported to the user as a bug waiting to happen, with an invented `.xyz` example — user: "why would you fucking pass a .xyz to a parser and ask which step this run belongs to?" Nothing does.* What is real is one grammar with two readers, worth collapsing on the one-home rule and on nothing more urgent. **③** a phantom rung for an unstaged calculation, fixed by ②. *The migration framing is gone with § 5l — these are ordinary defects in `parse/` and `identity`* | § 5l's inventory, re-measured 2026-09-17 | **① FIXED 2026-09-17. ② and ③ open.** *The fix was already in the same module.* `_siesta_fdf_path_for` — the function `model/parse.md` § 5.3 names as the shape a companion lookup may legitimately take — solves the identical problem identically: try the exact stem, then *"fall back to a single `*.fdf` in the same directory"*. `read_frozen_atoms` never got that fallback. It has one now, asked through `sidecars.molstruct.sidecars_in` (the framework's own search, § 4.5) rather than a hand-rolled glob, and **guarded twice**: the lone sidecar's label must be a prefix of the artifact's on a `_` boundary (so an unrelated sidecar that merely happens to be alone is refused), and two candidates decline rather than pick. Licensed by `project-layout.md` § 1.4 — a run directory holds one invocation's output. **Strictly additive**: it runs only where the answer was already nothing. ② and ③ remain, and are one deletion: `identity.parse_stage_token` goes, its three callers (`parse/dirs/job.py:81`, `materialize.py:394`, `:433`) move to `runfiles.parse`, which is right on both shapes the two disagree about |

> **Verdict (2026-10-08 validation):** CONFIRMED done — M5 step 3, 2026-10-02; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **W27** | engines / execution | **Transport is not on the seven-floor stack — put it there.** *(user: "fix your fucking plan by having first a correct top-down architecture"; "why the fuck is the emit not based on template based approach".)*  The architecture and the derivation are `engines/transport.md` §§ 3.2–3.7.  **THE ROOT, in the project's own vocabulary** (`execution/architecture.md` § 2): `molbuilder/transport/` appears in **none of the seven floors**, while that document's one mention of transport claims its stages "run INSIDE the job system (each an ordinary prep/launch rung)".  Four rules broken: floor 3 renders the text of every file from a `ParameterSet` through `prepare_deck` — transport's `render_script` concatenates literal f-strings; floor 2 holds what the person asked for — transport had no template, so `TransportConfig` became the definition; `prep` is the conductor and may never decide — `_prep_transport` is a second conductor that does; floor 2 must never name a machine — `max_memory_mb`/`num_threads` sit on it.  **WHY, from the history:** `transiesta.py::render_script` 2026-06-10; the pipeline landed 2026-08-19 (`refactor(prep): the seam carries the engine's FORM`) and migrated siesta + pyscf in one commit, leaving transport behind; the composite was then built outward from the unmigrated emitter.  `template.md` § 9.2 has recorded the missing arm all along, filed as one lost feature (USER-CUSTOM) rather than as *transport cannot render from a template*.  **MEASURED:** the seed deck `prep` renders carries **13 keywords / 4 blocks** against a template offering **45 deck-reaching items**.  **EVERY KNOWN DEFECT IS DOWNSTREAM:** the seed dying at 1000 SCF iterations (`MaxSCFIterations` cannot travel from the citation — no transport emitter writes it and no transport field held it); the device deck aborting on "the continued fraction method requires at least 20 poles" (~~the pole *energy* is written, never `TS.Contours.Eq.Pole.N`~~ — withdrawn: the keyword is read (`m_ts_chem_pot.F90:113`) but cannot act on this deck shape — its continued-fraction branch overwrites the count from the energy, `N = int(E_pole/π/kT)` (`:319`), so the energy is the handle, and the abort was our 1.5 eV default giving 18 — § 5p.3o); `TBT.k` as a bare scalar the parser rejects *(the bracketed list tbtrans reads since 2026-09-29, § 5u step 1)*; `tbt_k_grid`'s unguarded transport axis *(guarded since: the settings gate refuses a third component other than 1, `validation/__init__.py`)*; the electronic contract as two frozensets and a twice-spelled predicate.  **ORDER OF WORK IS FLOOR ORDER**, § 3.6: (1) render through `spec_for`/`DeckSpec`/`prepare_deck`, (2) no value syntax by hand, (3) validation report + read-back check, (4) `_prep_transport` stops deciding, then the floor-2 items.  My first draft of this plan put the pipeline at step 6 of 7 because it was written from the parameters down; read from the floors down it is step 1.  **DONE:** 4a the `citation` marker (`template.md` § 6.4's sibling answerer, per-kind); 4b/4c transport's parameters as 17 catalogue rows + 7 shared rows tagged `citation = ["transport"]` + 9 relaxation rows tagged `optimization`, with matching `SiestaConfig` fields — which also made `electrode_kz` (invariant I9) reachable from a description for the first time. | `engines/transport.md` §§ 3.2–3.7 · 2026-09-15 | 4a/4b/4c **done** · the floor-3 migration: rendering through `spec_for` done 2026-09-16 (`engines/transport.md` § 3.6 item 1), the NEGF block on the catalogue built 2026-09-29 (§ 5u step 1, § 3.6 item 2), the rest (§ 3.6 items 6, 7, 12) **built 2026-10-02 (M5 step 3)** |

> **Verdict (2026-10-08 validation):** CONFIRMED done — 32 rows; `TS.Voltage` `:2473`; `TBT.Verbosity` `:2428`; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **TR1–TR3** | engines / transport | **Transport's description gains a template, and the catalogue gains what the design needs.** T1: `jobset init --calculation transport` writes `<label>.template.toml` with its Class A values defaulted from the cited relaxation — today a transport folder has **no template at all**, so the shared baseline has no home and the stage table has nothing to read. T2: the three keywords with no catalogue row — `TS.HS.Save`, the equilibrium pole **count**, `TS.Voltage`. T3: `kgrid` split, because its three components fall in three classes. **Reuses `build_description` / `template_with_values`, which already narrows the catalogue by `calculation`** | § 5p · `transport.md` § 2a | **ALL DONE 2026-09-16** — TR1 § 5p.3c, TR2/TR3 § 5p.3d |

> **Verdict (2026-10-08 validation):** CONFIRMED done; archive.

| # | area | item | from | state |
|---|---|---|---|---|
| **TR4–TR6** | engines / transport | **Transport renders through the one path.** T4: `_prep_transport` keeps the compose and hands to `resolve`, so a transport run has a `ParameterSet` with provenance and `--pipeline-log` stops being a no-op. T5: the remaining four rungs onto `spec_for` — ⚠️ **blocked on a seam question**, what a composite kind hands its renderer, since `spec_for(struct, cfg, stage_token=)` does not carry the `ComposedJunction`. T6: `TransportConfig` retires | § 5p · § 2a | **TR4 done 2026-09-16** (§ 5p.3e); TR6 half-done — `siesta_config_for` already deleted; **TR4 and TR5 DONE**; TR6 all but done — one projection survives, feeding the lifted NEGF block, and goes with it |

> **Verdict (2026-10-08 validation):** CONFIRMED all closed — `889a9ca9`, `f8ffd458`; `diagnostics.py:310-317`; `recipes.py:856-857,1327,2640`; archive whole.

### 2a. From the env-config hand-over, when it was archived *(2026-09-20)*

[`2026-09-12-env-config-handover.md`](?doc=archive/2026-09-12-env-config-handover.md)
is spent — **86 of ~95 items DONE**, re-derived from the code, its invariants
(§ 0's targets, § 3.0's Z1–Z9) reached on eight of nine phases. These are what
was left. Each carries the measurement that proved it open, so none needs
re-deriving before it is acted on.

**Seven of the eight closed on 2026-09-21, and one row is left.** Each was
re-derived against the tree before its row was removed:

* **J2** — `scheduler/probe.py` named `write_config_scope`; `probe --write`
  writes `environment.json` through `write_environment`. Docstring fixed.
* **J3** — was never open here. `diagnostics.manager_info` became the one
  reader of `conda info --json` on **2026-09-13**, the day after the hand-over
  was written, which is why it was not carried into this list at all. The
  archived body still describes the old state and its header now says so.
* **B8**, **H8**, **H8b** — `envs` (`889a9ca9`): a comment naming a module with
  no such function, two `__all__` entries that made `import *` raise, and a
  strip the control flow could not reach.
* **D9**, **D10** — `jobset` + `scripts` (`f8ffd458`): the bare `isatty` that
  could raise on the very input it was written for, and a script that honoured
  two thirds of the host-env rule.
* **F2** — **closed as a decision, 2026-09-21.** The question was whether to
  restore a deliberately broken PyPI `pyscf-properties` in the HOST env so the
  2026-09-11 observation could be pointed at. No: the fact is permanent and
  readable from what is installed — the dist-info reads `0.1.0` while
  `direct_url.json` names a git commit, which *is* the version collision
  `force=True` exists for. `recipes.py`'s comment now states that reason
  instead of citing the vanished artifact, so nothing depends on the evidence
  being recreated — and recreating it would put pyscf back in the env it must
  never be in.

| # | item | measured 2026-09-20 |
|---|---|---|
| **F1** | the A-rules whose checker was deleted | ✅ **CLOSED 2026-09-21 — by building the one checker that looked viable and measuring it.** **A7** is partly mechanised and its row now says which half. **A8** was the candidate; the checker was written (§ 3's eleven classes' fields × every signature, all from the AST) and run: **two candidates, both correct code, zero violations** — so the rule holds, and the check cannot tell a `Stage`'s `name` from *"what the user called this calculation"*, which is § 3.1's own invocation-vs-job carve-out and is semantic. Rejected, and recorded in A8's row so it is not re-attempted. **A1 and A4** cannot be checked at all. Nothing left to decide |

Two more are recorded there and are **out of scope by that document's own § 5**,
not by neglect: `script-preparation.md:202` vs `runwrap.py:2295` on who reads
the rank count, and `oauth.py`'s two independent `expanduser` readings of
`client_secret_file`. One was **deferred by your ruling**, and is **fixed 2026-09-29** (M2c, W36 ⑥):
`recipes.py` resolved the CUDA version at import, so `nvidia-smi` ran on every
invocation; the registry is now built on its first ask (`builtin_recipes()`).

~~**Three fresh residues the audit turned up, not on anyone's list:**~~
✅ **ALL THREE CLOSED 2026-09-21.** `cli.py`'s two unused imports are gone.
The GHOST advice was wrong in **both** halves, not the one recorded here:
`conda env remove -n <name>` does not clear a ghost either — it resolves the
name to a prefix and refuses with `EnvironmentLocationNotFound`, there being no
directory, which is what GHOST *means* (measured 2026-09-21). And `--clean`
does fix it, by the route the note missed: the REMOVE step is skipped, but
`can_resume` is PRESENT-only, so the create still runs and restores the
directory the entry names. The message says that now.

---

## 5q. The engine offset — scope and order of work *(W33, 2026-09-25)* (archived 2026-10-08)

*Removed from the live section by the 2026-10-08 validation walk. Each block
is verbatim, with the walk's verdict. What stays live is in the trimmed § 5q.*

### 5q.0 Why — four measured facts, one cause

> **Verdict:** CONFIRMED, history — all four causes are closed.
> `watch.py::_run_periodicity_json` is gone (the Results load asks
> `runs.Declared.frame`, `watch.py:207–217`); `frame_shift` has 0 hits in the
> tree.

All four were measured on 2026-09-25, on the fake-junction ladder
(`projects/claude-vib-ui`).

1. **TranSIESTA refused the device deck** — *"Electrode: L lies outside the
   unit-cell"*, 4 s in, before any SCF. The junction's box was anchored at its
   lowest atom by the electrode builder, deliberately (`modify.py:872–881`,
   *"the padding opens at the TOP"*). The stored corner was `−17.355` against an
   atom at `−17.355016`, so that atom sat 1.6e-5 Å below the face in every deck. The
   relaxation and the seed ran with the same atom, because plain SIESTA treats z
   as periodic; only TranSIESTA checks. molbuilder's own `.validation.txt`
   passed the deck.
2. **The Results tab drew a box the engine never had.** `watch.py::_run_periodicity_json`
   sends no corner — so the server derives one — and takes the axis kinds from a
   `.source` pair searched in the **run** directory. A ladder keeps that pair at
   the calculation root, so none was found, the axes defaulted to `isolated ×3`,
   and the derived corner (`−0.48, −0.48, −1.177`) drew the atoms centred.
   SIESTA's corner was `(0,0,0)`, and every file SIESTA wrote is flush.
3. **The same kind of junction was centred on 2026-08-29 by accident.** Before
   `20f1cca0` (2026-09-21), transport form A left every axis `isolated`, so the
   derived corner centred it (the `Au-BDT-Au` device deck: 1.2004 Å each side).
   That commit correctly stated z = `transport`, and a transport axis's corner
   rule is `bbox_min`, which is flush. Nothing caught it because no device had
   run since.
4. **The provenance never existed, and the trajectory mixes frames.** `frame_shift`,
   the stamp `structure-periodicity.md` § 6.1 clause 5 promises, is written and read by no code. The molwatch
   log's step 0 is in the design frame while the deck from the same prep call is
   in the engine frame (measured by the structure-API audit, § 1.18: a 45 Å jump between step 0 and step 1).

**The cause, once:** placement is re-derived by each reader, with rules that
depend on the axis kind, instead of computed once by one rule and recorded
where it was applied.

### 5q.1 Data structure

> **Verdict:** CONFIRMED built, one stale cell — `structure.py:122,166`;
> `cell.py:473,509,615`; `periodicity_gate.py:78,83`; `modify.py:830–834`.
> The frame-set row is NOT built: it is § 5u step 11's (the trim's § 5q.5
> keeps it).

| | today | becomes |
|---|---|---|
| `Structure.cell_origin` | a stored field (`structure.py:132` `METADATA_FIELDS`, the check `:585`, to/from dict `:896`/`:979`, copy `:1979`, the view block `:1078–1095`, the periodicity blocks `:2140`, `:2255`) | **replaced** by `Structure.engine_offset` — absent means the rule; present only when the person assigned the origin (contract § 6.0, **D1**) |
| the corner derivers | `resolve_cell_origin` `:679`, `expected_cell_corner` `:780`, `_derived_corner_under_explicit_cell` `:840` | **removed**. `resolve_cell` and `effective_vacuum` stay: they size a box nobody typed |
| `cell.py` | composes box + corner, and `ResolvedCell` carries `origin_is_user_owned`, `corner_was_derived`, `contains_at_world_origin` and two fractional projections | **gains** `engine_offset`, `EngineFrame`, `to_engine`, `engine_frame` (contract § 6.0); `ResolvedCell` carries `engine_offset` / `box_corner`, and the three origin fields and the second projection **go**; `EngineFrame` states whether its offset was assigned |
| `periodicity_gate.py` | the `cell_origin` op (`:370–:624`), `BLOCK_KEYS`, the manual-origin regime and its notices | the op becomes `box_corner` (assign on a typed cell, or clear back to the rule), `BLOCK_KEYS` with it; the manual-origin regime becomes the assigned offset; the gate has three box states — derived, typed, typed with an assigned origin — not four |
| `modify.py` | the electrode builder states the flush corner (`:872–:902`); `calibrate_to_cell` (`:1137–:1170`) | the builder states **no** origin; `calibrate_to_cell` **retired** (**D3**) |
| a frame set | — | **one** offset, from frame 0, for every frame (contract § 6.0) |

### 5q.2 File access

> **Verdict:** CONFIRMED — `sidecars/molstruct.py:89,128`;
> `parse/sidecars/molstruct.py:205–226`; `script_emit.py:461–490`; 0 stated at
> `siesta/input.py:807`, `pyscf/input.py:1311–1336`, `jobset/prep.py:459`,
> `cli.py:1048–1051`, `transport/compose.py:514–522`;
> `trajectory_log/format.py:57–60`.

| file | today | becomes |
|---|---|---|
| `.molstruct.json` sidecar — `sidecars/molstruct.py` (writer), `parse/sidecars/molstruct.py` (reader) | schema v9, carries `cell_origin` | **v10**: `cell_origin` is replaced by `engine_offset`, always written — `null` means the rule, `[0, 0, 0]` an engine's output, `−P` an origin the person assigned. The writer writes v10; the reader accepts v9 and ignores its `cell_origin`, as `model/structure.md` § 2.2 already rules for a removed key (**D2**) — a v9 corner may be the electrode builder's flush one, the placement that failed |
| every deck molbuilder writes (SIESTA `.fdf`, PySCF `.py`) | an `atom-metadata` block only when there are labels; no placement record | + a **`molbuilder engine-offset` block** for every deck (cell, offset applied, whether it was assigned, axis kinds). One writer beside `script_emit.emit_atom_metadata`, one reader beside `_extract_atom_metadata_dict` |
| engine output (`.XV`, `.out`, `.STRUCT_OUT`, `.ANI`, `.MD`, the PySCF logs) | read as bare coordinates plus a lattice | read into `engine_frame()`, offset 0 stated |
| the molwatch log (`trajectory_log/format.py` ← `jobset/prep.py`; the audit's § 1.18) | step 0 written in the design frame | step 0 written from `to_engine()` — closes that finding |
| exports — the Results export, and `cli.py:1163–1174` (`.XV` → pair) | set `cell_origin = None`, which then derives | engine coordinates + cell, v10, stating offset `0` (contract § 6.0): every run's export reloads as a typed cell with a stated origin of 0 (`watch.py`, `cli.py`) |
| transport artifacts | form A composes the `.XV` with `cell_origin: None` (`compose.py:623`) and derives | the `.XV` through `engine_frame()`, each rung through `to_engine()`. `atom-permutation.json` and `slot-provenance.json` are unaffected (indices, provenance) |

### 5q.3 Protocol agreement — the wire

> **Verdict:** PARTLY — built (`structure.py:896–950`;
> `render-engine.js:313–317`; `model.js:879–884`; `periodicity.js:508–621`).
> Stale: `_shared._stated_periodicity` is `validate_periodicity`
> (`_shared.py:36–37, 354–385`); D5's `.source` pair fallback was deleted by
> B12 4b (plan § 0c unit 4); the transport citation viewer is § 5u step 9's.
> The corrected state is the trim's § 5q.3.

* **One periodicity block**, built by one server function, `Structure.to_wire`:
  `{cell, engine_offset, resolved_cell, box_corner, axis_kind, vacuum,
  resolved_vacuum}` — `engine_offset` the offset the structure STATES (raw,
  echoed back by the client), `box_corner` where the box is drawn
  (`−engine_offset` of the coordinates it rides with). No `coordinates` tag:
  D4 made every viewer draw the coordinates of the structure it shows, so the
  structure says which they are. Every door that sends a structure spreads the
  block, and none composes its own. *(In the swap — `6c705058`, the commit
  that retired `cell_origin` for the stated `engine_offset` — 2026-09-25.)*
* **The doors**: `/api/structure/periodicity` (the Cell page), the structure
  load/save payload (`_shared._stated_periodicity` reads `engine_offset` back),
  and `/api/watch/load` / `/api/watch/data` — the Results door, which reads the
  run deck's ENGINE-OFFSET record for the axis kinds (D5; the `.source` pair for
  a run made before it) and states `engine_offset = 0` for the engine's frames.
  The transport citation viewer is still open.
* **MolView** draws the box at `box_corner`, verbatim (`render-engine.js`,
  `model-jobs.js`'s effective cell, `ui.js`'s Origin row, `model.js`'s
  `getUnitCellOrigin` — the ASSIGNED origin or `null`). The browser never
  computes a corner. *(In the swap.)*
* **The Cell page** (`modify/periodicity.js`, `templates/modify.html`): the
  origin group assigns the box's origin — three numbers, or *Use selection* at
  the atom's own unrounded position — sent as the block's `box_corner`, and
  *Derive it* became *Automatic* (**D1**). Assigned values are shown in full,
  so an Apply never re-sends a rounded origin. *(In the swap.)*
* `/api/modify/calibrate` (`web/blueprints/modify.py:656–675`) is **retired**
  (**D3**), with the `calibrate` op in `lib/molview/model-jobs.js` and
  `modify.calibrate_to_cell`.

### 5q.4 Validation

> **Verdict:** T1 CONFIRMED (`tests/test_engine_offset_reaches_every_deck.py`);
> T2 REFUTED as located — its pin is `tests/test_siesta_flat_run_e2e.py:168–191`,
> not `tests/watch/test_api_load.py`; T3 CONFIRMED
> (`tests/test_results_export_e2e.py`); T4 OBSOLETE — TD3's
> `cell.transport_vacuum` (`validation/__init__.py:459–500`) refuses a face
> gap, ERROR, not a `d/2` warning; T5 CONFIRMED (`tests/test_cell.py:543–576`).
> The retirements: done 2026-09-25 (the P4 row below). The checks and the
> test map are restated in the trim's § 5q.4.

**The checks** (contract § 6.0): *the atoms fit* (fractional span `< 1`) at the
edit; *transport clearance along c* in the transport kind validator
(`validation/__init__.py`, beside the `TBT.k` checks).

**The tests** — through the road, and fewer than today:

| | what it drives | what it asserts |
|---|---|---|
| **T1** | per engine, `jobset init → prep`: a SIESTA relaxation, a SIESTA single point, the SIESTA and PySCF vibration ladders, PySCF, the five transport rungs | the deck's coordinates are design + `engine_offset`, every atom is inside with the rule's margins, and the deck's record equals the computed offset; on a typed cell with an assigned origin the deck is design − origin, its record says assigned, and an origin that leaves an atom outside is refused naming it; the vibration `freq` deck writes the relax output unchanged (a stated `0`); an atom beyond a periodic face is a warning in the report |
| **T2** | the Results door on a finished run | `box_corner = 0` and the coordinates are verbatim from the output, and the axis kinds are the deck record's (D5). The run directory is BUILT under `tmp_path` with a flush deck and output — never read from `projects/` (`testing.md` § 2a); a run with no record takes its kinds from the `.source` pair |
| **T3** | export from Results → reload | coordinates + offset is what the engine had: every run's export reloads as a typed cell with a stated offset of `0` (contract § 6.0), the box where the engine had it |
| **T4** | `prep device` on the fixture | every atom inside the cell along c; a face gap below the lead's `d/2` is warned, not refused (risk **R1**) |
| **T5** | the rule itself, API-level on constructed fixtures shaped like the junction (`tests/test_cell.py`) — the header says so | a Cartesian implementation fails it (a skewed cell's x-extent exceeds \|a\|); a widest-gap cut fails T1's slab, whose widest gap along c is inside it, as the junction's Au–S contacts (2.399 Å) were wider than its seam (2.355 Å). The junction's `[6.3684, 3.746, 18.5325]` is recomputed in the contract, not pinned |

Each is mutation-tested: break the rule, watch it fail. T1 also pins a rule
the audit found unpinned — the explicit-cell centring branch, whose deletion
left 441 tests green (A1.17).

**Retired**: the tests that pin the obsolete designs — the derived corner per
axis kind, the user-owned origin, the `cell_origin` op, the containment regimes,
the `resolved_cell_origin` wire shapes, and the nine tests asserting the
literal corner `[7.5, 7.5, 7.5]` on one fixture (the audit's § 5a). Counted at planning: 24 files and 210
references (`test_periodicity_gate.py` alone has 82, then
`test_structure_authority_roundtrip.py` 22, `test_web.py` 18,
`test_structure_periodicity.py` 16, `test_molview_model.py` 13). Unifying must
lower the count.

### 5q.5 The inventory — every code site, and its phase

> **Verdict:** the P1–P3 rows CONFIRMED built, the rows marked done are done;
> the P4 rows (the design-frame exports `structure.py:1603,1725,1752`,
> `cli.py:146–147`; the transport validation subject `transport/deck.py:657`;
> `wizard.py:395`; multi-frame pairs → § 5u step 11) stay OPEN and are the
> trim's § 5q.5. The document list: the owner's four clauses are still in
> `structure-periodicity.md` (`:606`, `:697`, `:781`, `:961`) — the trim's
> list names them by line.

| site | role today | phase |
|---|---|---|
| `structure.py` | stores the corner; derives it three ways | P1 |
| `cell.py` | composes box + corner (`:186`, `:208`, `:234`) | P1 |
| `periodicity_gate.py` | the Cell-page door, the origin op and regime | P1 |
| `sidecars/molstruct.py`, `parse/sidecars/molstruct.py` | schema v9 | P1 |
| `modify.py` | builder's flush corner; `calibrate_to_cell` (retired, **D3**) | P1 |
| `cell.py` `_contains`, the three `_CONTAIN_EPS` / `_EPS` | containment re-implemented beside `Structure.cell_contains_atoms` (A1.13's containment half) | P1 |
| `validation/siesta.py:802–803` | a containment check that ignores the corner — `inv(cell)` on raw positions; the owner says inside, the check says outside (the audit's § 1.6) | P1 |
| `periodicity_gate.py:385,422,592`, `_shared.py:156` | user-facing error text citing a `§ 3c` that no longer exists | P1 / P3 |
| `siesta/input.py` | hand translation (`:871`); validators get a re-derived struct (`:990`) | P2 |
| `siesta/input.py` `_wrap_into_cell` (`:295`, `:948–956`) | the `wrap_into_cell` knob fractional-wraps atoms into a user-supplied cell — the re-wrapping contract § 6.0 forbids. Missed by the first inventory; the audit's § 8 named it an origin-rule site | P2 — **retired 2026-09-25**: the knob, its catalogue row, its validator check, the wrap branch and the tests that pinned it. A template written before then still names the item with a value, and `prep` refuses it like any unknown name until the block is deleted — kept so *(user, 2026-09-25: "retiring this unused item is correct and should be refused")*; a read-and-ignore rule was tried and reverted |
| `transport/transiesta.py` | a second hand translation (`:401–404`) | P2 |
| `pyscf/input.py` | coordinates verbatim into `gto.M` (`:137`, `:547`) | P2 |
| `script_emit.py` | the metadata block writer/reader; + the record's | P2 |
| `trajectory_log/format.py` ← `jobset/prep.py` | molwatch step 0 in the design frame | P2 |
| `transport/compose.py` | form A composes with no corner, then derives | P3 |
| `cli.py` | the `.XV` export (`:1163–1174`) | P3 |
| `web/blueprints/watch.py` | the Results periodicity block | P3 |
| `web/blueprints/_shared.py` | the payload's periodicity block | P3 |
| `web/blueprints/build.py` | the periodicity door's origin op | P3 |
| `web/blueprints/modify.py` | `/api/modify/calibrate` — retired (**D3**) | P3 |
| `lib/molview/render-engine.js`, `model-jobs.js`, `ui.js`, `model.js`, `demo.js` | draw / choose / show / commit an origin; `model-jobs.js`'s `calibrate` op (retired, **D3**) | P3 |
| `modify/periodicity.js`, `templates/modify.html` | the origin group — kept, re-founded on the assigned offset (**D1**) | P3 |
| `validation/__init__.py` | no transport clearance rule | P4 |
| `transport/sort.py`, `config/siesta.py` | comments naming the old field | P4 |
| *found by the structure review, 2026-09-25 (at `effb6ca3`):* | | |
| `jobset/prep.py::_vibration_stage_geometry` | hands `freq` the relax run's output coordinates, which the rule then re-centres — a relaxed geometry moved against the mesh (`engines/vibration.md` § 5.2a; −0.0168 Å on the H₂ e2e). A P2 regression | P1 — the output states `0` |
| `siesta/input.py` validation subject → `validation/sidecar.py` | the relaxation record's `geometry_sha256` is compared with the SHIFTED subject, so on the road a relaxed pair's record stops vouching and its checks are skipped. A P2 regression | P1 |
| `script_emit.render_deck` | a spec without `engine_frame` skips the gate and the record with a log note | P1 — refused |
| `trajectory_log/format.py` ← `jobset/prep.py` | the preview calls `to_engine(struct)` with no box, so step 0 differs from the deck whenever prep passes `cell=` | P1 — from the spec's frame |
| `cell.py` `cell.vacuum_ignored` | *"On an isolated axis it still sets where the structure sits"* — false since P2, and in every SIESTA report for a typed cell with a stated vacuum | P1 |
| `pyscf/input.py`, `pyscf/vibration_deck.py` | the pair writer bakes the design structure's sidecar beside the engine coordinates it saves | **done** (`dba66a9a`: `_sidecar_for` states 0, both decks) |
| `structure.py` `to_extxyz` / `to_ase` / `to_pyscf`, `cli.py --pyscf-atom-block` | design coordinates beside a lattice that implies the box at the origin | P4 |
| `transport/deck.py`, `validation/__init__.py` | no validation subject: the transport validators judge design coordinates in `resolve_cell()`, which for an isolated lead is not the deck's box | P4 |
| `transport/wizard.py:431` | the lead's z hand-shifted by its lowest layer — harmless (re-centred), and a translation outside the one door | P4 review |
| multi-frame pairs | "one offset from frame 0" is implemented nowhere, and no emitter takes frames | P4, with W32 |

**The documents to sweep** in P4 (restatements; the owner is
`structure-periodicity.md`, whose superseded clauses are deleted then):
`web/molview.md` (11), `model/structure.md` (11), `web/web-api.md` (6),
`model/structure-molstruct.md` (5), `engines/transport.md` (3),
`engines/vibration.md` (2), `architecture.md` (2), `science/normal-modes.md`,
`README.md`, `model/overview.md`, `backend-architecture.md` (1 each); and, found by the review, `model/parse.md` (the premise *"no deck shifts a structure that states one"*), `model/structure.md` § 2.2 (the removal rule D2 relied on), `execution/job-contracts.md` § 3.1 (the block grammar lacks ENGINE-OFFSET), `engines/vibration.md` § 5.2a, `engines/siesta.md` (centring described for the derived box only). Dated
plans and handovers are history and are not rewritten; `plans/plan.md`'s live
rows are.

### 5q.6 Phases, each with its done-condition

> **Verdict:** P0–P2 CONFIRMED done; P3 PARTLY — done, but T2's file is wrong
> (the pin is `tests/test_siesta_flat_run_e2e.py:168–191`); P4 PARTLY —
> rewritten in the trim as three open items (the `d/2` warning is OBSOLETE,
> TD3); P5 OBSOLETE — TD8 made `claude-au-bdt-au` the one acceptance ladder,
> § 5u step 4 ran it to its record on 2026-09-29, and the viewer half is
> § 5x R9. The commits paragraph is history; the rungs 1–3 paragraph is P5's;
> the fixture paragraph describes `claude-vib-ui`, retired as an acceptance
> target by TD8.

| | work | done when |
|---|---|---|
| **P0** | the name, contract § 6.0, this plan | **Done** 2026-09-25 (`a2901f4d`), and the contract revised as the decisions landed: the stated offset and the gate (`b13810f8`), centring for robustness (`790694f4`) |
| **P1** | data structure + file access: §§ 5q.1–5q.2's model and sidecar rows | T5 passes; the codec reads v9 and writes v10. **Committed**: the rule, `EngineFrame`, `to_engine`, `engine_frame`, `require_placed` (`dcfb371b`); the stated offset (`Structure.engine_offset`, in memory), the gate's containment, and the review's two regressions fixed — the vibration `freq` deck and the SIESTA relaxation record (`1df5cc24`); moving atoms only moves atoms, D6 (`4adc6832`). **Done** with the swap (`6c705058`): `cell_origin` retired to `RETIRED_METADATA_KEYS` (D2) and its derivers deleted; `cell.resolve` places the box at `−engine_offset`, with `cell.beyond_periodic_face` and a stated-offset `cell.atoms_outside`; the sidecar at v10 with `engine_offset`, readable {7, 8, 9, 10}; the gate's op `box_corner`; the electrode builder states no corner; both `.XV` readers state 0 |
| **P2** | every emitter through `to_engine`, each deck carrying its record | T1 passes for every engine. **Done** 2026-09-25 (`4a3172e1`): SIESTA, all five transport rungs, PySCF (both atom writers) and the molwatch preview; `render_deck` writes ENGINE-OFFSET and runs the gate; every prep renders from the compose record. The review of 2026-09-25 found three regressions: old templates carrying `wrap_into_cell` are refused (kept so, by decision — `d17c3ad4`); the `freq` re-centring and the relaxation record are fixed (`1df5cc24`). T1's assigned-origin clause landed with the swap (`6c705058`), for SIESTA and PySCF — and found the outside-origin refusal reaching `prep` as a traceback, fixed there (it is a `ValidationError` now). **T1 complete** (`dba66a9a`, after the review): every transport rung through prep, over a cited deck with its record and one from before the rule (D7), and the PySCF vibration deck |
| **P3** | readers and the wire (§ 5q.3), MolView and the Cell page | T2 and T3 pass, and on the dev server the browser draws what the deck says. **Committed** (`6c705058`): the wire's `engine_offset` + `box_corner` (`cell_origin` / `resolved_cell_origin` gone); MolView draws at `box_corner`; the Cell page's origin group sends `box_corner`, *Automatic* replacing *Derive it*; the Results door reads the run deck's ENGINE-OFFSET record for the axis kinds (D5) and states 0. The 28 test files naming the retired design were read in full against the contract (three agents) and revised or retired by that review: 6568 → 6552 test functions, every new or strengthened test mutation-checked (17 mutants, all red). T2 passes (`tests/watch/test_api_load.py`, its run built under `tmp_path`). **T3 passes** (2026-09-27, `tests/test_results_export_e2e.py`, one case per route a run's output takes out of the Results tab -- a SIESTA trajectory, whose cell is its output's; a PySCF trajectory, whose cell is its deck's record; SIESTA's own `<label>.xyz` through the structure preview -- each run made on the road, saved through Export → Data → Save to project and read back through the codec: the engine's cell, a stated offset of 0, the engine's coordinates. Each case was broken on purpose. M1's review found the third route stated no frame at all, and the export offering " .log_frame6"; both fixed in the same milestone). The dev-server browser check is done: 2026-09-25, on a cleared workspace store and a new project (`claude-w33`), the junction built from SMILES, saved as a v10 pair at 6 decimals, relaxed and cited for transport through the tabs, every deck carrying its record |
| **P4** | the checks, the test retirement, the document sweep | T4 passes; the review of § 5q.5 finds no hand translation. Containment is refused at the hand-off already (P1). The swap fixed what it made false in the documents — the contract's field table, op list and op table, backend rows and status lines, `structure.md` § 5, `transport.md`'s box section — and the 1.6e-5 Å miss misreported as "16 fm" in six files. **Open:** the `d/2` face-gap warning for transport rungs; the document sweep proper — deleting the superseded §§ 6, 6.1 clause 4 and 6.1a's corner column (§ 7 was rewritten to the one-Apply page on 2026-09-25, and clause 5 to the record); the test retirements the review named, **done 2026-09-25** (five confirmed by `tools/verify_subsumption.py`, one refocused where the tool showed it was the only pin, two with the API they tested); the § 5q.5 inventory rows marked P3/P4 that the swap does not reach (the design-frame exports, the transport validation subject — narrowed to a lead that states no cell — frame sets) |
| **P5** | **acceptance** — the ladder rebuilt from scratch in a new project (`claude-w33`, user 2026-09-25: "re-create the transport and spectrum tasks from scratch"): junction `structure/au333bdt`, relaxation `optimization/au333bdt-loose` (one CG step, 4926.6 s at 10 ranks, 67 SCF iterations, max \|F\| on the free BDT 1.90 → 0.79 eV/Å, every atom d/2 inside z), transport `transport/au333bdt-t` (bias 0; seed launched 2026-09-25) | the device reaches its SCF; the record is written; the Results tab shows each rung's engine frame. **In progress** — the device reached its NEGF SCF on 2026-09-26, twice: run-0 (TranSIESTA's own 42 poles) diverged, 1000 iterations with the charge off by 584 electrons; run-1 (`TS.Contours.Eq.Pole 10 eV`, 123 poles — the two decks differ in that one line) held the charge to 0.004–0.08 and was converging when it stopped at NEGF step 9. No transmission ran, so no record yet. Which ladder closes P5 is § 5u's TD8 **Re-pointed 2026-09-29 (TD8): the acceptance ladder is `claude-au-bdt-au`**, `claude-w33` retired as a target. On it: the device reached its NEGF SCF and converged (13 iterations, TranSIESTA's 123 poles); the record is written (`aubdtau-T.transport.json`). **Left: the Results tab's view of each rung's engine frame**, not yet looked at |

**The commits run add → move every caller → delete**, each one green: the
old API goes in the same work item, never left as a shim, but not before its
readers have moved.

The ladder's rungs 1–3 concluded on flush decks. Re-prepped under the rule their
decks differ, so strict composition will not reuse them — that is correct, and
is why P5 re-runs them.

**The fixture**, all of it mine (`projects/claude-vib-ui`).
`structure/au333x6_bdt.xyz` + `.molstruct.json`: 120 atoms, two Au(111) 3×3×6
slabs around a BDT with its thiol hydrogens removed, built by the documented
CLI pipeline — `molbuilder smiles "SC1=CC=C(S)C=C1"` → `modify --delete 8,11`
→ `modify --orient-axis 0,5 --center midpoint` → `modify --electrode
Au:111:3x3x6@contact=2.4:registry=B:+z=5 --electrode
Au:111:3x3x6@contact=2.4:registry=A:-z=0`. Labels (0-based) `L-electrode`
66–92, `R-electrode` 39–65, `bridge` the rest; all 108 Au frozen; `c` =
37.065 Å (span + one layer spacing, set on the Cell page). Built on the
experimental `a` (4.078 Å; PBE's is 4.158) — slightly compressed gold,
consistent within the ladder. The relaxation `optimization/au333bdt-loose`:
one CG step at 0.5 eV/Å, 4949.6 s and 52 SCF iterations, max |F| on the free
BDT 3.31 → 1.15 eV/Å. The transport `transport/au333bdt-t`: the seed 4180.7 s
and 44 iterations; the leads 159.4 and 161.6 s, 15 iterations each, E_F
−1.203935 and −1.203934 eV; the device refused before its SCF (§ 5q.0, fact 1).

### 5q.7 Risks

> **Verdict:** R1 OBSOLETE — TD3's rule refuses the face gap
> (`validation/__init__.py:459–500`); R2 and R3 are facts of the 2026-09-25
> transition, not work; R4 → § 5u step 11 (the trim's frame-sets row carries
> it); R5 history — P5 is closed and the record has been read by
> `same_calculation` since.

* **R1 — the device's gaps against the lead's** *(reframed twice, 2026-09-25)*.
  What TranSIESTA requires is containment — every atom inside the cell (its
  own words, contract § 6.0, *Why centring*); `d/2` at each face is what its
  recipe recommends, and centring gives it. So the 0.21 mÅ by which the
  device's half-gap (1.1775) exceeds the lead's (1.17725) — `c` typed 37.065
  against span + spacing 37.0645 — is irrelevant to the requirement: either
  way every atom is more than a full ångström inside the cell. A lead and
  device placed by different rigid shifts is not what TranSIESTA checks (the
  leads ran flush in their own cells). T4 asserts containment and the `d/2`
  warning; only the P5 device run shows TranSIESTA accepting it.
* **R5 — the record is load-bearing before anything reads it.** `same_calculation`
  masks only volatile fields, so any change to the ENGINE-OFFSET block's text
  (a key added, the format bumped) makes every concluded rung "not the same
  calculation". Freeze the record's format before P5's ≈ 70-minute seed re-run.
* **R2 — existing runs.** A re-prep of any existing calculation renders shifted
  coordinates, so its old concluded attempts no longer count. Intended for
  transport; for a finished relaxation it means its runs do not match a re-prep.
* **R3 — egg-box.** Moving atoms against SIESTA's real-space mesh shifts total
  energies at the meV level, so an energy from before this change and one from
  after are not bit-comparable for the same structure.
* **R4 — frame sets** need one offset from frame 0, or the electrode-identity
  gate of `engines/transport.md` § 2a.9 breaks between frames.

### 5q.8 Decisions

> **Verdict:** D1–D4, D6–D16 CONFIRMED built; the rules live in the
> contract's § 6.0. D5 PARTLY superseded — the axis kinds are the deck
> record's, and the `.source` pair fallback for a run made before the record
> was retired by B12 4b (plan § 0c unit 4).

* **D1 — the origin: the rule, unless the person assigns it** *(user,
  2026-09-25, revising the first answer: "we should add one that can allow
  user to explicitly assign the origin of the cell box as the user desire (for
  further modificaiton convenience etc). this could be handled by just write
  the -origin to the frame_offset such that the next treatment of the file will
  force the 0,0,0 to be the user specified position")*. The Cell page's origin
  group stays, on a typed cell only, and stores the offset; its rules are the
  contract's (§ 6.0, *A stated offset*). The first answer —
  *"always automatically calculated based on the cell unit and all the atoms"*
  — is still what happens wherever the person assigns nothing.
* **D2 — existing sidecars: the old corner is retired** *(user, 2026-09-25:
  "your option (a) is fine, and when files are saved, make sure no old retired
  key is written again")*. Corrected first by the structure review: three
  gates REFUSE an unknown sidecar key (`model/structure.md` § 2.2's *"ignores
  unknown keys"* contradicts its own section and the code), and every sidecar
  carries `cell_origin`. So `cell_origin` joins `RETIRED_METADATA_KEYS` — read
  and ignored, and an old structure opens with the rule's placement — and no
  writer ever emits it again. An origin a person typed by hand is assigned
  again on the Cell page. Chosen over migrating it as a stated offset because
  v9 cannot tell a typed corner from the electrode builder's flush one, and
  the flush one is the placement TranSIESTA refused.
* **D3 — calibrate: retired** *(user, 2026-09-25: "we can retire the
  calibrate button")*. The `calibrate` op, `/api/modify/calibrate`,
  `modify.calibrate_to_cell` and their tests go in P3. What calibrate was
  for — a structure saved with its box where the person wants it — is D1's
  assigned origin. **Done 2026-09-25**: the op, the route, the function, their
  tests and the clauses that described them.
* **D4 — every viewer draws the structure's own coordinates** *(user,
  2026-09-25: "the 3d viewer should just follow what coordinate is in the
  structure, and draw the cell box with the offset in mind (start from
  -offset). i don't believe the 3d viewer should do the job of translation.
  the siesta receives the correct result because the script generator/engine
  will do the correct thing to gate and translate the coordinate to siesta.
  while the 3d viewer sees the result it would see what siesta see and print
  so there is no inconsistency issues")*. A calculation page shows the design
  coordinates with the box at `−engine_offset`; the Results tab shows the
  engine's, with the box at `(0,0,0)`. No viewer translates.
* **D5 — the Results tab's axis kinds are the structure's** *(user,
  2026-09-25: "we should show the axis_info as in structure. siesta is always
  periodic, true, but the isolate axis get our additional gate of vacuum
  surrounding them and that shows in the cell box too")*: read from the
  deck's record, which carries them as the structure had them when the deck
  was written.
* **D6 — moving atoms only moves atoms** *(user, 2026-09-25: "leave the cell
  alone, moving atoms only moves atoms")*. A translation, rotation or
  orientation — of some atoms or all — changes coordinates only: the cell's
  vectors and a stated offset stay where they were. Under *Automatic* there is
  no stored corner to keep: the box is the rule's, centred on wherever the
  atoms now are. Retires § 6
  clause 5 (a whole-structure transform moved the box with the atoms) and the
  tests that pinned it. **Done 2026-09-25** (`Structure.affine`).

**D7–D16 — the review's ten questions** *(user, 2026-09-25: "go with your
recommendations on all ten"; and, on the workspace store: "you can clean up
the persistence and timeline record so you have a clean baseline to test e2e.
new project, new file, no residue state to deal with")*. Asked after the
four-agent review of the swap; each is the recommendation as put:

* **D7 — a relaxation from before the rule stays citable.** The transport
  citation states `0` for the `.XV` only when the cited deck carries its
  `engine-offset` record; otherwise the rule centres the junction, a rigid
  shift (contract § 6.0). **Done 2026-09-25** (`transport/compose.py`).
* **D8 — the dipole estimate is taken about the centre of mass**, so a charged
  system's number does not depend on where it sits in the box.
* **D9 — an append that centres the incoming structure drops its stated
  offset**: the centring reframed its coordinates, so the box centres on the
  joined atoms, and the append says so.
* **D10 — the unused placement API goes.** `engine_frame`,
  `EngineFrame.box_corner`, `ResolvedCell.corner` and `.offset_stated` have no
  reader; each engine-output door states `engine_offset=0` itself, and the
  contract's table says so.
* **D11 — the hand-off refusal has its own id**, `deck.atoms_outside`, so
  `cell.atoms_outside` carries one severity everywhere.
* **D12 — the Cell page names the atoms outside**, per axis, as the deck
  refusal does.
* **D13 — no box without a corner.** MolView draws no box when the server sent
  no `box_corner`. The store of drafts from before W33 was backed up and
  cleared rather than migrated (the second ruling above), so nothing restored
  can lack one.
* **D14 — a retired `cell_origin` is said, not dropped silently**: the load
  door names the corner it did not apply.
* **D15 — the load door's text branch loses its side blocks**
  (`atom_metadata`, `periodicity`, `info`: no caller sends them), and the
  Results door builds its structure directly.
* **D16 — the Cell-page gestures say what they read**: the ruler's picks, in
  order (user, 2026-08-31) — *Use picked atoms* / *Use picked atom*, and
  *Automatic*'s title says it empties the boxes for Apply.

**D8–D16 done 2026-09-25**, each with its test broken on purpose to watch it
fail; the review's retirements went with them, each confirmed first by
`tools/verify_subsumption.py` (§ 5q.6, P4).

---

## 5r. The structure-API cleanup — its order, and what not to fix *(A1; from the unification audit, 2026-09-25)* (archived 2026-10-08)

*Removed from the live section by the 2026-10-08 validation walk: one
sentence. The order of work (§ 5r.1) and the other non-findings (§ 5r.2) stay
live; the `transport/transiesta.py` bullet was refreshed in place with the
four `+1` sites (`:387`, `:521`, `:600–601`) and the torn-frame reader with
its location (`parse/engines/pyscf.py:405–439`) — no text removed there.*

### 5r.2 Do NOT "fix" these — confirmed non-findings (the companion-lookup bullet's last sentence)

> **Verdict:** REFUTED, stale — `siesta_mdnc.sibling_md_nc` has had the guard
> since B12 4c (2026-10-04): it finds `<label>.MD.nc` by the label the `.out`
> prints (`reinit: System Label:`), never the lone `.MD.nc` of a folder
> (`parse/engines/siesta_mdnc.py:384–398`). Deleted from the bullet.

  A third copy of the
  shape (`sibling_md_nc`) has neither guard.

---

## 5s. The electronic state — charge and spin as one answer *(W34, 2026-09-25)* (archived 2026-10-08)

*Removed from the live section by the 2026-10-08 validation walk. Each block
is verbatim, with the walk's verdict. What stays live is in the trimmed § 5s.*

### 5s.0 Why — measured on 2026-09-25, every one read in the code or the engine's source

> **Verdict:** CONFIRMED, history — each cause is addressed: the adapters,
> `_applyAutoDetectToForms`, `applySuggested` and `resolve_net_charge` have
> 0 hits; the Makov–Payne notice and script are keyed on a finite system
> (`makov_payne.py:51–53, 339–342`); PySCF's class is explicit
> (`pyscf/input.py:253`); the citation brings its spin
> (`citation_defaults.py:56–72, 126–129, 161–194`). What is still unread —
> the engine's own charge, moment and ⟨S²⟩ — is P5, kept live.

* **Every layer read the raw fields and interpreted them itself.** The analyzer
  counted electrons at charge 0 and ignored periodicity (`chemistry.py`
  `analyze_structure`), so formate at −1 and a bulk gold lead were both told to
  go open-shell; the parity check used the run's charge on PySCF and, on SIESTA,
  only when the charge was typed (`validation/siesta.py`); for a metal-free
  structure the two reported one fact and could disagree.
* **Auto-detect overwrote.** It wrote `net_charge = 0` over a blank charge (the
  phosphate rule switched off), `spin_total = 0.0` beside `non-polarized`
  *(stopped 2026-09-28 by a closed shell's SIESTA suggestion carrying no total
  spin -- a stop-gap the amendment retires with the adapter)*, and
  RKS/UKS over HF (`structure-optimization/viewer.js` `_applyAutoDetectToForms`,
  `lib/spectra/core.js` `applySuggested`, the two adapters); the chip never read
  the charge (`lib/detection-chip.js`).
* **Nothing travelled.** A transport calculation's spin started at the class
  default whatever the cited relaxation ran (`transport/citation_defaults.py`
  carries six pairs and the k-grid), under a caption naming the cited run; a
  charged cited run was not detected; a vibration built from a relaxed structure
  started neutral and closed-shell; a ladder stage could change either, with the
  `.DM`/`.chk` carried across unconditionally (`resolve.py`, `warm-files.toml`).
* **The charged-species science ignored the cell.** The Makov–Payne notice and
  script fired for slabs and crystals too, and SIESTA applies the same
  correction itself for a molecule in a cubic cell (`madelung.f`, printed as
  `siesta: Emadel`), which the script never read — a double count.
* **Nothing read back what ran**: no SIESTA charge or moment, no PySCF ⟨S²⟩, the
  stability outcome printed but not recorded, and a spin-polarized TBtrans run's
  `TBT_UP`/`TBT_DN` files invisible to the transport record's glob.
* **PySCF re-ruled silently**: `dft.RKS`/`scf.RHF` with `mol.spin != 0` become
  ROKS/ROHF inside PySCF.
* **The documents were wrong in four places**: *"a polarized run starts from zero
  net spin"* (SIESTA starts at maximum aligned moments), *"the parity rule runs in
  `_validate_siesta`"* (only for a typed charge), `cfg.charge` for PySCF (the
  field is `net_charge` since 2026-08-19), *"PySCF UKS at spin 0 is a free
  moment"* (it is pinned).
* **And one of those errors is printed to users.** The `config.spin_total`
  warning (`validation/siesta.py`, `_check_siesta_spin_treatment_needs_spin_total`)
  says *"the SCF then starts from zero net spin on every atom"* — SIESTA starts
  every atom at its maximum moment, aligned (`m_new_dm.F90`) — and offers
  `spin_total` as a starting point, when with `Spin.Fix` it constrains the total
  moment for the whole run.

### 5s.2 Decisions *(user, 2026-09-25: "go with your recommendations on all seven")* — decisions 1–3, 6–9 and the amendment

> **Verdict:** CONFIRMED built — decision 1 (`electronic_state.py:160–252`,
> 14 call sites; the analyzer judges the resolved charge,
> `validation/chemistry.py:234–250`); decision 2 (the four items `shared`:
> catalogue `:1044–1048, 1077, 1120, 1191`; the stage-override refusal
> `chemistry.py:277–282`); decision 3 replaced by 9; decision 6 (the
> capability table; the PySCF vibration's ROHF/ROKS refusal); decision 7
> (`jobset/migrate.py:340–372`, `_cli.py:2170–2175`); decision 8 (`free`,
> `model.py:579`); decision 9 (`ElectronicState`, every consumer reads it;
> the adapters and `resolve_net_charge` 0 hits). Decisions 4 and 5 (PARTLY)
> stay live in the trim.

1. **The framework**, contract first, then code in phases (§ 5s.3). It absorbs
   the two questions held open earlier the same day — the analyzer judges the
   resolved charge (was D17) and does not read parity in a repeating cell (was
   D18).
2. **Charge and spin belong to the calculation** — a stage override of any of the
   four items is refused, as transport already refuses its shared items.
3. ~~**Auto-detect fills only fields still at their default**~~ — *replaced by
   decision 9 (2026-09-28): there is no fill.* It read: never `0` over a blank
   charge, never a pin beside a restricted treatment, and the HF/DFT choice
   kept.
6. **Restricted-open is an explicit choice**, with the capability table: not
   offered on SIESTA (no such formalism), offered for a PySCF optimization or
   single point (ROHF/ROKS gradients exist), refused by name for a PySCF
   vibration (no analytic ROHF/ROKS Hessian in PySCF; the deck uses the
   analytic Hessian). `restricted` with unpaired electrons stays refused and
   names both ways out.
7. **Existing templates are migrated, not tolerated** — a one-time `jobset`
   command rewrites the old items (`spin`, the R/U inside `method`, SIESTA's
   `spin_treatment` spellings and `spin_total`); nothing reads the old names
   afterwards, and a template that still carries one is refused naming the
   command.

*Amended 2026-09-28 (user: "i believe we had a framework for charge handling
etc, and spin/close-shell/open-shell can be handled by a similar class
level/framework level such that all engine can use to detect and decide"; "or
maybe this could be merged to that class too"; "go ahead with the contract, add
free, make sure api and users are unified, after refactoring, use agent to
detect residue handcrafted or duplicated/redundant code/design"):*

8. **A floating moment is the value `free`** of `unpaired_electrons`, so that a
   blank can mean *auto* for every item. SIESTA only; PySCF refuses it by name.
9. **One class decides the state, the way the charge rule decided the charge.**
   `ElectronicState`, built by `electronic_state(struct, cfg, kind=)`, answers a
   blank item in one order — stated, implied by a stated item, recorded by the
   run the structure came out of, detected from the structure; `method` is never
   blank — and carries each value's source and reason. Every consumer reads it: the deck writers, the checks, the
   hand-over, the forms (the chemistry card and chip show the resolved state;
   the Auto-detect button and its fill are retired), the read-back. The
   per-engine adapters and `resolve_net_charge` go. After the
   refactor, agents review the whole of it for hand-built, duplicated or
   redundant code and design.

The original decisions 4 and 5, before the trim's refresh:

4. **Transport's spin defaults from the cited run**, and a cited run carrying a
   net charge is refused.
5. **A vibration built from a relaxed structure inherits its state**, and a
   change is warned.

### 5s.3 Phases, each with its done-condition — the P0–P3 rows, the original P4 and P5 rows, and *Where the phases stand*

> **Verdict:** P0–P3 CONFIRMED (`chemistry-correctness.md:271–606, 606–720`;
> `tests/test_electronic_state.py`; `tests/test_chemistry_card_e2e.py`;
> `lib/detection-chip.js`; `web/blueprints/build.py:122–151`;
> `runwrap.py:3289–3305`). P4 PARTLY (`parse/fdf.py:158–175`; the comparison
> `validation/spectra.py:638–640`; the charged-citation refusal site not
> located) — refreshed in the trim. P5 OPEN — kept in the trim with the
> TBtrans half pointed at § 5x B6 / R1 (`record.py:453`). The review's
> fixes CONFIRMED (`cell.py:661`; `electronic_state.py:240–245, 217–224`;
> `jobset/prep.py:1549–1557`; `transport/deck.py:710`; `migrate.py:366–371`;
> `build.py:148–151`); the third pass PARTLY (no site named by the walk).

| | work | done when |
|---|---|---|
| **P0** | this section, the contract (§§ 2a–2b), and a pointer at every restatement the contract supersedes (`engines/template.md` § 6.3's spin paragraph, `validation.md` §§ 2, 2.1, 9, `engines/siesta.md` §§ 4–5, `engines/pyscf.md`, `engines/vibration.md`, `engines/transport.md`, `engines/stages.md`, `science/overview.md`, `model/chemistry.md`) | the user has read the contract |
| **P1** | the items and the class: the catalogue rows, both configs' fields, `electronic_state.electronic_state()` with its one order (§ 2a.1a) and its detection table (§ 2a.1b) — the analyzer's facts feed its detection step, the phosphate rule its charge step — every deck writer composing from the state (PySCF's class explicit, ROKS/ROHF included; each value's source in the deck comment), the migration command, the old items deleted everywhere | every deck kind — SIESTA optimization, vibration and the five transport rungs; PySCF optimization and vibration — is written from the state, pinned through `jobset prep` per kind; the migration command turns an old template into one prep accepts |
| **P2** | the checks, all reading the state: parity for a finite system at the resolved charge (SIESTA's auto charge included); a stated item against `recommended`, one family with parity (ES9); a metal-driven decision warns until the count is stated (ES8); the capability refusals (ES4, ES5, `free` on PySCF — ES6); the charged-species checks keyed on `finite`; `check_open_shell_metal` deleted; the correction script reads `siesta: Emadel`; the wording defects (*"closed-shell doublet"*, *"small Au cluster (27 atoms) … needs n ≥ 4"*, the empty `()`); the retired `config.spin_total` warning's premise (§ 5s.0) and the wrapper's IMAX hint (§ 5s.4) | formate at −1 and the gold lead prep without a spin finding; a radical still gets one; a charged slab gets no Makov–Payne script; every new test mutation-checked |
| **P3** | the forms: the chemistry card and the chip show the state `/api/structure/analyze` resolves for the form's own four items, on load and on every change to one of them; the Auto-detect button, its fill and the adapters are deleted; the count is a list (*(auto)*, 0–10, `free` where the engine can float it) and a choice the engine cannot run is not offered; the four items are `shared` for their kinds, so the Task setup columns exclude them and `prep` refuses a stage override of one for every kind; the transport caption names each value's source | on the dev server, a blank charge stays blank and the card says what it resolves to and why; a typed −1 changes the card; Fe with the spin blank shows *unrestricted, 2S = 2* with its warning |
| **P4** | the hand-over: `parse/fdf.py` reads `NetCharge`/`Spin`/`Spin.Fix`/`Spin.Total`; transport defaults its spin from the citation and refuses a charged one; the relaxation record carries the state, the vibration defaults from it, and the record check compares it | a polarized relaxation cited for transport yields polarized rungs; a charged citation is refused by name; a vibration of a charged relaxation starts charged |
| **P5** | the read-back: SIESTA's `.out` (net charge, fixed or converged moment) into the run record; PySCF's class, ⟨S²⟩ and stability recorded; the transport record reads both TBtrans channels; the Results tab shows asked against used, and a difference is a finding | a spin-polarized run's moment and a UKS run's ⟨S²⟩ appear on the Results tab; a two-channel transmission is read |

**Where the phases stand (2026-09-29).**

* **P0–P4 built.** The contract is § 2a–2b of `science/chemistry-correctness.md`
  and every restatement the documents review found is swept.
  * **P1:** every deck kind is written from the state and pinned through prep —
    `tests/test_electronic_state.py`, and `tests/test_transport_prep.py` for a
    blank spin decided once on the junction. The migration keeps what the file
    states and refuses an old item that disagrees with a new one.
  * **P2:** the one family, `check_electronic_state`. The Makov–Payne notice and
    script are keyed on a finite system and the script reads `siesta: Emadel`
    first. The IMAX hint no longer names a spin cause.
  * **P3:** the card and the chip. They are asked about the structure the page
    holds, and hide when there is nothing to answer. Driven in a browser on all
    three tabs in `tests/test_chemistry_card_e2e.py`, and on the dev server.
  * **P4:** the fdf reader knows every word SIESTA accepts for `Spin`. A cited
    deck or record brings its spin. A charged citation is refused by name, and a
    record the structure was edited since is not assumed.
* **The review** (four agents, 2026-09-29, the user's *"use agent to detect
  residue handcrafted or duplicated/redundant code/design"*). It found:
  * A PySCF calculation of a periodic structure was judged a repeating cell.
    Fixed: *finite* is the calculation's (`MOLECULAR`, § 2a.1b).
  * The transport rungs named no source for their spin. Fixed: they are handed
    the junction's state.
  * PySCF raised a bare `KeyError` before the gate on a label naming no element.
  * The migration overwrote values it should have kept.
  * The card was handed a file path and re-read the disk.
  * Charge-step tests were duplicated ten times API-level. One road test
    replaced them.

  Each was verified against the code text, fixed and mutation-checked. What was
  left is below.

  A **third pass** over those fixes (one agent, the same day) found six more.
  Each was verified, then fixed (the first two and the folder one
  mutation-checked):
  * a spin left blank on the Transport panel was written as the citation's
    value while the card showed the junction's answer;
  * a record edited since still gave the citation its spin and charge;
  * the peptide-charge advice ran on transport, whose charge is refused;
  * a refused charged citation left an empty folder;
  * a Cu(II) complex was given a Cu(0) reason;
  * the migration quoted a method the file did not contain.
* **P5 open:** the read-back — the engine's own account of the charge and spin
  it used (ES10).

### 5s.4 Found on the way — the items closed or re-homed

> **Verdict:** the IMAX hint PARTLY — the fix stands (`runwrap.py:3289–3305`)
> but `test_runwrap.py` carries no IMAX text: that test was retired in
> `898bee5d`; the trim keeps the item without the test clause. The shared
> panel → § 5u step 8, which already absorbs it (the trim points there).
> `engines/transport.md` § 3.1 CONFIRMED (`:1901–1905`). The Fe hint
> CONFIRMED (`electronic_state.py:384–394, 440`; `validation/chemistry.py:564`).
> A record from before 2026-09-28 carrying no charge: OBSOLETE by rule — not
> a state today's road produces.

* ~~The wrapper's failure hint (`runwrap.py`) still says *"SpinPolarized with
  Spin.Total unset or 0 on a d/f-shell metal also triggers IMAX=0"*~~ — **fixed
  2026-09-29** (P2): the retracted cause is gone and `test_runwrap.py` says so.
* The transport tab's shared panel is not persisted: an adopted citation or a
  restored session resets it to the defaults (UI inventory, not yet reproduced).
* ~~`engines/transport.md` § 3.1's *"the six numbers"* names seven items~~ —
  **fixed 2026-09-29**: it lists what every stage must agree on, the spin
  included.
* **Left by the review (2026-09-29), each named for a decision or a later pass:**
  * ~~**Fe's blank count reads *"2S = 2 is its usual count -- Fe(II),
    intermediate-spin … rare"*.**~~ **Decided and fixed 2026-09-29 -- 5a643599** *(user: "go
    with your recommendations on all three")*: 2 stays -- the four-coordinate
    porphyrin of the hemeC work, the first row of `chemistry-correctness.md`
    § 2.1's own table -- said as *the starting guess*, and the hint names where
    intermediate-spin Fe(II) holds (four-coordinate porphyrins, phthalocyanines)
    instead of calling it rare.
  * **A record written before 2026-09-28 carries no charge**, so a transport
    citation of one reads neutral and cannot refuse a charged run by name.

---

## 5t. The run record — scope and order of work *(W35, 2026-09-26)* (archived 2026-10-08)

*Removed from § 5t on 2026-10-08 by the validation of the same day, which
re-read every item against the code. Each block verbatim, with its verdict.
What stays open is in § 5t; its ids are unchanged. Facts the validation set
first: `JobDirParser` / `parse_dir` are gone (`runs.folder_answer`,
`runs.py:429`); the "folder read alone" rule is retired (`parse.md:1412-1416`);
the tests P1 cites were deleted under the no-saved-run rule (`2b83947f`,
`cdec94da`).*

### 5t.0 Why — measured, by two inventories and a device that diverged unseen

*As measured on 2026-09-25/26, before P1 — P1's row says what changed.*

* **Three regexes for one line, all blind to TranSIESTA.** The wrapper's
  timing tee (`runwrap.py`), the monitor (`monitor.py` `_SCF_LINE`) and the
  parser (`parse/engines/siesta.py`) each match `scf:` and none matches
  `ts-scf:`: the device's timing log held 7 iterations against 1000 and gave
  *"4049.94 s/iter"* (the true rate was 27.5 s); the monitor logged *"no SCF
  progress"* for 7.6 hours; the parser gave the periodic initialization's
  energy (−437,029 eV) for a NEGF loop at −205,444 eV and 584 electrons short,
  and a live device reads *converged* from the initialization's marker.
* **Parameters are recorded as asked, never as used.** SIESTA's own
  `fdf.<timestamp>.log` has no reader; the wrapper's absent list prints
  catalogue defaults under *"the engine default applies"*
  (`negf_eq_pole_ev (catalogue default 0.0)` where SIESTA used the continued
  fraction's own 0.2507 Ry);
  PySCF's three-column read-back is printed only by the optimization script
  and read by nothing; the pseudopotentials are recorded nowhere.
* **The monitor never writes its closing lines**: the wrapper kills it and it
  handles no signal — 0 of 9 monitor logs in `claude-w33` carry
  `[UTIL-SUMMARY]`, so no run records its CPU and GPU means, nor the limit
  `[UTIL-BASIS]` states.
* **The Results tab shows no parameters**: the deck, the validation report
  and `run.json` are reachable only as raw text.
* **The transport record**: written only by `summarize run` and refused
  until the transmission has output; rung attempts picked by file time while
  `.gathered-from` goes unread; `energies_relative_to_ef` hard-coded; the
  spin channels' files missed; a TBtrans run labelled *SIESTA binary* and
  its `.out` claimed by no parser.
* **The contour**: no `contour.eq`, so TranSIESTA's 42-pole continued
  fraction — dQ −29 at step 1 and −584 at the cap; 123 poles on the same deck
  conserved the charge to 0.024 by step 4.
* **The retry**: the wrapper warm-retried the diverged device from its own
  density for three more hours.

> **Verdict:** OBSOLETE — history. The one-line-three-regexes, the unread `fdf` log, the unsigned monitor, the bare Results tab and the record's reader are answered (`siesta_fdflog.py:139-158`; `monitor.py:2231-2233`; `results.md:646`; `record.py:392-399`; `tbtrans.py:21`). Still true, carried by § 5t.3 P3 → § 5x B6: `energies_relative_to_ef` hard-coded (`transport/record.py:537`); the spin channels' files missed (`:453`). The contour and the retry are P4's why, restated in its row.

### 5t.2 Decisions — the rows removed

1. **The run record as scoped**, for every kind — not only transport.

> **Verdict:** CONFIRMED (`parse/dirs/record.py:1-30,509-517`).

4. **The parameters are reported as optimization's are — and better**: with
   what the engine used, and with the deck as it ran (user).

> **Verdict:** PARTLY — built (`parse/dirs/setup.py:1-18,151-205`; `siesta_fdflog.py:126-135`; `record.py:467`); no `echo` column, though `parse.md:1538` lists it. The owed column stays as D4 / P2 in § 5t.

6. **Framework, never a patch** (user: *"we need a systematic and framework
   level design and fix not a patching or hacking … such that all the other
   users can benefit"*). So the record is a declared table of contributors,
   one reader per file (`model/parse.md` § 5d.1b); the Run panel renders any
   record; the SCF plots draw whatever phases and criteria a run states.

> **Verdict:** CONFIRMED (`record.py:509-569`; `parse.md:1466`; `run-panel.js:204-216,282-308`; `scfplot.js:9-15,20-30,89-108`).

7. **The transport result is shown, not listed** (user: *"plots that show
   the convergence of the calculation and … the DOS … in a more graphical
   way"*): each rung's convergence, and T(E), the DOS and the eigenchannels
   as plots. **TBtrans writes everything by default** — device DOS
   (`TBT.DOS.Gf`), spectral DOS from each electrode (`TBT.DOS.A`), electrode
   bulk DOS and transmission (`TBT.DOS.Elecs`), eigenchannels (`TBT.T.Eig`) —
   none of which it writes for a two-electrode junction unless asked
   (`Util/TS/TBtrans/m_tbt_options.F90`). P3.

> **Verdict:** PARTLY — TBtrans's outputs on by default (`data/catalogue.template.toml:2301-2347`); T(E), the DOS and the I–V drawn (`lib/inspectors/transport.js:142`); the eigenchannels and each rung's SCF plot → M5 step 9 / § 5x B6. The remainder stays as D7 in § 5t.

8. **One monitor, every engine, reporting what is going on** (user: *"the
   monitor is a script that will be exposed to everyone. It is run as a
   batch job"*; *"do not reinvent the wheels"*): the existing
   `mb_monitor.py`, launched by the shared part of every wrapper — today its
   launch sits in the SIESTA branch and its files ship only beside `.fdf`
   decks (`runwrap.py`) — reading each run's state through a stdlib reader
   chosen by the output file's role and shipped beside it, and reporting that
   state to the webhook: phase, iteration, energy, each residual against its
   criterion, the charge in a NEGF loop, the forces in a relaxation, and the
   warnings. P4.

> **Verdict:** CONFIRMED via D10 (`runfiles.py:1017-1022`; `tests/test_vibration_e2e.py:172-180`; `monitor.py:113-122`).

10. **The monitor reads through the framework, for every engine — pulled
   forward from P4 as a clean-up** (user, 2026-09-26: *"we have a job
   directory parser that handles this … a whole framework on how to parse the
   directory and judge its status and then read the results"*; *"there's a file
   that can give a copy over. We did this for everything else"*; *"clean up
   all the shit you put together … do this holistically"*). P1 fed the monitor
   the grammar as command-line flags beside a reader of its own; now the
   framework's readers travel as `runwrap.MONITOR_COMPANIONS`
   (`execution/run-reports.md` § 2.3) — `runfiles`, the grammars, the timing
   instrument's rows, the ending scan, `run_status` — the monitor launches
   for every engine and names its files through `runfiles`, and one reader
   per line: the SCF row's values, the forces' `Max` line, the `redata:`
   targets and the molwatch log's lines leave the parsers for the tables the
   parsers and the monitor both read. The record's verdict reads the same
   scan `run_status` makes.

   **Done 2026-09-26, and further than the paragraph above** (user: *"the
   whole output parser should be already written"*; *"the type of the
   calculation can be used as an input"*): **each parser is split into its
   reading pass** — `siesta_reader.SiestaReader`, `molwatch_reader.
   MolwatchReader`, stdlib, shipped — and the registered parser only builds
   arrays from it; the monitor feeds that same pass the output as it grows,
   so both side readers (`live_state`, `criteria`) are gone. The step is the
   one SIESTA states (`Begin <CG|Broyden|FIRE> opt. move = N`, `Begin FC step
   = N`; a single point states none), so the output says what a step is and
   no calculation needs telling. **The wrapper asks, it does not grep**: its
   propor hint and warm retries ask `_run_ending.py` (`_mb_ending`) over the
   output and SIESTA's stderr, whose `die` flushes stdout on node 0 alone;
   the cause is the FIRST fatal line (it was the last, which `die`'s own
   `Stopping Program` cascade always is). **What a report carries is one
   declaration** (`report_fields.py`), each field saying which engines and
   calculations can state it: the Task-setup card is offered only those
   (`GET /api/notify/report-fields`), a description naming another is
   refused at save, and the four hand-kept copies are gone
   (`engines/stages.md` § 6.9). **In direct mode the job is its process
   tree** over the cores it was launched on, and a GPU is sampled only for a
   run that uses one (`run-reports.md` § 2.1a). Checked on real runs through
   the road — SIESTA relax and FC, PySCF water (`tests/test_siesta_vibration_
   e2e.py`, `tests/test_vibration_e2e.py` now read the monitor's closing
   lines) — which found two defects the unit tests could not: the first CPU
   sample taken over two microseconds (262144 %), and a progress log the
   PySCF deck rewrites in place read on from the seed's old offset, so the
   force had no tolerance beside it. Both fixed and mutation-checked.

   **Reviewed the same day by two agents, every claim re-verified against the
   code, and corrected.** The door's cause is what SIESTA says stopped the
   run — `SCF_NOT_CONV … (required).` makes the SCF's failure fatal
   (`Src/siesta_forces.F90`), and the warm retry asks exactly that, so a
   tolerated non-convergence followed by another crash is not retried; a
   run with no output still has its stderr heard. The monitor's cores are the
   launcher's (serial, OpenMP-only, pure MPI, hybrid), a PySCF GPU run is
   sampled, a deck name the catalogue cannot read back is said in the log
   instead of killing the monitor at start. Each parser states its own
   residuals; a PySCF step's SCF iteration and ΔE, |g|, ddm reach the report;
   `n_iters` unstated is absent, not 0; every finished step is an SCF-converged message, a
   PySCF relaxation's first included; the seconds per iteration exclude a
   relaxation's step boundaries (0.84 → 0.58 s on a real H2 run); the
   process-tree reading takes no rate from a read a reaped rank crossed.
   `max_force` is offered for every run (force-constant runs and transport
   rungs state one). The shipped modules are nobody's deck: a `.py` naming no
   `JOB` is no label, so a prepped PySCF attempt has its record again, and the
   SIESTA sniffer no longer claims `siesta_reader.py` as an output. The
   Task-setup card keeps a description's report list when it could not ask
   which fields apply, and takes only the latest answer.

> **Verdict:** CONFIRMED (`siesta_reader.py:1-20`; `molwatch_reader.py:1-10,30`; `siesta_grammar.STEP_BEGIN:61`; `runwrap.py:1468-1510`; `_run_ending.py:82-87`; `report_fields.py:49-66`; `notify_setup.py:282`; `stages.md:1644,1689`; `monitor.py:661,786-805,1862,2012-2014`; `runwrap.py:3322-3336`; `siesta_grammar.py:525,546`; `MONITOR_COMPANIONS` `runwrap.py:3644`; `MONITOR_BUNDLE` `runfiles.py:835`). One sentence of it is now wrong by design and stays open in § 5t.4: `max_force` offered to a transport rung (`report_fields.py:58-61`) → § 5x B6.

### 5t.3 Phases — the rows removed

| | work | done when |
|---|---|---|
| **P0** | the contract and this plan | the user has read them |

> **Verdict:** CONFIRMED.

| | work | done when |
|---|---|---|
| **P1** | **the grammar and the readers**: the SIESTA-family line table; the parser's NEGF phase (`ts-scf:`, `ts-q:`, `ts-Vha:`), the start-up echo, the charge distribution, the electrode checks, `Emadel`, start and end of run; the `fdf`-log reader; the TBtrans reader, spin channels included; the timing tee and the monitor rendered from the table; the monitor's closing summary on SIGTERM | the 2026-09-25/26 device outputs parse to 7 periodic + 1000 NEGF cycles carrying dQ; the timing log counts NEGF rows; the monitor reports progress through a NEGF loop — each pinned on a trimmed real output, mutation-checked. **Done** 2026-09-26, then **reviewed the same day by five agents and corrected**: `parse/engines/siesta_grammar.py` is the table — the SCF row of both phases built from one spelling and case-blind in every reader (the user's 2026-05-28 rule), the ending markers (moved from `_run_ending`, whose four literals the parser had retyped), `SCF cycle continued`, the TranSIESTA lines (spin-polarized columns, contour segments), the build header and the launch lines with one reader each (`read_launch_line`: serial mode is one rank, and `bench`'s private rank regex is gone), the geometry-step line; the parser's rules read the table's patterns; the cheap ending scan is phase-aware, guarded on both devices; `siesta_fdflog.py` keeps every reading of a key read twice (the pole energy: 0.1102 then 0.2507 Ry, the second in effect) on fdf's own label rule; `tbtrans.py` per spin pass; the tee and the monitor rendered from the table — the monitor reporting E_KS, not Eharris; the timing instrument per phase, one level, the headline the NEGF loop's; the monitor's stop: a lock-free flag, SIGTERM for the job's end and SIGUSR1 for a warm retry — which sent a false *it ended* until the review — no sample counted after the stop, and the wrapper waiting for its closing lines (a race the batch caught). ~~Pinned in `tests/parse/test_siesta_negf_phase.py`, `test_engine_used_parameters.py`, `test_tbtrans_out.py`, `tests/test_monitor.py` (the shipped copy, molbuilder unimportable, both signals) and `tests/test_run_ending_one_table.py`, every one mutation-checked.~~ **Not measured, so not pinned:** the spin-polarized TranSIESTA rows and TBtrans passes (no polarized transport run exists), `SCF cycle continued`, and the per-phase timing — its rows are the device's own, its epochs constructed until P3's first device run writes a two-phase log |

> **Verdict:** PARTLY — the mechanisms are present (`siesta_grammar.py:27-46,504,552,568-608,661,887-944`; `bench/result.py:174`; `siesta_fdflog.py:40-48,126-135`; `tbtrans.py:71-75`; `monitor.py:2103-2119,2231-2233`; `runwrap.py:3252`; `scf_timing_rows.py:1-58`). The test sentence is struck: those tests were deleted under the no-saved-run rule (`2b83947f`, `cdec94da`); only `tests/parse/test_tbtrans_out.py` remains. Nothing of P1 is open.

| | work | done when |
|---|---|---|
| **P2** | **the record**: `RunDirResult.record` through `parse_dir`; `/api/results/dir` serves it; a Run panel for every kind — computation, setup with asked ≠ used first, the deck, the verdict — **with the review's eight corrections, § 5t.5, decided 2026-09-26**; the record composed from a declared table of contributors; one parameters fence for both engines | on the dev server, an optimization, a vibration and each transport rung show their Run panel; the SCF plots draw each phase against its own criterion |

> **Verdict:** PARTLY — built (`web/results.md:646`; `scfplot.js:89-108`; `parse/dirs/setup.py:151-153` → `script_emit.declarations:796-802`); `parse_dir` is gone (`runs.folder_answer`, `runs.py:429`). Owed, and rewritten as § 5t's P2 row: the `echo` column; `compare_asked_to_ran` called only by the bench (`jobset/summarize.py:178`).

| | work | done when |
|---|---|---|
| **P3** | **the transport report** from the rung records, composed on read: `.gathered-from` provenance, the E_F reference checked, both spin channels, the device's NEGF facts; **shown, not listed** — each rung's convergence plotted by the SCF-progress component the trajectory viewer uses, and the results plotted: T(E) per bias point and channel, the I–V curve, the device DOS, each electrode's spectral and bulk DOS, the eigenchannels — TBtrans writing all of them by default (decision 7), read by one reader of its output kinds (`parse/engines/tbtrans.py`) | the `claude-au-bdt-au` ladder (TD8; it was `claude-w33`), read in the browser, its plots drawn from a real TBtrans run — that run exists since 2026-09-29, the DOS and eigenchannels among its files |

> **Verdict:** OBSOLETE shape (`transport/record.py:87,172,453`) — the work is § 5x B6 + M5 step 9; § 5t's P3 row is that one pointer.

| | work | done when |
|---|---|---|
| **P5** | **the document sweep** — § 5t.4 and the inventories' stale statements | the review finds none |

> **Verdict:** PARTLY — the Watch tab gone from the documents; byte-for-byte corrected (`transport.md:3234`); the PROVENANCE sentence corrected (`job-contracts.md:1145`). Left, in § 5t's P5 row: the Makov-Payne script; § 5v.

### 5t.4 Found on the way — the rows removed

* **A TranSIESTA device's forces are reported as if they meant something.**
  Measured on `claude-au-bdt-au`'s device, 2026-09-29: TranSIESTA prints
  *"TranSiesta will NOT update forces … ALL FORCES AFTER TRANSIESTA HAS RUN
  ARE WRONG"*, and the monitor's closing line still says *"max force 2.792
  eV/Ang"* — the device deck writes `WriteForces .true.` from the shared
  output section. The device's record reports no force, and the deck need not
  ask for one. P3 (the device's NEGF facts).

> **Verdict:** OBSOLETE half — the deck and the record are K1's (`transport/record.py:293-340`); the open half, `max_force` still offered to every run (`report_fields.py:58-61`), is one line in § 5t.4 → § 5x B6.

* The Watch tab was retired on 2026-05-19 and documents still name it —
  `execution/run-reports.md` § 2, `execution/job-system.md` and
  `execution/job-contracts.md` (not `web/trajectory.md`, which names only the
  live `/api/watch/*` routes, as this item said). P5.

> **Verdict:** CONFIRMED fixed.

* ~~A PySCF run's manifest promises the monitor, `util.csv` and SCF-timing
  files, which the wrapper never writes for PySCF (`runfiles.py`).~~ **Fixed**
  in P1: the three rows are SIESTA's.

> **Verdict:** REFUTED as written (`runfiles.py:1017-1026`): the monitor's two files are every engine's since decision 10; only the SCF-timing tee's row is SIESTA's. Rewritten in § 5t.4.

* The SIESTA SCF-residual plot draws `DM.Tolerance` over the dHmax trace.
  SIESTA's dHmax criterion is in the `.out` for both phases —
  `redata: Hamiltonian tolerance for SCF` and `ts: SCF Hamiltonian tolerance`
  (this item said "only in the `fdf` log") — beside which of the criteria are
  required. P2.

> **Verdict:** CONFIRMED fixed (`scfplot.js:89-108`; `siesta_grammar.py:770-786`).

* ~~`web/trajectory.md` says a SIESTA `.out` carries no time of day; it
  carries `>> Start of run` and `>> End of run`.~~ **Fixed** in P1.

> **Verdict:** CONFIRMED (`web/trajectory.md:138-141`).

* `engines/transport.md` §§ 6.1 and `jobset/prep.py` describe the gather's
  gate as *byte-for-byte*; it is `same_calculation`. P5.

> **Verdict:** CONFIRMED fixed (`prep.py:1800-1818`; `script_emit.py:123-139`; `transport.md:3234`).

* `execution/job-contracts.md`, `runwrap.py` and `parse/contract.py` say a
  TranSIESTA deck carries no PROVENANCE block; it does. P5.

> **Verdict:** CONFIRMED fixed (`parse/contract.py:338-342`; `runwrap.py:3425-3453`; `job-contracts.md:1145`).

* `transport/record.py` reads tbtrans's reported voltage and drops it, keeps
  its own copy of the current line (`_CURRENT_RE`) and globs `.TBT.AVTRANS_*`
  alone, so a polarized point would read as pending. P3: it moves onto
  `parse/engines/tbtrans.py`.

> **Verdict:** OBSOLETE → § 5x B6 (R1: `tbtrans.transmission_files` replaces the record's own glob); the glob at `record.py:453` is named in § 5t's P3 row.

* **Found by the 2026-09-26 review, before W35** — the solver lines
  (`parse/engines/_diag.py`): SIESTA 5.4.2 prints `diag: Algorithm`, never the
  `redata: Diagonalization algorithm` / `Diag.ELPA.GPU` lines `REQUESTED`
  matches (its `diag:` and `* ProcessorY, Blocksize:` patterns do match real
  lines), so `runtime_info["siesta_diag"]` is empty for every real run and
  the trajectory viewer's solver badge is fed only by
  `tests/test_siesta_runtime_info_build.py`'s invented header — whose other
  lines (`* Running on 4 MPI processes`, `ELPA support`, `End of run:` without
  `>>`) no SIESTA prints either; `tests/test_cli_runtime_info.py` uses the
  same stub. P2 folds the solver lines into the table and retires both.

> **Verdict:** PARTLY by the report (`siesta_grammar.read_diag_line:698-708`; `siesta_reader.py:939-940`; `tests/test_cli_runtime_info.py` 18 lines); read 2026-10-08 while trimming: `tests/test_siesta_runtime_info_build.py` does not exist, and `test_cli_runtime_info.py` is a CLI refusal on a file no parser reads — no invented header. Nothing open.

* `parse/instruments/monitor.py` reads `[UTIL-SUMMARY]` only; the limit and
  the kernel's peak on `[UTIL-BASIS]` have no reader (`model/parse.md`
  § 5c.1). P2.

> **Verdict:** CONFIRMED fixed (`parse/instruments/monitor.py:41-47,81-90`).

* **Decided (user, 2026-09-26)** — *"the monitor … is different from the
  result tab … you can make up your correct decision about that side"*: the
  monitor's closing line says whether a relaxation's geometry relaxed, as the
  output states it (`RunEnding.relaxed`). The Results tab's verdict is P2's.
* **Decided (user, 2026-09-26)** — *"can't we just put it in one single
  Python file?"*: the monitor and the modules it reads through travel as ONE
  file, `mb_monitor.pyz` (`runwrap.MONITOR_BUNDLE`), a Python zip application
  holding each module's own file unchanged; the wrapper runs it, and asks
  `mb_monitor.pyz ending …` how a run ended. Fourteen files had stood beside
  every deck.
* **Decided (user, 2026-09-26)** — *"the status can be detected … part of the
  parser for the job directory"*: the directory door answers a prepped,
  never-launched attempt `pending` and a launched, silent one `queued`, from
  its `run.json` (`run_status`'s `launch`) — the rule the jobset layer kept
  above the door, which answered the same attempt "running".
* **Decided (user, 2026-09-26)** — *"Starting and finishing message is okay.
  No stalling"*, and the SCF-convergence messages are by design: a run tells
  its channels when it starts, on each converged SCF and every N hours as
  ticked, and when it ends; the monitor judges no stall — no `[STALL]` line,
  no stall message, no `--stall-heartbeat`, no rate hidden as "stalled"
  (`run-reports.md` § 2). The stall ping came in on 2026-06-27 (`01ba00d5`)
  and went to every channel from 2026-08-26 (`958d46d2`).
* **Decided (user, 2026-09-26)** — *"It shows what it is. Not finished, error,
  or finish"*: `run_status` has no `stale` state and no age rule — a file with
  no ending is `running`, not finished, however long it is quiet
  (`running-a-job.md` § 4.2). The monitor, which sees the PID go, says a run
  with no ending and no exit recorded `failed`, stopped before its end.
  `stale` came in on 2026-06-19 (`d620b1da`).

> **Verdict:** CONFIRMED, all five (`_run_ending.py:79-81`; `monitor.py:173-175,209-210,409`; `runfiles.py:835`; `runwrap.py:3642-3674,3725-3729`; `run-reports.md:206-207`; `job.py:231-238,398-408`; `runstatus.py:232-249`; `monitor.py:75,1826`; `run-reports.md:105`; `job.py:237,350-351,395-397,472-474`; `running-a-job.md:596`).

* **Built ahead of P2, found by the 2026-09-26 review**: the run record
  (`parse/dirs/record.py`, `setup.py`), `JobDirParser` serving `status` and
  `record`, `/api/results/dir` carrying it; the Run panel
  (`lib/results/run-panel.js`, `web/results.md` § 3a, 2026-09-27); a folder
  read alone -- a SIESTA run named by hand, an output copied without its deck
  -- has its record too (`model/parse.md` § 5d.1, 2026-09-27). **Decided
  (user, 2026-09-27)**: a fact shown twice is not a fault; one fact from two
  sources is -- *"the key is to have information source unified rather than
  worrying about repeats"*; the viewers keep their lines. The trajectory
  viewer's seconds per iteration and its badge's *ended* time are read from
  their one source -- the SCF-timing log through the record's reader, and the
  output's own end (2026-09-27; its three estimates are gone). **P2 still
  owes**: the setup rows' `echo` column (§ 5d.3);
  comparing the launch rows asked with those run
  (`bench.compare_asked_to_ran`); the SCF plots by phase; the parameters
  fence narrowed to the calculation's items.

> **Verdict:** PARTLY (`results.py:140-180`; `run-panel.js:4-6`; `trajectory/core.js:1615-1634`). `JobDirParser` is gone and the "folder read alone" rule is retired (`parse.md:1412-1416`); the SCF plots by phase and the narrowed fence are built. The two still owed — the `echo` column, `compare_asked_to_ran` — are § 5t's P2 row.

* **The 2026-09-26 review's eight questions — decided by the user
  (2026-09-26/27)**:
  1. A force-stopped run reads `failed · stopped before its end`: `run_status`
     reads the monitor's closing record after the output's ending and the
     marker (`ffe0e8c4`).
  2. Flat stages record their launch too: `<basename>.run.json` beside the
     stage's deck, read by the ladder, the folder and the run record
     (`project-layout.md` § 1.6.3, 2026-09-27).
  3. A folder read alone that holds only a product: obsolete — confirmed in
     the browser, a folder holding only a spectrum lists it and opens it in
     the spectra viewer (2026-09-27).
  4. One SIESTA ending reader, the reading pass; `ending_of(path)` the one
     door (`ffe0e8c4`).
  5. `run_status` reads SIESTA's stderr as the wrapper does: the session
     log whose first section is the run (`wrapper_log`, which now travels),
     and a stop's detail quotes the line that stopped it (2026-09-27).
  6. The memory peak shown is the job's own: the kernel's counter when the
     job has its own cgroup, the largest sample otherwise — `utilisation()`
     decides, the record and the bench summary carry it (2026-09-27).
     *Unmeasured:* on Sol's cgroup v1 the counter and the samples are read
     from the monitor's own cgroup and the limit from the job's; whether
     ranks launched into another step are inside it is for a Sol run to say.
  7. Tolerances in the residuals' units: PySCF's SCF criteria stated in eV
     in `runtime_info.scf_criteria` beside SIESTA's; the viewer draws every
     residual's line from them (2026-09-27).
  8. No `notify` block, no notification (`ffe0e8c4`).

> **Verdict:** CONFIRMED (`job.py:422-431`; `runrecord.py:177-184`; `_run_ending.py:284,104-133`; `runwrap.py:1507`; `utilisation.py:53-62`; `parse/engines/pyscf.py:229-230`; `trajectory/core.js:1021-1026,1089-1100`; `prep.py:2161-2168`).

* **Found reading the Results tab's code (2026-09-27), open**:
  * ~~the run record is composed on every folder scan and no page shows
    it~~ — the Run panel shows it (2026-09-27);
  * ~~every visit scanned its folder twice~~ — `pageshow` re-scanned on a
    fresh load too, emptying the menu the first scan had filled; the picker
    test that read it failed under load (`2d3b67e0`, 2026-09-27);
  * ~~the vibration deck states no SCF criteria~~ — both PySCF decks read
    them back off their solver through one emitter
    (`pyscf/input.emit_scf_criteria_readback`, 2026-09-27);
  * ~~two tests read old frozen outputs~~ — replaced by a SIESTA run the road
    makes and stops (`tests/test_siesta_stopped_run_e2e.py`, 2026-09-27); the
    device's `fdf` log still answers what only a device run shows (a key read
    as `CG` and `cg`).

> **Verdict:** CONFIRMED. Item (a) stays in § 5t.4, unverified (`_run_ending.py:338-371`).

* **W34's contract wrote `Spin.Fix` beside `Spin non-polarized`** for every
  closed-shell molecule; SIESTA stops on `Spin.Fix` at any spin but
  collinear-polarized (`read_options.F90`). Corrected in the contract
  (`science/chemistry-correctness.md` § 2a.3–2a.4, ES6) with its
  restatements (`science/overview.md`, `engines/siesta.md` — whose "below
  ~1 meV" image energy is ≈ 0.37 eV — `science/validation.md`,
  `execution/running-a-job.md`); W34's P1 builds it so.

> **Verdict:** CONFIRMED (`chemistry-correctness.md:494-495,548-587`; `siesta/input.py:1292-1335`; `electronic_state.py:68-70`).

### 5t.5 P2's design, corrected by the 2026-09-26 review — the corrections removed

1. **Cheap reads only.** The record never builds a trajectory on a folder
   scan — a device `.out` is MBs, and the viewer parses it on mount anyway:
   the `.out` head and tail through the table, the `fdf` log, the wrapper
   log, the instruments, the deck, `.concluded`. Per-phase convergence comes
   from the phase-aware ending scan. **`evolution` leaves the record**: its
   one reader, the plots, reads the trajectory.

> **Verdict:** CONFIRMED (`record.py:22-29,66-73,171-202,266-281`; `siesta_grammar.py:732-739`).

2. **One door.** `/api/results/dir` asks `parse_dir` — the container and
   read-alone rules move into `JobDirParser` first, and the unread `files`
   and `active` go — and the benchmark's `summarize.parse_point` composes
   from the same computation reader.

> **Verdict:** OBSOLETE — `parse_dir` / `JobDirParser` are gone (`runs.folder_answer`, `runs.py:429`); `parse_point` is `bench.parse_effective_run` (`jobset/summarize.py:41`).

3. **Which run.** A record describes the attempt's latest `-runN` and lists
   the earlier ones with how each ended; the `fdf` log pairs with the `.out`
   by its stamp (exact on 122 of 122 real outputs), the wrapper log by the
   first `run index` line it states.

> **Verdict:** CONFIRMED (`record.py:171-202`).

5. **Pseudopotentials** by file, uuid (the `.out` names both), sha256 and
   header — not "matches the library today".

> **Verdict:** CONFIRMED (`record.py:266-281`).

7. ~~**One home per fact on the page**: the trajectory viewer's runtime line
   and the spectrum viewer's Host/CPU/GPU rows go, in favour of the Run
   panel~~ — **superseded (user, 2026-09-27): one SOURCE per fact; a fact
   shown twice is not a fault**, and the viewers keep their lines, the
   redundancy a run copied out of its calculation keeps. The panel is carried
   by the picker's selection event and cleared when the folder changes
   (built, 2026-09-27).

> **Verdict:** CONFIRMED (`run-panel.js:4-21`).

8. **The SCF plots by phase**, each drawn against its own required
   criterion — dHmax against the H tolerance, dDmax against the DM tolerance,
   the NEGF dQ against the charge tolerance.

> **Verdict:** CONFIRMED (`scfplot.js:89-108`).

---

## 5u. Transport — the one list (M5, W43) *(consolidated 2026-09-29)* (archived 2026-10-08)

*Trimmed 2026-10-08 against the validation report (its § 5u rows). Each block
below is the removed text, verbatim, under its verdict. The live § 5u keeps
steps 5, 7, 8, 10, 11 (corrected), the two owed contract texts, TD1–TD14 as a
compact decision record, and the § 5u.3 / § 5u.4 residue as § 5x B9's input.
§ 5x holds the order for transport since 2026-10-08.*

### From the preamble

> **Verdict:** HISTORY — the 2026-09-29 inventory that made § 5u the order; § 5x is the order since 2026-10-08 (report: header pointer H1 → "§ 5u → § 5x holds the order"). The paragraph explains no open item. Replaced by a short note in the live section.

**Why this section exists.** Transport's open work sat in six § 2 rows (W10,
W24, W25, W27, W30, W32), three plan sections (§ 5c.3, § 5o, § 5p with TR1–TR10
and § 5p.3p's steps) and two contract tables (`engines/transport.md` § 3.6 and
§ 3.8.5), written between 2026-09-15 and 2026-09-24. On 2026-09-29 an inventory
read every one of them against the code at `3e55f97e` (file and line for each
verdict), and the load-bearing verdicts were re-read here before this was
written. The finding: much marked open is built, some marked built is not, and
six orders of work disagreed. **This section is the order.** The rows it absorbs
point here (§ 5u.5); § 5c.3, § 5o and § 5p keep the *why* and no longer the
order.

### From 5u.0 What is true today — measured

> **Verdict:** PARTLY (rewritten) — the verb is stale: `_KINDS = ("task", "bench")`, `jobset/_cli.py:768`; every `prep run` / `launch run` / `summarize run` is `… task`. The bullet is kept in the live section with the verbs corrected.

* **The road is the ordinary one.** Five rungs (seed, electrode_L, electrode_R,
  device, transmission), each prepped by `molbuilder jobset prep run <stage>`
  (or the Task setup tab — one entry, `prep_stage`) and launched by
  `molbuilder jobset launch run <stage>`; every rung renders through
  `spec_for` → `prepare_deck` with its `.validation.txt`; `summarize run` writes
  `<label>.transport.json`.

> **Verdict:** OBSOLETE (superseded) — the road junction `claude-transport-walk/transport/au-dta-t` ran every rung to its record on 2026-10-08, the device as a three-point sweep (§ 5x.7). `claude-w33` was retired as an acceptance target by TD8. One superseded line stands in the live section.

* **No ladder on this machine has concluded a device rung** — read in the
  engine's own output. `claude-w33/transport/au333bdt-t` device run-0: 1000
  NEGF iterations, the charge off by 584 electrons (`ts-q` dQ), `SCF_NOT_CONV`,
  abnormal stop. Run-1 (`TS.Contours.Eq.Pole 10 eV`, which TranSIESTA turned
  into 123 poles): the warm-up SCF converged in 7 iterations and the NEGF cycle
  held the charge to 0.004–0.08 electrons and was converging — stopped at NEGF
  step 9, 2026-09-26. **No transmission rung has run since the 2026-08-29 chain
  walk**, whose saved output is a trimmed test fixture.

> **Verdict:** PARTLY stale (superseded) — `web/results.md` § 2.5 (:440-463) built 2026-10-07 (step 9 ①②: DOS, eigenchannels, the chain, per-rung SCF); the remainder is § 5x B6. One superseded line stands in the live section.

* **The Results surface exists** — the `transport-json` parser and
  `lib/inspectors/transport.js`: the I–V table, per-rung cards, T(E) per bias
  with its treatment named. Not drawn: an I–V curve, DOS, eigenchannels, the
  chain of runs that fed a curve. (W10's *"the reader does not exist"* is no
  longer true.)

### From 5u.1 The order — the done steps

> **Verdict:** CONFIRMED done (`4c612262`; `transport.md` :3431-3474, :2284-2292; catalogue `tbt_verbosity` :2422, `electrodes_bulk` :2544; `transport/deck.py:57-74,109-121`). Its two design questions are TD11 and TD12, which stay in step 7.

| **1** | **The NEGF block on the catalogue** — the rulings of `engines/transport.md` § 6.1b *(user, 2026-09-29)*. **Built and reviewed 2026-09-29 (`4c612262`)** — every verified finding fixed; the review's two design questions are TD11 and TD12 | W27 floor 3 (§ 3.6 item 2); § 3.8.5's device-deck row; the `TBT.Verbosity` gap (§ 2a.13) | every `TS.*` / `TBT.*` value in a device or transmission deck is written by the section walk with its note (so the check gate sees it), saying what it decides and which program reads it; the device deck carries no `TBT.*`; `TBT.k` is written `[a b c]`; `electrodes_bulk` (was `elecs_bulk`) is one shared value in both decks, and a template still naming `elecs_bulk` is refused with the new name; `TBT.Verbosity` has its row; `_legacy_view` is gone; the deck's comments state what each program reads — road tests prep a fixture ladder's device and transmission and read the decks |

> **Verdict:** CONFIRMED done, code + contract (`config/siesta.py:1045`; catalogue :2452-2463; `transiesta.py:130,92`; the kind gate `validation/__init__.py:416-544`; `tbt_dos_*` True `config/siesta.py:980-998`; § 6.1c :3591-3607; `compose.py:436-521`). The measurement on the real ladder is UNVERIFIED — `projects/claude-au-bdt-au` is absent; the 10 eV / 123-pole rule is now the contract's (§ 6.1c :3510-3517). TD13 and TD14 stay in step 7; the interim 10 eV is § 5x's parked P3.

| **2** | **Before any device runs: the pole energy stated, vacuum by axis, and TBtrans's outputs on** — TD1 and TD3's rulings, and W35 decision 7 (decided 2026-09-26). **Built and reviewed 2026-09-29 (`b63f219d`)** — the contract is `engines/transport.md` § 6.1c. Found building it: `compose` stated every junction periodic across, so TD3's isolated wire was unreachable; it now carries the relaxation deck's recorded transverse kinds (a deck from before that record is read periodic across, as before; a record saying anything else is refused). Read by a fresh agent in full, every finding re-read against the code and the engine source, and fixed: the transverse measure taken lead by lead and naming the lead (pooled, two leads answered for each other); TranSIESTA's 10 K temperature floor (`m_ts_options.F90`); the refusal's least energy rounded up so it is accepted (`pole_energy_for`); the pole range 1–40 eV (10 eV was both default and ceiling); what an explicit 0 eV does (TranSIESTA keeps 8 poles and stops); a statement beside a non-plain line refused, not dropped; help text that promised what the Results tab draws; two tests of the retired rules retired; the default added to the catalogue guard. Two design questions from it are TD13 and TD14. **Measured on the real ladder**: the Au–BDT–Au device deck states `10.0000 eV # 123 poles at 300 K`, and TranSIESTA reported 123 | TD1; TD3; I12's fixed 3.0 Å threshold; W35 decision 7's defaults | `negf_eq_pole_ev` defaults to **10 eV** and is **always written**; the deck line states the point count it gives at the run's temperature, by TranSIESTA's own rule `N = int(E / (π·k_B·T))`; the note and the form's help explain why an energy sets a count (the measured table in D1's ruling); a stated energy under the 20-point floor stays refused. **The 10 eV is interim**: M3 P4's stated contour (`contour.eq`, W35 decision 2) replaces it, and it supersedes § 5p.3o's *0 = let the engine choose*. The transmission's DOS and eigenchannel outputs default ON (W35 decision 7), so the first ladder writes what M3 P3 draws. The kind gate **refuses** vacuum along the transport axis beyond one layer spacing of the lead (measured from the lead, not a fixed 3.0 Å) and on a transverse axis declared periodic, and **allows** it on an isolated transverse axis, taken from the structure's stated vacuum. Road tests on each rule |

> **Verdict:** CONFIRMED done (`325cb1d4`; `transport/sort.py:52-58`; `compose.py:1028`; `siesta/input.py:603-624`; `ELECTRODE_LABELS` `transport.md:2823-2835`). Step 8 waited on it; that wait is over.

| **3** | **`TransportConfig` retires** — TD4's ruling. **Built and reviewed 2026-10-02 (`325cb1d4`); its three questions decided the same day and built as § 0b items 1–3** (D2: the record speaks the catalogue's names, `e1d88b22`; D3: the page code gone, `2ec16acd`; D4: the leads are two exact names) | TR6; § 3.6 items 6, 7, 12; § 5p.3p steps 3 and 6; X1 ②–⑤; the kind-narrowing defect below (M2l's *transport `restart` refused* is this); V1.11 / A1.16 U10 (compose's permutation record); § 8 row 4 (`labeled_citation_structure`'s globs) | `config/transport.py` holds no config class (the region-label constants and `is_electrode_label` move to one home — five importers); `config_for`, `SEALED_*`, `dataclass_to_form_schema` and the tests that pin them are gone; the recorded-contract vocabulary (`parse/contract.RECORD_TO_SIESTA_FIELD`) is decided; an override naming an item of another kind is refused on every road; compose writes the permutation through `write_permutation`, stamping its key; the transport arm of `spec_for` stops dropping `cell=` in silence (`transport.md` § 3.6a) |

> **Verdict:** UNVERIFIED as a measurement — **FLAG: `projects/claude-au-bdt-au`, TD8's acceptance ladder, is not on disk** (on disk: `AuBDTAu`, `Au-BDT-Au.old`); the record below is a recorded measurement with no folder behind it. The acceptance is § 5x.0's road junction (`claude-transport-walk/transport/au-dta-t`, run to its record 2026-10-08, § 5x.7). Kept here as the record of what was measured on 2026-09-29.

| **4** | **The first ladder to its end** — Au–BDT–Au (`claude-au-bdt-au`), one computation at a time, each rung's engine report read against its deck (the first `tbtrans` report is the check on § 6.1b). The seed and both leads **concluded 2026-09-29** (39, 15 and 13 SCF iterations). **The device and the transmission concluded the same day, on step 2's code** — the device with warm retries off (`continue_retries = 0`; its wrapper: *"Retry policy: none"*): TranSIESTA took 123 poles, its NEGF cycle converged in 13 iterations (*"SCF Convergence by DM+H+dQ criterion"*), the charge held to −0.0255 of 1408 electrons; `tbtrans` wrote T(E), the eigenchannels, the device and spectral DOS and the leads' bulk DOS and transmission (973 s, 6 ranks); `summarize` wrote `aubdtau-T.transport.json`: **T(E_F) = 0.161, G = 0.161 G₀** at 0 V, one eigenchannel carrying 98.5 % of it — inside the 0.1–0.4 G₀ GGA-NEGF range the contract's caveat names against ≈ 0.011 G₀ measured (§ 2). Energies are relative to the device's Fermi level, traced: `tbtrans` subtracts the `Ef` stored with the Hamiltonian (`m_handle_sparse.F90` `reduce_spin_size`, from `m_tbt_hs.F90:417`). **Shown on the Results tab** the same day, on the restarted dev server: the I–V row (0 V, G(E_F) 0.1613 G₀, 0 A), each rung's state — the leads' E_F agreeing at −1.416 eV — the treatment named (*single bias … the LINEAR-RESPONSE approximation*), T(E) drawn against E − E_F, and the provenance with the cited files' hashes. **The first `tbtrans` report, read against § 6.1b**: *"Bulk H, S in electrode region = T"* on both leads from `TS.Elecs.Bulk` alone; 300 K from `ElectronicTemperature`; the leads' eta the deck's 1 meV and the device's 0.1 meV by `tbtrans`'s own rule; `TBT.k` read as a list (its fdf log: `#:list? TBT.k T`); and the lead's inter-layer distance 2.4006 Å — the spacing § 6.1c's vacuum rule reads. **Step 4 done 2026-09-29.** Whether this ladder is also the acceptance of E13's workstation half, W33 P5 and W35 P3's done-when is TD8's. Steps 1–3 leave the seed and lead decks unchanged — step 1's review found them byte-identical — so those concluded runs stay valid | E13 (the workstation half); W32's gate; W35 P3's done-when; W33 P5 (TD8) | a `<label>.transport.json` from a run we watched finish, shown on the Results tab |

> **Verdict:** PARTLY (rewritten) — the group is built (`jobset/group.py`; `submit.py:2162`; `model.py:382`; `project-layout.md:1056-1074`). The done-when column is OBSOLETE: `_plan_chain` is gone (§ 5x B4's one walker `_walk_script` replaced it), and *"a rung whose concluded attempt matches its deck is never rerun"* contradicts § 5x decision 4 (a swept stage launched again is rule 3: warm takes the done points over; `--cold` runs every point). The verbs `prep run` / `launch run` are stale. The live row keeps the built summary, the *Left* (the frame walk → step 11; the group's header on a scheduler) and a done-when for the frame walk only.

| **5** | **Submission groups — one queue wait per rung, not per point** — TD2's ruling; its order against M2f, M2g, M2i and M2k is TD9. **The group built 2026-10-07** *(user: "group at prep")*: `prep run seed electrode_L electrode_R` names a group -- each member prepped as alone, one save, none building on another (refused by name), one shared allocation; `Job.group`; the group's header at prep; `launch run` by the same names sends one job walking them (`project-layout.md` § 1.6.6; `jobset/group.py`). Walked on the carbon chain (`chain-g`): one job, three runs, the ladder's result unchanged. **Left**: the frame walk (step 11); the group's header checked on a machine with a scheduler | TR9; § 2a.7's *default grouping*; W32 ③'s *the gather runs per frame* | the framework the bias chain already is (`_plan_chain`: one submission walking prepared attempts, each running its own `.run.sh`), generalised: the seed and both leads as ONE submission, shared by every bias point and every frame; the device and the transmission ONE submission per scan, walking bias points (warm-chained, stopping at a failure) or frames (each from the shared seed, continuing); members stay prepared attempts with their own records; the allocation sized from the members' own requests, a member that cannot share it refused by name; the order and the stop-or-continue rule read from the ladder's own `stage_inputs` and the axis's declaration. Reuse is the gather's: a rung whose concluded attempt matches its deck is never rerun. Written into the contract first — which amends `project-layout.md` § 1.6.6 (*"a `JobSet` carries no scheduler dependency … no field in a description, and no flag at launch, is permitted to make it"*) as well as `task-setup.md` § 1; § 5p.5's argument covered the independent preparatory rungs only |

> **Verdict:** OBSOLETE — superseded by § 5x B1 (`7351a3ff`): the treatment is one explicit switch, `bias.low_bias_approximation` (`task.py:420`; `_cli.py:519`; `migrate.py:201-206` migrates the old `treatment` word; `record.py:531-533,660-674`; `transport.md:1233-1255`); the tab's half (`transport_calculation.html:169-180`, `core.js:836-844`) was not seen in a browser by the report. The `chain-*` walks are retired as evidence (§ 5x.0). TD7's ruling stays in the decision record.

| **6** | **The bias treatment, named** — TD7's ruling. **Built 2026-10-08** (`transport.md` § 2a.10, *The choice, as stated and as run*): `task.json` `bias.treatment` (`low-bias` / `re-converged`), stated for several voltages or refused, `jobset init --bias-treatment`, migrate writes `re-converged`; under `low-bias` the device has no axis and each transmission point gathers the 0 V Hamiltonian, TBtrans integrating its own window; the record and the report say the treatment. Walked on the chain (0/0.2/0.4 V). The tab's choice and the start/stop/step builder built with Q13's T7 (2026-10-08) | § 2a.10's advisory; § 2a.12's label | the tab asks which: *the low-bias approximation* (one device converged at 0 V; every I–V point from its T(E)) or *re-converged at every nonzero voltage*; the result's label and the record carry the treatment and the voltages it was computed at *(and, user 2026-09-30: "as long as the sweep can be easily setup in the UI" -- the bias list is set up by a start / stop / step builder that fills the list, still editable and always 0 V first, one list for both treatments: under the low-bias approximation the voltages reported, under re-convergence the chain of runs)* |

> **Verdict:** fragment of step 7 — `§ 11 P10 (the T(E) window test)` is done (§ 11 archived 2026-10-08); dropped from the live row's done-when. The rest of step 7 is open and kept.

`§ 11 P10 (the T(E) window test);`

> **Verdict:** fragment of step 8 — "after step 3": step 3 is done (`325cb1d4`, 2026-10-02); the wait is over. The row is open and kept with the report's cites (`transport.md:2771`, :2688).

`**One panel per engine** — after step 3`

> **Verdict:** PARTLY (rewritten) — ①② built (`results.md` § 2.5 :440-463); ③ waits on step 11; the rest is § 5x B6 (R1–R4, R6, R7, R9, the low-bias I(V)). The done-when's preconditions are superseded: M2l is done; the record picks a sweep's points from its newest run (`record.result_folders`, § 5x B2); the chain and the caveat are on the road junction's report (§ 5x.7). The live row points at B6 and names the two done-when items B6's row does not spell out (the presenters' door; W30 ④'s deck viewer).

| **9** | **The Results surface, finished** -- **designed 2026-10-07, `web/results.md` § 2.5, approved (user: "yes, go ahead with the build"); building: ① record fields (DOS, PDOS by label, eigenchannels, per-rung SCF, NEGF figures, chain, caveat) + the shared SCF-plot module; ② the report's cards, tabs, two selections, PDOS of a selection by atoms and orbital type; ③ the frame axis with step 11** | W10; § 5c.3 (e) and (f)'s rung half — (f)'s citation half is built (`transport.js` draws `provenance.slot`); TR10's chain; W30 ④ (the deck viewer) | after M2j (status owns the ladder) and W38 M4 in M2l (the record picks attempts by file time), and planned with M3 P3 and M6 P5, which change the same record: the curve carries the chain of runs that fed it (`.gathered-from`); presenters read through a door rather than `JSON.parse` (a framework door, not transport's alone); the I–V curve; the decks viewable per rung, attempt and bias point beside their `.validation.txt`; the DFT–NEGF caveat carried on the result (`transport.md` § 2: *"surface this caveat honestly rather than imply DFT-NEGF == experiment"*) |

> **Verdict:** fragment of step 11 — "after steps 4 and 5": step 4 is done; step 5's remainder *is* the frame walk. The row is open and kept, pointing at step 5's *Left*.

`**The frame axis** — after steps 4 and 5`

> **Verdict:** rewritten — "swept once at M5's end" is § 5x B9 now; the live line says so.

**Beside every step: the documents.** Each step fixes the false claims and stale
citations in the documents it touches (§ 5u.3, § 5u.4, § 5v's transport part);
what is left is swept once at M5's end.

### From 5u.2 Rulings

> **Verdict:** the decision record — COMPACTED to a list in the live section (every id kept; the rulings' gist and where each lands). The questions as put, the user's words and the *(was: recommended …)* notes are kept here verbatim. TD9 is still open. TD8 is PARTLY: its ladder's folder is absent — § 5x.0's road junction supersedes it (flagged to the user in the live list). TD7's form is superseded by § 5x decision 1.

| # | the question | the ruling | lands in |
|---|---|---|---|
| **TD1** | the device's equilibrium contour: TranSIESTA's fallback (42 points at 300 K) lost 29 electrons at step 1 and 584 by step 1000 on `claude-w33`'s device; the same deck with `TS.Contours.Eq.Pole 10 eV` (123 points) held the charge to 0.024 by step 4 — the two decks differ in that one line | *"is there a best default value … so we can put in the template as a good starting point, with correct WebUI and script comment … and make it always enter the script? … all parameter will be explicitly set in the .fdf"* — **10 eV, always written, the count stated beside it** — interim until M3 P4 states the contour. The unit confused the user and will confuse readers, so the note says it plainly: an ENERGY, because TranSIESTA counts its points by how far up the imaginary energy axis they reach — at 300 K, 1.5 eV = 18 (refused), nothing = 42 (lost the charge), 4 eV = 49 (the manual's *at least 50*), 10 eV = 123 (held it); the same energy gives fewer points hotter. The stated contour itself (`contour.eq`, W35 P4) stays in M3 unless the first device run says otherwise | step 2 |
| **TD2** | grouping the ladder's submissions (TR9) | *"in a HPC environment submit a task is heavy … combining small chunk of task into one where in a sequence of tasks the components are not required to be repeated … can benefit from less such scheduling burden"*; *"the transport is also built to work with small variation in structure (… multi-frame structures where only the molecule bridge changes), so this combined task could help"*; *"not a hack but a sequence-aware design framework"* — **build it, on the chain submission** | step 5 |
| **TD3** | a box for a lead; vacuum in a transport structure | *"we allow user to explicitly set vacuum in structure … we will refuse vacuum in periodic axis … for transport axis … no vacuum is allowed either"* — **checked and confirmed, with one precision and one exception.** Along transport the leads continue through the cell boundary into the periodic image, so a gap makes the lead a surface: refuse — but the empty span there is exactly one layer spacing of the lead (2.35 Å for this gold, as TranSIESTA reported), not zero. On a transverse axis declared periodic, vacuum contradicts the declaration: refuse. On a transverse axis declared **isolated** (a wire or chain lead) vacuum is what isolates it — the standard TranSIESTA setup — so it stays, from the structure's stated vacuum, and so does the box built from it. The k-grid along transport is 1 on the device and the seed and dense on the electrode rungs (a lead is bulk along the wire — the same continuity) | step 2 |
| **TD4** | retire `TransportConfig`? | *"if this is only related to parameter/script generation, then i agree this can be retired and result presentation can use other .json file"* — it is: the Results tab reads the record (`<label>.transport.json`) and the ladder's status, never the class. **Retire** | step 3 |
| **TD5** | how a value records where it came from (`engines/template.md` § 6.6's three proposals) | *"yes to your recommendation"* — **all three, as proposed** | step 10 |
| **TD6** | per-rung convergence defaults (§ 2a.7) | *"yes, but in general i believe the convergence condition is always exposed to user and the user can have the final say"* — **decided after the first ladder**; every rung's convergence settings stay on its own tab, the person's to change | after step 4 |
| **TD7** | the bias treatment as a named choice | *"the user can enforce low-voltage approximation for all I-V or require re-seed/re-convergence for non-zero bias? if so, i agree, and result presentation should have the correct label/comment … same with the actual data record"* — **build it** | step 6 |

| **TD8** | **one acceptance ladder.** Three transport ladders are each named as some milestone's acceptance: `claude-w33` (W33 P5, W35 P3's done-when), `claude-au-bdt-au` (step 4), and `claude-vib-ui`'s fake junction (W33: *"paused at rung 4 until P5"*) | **ruled 2026-09-29** *(user: "TD8 yes")* — **`claude-au-bdt-au` is the one acceptance ladder**: it closed step 4, and E13's workstation half, W33 P5 and W35 P3's done-when are checked against it; `claude-w33` and `claude-vib-ui`'s fake junction are retired as acceptance targets (their folders stay) *(was: recommended: `claude-au-bdt-au` closes step 4, E13's workstation half, W33 P5 and W35 P3's done-when; the other two are retired as acceptance targets)* | step 4 |
| **TD9** | **step 5 against M2.** Grouping reads what four M2 milestones rebuild: which attempt *concluded* (M2f, F3), the bias chain's hand-spelled restart list (M2g), a seed skipped by removal rather than `enabled` (M2i), one placement decision and record for an allocation (M2k, F1/F6) | **open** — recommended: steps 1–4 now, then M2f, M2g, M2i, M2k, then step 5 onward | step 5 |
| **TD10** | **the displacement generator's one home.** W32 ④ (V1.25, the 2026-09-24 contract) and W42 ③ (M10: many modes, Gauss–Hermite nodes) both build it | **ruled 2026-09-29** *(user: *"we don't need a procedure to build displacement along a mode. the transport is presented with a multi-frame structure when a serious of positions need to be done in sequence. the user is responsible to make sure they are the same structure with only relevant atoms moved, and that the meta data \"customized\" section contains parameter sets that will tell the details of each frame (such as displacement of each frame, the frequency of the mode, etc). the multi-frame structure can be generated by a set of backend procedure that can be designed separately that takes the equilibrium relaxed structure and the normal mode as input, and correctly handle meta data."*)* — **transport builds no displacement.** It is handed a multi-frame structure (one pair, many frames); the person is responsible for the frames being one structure with only the relevant atoms moved, and the structure's `customized` section (W39) carries each frame's parameter set — its displacement, the mode's frequency and whatever else tells that frame's details. The generator is **a separate backend procedure**, designed on its own (V1.25): the equilibrium relaxed structure and a normal mode in, the multi-frame structure out, its metadata handled. W32 loses ④; W42 ③ uses the procedure | step 11 · V1.25 · W39 |
| **TD11** | **`electrodes_bulk`'s scope** *(step 1's review)*. Only the device (`TS.Elecs.Bulk`) and the transmission (`TBT.Elecs.Bulk`, which defaults to it) read it. It is declared `shared = ["transport"]` — one value, bound to every rung of the kind, the seed and the leads included, where nothing reads it. The markers cannot yet say *one value, carried by these named rungs*: `shared` is every rung, and `stages` names rungs that may each answer differently (`engines/template.md` § 5, the `stages` table) | **ruled 2026-09-29** *(user: "TD11 a")* — `shared` and `stages` declared together mean **one value, carried by the named rungs only**; `electrodes_bulk` declares both, and the shared panel says which rungs read it *(was: recommended: `shared` and `stages` declared together mean one value carried by the named rungs only; `electrodes_bulk` declares both, and the shared panel says which rungs read it)* | step 7 |
| **TD12** | **what the transmission deck carries** *(step 1's review)*. It carries the SCF and output controls a `siesta` rung needs, which `tbtrans` never reads — but `tbtrans` does read some SIESTA labels by fallback (the temperature chain; `TBT.k` falling to `kgrid.MonkhorstPack`) | **ruled 2026-09-29** *(user: "for TD12 would it be necessary to cut redundancy? i feel a comment or log showing what are the parameter taken by tbtrans would be enought. there is no pont adding complexity between steps")* — **nothing is cut.** The transmission deck keeps the ladder's shared settings; its header says which program reads what — `tbtrans` reads the `TBT.*` settings, the `TS.*` declarations and, by fallback, `ElectronicTemperature`, and the SCF and output settings are the ladder's shared set it does not read — and points to `tbtrans`'s own fdf log in the run (`fdf.<time>.log`), which records every key it read, the engine's word rather than ours | step 7 (the header only) |
| **TD13** | **correcting a junction's transverse kinds without a new relaxation** *(step 2's review)*. `compose` takes the kinds from the cited attempt's deck, a launched attempt is never rewritten, and a calculation composed before keeps its junction's kinds (`load_compose_record`) — so today a wire declared periodic by mistake needs its structure re-declared, relaxed again and re-cited. Options: a transport-side declaration of the transverse kinds that defaults to the recorded ones; an edit door on the cited deck's record, as `swap_electrode_labels` is for the labels; or the re-run, stated plainly (what the refusal and § 6.1c say now) | **ruled 2026-09-29** *(user: "TD13 a")* — **a transport-side declaration of the transverse kinds**, one shared item defaulting to what the relaxation's deck recorded: the person's word is stated where the calculation is described, and the record stays what the relaxation ran *(was: recommended: the transport-side declaration, one shared item defaulting to the record, so the person's word is stated where the calculation is described and the record stays what the relaxation ran)* | step 7 |
| **TD14** | **whether the transverse check should also catch a cell that does not tile** *(step 2's review)*. It catches vacuum; a multi-layer fcc(111) lead in a cell one atom column too long is still bonded across the boundary through its next layer (1.41 of its bond) and passes; a single-layer lead does not. § 6.1c states the limit | **ruled 2026-09-29** *(user: "TD14 a")* — **the transverse check measures within each atomic layer of the lead**: it asks whether each layer tiles, not whether the stack bonds, so a cell one column too long is caught on a thick lead too *(was: recommended: measure within each atomic layer of the lead, so the check asks whether each layer tiles, not whether the stack bonds)* | step 7 |

> **Verdict:** not an open item here — the deferral is § 2a.7's own ruling and lives there; nothing in § 5u schedules it.

*(Automatic resubmission is deferred by § 2a.7's own ruling — it waits on whether compute nodes may submit jobs — and is not scheduled.)*

> **Verdict:** CONFIRMED built — K4's one door `template.unread_overrides` (found built when step 3 took it up, 2026-10-02); pinned through the road by `test_where_an_item_binds_e2e.py`. Nothing open.

**Found, and a defect rather than a decision** (a written rule is broken, so it
is fixed in step 3): an override is never narrowed to the calculation kind —
the describe door and `prep` accept any `SiestaConfig` field, so `restart`,
`relax_*` or `md_*` on a transport rung are accepted and do nothing, against
`engines/template.md` § 6.3's kind protocol. *(Found built when step 3 took it
up, 2026-10-02: K4 made it one door on 2026-09-30, `template.unread_overrides`,
asked by the description check and by `resolve`, on every kind and road —
pinned through the road by `test_where_an_item_binds_e2e.py`.)*

### 5u.3 What the documents say that the code contradicts — found 2026-09-29, each re-read when fixed

> **Verdict:** PARTLY — rewritten in the live section as the residue only. Four fixed CONFIRMED (§ 4's tip-electrode `transport.md:2823-2835`; § 3.6a's `config_for` :2420; § 3.7's LIVE rows :2455-2459; `model/overview.md` § 2.2 :148 — compose writes through `write_permutation`, `compose.py:1028`). Closed since: § 2a.12 *"asks TBtrans for all"* (`tbt_dos_*` True); § 2a.7's *"bias treatment built"* (B1); `job-contracts.md` *"the composite has no template"*; the registry reader (`web/results.md` § 0.2, `web/presenters.md`); plan § 5o.6 (§ 5o archived 2026-10-08); the stale-the-other-way list (built). Residue OPEN → § 5x B9's input: § 8 :3898 and :3901; § 6 :3043 `parse_fdf_params`; § 2a.13 :1635 + `science/overview.md:212` (the net-charge gate); § 0.5's folder (UNVERIFIED). Routed: § 2a.12's *"composed on read"* → § 5x B6 (R7); § 2a.13's stated contour → § 5x's parked P3.

`engines/transport.md`: § 8 (*"one panel per ENGINE since 2026-09-15"* — not
built); § 2a.12 (*"the transmission deck asks TBtrans for all of them"* — the
outputs default off; the report *"composed on read"* from `.gathered-from` — it
is written at `summarize` and picks attempts by file time); § 2a.13 (the stated
equilibrium contour and its gate — neither exists); § 4 (*"tip-electrode,
gate-electrode work without code changes"* — the sort accepts L, R, bridge and
buffer only); § 2a.7's correction (*"the bias treatment [is] built"*); § 6 and
§ 3.6a (`config_for` as the fill — dead); § 6, § 5, § 1 (`parse_fdf_params` in
`transport/preflight.py` — module deleted); § 3.7 (`SEALED_*` and
`dataclass_to_form_schema` *"LIVE"* in the blueprint — not imported); § 2a.13
(`_validate_transport_kind` refuses a net charge — `validation/chemistry.py`
does); § 0.5 (the user's `AuBDTAu-CT` holds a deck each and its
`.validation.txt` — it holds no template). `model/overview.md` § 2.2 (one
permutation writer for every kind — compose bypasses it). `web/results.md`
§ 0.2 and `web/presenters.md` (every presenter reads through the registry;
transport's table only). `execution/job-contracts.md` (*"the composite has no
template"* — and the same sentence in `prep.py` and `_cli.py`). Plan § 5o.6
(*"the inventory lives in `engines/transport.md` § 3.3"* — deleted 2026-09-15).
And the stale-the-other-way: E15, § 5p.3n's gather, § 2a.14's two bias homes,
§ 3.6 item 11, TR5, TR7, W10, X4 ⑥ are built. *(M5 step 3, 2026-10-02, fixed
§ 4's tip-electrode claim, § 3.6a's `config_for` and § 3.7's LIVE rows; and
`model/overview.md` § 2.2 is true again — compose writes through
`write_permutation`.)*

### 5u.4 The stale citations

> **Verdict:** PARTLY — rewritten in the live section as § 5x B9's input: ~22 `transport-design.md` citations remain (the report's list; 54 on 2026-09-29); the walkthrough cited at `transport.md:2808` does not exist; the deleted names (`route_overrides`, `_validate_transport`, `render_script`, *"the registry"*) — none left (`overview.md:338-352`; `transport/__init__.py:26-27`), that sentence dropped. The five other missing documents were not re-checked.

**54 citations of `transport-design.md` in live files** — a document that
exists only as `archive/2026-09-01-transport-design.md` — 10 in `jobset/prep.py`
alone, several in refusal messages and CLI help, and three written into every
device deck. Where each cited section lives now: § 4.1 → `engines/transport.md`
§§ 1, 3, 3.1; § 4.3 → § 2a.10, § 2a.11; § 4.1b → § 3.1; § 4.1a → § 4 and
`model/overview.md` § 2.2; § 4.2 → §§ 1, 6.1, 2a.11, 2a.15; § 3 → § 5 (I10,
I11), § 7.1; § 7 → § 8, § 2a.12. Also stale: `transport.md` section numbers
written into decks (*"4.2"*, *"3.3"* as the parameter inventory), six other
missing documents (`structure-info-plan.md`, `protocols/runtime-registry.md`,
`spectra-migration-plan.md`, `cell-plan.md`,
`execution/walkthrough-2026-09-15-junction.md` — cited by `transport.md` and
`junction-cell.md` — and the bare `staged-runs-implementation-plan.md` in
`generator.md`, linked to the archive 2026-09-29), and deleted names still spoken in
code (`route_overrides`, `_validate_transport`, `render_script`, *"the
registry"*).

### 5u.5 What this absorbs — read here, not acted on separately

> **Verdict:** PARTLY — rewritten in the live section with the open pointers only. Removed as done: W27 → steps 1–3; § 5o.5 → step 1; § 5p.4's TR1–TR5, TR7, TR8 built and TR6 → step 3; § 5p.3p's step 3 (TR6) and step 6 (→ step 3); X1 ②–⑤ → step 3; X4 ① built (`species_order` shared, catalogue :1006-1010; the invariant table re-derived, `transport.md` § 5 :2961-2976); the face gap (TD3 made the gate an error — `cell.transport_vacuum`, `validation/__init__.py:459-500`; M4 P4's d/2 warning OBSOLETE); X4 ③ → TD3 and step 2; § 11 P10 → step 7 (done); § 8 row 4 → step 3; V1.11 / A1.16 U10 → step 3; the archive line (W26, W29, W31, X2, E15 — closed 2026-09-29). (a) REFUTED as worded: there is no `calcdirs.container_or_run`; the door is `runs.folder_answer`'s `place` — corrected in the live line.

W27 → steps 1–3 · W30 → ③'s remainder is step 10, ④ is step 9, ⑤ is step 8
(①, ② built) · W25 → step 7 (its headline, the four dead scalars, done
2026-09-15) · W24 → step 8 · W10 → step 9 · W32 → step 11 · § 5c.3 → (b)–(d)
built, (e)(f) step 9, (a) superseded by `calcdirs.container_or_run` · § 5o.5 /
§ 5o.6 → steps 1 and 7 · § 5p.4 → TR1–TR5, TR7, TR8 built; TR6 step 3; TR9
step 5; TR10 step 9 · § 5p.3p → its step 3 is TR6, its step 6 is step 3, its
step 10a stays with the route-catalogue sweep (§ 0a's *Unscheduled*) · X1 ②–⑤
→ step 3 · X4 ① built (`species_order` shared, one species rule, the config
passed through — the invariant table's row naming its holder is the document
sweep's) · X4 ⑤ (`structure_hash`) → with V1.9 before M2m · § 5q.3's transport
citation viewer → § 5q.6 P4 · the face gap has two homes that are one
validator — step 2's vacuum rule and M4 P4's d/2 warning are ruled together · X4 ② (whether a lead keeps `info`, annotations and identity)
— the user's, not yet asked · X4 ③ → TD3 (a box only for an isolated axis) and
step 2 · X4 ④ (every transport deck told its labels are *not consumed*) → step
7 · § 5o.6's open rows → step 7 · A1.12 → step 7 · § 11 P10 → step 7 · § 5s.4's
unpersisted shared panel → step 8 · § 8 row 4 → step 3 · V1.11 / A1.16 U10 →
step 3 · N10's two predicates for a calculation root → M3 P2 (§ 5t.5 ②) · E13
→ step 4 (the Sol half after it) · S13 (the convergence sweep) → after step 4,
unscheduled · X3 (the Cell page's seam verdict) → unscheduled · E14 (two
uncited references) → the document sweep.
**Moved to the archive the same day, each closed by its own status and
re-read:** W26, W29, W31, X2, and E15 (the pipeline log has been opened for
transport since 2026-09-16).

---

## 5v. Documents that lag the code — the document sweeps' input *(2026-09-29)* (archived 2026-10-08)

*Entries removed from § 5v on 2026-10-08 by the validation of the same day,
which re-read each against its document and the code. Each fragment
verbatim, with its verdict. The list's own rule put them here: an entry
leaves it when its document is corrected.*

### Transport

(the pole COUNT row, an overturned premise, was struck with M5 step 2)

> **Verdict:** CONFIRMED (`transport.md:1637-1648`) — the row is gone from § 2a.13; nothing left to sweep.

§ 3.6a's `config_for` validating against
`TransportConfig` names

> **Verdict:** FIXED (`transport.md:2420,2435-2440`).

`execution/job-contracts.md`'s TranSIESTA rows
(*"TRANSPORT DOES NOT"*, a bare `write_text`, *"one writer and one reader"* of
the permutation)

> **Verdict:** PARTLY — two fixed; the *"one writer and one reader"* sentence (`job-contracts.md:2135`) is true and stands. The row is rewritten so in § 5v.

`web/form-schema.md`'s *"still calls
`dataclass_to_form_schema`"*

> **Verdict:** FIXED — `dataclass_to_form_schema` has no hit in `docs/` or the code.

`model/parse.md`'s transport record *composing
the rungs' records* (planned, written as built)

> **Verdict:** NOW TRUE (`parse.md:292-293`; `transport/record.py:219-246`).

`render_script`'s open question

> **Verdict:** FIXED (`template.md:296-304`).

*"ONE ARM
IS STILL MISSING"*

> **Verdict:** FIXED.

### Elsewhere

~~`engines/transport.md` § 3.7's *`dataclass_to_form_schema` LIVE*~~
(done 2026-10-02 with § 5u step 3: the builder deleted, the row says so).

> **Verdict:** CONFIRMED (`transport.md:2459`).

### Second open-lists outside this plan

*task #N*
references — only `spectra.md`'s #102 is left (W15's presenters pass); the rest
were rewritten to their homes on 2026-09-29 (#102, #103, #106, #107 and #105's
module half → W15; #105's feature half → § 0a's *smaller open items*; #108 →
§ 0a's *Unscheduled*; #73 and #104 done)

> **Verdict:** CONFIRMED — only `spectra.md:652`'s #102 remains; the history of the rest is removed, the open reference stays in § 5v.

archived plans named as *the plan* (`stages.md`, `siesta.md` — the
four others corrected 2026-09-29)

> **Verdict:** OPEN for the two named (`stages.md:16-17`, `siesta.md:164-165`), kept in § 5v with their lines; the "four others corrected" history is removed.

`worked-example.md`'s gaps

> **Verdict:** all CLOSED — dropped from the *owed* tables list.

---

## 5w. The M11 review — its findings, by the mechanism that produced them *(W50, 2026-09-29)* (archived 2026-10-08)

> **Verdict (2026-10-08 validation):** the header and its archive pointer confirmed (`docs/archive/2026-09-29-m11-static-review.md`); of the 22 classes, K1–K8, K10–K12, K19 and K20 are done in the code (file:line per row below); K9, K13–K15, K16 (built, unmarked), K17's rest, K18, K21, K22 and the § 5w.2 sweep stay open and are kept in the plan under § 5w; § 5w.4 = Q9 (§ 0a). Over the two validation passes on § 5w: K rows 11 confirmed, 3 partly (K5, K16, K17), 1 superseded in part (K10), 7 open; § 5w.5 bullets 8 confirmed, 4 partly (K6, K1, K3, K5), 1 rewritten (K10). What follows is the text removed from § 5w, verbatim, each block behind its verdict; the rows and bullets the plan keeps are not repeated.

### The framework paragraph's last line

> **Verdict (2026-10-08 validation):** "K6 is first" — K6 done 2026-09-29; the plan keeps the line without the clause.

as a milestone does (§ 0a). **Approved 2026-09-29** (§ 5w.3); K6 is first.

### 5w.1 The classes — the rows archived or rewritten

> **Verdict (2026-10-08 validation):** the rows below are the ones the plan now carries as one-line pointers, or rewrote (K10, K16, K17) — verbatim here. K1 CONFIRMED (`catalogue:676-698,164-165,2466-2499`; `template.py:344,1747,1865-1911`; `resolve.py:304-320`; `template.md:740-742`) · K2 CONFIRMED (`template.py:269,1974-1991`; `electronic_state.py:82-104`; `catalogue:167,432,1104`) · K3 CONFIRMED (`template.py:254,1914-1967`; `kmesh.py:34-38,95-200`; `contract.py:68`; `metadata.py:40,169`; `catalogue:380,537,847,2224,2578`; `siesta.md:392`) · K4 CONFIRMED (`template.py:1774-1853,255-260,398-407,481-491`; `pyscf/stages.py:140-149`; `prep.py:2084-2097`; `transport/stages.py:47`; `validation/stages.py:47-148`; `task-setup/viewer.js:1305-1394`; `tests/test_where_an_item_binds_e2e.py`) · K5 PARTLY (`template.py:1724-1742`; `build.py:1513-1530`; `task.py:566-581`; `resolve.py:637-666`; `viewer.js:505`; `stages.py:70-92`; `prep.py:1571-1583`; no fit panel, C1) — the GPU-type half superseded by W52 (9), `scheduler/quantities.py:288` refuses a card; carried in § 5w.5 · K6 CONFIRMED (`relax_policy.py:26-123`; `template.py:2230`) · K7 CONFIRMED (`template.py:1168-1190,2044`; `_shared.py:960-986,648`; `transport.py:713-740`; `form-schema.js:888,1073`; `migrate.py:412-420`; `form-schema.md:69-70`) · K8 CONFIRMED (`cell.py:654-694`; `vibration_deck.py:81`; `vibration_emitters.py:304`; `structure-periodicity.md:153-160,781`; `siesta.md:451`; `vibration.md:2353`) · K10 OBSOLETE in its transport half (`rung_containers` gone — § 5x B2–B4), PARTLY in its PySCF half (`per_point_rungs` `stages.py:111-118`; `pyscf/warm-files.toml:61-69`; `runwrap.py:906`; `runfiles.py:70`; `siesta/input.py:1278-1284`); the row rewritten in § 5w.5 · K11 CONFIRMED (`prep.py:1868-1886,1571-1583,1665-1686,1806-1824`; `script_emit.py:123`; `transport.md:3234`); its road test not found in `tests/`, carried in § 5w.5 · K12 CONFIRMED (`identity.py:218-242,287-294,387-424`; `siesta/input.py:559-571,989-1006`; `pyscf/input.py:280-281`; `template.py:1786,1813`; `job-system.md:1148,1185-1196`; `stages.md:582`; `tests/test_stage_names.py:6,81-82`) · K16 PARTLY — built, unmarked (`script_emit.py:796-802`; `setup.py:151-153`); the plan's row says so · K17 PARTLY — SO-C3 closed (`tuning.md:314,420`), T-F2 built with K3 (`kmesh.with_fixed` `kmesh.py:187`; `contract.py:68`), berny retired (`template.py:2230`); the rest open, kept in the plan's row without the three · K19 CONFIRMED (`electronic_state.py:88-107`; `parse/fdf.py:156-193`) · K20 CONFIRMED (`template.py:375-381`; `catalogue:967,978`; `pseudos.py:112-116`; `validation/siesta.py:196-210`; `build.py:999-1001`; `validation-findings.js:35,283`; `tests/test_build_e2e.py:942-975`).

| | what escaped the template | the declaration, and the door every reader asks | closes |
|---|---|---|---|
| **K1** | **who answers, per kind** — the markers exist and are missing | `write_forces` / `write_coor_step` fixed on for **every SIESTA kind** *(user, 2026-09-29: "siesta always write forces and always write cordinates - that's needed for later use of data")* — answered by the framework, never a control, echoed read-only: SIESTA ships both off (`LongOutput`), and what reads them later is the per-step trajectory, the relaxation record's held-atom-aware force, and the vibration finish's step 0 (§ 5.5), which fails after the whole force-constant run without them; `shared` for **every** kind on the identity items (`system_label`, `psml_lib`, `species_order`: one calculation, one name, one species table); `resolve` refuses a stage override of a `role` item as it refuses a `shared` one — one door beside `why_shared` | SS-C1, SO-C2, T-F5 |
| **K2** | **which values a kind may take** | a per-kind choice set on the item, the sibling of `recommended` (§ 6.3a), read by the form, the stage table's cells, the preflight and `prep` — so a value the kind cannot run is never offered and is refused by name if written | SS-C4 (Verlet / Nose / none on a vibration's relaxation), PO-C3 (berny), SO-C13 (transiesta on an optimization), T-F20 (spin treatments per engine and kind — `electronic_state.CAPABILITY` is the same fact in a second home), PS-C2's refusal half |
| **K3** | **hard bounds against a recommended range** — and the k-point mesh *(user, 2026-09-30)* | `range` stays advisory, one warning on every surface; a declared hard domain is refused on every surface with one message — never an error in one door and a warning in another; every rung's k-point sampling decided in one place (`kmesh.py`) | SS-C5 (`fc_displacement` > 0 — SIESTA divides by it; the `relax_steps` half moved to K4), PS-C22 (temperature and pressure > 0), T-F15 (bias points distinct, and warned outside the item's range), SO-N4 (one range, two severities), T-F2 (from K17) |
| **K4** | **where an item binds, per rung ROLE** — **done 2026-09-30** (§ 5w.5) | `stages` names a rung's role, and the kind says which role a stage plays — transport: its name; vibration: a relaxation or a force-constant rung, whatever the stage is called (`vibration_render_kind`); optimization: no roles. The stage table offers each rung only what its deck reads and echoes the rest (§ 6.6 obligation 3); a preset fills only the rungs that read it; each item's tightening direction is declared, so R3 reads it for every engine and skips a kind whose rungs are different programs; a relaxation rung relaxes — `relax_steps = 0` there relaxes nothing, while `prep bench`'s single-point pin of it is legitimate | SS-C6, PO-C14, T-F14, SS-C5's `relax_steps` half *(from K3, 2026-09-30)* |
| **K5** | **execution values with several homes** | one home per rung — the run card, `stages[i].execution` — and one resolved answer every reader asks (`resolve`'s, with its provenance): the deck, the wrapper, the scheduler's GPU ask, the bench, Task setup's cards and hints; the stage table stops offering execution items as columns; a transport rung takes its run card like any other rung | SO-C1 (a rung's `use_gpu` never reaches `--gres`), T-F3, SO-N12, the `--from` hint |
| **K6** | **the engine's own outcome, assumed instead of read** | one relaxation outcome record and one policy for both engines: the PySCF decks read geomeTRIC's convergence flag (`geometric_solver.kernel`) — `halt` raises before anything is written, `continue` re-enters from the geometry it stopped at (`pyscf.md` § 3 already says *extend this rung*), `proceed` records *not converged*; one remedy text for a non-stationary reference, read by `prep` and the finish; whether a retry resumes is the kind's warm-state declaration (`warm-files.toml`: an FC run restarts at `FC.First`), read by the wrapper; a capability the engine lacks (gpu4pyscf's `stability`) is declared and asked, never called and caught | PS-C1 = PO-C1, SS-C2, SS-C3, PO-C2, PO-C15 |
| **K7** | **the form's value model** — **done 2026-10-01** (§ 5w.5) | one field state on every surface: the template's value, the kind's default and the value's source (§ 6.6 obligation 2's four states); blank is *not chosen* for every field and never the first choice; the rung surface shows the template the rung will run; a set optional field is sent; a value that will not coerce is refused naming its field | T-F25, T-F1, T-F24, SS-C16, SO-N5, PS-C12; the K3 review's fractional k count and a triple's findings beside it |
| **K8** | **what the engine sees** — **done 2026-10-01** (§ 5w.5) | one door for the engine's frame facts: the axis kinds PySCF computes with (a cluster: isolated on all three), read by the deck, the Methods count and the R7 note alike; the box checks run for an engine that uses a cell | PS-C4, PO-C13; the K3 review's dipole advisory, keyed on the k count |
| **K10** | **where things are on disk** — **done 2026-10-01** (§ 5w.5) | one reader of a rung's attempts, a bias scan's per-point folders included; carry names from the rung's own naming door; a kind's warm set carries what its rungs read | T-F27/F13, PO-C16, SS-C14 |
| **K11** | **comparing against a stale render** — **done 2026-09-30** (§ 5w.5) | the gather check renders each upstream rung now, in memory, through the one render door, and compares that | T-F30 |
| **K12** | **stage names** — **done 2026-10-01** (§ 5w.5) | one resolver and one printer: a deck header prints the name `launch` accepts; names fold case everywhere (`stages.md`) | SS-C11, SS-C15 |
| **K16** | **the run record's parameter rows** | `declarations(engine, calculation, stage)` — the rung's own items | SS-C10 |
| **K17** | **physics each needing a build or a refusal** (not a framework gap) | PS-C3 (PCM's solvent terms on the held-atom, IR-only and Raman routes — build and measure, or refuse); PS-C2 (two spin channels in the spectrum record — build, or K2 refuses); SO-C3 (GPU-ELPA `BlockSize` — **closed**: SIESTA 5.4.2 rounds the diagonaliser's block down to a power of two itself under GPU ELPA (`diag_option.F90` `elpa_gpu_block_size`), so a set value written verbatim never moves ELPA to the CPU; `tuning.md § 2.11`); SO-C10/C11 (`ParallelOverK` — SIESTA's default unless set; ELPA forces it off); SO-C12 (pseudopotentials by exact name); PO-C4 (the geomeTRIC log — write it or drop the promise); PO-C10 (the ECP read back from the molecule); PO-C12 (a `-V` functional with D3); T-F35 (the T(E) window covers the bias window); T-F26 (only the L and R electrode labels); T-F4/F34 (a fixed ladder's controls); T-F2 *(moved to K3, done 2026-09-30)*; T-F28 (the device's E_F, iterations and poles in the record); PS-C13 (mode numbers bounded at `prep`); the K3 review's `ParallelOverK` counted before time reversal, and the vibration's level-of-theory check blind to the recorded k-mesh | as listed |
| **K19** | **a deck's spin in SIESTA's older words — warned, not read** *(found by the K2 review; user 2026-09-30: "a warning is all needed")* — **closed without code 2026-09-30**: a transport citation drops a fixed count whatever words carried it (TranSIESTA cannot hold one, `count_must_float`), so the older words change no template; the only other reader of such a deck is a foreign run opened on the Results tab, which molbuilder did not write — dropped by the user's rule | molbuilder writes only `Spin.Fix` / `Spin.Total`; a deck it did not write that uses the older `FixSpin` / `TotalSpin` (SIESTA still honours them, `read_options.F90`) is read without them, so its fixed moment would read as floating — a warning says so where the deck is read, naming the words to write | the K2 review's outside finding |
| **K20** | **the pseudopotential directory, settled before the Build tab is left** — **done 2026-09-30** (§ 5w.5) *(user, 2026-09-30: "it seems user easily misses this in the first setup and only finds out after the script is generated")* | on a SIESTA form the field is marked required in the setup card, red while empty; a live check beside it runs `prep`'s own coverage check as the folder is typed or picked (each element found, and its XC family against the functional); a suggestion, never a guess -- *use `projects/pseudopotential`, covers all N elements* -- when that folder covers the structure; and Send refuses until covered, the hand-over asking the same check of the folder it writes into (pseudopotentials already beside the calculation count, as at `prep`), the page scrolling to the field. Not PySCF's (none) nor transport's (they come with the citation) | the Build preflight's warn-only case; the hand-over checking none |

### 5w.3 What needs the user's word — **ruled 2026-09-29** *(user: "agree with your recommendation")*

> **Verdict (2026-10-08 validation):** every ruling built: K5's home `template.py:1724-1742`; K7's source key `template.py:1168-1190`, `form-schema.md:69-70`; K14's blank budget as the deck's stated counts `runwrap.py:3436-3445`; K17's offset `kmesh.py:187`, `contract.py:68`; berny retired `template.py:2230` (with K6). The plan keeps the rulings compact and cited; the berny clause and the hexagonal-cell prose (now the item's help) are archived here.


* **The approach** — the classes above, each a declaration plus one door, in
  the order of § 5w.4. **Ruled: yes.**
* **K5's home** — execution values live on the rung's run card
  (`stages[i].execution`), and the stage table stops offering them. **Ruled: yes.**
* **K7's source key** — `template.md` § 6.6's open choice 1: `source`, one of
  `cited` · `record` · `person` · `default`. **Ruled: yes.**
* **K14's blank budget** — **ruled: resolved at `prep` from what the machine
  granted, and written in the deck.**
* **K17's forks** — **ruled:** PCM on the held-atom, IR-only and Raman
  routes is refused now (K2), built and measured later; unrestricted PySCF
  vibration is refused now (K2), the two-spin record built later; `berny` is
  retired (not installed, and the deck could not pass it the rung's criteria,
  held atoms, trajectory or live log); the geomeTRIC log's promise is dropped
  (its text is in the run's log); `kgrid_displacement` on transport carries the
  cited run's offset *(built with K3, 2026-09-30)* — a `citation` item, written through the one door on every
  rung, the transport axis 0 as both engines use it, and the transmission's
  `TBT.k` in its block form when displaced (the list form carries no offset); one value on every rung is what TranSIESTA itself demands — it compares each lead's grid and offset with the device's and stops on *"found incompatible k-grids"* (`ts_electrode.F90`). The item's help gains the caveat for a hexagonal cell (Au(111)): an offset of 0.5 does not respect the six-fold symmetry, so Γ-centred is the usual choice there — SIESTA folds only k with −k, so nothing is computed wrong, the sampling is lopsided and converges more slowly.
* **K9's scope** — **ruled: the declaration first; the check against a measured
  run comes with the e2e step.**

### 5w.4 Order

> **Verdict (2026-10-08 validation):** = Q9 (plan § 0a). The order was followed: K6, K1–K3, K19, K20, K4, K5, K11, K7, K8, K10, K12 done in it (§ 5w.5 below); what remains is Q9's list.


K6 first — a relaxation that ran out of steps is recorded as converged on both
PySCF decks, which is a wrong answer given silently. Then the declarations
K1–K3; then K19 (the reader) and K20 (the pseudopotential directory), ruled
to come before K4 (user, 2026-09-30); K4, which the surfaces read; K5; K11; K7;
K8; K10; K12–K16; K18; K17; K21; K22 once designed; K9 and the text sweep last. Then W50's step 3, the road in the browser per track, with the
reviewers' probes (the archive's § E lists) among its runs.

### 5w.5 Progress

*The heading and each bullet as written. The plan keeps, under § 5w.5, only K3's parked list, K5's W52 note, K11's missing test and K10's rewrite.*

#### K6

> **Verdict (2026-10-08 validation):** CONFIRMED in substance, PARTLY in names: `relax_policy.py:26-123`; `template.py:2230` (berny, optimizer retired); `tests/test_pyscf_relaxation_outcome_e2e.py`; `vibrational_analysis.py:250-294`; `vibration.md:1249-1260`; `resumes` in `siesta/warm-files.toml:160` and `pyscf/warm-files.toml:82`, read by `warm_list(...).resumes` (`warmfiles.py:135,168-185`) — **`warmfiles.resumes_for` does not exist** (stale name in the text below); `model.py:371`; `prep.py:248-251,2084`; `runwrap.py:1597`; `running-a-job.md:465-500`; `job-contracts.md:1805`; `pyscf/input.py:1021-1079,1325-1326`; `pyscf.md:759,818-820`. R6 → K18, still open (`parse/contract.py:28-31`; `pyscf.md:135`) — the plan's K18 row carries it.

* **K6 — done 2026-09-29; R6 and V1.31 ruled the same day** (user: "agree with your recommendation on V1.31 and R6"). *Step 1 done 2026-09-29*: both PySCF decks relax
  through one spliced function, `relax_policy.relax`, which calls
  `geometric_solver.kernel` and applies the policy to geomeTRIC's own flag —
  `halt` stops before the relaxed geometry is written, `continue` re-enters
  from the geometry reached, `proceed` keeps it and says so; every step's SCF
  must converge under every policy (PS-C1 = PO-C1). The vibration result's
  `relaxation.converged` has one meaning on every route, the judged force
  against the criterion, set after the relaxation and under the person's
  statement alike. `berny` retired, and with one choice left the `optimizer`
  item retired too (`template.RETIRED_ITEMS`: a template naming it is refused
  and the line deleted — the user's `PDT/spectrum/pySCF_PDT` template carries
  one; left untouched) — K17's berny fork, done here because the door is
  geomeTRIC's. PO-C15's comment corrected (the warm start it described is not
  there; unchanged behaviour). Tests: the emitted retry loop's text tests
  retired with it (`test_one_relax_retry_loop.py`, C3, the PySCF signature
  probe, the berny cases), anchors re-pointed; one road test added
  (`test_pyscf_relaxation_outcome_e2e.py`: halt, continue, proceed on H2),
  each assertion mutation-tested red. *Measured on it*: a re-entry starts
  geomeTRIC's step history afresh, so a batch of two steps makes about one good
  step — a cost of a step or two per batch at ordinary budgets. *No test pins*
  the step-SCF guard under `proceed`: the road cannot make a step SCF fail
  inside the declared ranges; it is one literal in the one function.
  *Step 2 done 2026-09-29* (SS-C2): the remedy for a reference geometry that
  is not stationary is one text, `vibrational_analysis.nonstationary_remedy`,
  written by `prep` at a force-constant stage, by the SIESTA finish and by the
  PySCF deck's gradient check — continue the ladder's stage that relaxed it, or
  relax first when the person stated it relaxed; the `vibration` block names
  the stage its relaxation record is of (`relaxation_stage`), so the finish
  names it without the ladder's vocabulary. The measured-fixture finish test
  asserts both routes, red under the old text.
  *Step 3 done 2026-09-29* (SS-C3, and SS-C14 with it): whether a re-run of
  a kind of run resumes is that kind's warm-state fact — one section-level key,
  `resumes`, in `warm-files.toml` (`[vibration] resumes = false`: SIESTA
  restarts a force-constant run at its first step), read by
  `warmfiles.resumes_for` of the section the rung's OWN kind reads
  (`prep._rung_kind`: a vibration's `relax` rung is an optimisation — its
  `.CG` now carries — its force-constant rungs the vibration) and baked into
  the job (`Job.resumes`) and its wrapper, whose banner and retry message say
  a retry repeats the run from its first step. The budget still travels — the
  wrapper's standing rule, *say it, do not decide it*, as for a `clean` stage.
  Contract: `job-contracts.md` § 4.2a, `running-a-job.md` § 3.5,
  `vibration.md` § 5.3. Test: one road test (the ladder prepped through the
  CLI on the measured relax), red under both mutations.
  *Step 4 done 2026-09-29* (PO-C2): the open-shell stability check asks the
  mean field whether it has one before calling it — gpu4pyscf's GPU classes
  declare `stability = NotImplemented`, and calling that killed every
  open-shell GPU run before its first step — and says NOT CHECKED and why
  (`pyscf.md` § 7.3); arrays come to the host before the comparison. Test: a
  real O2 UHF run on this machine's GPU through the road, red when the deck
  calls without asking; the stability text tests re-anchored, the duplicate
  gap test retired.
  *Step 5 done 2026-09-29* (PS-C10): a PySCF run's own `_optimized.xyz`
  pair leaves the input's run records behind — `info.relaxation` and
  `info.calculation` describe the run the input came out of, and were copied
  onto every pair, so a PySCF-relaxed geometry carried a SIESTA run's
  tolerance, force and level of theory (`pyscf.md` § 2). Test: an input
  carrying a SIESTA run's records through a PySCF relaxation, red when the
  records are copied. **V1.31, closed by the one-source rule (ruled
  2026-09-29)** rather than built — the relaxation record is one reader's
  (`parse.contract.relaxation_of`, over the run's own output), which the
  Results tab's export already writes onto the pair; a deck computing its own
  copy in the pySCF env would be a second source of the same record.
  *The full review, 2026-09-29* — nineteen findings (R1–R19), each re-read in
  the code and the engine's source, and every one but R6 fixed: a halted
  relaxation stops with a `RuntimeError`, so the live log records `# error:`
  rather than a clean end (R1); what `proceed` records is stated truly — the
  judged force at the geometry reached, which can pass while geomeTRIC's other
  criteria did not (R2); every wrapper text that speaks of a retry reads one
  description (`runwrap._retry_texts`), and the force-constant retry is stated
  exactly — it re-runs from the reference step, continuing an SCF that stopped
  there and meeting the same SCF again after one that stopped at a displacement
  (R4, R5); berny's reasons without the false one (R7); the restatements the
  step missed (R8); one `prep._rung_kind`, SIESTA-scoped, for the deck and the
  warm files (R9); a vibration block's relaxation record names its stage and
  the finish refuses one that does not (R10); one engine-aware remedy, prep's
  four other wordings gone, "relax first" on SIESTA meaning the ladder's
  `relax` stage (R11); the relaxation's wording and one reading of the policy
  (`relax_policy.policy_of`, R12, R13); the status text, the per-bias-point
  wrappers and transport jobs carry `resumes` (R14); the `continue` test made
  certain by geomeTRIC's own trust radius, and a road test for a structure
  stated relaxed on PySCF (R15); the `_initial.xyz` pair stated (R16); the
  comments, and SS-C7 / SS-C8 — the FC density and restart-file claims — in
  all their homes (R17); a re-entry's effects stated (R18); the vocabulary
  sentence (R19; `stages.md` § 1.3's counts go with PO-C21d's sweep). **R3
  measured and refuted**: an H2 relaxation with `use_gpu` and `scf_soscf` on
  this machine's GPU (gpu4pyscf 1.8.1) ran under `proceed` and `continue`, the
  second-order solver converging at every geometry and on re-entry — nothing
  to refuse. Full suite green (9 488 passed).
  **R6, ruled 2026-09-29 — build the reader, K18**: checked for the ruling,
  NO structure out of a PySCF run records its level of theory — neither its
  own pair nor the Results tab's export, since `contract_of` reads SIESTA
  decks only — so a later calculation with a blank charge or spin works them
  out from the structure (a relaxed formate anion, −1, becomes a neutral
  radical in a blank-charge vibration, said in the deck and warned nowhere).
  Ruled: the PySCF half of the one reader, § 5w K18, with the gap stated in
  `pyscf.md` § 2 until it lands.

#### K1

> **Verdict (2026-10-08 validation):** CONFIRMED in substance (`catalogue:676-698,164-165,2466-2499`; `template.py:344,1747,1865-1911`; `resolve.py:304-320`; `template.md:740-742`), PARTLY in names: **`foreign_overrides` is gone** — the `foreign` parameter of `validation/metadata.py:57-74`; **`TestTheRungFixesItsOwn` is absent** (retired `cdec94da`); `test_fixed_and_shared_items_e2e.py` has 3 tests, not eleven road tests.

* **K1 — done 2026-09-29** (SS-C1, SO-C2, T-F5; mechanism ruled the same day,
  the bias as (b): *"agree with (b), go ahead with K1"*). **What the rung fixes
  reaches the deck one way**: a `role` item's answer is the catalogue's `value`
  unless the item names a rung's own (`role_values = { device = "transiesta" }`,
  the new key); `resolve` lays the rung's answers on its config last, with
  provenance `role` (`template.role_answers`), so the gate, the record and the
  deck read one value, and the section walk writes a role item like any other,
  beside a note that it is fixed — transport's hand-typed `SolutionMethod`,
  `TS.HS.Save` and the block-written `TS.Voltage` went into sections; the
  transmission writes no `SolutionMethod` and, by the audit of its source, no
  output group. **Every door refuses it**: a stage override, a pin, a sweep
  axis (`fixed_by_role`, `why_role`), and a template value unless the item's
  answer is the same on every rung and the value is it
  (`fixed_on_every_rung`) — the 33 SIESTA templates carrying
  `write_forces = true` still prep; `foreign_overrides` leaves fixed items to
  that one refusal; the settings gate refuses a render that skipped `resolve`
  with a switch off. `write_forces` / `write_coor_step` are fixed on every
  SIESTA kind; the bias's one home is the description's list — a template
  value or a stage override naming it is refused (T-F5). **The calculation's
  identity** — `system_label`, `species_order`, `psml_lib` — is `shared` on
  every SIESTA kind, refused per stage with its own reason
  (`IDENTITY_ITEMS`). Contract: `template.md` § 5, § 6.4, § 10a.2;
  `stages.md` § 1.2, § 3, § 4, § 6.2, § 6.6; `transport.md` § 2a.3, § 2a.10,
  § 2a.13, § 2a.14, § 3.6, § 3.8, § 6.1a, § 6.1b; `generator.md`; `siesta.md`;
  `vibration.md` § 5.5. Tests: eleven road tests through `jobset prep`
  (`test_fixed_and_shared_items_e2e.py`, `test_transport_prep.py`
  `TestTheRungFixesItsOwn`), one API-level gate test (the road cannot skip
  `resolve`), the writer's table round trip; each red under its mutation
  (sixteen). The template-bias tests retired with the second home.
  *The full review* — seventeen findings, each verified in the code and
  SIESTA's source, all fixed: a live-SIESTA smoke test and the identity test
  rendered the device from a bare config (now laid through `role_answers`, as
  `transport.md` § 6.1a says a `spec_for` caller must); a browser test's
  example column was `write_forces`; the refusal order on a fixed item on the
  wrong rung; a false `.ANI` sentence; a template 0 V accepted beside a scan;
  the writer's missing inline tables (`recommended` too); `role_answers` with
  no rung; the transmission's unread output group; the restatements
  (the bias as Class C in five places, the precedence order in five). The
  read-only echo of a fixed item on the form is K7's.

#### K2

> **Verdict (2026-10-08 validation):** CONFIRMED (`template.py:269,1974-1991`; `electronic_state.py:82-104`; `catalogue:167,432,1104`). The owed note at its end — `parse/fdf._read_state` and SIESTA's older `FixSpin` / `TotalSpin` — is K19's, closed without code (`electronic_state.py:88-107`; `parse/fdf.py:156-193`).

* **K2 — done 2026-09-30** (SS-C4, PO-C3 — retired with K6 — SO-C13, T-F20,
  PS-C2's refusal half; K17's PCM fork; ruled 2026-09-29: *"agree with your
  recommendation, go ahead with K2"*). **One declaration**: `offered` on the
  item, the choices a kind may take — keyed by kind, or by engine and kind
  where an item two engines share differs (`[item.spin_treatment.offered]`,
  `siesta.transport = [...]`); a misspelled kind is refused at load, and a
  kind's starting value is checked to be among its set. **One door**,
  `template.offered`, read by the Build form and the stage table's cells
  (which show a stored value outside the set as itself, marked), by the
  description's own check (gate ③, before anything reaches a machine), by
  `resolve` (naming where the value came from) and by the settings gate —
  which holds the electronic state's RESOLVED values to the sets, a detected
  radical on a PySCF vibration included, on transport on the junction
  before the state is handed to the rungs; the reasons are
  `template.why_not_offered`'s. `electronic_state.CAPABILITY`, `_CANNOT`,
  `FLOATS`, `offered()` and `cannot_run()` retired; `engines_for` stays, on
  its own table. **The sets**: a vibration's relaxation CG / Broyden / FIRE;
  an optimization's and a vibration's solver diagon / OMM; SIESTA
  transport's spin restricted / unrestricted; a PySCF vibration's restricted
  alone and its count 0; PySCF's counts without `free`. **Two rules of two
  items**, in the gate: PCM on a PySCF vibration only on the frequencies-only
  route (no held atoms, IR or Raman — `vibration.md` § 4.6), refused
  otherwise; and ES6's float-only rule gains transport — one function,
  `electronic_state.count_must_float`, under which a count nobody stated
  floats and a stated one is refused, and the citation defaults neither a
  fixed count nor a treatment TranSIESTA cannot run. Contract: `template.md`
  § 5, § 6.3a, § 10a.2; `chemistry-correctness.md` § 2a.1a, § 2a.2, § 2a.3;
  `vibration.md` § 3.1, § 4.6; `transport.md` § 2a.13, § 3.1; `pyscf.md`,
  `siesta.md`, `overview.md`, `stages.md` § 6.6, `form-schema.md`,
  `task-setup.md` § 5.2, `validation.md` § 6, `web-api.md`. Tests: fifteen
  new, through `jobset prep`, the describe door and the surfaces' own routes
  (the browser for the two menus), plus the catalogue's misspelled keys at
  load; each red under its mutation (nineteen). Re-anchored: the charge
  step, two render-gate parity tests (a closed shell on methyl), the
  solvated end-to-end run (the frequencies-only route), the citation's
  spin, the form-honesty companions, the Methods Hessian module.
  *The full review* — fourteen findings, all verified and fixed: the gate
  reads the kind a deck RENDERS as, so a SIESTA vibration's `relax` rung
  (an optimization's render) is `resolve`'s alone to hold to the relaxers —
  stated in § 6.3a, the rung's own role K4's; gate ③ checked full
  `choices`; ES5's order and its way out on transport; the misspelled key;
  the menus' blank over a stored value; the cited treatment; the PCM rule on
  an unknown solvent; the junction's source words; the load-time guard's
  home; the grammar; a honesty probe that measured nothing; the
  restatements. *Found beside it, owed*: `parse/fdf._read_state` does not
  read SIESTA's older `FixSpin` / `TotalSpin` spellings, which SIESTA still
  honours (`read_options.F90`), so such a deck's fixed count reads as free —
  the fdf log's check (K9) is its natural catch.

#### K3

> **Verdict (2026-10-08 validation):** CONFIRMED in substance (`template.py:254,1914-1967`; `kmesh.py:34-38,95-200`; `contract.py:68`; `metadata.py:40,169`; `catalogue:380,537,847,2224,2578`; `siesta.md:392`; the one renderer `task-handover.js:27-33,60,228`, `validation-findings.js:1-30`, `handover-procedure.md:99`; `tests/test_k_point_mesh_e2e.py` and `test_hard_limits_e2e.py` exist), PARTLY in names: **`tests/test_transport_tab_e2e.py` retired `cdec94da`** (the "two browser tests on the Transport tab" below are gone). Of its parked list, `config_for` / `dataclass_to_form_schema` are gone, the dipole advisory is K8's and the fractional count K7's (both done); `transport/sort.py:74`'s own `TRANSPORT_AXIS` stays owed (`kmesh.py:30-33` says so) — the reduced list is in the plan's § 5w.5. The "owed, beside it" items (a transport finding in card 5's panel; the junction picker before the projects root resolves) were not re-checked by the validation.

* **K3 — done 2026-09-30** (SS-C5, PS-C22, T-F15, SO-N4; K17's offset fork,
  T-F2; ruled: *"go ahead with K3 after K2 is committed"*). **The k-point
  mesh joined it the same day** (user: *"check kgrid in different setting
  such that you have a unified system/logic. if we need a layer/api to
  handle this consistently and systematiclly, we should consider it"*;
  *"i want a holistic system solution, not a patching here and there. api
  level and data structure level unification is essential"*; ruled *"agree
  with your recommendations, go ahead with the k layer in K3"*). **One
  severity per bound**: a `range` is warned on every surface — the
  description's own check refused it until now (SO-N4); a hard limit is
  declared on the item, `above = { value, why }` — `fc_displacement`,
  `temperature_K`, `pressure_atm`, `block_size`, `kgrid`, `tbt_k_grid`
  above 0, `electrode_kz` above 1 (its range a recommendation from 20) —
  and refused on every door with one message. **One per-value door**,
  `template.why_not`: a component the kind fixes, a choice the kind does not
  offer, a value at or below its limit — the first that holds, so one value
  draws one refusal; `resolve`, the description's check and the settings
  gate each ask it once (five passes became three calls of one door), and a
  refused value's range warning stands aside, as does the range of an item
  the kind does not carry. **The k-point mesh** (`kmesh.py`, `siesta.md`
  § 6.1): one `KMesh` per deck — each axis's kind, its role (sampled, Γ,
  open, a lead's), the count, the offset and the item that answered —
  derived once (`mesh_for`), written by one writer (SIESTA's block and
  `tbtrans`'s `TBT.k`, always the block, which carries the offset), checked
  once (`check`: an isolated axis sampled more than once is warned — on the
  transmission's grid too — a far-apart periodic axis hinted, an offset on a
  single point warned), its fixed components answered without a structure
  (`fixed`: a transport calculation's third component of `kgrid`,
  `tbt_k_grid`, `kgrid_displacement`), and read by `Diag.ParallelOverK`,
  counted on the mesh written — a lead's forty points — and the dipole
  advisory. Retired: four hand-built block writers, the transport kind
  validator's three k blocks, the SIESTA validator's k block, the
  displacement callable's single-point rule — "the transport axis is 1" in
  six places. The cited offset is carried (K17's ruling): the parser reads
  the offset column, the recorded contract gains `kgrid_displacement`
  (`K_MESH_RECORD_KEYS`, beside the counts' older name), and the citation
  lays the fixed axis on through `kmesh.with_fixed`; the form draws a fixed
  component locked with its reason; the composition reads
  `kmesh.TRANSPORT_AXIS`. **Rulings along the way** *(user, 2026-09-30)*:
  `k > 1` on an isolated axis stays a warning (the periodicity doc's
  "clamp" was never built, and is corrected); a `transport` axis outside a
  transport calculation is sampled like a periodic one; `relax_steps = 0` is
  not a limit — `prep bench` pins exactly that single point — so SS-C5's
  `relax_steps` half, a vibration's relax rung that relaxes nothing, is
  K4's, where the rung's role is. **Also**: the Transport tab's Send and
  `jobset init`'s transport arm run the description's own check (they ran
  only the codec; its warnings ride the Send's notices, `workflow.md` § 9);
  a repeated bias point is refused in the codec, one outside the item's
  range warned (T-F15); the template reader holds only an item's default to
  its limit, a person's value being the door's. Contract: `siesta.md` § 6.1
  (owner); `template.md` § 5, § 5.3; `stages.md` § 6.6; `form-schema.md`;
  `transport.md` § 0.2–0.4, § 2a.13, § 3.2, § 3.6, § 3.8, § 5, § 7;
  `structure-periodicity.md` § 2; `science/overview.md` § 4;
  `validation.md` § 4.1; `workflow.md` § 9; `job-contracts.md`;
  `spectra.md`; `molview.md`; `vibration.md`. Tests: thirteen new, each
  driving the road — `jobset init`, `jobset prep`, the Task-setup save, the
  Transport tab's Send — and each red under the mutations its docstring
  names: five k-mesh tests (`test_k_point_mesh_e2e.py`), five limit, range
  and bias tests (`test_hard_limits_e2e.py`), two browser tests on the
  Transport tab itself (`test_transport_tab_e2e.py`: the locked component
  and its reason; a refused and a warned Send, their findings in the
  panel), and the catalogue's starting values. Two are API-level, each
  saying why: the settings gate's one-refusal rule, which `resolve`
  refusing first keeps the road from reaching, and the catalogue, authored
  rather than described. The record path's offset rides its citation test.
  Retired with their rules' move: the transport kind's three k test classes
  (fourteen cases), a duplicate of the prep door's refusal, the description
  check's API-level one-refusal test (the save's road test says it); two
  range tests and two save tests each merged into one.
  *The full review* — twenty-two findings, each verified in the code, all
  fixed: `kgrid`'s recommended range was warned nowhere (the metadata pass
  stood aside for any field with a `validate` callable, and `kgrid`'s now
  held only its shape) — every triple is range-checked per component
  through one helper, `outside_range`, which the description's check asks
  too (it checked scalars alone); a wrongly typed value drew the type's
  refusal and the limit's — the door leaves it to the type check
  (`_TYPE_CHECKS`) and asks about the value as `resolve` will see it, one
  lossless canonical form (`template.as_declared`: a whole float is the
  count it names — the preflight accepted `8.0` for a count and the
  settings gate refused it); a refused value's range warning and its
  mesh's findings stand aside at the settings gate too; the description's
  check judged items the kind does not carry (`not_carried_by`); execution
  values and bench points were first refused at prep; a repeated bias point
  was compared by value while its folder is `bias_token`'s — the one
  spelling moved to the codec, and two points in one folder are refused;
  `jobset init`'s transport arm dropped its warnings; the Send lost its
  next-step line when warned; the shared offset was warned once per mesh; a
  kind's recommended value was not held to the limit; the triple's labels
  had a second home (`kmesh.AXES` now); the stale text (the old key's name,
  deleted writers and holders, a false history, two table placements, the
  offset's help and the hexagonal caveat the ruling asked for, a module's
  stated dependencies, the codec's check list, the bias rows of the
  check table, a door's docstring). **The findings reach the page through
  the one renderer** *(user, 2026-09-30: "we have facility/framework designed
  for that. don't handcraft another set")*: the hand-over
  (`lib/task-handover.js`) listed every notice as a bullet in its status
  line, and K3 had routed the Transport describe's warnings into it — a
  second renderer, which `science/validation.md` § 4.1 R2 forbids and R2a
  names. It hands them to the tab's findings panel now (`showFindings`),
  through `lib/validation-findings.js`: the Build and Spectrum tabs' own
  panels, the Transport tab's new one under Send; and Task setup's two
  renderers — the prep answer's finding lines and the save's notes — draw
  through it too. The Transport tab's browser tests drive it. *Owed,
  beside it*: an override of an item the kind does not carry is inert and
  said nowhere — which rung and kind an item binds to is K4's; a transport
  finding lands in card 5's panel, not beside its rung's field, since the
  renderer does not yet route by rung; and the junction picker opens
  nothing, and says nothing, when pressed before the sidebar has resolved
  the projects root (`tree-picker.js` answers `null`) — found writing the
  tab's browser test.
  *The post-commit review* *(user, 2026-09-30: "use agent to do full code and
  document review … focus on the k-grid scope, from end to end", tests
  included)* — two reviewers, the shared layer with the optimization,
  vibration and bench roads, and the transport road; each finding re-read in
  the code and the engine's source, then triaged by the user's rule: *does it
  break molbuilder's own workflow?* **Fixed:** the Build and Spectrum tabs'
  Send replaced the live check's findings with its own notices — a Send now
  adds beside them (`standing`, `handover-procedure.md` § 2.1); the transport
  mesh blocks ignored `verbose_comments`; the single-point offset warning
  fired for 1.0, which both engines read as Γ (modulo 1); `triple_labels`,
  the triple's labels' second home, deleted; the K3 text that was wrong — a
  raised `kgrid` said to reach the transmission's `TBT.k` (§ 7, the deck's
  advice), "both engines force the transport axis" (only TranSIESTA's NEGF
  step and `tbtrans` do), "the same grammar" (`tbtrans` reads a block row
  only with its offset column), the offset's help, `TBT.k` as a list in
  `transport.md`, two rules `template.md` § 5.3 did not state, stale
  comments; the tests — the one-refusal test off the library onto the Build
  tab's live check and the Transport tab's Send, the offset's fixed
  component refused at prep, a relaxation's mesh by axis kind through
  `prep`, two duplicates retired. **Refused, for now** *(user: "refuse the
  stripe case for now. this is a special case that will take a lot of
  effort to get right")*: a junction periodic along one transverse axis and
  isolated along the other, sampled more than once across its vacuum —
  TranSIESTA stops the device on it (`ts_electrode.F90`, *"found
  incompatible k-grids"*) after the seed and both leads have run; refused at
  the seed's `prep` (`siesta.md` § 6.1). **Dropped** — inputs molbuilder
  does not produce: a hand-edited description's mistyped value (refused,
  with a range warning beside it), a wrong name in an `execution` block
  (refused twice), a cited deck's other k-block spellings, a bias of −0.
  **Parked, one line each:** the dipole advisory keyed on the k count, not
  the axis kinds (K8); a fractional k count rounded by the form, and a
  triple's findings on its card rather than beside it (K7); `ParallelOverK`
  counted before time reversal folds k with −k (K17, beside SO-C10/C11); the
  vibration's level-of-theory check ignoring the recorded k-mesh (K17); the
  transport axis stated again in `transport/sort.py` and indexed as `c` by
  the transport writers; `stages.config_for` and `dataclass_to_form_schema`
  without a production caller; the Build and Spectrum Send writing a
  template the live check refuses (prep refuses it); the non-transport
  `jobset init` discarding gate ③'s warnings (latent: its ladder presets
  hold none); duplicate k tests in `validation/test_geometry.py`; stale
  names from before K3 (§ 5w.2's sweep).

#### K19

> **Verdict (2026-10-08 validation):** CONFIRMED (`electronic_state.py:88-107`; `parse/fdf.py:156-193`).

* **K19 — closed without code 2026-09-30** (the row says why: a transport
  citation drops a fixed count whatever words carried it).

#### K20

> **Verdict (2026-10-08 validation):** CONFIRMED (`template.py:375-381`; `catalogue:967,978`; `pseudos.py:112-116`; `validation/siesta.py:196-210`; `build.py:999-1001`; `validation-findings.js:35,283`; `tests/test_build_e2e.py:942-975`).

* **K20 — done 2026-09-30** (user: *"it seems user easily misses this in the
  first setup and only finds out after the script is generated"*). The
  pseudopotential folder is settled before a SIESTA calculation is written:
  the item declares the kinds whose forms draw it **required** (`required`,
  a new item key — `template.md` § 5; `psml_lib` on optimization and
  vibration, not transport, whose files come with the citation), and the
  form marks it and draws it red while empty; the live check's unset
  warning names the tree's `pseudopotential` folder when it covers every
  element the calculation lacks (`pseudos.CONVENTIONAL_LIBRARY`, a
  suggestion, never a fill); the **Send asks the same check of the folder
  it writes into** (`validation.siesta.pseudopotential_findings`; the
  sender passes `dest`, and files already beside the calculation count, as
  at `prep`) and refuses until every element is covered, its findings
  beside the field and the first brought into view (the renderer's
  `reveal`). Contract: `science/pseudopotentials.md` § 1 (owner);
  `job-contracts.md` § 2.5a; `handover-procedure.md` § 2.2;
  `template.md` § 5; `form-schema.md` § 1.1. Tests: one browser test on the
  Build tab (the mark, the suggestion, the refusal in view with nothing
  written, then the Send writing) — red under each of its four mutations;
  the Send tests' calculations carry their pseudopotentials, as a person's
  do.

#### K4

> **Verdict (2026-10-08 validation):** CONFIRMED (`template.py:1774-1853,255-260,398-407,481-491`; `pyscf/stages.py:140-149`; `prep.py:2084-2097`; `transport/stages.py:47`; `validation/stages.py:47-148`; `task-setup/viewer.js:1305-1394`; `tests/test_where_an_item_binds_e2e.py`); one sub-claim unread — the vibration finish's criterion (before `prep.py:335`).

* **K4 — done 2026-09-30** (SS-C6, PO-C14, T-F14, SS-C5's `relax_steps` half,
  and K3's owed override of an item the kind does not carry; ruled: *"yes,
  catalogue as the one source, go ahead"*, after the user asked *"where is the
  information stored that tells what parameters will be used by what engine
  in what kind of setup?"* and *"we want logical and clean design that serves
  unified purpose"*). **The catalogue's `stages` names rung ROLES** — no new
  key: transport's are its five rungs, by name; a vibration's `relaxation`
  and `force_constants`; an optimization has none, so every rung reads every
  item (`template.KIND_ROLES`). **One rule** maps a stage to its role,
  `template.stage_role_rule` / `stage_role` — on SIESTA the stage named
  `relax` relaxes and every other measures force constants; a PySCF
  vibration relaxes inside its one deck — and `vibration_render_kind` and
  prep's `_rung_kind` are read off it (they were two encodings of it).
  **One door** for *does this rung read it*, `template.reads` /
  `unread_overrides` / `why_unread`, asked by the description's own check (at
  every describe and at `prep`) and by `resolve`: an override on a rung that
  does not read the item, or of an item the kind does not carry, is refused
  by name. Transport's own copy (`transport/stages.foreign_overrides`, asked
  by its describe route and its prep step) is deleted; `TRANSPORT_STAGES`
  reads the role vocabulary. Declared: the SIESTA relaxation settings
  `stages = ["relaxation"]`, `fc_displacement` and PySCF's
  `displacement_amplitude_ang` `stages = ["force_constants"]` (PySCF's
  `geom_*` stay every rung's — its vibration relaxes in-process); the catalogue
  refuses at load a `stages` name no carrying kind has, and a declared kind
  with roles it names none of. **The ladder check** (R3) reads each item's
  new `tightens` (`"up"`/`"down"`, set where `tuning.md` § 2 has a tier
  table) on every engine — the SIESTA-only table in `validation/stages.py`
  is deleted — and compares only rungs of one role, so transport's per-rung
  tolerances never read as a loosening. **The stage table** maps each row to
  its role with the rule handed beside its columns (`roles`), draws a cell
  its rung does not read disabled, naming the readers, and fills a preset
  only into the rungs that read each value. **The vibration finish** judges
  the reference geometry by the relaxation rung's own `relax_force_tol` (the
  template's when stated relaxed), not the force-constant stage's copy.
  **A rung that relaxes takes a step**: `relax_steps = 0` beside a moving
  `relax_type` is refused by the description's own check, on the rungs that
  read it (`validation.stages.check_a_relaxation_takes_a_step`) — not as a
  hard limit, since the settings gate cannot tell `prep bench`'s own
  measurement pin of 0 from a stated value, and the description check never
  sees the pin; its range warning stands aside. Contract: `template.md`
  § 6.4 (owner), § 5, § 5.3; `stages.md` § 4 R3; `vibration.md` § 5.2a,
  § 5.5; `task-setup.md` § 5, § 9. Tests: five on the road
  (`test_where_an_item_binds_e2e.py`: the save's refusals by role and kind,
  the ladder by role on PySCF and transport, the stepless relaxation, the
  finish's criterion through `prep`, the stage table in the browser) — red
  under each of their twelve mutations; one transport test's wording
  updated.

#### K5

> **Verdict (2026-10-08 validation):** PARTLY: the home, the column and bench-row refusals, the one door confirmed (`template.py:1724-1742`; `build.py:1513-1530`; `task.py:566-581`; `resolve.py:637-666`; `viewer.js:505`; `stages.py:70-92`; `prep.py:1571-1583`; C1 closed, no fit panel). **The GPU-type half is superseded by W52 (9)** — no GPU card anywhere; `parse_gres_flag` (`scheduler/quantities.py:288`) refuses a card. The note is in the plan's § 5w.5.

* **K5 — done 2026-09-30** (SO-C1, T-F3, SO-N12, the `--from` hint; the
  home ruled 2026-09-29, and the bench row's half by the user on
  2026-09-30: *"trials only"*). **A run setting has one home per rung, its
  run card** — the catalogue's `execution` items (`template.run_settings`),
  the machine's answers and a person's alike, stated in `execution`, the
  calculation's with the rung's over it (`stages.md` § 6.8d's "and nowhere
  else"). **Not a column**: the stage table offers none (`_column_items`),
  and the description's own check and `resolve` refuse one in `varies` or
  an override by name (`why_run_setting`, the machine-fact story folded in).
  **Not a bench row**: a one-point non-machine row pins the bench's trials
  alone — `prep_run_inputs` takes the run's pins from the card only, which
  also makes the bench card's *"the two are independent"* true (SO-N12).
  **One answer, one door**: `Task.run_condition` (moved from
  `prep_inputs.run_condition`) is what every reader takes — the deck's pins
  and the launch shape at `prep`, `run_uses_device`, the sequence checks
  (`resolved_ladder` lays each rung's card, so § 6.6a's *starts clean*
  reads the card's `restart`), the Task setup tab (`runConditionOf`: the
  `--from` hint reads what the rung's card STATES — it read a stage column;
  a rung that states nothing is the stage hand-over's, W37/M2h).
  **Transport takes its
  run card** (T-F3): `_prep_transport` refused every pin, so a run card's
  `use_gpu` on a transport rung stopped the prep; its rung now resolves
  with the card's pins. **The GPU ask, asked while checking the user's
  *"make sure bench and run are correctly honoring the options"***: a run
  that uses a device gets its `--gres` decided at prep for every engine —
  count from `gpu_count`, else one (G5), type by the bench's own producers
  (stated or probed, else the queue menu's inventory) — and the header and
  the placement ask the one door for any engine; they asked a SIESTA deck
  alone, so a PySCF run whose card said `use_gpu` reached the queue with no
  device. SLURM's untyped `gpu:N` read as a type (`--gres=gpu:gpu:N`) is
  fixed in `_parse_gres_flag`. None of the 51 descriptions under
  `projects/` used a run setting as a column or a one-point bench row.
  **Reviewed 2026-09-30** by an agent reading `af91d223` (code only), every
  finding re-read against the code: **A1** the hint fell back to the
  catalogue's default `continue`, so every transport rung after the first
  and a vibration's `freq` were taught a `--from` their prep refuses and
  their Prep buttons only previewed — it reads the card alone now, and a
  browser case pins it; **A2** the transport tab's rung vocabulary
  (`resolvable_override_names`) subtracted only the machine's answers, so it
  offered the solver, `block_size` and `parallel_over_k`, each refused at the
  save — it subtracts `run_settings`; **B1** a template `use_gpu` with an
  empty card returned before the device ask; **B2** the card type's queue
  menu was read from the folder, not the `--target` record in hand; **B3**
  the Measure card called a one-point row "chosen" — "every trial"; **B4** a
  `gpu_count` without `use_gpu` was dropped unsaid — `prep` says so; **B5**
  placement ignored a stated `gres`; **C2** the run card offered, and the
  description's check accepted, a run setting the kind does not carry
  (`restart` on a vibration) — narrowed by kind on both; **B6** nine stale
  sentences. **T2**: a transport rung's card reaching its deck, through
  `jobset prep run`. Not so: *"reverting each GPU fix turns them red"* —
  the header's gate is unreachable on the road (prep now always states the
  `gres`), and is kept because its old comment was false. **C1** — the run
  card's fit panel reads the run through the bench's enumerator — closed
  2026-10-05: the panel retired with the preview (unit 10e), and a run's fit
  is prep's checkpoint 4.

#### K11

> **Verdict (2026-10-08 validation):** CONFIRMED (`prep.py:1868-1886,1571-1583,1665-1686,1806-1824`; `script_emit.py:123`; `transport.md:3234`). **Its road test — `mesh_cutoff` changed, the device's prep refused — is not found in `tests/`**; the plan's § 5w.5 carries it, to be judged under § 5x's gate (B2–B4).

* **K11 — done 2026-09-30** (T-F30; user: *"mismatch is a mistake. period."*).
  The gather compares an upstream rung's concluded attempt with the deck that
  rung renders NOW — `_rung_deck_now`, through the rung's one door
  (`_transport_rung`: its structure, the template ⊕ its overrides ⊕ its run
  card resolved, the junction's electronic state; `_transport_spec` per bias
  point), the same pair `_prep_transport` writes the rung's decks with — and
  refuses a mismatch by name. It compared with the stage folder's LAST
  render, which a template value changed since, or a re-pointed junction,
  leaves as it was, so a stale lead's `.TSHS` was carried into the device and
  `.gathered-from` called it consistent. A transport deck records no machine
  sizing (no BENCH-MARKS block), so the render needs no allocation, and the
  comparison is `same_calculation`'s as it stood. `engines/transport.md` § 6's
  gate row says so. Test, through `jobset prep run`: the leads and the seed
  concluded, `mesh_cutoff` changed in the template, the device's prep refused
  — red against the stage-folder comparison.

#### K7

> **Verdict (2026-10-08 validation):** CONFIRMED (`template.py:1168-1190,2044`; `_shared.py:960-986,648`; `transport.py:713-740`; `form-schema.js:888,1073`; `migrate.py:412-420`; `form-schema.md:69-70`).

* **K7 — done 2026-10-01** (T-F25, T-F1, T-F24, SS-C16, SO-N5, PS-C12; the K3
  review's fractional k count and a triple's findings; K1's read-only echo;
  the source key ruled 2026-09-29). **Every template value records whose it
  is** — `source`: `cited` · `record` · `person` · `default` (`template.md`
  § 6.6 obligation 2), written by the one writer from what its caller knows
  (`jobset init` the name and pseudopotential folder given, the form doors
  what the form sent, the transport door the citation's answers with the
  person's over them — a value the panel holds as the citation answered it
  stays the citation's — `migrate` what the file said); a file written before
  reads *not recorded*. **One field state on every surface** (`form-schema.md`
  § 1.1): the schema carries the template's `value` and `source` beside the
  kind's `default`, and the source words once (`template.SOURCE_WORDS`); a
  field holds a value only where its surface edits the template (transport's
  shared panel), a rung's tab holds the rung's own and shows the template's
  as what a blank field runs (T-F1), a new calculation's form holds what the
  person gives; a caption under every field says whose its value is and
  follows each edit. **Blank is not chosen, for every field** — never a
  list's first choice, a triple's zeros or an unticked box (an unanswered
  checkbox is indeterminate, T-F25); `collectForm` reads a blank as `null`,
  and the one server door (`config_from_params`) reads it as not chosen, with
  the kind's recommendation now under every form's values — it raised
  `float(None)` naming nothing (SS-C16 / SO-N5), and PySCF's sentinel list
  went. **A value that will not read is refused naming its field**, beside it
  in the browser and at the door: a fractional count is never rounded
  (`int(4.5)` and `parseInt` gave 4), a triple holding only some components
  is refused, and a triple's id is on its wrapper, so its findings land beside
  it. **Transport's two surfaces are drawn from the template its describe
  writes** (`_panel_template`, one door for both): a rung's tab, fetched with
  the junction and what the panel holds, shows what the rung runs, the
  transmission grid following a mesh set on the panel (the one k-mesh rule
  applied to the person's mesh, T-F1); a rung's bag is what its tab holds, so
  an explicit 1 1 1 and `scf_must_converge` are sent (T-F24); a field emptied
  on the panel is not chosen. **An item the rung fixes is echoed read-only**
  at the rung's answer, with why (`locked`; `template.PER_POINT` names the
  bias, which shows no number) — never a control. **The Task setup hover**
  gives this kind's recommendation (the columns and run-card routes,
  PS-C12) and the template value's source. The Structure-optimization
  restore writes through `collectForm` / `setValues` (an unanswered box stays
  unanswered) while the form-dirty gate holds `_ignoreFormChanges` — the
  e2e caught a load stopped at *discard unsaved changes?* over values nobody
  had touched; the Recommended panel's reset blanks a field; a rung's tab
  saves on every edit, so a redraw cannot lose a keystroke. Contract:
  `form-schema.md` § 1.1, § 2, § 3, § 3.0a, § 3.1; `template.md` § 5,
  § 6.3a, § 6.4, § 6.6; `transport.md` § 3.8.2, § 3.8.3, § 3.8.5, § 3.8.7,
  § 3.8.9; `task-setup.md` § 5.1. Tests through the road: the hand-over's
  sources over the kind's recommendation, the door's blank and refusal, the
  transport tab's two surfaces and the written template's sources, the
  pickers' kind default; in the browser the form's states and refusals, the
  mesh's finding beside the mesh, a reload, and a rung's tab end to end — 22
  mutations, each red. `test_form_state_persistence_js.py` retired with the
  by-id reader it pinned, and the panel-ordering source pin with its premise.
  *The review* (an independent read of the commit, each claim verified in the
  code): `migrate` called every value it carries nobody's — a file written
  before sources records none, so what it carries is *not recorded* (the
  writer takes a source of `None`); a rung draw superseded while its answers
  were in flight pointed the Send at tabs already gone from the page (a
  sequence now); the rung tabs' saves from before K7, every field pre-filled,
  would restore as the person's overrides (a new key); a shared value that
  will not read redrew the rungs from the citation alone (they wait for it to
  read); one half-typed field wiped a form's save or stopped its later ones
  (`formSchema.heldValues` saves field by field). Pinned besides: a nobody's
  value held on the shared panel, the sources `init` and `migrate` write, the
  Recommended panel's own Reset. The transport door lays the kind's
  recommendation under the citation's answers, as § 6.3a says every door
  does; template.md § 5 has its `source` row, transport.md's role rows and
  form-schema.md's blank sentence say what the code does. Seven mutations,
  each red.

#### K8

> **Verdict (2026-10-08 validation):** CONFIRMED (`cell.py:654-694`; `vibration_deck.py:81`; `vibration_emitters.py:304`; `structure-periodicity.md:153-160,781`; `siesta.md:451`; `vibration.md:2353`).

* **K8 — done 2026-10-01** (PS-C4, PO-C13; the K3 review's dipole advisory).
  **What the engine computes with has one door** — `cell.engine_axis_kinds(
  engine, struct)` (`structure-periodicity.md` § 2.1): the structure's axis
  kinds on an engine that computes in a cell, a cluster's — isolated on all
  three — on one that builds the atoms as a molecule in free space
  (`cell.MOLECULAR`, moved from `electronic_state`, where it decided *finite*
  alone). Every question about the calculation asks it: the electronic
  state's *finite*; a vibration's surviving motions — on PySCF the deck's view
  carries them (`axis_kind`), the deck writes them as `AXIS_KIND`, and the
  Methods count and the settings check's R7 note read the same, where they
  counted the structure's (water in a periodic box, its oxygen held, was told
  six modes and none removed while the deck removed three, PS-C4); SIESTA's
  dipole advisory, asked of the axes, never the k count — a crystal or a
  junction sampled at Γ alone was told it sat in a 3-D vacuum cell. **The
  box's advice is an in-cell engine's** (`cell.computes_in_cell`): a PySCF
  deck no longer hears vacuum, image or face advice about a box it never uses
  (PO-C13). **Its refusals stay every engine's**: an impossible box is a broken
  structure, and the browser's request seam already refused it whatever the
  engine, so the settings gate says the same and the two roads cannot disagree
  — PO-C13's error half closes as that, not as a PySCF exemption. Contract:
  `structure-periodicity.md` § 2.1, § 6.1a; `chemistry-correctness.md` § 2a;
  `vibration.md` § 3, § 4.1; `validation.md` § 6. Tests through the road: the
  periodic water's vibration through `init` / `prep` (the deck's `AXIS_KIND`,
  its Methods count, the note), the box's advice on both engines through the
  live check, the hydrogen chain on PySCF's detection table (a molecule's
  doublet — the M6 fix had no test), the dipole advisory on a molecule at
  4×4×4 and a crystal at Γ — 8 mutations, each red. *The review* (an
  independent read of the commit, each claim verified in the code): the
  forecast of the hand-off's own refusal — an atom a stated origin leaves
  outside — was dropped with the advice, so PySCF's live check went silent on
  what its prep still refuses; it is every engine's now
  (`cell.box_findings_for`, the finding named once, `cell.ATOMS_OUTSIDE`).
  `normal-modes.md` § 3 still said the axes are *not an engine property*, and
  `overview.md` and `siesta.md` the dipole's k-count rule; `KMesh.single_point`
  lost its last reader (deleted); the vibration view answered `engine_name`
  with its class name (it states `ENGINE = "pyscf"`); `cell.py`'s header
  gained the engine block; § 2.1 claims what the code does; the dipole test
  moved onto the prep road. Two more mutations, each red.

#### K10

> **Verdict (2026-10-08 validation):** OBSOLETE in its transport half — `rung_containers` is gone, a bias sweep is one run with its points inside (§ 5x B2–B4); PARTLY in its PySCF half, which stands (`per_point_rungs` `stages.py:111-118`; `pyscf/warm-files.toml:61-69`; `runwrap.py:906`; `runfiles.py:70`; SS-C14's deck half `siesta/input.py:1278-1284`). Parked item 2 closed (`job-contracts.md:2136`; `results.md:45`); parked item 1 (the flat shape's relaxation re-run from the FC `.XV`) unmeasured. Rewritten in the plan's § 5w.5.

* **K10 — done 2026-10-01** (T-F27, T-F13, PO-C16, SS-C14). **Where a rung's
  attempts are has one door** — `transport.stages.rung_containers(base, task,
  stage)` (`transport.md` § 2a.11): one folder per bias point for a rung a scan
  runs at each point, the stage folder otherwise. Which rungs those are is the
  bias item's own `stages` (`per_point_rungs`, through `template.PER_POINT`),
  and `rung_container` refuses by name a point the scan does not hold. Every
  reader asks it: prep's decks, wrappers and attempts, the gather, the chain
  launch, prep's *already under way*, Task setup's count, the record, and
  `jobset status` with the Results tab's ladder. The record looked in the
  point folders for the transmission alone, so a finished device scan read
  *not run* (T-F27); Task setup's count and prep's question looked in the
  stage folder alone, so a re-prep re-rendered over a launched point without
  asking (T-F13). **What PySCF carries is what its rungs read back** (PO-C16):
  geomeTRIC's trajectory and scratch left `pyscf/warm-files.toml` — geomeTRIC
  1.1.1 rewrites the trajectory from the first step of every run, and on
  PySCF never reads the scratch back (no `read_result`, no Hessian asked) —
  so a rung launched again over what its finished run left no longer
  announces a warm restart from its own trajectory, and the banner names only
  what the engine loads. The wrapper's probe is `[ -s ]` alone, since no warm
  file is a folder; `runfiles._CARRIED_ROLES` is `_ENGINE_ROLES`, without
  `_geom_optim.tmp`, which nothing writes. **SS-C14's deck half**: the
  start-state comment said *one field in the description* decides, and a
  vibration's relaxation has no such field; it names the field, `restart`,
  as `run-identity.md` § 4 does. Contract: `transport.md` § 2a.11;
  `stages.md` § 1.1a, consequence 4; `job-contracts.md` § 4.2;
  `vibration.md` § 5.3. Tests through the road: a finished device scan
  through `summarize run`, a launched point seen by prep's question and Task
  setup's count, the wrapper's probe over a finished rung's leftovers — six
  mutations, each red. A products test faked a device run in a scan's stage
  folder, which the road never makes; it describes one point now. *The
  review* (an independent read of the change, each claim verified in the
  code): `jobset status` and the Results tab's ladder still read a rung's
  attempts from the stage folder, so a running scan read *prepped, not
  launched*; they ask the door now, and a scan's rung speaks from its first
  point not finished, naming the point (`results.md` § 2.4,
  `running-a-job.md` § 4.2). The deck comment's first rewrite listed whose
  value `restart` is and missed a benchmark trial's pin; the catalogue said no
  vibration deck reads `restart`, though the relaxation rung reads its
  default; the gather chose which upstream renders at a point by its name
  (`per_point_rungs` now). Stale since the rows went: `pyscf/stages.py`'s
  docstring, `runwrap`'s suffix note, the rules file's header,
  `job-contracts.md`'s `--cold` lesson (a name sweep since U17), two
  citations. The record's device test asserts `ran` over a measured SIESTA
  output; the probe test's scenario is a finished rung's trajectory — four
  more mutations, each red. *Parked*: on the flat shape a vibration's
  relaxation re-run after its force constants starts from the FC run's last
  `.XV`, a displaced geometry — the same minimum, a few steps more (read from
  the M11 review, not measured); the record's per-rung `state` words (`ran`,
  `no_output`, `not_run`, `unreadable`) are stated in no document (before
  K10).

#### K12

> **Verdict (2026-10-08 validation):** CONFIRMED (`identity.py:218-242,287-294,387-424`; `siesta/input.py:559-571,989-1006`; `pyscf/input.py:280-281`; `template.py:1786,1813`; `job-system.md:1148,1185-1196`; `stages.md:582`; `tests/test_stage_names.py:6,81-82`).

* **K12 — done 2026-10-01** (SS-C11, SS-C15). **What molbuilder prints, you
  can type** (`job-system.md` § 5.3): a printed command names a stage by its
  name. The SIESTA and PySCF relaxation decks' headers printed the token,
  `jobset launch run 02_freq`, which `launch` refused — the token being a
  legal name of another stage (SS-C11) — and the PySCF vibration deck a
  placeholder; a header holds only the token, so it prints the name through
  `identity.command_stage`. Never `#N` either, which bash reads as a comment
  unquoted (the contract now says to quote it). **One key**,
  `identity.stage_key`, compares a name the description holds with one from
  outside it (`stages.md` § 2): the description's two duplicate checks, the
  verbs' resolver (`launch run TIGHT` is `tight`), and the role rule,
  `template.stage_role` with the stage table's `roleOf` in the browser — a
  SIESTA vibration's row renamed `Relax` in Task setup rendered the
  force-constant deck, and prep, which found the relaxation by the exact
  name, refused `freq` for a ladder with no relax stage (SS-C15); it asks the
  role rule now. A verb resolves once, at its entry, and uses the
  description's spelling after that. **One resolver**: the bench verbs
  matched the exact name in a lookup of their own, so `launch bench '#1'`
  refused the stage `prep bench '#1'` had prepared; they resolve once,
  through `resolve_stage_ref`, and print the name back. Contract:
  `job-system.md` § 5.3; `stages.md` § 2; `template.md` § 6.4;
  `task-setup.md` § 5. Tests through the road: each deck's printed launch
  line typed back, and in another case, to `launch --dry-run`; a vibration's
  relax row renamed and saved, then prepped; the bench verbs by `#1` and in
  another case; the stage table's role after a rename in the browser — nine
  mutations, each red. The PySCF header test asserted the token, and asserts
  the name now. *The review* (an independent read of the change, each claim
  verified in the code): a benchmark trial's deck now printed
  `launch run coarse`, which launches the stage's run, not the trial — prep
  tells the deck it is a trial (`spec_for(trial=)`) and the header prints
  `launch bench <stage> <trial>`, typed back to plan that trial alone; a bare
  `launch run` on a ladder listed the tokens as what to type (the typeable
  `coarse (#1)` now); `launch` recorded the typed spelling in the decision
  ledger (the name now); `job-system.md` showed `status 3`, which names a
  stage called `3` (`status '#3'`); the printer and the key were claimed for
  more than they cover, and W37 still listed the header's `02_medium` — four
  more mutations, each red. The vibration remedy names the relaxation rung
  `relax`, the kind's name, which resolves in any case; a stage renamed
  `Relax` is named `relax` there (`vibration.md` § 5.3).

---

# § 7 — text removed by the 2026-10-08 cleanup

*(Each block below is verbatim what `plan.md` § 7 held before the cleanup;
the `Verdict` line is the 2026-10-08 validation's finding with its evidence.
The section itself stays open — § 8.5 and § 0a's carry line lean on it.)*

## 7.1 What is actually wrong — measured 2026-09-07, before proposing (archived 2026-10-08)

> **Verdict:** REFUTED (7-2) — the route-catalogue sweep (plan § 0a *Unscheduled*, `:585-587` before this cleanup) found 11 documented routes that do not exist and 9 live ones undocumented; 71 unique `@bp.route("/api…")` lines today (`grep -rhoE '@bp\.route\("/api[^"]*"' molbuilder/web | sort -u`). The 2026-09-29 parenthetical is folded into the rewritten paragraph.

**Not coverage.** The first guess was that documentation had fallen behind the
code, and it has not. Every one of the **77** `/api/*` routes the blueprints
serve is named in `web-api.md` — re-derived by expanding the doc's own braced
forms (`/api/checkpoint/{state,list,config}`), which a naive matcher scores as
26 missing. Do not plan a coverage sweep; there is nothing to catch up on.
*(2026-09-29: coverage in this sense still holds; truthfulness does not. The
coverage check of that day read every live document against the code and
found statements the code contradicts — § 5v lists them, and the survey
starts from that list rather than repeating it.)*

> **Verdict:** PARTLY stale (7-3) — re-measured 2026-10-08: 77 `.md` outside `archive/` and `plans/`, 68,796 lines, nine directories holding them (`docs/` and eight below it). Numbers replaced in place.

**Findability.** 75 live documents, **50,609 lines**, nine directories. The
information is there and a person cannot reach it.

> **Verdict:** CONFIRMED (7-4) — `docs/README.md:211` is the index; the longest row is 823 words (`:249`). Kept; the measured numbers were added.

**The existing index has the problem it is meant to solve.** `README.md`
§ Index is one row per DOCUMENT, and several rows run past 400 words. To find
out which document owns a fact you read essays about documents. It is a good
map of the tree and a poor answer to "where is X".

> **Verdict:** PARTLY (7-5) — `science/pseudopotentials.md` § 3 has been corrected (its note at `:470-479` records the removal); the `process/testing.md` § 6 quote was not re-located 2026-10-08. Kept as history, with those two facts added.

**Two documents contradicted themselves in one day**, both found by accident
while doing something else — `science/pseudopotentials.md` § 3 listed a check
as not-done that its own § 2a.1 documents as done, and `process/testing.md`
§ 6 quoted a number three sentences after telling the reader never to quote a
number from a document. Neither is exotic. Nothing systematic would have
found them, which is the survey's real justification.

## 7.2 The four indexes (archived 2026-10-08 — the API and Data-structure rows' old text)

> **Verdict:** CONFIRMED premise (7-7) for the Module row (`a291fbbf`). The API row gained the route sweep as a second argument; the Data-structure row gained the pointer to `execution/project-layout.md` § 5 (`:2570-2695`). The old cells:

| **API** | route | what it answers · who calls it · which doc owns it | **generated** — the route list is derivable from the blueprints, and a generated index cannot drift. The 2026-09-07 W8 finding is the argument: `web-api.md` claimed the Documents tab read `/api/docs/list`, which it has not since the commit after the one that added it |
| **Data structure** | persisted file / schema | its shape · its version · its one reader and one writer | **written** — the doors are a design fact, not derivable |

## 7.3 The reference index, which is the one with real content in it (archived 2026-10-08)

> **Verdict:** REFUTED (7-10) — `trajectory_log/emitter.py:40,43` imports `HARTREE_BOHR_EV_ANGSTROM_ASE` from `constants` (both the in-package and the beside-a-job branch); no `HARTREE_BOHR_TO_EV_ANG` exists anywhere under `molbuilder/` (grep 2026-10-08). The "why" is the comment at `constants.py:95-100`. The point — the reasoning has no documentation home — stands; the example was rewritten.

The worked example, found while measuring this:
`trajectory_log/emitter.py` retypes `HARTREE_BOHR_TO_EV_ANG = 51.42208619`
rather than deriving it from CODATA-2018 (which gives 51.422067476, ~0.4 ppm
apart) — deliberately, so emitted forces line up with what a person reads in
ASE, VASP and QE logs. That is a real convention choice with a real
justification, and it lives in a code comment nobody will find.

> **Verdict:** PARTLY (7-11) — `science/references.bib` holds 34 entries (`grep -c '^@'`); 7 live documents reference it; Reed 2006 and Stokbro 2003 are still absent (E14 open). Numbers refreshed; "at least 12 cite literature in prose" was not re-counted and is dropped.

Citations are split the same way: `science/references.bib` holds **21**
entries, **5** live documents reference it, and at least **12** cite
literature in prose instead. Two keys the roadmap called for — Reed 2006 and
Stokbro 2003 — are in neither (row **E14**).

## 7.4 Order, and the one risk (archived 2026-10-08 — step 1's old text)

> **Verdict:** PARTLY done (7-12) — step 1 is § 5v (2026-09-29, plan `:2627` before this cleanup: "Documents that lag the code — the document sweeps' input"). Rewritten to say so; steps 2–4 stay open. The risk paragraph is CONFIRMED history/rule (7-13) and kept.

1. **Survey first, and record findings without fixing them.** Every live doc
   read against the code, one pass, findings in a list. Fixing while reading
   is how a survey becomes a month.

---

# § 8 — text removed by the 2026-10-08 cleanup

*(Each block below is verbatim what `plan.md` § 8 held before the cleanup;
the `Verdict` line is the 2026-10-08 validation's finding with its evidence.
Rows 10, 11, 13, 14 stay in the plan.)*

## 8. One nature, fourteen instances — the reading doors have no guard (archived 2026-10-08 — the ten closed rows)

> **Verdict:** the premise ("fourteen instances, no guard") is gone: rows 1–9 and 12 are fixed, each by deleting the copy or routing it through the door. The 2026-09-07 head note is kept in the plan; its sentence "Every instance below was found by accident, chasing something else. None was found by a guard, because no guard covers this class." is folded into the rewritten § 8.1.

> **Verdict (row 1):** CONFIRMED fixed — `molbuilder/__init__.py:92-100`: "THERE IS NO `load()` HERE, and that is the rule ... THE DOOR IS `StructureCodec`".

| 1 | reading the `.xyz`+`.molstruct.json` pair (`StructureCodec`) | `molbuilder.load()` | dropped the whole sidecar; `jobset init` therefore wrote descriptions with no regions, frozen atoms or cell | **FIXED 2026-09-07** — deleted |

> **Verdict (row 2):** DONE — `molbuilder/selection.py` has no `_load_structure` (grep 2026-10-08).

| 2 | ” | `selection.py::_load_structure` | applies `regions` only — drops identity columns, cell, annotations, `info`; three rule kinds then return nothing | dead code, deletion pending |

> **Verdict (row 3):** DONE — `molbuilder/transport/_cli.py` does not exist (§ 0a already said "§ 8's rows 3 and 7 name deleted code").

| 3 | ” | `transport/_cli.py::_load_device` | applies the whole sidecar — correct, but a third copy of the walk | open |

> **Verdict (row 4):** CONFIRMED fixed (`6822f639`) — `transport/compose.py:253-256` pairs each `.xyz` through `sidecars.molstruct.sidecar_path_for`, not a glob.

| 4 | ” | `compose.py::labeled_citation_structure` | globs `*.molstruct.json` instead of asking the pairing rule | **FIXED 2026-09-08** (`6822f639`) — through the finder, `sidecars.molstruct.sidecars_in`; found closed when M5 step 3 took it up |

> **Verdict (row 5):** CONFIRMED closed (W56 4c, 2026-10-04) — the row said so already.

| 5 | the pairing rule (`sidecar_path_for`) | `parse/engines/_sidecar.py` | own suffix-strip table; **disagrees** on `x_optim.xyz` and `z.molwatch.log` — deliberate, but a third derivation | **closed 2026-10-04** (W56 4c) — `_sidecar.py` deleted: no parser looks for a sidecar beside an output |

> **Verdict (row 6):** DONE — `web/blueprints/files.py:246-255` `_paired_sidecar_path` asks `sidecars.molstruct.sidecar_path_for`; the residue is `_STRUCTURE_SUFFIXES` (`:243`, which files carry a structure — not the pairing).

| 6 | ” | `files.py::_paired_sidecar_path` | agrees today; a fourth copy is a coin-flip on the day one changes | open |

> **Verdict (row 7):** DONE — `web/blueprints/selection.py:123` serves only `/api/selection/eval`; `/api/selection/atoms` is gone. (Doc residue, § 5v's kind: `web/blueprints/__init__.py:13` still documents `/api/selection/atoms`.)

| 7 | per-atom rows (`_shared.atoms_list`) | `/api/selection/atoms` | skips the cell resolver, so it answers with a box that was never resolved | dead, deletion pending |

> **Verdict (row 8):** DONE — `molview/render-engine.js:96-98` `isolateInEffect(switches, selection)` is the one accessor; `mount.js:209-212` reads it before the click gate.

| 8 | "is isolate in effect" (`isolate && selection.length > 0`) | `mount.js:228` reads the raw switch | **live UI defect** — with nothing selected, the 3-D window stops accepting clicks, silently | open, fix agreed |

> **Verdict (row 9):** DONE — `molview/stores.js:129-131`: "ISOLATE IS A PREFERENCE, AND IT STAYS WHERE YOU PUT IT (user, 2026-09-07)"; no auto-off.

| 9 | ” | `stores.js:145` auto-off | a second mechanism for the same rule; it is also what makes the button un-press itself | open, fix agreed |

> **Verdict (row 12):** DONE — `molbuilder/structure.py:1925` writes `labels_modified` or `structure_modified` by what changed.

| 12 | what an edit invalidates | one `structure_modified` flag for geometry, cell AND labels | unusable by its only reader | **FIXED 2026-09-07** — split in two |

## 8.2 Why nothing caught them, and what the codebase already does about it (archived 2026-10-08)

> **Verdict:** REFUTED ("nine guards exist") — four remain (`test_config_dir_has_one_home.py`, `test_css_module_boundary.py`, `test_no_duplicated_ui_components.py`, `test_no_test_is_shadowed.py`); five files are gone — `test_layering.py` (`a291fbbf`), `test_one_home_for_a_constant.py`, `test_css_no_duplicate_selectors.py`, `test_vibrationview_module_boundary.py` (`082ba979`), `test_one_naming_authority.py` (`164ecbfe`) — and the scan inside `test_config_dir_has_one_home.py` with them (`164ecbfe`, its last commit). The report counted six deleted and named `2b83947f` among the commits; `git log --diff-filter=D` maps the five absent files to `a291fbbf`, `082ba979` and `164ecbfe` only. "The missing member of an existing family" has no family to join. Rewritten.

**This project already knows the answer.** Nine guards exist, each written
after one of these was found the hard way:

`test_layering.py` · `test_one_home_for_a_constant.py` (the Bohr radius, once
written out eight times) · `test_one_naming_authority.py` · `test_config_dir_has_one_home.py`
· `test_css_no_duplicate_selectors.py` · `test_css_module_boundary.py` ·
`test_vibrationview_module_boundary.py` · `test_no_duplicated_ui_components.py`
· `test_no_test_is_shadowed.py`

*(2026-09-26: the source scans in `test_one_naming_authority.py` and
`test_config_dir_has_one_home.py` were deleted on the user's rule — a test runs
the code; what each file keeps does. 2026-09-27: `test_layering.py`'s three
scans likewise -- the layer rule is review's, `code-audit.md` § 1c (e) -- and
its one test that runs the code moved to `test_monitor_bundle_runs_alone.py`.)*

They cover imports, constants, a naming rule, a config path, CSS, two module
seals, UI components and test names. **Not one covers a reading or writing
door** — which is where rows 1-7 live.

So this is not a new mechanism to invent. It is **the missing member of an
existing family.**

## 8.3 The framework fix (archived 2026-10-08)

> **Verdict:** OBSOLETE — ruled out. The guard it describes is an AST source scan "following `test_one_home_for_a_constant.py` exactly", and that test and its kind were retired by `082ba979` ("retire 60 tests that assert where code lives, not what it does") and `a291fbbf` (the layer rule is review's); § 7.3 records "there is no lint". Rows 2–7, its intended first findings, were closed by reading.

**A guard that enumerates who may read the pair, and fails when a second
implementation appears.**

Shape, following `test_one_home_for_a_constant.py` exactly — an owner, an
allowlist with a REASON per entry, and a check that each allowance still
describes the code:

* **owner** — `StructureCodec`.
* **the lint** — AST-walk `molbuilder/**`; any function that both reads a
  geometry path AND looks for a sidecar beside it, and is not the codec, is a
  finding.
* **allowlist** — entries carry the reason they are exempt, and the guard
  fails if an allowed site stops doing the thing it was allowed for (the
  existing constant guard does exactly this, so an allowance cannot outlive
  its reason).
* **the same for the pairing rule** — `sidecar_path_for` is the owner; rows
  5 and 6 are its findings.

**Why a hand-written list is not enough, and this is the load() lesson:** a
list of doors written by a person would have carried `molbuilder.load()` on it
looking entirely reasonable — right name, in `__all__`, docstring accurate for
what it did. Only *deriving* the readers from the code and comparing against
one declared owner surfaces a rival. **The index must be generated** (§ 7.2),
and the guard must read the code, not a list of names.

## 8.4 What is NOT an instance, and must be fixed on its own (archived 2026-10-08)

> **Verdict:** DONE, both — row 8: `molview/render-engine.js:96-98` `isolateInEffect`, read by `mount.js:209-212`; the auto-off deleted (`stores.js:129-131`). `lastApply`: `stores.js:128` clears it in the one `set` every selection door goes through (`:225`, `:248`, `:254`, `:264` likewise), kept only by `applyFilter` (`:283`), which produced it. § 0a's carry line still listed both as open.

Two findings are ordinary defects, not door duplication, and a guard will
never catch them:

* **Row 8, the isolate click gate** — a live UI defect. Nothing selected,
  press "Show selected only", and the 3-D window silently stops accepting
  clicks while the drawing does not change. Fix agreed with the user: one
  accessor in the store, read by all three, and the auto-off deleted (a
  preference stays where you put it; "nothing selected" means nothing to
  hide).
* **`lastApply` never cleared by a selection change** (`stores.js:136-147`) —
  the panel can read *"No atoms matched this filter"* beside *"N of N
  selected"*.

## 8.5 Order (archived 2026-10-08)

> **Verdict:** OBSOLETE — step 1 is the ruled-out guard (§ 8.3); steps 2–4 are done by reading (rows 2, 3, 6, 7 above); step 5 is what remains and now stands in § 8.1's state column. The "~10 rule tests" and the "atom-row wire-shape claim" of step 2 went with the dead route (`/api/selection/atoms` gone; `web/blueprints/selection.py:123`).

1. The guard for the pair readers **first**, with rows 2-6 as its initial
   findings — so the deletions that follow are checked by something rather
   than by me.
2. Delete row 2 and row 7 (dead), rehome the ~10 rule tests onto the live
   door, and re-home the atom-row wire-shape claim, which today is pinned
   only through the dead route.
3. Rows 3-6 by the guard's allowlist: each either goes through the door or
   earns a written reason.
4. Rows 8 and 9 as their own fix.
5. Rows 13-14 fold into § 7's survey.

---

# § 9 — text removed by the 2026-10-08 cleanup

*(Each block below is verbatim what `plan.md` § 9 held before the cleanup;
the `Verdict` line is the 2026-10-08 validation's finding with its evidence.
§ 9.2 stays in the plan whole — `.gitignore:85` and
`science/pseudopotentials.md` § 3 (`:446-470`) cite it by id.)*

## 9. ATOM as a pseudopotential validator — PENDING, not started (archived 2026-10-08 — the head note)

> **Verdict:** CONFIRMED as written; the status in the header moved to "§ 9.1 DONE, § 9.2 PENDING" and the head note now points at § 10's archive line ("§ 9.2 (ATOM) stays in § 9", plan `:3729` before this cleanup).

*(User, 2026-09-19. Recorded now so the scope is honest; the immediate work
is § 9.1, which is done, and § 9.2 is future.)*

## 9.1 What the configuration-time check must guarantee — DONE (archived 2026-10-08)

> **Verdict:** CONFIRMED DONE — `validation/siesta.py:60-80` (`_check_siesta_pseudo_coverage`: "folder known, and it lacks a species -> ERROR; no folder yet (Build tab, before a save) -> WARN"); `pseudos.py:626-637` (`misnamed` in `ERROR_STATUSES`, "the SINGLE source of truth for which statuses are ERROR", consumed by the preflight and `cli.cmd_pseudo_check`); `science/pseudopotentials.md` § C1a (`:139`, "the pseudopotential for element `E` is the file named `E.psml`"). Reduced to the one-paragraph pointer in the plan.

The user's rule for this level, verbatim: *"make sure that the file are
explicitly provided, the name matches the atom name, and that's pretty much
it at this level."* Two things, both at configuration time, both now enforced
by the checks the Build preflight and `prep` already share:

| | where |
|---|---|
| **explicitly provided** — `psml_lib` names a library, OR the calculation folder already holds them; never neither | `validation.siesta._check_siesta_pseudo_coverage`. With a calculation folder in hand this is answerable, so it is an **ERROR**; before one exists (Build tab) it stays a WARN, because whether the folder will supply them is not knowable yet |
| **the name matches the element** — the file for element `E` is `E.psml`, which is the name SIESTA opens | `pseudos.check_coverage` → `misnamed`, in `ERROR_STATUSES`. The map is keyed on what a file DECLARES, so a correct gold pseudopotential saved as `gold.psml` used to answer `ok` for `Au` while SIESTA could not start |

Both are refusals at the moment the calculation is configured, which is the
only moment they are cheap. Nothing below this line is required for that.

## 9.2 ATOM as an optional extra layer (archived 2026-10-08 — two sentences only)

> **Verdict:** the path is gone — `~/Downloads/atom-4.2.7-100` no longer exists on this machine (2026-10-08); the section's facts came from that 2026-09-19 reading and the plan now says so. Everything else in § 9.2 is kept verbatim: still absent, still wanted, no later ruling.

`~/Downloads/atom-4.2.7-100` — ATOM 4.2.7, the SIESTA project's own
pseudopotential program (Froyen / Troullier / Martins, maintained by Alberto
García). Not in the repo.

> **Verdict:** "added now" is history — `atom*/` is in `.gitignore:88`, with the § 9.2 pointer at `:85`. The cell now says "added (`.gitignore:85-88`)".

| `x3dna*/` in `.gitignore` | `atom*/` likewise, **added now**: the ignore has to exist *before* the folder does, or a redistribution-prohibited package lands in `git status` |

> **Verdict:** the § 3 citation gained its line range (`:446-470`) and the note that § 3 points back at § 9.2; no other change.

Those are precisely the two items
[`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) § 3 declares
out of scope today, and the ATOM tutorial's own warning is the argument for
them: *"You should thoroughly test a pseudopotential before using it."*

---

## 5x / 5u — the transport programme's record, closed 2026-10-08 (B0–B9, the two review rounds)

> **Verdict (2026-10-08, evening):** the transport calculation runs end to end on molbuilder's own road for its three cases, the milestone review (B8) came back clean after one revision in each of two rounds, and B9 swept the documents. What the programme left open is in `plans/plan.md` § 2 (F9, F11, F14, F19b, R2-15, S-T2, TD10 text, L1, L2) and § 5u.1 (steps 7, 8, 10, 11). The sections below are moved here whole.

### 5x.2 What was wrong on 2026-10-08, and why (validated; the evidence by id in § 5x.6, each closed by the step § 5x.6's map names)

1. **A sweep was built as five little stages** — each point its own folder of
   attempts, its own launch record, its own state. From that one wrong shape:
   points read `queued` forever; launching again re-ran done points and
   emptied the I–V meanwhile; the walk's hand-over was recorded nowhere, so the
   provenance was false; no clean seed density survived a first launch; the
   scan's stage folder held a deck nothing ran; the relaunch rule existed
   twice; `--cold` on an unlaunched scan was dropped.
2. **`low-bias` computed something else than it said**: TBtrans per voltage on
   the 0 V Hamiltonian with the leads shifted ±V/2 (`m_ts_electrode.F90:1461`)
   — neither linear response nor self-consistent, labelled "linear response".
3. **The record reads TBtrans files with a second reader** (its own glob, beside
   `parse/engines/tbtrans.py`): a spin-polarised point reads "failed". *(still so)*
4. **Three walk scripts** for one job: the benchmark's, the group's, the scan's.
5. Smaller, each one place: the electrode swap writes the cited run's files
   (§ 4 says the calculation's own copy); a lead's file name from two sources
   (`cfg.system_label` vs `task.label`); a relaxation that ended with an error,
   or a bare structure with no run, is citable — and a bare structure brings no
   pseudopotentials, so the contract's own refusal ends in *"put the files in
   `pseudos/` yourself"*, a hand step; the energy window is not checked against
   the bias; the treatment label is not beside the I–V; the E_F reference is
   asserted, not checked; the device facts shown can be from another run than
   the one the transmission read; `status` never shows what a rung gathered.
   *(all still so)*
6. **The contract** defined "run" three ways and kept the unbuilt sweep design
   beside the built rule, unmarked. *(§ 2a.10–2a.12 and rule 3 rewritten, B0;
   the rest of the text is B9.)*

### 5x.4 The build — step by step, with its state

Every step: its targeted tests **judged** (gate 5 — a test a change breaks is
judged for retirement, never repaired to pass) before its commit; its "done
when" seen on **the road junction**, through the page and the printed verbs.

| step | what exactly | files | done when | state |
|---|---|---|---|---|
| **B0** | **the contract** — § 2a.10 (the switch, the treatment table, low-bias = record-computed linear response), § 2a.11 rewritten as this design, § 1.1, § 2a.12; "run" one meaning in `job-system.md` *Words* and rule 3 read at the point; `project-layout.md` § 1.5–1.6 | `transport.md`, `job-system.md`, `project-layout.md` | read, and yes | **done**, d398ae86 — *it did not do the text sweep it claimed* (the "scan" passages, `_plan_chain` in § 3.5's table, "attempt" as the folder word): that is B9. Four sentences fixed 2026-10-08 to what the build settled: warm-all-done refused; the newest device run whole or refused; the doors' real names; each point its own copies |
| B1 | the switch `low_bias_approximation` in `task.json` (required for more than one voltage, refused for one, never inferred); `jobset init --low-bias-approximation / --no-…`; the describe route and the tab's yes/no; `jobset migrate` from the old `treatment` word; `sweep_points`; the record's three words | `task.py`, `_cli.py`, `migrate.py`, `stages.py`, `record.py`, the tab | `sweep_points` answers `()` / `()` / `(0, 0.2, 0.4)` for the three cases | **done**, 7351a3ff |
| B2 | the layout: `point_folders`, `points_in`, `open_sweep_run` / `SweepRun`, `result_folders`; every reader moved, the old doors deleted; prep opens `run-0/vX/` with the run's clean inputs and each point's own copies; no deck at the stage level; Task setup's count reads `attempts_in` | `transport/stages.py`, `jobset/materialize.py`, `jobset/prep.py`, `transport/record.py`, `jobset/runstatus.py`, `web/blueprints/build.py` | `prep task --stage device` on the road junction: the tree as § 2a.11 draws it | **verified on the road junction 2026-10-08** (`au-dta-t`): `04_device/run-0/{v0,v0.2,v0.4}`, the run's clean `.DM` + two `.TSHS`, `.gathered-from` once |
| B3 | the done-door `done` + `products_of`; the transmission's prep takes the newest device run whole or is refused | `jobset/continuation.py`, `jobset/prep.py` (`transport_inputs`) | the transmission prep refused while a device point is not done, then gathers each point's `.TS.HSX` from it | **verified on the road** — the transmission prep took `04_device/run-2` whole, each point's `.TS.HSX`; while run-0/run-1 had a point not done the transmission's row said so and why |
| B4 | the one walker `_walk_script` for bench, group and sweep; `_plan_sweep` (one `run.json`, `.continued-from` at plan time, warm take-over, cold, warm-all-done refused); `_carry_sweep_gather`; the `along` row; `_plan_chain`, `_bench_walk`, `never_started`, `_a_scan`, the per-point launch records deleted | `jobset/submit.py`, `warmfiles.py`, `siesta/warm-files.toml` | (i) the device sweep walks 0 → 0.2 → 0.4, each from the previous point's `.TSDE`, said in the log and `status`; (ii) **the stop**: the device stopped at 0.2 V through the road — `max_scf_iterations` 1 on Task setup is a description change, so a new calculation — `status` says *1 of 3 points done; 0.2 V: …*, 0.4 V *not run*; launched again warm: takes 0 V over, runs 0.2 and 0.4; (iii) every point done + warm → refused with the `--cold` line; `--cold` → `run-1/`, all three | **(i), (ii), (iii)-refusal verified on the road 2026-10-08** (§ 5x.7): three launches — run-0 did 0 V, stopped at 0.2 V; run-1 took 0 V over, did 0.2 V, stopped at 0.4 V; run-2 took 0 V and 0.2 V over, did 0.4 V; every `.continued-from` right; `status` *1 of 3 points done; 0.2 V: stopped before its end* → *3 of 3 points done*; a fourth warm launch refused with the `--cold` line. **Not run: `--cold`** |
| **B2–B4 gate** | the targeted tests judged: `test_transport_task.py`, `test_transport_record.py`, `test_launch_protocol.py`, `test_launch_door.py`, `test_launch_values.py`, `test_launch_ask_mode.py`, `test_prep_protocol.py`, `test_prep_bench_fold.py` (the walker), `test_status_lists_the_ladder.py`, plus every test naming a moved name (grep `tests/` recursively for `_bench_walk`, `rung_container`, `_plan_chain`, `scan_points`, `never_started`); then **one commit** naming S1–S10 and P4 | `tests/` | the tests judged; the server restarted; the commit | **done**: 206 tests passed, nothing to judge; commit `ddf12a43`; the record `NameError` the walk found fixed in `8dc08851` |
| **the road junction** | built on the Molbuilder tab in Chrome, relaxed through `init`/`prep`/`launch`, cited on the Transport tab (§ 5x.0) — the subject of every "done when" from here | `projects/claude-transport-walk/…` (a new calculation) | the relaxation finished; the transport described from its citation | **done 2026-10-08** (§ 5x.7): `structure/au-dithioacetylene` built and labelled on the page; `optimization/au-dta-relax` relaxed (finished, geometry NO); cited; `transport/au-dta-t` run to its record |
| B5 | `status <stage>`: the points (done / why / started from) and the gather shown on every rung (R5); the sweep row's converged column — `SCF yes` when every point is done, `SCF NO` when a point finished unconverged (today: empty unless all done) — settled at B5 | `jobset/runstatus.py`, `jobset/_cli.py` (status) | `jobset status` and `status device` on the road junction, after (i) and (ii) | **done 2026-10-08** (`status device`: the gather, *0 V, 0.2 V taken over from run-1; the rest walked*, each point's start; a transmission point's *took*; the ready line's gather once; `SCF NO` when a point finished unconverged) |
| B6 | the record and the Results tab: `tbtrans.transmission_files` (the channels) replaces `record.py`'s own glob (R1); the low-bias I(V) from T(E,0) (P1's record half); device facts from the run the transmission gathered (R4); E_F of device vs leads **checked** (R3); the treatment label beside the I–V (R2); R6's facts (the seed's E_F, the contour and poles, the window, points, TBT k-grid); the report composed on read and every run listed at the root, as § 2a.12 and `results.md` § 2.4 already say (R7); **R9 — the 3D viewer draws nothing in a real browser: its cause read before anything else on this tab** | `transport/record.py`, `parse/engines/tbtrans.py`, `inspectors/transport.js`, `results/viewer.js` | `summarize task` and the Results tab on the road junction, the three cases | record half committed (`cf26ac24`); Results half committed (`6d4ea7c4`): R2, R3 (the E_F frames), R9 (side-on), P1's table; R7 (decision 8: the root opens its report through `task.json`; every run listed and picked in place, `ac2d5741`); the low-bias (`au-dta-lb`) and single-bias (`au-dta-sb`) cases walked on the road junction (§ 5x.7, `fc413816`) — **done** |
| B7 | the small ones, each through the road: the swap applied to the calculation's copy (C1, § 4); the lead's file stem from `task.label` alone (C2); **the citation is a finished molbuilder relaxation run, nothing else** (C4; decision 7) — § 3.1's two saved-structure rows go, the Transport tab's picker offers runs, `compose` refuses the rest naming the relaxation to run first; C3 falls with form B; the window gate ± V/2 + 5 kT in the settings gate (P2) | `transport/compose.py`, `transiesta.py`, `web/blueprints/transport.py`, the tab, `validation/` | each refusal and each pass seen through the road | **done** (2026-10-08): F25 + P2 (`16409a44`), F16 + C2 + C1 (`548b4c74`), decision 7 (C4, C3, F4's rule; the citation's convergence carried, F14's B7 part) — the saved-structure forms, the recorded-contract lane and the cited-file rewriter are gone; compose tests cite a run made on the road (the stand-in leaves a `.XV` with the deck's geometry) |
| **B8** | **the milestone review** *(user, 2026-10-08)* — two rounds, fresh agents, the full code TEXT, every finding verified, one revision each: **the backend** (`jobset/`: prep, launch, status, continuation, materialize, runstatus, ledger; `transport/`: stages, compose, record, deck); **the CLI** (`init` / `prep` / `launch` / `status` / `summarize` for transport, the printed lines); **the UI, fact-checked against the contracts** — the Transport tab (the citation, the bias list and its builder, the switch, the template: is every parameter in its logical place, named by the contract's word, with no second home), Task setup (the stages, `execution`, `allocation`, the Prep ladder), the Results tab (the ladder with its points, T(E) per channel, the I–V with its label, the viewer — R9); **the checkpoint** (`molbuilder checkpoint save / restore` against *How the checkpoint supports you* and rule 1: a state before every prep; rollback as the only redo); **the stage/run configs** (`task.json`'s stages and run cards — `restart`, `continue_retries`, threads, `time`; the catalogue's stage rows; what prep states or refuses); **the CSS** — tokens only, the shared components (`lib/state-chip`, `ts-planlist`, `tm-kinds`), `[hidden]` guards, no magic numbers, the global sheets' reach | all of the above | both rounds clean | **round 1 under way** (condition met 2026-10-08): five read-only agents reported (backend, CLI + checkpoint, the Transport tab, the Results tab, the suite), every finding verified at its lines; batch 1 `8fab088a` (the take-over's origin, the status row at the point, the record's provenance and contour, the picked run); batch 2 (2026-10-08): U1 the eight SCF items declare the SCF rungs (`stages`), the transmission deck writes `ElectronicTemperature` alone; U4 `describe_attempt` answers `findings` (info / warn / error) under a one-line summary, rendered by the one renderer; U5 a refusal lands beside its control — rung ids carry the rung (`t-<rung>-…`), `fieldIds` keyed by stage, the codec's bias refusals name `where` (`task.DescriptionError`), the rung's tab is selected and the fold opened; U7 Task setup echoes the junction, the bias list and the treatment (`Task.treatment`, the folder answer's word); U8 the Measure card carries `bench_refusal`; U9 `BENCH_START` gone; U15 one shape (`form` / `contract` keys dropped); U16 wording, `.disabled-tip`; R1-10 `is-pick`; batch 3 (2026-10-08, the CSS and the keyboard): U10 one `.choice-list` / `.choice-row` in form-components.css (This machine's kinds, the bias treatment); U11 form-schema.css scoped to `.schema-section`; U12 `--tr-*` tokens; U13 the shared sheets' spacing on the grid; R1-05 the Selection panel's height is its content's with the extent as its minimum, the fill region a zero-basis flex item with a floor (`--molviewer-size-fill-rows`), the dead 340px cap gone; R1-07 the Results sheet's fallbacks dropped, weights and sizes from tokens; R1-08 one type family in MolView; U14 / R1-12 ui-contract.md § 4.1 (named, reachable without a mouse): triple cells named, the picker's rows focusable treeitems with the arrows and Enter, Task setup's `×` buttons named, the report's tabs the ARIA pattern with arrow keys, the PDOS controls named, the I–V pick a button with `aria-pressed`, the ladder's open buttons named per rung with `aria-pressed`; batch 4 (2026-10-08): C2 a `checkpoint restore` appends its own ledger line after the rollback (checkpointing.md § 7); C3 a directory the restore emptied is removed; F27 the deck's unit word is the catalogue's `unit` (`siesta/layout.unit_word`, fdf's `Ang` for Å, the bias's `eV` the one exception; the window block reads it too); R1-02 / F23 a transmission run's result is the report at its point — `every_run` answers `opens: task.json` with the point, the row's *open* re-announces the root with that bias selected (`selectBias`), `opens` root-relative; R1-09 the chart's height is the sheet's (`--tr-plot-h`, `--tr-plot-half-h`; no `height` in a layout); T4 the three headers say what their files hold; T5 the hand-built record tests retired; T6 the G0·V test says why it is API-level; T9 one water (`tests/validation/conftest.py` gone); **round 1 complete** (`23721631`); **round 2 complete 2026-10-08** — five fresh read-only agents verified round 1's fixes and read the scopes again; their findings verified at the lines and revised in one commit: the restore's ledger line reverted (A5: the ledger is part of the state), the record's per-point provenance from the gathered run, the gather carried from the run its record names, one product list for `done`, the transmission deck without the MPI lines, the rename echoed on Task setup, the held citation kept on a non-citable pick, the bias list's client refusals beside it, tabs.md § 4 rewritten, the picker's rows as treeitems, `record` out of the source vocabulary, no re-mount on a ladder open at a point, the device row naming its point, one sizing rule for MolView's three pages and its contract corrected, the report's handle in every contract, five engine-less road tests renamed out of `_e2e`, the codec's refusals driven through `jobset init`, the source-text pins retired, A5's test seeing directories; **open from round 2** (not fixed, recorded for B9 / the next unit): R2-15 the file card after a picked run (two folders announced), the stand-in engine leaving empty rung products so the transport road becomes testable (S T2, a proposal to the user), the walk-script member word on the group and bench callers' log lines (done), `test_results_file_picker_js` hand-typed dir rows (judged: pure-function inputs, kept); then B9 |
| B9 | the doc sweep: `transport.md`'s "scan" passages (the orientation at :40, § 2a.7's grouping row, § 2a.9's axis rule, § 2a.10's advice, § 3's example and table, § 5's bias rule, § 6a, § 8) → "sweep" and this design; § 2a.14's table, § 3.2, § 3.4, § 3.6a; "attempt" in `project-layout.md` § 1.5–1.6 — the folder's word is **run**, `attempt` stays the code's identifier, said once in *Words*; R8's two links; the 18 contradictions (T) re-read | `docs/` | a reader finds one rule per concept | not started |

**Parked, named** (not Q14's): **P3** — the device's equilibrium contour is the
interim 10 eV pole energy, not a stated `contour.eq` with the spectrum gate;
§ 6.1c says so itself (*"The 10 eV is interim — M3 P4's stated contour"*). It
is M3 P4's, independent of the sweep.

### 5x.6 The sweep's evidence — every confirmed finding, by id, and what closes it

| closes | ids |
|---|---|
| B1 | P1 (the switch half) |
| B2 | S4, S6 |
| B4 | S1, S2, S3, S5, S7, S8, S9, S10, P4 (each point's `.continued-from` says what it started from; the run's `.gathered-from` is the run's) |
| B5 | R5 |
| B6 | P1 (the record half), R1, R2, R3, R4, R6, R7, R9 (re-scoped 2026-10-08: the viewer draws, end-on down z — show the junction side-on) |
| B7 | C1, C2, C3 (falls with form B), C4, P2 |
| B9 | T, R8 |
| parked | P3 (M3 P4) |

Each build step's commit names the ids it closes.

**S — Structural: the bias sweep is not one run** (the root of most defects).
The contract designs a swept rung as ONE run holding its points and one record
(`transport.md` § 2a.11, *designed, not built*). The code instead makes every
point a stage of its own — `04_device/v0.2/run-N/`, its own launch record, its
own state, its own attempts. Confirmed consequences:

| id | what happens on a real run | evidence |
|---|---|---|
| S1 | a point the walk never reached, and every point of a job cancelled while pending, reads `queued as job N` forever — "queued" is the job's word, given to a point | `submit._go` writes every point's `run.json` at send; `parse/dirs/job.py:408` |
| S2 | launched again, **every** point is re-run, the converged ones too — and until they are, the report's I–V empties (each point's newest attempt is empty) | `submit._plan_chain` 1962–2018; `record.py:428` |
| S3 | the walk's hand-over between points is recorded nowhere: `status` says a point "started from the structure"; `.gathered-from` says `<label>.DM <- 01_seed` though the walk replaced it (and v0's own run rewrote it — `save_density_matrix.F90:138`) | `submit.py:2054`, `runstatus.py:591-595`, `materialize.py:753` |
| S4 | the shared inputs (seed `.DM`, both `.TSHS`) are copied into every point; no clean seed density survives a first launch, so `--cold` is not really cold | `prep.gather_sources` per point |
| S5 | the transmission gathers each point's newest device attempt separately — one I–V is one device run today only because every point advances together; skipping done points (S2's fix) would break it | `prep.transport_inputs` per point (validator: refuted today, real once S2 is fixed) |
| S6 | the scan's stage folder holds a deck, run script and header at the first point that nothing runs | `prep.py:1095-1098` |
| S7 | the launch-again rule for a point is a second copy of `continuation.relaunch` | `submit._plan_chain` 1986–1997 |
| S8 | `--cold` on a scan not yet launched is dropped silently | `_plan_chain` 1977–1981 |
| S9 | a group naming a scan rung is refused with a rollback that does not help | `_plan_group` → `_plan_member` |
| S10 | the walk records no stop: nothing says which points it never reached | the chain log only |

Today's patches on these (the "never-started" point, 2026-10-08) are symptoms of
S and go with it.

**P — Physics.**

| id | finding | evidence |
|---|---|---|
| P1 | **`low-bias` is not linear response.** Each transmission point runs tbtrans on the 0 V device H with `TS.Voltage V`; tbtrans shifts each lead's self-energy by its chemical potential (±V/2) and leaves the device H at 0 V. The I(V) recorded is neither ∫T(E,0)[f_L−f_R] (the contract's definition, § 1.1) nor a self-consistent one, and the record labels it "linear response" | `m_ts_electrode.F90:1461-1462`, `m_tbt_hs.F90:109-112, 287-298`; `record.py` `TREATMENT_NOTE` |
| P2 | the transmission energy window (±2 eV default) is never checked against the bias window (±V/2 + a few kT): above ~3.5 V the current is silently cut | catalogue rows; nothing in `validation/` |
| P3 | the equilibrium contour is the interim 10 eV pole energy, not the stated `contour.eq` with a spectrum gate the contract describes | `transiesta.py:579-585`; `transport.md:1661` vs `:3613` |
| P4 | device points at V≠0 gather the seed `.DM`, which TranSIESTA never reads there (it needs a `.TSDE`) — recorded as an input it is not | `stages.stage_inputs`; `m_new_dm.F90:487-494` |

**C — Composition.**

| id | finding | evidence |
|---|---|---|
| C1 | the electrode swap rewrites the **cited run's** files (and so every other calculation citing it); the contract says the swap is the calculation's own (`swap_electrodes: true`) | `compose.py:618-713`; `transport.md` § 4 |
| C2 | the leads' `.TSHS` name comes from two sources: the device deck uses the template's `system_label`, the lead and the gather use `task.label` — equal on every road today, unchecked | `transiesta.py:509` vs `prep.py:1528`, `stages.py:186-202` |
| C3 | a form-B junction (`.xyz` + sidecar) keeps the sidecar's z kind — a false "transport axis not periodic" deck warning | `compose.py:855-877` |
| C4 | a relaxation with no molbuilder record, or one that ended with an error, composes into a junction — and a bare structure brings no pseudopotentials (`prep._transport_provide_pseudos`: *"put the files in pseudos/ yourself"*) | `compose.py:846-853`; `transport.md` § 3.1 |

**R — Results and pages.**

| id | finding | evidence |
|---|---|---|
| R1 | the record reads TBtrans's files with its own glob — a **second reader** beside the family's (`parse/engines/tbtrans.py`, `transmission_files`, which already knows the channels) — and so a spin-polarised point (`<label>.TBT_UP.AVTRANS_*`) is reported **failed**; G is not (e²/h)(T↑+T↓) | `record.py:453`; `m_tbt_save.F90:2266-2270` |
| R2 | the treatment label is in the T(E) tab only, not beside the I–V | `transport.js` `_fillIV` |
| R3 | "energies relative to E_F" is a constant; the device's NEGF E_F is never compared with the leads' | `record.py:528` |
| R4 | the device facts beside T(E) come from the device's newest attempt, not the run the transmission read | `record._stage_facts` |
| R5 | `status` never shows what a transport rung gathered (every rung "started from the structure") | `runstatus.py:591-595` |
| R6 | the record lacks: the seed's E_F; the contour and pole count (already parsed); the window, points and TBT k-grid | `record._science` |
| R7 | no report until a first `summarize task`; every run of the calculation not listed at the root (both doc'd as built) | `transport.js:688-691`; `results.md` § 2.4 |
| R9 | **the 3D viewer draws nothing on the Results tab in a real browser** (2026-10-08, the user's Chrome, AMD GPU, WebGL context alive, no console error): the transport report's Device view and the plain structure viewer both show an empty scene with "16 atoms" loaded and a full-size canvas; Reset view changes nothing | seen in the browser; cause not yet read |
| R8 | two wrong section links on the transport tab | `transport_calculation.html:210, 240` |

**T — Contract text** (sweep A): 18 internal contradictions, the main ones —
"run" defined three ways (`job-system.md:74`, `project-layout.md:818`,
`transport.md:1284`); the 2026-10-07 rule *any stage launched again however it
ended* vs the 2026-10-05 sweep design *a done point is never run again*; per
point vs one run for what the transmission takes; present-tense passages
describing the unbuilt sweep; stale passages (§ 2a.14's table, § 3.2, § 3.4,
§ 3.6a, § 6a, § 8). *B0 resolved the first three in § 2a.10–2a.12, Words and
rule 3; the passages are B9's.*

### 5x.7 The road walk — 2026-10-08, the junction built on the page, run to its record

*(user: "stop fucking hand bake these fucking things"; "this is a good chance to test the continue/re-run mechanism"; "review the log output and the script generated to detect any potential error/issues that are not picked up by the code/parser")*

**What ran, every step through the page or a printed verb** — the evidence
§ 5x.0 asks for:

| step | door | what happened |
|---|---|---|
| the junction | Molbuilder tab in Chrome | SMILES `SC#CS` → the two H deleted → S–S oriented on z (the ruler picks the anchors) → Slab tab: Au(100), PBE `a`, 2×2×6, registry B above (+z, surface 4.69 Å) / A below → 52 atoms; labels by the Selection filter (`37-52` L-electrode + frozen, `13-28` R-electrode + frozen, the rest bridge); saved as `claude-transport-walk/structure/au-dithioacetylene` (cell 5.88 × 5.88 × 30.17 Å, 2.079 Å across the boundary = the lead's own spacing) |
| the relaxation | Structure optimization tab → Send to Task setup → Task setup (Hierarchical, one stage `coarse` with overrides SZP / k 1,1,1 / 3 steps / 150 SCF) → the printed verbs | `prep task --stage coarse --target this --np 4 --cpus-per-task 1`; `launch task` (direct, foreground): 245 s, 97 SCF rows over two tries (the retry warm), **finished, geometry NO**; the pseudopotentials came from `projects/pseudopotential` |
| the citation | Transport tab → *Choose junction directory* → `…/au-dta-relax/01_coarse/run-0` | the dialog's own verdict: *CONCLUDED (rc=0) · SZP · 150 Ry · GGA/PBE · k 1x1x1 · 52 atoms*; both leads: z-period derived 8.316 Å from 4 layers, the seam CONTINUES the crystal, the principal-layer condition met (orbital reach 6.49 Å < 10.39 Å); the shared panel filled from the cited `.fdf`; charge 0 by rule; 3836 electrons |
| the description | the bias builder 0 / 0.4 / 0.2 → *0.0, 0.2, 0.4*; the switch **no** (self-consistent); *Describe* | `transport/au-dta-t/task.json` + template; `slots.junction` = the run; `bias.low_bias_approximation: false` |
| seed + leads | `prep task --stage seed --stage electrode_L --stage electrode_R --target this --np 4 --cpus-per-task 1` → `launch task …` (one group job) | the three `.psml` **from the cited run folder**; seed 152 s / 68 SCF, each lead 23 s / 14 SCF; `status`: all three *finished, SCF yes* |
| the device sweep | `prep task --stage device` → `04_device/run-0/{v0,v0.2,v0.4}`, the run's clean `.DM` and `.TSHS`, `.gathered-from` once | **launch 1** (run-0): 0 V done (29 NEGF iterations, ~4 s each), killed during 0.2 V at the launching process's 10-min bound — `status`: *failed — 1 of 3 points done; 0.2 V: stopped before its end: no ending in its output and no exit recorded*; the transmission *waiting*, saying which point. **launch 2** (run-1): *takes over 0 V done in run-0; runs 0.2 V (from 0 V's .TSDE), 0.4 V* — 0.2 V done, killed during 0.4 V. **launch 3** (run-2): *takes over 0 V, 0.2 V done in run-1; runs 0.4 V (from 0.2 V's .TSDE)* — finished: *3 of 3 points done, SCF yes*. **launch 4, warm**: refused — *every point of 04_device/run-2 is done … launch it cold*. Every point's `.continued-from` names the run/point it came from (run-1/v0 ← run-0/v0; run-2/v0.2 ← run-1/v0.2; run-2/v0.4 ← run-2/v0.2); the ledger holds each launch's question and decision |
| the transmission | `prep task --stage transmission` → `05_transmission/run-0/{v0,v0.2,v0.4}`, each point's `.TS.HSX` **from `04_device/run-2`**, the leads' `.TSHS` from run-0 | `launch task` (one walk): *3 of 3 points done* |
| the record | `summarize task` | **crashed** — `record.py:494 NameError: att` (B2's `result_folders` rewrite left the old loop name); fixed `8dc08851`; then: G(E_F) 0.0698 / 0.0513 / 0.0457 G₀, I 0 / 1.78e−6 / 2.61e−6 A (total, both channels), `self-consistent`, the provenance (citation, hashes, the chain with device at run-2) |
| the Results tab | the sidebar on `au-dta-t`, *Reload from current project dir* | the ladder *5 rungs, every rung finished* with *3 of 3 points done* on both swept rungs; the report: *self-consistent · 3 points · energies relative to E_F*; T(E) for the three voltages and four eigenchannels; per-rung SCF plots with the tolerance line; *Rungs and provenance* (E_F, E, SCF, attempt); *What each rung took*; the citation with its hashes; the GGA caveat |

**Not run:** `launch --cold` on the finished sweep (the same walker with nothing
taken over); the low-bias and single-bias cases on this junction (B6's
"three cases").

**Found on the road** (each goes to the step that owns it; none is fixed by
a patch here):

| id | where | what | owner |
|---|---|---|---|
| F1 | `jobset status` | the ready device row's *takes …* detail is printed once per point — three times the same three files; `status device` on a sweep says *continued from nothing — it started from the structure*, lists no points and *warm files -* | B5 |
| F2 | the Results report | *Rungs and provenance* shows the device E_F **5.169 eV** beside the leads' **−1.920 eV** with no word — R3's check, unbuilt; *What each rung took* omits the transmission's per-point `.TS.HSX` (it lists the run's `.gathered-from` only) | B6 |
| F3 | the Results report | the Device view draws the junction **end-on, down z** (a 2×2 column reads as a cube, the molecule hidden); R9 re-scoped: not blank — the default must be side-on, the leads told apart | B6 |
| F4 | Transport tab | the § 1 text still offers *`.xyz + .molstruct.json` — a labelled structure* as citable (decision 7 removes it); the § 4 paragraph still says *low-bias … re-converged* instead of the switch's words; the citation dialog's refusal cites **`transport-design.md 4.1b`** (an archived document) | B7 (the rule), B9 (the words) |
| F5 | Molbuilder tab, Selection panel | with two cell notices shown, `.molviewer-filter-rows` (`molview.css:492`: `flex: 1 1 auto; min-height: 0`) collapses to **0 px** — the filter row cannot be seen or typed into; the *Match all* select overlaps the Clear/Invert/All row. A layout-system fix (the panel's height from its content; tokens), never a pinned height (user, 2026-10-08) | B8 (CSS) |
| F6 | Task setup | a new description arrives with a **benchmark grid already filled** (`mpi_np` 4/8/16, `omp_threads` 1/2, *measured · 3 points*) that nobody asked for, and the run card then says *measuring: 4, 8, 16* instead of taking a value; the launch values had to be stated on the command line | B8 (UI fact-check) |
| F7 | Task setup, Structure optimization | the accessibility tree exposes almost none of the page's controls (the add-column/add-setting selects and buttons are unnamed), so a screen reader — or a browser driver — cannot name them; the Structure optimization tab restored a stale 312-atom session on load | B8 (UI) |
| F8 | Transport tab console | `[transport] session restore failed Error: … is not a directory` thrown at `core.js:393` for a stale session — a restore that cannot find its folder should say so on the page, not throw | B8 |
| F9 | `prep` output | a group prep prints the pseudopotential copy lines and the `geometry.h_ratio` warning once **per rung** (three times each) | B8 |
| F10 | the citation dialog | every confirmation (*z-period DERIVED*, *seam CONTINUES*, *principal-layer condition MEASURED*) is prefixed ⚠ — the mark of a warning on a fact that is good news | B8 (UI) |
| F11 | `launch` in direct mode | the verb blocks until the walk ends (Popen + wait in the verb's own process group, which is what lets a bound on the verb reach the walk — C4, 2026-10-08); killed at a bound it leaves no orphan processes, the monitor's closing record says *stopped before its end*, and the next warm launch takes the done points over — the road handled it. **The wrapper's own kill line never reached its session log** (it went through the tee the kill took): **fixed** — the handler appends it directly | done |
| F12 | the junction's cell (**the reviews' blocker**, 2026-10-08) | `c` = 30.17 Å = the atoms' span, so the two leads' end layers meet through the periodic image (0 room; 8 Au per 2×2 cell in one plane, 2.079 Å apart): 19–23 eV/Å on every frozen atom and 587 kbar in the relaxation and the seed (vs −55 kbar in the lead's own cell), the Hartree reference plane on the fused layer — **no gate said a word** (the kind gate refused only room above 1.5 spacings; the Cell page's *collision* notice was on the page and ignored). **Fixed**: `sidecar.check_junction_boundary` — one rule both ways, a refusal on every transport rung, a warning at a labelled junction's relaxation, naming *set c to 32.249 Å*; § 6.1c's row says both; the case is in the kind gate's table. **The walk's junction is rebuilt with its cell closed** (the Cell page), relaxed and run again — its numbers above are not a result | done (gate); the redo next |
| F13 | the record / Results | the device E_F (**5.169 eV**) beside the leads' (**−1.920 eV**) is a reference convention, not a mismatch: TranSIESTA zeroes the Hartree potential on the boundary plane (`ts-Vha −7.241 eV` shifts the seed's −2.072 to +5.169) and fixes E_F, letting the charge float; the leads report in their own cells' frame; the lead Hamiltonians are aligned to μ = E_F ± V/2 inside. The report must **say the frames** (the row's `vha_ev` is stored) — nothing printed tests the physical alignment (device electrode-region potential vs bulk) | B6 (R3) |
| F14 | the parsers | the unconstrained max force and the pressure of a periodic slab are never shown (the monitor and wrapper print the constrained max — the 23 eV/Å on the frozen atoms stayed invisible); *ALL FORCES AFTER TRANSIESTA ARE WRONG* is not captured while `MESSAGES` carries only the harmless Γ-point note (TranSIESTA's *Removed N elements which connect electrodes across the device region* is **not** a defect to capture: the source's comment is "remove all electrode to other side connections ... that cross the boundary" — the leads' coupling through the periodic image, present in every closed cell: 10 880 in the closed walk, 12 068 in the fused one); the SCF table's `**********` overflow row passes silently; the hand-over geometry is `.XV` = `STRUCT_NEXT_ITER`, a CG interpolation never evaluated, and nothing says so; the monitor's final status offers `max_force` for a device run | B6 (the device's forces, the E_F frames); M3 (the slab's forces and pressure, the engine warnings, the overflow row); B7 (the citation says whether its geometry was evaluated) |
| F15 | the take-over copy (`_plan_sweep`) | a point taken over carries **every** file of the done point: the old run's session, monitor and util logs, `calcdir.json`, `.gathered-from` (old mtimes), the fdf log — and the scratch `.TSGFL`/`.TSGFR` (86–217 MB each; run-0 590 MB → run-2 1017 MB); the checkpoint before the transmission prep stored 2.7 GB, 2.6 GB of it TSGF; `.gitignore`'s binary classes name `*.DM *.HSX *.TBT.AVTRANS_* *.TBT.CC *.TBT.DOS *.TSHS` only, so 95 `.TSDE`/`.TBT.nc`/`.ion*` files went into git. **Decision owed**: what a taken-over point carries — its inputs, the engine results readers and rungs use, and the run's own account (`.out`, `.concluded`, `0_NORMAL_EXIT`, fdf log, timing); not the scratch self-energies; its `.continued-from` naming the folder that computed it (one hop); and the checkpoint's binary classes completed from the run-file catalogue | B7 (one rule: the catalogue says which files a point's result is; the copy and the checkpoint read it) |
| F16 | the decks | `SaveHS` (`write_hs`, a plain output preference with no `role`) is what writes the device's `.TS.HSX` the transmission reads — the lead's deck note says the opposite; `TBT.Elecs.Eta 0.001 eV` is written on the transmission while the device relies on the TranSIESTA default `TS.Elecs.Eta` (one value, two sources); the contour note describes circle + poles while the run uses the continued fraction (10 eV = 123 poles) — P3; the transmission deck's SIESTA keywords tbtrans never queries (the § 6.1b audit: it read `SystemLabel`, `SystemName`, `ElectronicTemperature`, `Spin`, `TBT.*`, `TS.Elec*`, `TS.ChemPot*`, `TS.Voltage`, `TS.Elecs.Bulk`) | B7 (`write_hs` a role item on the device; `TS.Elecs.Eta` with § 5u step 7); B9 (the notes; § 6.1b's list) |
| F17 | the ledger, `run.json` | a sweep's run-1/run-2 recorded `continued_from: null`; the three device launches were three identical *question* + *launched* pairs with no word of the run opened, the points taken over or walked — **fixed**: the run's `.continued-from` names the run it took points from; the `launched` entry carries each member's run and its walk | done |
| F18 | the docs | `project-layout.md` § 1.5 (*continues from its own last density*, `launch --skip`, *a point recorded done is never rewritten*) and `transport.md` :1628, :1763 (*continuing the point's own cycle*) restate the rule § 2a.11 settled the other way (*every point not done is redone from the same start*) | B9 |
| F19 | the walk log, the wrapper logs | the sweep's log uses the benchmark's words (*start trials=3*, *run_trial*); a retry's `exec` inherits the first try's tee so run0's session log holds run1's session too; the effective-parameters banner shows catalogue defaults (`bias_voltage_v 0.0` in the 0.2 V point's log); the tbtrans wrapper says *Retry policy: up to 1 retry on non-convergence* | B8 |
| F20 | this junction's physics (not code) | the relaxation's 2×3 CG moves were two line searches with a 0.05 Å cap (C≡C 1.20 → 1.37 → 1.31 Å, the S slid, the energy rose); the second SCF iteration after every move overflows (`dHmax **********`) and recovers in ~10 iterations; Γ-only sampling puts a folded lead band edge within ~10–30 meV of E_F (bulk T(E_F) 4.83 drifting over 20 meV); 150 Ry is low for semicore Au. A real calculation needs the closed cell, a converged relaxation on an evaluated geometry, transverse k (≥ 6×6, electrode ≥ device), ≥ 300 Ry | the walk's redo keeps the cheap settings (a mechanism test); the rest is the user's physics |

**Judged on the road, and by the three read-only reviews of its decks, logs
and wrappers (2026-10-08):** rule 3 at the point (§ 2a.11) holds in the code as
written — take-over, the same start for a point not done, the refusal; every
stage prepped, launched and gathered the right files; the kills were
detected and the endings unified; the currents parsed with the spin factor.
As a junction result it is **not trustworthy** (F12): the cell was fused. The
test subject is the mechanism, not the physics (SZP, Γ, a 3-step
relaxation): the numbers are not a result.

**The closed-cell walk (2026-10-08, `au-dta-junction` → `au-dta-relax2` →
`au-dta-t2`; the sweep case).** The junction rebuilt on the Cell page (*Span +
gap*, c = 32.249 Å); the new gate seen both ways on the road — the fused cell
refused on the transport rungs and warned at the relaxation, naming the c to
set. The relaxation (run-0 killed, run-1 finished, geometry **NO** at 1.878
eV/Å — loose on purpose), the citation, seed + leads, then the device sweep in
**three bounded launches**: run-0 and run-1 killed at 0.4 V (a point not done
restarts from the same start, by design — this one alone needs ~9 min, ~10 s a
NEGF step), run-2 *takes over 0 V, 0.2 V done in run-1; runs 0.4 V (from
0.2 V's .TSDE)* and finished, **3 of 3 points, SCF yes**; the transmission
run-0 at its three points (~2.5 min each, 4 ranks); `summarize task`; the
Results tab. The record: G(E_F) 0.083 / 0.024 / 0.021 G0, I 0 / 8.64e-7 /
1.22e-6 A (both channels, the printed one doubled); ts-Vha +0.257 eV at 0 V,
the device's −2.803 eV is −2.547 eV in the seed's frame against the seed's
−2.542 eV — one sentence under the rungs, printed by `summarize`. The Results
tab: *5 rungs, every rung finished*, the Device **side-on**, the I–V tab with
its treatment beside the table. **Read in the logs and decks** (the user's
ask): the tbtrans deck's window −2 … 2 eV, 401 points, `TBT.Elecs.Eta` 0.001
eV, `TS.Voltage` and the chemical potentials per point, the Γ `TBT.k`; the
*ensure entire Fermi function window* line is met at 0.4 V (±0.2 eV + 5 kT ≪
2 eV); the device logs carry no *FORCES WRONG*; the *Removed N elements* line
is the leads' coupling through the image, in every closed cell (F12, F14).

**Found on this walk:** **F21** the Results report's camera — the view
context put back the **end-on** pose this card had persisted before the
side-on view existed (its restore lands after the report's camera action; a
switch flip or the post-restore write persists whatever pose is current);
**fixed** as the viewer's home view (`lookAlong` — the load's fit and Reset
look along it; a camera action during a rebuild is held, `molview.md` § 10.9),
so Reset gives the side-on and the lane records it · **F22** a sweep's walk
script lives in `<stage>/launch/` and `run.json`'s command is relative to the
stage folder, while `status` prints *Directory: <stage>/run-0* — no document
says where the walk script is (B9) · **F23** a transmission point's folder
opens nothing — no parser claims TBtrans's output, so a point picked off the
root's ladder has no *open* (its result is the root's T(E) and I–V): either a
reader claims the file or `results.md` § 0.1 says a transmission point is read
at the root (B9) · **F24** the Transport tab's step 4 copy still says *several
values are a **scan** … low-bias converges the device once … **re-converged**
runs the device at every voltage* — the record's words are
*low-bias-approximation* / *self-consistent* and the design's word is *sweep*
(B9's sweep of "scan" reaches the tab's copy; B8's fact-check) · **F25** the
group launch prints *warn [structure.regions]: this structure carries region
label(s) ['SC#CS#'], which the transport ladder does NOT consume* — the
molecule's own label from the Molbuilder tab; molbuilder reads only the labels
it owns and never warns about another (user, 2026-10-02) — the warning
(`sidecar.check_unconsumed_region_labels`) and the two sentences that order it
(`validation.md` § 5 pattern B; `transport.md` § 4 *"a label this engine does
not consume is WARNED about"*) go (B7, B9; **approved**, user 2026-10-08) ·
**F26** the suite's runner: `tests/validation/test_siesta.py` run in one
batch with `tests/test_transport_sort.py` and `tests/test_vibration_render_gate.py`
under 8 workers loses every fixture of `tests/validation/conftest.py`
(*fixture 'water_struct' not found*, 11 errors; the same at `924d68c7`
with nothing changed; the pair `test_pyscf.py + test_siesta.py` passes) —
**diagnosed and fixed** (B8 round 1, `8fab088a`): pytest binds a
subdirectory's conftest to the first collector of that directory and a later
bare file argument from the parent re-collects it; the runner hands a targeted
set to pytest ordered by directory (`testing.md` § 6.1a rule 3) · **F27** — **fixed** (B8 batch 4: `siesta/layout.unit_word`, one source, the catalogue's `unit`) — the deck's unit words were a hand table beside
the catalogue's own `unit` (`siesta/layout.py` `_UNIT`, with its widths and
formats): an item added with `unit = "eV"` rendered `TS.Elecs.Eta 0.001`
bare, which TranSIESTA reads in rydberg — 13.6× — and nothing said so until
the deck was read (2026-10-08); one source (the catalogue's `unit`, the
bias's V→eV spelling the one declared exception) — B8 · a single
click on a record in the
sidebar does **not** switch the report — `results.md` § 3b's rule (a click is a
preview; the panel's dropdown chooses), not a defect.

**The low-bias case (`au-dta-lb`, 2026-10-08).** Described on the Transport
tab (the same citation; the builder 0 → 0.4 step 0.2 → *0.0, 0.2, 0.4*; the
switch **yes**; a new folder from the sidebar; *Describe*), then the printed
verbs: seed + leads as one job, the device once at 0 V (one run, no point
folders), the transmission, `summarize task`. **Found and fixed on the road
(`fc413816`)**: the record's integral called `np.trapz`, gone from this
numpy, so the first `summarize` crashed and the Results route answered an
HTML fault the inspector reported as a JSON syntax error; the I–V tab drew
only the measured 0 V point. The record: G(E_F) 0.083 G0 (the sweep's 0 V,
the same device run and numbers), I(V) computed from T(E, 0): 2.34 µA at
0.2 V and 5.01 µA at 0.4 V against the self-consistent sweep's 0.86 µA and
1.22 µA — the approximation's overestimate, with T(E_F) falling to 0.024 G0
under 0.2 V of bias; the Results tab opened the report through the
description before any summarize (R7 on a fresh calculation), the I–V tab
names the treatment and plots the computed curve dashed.

**The single-bias case (`au-dta-sb`, 2026-10-08).** The bias list *0.2* alone
is refused at *Describe* — *the bias list must start at 0.0 … a scan starts
from equilibrium* — so a single bias is the one equilibrium slice, *0.0*;
described, walked through the same verbs (device once at 0 V, the
transmission), summarized: the record says *single-bias*, G(E_F) 0.083 G0,
no I–V beyond the one point. The three cases of § 5x.0 have now run on the
road junction.

**Not run on this walk:** `launch --cold`; no point was killed, so F11's kill
line (`6d4ea7c4`) is still unseen.

### 5u.0 What is true today — measured

* **The road is the ordinary one.** Five rungs (seed, electrode_L, electrode_R,
  device, transmission), each prepped by `molbuilder jobset prep task --stage
  <stage>` (or the Task setup tab — one entry, `prep_stage`) and launched by
  `molbuilder jobset launch task --stage <stage>` (`_KINDS = ("task", "bench")`,
  `jobset/_cli.py:768`); every rung renders through `spec_for` → `prepare_deck`
  with its `.validation.txt`; `summarize task` writes `<label>.transport.json`.
  A swept rung is one run holding its points (§ 5x.1, § 5x.3).
* *No ladder on this machine has concluded a device rung* — **superseded
  2026-10-08**: the road junction `claude-transport-walk/transport/au-dta-t`
  ran every rung to its record, the device as a three-point sweep (§ 5x.7).
* *The Results surface … not drawn: an I–V curve, DOS, eigenchannels, the chain*
  — **superseded**: `web/results.md` § 2.5 (:440-463) built 2026-10-07 (step 9
  ①②); what is left to draw is § 5x B6.
* **What each program reads is settled in the engine's source** (SIESTA 5.4.2
  and its `libfdf`): `siesta` reads no `TBT.*`; `tbtrans` reads the `TS.*`
  junction description and takes several `TS.*` values as its defaults; a
  `TBT.k` line is read only as a bracketed list or a block — the deck's bare
  triple is skipped (`engines/transport.md` § 6.1b, :3383-3390). Step 7 is
  checked against this source the same way.

### 5u.3 What the documents say that the code contradicts — the residue, § 5x B9's input

Re-read 2026-10-08. `engines/transport.md`: § 8 :3898 (*"one panel per
ENGINE since 2026-09-15"* — not built; step 8) and :3901 (*"the tab's live
routes are four"* — six: `web/blueprints/transport.py:92,284,329,365,413,616`);
§ 6 :3043 (`parse_fdf_params` in `transport/preflight.py` — the module is gone;
the reader is `parse/fdf.py:196`); § 2a.13 :1635 and `science/overview.md:212`
(the kind gate refuses a net charge — the family is `check_electronic_state`,
`validation/chemistry.py:234`); § 0.5 (the user's `AuBDTAu-CT` holds a deck
each and its `.validation.txt` — it holds no template; on disk the folders are
`AuBDTAu` and `Au-BDT-Au.old`, not re-read). Routed elsewhere, not B9's:
§ 2a.12's report *"composed on read"* from `.gathered-from` → § 5x B6 (R7);
§ 2a.13's stated equilibrium contour and its gate → § 5x's parked P3 (M3 P4).

### 5u.4 The stale citations — § 5x B9's input

**~22 citations of `transport-design.md` remain in live files** (2026-10-08;
54 on 2026-09-29) — a document that exists only as
`archive/2026-09-01-transport-design.md`: `task.py` ×6, `transport/compose.py`
×3, `transport/sort.py` ×2, `transport/wizard.py:72`,
`siesta/warm-files.toml:139`, `lib/transport/core.js` ×2,
`lib/task-handover.js` ×3, `execution/job-contracts.md:2132,2135,2322`,
`execution/architecture.md:105`, `web/tabs.md:223` — and the citation dialog's
refusal shows one on the page (§ 5x.7 F4, *`transport-design.md 4.1b`*). Where
each cited section lives now: § 4.1 → `engines/transport.md` §§ 1, 3, 3.1;
§ 4.3 → § 2a.10, § 2a.11; § 4.1b → § 3.1; § 4.1a → § 4 and `model/overview.md`
§ 2.2; § 4.2 → §§ 1, 6.1, 2a.11, 2a.15; § 3 → § 5 (I10, I11), § 7.1; § 7 → § 8,
§ 2a.12. Also stale: `transport.md` section numbers written into decks
(*"4.2"*, *"3.3"* as the parameter inventory); the missing walkthrough
`execution/walkthrough-2026-09-15-junction.md`, cited at `transport.md:2808`
and by `junction-cell.md`; the five other missing documents named 2026-09-29
(`structure-info-plan.md`, `protocols/runtime-registry.md`,
`spectra-migration-plan.md`, `cell-plan.md`, and the bare
`staged-runs-implementation-plan.md` in `generator.md`, linked to the archive
2026-09-29) — not re-checked 2026-10-08.

### 5u.5 What this absorbs — read here, not acted on separately

W30 → ③'s remainder is step 10, ④ is step 9 (§ 5x B6), ⑤ is step 8 (①, ②
built) · W25 → step 7 (its headline, the four dead scalars, done 2026-09-15) ·
W24 → step 8 · W10 → step 9 (§ 5x B6) · W32 → step 11 · § 5c.3 → (b)–(d)
built, (e)(f) step 9 (§ 5x B6), (a) superseded by `runs.folder_answer`'s
`place` · § 5o.6 → step 7 · § 5p.4 → TR9 step 5 (built 2026-10-07; its *Left*),
TR10 step 9 (§ 5x B6) · § 5p.3p → its step 10a stays with the route-catalogue
sweep (§ 0a's *Unscheduled*) · X4 ⑤ (`structure_hash`) → with V1.9 before
M2m · § 5q.3's transport citation viewer → § 5q.6 P4 · X4 ② (whether a lead
keeps `info`, annotations and identity) — the user's, not yet asked · X4 ④
(every transport deck told its labels are *not consumed*) → step 7 · § 5o.6's
open rows → step 7 · A1.12 → step 7 · § 5s.4's unpersisted shared panel → step
8 · N10's two predicates for a calculation root → M3 P2 (§ 5t.5 ②) · E13 → the
workstation half was step 4 (done); the Sol half after it · S13 (the
convergence sweep) → after step 4, unscheduled · X3 (the Cell page's seam
verdict) → unscheduled · E14 (two uncited references) → the document sweep
(§ 5x B9). *(The pointers to steps 1–4 and 6 are in the archive.)*
