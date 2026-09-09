# The paths framework, M1–M8 — CLOSED 2026-09-08

**Role:** record
**Superseded by:** [`plans/plan.md` § 5l](?doc=plans/plan.md) — the STANDARD.
**Contract that survived:** [`execution/project-layout.md` § 4.5](?doc=execution/project-layout.md).

This is `plans/plan.md` § 5k as it stood when the work it describes shipped. It
is here for one reason: to recover *why* a decision was made, never to decide
what is open now. Everything still open moved to § 5l.

**What shipped.** Twenty-four handcrafted path searches went to zero and eight
hand-built counter-keyed names went to zero, both guarded by
`tests/test_path_framework.py` over `tools/classify_path_finders.py`. Commits
`6822f639` (M7, the search half) and `981534b9` (M8, the compose half plus three
defects found by re-reading M7's own diff).

**Why its METHOD was superseded the same day.** It closed each asymmetry by
adding a door for whatever question a call site happened to ask. After M1–M8 the
framework was ~40 public functions answering four questions four ways apiece,
five of those doors added on the final day to fit individual callers. § 5l
reverses the direction of fit: a use that does not match the standard is a use
that changes.

---

## 5k. The paths framework — COMPOSE has homes, FIND has none

*(Opened 2026-09-08, from the jobset round. Replaces the loose "P2–P5" note.
Contract when written: [`job-contracts.md` § 2.2a](?doc=execution/job-contracts.md)
owns the run-file grammar and [`project-layout.md` § 1](?doc=execution/project-layout.md)
the directory layout; this section is the plan, not a second contract.)*

### 5k.1 The measured problem

Every name molbuilder writes has a door that COMPOSES it. Almost nothing has a
door that FINDS it. A caller holding a bundle and a question — *is there a
sweep in here? which attempts exist? which run wrote this?* — has nowhere to
ask, so it spells a glob, and the layout rule gains another site.

**M1 IS DONE AND IT IS A TOOL, not the table that used to stand here** —
`tools/classify_path_finders.py`. Run it; do not quote the figures below,
which are its output on 2026-09-08 and are here to say what it found, not to
be cited later.

    python tools/classify_path_finders.py            # the summary
    python tools/classify_path_finders.py --list owned

| bucket | n | what it means |
|---|---|---|
| **owned — a door composes the name, none finds it** | **22** | the migration's subject |
| owned — workspace store (a separate grammar) | 2 | same fault, different vocabulary |
| door — a finder, not a caller | 3 | `sweep_set_paths` ×2, `named_environments` — what there should be more of |
| foreign — not a name we compose | 18 | conda-meta, `*.psml`, `*.md` |
| exact — a single named file | 10 | no pattern to own |
| listing — no pattern | 17 | `iterdir`/`scandir` |
| **unclassified** | **0** | every site was read; the twelve the syntactic pass could not settle carry a recorded reason |

**Every override is keyed by `(file, function, pattern)` and never by line** —
`classify_source_reads.py` keyed its reasons by line number and two came
unanchored the first time tests were deleted around them (§ 5h).

Three findings the survey produced that the hand table had not:

1. **The `-run<N>` glob is spelled twice** — `materialize.py:576` and
   `summarize.py:63`, both `f"{basename}-run*.{suffix}"`. `runfiles` exists so
   that grammar has one home, and both readers bypass it because it offers no
   reader. This is M3's first target.
2. **`validation/identity.py:273` is the gap in one function.** Its suffixes
   already come from the one rules file (`_engine_inventory` → `_warm_inventory`,
   U3) — the VOCABULARY asks, and only the SEARCH is spelled by hand.
3. **The first pass under-counted by four** because our names are usually built
   from a constant (`_JS`, `SUFFIX`, `{label}`) rather than written out. A
   survey that misses a name because the caller imported its constant reports
   the code as tidier than it is, and does so exactly where it is least tidy:
   a caller that knows the filename has a home and still assembles the search
   by hand. The tool resolves module-level constants now.

### 5k.2 What the jobset round established, and why it belongs here

Four findings from 2026-09-08, each one a caller of the framework-to-be:

1. **A verb must be finished by the verb.** `prep_calculation` placed a TRIAL
   through `trial_work_dir` (which knows the attempt layer) and a LADDER RUNG
   through a bare `stage_dir` (which does not), so a rung came out half-prepped
   and `launch` refused it. Fixed in `c23a743d`. **For the framework:** the
   attempt is not a side effect of one caller — it is part of the answer to
   *where does this job's files go*, and the framework must expose that as one
   question with one answer, not as `stage_dir` plus a step someone remembers.

2. **The contract and the code disagreed about which verb owns the attempt**
   (`project-layout.md` said "launch adds attempts"; `_launch_dir` refuses
   without one). Corrected in the same commit. **For the framework:** the
   layout document and the path doors must be checkable against each other,
   or a caller implementing the document produces something the code refuses.

3. **The browser is a caller, not an exception.** The bench row composed
   `bench-<token>/` client-side and was wrong in every layout (`6e20cc4b`,
   `cdd5ace1`). `viewer.js:2019` still re-implements `identity.stage_token`
   as `String(n).padStart(2,"0") + "_" + name` to build a `--from` argument;
   it agrees today and has no reason to keep agreeing. **For the framework:**
   the browser cannot import Python, so the server must SEND every path the
   page shows — the framework's web obligation is an endpoint contract, not
   a JS port.

4. **The finder gap surfaced twice in one function in one day** — `rglob` over
   the whole tree, then hand-spelled globs — before landing on
   `sweep_set_paths` beside its namer (`ef079932`).

### 5k.2a The design, agreed 2026-09-08

**One module, `molbuilder/paths.py`, L1 and stdlib-only.** The contract is
[`project-layout.md` § 4.5](?doc=execution/project-layout.md) — written there
because that document already owns the tree, the shapes and the attempt rule,
and because this week proved it is the document callers actually implement
(`prep` followed its "launch adds attempts" line and produced a folder the
launcher refused). The run-file GRAMMAR stays in `job-contracts.md` § 2.2a.

**What made one module possible.** `Shape` is floor 2 for a single import —
`from ..task import SHAPES`, where `SHAPES = ("flat", "hierarchical")` — and
that constant is imported from `task` by exactly one module, `shape.py`
itself. Moving it into `paths` and having `task` import it back inverts one
line and makes the whole naming-and-layout surface stdlib-only, which is the
property the monitor needs (`MONITOR_COMPANIONS` already ships `config_dir.py`
on the same grounds).

**What it absorbs:** `jobset/shape.py` whole. **What it leaves alone:**
`runfiles` (the filename grammar, already correct and already L1, imported by
`paths`), `identity.stage_token`, `task` (the description), and
`materialize`'s JobSet-level assembly, which delegates.

### 5k.3 The rule the framework adds

> **For every name it composes, it owns the search.** A door that can build
> `<NN>_<stage>/bench` must answer *where are the bench containers in this
> bundle* without the caller spelling a pattern.

Two constraints carry over and are not negotiable:

- **L1, stdlib-only.** `runfiles` already is, and that is load-bearing: the
  monitor ships beside a job (`runwrap.MONITOR_COMPANIONS`) and runs under the
  JOB's python with no molbuilder installed. A finder that imports `task` or
  `Shape` cannot go in the same module — which is why `Shape` (L2, it reads
  `task.SHAPES`) and `runfiles` (L1) are separate today and must stay so.
- **The layout is DECLARED, never inferred.** `flat` vs `hierarchical` comes
  from the description; a finder that guesses from what it sees on disk would
  re-introduce the drift the shapes exist to prevent.

### 5k.4 Migration — each step separately green

**NOTHING BELOW IS BUILT.** No framework module exists — there is no
`paths.py`, no `locator.py`; the `paths-framework` branch is an empty
placeholder pointing at an old `main`. `runfiles.py` (`3dfa76c9`) and the
composers in § 5k.1 predate this section and are its INPUTS, not its output.
`materialize.sweep_set_paths` was added 2026-09-08 while fixing a `status`
message — 27 lines, mostly docstring — and it is an EXAMPLE of the rule in
§ 5k.3, not a step of this migration. Recorded that way because listing it as
"M1, done" read as though the framework had started.

| step | what | done when |
|---|---|---|
| **M1** | *(done 2026-09-08)* The inventory, as a re-runnable tool: `tools/classify_path_finders.py`. 72 searches, 24 owned, 0 unclassified | shipped |
| **M2** | *(done 2026-09-08)* The design + contract: one L1 stdlib-only `paths.py`, contract at `project-layout.md` § 4.5 | agreed and written |
| **M3** | *(done 2026-09-08)* `runfiles.find` + `runfiles.latest_run`. Four hand-rolled readers of the `-run<N>` counter retired: the globs in `materialize` and `summarize`, and the index regexes in `summarize` and `parse/engines/pyscf.py` | shipped; survey's owned bucket 22 → 20 |
| **M4** | *(done 2026-09-08)* **a** — `Shape` moves to a floor-1 `paths.py` (`SHAPES` with it), plus `TRIAL_PREFIX` / `trials_in` / `launched_trials`. **b** — the ATTEMPT gets a composer at last: `attempt_name` / `attempt_dir` / `attempt_index` / `attempts_in` retire nine `f"run-{n}"` spellings across four modules and the regex two of them reached across for | shipped; owned 24 → 19 |
| **M5** | *(done 2026-09-08)* `/api/task-setup/attempts` answers from the DECLARED shape; `runsForStages` and the `--from` builder stop composing. Four browser re-implementations gone — `stage_token`, `/^run-\d+$/`, the flat `_<token>-run` form, and a shape inferred from disk | shipped |
| **M6** | *(done 2026-09-08)* `runfiles.find_by_role` — the label-less half, for the caller that has a folder and no label. `parse/contract.py`'s `*.fdf` and `*.molwatch.log` ask it | shipped; owned 19 → 17 |
| **M7** | *(done 2026-09-08)* the rest, each site read before it moved; three doors added where the door could not answer; the survey's `owned` bucket **empty**, guarded by `tests/test_path_framework.py` | shipped; owned 17 → 0 |

**M4b's number moved while it was being fixed.** The plan said eight
`f"run-{n}"` spellings; re-derived at the start of the work it was NINE, and
one of the nine had been added the same day by M4a — the commit that gave the
TRIAL directory a home reached for an f-string because the ATTEMPT still had
none. That is the argument for the framework in a single diff: a rule with no
door grows a new caller even while you are closing its old ones.

**Still spelling `-run<N>` after M3:** `siesta/makov_payne.py`, and it is the
interesting one — that code is SHIPPED beside a job
(`materialize` stages it with `MONITOR_COMPANIONS`), so it cannot import
`runfiles` unless `runfiles` ships too. It is stdlib-only precisely so it
can; adding it to the companion list is the step that closes the last one,
and it belongs after M4 rather than squeezed into M3.

**M3 found two properties nobody was testing**, both load-bearing in the code
it replaced and both green under mutation until tests were written for them:
`latest_run` must be the MAXIMUM (the minimum reads an earlier attempt's
goodbye as the latest word — what `attempt_concluded`'s own comment forbids),
and `find` must sort the counterless name FIRST (last, and the PySCF
fallback beats every real attempt — the 2026-08-13 bug its comment records).
Moving a rule into a shared door does not test it; it widens who depends on
it, which is the argument for writing the test at the same commit.

**Do M1 and M2 before any code.** § 5a's rule applies to this section as much as any other:
the table above was measured on 2026-09-08 and will be wrong by the time it is
acted on.

### 5k.4a A SECOND POPULATION, found doing M5 *(user, 2026-09-08)*

**~40 tests slice a source file by TEXT ANCHOR and then run the slice** —
`src.index("        let from = \"\";")` and its like, across ~20 files. They
look sound because what they do with the slice is EXECUTE it, and
`classify_source_reads.py` agrees: its `_RAN` rule credits them as *already
runs the code*, so § 5h has never counted them.

**The anchor is the pin.** M5 edited the `--from` block to stop composing a
token; the slice stopped matching, and the test died with `[eval]:17` —
telling nobody whether the page still worked. A test that cannot survive an
edit to the lines it quotes is measuring spelling, whatever it does next.

Two were converted in M5, to Playwright tests that read the command block a
person actually copies. The rest are open, and they are NOT § 5h's backlog:
that section counts assertions over a file's text, and these pass its filter.
**The instrument needs the rule before the population can be counted** — a
slice taken by literal offset is a pin even when the slice is executed.

### 5k.4c What M7 did, and the three doors it had to add

**Owned went 17 → 0**, and the finished state is a check rather than a count:
`tools/classify_path_finders.py --check` fails on any `owned` or
`unclassified` site and on any exemption that no longer matches one;
`tests/test_path_framework.py` is that check as an assertion, and
`tests/test_path_framework_doors.py` asserts what each migrated caller now
ANSWERS. Re-run the tool rather than trusting this paragraph (§ 5a).

**§ 5k.4b's table was right that the risk is not in the mechanical ones — and
wrong about three of the four rows.** Re-derived at the start of the work:

| § 5k.4b predicted | what reading it showed |
|---|---|
| `identity.py` `*{suffix}` needs `find_by_role` to take an unchecked role | **it needs no door.** The suffixes are the ENGINE's warm-restart vocabulary, which `WRITTEN` deliberately excludes, so § 4.5 — *for every name **it composes*** — does not reach it. `find_by_role`'s refusal is correct and stays |
| `identity.py` `{label}*` is a leftover to migrate | **it IS the door.** `warm_files_present` is the § 4.2 subtraction — everything named after the label that `is_ours` does not claim — and `check_id_change` and `runstatus` both ask it |
| `workspace_storage.py` ×2 need their own namer first | **they already have one.** `_state_path` composes and `_state_indices` finds, both from the one `_STATE_SUFFIX`, in the same module; nothing outside it spells `.wc.json` (the four hits in `workspace/dispatcher.js` are prose). A self-contained grammar already obeying § 4.5 |
| `template.py` `*.template.toml` — the door belongs beside `template` | **no: `.template.toml` is a row in `runfiles.WRITTEN`.** A role the catalogue declares is found by the catalogue; a second finder in `template` would be the two-readers fault § 4.5 exists to prevent |

**Three doors were added, and each because a caller genuinely could not ask:**

1. **`runfiles.role_matches`** — `.runwrap-*.log` is the one `WRITTEN` row that
   is a FAMILY, not a name (one file per launch, stamped with the clock), and
   `find` compared roles by equality. `summarize._wrapper_log` kept its own
   glob because the door could not answer, not because nobody had looked.
2. **`sidecars.molstruct.SUFFIX` / `sidecars_in` / `is_sidecar`** — the sidecar
   had a composer (`sidecar_path_for`) and no public suffix and no finder, so
   `transport/compose` carried four copies of the literal.
3. **`runfiles.find`'s `is_file()` filter** — its docstring said *our FILES*
   and it checked nothing, while `find_by_role`, written later, did. Found by
   reading the module end to end, not by a failing test.

**Two sites were re-declarations rather than searches, and were fixed as such:**
`parse/contract._declared_in_provenance` globbed `.run.sh` / `.fdf` / `.py` —
three catalogue rows spelled as patterns, the § 4.2a fault R1 closed elsewhere
in the same module — and `projects._GEOM_OUTPUT_PATTERNS` was a FOURTH copy of
`_geom_optim.xyz`, the spelling `pyscf/input.py`'s own comment says *"came to
have six spellings."* Both now ask their home; the picker keeps the CURATION
(which engine outputs are a startable geometry — the rules file has no field
for that) and asks only for the spellings.

**Where the answer changed, it changed on purpose.** Three of them:

- `runstatus._stage_state` is now label- and stage-scoped through the grammar,
  and **asks for the role FAMILIES `*.out` / `*.log` rather than the exact
  roles.** Narrowing to `.out` and `.log` loses `.pyscf.log` — the catalogue
  says PySCF *"writes here and not to `.out`"* — so a finished PySCF rung would
  have answered **queued**, § 1.6's one forbidden line. Caught by reading the
  catalogue, held by a test, and confirmed by mutation.
- `summarize._wrapper_log` returns `None` where it returned
  `<basename>.runwrap-none.log` — a composed name for a file that cannot
  exist, the same handcraft pointing the other way.
- `transport/compose`'s `.XV` / `.xyz` globs gained `is_file()`, matching what
  `find_by_role` guarantees for the roles beside them.

**Nine mutations, nine killed.** Including two on the guard itself: an override
allowed to name a failing verdict (the back door), and `FAILING_VERDICTS`
emptied — which leaves the classifier working perfectly and every assertion
green. That one **survived** until the mutation test was made to assert that
the verdict it recognises is one the guard fails on.

**`tests/` is out of the survey's scope, and that is a design position, not a
deferral.** 193 path searches live there, 73 of them spelling one of our names.
A test that located our files through the door would be asserting that the door
agrees with itself; the literal in a test IS the independent check on the
grammar. What is NOT settled is the subset that globs merely to LOCATE a file
it then reads — those gain nothing from the literal and can go stale silently.
Separating the two is a review, not a sweep, and it is open.

### 5k.4d M8 — the COMPOSE half, found by reviewing M7's own diff

**A duplicate composer performs no search, so M7's guard was blind to it.** Two
were live:

| site | what it was |
|---|---|
| `materialize.attempt_concluded` | `f"{basename}-run{newest}.concluded"`, on the line AFTER asking `runfiles.latest_run` for that counter |
| `submit.py:1940` | `f"{_names[_j.name]}/run-{_ns[-1]}"` handed to `prepare_attempt` as `continue_from` — a real path on the live continue-a-run route |

Six more were `run-<n>` in user-facing messages, which would have started lying
the day the prefix changed. All eight ask a composer now, and
`classify_path_finders.compositions()` keeps it that way — narrow by design
(`project-layout.md` § 4.5's compose half explains why 35 role-suffix matches
would be the wrong instrument).

**And re-reading M7's diff found three defects in it**, none of which any test
had caught:

1. **`summarize._read_system` raised on a missing bundle.** `glob("*/*.fdf")`
   yields nothing for a directory that is not there; the bare `iterdir()` that
   replaced it raises `FileNotFoundError` — against that function's own
   docstring, *"absence degrades rather than raises — this is a reporter."*
   The kind of difference a mechanical migration produces and a green suite
   hides.
2. **`_stage_state` gained `label: str = ""`.** An empty label matches nothing
   (`runfiles.parse` requires one), so a caller who forgot it would get
   *"prepped, not launched"* for every rung — § 1.6's forbidden line, reached by
   a signature. Now required, and asserted as an outcome pair rather than as a
   signature: refused without a label, answers with one, same directory.
3. **`_SIDECAR_SUFFIX = SUFFIX`** — a back-compat alias, against the standing
   no-shims rule. Deleted.

**And the first M8 spelling was itself a regression, caught the same way.**
`materialize.attempt_concluded` was moved onto `runfiles.compose`, which
**validates the label** — and that function is handed whatever the DECK is
called. A cited transport relaxation may be a person's own
`my.relaxation.fdf`, so composing turned *"this directory has no record"* into
a `RunFileError` at somebody who used a dot. `runfiles.tail` is the right door,
and the distinction is now written into it:

> **`compose` when you own the label; `tail` when you were handed a stem.**

`summarize._trial_deck` had the same shape (its `basename` is
`Path(job.script).stem`, read out of `job-set.json`) and moved too. Both are
readers whose modules promise to degrade rather than raise, which is exactly
what the label validation would have broken. **The lesson generalises past this
migration: a door with a stricter contract than the code it replaces is a
behaviour change, not a cleanup.**

**And auditing that class found a pre-existing crash it had already caused.**
`web/blueprints/watch.py`'s run resolver reads `JOB` and `SystemLabel` through
regexes bounded to `[A-Za-z0-9_-]+` — deliberately, so a malformed deck cannot
inject a path — but `py_stem` is the deck's **filename** stem, unbounded, and it
went straight into `compose`. A PySCF script named `my.job.py` made the Watch
tab raise `RunFileError` instead of falling through to its generic steps.
Introduced by `3dfa76c9` (the commit that moved those names onto the composer),
measured 2026-09-08, fixed by catching the grammar's own refusal and RECORDING
it in the resolver's `attempts` list — which is what a resolver is for.

**Seven more mutations, seven killed** — including the two guard-integrity ones
(`COUNTER_FRAGMENTS` emptied; the `label` default restored). The second
**survived** the first attempt, because nothing called `_stage_state` with
defaults; the outcome test above is what closes it.

**Still open, unchanged by M7 and M8:** `siesta/makov_payne.py` spells `-run<N>` and
cannot import `runfiles` until `runfiles` joins `MONITOR_COMPANIONS`; § 5k.4a's
~40 text-anchored source slices in tests; and `classify_source_reads.py`'s
line-keyed overrides (this survey's are keyed by `(file, function, pattern)`
and a dead key now FAILS, which is the pattern to copy there).

### 5k.4b Where M1–M6 left it, and what M7 was

**Owned went 24 → 17.** The seven closed were the ones with a door already
waiting or one step from it; the seventeen left are listed by
`classify_path_finders.py --list owned`, and re-run it rather than trusting
this paragraph.

Most are the SAME SHAPE M6 just gave a door to — a dotted role globbed with no
label: `*.fdf` ×3, `*.out` ×2, `*.molwatch.log`, `*.source.xyz`,
`*.molstruct.json`, `*.XV`, `*.template.toml`. Those are `find_by_role`
calls; the work is mechanical and the risk is in the two that are not:

| left over | why it is not mechanical |
|---|---|
| `validation/identity.py` `*{suffix}` | the suffix is a RUNTIME value from the engine's rules file, so the role cannot be checked against `WRITTEN` at the call — `find_by_role` would have to take an unchecked role, which is the check it exists for |
| `validation/identity.py` `{label}*` | has a label but no role: *everything this calculation left here*. `find(dir, label)` answers it, but the caller wants names it can subtract a known set from, so the shape of the answer matters |
| `web/blueprints/workspace_storage.py` ×2 | the workspace store's own grammar, not run files — same fault, different vocabulary, and it needs its own namer first |
| `template.py` `*.template.toml` | `template` owns that name; the door belongs there, beside it, not in `runfiles` |

**`os.path.join(dir, "*.fdf")` appears three times in `web/blueprints/watch.py`**
and the survey reads the pattern through the join, which is worth keeping in
mind when the count is re-derived: they are ordinary globs wearing a coat.

### 5k.5 What must not change

> **§ 5k's METHOD is superseded by § 5l** *(user ruling, 2026-09-08)*. Its RULE
> stands and is guarded; what stops is closing each asymmetry by adding a door
> for the question a call site happened to ask. Read § 5l before acting on
> anything in this section — five of the doors M7/M8 added are on its delete
> list.


`runfiles` stays stdlib-only; `Shape` stays the only reader of `task.SHAPES`;
`identity.stage_token` stays the one speller of `<NN>_<name>`; and no finder
may infer the shape from the disk. A migration step that needs one of these to
move is a step that has found a design problem, not a step to push through.


