# Testing — the strategy, the layers, and the browser tests

**Role:** reference
**Domain:** process
**Companions:** [`conventions.md`](?doc=process/conventions.md) — the guard tests
that enforce the conventions; [`package-layout.md`](?doc=process/package-layout.md)
— where `tests/` sits; [`ops/installation.md`](?doc=ops/installation.md) — the
conda envs the backend tests run against.

The suite is ~360 test files. Most are fast Python unit/module tests you'll write
constantly; a smaller set drives a real browser. This doc is the map: the pyramid,
where tests go, how the front-end JS is tested without a browser, and the handful
of Playwright patterns that keep the e2e tests from being flaky.

## 0. Three tiers — what a person runs, in development and in the field

molbuilder is checked in three tiers, each answering its own question. On a new
installation run them in this order; in development, run the tests your change
touched (§ 6.1), and a tier whole only when the change reaches that far.

| tier | answers | where | needs | command |
|---|---|---|---|---|
| **1 · basic** | is molbuilder installed whole, and does it do what its contracts say? | anywhere — the development box, a fresh install | the `molbuilder` env; no engine, no browser, no scheduler | `python tools/testrun.py run none2e` |
| **2 · field** | does molbuilder work with **this** target — its scheduler, its queues, its environments? | the target machine, or here with its record | the target's record, made there by `jobset probe --write`; for a field run, its engine environments | `MOLBUILDER_FIELD_RECORD=<record> python tools/testrun.py run field`, then a field run (below) |
| **3 · e2e** | does the science come out right, and does the whole workflow — the pages, the road, the engines — hold together? | a machine with the engine environments and a browser | `molbuilder-siesta`, `molbuilder-pySCF`, Playwright's Chromium | `python tools/testrun.py run e2e` — on the person's word: it runs real engines |

### Tier 1 — the basic suite: installation and correctness

**What it holds**: molbuilder's own behaviour, driven the way a person drives it —
the road (`jobset init → prep → launch`; the case tables, § 6), the suite's
stand-in engine for a run's process, measured fixtures read where they were
measured (`tests/fixtures/`), and machine records written in molbuilder's own
format to describe a machine — its cores, its queues and their limits — to the
logic that reads them.

**What it never holds** *(user, 2026-10-04, 2026-10-06: "the whole default test set
should never be based on fabricated text")*:

- **fabricated text** — engine output typed into a file (`.out`, a log), a
  scheduler's answer typed into a parser (`sinfo`, `sacctmgr`, `scontrol`), a
  command's answer swapped in-process. Text a machine or an engine prints is
  read where it was printed: tier 2 and tier 3;
- **a hand-built product** — a job set, a run record or a result made by hand
  where molbuilder makes it, or a state the road never reaches;
- **an expectation copied from the code** — an expected value is the
  contract's; a test that disagrees with the code names the code's defect, and
  is never edited to match what the code just printed.

**When a basic test fails**, ask first whether it stands on fabricated text or a
hand-built state: such a test is retired, never updated. Then whether the rule it
cites holds (§ 3b) — a test is not an authority.

### Tier 2 — the field: molbuilder on the target environment

On each machine molbuilder will run jobs on — a cluster, or a workstation — two
steps:

1. **Its record, probed there, and read.** On the target: set its `env_init`
   (`molbuilder envs init-config`), then `molbuilder jobset probe --write` — for a
   cluster prepared for from elsewhere, `--name <name>`, and its record copied to
   `~/.config/molbuilder/environments/<name>.json` on the machine you prep from
   ([`preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)).
   Then the field test reads it, here or there:

   ```bash
   MOLBUILDER_FIELD_RECORD=~/.config/molbuilder/environments/<name>.json \
       python tools/testrun.py run field
   ```

   It holds the record to its contract
   ([`configuration.md`](?doc=configuration.md) M-2,
   [`scheduler.md`](?doc=execution/scheduler.md) R13/R14): each machine fact a
   number or unknown, each queue named whole, each wall one the scheduler takes,
   the answers of one question arriving together. It asserts nothing about what
   a site's scheduler prints — the probe read that, on the machine. On a
   cluster, run ON it, it also asks the scheduler about a held H2's line
   (`launch run --mode ask`): asked once with `--test-only`, the answer read as
   a start time, nothing submitted or recorded — an `sbatch` that passes a
   question to the real one and refuses anything else stands first on PATH
   (`test_ask_the_target.py`). `tests/field/` is collected only when a record
   is named; tier 1 never sees it, and its own `sbatch` writes each call down
   and refuses it: what a scheduler answers is read here, never typed into a
   basic test *(the stand-in answered in SLURM's words until 2026-10-06)*.
2. **A field run.** The smallest real job, on the road, on the target: `jobset
   init` a held H2, `jobset prep run coarse --target <name>`, then on the target
   `jobset launch run coarse` with its mode, `jobset status`, and the Results tab.
   It shows the run script enters the target's environments, the header is one
   its scheduler takes, and the run's records read back -- the job id, the
   queue, wall and memory it was sent with, `queued as job <id>` while it
   waits: what the basic tier no longer stages with a scheduler's typed answer. A failure here is the
   target's record or its environments, said by the refusal: fix it there and
   probe again.

### Tier 3 — e2e: the science and the workflow

The `*_e2e.py` batch: real engines — SIESTA, PySCF — on small systems (H2, H2O, a
minimal junction), their results checked against measured or published values
(§ 3b, *the one exception: scientific validation*), and the browser walks through
the pages (Playwright, § 5). It is slow and spends the machine's cores: run it on
the person's word, after tiers 1 and 2 pass.

## 1. The pyramid — pick the lowest layer that covers the contract

Tests are marked by **layer**, and the marker is orthogonal to the directory (a
`tests/spectra/` file can hold unit *and* integration tests). The markers, from
`pyproject.toml`:

| Marker | Layer | What it tests |
|---|---|---|
| `unit` | L1 | a pure helper — no I/O, no globals (microsecond cost) |
| `module` | L2 | one submodule's public surface, end to end |
| `interface` | L3 | the contract between two modules (a registry, a severity map, a shape) |
| `integration` | L4 | several subsystems agreeing on one shared fact |
| `smoke` | — | subprocess-runs a *generated script* (slow; needs pyscf) |
| `e2e` | — | browser-driven Playwright (slow; needs chromium) |
| `slow` | — | > 1 s (`make test` skips them) |
| `engine` | — | starts a real SIESTA / TranSIESTA / PySCF process, itself or through a fixture; such tests run one at a time (§ 6.1a) |
| `capture_on_fail` | — | dump browser state + console to `test-artifacts/` on failure |

The rule of thumb: **cover a contract at the lowest layer that can see it.** A
severity-map bug is an `interface` test, not an e2e click-through. e2e is for the
things only a real browser exposes.

Config worth knowing (`pyproject.toml`): `testpaths=["tests"]`,
**`pythonpath=["."]`** (so the in-tree package imports without `pip install -e`),
`addopts="-ra"`.

## 2. Where tests go, and the one structural invariant

`tests/` is **flat at the top** with a few topic subdirs (`tests/parse/`,
`tests/spectra/`, `tests/validation/`, `tests/watch/`, and `tests/field/` — the
field tests, collected only when a probed record is named, § 0); fixtures live in
`tests/data/`. Naming is `test_*.py`.

The one *structural* rule is **layering**, and **review keeps it, not a test**
*(user, 2026-09-27: "a static code review problem")* — imports only point down:

- **L1** (core types — `structure.py`, `chemistry.py`, …) imports nothing higher;
- **L2** (domain verbs — builders, engines, parse, …) may import L1, not L3;
- **L3** (the two *surfaces*, `cli.py` and `web/`) may import anything.

`tests/test_layering.py` read every module's imports to check it until
2026-09-27; a scan of the package's own source earns no place (*What does NOT
earn a place*, below), and the two failures that matter at run time show up
without it — an upward import at a module's top fails every import of that
module, and a file that ships beside a job and reaches into molbuilder fails
there, which `tests/test_monitor_bundle_runs_alone.py` runs. **The proof is a
workflow that finishes end to end** *(user, 2026-09-27)*: the road tests that
launch real runs in each engine's own env and assert the monitor's closing
report (`test_siesta_vibration_e2e.py`, `test_vibration_e2e.py`). The question
review asks is `process/code-audit.md` § 1c (e). This is what lets `cli` and
`web` share one API without circular imports — see the thin-shell note in
[`conventions.md § 3`](?doc=process/conventions.md).

### 2a. A test never touches the real projects tree

**A test that builds a folder under the developer's own `projects/` is
reading and writing their data.** It also lies: it passes because that tree
happens to hold something, and it fails on a machine where it does not.

The rule is one door, the same one production uses. `projects_root()` is the
single definition and the only reader of `$MOLBUILDER_PROJECTS`; it feeds
`Capabilities.file_picker_roots()` → `GET /api/files/roots` →
`setProjectsRoot()` → `projects.getProjectsRoot()`. A test points that door
somewhere temporary and builds inside it:

- **`isolated_projects_root`** (function-scoped) — `$MOLBUILDER_PROJECTS` →
  `tmp_path/projects`. Use this by default.
- **`isolated_projects_root_module`** — the module-scoped sibling, for a
  fixture a whole file shares. It exists because a module-scoped fixture
  cannot ask for `tmp_path`; it uses `MonkeyPatch()` and `tmp_path_factory`
  directly.

Nothing is cached: `file_picker_roots()` calls `projects_root()` at CALL
time, so pointing the door is enough and the live server serves the tmp tree.

**Enforced, not just written here.** `test_no_tests_read_the_projects_tree.py`
AST-walks every test file for a path built from the repository root joined to
`projects`, and carries a ten-row truth table of what must and must not fire —
because a guard nobody has watched fail is a guard nobody has tested. It found
thirteen sites in seven files, all converted 2026-09-06.

**A fixture of your own that re-isolates `HOME` or `XDG_CONFIG_HOME` moves the
machine scope out from under `conftest`'s.** If the test then preps, call
`conftest.write_machine_record()` at the end of that fixture — prep refuses
without a probed record (`running-a-job.md` § 3.1).

## 3. Design tests *around* the envs, never the reverse

molbuilder dispatches into per-backend conda envs
([`installation.md`](?doc=ops/installation.md)), and the test rule follows from
that: **a test must not require an env to be reshaped to pass.** The whole suite
runs in the host `molbuilder` env; tests that genuinely need a backend
(`smoke` needs pyscf, `e2e` needs chromium) are **marked and gated** so they skip
cleanly when that backend is absent, rather than failing. You design the test to
fit the environment model — you never edit an env to make a test pass (that rule is
load-bearing elsewhere too).

## 3b. A test earns its place by the failure it would have caught *(user, 2026-09-08)*

**More tests are not better results.** A suite is a cost paid on every run and
every refactor, and a test that cannot fail — or that only restates a signature —
is cost with no signal. The question is never *is this true?* but **what would
have to break for this to go red, and has that ever happened?**

> **The default is REMOVAL.** *(user, 2026-09-08: "there is no real thing or bug
> that was detected by most of the tests that are designed this way. Only good
> tests fall out of this style. You should ask always: why should I keep this
> test — if there is no strong reason, then remove it.")*
>
> The burden of proof is on KEEPING. Not *can I justify cutting this* — **why
> should this exist**, answered with a concrete failure it would catch that
> nothing else catches. No strong answer, no test.

**Judge at the FRAMEWORK level, not the assertion level.** The question is not
whether an assertion is true. It is whether the test protects a property of the
system **that the framework does not already guarantee**. If a correct API, a
type, a frozen dataclass, a validator, or another test already makes the asserted
state impossible to reach, the test is rigidity with no benefit: it costs on every
run and on every refactor and buys nothing.

**And "it pins a contract" is not automatically a strong reason.** A contract test
earns its place when the contract could be violated *silently* — when nothing else
would go red. If breaking it fails ten other tests first, this one is a monument to
the rule, not a guard on it. That distinction is the whole of this section: the
objection is never to contract tests, it is to **mechanical** ones — written
because a rule existed, rather than because the rule can plausibly break unseen.

**Cutting mechanically is the same failure in the other direction.** For every
test removed, be able to say what now goes undetected and why that is acceptable.
If that sentence will not come, the honest answer is *unsure*, not *cut*.

### Every test states its GOAL and its CONTRACT, in its first lines *(user, 2026-09-08)*

> *"The tests should have their goal clearly stated upfront in their starting
> comments, and the related contract specified, so that it is easy to identify
> obsolete design pinners."*

**A test's docstring opens with two things:**

1. **The failure it catches** — the concrete thing that goes wrong if this test
   is deleted and the code drifts. Not what the test *does* (the body says
   that); what it *prevents*.
2. **The contract it serves** — the document and section that owns the rule, or
   the dated defect it commemorates. `job-contracts.md § 2.2a`. `2026-08-27, the
   card showed one arbitrary node's cores`.

```python
def test_a_flat_rung_that_never_ran_does_not_read_its_siblings_output():
    """A rung with no output of its own must not report a sibling's state.

    `project-layout.md` § 1: in flat, one directory holds every stage and the
    token in the filename is what selects. Measured 2026-09-08 — without the
    narrowing, a stage that had never run reported the newest stage's `.out`.
    """
```

**Why this is a rule and not a style note: it is what makes a test's own
obsolescence visible.** The audit that produced § 3b spent most of its effort
reconstructing, from the body outward, what each test was *for* — and the
verdicts that were hardest to reach were exactly the tests that never said.
Several turned out to guard code that had been deleted: a `-stage<N>` filename
spelling retired 2026-08-10, a `_mb_mem_est` block with zero hits in the
package, a `MOLBUILDER_PREACTIVATE_CMDS` hook that no longer exists. **Each of
those was one grep from being retired, and stayed for months, because the test
never named the thing it was protecting.**

A stated contract also fails usefully: when the contract moves, the citation
goes stale and the test announces its own obsolescence instead of quietly
pinning a retired design.

> **The two guards that used to make that automatic are gone**, retired
> 2026-09-10 in the sweep that cut 18 files asserting the shape of the
> repository rather than a result. That was right by this page's own rule — *a
> doc path still resolves* is a static-review finding, not a test. The cost is
> worth stating plainly: **a retired guard moves the work to review, it does not
> delete the work.**

**The four questions a header should let a reader answer without reading the
body:** what breaks if this goes; who owns the rule; has it ever actually
happened; and is that rule still live. A test that cannot answer the fourth is
the obsolete design pinner this rule exists to surface.

### The one exception: scientific validation *(user, 2026-09-08)*

> *"The only test I ask you to be careful about is the scientific validation
> ones. These must be correct and rigid — but how they should be designed can be
> discussed."*

**Everything above is suspended for a test that asserts a physical or chemical
fact, or gates a calculation against a science rule.** Rigidity there is the
feature, not the cost: a science test that looks over-strict is doing its job,
and *"another test would fail first"* is not a reason to drop one — physical
correctness is the property with no cheap second opinion.

In scope: numbers with physical meaning (frequencies, IR/Raman intensities,
thermochemistry, energies, forces, transmission, conductance, charge, dipole);
comparisons against literature values, windows or orderings; the `validation`
package and its gates; unit conversions and constants; and the structural facts
that decide **which atoms are computed** — net charge and protonation, species
order, basis and pseudopotential coverage, cell and periodicity, atom-index
convention, frozen atoms and region labels. Anything citing `docs/science/`.

**DESIGN is open; correctness is not.** A science test may fairly be criticised
for asserting an implementation detail instead of the physical quantity, for an
arbitrary or undocumented tolerance, for being able to pass for a physically
wrong reason, or for lacking a stated reference. Those are improvements to
propose — never grounds to delete.

### The three kinds that earn a place

| kind | what it pins | why a review cannot replace it |
|---|---|---|
| **a measured regression** | a defect that ACTUALLY happened, with its numbers | a person forgets; the file does not |
| **an artifact lint** | a property of *every* shipped artifact of a class — each served template, each stylesheet, each generated deck | it replaces the manual sweep entirely (§ 3a) |
| **a contract a document states** | a rule someone wrote down and code must obey | the document and the code drift silently |

### What does NOT earn a place

**A scan of molbuilder's own source** *(user, 2026-09-26)*. A test that opens a
`.py` or `.js` file of the package — `ast.parse`, `inspect.getsource`,
`read_text()` — to check a spelling, an import, or that a rule is written in
only one module is not an artifact lint, whatever it quantifies over. The
framework is data-driven: each rule is declared once and imported everywhere,
so a second copy is a **review** finding
([`process/code-audit.md`](?doc=process/code-audit.md) § 1c) — while it
agrees with the first it gives the same answer, and nothing a run observes
can tell them apart. Where the failure such a scan reached for matters, the
test RUNS the code — the route, the CLI, `jobset init → prep → launch` — and
asserts what came out.

**An API-shape assertion.** *"`find_by_role` returns sorted paths"*, *"`compose`
refuses an unknown field"* — that is the signature restated in a second
language. A correct API plus static review covers it, and the test only fails
when someone deliberately changes the thing it copies.

**A test per call site — and the sharp form of this is not what it first looks
like** *(user, 2026-09-08: "isn't it more important to check the return values
or the outcome of a module/function?")*.

Yes — a return value is exactly what to assert. The mistake is asserting the
return of the **wrong function**. A thin caller of a door returns the door's
answer, so a test of that caller observes the DOOR's outcome through a wrapper:

```
test_the_watch_resolver_finds_a_molwatch_log_first   -> parse.dirs.rundir.openable_in
test_find_template_still_refuses_two_answers         -> template.find_template
test_the_provenance_step_reads_the_wrapper           -> parse.contract._declared_in_provenance
test_read_system_degrades_on_a_missing_bundle        -> summarize._read_system
```

All four are callers of `runfiles.find_by_role`. If the door is right they all
pass; if it is wrong they all fail together — **four tests carrying one bit**.
*(The first two are gone — retired 2026-10-04, it stood on outputs written
by hand, and 2026-10-05, a duplicate of the refusal pinned where a person meets
it; the example stands as the pattern.)*

> **The first one stopped being an example on 2026-09-18, and the reason is
> this section's own rule.** `openable_in` still calls `find_by_role`, so the
> claim above is still literally true of it — but it is no longer THIN: it
> asked `task.json` what calculation this is (the run door, `runs`, asks the
> description since 2026-10-04), `runfiles.result_roles` what
> that produces, and `detect()` whether anything can open the answer
> (`model/parse.md` § 5.2). Three rules the door cannot know, so by the rule
> below it now earns its own tests, and it has them. Left in place rather
> than swapped, because a caller *becoming* a decider is the transition worth
> showing.

> **Test the DOOR's outcome. Test a caller only where the caller DECIDES
> something the door cannot know.**

A caller earns its own test when it adds a rule of its own — and those are worth
a lot, because that is where the real defects were:

- `runstatus._stage_state` deciding *queued* vs *pending* from which roles are
  present — the policy is the caller's, and a finished PySCF rung reported
  **queued** because `.pyscf.log` was not in the set it asked for;
- `runrecord.ending` **degrading rather than raising** on a deck
  name it was handed — a reporter's contract, not the grammar's;
- `parse.engines._sidecar.read_frozen_atoms` reconstructing a name instead of
  asking by role — every staged run silently lost its frozen atoms.

None of those three is visible from the door. All three are real, measured, and
each is worth more than the twelve wrapper tests put together.

### The rule that follows, and it is the sharp one

> **Unifying an API must REDUCE the test count.**

If N call sites each needed their own test, and they now share one door, those N
tests collapse into one door test plus the guard that keeps callers on the door.
A unification that *adds* tests has usually not unified anything — it has added a
layer and kept the old surface, and the test count is the first place that shows.

*This section exists because the opposite happened. The paths framework
(`plans/plan.md` § 5k) was undertaken to reduce complexity, and the migration
shipped **five new test files in one day** — one test per migrated call site,
alongside the guard that already made most of them unnecessary. The count is
evidence: it went up while the API was being unified, which is precisely the
signal this rule names.*

> **And the follow-on proves the rule twice.** § 5l answered the same problem by
> specifying a replacement API — one address, three verbs — and `ref.py` shipped
> with a full test file and **no production caller at all**. For over a week the
> repo carried a module and its tests that answered nothing, while the ~40
> functions it was meant to replace kept being used. **A test file is not
> evidence that a unification landed**; a retired call site is. The standard was
> retired and both files deleted 2026-09-17 (`plans/plan.md` § 5l).

### How to review a test for retirement

Ask in order, and stop at the first *yes*:

1. **Is it subsumed?** Would another test already fail for this cause?
   **A subsumption verdict needs a MUTANT, not an argument** — break the code and
   watch the *coverer* go red too: `python tools/verify_subsumption.py pairs.json`.
   Measured with a FIXED operator set 2026-09-09: this reasoning was **wrong on
   9 of 19 candidates -- nearly half**. (The earlier figures of 1-in-5 and
   1-in-3 were measured with a harness that had no operator for dropping a
   clause from an `or`, which is the commonest shape in a validator: a guard is
   `not isinstance(x, T) or not x` and its two tests reach ONE CLAUSE EACH.)
   Coverage cannot rescue it either, because two tests can cover one line and
   still assert different things about it. A verdict of INCONCLUSIVE means the harness has no
   opinion — never that the cut is safe.
   *(A subsumption that DID hold: a `RETIRED` doc-path pattern — four alternation
   arms, three lookbehinds — enumerated locations all deleted from disk, so the
   neighbouring "every cited path must exist" test flagged every one.)*
2. **Can it fail?** Break the code it names and watch. A test that stays green is
   not a test (`feedback: mutation-test the test`).
3. **Does it restate a signature?** Then delete it and let review carry it.
4. **Does it name a defect that happened?** Keep it, and keep the numbers in it.

### What the 2026-09-08 audit left open

The audit that produced this section gated ~3,000 of 4,448 test functions. Its
**applied** cuts live in their commit messages; everything it found and did not
act on lives in two records, and neither has been actioned:

- [`test-audit-findings.md`](?doc=archive/2026-09-08-test-audit-findings.md) — two real code
  defects found by auditing tests, four tests owed a redesign rather than a
  deletion, one coverage gap a deletion exposed, and 18 unapplied subsumption
  verdicts with their reasoning.
- [`archive/2026-09-08-test-design-findings.md`](?doc=archive/2026-09-08-test-design-findings.md) — the
  protected class: cases where a science test would pass for a physically wrong
  reason. **No science test was cut**; ~120 drafted cuts were withdrawn when the
  exception below landed.

## 3a. A test asserts on the END PRODUCT, never on the source that made it

*(User ruling, 2026-09-03: "tests should be focusing on end product and
behavior. Tests pinning source files are stupid, retarded, and absurd.")*

> **Never assert that a string appears in a file the project ships.**

```python
js = client.get("/static/lib/spectra/core.js").data.decode()
assert 'transition("IDLE")' in js          # ← this is not a test
```

That passes when the call is **deleted and its name left in a comment**, and
fails when the code is correct and someone renames a local. It measures
spelling, and spelling is not behaviour.

**What to assert instead — the end product, which is whichever of these the
code actually makes:**

| the thing under test | the product to assert on | how |
|---|---|---|
| a route | the response — status, body, headers | the Flask test client |
| a deck emitter | the rendered deck text | call the renderer |
| a JS module | what it *does* — call it and check the result or the DOM it changed | `tests/_node_esm.py` (§ 4) |
| a whole page's flow | what a person sees after acting | a Playwright `*_e2e.py` (§ 5) |

**Reading a shipped file is not automatically wrong — asserting on its
spelling is.** These are legitimate, because the *artifact itself* is the
thing under test:

- no inline `<script>` in a template — a CSP property of what ships;
- a static file is served at all, with the right status and caching headers;
- every asset a template names exists on disk
  (`test_wheel_ships_the_front_end.py`);
- a stylesheet declares no raw hex colours, or no duplicate selector.

The test is *"which question is this asking?"* — a property of the shipped
artifact, or the presence of one line of implementation.

**What replaces a retired pin is REVIEW, not silence.** The claims those
tests make are real; what is wrong is asking a grep to check them. The four
classes of silent failure they were reaching for — a surface that answers a
question a producer owns, a reference that resolves to nothing, an order
nothing enforces, state carried between subjects — are written up as review
guidance in [`process/code-audit.md`](?doc=process/code-audit.md) § 1c, with
what each one looks like and why it cannot be seen at runtime.

**When you meet an old one, investigate before deleting.** Work out what it
was trying to establish, then either **redesign** it to observe that through
behaviour, or **retire** it because the intent has no observable consequence.
Deleting without reading loses coverage that was real; keeping without reading
keeps a check that cannot fail. B3 in [`plans/plan.md`](?doc=plans/plan.md)
tracks the backlog.

### 3a.1 A test that admits it cannot see the behaviour has already failed

*(User ruling, 2026-09-03: "If a test already acknowledged that only an e2e
test is the only way to verify certain behaviours, then that test should be
retired. And then an e2e test that actually pins those behaviours should be
designed.")*

Some pins say so in their own docstring:

> *"These are static / string-pin checks; the runtime behaviour is exercised
> by Playwright E2E."*
> *"Source-text pins (this page has no node harness — the live browser walk
> covers behavior)."*
> *"…only a browser would show it."*

**Read that as a verdict, not a caveat.** The author already worked out what
would actually verify the thing and then wrote something else. There is
nothing left to investigate: **retire the pin, and write the test it named.**
Which of the two you owe depends on whether the named replacement is real, and
that is a claim to check, never to believe:

| the docstring names… | and it… | then |
|---|---|---|
| another test | **exists and drives the behaviour** | delete the pin — it is a duplicate |
| another test | **does not exist** | the pin was the only coverage there ever was — write the e2e, *then* delete |
| "a browser", no test named | — | write the e2e, then delete |

On 2026-09-03 all three occurred. `test_this_machine_js.py` cited
`test_config_dir_has_one_home`, which **had never been written**, and no
browser test visited `/this-machine` at all — so a page nothing tested carried
a docstring claiming it was tested elsewhere.

**Why an admission is worse than an ordinary pin.** A pin that says nothing is
merely weak. A pin that names its own replacement is *load-bearing
misinformation*: the next reader sees coverage, stops looking, and the gap
closes over. `spectra/test_blueprint.py`'s listener pin stayed green through a
**total teardown leak** — it named the array, the array survived, the contract
died — and its docstring said the behaviour was covered.

**Distinguish the honest neighbour.** A docstring that says *"testing this on
a never-launched stage would prove nothing, so the fixture launches one"* is
the opposite thing: mutation reasoning, explaining why the test can fail for
the right reason. That is the best kind of docstring in this repo. The phrase
to react to is *"the real check is elsewhere"*, never *"proves nothing"*.

## 4. Testing the front-end JS without a browser

> **Neither test tool is installed by default. Both ARE in the host recipe**,
> marked `opt_in`, and one command adds them:
>
> ```
> bash scripts/install-env.sh install molbuilder --with-dev-tools --yes
> ```
>
> That installs `nodejs`, `playwright`, `pytest-playwright` and a headless
> chromium **into the host env**, with the browser confined to
> `$CONDA_PREFIX/share/ms-playwright` (an `activate.d` hook keeps the path set
> for the pytest process) and no step requiring root.
>
> `bash scripts/install-env.sh doctor` shows what you have: the `opt-in:` row
> for `molbuilder` lists whatever is still absent.
> Neither is needed to RUN molbuilder, and chromium is a large download
> nobody should pay for by accident — which is why `bootstrap` leaves them
> out (`envs/recipes.py` records the ruling: a browser-tooling-only env
> cannot start the app, so e2e runs in the host env or not at all).
>
> **Know what a missing tool costs you.** Without `node`, **717 tests skip**
> — and pytest counts a skip toward a green run, so the suite reports
> healthy while every one of those modules goes unexercised. A full lane
> here reads `9,414 collected` and runs about 7,800.


Most front-end logic is tested **in Node, no browser** — the `*_js.py` tests,
plus a handful of others. `tests/_node_esm.py` is **the harness to use**, and
its `run_node(files, snippet)` loads an ordered list of module files via
dynamic `import()` and then runs a JS snippet against them.

> **It is the harness to use, not the harness in use.** Measured 2026-09-29:
> of the **41** `*_js.py` files, **8** drive `_node_esm` (23 files across the
> whole suite do), and **27** of the rest hand-roll their own
> `subprocess.run(["node", ...])` — the duplication
> recorded as **B4** in [`plans/plan.md`](?doc=plans/plan.md). This paragraph
> claimed all of them did, which is the shape of drift worth naming: a
> document describing the intended state in the present tense, so nobody
> looking for the gap can see it. Reach for `_node_esm` in anything new;
> converting the rest is B4's job.

The clever part is that it spans the ESM migration: a classic IIFE file publishes
its `window.molbuilder.*` global as a side effect, and a converted ES module
publishes the *same* global **and** exposes exports — a dynamic `import()` runs
either kind. So a test that reads through the **global** (`window.molbuilder.X`)
passes *before and after* its module converts to ESM — no per-module test churn at
conversion time. (This is why the ESM tasks #103–#107 don't drag a test rewrite
behind them.)

Reach for a JS unit test for anything you can express as "load these modules, call
this, assert the result." Save the browser for what needs the DOM + 3Dmol.

## 5. The browser tests (Playwright / e2e)

The 19 `*_e2e.py` tests use **Playwright** against a real headless Chromium. Each
spins up a **live Flask server** in a fixture (`flask_server`) built from
`create_app`, and drives the `page` fixture (`page.goto(f"{base}/molbuilder")`, …).
The default `web_client` fixture builds the app with **rate-limiting disabled**
(`create_app(config={"rate_limit": {"enabled": False}})`) so the limiter never
trips the test client; the rate-limit tests build their own enabled client.

These are the durable patterns — follow them and the e2e tests stay stable:

- **Locate what the *user* clicks, not what the DOM says.** Target the visible
  label/button, not a hidden input by id. Where a control is backed by an invisible
  element (a hidden radio, a 3Dmol atom), don't fight the click pipeline — set state
  via `page.evaluate(...)` and dispatch the event the JS listener actually cares
  about. (3Dmol atoms live inside a canvas — there's nothing to click; you go
  through `page.evaluate`.)
- **Wait on *state*, not time.** Prefer `page.wait_for_function(...)` /
  `expect(locator).to_*` over `sleep`. A time-based wait is either flaky or slow;
  a state-based wait is neither.
- **Assert with `expect(locator).to_*`**, which auto-retries, rather than reading a
  value once and asserting on the snapshot.
- **Set the viewport explicitly when layout matters.** A `force=True` click bypasses
  the actionability checks but the final coordinate still must land on the element's
  box — a zero-size or off-screen element fails "outside of the viewport." If layout
  is under test, pin the viewport.
- **Every failure should point at the root cause.** Wire `page.on("pageerror", …)`
  and `page.on("console", …)` so a JS error or console message surfaces in the test
  output; use `capture_on_fail` to dump browser state + console to
  `test-artifacts/` for the intermittents.

## 6. A few named test patterns

- **Artifact lints** — assert a property that must hold across *every* file in a
  class, checked on the artifact rather than on a runtime. E.g.
  `test_no_inline_scripts.py` scans every served template for an inline `<script>`
  (the CSP `script-src 'self'` rule would break at runtime otherwise), and
  `test_negative_body_assert_lint.py` AST-lints the *test suite* so a "body lacks X"
  assertion is always paired with a status check. Cheap, and they catch a class of
  bug no unit test would. The artifact is what SHIPS or what the code
  GENERATES; a scan of the package's own `.py`/`.js` is not one (§ 3b, *What
  does NOT earn a place*).

  **The boundary — and it is the whole pattern.** A lint quantifies over a class
  and names no line; the moment it names one file and one spelling it is a *source
  pin*, which § 3a forbids. `assert "function _themeColors" in core_js` is not a
  thin lint, it is a different thing wearing the name. Both examples above pass the
  test: delete the code they guard and they fail; rename a local and they don't
  care. Ask that of anything you add here.

  **A vocabulary survey cannot tell a retired design from an undocumented
  one — and the two want opposite actions** *(2026-09-02)*. 58 tests were
  proposed for retirement because their vocabulary (`fileState`, `fetchSeq`,
  `uiPrefs`) appeared in no live document. It was the shipped code's own
  structure: `lib/trajectory/core.js` is built exactly that way, `fileState`
  alone appears 54 times in it, and retiring them would have deleted the ONLY
  enforcement of `results.md` § 4. What was true is that the names had no live
  home — so the fix was to WRITE the section, not to delete the tests. Before
  retiring a test for naming something undocumented, open the code: an
  undocumented rule that the code holds is a documentation gap, and the test
  is the thing keeping it honest.

  **The instrument — `tools/classify_source_reads.py`** *(2026-09-06)*. The
  boundary above is easy to state and easy to get wrong by eye: this population
  had been counted three times with three answers (233, 256, 173) because each
  count used a different definition and none wrote it down. The tool states the
  definition in code and can be re-run, so **quote its output, never a number
  from a document**. It separates the two things that look identical to a grep
  — an assertion on GENERATED output (a deck, a wrapper, an `.sbatch`: a real
  property of a real product, and the great majority of them are this) from one
  on a file a person wrote — then splits the second by the rule above, and routes
  each pin by *what would have to be true for the check to be honest*:
  **browser** where the answer depends on the cascade, layout or real
  visibility, which jsdom cannot give; **node** where the code must run but
  nothing needs painting; **python** where calling the function is cheaper than
  reading its source. Corrections made by READING a site live in the tool's own
  `_OVERRIDES`, with the reason — *"the regex said so"* is the reasoning it
  exists to replace. The standing backlog it measures is `plans/plan.md` § 5h.

  *(This paragraph carried "1,147 of the 1,255" until 2026-09-07, three
  sentences after telling you not to quote a number from a document. Both
  figures had moved. The tool is the answer; there is no number here now.)*
- **A contract's cases are a TABLE, run down the road** *(user, 2026-10-01:
  "why can't test be just data/config file driven query kind of test?";
  "(1) query, allow or deny, (2) correctly produce slurm command/header, (3)
  correctly execute locally if that's the target")*. Where a contract is a
  set of cases — what a person or a probe writes, and what molbuilder must
  answer — the cases are rows of one data file, and ONE runner drives each
  down the road a person runs (`init → prep → launch --dry-run`; on a
  machine with no queue, the run script's own dry run), checking three
  layers: **allowed or refused**, with the words; **what is produced** — the
  header, the `sbatch` line, the deck, the run script; **what the run script
  does here** — and, before them, **what the probe records** when a row runs
  the probe itself: what it said and what the record holds
  (`tests/data/machine_record.toml`, whose rows end there). A stand-in plays
  what the box cannot be (`sbatch` that queues nothing, `nvidia-smi` that
  reports the GPUs a row gives), from `tests/support/road.py`. A rule changes; its rows change — never a function
  per case, and never a test of an internal step a row already reaches. The
  first is the GPU contract's (`tests/data/gpu_contract.toml`,
  `tests/test_gpu_contract.py`; `execution/gpu.md` § 7).
- **A run is made on the road, or not at all** *(user, 2026-10-03: "why would
  any fucker copy an .out to some fucking other directory and call it the
  result"; "retire all")*. A test that asserts what a run ended as, produced,
  hands on or shows makes that run with the engine, through `init → prep →
  launch`. Everything else is a run that never happened, and a test built on
  one is not written: an output copied into a stage's folder, renamed, cut
  or edited; an output invented as text — handed straight to a parser as
  much as laid in a folder; a conclusion marker, launch record, monitor
  record, restart or product file written by hand, or one of ours edited by
  hand into a shape our writer never writes; a structure carrying the record
  of a calculation nothing ran. A measured fixture is read where it was
  measured, as itself — a parser on the file, the monitor replaying the
  measured run's own files under their own names — never called another
  run's result. *(282 tests and 10 case-table rows were retired for it that
  day, the transport suite's citation fixture and the run-ending and viewer
  cases of plan W55 9b among them. The rest went on 2026-10-04 — user: "any
  fucking faking tests should be retired"; "if you find any such tests, need
  to carefuly justify why we need them" — 446 tests in 72 files, 16 of them
  deleted whole: parser tests on made-up output text, refusals of
  hand-edited result files, restart files planted for the cold start and the
  carry, and the two fixtures that had been cut from their runs.)* **A test
  that fakes output is kept only with its reason stated to the user first.**
- **Framework and API tests; end-to-end only when the user asks** *(user,
  2026-10-04: "you have to rely on more api and framework test rather than
  fucking e2e test. e2e test is only necessary when i say so")*. A run a
  framework test needs is made on the road with the suite's **stand-in
  engine** (`conftest._STUB_BODIES`), which plays the engine's part and no
  other: a road row says how it ends (`stand_in`: the exit code, whether it
  leaves the restart file SIESTA leaves, empty, and whether it waits until
  the calculation's ledger holds a decision — what a launch has written
  down while its run runs), and our wrapper, running
  it, writes the run's records as it concludes — so what a stage builds on,
  what it continues from and whether a viewer follows a run are tested
  through our own code with no engine (`tests/data/hand_overs.toml`). What
  depends on the ENGINE's own files — the `.MD.nc`, an `.XV` — is read from a
  **measured fixture**: a real run of ours, the smallest that shows the
  mechanism (H2; H2O at most — user, 2026-10-04: *"run are supposed to verify
  mechanism ... a Au-BDT-Au is too much"*), made once on the road and kept
  as itself with a README saying how (`tests/fixtures/siesta_flat_h2`). A
  browser or a real engine in a test is for when the user asks for one.
- **State-composition tests** — the molview class of bug: a value is correct in
  isolation but wrong once composed with a sibling piece of state. These get an
  explicit test that exercises the *combination*, not each part alone.

## 6.1 How to actually run the suite — `tools/testrun.py`

Use the project's runner. It exists because a bare `pytest` gives you nothing
until it exits, and piping it to a file or through `tail` is worse than nothing:
the output buffers until the process ends, and a `tail` silently truncates the
failure list, so you draw conclusions from a fraction of the failures.

```bash
python tools/testrun.py run tests/test_a.py tests/test_b.py   # a targeted set
python tools/testrun.py run none2e   # tier 1: every non-e2e test (§ 0)
MOLBUILDER_FIELD_RECORD=<record> python tools/testrun.py run field   # tier 2
python tools/testrun.py run e2e      # tier 3: the engine and browser batch
python tools/testrun.py run lf       # rerun ONLY the last run's failures
python tools/testrun.py status              # live summary of every batch
python tools/testrun.py status --fails      # every failed id + its reason
python tools/testrun.py failed none2e       # bare node-ids, to feed back to pytest
```

`tools/progress_plugin.py` streams **each test outcome** to
`.test-progress/<batch>.jsonl`, flushed per test, so `status` is accurate *while
the run is in flight* and readable from any other shell or session — the path is
stable and git-ignored, never a job-specific temp file.

The working loop this enables: run the tests the change touched, read
`status --fails`, fix, then `run lf` to verify just those failures instead of
paying for a whole re-run. A whole batch is for a change that moves or removes a
whole surface, not a step before every commit.

> **Why this is documented so emphatically.** On 2026-07-29 a session ran three
> full sweeps as `pytest … | tail -14`, saw 7 of 31 failures, inferred the rest
> from whichever files it was already touching, and pushed two regressions as a
> result — a half-migrated per-slot map in `warning-modal.js`, and a new embed
> handle method missing from its documented-surface list. Both were sitting in
> the 24 failures the `tail` had cut off. `status --fails` shows all of them in
> one line each.

Two habits that go with it: never conclude "the suite is green" from a truncated
view, and when an unfamiliar test fails, check it against `HEAD` in a throwaway
`git worktree` before assuming it is someone else's problem — that is how you
tell a regression you just pushed from breakage that was already there, and it
takes a few seconds.

### 6.1a A run is spread over the machine's cores

`run` hands the tests to pytest-xdist: `--workers N` processes, by default a
quarter of the cores and at most 8; `--workers 1` is one process. Two rules
decide what may run beside what, and `tests/conftest.py` applies both as the
tests are collected:

1. **A file runs whole, in order, in one worker.** Its module fixtures are built
   once and its tests meet each other exactly as in one process, so spreading a
   run changes *when* a file runs, never *how*.
2. **A file that runs a real engine runs alone among those that do.** A test
   that starts SIESTA, TranSIESTA or PySCF — itself or through a fixture —
   carries `@pytest.mark.engine` (§ 1), on itself or on its module, and every
   file holding one shares one worker. Two computations never overlap (the
   same one-at-a-time rule as job submission), and an engine's threads never
   compete with another engine's.

`run lf` stays in one process: it re-runs a handful of tests, picked by their
ids as they are collected. It runs only when pytest's last-failed list names a
test file here; a list naming none — empty after a green run, or holding only
xdist's own report of workers that disagreed — is nothing to rerun, where
pytest's `--last-failed` would run every test.

**A test's id and its place in the collection may not depend on the process.**
Every worker collects the whole run, and xdist refuses a run whose workers
disagree on the list, order included — so a clock value, an unsorted set, a
random draw, a pid or a temporary path belongs inside the test, never in its
parameters. The first spread run (2026-09-30) stopped at collection on a
timestamp in `test_notify_listener.py`'s parametrize ids.

What a spread run keeps from a one-process run:

- **One progress file, one writer.** Only the process `testrun` starts writes
  `.test-progress/<batch>.jsonl`. The workers receive the run's own arguments,
  so they load the plugin too, and it stays silent in them. Every result is
  relayed to the writer, so the file reads exactly as a one-process run's
  does.
- **A crash ends the run.** A worker that dies ends it: the groups the live
  workers hold finish, and no group is handed out again. The test it was
  running is recorded as failed with the crash as the reason, so
  `status --fails` names it, and a test is counted once however many records
  name it — so the tests no worker reached leave `ran` short of `collected`,
  and `status` calls the run PARTIAL. `testrun` passes
  `--max-worker-restart 0` because a replacement worker is wrong when the
  dead one had finished groups: xdist 3.8.0 hands those back as work, the
  replacement is given one with nothing left in it, and the run waits for
  ever (measured 2026-09-30).
- **A file that fails to collect is a failure of its own.** One process stops
  at it; a spread run goes on with the other files. Either way the file is
  recorded, named by `status --fails` with its error, counted in FAIL and
  never in `ran`.
- **Plain test ids.** xdist appends the group to every id
  (`…::test_x@tests/test_y.py`); `tests/conftest.py` takes it off each result
  before anything reads it, so `status --fails`, `failed`, the terminal's
  summary and the last-failed cache carry ids pytest accepts back.
- **The canaries.** The `the_suite_leaves_your_*_alone` fixtures run once per
  worker. Each worker's window covers every test it runs, so together they
  cover the run.

Two batches launched side by side each get their own workers, and each its
own engine worker; `run all` covers both in one run.

**Why a quarter of the cores, at most 8** (the one-process runs of 2026-09-29,
by file): `none2e` took 50 minutes, 49 of them outside the engine tests and
most of that per-test setup, which spreads evenly; its longest file takes 3.4
minutes, so past ~8 workers that file sets the wall time. `e2e` is bound by its
engine tests — 15 of its 23 minutes — which run one at a time at any worker
count. The other three quarters stay with the dev server and the engine
tests' own threads. Measured on 2026-09-30: `run all` on eight workers ran all
9540 tests in 20 minutes, where the two batches took 73 one after the other.
The 64 engine tests ran one at a time from minute 1.4 to minute 20.0, and the
rest finished inside that window on the other seven.

## 7. What gates a commit

There is **no CI**. Two things gate a commit, and only the first is a hook:

- **The pre-commit hook** (`.pre-commit-config.yaml`, installed by
  `make precommit`): file hygiene, pyflakes, and a `node -c` syntax check on
  each changed `*.js`.
- **The tests the change touched**, run through `tools/testrun.py` (§ 6.1)
  before committing: the unit's own tests and every test file that touches a
  name the change moved or added. The set is found by what the change touched,
  never by layer, so it holds the touched areas' e2e tests too — the
  2026-06-14 workspace-dispatcher clobber shipped through three commits
  because the e2e test that caught it sat behind `-m "not e2e"`.

The whole suite is not a step before every commit *(user, 2026-09-30)*. It
runs when the touched set cannot bound what could break — a change that moves
or removes a whole surface — and the reason is said before it is launched.

A ready-to-use GitHub Actions workflow sits beside this doc at
`process/github-workflows-test.yml`. It is a **template, not a live workflow** —
copy it to `.github/workflows/test.yml` yourself if you want CI; it is
deliberately not installed there.

## 8. Test map (the meta-tests)

- `test_monitor_bundle_runs_alone.py` — the monitor's shipped files read a real
  run with molbuilder absent (§ 2; the layer rule itself is review's).
- `test_pyscf_bundle_runs_alone.py` — every module the PySCF script imports loads
  from `mb_pyscf.pyz` with molbuilder absent, and writes a pair the package's
  codec reads back.
- `_node_esm.py` — the Node ESM load-sim harness the `*_js.py` tests use (§4).
- `test_no_inline_scripts.py`, `test_negative_body_assert_lint.py` — the
  artifact lints (§ 6).
- `test_no_tests_read_the_projects_tree.py` — no test builds a folder inside
  the developer's real `projects/` tree (§ 2a); carries its own truth table.
- `test_one_door_reads_a_structure.py` — the author's frozen atoms, regions
  and cell, saved as the `.xyz` + `.molstruct.json` pair, reach the deck
  through `jobset init → prep`, because the pair is one file. Written
  2026-09-07 after four readers of that pair were found, one of which
  (`molbuilder.load()`) dropped the sidecar entirely and cost `jobset init`
  the author's regions and frozen atoms. It scanned the package for direct
  reader calls until 2026-09-26; it drives the road now, so a reader that
  drops the sidecar anywhere on it shows up as the deck that relaxes every
  atom.
- `conftest.py` — the `web_client` fixture (rate-limit-off) + the shared fixtures
  (`isolated_projects_root` and its module-scoped sibling, `write_machine_record`);
  each e2e file carries its own `flask_server`.
