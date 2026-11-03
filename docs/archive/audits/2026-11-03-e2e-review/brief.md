# E2E review brief — does each e2e test still pin the right detail of the contract as it now reads?

You are one of ten readers reviewing molbuilder's end-to-end tests (`tests/*_e2e.py`, 39 files, 565 tests) at
/home/qqing/molbuilder. The user, 2026-11-02: *"all e2e tests should be re-evaluated and reviewed in their full code
since many details in contract and design has been refined/revised. e2e tests may no longer pin the correct details.
they need more work. so make sure this happens before you run e2e tests to validate 2.65.4"*.

The e2e batch last ran on 2026-10-31 09:11, at release ~2.56.2. Since then 2.57.0 … 2.65.3 shipped with no e2e run
(the user's rule: no e2e before F9, except the approved 2.65.4 run). Those releases changed contracts and roads the
e2e tests walk. Read their rows in `docs/process/changelog.md` (2.57.0 to 2.65.3) first, so you know what moved:
- 2.57.0 a species is a label and an element; 2.58.0 a displacement sweep relaxes once, optical frames keep each spin's
  Fermi level; 2.59.0 the results views (the run door answers its own folder; `point_of`, `views_of`);
- 2.60.0 **a run is one start** — the automatic in-place retry and the try counter REMOVED; a retry is a re-run, a new
  `run-<n>/` folder; 2.61.0 a run's end known wherever it runs; only `jobset launch` starts a run;
- 2.62.0/2.62.1 what a stage was prepared with, a sequence's header, a run of points' continuation, the label cap
  counts the stage token, one word for a point;
- 2.63.0 one run script for every engine (blocks decided by the engine record); 2.64.0 an engine's facts are its
  record's, a run's files read by what they are; 2.64.1 a SIESTA GPU run on ScaLAPACK said, never refused;
- 2.65.0 a kind's facts are its record's, pages read them from the server; 2.65.1 **an engine's own stop is said,
  never refused**; 2.65.2 launch asks the kind's reuse part; 2.65.3 one transport describe door, an extension runs on
  the finished calculation's engine (another refused naming both), the label by one rule (`task.label_of`), the
  describe writer's whole preflight, a SIESTA describe refuses a species with no pseudopotential.
Earlier rules that tests may still contradict: the IR is the free atoms' (no atom choice); a page states a method
neutrally, never a verdict or a branch on a category molbuilder does not know (e.g. "metal"); a solver and the GPU are
two of the person's choices (never refused, warned or noted together); the warm retry answers any SCF stop within the
person's budget (no divergence test); prep decides, launch only runs; a refusal names the way out.

## Rules for you (hard)
- READ ONLY. Never edit, create or delete a file in the repository. Write only your report file (below).
- Never run pytest, `tools/testrun.py`, `molbuilder`, `jobset`, an engine, a browser or a server. Never import
  anything from `tests/` in a script. Never touch `~/.config/molbuilder`, `~/.local/state/molbuilder` or `projects/`.
  You may read code and docs, `grep` (use /usr/bin/grep), and run read-only `git log` / `git show` / `git diff`.
- Every claim gives file:line. A claim that the code no longer does something is made only after reading that code
  path (molbuilder/…:line) — never from a grep hit alone. A contract claim quotes the sentence (≤ 2 lines) and its §.
- If a claim cannot be checked, say "unverified" — never guess.

## Read first, whole
1. `docs/process/testing.md` — tier 3 (§ 0), § 2b (the guards), § 3b (a test earns its place; the GOAL and CONTRACT in
   its first lines; the science exception), § 3a (end product, never source), § 5 (browser), § 6 (patterns).
2. The owning contract chapters named for your group, **whole** — as they read today. If a chapter is too long for
   your context, read whole the sections your tests cite and every section a changelog row since 2.57.0 touched.
3. Each file of your group **whole**, and every helper/fixture/data file it uses (`tests/support/*.py`,
   `tests/_road.py`, `tests/conftest.py` parts it calls, `tests/data/*.toml` it reads).
4. `git log --format='%as %s' -- <file>` per file, to see when and why it was last revised.

## The user's rules for a test (judge every test against them)
- **Outcome only** — never a test that greps source or doc text, or pins one file's text to another's.
- **Category stated**: e2e with the real engine, or e2e with the browser; an e2e is only for a failure only the engine
  or the browser shows — a check that needs neither belongs in tier 1 (say so: "not e2e").
- **No record a run did not make** — no handcrafted `task.json`, `job-request.json`, `run.json`, launch record,
  deck, `.out`, planted mtime, faked server reply, planted sessionStorage where a click reaches the state, a
  hand-edited bundle. A person's edit of a file molbuilder hands the person (the template, `job-request.json`
  values) is a fair input when the contract says the person edits it.
- **Artifacts read through their PARSER** (molbuilder's reader door), not re-parsed by the test's own regex.
- **Expectations from the CONTRACT** — never copied from the code's output; a test that disagrees with the code
  names the code's defect.
- **A case is a table ROW** — repeated test bodies that differ by data are one parametrized test.
- **Retire first** — a test that no longer guards a real failure a person would meet is retired, not repaired.

## What to find, per TEST (not just per file)
For each test function (and each fixture that makes a run or a state), decide one or more of:
- **RIGHT** — still pins a rule the contract states today; its expectation matches the contract's words.
- **WRONG DETAIL** — its expectation contradicts the contract as it now reads (quote both: the test's line and the
  contract sentence with its §). E.g. it expects a refusal the contract now says is only said; a word, file name,
  folder layout (`run-<n>/`, no try counter), count, order or state the contract changed.
- **STALE ROAD** — it walks a road the code no longer has: a flag, door, file name, layout, verb, step, CLI option
  or record field removed or renamed; a fixture that writes or expects files describe/prep/launch no longer write.
  Say whether it will FAIL or pass VACUOUSLY (worse: green while checking nothing). Cite the code line.
- **WEAK** — it cannot fail on the defect its docstring names (mutation reasoning: which code change would it
  survive?), or it asserts only presence.
- **HANDCRAFTED** — input not made by the road (see above); say what the road would make instead.
- **TEXT-PINNING** — asserts on source/doc text or exact UI copy the contract does not fix.
- **DUPLICATE** — of which test (file:line); **NOT E2E** — needs no engine and no browser.
- **NO GOAL** — states no goal/contract/silent failure in its first lines (testing.md § 3b).
- **RETIRE** — guards no real failure today (say why).
Then also list **OWED**: rules on your group's roads that changed since 2.57.0 and that only the engine or browser
can show, which no e2e test pins now (name the rule's § and the run that would show it — prefer a check on a run the
pass already makes, testing.md tier 3: "one pass makes each run once").

For every non-RIGHT verdict, give the **revision**: concretely what the test should assert instead (the new
expectation and its contract source), or what to delete, or where it moves. Do not write code; one or two lines each.

## Report
Write `/home/qqing/.claude/jobs/142ef3ba/tmp/e2erev/<GROUP>.md` (your group id is in your task):
1. **Summary**: counts per verdict; then every WRONG DETAIL and STALE ROAD finding first (one line each, file:line,
   the contract §), then WEAK/HANDCRAFTED, then the rest.
2. **Per file**: one table row per test: `test (line) | what it asserts | contract § (as it reads today) |
   verdict(s) | evidence (file:line, quotes) | revision`.
3. **OWED** list.
4. One short paragraph: what in these files drifted from the contract, and why (the pattern).
Reply with the summary only.
