# F12 fix 15 (every printed command is one you can type, 2.77.23, R3c2) — the review

**Verdict: closed with 2.77.23.** The user, 2026-11-21: *"Spell in full, one printer"* — every line molbuilder
prints that tells a person to run `molbuilder <verb>` spelled in full through one printer,
`"${CONDA_PREFIX:?}/bin/python" -m molbuilder …` (`execution/job-system.md` § 5.3); the guides keep their
stated shorthand in prose. And, asked mid-release when a printed line typed in a calculation's folder
answered *No module named molbuilder*, 2026-11-22: *"Host env points at the repo"* — the host env holds a
path file naming the checkout, written by its recipe and again by every `bootstrap`
(`ops/deployment.md` § 0, new, the user: *"in the deployment section/part, explain what is the
design/setup after bootstrap"*).

## What was built

- **One spelling** (`identity`): `as_typed`, `run_as`, `PROG`, `JOBSET_PROG`, `JOBS_PROG`, `in_template`
  for a line placed in a `str.format` template; every printed line to type across ~30 modules through it
  — next steps, refusals, deck headers, README files `init-config` writes, the pages (`molbuilder.typed`,
  read from the head's `molbuilder-typed` meta; `{{ molbuilder_typed }}` in templates).
- **Where a line lands**: a generated script says it through `runwrap.shell_literal` (never read by the
  shell); the CLI's help prints each indented paragraph as written (`cli._kept_as_written`) and every
  `Usage:` as typed; option help (a wrapped column), the JSON template (quotes escaped) and help prose name
  molbuilder's verb in words; a value only the person knows — a port, a state, a user — is said in words,
  never a `<placeholder>`; a server survey prints a detail line per port it found.
- **The path file** (`envs.recipes.SOURCE_PATH_STEP`, `molbuilder-source.pth`): the host recipe's first
  extra step; `bootstrap` runs it every time through `install.extra_step_for` (one place per
  `ops/env-framework.md` § 5), said in its dry run, counted in its exit; doctor's host check imports
  molbuilder isolated, and a failure names `bootstrap` after `repair`.

## Round 1 — three readers (correctness, a whole-repository sweep, one-home and tests)

| # | finding | verdict |
|---|---|---|
| 1 | the real-run helper read a bare `molbuilder` stop line | fixed |
| 2–6 | bare or fragment lines: `modify` refusals, `serve` pairs, `xv2xyz`/`atom-index` pointers, `jobs`, the config templates and READMEs, `checkpoint`, `migrate`, `bootstrap` flags, the transport chain, `--name` help, the test runner's repair line | fixed — through `identity` / `envs.hints`; the transport chain once, `compose.RELAX_THROUGH` |
| 7 | a printed line typed in a calculation's folder: *No module named molbuilder* (a test's PYTHONPATH hid it) | the user's design: the host env's path file; deployment § 0 |
| 8 | click refilled every indented help paragraph into run-on lines | fixed — `_kept_as_written`; every command's help walked |
| 9–17 | `monitor --help` named the bundle file; option help wrapped lines; the JSON template's escaped quotes; an alias; argv module names; the ADVICE note; a test helper letting a bare line pass; the page chapters; the config chapter | fixed |
| 18 | (the fast suite) a new import in `config_dir`, which travels beside every job and is stdlib-only | fixed — the sentence in words |

## Round 2 — three new readers (correctness, one-home and tests and sweep, pages)

| # | finding | verdict |
|---|---|---|
| 1 | a moved checkout: doctor named `repair`, which writes no path file | fixed — `bootstrap` after `repair`; deployment § 0's moved/second-clone sentences |
| 2 | deployment § 0's diagram did not parse (quotes in a mermaid label) | fixed |
| 3 | typed lines broken where click wraps help prose and option help | fixed — the verb in words; the walk fails on a broken line |
| 4 | the install script's usage said PYTHONPATH must name the repo | fixed |
| 5 | bootstrap built its own extra step | fixed — `install.extra_step_for` |
| 6 | the tests took the spelling from `identity`, so a spelling breaking M6 would pass | fixed — `support.road.TYPED`, § 5.3's words |
| 7 | the shell test's venv read the real env's path files | fixed — a venv of its own, the recipe's steps run in it |
| 8 | the run script's literal sentences untested | fixed — the gate rows pin the typed launch word for word |
| 9 | two printed lines short; the plan banner's order; the page placeholder; an untrue page sentence; the page head's order | fixed |
| 10 | the notebook pass-through's JSON escapes the quotes | dropped — data for the page's reader, which decodes it |

## The two independent full-text readers, and the discipline reader

| # | finding (reader) | verdict |
|---|---|---|
| 1 | env commands print `bash scripts/install-env.sh …`, typable only in the checkout (discipline) | the user's choice, asked at the commit |
| 2 | prose on a typed line without `#`; a label before a command (discipline, B) | fixed; the walk fails on both |
| 3 | `<placeholder>` values the shell reads as redirects (discipline) | fixed — in words; per-port survey lines |
| 4 | single quotes around the typed launch in two sentences (discipline) | fixed — backticks, every shell landing escaping them |
| 5 | typed lines past their specific identity door; local imports; notes untrue (discipline, B) | fixed |
| 6 | two help sentences broke below 80 columns (A, B) | fixed; the walk runs at 80 and 52 columns |
| 7 | the guides said the typed command runs from the clone (A) | fixed — any folder, the host env active |
| 8 | nothing pinned doctor's isolated host check (A) | fixed — a row on a scratch env the real manager made |
| 9 | `jobset summarize task` printed bare (B) | fixed |
| 10 | a second scratch-env maker in tests (discipline) | left — R4 (tests) |

## The reader of the fixes

| # | finding | verdict |
|---|---|---|
| 1 | the doctor row ran the person's own host env (the manager knows a scratch env by no name) | fixed — the snapshot handed in |
| 2 | running-a-job.md quoted the refusal in single quotes | fixed |
| 3 | no guard for a label before a typed command | fixed |
| 4 | no test saw the surveys' per-port lines | fixed |

## Evidence

- The whole fast suite: 6343 passed before round 1; 6398 after it (8 failures, one cause, fixed; the
  files rerun, 318 passed); every touched file after round 2, 933 passed; 6469 after the full-text fixes
  (2 rows judged against § 5.3 and rewritten; the files rerun, 371 passed); the last fixes' files, 262
  passed. pyflakes clean.
- End to end, after the full-text fixes: This machine, the inspectors, Task setup's prep, and the real
  SIESTA relaxation, transport decks, sequence of groups and real junction: 376 passed.
- Every new or changed row seen failing on 2.77.22 at its assertion: the help walk (`Usage: -c …`; and the
  label guard on the pre-fix probe help), the page head and runtime (`typed` missing), the gate refusals,
  the bootstrap path-file rows, the opt-in plan, the doctor rows, the shell test (`molbuilder: command
  not found`), the survey row, the advice and notify rows.
- The real host env holds no path file until the person runs `bootstrap` once; nothing in the review or
  the suite wrote into it.
