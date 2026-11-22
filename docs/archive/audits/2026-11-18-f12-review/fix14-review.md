# F12 fix 14 (what runs in an environment is its own, 2.77.22, R3c1) — the review

**Verdict: closed with 2.77.22.** The user, 2026-11-21: *"we should never try to reach out of designated
environment"*; *"why would it try to reach to an unknown python? what's the theory behind that?"*; *"make
sure that no bare python or out-of-env python is ever tried. period. i don't think you have an excuse that
why this was not detected before in multiple reviews"*; *"apply the same rule to ctest, make, mpirun and
tleap too"*. Three answers of the user's this release: a person's typed command names the env's Python by
path (*"Env's Python by path"*); the job's program and `mpirun` are its env's own (*"Env's own, else
stop"*); the ~200 printed `molbuilder <verb>` lines are spelled in full through one printer in the next
release (*"Spell in full, one printer"*, R3c2). And *"ok remove them"*: the repository Makefile's dead
`web` targets.

## Why the reviews before did not find it

No chapter stated the rule for the whole project — only the job's own Python (`running-a-job.md` § 2.2a,
2.35.1) — so no reviewer was asked it. The reviews read each release's changes, and the places that
reached another program had not changed since June (`envs/validate.py`, the recipes' self-tests, the
notebook's start, the correction script's interpreter line, the tool router's PATH step, the Makefile's
`web` targets, broken since they were written). And no test ran any of them with another program first on
`PATH`. Now the rule is `ops/installation.md` M6, the project profile asks it of every review, and each
round of this one had a reader of the WHOLE repository for it — the readers that found what diff-readers
could not.

## What was measured

conda 26.7.1, 2026-11-21: `conda run --prefix <env> <program>`, called from a shell where an env is active,
puts `<env>/bin` on `PATH` where the active env's `bin` stood — behind whatever is ahead of it — so a
`python` (or `tleap`, `make`…) placed ahead answered, for an env without it AND for one holding its own;
from a shell with none active it puts `<env>/bin` first. The door now hands the manager no active env.

## Round 1 — four readers (correctness, a whole-repository sweep, one-home and discipline, notes and tests)

| # | finding (reader) | verdict |
|---|---|---|
| 1 | the SIESTA propor advice is a format template: the new `${CONDA_PREFIX:?}` raised KeyError, and a failed job lost its advice (correctness) | fixed — braces doubled; a test renders the advice through its door |
| 2 | a prep-protocol row expected the old deck line (notes-tests; the whole suite) | fixed — the row is the chapter's new line; a PySCF row added |
| 3 | bare `python` lines left in field tests, a test docstring, install-env.sh, the profile (one-home) | fixed |
| 4 | the screenshot tool looked up the host env's prefix instead of its own interpreter, and would start a server on the person's real config (one-home, correctness) | fixed — `sys.executable`, its own config/state/runtime folders, `MOLBUILDER_PROJECTS` |
| 5 | notes untrue: the notebook's `jupyter lab`, the CUDA probe's "every probe", the door's spelling; § 5.6 silent on M6; a tombstone (one-home, correctness) | fixed |
| 6 | the CUDA probe named no repair; `repair` dropped the door's refusal (correctness, notes-tests) | fixed — `doctor` named; `install.unlaunched_reason` shared |
| 7 | the notebook's refusal to its log, the unknown-folder refusal, the PySCF line: untested (notes-tests) | fixed — rows |
| 8 | `envs install --dry-run` printed a bare `python` (sweep) | fixed — `builds.as_carried` |
| 9 | `pip install` remedies in the auth providers and `structure.py` (sweep) | fixed — the env's repair |
| 10 | the notebook kernel could be a person's `--user` kernel on another Python (sweep, measured) | fixed — `JUPYTER_PREFER_ENV_PATH=1` in `jupyter.notebook_environment` |
| 11 | the user's word mid-round: `ctest`, `make`, `mpirun`, `tleap` | done — the door hands no active env; validate's probes by path; the router never the host's PATH; self-tests by path |
| 12 | the user's answer: the job's program and `mpirun` | done — the run script's program from a folder of the env, its `mpirun` the env's, else it stops |

## Round 2 — three new readers (correctness, a whole-repository sweep of every program, one-home and tests)

| # | finding (reader) | verdict |
|---|---|---|
| 1 | `envs validate`'s ELPA check ran its validators under PATH's `mpiexec` — the machine's MPI (correctness, sweep) | fixed — through the door, the env's `make` and `mpiexec` by path |
| 2 | the GPU SIESTA self-test left `siesta` to PATH; a decoy passed it OK (all three) | fixed — the hook's folder by path |
| 3 | the job's launch under its env's own `mpirun` unpinned (a mutation passed 152 tests) (one-home-tests) | fixed — a launch-gate row |
| 4 | the SIESTA deck's by-hand line; source builds' bare `cmake`/`make`/compilers; `npm`; the repo Makefile and hook (sweep) | fixed — by path |
| 5 | notes: the router's docstring, `diagnostics`, `runtime_config`, the job chapter's lists and example log, `job-contracts` (all) | fixed |
| 6 | one home: an env's program spelled twice; the dry run re-spelled the door's rewrite (one-home) | fixed — `Capabilities.program_of`; `builds._started_python`/`_starting` |
| 7 | the ~200 printed `molbuilder <verb>` lines (sweep) | the user's choice — R3c2, the next release |
| 8 | the Makefile's `web` targets call a verb that never existed (sweep) | removed, the user's word |

## The two independent full-text readers

| # | finding (reader) | verdict |
|---|---|---|
| 1 | netCDF-Fortran's MPI compilers bare (both) | fixed |
| 2 | the job's program/`mpirun` stops named no repair (both) | fixed — `runwrap._env_repair`, one sentence for every env stop |
| 3 | "Nothing was started" untrue where MPS started first (A) | fixed — "the engine was not started" |
| 4 | the Makefile printed a `pip install` remedy; the hook's missing `node` said nothing (A, B) | fixed — said, no install named |
| 5 | install-env.sh notes; env-framework § 5.4/§ 5.6; running-a-job § 2.1 (A, B) | fixed |

## The reader of the fixes

| # | finding | verdict |
|---|---|---|
| 1 | the pre-commit hook's new message held `: `, so `.pre-commit-config.yaml` no longer parsed and every hook would fail for a person who installed them | fixed — the message reworded; the file parsed with a YAML loader |
| 2 | "before anything starts" left in `job-contracts.md` § 2.6, the changelog row and a test docstring | fixed — "before its engine starts" |
| 3 | the sequence-of-groups walk's env stop named no repair | fixed — `runwrap.env_repair` (now public), the host env's |
| 4 | `_program_launcher_block`'s `recipe=` undocumented; a sentence starting in lowercase | fixed |

## Evidence

- The whole fast suite: 6313 passed before round 1 (1 row expecting the old deck line); 6327, 6329 after
  round 1; 6336 after round 2 (2 rows standing in `cmake` on PATH, moved to the env's bin); 6338 after the
  full-text readers' fixes; 6338 after the fixes reader's. pyflakes clean throughout.
- Real-engine SIESTA relaxation and transport tests: 52 passed after the job-side change, after round 2 and
  after the full-text fixes.
- Every new row seen failing on 2.77.21 at its assertion (the decoy reached, the site program run, the old
  deck line, the interpreter line); the notebook-kernel row there fails at import, its door being new.
