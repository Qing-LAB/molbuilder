# F12 review, round 2 — holistic: one door per fact, one data shape, nothing duplicated or handcrafted

molbuilder (checkout `the checkout`, version 2.77.2) builds molecular
structures and describes, prepares, launches and reads SIESTA / PySCF /
TranSIESTA calculations through a browser UI and a CLI. This is the closing
holistic review of the finish line (F0–F11 in `docs/plans/plan.md`). Its
rules: `docs/architecture.md` and `docs/execution/architecture.md` (the
layers and the module doors — § 3.2's table names the owner door of each
fact), `docs/process/conventions.md`, `docs/process/testing.md`.

## Your job

You are given whole files — a slice of the coverage table. Read each one
**whole**, then, for each, ask:

1. **One door per fact.** Is any fact (a file name, a path rule, a vocabulary,
   a default, a unit, a parse of an engine's output, a rule of the contract)
   computed here when an owner door already answers it elsewhere? Name both
   places (`file:line` each) and which is the owner (the architecture table,
   or the module that the most readers ask).
2. **One data shape.** Is the same record carried in two shapes (a dict here,
   a dataclass there; two spellings of the same key; a list re-keyed by hand)?
3. **Duplicated or redundant code.** Two functions doing the same job;
   a helper that re-implements a library/vendored function or a framework
   door; dead branches; a parameter nobody passes.
4. **Handcrafted code.** A branch on a name (an engine, a kind, a file name)
   in shared code where a table row/record should decide; a hand-kept list
   that a table already holds; a value typed in that a door derives.
5. **In tests too**: a test that greps source or doc text instead of checking
   an outcome; hand-typed input where the road (the real writer) should make
   it; a fixture/prelude copied per file instead of the shared support door
   (`tests/support/`); two tests pinning the same thing.

For EVERY finding: both `file:line` places (quote the lines), which is the
owner, what goes wrong or could drift (a concrete case: change X here and Y
there silently disagrees), and the smallest fix in words (no code). Say how
sure you are (read both sides, or plausible).

## Not findings — leave these out

- Style, naming taste, line length, docstring wording.
- New features, or a rewrite of a working design you merely prefer.
- Anything proposing to merge, share code between, or compare the MolView and
  VibrationView packages: they are sealed apart by design.
- A fact shown twice FROM ONE SOURCE (that is fine); only two SOURCES is the
  defect.
- Correctness against the contract (round 1 covered it) — unless it is caused
  by a duplicate.

## Strict rules

- **Read only.** Do NOT create, edit, move or delete ANY file under
  `the checkout` (docs included). No `git` command that writes.
- Do NOT run pytest, `tools/testrun.py`, the server, or any `molbuilder` verb
  that writes. Do NOT read or write `~/.config/molbuilder`,
  `~/.local/state/molbuilder` or `projects/`.
- You MAY run read-only searches (grep, find) and read-only Python
  (`<conda>/envs/molbuilder/bin/python -c "..."`) that writes
  nothing.
- Write your report — and nothing else — to the path you are given under
  ``.
- Plain language. Your final message: the number of findings, and the five
  that matter most in one line each.

## Your slice's tests

Read, whole, the test files that import or drive your slice's modules: find
them with `grep -rl "<module name>" tests` (and, for
page scripts, the `*_js.py` and `*_e2e.py` files that open those pages), and
apply point 5 to them. Do not spawn helper agents (a running limit applies);
if your slice is too large to read whole, say exactly which files you read
whole, in part, or not at all.

Round 1 (correctness) already reported in `r1-*.md`;
do not repeat its findings unless a duplicate is their cause.
