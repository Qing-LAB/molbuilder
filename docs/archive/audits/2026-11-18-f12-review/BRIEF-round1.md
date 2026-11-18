# F12 review, round 1 — correctness: each contract chapter against the code it governs

molbuilder (checkout `the checkout`, commit f578482db, version 2.77.2) is
a tool that builds molecular structures and describes, prepares, launches and
reads SIESTA / PySCF / TranSIESTA calculations (optimization, vibration,
transport), through a browser UI and a CLI. Its contracts are the chapters
under `docs/` (start from `docs/README.md` for the map). This is the closing
review of the finish line (items F0–F11 in `docs/plans/plan.md`): the product
as it stands now.

## Your job

You are given ONE slice: a set of contract chapters and the code they govern.

1. Read every chapter of your slice **end to end** first.
2. Read the code of your slice **whole** — every function of every listed
   module — and follow callers/callees outside the slice only as far as you
   need to trace a claim.
3. Report:
   - **Defects**: where the code does not do what its chapter says, or gives a
     wrong answer, crashes with a traceback, writes a wrong or partial file,
     loses a person's data, or shows a person something untrue — on a road a
     person actually uses (the browser tabs, the `molbuilder` CLI, a run).
   - **Untrue sentences**: a chapter sentence the code contradicts (listed
     separately; the code may be right and the chapter wrong).

For EVERY finding give: `file:line`; the chapter and section, with a short
quote (one or two lines) of the rule; what the code does instead, **as read
in the code** (quote the lines); a concrete failure scenario (the input or
state → the wrong result); and how sure you are (read-and-traced, or
plausible). A finding you could not trace through the code is marked
*plausible* and says what is missing.

## Not findings — leave these out

- Style, naming taste, refactor wishes, new features, "could be cleaner".
- Speculation not traced through the code.
- Anything proposing to merge, share code between, or compare the MolView and
  VibrationView packages: they are sealed apart by design.
- An engine's own physics or its own refusals: molbuilder records and shows
  them, it does not second-guess the engine.
- A scientific setting the person chooses: molbuilder writes it as set.

## Strict rules — a test batch is running on this checkout

- **Read only.** Do NOT create, edit, move or delete ANY file under
  `the checkout` (docs included) — a change in the checkout breaks the
  running batch. Do not run `git checkout`, `git stash` or anything that
  writes to the repository.
- Do NOT run pytest, `tools/testrun.py`, the server, or any `molbuilder` verb
  that writes (`prep`, `launch`, `describe`, `summarize`, `migrate`, `serve`,
  `jobs`). Do NOT read or write `~/.config/molbuilder`,
  `~/.local/state/molbuilder` or `projects/`.
- You MAY run read-only Python to check a claim:
  `<conda>/envs/molbuilder/bin/python -c "..."` importing
  `molbuilder`, provided it writes nothing anywhere.
- Write your report — and nothing else — to the path you are given under
  ``.
- Plain language. Your final message: the number of defects and untrue
  sentences, and the most severe five in one line each.
