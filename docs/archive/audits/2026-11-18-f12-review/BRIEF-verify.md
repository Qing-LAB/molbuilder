# F12 round 1 — verify every finding

Thirteen readers each checked one slice of molbuilder (checkout
`the checkout`, version 2.77.2) against its contract chapters
(`docs/`) and reported defects (D…) and untrue sentences (U…), in
`r1-*.md`. Reviewers are often
wrong in a confident voice. Your job is to **re-read every finding of the
report(s) you are given, trying to refute it**, and classify it.

For EACH finding (every D and every U, and any "plausible" item):

1. Open the cited `file:line` and read the code path whole — the function,
   its callers on the road a person uses, and the chapter sentence quoted.
   Line numbers may be off; find the code by its content. The tree may have
   changed by two files since the review (`molbuilder/web/static/task-setup/viewer.js`,
   `tests/test_task_setup_one_folder_e2e.py`) — that is expected.
2. Verdict: **CONFIRMED** (you traced it yourself and it happens), **REFUTED**
   (say exactly why — the guard, the caller, the sentence that makes it not
   so), or **PLAUSIBLE** (the mechanism is there but you could not show it
   happens; say what is missing).
3. For a confirmed or plausible one, its **class**:
   - **W** — a wrong result: a wrong number, structure, deck, file or record;
     data lost; a run started wrong.
   - **C** — a crash: a traceback or an HTML 500 where a refusal in words is
     owed, or the page breaking.
   - **M** — a person told something untrue on screen or in the terminal
     (a wrong status, a wrong file named, a wrong hint).
   - **N** — a cosmetic or comfort item (a control's look, a lost zoom, a
     stale icon), or a race whose timing a person hardly meets.
   - **T** — only the chapter is wrong (the code is right): the sentence to
     change. Or the reverse — say which side should change and why.
4. **Road**: does a person reach it with what molbuilder itself writes and
   offers (the tabs, the CLI, molbuilder's own decks and files)? Or only with
   hand-made or foreign input (a file edited by hand, a deck molbuilder did not
   write, a symlink, an unusual git setting)? Say which, in a few words.
5. Note a finding that duplicates another report's (name it).

## Strict rules

- **Read only.** Do NOT create, edit, move or delete ANY file under
  `the checkout`. No `git` command that writes.
- Do NOT run pytest, `tools/testrun.py`, the server, or any `molbuilder` verb
  that writes. Do NOT touch `~/.config/molbuilder`, `~/.local/state/molbuilder`
  or `projects/`.
- You MAY run read-only Python (`<conda>/envs/molbuilder/bin/python -B -c "..."`)
  to reproduce a claim; any scratch file goes ONLY under
  `scratch-<your letters>/`.
- Do not spawn helper agents.
- Write your verdict table — and nothing else — to the path you are given:
  one row per finding: `id | verdict | class | road | one-line reason`, then
  under it a short paragraph for each CONFIRMED W or C finding: what goes
  wrong, for whom, and the smallest fix in words.
- Your final message: counts per verdict and class, and the CONFIRMED W/C
  findings in one line each.
