# E2E fix brief — revise your group's e2e tests to pin the contract as it reads today

The e2e review is done. Each group's report is in /home/qqing/.claude/jobs/142ef3ba/tmp/e2erev/<GROUP>.md
(per-test verdicts, evidence, proposed revisions, OWED), and the findings I checked by hand are in
/home/qqing/.claude/jobs/142ef3ba/tmp/e2erev/verified.md. Your job: make the revisions in YOUR files only.
The e2e batch will be run ONCE after all fixes (by me) — you cannot run it.

## Hard rules
- Edit ONLY the files named in your task. Never edit `tests/support/*`, `tests/conftest.py`, `tests/_road.py`,
  `tests/data/*` (unless your task names a data file), any `molbuilder/` code, or any doc. If a revision needs a
  shared helper or a product/doc change, do NOT make it: list it under SHARED in your report.
- Never run e2e tests, an engine, a browser, a server, `molbuilder`/`jobset` commands, or `tools/testrun.py`.
  Never import `tests/` code in a script. Never touch `~/.config/molbuilder`, `~/.local/state/molbuilder`,
  `projects/`.
- You MAY run, from /home/qqing/molbuilder, exactly:
  - `/home/qqing/miniconda3/envs/molbuilder/bin/python -m pyflakes <your files>`
  - `HOME=/home/qqing/.claude/jobs/142ef3ba/tmp/fakehome XDG_CONFIG_HOME=/home/qqing/.claude/jobs/142ef3ba/tmp/xdg XDG_STATE_HOME=/home/qqing/.claude/jobs/142ef3ba/tmp/xdgs /home/qqing/miniconda3/envs/molbuilder/bin/python -m pytest --collect-only -q -p no:cacheprovider <your files>`
  Both must be clean when you finish.
- Read before editing: your group's report whole, `docs/process/testing.md` § 3b and § 5, each file whole, and
  the contract sentence behind every expectation you write. Read code paths you rely on (molbuilder/…).
- Match the files' own style (docstrings state GOAL, the contract §, and the silent failure; a MUTATION line where
  the file uses them). Keep each edit minimal and local.

## What to do (in this order)
1. **WRONG DETAIL / STALE ROAD / vacuous assertions** (a test that will fail, or passes checking nothing): fix
   each to the contract as it reads today — the expectation comes from the contract sentence (quote its § in the
   docstring), never from what the code printed. A negative assertion whose target text no longer exists anywhere
   is replaced by the positive assertion the contract states, or deleted if none.
2. **RETIRE** what the report says guards no failure today (pure duplicates, a pane that mounts nowhere, a test on
   a faked failure that is also wrong). Delete it whole — no skip, no xfail. Remove imports/helpers left unused.
3. **NO GOAL / stale citations / wrong docstrings** (an archived plan cited, a renumbered §, a stale reason): fix.
   Cite the LIVE chapter and §.
4. **OWED — add only items that meet ALL of**: (a) a rule changed in releases 2.57.0–2.65.3 (or named untested in
   those releases' changelog rows); (b) only the engine or the browser can show it; (c) it sits on a run or page
   your files ALREADY make — no new engine run, no new fixture that launches an engine; (d) a contract sentence
   states the expected value — quote it. Prefer one more assertion in an existing test on that run over a new
   test; read artifacts through molbuilder's reader doors (`runs.run_of`, `run_info`, the parsers), not your own
   regex. Each added check names in its docstring the failure it guards.
5. **Leave for F9** (do NOT change now; list in the report): handcrafted inputs in tests that are otherwise right
   (planted sessionStorage / `setShared` instead of clicks, the `Capabilities.file_picker_roots` patch,
   `page.route` fakes, synthetic DOM events, a typed GPU sampler, a forged mtime), NOT-E2E moves to tier 1,
   table-row merges of tests that are each right, text-pinning on copy that is still right.
6. **CHANGES WITH 2.65.4** (transport only): do not touch; I make them with 2.65.4.

## Contract and product text already corrected (pin these, not the old words)
- `results.md` § 6.3 + the GPU notice (`lib/system-load-monitor.js`): the server asks `nvidia-smi` again every few
  seconds; the notice says "The server asks the driver again every few seconds: once the host is fixed, this
  clears by itself." — no "restart the server".
- `molview.md` § 1.1 / § 11.2 / § 11.3 / test map, `workspace.md` § 3: **every edit lays down its own point**;
  Retract steps back one edit; an operation that declares no point leaves work no point holds (the badge), which
  the first Retract discards.
- `running-a-job.md` § 1/§ 3.6: the banner's labels are padded to the longest label that banner prints — a
  test reads a banner line by its label and value, whitespace-tolerant (e.g. `re.search(r"^\s*Wall\s*: 20s
  from the job's start", text, re.M)`), never a fixed column.
- `transport.md` § 3.1: under `unrestricted` a stated count is written as stated, TranSIESTA's stop said as
  information, never refused (ES6).
- `presenters.md` § 1–2: the trajectory heading = the folder answer's `engine_words` + `noun` (*SIESTA
  optimization*, *PySCF vibration*), engine words alone *SIESTA trajectory*, neither *Trajectory*.
- the chemistry card (`chemistry-correctness.md` l.~409, `/api/structure/analyze`): a PySCF method the form leaves
  blank is said **`default`** ("DFT — default"), a stated one "stated"; SIESTA's has no line (rule).
- `notifications.md` § 3: a save that succeeds lowers the save-error row (only that row).
- `tabs.md` § 5.5: the Documents reader starts dark; the ◑/◐ choice is kept for later pages in that browser.
- `trajectory.md` § 5: coming back to the tab (pageshow from cache, or visible again) asks
  `/api/watch/data` once, live run or not; a run no longer live is still not polled after that.
- `task-setup.md` § 5.2: a stored value the kind does not offer is shown as itself, marked
  "(not offered here)".

## Decisions already taken (apply them)
- `test_monitor_start_is_logged_e2e.py`: the two tests that rewrite a member of the prepared `mb_monitor.pyz`
  stand on a hand-edited product and now count wrongly → RETIRE both; keep the first test.
- `test_source_inspector_e2e.py`: the text pane mounts nowhere (`presenters.md` § 3) → retire the whole file.
- `test_notebook_through_one_port_e2e.py` outward-address test: the sandbox guard forbids any connection off
  loopback; assert the OUTCOME instead — the notebook's listening socket is bound to loopback only (e.g.
  `psutil.net_connections(kind="inet")` filtered by the notebook's port, every `laddr.ip` loopback) — no
  connection off this machine.
- `test_system_load_monitor_e2e.py`: keep the in-process GPU fake for now (F9 rules on it); fix the wrong
  "restart the server" assertion to the new notice; drop the tooltip copy pin.
- `test_wall_reaches_the_engine_e2e.py` test 2 (forged `.XV` mtime): leave for F9.

## Report
Write /home/qqing/.claude/jobs/142ef3ba/tmp/e2erev/fix-<GROUP>.md:
1. Per test touched: `test (file:line after edit) | action (fixed / retired / added / docstring) | contract §
   | what it now guards, and one code change that would turn it red`.
2. SHARED: helper/product/doc changes a revision needs that you did not make.
3. F9: what you left, one line each.
4. The pyflakes and collect-only results (counts).
Reply with a short summary only.
