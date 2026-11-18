# F12 review, round 1 — slice C: run reports, checkpoints, project layout, the run-file catalogue

Checkout `the checkout` at f578482db (2.77.2), read only.

**Read end to end:** `docs/execution/run-reports.md`, `checkpointing.md`, `project-layout.md`,
`generator.md`, `worked-example.md`, `overview.md`.
**Read whole:** `molbuilder/monitor.py`, `monitor_log.py`, `pipeline_log.py`, `report_cards.py`
(+ `data/report-cards.json`), `report_fields.py`, `checkpoint.py`, `projects.py`, `paths.py`,
`warmfiles.py`, `runfiles.py` (grammar, finders, row machinery; the `WRITTEN`/`ENGINE_FILES` data rows
checked by tool, below), `catalogue_tables.py`, `trajectory_log/` (all four modules).
**Followed outside the slice as far as a claim needed:** `runrecord.py` (notified record, launch record,
`sent_wall_s`), `runwrap._monitor_block` / `monitor_companions`, `web/blueprints/notify.py`, `web/app.py`
(listener registration), `cli.py` (checkpoint verbs, `notify-token`, `monitor`), `runtime_config`
(checkpoint section), `calcdirs.calculation_above` / `Where`, `jobset/materialize.py` (run numbering),
`jobset/shelves.py`, `molwatch_reader.now`.

**Read-only checks run** (nothing written): `python -m tools.manifest --check` → exit 0 (project-layout § 5
equals the catalogue); every `door` and every `reads` reference of `runfiles.WRITTEN` / `ENGINE_FILES`
resolves to an existing function; the shipped checkpoint families, scratch and never-carried families,
`TRIAL_PINS`, `GRID_AXES`, `CANONICAL_TOPICS`, the catalogue's item counts (134: 72 / 45 / 12 / 5) and
the monitor bundle's module list all agree with the chapters.

**Totals: 3 defects, 4 untrue sentences.**

---

## Defects

### D1 — git's own ignore settings decide what a save keeps: a small file they match is in no state, never shown unsaved, and deleted without warning by a restore  *(high; read-and-traced, git behaviour per git's documentation, not run here)*

- **Where:** `molbuilder/checkpoint.py:353-358` (`_run_git` passes the process environment through
  untouched), `:1765-1768` (save), `:1536-1537` (status), `:1901-1907` (restore's sweep).
- **The rules:** `checkpointing.md` § 2.1 — *"Checkpoint never chooses | whether a file is stored"*;
  S1 (l. 1044) — *"every regular file is in git or in the archive: never both, never neither"* but the two
  stores, an engine's scratch and molbuilder's temporary files; S1 (l. 1084) — *"No other category is
  exempt"*; I2a (l. 1218) — *"a restore is decided by what the save recorded, never by configuration"*;
  A5 (l. 1365) — *"warns first about anything that will be lost"*; § 7 (l. 626) — *"The decision ledger is
  part of the state"*.
- **What the code does:** the only thing that sends a small file to a store is
  ```python
  _run_git(["add", "-A", "--", ".", *excludes], cwd=self.path)
  ```
  and `git add -A` skips every path git considers ignored — not only the generated block, but a person's own
  lines outside it (which S1a says are *"left alone"*), `.git/info/exclude`, and the person's **global**
  excludes file (`core.excludesFile`, by default `~/.config/git/ignore`). `_run_git` sets only the author
  identity (`env.setdefault("GIT_AUTHOR_NAME", ...)`); nothing neutralises those sources (grep for
  `excludesFile` / `info/exclude` / `GIT_CONFIG_GLOBAL` in `molbuilder/` finds nothing). A small file so
  matched is not big (`big_files` only takes files over the limit), so it reaches neither store. `status`
  asks `git status --porcelain -z -uall`, which does not list ignored files, so it is never *unsaved*.
  `restore` then sweeps:
  ```python
  if never_carried(path.name) or (key not in tracked and key not in expected):
      path.unlink()
  ```
  — the file is neither tracked nor in the MANIFEST, so it is deleted, and since `status(deep=True)` never
  saw it, the A5 question never names it.
- **Failure scenario:** a person whose global git ignore holds `*.out` and `*.log` (common entries). Every
  save omits each run's `<base>.out` (under 10 MB), `.runwrap.log`, `.monitor.log`, `.scf-timing.log`,
  `.molwatch.log`, the pipeline logs and `jobset-decisions.log`; the sidebar pill reads `saved`;
  `checkpoint restore <any state>` (asked or `--force`) deletes all of them from every run folder, with no
  warning, and no state can bring them back.
- **Same root, plausible (not traced past the call):** a global `commit.gpgsign = true` or `core.hooksPath`
  applies to `git commit` at `checkpoint.py:1802` and `git tag -a` at `:1983`; with no gpg agent (the web
  server) the commit fails, `save_before` raises, and per § 9 the prep / first launch / Save is stopped.

### D2 — a run's `start` card never states its wall  *(medium-low; read-and-traced)*

- **Where:** `molbuilder/monitor.py:2284-2285` and `:2382`; `:3074-3095` (`_notify_main`).
- **The rule:** `run-reports.md` § 4.1c (l. 1065) — *"A start card and a test card are the title and the
  summary line, with no field grid: the summary says where the run is and its wall"*.
- **What the code does:** the start event is sent from `st0` before the wall is ever read:
  ```python
  if not member:
      _said(st0, "start", start)        # l. 2285 -- st0.wall_s is still None
  ...
  _count_down(st0, start)               # l. 2382 -- the first read of the wall
  ```
  `JobStatus.as_text` prints *time limit …* only `if self.wall_s is not None and self.wall_left_s is not None`,
  so the start card's summary (and the `[MONITOR] start` / `[NOTIFY] (stub) start` lines) never carry it.
  A run of points' `start`, sent by `_notify_main`, builds `JobStatus(state=a.state, detail=a.detail)` and
  never counts the wall at all.
- **Failure scenario:** a job sent to a queue with `--time 12:00:00`, its `run.json` written before the
  monitor starts; the Slack/Discord start card reads `running | the run 0s`, without *time limit 12h: 12h left*.

### D3 — `molbuilder notify-token` tells the person a failed report leaves no trace  *(low; read-and-traced)*

- **Where:** `molbuilder/cli.py:2236-2240`, `:2246-2247`.
- **The rule:** `run-reports.md` § 3 (l. 735) — *"A failure costs the run nothing … and is said in the
  monitor log"*; § 4.4 (l. 1149) — *"every report gets a 404 — costing the run nothing, said in its
  monitor log"*.
- **What the code prints:** *"until it is restarted every report gets a 404 -- and a notifier is silent on
  failure, so you would see no error, just no reports."* and *"Every key issued under the old segment stops
  working, and silently."* The monitor does write `[NOTIFY] channel 'molbuilder': start not delivered -- the
  service answered HTTP 404` (`monitor.py:2017-2020`).
- **Failure scenario:** a person issues the first key without restarting, sees no reports, and — told
  there is no error to find — never opens the run's monitor log that names the 404.

---

## Untrue sentences

### U1 — `checkpointing.md` § 15 (l. 1936) and § 13.4 (l. 1878): *"`CheckpointError` and its five subclasses … `CalculationNameError` and `NestedRepoRefusedError` are the user's to resolve; `GitNotInstalledError` and the rest are faults"*  *(read-and-traced)*
`checkpoint.py:230-265` defines six: `InputError`, `GitNotInstalledError`, `DirtyWorkingTreeError`,
`NoSuchRefError(InputError)`, `NestedRepoRefusedError(InputError)`, `CalculationNameError(InputError)`.
"The rest are faults" is untrue for `NoSuchRefError` and `InputError` itself (blank note, a tag name taken),
which every surface buckets as the person's — the CLI exits 2 for them (`cli.py` `except InputError: sys.exit(2)`),
as § 5 *Exit codes* says (*"an unknown state"* is the person's input).

### U2 — `run-reports.md` § 3.0 (l. 808-809): *"A machine with no channels file at all is said the same way, once per run (`[NOTIFY] no channels set up on this machine …`)"*  *(read-and-traced)*
For a run of points the line is said far more than once: each point's member monitor calls
`_install_env_notifiers` → `load_channels` (`monitor.py:2269`, `:1802`) into its point's monitor log, and
every event the `notify` verb sends (`start`, each `milestone`, each new `part_failed` reason, `finish`)
calls it again into the run root's monitor log (`monitor.py:3070-3073`). A transport sweep of 12 points
with `channels: ["*"]` and no channels file writes it 12 times in the points' logs and once per event at the root.

### U3 — `project-layout.md` § 4.3 (l. 1079-1080) and invariant 4b (l. 1447): a run's number is *"never reused"*  *(read-and-traced)*
The next number is the newest folder's plus one (`jobset/materialize.py:242-256`, `resolve_attempt`;
`paths.attempts_in`). A restore to a state that never held `01_coarse/run-2/` removes that folder whole
(`checkpoint.py:1901-1920`, as `checkpointing.md` § 7 says), and the next launch opens `run-2/` again —
a second, different `run-2` beside the one a later state holds. (A run a refused send opened is also
refilled under its number, § 1.6.2 — arguably not a reuse of a run that happened.)

### U4 — `project-layout.md` invariant 15 (l. 1531): *"Every big regular file is in git or in the archive, never both, never neither — but an engine's scratch"*  *(read-and-traced)*
A big file is never in git (S1b; `checkpoint.big_files` → archive), so "big" misstates S1, which covers
every regular file; and the exemption omits molbuilder's temporary files, which S1 (item 3) and the code
(`warmfiles.never_carried` = scratch **or** `runfiles.is_transient`) also keep in neither store.

---

## Checked and agreeing (no finding)

- Monitor: events and who sends each (§ 2 table), change-gated `[STATUS]`/`util.csv`, N-hour clock and the
  member's re-read of the record, `scf_converged` only on geometry-moving steps (`step_moves`), the step-
  and pace-based `wall_near` texts, the `[WALL]` "not counted" cases, `run_status`/closing-record order,
  `[UTIL-SUMMARY]` verdict bands, `[UTIL-BASIS]` wording, cgroup/tree basis rules, `notified.json` lock and
  dedup of `start` / `part_failed` reasons, the summary-line order (§ 2.8), `--notify-*` flag semantics
  (absent / `*` / `""`), `MB_NOTIFY_URL` override bypassing `channels: []`, `address_refusal`,
  `delivery_failure` never quoting the address, the three envelopes, User-Agent, signing.
- Listener (`web/blueprints/notify.py`): 404 for every refusal, 8 KB cap before parse, NaN-safe 15-min skew,
  per-key 60/min, fixed field set, 500-char cut, `v`/`user`/`received_at`, 1 MB × 5 rotation; registered
  only when the key file names a route and holds keys.
- Card layout: shipped file matches § 4.1c (Slack's swap of `n_iters`/`per_iter_s` and `geom_step` first),
  every load-time refusal § 4.1c lists.
- Checkpoint: save order (§ 6), shallow check on reuse / deep on restore, staging dir per saver, empty-MANIFEST
  trailer, `prune_scratch` never failing a save, restore order and sweep, `--force` skipping the question,
  `list` printing states before damage and exiting 1, `save_before` note and its two printed lines, L1/L3
  refusals, `name_engine` never replacing.
- Catalogue/layout: `tools.manifest --check` clean; all doors resolve; topics, trial/attempt/bench naming,
  `GroupNames.of_*`, shelf order (`_width`), trial pins, catalogue counts.

## Observation, not counted (the code does what its chapter says)

A `<config dir>/report-cards.json` copy that does not read stops the monitor at its start
(`monitor.py:2262-2266`, as § 4.1c says) — for every run on that machine, including runs with no `notify`
block — so those runs get no `util.csv`, no `[STATUS]` lines and no `[MONITOR] job ended` closing record,
the evidence a forced stop is read by. § 3 argues the opposite for the channels file (*"the tail wagging
the dog"*); § 2.5's tag table also lacks the `[MONITOR] the card's layout is refused` line. A decision, if
anyone wants one.
