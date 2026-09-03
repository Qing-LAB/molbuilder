# Bench summary — a whole sweep on one page

**Role:** contract
**Domain:** web

**Companions:**
[`web/results.md`](?doc=web/results.md) — the tab that hosts this view;
[`web/presenters.md`](?doc=web/presenters.md) — the registry that picks it;
[`execution/project-layout.md`](?doc=execution/project-layout.md) — what a
`bench/` directory is;
[`engines/tuning.md`](?doc=engines/tuning.md) — what the measurements mean.

A sweep answers *which configuration should I run the real thing with?* Until
now that answer arrived as a text report written after the fact, while the
trials themselves could only be opened one file at a time. This view puts the
whole sweep on one page **while it is still running**: the comparison across
trials, and each trial's own progress underneath it.

---

## 1. What it is, and where it comes from

**It is a presenter, not a mode of the tab.** `results.md` § 1 says adding a
result type is a new presenter module and never an edit to the controller, and
this obeys that: the presenter claims **`job-set.json`**, which is a real file
that a sweep already writes into its own `bench/` directory
(`job-contracts.md` § 6.3). Its presence *is* what makes a directory a sweep,
so there is nothing new to detect and no new kind of thing for the picker to
list.

Pick that file and you get the sweep; pick a trial's `.out` and you get the
trajectory viewer you always got.

---

## 2. The rule — it composes, it never recomputes

**B1 — every number on this page is read from the one place that already owns
it.** The view runs no measurement and no arithmetic of its own:

| what the page shows | the one door it comes through |
|---|---|
| which trials exist, and their coordinates | `job-set.json` via `summarize.discover_points_from_jobset` |
| what each trial asked for, ran, and measured | `summarize.parse_point` → `BenchPoint` |
| whether a trial is queued / running / finished / failed | `runstatus.jobset_status` → `StageStatus` |
| the per-trial plot | `parse.registry.parse` → `trajectory_result_to_legacy_dict` — the same reader a single file already goes through |

**B2 — a second path that computes the same figure is the defect, not the
feature.** `submission.md` § 3 records a summary that showed "170 minutes" for
five 38-minute jobs because it computed its own total a second way. A page that
compares six trials has six chances to do that, so it gets none: if a figure is
not already produced by one of the four doors above, it does not appear here
until that door is the one that produces it.

**B3 — a trial with nothing to show says so.** A failed or not-yet-started
trial still gets its card, carrying its state and whatever the run left behind.
Hiding it would answer *"where did my third trial go?"* with silence — the same
reason `submission.md` § S5 lists the queues that cannot take a job.

**B4 — the page is live, and says when it last looked.** Status is polled on
the trajectory viewer's cadence (15 s). A sweep is watched precisely while it
runs, so a page that silently showed a stale verdict would be worse than one
that showed nothing.

**Two clocks, because they fail differently** *(2026-08-25)*. The foot carries
*"last looked HH:MM:SS · measured HH:MM:SS"*: the browser's poll time, and the
server's own `generated_at` stamp on the composition. The first says the poll
is still running; it cannot say the ANSWER is old, because a response from a
cache — or from a server running six-day-old code, which is how this was
found — ticks just as freshly. A widening gap between the two is the tell, and
it is only visible if both are printed.

> **The plot's door is the PARSER, not `/api/watch/*` (corrected 2026-08-25).**
> This table named the watch endpoint, which cannot serve a sweep: that
> blueprint keeps *"a single global 'current file' dict guarded by a Lock"*
> (`watch.py`), so loading six trials would evict five of them — and clobber
> whatever the reader has open in the trajectory viewer besides. The
> single-slot state is the HTTP layer's, not the reader's:
> `parse.registry.parse` and `trajectory_result_to_legacy_dict` are ordinary
> functions. The route calls those per trial, server-side, which is § 4's rule
> anyway. So the figure still comes from the one reader that owns it — B1 is
> satisfied by naming the *reader*, and was never satisfied by naming an
> endpoint that could only hold one answer at a time.

**B5 — when trials ran on different machines, the page says so and stops
there.** *(2026-08-27, with `generator.md` § 4.4b.)* A partition is a queue
holding many machines (`scheduler.md` R0), so a sweep can span hardware without
anyone choosing that — and a GPU axis spans it *by construction*, since no
queue offers a device and its widest CPU node at once.

**What the page does about it is: show it.** Each card names its machine, and
where they differ the header says which machines are in play instead of
implying one. That is the whole of the rule.

> **It does not rank, discount, annotate or refuse** — *"speed comparison is
> speed comparison; you are not the analyzer, you present the data"* (user,
> 2026-08-27). Two machines may be exactly the comparison that was wanted. The
> page's contribution is that the reader knows which two; what the ratio means
> is theirs to decide — B1's *composes, never recomputes*, applied to a
> judgement instead of an arithmetic.
>
> **The header's one-line summary is the concrete case.** It reads *"444 atoms ·
> siesta · 128 cores · slurm"* — one core figure for the whole sweep, which is
> the sweep's **allocation**, not any trial's machine. Where trials disagree,
> one number there would be picking a winner silently.

---

## 3. What you see

```
┌─ siesta-AuBDTAu ──────────────────── 6 trials · 6 done ─┐
│  444 atoms · siesta · 128 cores · slurm                 │  ← what it is OF
│                                                         │
│  s/iter  ┤ ●                                            │
│          ┤    ●───●                                     │
│          ┤                                              │
│  GB      ┤ ●───●        peak RAM                        │
│          ┤ ○───○        peak VRAM                       │
│          ┤                                              │
│  % busy  ┤ ●───●        CPU                             │
│          ┤ ○───○        GPU SM                          │
│          └──────────────────────────  G →               │
└─────────────────────────────────────────────────────────┘

│ trials ran on 2 kinds of node:                          │  ← B5: stated,
│   48c 500G A100 (4 trials) · 128c 500G no gpu (2 trials)│     never judged
│                                                         │
┌ G4K12C1ELPA1STAGE   ● finished        62.6 s/iter ──────┐
│   np 48 · thr 1 · gpu:a100:4                            │
│   on 48c 500G A100 (sol-g042)                           │  ← what the numbers
│   peak 71.1 GB · cpu 88% · gpu 28% · vram 7.0 GB · 202 s│     below measure
│   ran with: blocksize 64 · elpa_gpu nvidia-gpu          │
└─────────────────────────────────────────────────────────┘
┌ G0K48C1ELPA2STAGE   ● finished        91.1 s/iter ──────┐
│   np 48 · thr 1 · no gpu                                │
│   on 128c 500G no gpu (sol-c221)                        │
│   peak 51.3 GB · cpu 27% · 15 s                         │  ← no GPU row:
└─────────────────────────────────────────────────────────┘     not measured,
                        last looked 21:14:02 · measured 21:13:58   not zero
```

> **Not built yet: the per-trial SCF plot.** The sketch above shows one on
> each card, and the cards ship without it. Every figure that IS shown comes
> through a door that already owns it (§ 2); a convergence *series* does not —
> `parse_scf_timing` yields `s_per_iter` and `iters_measured`, both scalars.
> Drawing the curve means reading each trial's `.out` through
> `parse.registry.parse`, which is the corrected door named above, and that
> work is not done. The cards say what they have rather than implying more.

**The comparison charts** plot every measured quantity against the coordinate
the sweep actually varied, read from each trial's `point` (`generator.md`
§ 4.3a). A sweep that varied nothing comparable gets the cards alone rather
than a chart of one column.

**Three panels, one x axis** *(2026-08-25)*, because the quantities do not
share a scale: s/iter runs 60–135, memory 7–101 GB, utilisation 0–100 %. On one
axis the percentages would be a flat line along the bottom. Stacked and
aligned, a person reads *down* a coordinate — *"at G=4 it got faster, used less
memory, and the GPU still sat at 28%"* — which is the sentence a benchmark
exists to produce.

**A series the monitor could not measure is absent, never drawn as zero.** A
CPU-only shelf has no `gpu_sm_mean_pct`; plotting it at 0 would read as
*measured, and idle* — the opposite of the truth. A panel whose every series is
absent is dropped.

The axis is chosen among the trials **being drawn**, not across the whole
sweep. `varied` answers *"what did this sweep vary"*, which is the right
question for the sweep and the wrong one for the chart: early on only a few
trials have finished and they may share a value of the first varied
coordinate. Taking `varied[0]` regardless put every finished trial at the same
x — a vertical line that looks like data, and a defect only a screenshot
caught.

**A trial card** carries its label, its state, and its headline measurement.
Underneath, in the order a person reads them:

| line | from | why it is there |
|---|---|---|
| the knobs | `knobs`, `bound` | what it was asked to run |
| **where it ran** | `machine_brief` — the monitor's `[MACHINE]` line, spelled once by the composer (`scheduler.md` R12) | *what the other numbers on this card are a measurement of.* A queue holds many machines (R0), so two cards of one sweep can name two — see **B5**. The header's own machine fact comes from the composer's `machines` census; `effective.node_phys_cores` remains only as the fallback for records that predate the `[MACHINE]` line |
| **what it used** | `metrics` — `peak_rss_gb`, `cpu_mean_pct`, `gpu_sm_mean_pct`, `gpu_vram_peak_gb`, `wall_s` | the numbers a RUN script is written from: how much memory to ask for (the peak, not the request), and whether the accelerator paid for itself. **All of these are the JOB's own since 2026-08-26** — read from its cgroup, with `cpu_mean_pct` a fraction of the cores it held. Before that they were the node's, so `peak_rss_gb` included every other job on the machine and a trial using all 48 of its cores on a 128-core node read 32%. Trials recorded earlier are **not comparable** with ones recorded since; the monitor log's `[UTIL-BASIS]` line says which basis a run used.  **Where the two MEANS come from** *(2026-09-03)*: from the monitor's own `[UTIL-SUMMARY]`, which averages every tick, when it wrote one — and from `util.csv`, whose rows are change-gated, when it did not (a killed trial has no summary).  The record's `util_basis` field says which, so an exact figure is never mistaken for a reconstruction |
| **ran with** | `effective` | the settled truth — the block size SIESTA chose, the ELPA build that answered |
| asked vs ran | `mismatch` | only the DISAGREEMENTS |

The middle two shipped on 2026-08-25. Every field had been crossing the wire
since the feature landed and the card drew none of them, so the page reported
*"62.6 s/iter"* with no way to see that the job asked for 256 GB and peaked at
71, or that the GPU sat at 28 % while the CPU ran at 88. `mismatch` shows only
what disagreed; `effective` is what actually happened, and a sweep over
solvers is largely a question about those values.

A silent eigensolver fallback is exactly what a sweep exists to catch —
`BenchPoint.mismatch` already computes that disagreement, and where it is
non-empty the card shows both.

---

## 4. The route

`GET /api/bench/summary?path=<path to a job-set.json>` — read-only, no side
effects, safe to poll. Shape pinned in
[`web-api.md`](?doc=web/web-api.md).

Composition happens **server-side**, in one place, for the same reason B1
exists: the browser cannot import `summarize.py`, so a browser that assembled
the sweep itself would be the second path B2 forbids. The route is a thin
surface over `summarize.sweep_view` — an L2 verb, testable without a Flask
app — and keeps only what a route owns: the picker's fence (the same
`_resolve_within_roots` `results.py` imports, so there is one answer to *"may
this be read"*), the HTTP refusals, and JSON.

> **The file's directory is NOT the bundle.** `job_dir_names` hands out trial
> directories relative to the **calculation root** (`01_coarse/bench/
> bench-G1K1C1`), while a stage's sweep file sits at
> `<calc>/<NN>_<stage>/bench/job-set.json`. Resolving the trials against the
> file's own directory points at paths that do not exist — and the failure is
> silent, not loud: every trial reads `unknown`, every measurement is `None`,
> and the page shows a sweep that looks like one which simply has not started.
> `bundle_for_sweep_file` asks the naming authority where the trials should be
> and walks up until they are actually there, so the answer is checked against
> the disk rather than guessed by climbing a fixed number of levels.

**Read-only means the record and the proposal, not every byte.** `sweep_view`
never writes `bench-result.json` or `bench-recommendation.txt` — the latter especially,
because once it exists it is the *user's* file, possibly edited, and a page
that polls every 15 s must not race it. It does not promise to touch nothing:
`jobset_status` decodes each run directory, and every parser appends its own
`<input>.parse.log` sidecar by design (`parse/_log.py` — the trajectory
viewer's polling already does exactly this).
