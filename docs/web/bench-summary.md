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

---

## 3. What you see

```
┌─ 01_coarse / bench ───────────────── 6 trials · 4 done ─┐
│  s/iter                                                 │
│    40┤ ●                                                │
│    30┤    ●                                             │
│    20┤       ●───●                                      │
│      └────────────────────────────  ranks →             │
└─────────────────────────────────────────────────────────┘

┌ G1K4C1    ● running     12.4 s/iter · 4m2s ─────────────┐
│   dDmax ╲___                                            │
│          ╲___                                           │
│   np 4 · thr 1 · a100×1 · elpa-2stage                   │
└─────────────────────────────────────────────────────────┘
┌ G0K48C1   ✗ failed          --                      ────┐
│   no SCF data — the run stopped at iteration 3          │
│   np 48 · thr 1 · no gpu                                │
└─────────────────────────────────────────────────────────┘
```

> **Not built yet: the per-trial SCF plot.** The sketch above shows one on
> each card, and the cards ship without it. Every figure that IS shown comes
> through a door that already owns it (§ 2); a convergence *series* does not —
> `parse_scf_timing` yields `s_per_iter` and `iters_measured`, both scalars.
> Drawing the curve means reading each trial's `.out` through
> `parse.registry.parse`, which is the corrected door named above, and that
> work is not done. The cards say what they have rather than implying more.

**The comparison chart** plots the sweep's verdict axis — `s/iter` — against
the coordinate the sweep actually varied, read from each trial's `point`
(`generator.md` § 4.3a). A sweep that varied nothing comparable gets the table
alone rather than a chart of one column.

The axis is chosen among the trials **being drawn**, not across the whole
sweep. `varied` answers *"what did this sweep vary"*, which is the right
question for the sweep and the wrong one for the chart: early on only a few
trials have finished and they may share a value of the first varied
coordinate. Taking `varied[0]` regardless put every finished trial at the same
x — a vertical line that looks like data, and a defect only a screenshot
caught.

**A trial card** carries its label, its state, and its headline measurement. Underneath: what it **asked** for beside what it
**ran**, because a silent eigensolver fallback is exactly what a sweep exists
to catch — `BenchPoint.mismatch` already computes that disagreement, and where
it is non-empty the card shows both.

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
never writes `bench-result.json` or `run-config.toml` — the latter especially,
because once it exists it is the *user's* file, possibly edited, and a page
that polls every 15 s must not race it. It does not promise to touch nothing:
`jobset_status` decodes each run directory, and every parser appends its own
`<input>.parse.log` sidecar by design (`parse/_log.py` — the trajectory
viewer's polling already does exactly this).
