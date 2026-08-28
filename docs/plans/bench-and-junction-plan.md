# Benchmark measurement + junction placement — the standing work

**Role:** plan (partly built, partly designed)
**Domain:** execution · science · web
**Started:** 2026-08-26
**Companions:** [`model/parse.md`](?doc=model/parse.md) § 2b — how a run ended ·
[`science/junction-cell.md`](?doc=science/junction-cell.md) §§ 3.1–3.3 — the seam ·
[`web/bench-summary.md`](?doc=web/bench-summary.md) — the sweep page

**Why this file exists.** A long session found more than it fixed, and the
findings are worth more than the memory of them. Everything below is either
verified in code, measured on a real sweep, or explicitly marked as an open
question. **Nothing here is a guess** — where something is unverified it says
so, and says what would settle it.

---

## 0. State when this was written

| | |
|---|---|
| pushed | `0a221f77` |
| committed, unpushed | 4 — the monitor PID fix, the config-directory consolidation, notifications, the listener |
| uncommitted | docs only: this plan, `run-reports.md` § 4, `access-control.md` § 8 |
| lane | last full run **7588 ran, 0 fail, 6 skip**; doc guards green since |

**Consolidated 2026-08-26.** Three of the findings recorded that morning
(§ 3 A, C, D) turned out to be one metric measuring the wrong thing — see
§ 2.12. Two proposals of mine were withdrawn the same day and both are recorded
as withdrawn rather than deleted: preserving `sinfo`'s `+` range (§ 2.13 P4),
and a cleverer name for the listener route (§ 2.11).

---

## 1. Shipped 2026-08-26

Committed in five splits and lane-verified (**7588 ran, 0 fail**). Kept as a
record of what moved, not as work.

1. **The run-ending vocabulary** (`parse.md` § 2b) — `run_state` says how a run
   *ended* and never whether it was any good, which is what made six healthy
   benchmark trials read as failures. `scf_converged` reported beside it.
2. **`parse/engines/_run_ending.py`** — one marker table, stdlib-only;
   `summarize.py`'s private copy deleted. The monitor's own private table went
   too: it could stop sampling mid-run, because `siesta: Final energy` prints
   early.
3. **The bench display** — three stacked panels, values on bars, usage and
   `ran with` lines, two clocks.
4. **Measurement fixes** — node facts read from the run's own wrapper log,
   `iters_measured: 1` disclosed rather than hidden.
5. **Notifications and the listener** — §§ 2.9, 2.10.

> **The one that cost real time:** a `watch tail` test compared against two
> state names the rename had deleted, so it **hung the lane for 11½ hours**
> instead of failing. Fixed by `CONCLUDED` plus a `signal.alarm` guard in the
> test — a hang is now a failure.

---

## 2. To build

### 2.1 Benchmark iteration count — settable per calculation

`_MEASUREMENT_PINS` in `jobset/_cli.py` hardcodes `max_scf_iter: 3`. It is
applied over every declared value and refused as a sweep axis.

At 3 iterations `parse_scf_timing` yields **one** timing sample — iteration 1
forms no delta, the iteration-2 delta is dropped as warm-up-adjacent. So the
verdict ranks on n=1 with no spread, and the Au-BDT-Au winner leads by 5%.

| iterations | deltas | samples averaged |
|---|---|---|
| 3 (today) | 2 | **1** |
| 5 | 4 | 3 |
| 8 | 7 | 6 |

**Design.** A scalar in `task.json`'s `bench` block — `"scf_iterations": 3` —
overriding the pin. It belongs to the description because it is a property of
*this calculation's* measurement, it travels to the cluster, and the
Task-setup tab already edits that block. A `--iterations` flag was rejected:
the same calculation prepped twice could then measure differently with nothing
on disk saying so.

**Keep:** default 3 (nothing existing changes); still **not sweepable**
(trials at different counts measure different things under one label);
`scf_must_converge: False` still pinned (a longer cap must still end clean).

### 2.2 CPU/GPU mean — read the monitor's own figure

`parse_util_csv` now time-weights, which reconstructs the mean by assuming a
value held constant across each gap. **True by construction, not exactly
true.** The exact figure already exists: `UtilAccum.add()` runs on *every*
tick, ungated (`monitor.py:533`), and writes the mean into `[UTIL-SUMMARY]`.

`parse_util_bound` stopped reading those numbers on 2026-08-19 justifying it
as *"a digest of the same samples `util.csv` records raw"* — **which the code
contradicts**: the digest covers every sample, the CSV only the changed ones.

**Do:** read `[UTIL-SUMMARY]` for the mean; keep time-weighting as the
fallback when the line is absent (a trial that dies before the monitor's final
write has a CSV and no summary). Correct the docstring's claim.

### 2.3 Junction placement — `mirror` | `translate`

The geometry is `junction-cell.md` § 3.3; this is the parameter, which does not
exist yet. `modify.py:974` unconditionally mirrors for `side="-z"`.

```
mirror     z' = anchor − gap/2 − (z − z_min)      layer order FLIPS
translate  z' = z + (anchor − gap/2 − z_max)      layer order KEPT
```

**The trap, and why the two lines are not symmetric.** Translation must
reference `z_max`. Using `z_min` — the mirror's reference — displaces the slab
by its own full thickness, landing it on the far side of the molecule. Verified
on a built 6-layer Au(111) slab: the face arrives a whole slab-thickness away
from where it belongs, in the wrong direction.

**Add** `placement` to `add_electrode_slab` and `add_symmetric_electrodes`,
plus a Junction-panel control. **Default `mirror`** — every existing structure
used it, and contact symmetry is usually the intent.

### 2.4 Seam detector + warning

Nothing measures the boundary today. Put `classify_seam` in `cell.py` (which
already owns `detect_layers` / `bulk_z_period` / `STACKING_PERIOD`).

Compare the top layer against the bottom layer's image one cell up:

| verdict | evidence |
|---|---|
| `continues` | lateral step matches the in-slab step (mod 120° on 111) |
| `eclipsed` | `Δxy = (0,0)` at ~`d` |
| `twin` | correct `a/√2` bond, **reversed** step |
| `collision` | ~0 Å — the unpadded case |

**The distance alone is not the test** (`junction-cell.md` § 3.1). A twin has
the correct bulk bond length, so a distance check passes it; only the *step*
separates continuation from twin.

Surface as a **warning, not a refusal** — an eclipsed seam is wrong for a
periodic crystal and harmless in a relaxation whose outer layers are frozen.
Name which condition failed: layer count (§ 3.1) or placement (§ 3.2).

**Evidence that this will fire, and that build time is the only moment it can.**
Measured on `projects/Au-BDT-Au` — the reason this is worth building, and the
reason it belongs here rather than in the science doc:

| | as built | after relaxation |
|---|---|---|
| outer 3 layers/side | `d` = 2.4006 Å | 2.4006 Å, z-spread **0.000** — frozen by `Geometry.Constraints` |
| interior | `d` = 2.4006 Å | ≈ 2.289 Å (−4.7 %), buckled ≤ 0.26 Å |
| gap across molecule | 10.400 Å | 11.052 Å |
| **seam** | **2.4008 Å, step (0.000, 0.000)** | **2.4008 Å, step (0.000, 0.000)** |

It has 6 layers/side, so § 3.1 holds and only the mirror fails — the detector
would warn, correctly. The seam is **bit-identical** before and after: the
layers forming it are exactly the ones the relaxation pins, so relaxation
cannot touch it. That is why the warning has to arrive when the structure is
built.

(`d = a/√3` at `a` = 4.158 Å, the PBE value this junction was built with.)

---

### 2.5 One answer to "how did this run end" — decided 2026-08-26

Today there are **two** entry points: `scan_ending(text)` and
`SiestaParser.parse(path)`. Callers pick. That is drift, and the decision is to
remove the choice.

**The constraint behind the split is real.** `summarize` runs **on the target**
(`job-system.md` § 5) — a cluster where numpy is not guaranteed — so the prep
layer is stdlib-only and a door that needs arrays cannot be the only door. What
was wrong was expressing that as two public functions rather than as one
interface with two implementations behind it.

**Numpy is not the thing to remove** (checked 2026-08-26). The ending path is
already free of it — `_run_ending`, `_helpers`, `bench/result` and
`jobset/summarize` import none. `siesta.py` uses it in exactly **three** places,
all building `Frame` positions and forces, and `frame.py` needs it because those
fields *are* arrays. That is the right representation for coordinates and it
stays.

**The shape to build.** The stdlib-only scanner becomes **the** implementation —
the fundamental one, and the copy that travels with the monitor script. The
heavy parser stops carrying its own fatal-marker rules and its own precedence,
and instead **drives that scanner** with the lines it is already walking, adding
frames on top. Two entry points remain (one takes text; one takes a path and
also builds frames) but they stop being two *implementations*.

Above that sits one engine-neutral question — how did this run end — dispatching
per engine (SIESTA, PySCF, molwatch). Callers ask once and never choose.

*Why this is not cosmetic.* The two doors already disagree: a line carrying both
a non-convergence message and an abort marker reads `running` through the scan
and `stopped` through the parser, because each implements its own precedence
over a shared word list. Sharing the words was never enough — **the drift lives
in the priority**, which is exactly what one implementation removes. A parser
with no precedence of its own has nothing to disagree with. The same shape shows up five more times as `("stopped",
"out_of_memory")` written out longhand (`siesta.py` ×2, `_run_ending.py`,
`core.js` ×2); one unified answer retires all of them.

### 2.6 Out of memory is a real state — and the limit must be declared

**It is reachable, on more than one path.** The scheduler kills a job that
exceeds its allocation; a local run under an explicit memory constraint can hit
the same wall; and MPI itself can report the failure. What is missing is not the
state but the evidence: those messages land on the job's **error** stream, and
the wrapper deliberately keeps that out of the `.out` the parser reads. Read the
wrapper log — it sits beside the `.out` and already has them.

**The larger requirement: the memory limit is an explicit parameter.** Not a
number inferred from what a node happened to have free. `peak_rss_gb` is
currently `MemTotal − MemAvailable` — whole-node memory, correct only when the
job owns the node (§ 3 A). A declared limit is what makes "out of memory"
meaningful: without one, exceeding it is undefined.

*Open, worth an experiment:* whether SIESTA and the MPI runtime actually respect
a declared limit, and what they do at it. Testable locally by setting one and
watching.

### 2.7 The benchmark summary does not need to poll — decided 2026-08-26

It refreshes on a timer today. It should not: a summary of finished trials is
not a running calculation, the refresh control already exists, and the user
knows when there is more to see. Removing the timer also retires the cost of
re-reading every trial's `.out` in full on each tick.

### 2.8 A sweep must hold hardware constant — decided 2026-08-26

**Not partition pinning.** Running the benchmark on a shorter-queue partition is
deliberate and correct: you want it scheduled in minutes, not after a day, and
you pick a queue whose hardware matches the machine the real run will use. A
different queue is fine. A different *machine* inside one sweep is not — that is
what made G2-vs-G4 compare silicon as much as settings (§ 4).

**So the requirement is a hardware match — CPU, GPU and memory — and the user
owns the judgement.** The system's job is to let a benchmark state what it needs,
to record what each trial actually landed on, and to say plainly when trials in
one sweep disagree. **It never overrides the user's choice.**

### 2.9 Notifications — BUILT 2026-08-26

Moved to [`execution/run-reports.md`](?doc=execution/run-reports.md), which is
now the contract for it. A plan is the only document allowed to describe
something that does not exist (`execution/overview.md` § 1), so what exists
does not belong here.

What landed: the `notify` block in `task.json`; the monitor's triggers, its
destination file and its 2 s guarded POST; the policy riding `Resources` to
the wrapper; the Task-setup card. § 2.10 below stays because the receiving end
does not exist yet.

### 2.10 The listener — BUILT 2026-08-26

Moved to [`execution/run-reports.md`](?doc=execution/run-reports.md) § 4.
`POST /api/notify`, registered only when `notify_tokens_file` is configured;
per-user bearer tokens; append-only to a rotating log; `molbuilder
notify-token` issues them.

---

### 2.11 The listener — hardened, BUILT 2026-08-27

Contract in `execution/run-reports.md` § 4, with the rules it rests on in
`ops/access-control.md` § 8 (rule 2 amended; rules 7 and 8 new).

**Built 2026-08-27**, once the egress and certificate tests came back clean.
Four parts, one change:

1. **HMAC-SHA256 over the body** replaces the bearer token. The key stops
   travelling; what travels is valid for one body only. Stdlib both ends.
2. **The route segment is generated per deployment** by `notify-token`, not a
   word chosen in the source — this repo is public, so a fixed word is not
   obscurity. `molbuilder.json` gains `notify_route`; `notify_tokens_file`
   becomes `notify_keys_file`, and **both** are required for the route to
   exist.
3. **Every failure answers a plain `404`**, so a wrong signature is
   indistinguishable from a path that was never registered. Verified this
   keeps the limiter: `rate_limit.py:305` counts any `400 <= s < 500`, and a
   notify request sets neither `molbuilder_auth_challenge` nor an
   authenticated session.
4. **`serve` never mints or rotates these.** The counterpart lives on a
   cluster molbuilder cannot reach, and a notifier is silent on failure by
   design — so a startup rotation would stop every running job's reports with
   nothing saying why.

**Named, not fixed:** HMAC is symmetric, so reading the server's key file is
enough to forge. Ed25519 would close it and is not in the stdlib, which the
monitor is restricted to. Recorded in § 4.2 rather than left to be
rediscovered.

**A fifth part the egress test produced.** The `302` measured from a compute
node is the sign-in redirect — and it is exactly what this route would return
if it ever fell out of `auth.py`'s `_PUBLIC_ENDPOINTS`. `urlopen` **follows**
a redirect, the POST body is dropped on the way to a login page, and the
notifier sees no error at all. A guard test now holds that the route never
answers 3xx.

39 listener tests; six mutations tried, six caught — including the route
segment going unchecked, the signature covering the body but not the
timestamp, `404` reverting to `401`, and the monitor sending its key instead
of a signature.

---

### 2.14 Ask the scheduler when — BUILT 2026-08-27

User: *we should provide another verb to prove if a job is going to be
submitted, what's the waiting time/expected start time, and the user can
adjust when they know the diff* — then, on where it belongs: *instead of
submit, we can just say ask. We don't have to reinvent something.*

**It is a MODE, not a verb, and that answers two of the three open
questions at once.** `--mode ask` walks the identical path `--mode submit`
walks and inserts one flag; the line asked about **is** the line that would
be sent. A separate verb would have re-rendered the flags, and two
renderings of one fact are two things that can disagree.

It also settles *how many queries*: **one**, for the job as configured, the
same way `submit` sends one. The comparison the person wants comes from
re-running with a different `--domain` or fewer cores — which is the loop as
they described it: *come back, tune it, submit for a different cluster or
reduce those resources and see if they can get a better waiting time, or just
say okay, I can live with that.*

**The third question is answered in the code rather than by measurement.**
SLURM sometimes declines to predict. That is reported as **unknown, never as
soon** — a missing answer dressed as a good one is how a person waits a day
for a queue that looked instant — and the reason it gives is printed, because
it is often the whole answer (*"Requested node configuration is not
available"* says the ask fits no machine in that queue). No Sol command was
needed after all: every branch is exercised by feeding the parser real SLURM
output, which is pure and needs no cluster.

**Not** gated one-at-a-time — that was a misreading, caught by the user on a
4-trial bench: the rule is *about the scheduler, not about doing several
things*, and `--test-only` enqueues nothing. The sweep is where asking pays,
since a grid's trials ask for different shapes and their waits differ. Query
count is capped for politeness; anything past it is named as unasked. Nothing is recorded either — a launch
record says a job exists, and after this one does not.

17 tests; five mutations tried, five caught — including reporting an unknown
as a time, sorting the answers (which would be a recommendation), and letting
`ask` escape the one-at-a-time gate.

**Tested against Sol's real `sbatch --test-only`, 2026-08-27 — and it found a
defect.** Both branches, verbatim:

```
sbatch: Job 62266174 to start at 2026-08-27T11:22:03 a using 4 processors
        on nodes sc078 in partition htc
allocation failure: Requested node configuration is not available
```

**The refusal worked, despite the guess being wrong.** It was written against
an invented `sbatch: error: …` prefix; Sol says `allocation failure: …`. It
survives because the parser keeps the raw line rather than matching a known
prefix — a design choice that turned out to be load-bearing rather than
merely tidy.

**The prediction lost two of its three fields.** There is a token between the
timestamp and `using` — what it is, is SLURM's business. One regex chaining
the three facts with optional tails required them to be adjacent, so the time
parsed while the processor count and the node name vanished silently. They are
read independently now: whatever SLURM inserts is ignored, and one missing
field cannot take the others with it. Both lines are pinned as fixtures.

**Still not run end-to-end**, because every stage on Sol has already been
launched and `--mode ask` correctly skips those. A fresh `prep` would give an
unlaunched deck; the piece that was actually unverified — what SLURM prints —
is now measured.

---

### 2.15 A bench trial gets attempts — CONTRACT SETTLED 2026-08-27

Contract: `project-layout.md` § 1.5a. **Shape decides**, exactly as it does for
a stage — hierarchical makes `run-1/`, flat writes `-run1.out`. The sweep's
opt-out is deleted rather than a fourth case added.

**No migration, no reader for the old layout** (user: *"new dir becomes
standard. No historical burden. Get this right at this stage is cheaper than
dragging obsolete code and carry bugs."*). Existing sweeps stop being readable
by `summarize`; nothing is deleted.

**Scope, measured rather than estimated.** `job_dir_names` is the naming
authority and has **33 call sites in 5 modules**, all inside `jobset/`:
`materialize` 8, `submit` 11, `summarize` 8, `prep` 4, `runstatus` 2. Nothing
in `bench/` or the web layer resolves a trial directory itself. An earlier note
here said ~85 across 9 — that counted mentions of three different names, not
seams, and it was wrong enough to have changed the decision.

**Five pieces — 1, 2, 3, 4 and 5 BUILT 2026-08-27**, after the naming
authority was unified first (which is what made the rest small):

**1. The run index must cover every per-run artifact — BUILT 2026-08-27.** The wrapper indexes the
`.out` and the timing log; `<basename>.monitor.log` is appended and
`<basename>.util.csv` is **truncated** by `write_text`. `util.csv` is the
benchmark's measurement, so a flat re-run destroys what it set out to repeat.
The wrapper already has `_run_n` — the two monitor filenames take it too.

*Paired with the reader, or nothing is found:* `summarize.py` already uses
`_latest_run_file` for the `.out` and the timing log, and the exact unindexed
name for the other two. Those two lines change with the writer.

*This is a live defect today*, independent of sweeps: a flat ladder stage
re-run loses its `util.csv` already.

**2. The sweep stops opting out.** `submit._launch_dir` carries the special
case — *"an attempt-less dir (a sweep trial) is ITS OWN attempt"* — and with it
the refusal that started this. Deleting it lets `attempts()` and
`keeps_attempts_as_directories` answer for trials as they already do for
stages.

**3. `prep` opens an attempt for a trial** in a hierarchical calculation, the
way it does for a stage, so there is a `run-<n>` to launch into.

> **THE BLOCKER, found by attempting it 2026-08-27 and then reverting.**
> `materialize` opening the attempt is four lines and reuses `resolve_attempt`,
> the same rule stages already use — it cost **2 failures in 427**. But the
> deck did not follow, because **`prep` computes the trial directory a second
> time**, in a different function, from the same two facts rather than from
> the naming authority:
>
> ```python
> # prep_calculation, ~line 852 -- NOT job_dir_names
> _jdir = base / bench_container(_shape, token) / f"bench-{_pt(element.point)}"
> ```
>
> Its own comment states the coupling and calls it safe — *"the same one
> `job_dir_names` will answer for this job, computed from the same two facts …
> so the deck is born where the launch will look for it."* Two doors on one
> fact, kept in step by hand. They agreed until an attempt layer existed, and
> then the deck landed in the container while the shared package landed in
> `run-0`.
>
> `prep_jobset` (line 188) uses the authority; `prep_calculation` (line 852)
> restates it. **Reconciling them is the work**, and it is a refactor of the
> naming seam rather than an addition to it — which is why the attempt was
> reverted rather than half-landed. The diff is kept at
> `scratchpad/p2-attempt.diff`.
>
> **DONE 2026-08-27.** `materialize.trial_dir` is the rule, and it had **three**
> spellings, not two: `job_dir_names` composed it for a tokened trial and again
> for a tokenless sweep, and `prep_calculation` a third time. All three ask now.
>
> `prep` could not simply call `job_dir_names` — it is *building* the JobSet in
> the loop that needs the directory — which is what made a shared **rule** the
> fix rather than a shared lookup. A test asserts the two agree on a real
> bundle, which is the property the old comment asserted and nothing checked.

**4. The GROUPED launch resolves the attempt too**, and its two gates read the
deck where the deck now is. Not a follow-up: leaving it means the cold gate
goes quiet on the one door that submits several trials at once.

**5. The readers resolve the attempt.** `summarize`'s two entry points take
`bundle / dirs[name]` as the artifact directory; in hierarchical that becomes
the container and the attempt is one level in.

**What the build actually cost, against the estimate.** The layout change
touched **six modules and four test files**, and every one of the gaps
predicted below turned up — including the sharp one. The two that were *not*
predicted are the interesting ones:

* **The grouped launch's container check.** `containers = {d.parent for d in
  trial_dirs}` was one shared `bench/` and became one per trial the moment
  `d` was an attempt, so every grouped submission was refused with *"the
  sweep's trials do not share one container"*. The container question belongs
  to the naming authority, the artifact question to the attempt; they are two
  questions and each now asks the right thing.
* **The refusal gave stage advice to a trial.** With attempts, a launched
  trial reaches the branch that says *"prepare a fresh one `--from
  <attempt>`"* — which is a stage's remedy: it continues from what the last
  attempt produced. A trial does not continue; it re-measures from cold. The
  refusal now says how to do the thing that is newly possible: prep again,
  which opens `run-<n+1>` and leaves the earlier measurement untouched.

**Gaps found while writing this, and NOT closed by it:**

* **Which attempt does the summary report?** `_latest_run_file` takes the
  highest index, so a re-run silently supersedes — and comparing two
  measurements of one point is the whole reason to re-run. At minimum the
  summary must say how many attempts a trial has; § 2.15's second half (the tab)
  is where that lands.
* **`check_trial_starts_cold`** runs against the trial directory. With an
  attempt layer it must check the attempt, or a warm file left beside it in the
  container reads as cold.
* **The bench container holds a `launch/` directory** (seen on Sol:
  `01_coarse/bench/launch`). Whether that is an attempt-bearing thing or a
  sibling of the trials needs reading before step 3 touches layout.
* **`identity.py`** lists `"{label}.util.csv", "{label}_*.util.csv"` among a
  run's files. If the name gains `-runN`, that pattern list is a third place
  the spelling lives.
* **THE GROUPED LAUNCH IS A SECOND PATH, and the sharp one.** A bench can be
  submitted in groups — `bench/launch/bench-group-cpu.sbatch` and friends,
  seen on Sol — and `submit.py:893` builds its trial directories itself
  (`base / dirs[j.name]`) rather than through `_launch_dir`. Two consequences,
  the second worse than the first:
  - the sequencer must `cd` into each trial's **attempt**, not its container;
  - `check_trial_starts_cold(base / dirs[j.name], j)` reads the deck from the
    container, and with the deck one level in it finds none — whereupon
    *"absence says nothing"* and the gate **silently stops checking**. A warm
    deck would then launch unverified through the group door while the
    by-name door still refused it. That is precisely the *"guard-only-a-
    surface-applies"* failure this module already names elsewhere.

  So the grouped path changes **with** the single path or not at all.

---

### 2.16 The label budget reserves for the wrong inventory — found 2026-08-27

`identity.MAX_LABEL_BYTES` bounds a label so that *label + stage + extension*
fits a filename. It is derived, not picked:

```python
MAX_LABEL_BYTES = 255 - _STAGE_BUDGET(32) - len(".STRUCT_NEXT_ITER")(17)  # = 206
```

**The 17 is SIESTA's longest extension, and molbuilder writes longer ones.**
The constant's own docstring says where it came from — *"from
`job-contracts.md` § 4.2's SIESTA inventory"* — and molbuilder's own files
were never counted. Its longest tail after the stage is **28**
(`.runwrap-<timestamp>.log`). So a label at the documented cap yields:

```
206 + 32 + 28 = 266   >   255      -- over by 11 bytes
```

**Latent, not live.** Real labels are ~32 characters and the longest name in a
real bench trial is 70, so nothing is failing today. It bites only a label near
the cap, and then as `ENAMETOOLONG` at whatever moment the wrapper opens its
log — after the run has started.

**The fix is to derive the reserve from `OUR_FILE_PATTERNS`**, the list that
already enumerates molbuilder's own files, so adding a file adjusts the budget
instead of silently eating into it. One inventory, one derivation — the same
shape as every other *one door per fact* correction on this page.

*(User raised it as "the longer and longer file name problem". The run index
did not cause it: today's two new suffixes are 17 and 14, both under the 28
already there. What the index does cost is path depth — `run-N/` adds 7
characters against a 4096 limit, where the deepest real path is 232.)*

**Does SIESTA care?** Not about the run index — that is the wrapper's stdout
redirect, and SIESTA's own filenames come from `SystemLabel`, which never
carries it. Whether SIESTA bounds `SystemLabel` itself is **unmeasured**: it
reads into a fixed-length Fortran string, and a 206-character label could in
principle be truncated or refused there long before the filesystem objects.
One `.fdf` with a very long `SystemLabel`, run through the site's own build,
would settle it — and it matters, because the cap above assumes only the
filesystem constrains the label.

---

### 2.12 What a percentage is a fraction OF — decided 2026-08-26

**The rule.** *A run reports how well it used **what it was given**. Cores it
did not ask for are not its business — they are unpredictable, and a fraction
taken over them measures the cluster rather than the calculation.*

Both readings in `monitor.py` are node-wide today, numerator and denominator
alike:

| | today | to build |
|---|---|---|
| busy time | `/proc/stat` aggregate line — every process on the node | the job's own cgroup (see the two spellings below) |
| cores | implicit: all of them | `os.sched_getaffinity(0)` |
| memory | `MemTotal − MemAvailable` — includes other people's jobs | the cgroup's own usage, against the enforced limit |

**Two cgroup generations, and Sol is the older one.** Measured on an
`htc`/`debug` node 2026-08-26: `/proc/self/cgroup` came back in **v1** form —
numbered `id:controller:path` lines, no `0::` line — while this workstation is
v2. An earlier draft of this section named v2 files only (`cpu.stat`,
`memory.current`, `memory.max`); **none of those exist on Sol.** The reader
must know both:

| | cgroup v1 (Sol) | cgroup v2 (this workstation) |
|---|---|---|
| mount layout | `/sys/fs/cgroup/<controller>/<path>` | `/sys/fs/cgroup/<path>` |
| CPU time | `cpuacct.usage` (ns) | `cpu.stat: usage_usec` |
| memory now | `memory.usage_in_bytes` | `memory.current` |
| memory peak | `memory.max_usage_in_bytes` | `memory.peak` *(newer kernels only)* |
| the limit | `memory.limit_in_bytes` | `memory.max` |

v1 turns out to be the **better** of the two here: it keeps a running peak
(`memory.max_usage_in_bytes`) as a kernel counter, so `peak_rss_gb` becomes a
measurement rather than the maximum of whatever the sampler happened to catch.
The enforced limit is what § 2.6 needs, and it is a fact rather than an
estimate.

**Measured on Sol 2026-08-26** (`htc`/`debug`, `-c 4`), every file read
successfully:

```
cpuacct.usage           52132694          ns of CPU time, this job's cgroup
memory.usage_in_bytes    1658880
memory.max_usage_in_bytes 2879488         a real peak — above current
memory.limit_in_bytes   9223372036854771712
SLURM_CPUS_ON_NODE=4  SLURM_CPUS_PER_TASK=4  nproc=4
```

So the numerator and both memory readings work, and `max_usage_in_bytes` is
confirmed to be a genuine running peak rather than a copy of current.

**The limit is on the JOB cgroup, not the task one.** Measured with
`--mem=8G`:

```
/slurm/uid_.../job_62238108/step_0/task_0   limit_in_bytes  9223372036854771712
/slurm/uid_.../job_62238108                 limit_in_bytes           8589934592
```

`8589934592` is exactly 8 GiB — the ask, enforced. The task cgroup carries the
`2^63` "no limit" sentinel. So **the reader must strip `/step_*` from the path
and ask the job cgroup**; asking the task cgroup returns a sentinel that any
arithmetic silently turns into 0%.

This changes § 2.6 for the better: the enforced limit is **readable**, not
something a person must restate. What they declare is the *ask*; what the
kernel holds is the *truth*, and OOM detection wants the second. A sentinel is
still not a limit and must be recognised as *no limit stated*.

**The denominator needs no cgroup at all.** The same measurement returned
`SLURM_CPUS_ON_NODE=1` and `nproc=1` on a one-core allocation, so SLURM sets
the affinity mask correctly and `os.sched_getaffinity(0)` answers on both
generations with no path parsing. That is the rung to try first.

**Two things this buys.** The arithmetic in § 4 — *48 ranks on 128 cores caps
node CPU at 37.5%* — stops being needed: both trials read ~86–90% of their own
allocation and are directly comparable. And the metric stops arguing for the
wrong decision: a job saturating its 48 cores currently *looks* starved, which
argues for a bigger machine, which is the queue this practice exists to avoid.

**The ladder, and saying which rung.** Denominator: affinity mask → SLURM env
→ node. Numerator: cgroup v1 → cgroup v2 → `/proc/stat` (node-wide, and
labelled as such). Whichever rung answered is recorded in `[UTIL-SUMMARY]`:
**a percentage whose denominator is invisible is how this went wrong in the
first place** — and with two cgroup generations in play, a number that does not
say where it came from cannot be checked at all.

*Verified, not assumed:* the path must come from `/proc/self/cgroup`, and on v1
it must be joined to the **controller's own** mount directory. Reading
`/sys/fs/cgroup/cpu.stat` directly lands on the root cgroup and silently gives
node-wide totals again — the same defect in a new spelling.

---

### 2.13 The probe: one meaning per field — reviewed 2026-08-26, P1 BUILT 2026-08-27

A full read of `scheduler/probe.py`, `scheduler/record.py` and the two grid
gates in `jobset/_cli.py`, against `scheduler.md` § 3's rules. **The probe
reads the right things and then loses them on the way to the record.** Four
findings; the first is the one that matters.

**P0 — a partition is a QUEUE, not a machine type.** Asked and answered
2026-08-27 (user: *how could htc clusters have several different
configuration? does that mean sol actually combines different configuration
and call them the same cluster?*). **Yes.** Measured:

```
htc      51 nodes   48 cores  gpu:a100:4
htc       3 nodes   64 cores  gpu:a100.20gb:16
htc     134 nodes  128 cores  (none)
```

`general` and `public` have the same shape. What separates the three is the
**clock** — htc 4 h, public 7 d, general 14 d, debug 15 min — not the hardware.
Four others (`highmem`, `gaudi`, `arm`, `fpga`) genuinely *are* hardware
classes; the three big general-purpose ones are not.

**So hardware cannot be chosen by choosing a partition** — that is `--gres` or
`--constraint`. § 2.8 already says pinning the partition is the wrong lever for
holding a sweep's hardware constant; this is why.

**It also corrects an argument made in P1 below.** The 128-core nodes are
**134 of htc's 188**, the clear majority — so asking for 128 cores there is the
common case, not the long wait. The queue-time case for reporting a *floor* is
much weaker than it was put, and neither collapsed number is the answer.

---

**P1 — `max_cores` collapses two genuinely different ceilings into one.**
`probe.py:153-158` keeps both `gpu_cores` (**min** across groups that have
devices) and `max_cpus` (**max** across all groups). Then `probe.py:319-324`
writes ONE field from whichever applies:

```python
if part.gpu_types:  row["max_cores"] = part.gpu_cores   # a MINIMUM
elif part.max_cpus: row["max_cores"] = part.max_cpus    # a MAXIMUM
```

**Attempted 2026-08-26 and REVERTED — the fix was wrong, and building P2 on it
is what proved it.** The attempt made both a floor (`min_cpus`, saved at
`scratchpad/p1-attempt.diff`). Its own tests passed. Then P2 routed the bench
grid's ceiling through the same field and four grid tests failed with:

```
declared bench point mpi_np=64 needs 64 cores but domain 'general' holds 48
```

**That refusal is wrong.** `general`'s node groups are 48 (a100), 64 (l40) and
**128 (CPU-only)**. A 64-rank *CPU* trial does not need a GPU node; SLURM
places it on the 128-core group and it runs. Refusing it because the GPU nodes
are smaller denies a CPU family a rank count only the GPU nodes cannot hold —
which is the exact reasoning the existing per-family cap comment already gives.

**The two questions the one field is being asked:**

| | the honest ceiling | why |
|---|---|---|
| *can this run here at all?* | the **widest** node | `admit.admits` is a refusal, and its own docstring says it *"only refuses what the record positively rules out"* (R3). SLURM will not place a job on a node too small; it waits for one that fits |
| *which cells should a sweep enumerate?* | the **floor** | a benchmark wants cells that schedule promptly. The user's reason: *we intend to allocate a specific number of cpu to avoid being promoted to request higher cpu-number/memory machine which will take long time to wait* |

Both are true; neither is the other. **The data to serve both is already
parsed** — `gpu_cores` and the widest-node figure are both computed and then
thrown away at the collapse. So the fix is to stop collapsing, not to pick a
winner.

**Measured, so the stakes are clear.** On the real Sol record every CPU-only
partition (`highmem`, `fpga`) has a *single* node group, so min == max and the
reverted change was a **no-op on Sol**. Every partition where they differ
(`htc`, `general`, `public`) has GPUs and so already used the floor. The
inconsistency is real but currently silent; P2 is what would have made it bite.

**Where P0 leaves it: keep the groups, stop collapsing.** The probe already
parses count, cores, memory and gres **per node group** and then discards all
but one figure. Carrying the groups onto the record answers every version of
the question from one measured fact — *can this run at all* (the widest), *how
many nodes could take it* (the count, which is what queue time actually turns
on), *which of them have devices* — with no second field to keep consistent and
no meaning to document twice.

R2 still governs what `admits` compares: a declared limit arrives with its
comparison. A derived ceiling over the groups is that comparison, and it can
finally say *which* group bound the ask, which R10 wants anyway.

**RESOLVED 2026-08-27.** The machines are the record; `max_cores` is the
widest, which is the only figure a refusal can honestly use; and what a person
reads is the **maximum core range** with the fitting node count beside it.

The user proposed the range and then caught its own naming trap: *"minimum
cores sounds like the user has to meet the minimum — maybe it's just the
maximum core range."* Exactly right, and it is measurable: a `-c 4` job gets
four cores on a 48-core node, so a low end presented as a floor would say the
opposite of what is true.

**The count had to ride with it, and the data is why.** Reading `48-128` you
would take 128 for the rare extreme; on `htc` it is 134 of 188 nodes. The same
64-core ask lands on 94% of `general` and 72% of `htc`, and neither is a field
— both fall out of the machines.

**P2 — `topology` is one arbitrary node, and a refusal is gated on it.**
`record.py:_slurm_pick_node` takes whichever GPU node `sinfo` prints **first**
and stores that machine's shape as the partition's topology. `_cli.py:1233`
then refuses a declared bench point against `sockets × cores_per_socket`.
Measured on the real record: that gate says **64**, the GPU family cap says
**48**, and the nodes the sweep actually ran on were 48 and 128. A declared
8×8 point passes the first gate and is dropped by the second — the two
disagree by construction. Core ceilings belong to `domains[].max_cores` (P1),
which is measured across every group; `topology` keeps only what it can
honestly answer, namely this node's own shape when we are standing on it.

**P3 — the CPU family has no core cap at all.** The per-family drop at
`_cli.py:1326` runs under `if fam`, i.e. GPU only. A CPU cell is bounded solely
by P2's arbitrary node.

**P4 — the `+` is stripped, and that is correct.** `parse_sinfo` does
`int(cols[4].rstrip("+"))` and keeps the base. An earlier draft of this plan
proposed preserving the range; **withdrawn** — SLURM promises nothing above the
base, so the base is the honest number and expanding it would invent a
guarantee. What was wrong was never the stripping; it was using a *max* as a
ceiling anywhere (P1).

**P6 — nothing in the record knows x86 from ARM.** User, 2026-08-26: *x64 vs
arm are different, don't mix them.* `Topology` holds seven fields and
architecture is not one of them; `_parse_scontrol_node` reads `Sockets`,
`CoresPerSocket`, `ThreadsPerCore`, `RealMemory` and `Gres` and **skips
`Arch=`**, which `scontrol show node` prints. `_parse_lscpu` ignores
`Architecture:` the same way. Sol has an `arm` partition — `gh200: 1`, a
Grace-Hopper part whose CPU is aarch64 — sitting in the same menu as eight
x86 ones, distinguishable only by a name ASU happened to choose.

**This is not a performance difference; it is a hard failure.** Three ways:

* **`conda_envs` is architecture-specific and the record does not say so.** The
  135 names were enumerated wherever the probe ran. An x86 env does not
  activate usefully on aarch64, so `prep --target sol` can name an environment
  that cannot exist on the node it is sent to.
* **Binaries do not cross.** A SIESTA built with AVX-512 does not run on
  aarch64 at all — not slower, absent. The toolchain sensitivity is already
  known here (a gcc version miscompiles one SIESTA source file); architecture
  multiplies it.
* **§ 2.8 is exactly this rule, one size larger.** *"What must be constant is
  the hardware within one sweep."* Two different ISAs is the most extreme way
  to violate it, and nothing detects it.

**And we do not record our OWN architecture either** (user, 2026-08-26: *we
should know our compiled/installed architecture*). That is the half that
actually decides whether a job runs, because **a mismatch is what fails, so the
check needs both numbers**:

* `conda_envs` is `List[str]` — bare names. `molbuilder-siesta` on an x86 root
  and on an aarch64 root are different software under one string.
* The build recipes assume x86 **in string literals, with no check**:
  `builds.py:701` maps `gcc_linux-64` → `x86_64-conda-linux-gnu-gcc`;
  `builds.py:471` looks for that binary by path; `recipes.py:1038` hard-codes
  `targets/x86_64-linux/include`. On aarch64 those packages are spelled
  `gcc_linux-aarch64`, so the lookup simply finds nothing — and a missing
  compiler reads as *unknown version*, not as *wrong architecture*.

**The rule this needs.** *An environment is not portable, and a record that
names one must say what it was built for.*

**Decided 2026-08-26 (user):** *the UI would just not list the cluster that is
incompatible; so far we don't have anything other than x64.* So this is a
**menu filter, not a refusal** — and that is the better shape, because it is
`ops/access-control.md` § 8 rule 2 applied to hardware: *a capability that
cannot be exercised safely should not appear.* An `arm` row that can only ever
fail is not a choice; offering it and then refusing the submission is two
messages where none was needed.

Two facts and one filter:

| where | status | |
|---|---|---|
| the probe | **BUILT 2026-08-27** | `Topology.arch` from SLURM's `Arch=` and `lscpu`'s `Architecture:` |
| the env list | **BUILT 2026-08-27** | `Environment.env_arch` — `platform.machine()` where the envs were enumerated |
| the domain menu | **BLOCKED** | needs the arch **per partition**, and nothing measured yet says it can be had |

**Why the filter is blocked.** `topology.arch` describes whichever node
`_slurm_pick_node` happened to pick — P2 again — so it says nothing about the
`arm` partition specifically. The filter needs a per-domain answer, and the
probe's one `sinfo` call (`%P|%30l|%D|%40G|%c|%m`) has no architecture column
in it. Whether one exists, or whether the site tags it as a node *feature*
(`%f`), is a measurement nobody has taken:

```bash
sinfo -h -o "%P|%f|%c" | sort -u | head -20
scontrol show node $(sinfo -h -p arm -N -o %N | head -1) | tr " " "\n" | grep -i arch
```

**Filtering on the partition's NAME is not the fallback.** `parse_gres`'s
docstring already argues that case and it holds here: *a list of names cannot
keep up with a site's hardware.* `arm` is what ASU chose to call it, not a
measurement.

Recording both facts is worth having on its own — the record stops being
silent, and the pair is visible to anyone reading it — but the menu keeps
offering every queue until the measurement above says how to know.

**R3 still governs, and here it decides the default.** An unstated architecture
never bars, so a record written before this field — every existing one — filters
nothing and the menu is unchanged. Only two *stated* architectures that disagree
remove a row. That is what keeps a partially-probed cluster usable, and it is
why the filter cannot be written as *show only x86*.

**The cheap part is already reachable.** The probe *already* runs `scontrol
show node` once, so `topology.arch` costs one more `kv.get("Arch")`.

**The part that needs measuring:** whether architecture can be read
*per-partition* in one call, or whether it needs one `scontrol` per partition.
It matters because that is what would let admission refuse a mismatch rather
than merely describe one. Worth one command on a login node when convenient:

```bash
sinfo -h -o "%P %f %c" | sort -u | head -20      # do features carry the ISA?
scontrol show node $(sinfo -h -p arm -N -o %N | head -1) | tr " " "\n" | grep -i arch
```

**Ordering note:** P6 goes with P1 — both are about a record that describes a
partition it has actually measured. P1 stops a field meaning two things; P6
stops a field being missing while its absence looks like agreement.

**P5 — the device menu shows one card under two names.** User, 2026-08-26:
*the `a100` returned `a100.40gb`* — the same hardware, reported by different
nodes under different gres tokens. Both appear in `domains[].gpu` as separate
entries, so `htc` offers `a100: 4` **and** `a100.40gb: 4`, which reads as two
choices and is one.

`parse_gres` is **not** the thing to change. It reads the type positionally and
its docstring explains why at length: an earlier reader matched against a
hard-coded name list and flattened `gh200` into `h200` and the slices into
whole cards, and *"a list of names cannot keep up with a site's hardware."*
Positional reading is right, and it is what makes the alias visible at all.

The gap is that **`a100 == a100.40gb` is site knowledge and nothing measures
it**. Two entries genuinely differ when one is a MIG slice (`a100.20gb` is half
a card, and `--gpus` calls slices *separate askable types, not a smaller ask of
the same one*) and genuinely coincide when one token is just more specific.
Nothing in `sinfo` distinguishes those cases.

**DECIDED 2026-08-27 (user): be explicit.** *"I really don't know because
that's what the slurm returned. I suspect that's just putting vmem together
with gpu in the value. We might just as well be explicit."*

The menu states the tokens exactly as SLURM spells them. No alias table: it
would be site knowledge nothing can verify, it goes stale silently when a site
relabels, and a wrong entry would merge two genuinely different devices — the
one failure mode worse than the duplicate it fixes.

The user's reading of the token is very likely right: `a100.40gb` looks like
the card plus its memory, the same shape as `h200.35gb` beside `h200`. That
makes it a naming convention rather than a fact about hardware, and **naming
conventions are exactly what a measurement must not infer from.**

**Related, and now explained:** `topology.gpu_type` read `a100` on
2026-08-25 and `a100.40gb` on 2026-08-27 with **the same GPU inventory both
times** — the partition's device set is identical between the two probes. That
is P2 with a concrete symptom: the field is not stable across probes of an
unchanged cluster, because it describes whichever node was picked and those
nodes spell the same card differently.

**Not a defect:** `domains[].gpu` enumerating nine GPU types, and
`domains[].max_cores = 48` on `htc`. Both are correct as they stand. The
record dated 2026-08-25T03:28 UTC is not stale either — **re-probing today
would print the same numbers**, because the loss is in what the record keeps,
not in what the probe reads.

---

## 3. Found, not fixed

| | finding | evidence |
|---|---|---|
| **A** | `peak_rss_gb` is `MemTotal − MemAvailable` — **node** memory, not process RSS. ≈ correct only when the job holds the whole node. **Superseded by § 2.6:** the fix is a declared limit, not a better guess. | `monitor.py:_read_mem_used_gb` |
| **B** | `wall_s` is the monitored window (first→last *written* row), not job wall time. **Still open**, and unaffected by § 2.12 — a clock, not a fraction. | `bench/result.py` |
| **C** | **Corrected 2026-08-26 — my earlier note said the probe "describes one node of a partition, faithfully", and that was too generous.** `record.py:_slurm_pick_node` takes whichever GPU node `sinfo` prints **first** and stores that one machine's shape as the partition's `topology`. The probe already knows the partition is heterogeneous — the same record's `domains[htc].gpu` enumerates nine GPU types, and `domains[].max_cores` is a deliberate **min** across GPU node groups (`probe.py:155`), i.e. a ceiling safe on any of them. So the heterogeneity is measured and then discarded by the one field the sizing code reads. Not staleness: `sol.json` is dated 2026-08-25T03:28 UTC, source `scontrol`/`sinfo`. | `topology` says 2×32=64 (the 4×a100 node); `domains[].max_cores` says 48; nodes used were 48 and 128. All three are real htc machines. **Resolved by § 2.13 P1+P2:** the ceiling moves to `domains[].max_cores` as a floor, and `topology` stops gating one |
| **D** | CPU-only trials log **no** node line — `detected phys_cores` appears only on the GPU path. **Shrunk by § 2.12:** core *count* stops mattering once the fraction is taken over the allocation. Core *identity* still does — clock and memory bandwidth set seconds-per-iteration — so what survives is recording node name and CPU model on **every** path, as provenance for comparing throughput and never as a divisor. A log line, not a probe change. | the six wrapper logs |
| **E** | A sweep can span node types with nothing saying so. **Answered — see § 2.8.** | § 4 |

**Not doing:** the dead-CSS list. Three verification attempts were each wrong
in a different way and the one entry chased properly (`workflow-group--stage`)
was live. It needs a runtime instrument across deliberately-triggered error
states, not another grep. Recorded in `ui-contract.md` § 8 with the traps.

---

## 4. The finding that is not a code defect

**The Au-BDT-Au sweep ran on three nodes:**

```
G0 × 2  →  sc004     cores not logged
G2 × 2  →  scg011    128 cores
G4 × 2  →  sg013      48 cores
```

So G2 and G4 are on **different silicon**, and 82.1 → 62.6 s/iter mixes *more
GPUs* with *a different machine*. Combined with n=1 timing, the comparison
cannot support the ranking it produces.

It also explains the CPU percentages completely — 48 ranks on 128 cores caps
node-wide CPU at 37.5% (observed 32.2%); on 48 cores the cap is 100%
(observed 89.3%). **Both were near their achievable ceiling; there is no idle
to explain.**

`task.json` carries `domain: "htc"` (qos `public`, 4 h). The record also
offers `debug` (same partition, qos `debug`, 15 min) — so **choosing `debug`
would not have fixed this**: same partition, same node pool.

**Answered 2026-08-26.**

*Was `debug` intended?* **Left open deliberately.** The user will run it again
and supply two recorded setups; consistency between what the tab showed and what
`task.json` holds is then checkable against real data rather than argued from
one sample. Nothing to chase until then.

*Should a benchmark pin the node type?* **Reframed — see § 2.8.** Pinning the
partition is the wrong lever and would break a deliberate practice: a benchmark
is sent to a short-queue partition on purpose, so it schedules in minutes
instead of a day, chosen to match the hardware the real run will use. A
different queue is fine and intended. What must be constant is the **hardware
within one sweep**, and the choice of hardware belongs to the user.

---

## 5. Order

Consolidated 2026-08-26, after the notification work shipped and the probe was
reviewed. **Measurement before anything that reads a measurement** — three of
today's findings turned out to be one metric measuring the wrong thing.

1. ~~Lane + commit § 1.~~ **Done 2026-08-26.**
2. ~~Notifications + the listener.~~ **Built 2026-08-26** (§ 2.9, § 2.10).
3. **§ 2.12 — the allocation is the denominator.** First, because it decides
   what every displayed percentage *means*, and because § 2.2, § 2.6 and § 3 B–D
   all read a number it changes. `memory.max` also hands § 2.6 the declared
   limit it was going to invent.
4. **§ 2.13 — one meaning per probe field.** P1 (`max_cores` becomes a floor,
   consistently) then P2/P3 (the core ceiling moves off `topology`). Independent
   of everything else, and it closes § 3 C.
5. **§ 2.2** — read the monitor's own mean from `[UTIL-SUMMARY]`. After § 2.12,
   because it is that number it reads.
6. **§ 2.5** — one engine-neutral answer to *how did this run end*. Retires the
   two-door split, the precedence divergence and five longhand copies; anything
   that adds a reader must come after it.
7. **§ 2.1** — settable benchmark iterations. Unlocks n>1 timing, which § 4
   needs before any sweep can support a ranking.
8. **§ 2.11** — harden the listener. Contract settled; **gated on the egress
   test** (§ 6), which could invalidate the destination entirely.
9. **§ 2.7** — drop the summary's poll timer. Removes a cost rather than adding
   a feature.
10. **§ 2.3 + § 2.4** — junction `placement`, and the seam detector that makes
    the choice informed. One feature.
11. **§ 2.6** — the declared memory limit, then OOM detection off the wrapper
    log. Mostly delivered by § 2.12's `memory.max`.
12. **§ 2.8** — record each trial's hardware; let a benchmark state what it
    needs. § 3 B and the shrunken D fold in here; A is retired by § 2.6, C by
    § 2.13.

---

## 6. What needs a human, and why

Two things nothing here can do for itself.

**Egress — MEASURED 2026-08-26, and it works.** From an `htc`/`debug`
compute node:

```
http=302  connect=0.014204  exit=0
```

TCP, TLS and HTTP all completed in 14 ms; the `302` is the auth gate sending
`/` to sign-in, which is the correct answer for an unauthenticated browser
request. **Port 8888 outbound from a Sol compute node is open**, so the
destination in § 2.11 stands and the hardening work is unblocked.

Two things that measurement did *not* settle, both of which would make reports
**vanish in silence** — the notifier catches every exception by design, so an
unreachable server never costs a run anything:

* **The certificate chain — SETTLED 2026-08-26, it validates.** Re-run without
  `-k` from a compute node: `http=302`, `curl_exit=0`. The CA that signed
  qlabsrv's certificate is trusted there, so `monitor.py:578`'s
  `urllib.request.urlopen` — which uses Python's default validating context —
  will connect. **§ 2.11 has nothing left gating it.**
* **A redirect would eat the POST.** The `302` we measured is exactly what a
  notify route would return if it ever fell out of `auth.py`'s
  `_PUBLIC_ENDPOINTS`: `urlopen` follows the redirect, the body is dropped on
  the way to a login page, and nothing anywhere records a failure. The route is
  exempt today; **it needs a guard test that it never redirects**, in the same
  shape as the other listener guards.

**A re-probe on Sol — after § 2.13, not before.** Re-probing today would print
the same numbers, because the loss is in what the record keeps. Once P1 lands,
one command on a login node produces a record whose `max_cores` means one
thing, and the same cgroup check below tells § 2.12 which rung of its ladder
Sol will actually answer on.

**Validation is against ASU's own documentation**, not against the record
itself — a record checked only against itself proves nothing.
