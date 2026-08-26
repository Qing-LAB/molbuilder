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
| pushed | `2107e3e3` |
| committed, unpushed | `f9078b65` — SCF_NOT_CONV is a cause, not proof |
| uncommitted | ~41 files (§ 1) |
| lane | **not run** since the § 1 work — run `tools/testrun.py run none2e` before committing |

---

## 1. Uncommitted work — verify, then commit

Done and locally tested; needs a full lane and a commit split.

1. **The run-ending vocabulary** (`parse.md` § 2b). `run_state` =
   `running`/`ended`/`stopped`/`out_of_memory`/`unknown`; `scf_converged`
   reported beside it, never folded in. Touches `frame.py`, `parse/types.py`,
   `engines/{_helpers,siesta,pyscf,molwatch}.py`, `parse/dirs/job.py`,
   `web/blueprints/watch.py`, `trajectory/core.js`, ~9 test files, 6 goldens.
2. **`parse/engines/_run_ending.py`** — the marker table, stdlib-only. Two
   doors onto it (`scan_ending` 21 ms vs full parse 272 ms on a six-trial
   sweep); the heavy parser builds its fatal rules from the same table.
   `jobset/summarize.py`'s private `_DONE_MARKERS` deleted.
3. **The bench display** — three stacked bar panels, values on bars,
   token-styled tooltips, usage line, `ran with` line, context header,
   two clocks, caption.
4. **Measurement fixes** — time-weighted util means, node facts read from the
   run's own wrapper log, `iters_measured: 1` disclosed on the card.
5. **Docs** — `parse.md` § 2b, `junction-cell.md` § 3.3,
   `bench-summary.md` § 3, `deployment.md` § 1.1, `ui-contract.md` §§ 5 + 8.

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

---

## 3. Found, not fixed

| | finding | evidence |
|---|---|---|
| **A** | `peak_rss_gb` is `MemTotal − MemAvailable` — **node** memory, not process RSS. ≈ correct only when the job holds the whole node. **Superseded by § 2.6:** the fix is a declared limit, not a better guess. | `monitor.py:_read_mem_used_gb` |
| **B** | `wall_s` is the monitored window (first→last *written* row), not job wall time. | `bench/result.py` |
| **C** | The probed `Topology` holds ONE shape; `_slurm_pick_node` describes one node of a partition, faithfully. A heterogeneous partition has no single value. `resolve_environment`'s own override example is `{"cores_per_socket": 24}` — someone hit this and hand-patched. | record says 2×32; nodes used were 48 and 128 |
| **D** | CPU-only trials log **no** node line — `detected phys_cores` appears only on the GPU path, so `sc004`'s size is unrecorded. | the six wrapper logs |
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

1. ~~Lane + commit § 1.~~ **Done 2026-08-26.**
2. § 2.2 — smallest, and it makes every displayed number trustworthy.
3. § 2.5 — one ending answer. Retires the two-door split, the precedence
   divergence and the five longhand copies in one change; everything that reads
   a run's fate depends on it, so it goes before anything that adds a reader.
4. § 2.1 — unlocks better statistics, which § 4 needs.
5. § 2.7 — drop the summary's timer. Small, and it removes a cost rather than
   adding a feature.
6. § 2.3 + § 2.4 together — one feature; the detector makes the choice informed.
7. § 2.6 — the declared memory limit, then OOM detection off the wrapper log.
   The limit comes first: without it the state has nothing to mean.
8. § 2.8 — record each trial's hardware, then let a benchmark state what it
   needs. § 3 B–D fold in here; § 3 A is retired by § 2.6.
