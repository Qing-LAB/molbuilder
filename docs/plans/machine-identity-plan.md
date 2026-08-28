# Machine identity — migrating the code to R11/R12/R13

**Role:** plan (built 2026-08-27; two facts still need a Sol run)
**Domain:** execution

**Status: ALL SEVEN PIECES BUILT 2026-08-27.** What remains needs a machine, not code: a Sol run to see a real `[MACHINE]` line, and a Sol re-probe to settle `lightwork`'s cap (§ 6). The rules are in
[`scheduler.md`](?doc=execution/scheduler.md) **R11**, **R12**, **R13** and R3's
second half, with the bench half in
[`generator.md`](?doc=execution/generator.md) § 4.4b and
[`bench-summary.md`](?doc=web/bench-summary.md) **B5**. This page is the work
list that follows from them, and nothing here re-decides a rule — where a
question is open it is marked and left open.

---

## 0. Why this exists

A contract consolidation on 2026-08-27 found that
[`submission.md`](?doc=execution/submission.md) **S3** — *a measurement is not
portable; applying a benchmark across a machine boundary is refused, not warned
about* — **had never once fired**. It read a scalar `node_type` on the domain.
The probe never wrote it: all nine Sol domains record `node_type: null`.

**And nothing could honestly have written it.** A partition is a queue drawing
from many machines (**R0**): Sol's `htc` holds fourteen kinds, `public` four. A
single machine type over a mixture is an opinion, so the scalar is retired and
the fact is taken from the machine a trial **actually ran on**.

> **The tests passed throughout, and that is the part worth carrying forward.**
> `test_measurement_transfers.py` builds a fake domain carrying
> `node_type="standard"` and asserts the plumbing moves it. Both ends are
> fixtures, so it proved the wire and never the current. **Every piece below
> states how it is verified against a real record, not a constructed one.**

---

## 1. What is wrong today, with evidence

| # | defect | evidence |
|---|---|---|
| **D1** | the machine a run landed on is **not recorded anywhere** | `_placed_on` (`submit.py:1179`) writes the *queue*; `summarize.py` contains no reference to host or nodelist |
| **D2** | S3's refusal reads a field nothing writes | 9/9 Sol domains `node_type: null` → `_measured_on` returns `None` → `_refuse_if_measured_elsewhere` returns early, silently |
| **D3** | *trials disagree* and *nothing says* collapse into one `None` | `_measured_on` (`_cli.py:676`) returns `None` for both; its own docstring calls the first *"no single node type to carry anywhere"*, a stronger claim than unknown |
| **D4** | admission's core ceiling ignores devices | `_widest_node` (`admit.py`) takes the widest node overall. On `public`: 128 cores across 107 standard nodes, **48** across the 52 with A100s — a 64-rank GPU trial is admitted and is unplaceable |
| **D5** | policy ceilings are never read | `sacctmgr -nP show qos format=Name,MaxWall,Flags` — `MaxTRES` is not requested; `scontrol show partition` is run and parsed for `DefMemPerCPU` alone, dropping `MaxCPUsPerNode` |

**D4 hits a benchmark's GPU side first**, because a sweep's rank axis is
normally sized against the CPU side. **D5 is why `lightwork` carries
`max_cores: 128` beside a standing note that it caps at 8** — with nothing able
to say whether both are true.

---

## 2. The two traps

Both would produce a feature that runs, reports, and means nothing. **Each gets
a guard test whose failure mode is named, not a general assertion.**

**T1 — the kind, not the box.** SLURM spreads an array over whatever is free,
so two trials of one sweep almost always land on different hosts. Comparing
**names** reports *"different machines"* for every sweep ever run.
*Comparability is decided on the node's kind* — physical cores, memory, device
model. The name is provenance: which box, for tracing a bad one.

**T2 — the node's size, not the allocation.** `monitor._alloc_cores()` answers
*what this job was given* (the affinity mask), deliberately — it is the
denominator every percentage is a fraction of. The node's own size is
`os.cpu_count()`. **Reading the allocation as the machine would make a
rank-scaling sweep — 48 against 64 against 128 — report a different machine per
trial**, breaking precisely the sweep this work exists to serve.

> Measured on this workstation: `os.cpu_count() = 40`,
> `len(os.sched_getaffinity(0)) = 40`. **They agree when nothing constrains the
> job, which is why a workstation test cannot catch T2** — it needs a cgroup or
> an affinity mask, i.e. a Sol run or a fixture that sets one.

---

## 3. Order

**P1 first and alone**: nothing downstream can compare what is not recorded.
P5 comes after P4 because the field cannot be deleted while a reader still
holds it. P6 and P7 are independent and may be taken at any point.

```
P1 record ──► P2 read ──► P3 show          ← BUILT 2026-08-27
      └─────► P4 refuse ──► P5 retire the scalar   ← BUILT 2026-08-27
P6 admission by device        ← BUILT 2026-08-27
P7 probe the policy caps      ← BUILT 2026-08-27
```

> **Built with the guards § 5 asked for**: `test_machine_identity.py`
> (T2 by restricting this process's own affinity; model-never-count;
> first-line placement), `test_machine_in_summary.py` (T1 with Sol's real
> jittered MemTotal figures; the no-judgement wording; legacy absence),
> and the census in `test_prep_bench_fold.py`'s sweep-view section.  Every
> fixture writes a real monitor log and reads it through the real parsers
> — § 7's acceptance rule — and each guard was mutation-tested (swap the
> cores source, drop the statement, force the column) before it counted.

---

## 4. The pieces

### P1 — the monitor records the machine — BUILT 2026-08-27

**Where:** `molbuilder/monitor.py`.

The monitor already runs on the compute node, already calls `nvidia-smi -L`
**once** per run (not per tick), and already writes a provenance line. So this
is one more line in a file that exists for exactly this purpose, not new
machinery.

| fact | source | why this one |
|---|---|---|
| node name | `os.uname().nodename` | provenance (T1) |
| physical cores | `os.cpu_count()` | the node's size — counts *online processors*, which a cgroup does not shrink; **never** `_alloc_cores()`, which reads the affinity mask a scheduler *does* shrink (T2) |
| memory | `/proc/meminfo` `MemTotal` | the node's own total; the cgroup limit is the allocation's |
| device **model** | `nvidia-smi -L` | the model survives the job's device filter; the **count does not** — inside a scheduled job `nvidia-smi` lists only the devices the job was granted, so a count would be the allocation's (T2 again, for devices) |

**Resolved 2026-08-27 (was open): a separate `[MACHINE]` line, written once at
monitor start.** Not fields on `[UTIL-BASIS]` — that line is written when the
job *ends*, so a monitor killed with its allocation would take the machine
record down with it; the machine is known at start. And one line keeps one
meaning: `[UTIL-BASIS]` answers *what was this job given*, `[MACHINE]` answers
*what kind of node was under it* — the two numbers T2 exists to keep apart.
Recorded in **R12**'s own text.

```
[2026-08-27T14:02:11] [MACHINE] node=sol-g042 cores=48 mem_gb=503.5 gpu=NVIDIA A100-SXM4-80GB
[2026-08-27T14:02:11] [MONITOR] start (interval=30s watch_pid=12345) ...
```

**Verified by:** a run on Sol whose line names an actual node; and a mutation —
swap `os.cpu_count()` for `_alloc_cores()` and a T2 guard must fail.

### P2 — `summarize` reads it — BUILT 2026-08-27

**Where:** `molbuilder/jobset/summarize.py` — `parse_point`, `BenchPoint`.

The monitor log is **already** read per trial (`_latest_run_file(d, basename,
"monitor.log")`), so this adds a field, not a door — **B1** holds: the figure
comes through the reader that already owns it.

**Verified by:** a fixture bundle with two trials whose logs name different
kinds → the point objects differ; same kind on different hosts → they do
**not** (T1).

### P3 — the summary shows it — BUILT 2026-08-27

**Where:** `summarize`'s table, and the bench summary page (**B5**).

Each trial names its machine. Where trials differ, the sweep header says which
machines are in play instead of printing one core figure — today it reads
*"444 atoms · siesta · 128 cores · slurm"*, which is the sweep's **allocation**,
not any trial's machine.

**It shows and stops there** — no ranking, discounting, annotating or refusing
*(user, 2026-08-27: "speed comparison is speed comparison, don't overstep — you
are not the analyzer, you present the data")*.

**Verified by:** a mixed CPU/GPU sweep renders both machines and no verdict.

### P4 — the transfer refusal reads the machine — BUILT 2026-08-27

**Where:** `_cli.py` `_measured_on`, `_refuse_if_measured_elsewhere`.

**The source is the verdict's own record.** What `prep run` applies is
`bench-result.json`, and since P2 its points carry `machine` — so the check
reads the thing being applied, not a second walk over `run.json` files.
One door; the old glob was a reader of a field `submit` wrote from the
*queue* (D2).

**What is compared, and why only that** *(decided 2026-08-27, building
this)*. The measured kind and the target's menu state exactly one fact in
one shared vocabulary: **the per-node core count**. Devices do not — the
menu speaks gres tokens (`a100`), the measurement speaks model names
(`NVIDIA A100-SXM4-80GB`) — and a substring bridge between them would be a
guess wearing a check's clothes. So cores decide, devices and memory stay
out until the vocabularies unify, and that limit is stated here rather
than papered over. The Sol hazard every document cites — a 48-core GPU
measurement carried to a 128-core CPU run — is a cores mismatch, so the
honest comparison is also the load-bearing one.

| the verdict's trials say | the target row says | outcome |
|---|---|---|
| **several kinds** | — | **refuse, naming them** — a verdict measured on two machines has no single basis to carry (D3: *disagree* stops being spelled like *unknown*) |
| one kind | a `node_types` row with those cores exists | silent — nothing is ruled out: the scheduler may land a job on any node of the queue, including that kind |
| one kind | `node_types` listed, **none** with those cores | **refuse** — positively ruled out, which is R3's standard for refusing |
| one kind | no `node_types` on the row | silent — cannot tell |
| nothing | — | silent — cannot tell (a pre-`[MACHINE]` record) |

**S3 keeps its refusal and this does not contradict § 4.4b's report-only
stance.** Carrying a verdict into `prep run` applies a number with nobody
looking; a summary is read by someone who can weigh it.

**Verified by:** the mutation that would have caught the original bug — a
record whose machine field is **absent** must leave the check silent, and one
with two kinds must refuse. Neither may be satisfied by a fixture that
supplies what the probe does not.

### P5 — retire the scalar `node_type` — BUILT 2026-08-27

Only after P4 stops reading it. **No shim, no fallback** — a cluster that
genuinely is uniform declares one entry in `node_types`.

| file | line | change |
|---|---|---|
| `scheduler/record.py` | 140 | drop the field |
| `scheduler/record.py` | 173 | drop from `_KNOWN` |
| `scheduler/record.py` | 829 | drop from the serialised row |
| `jobset/submit.py` | 1194 | `_placed_on` drops it — it records the queue |
| `jobset/materialize.py` | 731 | docstring says `placed_on` is *"WHERE IT RAN"*; **R12 says it is where it was SENT** |
| `runtime_config.py` | 1685 | comment lists it as an operator column |

### P6 — admission's ceiling filters by device (R3 second half) — BUILT 2026-08-27

**Where:** `scheduler/admit.py` `_widest_node`.

When a request names a device, the ceiling is the widest node **that offers
it**, not the widest overall. *"SLURM will not place a job on a node too
small"* — the corollary's own argument — is equally true of devices.

**Verified by:** a `public`-shaped record; 64 ranks + `a100` must be refused
naming 48, and 64 ranks with no device must still pass at 128. **The second
half is the mutation** — a filter that always applies would re-break the
64-rank CPU trial R3's corollary was written for.

### P7 — the probe reads the policy caps (R13) — BUILT 2026-08-27

**Where:** `jobset/_cli.py:2690`, `scheduler/probe.py` `parse_qos`,
`parse_scontrol_partitions`.

| cap | fix |
|---|---|
| QOS `MaxTRESPerJob` (`cpu=N`) | add `MaxTRES` to the `sacctmgr` format list and parse it |
| partition `MaxCPUsPerNode` | already fetched by `scontrol show partition`; keep it |

The smaller of *policy* and *widest machine* governs. **This is what settles
`lightwork`** — and it is a probe gap, not a question for a person *(user,
2026-08-27: "why do I need to separately get info on lightwork? your jobset
probe did not do its job?")*.

**Verified by:** a fresh probe on Sol where `lightwork` reports a policy cap or
demonstrably has none. **Both outcomes are results**; what is not acceptable is
the current state, where the question cannot be asked.

---

## 5. Tests — what retires, what is rewritten

Per the standing rule: *tests serve the contract; never assert a rule no
document states.*

| test | verdict | why |
|---|---|---|
| `test_measurement_transfers.py` | **rewrite** | the property (S3) survives; its mechanism is retired. Its docstring quotes `asu-sol.md` § 5.3 on `node_type` — text that no longer exists. Must drive off a recorded machine, and **must not supply through a fixture what the probe does not write** |
| `test_routing_domains.py` (≈153–178) | **rewrite** | asserts `row.node_type == "gpu-a100"` for a declared row; becomes `node_types` |
| `test_scheduler_config.py:272` | **amend** | a comment naming the drafted column |
| everything under `test_domain_machines.py` | **keep** | it already tests `node_types`, the surviving spelling |

**New guards, each named for the failure it prevents:** T1 (same kind, two
hosts → comparable), T2 (rank-scaling sweep → one machine), D3 (absent → silent,
several → refuse), P6's second half (no device → unfiltered ceiling).

---

## 6. What needs a human

**Nothing needs a person to answer a question** — P7 removes the only one that
did. Two need a person to *run* something:

- a Sol run to confirm P1's line names a real node and P7's probe reads a real
  cap (both are `srun` one-liners, not investigations);
- the standing rule that jobs are submitted **one at a time, by hand**.

---

## 7. Risk

**The migration's own failure mode is the one it is fixing.** Every piece here
can be made to pass with a fixture that supplies the value reality does not —
which is exactly how S3 stayed dead through four days of green tests. **The
acceptance question for each piece is not "does the test pass" but "was this
value written by the thing that will write it in production".**
