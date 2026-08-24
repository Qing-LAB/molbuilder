# What a job asks for — one question, one answer, one output

**Role:** contract
**Domain:** execution

**Companions:**
[`execution/scheduler.md`](?doc=execution/scheduler.md) — whether a request
fits a queue and which queue it lands in (this document decides *what the
request is*);
[`execution/gpu.md`](?doc=execution/gpu.md) — the GPU decision's own travel.

**This is a tool, not a research project.** A scientist supplies what they
know and makes the choices that are theirs. It must not hand back questions
that are not about the science, and it must not answer questions that are.

---

## 1. The rule

**Ask. Do not derive.**

A scientist knows what their calculation needs better than any rule this
framework can write. So it asks two questions, once:

> **How much total time?**  **How much total memory?**

Everything else follows, and nothing needs explaining afterwards because
nothing was invented.

Five consequences, and they are the whole contract:

**S1 — Unanswered is `None`, never a default wearing a number's clothes.** A
value the person did not give is absent. The scheduler's own default may then
decide — a real and legitimate outcome — but the framework does not
manufacture a number and present it as an answer.

**S2 — Nothing is derived, for either axis** *(amended 2026-08-24; the
first version let a stated total become a per-trial bound by arithmetic, and
an UNSTATED total become fifteen invented minutes a trial — which sent five
38-minute jobs to Sol for a system nobody had sized)*. **Time**: `--time` is
the wall, at prep or at launch; unstated, the target queue's **own ceiling**
is requested — the full amount the cluster allows there — and a queue that
states none gets no wall at all. `--trial-timeout` exists so one hung trial
cannot eat a group's whole wall; unstated, no per-trial bound exists.
**Memory**: `--mem`, at prep or at launch, else the user's own config
(`defaults.mem` / `gpu.mem`), else nothing — the scheduler's default
decides, and the plan says so before anything submits. Hitting a limit is a
result; inventing one is not.

**S3 — A measurement is not portable.** Numbers taken on one kind of node do
not describe another. Applying a benchmark across a `node_type` boundary is
refused, not warned about.

**S4 — Nothing is submitted unseen.** The full request, before the
irreversible step. `--yes` is how a person says *I have decided to trust
this*; its absence is not permission.

**S5 — The queue is named, never inferred.** Which queue to spend a day of
wall-clock in is a judgement about priority, contention and what else is
running — none of it on the machine's record, all of it the person's. So the
options are **listed** and the person names one with `--domain`. A queue that
cannot take the job is listed too, with the reason: hiding it answers *"why is
my queue not an option?"* with silence.

*`execution.domain` in the config is that answer given once for a machine
instead of once per call — a decision, not a guess. A split sweep needs one
per side, because a cpu-only partition cannot take a GPU group; `--gpu-domain`
**refines** `--domain` and is needed only when the two differ.*

---

## 2. Why this replaced a bigger design

The first version of this document was 350 lines: five provenance categories,
a rule that assumed numbers must announce themselves, and a display labelling
every figure with where it came from.

**All of that machinery existed to cope with numbers nobody chose.** Ask, and
there are none to label.

What prompted it: a job asked for 128 GB because SLURM grants 2 GB a core and
it had 64 of them, and for 38 minutes because a per-trial default nobody set
was multiplied by a trial count nobody saw. Both were *correct arithmetic on
inputs the person had never been offered*. The instinct was to make the
arithmetic visible. The better answer was to stop doing it.

> **Recorded because the method matters more than the conclusion.** Chasing
> *why* those numbers appeared produced three confident explanations, each
> falsified: memory could not have caused the queue fall-through (the ceiling
> was never populated, and an unstated limit never bars), `htc` holds far more
> than 128 GB anyway, and the old placement rule picks `htc` regardless. Hours
> went into explaining a number instead of removing the need to explain it.

---

## 3. The four things, and where they live

`jobset/ask.py` — and the CLI and the browser call the same four. *Two
surfaces asking one question two ways is how they come to disagree about what
was asked.*

| | |
|---|---|
| `Ask` | the question, and the answer to it |
| `queue_table` | the queues this machine offers, and which can take the job |
| `confirm` | the one interface — approve, or don't |

*The one output is the launch door's **plan** — the exact `sbatch` command
of every job, printed by the code that submits it. A `render` summary lived
here until 2026-08-24 and could disagree with the submission it described.*

```
$ molbuilder jobset launch bench --mem 900G
this machine offers:
     name         partition/qos           max time  cores    memory  gpu
!  1  debug        htc/debug                    15m    128    251 GB  -
      -> needs 900 GB but debug allows 251
!  2  htc          htc/public                    4h    128    251 GB  -
      -> needs 900 GB but htc allows 251
!  3  general      general/public              168h     48  502.9 GB  a100 x4
      -> needs 900 GB but general allows 502.9
   4  highmem      highmem/public               48h    128   2002 GB  -

  choose one with --domain <name>.  Nothing is submitted until you do.

$ molbuilder jobset launch bench --mem 128G --domain htc
about to submit:
  bench-group-cpu
    sbatch -J AuBDTAu_bench-group-cpu -p htc -q public -n 48 -c 1 -t 0-04:00:00 --mem=128G ... launch/bench-group-cpu.sbatch
  bench-group-gpu-G1K48C1
    sbatch -J AuBDTAu_bench-group-gpu-G1K48C1 -p htc -q public -n 48 -c 1 --gres=gpu:a100:1 -t 0-04:00:00 --mem=128G ... launch/bench-group-gpu-G1K48C1.sbatch
  gpu share  48 rank(s) / 1 GPU(s) = 48 rank(s)/GPU
  NOTE 48 ranks/GPU; this stack's tuned point (no NCCL) is ~4 (engines/tuning.md § 2.12).
  per-trial bound: none -- each trial runs until the wall
  submit this? [Y/n]
```

*Every line is read off the very `sbatch` commands about to be sent. Only
what a queue can actually refuse appears as a bar — with no `--time` stated
there is no walltime to fail, and `-t 0-04:00:00` above is `htc`'s own
ceiling, requested because nothing else was said. The GPU-sharing lines say
once per RATIO what several shelves may share, and an unstated `--mem`
would add its own line here rather than being defaulted in silence.*

*The `-t 0-04:00:00` is `htc`'s own ceiling — the full amount that queue
allows — because no `--time` was stated. The display is the exact command,
from the same code that submits it; a summary computed a second way is how
"170 minutes" was once shown for five 38-minute jobs.*

The listing reuses the scheduler's own admission, so it cannot say yes where
the submission says no. *A table that disagrees with the check is worse than
no table.*

`--yes` skips the question, never the output: a person scrolling back must be
able to see what was sent.

---

## 4. What the machine record is still for

Asking does not make the record redundant — it changes what it is **for**. It
no longer invents your numbers; it checks them, and tells you the truth about
the hardware:

* **memory per node and per core** — measured, so *"you asked 900 GB, the
  largest queue here holds 503"* arrives while changing the number is free,
  rather than as a scheduler rejection after a day in the queue;
* **`node_type`** — so a measurement taken elsewhere is refused rather than
  silently applied (S3);
* **queue ceilings** — so the listing can mark what fits and say why the rest
  does not. *They no longer pick a queue for you* (S5); the ordering machinery
  that once did survives only where a queue is named for a machine rather than
  for a call ([`scheduler.md`](?doc=execution/scheduler.md) § 5a).

All four of those fields were declared on the record and read by nothing. That
is now fixed, and it is worth stating as a pattern rather than as four
incidents: **a field the record carries and no code reads is a check somebody
designed and nobody wired.**
