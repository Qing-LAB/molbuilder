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

**S1 — Unanswered is refused, never a default wearing a number's clothes.** A
value the person did not give is absent, and prep refuses the run, naming
where to state it ([`architecture.md` § 5.2](?doc=execution/architecture.md),
2026-10-02). Neither the framework nor the scheduler's own default answers it:
both would be a number nobody chose.

**S2 — Nothing is derived, for either axis** *(amended 2026-08-24; the
first version let a stated total become a per-trial bound by arithmetic, and
an UNSTATED total become fifteen invented minutes a trial — which sent five
38-minute jobs to Sol for a system nobody had sized)*. **Time**: the wall is
stated — `allocation.time`, the run card's `time`, `--time` at prep or at
launch — and on a target with a scheduler an unstated one is refused at prep
(until 2026-10-02 the queue's own ceiling stood in for it). `--trial-timeout`
exists so one hung trial cannot eat a group's whole wall; unstated, no
per-trial bound exists. **Memory**: `allocation.mem`, `--mem` at prep or at
launch — unstated, refused at prep the same way (until 2026-10-02 a
`molbuilder.json` default, else the scheduler's, stood in). Hitting a limit is
a result; inventing one is not.

**S3 — A measurement is not portable.** Numbers taken on one kind of node do
not describe another.

**Nothing enforces this, and nothing needs to** *(corrected 2026-09-04)*.
This said the crossing was "refused, not warned about", and named a refusal
(`_refuse_if_measured_elsewhere`) that **no production path ever called** —
its only caller was a test, so a green suite proved a rule the product did
not apply. The refusal is deleted rather than wired, because the route it
guarded is gone: since 2026-09-02 no verdict reaches a launch by itself
(`prep_inputs.prep_run_inputs`, *"THERE IS NO SECOND RUNG"*), so there is no boundary left to
cross. What a run uses is what a person wrote in `execution` — and a person
reading a report about another machine's node is making a judgement, which
is theirs to make.

> **The boundary is the machine, not the queue** (`scheduler.md` **R11**,
> 2026-08-27). This rule read `node_type` — a scalar on the domain — until the
> probe showed a domain holds 1–14 machine types and never wrote the field at
> all, so the refusal had never once fired. The comparison it makes is now
> against what a trial *landed on* (R12), which is a fact rather than a
> description of a mixture.
>
> **And this is a refusal only because nobody is looking.** Carrying a verdict
> into `prep run` applies a number automatically. **Presenting two measurements
> side by side is not the same act** — there the person is reading the table and
> can weigh it, so the bench summary *states* the machines and never withholds
> the comparison (`generator.md` § 4.4b). Same fact, two stances, and the
> difference is whether a human is in the loop at that moment.

**S4 — Nothing is submitted unseen.** The full request, before the
irreversible step — on every door: a stage, a grouped bench, a bias chain
*(the stage's door sent straight away until 2026-10-01, on the premise that
`prep`'s printout had been the look; a launch flag can change the queue and
the wall after it, so the line as sent was never seen — ruled that day)*. `--yes` is how a person
says *I have decided to trust this*; its absence is not permission. A
question that carries a judgement only the person can make — following a run
that was launched and never concluded — takes **no** as Enter's answer.

**S5 — The queue is named, never inferred — and checked where it is named.** Which queue to spend a day of
wall-clock in is a judgement about priority, contention and what else is
running — none of it on the machine's record, all of it the person's. So the
person names one — in the description (`allocation.domain`, the run card's
`domain`) or with `--domain` — and a run on a scheduler that names none is
refused at prep. **Prep admits the job on the queue it names** — its wall,
memory, ranks, cores and GPUs against what the record says that queue takes —
and records the placement with each value's source
([`job-system.md`](?doc=execution/job-system.md) § 6.0); a queue that cannot
take it is refused there, naming what was asked and what the queue offers.
Where the queues are listed, a queue that cannot take the job is listed too,
with the reason: hiding it answers *"why is my queue not an option?"* with
silence. **And a page that offers queues proposes no value**: choosing a queue
on the Task setup card fills neither the wall nor the memory — its limits are
shown beside the fields, and an empty field stays empty until it is stated
*(it wrote the queue's ceiling and 95 % of its memory until 2026-10-03, which a
Save made "stated" — S1's default wearing a number's clothes)*.

*There is no queue given once for a whole machine: `execution.domain` in
`molbuilder.json` did that until 2026-10-02, and every job there received a
queue it never named. A split sweep needs one per side, because a cpu-only
partition cannot take a GPU group; `--gpu-domain` **refines** `--domain` and is
needed only when the two differ.*

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

*The one output is the launch entry's **plan** — the exact `sbatch` command
of every submission, made once and then sent as it was shown
([`job-system.md`](?doc=execution/job-system.md) § 6.0): the send checks the
folder is still the one the plan was made from, and refuses otherwise. A
`render` summary lived here until 2026-08-24 and could disagree with the
submission it described; until 2026-10-03 the plan was made twice — once to
show, once to send — with nothing comparing the two.*

```
$ molbuilder jobset launch bench coarse --mem 900G
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

$ molbuilder jobset launch bench coarse --mem 128G --domain htc
about to submit:
  bench-group-cpu
    sbatch -J AuBDTAu/bench-group-cpu -p htc -q public -n 48 -c 1 -t 0-04:00:00 --mem=128G ... launch/bench-group-cpu.sbatch
  bench-group-gpu-G1K48C1
    sbatch -J AuBDTAu/bench-group-gpu-G1K48C1 -p htc -q public -n 48 -c 1 --gres=gpu:1 -t 0-04:00:00 --mem=128G ... launch/bench-group-gpu-G1K48C1.sbatch
  gpu share  48 rank(s) / 1 GPU(s) = 48 rank(s)/GPU
  NOTE 48 ranks/GPU; this stack's tuned point (no NCCL) is ~4 (engines/tuning.md § 2.12).
  per-trial bound: none -- each trial runs until the wall
  submit this? [Y/n]
```

*Every line is read off the very `sbatch` commands about to be sent. Only
what a queue can actually refuse appears as a bar. The `-t 0-04:00:00` is the
wall the description states (`allocation.time`); a run stating none is refused
at prep. The GPU-sharing lines say once per RATIO what several shelves may
share.*

*The display is the exact command, from the same code that submits it; a
summary computed a second way is how "170 minutes" was once shown for five
38-minute jobs.*

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
* **`node_types`** — so a measurement taken elsewhere is VISIBLE rather than
  silently applied. It is a record, not a gate: nothing refuses on it (S3). *Corrected 2026-08-27: this said `node_type`, the
  scalar, and claimed below that the check was wired. It was wired to a field
  the probe never wrote, so it read `None` and returned in silence — see
  `scheduler.md` R11. The machine now comes from where the trial ran (R12),
  not from the domain's description of itself;*
* **queue ceilings** — so the listing can mark what fits and say why the rest
  does not. *They never pick a queue for you* (S5), *and never stand in for a
  wall nobody stated* (S2).

All four of those fields were declared on the record and read by nothing. That
is now fixed, and it is worth stating as a pattern rather than as four
incidents: **a field the record carries and no code reads is a check somebody
designed and nobody wired.**

> **The pattern has a mirror image, and `node_type` was it** *(2026-08-27)*.
> Wiring the reader is only half: this document claimed the check was fixed
> because code now read the field, and nothing checked that anything *wrote* it.
> The probe never did — nine domains, `node_type: null` — so S3 read `None` and
> returned in silence for four days. **A check is wired when both ends are, and
> the honest test is to look at a real record and see a value there.** The
> re-derivation that would have caught it is R0's: nothing could write a scalar
> machine type for a queue that holds fourteen.
