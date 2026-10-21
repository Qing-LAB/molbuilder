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

**S2 — Nothing is derived, for either axis** *(a derived bound — fifteen
invented minutes a trial, for a total nobody stated — sent five 38-minute jobs
to Sol for a system nobody had sized)*. **Time**: the wall is
stated — `allocation.time`, the run card's `time`, `--time` at prep, or for a
submitted job `--time` at launch — and on a target with a scheduler an
unstated one is refused at prep. `--trial-timeout` exists so one hung trial
cannot eat a group's whole wall; unstated, no per-trial bound exists.
**Memory**: `allocation.mem`, `--mem` at prep — unstated, refused at prep the
same way. Hitting a limit is a result; inventing one is not.

**S3 — A measurement is not portable.** Numbers taken on one kind of node do
not describe another.

**Nothing enforces this, and nothing needs to**: no verdict reaches a launch
by itself (`prep_inputs.prep_run_inputs`, *"THERE IS NO SECOND RUNG"*), so
there is no boundary to cross. What a run uses is what a person wrote in
`execution` — and a person reading a report about another machine's node is
making a judgement, which is theirs to make.

> **The boundary is the machine, not the queue** (`scheduler.md` **R11**,
> 2026-08-27): a domain holds 1–14 machine types, so the comparison is
> against what a trial *landed on* (R12), which is a fact rather than a
> description of a mixture.
>
> **And this is a refusal only because nobody is looking.** Carrying a verdict
> into `prep task` applies a number automatically. **Presenting two measurements
> side by side is not the same act** — there the person is reading the table and
> can weigh it, so the bench summary *states* the machines and never withholds
> the comparison (`generator.md` § 4.4b). Same fact, two stances, and the
> difference is whether a human is in the loop at that moment.

**S4 — Nothing is submitted unseen**, and nothing run here unseen *(user,
2026-10-05: a run here is shown and asked as a submission is)*. The full
request, before the irreversible step — on every door: a stage, a group, a
benchmark's shelf, a run of points, a run here, a sequence of groups — its
job's line and header, its first group's plan in full, and each later group
said as launched in the job when the one before has finished, its plan made
then (`job-system.md` § 6.0). Prep prints every value it
decided with where it came from, and every value of the header it writes; the
launch shows the line as sent and every value of the header it sends, a wall
it restates on the line. `--yes` is how a person
says *I have decided to trust this*; its absence is not permission — with
nobody to ask (no terminal), nothing goes and the launch is refused.

**S5 — The queue is named, never inferred — and checked where it is named.** Which queue to spend a day of
wall-clock in is a judgement about priority, contention and what else is
running — none of it on the machine's record, all of it the person's. So the
person names one — in the description (`allocation.domain`, the run card's
`domain`) or with `prep --domain` — and a run on a scheduler that names none
is refused at prep; launch sends to that queue and names none of its own. **Prep places the job on the queue it names** and prints
its fit — its wall, memory, ranks, cores and GPUs against that queue's
machines on the record: the nodes that hold it, and each limit it is over —
and records the placement with each value's source
([`job-system.md`](?doc=execution/job-system.md) § 6.0). It refuses none of
it: the scheduler decides, and its answer is shown
([`scheduler.md`](?doc=execution/scheduler.md) R6). Where the queues are
listed, each is listed with its machines and the fit of what the job states.
**And a page that offers queues proposes no value**: choosing a queue
on the Task setup card fills neither the wall nor the memory — its wall and
its machines' memory are shown beside the fields, its Preview gives the fit,
and an empty field stays empty until it is stated
*(a value the page proposed, a Save would make "stated" — S1's default wearing
a number's clothes)*.

**A benchmark measures on one queue** — the one its prep names; measuring
another queue is another benchmark (the user, 2026-10-17).

---

## 3. The three things, and where they live

The CLI and the browser call the same three. *Two surfaces asking one question
two ways is how they come to disagree about what was asked.*

| | |
|---|---|
| `placement.queue_table` | the queues the target's record lists, their machines, and the job's fit on each — shown in prep's refusal of a run that names none; placement's, floor 3, which places the job on them |
| `ask.gpu_share_notes` | what a run's GPU request means — its ranks over its GPUs — said at its prep |
| `ask.confirm` | the one interface — approve, or don't; for a run here, also approve the other way, in the foreground or in the background (`Said.other`, `job-system.md` § 6.0) |

*The one output is the launch entry's **plan** — the exact `sbatch` command
of every submission and the header it sends, made once and then sent as it
was shown ([`job-system.md`](?doc=execution/job-system.md) § 6.0): the send
checks the folder is still the one the plan was made from, and refuses
otherwise.*

```
$ molbuilder jobset launch bench coarse --mode submit
about to submit:
  bench-group-cpu-G0K48C1   -> domain htc
    sbatch -J AuBDTAu/bench-group-cpu-G0K48C1 --export ALL,MB_LAUNCHED_BY=jobset-launch launch/bench-group-cpu-G0K48C1.sbatch
    header: -J bench-group-cpu-G0K48C1  -N 1  -n 48  -c 1  -t 0-04:00:00  -p htc  -q public  --mem=64G  -o launch/slurm.%j.out  -e launch/slurm.%j.err
  bench-group-gpu-G1K48C1   -> domain htc
    sbatch -J AuBDTAu/bench-group-gpu-G1K48C1 --export ALL,MB_LAUNCHED_BY=jobset-launch launch/bench-group-gpu-G1K48C1.sbatch
    header: -J bench-group-gpu-G1K48C1  -N 1  -n 48  -c 1  -t 0-04:00:00  -p htc  -q public  --gres=gpu:1  --gres-flags=enforce-binding  --mem=64G  -o launch/slurm.%j.out  -e launch/slurm.%j.err
  per-trial bound: none -- each trial runs until it ends, or the job's wall
  submit this? [Y/n]
```

*Every line is read off the very `sbatch` commands about to be sent and the
headers they send: each shelf's header written at the benchmark's prep, from
what it placed on the benchmark's one queue, its wall the description's
(`allocation.time`). What a GPU shelf's ranks mean for its device — the
GPU-sharing note — is said at that prep, where the grid can still change.*

*The queues the target offers, and the job's fit on each, are listed at prep
when a run names none: the listing asks the one fit the placement asks
(`scheduler.fit`), so it cannot say a queue holds the job where prep's own
line says it does not.*

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
