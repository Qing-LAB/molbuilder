# Sending a job — nothing unseen, one at a time

**Role:** contract
**Domain:** execution
**Owns:** the sending rules — the full request is shown before anything
irreversible goes (S4), and a scheduler is handed one job at a time.
**Read with:** [`job-system.md`](?doc=execution/job-system.md) § 6.0 — the
launch protocol that carries these rules out (plan, show, ask, send, record) ·
[`job-system.md`](?doc=execution/job-system.md) § 6.0a — the jobs sent from this
machine, asked after and stopped · [`scheduler.md`](?doc=execution/scheduler.md)
R6, § 6a — placement, the header, the send line ·
[`generator.md`](?doc=execution/generator.md) § 4.3a — a benchmark's shelves ·
[`architecture.md`](?doc=execution/architecture.md) § 5.2 — where every launch
value is stated

## The short version

molbuilder sends nothing a person has not seen. Before a job is submitted, or a
run started here, the launch prints the exact line it will send and every value
of the header it sends, and asks; `--yes` is how a person says *I have decided
to trust this*, and with nobody to ask, nothing goes (S4). `--mode ask` puts the
same lines to the scheduler and sends nothing (§ 1, *Asking without sending*). It
hands a scheduler few, deliberate jobs — one per launch of a task, one per shelf
of a benchmark, sent one after another — never a flood (§ 1, *One job at a
time*). The other rule ids — S1, S2, S3, S5 — point to where each rule is decided
(§ 2); § 3 shows what the person sees.

---

## 1. The sending rules

**S4 — Nothing is submitted unseen, and nothing run here unseen** *(user,
2026-10-05: a run here is shown and asked as a submission is)*.

- **What is shown — the full request, before the irreversible step, on every
  door**: a stage, a group, a benchmark's shelf, a run of points, a run here, a
  sequence of groups. For each job: its line and its header; a sequence's first
  group's plan in full, and each later group said as launched inside the job
  when the one before has finished, its plan made then
  ([`job-system.md`](?doc=execution/job-system.md) § 6.0).
- **Prep** prints every value it decided with where it came from, and every value
  of the header it writes. **The launch** shows the line as sent and every value
  of the header it sends, and a wall it restates on the line.
- **The plan is sent as it was shown**: made once, then sent; the send checks the
  folder is still the one the plan was made from, and refuses otherwise.
- **`--yes` skips the question, never the output**: a person scrolling back must
  be able to see what was sent. Its absence is not permission — **with nobody to
  ask (no terminal), nothing goes and the launch is refused**, naming `--yes`, so
  a script reading the exit code is not told it went.
- **Every question and its answer is written down**, a *no* too
  ([`job-system.md`](?doc=execution/job-system.md) § 6.0).

**Asking without sending — `--mode ask`.** `jobset launch … --mode ask` submits
nothing: in place of the question, each submission's line is put to the
scheduler with its kind's ask — SLURM's `--test-only` after the command
([`scheduler.md`](?doc=execution/scheduler.md) § 6a) — and the answer is printed
(`submit.ask_launch`).

- **One question per submission, as `submit` would send it** — so one per shelf
  of a benchmark, each with the header its prep wrote. A stage launched before is
  asked as the run its next launch would open, and nothing is opened; a trial
  already run is said as *already run, so not asked*.
- **It is not held to one job at a time**: asking enqueues nothing. It stops at
  `submit.ASK_MAX_QUERIES` (24) questions out of politeness, and names the rest —
  `NOT asked (past 24 queries): …` — never drops them.
- **The answer, job by job, in the order asked** — never sorted, since a sort
  would be a recommendation: when it would start, on which nodes, with how many
  processors, and that the time is an estimate from the queue as it is now. A
  prediction the scheduler declines or refuses reads *no prediction*, with the
  scheduler's own words — **unknown, never soon** (`ask.Prediction`).
- **On a machine with no queue** there is nothing to wait for, and it says so.
- **What was asked is written down** in the calculation's ledger.
- **It needs a login node**: there is no prediction without the cluster.

**One job at a time.**

> *"SLURM should never submit jobs in parallel. Submission is manual and one by
> one. It is a disaster to do parallel job submission on HPC."* — the user

Two reasons, and the second is scientific:

- **On a shared cluster it is antisocial.** N jobs entering the queue together
  start together if there is room, and the allocation goes with them.
- **For a benchmark it is not merely rude, it is invalid.** Points that run
  concurrently in one allocation contend for the same cores, memory bandwidth and
  interconnect, so the sweep measures **contention rather than scaling** — and
  reports a number that looks fine.

So `--mode submit` hands the scheduler few, deliberate jobs:

| launch | jobs it sends |
|---|---|
| a task | one job per invocation — a stage, a group of stages, or a sequence of groups launched one after another inside it |
| a benchmark | one job per resource shelf, sent one after another; a named trial goes alone |

A shelf's trials run one after another inside its one job, so no two contend —
how shelves are made, ordered and sent is
[`generator.md`](?doc=execution/generator.md) § 4.3a. **`--mode direct` is not
submission**: it runs each job here, one after another —
in the background unless `--foreground`
([`job-system.md`](?doc=execution/job-system.md) § 6.0). The rule lives in the
launch entry (`submit.plan_launch`), not in the command line, so every caller
meets it.

## 2. The other rule ids — where each rule is decided

Code cites these ids, so each is kept here as a pointer to the rule's home:

| id | rule | decided in |
|---|---|---|
| **S1** | a launch value nobody stated is refused at prep, never filled with a default — the framework's or the scheduler's | [`architecture.md`](?doc=execution/architecture.md) § 5.2 |
| **S2** | nothing is derived: the wall and the memory are stated, never worked out from other numbers | [`architecture.md`](?doc=execution/architecture.md) § 5.2 |
| **S3** | a measurement is not portable: comparability is the machine a run landed on, and no verdict reaches a run by itself | [`scheduler.md`](?doc=execution/scheduler.md) R11 |
| **S5** | the queue is named by the person, never inferred; prep places the job on it and prints its fit; a page that offers queues proposes no value | [`scheduler.md`](?doc=execution/scheduler.md) R8, R6 |

## 3. What the person sees

Each is one function, reached through the one prep and launch entries, so the
terminal and Task setup's Prep cannot disagree about what was asked:

| function | what it gives |
|---|---|
| `placement.queue_table` | the queues the target's record lists, their machines and the job's fit on each — printed in prep's refusal of a run that names none ([`scheduler.md`](?doc=execution/scheduler.md) R8) |
| `ask.gpu_share_notes` | what a run's GPU request means — its ranks over its GPUs — said at its prep, where the grid can still change |
| `ask.confirm` | the one question — approve, or don't; for a run here, also approve the other way, in the foreground or in the background (`Said.other`) |

A benchmark's launch, as the person sees it — every line read off the very
commands about to be sent and the headers they send, each shelf's header written
at the benchmark's prep, its wall the job request's `allocation.time`:

```
$ molbuilder jobset launch bench coarse --mode submit
about to submit:
  bench-group-cpu-G0N48C1   -> domain htc
    sbatch -J AuBDTAu/bench-group-cpu-G0N48C1 --export ALL,MB_LAUNCHED_BY=jobset-launch,MB_WALL_S=14400 launch/bench-group-cpu-G0N48C1.sbatch
    header: -J bench-group-cpu-G0N48C1  -N 1  -n 48  -c 1  -t 0-04:00:00  -p htc  -q public  --mem=64G  -o launch/slurm.%j.out  -e launch/slurm.%j.err
  bench-group-gpu-G1N48C1   -> domain htc
    sbatch -J AuBDTAu/bench-group-gpu-G1N48C1 --export ALL,MB_LAUNCHED_BY=jobset-launch,MB_WALL_S=14400 launch/bench-group-gpu-G1N48C1.sbatch
    header: -J bench-group-gpu-G1N48C1  -N 1  -n 48  -c 1  -t 0-04:00:00  -p htc  -q public  --gres=gpu:1  --gres-flags=enforce-binding  --mem=64G  -o launch/slurm.%j.out  -e launch/slurm.%j.err
  per-trial bound: none -- each trial runs until it ends, or the job's wall
  submit this? [Y/n]
```

What each part of the line and the header is:
[`scheduler.md`](?doc=execution/scheduler.md) § 6a.
