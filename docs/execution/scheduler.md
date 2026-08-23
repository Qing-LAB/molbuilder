# The scheduler subsystem — what a machine offers, and what a job may ask of it

**Role:** contract
**Domain:** execution

**Companions:**
[`execution/job-system.md`](?doc=execution/job-system.md) § 6 — submission and
routing as the job system sees them (**this document is the mechanism its
design decision #4 always needed**);
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 3.1, § 5.3 —
what each `#SBATCH` line *means*, which stays there;
[`configuration.md`](?doc=configuration.md) § 5 — **M-1**, the fact/preference
split this subsystem must not blur;
[`execution/preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)
— which machine a `prep` is *for*, and how a record gets here.

Scheduling work was spread across five modules — `environment.py`,
`scheduler_probe.py`, `jobset/submit.py`, `runwrap.py`, `runtime_config.py` —
carrying roughly 450 mentions of partitions, queues, ceilings and directives
between them. This document says what that subsystem *is*, so the pieces have
one home and one set of rules instead of one treatment per place someone
noticed a problem.

> **Status.** Phase 1 landed 2026-08-23: `environment.py` and
> `scheduler_probe.py` are now `scheduler/record.py` and `scheduler/probe.py`,
> and `molbuilder.scheduler` is the package. Phases 2–4 (§ 7) are open, so the
> module names in the tables below still describe where placement and emission
> live **today** — that is the point of them.

---

## 0. What this document owns

**Owns:** whether a request fits a queue; which queue a request is placed in;
and the directives that placement produces. The vocabulary below (*machine*,
*domain*, *request*, *placement*) is defined here.

**Does not own:** the meaning of an individual `#SBATCH` line
(`running-a-job.md` § 5.3), the on-disk shape of `environment.json`
(`configuration.md` § 5), which machine a bundle is prepared for
(`preparing-for-another-machine.md`), or the schedule of the work
(`roadmap.md`, rule R3).

---

## 1. Why a subsystem, and not five modules that cooperate

Two failures on ASU Sol, three days apart, were the same defect seen twice.

**2026-08-23, a grouped benchmark was refused by the scheduler.**
`QOSMaxWallDurationPerJobLimit`. Sol records domains cheapest-ceiling-first,
so the first GPU-capable row is `htc/debug` at 15 minutes. The GPU branch of
routing took that first row without regard to duration, asked afterwards
whether it fit, got "no", and returned "no preference" — which meant *the
rendered header's directives stand*, and the header named that same row. A
38-minute job went into a 15-minute ceiling while `htc/public` (4 h) and
`general` (14 d) sat further down the same menu. The CPU branch, four lines
below, had always walked the menu for a row that fits.

**The same day, by hand.** `sbatch <trial>.sbatch` died the same way, for the
opposite reason: the header named a queue and stated **no** wall at all, so
SLURM applied the partition default — longer than the QOS it had just named
permits. `jobset launch` never saw it, because it passes `-t` on the command
line where flags win. The trap was armed only for a human.

Neither is a coding slip. They are what happens when **five things that are
one thing** have no home:

| responsibility | where it lived |
|---|---|
| describe a machine (probe → record) | `scheduler_probe.py` + `environment.py` |
| read records (scopes, named targets) | `environment.py` |
| **decide whether a request fits a queue** | nowhere — see § 2 |
| **choose a queue for a request** | inside `jobset/submit.py`, the *bench submission* module |
| **emit directives** | **two emitters**: `runwrap.py` (the header) and `jobset/submit.py` (the flags) |

Placement living inside bench-submission is why the CPU and GPU branches were
written separately and disagreed. Two emitters is why one half of a placement
could name `debug` while the other asked for 38 minutes.

And the rule was already written. `job-system.md` § 6 says the framework
*"refuses to emit a header it knows will be rejected (design decision #4)"*.
The rule existed; nothing was in a position to enforce it.

---

## 2. The check that was missing

`Domain` has carried four constraints since it was written. What compared a
**request** against them, before 2026-08-23:

| constraint | who checked it | when |
|---|---|---|
| `gpu` | the GPU row selector | prep and launch |
| `max_cores` | one call site, prep's per-family cap | prep only |
| `max_time` | nothing | never |
| `max_mem_gb` | **nothing. Declared, serialised, round-tripped, read by no code at all.** | never |

Four facts, four treatments, three moments, one never implemented. That is not
four bugs. It is one missing function.

> **A field the record carries and no code reads is the signature of this
> defect.** `max_mem_gb` was not forgotten by accident — there was no place
> where *"does this fit?"* was a question anybody had to answer, so nothing
> ever pulled it into use. When the subsystem exists, adding a column to the
> record and not comparing it becomes visible: § 3's R2 makes admission total.

---

## 3. The rules

**R1 — A placement is decided once.** The `#SBATCH` header and the `sbatch`
command line are two *renderings* of one placement, never two decisions. They
cannot disagree, because there is nothing to disagree about.

**R2 — Admission is total.** Every constraint the record carries is compared
against the request, or the record should not carry it. A new column arrives
with its comparison or it does not arrive.

**R3 — An unstated limit never bars.** A domain that does not state a ceiling
is not claiming a small one; a ceiling that cannot be parsed is not a
refusal. Silence is permission — the opposite reading would make a
partially-probed cluster unusable.

**R4 — A refusal names the numbers.** *"needs 38 min but debug allows
00:15:00"*, not *"does not fit"*. We hold the record; sending a user to read
`scontrol` for numbers already on disk is the failure this subsystem exists to
end.

**R5 — A header must be submittable on its own.** If it names a queue it
states a wall that queue accepts. When nothing else supplies one, the queue's
own ceiling is the value — the only number that queue can never reject as too
long.

**R6 — Refuse before submitting, when the record already says so.** The
existing `job-system.md` § 6 rule, now with somewhere to live. A machine whose
record lists **no** domains is a different situation: nothing was promised,
and the header stands.

**R7 — Callers ask what they know.** `prep` knows cores and devices but not
duration; `launch` knows all of them; a caller asking about *capability* asks
about nothing else. An unasked constraint is `None`, and `None` is never a
refusal. This is what lets one admission function serve every caller without
each growing its own variant.

**R8 — Measurements from the record, preferences from the config.** M-1
(`configuration.md` § 5) is not relaxed here. What a queue *allows* is a
measurement and is read from `environment.json`. Which queue you *want* is a
preference and is read from `molbuilder.json` or `--domain`. This subsystem
reads both and mixes neither.

---

## 4. The vocabulary

- **Machine** — one host or cluster, as measured: scheduler kind, topology,
  site, and its domains. Persisted as `environment.json`
  (`configuration.md` § 5).
- **Domain** — one reachable `(partition, qos)` pair and what it allows:
  `max_time`, `max_cores`, `max_mem_gb`, `gpu`. *A fact, never a preference —
  it says "you may submit here, for this long", never "submit here".*
- **Request** — what one job asks for: ranks, cpus-per-task, GPUs, memory,
  wall, exclusivity. Fields the caller does not know are `None` (R7).
- **Placement** — a request bound to a domain, with the reasoning that chose
  it. The single decision R1 refers to.
- **Directives** — a placement rendered for the scheduler, in the two
  spellings it needs: header lines and command-line flags.

A **workstation** is a machine with no domains. It is not an error state: it
yields no header at all, and the job runs directly.

---

## 5. The shape

```
scheduler/
  record.py   Machine · Domain · Topology · Site; read/write; scopes; named targets
  probe.py    detection (sinfo / scontrol / lscpu) -> a record
  admit.py    admits(domain, request) -> [reasons]        # R2, R3, R4, R7
  place.py    place(machine, request, *, prefer_gpu) -> Placement   # R6
  emit.py     Directives(placement) -> header_lines() · sbatch_flags()   # R1, R5
```

Honestly graded, these are not equally forced:

- **`admit.py`, `place.py`, `emit.py` are forced by measured defects** — the
  four-treatments table in § 2, the two divergent branches, the two emitters.
  Each has a failure on Sol behind it.
- **`record.py` and `probe.py` are relocations.** They work today. They move
  because the three above need a home to be *in*, and leaving the data model
  in a general-purpose `environment.py` is what let the check drift away from
  the record it checks.

`emit.py` is unapologetically SLURM syntax. The *boundary* is not: the code
already models two scheduler kinds (`slurm` and `workstation`), and a header
renderer that returns nothing for a workstation is the second kind being
handled. Naming the package for SLURM would bake one kind into the boundary
and be wrong the first time anything else appears.

---

## 6. What a caller sees

```python
req = Request(ranks=64, cpus_per_task=1, gpus=2, gpu_type="a100",
              mem_gb=390, walltime_s=38 * 60)

placement = place(machine, req, prefer_gpu=True)
#   -> htc/public          (debug is skipped: 15 min < 38 min, R3 not engaged)
#   -> Unplaceable(reasons=[...]) when nothing admits it (R6), each reason
#      naming its numbers (R4)

d = Directives(placement)
d.header_lines()    # what the .sbatch carries, wall included (R5)
d.sbatch_flags()    # what the command line carries
#   the same placement, twice (R1)
```

`--domain NAME` names the placement instead of choosing one. It is still
**admitted**: naming a queue that cannot hold the job earns R4's refusal, not
silent acceptance — the user's explicit choice is honoured as a choice, not as
permission to skip the check.

---

## 7. Migration

Smallest risk first; each step separately testable and separately revertable.
The schedule is `roadmap.md` § 7.6, not here (R3).

1. **Move the record and the probe.** Pure relocation.
2. **Move admission.** `domain_admits` already exists and has one caller.
3. **Move placement out of `jobset/submit.py`.** The CPU and GPU branches
   become one walk over the menu with one admission call.
4. **Unify the emitters.** The last and most valuable step: the header and the
   flags stop being two functions.

**The gate for step 4** is a test that renders both spellings from one
placement and asserts they name the same queue and the same wall — the
assertion neither emitter could have made alone, and the one that would have
caught both Sol failures before they left the workstation.
