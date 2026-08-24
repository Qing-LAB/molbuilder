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

Four consequences, and they are the whole contract:

**S1 — Unanswered is `None`, never a default wearing a number's clothes.** A
value the person did not give is absent. The scheduler's own default may then
decide — a real and legitimate outcome — but the framework does not
manufacture a number and present it as an answer.

**S2 — A benchmark is bounded, never estimated.** It exists to measure the
per-cycle cost, so feeding it an estimate of that cost is circular. The person
states the **total**; the per-trial bound is arithmetic on top. **A bound
cannot be wrong** — it can only be reached, and reaching it is a result.

**S3 — A measurement is not portable.** Numbers taken on one kind of node do
not describe another. Applying a benchmark across a `node_type` boundary is
refused, not warned about.

**S4 — Nothing is submitted unseen.** The full request, before the
irreversible step. `--yes` is how a person says *I have decided to trust
this*; its absence is not permission.

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
| `fits` | whether this machine can honour that answer |
| `render` | the one output — what is about to be requested |
| `confirm` | the one interface — approve, or don't |

```
$ molbuilder jobset launch bench --budget 4h --mem 128G
about to request:
  time     4h 00m
  memory   128 GB
  36 trial(s), 5 min each -> 239 min total
  queue    htc / public
  submit this? [Y/n]
```

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
* **queue ceilings** — so placement takes the cheapest ceiling that *fits*, in
  an order the site declares ([`scheduler.md`](?doc=execution/scheduler.md)
  § 5a).

All four of those fields were declared on the record and read by nothing. That
is now fixed, and it is worth stating as a pattern rather than as four
incidents: **a field the record carries and no code reads is a check somebody
designed and nobody wired.**
