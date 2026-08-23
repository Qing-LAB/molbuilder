# Preparing for another machine — probe there, prep here, run there

**Role:** contract.
**Domain:** which machine a `prep` is *for* — how that machine is named,
how the choice is refused when nobody made it, and what must travel with a
bundle for it to run somewhere else.
**Reader:** anyone preparing a calculation on one machine to run on
another — the normal case for a cluster, and the case the Task-setup tab
needs before it can offer to prepare anything.

---

## 1. The workflow

```
  on the TARGET            on YOUR machine                 on the TARGET
  ─────────────            ───────────────                 ─────────────
  jobset probe             (copy the record over)
    --write --name sol  ─────────────────────────►  jobset prep run <stage>
                                                      --target sol
                                                          │
                                                    rsync the bundle  ───►  ./…run.sh
```

1. **Probe on the target.** `jobset probe --write --name sol` measures that
   machine — cores, GPUs, scheduler, the `(partition, qos)` domains the
   account can reach — and writes `~/.config/molbuilder/environments/sol.json`.
2. **Carry the record to your machine**, into the same directory. It is a
   plain JSON file; copying it is the whole step.
3. **Prepare here, for there.** `prep --target sol` resolves capability from
   that record instead of from the machine you are sitting at, snapshots it
   as `environment.json` beside the bundle, and renders decks and wrappers
   against the target's numbers.
4. **Send the bundle** and run it there.

**Why this shape.** Rendering a deck needs the machine's capability — rank
counts, memory, whether there is a queue — and a browser or a laptop does
not have it (`project-layout.md` § 2.3.1, invariant **M1**). `--target`
supplies it *as a measurement taken on the machine itself*, which is the
only honest source. Preparing without it measures the desk and calls it the
cluster.

---

## 2. What travels, and what does not

| | travels with the bundle | why |
|---|---|---|
| the deck, the wrappers, `task.json`, the template | ✅ written by `prep` | that is the point |
| the pseudopotentials | ✅ copied in at `prep` | the files are the same everywhere; the library path is not (`project-layout.md` § 2.6) |
| the structure pair | ✅ copied in | same reason |
| `environment.json` | ✅ snapshotted beside the bundle | so the target's capability is a fact of the calculation, not of whoever prepped it |
| **`script_generation` — the preamble and activation** | ⚠️ **only from the bundle's own `.molbuilder.json`** | see § 3 |
| conda env NAMES (`envs`) | ⚠️ same | a name that exists here need not exist there — § 3 applies to it identically |

**Everything else prep writes is machine-free.** Verified by inspection: of
every file `prep` produces, the only lines carrying a local absolute path
are the preamble lines, and they come from configuration, not from the
renderer.

---

## 3. The bootstrap does not travel by itself

The machine record holds **measurements** — `scheduler`, `topology`,
`site`, `domains`. It does **not** hold `script_generation`, and it must
not: `configuration.md` § 5 **M-1** puts measurements in the machine record
and preferences in `molbuilder.json`, and a preamble is a preference.

So `prep --target sol` renders the *right numbers* into `.run.sh` and the
*local machine's bootstrap*:

```bash
# what a laptop bakes, for a job that will run on a cluster:
source /home/you/miniconda3/etc/profile.d/conda.sh   # does not exist there
```

That script then fails on a compute node, unattended, after a queue wait
(`running-a-job.md` § 2.0a: a failed preamble is fatal by design, and
correctly so — the alternative is a job that runs in the wrong environment).

**The mechanism that fixes it half-exists:** the bundle's own
`.molbuilder.json`. For `activation` it is a clean override — the bundle's
value wins and the machine's is not used.

**For `preamble` it is not an override, and this is the sharp edge.**
Preambles **concatenate** — server first, then the bundle's
(`architecture.md` § 8.2) — so a machine-scope preamble is emitted into the
wrapper *even when the bundle supplies its own*:

```bash
# === SERVER PREAMBLE (from molbuilder.json) ===
source /home/you/miniconda3/etc/profile.d/conda.sh   # travels anyway
# === PROJECT ADDITIONS (from .molbuilder.json) ===
module load mamba                                    # the target's
```

Measured, not assumed: `config_provenance` reports that value's origin as
`server+project (concatenated)`, and the rendered wrapper carries both
lines under their own headers.

**So to prepare for another machine, the preparing machine must not set a
machine-scope `preamble`** — or must accept that its lines run there. A
per-calculation `.molbuilder.json` cannot subtract what the join has
already added. That is a consequence of the merge rule, not a defect in
it: joining is the useful answer when both scopes describe the *same*
machine, which is every case except this one.

> **The rule:** when `--target` names a machine that is not this one, the
> bootstrap must come from the **bundle**. If it came from the local
> machine scope, `prep` says so — that is a wrapper that will not start.

**A record can also be stale.** Nothing expires it: a cluster that changed
its partitions or its node mix since the probe will be prepped against the
old answer. `environment.json` carries `detected_at`, so the age is
knowable; surfacing it is left to the surfaces rather than made a refusal,
because a six-month-old record is often still exactly right.

---

## 4. Choosing the machine is the user's, always

Four ways the wrong machine gets chosen **silently** today. Each is the same
defect: a question with several answers, answered by default.

Each row below was **exercised through the `prep` command itself**, not
reasoned about — two of the four turned out to behave correctly already,
and one of those I had first written up as broken.

| # | situation | today | required |
|---|---|---|---|
| **C1** | several records reachable, no `--target` | ❌ silently uses **this** machine — two cluster records present, a workstation chosen, exit 0 | refuse, and name the choices |
| **C2** | `--target` given, but the bundle already carries a different machine's record | ✅ **already refuses** — `UnknownTarget.conflict`, naming both ways out ("delete the snapshot and re-prep" / "drop `--target`") | *(correct today)* |
| **C3** | `--target sol` where `sol.json` exists but is malformed or a future schema | ❌ falls through to **this** machine, exit 0 | refuse: the user named it, so an unreadable record is an error, not a miss |
| **C4** | `--target` names a record that does not exist | ✅ raises `UnknownTarget`, listing the known ones and how to write the missing one | *(correct today)* |

> **C2 is why this section is written from measurements.** Reading
> `resolve_target` alone says the target is ignored — it returns an
> existing `environment.json` before ever consulting `target`. The guard
> lives further along the prep path, and running the case is what showed
> it. A contract written from the first reading would have added a second
> refusal on top of a working one.
>
> **And measurements alone were not enough either.** Three defects in the
> first implementation survived a passing test suite and only a full read
> of `machine_for` showed them, because each needs a *combination* no
> single case reaches:
>
> * **C1 asked for a local record first.** With named targets and nothing
>   probed locally — the commonest cluster setup — the question went
>   unasked and a fresh probe of the machine the user was sitting at
>   answered it. "This machine" is always a candidate, so any named record
>   makes the question real.
> * **C3 lived inside the resolution loop**, which `record_scopes` orders
>   calculation-first. An already-prepped bundle returns at the first scope
>   and never reaches the target one, so the same flag refused for a fresh
>   folder and stayed silent for a prepped one. The named target is now
>   validated whole, before any scope is walked.
> * **A dead branch.** `named` was assigned and then returned unconditionally
>   at the end of every loop iteration, so the `if named is not None` after
>   the loop could never run. Removed.

**Why refusing beats defaulting.** The cost is asymmetric. Being asked
which machine costs one flag. Being given the wrong one costs a queue wait,
an allocation, and a set of numbers that look plausible — the failure mode
`--target` was introduced for in the first place, after a benchmark prepped
on a workstation was measured against the desk.

**One record and no `--target` still proceeds silently.** There is no
ambiguity to resolve, so there is no question to ask.

**What a bundle cannot currently answer: *which machine am I for?*** The
snapshot beside a bundle carries capability — scheduler, topology, site,
domains — and `source` records *how* each part was measured (`scontrol`,
`lscpu`, `flag`), never *whose* machine it is. `Site` is
`partition`/`qos`/`account`; there is no machine name anywhere in the
record. So C2's refusal compares the two records' **content**, which is
sound (identical records make the question moot) but leaves a person
holding an rsync'd bundle unable to confirm what it was prepared for.

`source` is a free-form `Dict[str, str]` that round-trips, so recording
`source["target"] = "sol"` when `--target` was used costs no schema change
and makes the bundle self-describing. Worth doing; not required by the
rules above, so it is named here rather than folded in silently.

---

## 5. What the browser may offer

The Task-setup tab may offer to prepare, subject to this section. It does
not get its own rules: it surfaces the ones above.

- It lists the reachable machine records and **requires a choice** when
  there is more than one (§ 4, C1).
- It shows what `prep` resolved, using the provenance `prep` already
  computes (`runtime_config.config_provenance` — `sources`, `effective`,
  `domains`), not a hand-written notice.
- It shows **which config supplied the bootstrap**, and flags the § 3 case:
  a remote target whose preamble came from the local machine.

> **This narrows a stated boundary, deliberately.** `job-system.md` § 4 says
> *"`prep` and `launch` stay on the terminal by design"*, written when
> `prep` necessarily resolved capability from the machine it ran on.
> `--target` removed that necessity: capability comes from a record measured
> on the target. **`launch` is unchanged** — starting a job still happens
> where the job runs.
