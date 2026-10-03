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
   plain JSON file; copying it is the whole step — see § 1a for the commands.

   ```
   scp cluster:~/.config/molbuilder/environments/sol.json \
       ~/.config/molbuilder/environments/
   molbuilder jobset machines        # confirm it arrived AND parses
   ```
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

---

## 1a. The commands, end to end

Verified 2026-08-22 by running each step against an isolated `HOME`, so the
paths below are what the code does rather than what it intends.

**On the target** (a login node is fine — a scheduler is read from `sinfo`,
not from being on a compute node):

```
molbuilder jobset probe --write --name sol
# -> wrote ~/.config/molbuilder/environments/sol.json
```

The name is yours; it is the filename stem, and it is what `--target` will
say. `--write` shows what it measured and asks before overwriting an existing
record, difference by difference — silence keeps the record, `--yes` takes
every probed value.

**On your machine**, copy the file into this directory and confirm:

```
scp cluster:~/.config/molbuilder/environments/sol.json \
    ~/.config/molbuilder/environments/
molbuilder jobset machines
```

`machines` prints every record, its path, and when it was measured. It is the
only step that answers *"did the copy land, and does it parse?"* — a record
that is present but corrupt is **listed and marked**, never skipped, because a
silently-dropped record looks exactly like one that was never copied.

**Then prepare, and send it:**

```
molbuilder jobset prep run coarse --bundle Au-BDT-Au/optimization/Relax --target sol
rsync -a Au-BDT-Au/optimization/Relax/ cluster:~/molbuilder/projects/Au-BDT-Au/optimization/Relax/
```

`launch` is **not** run here — starting a job happens where the job runs
(§ 5's closing note).

**Where the files are.** In the config directory —
[`configuration.md`](?doc=configuration.md) § 2.1c: `$MOLBUILDER_CONFIG_DIR`,
else `$XDG_CONFIG_HOME/molbuilder`, else `~/.config/molbuilder`. With neither
variable set:

| | path |
|---|---|
| this machine's record | `~/.config/molbuilder/environment.json` |
| a named target | `~/.config/molbuilder/environments/<name>.json` |
| the bundle's snapshot | `<bundle>/environment.json`, written by `prep` |

The three are different scopes, not copies: `record_scopes()` walks them
calculation → target → machine, first match wins.

## 2. What travels, and what does not

| | travels with the bundle | why |
|---|---|---|
| the deck, the wrappers, `task.json`, the template | ✅ written by `prep` | that is the point |
| the pseudopotentials | ✅ copied in at `prep` | the files are the same everywhere; the library path is not (`project-layout.md` § 2.6) |
| the structure pair | ✅ copied in | same reason |
| `environment.json` | ✅ snapshotted beside the bundle | so the target's capability is a fact of the calculation, not of whoever prepped it |
| **`env_init` — the preamble and activation** | ✅ **on the target's own record**, since 2026-08-24 — copied there by the probe from that machine's `molbuilder.json` | how a shell enters an environment there is a fact about that machine (`configuration.md` § 5 M-1) — see § 3 |
| the target's **env inventory** | ✅ same | whether `molbuilder-siesta-gpu` exists there is a fact; *which* env you want is a preference and stays in `molbuilder.json` |

**Everything else prep writes is machine-free.** Verified by inspection: of
every file `prep` produces, the only lines carrying a local absolute path
are the preamble lines, and they come from the target's record, not from
the renderer.

---

## 3. The bootstrap travels on the record

**How a shell enters its environment is a fact about the machine**, so it
rides the machine's record with the core count and the queue walls:
`Environment.env_init` carries `{preamble, activation}`. It is
**declared once, where molbuilder is installed** — in that machine's own
`molbuilder.json`, which `envs init-config` asks for at install, because
molbuilder needs it there before any record exists — and **`jobset probe
--write` copies it into every record it writes**, this machine's or a named
one *(user, 2026-10-02)*. Probe Sol on Sol, copy `sol.json` here, and `prep
--target sol` bakes `module load mamba` + `source activate` — Sol's answer,
not this workstation's. A copy that is wrong for the machine it describes is
edited by hand, in that record.

The generator reads that field and nothing else. There is **one rule for
local and remote**: the record states it, so the record is the answer — for
this machine too, whose record carries the copy of its own `molbuilder.json`
([`configuration.md`](?doc=configuration.md) § 4). A record that states none
is refused at prep, and the refusal says where to declare it
([`architecture.md`](?doc=execution/architecture.md) § 8.3).

> **What this section said until 2026-08-24, and what it cost.** It said the
> record *"does not hold `script_generation`, and it must not"* — reading
> M-1 as putting a preamble on the preference side. It then predicted, in
> detail, the failure that followed from its own rule:
>
> ```bash
> # what a laptop bakes, for a job that will run on a cluster:
> source /home/you/miniconda3/etc/profile.d/conda.sh   # does not exist there
> ```
>
> That is exactly what a browser-prepped bundle carried to Sol, and every
> trial died on it after a queue wait.
>
> The misclassification was the defect, not the merge rule. Put a preamble
> to M-1's own question — *what is this machine* versus *what do I want from
> it* — and `module load mamba` is not something anyone wants from Sol; it
> is how Sol works. Once it sits with the other facts, the bundle needs no
> per-calculation override, the join has nothing local to add, and the
> answer comes from the machine that has it.

**Which environment is still a preference.** `envs.<category>` says which
env you want for a category; the record says which envs that machine *has*.
The first is yours and lives in `molbuilder.json`; the second is a fact and
is checked against the target so a name that exists here but not there is
caught at prep rather than at `conda activate` on a compute node.

**A record can also be stale.** Nothing expires it: a cluster that changed
its partitions, its node mix, or its installed environments since the probe
will be prepped against the old answer. `environment.json` carries
`detected_at`, so the age is knowable; surfacing it is left to the surfaces
rather than made a refusal, because a six-month-old record is often still
exactly right. Re-probe when the machine changed. A calculation keeps the
record it was first prepped with (`configuration.md` M-3), so one already
prepped follows the new record only through a new prep, from the state saved
before its first (`molbuilder checkpoint restore`; `job-system.md` § 5.0).

---

## 4. Choosing the machine is the user's, always

Four ways the wrong machine was chosen **silently** when this section was
written. Each is the same defect: a question with several answers, answered by
default. **All four refuse now** *(re-read against `scheduler/record.py::machine_for`,
2026-09-29)* — the "today" column is the record of what was measured.

Each row below was **exercised through the `prep` command itself**, not
reasoned about — two of the four turned out to behave correctly already,
and one of those I had first written up as broken.

| # | situation | today | required |
|---|---|---|---|
| **C1** | several records reachable, no `--target` | ❌ silently uses **this** machine — two cluster records present, a workstation chosen, exit 0 | refuse, and name the choices — **done**: `AmbiguousTarget`, naming every record and *(this machine)* |
| **C2** | `--target` given, but the bundle already carries a different machine's record | ✅ **refuses** — `UnknownTarget.conflict`, by the name the copy carries since 2026-10-02 (`configuration.md` M-3: a calculation is set to the machine of its first prep; `this` is checked like any name, and a re-probe of its own machine is no conflict) — its way out the machine it is set to, or a new prep from a saved state (`checkpoint restore`) | *(correct today)* |
| **C3** | `--target sol` where `sol.json` exists but is malformed or a future schema | ❌ falls through to **this** machine, exit 0 | refuse: the user named it, so an unreadable record is an error, not a miss — **done**: `UnknownTarget.unreadable`, the named target validated whole before any scope is walked |
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

**A bundle says which machine it is for.** The calculation's copy of the
record names the machine of its first prep — `machine`, the `--target` name or
`this` — and only the copy carries it (`configuration.md` M-3). C2's refusal
compares that name with the one typed, and a person holding an rsync'd bundle
reads what it was prepared for in its `environment.json`.

---

## 5. What the browser may offer

The Task-setup tab may offer to prepare, subject to this section. It does
not get its own rules: it surfaces the ones above.

- It lists the reachable machine records and **requires a choice** when
  there is more than one — `GET /api/task-setup/machines` answers with the
  records and a `choice_required` flag computed by the same rule the CLI
  refuses on (§ 4, C1), so the two cannot disagree about what is ambiguous.
- **It reuses the components that exist.** The card is the shape chooser's
  `.ts-choice`/`.opt`, which the design already uses for exactly this kind
  of question — a choice with no default — plus `.card`, `.hint`,
  `.ts-needs` and `.ts-state`. No stylesheet was touched: a second chooser
  would be two components for one idea, and the pressed and hover states
  would then be maintained twice.
- An **unreadable** record is listed and marked, not hidden. The user wrote
  it; hiding it leaves them waiting for something that cannot happen.
- The commands the tab teaches carry `--target` whenever the choice was
  required — the record's name for a remote machine, **`this` for the box
  you are on** (the CLI's own name for it, since with several records `prep`
  refuses to guess) — and **`launch` does not**, because launching happens
  on the machine.

- **It shows what a `prep` would resolve, and from which file** —
  `GET /api/task-setup/resolved` serves `config_provenance`, the same block
  `prep` prints. Served rather than restated: a hand-written notice in the
  browser would be a second account of the same facts, free to drift from
  the one the terminal shows. Safe for a page by construction — provenance
  carries paths, presence and an allowlisted set of effective values, never
  file contents, so a TLS key or an OAuth secret cannot reach it.
- **There is no separate "the bootstrap will not travel" warning, and
  there must not be.** One existed until 2026-08-25 and both surfaces
  showed it. It asked the LOCAL config cascade whether the bootstrap came
  from the bundle's own file — the test § 3 states above it retracted — so
  it fired on every named-target prep, including the ones where the
  target's record had answered and the wrapper was correct. The rule now
  has one enforcement point and it is a refusal, not a warning: `prep`
  refuses any target — this machine or a named one — whose record cannot
  say how to enter its environment. A wrapper that is generated is a wrapper whose bootstrap
  came off the record it names.

> **This narrowed a stated boundary, deliberately.** `job-system.md` § 4 said
> *"`prep` and `launch` stay on the terminal by design"*, written when
> `prep` necessarily resolved capability from the machine it ran on.
> `--target` removed that necessity: capability comes from a record measured
> on the target — which is why the Task setup tab's Prep buttons may call the
> one prep entry (`job-system.md` § 5.3). **`launch` is unchanged** —
> starting a job still happens where the job runs.
