# Configuration — every file that is *set* rather than produced, and who writes each

**Role:** contract
**Domain:** (tree-wide)

**Companions — the contracts that own each file's internals, and where this
document and one of them disagree about a file's OWN contents, that one wins:**
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 6.1 (the
artifact registry — every file's schema string and authoritative module) ·
[`engines/template.md`](?doc=engines/template.md) (the catalogue and the
template) · [`engines/stages.md`](?doc=engines/stages.md) (`task.json`) ·
[`execution/project-layout.md`](?doc=execution/project-layout.md) § 2.3.1 (the
capability/allocation model these files serve) ·
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 5 (how the
`molbuilder.json` scopes merge) · [`workflow.md`](?doc=workflow.md) (what flows
into what).

## 0. What this document owns

**A file in this project is either *set* or *produced*.** A produced file is an
answer the program worked out — a deck, a job plan, a benchmark result — and the
registry in `job-contracts.md` § 6.1 lists every one of them with its schema.
This document covers the other kind: **the files somebody, or something,
configures.** Until it existed the question *"where do I set that, and who else
writes this file?"* was answered in five documents and completely in none.

| this document owns | it does not own |
|---|---|
| **which files are configuration**, and the one name each is called by | the keys inside any one of them |
| **who writes each file** — a person, a probe, a producing verb, or the engine's own package | what the writer does internally |
| **the scopes**, and which file wins when two of them speak | the merge algorithm, which is `running-a-job.md` § 5 |
| **the machine-facts rules** (§ 5) — the split between what is probed and what is chosen | the topology fields themselves, which are `scheduler/record.py`'s |
| **what is refused where**, and why refusal beats silence | the error text, which belongs to the validator |

**Two rules keep this document true**, and they are the same two that keep
[`workflow.md`](?doc=workflow.md) true:

> **R-C1 — this page states *who writes what, where*. It never restates a
> file's contents.** A key is named here only when the rule is about the key's
> *home*. The list of settings inside `molbuilder.json`'s `scheduler` block
> lives in `job-system.md`; the list of items in a template lives in
> `template.md`. A second copy is a copy that drifts.

> **R-C2 — where this page and a file's owning contract disagree about that
> file's insides, the owning contract wins. Where they disagree about *who
> writes it*, this page wins.** That is the one question this document was
> created to have a single answer to.

---

## 1. The sorting question — who writes it?

Every configuration file in this project has exactly one kind of writer. That is
the whole taxonomy, and it is what § 3's table is sorted by.

| writer | means | files |
|---|---|---|
| **a person** | somebody decided it. Nothing in the program may overwrite it | `molbuilder.json`, `.molbuilder.json` |
| **a probe** | a machine was asked, and it answered | `environment.json` |
| **a producing verb** | `describe` (or the web's Task-setup tab) wrote down what a calculation *is* | `<label>.template.toml`, `task.json`, `task.1st.json` |
| **the engine's package** | it ships with the code, as data rather than as Python | `catalogue.template.toml`, `<engine>/warm-files.toml` |

**The taxonomy is not descriptive; it is enforced.** Two rules follow from it and
both already exist in code:

- **A probe never writes a person's file, and never writes a preference** (§ 5,
  M-1). What partitions a cluster has is a fact. Which one you want is not.
- **A producing verb never writes a person's file.** `describe` records the
  calculation; it does not touch `molbuilder.json`. The one verb that ever wrote
  into a person's config was the scheduler prober, and § 5 M-1 is the rule that
  stopped it.

---

## 2. The scopes, and who wins

Three scopes exist. They are read in this order, and **later wins**:

| # | scope | where | what may live here |
|---|---|---|---|
| 1 | **machine** | `molbuilder.json` in the working directory, else `$XDG_CONFIG_HOME` | every section |
| 2 | **project** | `.molbuilder.json` in a project or calculation folder | `execution`, `script_generation`, `scheduler` — **and nothing else** |
| 3 | **calculation** | the folder itself: `task.json`, `<label>.template.toml`, `environment.json`, an optional `warm-files.toml` | what this one calculation is |

**A section that may not live in a scope is refused there, never ignored.**
`runtime_config._read_project` states the reason, and it is the argument behind
every refusal in this document: *"a section that is read, validated and then
silently dropped is worse than one that was never allowed — it looks effective,
and the folder is saved under rules nobody applied."*

`checkpoint` carries its own refusal message on top of the general one, because
the operator meeting it is mid-mistake about that specific thing: a
project-scope copy is a file somebody can edit *between a save and a restore*.

> **Scope-name drift, recorded rather than hidden.** Scope 2 is called
> `"project"` in the section registry, `"bundle"` in `config_provenance`'s
> output, and *"a project or calculation folder"* in the refusal message —
> three names for one scope. Naming is § 6.3's territory, so the fix belongs
> there; it is noted here because this is the page a reader arrives at with the
> question.

### 2.1 Where each file is looked for, in order

**Every lookup is stated here, and every one of them is a *first-found-wins*
fallback except where the table says merge.** Two different combining rules for
two files would be two things to remember, so there is one exception and it is
named.

| file | looked for, in order | combining rule |
|---|---|---|
| `molbuilder.json` (machine) | 1. `./molbuilder.json` — the **working directory**, legacy and **warned about** (§ 2.1a)<br>2. `$XDG_CONFIG_HOME/molbuilder/molbuilder.json` if that variable is set — **the home**<br>3. `~/.config/molbuilder/molbuilder.json` — the home when `XDG_CONFIG_HOME` is unset | **one file, never both** — the first found is the machine scope, entire |
| `.molbuilder.json` (project) | `<project-dir>/.molbuilder.json` | **deep-merged** over the machine file, project wins — *the one merge in this document*. Objects recurse, scalars and arrays replace |
| `environment.json` | 1. `<calculation>/environment.json`<br>2. a **named target**, when one was asked for: `<machine scope>/environments/<name>.json`<br>3. `$XDG_CONFIG_HOME/molbuilder/environment.json`, else `~/.config/molbuilder/environment.json`<br>4. a fresh probe — **only when the caller asked for one** | **whole record**, first found wins (M-3). No field merge |
| `catalogue.template.toml` | `molbuilder/data/` inside the installed package | one file; it ships with the code |
| `<engine>/warm-files.toml` | 1. `<calculation>/warm-files.toml`<br>2. `molbuilder/<engine>/warm-files.toml` in the package | first found wins — a calculation's tuned copy replaces the shipped one |
| `<label>.template.toml`, `task.json` | the calculation folder, and nowhere else | there is nothing to combine — one calculation, one description |

**`environment.json` has no working-directory step, and that is deliberate.** A
calculation folder is very often the working directory, so a cwd step would make
the machine scope and the calculation scope the *same file* whenever you ran
from inside a bundle — and M-3's precedence would be comparing a record against
itself. `molbuilder.json` can afford the cwd step because its calculation-scope
counterpart has a different name (`.molbuilder.json`, with the dot).

> **A search order is not a merge order.** `molbuilder.json`'s three locations
> are alternatives — finding one *stops the search*, so a cwd file with only a
> `tls` block does not inherit the XDG file's `execution`. Only the machine ←
> project pair merges, and only after each has been resolved to one file.

### 2.1a The machine scope has ONE home, and a cwd file is warned about

*(User, 2026-08-31: "I had instances where information are saved in two places
and I did not realize which one was the effective one … I prefer consistency
rather than all based on implicit rules.")*

**The home is the per-user config directory** — `$XDG_CONFIG_HOME/molbuilder/`
if that variable is set, else `~/.config/molbuilder/`. That is where
`auth-setup` writes, where `environment.json` already lives, and what every
instruction should name.

**`./molbuilder.json` still wins when it exists, and that is exactly the
problem.** Step 1 of § 2.1's search is a *first-found-wins* stop, so a working-
directory file silently stands in front of the per-user one — the per-user file
is not merged, not consulted, and not mentioned. Two files hold configuration,
one takes effect, and nothing says which.

Worse, **the cwd step is redundant**: per-directory configuration already has a
scope of its own, `.molbuilder.json`, which *merges* properly and is documented
as the one merge in this document. Anything a cwd `molbuilder.json` can express,
the project scope expresses better and without shadowing anything.

So the rule is:

1. **A machine-scope file belongs in the per-user config directory.** New
   installs write there; documentation names that path.
2. **Per-directory configuration is the project scope's job** —
   `<project-dir>/.molbuilder.json`, which merges rather than replaces.
3. **A `./molbuilder.json` is honoured and WARNED about.** It is not silently
   obeyed and not refused: refusing would break a machine that has one today,
   and obeying quietly is what caused the confusion. Every surface that resolves
   the machine scope says so — `runtime_config.machine_config_shadow()` is the
   one place that phrasing lives, and it names **both** paths, says which is in
   effect, and says plainly when a per-user file **exists and is being
   ignored**. That last case is the one worth the noise: it is the only state in
   which the same setting can be written twice and read once.
4. **The cwd step is on its way out.** It is kept for the machines that have one
   and is not a place to put a new file. Retiring it is its own decision,
   recorded here rather than left implicit.

The warning is not a diagnosis aid bolted on afterwards; it is the price of
keeping a search step whose whole failure mode is being invisible.

### 2.1b It holds secrets, so it is `0600` — checked, not just written

*(User, 2026-08-31: "we should constrain its chmod in contract and in
practice?")*

`molbuilder.json` is not ordinary configuration. It carries `tls.cert` and
`tls.key`, `secret_key_file`, and the `auth.providers` block — paths to private
keys, and provider credentials. A world-readable copy on a shared login node is
a real exposure, not a tidiness question.

**The rule, for the file and its directory:**

| | mode | why |
|---|---|---|
| `molbuilder.json` (either location) | **`0600`** | owner reads and writes; nobody else has any business with it |
| the per-user config directory | **`0700`** | a listable directory names the file even when the file itself is shut |

**Writing it this way was already done; checking it was not.** `auth_setup`
creates the file with `os.open(..., 0o600)` and `fchmod`s the descriptor
*before the first byte* — the mode is right before there is anything to read,
rather than being fixed afterwards by a `chmod` that races the write. That care
is worth keeping and is not what this section adds.

What it adds is that **an existing file's mode is checked on the way in**. A
file arrives loose in ways no writer controls: copied from another machine,
restored from a backup, created by an editor, `git checkout`-ed, or unpacked
from an archive that did not preserve modes. The careful writer never sees those,
so a file that is `0644` today is `0644` silently.

**A warning, never a refusal**, for the same reason as § 2.1a: refusing locks a
person out of their own tooling over a condition they can fix in one command,
and the fix is named in the message. `runtime_config.machine_config_mode_warning()`
is the one place it is phrased, so every surface says the same thing, and it
names the exact `chmod` to run. It says nothing when the mode is already tight —
the quiet case is the correct one.

### 2.2 Which file actually took effect is displayed, never inferred

Three files can supply a `scheduler` block — the working directory's, the
per-user XDG one, and the bundle's — so *"it read the wrong config"* is a real
and frequent diagnosis. Two rules make it a readable one:

- **Every refusal names the resolved path**, not the generic filename.
  `runtime_config.machine_config_path()` exists for exactly this and returns an
  absolute path; a message quoting `molbuilder.json` names three possible files
  and is therefore no answer. This was already learned once here (R10,
  2026-08-12) and reintroduced on 2026-08-17, where it cost thirteen confusing
  test failures whose real cause was a config file two directories up.
- **`config_provenance` lists every scope it consulted** — path, found or
  absent, and how it was reached — including `environment.json`'s two scopes,
  and then which file supplied each effective value. It is safe for logs by
  construction: paths and presence always, values only for the sections flagged
  printable (§ 4).

```text
config:
  machine     /home/you/.config/molbuilder/molbuilder.json   (found, via xdg)
  bundle      /work/calc/.molbuilder.json                    (absent)
  environment /work/calc/environment.json                    (found, via calculation)
  environment /home/you/.config/molbuilder/environment.json  (found, via machine)
  execution.mode = 'submit'   <- machine
  environment.domains: general
```

The two `environment` rows are listed in precedence order, so the record that
won is the first one marked found — here the calculation's, which is why the
menu reads `general` rather than the machine record's own.

---

## 3. The map — every configuration file

Sorted by writer, per § 1. **Schema strings and authoritative modules are
deliberately absent**: that is § 6.1's registry, and R-C1 forbids the copy.

| file | writer | scope | what it answers |
|---|---|---|---|
| `molbuilder.json` | a person | machine | **what this installation is and what you want from it** — the server's own settings, and the defaults every calculation inherits |
| `.molbuilder.json` | a person | project | the three sections above, overridden for one project or folder |
| `environment.json` | a probe | machine *and* calculation (§ 5 M-3) | **what the target machine is** — cores, GPUs, scheduler, and the queues you can actually reach |
| `<label>.template.toml` | `describe` / the Task-setup tab | calculation | **every parameter of this calculation**, with the value in force |
| `task.json` | `describe` / the Task-setup tab | calculation | **what changes** — the ladder, what varies, the structure reference |
| `task.1st.json` | the Task-setup tab | calculation | a partial description in flight; **removed** when the real one is saved |
| `catalogue.template.toml` | shipped with the code | the package | **the master list** — every parameter both engines know, with its metadata. `<label>.template.toml` is made from it |
| `<engine>/warm-files.toml` | shipped with the code | the engine's package | which files a warm restart carries. A calculation may carry its own tuned copy, and that copy wins |

**One producer, two surfaces.** `<label>.template.toml` is written by
`describe.py` and by the web's build blueprint through **the same function**,
`template.template_with_values`. That is what makes *"the web writes the same
bytes as the CLI"* a checkable claim rather than an intention.

---

## 4. `molbuilder.json` — what you want

Ten sections. Each declares which scopes may carry it, and whether its values
may be printed in a provenance log. Both facts live in one registry in
`runtime_config._SECTIONS`, which is why they cannot disagree.

| section | scopes | in provenance logs? |
|---|---|---|
| `execution` | machine · project | **yes** |
| `script_generation` | machine · project | **yes** |
| `scheduler` | machine · project | no — except the routing **domain names**, which `config_provenance` prints |
| `tls` | machine | no |
| `auth` | machine | no |
| `secret_key_file` | machine | no |
| `admin` | machine | no |
| `envs` | machine | no |
| `checkpoint` | machine | no |
| `rate_limit` | machine | no |

**Why the provenance column exists and why most rows say no.**
`config_provenance` answers *"where did that setting come from?"* at the moment
a setting takes effect — the question an inert fixture makes unanswerable. It is
safe to log **by construction**: it prints only the sections flagged safe, plus
the scheduler's routing-domain *names*. A section holding a secret, or a path to
one, is never printed, so the flag is a security boundary rather than a
verbosity preference.

**Why only three sections reach the project scope.** Those three are the ones a
*folder* can legitimately differ on — how it runs, how its scripts are built,
what it asks the scheduler for. The other seven are properties of the
installation: two folders differing on `auth` or `checkpoint` would be two
behaviours with nothing on disk explaining the difference.

---

## 5. `environment.json` — what was probed

*(Contract 2026-08-17, user decision. Stated in `job-contracts.md` § 6.1a until
this document existed; moved here 2026-08-17 because the machine-facts split is
the configuration model, and having it in the artifact registry made the
registry answer two questions.)*

**Two files answered the same question and neither knew the other existed.**
`environment.json` recorded `topology.gpu_type`, probed from `scontrol`.
`molbuilder.json`'s `scheduler.gpu.default_type` recorded the same physical
fact, probed from `sinfo`. Only the first reached the code that builds the ask.

The disagreement went deeper than a duplicated value. `scheduler/record.py`'s
`detect_site` leaves `qos` and `account` unset and says why: *"they are site
policy, not reliably derivable from `sinfo`, so they come from the user's
config, not detection."* In the same tree, `scheduler_probe.parse_allowed_qos`
derives exactly that from `sacctmgr -nP show assoc user=$USER format=QOS`. Two
modules disagreed about whether a fact is detectable — one probed it, the other
declared it unprobeable — and `Site.qos` / `Site.account` have been dataclass
fields that **nothing has ever written**.

### M-1 — the split is **fact vs preference**, not probed vs declared

*(Corrected 2026-08-17, hours after the first draft, by the user pointing at
the machine this actually runs on. The first version sorted by **probed vs
chosen** and was wrong — see the box below, which is kept because the mistake
is the clearest statement of the rule.)*

| | a **fact** about a machine | a **preference** of yours |
|---|---|---|
| answers | *what is this machine* | *what do I want from it* |
| file | `environment.json` | `molbuilder.json` |
| arrives by | **probe** when you are standing on the machine · **declaration** when you are not | always a person |
| examples | cores, GPUs and their type, memory, scheduler kind, the partitions and QoS you can reach and their walls, **how a shell enters an environment there** (`script_generation`), **which environments exist there** | which partition to default to, `gpu.exclusive`, `gpu.mem`, `defaults`, **which environment to use** |

> **The bootstrap is a fact, and it took a wasted afternoon to place it
> correctly** *(2026-08-24)*. `module load mamba` is not something a person
> wants from ASU Sol; it is how Sol works, and no other answer is available
> there. Put it to the question in this table's first row — *what is this
> machine* — and it lands in the fact column with the core count.
>
> `preparing-for-another-machine.md` § 3 read it the other way, called a
> preamble "a preference", and concluded the record **must not** carry it.
> The consequence was the failure that section itself predicts, in the same
> words it uses to predict it: a bundle prepped on the workstation baked
> `source /home/u/miniconda3/etc/profile.d/conda.sh` and every job on
> Sol died on a path that exists on neither the cluster nor anywhere it was
> sent.
>
> **The distinction that keeps the two apart**: which environment you want
> is a preference (`envs.<category>`); whether it EXISTS on that machine is
> a fact, and it is knowable by the probe — `conda env list` enumerates
> without entering, so the probe running in one env reports all of them.
> The circularity is only apparent: the probe needs *an* env to run, never
> the ones the generated script will use.

> **Why "probed" is the wrong axis.** *You can only probe the machine you are
> standing on.* Describe a calculation on a workstation to run it on a cluster
> — the ordinary workflow — and the cluster cannot be probed from where you
> are. Its partitions and walls get written down by hand. Those rows are
> **facts**; they simply arrived by declaration.
>
> Sorting by *probed* made the one case that MUST declare an error. It bricked
> `prep` on a workstation over a config block describing a machine elsewhere,
> and the refusal told the user to delete rows carrying `node_type`,
> `max_cores`, `max_mem_gb` and a GPU memory figure the prober's own note says
> **cannot be probed** — data with nowhere else to live.
>
> The model already carried the right answer and it was not read:
> `Environment.source`'s vocabulary is `scontrol` / `lscpu` / **`flag`**, and
> `flag` *is* the declared case; `resolve_environment(overrides=…)` is its
> door — fed by `jobset probe --set key=value` (typed by the `Topology`
> schema itself, unknown keys refused by name) and `--scheduler`
> *(2026-08-19)*.

**Probed beats declared where both exist** — standing on the machine beats a
hand-written note about it — so `scheduler.routing` is read as declared
capability and used when nothing has been probed. Declared rows ride through
**whole**, keeping the operator's own columns (R10, 2026-08-12: rebuilding a
row from a known-key list made drafting a column indistinguishable from not
writing one).

**A probe still never writes a preference.** That half of M-1 stands: what
partitions exist is a fact either way; which one you want is not.

**A probe never writes a preference.** `derive_scheduler_block` already draws
this line for itself — *"exclusivity + memory are POLICY, not probed … `gpu.mem`
must be configured as a site policy (it cannot be probed)"* — and then crosses
it, emitting a `directives` block whose partition is `route_parts[0]`, *the
cheapest*. Cheapest is a preference. What partitions exist is a fact and moves
to `environment.json`; which one you want stays a choice in `molbuilder.json`.

### M-2 — one shape, cluster or workstation

`environment.json` carries `scheduler: "slurm" | "workstation"` and the same
fields either way; a field that could not be detected is `null`, kept and never
omitted, so a consumer can tell *absent* from *unknown*.

`molbuilder.json`'s scheduler block can never serve this role — it is
SLURM-shaped by construction, down to its `kind` enum. That is why the probe's
target is this file. The rule was already recorded as an amendment to the
prober's own refusal message (`project-layout.md` § 2.3.1 M6, 2026-08-17: *"a
workstation records its capability in the same shape a cluster does"*); this is
the artifact that satisfies it.

### M-3 — two scopes, precedence and not merge

1. **the calculation** — `<calculation>/environment.json`, snapshotted by `prep`
   step 1 and, once written, never overwritten;
2. **the machine** — written by `jobset probe`, shared by every calculation here;
3. **a fresh probe** — when neither file exists, and only when the caller asked
   for one (M-4).

**And one more, which is a name rather than a location.** It is consulted
**second** — after the calculation's own snapshot, before this machine's
record — because asking for a target by name is more specific than asking for
wherever you happen to be, and less specific than an answer this calculation
has already taken. `jobset probe --name sol`
writes a record to `<machine scope>/environments/sol.json`, and `prep --target
sol` asks for it by name. That is how you prep for a cluster from a workstation
— the machine you are describing is not the machine you are on, so *which
record* stops being answerable by location alone.

Naming one is **refused when it is not there**, and refused again when it
contradicts a snapshot the calculation already carries. Both refusals are the
same rule: a target the user typed is an instruction, and silently ignoring an
instruction is worse than stopping. A typo'd `--target` on an already-prepped
folder used to prep happily against whatever was snapshotted, which is exactly
the mistake the flag exists to catch.

The first one found is the whole answer. **There is no field-level merge.** Two
partial records blended at read time would describe a machine that exists in no
file — and it would silently defeat `resolve_target`'s standing guarantee that
two stages of one calculation cannot disagree about their own target. A
calculation that should follow a re-probed machine deletes its file.

The order matches § 2's, deliberately: a second precedence rule with different
edges is a second thing to remember.

### M-4 — one door, and the filename has one home

`scheduler/record.py` owns the schema, the dataclasses and the JSON round-trip. It
does **not** own the file, and that gap is why three call sites grew three
different shapes — a raw `write_text`, a read returning an `Environment`, and a
second read returning a plain `dict`.

| name | answers |
|---|---|
| `FILENAME` | the name `"environment.json"`, which was a string literal in three modules |
| `read_environment(path)` · `write_environment(env, path)` | one record at **one file**, or `None` — malformed is `None`, not an exception |
| `machine_scope_path()` · `environments_dir()` · `named_environments()` | **where** the records live: this machine's, and the named ones |
| `record_scopes(bundle_dir, target)` | the precedence as **data** — `[(label, path), …]`, in order |
| `machine_for(bundle_dir, *, target=, probe=)` | M-3's precedence, entire — the one function a caller asks |
| `UnknownTarget` | a named target that does not exist, or one that contradicts the calculation's snapshot |

**No consumer reads either file directly.**

> **Two shapes here were corrected on 2026-08-17 and the reasons generalise.**
>
> **The reader takes a FILE, not a directory.** It took a directory and joined
> `FILENAME` itself — which reads as tidy until a second location exists.
> Named targets are a second location, so a private `_read_named` grew beside
> it and there were two readers of one format again. A path-keyed door has one.
>
> **`probe` is off by default.** `machine_for` used to detect whenever no
> record answered, and `get_routing` calls it on every lookup — so a read-only
> getter shelled out to `sinfo`, `scontrol`, `lscpu` and `nvidia-smi`, 56 ms a
> call, and on a login node a round trip to the scheduler. Probing is opt-in
> and `prep` step 1 is the caller that opts in, because it is the one that
> writes the answer down afterwards.

### M-5 — this record stays JSON

§ 3's rule is *TOML when a person reads and edits it* — the reason
`<label>.template.toml` and `warm-files.toml` are TOML. Under M-1 no person
edits `environment.json`: a probe writes it and a person re-probes. A
machine-written, machine-read file stays JSON.

The cost of doing otherwise is concrete rather than aesthetic. `tomllib` reads
TOML and does not write it, so the only TOML emitter in this tree is
`template.py`'s, hand-rolled and guarded by round-tripping its own output back
through `tomllib` and comparing (*"the writer checks itself"*).
`scheduler/record.py` is stdlib-only on purpose — it ships to the target and runs in
a backend env with no molbuilder on it — so it could not import that emitter and
would have to carry a second one.

**The declared-override door is fed by flags, not a file** *(2026-08-19)*:
`jobset probe --set gpus_per_node=4` answers *"how do I tell it this machine
has 4 GPUs"*, and what persists is the probe's own record — this file, with
`source: flag` admitting how the fact arrived. If a *standing* declared file
is ever added instead, it is edited by a person, and § 3's rule then chooses
TOML for it — the *chosen* side of M-1, and a separate file from this one.

### M-6 — the probe asks before it overwrites *(2026-08-19)*

`jobset probe --write` over an **existing** record shows each place the probe
disagrees with the record and asks, **per difference**, which value survives.
The default is No — the record stays — so a weaker probe cannot erase a
declared fact: a login node that sees no GPUs probes `null`, and keeping the
recorded `4` is one keystroke. EOF keeps everything — a scripted probe
without `--yes` changes nothing (silence is no, the standing doctrine) —
and `--yes` takes every probed value. The reachable-domain **set** is one
question, not one per row. `detected_at`/`source` follow the new probe
either way: the kept values were re-confirmed now, and the stamp says when
the record was last looked at.

Creating a record where none exists is one consent — there is nothing to
clobber.

### The schema is `molbuilder/environment@2`

The reachable `(name, partition, qos, max_time)` **domains** land in the
record — the prober's `routing`, minus the preference M-1 removes.
`scheduler.routing` is **refused** in `molbuilder.json`, naming the file it
found the key in, because a stale hand-written menu silently dropped is the
case where "looks effective" ends in a job the scheduler rejects.

**`Site.qos` is still `None`, and that is deliberate** *(corrected 2026-08-17,
after the code was written)*. The plan said this field would finally be filled.
It should not be, by either route: a single QoS value is *which one you use* —
a preference, which M-1 keeps in `molbuilder.json` — and the *entitlement* it
was standing in for is the whole `domains` list, which is plural. The one
fact-shaped thing it could hold is SLURM's per-association **default** QoS, and
reading that needs a `sacctmgr` format string this was written without a
cluster to verify against. An unverified parse is how a probe starts reporting
something that isn't there, so the field stays empty until there is real output
to check it with.

A major bump rather than a minor, deliberately. `from_dict` already tolerates
missing and unknown keys, so an `@1` file would parse — and would read as *a
cluster with no reachable domains*, indistinguishable from a real cluster where
you hold no QoS. The bump is what makes an old record say *"I predate the
probe"* instead of answering the question wrongly.

---

## 6. The calculation's own files

Three files describe one calculation, and the split between them is the reason
a calculation folder is portable at all.

| file | holds | why it is separate |
|---|---|---|
| `<label>.template.toml` | **every** parameter, with the value in force | what the calculation *is*. Made from the catalogue, so a parameter exists in one place |
| `task.json` | **what changes** — the ladder, what varies, the structure reference | what the calculation *does*. Keeping it out of the template is what lets one template serve every stage |
| `environment.json` | the machine (§ 5) | **not portable** — it is the one file that describes where you are rather than what you asked for |

**The machine is deliberately not in the first two.** That is what lets you hand
the folder to a colleague on a different cluster, or benchmark it on a short
queue and run it on a long one, without editing it. The rule is
[`generator.md` § 4.1](?doc=execution/generator.md).

An engine's `warm-files.toml` ships in its package and a calculation may carry a
tuned copy that wins — the same *most-specific-scope-wins* shape as § 2, applied
to a file whose default is shipped rather than written.

---

## 7. What this document does not cover

**Produced files are not configuration**, and none of them appear above:
`job-set.json`, the rendered decks, `bench-result.json`, `run.json`,
`jobset-decisions.log`, checkpoint manifests. Every one is
registered in [`job-contracts.md` § 6.1](?doc=execution/job-contracts.md) with
its schema and its authoritative module.

The line is: **if deleting it loses a decision somebody made, it is
configuration. If deleting it only costs the time to recompute, it is a
product.** `environment.json` sits on the configuration side by that test even
though a probe wrote it — deleting it loses which machine an answer described.

---

## 8. Known drift

Recorded here because this is the page the question arrives at, and fixed in the
document that owns each.

| what | owner | status |
|---|---|---|
| One scope, three names — `"project"` · `"bundle"` · *"a project or calculation folder"* | `job-contracts.md` § 6.3 (identifier conventions) | open |
| ~~`verbose_comments` and `write_molwatch_log` are items in the catalogue for neither engine~~ — **withdrawn 2026-08-17: this was my misreading.** All three (`max_memory_mb` too) *are* catalogue items; they declare **no `engines` list**, so a per-engine query misses them while a plain lookup finds them. That is correct for what they are — a machine fact and two emitter switches, none of them engine-specific | — | **closed.** The rule they follow is *an item with no `engines` applies to every engine*, and [`engines/template.md`](?doc=engines/template.md) states it twice — in § 5's key table and in § 6.3's writer rule. This row claimed no document said it, which was the second half of the same misreading |
| `resolve_environment(overrides=…)` — **`jobset probe` is the caller** (`jobset/_cli.py`) and passes no `overrides`, so a machine fact cannot be declared through the verb yet. The missing sliver is the flag surface (`--set key=value`, a scheduler override), not the door or its caller — misread once (2026-08-19) as "the function has no caller" | this document, § 5 M-5 | open by design — the door and its caller exist; the flags are not built |
