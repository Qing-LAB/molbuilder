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
| **who writes each file** — a person, a probe, a producing verb, or the engine's own package | what the writer does internally, beyond the two properties in the next row |
| **the mode and the durability of every file listed here** — `0600`/`0700`, and that a write is atomic (§ 2.1b, § 2.3) | the bytes, the order of keys, the error text |
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
| 1 | **machine** | `molbuilder.json` — **one file**, in the config directory named by `$MOLBUILDER_CONFIG_DIR` or the XDG default. A copy in the working directory is **not read**, and you are told so (§ 2.1a) | every section |
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
| `molbuilder.json` (machine) | 1. `$MOLBUILDER_CONFIG_DIR/molbuilder.json` if that variable is set — **the root, exactly as given** (§ 2.1c)<br>2. `$XDG_CONFIG_HOME/molbuilder/molbuilder.json` if that variable is set<br>3. `~/.config/molbuilder/molbuilder.json` | **one file.** There is no search: the branches above choose a DIRECTORY, and the file is the one in it. A `./molbuilder.json` in the working directory is not read (§ 2.1a) |
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

**`./molbuilder.json` is NOT READ** *(landed 2026-08-31)*. It was step 1 of a
first-found-wins search until then, which is the state this section was written
to end: a working-directory file stood silently in front of the per-user one —
not merged, not consulted, not mentioned — so two files held configuration, one
took effect, and nothing said which.

It was also redundant. Per-directory configuration already has a scope of its
own, `.molbuilder.json`, which *merges* properly and is documented as the one
merge in this document. Anything a cwd `molbuilder.json` could express, the
project scope expresses better and without shadowing anything.

So the rule is:

1. **A machine-scope file belongs in the config directory** — the one named by
   `$MOLBUILDER_CONFIG_DIR` (§ 2.1c), else `$XDG_CONFIG_HOME/molbuilder/`, else
   `~/.config/molbuilder/`. `machine_config_path()` has one branch and no
   search; there is nowhere else for it to stop.
2. **Per-directory configuration is the project scope's job** —
   `<project-dir>/.molbuilder.json`, which merges rather than replaces.
3. **A `./molbuilder.json` is reported, not obeyed.** Leaving it silently
   ignored would recreate the confusion in the other direction — a file you
   edited that does nothing. So `runtime_config.machine_config_shadow()` says
   it is **not read** and names the file that is. That phrasing lives in one
   place, and every surface that resolves the machine scope shows it.

Retiring the step took **no migration and no compatibility layer** (user,
2026-08-31: *"no old design should be expected"*).

> **All of it landed on 2026-08-31**, and each piece is described where it
> belongs: the root override in § 2.1c, the state and runtime directories with
> their `paths` block in § 2.1d, and the session key's one home in § 2.1e.
> `archive/2026-09-01-config-access-plan.md` holds the reasoning and the order it was taken
> in; **this document describes what is.**

### 2.1b It holds secrets, so it is `0600` — checked, not just written

*(User, 2026-08-31: "we should constrain its chmod in contract and in
practice?")*

`molbuilder.json` is not ordinary configuration. It carries `tls.cert` and
`tls.cert` and `tls.key`, and the `auth.providers` block — paths to private
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

### 2.1c Naming the root outright — `MOLBUILDER_CONFIG_DIR`

*(Built 2026-08-31. `archive/2026-09-01-config-access-plan.md` § 3.1.)*

`config_dir()` resolves the per-user root in three steps:

1. `$MOLBUILDER_CONFIG_DIR`, when set — **used exactly as given**.
2. `$XDG_CONFIG_HOME/molbuilder`, when that variable is set.
3. `~/.config/molbuilder`.

Spelled like `MOLBUILDER_DATA_DIR` and `MOLBUILDER_PROJECTS`, which are already
the convention for *"the program's own `<thing>` directory"*.

**No `molbuilder` component is appended to the override, and the asymmetry with
`XDG_CONFIG_HOME` is deliberate.** `XDG_CONFIG_HOME` names a root shared by
every application, so ours must add its own name beneath it.
`MOLBUILDER_CONFIG_DIR` names *our* directory; appending to it would put the
files somewhere the person did not ask for. The two variables answer different
questions.

**It is an override, not a search step.** Set it and that is the root, entire —
nothing falls back past it, and a file in either of the other two places is not
consulted. A fallback here would recreate the shadowing § 2.1a exists to warn
about: one setting, two files, one silently winning.

An empty value is *not set* — `MOLBUILDER_CONFIG_DIR=` is how a shell clears a
variable it cannot unset, and treating it as a root would put the config at the
filesystem root.

`tests/test_config_dir_has_one_home.py` pins all of it, including that no other
module reads either variable itself.

**Who creates it.** Nobody had to, and that was the gap: every reader treats an
absent file as *unset*, so a machine on which the directory had never been made
was indistinguishable from one deliberately left unconfigured — and
`script_generation.activation`, which has no default, made that state a refusal
to render any wrapper (`execution/running-a-job.md` § 5.2). **`molbuilder envs
init-config` creates it**, and `bootstrap` runs it at the end of a first
install ([`ops/installation.md`](?doc=ops/installation.md) § 2.1). Nothing else
changes: the directory is still not *required* to exist, every caller still
writes on demand, and the seeding never overwrites — a file that is there is
reported and left byte-for-byte as it is.

### 2.1d Operational state — `$XDG_STATE_HOME`, and `paths` may name it

*(Built 2026-08-31. `archive/2026-09-01-config-access-plan.md` § 3.2.)*

Logs, pidfiles and reports are **not configuration**, and they do not live in
the config root. XDG has directories for exactly this:

| ours | variable | default |
|---|---|---|
| `logs/`, `reports/` | `XDG_STATE_HOME` | `~/.local/state/molbuilder/` |
| `run/` — pidfiles | `XDG_RUNTIME_DIR` | `<state>/run` when unset |

`XDG_STATE_HOME` entered the Base Directory spec in 0.8 for state that persists
across restarts but is not portable enough for `$XDG_DATA_HOME` — and the spec
names *logs* first. `XDG_RUNTIME_DIR` is cleared when the session ends, which
is right for a pidfile; when it is unset (cron, a detached ssh, some
containers) the fallback is the **state** directory rather than a temp dir,
because a supervisor's pidfile that vanished underneath it would leave a
running server nothing can find.

**The variables are the only way to move them.** `paths` holds `projects`
and nothing else — see the note below.

> **The default answers before the config is read**, including when reading it
> fails. A `paths` override therefore takes effect for everything *after*
> config load, and a `molbuilder.json` that will not parse still has somewhere
> to say so. A log that could only be written after parsing a file that failed
> to parse is the one log nobody gets.
>
> **And that is why `paths.logs` / `paths.run` / `paths.reports` no longer
> exist.** They were added on 2026-08-31 and retired the same day. The `serve`
> supervisor is L1 and the config reader is L2, so the supervisor could not
> reach a config-derived answer — leaving two answers to one question, which is
> the defect this whole change removes. The keys were also a *second way to say
> one thing*: `$XDG_STATE_HOME` already moves logs and reports, and
> `$XDG_RUNTIME_DIR` already moves pidfiles, which is `config_dir.py`'s own
> 2026-08-23 reasoning against a `paths.state` key. Deleting them removed the
> inversion instead of working around it with an injection point. A config
> still naming one is **refused**, with the variable that replaces it named.

`~/.molbuilder/` — a second per-user root that moved with no variable at all —
**no longer exists**. `tests/test_config_dir_has_one_home.py` asserts that no
module builds a per-user path itself.

### 2.1e The session key has one home, and the config cannot name it

*(Built 2026-08-31. Cited by `runtime_config._SECRET_KEY_MOVED`,
`web.auth._install_secret_key` and `auth_setup.build_auth_block`.)*

**The key is `<config dir>/secret_key`.** It is created there on first run, at
mode `0600`, and both the server and `molbuilder auth-setup` resolve it through
`auth_setup.secret_key_path()` — one function, so the reader and the writer
cannot name different files.

**`secret_key_file` is retired**, and a config still carrying it is **refused**:

```
molbuilder.json: 'secret_key_file' is no longer configured.  The session key
has ONE home -- <config dir>/secret_key, beside this file -- and is created
there on first run …
```

Refused rather than ignored, for this document's usual reason: a setting that
is read and silently dropped looks effective, so someone would point it
somewhere and wonder why their sessions still died on every restart. The
message is its own sentence rather than the generic *unknown top-level key*,
because that one reads as a typo and sends the reader back to retype a key
that has no spelling.

**Why a configurable path was the wrong shape.** It is how the key came to live
in two places at once — this machine's config said `~/.molbuilder/secret.key`
while the wizard wrote `<config dir>/secret_key`, so running `auth-setup`
produced a fresh key the server never read *and reported success*. One file
with one home cannot do that.

**There is no ephemeral fallback any more**, and its absence is deliberate. It
existed for "no path configured", which cannot happen when the path is not
configurable — and it degraded silently into sessions that died on every
restart, behind a warning in a log nobody reads.

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

### 2.3 How a file molbuilder writes is written — atomically, and privately when it carries a credential

*(Built 2026-09-12, measured while inventorying every secret this package
touches. This row of § 0's table was added with it: the mode rule in § 2.1b had
already taken this ground, for one file.)*

**Two properties, and they are separate questions.** A file molbuilder writes
must survive being interrupted; a file carrying a credential must never be
readable by anyone but its owner, not even for the length of a write.

| property | means | what breaks without it |
|---|---|---|
| **atomic** | a reader never sees half a file, and a failed write leaves the PREVIOUS content in place | a crash, a full disk or a kill mid-write destroys what was there. For `notify_keys` that is every key ever issued |
| **private** | `0600` from the moment the inode exists — not fixed up afterwards | the bytes sit on disk readable by every account on the box for the length of the write |

**This package had one function for each, and neither had both.**

| writer | atomic | private |
|---|---|---|
| `persist.write_bytes` — *"THE atomic writer"*, adopted package-wide at U8 | **yes** | **no, deliberately** — its own docstring: *"`mkstemp` creates 0600, which is not what a shared artifact should end up as"*, so it widens to `0644` |
| `auth_setup.write_secret_file` | **no** — in-place `O_WRONLY|O_CREAT|O_TRUNC` on the target itself | **yes** |

And **every secret this package writes went through the second one** — the
session key, the Google client secret, `notify_keys`, `notify` — while the
non-secret `molbuilder.json` got the atomic one. That is the wrong way round: it
is the same inversion [`run-reports.md`](?doc=execution/run-reports.md) records
one level down, where *the key file was `0600` and the data it protects was not*.

**Why a rule that should have caught it did not.** R10 (2026-08-12) aligned what
it called *"the last in-place `O_TRUNC` write"* with the atomic writer, for
exactly this reason — *"a crash mid-write left a truncated config for every later
read to refuse."* It was not the last. `write_secret_file` is one, and it could
not be aligned, because going through `write_bytes` meant **losing `0600`**. The
two rules were in tension, the tension was never written down, and the writer
that needed both kept the weaker half.

> **THE RULE — privacy is a PARAMETER of the one writer, never a second
> writer.** `persist.write_bytes(target, data, mode=…)` replaces a whole file
> this package means to keep, and **a credential is not a reason to write one's
> own**: that is how the package ended up with two writers neither of which was
> both safe. Two shapes legitimately sit outside it — staging for validation,
> and appending — and each is named below with the reason; **nothing else may**,
> and a new one is the finding, not the fix. `mode=None` — the
> default, and every caller that existed before this section — keeps a shared
> artifact's mode. `mode=0o600` is how a credential is written, and it is
> strictly the **stronger** path rather than a compromise: `mkstemp` creates the
> temp at `0600`, so there is no moment at any other mode at all, and the target
> is never opened for writing.

**How the API is used.** A caller never picks a writing strategy. It picks the
door for the kind of file it has, and the door knows:

| to write | call | which is |
|---|---|---|
| a secret, whole | `auth_setup.write_secret_file(path, text)` | parent at `0700`, then `write_bytes(…, mode=0o600)` |
| `molbuilder.json` / `.molbuilder.json` | `runtime_config.write_config_scope(patch, …)` | merge over what is there, validate the merge, `write_bytes`, then `0600` |
| `molbuilder.json`, from the auth wizard | `auth_setup.emit_molbuilder_json` | stage privately, **validate the bytes on disk**, then replace |
| anything else, whole | `persist.write_json` / `write_bytes` | the shared-artifact mode |
| a log, **appended** | `serve_daemon._open_private` | the one case temp-and-rename cannot serve |

**Two shapes that are not `write_bytes` calls, and both are deliberate.**

The auth wizard *stages, validates, then replaces*: it writes a private temp,
reads that temp back through `read_config`, and replaces the target only if the
server would accept it. `write_bytes` cannot express this, because it replaces
immediately — there is no moment in it where the new bytes exist on disk and the
old file is still there to keep. The order matters for a reason that was measured
(2026-09-10): the file used to be rewritten and validated *afterwards*, so
`providers=[]` left the machine with a config the server refuses. It has both
properties § 2.3 asks for; what it does not share is the implementation, which is
the right trade for a writer whose whole job is the validation step.

**An append is the other, and it is a real exception**, not an oversight: a
log is added to rather than replaced, so there is no previous content to
preserve and temp-and-rename would discard the file on every line. Its mode
therefore goes on the descriptor at create time instead — `0600` in a `0700`
directory, fixed 2026-09-12 after the server log was measured at `0664` while
carrying a provider's `client_secret` (`web/auth_providers/oauth.py` routes one
there deliberately, to keep it out of the user-visible response). Same
principle, different mechanism, because the file has a different shape.

and **the path is always asked for, never assembled**: the directory from
`config_dir` (§ 2.1c), the filename from whichever module owns that file's
format (A11). § 3.1 is the whole tree with the resolving function beside each
entry.

**There is no `retrieve_secret("name")` door, and there must not be one.** A
single name-keyed registry of secrets has to re-spell filenames their format
owners own — which is the change that was tried and reverted inside one day on
2026-08-31 (`config_dir.py` records it) and that
`test_config_dir_has_one_home.py::TestNoModuleNamesOneOfThoseFilesItself`
refuses: that test asserts each of these filenames appears in **exactly** the
module entitled to spell it. The door a caller wants already exists and is
already path-free — it is the owning module's own resolver, and for the four
secrets that is `config_dir.session_key()`,
`config_dir.google_client_secret()`, `monitor.default_notify_path()` and
`monitor.notify_keys_path()` (§ 3.1 lists every file's). Measured 2026-09-12:
**no module outside an owner joins a secret filename to a directory**, so the
property a unified resolver would have been built to guarantee is one the code
already has.

**Three gains over careful in-place writing**, and they are the general reasons
to prefer temp-and-rename:

1. **the previous secret survives a failed write** — a full disk, a crash, a kill
2. **no loose window, rather than a short one** — the old code `fchmod`ed an inode that already existed; this one's inode is `0600` before it has a name
3. **a planted symlink is replaced, not followed** — `os.replace` acts on the symlink itself, so `O_NOFOLLOW` stops being load-bearing

**What this is not about.** It protects a write from being INTERRUPTED — a
crash, a full disk, a Ctrl-C — which needs only one person at one keyboard.
It is not about two programs writing at once, and nothing here should grow to
be: molbuilder is single-user software, and a second simultaneous writer is not
a situation it has.

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
| `secrets/README` | `envs init-config` | machine | **how to treat the secret files this directory is for** — the `0700`/`0600` rule, what belongs there (things `molbuilder.json` names by PATH), mock `notify` channel examples for all three kinds, and the **four** secrets that cannot live there because they have one fixed home each: `secret_key` (§ 2.1e), `google_client_secret`, `notify`, `notify_keys` |
| `environments/README` | `envs init-config` | machine | **that the probe runs on the TARGET, not here** — the three commands (probe there, copy here, `jobset machines` to confirm), the `--set` fallback when molbuilder cannot be installed there, and that this machine's own record is `../environment.json` and not in that directory |

### 3.1 The tree — where all of it actually sits

One picture, because *"the config directory"* is three roots and a person
configuring this should not have to assemble them from five sections. **Beside
every entry is the function that resolves it**, which is the whole of the access
rule: molbuilder asks for a path and is given one; nothing joins a directory to a
filename except the module that owns that filename (§ 2.1c for the directory,
A11 for the name).

```text
$MOLBUILDER_CONFIG_DIR, else $XDG_CONFIG_HOME/molbuilder, else ~/.config/molbuilder
│                                      0700   config_dir.config_dir()
├── molbuilder.json        what you want              0600   runtime_config.machine_config_path()
├── environment.json       what THIS machine is              scheduler/record.machine_scope_path()
├── secret_key             the session key            0600   config_dir.session_key()
├── google_client_secret   Google's OAuth secret      0600   config_dir.google_client_secret()
├── notify                 run-report channels        0600   monitor.default_notify_path()
├── notify_keys            run-report signing keys    0600   monitor.notify_keys_path()
├── environments/          one record per OTHER machine  0700
│   ├── README                  written by `envs init-config`
│   └── <name>.json             probed ON that machine, copied here
└── secrets/               files molbuilder.json names by PATH   0700
    ├── README                  written by `envs init-config`
    └── …                       a TLS key/cert, a provider's client secret —
                                your names, because your config names them

$XDG_STATE_HOME/molbuilder, else ~/.local/state/molbuilder
│                                             config_dir.state_dir()
├── logs/                  diagnostics — delete when fixed  0700   config_dir.logs_dir()
│   ├── serve-<port>.log        everything the server prints 0600   config_dir.serve_log()
│   └── serve-<port>.stacks.log thread stacks on SIGUSR1     0600   config_dir.serve_stacks_log()
└── reports/               per-run measurements — KEPT              config_dir.reports_dir()

$XDG_RUNTIME_DIR/molbuilder, else <state dir>/run
│                                             config_dir.runtime_dir()
└── serve-<port>.pid       the address stop/restart act on          config_dir.serve_pidfile()
```

**Three roots, not one, and the split is what each kind of file deserves.**
Configuration is edited and backed up; state grows and is deleted; a runtime
directory is *erased when the session ends*, which is right for a pidfile and
wrong for anything meant to outlive a logout. A person who wants them
somewhere else moves them with `$XDG_STATE_HOME` / `$XDG_RUNTIME_DIR` — **not
with a key in `molbuilder.json`**, which § 2.1d explains and which a
`paths.logs` / `paths.run` / `paths.reports` is refused for. (This sentence said
*"points `paths` at one place"* until 2026-09-12, restating advice retired on
2026-08-31.)

**The four files `molbuilder.json` cannot name** are `secret_key`,
`google_client_secret`, `notify` and `notify_keys`. Each has one fixed home so a
reader and a writer cannot mean different files — § 2.1e is the worked example of
what happens otherwise, and `secret_key_file` / `notify_keys_file` are **refused**
in config rather than ignored. Everything under `secrets/` is the opposite case
by design: `molbuilder.json` names those by path, so the name is yours and the
directory is a suggestion.

**`secrets/` may be empty on a working installation** and often is — a
workstation with no HTTPS and no sign-in needs nothing in it. An empty
`environments/` likewise means you only ever run locally.

### 3.2 What a first install seeds

`molbuilder envs init-config` writes it, and `envs bootstrap` runs that at the
end — so a first install arrives with a usable starting point rather than an
empty directory. It seeds the
config directory at `0700`, `molbuilder.json` at `0600` **as a template** —
every section that can be empty present and empty, each with a `_`-prefixed
comment saying who fills it (you, a command, or a probe) — plus `secrets/` and
`environments/` at `0700`, and `environment.json` from the probe. Two values are
**asked**, never detected, because install time is the one moment the answer is
known: `script_generation.activation` (it has no default) and `paths.projects`
(its default is inside the checkout, which is often not where you want it).
`--yes` takes both defaults **and prints them**.

**Seeding runs at the END of `bootstrap`, and its preconditions are checked at
the START** *(2026-09-12)*. Running it last is right: a failure there must not
throw away forty minutes of built environments, so it is reported and never
fatal. But the two things that make it fail are already true before the first
install — the config root is unwritable (or is a file), or there is no terminal
to ask the two questions in and `--yes` was not passed — so both are stated up
front, ahead of the `Proceed?` prompt, where declining still costs nothing.
`initconfig.seeding_blockers()` answers the first (it owns the directory it
writes); the CLI answers the second (it owns the prompting).

**And a bootstrap that seeded nothing exits non-zero.** It reported success until
2026-09-12, because the failure was printed but never counted and the exit code
came from the doctor pass alone — so a CI or `nohup` run that installed five
environments and created no config at all looked clean, and every later verb then
refused for want of `script_generation.activation`.

### 3.3 One producer, two surfaces

`<label>.template.toml` is written by
`describe.py` and by the web's build blueprint through **the same function**,
`template.template_with_values`. That is what makes *"the web writes the same
bytes as the CLI"* a checkable claim rather than an intention — the same shape as
§ 2.3's one writer, applied to a produced file instead of a configured one.

---

## 4. `molbuilder.json` — what you want

Twelve sections. Each declares which scopes may carry it, and whether its values
may be printed in a provenance log. Both facts live in one registry in
`runtime_config._SECTIONS`, which is why they cannot disagree.

**Two of the twelve are gravestones.** `notify_keys_file` and `notify_route`
were retired on 2026-08-31 and are now **refused wherever they appear** — they
carry no scope and reach nothing. They stay in the registry so that writing one
is answered *by name*, with what to do instead; dropping them would make the
same file fail as an unknown key, which tells the person nothing
([`run-reports.md`](?doc=execution/run-reports.md) § 4.3).

| section | scopes | in provenance logs? |
|---|---|---|
| `execution` | machine · project | **yes** |
| `script_generation` | machine · project | **yes** |
| `scheduler` | machine · project | no — except the routing **domain names**, which `config_provenance` prints |
| `tls` | machine | no |
| `auth` | machine | no |
| `admin` | machine | no |
| `envs` | machine | no |
| `checkpoint` | machine | no |
| `rate_limit` | machine | no |
| `notify_keys_file` | — | — |
| `notify_route` | — | — |
| `paths` | machine | **yes** |

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
