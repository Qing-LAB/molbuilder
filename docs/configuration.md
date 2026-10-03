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
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 5 (what of
`molbuilder.json` reaches a calculation) · [`workflow.md`](?doc=workflow.md) (what flows
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
| **the mode and the durability of every file listed here** — `0600`/`0700`, and that a write is atomic (§ 2.1b, § 2.3). The modes are `placement.py`'s table since 2026-09-13, one row per file, and a file with no mode requirement says so there (`mode=None`) rather than being silently absent — which is what made this row an overstatement | the bytes, the order of keys, the error text |
| **where each file is looked for**, and which one answers when several could | how a reader parses it |
| **the machine-facts rules** (§ 5) — the split between what is probed and what is chosen | the topology fields themselves, which are `scheduler/record.py`'s |
| **what is refused where**, and why refusal beats silence | the error text, which belongs to the validator |

**Two rules keep this document true**, and they are the same two that keep
[`workflow.md`](?doc=workflow.md) true:

> **R-C1 — this page states *who writes what, where*. It never restates a
> file's contents.** A key is named here only when the rule is about the key's
> *home*; the list of items in a template lives in `template.md`. A second copy
> is a copy that drifts. **`molbuilder.json` is the one exception, and § 4 is
> it:** *what belongs in this file* is a question about homes, so § 4 lists
> every key the file may hold and what each is for, and leaves what a key may
> be set *to* with the contract it names.

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
| **a person** | somebody decided it. Nothing in the program may overwrite it | `molbuilder.json` |
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

Two scopes exist:

| # | scope | where | what may live here |
|---|---|---|---|
| 1 | **machine** | `molbuilder.json` — **one file**, in the config directory named by `$MOLBUILDER_CONFIG_DIR` or the XDG default. A copy in the working directory is **not read**, and you are told so (§ 2.1a) | every section of it (§ 4) |
| 2 | **calculation** | the folder itself: `task.json`, `<label>.template.toml`, `environment.json`, an optional `warm-files.toml` | what this one calculation is |

### 2.1 Where each file is looked for, in order

**Every lookup is stated here, and every one of them is *first-found-wins*.**
No file is merged with another.

| file | looked for, in order | combining rule |
|---|---|---|
| `molbuilder.json` (machine) | 1. `$MOLBUILDER_CONFIG_DIR/molbuilder.json` if that variable is set — **the root, exactly as given** (§ 2.1c)<br>2. `$XDG_CONFIG_HOME/molbuilder/molbuilder.json` if that variable is set<br>3. `~/.config/molbuilder/molbuilder.json` | **one file.** There is no search: the branches above choose a DIRECTORY, and the file is the one in it. A `./molbuilder.json` in the working directory is not read (§ 2.1a) |
| `environment.json` | 1. `<calculation>/environment.json`<br>2. a **named target**, when one was asked for: `<config dir>/environments/<name>.json`<br>3. `<config dir>/environment.json` — the config directory is § 2.1c's three branches, **`MOLBUILDER_CONFIG_DIR` included**; this row stated only the last two until 2026-09-12 — and when none answers, nothing: **no reader probes** (M-4; until 2026-10-01 a fourth row read *a fresh probe, only when the caller asked*) | **whole record**, first found wins (M-3). No field merge |
| `catalogue.template.toml` | `molbuilder/data/` inside the installed package | one file; it ships with the code |
| `<engine>/warm-files.toml` | 1. `<calculation>/warm-files.toml`<br>2. `molbuilder/<engine>/warm-files.toml` in the package | first found wins — a calculation's tuned copy replaces the shipped one |
| `<label>.template.toml`, `task.json` | the calculation folder, and nowhere else | there is nothing to combine — one calculation, one description |

**`environment.json` has no working-directory step, and that is deliberate.** A
calculation folder is very often the working directory, so a cwd step would make
the machine scope and the calculation scope the *same file* whenever you ran
from inside a bundle — and M-3's precedence would be comparing a record against
itself.

> **A search order is not a merge order.** `molbuilder.json`'s three locations
> are alternatives: the first that applies is the one file there is.

### 2.1a The machine scope has ONE home, and a cwd file is warned about

*(User, 2026-08-31: "I had instances where information are saved in two places
and I did not realize which one was the effective one … I prefer consistency
rather than all based on implicit rules.")*

**The home is the per-user config directory** — § 2.1c's three branches,
`$MOLBUILDER_CONFIG_DIR` first. That is where `auth-setup` writes, where
`environment.json` already lives, and what every instruction should name. (This
paragraph gave only the last two branches until 2026-09-12; § 2.1c is the rule
and restating two thirds of it here is how a reader comes to put a file where
nothing looks.)

**`./molbuilder.json` is NOT READ** *(landed 2026-08-31)*. It was step 1 of a
first-found-wins search until then, which is the state this section was written
to end: a working-directory file stood silently in front of the per-user one —
not merged, not consulted, not mentioned — so two files held configuration, one
took effect, and nothing said which.

So the rule is:

1. **A machine-scope file belongs in the config directory** — the one named by
   `$MOLBUILDER_CONFIG_DIR` (§ 2.1c), else `$XDG_CONFIG_HOME/molbuilder/`, else
   `~/.config/molbuilder/`. `machine_config_path()` has one branch and no
   search; there is nowhere else for it to stop.
2. **A `./molbuilder.json` is reported, not obeyed.** Leaving it silently
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

### 2.1b It names a private key, so it is `0600` — checked, not just written

*(User, 2026-08-31: "we should constrain its chmod in contract and in
practice?")*

`molbuilder.json` is not ordinary configuration. It carries `tls.cert` and
`tls.key` — where the server's private key is — and the `auth.providers` block,
who may sign in and with which client. A world-readable copy on a shared login
node is a real exposure, not a tidiness question. *(It also named the
providers' client-secret files until 2026-10-02; § 3.1.)*

**The rule, for the file and its directory:**

| | mode | why |
|---|---|---|
| `molbuilder.json` | **`0600`** | owner reads and writes; nobody else has any business with it |
| the per-user config directory | **`0700`** | a listable directory names the file even when the file itself is shut |

**Writing it this way was already done; checking it was not.** `write_config_scope`
writes it through `write_bytes(mode=0600)`, whose temp is `0600` *before it has a
name* (§ 2.3) — the mode is right before there is anything to read, rather than
being fixed afterwards by a `chmod` that races the write. (This paragraph named
`os.open` + `fchmod` until 2026-09-13, an implementation that has since been
replaced; the property it describes is unchanged.) That care is worth keeping
and is not what this section adds.

What it adds is that **an existing file's mode is checked on the way in**. A
file arrives loose in ways no writer controls: copied from another machine,
restored from a backup, created by an editor, `git checkout`-ed, or unpacked
from an archive that did not preserve modes. The careful writer never sees those,
so a file that is `0644` today is `0644` silently.

**A warning, never a refusal**, for the same reason as § 2.1a: refusing locks a
person out of their own tooling over a condition they can fix in one command,
and the fix is named in the message. **One sentence and one rule, the table's**
(§ 3.1): `placement`'s `molbuilder.json` row phrases it, naming the mode, why it
matters and the exact `chmod` to run. The terminal prints it among the tree's
findings (`placement.machine_config_warnings()`), and the Task-setup card asks
for that row alone (`placement.machine_config_finding()`), so both say the same
words. It says nothing when no bit is set beyond `0600` — the quiet case is the
correct one. *(Until 2026-10-02 a second sentence,
`runtime_config.machine_config_mode_warning()`, was this paragraph's "one
place" while the terminal printed the table's — two phrasings and two rules for
one fact, so a `0700` file was quiet on the card and a finding in the terminal
(T24, C8; user: one sentence, the one with the fix).)*

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

`tests/test_config_dir_has_one_home.py` pins all of it by setting the variables
and asking every door where its file went. That no other module reads either
variable itself is review's to hold (`process/code-audit.md` § 1c): a second
reader that agrees with `config_dir()` answers exactly as it does, so no run can
tell the two apart until the rule moves.

**The host env is always `molbuilder`** *(user, 2026-10-02: "let's enforce one
name")* — the env molbuilder itself runs from, which `scripts/install-env.sh`
creates and every `envs` verb reports against. Nothing renames it: `envs.host`
is refused by name (§ 4), and `MOLBUILDER_HOST_ENV` is not read.
*(Until that day the variable named the host env for one invocation, recorded
here 2026-09-13, and `envs.host` was its persistent home — but the installer
read only the variable and the `envs` verbs read the key too, so a name set in
the file alone had `bootstrap` build `molbuilder` while `envs list` and
`doctor` watched another env: two answers to one name, D21 and C23.)*

**Who creates it.** Nobody had to, and that was the gap: every reader treats an
absent file as *unset*, so a machine on which the directory had never been made
was indistinguishable from one deliberately left unconfigured — and the
activation, which has no default, made that state a refusal to render any
wrapper (`execution/running-a-job.md` § 5.2). **`molbuilder envs
init-config` creates it**, and `bootstrap` runs it at the end of a first
install ([`ops/installation.md`](?doc=ops/installation.md) § 2.1). Nothing else
changes: the directory is still not *required* to exist, every caller still
writes on demand, and the seeding never overwrites a file a person wrote — a
file that is there is reported and left as it is. Two things it rewrites, and
says so: molbuilder's own two READMEs, when their text is out of date, and a
`molbuilder.json` with no `env_init.activation`, which gains the one asked
([`ops/installation.md`](?doc=ops/installation.md) § 2.1).

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
**no longer exists**, and no module builds a per-user path itself — a rule
review holds, for the reason above.

### 2.1e The session key has one home, and the config cannot name it

*(Built 2026-08-31. Cited by `runtime_config._SECRET_KEY_MOVED`,
`web.auth._install_secret_key` and `auth_setup.build_auth_block`.)*

**The key is `<config dir>/secrets/secret_key`.** The server creates it there on its
first start, at mode `0600`, and reads it from then on — one resolver
(`config_dir.session_key()`) and one creator (`web/auth._install_secret_key`),
so the reader and the writer cannot name different files. `molbuilder
auth-setup` does not touch it: until 2026-09-13 it regenerated the key on every
run, logging every signed-in person out, with a different encoding from the
server's own.

**`secret_key_file` is retired**, and a config still carrying it is **refused**:

```
molbuilder.json: 'secret_key_file' is no longer configured.  The session key
has ONE home -- <config dir>/secrets/secret_key -- and is created
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
while the wizard wrote the one the resolver named, so running `auth-setup`
produced a fresh key the server never read *and reported success*. One file
with one home cannot do that.

**There is no ephemeral fallback any more**, and its absence is deliberate. It
existed for "no path configured", which cannot happen when the path is not
configurable — and it degraded silently into sessions that died on every
restart, behind a warning in a log nobody reads.

### 2.2 Which file actually took effect is displayed, never inferred

One file supplies `launch` — this machine's — and three can supply a machine
record (§ 5 M-3), so *"it read the wrong config"* is a real
and frequent diagnosis. Two rules make it a readable one:

- **Every refusal names the resolved path**, not the generic filename.
  `runtime_config.machine_config_path()` exists for exactly this and returns an
  absolute path; a message quoting `molbuilder.json` names three possible files
  and is therefore no answer. This was already learned once here (R10,
  2026-08-12) and reintroduced on 2026-08-17, where it cost thirteen confusing
  test failures whose real cause was a config file two directories up.
- **A malformed file is refused by the command that reads it**, in the
  resolved path's words — `Error: <path>: invalid JSON (…)`, exit 2, the CLI's
  usage-error surface — and a command that reads none runs: `molbuilder
  --help` and a group's own `--help` do, while a `jobset` verb's `--help` is
  refused with the file, because the group's header reads it first
  (`jobset/_cli._echo_config_root`). The CLI takes its machine snapshot on
  first use (`diagnostics.get_capabilities`), not before it parses its
  arguments, which until 2026-09-29 made a broken file fail even `--help`
  (plan W36 ⑥). The server reads the file at start and refuses to start
  ([`ops/deployment.md`](?doc=ops/deployment.md) § 5), and **a verb that makes
  a running process read it again reads it first** — `serve restart`, the
  Reload button, `jupyter start` and `restart` — so a broken file leaves the
  running process alone (§ 1.0b there).
- **`config_provenance` lists every scope it consulted** — path, found or
  absent, and how it was reached — including `environment.json`'s two scopes,
  and then which file supplied each effective value. It is safe for logs by
  construction: paths and presence always, values only for the sections flagged
  printable (§ 4).

```text
config:
  machine     /home/you/.config/molbuilder/molbuilder.json  (found)
  environment /work/calc/environment.json  (found, via calculation)
  environment /home/you/.config/molbuilder/environment.json  (found, via machine)
  launch.mode = 'submit'   <- machine
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
non-secret `molbuilder.json` got the atomic one. That is the wrong way round.

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
> both safe. A handful of shapes legitimately sit outside it — **each named
> below with its reason, and that list is the rule**. A write that is not on it
> is the finding, not the fix. *(This said "two shapes ... nothing else may"
> until 2026-09-13, when five were measured. An exception with a reason is a
> rule; an unnamed one is drift, and "nothing else may" was simply untrue.)* `mode=None` — the
> default, and every caller that existed before this section — keeps a shared
> artifact's mode. `mode=0o600` is how a credential is written, and it is
> strictly the **stronger** path rather than a compromise: `mkstemp` creates the
> temp at `0600`, so there is no moment at any other mode at all, and the target
> is never opened for writing.
>
> **`exclusive=True` is the same move for the second tension** *(2026-09-20)*.
> *Create if absent* and *replace* are different operations, and the session key
> needs the first: replacing it signs out everyone logged in. That was a named
> exception below — `web/auth.py` opening `O_EXCL` by hand — which bought *never
> replace* and paid with *never atomic*: a process dying between its `open` and
> its `write` left a 0-byte key that every later start refused. `os.link`
> creates the name or fails, as atomically as the rename it stands in for, so
> the parameter gets both. **Where a second writer exists to buy one property,
> check what it sold.**

**How the API is used.** A caller never picks a writing strategy. It picks the
door for the kind of file it has, and the door knows:

| to write | call | which is |
|---|---|---|
| a secret, whole | `auth_setup.write_secret_file(path, text)` | parent at `0700`, then `write_bytes(…, mode=0o600)` |
| a secret that must **never be replaced** once it exists — the session key | `write_bytes(…, mode=0o600, exclusive=True)`, catching `FileExistsError` | creates the name or fails; the caller then reads the existing secret back and signs with **that**, so losing the race costs nothing |
| `molbuilder.json` | `runtime_config.write_config_scope(patch)` | merge over what is there, validate the merge, then `write_bytes` at `0600` — the mode set as the file is written, never after — and the auth wizard's writer since 2026-09-13; it had one of its own |
| anything else, whole | `persist.write_json` / `write_bytes` | the shared-artifact mode |
| a log, **appended** | `serve_daemon.open_private` | the one case temp-and-rename cannot serve |
| anything written BY THE MONITOR on a compute node | `pathlib`, deliberately | it ships beside the job, where the only module of ours it can reach is one that travels with it — see below |

**The shapes that are not `write_bytes` calls, and why each one is not.**
*(The list was two long and said "nothing else may"; three more existed, which is
how that sentence came to be false — B2. Naming them is the fix: an exception
with a reason is a rule, an unnamed one is drift. Naming them is also what
retired one: written down, the session key's row read as a workaround rather
than a reason, and on 2026-09-20 it became `exclusive=True` on the one writer.
Four remain.)*

| outside the one writer | why |
|---|---|
| an appended log (`serve_daemon.open_private`) | below |
| **the supervisor's pidfile** | a few bytes rewritten at every start, holding an address rather than a secret, read by the next `stop`/`restart`. A truncated one is replaced on the next start; there is nothing in it to preserve |
| **a README this program seeds** (`envs init-config`, and a new project's skeleton) | written into a directory the same call just made — `envs init-config`'s two are also rewritten whole when their text is out of date, being molbuilder's own text — so there is no person's content to protect, and none of them carries a credential |
| **anything the monitor writes on a compute node** | it ships beside the job, and `persist` does not travel with it — see below |

**One of those needs the longer reason.** *(Two did until 2026-09-13: the
auth wizard staged a temp, validated the bytes on disk and replaced — a second
writer of `molbuilder.json` kept for the validation step, which the one door
already had: `write_config_scope` validates the merge before a byte lands. What
the door lacked was the wizard's one private finding — that a file can be
refused for a section the patch never touched — and that now lives in the door,
where every caller gets it.)*

**The other shape is outside it for a reason that is not about writing at all:
`monitor.py` SHIPS BESIDE A JOB.** It travels to the machine that runs the job
inside `mb_monitor.pyz` and is executed by **the job's own python**, in a
backend env where molbuilder is not installed (`runwrap.MONITOR_BUNDLE`,
`execution/running-a-job.md` § 2.0a). So its two files — the
`-runN.util.csv` it writes and the `-runN.monitor.log` it appends to
(`execution/run-reports.md` § 2.5) — use `pathlib`: importing `persist` here
would make the monitor die at import on every node, which costs the run's
status, its utilisation trace and its reports -- the session log says why
(`execution/run-reports.md` § 2.6), and nothing else does.

**And the reason is SHIPPING, not stdlib-ness.** `persist.py` is itself pure
stdlib — so is `config_dir.py`; the property is *depends only on stdlib*, never
*imports nothing of ours* (`config_dir.py`'s own header says so, and
`scheduler/record.py` imports `persist` on the strength of it). What kills the
import on a compute node is that `persist.py` is not THERE: what travels is
exactly the modules `runwrap.MONITOR_COMPANIONS` names, in one file
(`execution/run-reports.md` § 2.3). So the monitor's real rule is
**stdlib-only AND travels**, and each of those modules imports the next two
ways — from the package, or from the bundle.

**A second bundle, a second rule of the same shape.** A SIESTA force-constant
job also carries its FINISH, `mb_vibration.pyz` (`runwrap.VIBRATION_COMPANIONS`,
`engines/vibration.md` § 5.5): the analysis that turns the run's force
constants into its spectrum.  It needs arrays, so its rule is **the standard
library, numpy and ASE — which the SIESTA job envs carry for it — AND
travels**; its members import each other the same two ways.  The monitor's
set stays stdlib-only: the two bundles are built by one builder from two
tables, and nothing of the finish's reaches the monitor.

**What a missing module costs** — which only an incomplete bundle can cause:

| missing | what happens |
|---|---|
| `config_dir` | imported only when the channels are read; `load_channels` catches the `ModuleNotFoundError`, logs *reports off*, and the monitor keeps monitoring |
| any reader the monitor imports at start | the monitor cannot start: the session log holds its `monitor: starting` line and the error, and no `started` (`execution/run-reports.md` § 2.6); and the ending cannot be read either -- no failure hint, no warm retry (`execution/job-contracts.md` § 2.6) |
| the whole bundle | the wrapper logs *monitor: not started*, and `_mb_ending` answers that the ending cannot be read — no failure hint and no warm retry (`execution/job-contracts.md` § 2.6) |

**Do not route these through the one writer.** They are the one place in this
document where a truncated file is the cheaper risk, and the trade is
deliberate.

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
2026-08-31 (`config_dir.py` records it) and that review refuses: each of these
filenames is spelled in **exactly** the module entitled to spell it.

**A CONSUMER OF A SECRET IS HANDED THE SECRET, NEVER A PATH TO IT** *(user,
2026-09-20: "we should avoid user access the file directly, the api should
return the KEY/SECRET")*. Where a credential is stored is not part of the
contract a caller sees, because a path handed out is a path that reaches logs,
tracebacks and responses. The value doors are:

| credential | door | hands back |
|---|---|---|
| session key | `config_dir.read_session_key()` | bytes, or `None` when not yet made |
| an OAuth client secret | `runtime_config.provider_client_secret(entry)` | the string, read from its kind's fixed home, `config_dir.client_secret(kind)` — `molbuilder.json` names no secret |
| notify channels | `monitor.load_channels()` | the channels, already judged |
| run-report signing keys | `monitor.read_notify_keys()` | `(route, {user: key})` |

The path resolvers — `config_dir.session_key()`,
`config_dir.client_secret(kind)`, `monitor.default_notify_path()`,
`monitor.notify_keys_path()` (§ 3.1 lists every file's) — survive for
**management** only: creating the file, auditing its mode, and showing an
operator their own file. **TLS is the exception**: `get_tls` returns paths
because the server library that consumes them takes paths, and file
permissions are the control there. Measured 2026-09-12:
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
| `molbuilder.json` | a person | machine | **what you want from this installation** — the server's own settings, which environments to use, how a launch is sent. No value of a job, and no fact about a machine but one: `env_init`, how a shell enters an environment here, which the probe copies into every record it writes (§ 4) |
| `environment.json` | a probe | machine *and* calculation (§ 5 M-3) | **what the target machine is** — cores, GPUs, scheduler, the queues you can actually reach, and how a shell enters an environment there |
| `<label>.template.toml` | `describe` / the Task-setup tab | calculation | **every parameter of this calculation**, with the value in force |
| `task.json` | `describe` / the Task-setup tab | calculation | **what changes** — the ladder, what varies, the structure reference |
| `task.1st.json` | the Task-setup tab | calculation | a partial description in flight; **removed** when the real one is saved |
| `catalogue.template.toml` | shipped with the code | the package | **the master list** — every parameter both engines know, with its metadata. `<label>.template.toml` is made from it |
| `<engine>/warm-files.toml` | shipped with the code | the engine's package | which files a warm restart carries. A calculation may carry its own tuned copy, and that copy wins |
| `secrets/README` | `envs init-config` | machine | **how to treat the credentials this directory holds** — the `0700`/`0600` rule, the two kinds in it (fixed-home, which `molbuilder.json` cannot name, and operator-named), the **function each is reached through**, and mock `notify` channel examples for all three kinds |
| `environments/README` | `envs init-config` | machine | **that the probe runs on the TARGET, not here** — the three commands (probe there, copy here, `jobset machines` to confirm), and that this machine's own record is `../environment.json` and not in that directory |

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
├── environments/          one record per OTHER machine  0700   scheduler/record.environments_dir()
│   ├── README                  written by `envs init-config`
│   └── <name>.json             probed ON that machine, copied here   scheduler/record.named_environment_path()
└── secrets/               EVERY credential                     0700   config_dir.secrets_dir()
    ├── README                  written by `envs init-config`
    ├── secret_key         the session key            0600   config_dir.session_key()
    ├── <kind>_client_secret  an OAuth provider's secret  0600   config_dir.client_secret(kind)
    ├── notify             run-report channels        0600   monitor.default_notify_path()
    ├── notify_keys        run-report signing keys    0600   monitor.notify_keys_path()
    └── …                       a TLS key/cert, if you keep it here —
                                your name, because your config names it

$XDG_STATE_HOME/molbuilder, else ~/.local/state/molbuilder
│                                             config_dir.state_dir()
├── logs/                  diagnostics — delete when fixed  0700   config_dir.logs_dir()
│   ├── serve-<port>.log        everything the server prints 0600   config_dir.serve_log()
│   ├── serve-<port>.stacks.log thread stacks on SIGUSR1     0600   config_dir.serve_stacks_log()
│   └── jupyter-<port>.log      the notebook server's own —   0600   config_dir.jupyter_log()
│                               A SECRET SINK: jupyter prints
│                               its URL with the token in it
├── jupyter-lab/           the FRAMED Lab's own home        0700   config_dir.jupyter_lab_home()
│   ├── settings/               molbuilder's defaults, rewritten at every start
│   ├── user-settings/          what a person changes inside Lab — SHARED, never wiped
│   ├── workspaces/<port>/      which documents were open — PER SERVER, emptied at each start
│   └── jupyter_server_config.py  copied from data/ — suppresses .ipynb_checkpoints
└── reports/               per-run measurements — KEPT              config_dir.reports_dir()

$XDG_RUNTIME_DIR/molbuilder, else <state dir>/run
│                                             config_dir.runtime_dir()
├── serve-<port>.pid       the address stop/restart act on          config_dir.serve_pidfile()
├── jupyter-<port>.pid     the SHEPHERD's — jupyter stop's address  config_dir.jupyter_pidfile()
└── jupyter-<port>.json    where the notebook is AND ITS TOKEN 0600 config_dir.jupyter_runtime()
                           — the token reaches a live kernel
```

*(The notebook's four were absent from this tree, and from `placement.py`'s
table, until 2026-09-15 — so the audit below never checked the newest
credential in the tree. `plan.md` § 5n.8.)*

**The modes above are not prose: they are `molbuilder/placement.py`'s table**
*(2026-09-13)*. One row per entry drawn here — the OWNER's resolver, the expected
mode, whether it holds a credential, and why — with
`config_dir.ensure_private_dir` as the one creator and `placement.findings()` as
the audit that `envs doctor` prints and `machine_config_warnings` returns. **The
table is the authority and this tree cites it**, because three statements on this
page were measured false on 2026-09-12 while a reader had no way to check them:
the config root was said to be `0700` and nothing created it that way, the serve
log was said to be `0600` and `serve status` made it `0664`, and § 0's ownership
row claimed every file here had a stated mode when several had none. Where a row
says nothing about a mode, the table says so outright (`mode=None` — not
policed), which is a statement rather than a gap.

**Three roots, not one, and the split is what each kind of file deserves.**
Configuration is edited and backed up; state grows and is deleted; a runtime
directory is *erased when the session ends*, which is right for a pidfile and
wrong for anything meant to outlive a logout. A person who wants them
somewhere else moves them with `$XDG_STATE_HOME` / `$XDG_RUNTIME_DIR` — **not
with a key in `molbuilder.json`**, which § 2.1d explains and which a
`paths.logs` / `paths.run` / `paths.reports` is refused for. (This sentence said
*"points `paths` at one place"* until 2026-09-12, restating advice retired on
2026-08-31.)

**Every credential is in `secrets/`** *(2026-09-20, user: "all secret
key/sensitive files should stay in secret directory, and only unified api can
resolve them")*. The two kinds in there differ in **who names the file**, not in
where it sits:

| | named by | molbuilder's part | examples |
|---|---|---|---|
| **fixed home** | molbuilder, one function each; `molbuilder.json` **cannot** name them | creates the ones it writes, and polices the mode of all of them | `secret_key`, `notify`, `notify_keys`, and each OAuth kind's `<kind>_client_secret` (`google_client_secret`, `github_client_secret`, …) |
| **operator-named** | you, via a path in `molbuilder.json` | reads the path and hands it on — nothing else | the cert files only: `tls.cert` and `tls.key`, a CAS provider's `ca_certs` |

**No secret in `molbuilder.json` except the cert files** *(user, 2026-10-02)*.
A provider's client secret was operator-named until that day — the entry's
`client_secret_file` — and the wizard wrote the fixed home's own path into it,
so the server read whatever the file named while the mode audit looked at the
fixed home: one credential, two answers, and moving the config directory would
have split them. Now the kind names the file, `config_dir.client_secret(kind)`
resolves it, and a `client_secret_file` is refused by name (§ 4).

**molbuilder is not a security manager** *(user, 2026-09-20: "we don't operate
any TLS files — we just expose this information for any call that needs them.
File mode or access control is the system's business")*. It polices the mode of
what it NAMES — the first row, its own logs, and `secrets/` itself. A file you
name is yours and the operating system's: molbuilder reads it, passes it on, and
says nothing about its permissions.

That line was drawn after a check briefly crossed it. It read `tls.key` and told
the operator to `chmod 0600` a letsencrypt key at `0640 root:ssl-cert` — that
tool's *correct* permission, where tightening it breaks group access and renewal
undoes it anyway. Advice that is wrong to follow is worse than none. So `secrets/`
is **offered** as a home for an operator-named credential, never enforced as one.

`secret_key_file` and `notify_keys_file` remain **refused** in config rather than
ignored, and that is the property § 2.1e exists for: one home and one resolver, so
a reader and a writer cannot mean different files. **That was never a claim about
which directory** — only about there being exactly one — which is why moving the
four into `secrets/` costs nothing and removes the trap of a directory named
`secrets` that did not hold the secrets.

**The location rule is ENFORCED, not merely stated** *(2026-09-20)*. A `Place`
in `placement.py` carries `credential_store=True` when the file's *purpose* is
to hold a credential, and `placement.misplaced()` reports any such file whose
resolver points outside `secrets/`. That is a **code** defect rather than a
permissions one, so unlike the mode check it does not wait for the file to
exist. It had to become executable: measured the same day, `session_key()`
could be pointed back at the config root and **222 tests still passed**, because
every test and every audit row asks the same resolver and so moves with it.
Nothing compared the answer against the rule.

**`credential_store` means KEPT** — a credential that survives the session. It is
the only field the location rule reads, and defining it that way is what keeps
the rule free of exceptions. A serve log *contains* a `client_secret` and a
notebook runtime file *is* a token, but neither is kept: the log is an artifact,
the token is regenerated every start and deleted every stop. Both are `False`
here and both are still `0600`, because `mode` is the field that has always
policed that.

*This paragraph claimed something false for a few hours on 2026-09-20.* It said
`credential_store` existed because it and a second field, `holds_credential`,
"need different rules, so they are different fields" — but `holds_credential`
was **read by nothing, ever**. A third field then carried an exemption for the
notebook token whose own text read *"it is not a credential molbuilder KEEPS"*,
which was the definition admitting it was wrong. Three fields compensating for
one bad predicate; defining it as *kept* collapsed all three into this one and
deleted the checker's exemption branch.

**The one credential outside `secrets/`, and why.** The notebook's runtime file
(`jupyter-<port>.json`) holds a token that authenticates a browser to a live
kernel. It stays in the runtime directory, and the reason is carried as data on
its own row — `Place.credential_store` is `False` and `Place.why` says why —
rather than as a branch in the checker: it is not a credential molbuilder
*keeps*, being regenerated at every start, deleted at every clean stop,
meaningless without the live process, and one fact with the pidfile beside it.
Caveat recorded there too: `runtime_dir()` falls back inside the state root when
`$XDG_RUNTIME_DIR` is unset, so *"erased at logout"* is not guaranteed; the harm
is bounded because a token to a dead server authenticates nothing.

**Nothing builds these paths by hand.** Each is asked of its owner —
`config_dir.session_key()`, `config_dir.client_secret(kind)`,
`monitor.default_notify_path()`, `monitor.notify_keys_path()`, and
`config_dir.secrets_dir()` for the directory itself. The fifth credential, the
notebook token, is the same: `config_dir.jupyter_runtime()` owns its path, and a
consumer calls `jupyter.read_runtime()`, which hands back the token rather than
the path. `jupyter.py` itself does hold the path — it creates the file at
`0600` when the notebook starts and removes it when the notebook stops — and
that is the same carve-out every other door has: a path function survives for
MANAGEMENT, and it is the consumers that must be given the value instead.

`tests/test_config_dir_has_one_home.py` fails if any door stops moving with
`MOLBUILDER_CONFIG_DIR`, or if `placement.misplaced()` — which reads the table
`envs doctor` prints — finds a credential resolving outside `secrets/`. A
filename built into a path outside its owner is review's to find: while the copy
agrees with the door it gives the door's answer, so nothing a run can observe
separates them.
*(An AST check over the package's source stood here until 2026-09-26; it was
retired with the other source scans, `process/testing.md` § 3a.)*

**Human-facing text derives from the table too** *(2026-09-20)*.
`config_dir.relative_home(resolver)` renders a location as `secrets/notify` —
config-relative, because the text `notify-token` prints is a shell recipe that
resolves on a *cluster*. It was spelled by hand in three places and all three
went stale in the move: the printed recipe, `this_machine.html`, and the
issued-key panel in `this-machine/page.js`. Each told an operator to write a
webhook where the monitor does not look, and a notifier swallows failures by
design, so nothing would have said — and no scan of the Python source would
have either: the strings were `"$cfg/notify"` and HTML.

**`secrets/` is no longer empty on a working installation** — the session key
alone appears there on first server run. An empty `environments/` still means you
only ever run locally.

### 3.2 What a first install seeds

`molbuilder envs init-config` writes it, and `envs bootstrap` runs that at the
end — so a first install arrives with a usable starting point rather than an
empty directory. It seeds the
config directory at `0700`, `molbuilder.json` at `0600` **as a template** —
every section that can be empty present and empty, each with a `_`-prefixed
comment saying who fills it (you, or a command) — plus `secrets/` and
`environments/` at `0700`, and `environment.json` from the probe. Two values are
**asked**, never detected, because install time is the one moment the answer is
known: the **activation** (it has no default), written into `molbuilder.json`
as `env_init` beside the preamble found from the conda installation —
molbuilder needs it on this machine before any record exists, and `jobset
probe` copies it into every record it writes (§ 4) — and `paths.projects` (its
default is inside the checkout, which is often not where you want it), written
there too. `--yes` takes both
defaults **and prints them**.

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
refused for want of an activation.

### 3.3 One producer, two surfaces

`<label>.template.toml` is written by
`describe.py` and by the web's build blueprint through **the same function**,
`template.template_with_values`. That is what makes *"the web writes the same
bytes as the CLI"* a checkable claim rather than an intention — the same shape as
§ 2.3's one writer, applied to a produced file instead of a configured one.

---

## 4. `molbuilder.json` — what you want

**This file holds your preferences about this installation, and how a shell
enters an environment on this machine** *(user, 2026-10-01 and 2026-10-02)*.
Two kinds of value are refused in it:

- **A fact about a machine** — its cores, GPUs, scheduler and queues. That is
  the machine's record, `environment.json`, written ON that machine by `jobset
  probe` (§ 5 M-1) and copied to where you prep. One fact in two files is two
  answers, and the record is the one every prep reads.
- **A value of a job** — its queue, wall, memory, rank count, cores per rank and
  GPU count. A job states each one itself, in its description or on the command
  line, or prep refuses it and names where to state it
  ([`execution/architecture.md`](?doc=execution/architecture.md) § 5.2). A
  machine-wide default would be a value nobody stated for the job it lands on.

**How a shell enters an environment on THIS machine is the one fact kept
here** — `env_init` *(user, 2026-10-02)*. molbuilder runs on this
machine and needs it before any record exists, so it is declared here once, at
install. `jobset probe --write` copies it into every record it writes — this
machine's `environment.json`, or a named `<name>.json` written on the machine it
names and copied to where you prep — and every prep reads it from the TARGET's
record. **It is required** *(user, 2026-10-03: "this is required explicitly")*:
`jobset probe --write` refuses on a machine whose `molbuilder.json` states no
`env_init.activation`, naming `molbuilder envs init-config`, so a record is
never written without the copy and never keeps one from an earlier probe. A
copy that is wrong for the machine it describes is edited by hand, in that
record.

**Every key the file may hold.** A key not in this table is refused, never
ignored — inside a section as at the top level, one sentence naming what the
section holds (`runtime_config._refuse_unknown`) — and a key starting with `_`
is a comment. What a key may be set *to*
belongs to the contract in the last column; the table says what each key is for
and which code reads it, so *"is this setting doing anything?"* has an answer.
The registry `runtime_config._SECTIONS` holds the provenance flag, which is why
that column cannot disagree with the code.

| key | what it is for | read by | printed in provenance logs | owner |
|---|---|---|---|---|
| `launch.mode` | how `jobset launch` sends a job when no `--mode` is given: `direct` runs it here with bash, `submit` hands it to the scheduler. Unset, launch refuses and asks for `--mode` | `runtime_config.get_launch_mode` ← `jobset launch` | yes | [`running-a-job.md`](?doc=execution/running-a-job.md) § 5.4 |
| `envs.<category>` | which conda environment a backend's work runs in, when it is not the default name. Categories: `siesta`, `siesta-gpu`, `pyscf`, `mdtools`, `jupyter` — the host env is always `molbuilder` (§ 2.1c) | `diagnostics.Capabilities.env_for_category` ← the run script, the `envs` verbs | no | [`ops/installation.md`](?doc=ops/installation.md) |
| `envs.manager` | the absolute path of the conda-compatible command (`mamba`, `micromamba`, `conda`) when the one on PATH is not the one to use | `runtime_config.get_env_manager` ← the `envs` verbs | no | [`ops/installation.md`](?doc=ops/installation.md) |
| `env_init.activation` · `env_init.preamble` | how a shell on THIS machine enters a conda environment — `conda activate` or `source activate`, with no default — and the shell run before it (`module load mamba`, or sourcing conda's hook). Asked by `envs init-config`; `jobset probe --write` copies both into every record it writes, and refuses without the activation; prep reads them from the target's record | `runtime_config.get_env_init` ← `jobset probe` | no | [`running-a-job.md`](?doc=execution/running-a-job.md) § 5.2 |
| `paths.projects` | where the project tree is | `projects.projects_root` ← every surface | yes | § 2.1d |
| `tls.cert` · `tls.key` | the paths of the server's HTTPS certificate and key | `serve` | no | [`ops/deployment.md`](?doc=ops/deployment.md) § 5 |
| `auth.providers` · `auth.trust_proxy` | who may sign in and how; whether a proxy's forwarded headers are honoured. `providers` is written by `molbuilder auth-setup`, which leaves `trust_proxy` as it is | `web/auth.py` | no | [`ops/deployment.md`](?doc=ops/deployment.md) · [`ops/access-control.md`](?doc=ops/access-control.md) |
| `admin.emails` | who may use the operator-only web actions: restarting the server, the rate limiter's block list | `runtime_config.get_admin_emails` ← `web/admin.py` | no | [`ops/access-control.md`](?doc=ops/access-control.md) § 5–6 |
| `rate_limit.enabled` · `.window_404_s` · `.threshold_404` · `.window_total_s` · `.threshold_total` · `.cooldown_s` · `.trust_proxy` · `.max_tracked_ips` · `.allowlist` | the web server's request limiter: on or off, its thresholds, who it never blocks | `web/rate_limit.py` | no | [`ops/deployment.md`](?doc=ops/deployment.md) |
| `checkpoint.size_limit_bytes` · `checkpoint.engines.<engine>` | which run files are too large to save with a folder | `runtime_config.get_checkpoint` ← `checkpoint.py` | no | [`execution/checkpointing.md`](?doc=execution/checkpointing.md) § 4 |

**Refused by name**, each with what to do instead. A file ported from an older
install is answered, not merely rejected; dropping these from the registry would
make the same file fail as an unknown key, which tells the person nothing.

| key | since | what to do instead |
|---|---|---|
| `scheduler` — every key in it: `kind`, `directives` (partition, QoS, account, mail, export), `defaults` (time, cores per task, memory), `placement_priority`, `routing`, `gpu` | 2026-10-02 | A target's queues are its record's: `jobset probe --write` ON that machine, the record copied here (§ 5). A job's queue, wall, memory and shape are the job's own (`execution/architecture.md` § 5.2) |
| `script_generation` | 2026-10-02 | Renamed `env_init`, for what it holds: how a shell on this machine enters an environment — the same two keys |
| `execution` | 2026-10-02 | Renamed `launch`. `execution` is the run card in `task.json`, and means only that |
| `notify_keys_file` · `notify_route` | 2026-08-31 | The key file carries its own route ([`run-reports.md`](?doc=execution/run-reports.md) § 4.3) |
| `envs.host` | 2026-10-02 | Delete it: the host env is always `molbuilder` (§ 2.1c) |
| `secret_key_file`, at the top level or inside `auth` | 2026-08-31 | The session key has one home, `secrets/secret_key` (§ 2.1e) |
| `auth.providers[].client_secret` · `auth.providers[].client_secret_file` | 2026-10-02 | Neither the secret nor its path sits here: write the secret to its kind's home, `secrets/<kind>_client_secret` (§ 3.1; `molbuilder auth-setup` writes Google's) |
| `auth.providers[].service_validate_url` | 2026-10-02 | Delete it: python-cas derives the validate endpoint from `login_url`, and nothing ever read this key (accepted and ignored from 2026-09-12) |
| `rate_limit.admin_emails` | 2026-08-03 | The top-level `admin` section — one list for the block list and the restart ([`ops/access-control.md`](?doc=ops/access-control.md) § 5) |
| top-level `cert` · `key` | 2026-09-02 | The `tls` section |
| `paths.logs` · `paths.run` · `paths.reports` | 2026-08-31 | `XDG_STATE_HOME` / `XDG_RUNTIME_DIR` (§ 2.1d) |

**Why the provenance column exists and why most rows say no.**
`config_provenance` answers *"where did that setting come from?"* at the moment
a setting takes effect — the question an inert fixture makes unanswerable. It is
safe to log **by construction**: it prints only the sections flagged safe, plus
the names of the record's queues. A section holding a secret, or a path to one,
is never printed, so the flag is a security boundary rather than a verbosity
preference.

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
*(The whole `scheduler.gpu` block is gone since 2026-10-01 and refused by
name: a card is a machine's fact and no GPU ask names one, and its partition,
whole-node and memory settings put this machine's choices into every target's
GPU job — `execution/gpu.md` § 1.2.)*

The disagreement went deeper than a duplicated value. `scheduler/record.py`'s
`detect_site` leaves `qos` and `account` unset and says why: *"they are site
policy, not reliably derivable from `sinfo`, so they come from the user's
config, not detection."* In the same tree, `scheduler/probe.py::parse_allowed_qos`
derives exactly that from `sacctmgr -nP show assoc user=$USER format=QOS`. Two
modules disagreed about whether a fact is detectable — one probed it, the other
declared it unprobeable — and `Site.qos` has been a dataclass field that
**nothing has ever written** (as was `Site.account`, removed 2026-10-02).

### M-1 — the split is **fact vs preference**, not probed vs declared

*(Corrected 2026-08-17, hours after the first draft, by the user pointing at
the machine this actually runs on. The first version sorted by **probed vs
chosen** and was wrong — see the box below, which is kept because the mistake
is the clearest statement of the rule.)*

| | a **fact** about a machine | a **preference** of yours |
|---|---|---|
| answers | *what is this machine* | *what do I want from it* |
| file | `environment.json` | `molbuilder.json` |
| arrives by | **`jobset probe`, run on that machine** — measuring what it can, recording what it cannot see as you declare it there (`--set`, `--scheduler`), and copying how a shell enters an environment there from that machine's `molbuilder.json` (§ 4) | always a person |
| examples | cores, GPUs and their type, memory, scheduler kind, the partitions and QoS you can reach and their walls, **how a shell enters an environment there** (activation and preamble), **which environments exist there** | how a launch is sent (`launch.mode`), **which environment to use** (`envs`), where the project tree is, the server's own settings. **Never a value of a job** — no queue, wall, memory, rank count or core count (§ 4) |

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

> **Why "probed" is the wrong axis.** A fact can be one no probe can see:
> the activation is how Sol works, yet nothing on Sol reports it. So the
> probe both measures and records what you declare to it on that machine —
> `Environment.source`'s vocabulary is `scontrol` / `lscpu` / **`flag`**, and
> `flag` *is* the declared case; `resolve_environment(overrides=…)` is its
> door, fed by `jobset probe --set key=value` (typed by the `Topology` schema
> itself, unknown keys refused by name) and `--scheduler`.  The activation is
> declared once, in that machine's own `molbuilder.json`, and the probe copies
> it into the record (§ 4).

**A target's queues are its record's and nothing else's** *(user,
2026-10-02)*: probed on that machine, the record copied here. A queue list
typed into this machine's preferences (`scheduler.routing`, refused since)
described a machine its author was not on, and every job prepped for it
trusted a menu nobody measured.

**A probe never writes a preference.** `derive_scheduler_block` (replaced
2026-08-17) drew this line for itself — *"exclusivity + memory are POLICY, not
probed"* — and then crossed it, emitting a `directives` block whose partition
is `route_parts[0]`, *the cheapest*. Cheapest is a preference. What partitions
exist is a fact, in `environment.json`; which one a job uses is the job's own
statement (`allocation.domain`, `--domain`), never a default anywhere.

### M-2 — one shape, cluster or workstation

`environment.json` carries `scheduler: "slurm" | "workstation"` and the same
fields either way; a field that could not be detected is `null`, kept and never
omitted, so a consumer can tell *absent* from *unknown* — **but for the three
facts that travel with the record**, `env_init`, `conda_envs` and `env_arch`,
which are left out when empty *(user, 2026-10-03)*, and the probe says in its
notes which it left out and why (no environment manager answered, or it listed
none). Left out reads as unknown, and nothing is checked against an unknown
inventory.

`molbuilder.json` held a SLURM-shaped `scheduler` block until 2026-10-02 and
could never serve this role; the record is the one place a machine's kind is
written. The rule was already recorded as an amendment to the
prober's own refusal message (`project-layout.md` § 2.3.1 M6, 2026-08-17: *"a
workstation records its capability in the same shape a cluster does"*); this is
the artifact that satisfies it.

### M-3 — two scopes, precedence and not merge

1. **the calculation** — `<calculation>/environment.json`, snapshotted by `prep`
   step 1 and, once written, never overwritten; it names the machine it was
   taken for (`machine`: that prep's `--target`, `this` for the machine it was
   prepped on). A preview reads the record it would snapshot and writes
   nothing (`web/task-setup.md` § 11.1);
2. **the machine** — written by `jobset probe`, shared by every calculation here.

When neither answers, nothing does: **no reader probes** (M-4). A machine is
measured only when a person asks — `jobset probe`, or `envs init-config`
seeding this machine's own record through the same prober — and the refusal
names the probe.

**And one more, which is a name rather than a location.** It is consulted
**second** — after the calculation's own snapshot, before this machine's
record — because asking for a target by name is more specific than asking for
wherever you happen to be, and less specific than an answer this calculation
has already taken. `jobset probe --write --name sol`, run ON Sol, writes Sol's
record as `environments/sol.json`; copied to where you prep, `prep --target
sol` asks for it by name. That is how you prep for a cluster from a workstation
— the machine you are describing is not the machine you are on, so *which
record* stops being answerable by location alone.

**A calculation is set to the machine of its first prep, and that does not
change** *(user, 2026-10-02: "when a machine is set for a job, it is set, no
changing")*. A later prep's `--target` is checked against the name its copy
carries: the same name passes, whatever has been re-probed since — the
calculation reads its copy — and any other, `this` included, is refused.
Naming a target is also **refused when it is not there**. Both refusals are
one rule: a target the user typed is an instruction, and silently ignoring an
instruction is worse than stopping. A typo'd `--target` on an already-prepped
folder used to prep happily against whatever was snapshotted, which is exactly
the mistake the flag exists to catch.

**Preparing a calculation for another machine — or for a re-probed record of
its own — is a new prep**, from a saved state before its first: `molbuilder
checkpoint restore` ([`execution/checkpointing.md`](?doc=execution/checkpointing.md)
§ 7), then `prep` *(user: "we have persistency to roll back and start for a
new prep if we need to")*. Every refusal whose way out is another record says
so.

The first one found is the whole answer. **There is no field-level merge.** Two
partial records blended at read time would describe a machine that exists in no
file — and it would silently defeat `resolve_target`'s standing guarantee that
two stages of one calculation cannot disagree about their own target.

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
| `machine_for(bundle_dir, *, target=)` | M-3's precedence, entire — the one function a caller asks |
| `probe_command(name)` · `probe_line(name)` | the command that writes a machine's record — bare for this machine (`this` is reserved, and `probe` refuses it), `--name` for a named target, run on that machine; every refusal that asks for a record prints it, a line of its own with the note after `#` *(W52: four spellings, one of them a command `probe` refuses)* |
| `UnknownTarget` | a named target that does not exist, or one other than the machine the calculation is set to |

**No consumer reads either file directly.**

> **Two shapes here were corrected on 2026-08-17 and the reasons generalise.**
>
> **The reader takes a FILE, not a directory.** It took a directory and joined
> `FILENAME` itself — which reads as tidy until a second location exists.
> Named targets are a second location, so a private `_read_named` grew beside
> it and there were two readers of one format again. A path-keyed door has one.
>
> **No reader probes** *(W52, 2026-10-01)*. `machine_for` used to detect
> whenever no record answered, and `get_routing` calls it on every lookup — so
> a read-only getter shelled out to `sinfo`, `scontrol`, `lscpu` and
> `nvidia-smi`, 56 ms a call, and on a login node a round trip to the
> scheduler. Probing became opt-in on 2026-08-17, and `prep` step 1 stopped
> opting in when it stopped guessing (`project-layout.md` § 2.3.1: *step 1
> reads, it does not probe, ever*) — but a GPU run's sizing and every Task
> setup preview still opted in, measuring a machine with no record a moment
> before step 1 refused for want of one. The parameter is gone: a record is
> read, or the caller refuses with `probe_command`.

### M-5 — this record stays JSON

§ 3's rule is *TOML when a person reads and edits it* — the reason
`<label>.template.toml` and `warm-files.toml` are TOML. Under M-1 no person
edits `environment.json` but to correct a copied `env_init` (§ 4): a probe
writes it and a person re-probes. A
machine-written, machine-read file stays JSON.

The cost of doing otherwise is concrete rather than aesthetic. `tomllib` reads
TOML and does not write it, so the only TOML emitter in this tree is
`template.py`'s, hand-rolled and guarded by round-tripping its own output back
through `tomllib` and comparing (*"the writer checks itself"*).
Writing this record as TOML would pull that emitter into the prober, or grow a
second one, for a file a person edits only to correct one copied value.

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
question, not one per row. `detected_at`/`source` follow the new probe: the
kept values were re-confirmed now, and the stamp says when the record was last
looked at — but **a declared fact kept stays declared** *(user, 2026-10-03)*:
`source` is noted per section, and a section whose kept value was declared
keeps `flag` in its note (`lscpu+flag`), as M-5 says it admits.

Creating a record where none exists is one consent — there is nothing to
clobber. **A file at the record's path that does not read** — a newer schema, a
hand edit gone wrong — is not "none": the probe says it is there and that
writing replaces it, and replaces it only with that one consent (`--yes` gives
it, and the line is still said) *(W52, 2026-10-01: the reader answers absent
and unreadable alike, and the probe took the one for the other)*.

These are rows of `tests/data/machine_record.toml`, run down the road.

### The schema is `molbuilder/environment@2`

The reachable `(name, partition, qos, max_time)` **domains** land in the
record — the prober's `routing`, minus the preference M-1 removes. The whole
`scheduler` block is **refused** in `molbuilder.json` (§ 4), naming the file it
found the key in.

**`Site.qos` is still `None`, and that is deliberate** *(corrected 2026-08-17,
after the code was written)*. The plan said this field would finally be filled.
It should not be, by either route: a single QoS value is *which one a job
uses* — the job's own statement — and the *entitlement* it
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
| One scope, three names — `"project"` · `"bundle"` · *"a project or calculation folder"* | `job-contracts.md` § 6.3 (identifier conventions) | **closed 2026-08-23** — `project` everywhere; marked here 2026-09-29. *(The project scope itself was removed on 2026-10-02: `molbuilder.json` is one file, § 2.)* |
| ~~`verbose_comments` and `write_molwatch_log` are items in the catalogue for neither engine~~ — **withdrawn 2026-08-17: this was my misreading.** All three (`max_memory_mb` too) *are* catalogue items; they declare **no `engines` list**, so a per-engine query misses them while a plain lookup finds them. That is correct for what they are — a machine fact and two emitter switches, none of them engine-specific | — | **closed.** The rule they follow is *an item with no `engines` applies to every engine*, and [`engines/template.md`](?doc=engines/template.md) states it twice — in § 5's key table and in § 6.3's writer rule. This row claimed no document said it, which was the second half of the same misreading |
| `resolve_environment(overrides=…)` — **`jobset probe` is the caller** (`jobset/_cli.py`) and passes no `overrides`, so a machine fact cannot be declared through the verb yet. The missing sliver is the flag surface (`--set key=value`, a scheduler override), not the door or its caller — misread once (2026-08-19) as "the function has no caller" | this document, § 5 M-5 | **CLOSED — re-measured 2026-09-20.** Both flags exist: `jobset probe --help` lists `--set KEY=VALUE` (*"declare a topology fact the probe cannot see"*) and `--scheduler [slurm\|workstation]`. The row said *"the flags are not built"* long after they were |
| **One fact, two keys:** *is a proxy in front of this server?* is `auth.trust_proxy` — which installs werkzeug's `ProxyFix` for one hop, so sign-in builds the public URL and the client address the WHOLE server reads, the limiter's included, becomes the one the proxy added — AND `rate_limit.trust_proxy`, with which the limiter alone takes the FIRST `X-Forwarded-For` entry: the one a visitor can write when the proxy appends to the header, as nginx's usual setting does. With the first on, the second adds only that forgeable reading; and the first cannot be set without sign-in (`auth` requires providers), so a server with no sign-in behind a proxy has only the forgeable one | this document, § 4 | **OPEN — found 2026-10-02**, both `false` on the machine it was found on. One key, read once by the server and installing `ProxyFix` for everything, is the fix; which section holds it is the user's call. *(This row first said the limiter would count every client as the proxy with only the sign-in key on — wrong: `ProxyFix` rewrites the address the limiter reads.)* |
