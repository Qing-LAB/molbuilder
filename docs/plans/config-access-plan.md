# One root for the server's own configuration

**Role:** plan (steps 1–4 built 2026-08-31; step 5, the read/write door, open)
**Domain:** ops · execution
**Started:** 2026-08-31
**Companions:** [`configuration.md`](?doc=configuration.md) §§ 2.1a–2.1e — the
contract this settles, edited as each step landed ·
[`ops/deployment.md`](?doc=ops/deployment.md) § 5 — the operator's view


> **Status.** Design, settled 2026-08-31. **No code yet** — written first on
> the user's instruction, and consolidated from four rounds of correction into
> one reading rather than a chronology of them.
>
> The rules here are what the code will be held to. Where they change a rule
> `configuration.md` already states, that document is the one that must be
> edited when the code lands — this file is a plan, not a contract.

---

## 1. What is on this machine

Measured, not assumed — this is the evidence the design answers to.

**Two roots, and the split between them is not by kind:**

| root | moved by `XDG_CONFIG_HOME`? | holds |
|---|---|---|
| `~/.config/molbuilder/` | **yes**, via `config_dir()` | `environment.json`, `environments/`, `notify`, `notify_keys` |
| `~/.molbuilder/` | **no** — `os.path.expanduser`, hardcoded | `secret.key`, `google_client_secret`, `logs/`, `run/` |

A third location wins over both for the machine config itself: a
`./molbuilder.json` in the working directory (`configuration.md` § 2.1).

### 1.1 Moving one root does not move the other

`molbuilder.json`'s `secret_key_file` reads `~/.molbuilder/secret.key` — the
running server's session key. `config_dir.py`, quoting `auth_setup`, says what
`XDG_CONFIG_HOME` is for:

> *"a user with `$XDG_CONFIG_HOME=/scratch/$USER` keeps secrets off the
> NFS-mounted `$HOME` on HPC nodes."*

Setting that variable moves `environment.json` and the notify tokens. It does
**not** move `secret.key`: nothing consults XDG to find it, because the path is
a literal in the config file under a root computed with a bare `expanduser`. A
person who does exactly what that sentence says moves some of their files and
not the one they were told to care about.

> **That sentence also claims more than this program can know.** Nothing here
> can tell whether a path is NFS, exported, group-readable by site policy, or
> backed up elsewhere. **The defect is not a broken safety promise — it is
> that there are two roots, so pointing the program at one place does not move
> everything.** That is a statement about our own layout, and we can be held to
> it. "Your secrets are off the shared filesystem" is not.

### 1.2 The same secret has two homes under two names

| | path | filename |
|---|---|---|
| what `auth_setup` writes by default | `<config_dir>/secret_key` | `secret_key` |
| what this machine actually reads | `~/.molbuilder/secret.key` | `secret.key` |

Different directory **and** different name. Running `auth-setup` today writes a
fresh session key where the server does not look, and reports success.

### 1.3 Who bypasses the one door that exists

`config_dir()` is the settled answer for the config root, and its docstring is
right about why. Callers: `runtime_config`, `scheduler/record`, `auth_setup`,
`monitor`. Bypassing it: `serve_daemon.py` (`run`, `logs`), `envs/_cli.py`
(`logs`), `web/blueprints/notify.py` (`reports`).

---

## 2. Scope — the server's own configuration, and nothing else

> *"I'm only talking about the server wide configuration file that stores
> secret and setups for the whole server."*

**In scope.** What configures *this installation*, the same for every project
on it:

| file | what it is |
|---|---|
| `molbuilder.json` | machine config — TLS paths, auth providers, scheduler, execution |
| `secret_key`, `google_client_secret` | the server's session and provider secrets |
| `environment.json`, `environments/<name>.json` | machine-scope environment records |
| `notify`, `notify_keys` | the server's notification tokens |
| `logs/`, `run/`, `reports/` | operational state — § 3.2 |

**Out of scope, by design and not as a later phase.** These have their own
owners and their own APIs, and that is intentional:

- the **Projects** tree and its file-management API — a per-user workspace,
  exposed on purpose;
- **per-calculation** files — `.molbuilder.json`, `task.json`,
  `<label>.template.toml`, `warm-files.toml` — which belong to a calculation;
- **sidecars and run records** — `molstruct`, `spectra`, `transport`, the run
  index — *data*, with schemas owned by their modules;
- **web request/response bodies**, and parse artifacts.

The survey found ~90 `json.load`/`dump` sites; the great majority are that
second list. Pulling them in is the over-reach this section exists to prevent.

---

## 3. Where things live

### 3.1 The config root

1. `$MOLBUILDER_CONFIG_DIR`, when set — **used exactly as given**.
2. `$XDG_CONFIG_HOME/molbuilder`, when that variable is set.
3. `~/.config/molbuilder`.

One variable, matching the `MOLBUILDER_DATA_DIR` / `MOLBUILDER_PROJECTS`
convention already in use. **An override, not a search path**: set it and that
is the root, entire — nothing falls back past it.

```
<root>/
  molbuilder.json          the machine config
  secret_key               the session key
  google_client_secret
  environment.json
  environments/<name>.json
  notify  ·  notify_keys
```

`config_dir()` keeps its name and its job, and gains the override.

### 3.2 Operational state

XDG has a convention and we follow it: **`$XDG_STATE_HOME`** (default
`~/.local/state`) was added in Base Directory spec 0.8 for state that persists
across restarts but is not portable enough for `$XDG_DATA_HOME` — the spec
names *logs* first. **`$XDG_RUNTIME_DIR`** (typically `/run/user/$UID`,
owner-only, cleared on logout) is the one for pidfiles and sockets.

Neither `~/.var/log` nor `~/.local/log` is a convention: `~/.var/app/` is
flatpak's, and `~/.local/log` is not in the spec.

| ours | variable | default |
|---|---|---|
| `logs/`, `reports/` | `XDG_STATE_HOME` | `~/.local/state/molbuilder/` |
| `run/` — pidfiles | `XDG_RUNTIME_DIR` | `<state>/run` when unset |

**And `molbuilder.json` may name them instead**, which is what makes a
standards-correct default acceptable rather than a second scattering. A person
who wants one directory holding everything sets one key; a person with a small
`$HOME` and a large scratch puts the logs on scratch without moving secrets.

```json
{ "paths": { "logs": "/scratch/$USER/molbuilder/logs",
             "run":  "/scratch/$USER/molbuilder/run" } }
```

**`~/.molbuilder/` ceases to exist**, as a root and as anything else.

#### This does not reverse the 2026-08-23 decision

`config_dir.py` records a user decision against a `paths.state` key, and the
reasoning still holds:

> *"A config key would be a second way to say one thing … It would also be
> circular for the first caller, which uses this to FIND `molbuilder.json`."*

That decision is about overriding **the config root**, and both objections are
specific to it: `XDG_CONFIG_HOME` already moves it, and a key *inside*
`molbuilder.json` cannot say where `molbuilder.json` is. **Neither applies to
operational state.** Resolution runs one way with no loop —

```
root (env or XDG)  →  molbuilder.json  →  logs / run / reports
```

— and there is no second way to say it, because we publish no environment
variable of our own for the state directory. `XDG_STATE_HOME` is not ours; it
moves every application at once, which is an account-wide setting rather than
one for this program.

#### The bootstrap constraint

Anything written **before** `molbuilder.json` is read — including the failure
to read it — has no configured destination and goes to the default. A
`paths.logs` override therefore takes effect for everything *after* config
load. A log that could only be written after parsing a file that failed to
parse is the one log nobody gets.

### 3.3 One location for the machine config

`configuration.md` § 2.1 makes the machine scope *first-found-wins* across
three locations, while machine ← project **deep-merges**. One configuration
system, two combining rules — the inconsistency the user objected to.

**Layering the three locations is the wrong repair.** It makes every key's
origin a thing to explain, and keeps three places a file may live. The repair
is the other direction, and it is this project's stated ordering — *delete >
one home > parameter > abstraction*:

> **There is ONE location for the machine config. Then nothing stops, because
> there is nothing to stop at.**

The cwd step goes. It is redundant — per-directory configuration already has a
scope, `.molbuilder.json`, which merges properly — and it is the entire source
of the shadowing § 2.1a had to grow a warning for.

**No migration command, and no compatibility with the old layout**
(*"no old design should be expected"*). A file in an old place is **named and
ignored, loudly**, with the path it should be at. § 2.1a's warning changes job
from *"this wins, and you may not want it to"* to *"this is not read; move
it"*.

---

## 4. What we guarantee, and what we do not

> *"I don't think we should overstep to be the nanny of everything, but rather
> to remind the user what is the right way to do this."*

**No detection, and no confirmation prompt.** There is no probe for NFS, no
"is this directory shared?" gate. A confirmation the program cannot verify is
theatre, and it trains people to click past it.

Instead:

- **Guarantee the layout** — one root, a stated set of files under it, nothing
  anywhere else. Checkable, and § 6's pin checks it.
- **Say what the root holds** — the setup path and `--help` name the directory
  and say plainly that it holds session and provider secrets, so the choice is
  made with the relevant fact in hand.
- **Report the mode** — `0600`/`0700`, warned about, never refused
  (`configuration.md` § 2.1b, already built).

**We are answerable for where our files are; the person is answerable for
where that is.**

---

## 5. One API — callers ask for the file, never for a directory

> *(User, 2026-08-31: "Variables are useful when it is used for those unified
> API to deduce the result … users … should go through API rather than go
> through directly for some variables. The variables are only concentrated and
> used within those APIs … and the users never sees them because they don't
> need to handcraft anything or derive anything. … that API should stay or be
> designed to be in the correct layer, so there is no reversion of
> dependency.")*

### 5.1 What is still handcrafted

Steps 1–4 stopped modules reading the environment: `tests/test_config_dir_has_one_home.py`
holds `MOLBUILDER_CONFIG_DIR` and every XDG variable to one module. **The
derivation moved up a level rather than going away.** Nine sites in seven
modules each hold a filename and join it onto a directory:

| module | what it derives |
|---|---|
| `runtime_config` | `<config>/molbuilder.json`, `<state>/logs`, `<state>/reports` |
| `auth_setup` | `<config>/secret_key`, `<config>/google_client_secret` |
| `scheduler/record` | `<config>/environment.json` |
| `monitor` | `<config>/notify` |
| `cli` | `<config>/notify_keys` |
| `serve_daemon` | `<state>/logs`, `<run>/serve-<port>.pid` |

Each is small and each is correct today. Together they are the same defect one
level up: seven modules that must agree about a filename, with nothing making
them. `configuration.md` M-4 already recorded this exact failure for a single
file — *"it gave `environment.json` one home for its FILENAME, which was a
string literal in three modules"* — and fixed that one. The rule was never
generalised, so the next six grew back.

### 5.2 The rule

**A caller names the FILE it wants, and gets a path. It never names a
directory, and never joins.**

```python
paths.machine_config()          # not config_dir() / "molbuilder.json"
paths.session_key()             # not default_secret_dir() / "secret_key"
paths.notify_token()            # not _config_dir() / NOTIFY_FILENAME
paths.serve_pidfile(port)       # not run_dir() / f"serve-{port}.pid"
paths.logs()                    # a directory, because it IS the answer
```

Filenames become constants inside that module and appear nowhere else. The
environment variables stay where they already are — read in one place, to
*derive* these answers — and no caller sees them.

### 5.3 Which layer, and why it can be the lowest one

**L1, stdlib-only, no molbuilder imports** — the property `config_dir.py`
already has, which is what lets `runwrap` ship its source beside a job.

That is only possible if the answers depend on nothing above L1. Today one
thing breaks it: `paths.logs` / `paths.run` / `paths.reports` are read from
`molbuilder.json`, which drags the API to L2 — and L2 is out of reach for
`serve_daemon` (L1) and for the shipped monitor. **That is the whole of the
`serve_daemon` conflict**: not a layering accident, but a config key placed
where the layer below it needs the answer.

So those three keys are **retired**, and the reasoning is `config_dir.py`'s own,
from 2026-08-23:

> *"`XDG_CONFIG_HOME` already moves this directory … A config key would be a
> second way to say one thing."*

`XDG_STATE_HOME` moves logs and reports; `XDG_RUNTIME_DIR` moves pidfiles. The
keys said the same thing a second way, and being a second way is exactly what
put them out of reach. **Deleting them removes the inversion rather than
working around it** — no injection point, no registry, no resolver installed at
startup. `paths.projects` stays: it is data rather than operational state, it
has no XDG equivalent, and `$MOLBUILDER_PROJECTS` is its documented override.

### 5.4 What the door still guarantees when it reads and writes

Reading and writing a file in § 2's list goes through the same module, so four
properties hold by construction rather than per caller:

1. **The path is computed in one place** — never `expanduser`, and now never a
   join, at a call site.
2. **The mode is enforced, not hoped for** — `0600` for a file holding secrets,
   `0700` for its directory, on write *and* checked on read. § 2.1b already
   does this for `molbuilder.json`; this generalises it.
3. **Writes are atomic** — `persist.write_bytes` does unique-temp + `os.replace`.

   > **`persist` cannot write a secret as it stands.** `write_bytes` takes no
   > `mode`: it uses the target's existing mode, or `0644` when the file does
   > not exist — deliberately, and its docstring says why (*"not what a shared
   > artifact should end up as"*). Right for a shared artifact, wrong for a
   > session key: a **new** secret written through it lands world-readable, and
   > § 2.1b's check would warn about a file we had just created. The door
   > creates the file at `0600` before handing the write over, rather than
   > giving `persist` a `mode=` — `persist`'s default is correct for what
   > `persist` is for, and only the caller knows it is writing a secret.
4. **An absent file is *unset*, never an error.**

## 6. Order

Contract first, then one mechanical change per commit, each with its own tests.

| # | step | notes |
|---|---|---|
| 1 | `MOLBUILDER_CONFIG_DIR` in `config_dir()` | plus the pin that nothing computes a per-user path itself |
| 2 | `state_dir()` / `runtime_dir()` (§ 3.2), move `logs`/`run`/`reports` off `~/.molbuilder/`, land the `paths` override | a default nobody can change is not the design |
| 3 | **the secret's one home** — `<root>/secret_key`, one filename | touches a live credential; **taken alone**, and it is the only step that can log a person out of their own server |
| 4 | delete the cwd step; `configuration.md` § 2.1a's warning becomes *"not read, move it"* | |
| 5 | **the one API** (§ 5): every per-user file named by a function, filenames held there and nowhere else, `paths.logs`/`run`/`reports` retired so it can be L1 | the layering conflict dissolves rather than being worked around |

**The pin asks the opposite of the usual question**: not *"is everything using
the door"* but *"does anything in § 2's first list compute a per-user path
without it"*.

Steps 2 and 3 change where files are looked for. They are not migrations —
there is no compatibility layer — so each names what must move, and the program
says so at the moment it cannot find something.
