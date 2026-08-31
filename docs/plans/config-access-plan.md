# One root for the server's own configuration

> **Status.** Design, 2026-08-31. Written before any code, on the user's
> instruction: *"we make sure we have one design, no old design should be
> expected."* Nothing here is built yet.

The user's diagnosis, in their words:

> *"why would the search stops at the cwd file? … this logic is wrong because
> it should check all options. another reason for us to unify config file
> access overall … detect any scattered .json file read/write or other config
> file read/write and put them all under the unified config access
> api/framework"*

---

## 1. What is actually on this machine

Measured, not assumed. Two roots, and the split between them is not by kind:

| root | moved by `XDG_CONFIG_HOME`? | holds |
|---|---|---|
| `~/.config/molbuilder/` | **yes**, via `config_dir()` | `environment.json`, `environments/`, `notify`, `notify_keys` |
| `~/.molbuilder/` | **no** — `os.path.expanduser`, hardcoded | `secret.key`, `google_client_secret`, `logs/`, `run/` |

And a third location wins over both for the machine config itself: a
`./molbuilder.json` in the working directory (§ 2.1 of `configuration.md`).

### 1.1 The secret is in the root that moving the other root does not move

`molbuilder.json`'s `secret_key_file` reads `~/.molbuilder/secret.key`. That
file is **in use** — it is the running server's session key.

`config_dir.py`'s docstring, quoting `auth_setup`, says what
`XDG_CONFIG_HOME` is for:

> *"a user with `$XDG_CONFIG_HOME=/scratch/$USER` keeps secrets off the
> NFS-mounted `$HOME` on HPC nodes."*

Setting that variable moves `environment.json` and the notify tokens. It does
**not** move `~/.molbuilder/secret.key`, because nothing consults XDG to find
it — the path is a literal in the config file, under a root computed with a
bare `expanduser`. So a person who does exactly what that sentence tells them
moves some of their files and not the one they were told to care about.

> **And that sentence claims more than this program can know** *(user,
> 2026-08-31)*. Nothing in the code can tell whether a path is NFS, exported,
> group-readable by a site policy, or backed up to somewhere else. **The
> defect is not a broken safety promise — it is that there are two roots, so
> pointing the program at one place does not move everything.** That is a
> statement about our own layout, which we can be held to; "your secrets are
> off the shared filesystem" is not.

### 1.2 The same secret has two homes under two names

| | path | name |
|---|---|---|
| what `auth_setup` writes by default | `<config_dir>/secret_key` | `secret_key` |
| what this machine actually reads | `~/.molbuilder/secret.key` | `secret.key` |

Different directory **and** different filename. Running `auth-setup` today
would generate a fresh session key into a file the running server does not
read, and report success. That is the user's complaint exactly — *"information
are saved in two places and i did not realize which one was the effective
one"* — and it is not hypothetical here.

### 1.3 Who bypasses the one door that exists

`config_dir()` is the settled answer for the config root and its docstring is
right about why. Its callers are `runtime_config`, `scheduler/record`,
`auth_setup`, `monitor`. Bypassing it:

- `serve_daemon.py` — `~/.molbuilder/run`, `~/.molbuilder/logs`
- `envs/_cli.py` — `~/.molbuilder/logs`
- `web/blueprints/notify.py` — `~/.molbuilder/reports`

---

## 2. Why the search stops, and why that is the wrong question

`configuration.md` § 2.1 makes the machine scope *first-found-wins* across
three locations, while machine ← project **deep-merges**. One configuration
system, two combining rules.

The user is right that this is inconsistent. But **layering the three machine
locations is the wrong repair**, because it makes every key's origin a thing
to explain — and it would keep three places a file may live.

The repair is the other direction, and it is the ordering this project already
states (*delete > one home > parameter > abstraction*):

> **There is ONE location for the machine config. Then nothing stops,
> because there is nothing to stop at.**

The cwd step goes. It is redundant — per-directory configuration already has a
scope, `.molbuilder.json`, which merges properly — and it is the entire source
of the shadowing § 2.1a had to grow a warning for. Delete the step and the
warning's job changes from *"this wins, and you may not want it to"* to
*"this file is not read; move it"*.

**No migration command, and no compatibility with the old layout** (user,
2026-08-31: *"no need for config migrate, we make sure we have one design, no
old design should be expected"*). A file in the old place is **named and
ignored**, loudly, with the path it should be at.

---

## 3. The framework

### 3.1 ONE root, named by the user or defaulted — and a stated layout under it

*(User, 2026-08-31: "We could allow, for example, an environment variable to
give the server the root directory for the config files. And by default that
should be the XDG directory. If not, if the user supplies an actual directory,
then everything goes from there … We should not guess from within the code
itself the nature of the directory that we detected. We can only guarantee
that once we have one directory detected or supplied, then all the config
files are consistently organized in that directory.")*

**The root, in order:**

1. `$MOLBUILDER_CONFIG_DIR`, when set — used exactly as given.
2. `$XDG_CONFIG_HOME/molbuilder`, when that variable is set.
3. `~/.config/molbuilder`.

One variable, matching the `MOLBUILDER_DATA_DIR` / `MOLBUILDER_PROJECTS`
convention already in use. It is an override, not a search path: **set it and
that is the root, entire** — nothing falls back past it, for the same reason
§ 2 gives for the machine config.

**Everything lives under that root, in a stated layout:**

```
<root>/
  molbuilder.json          the machine config
  secret_key               the session key
  google_client_secret
  environment.json
  environments/<name>.json
  notify  ·  notify_keys
  logs/  ·  run/  ·  reports/
```

**One root, not XDG's three** (`CONFIG` / `STATE` / `RUNTIME`), and that is a
deliberate reversal of this plan's first draft. The need being served is
*"point the program somewhere and have everything be there"*; three roots means
three variables to set and three chances to move two of them. A person putting
their session key on scratch wants the logs and the pidfile there too. The XDG
split is the better answer for a desktop application with a packager; this is
one directory a person is told about once.

`config_dir()` keeps its name and its job, and gains the override. **The
`~/.molbuilder/` root ceases to exist.**

### 3.1a What we guarantee, and what we do not

**We do not detect, and we do not ask for confirmation** *(user: "I don't
think we should overstep to be the nanny of everything, but rather to remind
the user what is the right way to do this")*. There is no probe for NFS, no
"is this directory shared?" prompt, no gate that must be cleared before the
program will run. A confirmation the program cannot verify is theatre, and it
trains people to click past it.

What we do instead:

- **Guarantee the layout.** One root, everything under it, no file anywhere
  else. This is checkable, and § 3.3's pin checks it.
- **Say what the root holds.** The setup path and `--help` name the directory
  and say plainly that it holds session and provider secrets, so a person
  choosing where to point it is choosing with the relevant fact in hand.
- **Report the mode**, as § 2.1b already does — `0600`/`0700`, warned about
  and never refused.

That is the whole of it. Whether the chosen directory is on a shared
filesystem is the person's knowledge, not ours, and the honest division is:
**we are answerable for where our files are; they are answerable for where
that is.**

### 3.2 The one door for reading and writing

Every read and write **of the files in § 3.3's list** goes through one module,
so that four properties hold by construction instead of per caller:

1. **The path is computed in one place** — never `expanduser` at a call site.
2. **The mode is enforced, not hoped for** — `0600` for a file holding
   secrets, `0700` for the directory, applied on write *and* checked on read
   (`configuration.md` § 2.1b, already built for `molbuilder.json`; this
   generalises it).
3. **Writes are atomic** — `persist.write_json` already does temp-file +
   `chmod` + rename; this is its caller, not a second implementation.
4. **An absent file is *unset*, never an error** — the rule every current
   reader implements separately.

### 3.3 Scope — the server's own configuration, and nothing else

*(User, 2026-08-31: "this is just about the configuration for the server
itself … project sidebar has its own directory and file management API exposed
to all the users, and that is okay because that is by design our intention. So
do not overcomplicate this situation. I'm only talking about the server wide
configuration file that stores secret and setups for the whole server.")*

**In scope — one root, one door.** The files that configure *this installation*
and are the same for every project on it:

| file | what it is |
|---|---|
| `molbuilder.json` | the machine config — TLS paths, auth providers, scheduler, execution |
| `secret_key`, `google_client_secret` | the server's session and provider secrets |
| `environment.json`, `environments/<name>.json` | the machine-scope environment records |
| `notify`, `notify_keys` | the server's notification tokens |
| `logs/`, `run/`, `reports/` | **see the note below** |

**Out of scope, and deliberately so — not an oversight, and not a later
phase.** These have their own owners and their own APIs, which is the design:

- the **Projects** tree and its file-management API — a per-user workspace,
  exposed on purpose;
- **per-calculation** files — `.molbuilder.json`, `task.json`,
  `<label>.template.toml`, `warm-files.toml` — which belong to a calculation,
  not to the server;
- **sidecars and run records** — `molstruct`, `spectra`, `transport`, the run
  index — which are *data* with schemas owned by their modules;
- **web request and response bodies**, and parse artifacts.

The survey found ~90 `json.load`/`dump` sites. The great majority are the
second list. Pulling them in would be the over-reach this section exists to
prevent.

> **`logs/`, `run/`, `reports/` are the one judgement call**, and it is stated
> rather than assumed. They are operational state, not configuration, so on a
> strict reading they do not belong in the table above. They are included for
> one reason: **they are the only things keeping `~/.molbuilder/` alive as a
> second root.** Leave them and the schema is not unified — a person who
> points `MOLBUILDER_CONFIG_DIR` somewhere still has files in two places,
> which is the whole complaint. Say so, and if the answer is to leave them
> where they are, then `~/.molbuilder/` survives and § 3.1's guarantee shrinks
> to "every *config* file", which is a smaller promise but still an honest one.

The pin that keeps this honest asks the opposite question of most: not *"is
everything using the door"* but *"does anything in the first list compute a
per-user path without it"*.

---

## 4. Order

Contract first, then one mechanical change at a time, each with its own tests.

1. **The root and its override** — `MOLBUILDER_CONFIG_DIR` in `config_dir()`,
   plus the pin that no module computes a per-user path itself.
2. **Move `logs/`, `run/`, `reports/`** off `~/.molbuilder/` and under the
   root.
3. **The secret's one home** — `<config_dir>/secret_key`, one filename. This
   one touches a live credential and is the only step that can log a person
   out of their own server, so it is taken deliberately and alone.
4. **Delete the cwd step**; the § 2.1a warning becomes "not read, move it".
5. **The read/write door**, and move each config reader onto it.

Steps 2 and 3 change where files on this machine are looked for. They are not
migrations — there is no compatibility layer — so each one names what must
move, and the program says so at the moment it cannot find something.
