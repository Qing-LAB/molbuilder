# One door for every per-user file

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

### 1.1 The secret is in the root that cannot be moved

`molbuilder.json`'s `secret_key_file` reads `~/.molbuilder/secret.key`. That
file is **in use** — it is the running server's session key.

`config_dir.py`'s docstring states the promise this breaks, quoting
`auth_setup`:

> *"a user with `$XDG_CONFIG_HOME=/scratch/$USER` keeps secrets off the
> NFS-mounted `$HOME` on HPC nodes."*

Setting `XDG_CONFIG_HOME` moves `environment.json` and the notify tokens. It
does **not** move `~/.molbuilder/secret.key`, because nothing consults XDG to
find it — the path is a literal in the config file, under a root computed with
a bare `expanduser`. The one file the promise exists for is the one file it
does not cover.

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

### 3.1 The roots — one module, and XDG's own separation

Two roots today, split by accident. The XDG spec already separates these by
kind, and the separation is real: configuration is edited and backed up,
runtime state is not.

| function | env var | falls back to | for |
|---|---|---|---|
| `config_dir()` | `XDG_CONFIG_HOME` | `~/.config/molbuilder` | `molbuilder.json`, `environment.json`, `environments/`, secrets, notify tokens |
| `state_dir()` | `XDG_STATE_HOME` | `~/.local/state/molbuilder` | `logs/`, `reports/` |
| `runtime_dir()` | `XDG_RUNTIME_DIR` | `state_dir()/run` | pidfiles, sockets |

`config_dir()` exists and does not change. The other two are new and replace
every hardcoded `~/.molbuilder/…`. **`~/.molbuilder/` ceases to exist as a
root.**

### 3.2 The one door for reading and writing

Every config read and write goes through one module, so that four properties
hold everywhere by construction instead of per caller:

1. **The path is computed in one place** — never `expanduser` at a call site.
2. **The mode is enforced, not hoped for** — `0600` for a file holding
   secrets, `0700` for the directory, applied on write *and* checked on read
   (`configuration.md` § 2.1b, already built for `molbuilder.json`; this
   generalises it).
3. **Writes are atomic** — `persist.write_json` already does temp-file +
   `chmod` + rename; this is its caller, not a second implementation.
4. **An absent file is *unset*, never an error** — the rule every current
   reader implements separately.

### 3.3 What counts as config, and what does not

The survey found ~90 `json.load`/`dump` sites. **Most are not configuration**
and must not be dragged in: sidecars (`molstruct`, `spectra`, `transport`),
run records, web request/response bodies, parse artifacts. Those are *data*,
and they belong to the modules that own their schemas.

The framework covers files that **configure the program**: the machine config,
`environment.json` and `environments/`, notify tokens and keys, session and
provider secrets. The test that keeps this honest asks the opposite question
of most: not *"is everything using the door"* but *"does anything compute a
per-user path without it"*.

---

## 4. Order

Contract first, then one mechanical change at a time, each with its own tests.

1. **`state_dir()` / `runtime_dir()`**, and the pin that no module computes a
   per-user path itself.
2. **Move `logs/`, `run/`, `reports/`** off `~/.molbuilder/`.
3. **The secret's one home** — `<config_dir>/secret_key`, one filename. This
   one touches a live credential and is the only step that can log a person
   out of their own server, so it is taken deliberately and alone.
4. **Delete the cwd step**; the § 2.1a warning becomes "not read, move it".
5. **The read/write door**, and move each config reader onto it.

Steps 2 and 3 change where files on this machine are looked for. They are not
migrations — there is no compatibility layer — so each one names what must
move, and the program says so at the moment it cannot find something.
