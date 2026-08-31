# Getting started — from a fresh clone to a first calculation

**Role:** guide
**Domain:** ops

**Companions — every step below is a pointer to the document that owns it,
never a second copy of its rules (`workflow.md` R-W1):**
[`ops/installation.md`](?doc=ops/installation.md) — the environments;
[`ops/deployment.md`](?doc=ops/deployment.md) — the server, TLS, sign-in;
[`ops/access-control.md`](?doc=ops/access-control.md) — who may do what;
[`configuration.md`](?doc=configuration.md) — every key `molbuilder.json`
takes, and the fact/preference split;
[`execution/preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)
— machine records and `--target`;
[`execution/run-reports.md`](?doc=execution/run-reports.md) — notifications;
[`execution/worked-example.md`](?doc=execution/worked-example.md) — the whole
road once, with a real molecule.

**Who this page is for.** You have a machine (a workstation, and maybe a
cluster account), a molecule, and none of molbuilder's files yet. This page
says **which files need to exist, where, in what order** — and hands each
step to the document that owns its details.

---

## 0. The one-look map — every file you will set up

| file | machine | what it is | written by |
|---|---|---|---|
| the clone + conda envs | each machine | the program and its engines | you + `envs bootstrap` (§ 1) |
| `molbuilder.json` | each machine | what **you want** from this machine: run directly or submit, default queue, where the projects tree lives | you, by hand (§ 2) |
| `~/.config/molbuilder/environment.json` | each machine | what this machine **is** — cores, GPUs, scheduler, queues | `jobset probe --write` (§ 3) |
| `~/.config/molbuilder/environments/<name>.json` | your workstation | another machine's record, so you can describe work *for* it from here | probe there, copy here (§ 4) |
| TLS cert/key + `auth` section | the machine that serves | HTTPS and sign-in — **only** when others reach your server | you (§ 5) |
| `~/.config/molbuilder/notify` | the machine that runs jobs | where run reports go, and the signing key | `molbuilder notify-token` + the Task-setup card (§ 6) |

Everything else — templates, task files, decks, results — is made by the
workflow itself, inside the projects tree.

---

## 1. Install: the clone and the environments

One prerequisite on every machine: **a working conda or mamba**
(`workflow.md` § 6.1). Then:

```bash
git clone <the molbuilder repository> && cd molbuilder
bash scripts/install-env.sh bootstrap    # finds conda, creates the host
                                         # env, then runs `envs bootstrap`
```

The bootstrap creates one env per engine (SIESTA, PySCF, …). What each env
holds, GPU builds, and what to do when one refuses:
[`ops/installation.md`](?doc=ops/installation.md). Every
`molbuilder <verb>` below means `python -m molbuilder <verb>` from the
activated host env in the clone — molbuilder is deliberately not
pip-installed.

---

## 2. `molbuilder.json` — what you want from this machine

One small hand-written file, in the config directory —
`$MOLBUILDER_CONFIG_DIR` if you set it, else
`$XDG_CONFIG_HOME/molbuilder/`, else `~/.config/molbuilder/` ([`deployment.md` § 5](?doc=ops/deployment.md)).
The two shapes you will actually write:

**A workstation** (jobs run right here):

```jsonc
{ "execution": { "mode": "direct" } }
```

**A cluster login node** (jobs go to the scheduler):

```jsonc
{ "execution": { "mode": "submit", "domain": "public" } }
```

`domain` is your default queue — a *preference*, so it lives here and never
in a probe's record ([`configuration.md` § 5](?doc=configuration.md), M-1).
If your calculations should live somewhere other than the clone's
`projects/`, add `"paths": {"projects": "/scratch/you/projects"}` — every
surface follows it at once (`workflow.md` § 6.2).

---

## 3. The machine record — what this machine is

On **each** machine, once (and again whenever the cluster changes):

```bash
molbuilder jobset probe --write --name <name>    # e.g. --name sol
```

On a cluster login node this reads the scheduler itself — every reachable
queue, each queue's machines, walls, memory, per-job policy caps. On a
workstation it records cores and GPUs. Facts only; nothing you would
rather choose is decided here
([`preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)).

---

## 4. The remote pair — describe here, run there

The ordinary workflow is: describe a calculation on your workstation **for**
a cluster you are not on. That takes exactly one file moved by hand:

```
cluster:      jobset probe --write --name sol     → ~/.config/molbuilder/environments/sol.json
workstation:  copy that file to the same path here
```

Now `prep --target sol` (and the Task-setup tab's machine card) can answer
with the cluster's real numbers, and a calculation folder travels there
**unchanged** — the folder never names a machine (`workflow.md` § 3). The
full story of records, `--target`, and what refuses when a record is stale:
[`preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md).

---

## 5. The server — local first, sign-in only when shared

**Just you, on your own machine:** nothing to configure.

```bash
molbuilder serve start        # background; log under $XDG_STATE_HOME/molbuilder,
                              # pidfile under $XDG_RUNTIME_DIR/molbuilder
molbuilder serve status       # is it up, is it answering, where
```

(`serve foreground` keeps it in your terminal; `restart`/`stop` act on your
own instance only — [`deployment.md` § 1](?doc=ops/deployment.md).)

**Shared with others:** two more things, both in
[`deployment.md`](?doc=ops/deployment.md) —

1. **TLS** — a cert/key pair, `--cert`/`--key` or the config's `tls`
   section. The server refuses a non-loopback bind without it.
2. **Sign-in** — the `auth` section (§ 3 there): an OAuth provider and its
   `allowed_users` list. **Auth is opt-in**: without that section a
   non-loopback server is a public read-write projects tree, which is the
   one mistake that page exists to prevent. The fastest path is
   `molbuilder auth-setup` — and § 3.1 there walks the Google-console
   half (creating the OAuth client, the redirect URI, rotating the
   secret).

Who may then do what — one person, a lab, admin rights, the reload button —
is [`access-control.md`](?doc=ops/access-control.md).

---

## 6. Notifications — two halves, only one travels

A run can tell you how it is going
([`run-reports.md`](?doc=execution/run-reports.md)). The split to keep
straight:

| half | file | set up |
|---|---|---|
| **when** to speak | the calculation's `task.json` | tick it on the Task-setup card — portable, no secrets |
| **where** to send | `~/.config/molbuilder/notify` on the machine that **runs the job** | `molbuilder notify-token <you>` prints the address + key and says exactly where to put them; the Task-setup card can write the file for you when the job runs on the same machine |

On a cluster, that file lives in **your cluster home** (the card hands you
the exact command to run there). Absent means silently off — a run without
the file behaves exactly as it always did.

---

## 7. First calculation — the road

Everything is now in place. The road, each stage owned elsewhere:

```
build the molecule            the Molbuilder tab (or bring an .xyz)
set the physics               the Structure-optimization tab
Send to Task setup            writes the portable folder
shape · stages · bench        the Task-setup tab; saving writes task.json
prep bench <stage>            on the target; renders trials
launch bench <stage>          one grouped job; summarize writes the verdict
prep run · launch run         the production stage, verdict applied
the Results tab               watch it run; read what it made
```

Done once, end to end, with a real molecule and every file shown:
[`worked-example.md`](?doc=execution/worked-example.md). What each command
means and why the order is load-bearing: [`workflow.md`](?doc=workflow.md).
