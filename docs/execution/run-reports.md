# Run reports — how a job tells you it is going

**Role:** contract
**Domain:** execution
**Companions:**
[`running-a-job.md`](?doc=execution/running-a-job.md) § 4.1 — the wrapper's
instruments, of which this is one;
[`job-contracts.md`](?doc=execution/job-contracts.md) — the monitor's place in
a run directory, and the boundary it may not cross;
[`web/task-setup.md`](?doc=web/task-setup.md) § 7 — the card that writes the
policy;
[`ops/access-control.md`](?doc=ops/access-control.md) — the gates the listener
sits behind.

A run takes hours or days on a machine you are not sitting at. Something
beside it already watches — `mb_monitor.py`, backgrounded by the wrapper — and
this is the rule for how it tells you what it sees.

**The sentence the whole design follows:**

> **When** to speak belongs to the calculation. **Where** to speak belongs to
> the machine. They are never written in the same file.

---

## 1. Why the split is the whole thing

A `task.json` **travels**. It goes to a cluster, into a handoff bundle, into a
colleague's copy of your project. That is what a description is for.

A webhook URL does not travel — it is a fact about one person's Slack, and for
Slack and Discord it *is* the credential, with no separate token beside it.
Anything that travels must therefore not carry it.

So the two halves live apart, and every rule below is a consequence:

| | lives in | travels? |
|---|---|---|
| **the policy** — when to speak | `task.json`'s `notify` block | **yes**, and safely: *"tell me every six hours"* is true wherever the file is opened |
| **the destination** — where to speak, and the secret | the user's own file, `$XDG_CONFIG_HOME/molbuilder/notify` (else `~/.config/molbuilder/notify`), mode `0600` | **no**, ever |

The portability test is [`task-setup.md`](?doc=web/task-setup.md) § 6.1's, and
it is the same one that lets a queue name into a description while keeping
*"use 16 ranks"* out.

> **On the directory.** Secrets go where every other molbuilder secret goes —
> `config_dir()`, which honours `$XDG_CONFIG_HOME`. That is not a style
> preference here: on an HPC login node `$HOME` is NFS-mounted and often
> snapshotted, and `XDG_CONFIG_HOME=/scratch/$USER` is how a person keeps a
> token off it. The monitor reads this file **on a compute node**. A path
> hardcoded to `$HOME` would have no such escape.
>
> *(`deployment.md` § 5.1 said `~/.molbuilder/` until 2026-08-26 while the code
> had always used `config_dir()` — the wizard and the hand-written instructions
> naming different directories. Corrected to the code's answer.)*

---

## 2. When it speaks

Three occasions. They are **not** a choice between: the two settable ones
combine with OR, and the third is not settable at all.

| occasion | set by | fires |
|---|---|---|
| an SCF cycle converged | `notify.on_scf_converged` | once per geometry step in a relaxation; a single point has none, and its finish message is the whole report |
| every N hours | `notify.every_hours` | N is a **number of hours**, not a duration string |
| **it ended** | nothing — always on | when the watched PID goes, however it went |
| **it stalled** | nothing — always on | no progress for `stall_heartbeat_s`, throttled to one per window |

**Nothing is reported before an SCF converges.** A half-finished cycle is not
news; the iteration count and running energy are already in the monitor log and
on the Watch tab. The exception is trouble — an abort, a scheduler timeout, a
stall — which is worth saying whenever it happens, whatever the policy says.

**Ending is not settable** because switching it off is the one thing nobody
wants: a run that finishes at 3am saying so is the reason the hook exists.

### 2.1 Looking often and speaking rarely are different numbers

The monitor wakes every `MB_MONITOR_INTERVAL` seconds (default **10**) and
samples utilisation into `util.csv`. That record stays dense — its whole point
is showing whether the CPU/GPU/memory allocation is being used.

**Notifying is separate and rare.** Until 2026-08-26 it was not: the notifier
fired on every *changed* sample, and on a running job the iteration count
advances constantly, so a webhook configured against it received a message
every few seconds for the length of the run. The per-sample event is a line in
the monitor log; who gets told, and when, is the table above.

### 2.2 A converged SCF is read from the geometry step

The monitor sees a cycle converge because `geom_step` advanced — SIESTA prints
`Begin CG move = N` when it begins the next one.

It is read that way, and not by scanning for a convergence phrase, because
**this module keeps no marker table**. The one it used to keep decided that a
run was *over*, and was wrong about it: `siesta: Final energy` prints before
the end, so the monitor could stop sampling while the job still held its GPUs.
`job-contracts.md` states the rule — the monitor *"follows the launcher's PID,
so it knows authoritatively when the run ended, rather than guessing from
output markers."*

Reading the artifacts to **report progress** is this module's job. Reading a
marker to decide **the run is over** is not. Same file, different question,
different authority.

---

## 3. Where it speaks

One mechanism, two kinds of destination. Both are a `POST` with a short
timeout, individually guarded, **silent on failure** — an unreachable server
must never cost the run anything.

`config_dir()/notify` is a JSON object:

```json
{ "url": "https://…", "headers": { "Authorization": "Bearer …" } }
```

| destination | where the credential is | note |
|---|---|---|
| **Slack / Discord** | **in the URL** — there is no separate token, and possession of the string is the whole authorization | port 443, so cluster egress is not a question |
| **a molbuilder listener** | in a **header**; the URL is a plain address | § 4 |

**Absent is off.** No file means no notifier is registered, and the run
proceeds exactly as it does for everyone who has never set this up. A
malformed file is not an error either — this is a monitor, and refusing to
watch a job because a notification could not be configured would be the tail
wagging the dog. It says so **in the monitor log**, because the wrapper
backgrounds the process as `>/dev/null 2>&1 &` and anything printed goes
nowhere.

`MB_NOTIFY_URL` overrides the file, for testing a destination once without
editing anything.

---

## 4. The listener

molbuilder's own receiving end is **one route**: `POST /api/notify`. It
appends one line to a record log, answers `{"ok": true}`, and does nothing
else. There is no `GET`, it never echoes the payload, and nothing stored is
readable through it — reading is a logged-in browser's job on the ordinary
tabs.

**It exists only when it is configured.** `molbuilder.json`'s
`notify_tokens_file` names the operator's token file; without that key the
blueprint is never registered and the path 404s like any other nonexistent
one. `access-control.md` § 8 rule 2 — *"a capability that cannot be
exercised safely should not appear. 404 is not rudeness; it is the honest
statement that there is nothing there."*

**Public means the SSO check does not apply, not that it is
unauthenticated.** `api_notify` is in `auth.py`'s `_PUBLIC_ENDPOINTS` because
a monitor on a compute node cannot do a browser sign-in. Its first act is
comparing a **bearer token** with `hmac.compare_digest`, against every entry
and without an early exit, so the time taken does not depend on where in the
file a token sits.

**One token per user, and the sender never states who it is** — the secret
is the claim. That is what stops a valid token being used to write into
somebody else's record, and it lets one be revoked without disturbing
anybody else.

**A bad token counts against the rate limiter.** The SSO gate marks its own
401 as *not evidence* (`g.molbuilder_auth_challenge`), because an expired
session is an ordinary visitor and counting it once locked a user out of
their own site for an hour. **That reasoning does not carry here**: nobody
reaches this route by accident, so a wrong token is somebody trying one.
This route never sets that flag.

The rest of the narrow surface: JSON only, a hard body cap checked before
parsing, a fixed field set — anything else is dropped rather than stored,
because the log is rendered in a browser and an open-ended blob is an
open-ended rendering problem — and strings length-capped. A user id is
constrained to what is safe as a **filename**, since that is what it
becomes.

### 4.1 The two files, and issuing them

`molbuilder notify-token <user>` generates the secret, writes the server
half, and prints the cluster half:

| | file | mode |
|---|---|---|
| the server | `notify_tokens` in the config directory — `{user: token}` | `0600` |
| the cluster | `notify` in the config directory — `{"url": …, "headers": {…}}` | `0600` |

**The token is printed once, and that is a deliberate exception.**
`auth-setup` never prints a secret and is right not to — a session key never
leaves the server that made it. This one is a *shared* secret by design: it
has to reach a second machine, and molbuilder has no channel that could
carry it there without showing it to you.

Neither token ever enters `molbuilder.json`, which carries paths
(`ops/deployment.md` § 5.1).

---

## 5. The boundary

> **The monitor observes and notifies. It never decides, and never mutates the
> calculation.**

That is `job-contracts.md`'s rule for the monitor, and everything here inherits
it. A notification is a message about a run, never an input to one. Nothing in
this path can start, stop, retry or alter a job — which is also what bounds the
damage if a destination is ever compromised: the worst it buys is noise.

---

## 6. Where each piece lives

| the question | the answer |
|---|---|
| when should this calculation speak | `task.Notify` — `task.json`'s `notify` block |
| how it reaches the wrapper | `jobset.Resources`, the road `continue_retries` already rides |
| how it reaches the monitor | `--notify-on-scf` / `--notify-every-hours` on the `mb_monitor.py` line |
| when to fire | `monitor.run_monitor` |
| where to send | `monitor.load_destination` → `config_dir()/notify` |
| the card that sets it | the Task-setup tab, `task-setup.md` § 7 |
| the receiving end | `web/blueprints/notify.py`, registered by `web/app.py` only when configured |
| issuing a token | `molbuilder notify-token` |
