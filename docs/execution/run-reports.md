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
{ "url": "https://…", "key": "…" }
```

| destination | how it is authorized | note |
|---|---|---|
| **Slack / Discord** | **the URL is the credential** — there is no separate secret, and possession of the string is the whole authorization | port 443, so cluster egress is not a question |
| **a molbuilder listener** | `key` **signs the body and never travels**; the URL is an address, not a secret | § 4 |

The two are shaped differently because only one of them is ours. A third
party that can be handed nothing but a URL has nowhere else to put a secret;
our own listener does, so it uses it.

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

molbuilder's own receiving end is **one route**. It appends one line to a
record log, answers `{"ok": true}`, and does nothing else. There is no `GET`,
it never echoes the payload, and nothing stored is readable through it —
reading is a logged-in browser's job on the ordinary tabs.

**Append-only is the security model, not a detail.** § 5's rule for the
monitor — *it observes and notifies, never decides, never mutates the
calculation* — holds one hop out. A message that arrives becomes a line in a
file. It is not parsed into application state, does not touch a project, and
cannot start, stop, retry or alter a job.

### 4.1 Four gates, and the question each one answers

| | the gate | the question it answers |
|---|---|---|
| **1** | the route exists only when **both** config keys are set | *has anyone enabled this?* |
| **2** | its path is a **per-deployment random segment**, never a fixed word | *where is it?* |
| **3** | the body carries an **HMAC-SHA256 signature**; the key never travels | *may this sender write?* |
| **4** | anything that fails answers a **plain `404`** | *— nothing. That is the point.* |

Gate 3 is the control. Gates 1, 2 and 4 exist so that a stranger cannot learn
whether gate 3 is even there.

**Why the path is generated, not just renamed.** A cleverer word gets
committed to a public repository, which makes it exactly as public as
`notify` and less honest about what it does; fixed strings are also how
wordlists get written. So `notify-token` generates the segment the same way it
already generates the secret, and it enters no source file, no document and no
example (`access-control.md` § 8 rule 7). The path is **not** a secret — it
appears in every access log, as any path does. It is merely unguessable, which
is a different and much smaller claim.

**Why a signature rather than a bearer token.** A bearer token proves *I know
the secret* by sending it, so it is on the wire on every report and a single
capture yields a credential that works forever, on any body. A signature over
the body proves the same thing while the key stays on the cluster: what
travels is valid for **one exact body** and nothing else. Both ends compute it
with the standard library, which the monitor is restricted to
(`job-contracts.md`) — it ships to a compute node where molbuilder is not
installed and runs under whatever Python the job's environment has.

**Why every failure is `404`.** A `401` would answer the question the other
three gates protect: it says *there is something here and you got it wrong*.
Answering with the router's own `404` makes a wrong signature
indistinguishable from a path that was never registered (`access-control.md`
§ 8 rule 2). It costs nothing — `404` is still `4xx`, so the rate limiter
counts it exactly as a `401` would.

**A report also carries a timestamp, signed with the body.** It bounds how
long a captured report stays replayable — a signature covers one body, so a
replay can only ever duplicate a line in a capped log. Fifteen minutes, which
is generous on purpose: a compute node's clock is not ours to trust closely,
and a run that reports late is not a run that is lying. The timestamp is
*inside* the signed material rather than beside it, or it could be rewritten
freely and the window would mean nothing.

**One key per user, and the sender never states who it is.** The server tries
each key it holds, without an early exit, and the one that verifies *is* the
identity. There is no user field to send, so a valid key cannot be used to
write into somebody else's record, and one can be revoked without disturbing
anybody else.

The rest of the narrow surface: JSON only, a hard body cap checked before
parsing, a fixed field set — anything else is dropped rather than stored,
because the log is rendered in a browser and an open-ended blob is an
open-ended rendering problem — strings length-capped, and a user id
constrained to what is safe as a **filename**, since that is what it becomes.

### 4.2 What someone probing this actually gets

Read downwards: each row assumes everything above it already went the
attacker's way.

| what they try | what they get | why |
|---|---|---|
| sweeps `/api/notify`, `/webhook`, `/api/hooks`, a wordlist | `404`, every one | there is no route at any fixed path; the only one sits at a segment that was never committed anywhere. A repo guard holds that: `test_no_fixed_notify_path_exists_anywhere_in_the_source` |
| guesses the segment | `404` | the space is far too large to walk, and every attempt is `4xx` and rate-limited |
| **reads the real URL** — an access log, a leaked destination file, over your shoulder | `404` on every request | the URL is an address. Without the key nothing can be signed, and the `404` will not even confirm they found the right place |
| captures a report in flight | one signature, useless | TLS is in front; and a signature covers **one body**. They cannot alter a field, cannot mint a new report, and a replay only adds a duplicate line to a capped, rotating log |
| **steals the cluster's `notify` file** | writes reports as that one user | bounded by append-only: no project is touched, no job starts, stops or changes. Revocation is one line out of the server's key file and disturbs nobody else |
| **steals the server's key file** | forges reports as **any** user | the one attack this does not stop — see below |
| floods the route | rate-limited, and the disk holds | every failure is `4xx` and counted; the log is capped and rotates, so a flood cannot fill the disk the app runs on |

**The honest gap.** HMAC is symmetric: both machines hold the same key, so
reading the server's key file is enough to forge. Ed25519 would close it — the
cluster would hold a private key and the server only a public one, making the
server-side file not a secret at all. It is not used because it is **not in
the standard library**, and the monitor may not carry a dependency to a
compute node. The trade is written down here so that if that constraint ever
changes the decision can be revisited rather than rediscovered.

### 4.3 The two files, and issuing them

`molbuilder notify-token <user>` generates the key **and**, on first use, the
route segment; writes the server half; and prints the cluster half:

| | file | mode |
|---|---|---|
| the server | `notify_keys` in the config directory — `{user: key}` | `0600` |
| the cluster | `notify` in the config directory — `{"url": …, "key": …}` | `0600` |

`molbuilder.json` needs **both** keys before the route exists at all:
`notify_keys_file` (a path, never the keys) and `notify_route` (the generated
segment, which is not a secret). Either one missing means no route is
registered — `access-control.md` § 8 rule 1, *the safe state is the one you get
by doing nothing*.

**The key is printed once, and that is a deliberate exception.** `auth-setup`
never prints a secret and is right not to — a session key never leaves the
server that made it. This one has to reach a second machine, and molbuilder
has no channel that could carry it there without showing it to you.

### 4.4 Restarting the server changes nothing

`serve` **reads** these values. It never generates them and never rotates
them, and that is the lifecycle rule that matters here.

The session key is the contrast: it lives on one machine, so `serve` generates
it on first run, and losing it only asks people to sign in again. A notify key
has a counterpart on a cluster molbuilder cannot reach. If `serve` minted a
new one at startup, every job already running would keep signing with the old
key, be refused, and — because a notifier is **silent on failure** by design,
so an unreachable server never costs a run anything — the reports would simply
stop, with nothing anywhere saying why.

So rotation is an act, never a side effect: re-run `notify-token --replace`
and copy the new destination file to the cluster (`access-control.md` § 8
rule 8).

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
| how a report is signed | HMAC-SHA256 over the body, both ends, standard library only (§ 4.1) |
| the receiving end | `web/blueprints/notify.py`, registered by `web/app.py` only when **both** `notify_keys_file` and `notify_route` are set |
| issuing a key, and the route segment | `molbuilder notify-token` (§ 4.3) |
| who may rotate a key | a person, never `serve` (§ 4.4) |
