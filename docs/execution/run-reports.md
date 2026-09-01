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

> **When** to speak, and **which channels**, belongs to the calculation.
> **What a channel actually is** — an address, and usually a secret — belongs
> to the machine. They are never written in the same file.

---

## 1. Why the split is the whole thing

A `task.json` **travels**. It goes to a cluster, into a citation's composed
copy, into a colleague's copy of your project. That is what a description is for.

A webhook URL does not travel — it is a fact about one person's Slack, and for
Slack and Discord it *is* the credential, with no separate token beside it.
Anything that travels must therefore not carry it.

So the two halves live apart, and every rule below is a consequence:

| | lives in | travels? |
|---|---|---|
| **the policy** — when to speak, and which channels **by name** | `task.json`'s `notify` block | **yes**, and safely: *"every six hours, to `slack`"* is true wherever the file is opened |
| **the channels** — what each name resolves to: the address, and the secret | the user's own file, `$XDG_CONFIG_HOME/molbuilder/notify` (else `~/.config/molbuilder/notify`), mode `0600` | **no**, ever |

**A name travels; what it points at does not** — and that is what lets a
description say where its reports go at all. `"slack"` is a label the person
chose on their own machine. It is not a credential, it grants nothing, and on a
machine that has no channel by that name it resolves to nothing and says so in
the monitor log. Before 2026-08-31 there was one unnamed destination per
machine, so a description could not express a choice and pointing at Slack
silently replaced the listener you were already using.

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

`config_dir()/notify` is a JSON object of **named channels**:

```json
{
  "channels": {
    "slack":       { "url": "https://hooks.slack.com/services/…" },
    "my-listener": { "url": "https://molbuilder.example.edu:8888/api/GfmVt99",
                     "key": "…" }
  }
}
```

A name is the person's own label — letters, digits, `-` and `_`, so it can be
written into a description and read back without quoting. Nothing generates one
and nothing has a special meaning.

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

**And one bad channel does not cost the others.** A file with three channels
and a typo in the second reports on two, and names the third in the log.
Refusing the file whole is what the single destination had to do, because
there was nothing else to keep; keeping it now would turn one mistake into
total silence, which is the failure this whole area keeps producing.

A third destination shape exists and is not in the table because nothing we
ship needs it: **`headers`**, for a generic endpoint that wants a credential in
a header and has no other way to be told who is calling. It is read if present
and never required.

`MB_NOTIFY_URL` overrides the file, for testing a destination once without
editing anything. It names no channel and needs none — one URL, used directly,
**and § 3.0's selection does not apply to it**: a run whose description says
`channels: []` still posts there, because the override is not one of the named
channels the description was talking about. Setting it is the act of asking for
this one report.

### 3.0 Which channels one run uses

The description names them, and the two ways of saying nothing mean different
things — which is the price of letting a checkbox list mean what it looks like:

| `notify.channels` in `task.json` | the run reports to |
|---|---|
| **absent** | **every channel on this machine.** The reading of a description that predates channels, and of one written by hand |
| `["slack", "my-listener"]` | those, and only those |
| `[]` | **nothing.** Not an error: reports off for this calculation, on a machine where they are otherwise set up |

Absent and empty are two spellings only because they are two intentions, and
the serializer writes `[]` explicitly rather than dropping it the way it drops
every other falsy field. That is the one exception in `task.py`'s round-trip
rule (S1) and it is here because the alternative — an unticked list quietly
meaning *all of them* — sends a report to a channel the person just unticked.

**A named channel this machine does not have is skipped, and said in the
monitor log.** That is the travelling case: a description written at a desk,
opened on a cluster. It cannot be an error — the run is not wrong — and it must
not be silent, because silence here is indistinguishable from working.

### 3.1 Setting the channels up — BUILT 2026-08-27, named 2026-08-31

**The format is the hard part, and it is the part a person should not have to
get right from memory.** Without a surface the flow is: run `notify-token`,
copy the JSON, reach the machine that runs the jobs, `mkdir -p -m 700`, paste,
`chmod 600`, and remember the directory. Four chances to be wrong and **every
one of them fails silently** — absent or malformed means no notifier, which is
indistinguishable from never having set it up. The wrong-path defect found in
the browser on 2026-08-27 came from exactly this, and it is still the flow on
a machine this server cannot write to — which is why the page hands over the
exact bytes rather than pretending otherwise.

**This does not touch § 1's split.** That rule is about what *travels*: the
policy and the channel **names** into `task.json`, the addresses and their
secrets into a file that never leaves the machine. A surface that writes the
**non-travelling half, on the machine it belongs to** is not carrying a secret
anywhere — it is putting one where the contract says it lives. The Task-setup
card's own rule (*it sets policy; it never sees a key*) is stronger than it
was, not weaker: that card writes `task.json` and, since 2026-08-31, has no
control that could reach anything else.

**Where it is.** Signed-in routes under `/api/notify/channels` and
`/api/notify/listener` (`web/blueprints/notify_setup.py`, and `web-api.md` § 4
lists them), behind the **This machine** tab (`this-machine.md`). A **separate
blueprint from the listener**, which is the public receiving end: that one has
no session, appends only, and its whole value is being small enough to reason
about.

**It is its own tab because it is not about a calculation.** It lived inside
the Task-setup notify card until 2026-08-31, which put a machine-wide setting
behind *having a calculation open* and needed a warning comment in the template
to keep two files straight in one card. Task setup now writes `task.json` and
nothing else, and the rule *it sets policy; it never sees a key* is true by
construction rather than by discipline.

**Nothing on that tab reads a secret back.** A stored key reports only
*present* or *absent*, and **every address is masked to its host and last few
characters** — because for Slack and Discord the URL *is* the credential
(§ 3), so returning it whole is returning the secret. The route that preceded
this returned every stored URL in full on the strength of *"an address, not a
secret"*, which was true of a listener URL and false of the Slack one actually
in the file.

**Every address, not only a webhook's**, because the alternative asks *which
kind is this* — and the kind is derived from whether a key is stored, so a
Slack URL saved with a key in the box would be classed a listener and printed
whole. A rule that mislabelling can defeat is not a rule. A listener address
loses nothing by it: the masked tail still names the segment, § 4's own
section shows the route in full, and an address is proved by **testing** it
rather than by reading it.

**A key is shown exactly once**, when it is issued, and never again — the same
deal a terminal gives you (§ 4.3). Issuing a secret and displaying a stored one
are different acts; only the second is the one that must never happen.

**One case: it writes the file, here.** *(user, 2026-09-01: every config file
molbuilder manages is saved on the machine molbuilder runs on.)* Through
`auth_setup.write_secret_file`, which sets the mode before the first byte, and
to `config_dir()` — the same function the monitor reads from, so the two
cannot name different directories.

> **It read `execution.mode` and branched on it until 2026-09-01**, treating
> `submit` as *"the jobs run on a cluster this server cannot reach"* and
> offering a copy-paste recipe instead of a save. That is a misreading of the
> setting: § 5.4 of [`running-a-job.md`](?doc=execution/running-a-job.md)
> defines `mode` as `direct` (run in place) or through the scheduler, gating
> `.sbatch` **on this machine**. A login node with SLURM is `submit` and is
> exactly where the file belongs, so the branch refused the machine that
> needed it most — while the genuine remote case, a laptop preparing for a
> cluster, is `direct` and was never detected.

**The remote case is real and is the user's to carry.** A bundle is prepared
on one machine and copied to the one that runs it
([`preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)
§ 1); the monitor reads its channels *there*. Putting the secret on that
machine is the user's job **by design**, and the wrapper carries **no
cleartext secret** — embedding one would violate the security protocol. What
a surface may do about it is **say what the script will look for and where**,
which is `this-machine.md` § 3.1. It may not generate a file for a machine it
cannot see.

Either way the **names** are what Task setup offers, so a description can be
written against channels that only exist on the far machine.

**A save updates one channel; it does not replace the file.** Two merges,
and both are load-bearing. *Across* channels: saving `slack` must not disturb
`my-listener`, so the write is keyed by name and touches nothing else.
*Within* one: the fields the page manages go over whatever that channel already
holds. Writing a fresh object instead destroyed the rest of it two ways — the
key (the page clears that field after each save, so the ordinary next action,
fixing a typo in the address, arrived with none) and a `headers` block the page
has no input for but the monitor reads. **Both failed silently**, because an
unsigned report gets the listener's `404` and the notifier swallows it.
Removing a channel deliberately is *Remove*.

**The single exception is the previous format.** A save clears the top-level
`url`, `key` and `headers` a one-destination file carried: once the file has a
`channels` map nothing reads them, and a credential nothing reads is a
credential nobody will think to rotate. By name, not by *"anything that is not
`channels`"* — the merge rule above is what protects a field added later, and
it still does.

**Whose file is it?** `config_dir()` belongs to the OS account the server runs
as, while a molbuilder login is a person. **molbuilder does not manage that
mapping and does not try to** (user, 2026-08-27: *each user is expected to
manage and align his login and OS account; we don't manage that*) — which is
`access-control.md` § 8 rule 3, *identity is borrowed, never stored*, applied
to the filesystem.

**And it can verify, which is the part worth the most.** A destination is only
known to work when a report arrives. Sending one test report and saying whether
it landed turns *"I think it is set up"* into an answer — and it is the only
check that exercises the whole path: the file, the URL, the route segment, the
signature, egress, and TLS. **Pair it with `MB_NOTIFY_KEY`** when the destination is a
molbuilder listener — an unsigned report is refused there, and refused with a
`404` that the notifier swallows, so the one destination you most want to test
would fail in silence.

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

### 4.1a Where the results go, and what one line holds

**`<state dir>/reports/<user>.jsonl` — `reports/`, not `logs/`.** The
distinction is the point: `<state dir>/logs/` is molbuilder's own
operational output, the kind you read when something is wrong and delete when
it is fixed. **These are measurements from calculations** — energies,
iteration counts, when a relaxation step landed. You keep them, grep them a
year later, and plot them. Filing them under `logs/` invited exactly one
mistake: treating results as disposable.

One file per user, **JSON Lines**, so `jq` and `pandas` read it directly with
no parser of ours in the middle. Mode `0600` in a `0700` directory — the key
file was always `0600` and the data it protects was not, which was the wrong
way round on a shared server.

**Every line stands on its own.** A line used to read `{"event":
"scf_converged", "energy": "-1740.2"}` and nothing said *which* calculation,
on *what* machine, or *when* it was sent — two jobs running produced
indistinguishable records. Somebody parses this file later with no session to
ask, so:

```json
{"v": 1, "user": "jdoe@asu.edu", "run": "BDT_Au_relax", "job": "62238108",
 "host": "sg013", "event": "scf_converged", "sent_at": 1756000000.5,
 "received_at": 1756000000.8, "state": "running", "elapsed_s": 1234.5,
 "n_iters": 7, "energy": "-1740.21", "geom_step": 3, "per_iter_s": 12.8,
 "text": "state=running elapsed=1234s scf_iters=7 geom_move=3 energy=-1740.21"}
```

`array` joins them for an array task (`SLURM_ARRAY_TASK_ID`), and is absent
otherwise — a field the monitor could not determine is left out rather than
sent empty, so a reader can tell *unknown* from *wrong*.

| field | where it comes from |
|---|---|
| `run` | the **label** (`run-identity.md` § 2 — *the stem of every file*), taken off the `.out` the monitor watches, `-runN` stripped |
| `job`, `host` | `SLURM_JOB_ID` and the node's own name |
| `sent_at` / `received_at` | the sender's clock and ours. **Both**, because when they disagree that is itself worth seeing |
| `text` | the same one-line summary a Slack or Discord channel renders, so one body is readable in a chat window and parseable here |
| `user` | **stamped from the key that verified**, never read from the payload — there is no user field to send |
| `v` | the record's shape, so a reader a year from now does not infer it from which keys happen to be present |

Anything the monitor could not determine is simply **absent**, which a reader
can tell from a wrong value.

**A volume cap, per key.** Sixty reports a minute — generous by orders of
magnitude against a monitor that speaks on convergence, every N hours, and
once at the end. It is not about disk: the record rotates at 1 MB × 5, so an
unbounded flood would **silently push a run's real reports out of the
window**. The cap is what keeps the results the results. It is a rolling
window, not a quota: a burst costs a minute, never the rest of the run.

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
| **floods the route with a VALID key** | 60 a minute, then `404` | `rate_limit.py` bounds failures only, and its total-request threshold ships disabled — so this is the listener's own per-key cap (§ 4.1a). Without it a flood would rotate a run's real reports out of the record |
| floods the route | rate-limited, and the disk holds | every failure is `4xx` and counted; the log is capped and rotates, so a flood cannot fill the disk the app runs on |

**The honest gap.** HMAC is symmetric: both machines hold the same key, so
reading the server's key file is enough to forge. Ed25519 would close it — the
cluster would hold a private key and the server only a public one, making the
server-side file not a secret at all.

**Decided 2026-08-27: not doing it, and the reason is a judgement rather than
an obstacle.** An earlier draft of this paragraph said it was impossible
without a dependency the monitor may not carry. That was checked and is not
true: the monitor only *signs*, signing is ~60 lines of stdlib arithmetic
(RFC 8032 publishes reference code), and a probe passed the RFC vectors,
interoperated with the server's `cryptography`, and cost 7.6 ms per signature
against a monitor that signs a handful of times per run.

It is not done because **what it protects against is already the least of the
risks here**. Ed25519 defends the server's own key file — and anyone who can
read `notify_keys` on that machine can read a great deal else. Against
everything on the wire, HMAC is already sufficient, and it sits behind a
generated route, a per-key volume cap, `0600` files, and a record that is
append-only and size-capped by construction. *(User, 2026-08-27: "we have
rate-limit, file access limit, encrypted/mutated api entry — this is enough
for a simple task that has low risk, just recording logs with a cap in size
and number.")*

Two costs that would come with it, stated so a future reader weighs the same
trade: a pure-Python signer is **not constant-time**, and it is cryptographic
code this project would own forever.

### 4.3 The two files, and issuing them

`molbuilder notify-token <user>` generates the key, writes the server half, and
prints the cluster half. `--channel` names the channel it prints (default
`molbuilder`); that name is what a description ticks, so re-issuing under the
same name replaces the credential without touching any description:

| | file | shape | mode |
|---|---|---|---|
| the server | `notify_keys` in the config directory | `{"route": …, "keys": {user: key}}` | `0600` |
| the cluster | `notify` in the config directory | `{"channels": {"<name>": {"url": …, "key": …}}}` | `0600` |

**`molbuilder.json` needs nothing.** The key file **is** the switch: the
listener is registered when that file exists and carries a route, and not
otherwise — `access-control.md` § 8 rule 1, *the safe state is the one you get
by doing nothing*, unchanged. No file, no route in it, no listener.

> **It took two settings until 2026-08-31, and they were the defect**
> *(user: "why would `notify_keys_file` not by default be the one that the task
> setup writes to? … the fact that you have to repeat the `notify_route` twice,
> with one in the file and one in the `molbuilder.json` which has its own
> pointer to that same file")*.
>
> `notify_keys_file` was a path to the file molbuilder had itself chosen and
> written; `notify_route` was a copy of the segment molbuilder had itself
> issued. Neither was information the operator held. What they bought was a
> working key file sitting beside a listener that had never been registered —
> answering 404 to everything, which by design is indistinguishable from an
> unconfigured server, so nothing said what was wrong.
>
> The duplication had a second cost, which this section used to describe as a
> procedure: because the command could not read `molbuilder.json`, issuing a
> second person's key generated a **new** segment, and pasting it moved the
> route out from under everyone already set up — silently, since a notifier
> swallows failures. With the route in the file the command reads what it
> issued, so **the second key joins the first automatically** and there is
> nothing to pass.
>
> ```console
> $ molbuilder notify-token alice --host https://molbuilder.example.edu:8888
>   first key here, so the route segment was generated: GfmVt99yUpOzGyp2
>
> $ molbuilder notify-token bob   --host https://molbuilder.example.edu:8888
>   joined the route already in that file (GfmVt99yUpOzGyp2), so everybody
>   already set up keeps working.
> ```
>
> **`--route` survives for one job**: adopting a segment that is already live
> somewhere else — a destination file issued before the file carried its route.
> Passing one that differs from the file's says so, in as many words, because it
> stops every key issued under the old segment.

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

So rotation is an act, never a side effect: re-run `notify-token` with
`--replace`, **and with `--route` naming the segment already in service**, then
copy the new destination file to the cluster (`access-control.md` § 8 rule 8):

```console
$ molbuilder notify-token alice --replace --route Ie8PB3cbJBoGoC \
      --host https://molbuilder.example.edu:8888
```

Without `--route` this rotates one person's key **and moves the route for
everybody else** — two changes where one was intended, and the second one
silent.

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
| what a channel name resolves to | `monitor.load_channels` → `config_dir()/notify` |
| which channels one run uses | `task.Notify.channels` → `--notify-channels` (§ 3.0) |
| where results land | `$XDG_STATE_HOME/molbuilder/reports/<user>.jsonl` (default `~/.local/state/molbuilder/`), JSON Lines, 0600 (§ 4.1a).  `$XDG_STATE_HOME` moves it -- `paths.reports` was retired 2026-08-31 and is now refused (`configuration.md` § 2.1d) |
| what identifies a report | `monitor.run_identity` — label, job id, host (§ 4.1a) |
| overriding that once | `MB_NOTIFY_URL` + `MB_NOTIFY_KEY` (§ 3) |
| the card that sets **when**, and ticks the names | the Task-setup tab, `task-setup.md` § 7 |
| the tab that sets **what a name is** | *This machine*, `this-machine.md` |
| how a report is signed | HMAC-SHA256 over the body, both ends, standard library only (§ 4.1) |
| the receiving end | `web/blueprints/notify.py`, registered by `web/app.py` only when the key file exists and carries a route (§ 4.3) |
| issuing a key, and the route segment | `molbuilder notify-token` (§ 4.3) |
| who may rotate a key | a person, never `serve` (§ 4.4) |
