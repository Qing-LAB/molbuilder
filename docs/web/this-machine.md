# This machine — the tab that holds the secrets

**Role:** contract
**Domain:** web

**Companions — the contracts this surface is built against, and where the two
disagree those win:**
[`execution/run-reports.md`](?doc=execution/run-reports.md) § 1 — the split this
tab exists to keep, § 3 the channel format, § 3.1 the surface itself, § 4 the
listener;
[`ops/access-control.md`](?doc=ops/access-control.md) — the gates every route
here sits behind, and § 8 rule 3, *identity is borrowed, never stored*;
[`web/task-setup.md`](?doc=web/task-setup.md) § 9b — the card that ticks the
names this tab defines;
[`web/web-api.md`](?doc=web/web-api.md) § 4 — the routes;
[`web/ui-contract.md`](?doc=web/ui-contract.md) — the sheet layering and the
`tm-` prefix.

**What this tab is for, in one sentence.** Every other tab is about a
calculation; this one is about **the box you are signed in to** — what its
notification channels are, and whether it is receiving run reports — and it is
the only surface in molbuilder where a secret is typed.

---

## 1. Why it is a tab and not a card

It was the second half of Task setup's *Tell me how it is going* card until
2026-08-31. Three things were wrong with that, and they are the reasons this
page exists:

- **It is not per-calculation.** One destination per machine, set on a page
  that only opens with a calculation loaded. Changing it there changed it for
  every run on the box, and the card had to say so in a hint.
- **It put two files in one card**, which the template kept straight with a
  comment: the ticks go into `task.json`, which travels; the address and key go
  into `config_dir()/notify`, which must not. A rule held by a comment is a
  rule waiting to be broken by the next edit.
- **There was nowhere to put the other half.** Issuing a listener key was
  `molbuilder notify-token` and nothing else, so setting this up end to end
  always dropped to a shell — which is what a person hit on 2026-08-31 and
  reported as *"it seems the code is missing"*.

Task setup now writes `task.json` and nothing else. That rule is no longer
maintained by discipline; there is no other file it can reach.

---

## 2. The one rule: this page writes secrets and never reads them back

> **A settings page that can show you a secret is a settings page that can
> leak one.**

So every response from every route here is built to be safe to look at:

| | what the page can show |
|---|---|
| a stored **key** | that it is *there*, never what it is |
| **any** address | **masked to host + last few characters** — enough to tell two apart, nowhere near enough to use |
| a **newly issued** key | once, at the moment it is issued, and never again |

The masking rule is not caution for its own sake. The route this replaced
returned every stored URL in full until 2026-08-31 on the strength of *"an
address, not a secret"* — true of the listener URL it was written for, false of
the Slack webhook that was actually in the file, and readable by anyone signed
in to the server.

**And it covers every address, not just a webhook's.** Masking only the kind
that needs it means asking *which kind is this*, and § 3's answer is derived
from whether a key is stored — so a Slack URL saved with a key in the box would
be classed a listener and printed whole. A rule mislabelling can defeat is not
a rule. A listener address loses nothing: the tail still names the segment, § 4
shows the route in full, and an address is proved by **testing** it, which is
better evidence than reading it.

Issuing a secret and displaying a stored one are different acts. Only the
second one can never happen: a key that is never shown at the moment it is made
cannot reach the machine it is for, which is why `notify-token` prints it too
(`run-reports.md` § 4.3).

---

## 3. Channels

A **channel** is a name the person chose and what it resolves to. The file is
`config_dir()/notify`, mode `0600`, and `run-reports.md` § 3 owns its format.

Each row shows the name, the kind, the address under § 2's rule, and how the
last test went. The actions are **Add**, **Test**, **Remove**.

*Whether a key is stored* is not a separate column, because the kind already
says it: a channel with a key is a listener and one without is a webhook (§ 3
of [`run-reports.md`](?doc=execution/run-reports.md)). Two columns for one fact
would be free to disagree.

**Two kinds, and the page asks which**, because the two are authorized
differently and a person cannot be expected to infer it from a URL:

- a **webhook** (Slack, Discord) — the address is the whole credential and
  there is no key;
- a **molbuilder listener** — a plain address plus a key that signs the body
  and never travels.

**Saving merges, twice over** (`run-reports.md` § 3.1). Saving one channel
leaves the others exactly as they were, and within a channel the fields this
page manages go over whatever is already stored — so the empty key box means
*unchanged*, and a `headers` block this page has no input for survives an edit
to the address. Removing a channel is **Remove**; it is never a side effect of
a save.

**One thing a save does clear**, and it is the previous format rather than
anything anyone put there: a single-destination file's top-level `url`, `key`
and `headers`. Once the file has a `channels` map those three are read by
nothing, so leaving them leaves a live credential in a file whose whole purpose
is holding one deliberately — and it would make this page's own message
(*"save a channel below and it becomes a named one"*) false. Keyed by name, not
by *"anything that is not `channels`"*, so the merge rule above still holds for
a field added later.

**Test is the part worth the most.** A channel is only known to work when a
report arrives, and this is the only check that exercises the file, the
address, the route segment, the signature, egress and TLS together. The result
is stored beside the channel so Task setup can show it — a name with no
evidence behind it is exactly the silent failure this whole area keeps
producing.

### 3.1 The file is written here, always

**Every config file molbuilder manages is saved on the machine molbuilder
runs on** *(user, 2026-09-01)*. There is no mode, no probe and no branch: this
page writes `config_dir()/notify` on this box.

> **It gated on `execution.mode != "submit"` until 2026-09-01**, reading
> `submit` as *"the jobs run somewhere this server cannot reach"* and refusing
> to save. That is not what the setting means —
> [`running-a-job.md`](?doc=execution/running-a-job.md) § 5.4 defines it as
> `direct` (run in place) or through the scheduler, gating `.sbatch`
> **on this machine**. So the gate refused a login node with SLURM, which is
> precisely where the file belongs, and missed the real cross-machine case
> entirely: a laptop preparing a bundle for a cluster is `direct`.

**The cross-machine case is real, and it is not this page's to solve.** A
bundle is prepared here and copied to the machine that runs it
([`preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)
§ 1), and the monitor beside the job reads its channels *there*. Getting a
secret to that machine is **the user's job, by design** — and the generated
run script assumes it happened, carrying **no cleartext secret**, because
embedding one would violate the security protocol.

**So the page tells rather than generates.** A disclosure names what the
script will look for and where:

| | |
|---|---|
| `<config dir>/notify` | the channels file, `0600`, same shape as here. A channel a description ticks must exist there under the same name, or nothing is sent to it — the monitor log says which |
| `<config dir>` is | `$MOLBUILDER_CONFIG_DIR` exactly as given; else `$XDG_CONFIG_HOME/molbuilder`; else `~/.config/molbuilder` — resolved **on that machine**, so it need not match this one's |

It emits no shell and holds no key. An earlier version generated a
copy-paste block that wrote the file with the key in it; that went with the
gate that revealed it. Running molbuilder on the target and using this page
is the simplest way to put the file in place — copying the file across works
too.

---

## 4. The listener

The other half, and the reason the round trip used to need a terminal. This
section shows whether this server is receiving run reports at all: the route
segment, who holds a key, and one action — **Issue a key**.

It is the same act as `molbuilder notify-token` and it writes the same file, so
`run-reports.md` § 4.3 and § 4.4 govern it unchanged. In particular:

- **the key file is the switch** — the listener is registered because that file
  exists and carries a route, not because a setting says so;
- **a second key joins the route already in the file**, so issuing one for a
  colleague cannot move the route out from under everybody;
- **rotation is an act, never a side effect.** `serve` reads these values and
  never mints them.

**Whose file is it?** `config_dir()` belongs to the OS account the server runs
as; a molbuilder login is a person. molbuilder does not manage that mapping and
does not try to — `access-control.md` § 8 rule 3 applied to the filesystem.

---

## 5. What this tab does not do

- **It does not read a calculation** and does not care whether one is open.
  Nothing here is scoped to a project.
- **It does not decide anything about a run.** A channel is where a message
  goes; the monitor's boundary (`run-reports.md` § 5) is unchanged — it
  observes and notifies, and never mutates the calculation.
- **It does not manage people.** There is no user list, no roles, and no
  mapping between a molbuilder login and an OS account.
