"""Setting up **this machine** — the notification channels, and the listener.

`run-reports.md` § 3.1 and `this-machine.md`.  The format is the hard part,
and it is the part a person should not have to get right from memory: the
alternative is running ``notify-token``, copying the JSON, reaching the
machine that runs the jobs, ``mkdir -p -m 700``, pasting, ``chmod 600``, and
remembering the directory.  **Four chances to be wrong, and every one fails
silently** — absent or malformed means no notifier, which is indistinguishable
from never having set it up.  A wrong-path defect found on 2026-08-27 came from
exactly that.

**This is not `notify.py`, and the separation is deliberate.**  That module
is the *public receiving end*: one route, no session, append-only, and its
whole value is being small enough to reason about.  This one is the opposite
in every respect — it is **login-gated**, it writes files, and it reads local
configuration.  Putting them together would blur a boundary that took real
work to make crisp.

**Nor does it touch § 1's split.**  That rule is about what *travels*: policy
and channel NAMES into ``task.json``, addresses and secrets into files that
never leave the machine.  Writing the **non-travelling half, on the machine it
belongs to**, is putting a secret where the contract says it lives rather than
carrying it anywhere.

**Whose files are they?**  ``config_dir()`` belongs to the OS account the
server runs as; a molbuilder login is a person.  molbuilder does not manage
that mapping and does not try to (user, 2026-08-27) — `access-control.md` § 8
rule 3, *identity is borrowed, never stored*, applied to the filesystem.

**IT ALWAYS WRITES, AND `execution.mode` HAS NOTHING TO SAY ABOUT IT.**  This
gated on ``mode != "submit"`` from 2026-08-27 until 2026-09-01, reading
``submit`` as *"the jobs run somewhere this server cannot reach"* and refusing
to save.  That is not what the setting means: `running-a-job.md` § 5.4 defines
it as ``direct`` (run in place) or through the scheduler, and it gates
``.sbatch`` submission **on this machine** — so a login node with SLURM is
``submit`` and is exactly where the file belongs.  The gate refused the
machines that needed it most, and did not detect the real cross-machine case
at all: a laptop preparing a bundle for a cluster is ``direct``.

The rule instead *(user, 2026-09-01)*: **every config file molbuilder manages
is saved on the machine molbuilder runs on.**  Getting a secret to the machine
that will construct or run the task is the user's own job, by design — the
generated run script assumes the transfer happened and carries **no cleartext
secret**, because embedding one would violate the security protocol.  So this
page writes here, and offers no recipe for writing anywhere else.

**Nothing here reads a secret back**, and that rule is now wider than it was.
A stored key is reported as present and never as itself, and a **webhook
address is masked too**: for Slack and Discord the URL *is* the credential
(`run-reports.md` § 3), so returning it whole returns the secret.  The route
this replaced returned every stored URL in full on the strength of *"an
address, not a secret"* — true of the listener URL it was written for, and
false of the Slack webhook that was actually in the file.

**One risk, named rather than hidden.**  :func:`test_channel` makes the
*server* POST to a URL the caller supplied, which is a request-forgery shape.
It is bounded rather than eliminated: the caller is already logged in and could
write the file and run a job to the same effect, the answer carries only a
status code and never the response body, and the timeout is short.  On a shared
server this is a real (small) capability handed to any signed-in user, and that
is the trade.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from flask import Blueprint, current_app, jsonify, request

bp = Blueprint("notify_setup", __name__)

#: How long the test POST may take.  Short: a person is watching, and an
#: unreachable destination is an answer worth getting quickly.
TEST_TIMEOUT_S = 8.0

#: How much of a masked address survives.  Enough to tell two webhooks apart
#: at a glance, far too little to use.  Slack's own UI shows a comparable
#: tail for the same reason.
MASK_TAIL = 4


def _dest_path() -> Path:
    """The channel file, from the **monitor's own** function.

    Imported rather than restated, so this page and the process that reads
    the file on a compute node cannot name different directories.  They did
    once: the Task-setup card said ``~/.molbuilder/notify`` while the
    monitor read ``config_dir()/notify``, and following the card put the
    file where nothing looks.
    """
    from ...monitor import default_notify_path
    return default_notify_path()


def _document() -> Dict[str, Any]:
    """The file as it stands, or ``{}``.

    Unreadable or malformed reads as empty, which is the safe direction: a
    save then writes a fresh, valid file rather than refusing, and the
    person is not stuck with a broken one they cannot fix from here.
    """
    try:
        obj = json.loads(_dest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(obj) if isinstance(obj, dict) else {}


def _stored() -> Dict[str, Any]:
    """The channels sub-object, raw — **keys and all**.

    The only function that hands out a stored secret, and it hands it to the
    merge in :func:`save_channel`, never to a response.  Every route answers
    through :func:`_row`.
    """
    chans = _document().get("channels")
    return dict(chans) if isinstance(chans, dict) else {}


def _write(doc: Dict[str, Any]) -> None:
    """Through `auth_setup.write_secret_file`, which creates the parent at
    0700 and sets the mode on the descriptor **before the first byte** — so
    the secret is never briefly on disk at a looser mode."""
    from ...auth_setup import write_secret_file
    write_secret_file(_dest_path(), json.dumps(doc, indent=2) + "\n")


def _mask(url: str) -> str:
    """An address, safe to show.  Scheme, host, and a short tail.

    **Not decoration.**  For Slack and Discord the URL is the whole
    credential, so this is the same rule as *never show a key* applied to
    the thing that IS the key.  The host survives because it is what tells
    you which service you are looking at, and the tail because it is what
    tells two of them apart.

    **Every address, not just a webhook's.**  Masking only the kind that
    needs it means asking *which kind is this*, and the answer is derived
    from whether a key is stored -- so a Slack url saved with a key in the
    box would be classed a listener and printed in full.  A rule that can be
    defeated by mislabelling is not a rule.  A listener address is not a
    secret and loses nothing here either: the tail still names the segment,
    the listener section below shows the route in full, and the address is
    proved by testing it rather than by reading it.
    """
    try:
        u = urllib.parse.urlsplit(url)
    except ValueError:                                  # pragma: no cover
        return "…"
    host = u.netloc or "?"
    tail = (u.path or "").rstrip("/")[-MASK_TAIL:]
    return f"{u.scheme}://{host}/…{tail}" if tail else f"{u.scheme}://{host}/…"


def _row(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """One channel, as everything outside this module may see it.

    **One door out**, so no route can forget: the key never appears, and the
    address appears in full only for the kind whose address is not a secret.

    The kind is DERIVED rather than stored, from the one thing that actually
    differs: a channel with a key is a molbuilder listener (the key signs the
    body, so the URL is only an address); a channel without one keeps its
    credential in the URL, because a third party handed nothing but a URL has
    nowhere else to put it (`run-reports.md` § 3).  Storing a `kind` field
    would be a second answer to a question the file already answers, free to
    disagree with it.
    """
    has_key = bool(spec.get("key"))
    ok = spec.get("tested_ok")
    return {
        "name":      name,
        "kind":      "listener" if has_key else "webhook",
        "where":     _mask(str(spec.get("url") or "")),
        "has_key":   has_key,
        "headers":   sorted(spec.get("headers") or {}),
        "tested_ok": ok if isinstance(ok, bool) else None,
        "tested_at": spec.get("tested_at") if isinstance(
            spec.get("tested_at"), (int, float)) else None,
    }


def _rows() -> List[Dict[str, Any]]:
    return [_row(n, s if isinstance(s, dict) else {})
            for n, s in sorted(_stored().items())]


def _state() -> Dict[str, Any]:
    """**The whole state of this machine's channels, from one function.**

    Every route that touches them answers with this, and it is not tidiness:
    the page repaints from whatever the response carries, so a mutation that
    replied with a narrower object left the painter reading fields that were
    not there.  It said *"2 channels in undefined"* after a test -- found in
    the browser on 2026-08-31, because `path` was in the GET's answer and in
    no other.

    A response is the state, or it is a trap for the next painter.
    """
    path = _dest_path()
    try:
        file_mode = oct(path.stat().st_mode & 0o777)
    except OSError:
        file_mode = ""
    return {"path": str(path), "channels": _rows(), "problem": _file_note(),
            "mode": file_mode}


def _file_note() -> str:
    """What is wrong with the file, in words, or ``""``.

    A broken file and no file both mean nothing is sent and look identical
    from outside.  Saying which is most of what this page is worth.
    """
    path = _dest_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        return f"not valid JSON ({exc})"
    if not isinstance(obj, dict):
        return "needs a JSON object"
    if not isinstance(obj.get("channels"), dict):
        if isinstance(obj.get("url"), str):
            return ("this is the old single-destination file — save a "
                    "channel below and it becomes a named one")
        return "needs a 'channels' object"
    return ""


# ── channels ────────────────────────────────────────────────────────────────

@bp.route("/api/notify/channels", methods=["GET"])
def channels():
    """What this machine can report to.  **Names, never secrets.**

    The Task-setup tab calls this and nothing else: it needs the names to
    offer as ticks and the evidence to show beside them, and it has no
    business with anything else here.
    """
    return jsonify({"ok": True, **_state()})


@bp.route("/api/notify/channels/<name>", methods=["PUT"])
def save_channel(name: str):
    """Add or update one channel, `0600`, in the one right place.

    **A save merges, twice over, and both merges are load-bearing.**

    *Across* channels: this writes one name and leaves every other exactly
    as it was.  The file used to hold a single destination, so a save was a
    whole-file write; keeping that here would make configuring Slack delete
    the listener — which is the shape of the bug the single destination
    already had, promoted to a data loss.

    *Within* a channel: the fields this page manages go over whatever that
    channel already holds.  Writing a fresh object destroyed the rest of it
    two ways, both found by round-tripping on 2026-08-27: the key (the page
    clears that field after each save — a secret left in the DOM ends up in
    a screenshot — so the ordinary next action, fixing a typo in the
    address, arrived with none), and a ``headers`` block the monitor reads
    and this page has no input for.  **Both failed silently**, because an
    unsigned report gets the listener's 404 and the notifier swallows it.

    Removing something deliberately is `DELETE`.
    """
    from ...monitor import is_channel_name
    if not is_channel_name(name):
        return jsonify({"ok": False,
                        "error": "a channel name is letters, digits, '-' "
                                 "and '_' (up to 64) — it gets written into "
                                 "a description and read back out"}), 400
    body = request.get_json(silent=True) or {}
    url = str(body.get("url") or "").strip()
    key = str(body.get("key") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "a url is required"}), 400
    if not (url.startswith("https://") or url.startswith("http://")):
        return jsonify({"ok": False,
                        "error": "the url must start with https:// "
                                 "(or http:// for a host you trust)"}), 400
    doc = _document()
    chans = doc.get("channels")
    if not isinstance(chans, dict):
        chans = {}
    spec = dict(chans.get(name) or {}) if isinstance(
        chans.get(name), dict) else {}
    spec["url"] = url
    if key:
        spec["key"] = key
    # A CHANGED ADDRESS IS AN UNTESTED ONE.  Carrying the old verdict over
    # would leave a green tick beside a channel nobody has ever reached, and
    # this page's whole claim is that the tick means something.
    spec.pop("tested_ok", None)
    spec.pop("tested_at", None)
    chans[name] = spec
    doc["channels"] = chans
    # THE RETIRED SHAPE IS CLEARED, and only it.  A single-destination file
    # carried `url` / `key` / `headers` at the TOP level; once this file has
    # a `channels` map those three are read by nothing, so leaving them is
    # leaving a live credential in a file whose whole point is holding one
    # deliberately.  It is also what `_file_note` promises -- it tells the
    # person that saving a channel turns the old file into a named one.
    #
    # Narrow on purpose: the merge rule above says a field added LATER must
    # survive a page that predates it, and it still does.  These three are
    # not unknown fields, they are the previous format's, by name.
    for retired in ("url", "key", "headers"):
        doc.pop(retired, None)
    try:
        _write(doc)
    except OSError as exc:
        return jsonify({"ok": False,
                        "error": f"could not write {_dest_path()}: {exc}"}), 500
    return jsonify({"ok": True, "channel": _row(name, spec), **_state()})


@bp.route("/api/notify/channels/<name>", methods=["DELETE"])
def remove_channel(name: str):
    """Remove one channel.  **Absent is off**, and off is a state a person
    is entitled to reach without a shell.

    Removing the last one leaves ``{"channels": {}}`` rather than deleting
    the file: an empty map and no file mean the same thing to the monitor,
    and leaving the file says *somebody configured this machine and then
    emptied it*, which is worth being able to tell.
    """
    doc = _document()
    chans = doc.get("channels")
    if isinstance(chans, dict) and name in chans:
        del chans[name]
        doc["channels"] = chans
        try:
            _write(doc)
        except OSError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **_state()})


@bp.route("/api/notify/channels/<name>/test", methods=["POST"])
def test_channel(name: str):
    """Send one report to one channel and say what happened.

    **This is the only check that exercises the whole path** — the file,
    the URL, the route segment, the signature, egress and TLS — and until
    it existed the only way to know a setup worked was to run a job and
    notice nothing arrived.

    It signs with the monitor's own :func:`sign_report`, and reads the
    channel through the monitor's own :func:`load_channels`, so a signature
    this accepts is one the listener accepts and a channel this can reach is
    one a job can reach.  A second implementation here could pass while the
    real one failed.

    The verdict is **stored beside the channel**, because that is what lets
    the Task-setup tab show evidence rather than a bare name — and a name
    with nothing behind it is exactly the silent failure this area keeps
    producing.
    """
    from ...monitor import load_channels, sign_report
    dest = load_channels().get(name)
    if not dest:
        return jsonify({"ok": False,
                        "error": f"no channel called {name!r} is set up "
                                 f"here"}), 404
    body = json.dumps({
        "event": "test",
        "text": f"a test report from molbuilder, for the channel {name!r}",
        "state": "test",
        "run": "notify-setup-test",
    }).encode()
    headers = {"Content-Type": "application/json", **(dest.get("headers") or {})}
    if dest.get("key"):
        ts = "%d" % int(time.time())
        headers["X-Molbuilder-Timestamp"] = ts
        headers["X-Molbuilder-Signature"] = sign_report(dest["key"], ts, body)
    req = urllib.request.Request(dest["url"], data=body, method="POST",
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TEST_TIMEOUT_S) as r:
            code = r.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:                            # noqa: BLE001
        # NOT an HTTP status: the request never completed.  Said plainly,
        # because "could not connect" and "was refused" send a person to
        # two completely different places.
        # EXPLICIT 200, and it is the right code: the API call worked, and
        # what it did on your behalf did not.  A 4xx would blame the
        # caller for a cluster's firewall; a 5xx would claim this server
        # broke.  `web-api.md` § 1 calls this the advisory bucket and asks
        # for the status to be spelled out so the intent is visible.
        _record(name, False)
        return jsonify({"ok": False, "reached": False,
                        "error": f"could not reach it: {exc}",
                        **_state()}), 200
    good = bool(code and 200 <= code < 300)
    _record(name, good)
    # A 404 IS THE LISTENER'S REFUSAL.  Every gate answers with one so a
    # stranger cannot tell them apart (`run-reports.md` § 4.1) -- which
    # means it cannot tell YOU apart either, and the honest thing is to
    # name all of the possibilities rather than guess between them.
    hint = ""
    if code == 404:
        hint = ("404 — the listener refuses everything the same way, so "
                "this is a wrong route segment, a wrong or missing key, or "
                "a server with no listener configured. Re-check the address "
                "and key against what issued them.")
    elif good:
        hint = "it arrived."
    return jsonify({"ok": good, "reached": True, "status": code,
                    "hint": hint, **_state()}), 200


def _record(name: str, ok: bool) -> None:
    """Remember how the last test went.  Best-effort: a verdict that could
    not be written is not worth failing a test that already happened."""
    doc = _document()
    chans = doc.get("channels")
    if not isinstance(chans, dict) or not isinstance(chans.get(name), dict):
        return
    chans[name]["tested_ok"] = bool(ok)
    chans[name]["tested_at"] = round(time.time(), 3)
    try:
        _write(doc)
    except OSError:                                     # pragma: no cover
        pass


# ── the listener ────────────────────────────────────────────────────────────

@bp.route("/api/notify/listener", methods=["GET"])
def listener():
    """Is **this server** receiving run reports, and who can send them?

    Two different questions, and the page needs both: ``configured`` is what
    the key file says, ``live`` is whether the route is actually registered
    in the running app.  They disagree for exactly as long as it takes to
    restart — `app.py` registers the listener at startup from the file, so
    the first key ever issued does not open the route until then, and a
    person watching a 404 deserves to be told that rather than left to guess
    between the four things a 404 can mean.

    The route segment is not a secret: it appears in every access log, as
    any path does.  What it buys is that a scanner sweeping fixed paths
    finds nothing (`access-control.md` § 8 rule 7).  The keys are secrets and
    only their user names appear.
    """
    from ...monitor import notify_keys_path, read_notify_keys
    route, keys = read_notify_keys()
    live = current_app.config.get("MB_NOTIFY_ROUTE") or ""
    return jsonify({"ok": True,
                    "path": str(notify_keys_path()),
                    "configured": bool(route and keys),
                    "route": route or "",
                    "live": bool(live),
                    "live_route": live,
                    "users": sorted(keys)})


@bp.route("/api/notify/listener/keys/<user>", methods=["POST"])
def issue_key(user: str):
    """Issue or rotate one person's signing key.

    Through `auth_setup.issue_notify_key`, the same door `notify-token`
    uses — so the two cannot generate different route segments from the same
    file, which is the failure `run-reports.md` § 4.3 records from when the
    route lived in two places.

    **The key comes back once**, and that is the deliberate exception to
    this module's own rule.  A key that is never shown at the moment it is
    made cannot reach the machine it is for; a key read back later is a leak
    with no purpose.  Issuing and displaying are different acts, and only
    the second one can never happen (`this-machine.md` § 2).
    """
    from ...auth_setup import NotifyKeyError, issue_notify_key
    body = request.get_json(silent=True) or {}
    replace = bool(body.get("replace"))
    try:
        key, seg, previous = issue_notify_key(user, replace=replace)
    except NotifyKeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    host = request.host_url.rstrip("/")
    return jsonify({"ok": True, "user": user, "key": key, "route": seg,
                    "url": f"{host}/api/{seg}",
                    "joined": bool(previous),
                    "live": bool(current_app.config.get("MB_NOTIFY_ROUTE"))})
