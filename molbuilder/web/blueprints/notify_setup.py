"""Setting up **where** a run's reports go — the authenticated half.

`execution/run-reports.md` § 3.1.  The format is the hard part, and it is
the part a person should not have to get right from memory: today you run
``notify-token``, copy the JSON, reach the machine that runs the jobs,
``mkdir -p -m 700``, paste, ``chmod 600``, and remember the directory.
**Four chances to be wrong, and every one fails silently** — absent or
malformed means no notifier, which is indistinguishable from never having
set it up.  A wrong-path defect found on 2026-08-27 came from exactly that.

**This is not `notify.py`, and the separation is deliberate.**  That module
is the *public receiving end*: one route, no session, append-only, and its
whole value is being small enough to reason about.  This one is the
opposite in every respect — it is **login-gated**, it writes a file, and it
reads local configuration.  Putting them together would blur a boundary
that took real work to make crisp.

**Nor does it touch § 1's split.**  That rule is about what *travels*:
policy into ``task.json``, destination and secret into a file that never
leaves the machine.  Writing the **non-travelling half, on the machine it
belongs to**, is putting a secret where the contract says it lives rather
than carrying it anywhere.  The Task-setup policy card keeps its own rule —
*it sets policy; it never sees a key* — because that card writes
``task.json``.

**Whose file is it?**  ``config_dir()`` belongs to the OS account the server
runs as; a molbuilder login is a person.  molbuilder does not manage that
mapping and does not try to (user, 2026-08-27) — `access-control.md` § 8
rule 3, *identity is borrowed, never stored*, applied to the filesystem.

**The key is never read back.**  :func:`destination` says whether one is
present and never what it is: a settings page that can show you a secret is
a settings page that can leak one.  Writing it is fine; reading it back is
not, and there is no reason to.

**One risk, named rather than hidden.**  :func:`test_destination` makes the
*server* POST to a URL the caller supplied, which is a request-forgery
shape.  It is bounded rather than eliminated: the caller is already
logged in and could write the file and run a job to the same effect, the
answer carries only a status code and never the response body, and the
timeout is short.  On a shared server this is a real (small) capability
handed to any signed-in user, and that is the trade.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, request

bp = Blueprint("notify_setup", __name__)

#: How long the test POST may take.  Short: a person is watching, and an
#: unreachable destination is an answer worth getting quickly.
TEST_TIMEOUT_S = 8.0


def _dest_path() -> Path:
    """The destination file, from the **monitor's own** function.

    Imported rather than restated, so this page and the process that reads
    the file on a compute node cannot name different directories.  They did
    once: the Task-setup card said ``~/.molbuilder/notify`` while the
    monitor read ``config_dir()/notify``, and following the card put the
    file where nothing looks.
    """
    from ...monitor import default_notify_path
    return default_notify_path()


def _read() -> Dict[str, Any]:
    """What is on disk, **without the key**."""
    path = _dest_path()
    out: Dict[str, Any] = {"path": str(path), "configured": False,
                           "url": "", "has_key": False, "problem": ""}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return out
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        out["problem"] = f"not valid JSON ({exc})"
        return out
    if not isinstance(obj, dict) or not isinstance(obj.get("url"), str) \
            or not obj["url"]:
        out["problem"] = "needs an object with a 'url' string"
        return out
    out["configured"] = True
    out["url"] = obj["url"]
    out["has_key"] = bool(obj.get("key"))
    try:
        out["mode"] = oct(path.stat().st_mode & 0o777)
    except OSError:                                     # pragma: no cover
        pass
    return out


def _existing() -> Dict[str, Any]:
    """The destination file as it stands, or ``{}``.

    Unreadable or malformed reads as empty, which is the safe direction: a
    save then writes a fresh, valid file rather than refusing, and the
    person is not stuck with a broken one they cannot fix from here.
    """
    try:
        obj = json.loads(_dest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(obj) if isinstance(obj, dict) else {}


def _public(d: Dict[str, Any]) -> Dict[str, Any]:
    """The record minus anything private.  **One door out**, so a route
    cannot forget: `_read` carries the key for `save_destination`'s benefit
    and every response goes through here."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


@bp.route("/api/notify/destination", methods=["GET"])
def destination():
    """Is a destination set up, and where does it live?

    Reports the URL — an address, not a secret — and only *whether* a key
    is present.  ``mode`` rides along because ``0600`` is part of the setup
    being right, and a person who copied the file by hand may not have it.
    """
    from ...runtime_config import read_config
    try:
        cfg = read_config()
        mode = (cfg.get("execution") or {}).get("mode") or ""
    except Exception:                                   # noqa: BLE001
        mode = ""
    out = _read()
    # WHICH MACHINE RUNS THE JOBS decides what this page can do for you.
    # `direct` means the file belongs here, so it can be written. `submit`
    # means the job runs somewhere this server cannot reach, and the most
    # it can do is hand you the exact content to put there.
    out["execution_mode"] = mode
    out["can_write_here"] = (mode != "submit")
    return jsonify({"ok": True, **_public(out)})


@bp.route("/api/notify/destination", methods=["POST"])
def save_destination():
    """Write the destination file, 0600, in the one right place.

    Through `auth_setup.write_secret_file`, which creates the parent at
    0700 and sets the mode on the descriptor **before the first byte** — so
    the secret is never briefly on disk at a looser mode.
    """
    from ...auth_setup import write_secret_file
    body = request.get_json(silent=True) or {}
    url = str(body.get("url") or "").strip()
    key = str(body.get("key") or "").strip()
    # A SAVE UPDATES; IT DOES NOT REPLACE.
    #
    # Writing a fresh `{"url": ..., "key": ...}` destroyed every other field
    # the file held.  Two ways that bit, both found by round-tripping on
    # 2026-08-27:
    #
    #   * the card CLEARS the key field after each save -- correctly, since
    #     a secret left in the DOM ends up in a screenshot -- so the
    #     ordinary next action, fixing a typo in the url, arrived with no
    #     key and wiped the stored one;
    #   * a `headers` block, which `monitor.load_destination` reads and this
    #     page has no input for, vanished the first time anybody edited the
    #     url here.
    #
    # And both failed SILENTLY: an unsigned or unauthenticated report gets a
    # 404 and the notifier swallows it.  So the rule is that this writes the
    # fields it manages over whatever is already there -- which also means a
    # field added to this file later cannot be dropped by a page that
    # predates it.  Removing something deliberately is `Remove`, then save.
    doc = _existing()
    doc["url"] = url
    if key:
        doc["key"] = key
    if not url:
        return jsonify({"ok": False, "error": "a url is required"}), 400
    if not (url.startswith("https://") or url.startswith("http://")):
        return jsonify({"ok": False,
                        "error": "the url must start with https:// "
                                 "(or http:// for a host you trust)"}), 400
    try:
        write_secret_file(_dest_path(), json.dumps(doc, indent=2) + "\n")
    except OSError as exc:
        return jsonify({"ok": False,
                        "error": f"could not write {_dest_path()}: {exc}"}), 500
    return jsonify({"ok": True, **_public(_read())})


@bp.route("/api/notify/destination", methods=["DELETE"])
def clear_destination():
    """Remove the destination.  **Absent is off**, and off is a state a
    person is entitled to reach without a shell."""
    try:
        os.unlink(_dest_path())
    except FileNotFoundError:
        pass
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **_public(_read())})


@bp.route("/api/notify/destination/test", methods=["POST"])
def test_destination():
    """Send one report to the configured destination and say what happened.

    **This is the only check that exercises the whole path** — the file,
    the URL, the route segment, the signature, egress and TLS — and until
    it existed the only way to know a setup worked was to run a job and
    notice nothing arrived.

    It signs with the monitor's own :func:`sign_report`, so a signature this
    accepts is one the listener accepts; a second implementation here could
    pass while the real one failed.
    """
    from ...monitor import load_destination, sign_report
    dest = load_destination()
    if not dest:
        return jsonify({"ok": False,
                        "error": "no destination is set up yet"}), 400
    body = json.dumps({
        "event": "test",
        "text": "a test report from molbuilder's setup page",
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
        return jsonify({"ok": False, "reached": False,
                        "error": f"could not reach it: {exc}"}), 200
    # A 404 IS THE LISTENER'S REFUSAL.  Every gate answers with one so a
    # stranger cannot tell them apart (`run-reports.md` § 4.1) -- which
    # means it cannot tell YOU apart either, and the honest thing is to
    # name all of the possibilities rather than guess between them.
    hint = ""
    if code == 404:
        hint = ("404 -- the listener refuses everything the same way, so "
                "this is a wrong route segment, a wrong or missing key, or "
                "a server with no listener configured. Re-check the url "
                "and key against what `notify-token` printed.")
    elif code and 200 <= code < 300:
        hint = "it arrived."
    # Same advisory bucket: a destination that answered 404 is a fact
    # this endpoint reports successfully.
    return jsonify({"ok": bool(code and 200 <= code < 300),
                    "reached": True, "status": code, "hint": hint}), 200
