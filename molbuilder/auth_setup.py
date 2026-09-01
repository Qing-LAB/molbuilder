"""Auth-setup wizard helpers.

Pure functions for generating molbuilder.json's ``auth`` block + the
out-of-band secret files (Flask session key, OAuth client secret).
The Click-driven CLI wrapper lives in ``molbuilder.cli`` as
``cmd_auth_setup``; everything personal-data-handling lives here so
it's testable without prompting.

Privacy contract:
  * Secrets (Flask session key, OAuth client secret) are written to
    files with mode 0600 in ``$HOME/.config/molbuilder/``.  Their
    contents are NEVER printed, NEVER returned through the API, and
    NEVER land in molbuilder.json (only file PATHS land there).
  * The system user account name -- ``getpass.getuser()`` -- is the
    single source of identity.  No other identifier is hardcoded
    anywhere in molbuilder; the wizard derives the ASU CAS
    ``allowed_users`` entry as ``<user>@asu.edu`` and the Google
    ``allowed_users`` entry from an interactive prompt (no assumption
    that the Google account == system user).
  * molbuilder.json itself is written mode 0600 too: it carries no
    secret literals, but it carries the secret-file PATHS, which is
    enough for an attacker with read-only access to those paths.
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_dir import DIRNAME, config_dir


# --------------------------------------------------------------------- #
#  Path helpers                                                          #
# --------------------------------------------------------------------- #


def default_secret_dir() -> Path:
    """The config directory, where this installation's secrets live.

    :func:`molbuilder.config_dir.config_dir` outright -- ``MOLBUILDER_CONFIG_DIR``
    if set, else ``$XDG_CONFIG_HOME/molbuilder``, else ``~/.config/molbuilder``
    (`configuration.md` § 2.1c).

    **The ``home=`` parameter is gone** (2026-08-31).  It was an escape hatch
    for callers naming a root outright, it had no production caller, and it
    rebuilt the ``.config/<name>`` rule itself -- the one place `config_dir`'s
    convention was still spelled a second time.  A second override for a
    directory that already has an environment one is exactly the duplication
    this module's own history is about; a test that wants a different root
    sets the variable, like everything else.
    """
    return config_dir()


def secret_key_path() -> Path:
    from .config_dir import session_key
    return session_key()


def google_client_secret_path() -> Path:
    from .config_dir import google_client_secret
    return google_client_secret()


# --------------------------------------------------------------------- #
#  Secret generation + on-disk emission                                  #
# --------------------------------------------------------------------- #


def generate_session_secret() -> str:
    """Return a fresh 32-byte URL-safe Flask session key.

    Using ``secrets.token_urlsafe(32)`` (NOT ``os.urandom`` directly)
    because it returns the value as base64-urlsafe text -- safe to
    write straight to a file without binary-mode handling, safe to
    paste into a JSON string if the user prefers literal secrets.
    """
    return secrets.token_urlsafe(32)


def write_secret_file(path: Path, contents: str) -> None:
    """Write ``contents`` to ``path`` with mode 0600 (owner read/write).

    Creates parent dirs with mode 0700 if missing.  Refuses to write
    an empty secret (defense against accidentally truncating a real
    one with a placeholder).

    **The mode is set on the descriptor before the first byte**, so there
    is no window where the content exists at looser permissions.  This
    docstring claimed that property from the start and the code did not
    have it: the ``0o600`` argument to ``os.open`` applies to a newly
    created inode only, so overwriting an existing loose file wrote the
    secret at the OLD mode and tightened it afterwards.  Measured and fixed
    2026-08-27 -- see the comment below.
    """
    if not contents:
        raise ValueError(
            "write_secret_file: refusing to write an empty secret."
        )
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Tighten parent dir perms too -- 0700 keeps the directory listing
    # private from other users on shared boxes.  No-op if already
    # tighter.  Some umasks make mkdir create 0755; force it.
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass
    # The mode argument to os.open applies to a NEWLY CREATED inode only.
    # Overwriting a file that already exists with looser permissions left
    # it loose -- and the sequence was open, WRITE, close, chmod, so the
    # secret was on disk at the old mode for the length of the write.
    # Measured 2026-08-27 on a 0644 file: the bytes landed at 0644 and were
    # tightened afterwards.  A small window, but a real one, and every
    # secret this module writes went through it.
    #
    # fchmod on the open descriptor closes it: the mode is right before any
    # content exists.  O_NOFOLLOW refuses a symlink planted at the path --
    # the parent is 0700 so only the owner could plant one, but "the owner
    # would not" is not a mechanism.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags, 0o600)
    try:
        os.fchmod(fd, 0o600)              # BEFORE the first byte
        os.write(fd, contents.encode("utf-8"))
    finally:
        os.close(fd)


#: A user id, as the LISTENER will accept it.  It becomes a log filename on
#: the server, so it is limited to what `notify.py` enforces when it writes.
#: Refusing at issue time is cheaper than a key that authenticates and then
#: cannot be recorded.
NOTIFY_USER_RE = re.compile(r"^[A-Za-z0-9._@+-]{1,128}$")


class NotifyKeyError(ValueError):
    """A key could not be issued, with a reason a person can act on."""


def issue_notify_key(user: str, *, path: Optional[Path] = None,
                     route: Optional[str] = None,
                     replace: bool = False):
    """Issue one run-report signing key.  ``(key, segment, previous)``.

    **One door, because there are two of them.**  `cli notify-token` and the
    *This machine* tab both do this, and they must write the same file the
    same way -- a second implementation would be free to generate a second
    route segment and silence everybody already set up, which is the exact
    failure `run-reports.md` § 4.3 records from when the route lived in two
    places.

    ``previous`` is the segment the file held BEFORE this call, or ``None``
    for the first key.  Returned rather than swallowed because it is what
    tells the four cases apart -- first key, joined the file's route,
    adopted a segment the file did not have, or **moved** the route because
    ``route`` disagreed with the file.  Only the last one stops every key
    already issued, and a caller that cannot see it cannot warn about it.

    The key is **returned**, once.  There is no way to read it back out of
    the file in a form anyone can use, and that is deliberate
    (`this-machine.md` § 2).
    """
    from .monitor import (notify_keys_document, notify_keys_path,
                          read_notify_keys)
    if not NOTIFY_USER_RE.match(user or ""):
        raise NotifyKeyError(
            f"{user!r} is not usable as a user id here. It becomes a log "
            f"FILENAME on the server, so it is limited to letters, digits "
            f"and . _ @ + - (max 128).")
    path = Path(path) if path else notify_keys_path()
    # THE ROUTE COMES OUT OF THE FILE IT WENT INTO, so a second key joins the
    # first by default and there is nothing for the caller to remember.
    existing_route, existing = read_notify_keys(path)
    if user in existing and not replace:
        raise NotifyKeyError(
            f"{user!r} already has a key in {path}. Re-issue with `replace` "
            f"to generate a new one -- the old one stops working the moment "
            f"you do.")
    token = secrets.token_urlsafe(32)
    existing[user] = token
    # GENERATED, NOT NAMED.  A word chosen in the source would be committed
    # to a public repository and so be exactly as public as `notify`, only
    # less honest about what it does (`access-control.md` § 8 rule 7).
    seg = route or existing_route \
        or secrets.token_urlsafe(12).replace("-", "").replace("_", "")
    write_secret_file(path, notify_keys_document(seg, existing))
    return token, seg, existing_route


# --------------------------------------------------------------------- #
#  Provider entries (round-trippable through runtime_config validators) #
# --------------------------------------------------------------------- #


# ASU CAS endpoints.  These are public, documented at
# https://uto.asu.edu/sites/default/files/2022-10/ASU%20CAS%20Documentation.pdf
# Hardcoding is OK because they're institutional URLs, not secrets.
_ASU_CAS_LOGIN_URL = "https://weblogin.asu.edu/cas/login"
_ASU_CAS_VALIDATE_URL = (
    "https://weblogin.asu.edu/cas/p3/serviceValidate"
)
_ASU_EMAIL_DOMAIN = "asu.edu"


def build_asu_cas_entry(asurite: str,
                         *,
                         provider_id: str = "asu-cas",
                         label: str = "ASU CAS",
                         ) -> Dict[str, Any]:
    """Return a validated ASU CAS provider entry.

    ``asurite`` is the ASU username (==> CAS principal); the wizard
    defaults it to ``getpass.getuser()`` so the on-disk system user
    is the single source of identity, with no other hardcoded ID.

    ASU CAS does NOT release the email attribute, only the ASURITE
    principal.  We use ``email_domain='asu.edu'`` so the auth layer
    synthesises ``<asurite>@asu.edu`` for the allowlist match -- the
    same pattern documented in runtime_config._validate_cas.
    """
    asurite = (asurite or "").strip()
    if not asurite:
        raise ValueError(
            "build_asu_cas_entry: 'asurite' is required + non-empty "
            "(it keys the allowed_users entry)."
        )
    if "@" in asurite:
        raise ValueError(
            f"build_asu_cas_entry: 'asurite' should be the ASU "
            f"username, not an email.  Got {asurite!r}; expected "
            f"something like {asurite.split('@', 1)[0]!r}."
        )
    return {
        "id":                   provider_id,
        "kind":                 "cas",
        "label":                label,
        "login_url":            _ASU_CAS_LOGIN_URL,
        "service_validate_url": _ASU_CAS_VALIDATE_URL,
        "version":              3,
        "email_domain":         _ASU_EMAIL_DOMAIN,
        "allowed_users":        [f"{asurite}@{_ASU_EMAIL_DOMAIN}"],
    }


def build_google_entry(client_id: str,
                        client_secret_file: Path,
                        allowed_users: List[str],
                        *,
                        provider_id: str = "google",
                        label: str = "Google",
                        hosted_domain: Optional[List[str]] = None,
                        ) -> Dict[str, Any]:
    """Return a validated Google OAuth provider entry.

    ``client_secret_file`` is a Path to a 0600 file the wizard has
    already written; the secret literal stays out of molbuilder.json
    (which is the whole point of the file-pointer indirection).

    ``allowed_users`` is the list of Google-account emails that are
    permitted to sign in.  The wizard prompts for these separately
    from the ASU prompt -- a user's Google account is rarely the
    same as their ASU email.

    ``hosted_domain``, when set, restricts sign-in to Google Workspace
    accounts in the given domains (e.g. ``["asu.edu"]``).  Empty list
    (the default) means "any Google account in allowed_users".
    """
    client_id = (client_id or "").strip()
    if not client_id:
        raise ValueError(
            "build_google_entry: 'client_id' is required + non-empty."
        )
    if not allowed_users:
        raise ValueError(
            "build_google_entry: 'allowed_users' must contain at least "
            "one email.  An empty list = nobody can sign in."
        )
    cleaned_users = [u.strip() for u in allowed_users if u.strip()]
    if not cleaned_users:
        raise ValueError(
            "build_google_entry: 'allowed_users' had only whitespace."
        )
    return {
        "id":                  provider_id,
        "kind":                "google",
        "label":                label,
        "client_id":            client_id,
        "client_secret_file":   str(client_secret_file),
        "allowed_users":        cleaned_users,
        "hosted_domain":        list(hosted_domain or []),
    }


# --------------------------------------------------------------------- #
#  Top-level molbuilder.json shape                                       #
# --------------------------------------------------------------------- #


def build_auth_block(providers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the ``auth`` block as a dict ready for json.dumps.

    The block carries ``providers`` -- a list, in render order on the sign-in
    page.  No secret literals: ``client_secret_file`` points at an out-of-band
    file.

    **It carried ``secret_key_file`` until 2026-08-31**, and writing that key
    is what made the session key configurable.  It now has one home,
    :func:`secret_key_path`, which is where the server looks and where this
    wizard writes -- so the two cannot name different files, which they did
    (`configuration.md` § 2.1e).
    """
    if not providers:
        raise ValueError(
            "build_auth_block: at least one provider is required."
        )
    return {"providers": list(providers)}


def emit_molbuilder_json(output_path: Path,
                          auth_block: Dict[str, Any],
                          *,
                          force: bool = False,
                          existing: Optional[Dict[str, Any]] = None,
                          ) -> Path:
    """Write the machine ``molbuilder.json`` carrying the ``auth_block``.

    WHERE it goes is the caller's to decide and the CLI asks
    :func:`runtime_config.machine_config_path` -- the same helper the reader
    uses -- so the wizard writes the file the server will actually read.

    If ``existing`` is provided, the auth block REPLACES any prior
    auth section but every other top-level key is preserved (so an
    install that already has e.g. ``envs`` or ``tls`` sections stays
    intact).  When ``existing`` is None and ``output_path`` exists on
    disk, refuses to write unless ``force=True`` -- avoids silently
    clobbering a hand-written config.

    File mode is 0600: molbuilder.json carries secret-file PATHS, not
    secret literals, but a path pointing at a 0600 secret is itself
    sensitive (knowing the path is half the attack).
    """
    output_path = Path(output_path)
    # The per-user config directory may not exist yet.  0700, matching the
    # secret directory this file's paths point INTO: the config names those
    # files, and knowing the path is half the attack (see the mode note above).
    #
    # It happened to work before only because the wizard writes the session
    # key first, which creates the directory as a side effect -- correctness
    # resting on call order, one reordering away from a FileNotFoundError on
    # a fresh machine.
    output_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if existing is None and output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists.  Re-run with --force to "
            f"overwrite, or pass --output PATH to write somewhere else."
        )
    merged: Dict[str, Any] = dict(existing or {})
    merged["auth"] = auth_block
    rendered = json.dumps(merged, indent=2, sort_keys=False) + "\n"
    # Same 0600 trick as write_secret_file: create with mode bits at
    # open() time so there's no world-readable window.
    fd = os.open(
        str(output_path),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        os.write(fd, rendered.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(output_path, 0o600)
    return output_path
