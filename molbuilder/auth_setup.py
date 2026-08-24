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
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_dir import DIRNAME, config_dir


# --------------------------------------------------------------------- #
#  Path helpers                                                          #
# --------------------------------------------------------------------- #


def default_secret_dir(home: Optional[Path] = None) -> Path:
    """Return ``$HOME/.config/molbuilder`` (Path object, not created).

    Override via ``$XDG_CONFIG_HOME`` if set, mirroring the XDG Base
    Directory spec -- so a user with ``$XDG_CONFIG_HOME=/scratch/$USER``
    keeps secrets off the NFS-mounted $HOME on HPC nodes.  That convention
    is :func:`molbuilder.config_dir.config_dir`'s and is IMPORTED, not
    restated -- this function used to spell it out, one of three copies.

    ``home=`` stays an explicit-root escape hatch for callers (and tests)
    that name the directory outright; it deliberately does NOT consult
    ``XDG_CONFIG_HOME``, because a caller passing a root has already
    answered the question the variable exists to answer.
    """
    if home is None:
        return config_dir()
    return Path(home) / ".config" / DIRNAME


def secret_key_path(home: Optional[Path] = None) -> Path:
    return default_secret_dir(home) / "secret_key"


def google_client_secret_path(home: Optional[Path] = None) -> Path:
    return default_secret_dir(home) / "google_client_secret"


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
    one with a placeholder).  Uses os.open(O_CREAT | O_TRUNC | O_WRONLY,
    0o600) so the file's first byte is written with the right perms --
    a write-then-chmod sequence has a window where the file is
    world-readable.
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
    # Atomic-ish create: open with 0600 + write + close.  If a file
    # already exists at the path with looser perms, this overwrites it
    # AND tightens the perms in one step (O_TRUNC + the mode arg).
    fd = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        os.write(fd, contents.encode("utf-8"))
    finally:
        os.close(fd)
    # Belt-and-braces: an existing file's mode isn't changed by the
    # mode arg to os.open (the mode arg only applies to newly-created
    # inodes), so chmod here covers the overwrite case.
    os.chmod(path, 0o600)


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


def build_auth_block(providers: List[Dict[str, Any]],
                      secret_key_file: Path,
                      ) -> Dict[str, Any]:
    """Return the ``auth`` block as a dict ready for json.dumps.

    The block carries ``providers`` (a list, in render order on the
    sign-in page) and ``secret_key_file`` (path to a 0600 file holding
    the Flask session signing key).  No secret literals are in this
    block -- both client_secret_file and secret_key_file point at
    out-of-band files.
    """
    if not providers:
        raise ValueError(
            "build_auth_block: at least one provider is required."
        )
    return {
        "providers":        list(providers),
        "secret_key_file":  str(secret_key_file),
    }


def emit_molbuilder_json(output_path: Path,
                          auth_block: Dict[str, Any],
                          *,
                          force: bool = False,
                          existing: Optional[Dict[str, Any]] = None,
                          ) -> Path:
    """Write ``./molbuilder.json`` carrying the ``auth_block``.

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
