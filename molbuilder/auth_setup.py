"""Auth-setup wizard helpers.

Pure functions for building molbuilder.json's ``auth`` block, and the
one secret writer (`write_secret_file`) for the files that block names by
path.  The Click-driven CLI wrapper lives in ``molbuilder.cli`` as
``cmd_auth_setup``; everything personal-data-handling lives here so
it's testable without prompting.  The FILE is written by
`runtime_config.write_config_scope`, the one door for it -- this module
had a writer of its own, and a session-key generator the server never
used (it has its own creator, `web/auth._install_secret_key`), until
2026-09-13.

Privacy contract:
  * The OAuth client secret (and every secret `write_secret_file` is
    handed) is written to a file with mode 0600 in the config directory
    -- wherever
    :func:`molbuilder.config_dir.config_dir` resolves it, NOT a hardcoded
    ``$HOME/.config`` (``MOLBUILDER_CONFIG_DIR`` and ``XDG_CONFIG_HOME``
    both move it, which is how a person keeps secrets off an NFS $HOME).
    Their contents are NEVER printed, NEVER returned through the API, and
    nothing THIS MODULE writes into molbuilder.json is a secret literal --
    it emits ``client_secret_file``, a path, every time.

    That is a property of the wizard and not of the file format, which the
    sentence here claimed until 2026-09-12.  ``molbuilder.json`` *may*
    legally carry a literal ``client_secret``: ``runtime_config``'s
    ``_validate_secret_pair`` accepts exactly one of the two and says
    ``client_secret_file`` is merely *preferred*, and
    ``web/auth_providers/oauth.py`` reads a literal when it is there.  A
    reader who took this bullet as a guarantee about the file would be
    wrong about a config somebody hand-wrote.
  * The system user account name -- ``getpass.getuser()`` -- is the
    single source of identity.  No other identifier is hardcoded
    anywhere in molbuilder; the wizard derives the ASU CAS
    ``allowed_users`` entry as ``<user>@asu.edu`` and the Google
    ``allowed_users`` entry from an interactive prompt (no assumption
    that the Google account == system user).
  * molbuilder.json itself is written mode 0600 too (by its door): it
    carries no secret literals, but it carries the secret-file PATHS,
    which is enough for an attacker with read-only access to those paths.
"""
from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_dir import ensure_private_dir



# --------------------------------------------------------------------- #
#  Path helpers                                                          #
# --------------------------------------------------------------------- #


# NO PATH HELPERS HERE, and that is the change (I8, 2026-09-13).  Three stood
# here -- `default_secret_dir()` returning `config_dir()`, `secret_key_path()`
# returning `config_dir.session_key()`, `google_client_secret_path()`
# returning `config_dir.google_client_secret()`.  Each was a one-line
# pass-through, and each was a SECOND PUBLIC NAME for a door `config_dir`
# already owns: § 3.1 spelled one and § 2.1e the other for the same file.
# `default_secret_dir` had no production caller at all, only tests.
#
# A11: one home per filename.  A module that re-exports another module's
# resolver has not given the file a home, it has given it two names -- which
# is the shape `config_dir.py` was created to end, and this module's own
# docstring is quoted in that file as one of the three that had to agree by
# comment.  Callers ask `config_dir` directly now.


# --------------------------------------------------------------------- #
#  Secret generation + on-disk emission                                  #
# --------------------------------------------------------------------- #




def write_secret_file(path: Path, contents: str) -> None:
    """Write ``contents`` to ``path`` with mode 0600 (owner read/write).

    Creates parent dirs with mode 0700 if missing.  Refuses to write
    an empty secret (defense against accidentally truncating a real
    one with a placeholder).

    **Atomic and private, which took two tries.**  Through
    :func:`molbuilder.persist.write_bytes` with ``mode=0o600`` -- the one
    writer this package puts bytes through (`configuration.md` § 2.3).  It
    stages a ``mkstemp`` temp, which is owner-only from the moment it exists,
    and ``os.replace``s it over the target.  So:

    * there is **no window at a looser mode**, because no other mode is ever
      set on the inode the secret lands in;
    * a failed write -- full disk, crash, kill -- **leaves the previous secret
      in place**.  This was the defect: until 2026-09-12 this function opened
      the target ``O_TRUNC``, so an interrupted write destroyed it.  For
      ``notify_keys`` that is every key ever issued.  R10 (2026-08-12) had
      aligned what it called *"the last in-place ``O_TRUNC`` write"* with the
      atomic writer; this was another one, and it could not be aligned then
      because ``write_bytes`` widened the mode to 0644 on the way past;
    * a symlink planted at the path is **replaced, not followed**, so the
      earlier ``O_NOFOLLOW`` guard is no longer what carries that.

    The previous attempt (2026-08-27) got the first point only: it opened the
    target and ``fchmod``ed the descriptor before the first byte, which fixes
    the mode of an inode that already exists and cannot make the write atomic.
    """
    if not contents:
        raise ValueError(
            "write_secret_file: refusing to write an empty secret."
        )
    path = Path(path)
    # The parent through the ONE creator: made at 0700 when this call is what
    # makes it, and otherwise left as the operator set it.  Until 2026-09-13
    # this did its own `mkdir` and then `os.chmod(parent, 0o700)` -- which,
    # for the session key, the notify keys and the Google secret, is the
    # CONFIG ROOT: re-moded on every secret write, against the decision
    # `ensure_private_dir` records (on a cluster `XDG_CONFIG_HOME=/scratch/
    # $USER` is how a person keeps tokens off NFS `$HOME`, and silently
    # re-moding what they set up is the program deciding for them).
    # `write_config_scope` already honoured that; this writer contradicted
    # it.  Seeding seeds; `envs doctor` reports what is loose.
    ensure_private_dir(path.parent)
    # ONE WRITER, and 0600 is a parameter of it rather than a second writer
    # (`configuration.md` § 2.3).  The temp mkstemp makes is owner-only before
    # it has a name, so the mode is never wrong; os.replace is atomic, so the
    # old secret is either fully replaced or fully intact.
    from .persist import write_bytes
    write_bytes(path, contents.encode("utf-8"), mode=0o600)


#: A user id, as the LISTENER will accept it.  It becomes a log filename on
#: the server, so it is limited to what `notify.py` enforces when it writes.
#: Refusing at issue time is cheaper than a key that authenticates and then
#: cannot be recorded.
NOTIFY_USER_RE = re.compile(r"^[A-Za-z0-9._@+-]{1,128}$")


class NotifyKeyError(ValueError):
    """A key could not be issued, with a reason a person can act on."""


def issue_notify_key(user: str, *,
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
    from .monitor import (is_route_segment, notify_keys_document,
                          notify_keys_path, read_notify_keys)
    if not NOTIFY_USER_RE.fullmatch(user or ""):
        raise NotifyKeyError(
            f"{user!r} is not usable as a user id here. It becomes a log "
            f"FILENAME on the server, so it is limited to letters, digits "
            f"and . _ @ + - (max 128).")
    path = notify_keys_path()
    # THE ROUTE COMES OUT OF THE FILE IT WENT INTO, so a second key joins the
    # first by default and there is nothing for the caller to remember.
    existing_route, existing = read_notify_keys()
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
    # NEVER WRITE A ROUTE THE READER WOULD REFUSE.  `monitor.read_notify_keys`
    # returns `(None, {})` for a segment failing its rule -- not just the bad
    # route, the WHOLE FILE -- so writing one unchecked destroys every key
    # already issued and takes the listener with it.  Measured 2026-09-12:
    # `--route a/b` was accepted and reported success, after which the server
    # had no route and no keys, silently, because a notifier swallows
    # failures.  The check is `monitor`'s own door, not a second regex here:
    # the two ends must not be free to disagree about what a segment is.
    if not is_route_segment(seg):
        raise NotifyKeyError(
            f"{seg!r} is not a usable route segment -- letters, digits, '-' "
            f"and '_' only (max 128), because it becomes one path component "
            f"of the URL the job posts to.  Nothing was written; the keys "
            f"already in {path} are untouched.")
    write_secret_file(path, notify_keys_document(seg, existing))
    return token, seg, existing_route


# --------------------------------------------------------------------- #
#  Provider entries (round-trippable through runtime_config validators) #
# --------------------------------------------------------------------- #


# ASU CAS endpoints.  These are public, documented at
# https://uto.asu.edu/sites/default/files/2022-10/ASU%20CAS%20Documentation.pdf
# Hardcoding is OK because they're institutional URLs, not secrets.
_ASU_CAS_LOGIN_URL = "https://weblogin.asu.edu/cas/login"
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
        # NO `service_validate_url`.  It was written here and read nowhere --
        # python-cas derives the validate endpoint from the login URL's root and
        # takes no parameter for an explicit one.  Writing a key the client
        # cannot consult told the operator a lie about what their config did.
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
    :func:`molbuilder.config_dir.session_key`, where the server looks and
    creates it -- the wizard does not touch it (`configuration.md` § 2.1e).
    """
    # NO EMPTINESS CHECK HERE.  It raised `ValueError("at least one provider
    # is required")` until 2026-09-09 -- a SECOND implementation of
    # `runtime_config._read_auth`'s "non-empty list" rule, with a different
    # message and a different exception type, and nothing checking the two
    # agreed.  The block is written through `write_config_scope`, which
    # validates the merge with the server's own validator, so the rule has one
    # home and the wizard reports it in the server's own words.
    return {"providers": list(providers)}
