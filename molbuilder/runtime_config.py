"""Per-machine runtime configuration, read from the ONE machine config.

This module is named ``runtime_config`` (not just ``config``) because
``molbuilder.config`` is the engine-parameter dataclasses package
(``SiestaConfig``, ``PySCFConfig``, ``TransportConfig``).  Different
concerns:

* ``molbuilder.config.*``        -- L1 dataclasses, calculation
  parameters serialised into the generated input deck.
* ``molbuilder.runtime_config``  -- per-machine deployment knobs
  (TLS paths, conda env names) read at startup from a gitignored
  file in the per-user config directory (`configuration.md` § 2.1c).

The reader has zero UI dependencies: it raises a domain-level
:class:`RuntimeConfigError` on bad input; the CLI / web layer catch
and translate that into their own user-facing surface (``click.UsageError``,
HTTP 400, etc.).  Keeping config-reading at L1 means the same code
serves CLI, web blueprints, and any future Python-API user.

Schema (all sections optional)::

    {
        "tls":  { "cert": "/etc/letsencrypt/.../fullchain.pem",
                  "key":  "/etc/letsencrypt/.../privkey.pem" },
        "envs": { "siesta":  "molbuilder-siesta",
                  "pyscf":   "molbuilder-pySCF",
                  "mdtools": "molbuilder-MDtools" }
    }

For backwards compatibility with the flat shape shipped before the
2026-05-14 four-env design, top-level ``cert`` and ``key`` keys are
also honoured (folded into ``tls`` by :func:`_normalise`).  **An
unknown top-level key is REFUSED with the known sections named** (U7,
2026-08-12; ``_``-prefixed keys are comments) — this header taught
"ignored silently so the file can grow" until R8, the exact tolerance
that silently ate ``admin``/``rate_limit``; the one total list of
sections is the ``_SECTIONS`` registry below (architecture § 8.2a).

This reader is intentionally stateless: it reads the file each time
it's called, parses, validates.  Callers that want a single
process-wide read should go through :mod:`molbuilder.diagnostics`,
which builds the immutable :class:`~molbuilder.diagnostics.Capabilities`
snapshot once at startup.  Putting the cache there (not here) keeps
this module a plain pure function: easy to test, easy to reason about,
no hidden state.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config_dir import config_dir


#: A11: the module that owns the FORMAT owns the NAME.  This one validates
#: `molbuilder.json`'s schema, so the spelling is here -- and the join is here
#: too, once, in :func:`_machine_config_file`.  Everything else asks
#: :func:`machine_config_path`.
CONFIG_FILENAME = "molbuilder.json"
# Per-project config sidecar.  Per docs/execution/running-a-job.md § 5: hidden file in
# the project directory, same schema as the server-wide molbuilder.json.
PROJECT_CONFIG_FILENAME = ".molbuilder.json"

#: One message, two raisers -- `get_scheduler` (which knows the file) and
#: `_validate_scheduler` (which does not).  ``{path}`` is what a person edits.
_ROUTING_MOVED = (
    "{path}: 'scheduler.routing' is no longer configured here.  The reachable "
    "(partition, qos) domains are PROBED, not declared -- run `molbuilder "
    "jobset probe --write` and they land in environment.json, where every "
    "calculation on this machine reads one answer (docs/configuration.md "
    "§ 5).  What stays yours in this file is which of them you WANT: "
    "'scheduler.directives.partition' and '.qos'.")


#: Same shape as `_ROUTING_MOVED`: a retired key gets its own sentence, not
#: the generic "unknown top-level key".
_SECRET_KEY_MOVED = (
    "{path}: 'secret_key_file' is no longer configured.  The session key has "
    "ONE home -- <config dir>/secret_key, beside this file -- and is created "
    "there on first run (docs/configuration.md § 2.1e).  A key naming its own "
    "location is how it came to live in two places at once: this file pointed "
    "at ~/.molbuilder/secret.key while `auth-setup` wrote "
    "<config dir>/secret_key, so running the wizard made a key the server "
    "never read.  Delete the line; move the file if you want the sessions it "
    "signed to survive.")


class RuntimeConfigError(Exception):
    """Raised when ``molbuilder.json`` is present but unreadable / malformed.

    The CLI layer translates this into ``click.UsageError`` and the
    web layer into HTTP 400; the L1 reader itself stays UI-agnostic.
    """


def read_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``molbuilder.json`` from ``path``, or from the one place the
    machine scope lives.

    **The location is asked of :func:`machine_config_path`, never re-derived.**
    This function kept its own copy of the lookup until 2026-08-31, and the
    copy is what let the working-directory step survive its own deletion: the
    door stopped returning a cwd path and this reader went on loading one, so
    a file that every message called unread was still being applied.  There had
    already been a split-brain here once -- until 2026-08-13 this read was
    cwd-only while the section getters honoured both, so an operator with an
    XDG-only config got a server with no auth and no TLS (final review A-7).

    Returns the normalised dict (see :func:`_normalise`).  Returns
    ``{}`` if no file exists (not an error -- the file is optional).
    Raises :class:`RuntimeConfigError` when the JSON is malformed or the
    schema is invalid.
    """
    cfg_path = path if path is not None else machine_config_path()[0]
    if not cfg_path.is_file():
        return {}
    try:
        raw = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(
            f"{cfg_path}: invalid JSON ({exc.msg} at line {exc.lineno})"
        ) from None
    if not isinstance(raw, dict):
        raise RuntimeConfigError(
            f"{cfg_path}: top-level value must be an object, "
            f"got {type(raw).__name__}"
        )
    try:
        return _normalise(raw)
    except RuntimeConfigError as exc:
        # The validators speak in terms of the SCHEMA and spell the
        # generic name; the reader knows WHICH file refused.  A malformed
        # project .molbuilder.json or XDG file used to refuse naming
        # 'molbuilder.json' with no path (R10, 2026-08-12).
        msg = str(exc)
        if str(cfg_path) not in msg:
            raise RuntimeConfigError(f"{cfg_path}: {msg}") from None
        raise


def _read_section(raw: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Return ``raw[key]`` as a fresh dict, validated to be an object.

    Returns ``{}`` when the key is absent.  Raises
    :class:`RuntimeConfigError` when the value is present but not a
    mapping.  Section types beyond "object" (e.g. string-keyed,
    string-valued for ``envs``) are enforced by :func:`_normalise`.
    """
    val = raw.get(key, {})
    if not isinstance(val, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: {key!r} must be an object, "
            f"got {type(val).__name__}"
        )
    return dict(val)


# --------------------------------------------------------------------- #
#  Provider validators                                                  #
# --------------------------------------------------------------------- #
#
# One validator per supported "kind".  Each returns the
# (possibly-normalised, default-filled) provider entry.  Validators
# raise :class:`RuntimeConfigError` for any malformed entry; that
# error bubbles up through ``_normalise`` to the CLI / web layer.
#
# Adding a new backend = add a kind to ``_SUPPORTED_KINDS`` + a
# validator function + a registration handler in
# ``molbuilder/web/auth_providers/``.  The schema layer here knows
# nothing about HTTP, authlib, or python-cas -- only the contract of
# the JSON payload.


_SUPPORTED_KINDS = ("google", "github", "microsoft", "orcid", "cas")

# id must be a URL-safe slug because it appears in route paths
# ``/login/<id>`` and ``/oauth-callback/<id>``.  Restricting to
# [a-z0-9_-] guarantees no quoting issues regardless of WSGI server.
# Also explicitly reject the internal ``mb_`` prefix (which auth_providers/
# oauth.py uses to mangle the operator id before passing it to Authlib --
# preventing a future operator from picking ``id="mb_X"`` and colliding
# with that namespace).
_ID_RE = re.compile(r"^(?!mb_)[a-z0-9][a-z0-9_-]*$")


def _require_str(entry: Mapping[str, Any], key: str, idx: int) -> str:
    """Return ``entry[key]`` as a non-empty string or raise."""
    val = entry.get(key)
    if not isinstance(val, str) or not val:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].{key} is "
            f"required and must be a non-empty string; got {val!r}."
        )
    return val


def _require_str_list(entry: Mapping[str, Any], key: str, idx: int,
                       *, optional: bool = False) -> list:
    """Return ``entry[key]`` as a list[str] or raise.

    When ``optional`` is True, an absent key returns an empty list.
    The list itself may be empty regardless (a documented fail-closed
    case for ``allowed_users``).
    """
    val = entry.get(key)
    if val is None:
        if optional:
            return []
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].{key} is "
            f"required (list of strings; empty list = no one)."
        )
    if not isinstance(val, list) or not all(
            isinstance(s, str) for s in val):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].{key} must be "
            f"a list of strings; got {val!r}."
        )
    return list(val)


def _validate_secret_pair(entry: Mapping[str, Any], idx: int) -> None:
    """Enforce 'exactly one of client_secret / client_secret_file'."""
    has_literal = isinstance(entry.get("client_secret"), str)
    has_file    = isinstance(entry.get("client_secret_file"), str)
    if has_literal and has_file:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}]: set EITHER "
            f"'client_secret' OR 'client_secret_file', not both."
        )
    if not has_literal and not has_file:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}]: one of "
            f"'client_secret' (literal) or 'client_secret_file' "
            f"(path to a 0600 file) is required.  client_secret_file "
            f"is preferred so the config itself stays safe to share."
        )


def _validate_oauth_common(entry: Dict[str, Any], idx: int) -> None:
    """Mutate ``entry`` in place: validate OAuth shared fields."""
    _require_str(entry, "client_id", idx)
    _validate_secret_pair(entry, idx)


def _validate_google(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    _validate_oauth_common(entry, idx)
    entry["hosted_domain"] = _require_str_list(
        entry, "hosted_domain", idx, optional=True
    )
    return entry


def _validate_github(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    _validate_oauth_common(entry, idx)
    entry["allowed_organizations"] = _require_str_list(
        entry, "allowed_organizations", idx, optional=True
    )
    return entry


def _validate_microsoft(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    _validate_oauth_common(entry, idx)
    tenant = entry.get("tenant_id", "common")
    if not isinstance(tenant, str) or not tenant:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].tenant_id must "
            f"be a non-empty string; got {tenant!r}.  Common values: "
            f"'common' (any Microsoft account), 'organizations' (any "
            f"work/school account), a tenant GUID, or a verified "
            f"domain like 'asu.onmicrosoft.com'."
        )
    entry["tenant_id"] = tenant
    return entry


def _validate_orcid(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    _validate_oauth_common(entry, idx)
    return entry


def _validate_cas(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    _require_str(entry, "login_url",            idx)
    _require_str(entry, "service_validate_url", idx)

    version = entry.get("version", 3)
    # ``type(version) is int`` excludes bool (subclass of int) and
    # float; otherwise ``True in (1,2,3)`` and ``3.0 in (1,2,3)``
    # would slip through as valid.
    if type(version) is not int or version not in (1, 2, 3):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].version must "
            f"be 1, 2, or 3 (CAS protocol version); got {version!r}."
        )
    entry["version"] = version

    for opt_str in ("service_url", "ca_certs",
                     "email_attribute", "email_domain"):
        v = entry.get(opt_str)
        if v is not None and (not isinstance(v, str) or not v):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: auth.providers[{idx}].{opt_str} "
                f"must be a non-empty string when set; got {v!r}."
            )
        entry.setdefault(opt_str, None)

    # CAS doesn't always release email -- we need at least one path
    # to produce one for the allowlist match.  ASU CAS, for example,
    # releases only the ASURITE principal; that gets paired with
    # email_domain='asu.edu' to synthesise 'asurite@asu.edu'.
    if entry["email_attribute"] is None and entry["email_domain"] is None:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}] (kind=cas) "
            f"requires at least one of 'email_attribute' (the CAS "
            f"attribute name carrying the email, when the IdP "
            f"releases one) or 'email_domain' (used to synthesise "
            f"'{{principal}}@{{email_domain}}').  Without either "
            f"there's no way to produce an email to match against "
            f"allowed_users."
        )
    return entry


_KIND_VALIDATORS = {
    "google":    _validate_google,
    "github":    _validate_github,
    "microsoft": _validate_microsoft,
    "orcid":     _validate_orcid,
    "cas":       _validate_cas,
}


def _validate_provider(entry: Any, idx: int) -> Dict[str, Any]:
    """Validate one provider entry; return the normalised copy."""
    if not isinstance(entry, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}] must be an "
            f"object; got {type(entry).__name__}."
        )
    out = dict(entry)

    # --- common required fields ------------------------------------- #
    pid = _require_str(out, "id", idx)
    if not _ID_RE.match(pid):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].id {pid!r} "
            f"must be a URL-safe slug matching {_ID_RE.pattern} "
            f"(it keys the route path /login/<id>)."
        )
    _require_str(out, "label", idx)

    kind = _require_str(out, "kind", idx)
    if kind not in _SUPPORTED_KINDS:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].kind {kind!r} "
            f"is not supported.  Supported: {', '.join(_SUPPORTED_KINDS)}."
        )

    # allowed_users is required so the operator must explicitly think
    # about access control.  An empty list is a degenerate-but-valid
    # case (the provider is enabled but nobody can sign in -- useful
    # for temporarily locking out a backend).
    out["allowed_users"] = _require_str_list(out, "allowed_users", idx)

    # --- kind-specific dispatch ------------------------------------- #
    return _KIND_VALIDATORS[kind](out, idx)


# --------------------------------------------------------------------- #
#  Top-level normaliser                                                 #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  The SECTION REGISTRY -- one row per top-level section (U7,           #
#  2026-08-12).  Everything the loader knows about a section is in its  #
#  row: how it is read (its validator), which SCOPES it may live in,    #
#  and whether provenance may print its VALUES.  `_normalise`,          #
#  `_read_project`'s scope refusal, `config_provenance` and             #
#  `write_config_scope` all consult THIS table and nothing else.        #
#                                                                       #
#  Why a table: until it existed each of those four sites kept its own  #
#  partial list, and the gaps were live bugs -- `_normalise` never      #
#  learned `admin` or `rate_limit`, so it silently DROPPED them and     #
#  `get_admin_emails` read post-strip config: nobody could be admin,    #
#  and nothing said why.  A section is either in this table or its      #
#  presence is an ERROR; there is no third state in which it looks      #
#  configured and does nothing.                                         #
# --------------------------------------------------------------------- #

def _read_tls(raw: Mapping[str, Any]):
    # Flat-shape ``cert``/``key`` fold in; nested wins (see _normalise's
    # precedence note).
    tls = _read_section(raw, "tls")
    for flat_key in ("cert", "key"):
        if flat_key in raw and flat_key not in tls:
            tls[flat_key] = raw[flat_key]
    for k, v in tls.items():
        if not isinstance(v, str):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'tls.{k}' must be a string, "
                f"got {type(v).__name__}"
            )
    return tls or None


def _read_envs(raw: Mapping[str, Any]):
    envs = _read_section(raw, "envs")
    for k, v in envs.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'envs' entries must be string -> "
                f"string; got {k!r} -> {v!r}."
            )
        # An empty string would silently degrade dispatch (env_for_category
        # returns "", env_available("") is False, routed_env returns
        # None, the call falls through to host PATH or errors).  Catch
        # it at the config boundary instead.
        if not k or not v:
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'envs' entries cannot be empty "
                f"strings; got {k!r} -> {v!r}."
            )
    return envs or None


def _read_auth(raw: Mapping[str, Any]):
    # Optional; absent ``auth`` means no authentication (the right default
    # for the localhost-only single-user deployment shape).  Explicit-
    # presence check (rather than ``if auth:``): writing ``"auth": {}`` is
    # almost certainly a mistake and we want a clear error rather than a
    # silent degrade to no-auth mode.  Schema and the per-provider
    # ``allowed_users`` gate: ``_validate_provider`` and
    # ``docs/ops/deployment.md``.
    if "auth" not in raw:
        return None
    auth = _read_section(raw, "auth")
    providers = auth.get("providers")
    if not isinstance(providers, list) or not providers:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'auth.providers' must be a "
            f"non-empty list of provider entries when the 'auth' "
            f"section is present.  Got {type(providers).__name__}."
        )
    seen_ids: set[str] = set()
    validated: list[Dict[str, Any]] = []
    for idx, entry in enumerate(providers):
        v = _validate_provider(entry, idx)
        if v["id"] in seen_ids:
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: duplicate provider id "
                f"{v['id']!r} in auth.providers (each entry's "
                f"id keys its route path, so they must be unique)."
            )
        seen_ids.add(v["id"])
        validated.append(v)

    # Optional ``auth.trust_proxy`` flag.  When True, the web layer
    # installs werkzeug's ProxyFix so the FIRST upstream proxy's
    # X-Forwarded-* headers are honoured.  Default False -- the right
    # choice for direct-TLS deploys (see _setup_session_security in
    # molbuilder/web/auth.py for the security implications).
    trust_proxy = auth.get("trust_proxy", False)
    if not isinstance(trust_proxy, bool):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'auth.trust_proxy' must be a "
            f"JSON boolean (true / false); got "
            f"{type(trust_proxy).__name__}."
        )
    return {"providers": validated, "trust_proxy": trust_proxy}


#: One URL segment: letters, digits, '-' and '_'.  Deliberately narrow --
#: the value becomes part of a route, and anything with a slash or a dot in
#: it would silently mean a different path than the one written down.
_NOTIFY_ROUTE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _read_notify_keys_file(raw: Mapping[str, Any]):
    """Path to the run-report signing-key file.  A top-level SCALAR, like
    ``secret_key_file`` and for the same reason: the config carries the
    PATH, and the keys live in a 0600 file beside it.

    Renamed from ``notify_tokens_file`` 2026-08-27: the secret stopped
    being a bearer token and became a signing key that never travels
    (`run-reports.md` § 4.1).  No compatibility shim -- a stale key under
    the old name would leave the route unregistered, which is the safe
    reading and the one `access-control.md` § 8 rule 1 asks for.
    """
    value = raw.get("notify_keys_file")
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'notify_keys_file' must be a "
            f"string path; got {type(value).__name__}."
        )
    return value


def _read_notify_route(raw: Mapping[str, Any]):
    """The listener's URL segment, generated per deployment.

    **Not a secret** -- it is in every access log, as any path is -- but
    never a fixed word either: this repository is public, so a word chosen
    in the source is exactly as public as ``notify`` and less honest about
    what it does (`access-control.md` § 8 rule 7).  ``notify-token``
    generates it the same way it generates the key.

    There is no default.  A default would be a fixed word again, and it is
    what makes rule 1 hold here: with this key absent there is no route at
    all, so nothing can be probed for.
    """
    value = raw.get("notify_route")
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'notify_route' must be a string; "
            f"got {type(value).__name__}."
        )
    seg = value.strip().strip("/")
    if not seg or not _NOTIFY_ROUTE_RE.match(seg):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'notify_route' must be one URL segment "
            f"of letters, digits, '-' or '_' (got {value!r}).  "
            f"`molbuilder notify-token` generates one."
        )
    return seg


def _require_object_section(raw: Mapping[str, Any], name: str):
    """The lazy-validated sections (scheduler, execution, rate_limit):
    merged and/or validated by their getters or consumers, so here we
    only keep the key alive and reject a non-object early.  A partial
    block (e.g. project scope supplying only ``defaults.time``) is legal
    at this layer -- completeness is a merged-config property."""
    if name not in raw:
        return None
    section = raw[name]
    if not isinstance(section, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: '{name}' must be an object; got "
            f"{type(section).__name__}."
        )
    return dict(section)


def _read_admin(raw: Mapping[str, Any]):
    # Who may do the things only an operator should do (ops/deployment.md;
    # read back by get_admin_emails -- absent or empty means NOBODY).
    # Eagerly shape-checked: a mistyped emails list would otherwise fail
    # silently into the safe-but-wrong "nobody".
    if "admin" not in raw:
        return None
    section = raw["admin"]
    if not isinstance(section, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'admin' must be an object like "
            f'{{"emails": ["operator@example.edu"]}}; got '
            f"{type(section).__name__}."
        )
    emails = section.get("emails", [])
    if (not isinstance(emails, (list, tuple))
            or not all(isinstance(e, str) for e in emails)):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'admin.emails' must be a list of "
            f"email strings; got {emails!r}."
        )
    return dict(section)


#: name -> how it is read · where it may live · whether provenance may
#: print its values.  ``scopes``: "machine" = molbuilder.json (the config dir),
#: "project" = the .molbuilder.json in a project or calculation folder
#: -- ONE name for that scope, everywhere (2026-08-23; it answered to
#: "bundle" in provenance output until then).  A section absent from a
#: scope's tuple is REFUSED there, never silently ignored -- S1c's
#: argument, generalised: a section that is read, validated and then
#: dropped looks effective while nobody applied it.  ``provenance_safe``
#: gates `config_provenance`: True only where every value is printable
#: in logs (no secrets, no paths to secrets).
#: Every directory ``paths`` may name.  A closed set: a key nothing reads
#: would look effective and do nothing, which is the argument behind every
#: refusal in `configuration.md`.
#:
#: ``logs``, ``run`` and ``reports`` were here between 2026-08-31 and the same
#: day, and retiring them is what removed a dependency inversion rather than
#: working around one -- see :data:`_OPERATIONAL_PATHS_MOVED`.
_PATH_KEYS = ("projects",)

#: Same shape as `_ROUTING_MOVED` and `_SECRET_KEY_MOVED`: a retired key gets
#: its own sentence, so it does not read as a typo.
_OPERATIONAL_PATHS_MOVED = (
    "{path}: 'paths.{key}' is no longer configured.  Operational state follows "
    "XDG's own directories -- $XDG_STATE_HOME for logs and reports, "
    "$XDG_RUNTIME_DIR for pidfiles (docs/configuration.md § 2.1d).  A config "
    "key said the same thing a second way, and being a second way is what put "
    "the answer out of reach of the layer that needs it: the `serve` "
    "supervisor writes its log before any config is read.  Set the variable "
    "instead -- it moves every application's state together, which is the "
    "setting a person makes for their account rather than for this program.")


def _read_paths(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """``paths`` — where molbuilder keeps things that are not its own code.

    ``projects`` is the tree of projects.  It exists because
    the default (inside the checkout) is not always writable or wanted --
    a cluster home with a small quota, a scratch filesystem, a shared tree
    (user, 2026-08-22).  Everything that touches the tree goes through
    ``projects.projects_root``, so setting it here moves the tree for every
    surface at once: the sidebar, the CLI verbs, the workspace store, the
    pseudopotential anchor.

    A relative value is resolved against the molbuilder root, so the
    setting means the same thing whatever directory you run from.

    ``logs``, ``run`` and ``reports`` name where OPERATIONAL STATE goes,
    overriding the XDG directories `config_dir.state_dir` and
    `config_dir.runtime_dir` default to (`plans/config-access-plan.md` § 3.2).
    They are read through :func:`logs_dir`, :func:`run_dir` and
    :func:`reports_dir`, which is also where the defaults live -- so a person
    with a small ``$HOME`` and a large scratch puts the logs on scratch
    without moving their secrets.

    **Added here rather than in a section of their own** (2026-08-31), because
    this section already answers *"where does molbuilder keep things that are
    not its own code"* and a second block asking the same question is the
    fragmentation this change exists to end.  The first attempt DID add a
    second ``paths`` reader and registry entry: the duplicate dict key
    silently won, and ``paths.projects`` -- a live setting that moves the whole
    project tree -- began being refused as unknown.
    """
    if "paths" not in raw:
        return None
    section = _require_object_section(raw, "paths")
    if section is None:
        return None
    retired = sorted(set(section) & {"logs", "run", "reports"})
    if retired:
        raise RuntimeConfigError(
            _OPERATIONAL_PATHS_MOVED.format(path=CONFIG_FILENAME,
                                            key=retired[0]))
    unknown = set(section) - set(_PATH_KEYS)
    if unknown:
        raise RuntimeConfigError(
            f"molbuilder.json: unknown key(s) in `paths`: "
            f"{', '.join(sorted(unknown))}.  The keys are "
            f"{', '.join(_PATH_KEYS)} (architecture.md § 8.2, "
            f"plans/config-access-plan.md § 3.2).")
    for key in _PATH_KEYS:
        val = section.get(key)
        if val is None:
            continue
        if not isinstance(val, str) or not val.strip():
            raise RuntimeConfigError(
                f"molbuilder.json: paths.{key} must be a non-empty string "
                f"path; got {val!r}.")
    return section


_SECTIONS: Dict[str, Dict[str, Any]] = {
    "tls":               {"read": _read_tls,
                          "scopes": ("machine",), "provenance_safe": False},
    "envs":              {"read": _read_envs,
                          "scopes": ("machine",), "provenance_safe": False},
    "auth":              {"read": _read_auth,
                          "scopes": ("machine",), "provenance_safe": False},
    "notify_keys_file":  {"read": _read_notify_keys_file,
                          "scopes": ("machine",), "provenance_safe": False},
    "notify_route":      {"read": _read_notify_route,
                          "scopes": ("machine",), "provenance_safe": False},
    "execution":         {"read": lambda raw: _require_object_section(
                              raw, "execution"),
                          "scopes": ("machine", "project"),
                          "provenance_safe": True},
    "script_generation": {"read": lambda raw: (
                              _validate_script_generation(
                                  raw["script_generation"])
                              if "script_generation" in raw else None),
                          "scopes": ("machine", "project"),
                          "provenance_safe": True},
    "scheduler":         {"read": lambda raw: _require_object_section(
                              raw, "scheduler"),
                          "scopes": ("machine", "project"),
                          "provenance_safe": False},
    "checkpoint":        {"read": lambda raw: (
                              _validate_checkpoint(raw["checkpoint"])
                              if "checkpoint" in raw else None),
                          "scopes": ("machine",), "provenance_safe": False},
    "admin":             {"read": _read_admin,
                          "scopes": ("machine",), "provenance_safe": False},
    "rate_limit":        {"read": lambda raw: _require_object_section(
                              raw, "rate_limit"),
                          "scopes": ("machine",), "provenance_safe": False},
    "paths":             {"read": _read_paths,
                          "scopes": ("machine",), "provenance_safe": True},
}

#: Top-level keys that are NOT section names but are still known: the
#: legacy flat spelling of ``tls`` (folded by ``_read_tls``).
_FLAT_ALIASES = ("cert", "key")


def _normalise(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Read every registered section + validate values; refuse the rest.

    Precedence: when both the nested ``tls`` section AND the flat
    ``cert``/``key`` keys are present, the nested value wins.  Migrating
    from flat to nested is therefore non-destructive.

    **An unknown top-level key is an ERROR, not tolerance** (U7,
    2026-08-12; running-a-job.md § 5 amended).  "Ignored silently" was the
    documented behaviour, and it is exactly how `admin` and `rate_limit`
    -- sections with live getters -- were dropped before they reached the
    web layer: the file looked configured and nobody could be admin.  The
    same hole swallows every typo'd section name.  The registry makes
    "known" one total list, so refusing is precise and the message can
    name what IS known.

    Value-type validation lives in each section's ``read`` (see
    ``_SECTIONS``) so the ``get_*`` accessors stay trivial and callers
    never see a section whose entries aren't the documented types.
    """
    # A leading underscore marks a COMMENT key ("_comment_tls": ...) --
    # JSON has no comments and the committed templates lean on this idiom.
    # An explicit marker is not the typo class the refusal exists for.
    unknown = sorted(k for k in raw
                     if k not in _SECTIONS and k not in _FLAT_ALIASES
                     and not k.startswith("_"))
    if "secret_key_file" in unknown:
        raise RuntimeConfigError(_SECRET_KEY_MOVED.format(path=CONFIG_FILENAME))
    if unknown:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: unknown top-level "
            f"key(s) {', '.join(map(repr, unknown))}.  Known sections: "
            f"{', '.join(_SECTIONS)} (plus the flat tls aliases "
            f"{', '.join(_FLAT_ALIASES)}).  A key this loader does not "
            f"know would be silently ineffective -- refused instead, so a "
            f"typo cannot masquerade as configuration "
            f"(running-a-job.md § 5).  A key starting with '_' is a "
            f"comment and is ignored by design."
        )
    out: Dict[str, Any] = {}
    for name, spec in _SECTIONS.items():
        value = spec["read"](raw)
        if value is not None:
            out[name] = value
    return out


def get_auth(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``auth`` section, or ``{}``.

    Trivial accessor -- type validity is enforced upstream in
    :func:`_normalise`.  The returned dict has the shape
    ``{"providers": [...]}`` when auth is configured, or ``{}`` when
    it isn't.  Most callers should use :func:`get_providers` for the
    inner list directly.
    """
    return dict(cfg.get("auth", {}))


def get_providers(cfg: Mapping[str, Any]) -> list:
    """Return the list of provider entries, or ``[]`` when no auth is
    configured.  Ergonomic shorthand for ``get_auth(cfg).get("providers", [])``.
    """
    return list(cfg.get("auth", {}).get("providers", []))


def get_notify_keys_file(cfg: Mapping[str, Any]) -> Optional[str]:
    """The path to the run-report signing-key file, or ``None``.

    With this **or** ``notify_route`` absent the listener blueprint is never
    registered, so the route does not exist at any path -- `run-reports.md`
    § 4.3, and `access-control.md` § 8 rule 1: *the safe state is the one
    you get by doing nothing.*
    """
    return cfg.get("notify_keys_file")


def get_notify_route(cfg: Mapping[str, Any]) -> Optional[str]:
    """The listener's generated URL segment, or ``None``.

    Not a secret; it appears in every access log.  What it buys is that a
    scanner sweeping fixed paths finds nothing, and the value never entered
    this repository (`access-control.md` § 8 rule 7).
    """
    return cfg.get("notify_route")


def get_tls(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Return the ``tls`` section, or ``{}``.

    Trivial accessor -- type validity is enforced upstream in
    :func:`_normalise`.  Callers that pass a hand-constructed cfg
    (not via :func:`read_config`) are responsible for its shape.
    """
    return dict(cfg.get("tls", {}))


def get_envs(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Return the ``envs`` section's CATEGORY map, or ``{}``.

    Trivial accessor -- type validity is enforced upstream in
    :func:`_normalise`.  The ``manager`` key is NOT a category: it is
    this machine's package-manager fact (:func:`get_env_manager`), so
    it is excluded here rather than leaking into category iteration.
    """
    out = dict(cfg.get("envs", {}))
    out.pop("manager", None)
    return out


def get_env_manager(cfg: Mapping[str, Any]) -> str:
    """The RECORDED package manager for this machine, or ``""``.

    ``envs.manager`` in ``molbuilder.json`` -- an absolute path to the
    conda-compatible CLI this machine should use (mamba / micromamba /
    conda).  One recorded fact instead of a per-run PATH sniff: on a
    cluster where the manager arrives via ``module load``, the probe's
    PATH answer changes with the shell's module state, which is how
    "the script did not follow the correct pathway" happens (ASU Sol,
    2026-08-21).  Absent means "probe" -- the historical behavior.
    """
    return str(dict(cfg.get("envs", {})).get("manager", "") or "")


def get_admin_emails(cfg: Mapping[str, Any]) -> frozenset:
    """Who may do the things only an operator should do.

    Reads the top-level ``admin`` section::

        "admin": { "emails": ["operator@asu.edu"] }

    **Absent or empty means NOBODY**, and that is the whole point of the shape:
    the state you get by writing no config is the safe one.  Two subsystems ask
    this question -- who may read and clear the rate limiter's block list, and
    who may restart the server everyone is using -- and they get the same
    answer.

    IT LIVED UNDER ``rate_limit.admin_emails`` UNTIL 2026-08-03, where an empty
    list meant "any signed-in user".  That was defensible for reading a block
    list and wrong for stopping a shared process, so the restart route had to
    INVERT it for itself: one value, two opposite readings, depending on which
    subsystem asked.  Worse, it was reached through the limiter's own object,
    so turning the limiter off silently changed who was an admin -- a
    connection nothing in the names would suggest.

    Emails are lowercased and blanks dropped, matching how the auth layer
    stores ``session["user"]["email"]``, so membership is case-stable.
    """
    section = cfg.get("admin") or {}
    if not isinstance(section, Mapping):
        return frozenset()
    raw = section.get("emails") or []
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    return frozenset(
        e.strip().lower() for e in raw
        if isinstance(e, str) and e.strip()
    )


def get_rate_limit(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``rate_limit`` section, or ``{}``.

    The ``rate_limit`` block tunes the IP-based scanner-detection +
    blocklist installed by :mod:`molbuilder.web.rate_limit`.  See
    :file:`docs/ops/deployment.md` for the full schema.
    Trivial accessor -- defaults are applied inside ``RateLimiter``.
    """
    return dict(cfg.get("rate_limit", {}))


# --------------------------------------------------------------------- #
#  checkpoint section (docs/execution/checkpointing.md § 4)             #
# --------------------------------------------------------------------- #


# **Contract:** `execution/checkpointing.md` § 4 -- the classification lives
# here, molbuilder-wide, and NEVER in a calculation folder (S1c).  A per-folder
# copy would let one folder behave differently from another for no recorded
# reason, and would put the classification somewhere a person can edit between a
# save and a restore.
#
# ``size_limit_bytes`` is the whole of the decision § 3's diagram turns on: over
# it a file goes to the archive, under it to git (S1b).  It is a STORAGE
# threshold -- moving it changes where a file is kept, never whether it is kept
# (§ 2.1).
#
# ``engines`` name families that are ALWAYS large, so those skip the measuring.
# That is an effort saving and nothing else: "a hint can make a save faster; it
# can never make it store less."  ``generic`` names none, which is always
# correct and merely stats more.
_CHECKPOINT_SIZE_LIMIT_DEFAULT = 10 * 1024 * 1024        # 10 MB, § 4
_CHECKPOINT_DEFAULTS: Dict[str, Any] = {
    "size_limit_bytes": _CHECKPOINT_SIZE_LIMIT_DEFAULT,
    "engines": {
        "generic": [],
        "siesta":  ["*.DM", "*.HSX", "*.TSHS",
                    "*.TBT.AVTRANS_*", "*.TBT.CC", "*.TBT.DOS"],
        "pyscf":   ["*.chk", "*.cube"],
    },
}


def _validate_checkpoint(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one scope's ``checkpoint`` section (checkpointing.md § 4).

    Returns a normalised copy with defaults filled in.  Raises
    :class:`RuntimeConfigError` on shape errors -- a checkpoint config that is
    wrong in a way nobody notices is a folder saved wrongly, so nothing here is
    coerced or guessed.
    """
    if not isinstance(raw, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'checkpoint' must be an object; got "
            f"{type(raw).__name__}."
        )
    out: Dict[str, Any] = {
        "size_limit_bytes": _CHECKPOINT_DEFAULTS["size_limit_bytes"],
        "engines": {k: list(v)
                    for k, v in _CHECKPOINT_DEFAULTS["engines"].items()},
    }
    if "size_limit_bytes" in raw:
        v = raw["size_limit_bytes"]
        # bool is an int subclass and is never a size.
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'checkpoint.size_limit_bytes' must be a "
                f"positive integer number of bytes; got {v!r}."
            )
        out["size_limit_bytes"] = v
    if "engines" in raw:
        engines = raw["engines"]
        if not isinstance(engines, Mapping):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'checkpoint.engines' must be an object "
                f"mapping an engine name to its always-large patterns; got "
                f"{type(engines).__name__}."
            )
        for name, pats in engines.items():
            if not isinstance(pats, (list, tuple)) or not all(
                    isinstance(x, str) and x.strip() for x in pats):
                raise RuntimeConfigError(
                    f"{CONFIG_FILENAME}: 'checkpoint.engines.{name}' must be a "
                    f"list of non-empty glob strings; got {pats!r}."
                )
            cleaned = [x.strip() for x in pats]
            for pat in cleaned:
                # A FAMILY, matched on the file's name -- never a path.
                #
                # The same string is written verbatim into .gitignore and
                # matched against basenames by the classifier, and a slash makes
                # those two disagree: git honours `runs/*.bin` as a path, the
                # classifier never matches it, and the file ends up gitignored
                # AND unarchived -- in no store at all, which is S1's
                # data-losing branch reached through a config typo.
                if "/" in pat:
                    raise RuntimeConfigError(
                        f"{CONFIG_FILENAME}: "
                        f"'checkpoint.engines.{name}' entry {pat!r} contains "
                        f"'/'.  These name FAMILIES of files (`*.DM`), not "
                        f"paths: git would read the slash as a path while the "
                        f"size check reads only the file's name, and a file "
                        f"they disagree about is stored nowhere "
                        f"(checkpointing.md S1, S1a)."
                    )
            out["engines"][str(name)] = cleaned
    return out


def get_checkpoint_engines() -> list:
    """Every engine entry the effective classification defines (§ 4).

    Exists so a test can be **generated from the configuration** rather than
    hand-written: walk the engines, walk each one's patterns, assert every
    matching file is stored (checkpointing.md § 13.1).  A hand-written list of
    extensions is a second copy of the classification, and it agrees with the
    first until the day it matters -- which is how `*.MD` sat in no store for
    months.
    """
    section = _validate_checkpoint(_read_server_wide().get("checkpoint") or {})
    return sorted(section["engines"])


def get_checkpoint(engine: Optional[str] = None) -> Dict[str, Any]:
    """The effective checkpoint classification (checkpointing.md § 4).

    **Server-wide scope only, and that is the rule rather than an omission.**
    There is deliberately no ``project_dir`` parameter: reading a scope beside
    the folder being saved is exactly the per-folder classification S1c
    forbids, and it is what would let somebody change where files are stored
    between a save and a restore (I2c).  :func:`_read_project` refuses such a
    section outright, so there is no quiet second home either.

    ``engine`` is a **hint** and may be omitted or unknown: an engine nobody
    configured resolves to ``generic``, which names no always-large families and
    therefore measures every file.  That is always correct and merely slower,
    which is the direction this contract errs in -- an unknown engine must never
    make a save store less.

    Returns ``{"size_limit_bytes": int, "always_large": [glob, ...]}``.
    """
    cfg = _read_server_wide()
    section = _validate_checkpoint(cfg.get("checkpoint") or {})
    engines = section["engines"]
    always = engines.get(engine) if engine else None
    if always is None:
        always = engines.get("generic", [])
    return {
        "size_limit_bytes": int(section["size_limit_bytes"]),
        "always_large":     list(always),
    }


# --------------------------------------------------------------------- #
#  script_generation section (docs/execution/running-a-job.md § 5)                     #
# --------------------------------------------------------------------- #


# Two and only two keys -- see docs/execution/running-a-job.md § 5
#
# ``preamble``:   verbatim multi-line bash, default empty.
# ``activation``: how to activate the env.  NO DEFAULT -- the operator
#                 must set it explicitly in at least one scope, OR the
#                 generator refuses to emit a wrapper (per § 2).
_ACTIVATION_FORMS: tuple = ("source activate", "conda activate")
_SCRIPT_GENERATION_DEFAULTS: Dict[str, Any] = {
    "preamble":   "",
    "activation": None,  # explicit-only; no smuggled default
}

# Keys silently dropped at read time (formerly load-bearing; now no-ops
# per the v2 rewrite of docs/execution/running-a-job.md § 5).  A one-time WARNING is logged
# when seen so the operator knows to clean up their config.
_DROPPED_KEYS = ("preactivate_format", "autodetect_conda")
# Keys aliased to the new schema for one release (warning emitted).
_RENAMED_KEYS = {"preactivate": "preamble"}


def _validate_script_generation(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one scope's ``script_generation`` section.

    Returns a normalised copy with defaults filled in.  Raises
    :class:`RuntimeConfigError` on shape errors.  Emits a warning to
    stderr for legacy keys (renamed or dropped) but accepts the file.
    """
    import warnings as _warnings
    if not isinstance(raw, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'script_generation' must be an "
            f"object; got {type(raw).__name__}."
        )
    out = dict(_SCRIPT_GENERATION_DEFAULTS)
    # preamble (new name)
    if "preamble" in raw:
        v = raw["preamble"]
        if not isinstance(v, str):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'script_generation.preamble' "
                f"must be a string (multi-line bash); got "
                f"{type(v).__name__}."
            )
        out["preamble"] = v
    # preactivate (legacy alias) -- warn + accept
    for legacy, current in _RENAMED_KEYS.items():
        if legacy in raw:
            _warnings.warn(
                f"{CONFIG_FILENAME}: 'script_generation.{legacy}' is "
                f"renamed to '{current}' (docs/execution/running-a-job.md § 5).  "
                f"Treating as '{current}' for backward compatibility; "
                f"please update your config.",
                DeprecationWarning,
                stacklevel=2,
            )
            v = raw[legacy]
            if not isinstance(v, str):
                raise RuntimeConfigError(
                    f"{CONFIG_FILENAME}: 'script_generation.{legacy}' "
                    f"must be a string; got {type(v).__name__}."
                )
            # Honour the renamed value only if the NEW key wasn't also set.
            if current not in raw:
                out[current] = v
    # Dropped keys -- warn + silently ignore (no behaviour attached).
    for dropped in _DROPPED_KEYS:
        if dropped in raw:
            _warnings.warn(
                f"{CONFIG_FILENAME}: 'script_generation.{dropped}' is "
                f"no longer used and will be ignored "
                f"(docs/execution/running-a-job.md § 5).  Please remove it.",
                DeprecationWarning,
                stacklevel=2,
            )
    # activation -- no default; ``None`` is the "not set" sentinel.
    # Only reject genuine bad values, not the sentinel (so this
    # validator is idempotent -- the read pipeline normalises the
    # raw file, then get_script_generation may call us again on the
    # already-normalised dict which carries None).
    if "activation" in raw and raw["activation"] is not None:
        v = raw["activation"]
        if v not in _ACTIVATION_FORMS:
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'script_generation.activation' "
                f"must be one of {_ACTIVATION_FORMS!r}; got {v!r}."
            )
        out["activation"] = v
    return out


# --------------------------------------------------------------------- #
#  Multi-scope read/write API (docs/execution/running-a-job.md § 5)                      #
# --------------------------------------------------------------------- #


def _machine_config_file() -> Path:
    """The machine config, in the config directory.

    **The bootstrap caller.**  This is how ``molbuilder.json`` is FOUND, so it
    is the one location that cannot be declared inside ``molbuilder.json`` --
    which is why :func:`molbuilder.config_dir.config_dir` takes its override
    from the ENVIRONMENT (``MOLBUILDER_CONFIG_DIR``) rather than from a config
    key, which would be circular.

    It was called ``_per_user_fallback_path`` while there was a
    working-directory step to fall back FROM; there is not, so the name said
    something untrue about the only location there is.
    """
    return config_dir() / CONFIG_FILENAME


def _read_scope(path: Path) -> Dict[str, Any]:
    """Read one scope's JSON file; ``{}`` if absent.

    Reuses :func:`read_config` for the parse + normalise path; this
    wrapper just hides the missing-file vs read-error distinction
    so the caller can short-circuit on empty.
    """
    if not path.is_file():
        return {}
    return read_config(path)


#: The sections :func:`config_provenance` reports.  A deliberate ALLOWLIST:
#: the machine file also carries ``auth`` / ``tls``,
#: and provenance output lands in terminals, STAGE-PLAN.md and shipped run
#: logs -- material that must never travel there.
#: Derived from the registry -- a section's values may be printed in logs
#: only where its row says so (the one list, U7).
_PROVENANCE_SECTIONS = tuple(
    name for name, spec in _SECTIONS.items() if spec["provenance_safe"])


def machine_config_path() -> Tuple[Path, str]:
    """Which ``molbuilder.json`` the MACHINE scope resolves to, and how —
    ``(path, "config-dir")``.

    Split out 2026-08-17 so a refusal can name the file it is refusing.  This
    two-step lookup was computed inline in :func:`config_provenance` and
    re-derived inside :func:`read_config`, so the display that exists to answer
    *"which file said this"* and the reader that raises about it could describe
    different files.
    """
    # ONE LOCATION (`plans/config-access-plan.md` § 3.3).  A working-directory
    # `molbuilder.json` was step 1 of a first-found-wins search until
    # 2026-08-31, and it is gone: it was redundant with the project scope --
    # `.molbuilder.json`, which MERGES rather than replaces -- and it was the
    # entire source of one setting living in two files with nothing saying
    # which won.  Nothing stops now, because there is nothing to stop at.
    return _machine_config_file().resolve(), "config-dir"


def machine_config_shadow() -> Optional[str]:
    """A warning when a ``./molbuilder.json`` is sitting there UNREAD.

    `configuration.md` § 2.1a.  The machine scope has ONE location, the
    per-user config directory.  A working-directory file is **no longer read
    at all** -- so the danger inverts: it used to win silently, and now it
    loses silently, and a person editing it would watch their changes do
    nothing (user, 2026-08-31: *"I had instances where information are saved in
    two places and I did not realize which one was the effective one"*).

    THE PHRASING LIVES HERE, in one place, so every surface says the same
    thing.

    Returns ``None`` when there is no such file -- which is the normal case,
    and a message then would be noise on every invocation.
    """
    cwd_path = Path(CONFIG_FILENAME)
    if not cwd_path.is_file():
        return None
    here = cwd_path.resolve()
    # ASKED, not re-derived: a second copy of the resolution is the split-brain
    # this whole change removes, and it would go unnoticed because both answers
    # agree today.
    home = machine_config_path()[0]
    return "\n".join([
        f"{CONFIG_FILENAME} in the working directory is NOT READ: {here}",
        f"  The machine config has one location, and this is not it: {home}"
        + ("" if home.is_file() else "  (no file there yet)"),
        "  Move it there, or delete it.  For settings that should apply to "
        "one project only, use that project's .molbuilder.json, which merges "
        "(configuration.md § 2.1a).",
    ])


#: The only modes a file holding credentials may carry, and the directory that
#: names it (`configuration.md` § 2.1b).  `0600` and `0700`: owner only.
CONFIG_FILE_MODE = 0o600
CONFIG_DIR_MODE = 0o700


def machine_config_mode_warning() -> Optional[str]:
    """A warning when the machine config is readable by anyone but its owner.

    `configuration.md` § 2.1b.  This file carries `tls.key` and the
    `auth.providers` block, so a world-readable copy on a shared login node is
    an exposure rather than an untidiness.

    WRITING IT TIGHTLY WAS ALREADY HANDLED -- ``auth_setup`` opens with
    ``0o600`` and ``fchmod``s before the first byte, so the mode is right
    before there is anything to read.  What no writer can control is a file
    that ARRIVES loose: copied from another machine, restored from a backup,
    made by an editor, or unpacked from an archive that dropped its modes.
    Those never pass through the careful path, so the check belongs on the way
    IN.

    A warning and never a refusal (§ 2.1a's reasoning): the fix is one command
    and it is named here.  ``None`` when the mode is already tight, and when
    the file does not exist -- there is nothing to say about a file nobody
    wrote.
    """
    path, _via = machine_config_path()
    try:
        if not path.is_file():
            return None
        mode = path.stat().st_mode & 0o777
    except OSError:
        return None
    loose = mode & ~CONFIG_FILE_MODE
    if not loose:
        return None
    who = []
    if mode & 0o077 & 0o070:
        who.append("your group")
    if mode & 0o007:
        who.append("everyone on this machine")
    reach = " and ".join(who) if who else "more than its owner"
    return (f"{path} is mode {mode:04o}, so {reach} can read it. It holds "
            f"private-key paths and provider credentials "
            f"(configuration.md § 2.1b).\n"
            f"  Fix it with: chmod {CONFIG_FILE_MODE:04o} {path}")


def config_provenance(project_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Which config files this process consults, and which one supplied each
    execution-relevant value — the answer to *"where did that setting come
    from?"* at the moment it takes effect (user request, 2026-08-12: the
    inert-fixture bug class is invisible without it).

    Safe for logs **by construction**: paths, presence, and the effective
    values of :data:`_PROVENANCE_SECTIONS` plus the scheduler's routing
    domain names — never the file contents (see the allowlist note above).

    Returns ``{"sources": [...], "effective": {...}, "domains": [...]}``:
    ``sources`` lists each scope as ``{scope, path, found}`` in precedence
    order (project last = wins); ``effective`` maps ``section.key`` to
    ``{"value": ..., "from": "machine"|"project"}``.

    **The scope is called ``project`` everywhere** -- the registry's
    ``_SECTIONS[...]["scopes"]``, this output, and the refusals below.  It
    was ``"bundle"`` here and ``"project"`` in the registry until
    2026-08-23, three names for one thing counting the refusals' prose
    (`configuration.md` § 8).  ``bundle`` was the one that had to go: it
    already names a different artifact -- the portable prepped directory
    the JobSet framework's ``--bundle`` points at.
    """
    machine_path, machine_via = machine_config_path()
    sources = [{"scope": "machine", "path": str(machine_path.resolve()),
                "found": machine_path.is_file(), "via": machine_via}]
    # WHAT THIS SCOPE IS STANDING IN FRONT OF (§ 2.1a).  The row above says
    # which file was reached; it cannot say that another one exists and was
    # skipped, and that is the state where a setting is written twice and read
    # once.  Asked of the one place that phrases it, never re-worded here.
    shadow = machine_config_shadow()
    # Same split as § 2.1a's: WHICH file, and whether it is safe to hold what
    # it holds.  Asked of the one place that phrases each, never re-worded.
    mode_warning = machine_config_mode_warning()

    if project_dir is not None:
        project_path = Path(project_dir) / PROJECT_CONFIG_FILENAME
        sources.append({"scope": "project", "path": str(project_path),
                        "found": project_path.is_file(), "via": "project"})

    # RAW file bytes decide what a file "supplied" (R10, 2026-08-12: the
    # normalized scopes injected validator defaults, and provenance then
    # showed them as file-supplied values -- the display existing to
    # answer "which file said this" answered it about keys no file
    # said).
    def _raw_file(path: Path) -> Dict[str, Any]:
        try:
            obj = json.loads(path.read_text())
            return obj if isinstance(obj, dict) else {}
        except (OSError, ValueError):
            return {}

    machine_file = _raw_file(machine_path)
    project_file = (_raw_file(Path(project_dir) / PROJECT_CONFIG_FILENAME)
                    if project_dir is not None else {})
    effective: Dict[str, Dict[str, Any]] = {}
    for section in _PROVENANCE_SECTIONS:
        for scope_name, raw in (("machine", machine_file),
                                ("project", project_file)):
            block = raw.get(section)
            if not isinstance(block, Mapping):
                continue
            for key, value in block.items():
                # later scope overwrites: project wins, mirroring _deep_merge
                effective[f"{section}.{key}"] = {"value": value,
                                                 "from": scope_name}
    # ONE exception, asked of its owner rather than re-derived (R10):
    # script_generation.preamble does not merge project-wins -- it
    # CONCATENATES server-then-project (get_script_generation's bespoke
    # rule), and showing one scope as the source misreported the other
    # half away.
    if "script_generation.preamble" in effective:
        try:
            chunks = get_script_generation(
                project_dir=project_dir)["preamble_chunks"] or []
            if len(chunks) > 1:
                effective["script_generation.preamble"] = {
                    "value": " + ".join(t for _sc, t in chunks),
                    "from": "+".join(_sc for _sc, _t in chunks)
                            + " (concatenated)"}
        except Exception:
            pass                      # display must never break a prep

    # Domains come from the MACHINE RECORD since N4, not from these files, so
    # provenance follows them there -- a display that kept reporting the old
    # home would say "(none)" on a correctly-probed cluster.  The record's own
    # scopes join `sources`, because "which file supplied this" is the question
    # this function exists to answer and environment.json now answers part of
    # it (`configuration.md` § 5, M-3).
    from .scheduler import FILENAME as ENV_FILENAME
    from .scheduler import machine_for, machine_scope_path
    env_machine = machine_scope_path()
    env_scopes = ([(Path(project_dir) / ENV_FILENAME, "calculation")]
                  if project_dir is not None else [])
    env_scopes.append((env_machine, "machine"))
    for path, via in env_scopes:
        sources.append({"scope": "environment", "path": str(path),
                        "found": path.is_file(), "via": via})
    # Through `get_routing`, NOT a second resolution.  This read the probed
    # record directly and so reported "no domains" on a workstation whose
    # config declared two -- a display whose whole job is to say where a value
    # came from, disagreeing with the reader that actually answers it.  One
    # question, one function.
    try:
        domains = [d.name for d in get_routing(project_dir=project_dir)]
    except RuntimeConfigError:
        domains = []               # a malformed block is the caller's to raise
    return {"sources": sources, "effective": effective, "domains": domains,
            "shadow": shadow, "mode_warning": mode_warning}


def format_provenance(prov: Mapping[str, Any]) -> str:
    """The ONE rendering of :func:`config_provenance` — the CLI echo and
    STAGE-PLAN.md both use it, so they cannot drift."""
    lines = ["config:"]
    # Width from the WIDEST scope name present, not a literal: "environment"
    # (11 chars) arrived on 2026-08-17 and ran straight into the hardcoded 8,
    # printing `environment/home/...` with no gap -- a display whose whole job
    # is to make the source legible.
    width = max([len(s["scope"]) for s in prov["sources"]] + [8]) + 1
    for s in prov["sources"]:
        state = "found" if s["found"] else "absent"
        via = f", via {s['via']}" if s["found"] and s["via"] != "project" else ""
        lines.append(f"  {s['scope']:<{width}}{s['path']}  ({state}{via})")
    for key in sorted(prov["effective"]):
        e = prov["effective"][key]
        lines.append(f"  {key} = {e['value']!r}   <- {e['from']}")
    if prov["domains"]:
        # Named for where they LIVE.  This said "scheduler.routing domains"
        # until N4 moved them out of the scheduler block, and a label pointing
        # at a key that no longer exists is worse than no label.
        lines.append(f"  environment.domains: "
                     f"{', '.join(prov['domains'])}")
    return "\n".join(lines)


def _read_server_wide() -> Dict[str, Any]:
    """The machine config, from the one place it lives
    (`configuration.md` § 2.1c).

    Since 2026-08-13 this IS :func:`read_config`'s own default lookup
    (A-7 closed the split-brain where serve/TLS/auth read cwd-only while
    these getters fell back to XDG); the alias stays because the section
    getters read better naming the SCOPE than the mechanism.
    """
    return read_config()


def _read_project(project_dir: Path) -> Dict[str, Any]:
    """One project-scope file, refusing the one section that may not live here.

    S1c: the checkpoint classification has ONE home, and it is the server-wide
    config.  A project-scope copy is a file somebody can edit between a save and
    a restore, and it makes two folders behave differently with nothing on disk
    explaining why (checkpointing.md § 4, I2c).

    Refused rather than ignored: a section that is read, validated and then
    silently dropped is worse than one that was never allowed -- it looks
    effective, and the folder is saved under rules nobody applied.
    """
    scope = _read_scope(Path(project_dir) / PROJECT_CONFIG_FILENAME)
    if "checkpoint" in scope:
        # The registry says machine-only too, but checkpoint keeps its own
        # message: S1c is the section-specific WHY, and the operator
        # reading this refusal is mid-mistake about exactly that.
        raise RuntimeConfigError(
            f"{Path(project_dir) / PROJECT_CONFIG_FILENAME}: a 'checkpoint' "
            f"section may not live in a PROJECT-scope file "
            f"({PROJECT_CONFIG_FILENAME}, in a project or calculation "
            f"folder).  The "
            f"classification has one home -- the server-wide "
            f"{CONFIG_FILENAME} -- so that two folders cannot behave "
            f"differently for no recorded reason, and so that nobody can "
            f"change where files are stored between a save and a restore "
            f"(docs/execution/checkpointing.md S1c, I2c)."
        )
    misplaced = sorted(
        k for k in scope
        if k in _SECTIONS and "project" not in _SECTIONS[k]["scopes"])
    if misplaced:
        allowed = ", ".join(n for n, spec in _SECTIONS.items()
                            if "project" in spec["scopes"])
        raise RuntimeConfigError(
            f"{Path(project_dir) / PROJECT_CONFIG_FILENAME}: "
            f"{', '.join(map(repr, misplaced))} may not live in a "
            f"PROJECT-scope file ({PROJECT_CONFIG_FILENAME}, in a project "
            f"or calculation folder) -- machine sections have one home, the "
            f"server-wide {CONFIG_FILENAME}.  A project file may carry: "
            f"{allowed}.  (Refused rather than ignored: a section that is "
            f"read, validated and then silently dropped looks effective "
            f"while nobody applied it.)"
        )
    return scope


def _deep_merge(base: Dict[str, Any],
                 overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Per docs/execution/running-a-job.md § 5:
       * scalars: overlay replaces base
       * objects: recurse
       * arrays:  overlay replaces base (no element-wise merge)

    Side-effect-free: returns a new dict; neither input is mutated.
    """
    out = dict(base)
    for k, v in overlay.items():
        if (k in out and isinstance(out[k], dict)
                and isinstance(v, dict)):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_effective_config(
    project_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return the merged effective configuration.

    When ``project_dir`` is None: returns the server-wide layer alone.
    When provided: deep-merges server-wide ← project (project wins per
    the rules in :func:`_deep_merge`).

    NOTE: the merge here is the GENERIC merge.  Subsystems with
    field-specific merge rules (like ``script_generation.preactivate``,
    which concatenates rather than replaces) must use their dedicated
    getter -- e.g. :func:`get_script_generation` reads both raw scopes
    and concatenates ``preactivate``, ignoring the generic merge.
    Other ``script_generation`` fields (``autodetect_conda``,
    ``preactivate_format``) use the standard replace rule.
    """
    server = _read_server_wide()
    if project_dir is None:
        return server
    project = _read_project(Path(project_dir))
    return _deep_merge(server, project)


def get_script_generation(
    project_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return the effective ``script_generation`` section.

    Per docs/execution/running-a-job.md § 5 + § 4:
      * ``preamble``: server-wide + project concatenated (server
        first), joined by ``"\\n"``.
      * ``activation``: project wins if set; else server-wide if set;
        else None.  The generator is responsible for refusing to emit
        a wrapper when ``activation`` is None
        (:func:`require_activation`).

    Returns:
        {
            "preamble":    "<concatenated lines>",  # may be empty
            "activation":  "source activate" | "conda activate" | None,
            "_preamble_scopes": ["server", "project"] subset,
        }
    """
    server_raw = _read_server_wide().get("script_generation") or {}
    project_raw: Dict[str, Any] = {}
    if project_dir is not None:
        project_raw = _read_project(Path(project_dir)).get(
            "script_generation") or {}

    # Normalise both scopes through the validator (catches type errors
    # and applies the preactivate -> preamble alias).
    server   = _validate_script_generation(server_raw) if server_raw else dict(
        _SCRIPT_GENERATION_DEFAULTS)
    project  = _validate_script_generation(project_raw) if project_raw else dict(
        _SCRIPT_GENERATION_DEFAULTS)

    # Per-scope preamble chunks (server first, then project).  Empty
    # strings drop out so the renderer can emit per-scope sentinel
    # blocks without conditional logic.  ``preamble_chunks`` is the
    # API the renderer uses; ``preamble`` (joined) is the
    # convenience field for callers that just want the merged text.
    chunks: List[Tuple[str, str]] = []
    for label, src in (("server", server), ("project", project)):
        text = (src.get("preamble") or "").rstrip("\n")
        if text:
            chunks.append((label, text))
    preamble = "\n".join(c[1] for c in chunks)

    # activation: project wins; else server; else None (no default).
    activation: Optional[str] = (
        project.get("activation")
        if project.get("activation") is not None
        else server.get("activation")
    )

    return {
        "preamble":         preamble,
        "preamble_chunks":  chunks,        # list of (scope, text)
        "activation":       activation,
    }


def require_activation(project_dir: Optional[Path] = None) -> str:
    """Return the effective ``activation`` value, or raise.

    Per docs/execution/running-a-job.md § 5 (refuse-to-emit rule): the generator must
    refuse to emit a wrapper if ``script_generation.activation`` isn't
    set in either scope.  Use this helper at every wrapper-render
    entry point so the error message + doc reference are consistent.

    Raises :class:`RuntimeConfigError` with an operator-facing message
    when the key is missing.
    """
    sg = get_script_generation(project_dir=project_dir)
    if sg["activation"] is None:
        raise RuntimeConfigError(
            "script_generation.activation is not set in molbuilder.json "
            "(or .molbuilder.json).  The wrapper generator refuses to "
            "emit a script that can't activate its conda env.\n"
            "\n"
            "Fix: add to molbuilder.json (server-wide):\n"
            '    {\n'
            '      "script_generation": {\n'
            '        "preamble": "module load mamba",\n'
            '        "activation": "source activate"\n'
            '      }\n'
            '    }\n'
            "\n"
            "Use ``conda activate`` if your conda hook is sourced "
            "(typical for local dev installs).  Use ``source "
            "activate`` for HPC clusters where ``module load mamba`` "
            "is the toolchain.  See docs/execution/running-a-job.md § 5"
        )
    return sg["activation"]


# detect_conda_activation was deleted 2026-08-13 (V22): zero callers
# anywhere -- activation is DECLARED (script_generation.activation),
# never detected, per running-a-job.md § 5.


# --------------------------------------------------------------------- #
#  scheduler section (docs/execution/job-system.md)          #
# --------------------------------------------------------------------- #


# Supported scheduler kinds.  Only SLURM today; PBS/local are future.
_SCHEDULER_KINDS: tuple = ("slurm",)

# ``directives`` keys we recognise (stable site #SBATCH header values).
# Unknown keys are accepted verbatim (forward-compat) but these are the
# ones the emitter maps to canonical flags.
_SCHEDULER_DIRECTIVE_KEYS: tuple = (
    "partition", "qos", "mail_type", "mail_user", "export",
)

# Per-job defaults (running-a-job.md § 5.3, § 6).  ``time`` is a
# walltime string; ``cpus_per_task`` an int (OMP width per rank); ``mem``
# a string like "120G" or None (=> scheduler default).
_SCHEDULER_DEFAULT_KEYS: tuple = ("time", "cpus_per_task", "mem")


def _validate_scheduler(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate + normalise the ``scheduler`` block of one merged config.

    Returns the resolved ``{kind, directives, gpu, defaults}`` dict.
    Raises :class:`RuntimeConfigError` on shape errors AND on the
    refuse-to-emit rule (running-a-job.md § 5.3): a ``slurm`` site
    that omits ``directives.partition`` or ``directives.qos`` cannot
    produce a header that will allocate, so we fail at generate time
    while the user is at a terminal -- never after a job has queued.
    """
    if not isinstance(raw, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler' must be an object; got "
            f"{type(raw).__name__}."
        )

    kind = raw.get("kind", "slurm")
    if kind not in _SCHEDULER_KINDS:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler.kind' must be one of "
            f"{_SCHEDULER_KINDS!r}; got {kind!r}."
        )

    def _as_obj(key: str) -> Dict[str, Any]:
        v = raw.get(key, {})
        if v is None:
            return {}
        if not isinstance(v, Mapping):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler.{key}' must be an object; "
                f"got {type(v).__name__}."
            )
        return dict(v)

    directives = _as_obj("directives")
    gpu        = _as_obj("gpu")
    defaults   = _as_obj("defaults")

    # WHICH AXIS DECIDES between queues that all fit (2026-08-23, user).
    # A PREFERENCE, so it lives here and never in the machine record (M-1):
    # the record measures what the queues offer, this says which of those
    # facts matters most at this site.  Default in `place.PRIORITY_DEFAULT`.
    #
    # Refused when it names something placement cannot order by -- a
    # preference that is silently dropped looks honoured and is not.
    priority = raw.get("placement_priority")
    if priority is not None:
        if not isinstance(priority, (list, tuple)):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler.placement_priority' must be "
                f"a list of axis names; got {type(priority).__name__}.")
        from .scheduler.place import check_priority
        try:
            priority = list(check_priority(priority))
        except ValueError as e:
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler.placement_priority': {e}")

    # Refuse-to-emit: slurm needs a partition + qos (§ 10).
    for required in ("partition", "qos"):
        val = directives.get(required)
        if not (isinstance(val, str) and val.strip()):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler.directives.{required}' is "
                f"required for a slurm site but is missing/empty.  The "
                f".sbatch generator refuses to emit a header that won't "
                f"allocate.\n"
                f"\n"
                f"Fix: add to molbuilder.json (see the asu-sol preset in "
                f"docs/ops/examples/molbuilder.asu-sol.json):\n"
                f'    {{\n'
                f'      "scheduler": {{\n'
                f'        "kind": "slurm",\n'
                f'        "directives": {{"partition": "public", '
                f'"qos": "public"}}\n'
                f'      }}\n'
                f'    }}\n'
                f"\n"
                f"On ASU Sol use partition/qos \"public\" (the \"general\" "
                f"partition went private in May 2026).  See "
                f"docs/execution/job-system.md, § 7.0, § 10."
            )

    # String-typed directives must actually be strings (catch e.g. a
    # numeric partition).  Unknown keys pass through untouched.
    for k in _SCHEDULER_DIRECTIVE_KEYS:
        if k in directives and not isinstance(directives[k], str):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler.directives.{k}' must be a "
                f"string; got {type(directives[k]).__name__}."
            )

    # gpu block: partition/default_type strings, exclusive bool.
    for k in ("partition", "default_type"):
        if k in gpu and gpu[k] is not None and not isinstance(gpu[k], str):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler.gpu.{k}' must be a string; "
                f"got {type(gpu[k]).__name__}."
            )
    if "exclusive" in gpu and not isinstance(gpu["exclusive"], bool):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler.gpu.exclusive' must be a "
            f"boolean; got {type(gpu['exclusive']).__name__}."
        )
    # gpu.mem: GPU-specific memory default (string or null).  GPU nodes
    # typically have less RAM per rank than CPU nodes (e.g. 24 GB/GPU vs
    # 2 TB/node), so a single defaults.mem can't cover both.  GPU jobs
    # use gpu.mem when set; CPU jobs use defaults.mem.
    if "mem" in gpu and gpu["mem"] is not None \
            and not isinstance(gpu["mem"], str):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler.gpu.mem' must be a string "
            f"(e.g. \"24G\") or null; got {type(gpu['mem']).__name__}."
        )
    # ``gpu.mem_cap_per_gpu`` was VALIDATED here and read by nothing once
    # the memory clamp was deleted (2026-08-24) -- a key a person could
    # set, that this file accepted, and that changed nothing.  Removed
    # rather than left standing: an accepted setting with no effect is
    # worse than a refused one.  A GPU job's default is ``gpu.mem``.

    # defaults: time str|None, cpus_per_task int|None, mem str|None.
    if "time" in defaults and defaults["time"] is not None \
            and not isinstance(defaults["time"], str):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler.defaults.time' must be a "
            f"string (e.g. \"0-04:00:00\") or null; got "
            f"{type(defaults['time']).__name__}."
        )
    if "cpus_per_task" in defaults and defaults["cpus_per_task"] is not None \
            and not isinstance(defaults["cpus_per_task"], int):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler.defaults.cpus_per_task' must be "
            f"an integer or null; got "
            f"{type(defaults['cpus_per_task']).__name__}."
        )
    if "mem" in defaults and defaults["mem"] is not None \
            and not isinstance(defaults["mem"], str):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler.defaults.mem' must be a string "
            f"(e.g. \"120G\") or null; got {type(defaults['mem']).__name__}."
        )

    out = {
        "kind":       kind,
        "directives": directives,
        "gpu":        gpu,
        "defaults":   defaults,
    }
    # ABSENT when unset, so a reader can tell "this site did not choose" from
    # "this site chose the default" -- `place` supplies its own default and
    # says so, rather than the config pretending to have made a decision.
    if priority is not None:
        out["placement_priority"] = priority
    # routing: REFUSED here since 2026-08-17 (N4).  It used to pass through to
    # get_routing, which owned the domain schema.  A domain is what `sinfo` and
    # `sacctmgr` measured, so it belongs in the machine record, and a probe no
    # longer writes into a person's config file (`configuration.md` § 5, M-1).
    # Refused rather than ignored, for this file's own stated reason: a section
    # read, validated and then silently dropped looks effective while nobody
    # applied it -- and a stale hand-written menu is exactly the case where
    # "looks effective" gets a job rejected by the scheduler.
    # routing rides through verbatim: DECLARED capability (`_declared_routing`),
    # which `get_routing` uses when nothing has been probed.  It was refused
    # here for part of 2026-08-17, on a rule that made the workstation-
    # describing-a-cluster case an error -- the one case that must declare.
    if raw.get("routing") is not None:
        out["routing"] = raw["routing"]
    return out


def _declared_routing(project_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """``scheduler.routing`` read as **declared capability**, not as an error.

    *(Corrected 2026-08-17, same day, after the user pointed at the machine
    this actually runs on.)*  N4 refused this key outright, on the rule
    "domains are PROBED, not declared".  That rule sorted by the wrong axis.

    **You can only probe the machine you are standing on.** A person
    describing a calculation on a workstation, to run on a cluster, cannot
    probe the cluster -- so they write its partitions and walls down by hand,
    and those rows are *facts*, merely declared ones rather than detected
    ones.  Refusing them made the one case that NEEDS declaring an error, and
    bricked `prep` on a workstation over a block describing a machine
    elsewhere.

    The axis is **fact vs preference**, not probed vs chosen.  This module
    already knew that and I did not read it: ``Environment.source``'s
    vocabulary is ``scontrol`` / ``lscpu`` / **``flag``**, and ``flag`` is the
    declared case; ``resolve_environment(overrides=...)`` is its door.

    Probed still wins where both exist -- standing on the machine beats a
    hand-written note about it -- which is why this is a FALLBACK.
    """
    out: List[Dict[str, Any]] = []
    scopes = [_read_server_wide().get("scheduler")]
    if project_dir is not None:
        scopes.append(_read_project(Path(project_dir)).get("scheduler"))
    for raw in scopes:
        if not isinstance(raw, Mapping):
            continue
        rows = raw.get("routing")
        if isinstance(rows, list):
            # Rows ride through WHOLE.  An operator's own columns
            # (`node_types`, `max_cores`, `max_mem_gb`, `gpu{}`) are the point
            # of declaring -- R10, 2026-08-12: rebuilding a row from a
            # known-key list made drafting a column indistinguishable from
            # not writing one.  A reader owns only the keys it checks.
            out = [dict(r) for r in rows if isinstance(r, Mapping)]
    return out


def get_paths() -> Dict[str, Any]:
    """The effective ``paths`` block, or ``{}``.  See :func:`_read_paths`."""
    raw = _read_server_wide()
    try:
        return dict(_read_paths(raw) or {})
    except RuntimeConfigError:
        raise


def get_scheduler(
    project_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the effective ``scheduler`` block, or ``None`` if unset.

    Mirrors :func:`get_script_generation`'s lifecycle (read at generate
    time only).  Server-wide and project scopes are deep-merged (the
    generic ``_deep_merge`` rule -- objects recurse, scalars/arrays
    replace; project wins), then validated.

    Returns ``None`` when neither scope defines a ``scheduler`` block --
    the signal to emit only ``.run.sh`` (today's behaviour) and skip the
    ``.sbatch`` (running-a-job.md § 5.3).  When a block IS present it
    is validated strictly, so a malformed/partial site config raises
    here rather than producing a header that won't allocate.

    Returns:
        {
            "kind":       "slurm",
            "directives": {partition, qos, mail_type, mail_user, export, ...},
            "gpu":        {partition, default_type, exclusive, ...},
            "defaults":   {time, cpus_per_task, mem},
        }
        or None.
    """
    server_raw  = _read_server_wide().get("scheduler")
    project_raw: Optional[Mapping[str, Any]] = None
    project_path = None
    if project_dir is not None:
        project_path = Path(project_dir) / PROJECT_CONFIG_FILENAME
        project_raw = _read_project(Path(project_dir)).get("scheduler")

    if server_raw is None and project_raw is None:
        return None

    merged: Dict[str, Any] = {}
    if isinstance(server_raw, Mapping):
        merged = _deep_merge(merged, dict(server_raw))
    elif server_raw is not None:
        # A non-object scheduler at server scope is a hard error.
        merged = _validate_scheduler(server_raw)  # raises with the message
    if isinstance(project_raw, Mapping):
        merged = _deep_merge(merged, dict(project_raw))
    elif project_raw is not None:
        _validate_scheduler(project_raw)  # raises

    # Name the FILE the block came from.  `_validate_scheduler` sees only the
    # MERGED mapping, so every refusal it raises -- a bad `kind`, a missing
    # directive -- could say no more than the generic "molbuilder.json", which
    # names two possible files (machine, project) and answers neither.  R10
    # fixed exactly this for `read_config` on 2026-08-12 and the scheduler
    # getter was never given the same treatment.
    #
    # Where ONE scope defines the block we can pin it exactly; where both do,
    # both are listed rather than one guessed at.
    contributors = [str(path) for path, raw in
                    ((machine_config_path()[0], server_raw),
                     (project_path, project_raw))
                    if isinstance(raw, Mapping)]
    try:
        out = _validate_scheduler(merged)
    except RuntimeConfigError as exc:
        msg = str(exc)
        if contributors and not any(c in msg for c in contributors):
            where = contributors[0] if len(contributors) == 1 else \
                " + ".join(contributors)
            raise RuntimeConfigError(f"{where}: {msg}") from None
        raise

    # gpu.default_type: the PROBED answer is the default, the configured one is
    # an override (N4, 2026-08-17).  Which card exists here is a measurement
    # (`topology.gpu_type`); which card you want is a choice, and a site that
    # wants the a30 rather than the a100 still says so in this file.  Before
    # this, two files each held a probed GPU type -- `topology.gpu_type` and
    # `scheduler.gpu.default_type` -- and only the first reached the code that
    # sizes a run (`configuration.md` § 5, M-1).
    if not (out.get("gpu") or {}).get("default_type"):
        from .scheduler import machine_for
        env = machine_for(project_dir)
        probed = getattr(getattr(env, "topology", None), "gpu_type", None)
        if probed:
            out.setdefault("gpu", {})["default_type"] = probed
    return out


def get_execution(
    project_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return the effective ``execution`` block: the run-vs-submit launch
    policy, read at prep time on the target.  The nearest live contract is
    `running-a-job.md` § 5.4 — the section this block's retired home
    (*job-execution.md § 8.13*, 2026-07 migration) maps to; the key's own
    full contract is still to be written (`job-system.md` § "The loop").

    Server-wide and project scopes are deep-merged (project wins).

    Returns:
        {
            "mode":       "direct" | "submit" | None,   # None -> UNSET: the
                          # caller refuses or asks.  Never derived from the
                          # detected scheduler -- § 5.4: the mode, not the
                          # scheduler, gates submission
            "submit_via": "slurm",                       # backend when submit
        }

    ``mode`` is the source of truth for HOW a benchmark point is launched,
    decoupled from the DETECTED scheduler (which still drives topology
    detection): you can be *on* slurm yet launch ``direct``, or force
    ``launch`` from an interactive shell.  A malformed ``mode`` raises.
    """
    server_raw = _read_server_wide().get("execution") or {}
    project_raw: Dict[str, Any] = {}
    if project_dir is not None:
        project_raw = _read_project(Path(project_dir)).get("execution") or {}

    merged: Dict[str, Any] = {}
    if isinstance(server_raw, Mapping):
        merged = _deep_merge(merged, dict(server_raw))
    if isinstance(project_raw, Mapping):
        merged = _deep_merge(merged, dict(project_raw))

    mode = merged.get("mode")
    if mode is not None and mode not in ("direct", "submit"):
        raise RuntimeConfigError(
            f'execution.mode must be "direct" or "submit"; got {mode!r}.\n'
            "Fix it in .molbuilder.json (running-a-job.md § 5.4).")
    submit_via = merged.get("submit_via", "slurm")
    domain = merged.get("domain")
    if domain is not None and not isinstance(domain, str):
        raise RuntimeConfigError(
            f"execution.domain must be a string (a scheduler.routing name); "
            f"got {type(domain).__name__} (running-a-job.md § 5.4).")
    return {"mode": mode, "submit_via": submit_via, "domain": domain}


def get_routing(
    project_dir: Optional[Path] = None,
    *,
    local_only: bool = False,
) -> List["Domain"]:
    """Return the submission-domain menu: every ``(partition, qos)`` this
    account may actually reach, with its wall.

    **Sourced from ``environment.json``, not from this file** (N4,
    2026-08-17).  A domain is a MEASUREMENT -- ``jobset probe`` reads it from
    live ``sinfo``/``sacctmgr`` -- and `configuration.md` § 5 M-1 puts
    measurements in the machine record and preferences in ``molbuilder.json``.
    It lived under ``scheduler.routing`` here until the prober stopped writing
    into a person's config file.

    Each entry is a :class:`~molbuilder.scheduler.record.Domain` -- **typed
    since 2026-08-23**, phase 3 of `execution/scheduler.md` § 8.  This function
    always built them and then flattened them with ``to_row()`` on its last
    line, so every caller reached for ``row.max_time`` against a plain
    dict and nothing could tell a real column from a typo.  That is how
    ``gpu_partition`` came to redirect GPU work from inside ``extra``, the bag
    the record documents as uninterpreted.

    Returns ``[]`` when there is no record, or on a workstation, which is the
    same signal it always was: no named menu, so the rendered header's default
    directives stand.
    Order is preserved (cheapest ceiling -> most general); the FIRST fitting
    domain is the recommendation.

    ``project_dir`` selects the calculation scope, so a folder carried to a
    cluster reads the record `prep` snapshotted beside it (M-3's precedence,
    through the one door).

    ``local_only`` bypasses ``project_dir`` and asks a different question --
    :func:`molbuilder.scheduler.record.machine_for`'s ``local_only`` docstring
    has the reasoning.  The declared-routing fallback below still applies:
    this machine's own DECLARED menu counts as "what this machine knows"
    exactly as much as a probed one does.
    """
    from .scheduler import Domain, machine_for
    env = machine_for(project_dir, local_only=local_only)
    domains = list(env.domains) if env is not None else []
    if not domains:
        # No probed domains: either a workstation (there are none to have) or a
        # cluster nobody has run `jobset probe` on yet.  Fall back to what was
        # DECLARED -- the only source available when the target is not this
        # machine.
        domains = [d for d in (Domain.from_row(r)
                               for r in _declared_routing(project_dir))
                   if d is not None]
    # ONE shape, whichever branch ran.  Until 2026-08-17 the probed branch
    # emitted a 4-key mapping built here by hand and the declared branch passed
    # its rows through whole, so a caller got 4 keys or 6 depending on which
    # source answered -- from the same function.
    return domains


def write_config_scope(
    project_dir: Optional[Path],
    patch: Mapping[str, Any],
) -> Path:
    """Write a partial config patch into one scope.

    ``project_dir`` selects:
      * ``None``: the machine config, which has one location and is
        created there when absent (`configuration.md` § 2.1c).
      * a path: ``<project_dir>/.molbuilder.json``.

    The patch is deep-merged ONTO the existing file's contents (per
    :func:`_deep_merge`), preserving keys outside the patch.  A corrupt
    existing file REFUSES rather than being overwritten (R10,
    2026-08-12 -- the documented 'log nothing, overwrite' destroyed
    whatever a hand-edit broke).  Files are written atomically
    (persist.write_bytes) at mode 0600, matching
    :mod:`molbuilder.auth_setup`'s precedent -- a config file may carry
    secret-file PATHS, deploy context, or per-cluster setup commands
    that aren't meant for casual inspection.

    Returns the resolved target path.
    """
    if project_dir is None:
        # THE SAME DOOR THE READER USES.  A writer with its own idea of where
        # the machine file lives writes one nothing reads, which is the whole
        # failure this change removes.
        target = machine_config_path()[0]
    else:
        # The same scope rule reads enforce (the registry): refusing at
        # WRITE time beats producing a file every later read refuses.
        misplaced = sorted(
            k for k in patch
            if k in _SECTIONS and "project" not in _SECTIONS[k]["scopes"])
        if misplaced:
            raise RuntimeConfigError(
                f"{', '.join(map(repr, misplaced))} may not be written into "
                f"a project-scope {PROJECT_CONFIG_FILENAME}: machine "
                f"sections have one home, the server-wide {CONFIG_FILENAME}."
            )
        target = Path(project_dir) / PROJECT_CONFIG_FILENAME

    existing: Dict[str, Any] = {}
    if target.is_file():
        try:
            existing = json.loads(target.read_text())
            if not isinstance(existing, dict):
                raise RuntimeConfigError(
                    f"{target}: exists but is not a JSON object -- refusing "
                    f"to merge a patch over it.  Fix or remove the file "
                    f"first.")
        except (OSError, json.JSONDecodeError) as exc:
            # REFUSED, not overwritten (R10, 2026-08-12: 'log nothing,
            # overwrite' silently destroyed whatever a hand-edit broke --
            # a config carrying auth providers and TLS paths is exactly
            # the file a user cannot afford to lose to a typo).
            raise RuntimeConfigError(
                f"{target}: unreadable ({exc}) -- refusing to overwrite a "
                f"corrupt config.  Fix the JSON (or move the file aside) "
                f"and retry.") from exc

    merged = _deep_merge(existing, dict(patch))
    # Round-trip through the validator BEFORE writing so we never
    # produce a file that ``read_config`` would reject.
    try:
        _normalise(merged)
    except RuntimeConfigError:
        # The PATCH was invalid; surface the error untouched.
        raise

    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(merged, indent=2, sort_keys=False) + "\n"
    # Through the ONE atomic writer (U8's shape; R10 aligned this last
    # in-place O_TRUNC write with it -- a crash mid-write left a
    # truncated config for every later read to refuse).  chmod after:
    # a config may carry secret-file paths and deploy context.
    from .persist import write_bytes
    write_bytes(target, rendered.encode("utf-8"))
    os.chmod(target, 0o600)
    return target


__all__ = [
    "CONFIG_FILENAME",
    "PROJECT_CONFIG_FILENAME",
    "RuntimeConfigError",
    "read_config",
    "read_effective_config",
    "write_config_scope",
    "get_tls",
    "get_envs",
    "get_auth",
    "get_providers",
    "get_rate_limit",
    "get_script_generation",
    "require_activation",
    "get_scheduler",
]
