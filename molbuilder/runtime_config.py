"""Per-machine runtime configuration read from ``./molbuilder.json``.

This module is named ``runtime_config`` (not just ``config``) because
``molbuilder.config`` is the engine-parameter dataclasses package
(``SiestaConfig``, ``PySCFConfig``, ``SpectraConfig``).  Different
concerns:

* ``molbuilder.config.*``        -- L1 dataclasses, calculation
  parameters serialised into the generated input deck.
* ``molbuilder.runtime_config``  -- per-machine deployment knobs
  (TLS paths, conda env names) read at startup from a gitignored
  file at the repo root.

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
                  "mdtools": "molbuilder-MDtools",
                  "tests":   "molbuilder-tests" }
    }

For backwards compatibility with the flat shape shipped before the
2026-05-14 four-env design, top-level ``cert`` and ``key`` keys are
also honoured (folded into ``tls`` by :func:`_normalise`).  Unknown
keys are ignored silently so the file can grow new sections without
breaking older readers.

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
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


CONFIG_FILENAME = "molbuilder.json"


class RuntimeConfigError(Exception):
    """Raised when ``molbuilder.json`` is present but unreadable / malformed.

    The CLI layer translates this into ``click.UsageError`` and the
    web layer into HTTP 400; the L1 reader itself stays UI-agnostic.
    """


def read_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``molbuilder.json`` from ``path`` or cwd.

    Returns the normalised dict (see :func:`_normalise`).  Returns
    ``{}`` if the file doesn't exist (not an error -- the file is
    optional).  Raises :class:`RuntimeConfigError` when the JSON is
    malformed or the schema is invalid.
    """
    cfg_path = path if path is not None else Path(CONFIG_FILENAME)
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
    return _normalise(raw)


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


def _normalise(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Fold flat-shape keys into the nested schema + validate values.

    Precedence: when both the nested section AND the flat key are
    present, the nested value wins.  Migrating from flat to nested is
    therefore non-destructive -- adding the ``tls`` block doesn't
    require removing the top-level ``cert``/``key``.

    Value-type validation lives here so :func:`get_tls` and
    :func:`get_envs` can stay trivial accessors and callers never see
    a section whose entries aren't the documented types.
    """
    out: Dict[str, Any] = {}

    # --- TLS section ------------------------------------------------- #
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
    if tls:
        out["tls"] = tls

    # --- envs section ------------------------------------------------ #
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
    if envs:
        out["envs"] = envs

    # --- auth section ------------------------------------------------ #
    #
    # Optional; absent ``auth`` means no authentication (the right
    # default for the localhost-only single-user deployment shape).
    #
    # When present, the schema is::
    #
    #     "auth": {
    #         "providers": [
    #             { "id": ..., "label": ..., "kind": ..., ... },
    #             ...
    #         ]
    #     }
    #
    # Each provider entry is a self-contained sign-in option that
    # renders as its own button on the login page.  ``id`` keys the
    # route paths (``/login/<id>`` and either ``/oauth-callback/<id>``
    # or ``/cas-callback/<id>`` depending on kind), so it must be a
    # URL-safe slug and unique across the list.
    #
    # Per-provider ``allowed_users`` is the access gate -- there is
    # no global allowlist.  An identity authenticated through provider
    # X is matched ONLY against X's own ``allowed_users``; an email
    # missing from the list means access denied even if it appears
    # in some other provider's list.  This makes each entry self-
    # contained and avoids precedence-rule surprises.
    #
    # Supported kinds and the shape contract for each are documented
    # in ``_validate_provider`` below.  See ``docs/deployment.md``
    # for the operator walkthrough per backend.
    # Explicit-presence check (rather than ``if auth:``): writing
    # ``"auth": {}`` is almost certainly a mistake and we want to
    # catch it with a clear error rather than silently degrading to
    # no-auth mode.  Omit the key entirely to disable auth.
    if "auth" in raw:
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
        # installs werkzeug.middleware.proxy_fix.ProxyFix so the FIRST
        # upstream proxy's X-Forwarded-* headers are honoured.  Default
        # False -- the right choice for direct-TLS deploys.  See
        # _setup_session_security in molbuilder/web/auth.py for the
        # security implications of flipping it on.
        trust_proxy = auth.get("trust_proxy", False)
        if not isinstance(trust_proxy, bool):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'auth.trust_proxy' must be a "
                f"JSON boolean (true / false); got "
                f"{type(trust_proxy).__name__}."
            )

        out["auth"] = {
            "providers":   validated,
            "trust_proxy": trust_proxy,
        }

    # --- secret_key_file --------------------------------------------- #
    #
    # Path to the file holding the Flask session-signing key.  Auto-
    # generated on first use when missing (32 random bytes via os.
    # urandom).  Required when auth is enabled; quietly ignored
    # otherwise (no sessions to sign).
    secret_key_file = raw.get("secret_key_file")
    if secret_key_file is not None:
        if not isinstance(secret_key_file, str):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'secret_key_file' must be a "
                f"string path; got {type(secret_key_file).__name__}."
            )
        out["secret_key_file"] = secret_key_file

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


def get_secret_key_file(cfg: Mapping[str, Any]) -> Optional[str]:
    """Path to the session-signing key file, or ``None`` when not
    configured."""
    return cfg.get("secret_key_file")


def get_tls(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Return the ``tls`` section, or ``{}``.

    Trivial accessor -- type validity is enforced upstream in
    :func:`_normalise`.  Callers that pass a hand-constructed cfg
    (not via :func:`read_config`) are responsible for its shape.
    """
    return dict(cfg.get("tls", {}))


def get_envs(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Return the ``envs`` section, or ``{}``.

    Trivial accessor -- type validity is enforced upstream in
    :func:`_normalise`.
    """
    return dict(cfg.get("envs", {}))


__all__ = [
    "CONFIG_FILENAME",
    "RuntimeConfigError",
    "read_config",
    "get_tls",
    "get_envs",
    "get_auth",
    "get_providers",
    "get_secret_key_file",
]
