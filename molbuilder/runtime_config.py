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
                  "mdtools": "molbuilder-MDtools" }
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
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


CONFIG_FILENAME = "molbuilder.json"
# Per-project config sidecar.  Per docs/config.md § 2: hidden file in
# the project directory, same schema as the server-wide molbuilder.json.
PROJECT_CONFIG_FILENAME = ".molbuilder.json"
# XDG fallback for the server-wide config when one isn't present in cwd.
# Matches the convention auth-setup uses (see molbuilder/auth_setup.py).
_XDG_FALLBACK_RELATIVE = Path(".config") / "molbuilder" / CONFIG_FILENAME


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

    # --- script_generation section (docs/config.md § 3.6) ----------- #
    if "script_generation" in raw:
        out["script_generation"] = _validate_script_generation(
            raw["script_generation"])

    # --- scheduler section (slurm-integration.md § 4) --------------- #
    # Validated lazily by get_scheduler (which merges scopes first and
    # applies the refuse-to-emit rule on the MERGED result).  Here we
    # only need to (a) keep the key out of the allowlist's drop set and
    # (b) reject a non-object early.  A partial block (e.g. project
    # scope supplying only ``defaults.time``) is legal at this layer --
    # completeness is a merged-config property, enforced in get_scheduler.
    if "scheduler" in raw:
        sched = raw["scheduler"]
        if not isinstance(sched, Mapping):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler' must be an object; got "
                f"{type(sched).__name__}."
            )
        out["scheduler"] = dict(sched)

    # --- execution section (job-execution.md § 8.13) ---------------- #
    # The run-vs-submit launch policy.  Like ``scheduler``, it is merged +
    # validated lazily by its getter (:func:`get_execution`); here we only
    # keep it out of the allowlist's drop set and reject a non-object early.
    if "execution" in raw:
        execn = raw["execution"]
        if not isinstance(execn, Mapping):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'execution' must be an object; got "
                f"{type(execn).__name__}."
            )
        out["execution"] = dict(execn)

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


def get_rate_limit(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``rate_limit`` section, or ``{}``.

    The ``rate_limit`` block tunes the IP-based scanner-detection +
    blocklist installed by :mod:`molbuilder.web.rate_limit`.  See
    :file:`docs/protocols/rate-limit.md` for the full schema.
    Trivial accessor -- defaults are applied inside ``RateLimiter``.
    """
    return dict(cfg.get("rate_limit", {}))


# --------------------------------------------------------------------- #
#  script_generation section (docs/config.md § 3.6)                     #
# --------------------------------------------------------------------- #


# Two and only two keys -- see docs/config.md § 4.
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
# per the v2 rewrite of docs/config.md).  A one-time WARNING is logged
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
                f"renamed to '{current}' (docs/config.md § 7).  "
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
                f"(docs/config.md § 6/§ 7).  Please remove it.",
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
#  Multi-scope read/write API (docs/config.md § 4)                      #
# --------------------------------------------------------------------- #


def _per_user_fallback_path() -> Path:
    """``~/.config/molbuilder/molbuilder.json``, honouring XDG.

    Mirrors :func:`molbuilder.auth_setup.default_secret_dir`'s
    convention so the auth-setup wizard and config reader land in the
    same per-user directory.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "molbuilder" / CONFIG_FILENAME
    return Path.home() / _XDG_FALLBACK_RELATIVE


def _read_scope(path: Path) -> Dict[str, Any]:
    """Read one scope's JSON file; ``{}`` if absent.

    Reuses :func:`read_config` for the parse + normalise path; this
    wrapper just hides the missing-file vs read-error distinction
    so the caller can short-circuit on empty.
    """
    if not path.is_file():
        return {}
    return read_config(path)


def _read_server_wide() -> Dict[str, Any]:
    """Server-wide config: cwd ``./molbuilder.json`` first, XDG
    fallback ``~/.config/molbuilder/molbuilder.json`` second.

    The cwd-first preference matches existing behaviour (every test +
    deployment path written before 2026-06-24 reads from cwd).  XDG is
    a fallback for users who'd rather not litter their project dirs
    with config files.
    """
    cwd_path = Path(CONFIG_FILENAME)
    if cwd_path.is_file():
        return _read_scope(cwd_path)
    return _read_scope(_per_user_fallback_path())


def _read_project(project_dir: Path) -> Dict[str, Any]:
    return _read_scope(Path(project_dir) / PROJECT_CONFIG_FILENAME)


def _deep_merge(base: Dict[str, Any],
                 overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Per docs/config.md § 2:
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

    Per docs/config.md § 3 + § 4:
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

    Per docs/config.md § 2 (refuse-to-emit rule): the generator must
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
            "is the toolchain.  See docs/config.md § 4."
        )
    return sg["activation"]


def detect_conda_activation() -> Optional[Dict[str, str]]:
    """Best-effort detect a conda/mamba activation for THIS machine.

    Returns ``{"activation": "conda activate", "preamble": 'source
    "<base>/etc/profile.d/conda.sh"'}`` when a conda/mamba install is
    discoverable (via ``$CONDA_EXE`` / ``$MAMBA_EXE`` / ``conda`` on
    PATH / ``$CONDA_PREFIX``) and its activation hook exists; else None.

    The hook path is the GENERATING machine's — correct when the script
    runs on the SAME machine (a workstation where you generate and run).
    For a different target (e.g. an HPC cluster reached after
    ``module load mamba``) set ``script_generation`` explicitly; this
    detection is only the no-config convenience for the local case.
    """
    import shutil

    base: Optional[Path] = None
    exe = os.environ.get("CONDA_EXE") or os.environ.get("MAMBA_EXE")
    if exe:
        # <base>/bin/conda  or  <base>/condabin/conda
        base = Path(exe).parent.parent
    if base is None:
        which = shutil.which("conda") or shutil.which("mamba")
        if which:
            # Resolve symlinks: a PATH shim (e.g. ~/.local/bin/conda ->
            # <base>/bin/conda) would otherwise yield the wrong base.
            base = Path(which).resolve().parent.parent
    if base is None:
        pref = os.environ.get("CONDA_PREFIX")
        if pref:
            pp = Path(pref)
            # CONDA_PREFIX may point at an env (<base>/envs/<name>);
            # walk back to the base in that case.
            base = pp.parent.parent if pp.parent.name == "envs" else pp
    if base is None:
        return None
    hook = base / "etc" / "profile.d" / "conda.sh"
    if not hook.is_file():
        return None
    return {"activation": "conda activate", "preamble": f'source "{hook}"'}


# --------------------------------------------------------------------- #
#  scheduler section (docs/protocols/slurm-integration.md § 4)          #
# --------------------------------------------------------------------- #


# Supported scheduler kinds.  Only SLURM today; PBS/local are future.
_SCHEDULER_KINDS: tuple = ("slurm",)

# ``directives`` keys we recognise (stable site #SBATCH header values).
# Unknown keys are accepted verbatim (forward-compat) but these are the
# ones the emitter maps to canonical flags.
_SCHEDULER_DIRECTIVE_KEYS: tuple = (
    "partition", "qos", "mail_type", "mail_user", "export",
)

# Per-job defaults (slurm-integration.md § 4.1, § 6).  ``time`` is a
# walltime string; ``cpus_per_task`` an int (OMP width per rank); ``mem``
# a string like "120G" or None (=> scheduler default).
_SCHEDULER_DEFAULT_KEYS: tuple = ("time", "cpus_per_task", "mem")


def _validate_scheduler(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate + normalise the ``scheduler`` block of one merged config.

    Returns the resolved ``{kind, directives, gpu, defaults}`` dict.
    Raises :class:`RuntimeConfigError` on shape errors AND on the
    refuse-to-emit rule (slurm-integration.md § 10): a ``slurm`` site
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
    mem_model  = _as_obj("mem_model")

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
                f"docs/protocols/slurm-integration.md § 4, § 7.0, § 10."
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
    # gpu.mem_cap_per_gpu: ceiling on a NON-exclusive GPU job's --mem,
    # per requested GPU (the node's proportional host-RAM share, e.g.
    # 512G/4 = "128G" on Sol A100 nodes).  Keeps a shared-node GPU job
    # backfill-friendly; the emitter scales it by the job's GPU count.
    # gpu.mem is the matching FLOOR.  See render_sbatch.
    if "mem_cap_per_gpu" in gpu and gpu["mem_cap_per_gpu"] is not None \
            and not isinstance(gpu["mem_cap_per_gpu"], str):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'scheduler.gpu.mem_cap_per_gpu' must be "
            f"a string (e.g. \"128G\") or null; got "
            f"{type(gpu['mem_cap_per_gpu']).__name__}."
        )

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

    # mem_model: tunable coefficients for the CPU --mem estimator
    # (molbuilder/siesta/memory.py).  All values must be numeric; the
    # estimator's MemModel.from_config is tolerant of missing keys, so we
    # only type-check what IS present.
    for k, v in mem_model.items():
        if v is not None and not isinstance(v, (int, float)) \
                or isinstance(v, bool):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'scheduler.mem_model.{k}' must be a "
                f"number; got {type(v).__name__}."
            )

    out = {
        "kind":       kind,
        "directives": directives,
        "gpu":        gpu,
        "defaults":   defaults,
        "mem_model":  mem_model,
    }
    # routing: pass through verbatim (deep-validated by get_routing, which
    # owns the domain schema -- slurm-integration.md § 4.3).  Preserve it
    # here so get_scheduler -> get_routing can see it.
    if raw.get("routing") is not None:
        out["routing"] = raw["routing"]
    return out


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
    ``.sbatch`` (slurm-integration.md § 10).  When a block IS present it
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
    if project_dir is not None:
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

    return _validate_scheduler(merged)


def get_execution(
    project_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return the effective ``execution`` block (job-execution.md § 8.13):
    the run-vs-submit launch policy, read at prep time on the target.

    Server-wide and project scopes are deep-merged (project wins).

    Returns:
        {
            "mode":       "direct" | "submit" | None,   # None -> caller
                          # derives from the detected scheduler
            "submit_via": "slurm",                       # backend when submit
        }

    ``mode`` is the source of truth for HOW a benchmark point is launched,
    decoupled from the DETECTED scheduler (which still drives topology
    detection): you can be *on* slurm yet launch ``direct``, or force
    ``submit`` from an interactive shell.  A malformed ``mode`` raises.
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
            "Fix it in .molbuilder.json (job-execution.md § 8.13).")
    submit_via = merged.get("submit_via", "slurm")
    domain = merged.get("domain")
    if domain is not None and not isinstance(domain, str):
        raise RuntimeConfigError(
            f"execution.domain must be a string (a scheduler.routing name); "
            f"got {type(domain).__name__} (job-execution.md § 8.14).")
    return {"mode": mode, "submit_via": submit_via, "domain": domain}


def get_routing(
    project_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return the validated ``scheduler.routing`` list (slurm-integration.md
    § 4.3): the named submission-domain menu, read at prep time.

    Each entry is a domain ``{name, max_time, max_mem_gb?, partition, qos,
    gpu_partition?}``.  Returns ``[]`` when no table is configured (→ the
    single ``directives`` default behavior, unchanged).  The framework
    hardcodes NO names/limits — every value here is user data, seeded from
    the user's live ``sinfo``/``sacctmgr`` (§ 7.0).  Order is preserved
    (most-constrained → most-general); the FIRST fitting domain is the
    recommendation.
    """
    sched = get_scheduler(project_dir=project_dir) or {}
    raw = sched.get("routing")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RuntimeConfigError(
            "scheduler.routing must be a list of domain objects "
            "(slurm-integration.md § 4.3).")
    out: List[Dict[str, Any]] = []
    seen = set()
    for i, dom in enumerate(raw):
        if not isinstance(dom, Mapping):
            raise RuntimeConfigError(
                f"scheduler.routing[{i}] must be an object; got "
                f"{type(dom).__name__}.")
        for key in ("name", "max_time", "partition", "qos"):
            if not dom.get(key) or not isinstance(dom[key], str):
                raise RuntimeConfigError(
                    f"scheduler.routing[{i}].{key} is required and must be a "
                    f"non-empty string (§ 4.3).")
        name = dom["name"]
        if name in seen:
            raise RuntimeConfigError(
                f"scheduler.routing: duplicate domain name {name!r} (§ 4.3).")
        seen.add(name)
        mem = dom.get("max_mem_gb")
        if mem is not None and not isinstance(mem, (int, float)):
            raise RuntimeConfigError(
                f"scheduler.routing[{i}].max_mem_gb must be a number (GB).")
        gpu_part = dom.get("gpu_partition")
        if gpu_part is not None and not isinstance(gpu_part, str):
            raise RuntimeConfigError(
                f"scheduler.routing[{i}].gpu_partition must be a string.")
        out.append({
            "name": name, "max_time": dom["max_time"],
            "max_mem_gb": mem, "partition": dom["partition"],
            "qos": dom["qos"], "gpu_partition": gpu_part,
        })
    return out


def write_config_scope(
    project_dir: Optional[Path],
    patch: Mapping[str, Any],
) -> Path:
    """Write a partial config patch into one scope.

    ``project_dir`` selects:
      * ``None``: server-wide file at the highest-precedence existing
        location (cwd if present, else the XDG path).  When neither
        exists, the XDG path is created.
      * a path: ``<project_dir>/.molbuilder.json``.

    The patch is deep-merged ONTO the existing file's contents (per
    :func:`_deep_merge`), preserving keys outside the patch.  Files
    are written mode 0600 to match :mod:`molbuilder.auth_setup`'s
    precedent -- a config file may carry secret-file PATHS, deploy
    context, or per-cluster setup commands that aren't meant for
    casual inspection.

    Returns the resolved target path.
    """
    if project_dir is None:
        cwd_path = Path(CONFIG_FILENAME).resolve()
        target = cwd_path if cwd_path.is_file() else _per_user_fallback_path()
    else:
        target = Path(project_dir) / PROJECT_CONFIG_FILENAME

    existing: Dict[str, Any] = {}
    if target.is_file():
        try:
            existing = json.loads(target.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            # Corrupt file -- log nothing, overwrite.  Callers that
            # want to preserve it should make a backup before calling.
            existing = {}

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
    # Atomic create with 0600 perms (same trick as auth_setup).
    fd = os.open(
        str(target),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        os.write(fd, rendered.encode("utf-8"))
    finally:
        os.close(fd)
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
    "get_secret_key_file",
    "get_rate_limit",
    "get_script_generation",
    "require_activation",
    "get_scheduler",
]
