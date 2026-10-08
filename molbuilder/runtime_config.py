"""Per-machine runtime configuration, read from the ONE machine config.

This module is named ``runtime_config`` (not just ``config``) because
``molbuilder.config`` is the engine-parameter dataclasses package
(``SiestaConfig``, ``PySCFConfig``).  Different
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

A top-level ``cert`` / ``key`` is **refused by name** with the ``tls``
spelling shown (``_RETIRED_FLAT_TLS``).  **An unknown top-level key is
REFUSED with the known sections named** (U7; ``_``-prefixed keys are
comments); the one total list of sections is the ``_SECTIONS`` registry
below (architecture § 8.2a).

This reader is intentionally stateless: it reads the file each time
it's called, parses, validates.  Callers that want a single
process-wide read should go through :mod:`molbuilder.diagnostics`,
which builds the immutable :class:`~molbuilder.diagnostics.Capabilities`
snapshot once per process -- where it is first asked, or at the server's
start.  Putting the cache there (not here) keeps
this module a plain pure function: easy to test, easy to reason about,
no hidden state.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from .config_dir import PRIVATE_FILE_MODE, config_dir

if TYPE_CHECKING:                      # pragma: no cover - typing only
    # `get_routing` is annotated `List["Domain"]` and imports the real class
    # inside its body.  This is the binding static readers read.  It is NOT evaluated at run
    # time, so `typing.get_type_hints(get_routing)` still raises -- that is
    # the known cost of the idiom, and nothing calls it.
    from .scheduler.record import Domain


#: A11: the module that owns the FORMAT owns the NAME.  This one validates
#: `molbuilder.json`'s schema, so the spelling is here -- and the join is here
#: too, once, in :func:`_machine_config_file`.  Everything else asks
#: :func:`machine_config_path`.
CONFIG_FILENAME = "molbuilder.json"

#: THREE SECTIONS RETIRED ON 2026-10-02, each refused by name with what to do
#: instead (`configuration.md` § 4) -- the file holds a person's preferences and
#: nothing else (user, 2026-10-01: "all resources are explicit, and based on the
#: target machine's .json environment manifest"; 2026-10-02: "explicit job
#: config is the only way allowed").
#:
#: ``scheduler`` put THIS machine's choices into every job it prepared, for any
#: target: a queue nobody named for that job, `-c`/`-t`/`--mem` defaults, a
#: queue menu typed by hand, an order to pick queues by.
_SCHEDULER_RETIRED = (
    "{path}: 'scheduler' is no longer configured -- "
    "delete the block.  Nothing in it may be set for every job:\n"
    "  * a job's queue, wall, memory, ranks and cores per rank are the JOB's "
    "own: its description (`allocation`, or the run card `execution` in "
    "task.json) or the prep/launch flags (--domain, --time, --mem, --np, "
    "--cpus-per-task, --gpus) -- prep refuses one stated nowhere;\n"
    "  * a machine's scheduler and queues are its RECORD's: `{probe}` ON that "
    "machine -- with --name and the name you prep it by when it is not this "
    "one, and the file it writes copied into environments/ here.\n"
    "(docs/configuration.md § 4; docs/execution/architecture.md § 5.2)")

#: The section is named for what it holds -- how a shell on THIS machine
#: initialises a conda environment.  The machine record carries it under the
#: same name.
_SCRIPT_GENERATION_RENAMED = (
    "{path}: 'script_generation' is now 'env_init' -- "
    "how a shell on this machine enters a conda environment, the same two "
    "keys.  Write\n"
    "    \"env_init\": {env_init}\n"
    "instead.  (docs/configuration.md § 4)")

#: `execution` is the RUN CARD in task.json -- ranks, threads, GPUs, wall,
#: queue.  This file used the same name for how a launch is sent, and carried a
#: default queue (`domain`) every job without one received.
_EXECUTION_RENAMED = (
    "{path}: 'execution' is now 'launch', so that "
    "`execution` means only the run card in task.json.  Write\n"
    "    \"launch\": {launch}\n"
    "instead.{dropped}  (docs/configuration.md § 4)")


#: Same shape as `_SCHEDULER_RETIRED`: a retired key gets its own sentence,
#: not the generic "unknown top-level key".
_SECRET_KEY_MOVED = (
    "{path}: 'secret_key_file' is no longer configured.  The session key has "
    "ONE home -- <config dir>/secrets/secret_key -- and is created "
    "there on first run (docs/configuration.md § 2.1e).  Delete the line; "
    "move the file there if you want the sessions it signed to survive.")


class RuntimeConfigError(Exception):
    """Raised when ``molbuilder.json`` is present but unreadable / malformed.

    The CLI layer translates this into ``click.UsageError`` and the
    web layer into HTTP 400; the L1 reader itself stays UI-agnostic.
    """


def _naming(path: Path, exc: Exception) -> str:
    """The refusal with the FILE in front of it, once.

    The validators speak in terms of the schema and spell the generic
    ``molbuilder.json``; the reader and the writer know which file refused.
    The generic name is dropped when the real one is supplied.
    """
    msg = str(exc)
    if str(path) in msg:
        return msg
    generic = f"{CONFIG_FILENAME}: "
    if msg.startswith(generic):
        msg = msg[len(generic):]
    return f"{path}: {msg}"


def read_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``molbuilder.json`` from ``path``, or from the one place the
    machine scope lives.

    **The location is asked of :func:`machine_config_path`, never re-derived.**

    Returns the normalised dict (see :func:`_normalise`).  Returns
    ``{}`` if no file exists (not an error -- the file is optional).
    Raises :class:`RuntimeConfigError` when the JSON is malformed or the
    schema is invalid.
    """
    cfg_path = path if path is not None else machine_config_path()
    if not cfg_path.is_file():
        return {}
    raw = _load_raw(cfg_path)
    try:
        return _normalise(raw)
    except RuntimeConfigError as exc:
        raise RuntimeConfigError(_naming(cfg_path, exc)) from None


def _load_raw(path: Path) -> Dict[str, Any]:
    """The file's JSON object, unvalidated -- ``{}`` when there is no file.

    THE ONE PARSE of ``molbuilder.json`` (W54 C19).  `configuration.md`
    § 2.2: a malformed file is refused by the command that reads it, in the
    resolved path's words -- once, here."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except OSError as exc:
        raise RuntimeConfigError(
            f"{path}: cannot be read ({exc.strerror})") from None
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(
            f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})"
        ) from None
    if not isinstance(raw, dict):
        raise RuntimeConfigError(
            f"{path}: top-level value must be an object, "
            f"got {type(raw).__name__}")
    return raw


# --------------------------------------------------------------------- #
#  Provider validators                                                  #
# --------------------------------------------------------------------- #
#
# One validator per supported "kind".  Each returns the
# (possibly-normalised, default-filled) provider entry.  Validators
# raise :class:`RuntimeConfigError` for any malformed entry; that
# error bubbles up through ``_normalise`` to the CLI / web layer.
#
# Adding a new backend = a row in ``_PROVIDER_KINDS`` (its validator and
# the keys its entry holds) + a registration handler in
# ``molbuilder/web/auth_providers/``.  The schema layer here knows
# nothing about HTTP, authlib, or python-cas -- only the contract of
# the JSON payload.


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


#: What a provider entry carrying its secret, or a path to it, is told
#: (`configuration.md` § 3.1; user, 2026-10-02: "no secret in
#: molbuilder.json except the cert files").
_NAMED_SECRET_REFUSED = (
    "{path}: auth.providers[{idx}]: '{key}' is refused -- molbuilder.json "
    "names no secret but the cert files (docs/configuration.md § 3.1).  "
    "The {kind} client secret's home is {home}: write it there, mode 0600 "
    "(`molbuilder auth-setup` does it for Google), and delete '{key}'.")


def _refuse_a_named_secret(entry: Mapping[str, Any], idx: int) -> None:
    """The entry names no secret -- neither its bytes nor its file.

    The secret's home is its kind's, :func:`config_dir.client_secret`.  The
    bytes (``client_secret``) and the path (``client_secret_file``) are both
    refused by name, each saying where the secret goes."""
    from .config_dir import client_secret, relative_home
    for key in ("client_secret", "client_secret_file"):
        if key in entry:
            kind = entry["kind"]
            raise RuntimeConfigError(_NAMED_SECRET_REFUSED.format(
                path=CONFIG_FILENAME, idx=idx, key=key, kind=kind,
                home=relative_home(lambda: client_secret(kind))))


def provider_client_secret(entry: Mapping[str, Any]) -> str:
    """**The provider's client secret, as a string.**

    THE ONE DOOR A CONSUMER USES.  It hands back the secret itself -- never a
    path, and never a hint about whether there was a file at all *(user,
    2026-09-20: "the api should return the secret/key ... how to read and what
    to find out is concealed")*.

    What a secret IS belongs here, because this module owns the provider
    entry's schema.

    **The kind names the file** *(user, 2026-10-02)*: the secret is read from
    :func:`config_dir.client_secret`, and the entry names no path
    (`_refuse_a_named_secret`).  What is concealed is that anyone downstream
    ever learns where it is.  Adding another source later -- an env var, a
    keyring -- changes this function and nothing that calls it.

    Raises :class:`RuntimeConfigError` naming the provider, because every
    failure here is a configuration mistake and the id is what makes it
    actionable.  The message never contains the secret.
    """
    from .config_dir import client_secret, relative_home
    pid = entry.get("id", "?")
    kind = entry.get("kind", "")
    home = client_secret(kind)
    where = relative_home(lambda: home)
    try:
        text = home.read_text().strip()
    except OSError as exc:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[id={pid!r}]: the {kind} "
            f"client secret, {where}, could not be read ({exc.strerror}).  "
            f"Write it there, mode 0600 (`molbuilder auth-setup` does it "
            f"for Google), readable by the server."
        ) from exc
    if not text:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[id={pid!r}]: the {kind} "
            f"client secret, {where}, is empty (or only whitespace).  Write "
            f"the provider's client secret into it and restart."
        )
    return text


def _validate_oauth_common(entry: Dict[str, Any], idx: int) -> None:
    """Mutate ``entry`` in place: validate OAuth shared fields."""
    _require_str(entry, "client_id", idx)
    _refuse_a_named_secret(entry, idx)


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


#: A key nothing reads.
_CAS_VALIDATE_URL_RETIRED = (
    "{path}: auth.providers[{idx}]: 'service_validate_url' is no longer "
    "configured -- delete it.  Nothing ever read it: python-cas derives the "
    "validate endpoint from 'login_url' (its root, then p3/serviceValidate "
    "for CAS v3) and takes no parameter for another one, and a key nothing "
    "reads looks effective.")


def _validate_cas(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    _require_str(entry, "login_url", idx)
    # The validate endpoint is python-cas's own, from `login_url`'s root
    # (`web/auth_providers/cas.py`); a key naming another one is refused by
    # name.  A CAS site whose validate endpoint is NOT
    # `<login root>/p3/serviceValidate` is not supported today: honouring one
    # means overriding python-cas's `url_suffix`, a change to the sign-in path
    # that wants a live CAS to test against.
    if "service_validate_url" in entry:
        raise RuntimeConfigError(_CAS_VALIDATE_URL_RETIRED.format(
            path=CONFIG_FILENAME, idx=idx))

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


#: The keys every provider entry holds, whatever its kind.
_PROVIDER_COMMON_KEYS = ("id", "label", "kind", "allowed_users")
#: An OAuth entry names its client, never its secret: that is at
#: `config_dir.client_secret(kind)`.
_OAUTH_KEYS = ("client_id",)

#: Every kind molbuilder signs in with: its validator, and the keys its entry
#: holds beyond the common four (`ops/deployment.md` § 3).  ONE row per kind,
#: so a kind cannot be validated without its keys being listed -- a key not
#: listed is refused (`configuration.md` § 4), which is what stops a typo in
#: `hosted_domain` from dropping that restriction in silence.
_PROVIDER_KINDS: Dict[str, Tuple[Any, Tuple[str, ...]]] = {
    "google":    (_validate_google,    _OAUTH_KEYS + ("hosted_domain",)),
    "github":    (_validate_github,    _OAUTH_KEYS + ("allowed_organizations",)),
    "microsoft": (_validate_microsoft, _OAUTH_KEYS + ("tenant_id",)),
    "orcid":     (_validate_orcid,     _OAUTH_KEYS),
    "cas":       (_validate_cas,       ("login_url", "version", "service_url",
                                        "ca_certs", "email_attribute",
                                        "email_domain")),
}


#: The kinds that sign in with an OAuth client, and so keep a client secret
#: at `config_dir.client_secret(kind)` -- read off the table above, never
#: listed again.
OAUTH_KINDS = tuple(kind for kind, (_validate, keys) in _PROVIDER_KINDS.items()
                    if "client_id" in keys)


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
    if not _ID_RE.fullmatch(pid):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].id {pid!r} "
            f"must be a URL-safe slug matching {_ID_RE.pattern} "
            f"(it keys the route path /login/<id>)."
        )
    _require_str(out, "label", idx)

    kind = _require_str(out, "kind", idx)
    if kind not in _PROVIDER_KINDS:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: auth.providers[{idx}].kind {kind!r} "
            f"is not supported.  Supported: {', '.join(_PROVIDER_KINDS)}."
        )

    # allowed_users is required so the operator must explicitly think
    # about access control.  An empty list is a degenerate-but-valid
    # case (the provider is enabled but nobody can sign in -- useful
    # for temporarily locking out a backend).
    out["allowed_users"] = _require_str_list(out, "allowed_users", idx)

    # --- kind-specific dispatch, then every other key refused -------- #
    # The kind's validator runs first, so a retired key it names gets its
    # own sentence rather than the generic list.
    validate, keys = _PROVIDER_KINDS[kind]
    out = validate(out, idx)
    _refuse_unknown(out, _PROVIDER_COMMON_KEYS + keys,
                    f"auth.providers[{idx}]",
                    hint=f"(kind {kind!r}: docs/ops/deployment.md § 3)")
    return out


# --------------------------------------------------------------------- #
#  The SECTION REGISTRY -- one row per top-level section (U7,           #
#  2026-08-12).  Everything the loader knows about a section is in its  #
#  row: how it is read (its validator) and whether provenance may      #
#  print its VALUES.  `_normalise`, `config_provenance` and             #
#  `write_config_scope` all consult THIS table and nothing else.        #
#                                                                       #
#  A section is either in this table or its presence is an ERROR;       #
#  there is no third state in which it looks configured and does        #
#  nothing.                                                             #
# --------------------------------------------------------------------- #

def _read_tls(raw: Mapping[str, Any]):
    """The ``tls`` section, and only it.

    There is ONE spelling: a second one is a second place to look.
    ``_normalise`` refuses the flat keys by name and says what to write
    instead.
    """
    tls = _require_object_section(raw, "tls") or {}
    _refuse_unknown(tls, ("cert", "key"), "tls")
    for k, v in tls.items():
        if not isinstance(v, str):
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: 'tls.{k}' must be a string, "
                f"got {type(v).__name__}"
            )
    return tls or None


#: What a config naming the host env is told (`configuration.md` § 2.1c).
_HOST_ENV_FIXED = (
    "{path}: 'envs.host' is no longer configured -- delete it.  The host env "
    "is always 'molbuilder' (docs/configuration.md § 2.1c; user, "
    "2026-10-02: \"let's enforce one name\").")


def _read_envs(raw: Mapping[str, Any]):
    """``envs`` -- the conda environment each category runs in when it is not
    the default name, and the conda-compatible command (`configuration.md`
    § 4).  The categories are `diagnostics`'s own table, asked rather than
    re-listed; another key is refused, and ``envs.host`` by name: the host
    env is always ``molbuilder``."""
    envs = _require_object_section(raw, "envs") or {}
    if "host" in envs:
        raise RuntimeConfigError(_HOST_ENV_FIXED.format(path=CONFIG_FILENAME))
    from .diagnostics import DEFAULT_ENV_NAMES
    _refuse_unknown(envs, (*DEFAULT_ENV_NAMES, "manager"), "envs")
    for k, v in envs.items():
        if str(k).startswith("_"):
            continue
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
    auth = _require_object_section(raw, "auth") or {}
    # A session key's path INSIDE `auth` is refused by name: its one home is
    # `secrets/secret_key` (§ 2.1e).
    if "secret_key_file" in auth:
        raise RuntimeConfigError(
            _SECRET_KEY_MOVED.format(path=CONFIG_FILENAME))
    _refuse_unknown(auth, ("providers", "trust_proxy"), "auth")
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


#: The two notify settings this file does not take.  The listener is switched
#: on by the key file itself, which carries its own route.
_NOTIFY_SETTINGS_MOVED = (
    "{f}: '{key}' is no longer read.\n"
    "\n"
    "The listener is switched on by the KEY FILE, which carries its own\n"
    "route: `<config dir>/secrets/notify_keys`, written by `molbuilder\n"
    "notify-token`.  Both settings were things molbuilder already knew --\n"
    "it chose the path and it issued the route -- so requiring them typed\n"
    "here meant a working key file could sit beside a listener that was\n"
    "never registered, answering 404 to everything.\n"
    "\n"
    "Delete this key.  If you have no `notify_keys` yet, run\n"
    "`molbuilder notify-token <user>`; to keep a route already in\n"
    "service, re-issue with `--route <your existing segment>`."
)


def _read_notify_retired(key):
    """Refuse a retired notify setting by name, with what to do instead.

    Silently ignoring it would be worse than refusing: the operator would
    have a config that looks configured and a listener that is not, which
    is the exact state this change exists to end.
    """
    def read(raw: Mapping[str, Any]):
        if raw.get(key) is None:
            return None
        raise RuntimeConfigError(
            _NOTIFY_SETTINGS_MOVED.format(f=CONFIG_FILENAME, key=key))
    return read


#: The one key `launch` holds, and the two values it takes.
_LAUNCH_MODES = ("direct", "submit")


def _read_launch(raw: Mapping[str, Any]):
    """``launch`` -- how `jobset launch` sends a job when no ``--mode`` is
    given (`running-a-job.md` § 5.4).  ``mode`` is its one key; another is
    refused by name, because a key nothing reads looks effective and is not.
    A ``_``-prefixed key is a comment, as at the top level."""
    section = _require_object_section(raw, "launch")
    if section is None:
        return None
    _refuse_unknown(section, ("mode",), "launch",
                    hint="A job's queue, wall and shape are the job's own "
                         "(docs/execution/architecture.md § 5.2).")
    mode = section.get("mode")
    if mode is not None and mode not in _LAUNCH_MODES:
        raise RuntimeConfigError(
            f'{CONFIG_FILENAME}: launch.mode must be "direct" or "submit"; '
            f"got {mode!r} (docs/execution/running-a-job.md § 5.4).")
    return section


def _read_scheduler_retired(raw: Mapping[str, Any]):
    """``scheduler`` is refused by name (:data:`_SCHEDULER_RETIRED`)."""
    if raw.get("scheduler") is None:
        return None
    from .scheduler.record import probe_command
    raise RuntimeConfigError(_SCHEDULER_RETIRED.format(
        path=CONFIG_FILENAME, probe=probe_command()))


_ENV_INIT_KEYS = ("activation", "preamble")


def _read_env_init(raw: Mapping[str, Any]):
    """``env_init`` -- how a shell on THIS machine enters a conda environment
    (`configuration.md` § 4): ``activation``, one of :data:`ACTIVATION_FORMS`
    with no default, and ``preamble``, the shell run before it, verbatim.
    `jobset probe` copies both into every record it writes, and prep reads
    them from the target's record.  Another key is refused by name; a ``_``
    key is a comment."""
    section = _require_object_section(raw, "env_init")
    if section is None:
        return None
    _refuse_unknown(section, _ENV_INIT_KEYS, "env_init")
    activation = section.get("activation")
    if activation is not None and activation not in ACTIVATION_FORMS:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'env_init.activation' must be one of "
            f"{', '.join(map(repr, ACTIVATION_FORMS))}; got {activation!r}.")
    preamble = section.get("preamble")
    if preamble is not None and not isinstance(preamble, str):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'env_init.preamble' must be a string -- the "
            f"shell run before the activation; got "
            f"{type(preamble).__name__}.")
    return section


def _read_script_generation_renamed(raw: Mapping[str, Any]):
    """``script_generation`` is refused by name, with the ``env_init`` block
    to write in its place -- its own values, so the message moves them."""
    if raw.get("script_generation") is None:
        return None
    section = raw["script_generation"]
    section = section if isinstance(section, Mapping) else {}
    block = {k: section[k] for k in _ENV_INIT_KEYS
             if isinstance(section.get(k), str) and section[k].strip()}
    raise RuntimeConfigError(_SCRIPT_GENERATION_RENAMED.format(
        path=CONFIG_FILENAME, env_init=json.dumps(block)))


def _read_execution_renamed(raw: Mapping[str, Any]):
    """``execution`` is refused by name, with the ``launch`` block to write
    in its place and what has no replacement."""
    if raw.get("execution") is None:
        return None
    section = raw["execution"]
    section = section if isinstance(section, Mapping) else {}
    mode = section.get("mode")
    launch = json.dumps({"mode": mode}) if mode else '{"mode": "direct"}'
    gone = sorted(k for k in section
                  if k != "mode" and not str(k).startswith("_"))
    dropped = (f"  {', '.join(map(repr, gone))} "
               f"{'has' if len(gone) == 1 else 'have'} no replacement: a job "
               f"names its own queue (`allocation.domain` in task.json, or "
               f"--domain)." if gone else "")
    raise RuntimeConfigError(_EXECUTION_RENAMED.format(
        path=CONFIG_FILENAME, launch=launch, dropped=dropped))


def _require_object_section(raw: Mapping[str, Any], name: str):
    """A section that must be an object: keep the key alive and reject a
    non-object early.  Every caller then refuses the keys its section does
    not hold (:func:`_refuse_unknown`) and checks the types of those it
    does."""
    if name not in raw:
        return None
    section = raw[name]
    if not isinstance(section, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: '{name}' must be an object; got "
            f"{type(section).__name__}."
        )
    return dict(section)


def _refuse_unknown(section: Mapping[str, Any], allowed, where: str,
                    hint: str = "") -> None:
    """Refuse, by name, every key of *section* that *allowed* does not list.

    `configuration.md` § 4: *"a key not in this table is refused, never
    ignored"* -- inside a section as at the top level, because a key nothing
    reads looks effective and is not: a misspelled ``admin.emails`` made every
    signed-in user an admin, and ``rate_limit``'s typos were dropped.  A
    ``_``-prefixed key is a comment, as at the top level.  ONE sentence for
    every section.
    """
    unknown = sorted(str(k) for k in section
                     if k not in allowed and not str(k).startswith("_"))
    if not unknown:
        return
    keys = [repr(k) for k in allowed]
    holds = (f"one key, {keys[0]}" if len(keys) == 1
             else f"{', '.join(keys[:-1])} and {keys[-1]}")
    raise RuntimeConfigError(
        f"{CONFIG_FILENAME}: '{where}' holds {holds}; got "
        f"{', '.join(map(repr, unknown))}." + (f"  {hint}" if hint else ""))


def _read_admin(raw: Mapping[str, Any]):
    # Who may do the things only an operator should do (ops/access-control.md
    # § 5; read back by get_admin_emails).  Absent or empty means ANYONE WHO
    # CAN SIGN IN, so a misspelled key is the dangerous typo here: it read as
    # "nobody named" and made every signed-in user an admin -- refused.
    if "admin" not in raw:
        return None
    section = raw["admin"]
    if not isinstance(section, Mapping):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'admin' must be an object like "
            f'{{"emails": ["operator@example.edu"]}}; got '
            f"{type(section).__name__}."
        )
    _refuse_unknown(section, ("emails",), "admin")
    emails = section.get("emails", [])
    if (not isinstance(emails, (list, tuple))
            or not all(isinstance(e, str) for e in emails)):
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'admin.emails' must be a list of "
            f"email strings; got {emails!r}."
        )
    return dict(section)


#: Every directory ``paths`` may name.  A closed set: a key nothing reads
#: would look effective and do nothing, which is the argument behind every
#: refusal in `configuration.md`.
#:
#: ``logs``, ``run`` and ``reports`` are refused by name -- see
#: :data:`_OPERATIONAL_PATHS_MOVED`.
_PATH_KEYS = ("projects",)

#: Same shape as `_SCHEDULER_RETIRED` and `_SECRET_KEY_MOVED`: a retired key
#: gets its own sentence, so it does not read as a typo.
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

    ``projects`` IS THE ONLY KEY.  A config naming ``logs``, ``run`` or
    ``reports`` is **refused** -- `_OPERATIONAL_PATHS_MOVED` carries the
    message, and `configuration.md` § 2.1d carries the reasoning: the `serve` supervisor is
    L1 and this reader is L2, so the supervisor could never reach a
    config-derived answer, leaving two answers to one question.
    ``$XDG_STATE_HOME`` and ``$XDG_RUNTIME_DIR`` move those directories, and
    they answer before any config is read -- which is what a log has to do
    when the thing that failed is the config parse.
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
    _refuse_unknown(section, _PATH_KEYS, "paths",
                    hint="(docs/configuration.md § 2.1d)")
    for key in _PATH_KEYS:
        val = section.get(key)
        if val is None:
            continue
        if not isinstance(val, str) or not val.strip():
            raise RuntimeConfigError(
                f"{CONFIG_FILENAME}: paths.{key} must be a non-empty string "
                f"path; got {val!r}.")
    return section


#: Every key ``rate_limit`` holds, and its type (`configuration.md` § 4;
#: the defaults are `web/rate_limit.DEFAULTS`).  ``bool`` keys are JSON
#: booleans; ``secs`` and ``count`` whole numbers, ``secs`` above 0 and a
#: ``count`` of 0 switching its signal off (`ops/deployment.md` § 4);
#: ``allowlist`` addresses or networks.
_RATE_LIMIT_KEYS = {
    "enabled": "bool", "trust_proxy": "bool",
    "window_404_s": "secs", "window_total_s": "secs", "cooldown_s": "secs",
    "threshold_404": "count", "threshold_total": "count",
    "max_tracked_ips": "positive",
    "allowlist": "nets",
}


def _read_rate_limit(raw: Mapping[str, Any]):
    """``rate_limit`` -- the web server's request limiter, every key typed.

    Read here, so a wrong value is refused before any server is touched.
    """
    import ipaddress
    section = _require_object_section(raw, "rate_limit")
    if section is None:
        return None
    if "admin_emails" in section:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: 'rate_limit.admin_emails' moved to the "
            f"top-level 'admin' section on 2026-08-03 -- write "
            f'"admin": {{"emails": [...]}} instead: one list answers who may '
            f"clear the block list and who may restart the server "
            f"(docs/ops/access-control.md § 5).")
    _refuse_unknown(section, tuple(_RATE_LIMIT_KEYS), "rate_limit",
                    hint="(docs/ops/deployment.md § 4)")
    for key, kind in _RATE_LIMIT_KEYS.items():
        if key not in section:
            continue
        v = section[key]
        where = f"{CONFIG_FILENAME}: 'rate_limit.{key}'"
        if kind == "bool":
            if not isinstance(v, bool):
                raise RuntimeConfigError(
                    f"{where} must be a JSON boolean (true / false); got "
                    f"{v!r}.")
        elif kind == "nets":
            if not isinstance(v, list):
                raise RuntimeConfigError(
                    f"{where} must be a list of addresses or networks "
                    f'("127.0.0.1", "10.0.0.0/8"); got {v!r}.')
            for entry in v:
                try:
                    ipaddress.ip_network(str(entry).strip(), strict=False)
                except ValueError:
                    raise RuntimeConfigError(
                        f"{where} entry {entry!r} is not an IP address or "
                        f"network.") from None
        else:
            least = 0 if kind == "count" else 1
            # bool is an int subclass and is never a count.
            if isinstance(v, bool) or not isinstance(v, int) or v < least:
                what = ("a whole number of seconds above 0" if kind == "secs"
                        else "a whole number, 0 to switch the signal off"
                        if kind == "count" else "a whole number above 0")
                raise RuntimeConfigError(f"{where} must be {what}; got {v!r}.")
    return section


#: name -> how it is read · whether provenance may print its values · whether
#: it is retired.  Every section lives in THIS machine's molbuilder.json, the
#: one config file.  ``provenance_safe`` gates `config_provenance`: True only
#: where every value is printable in logs (no secrets, no paths to secrets).
#: A ``retired`` row exists only to refuse its section by name, so the
#: unknown-key refusal never offers it as a known one.
_SECTIONS: Dict[str, Dict[str, Any]] = {
    "tls":               {"read": _read_tls,
                          "provenance_safe": False},
    "envs":              {"read": _read_envs,
                          "provenance_safe": False},
    "auth":              {"read": _read_auth,
                          "provenance_safe": False},
    "notify_keys_file":  {"read": _read_notify_retired("notify_keys_file"),
                          "provenance_safe": False, "retired": True},
    "notify_route":      {"read": _read_notify_retired("notify_route"),
                          "provenance_safe": False, "retired": True},
    "launch":            {"read": _read_launch,
                          "provenance_safe": True},
    "env_init":          {"read": _read_env_init,
                          "provenance_safe": False},
    # RETIRED 2026-10-02 -- refused by name, each with what to do instead
    # (`configuration.md` § 4).
    "execution":         {"read": _read_execution_renamed,
                          "provenance_safe": False, "retired": True},
    "script_generation": {"read": _read_script_generation_renamed,
                          "provenance_safe": False, "retired": True},
    "scheduler":         {"read": _read_scheduler_retired,
                          "provenance_safe": False, "retired": True},
    "checkpoint":        {"read": lambda raw: (
                              _validate_checkpoint(raw["checkpoint"])
                              if "checkpoint" in raw else None),
                          "provenance_safe": False},
    "admin":             {"read": _read_admin,
                          "provenance_safe": False},
    "rate_limit":        {"read": _read_rate_limit,
                          "provenance_safe": False},
    "paths":             {"read": _read_paths,
                          "provenance_safe": True},
}

#: The flat spelling of ``tls``, ONLY to be refused by name: a person whose file says ``"cert"`` at the top
#: level has to be told the nested spelling, not handed the generic
#: unknown-key list and left to guess which of the sections it belongs in.
_RETIRED_FLAT_TLS = ("cert", "key")

#: What to say when one of them turns up.
_FLAT_TLS_RETIRED = (
    "{path}: 'tls' is a section, not top-level keys.  Found {found} at the "
    "top level -- one setting has one spelling.  Write:\n"
    '    "tls": {{"cert": "...", "key": "..."}}'
)


def _normalise(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Read every registered section + validate values; refuse the rest.

    ``tls`` is a SECTION.  The flat top-level ``cert``/``key`` spelling is
    refused BY NAME, showing the section to write, rather than as an
    "unknown top-level key".

    **An unknown top-level key is an ERROR, not tolerance** (U7;
    running-a-job.md § 5): ignoring one would swallow every typo'd section
    name.  The registry makes "known" one total list, so refusing is precise and the message can
    name what IS known.

    Value-type validation lives in each section's ``read`` (see
    ``_SECTIONS``) so the ``get_*`` accessors stay trivial and callers
    never see a section whose entries aren't the documented types.
    """
    # A leading underscore marks a COMMENT key ("_comment_tls": ...) --
    # JSON has no comments and the committed templates lean on this idiom.
    # An explicit marker is not the typo class the refusal exists for.
    unknown = sorted(k for k in raw
                     if k not in _SECTIONS and not k.startswith("_"))
    if "secret_key_file" in unknown:
        raise RuntimeConfigError(_SECRET_KEY_MOVED.format(path=CONFIG_FILENAME))
    # NAMED BEFORE THE GENERIC LIST: whoever wrote them is owed the
    # section's spelling, not "unknown top-level key(s) 'cert'".
    flat = [k for k in _RETIRED_FLAT_TLS if k in raw]
    if flat:
        raise RuntimeConfigError(_FLAT_TLS_RETIRED.format(
            path=CONFIG_FILENAME,
            found=", ".join(repr(k) for k in flat)))
    if unknown:
        raise RuntimeConfigError(
            f"{CONFIG_FILENAME}: unknown top-level "
            f"key(s) {', '.join(map(repr, unknown))}.  Known sections: "
            f"{', '.join(n for n, s in _SECTIONS.items() if not s.get('retired'))}"
            f".  A key this loader does not "
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
    2026-08-21).  Absent means "probe".
    """
    return str(dict(cfg.get("envs", {})).get("manager", "") or "")


def get_admin_emails(cfg: Mapping[str, Any]) -> frozenset:
    """Who may do the things only an operator should do.

    Reads the top-level ``admin`` section::

        "admin": { "emails": ["operator@asu.edu"] }

    **Absent or empty means NOBODY IS NAMED HERE**, and what that means is
    one rule, `web/admin.is_admin_request`'s, for both subsystems that ask --
    the rate limiter's block list and the server restart
    (`access-control.md` § 5): **anyone who can sign in**.  That is not an
    open door, because reaching a session at all requires being in a
    provider's ``allowed_users``, a REQUIRED field -- an operator has already
    written every person down by hand.  Naming addresses here narrows it.
    Anonymous is never an admin.

    The empty-set reading is § 5's: a second list repeating the allow-list is
    two lists to keep in step for one question, and on a single-operator
    server it is the same address written twice.

    Emails are lowercased and blanks dropped, matching how the auth layer
    stores ``session["user"]["email"]``, so membership is case-stable -- the
    one place that is done.  ``cfg`` has met :func:`_normalise`, whose
    :func:`_read_admin` holds the section's shape, so it is not checked again
    (W54 C21).
    """
    emails = (cfg.get("admin") or {}).get("emails") or ()
    return frozenset(e.strip().lower() for e in emails if e.strip())


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
    _refuse_unknown(raw, ("size_limit_bytes", "engines"), "checkpoint",
                    hint="(docs/execution/checkpointing.md § 4)")
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
    # `read_config` hands back the section already validated (the registry's
    # reader IS `_validate_checkpoint`); the defaults are asked for only when
    # the file has no section at all.
    section = read_config().get("checkpoint") or _validate_checkpoint({})
    return sorted(section["engines"])


def get_checkpoint(engine: Optional[str] = None) -> Dict[str, Any]:
    """The effective checkpoint classification (checkpointing.md § 4).

    **Server-wide scope only, and that is the rule rather than an omission.**
    There is deliberately no ``project_dir`` parameter: reading a scope beside
    the folder being saved is exactly the per-folder classification S1c
    forbids, and it is what would let somebody change where files are stored
    between a save and a restore (I2c).

    ``engine`` is a **hint** and may be omitted or unknown: an engine nobody
    configured resolves to ``generic``, which names no always-large families and
    therefore measures every file.  That is always correct and merely slower,
    which is the direction this contract errs in -- an unknown engine must never
    make a save store less.

    Returns ``{"size_limit_bytes": int, "always_large": [glob, ...]}``.
    """
    section = read_config().get("checkpoint") or _validate_checkpoint({})
    engines = section["engines"]
    always = engines.get(engine) if engine else None
    if always is None:
        always = engines.get("generic", [])
    return {
        "size_limit_bytes": int(section["size_limit_bytes"]),
        "always_large":     list(always),
    }


#: The activation's two legal values, in ONE home: `env_init` is checked
#: against them, and `envs init-config` offers them as the choices.
#: Presentation order is the caller's business.
ACTIVATION_FORMS: tuple = ("source activate", "conda activate")


# --------------------------------------------------------------------- #
#  The machine config file -- where it is, and how it is written        #
#  (`configuration.md` § 2)                                             #
# --------------------------------------------------------------------- #


def _machine_config_file() -> Path:
    """The machine config, in the config directory.

    **The bootstrap caller.**  This is how ``molbuilder.json`` is FOUND, so it
    is the one location that cannot be declared inside ``molbuilder.json`` --
    which is why :func:`molbuilder.config_dir.config_dir` takes its override
    from the ENVIRONMENT (``MOLBUILDER_CONFIG_DIR``) rather than from a config
    key, which would be circular.
    """
    return config_dir() / CONFIG_FILENAME


#: The sections :func:`config_provenance` reports.  A deliberate ALLOWLIST:
#: the machine file also carries ``auth`` / ``tls``,
#: and provenance output lands in terminals, STAGE-PLAN.md and shipped run
#: logs -- material that must never travel there.
#: Derived from the registry -- a section's values may be printed in logs
#: only where its row says so (the one list, U7).
_PROVENANCE_SECTIONS = tuple(
    name for name, spec in _SECTIONS.items() if spec["provenance_safe"])


def machine_config_path() -> Path:
    """Which ``molbuilder.json`` the MACHINE scope resolves to.

    The one lookup, so a refusal can name the file it is refusing, and the
    display that answers *"which file said this"* and the reader that raises
    about it describe one file.
    """
    # ONE LOCATION (`archive/2026-09-01-config-access-plan.md` § 3.3).
    return _machine_config_file().resolve()


def machine_config_shadow() -> Optional[str]:
    """A warning when a ``./molbuilder.json`` is sitting there UNREAD.

    `configuration.md` § 2.1a.  The machine scope has ONE location, the
    per-user config directory.  A working-directory file is **not read at
    all** -- so it loses silently, and a person editing it would watch their
    changes do nothing (user, 2026-08-31: *"I had instances where information are saved in
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
    home = machine_config_path()
    if here == home:
        # The working directory IS the config directory (`cd ~/.config/
        # molbuilder`): the file here is the one that is read.
        return None
    return "\n".join([
        f"{CONFIG_FILENAME} in the working directory is NOT READ: {here}",
        f"  The machine config has one location, and this is not it: {home}"
        + ("" if home.is_file() else "  (no file there yet)"),
        "  Move it there, or delete it (configuration.md § 2.1a).",
    ])


def config_provenance(project_dir: Optional[Path] = None, *,
                      target: Optional[str] = None,
                      record=None) -> Dict[str, Any]:
    """Which config files this process consults, and which one supplied each
    execution-relevant value — the answer to *"where did that setting come
    from?"* at the moment it takes effect (user request, 2026-08-12: the
    inert-fixture bug class is invisible without it).

    Safe for logs **by construction**: paths, presence, and the effective
    values of :data:`_PROVENANCE_SECTIONS` plus the names of the target
    record's queues — never the file contents (see the allowlist note above).

    Returns ``{"sources": [...], "effective": {...}, "domains": [...]}``:
    ``sources`` lists each file consulted as ``{scope, path, found}`` -- this
    machine's molbuilder.json, then the machine records; ``effective`` maps
    ``section.key`` to ``{"value": ..., "from": "machine"}``.  ``project_dir``
    names the calculation whose own machine record is listed; ``target`` the
    machine a prep names (``--target``), whose record is listed in its
    place among the scopes (`scheduler.record.record_scopes`, the one
    statement of their order) -- so the first found is the one that
    answered.  ``record`` is the record the caller read from them, whose
    queues are listed (`prep` reads it once, at its checkpoint 4); with none
    they are asked of :func:`get_routing`.
    """
    machine_path = machine_config_path()
    sources = [{"scope": "machine", "path": str(machine_path.resolve()),
                "found": machine_path.is_file()}]
    # WHAT THIS SCOPE IS STANDING IN FRONT OF (§ 2.1a).  The row above says
    # which file was reached; it cannot say that another one exists and was
    # skipped, and that is the state where a setting is written twice and read
    # once.  Asked of the one place that phrases it, never re-worded here.
    shadow = machine_config_shadow()
    # Same split as § 2.1a's: WHICH file, and whether it is safe to hold what
    # it holds.  Asked of the one place that phrases each, never re-worded --
    # the mode's is the placement table's row, the sentence the terminal
    # prints too (§ 2.1b).
    from .placement import machine_config_finding
    mode_warning = machine_config_finding()


    # RAW file bytes decide what a file "supplied" (R10): the normalised
    # scopes carry validator defaults no file said.
    machine_file = _load_raw(machine_path)
    effective: Dict[str, Dict[str, Any]] = {}
    for section in _PROVENANCE_SECTIONS:
        block = machine_file.get(section)
        if not isinstance(block, Mapping):
            continue
        for key, value in block.items():
            effective[f"{section}.{key}"] = {"value": value,
                                             "from": "machine"}
    # Domains come from the MACHINE RECORD (N4), not from these files, so
    # provenance follows them there: the record's own scopes join `sources`,
    # because "which file supplied this" is the question this function exists
    # to answer and environment.json answers part of it (`configuration.md`
    # § 5, M-3).  The scopes, and their order, are the record's OWN list
    # (`record_scopes`) -- a named target's included.
    from .scheduler.record import (AmbiguousTarget, UnknownTarget,
                                   record_scopes)
    for via, path in record_scopes(project_dir, target):
        sources.append({"scope": "environment", "path": str(path),
                        "found": Path(path).is_file(), "via": via})
    # The record in hand when there is one -- what answered; else through
    # `get_routing`, NOT a second resolution: a display whose whole job is to
    # say where a value came from must ask the reader that answers it.
    if record is not None:
        domains = [d.name for d in routing_of(record)]
    else:
        try:
            domains = [d.name for d in get_routing(project_dir=project_dir)]
        except (AmbiguousTarget, UnknownTarget):
            domains = []           # which machine is not decided yet: no queues
    return {"sources": sources, "effective": effective, "domains": domains,
            "shadow": shadow, "mode_warning": mode_warning}


def format_provenance(prov: Mapping[str, Any]) -> str:
    """The ONE rendering of :func:`config_provenance` — the CLI echo and
    STAGE-PLAN.md both use it, so they cannot drift."""
    lines = ["config:"]
    # Width from the WIDEST scope name present, not a literal.
    width = max([len(s["scope"]) for s in prov["sources"]] + [8]) + 1
    for s in prov["sources"]:
        state = "found" if s["found"] else "absent"
        via = f", via {s['via']}" if s["found"] and s.get("via") else ""
        lines.append(f"  {s['scope']:<{width}}{s['path']}  ({state}{via})")
    for key in sorted(prov["effective"]):
        e = prov["effective"][key]
        lines.append(f"  {key} = {e['value']!r}   <- {e['from']}")
    if prov["domains"]:
        # Named for where they LIVE.
        lines.append(f"  environment.domains: "
                     f"{', '.join(prov['domains'])}")
    return "\n".join(lines)


def _deep_merge(base: Dict[str, Any],
                 overlay: Dict[str, Any]) -> Dict[str, Any]:
    """How `write_config_scope` lays a patch over the file
    (`configuration.md` § 2.3):
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


def get_paths() -> Dict[str, Any]:
    """The effective ``paths`` block, or ``{}``.  See :func:`_read_paths`."""
    return dict(read_config().get("paths") or {})


def get_launch_mode() -> Optional[str]:
    """This machine's ``launch.mode`` -- ``"direct"``, ``"submit"`` or ``None``
    (`running-a-job.md` § 5.4).  ``None`` is UNSET, and `launch` then refuses
    rather than derive one: deciding ``submit`` from a DETECTED scheduler
    would gate submission on detection.  It arrives validated
    (:func:`_read_launch`), so nothing is checked here."""
    return (read_config().get("launch") or {}).get("mode")


def get_env_init() -> Dict[str, str]:
    """THIS machine's ``env_init`` -- ``activation`` and ``preamble``, each
    only when stated -- from this machine's own ``molbuilder.json``
    (`configuration.md` § 4).  What `jobset probe` copies into the record it
    writes."""
    section = read_config().get("env_init") or {}
    return {k: section[k] for k in _ENV_INIT_KEYS
            if isinstance(section.get(k), str) and section[k].strip()}


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

    Each entry is a :class:`~molbuilder.scheduler.record.Domain` (phase 3 of
    `execution/scheduler.md` § 8).

    Returns ``[]`` when there is no record, or on a workstation: no queue to
    name.  Order is the record's; nothing here chooses among them -- a job
    names its own (`execution/architecture.md` § 5.2).

    ``project_dir`` selects the calculation scope, so a folder carried to a
    cluster reads the record `prep` snapshotted beside it (M-3's precedence,
    through the one door).

    ``local_only`` bypasses ``project_dir`` and asks a different question --
    :func:`molbuilder.scheduler.record.machine_for`'s ``local_only`` docstring
    has the reasoning.
    """
    from .scheduler import machine_for
    return routing_of(machine_for(project_dir, local_only=local_only))


def routing_of(env) -> List["Domain"]:
    """The submission-domain menu OF a machine record in hand -- its probed
    domains -- for a caller that already holds the record it means (a
    ``--target`` prep reads the TARGET's, which ``get_routing`` cannot name
    before the calculation has snapshotted it).  :func:`get_routing` is this,
    asked of the record ``machine_for`` answers.

    **The record's queues and nothing else** *(user, 2026-10-02)*: a
    target's queues are probed on that machine and its record copied here
    (`configuration.md` § 5)."""
    return list(env.domains) if env is not None else []


def write_config_scope(patch: Mapping[str, Any]) -> Path:
    """Write a partial config patch into this machine's ``molbuilder.json``,
    which has one location and is created there when absent
    (`configuration.md` § 2.1c).

    The patch is deep-merged ONTO the existing file's contents (per
    :func:`_deep_merge`), preserving keys outside the patch.  A corrupt
    existing file REFUSES rather than being overwritten (R10).
    Files are written atomically
    (persist.write_bytes) at mode 0600 -- a config file may carry
    the TLS key's path, deploy context, or per-cluster setup commands
    that aren't meant for casual inspection.  THE ONE WRITER of this
    file.

    Returns the resolved target path.
    """
    # THE SAME DOOR THE READER USES.  A writer with its own idea of where the
    # machine file lives writes one nothing reads, which is the whole failure
    # this change removes.
    target = machine_config_path()

    try:
        existing = _load_raw(target)
    except RuntimeConfigError as exc:
        # REFUSED, not overwritten (R10): a config carrying auth providers
        # and TLS paths is exactly the file a user cannot afford to lose to
        # a typo.
        raise RuntimeConfigError(
            f"{exc}  Nothing was written: a corrupt config is refused, never "
            f"overwritten -- fix it (or move it aside) and retry.") from None

    merged = _deep_merge(existing, dict(patch))
    # Round-trip through the validator BEFORE writing so we never
    # produce a file that ``read_config`` would reject.
    try:
        _normalise(merged)
    except RuntimeConfigError as exc:
        # WHOSE FAULT?  The merge fails when the patch is bad -- or when the
        # file was already one the server would refuse before this write (a
        # bad `envs` entry, measured 2026-09-10 through the auth wizard,
        # which then blamed itself).  Ask the existing file alone: a person
        # sent to fix "the patch" for a section it never touched fixes the
        # wrong thing.
        if existing:
            try:
                _normalise(existing)
            except RuntimeConfigError as was:
                raise RuntimeConfigError(
                    f"{target} was ALREADY one the server would refuse, "
                    f"before this write -- {was}.  Nothing was written; fix "
                    f"that section and retry.") from None
        raise RuntimeConfigError(_naming(target, exc)) from None

    # 0700 through the one creator when this call is what makes the
    # directory.  It does not TIGHTEN
    # one that is already there: a writer is not the place that polices what
    # the operator set up (`envs doctor` is).
    from .config_dir import ensure_private_dir
    ensure_private_dir(target.parent)
    rendered = json.dumps(merged, indent=2, sort_keys=False) + "\n"
    # Through the ONE atomic writer (U8's shape): a crash mid-write must not
    # leave a truncated config for every later read to refuse.
    #
    # `mode=` rather than a chmod afterwards (D11): a chmod afterwards leaves
    # the temp loose with the content in it, the window `write_bytes`' own
    # `mode=` parameter exists to close.
    from .persist import write_bytes
    write_bytes(target, rendered.encode("utf-8"), mode=PRIVATE_FILE_MODE)
    return target


__all__ = [
    "CONFIG_FILENAME",
    "RuntimeConfigError",
    "read_config",
    "write_config_scope",
    "get_tls",
    "get_envs",
    "get_auth",
    "get_providers",
    "get_rate_limit",
    "get_launch_mode",
    "get_env_init",
    "ACTIVATION_FORMS",
]
