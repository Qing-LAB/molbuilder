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

    return out


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
]
