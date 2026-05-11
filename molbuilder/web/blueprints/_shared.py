"""Helpers shared across the build / modify / watch blueprints.

This module is the SINGLE source of truth for:

* the ``Issue`` -> JSON wire shape
* JSON -> Structure body parsing (xyz + per-atom metadata lists)
* Structure -> JSON response body construction
* JSON -> dataclass coercion (used by Build for SiestaConfig /
  PySCFConfig form values; available to Modify for any future op
  that takes a dataclass-driven body, e.g. M5's electrode panel)

If a helper is genuinely blueprint-specific (e.g. Build's
``/api/build/load`` accepts both multipart and JSON, Modify's body
parsing always carries the canonical state bundle), it stays in
the calling blueprint.  Promote here when at least two callers
need the same behaviour and drift would silently break wire
contracts.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import fields
from typing import Any, Dict, List, Tuple

from flask import jsonify

from molbuilder.structure import Structure
from molbuilder.validation import validate_geometry


# --------------------------------------------------------------------- #
#  Issues                                                                #
# --------------------------------------------------------------------- #


def issues_to_json(issues):
    """Serialise List[Issue] for the JSON wire.

    The web client reads ``issues[].severity / message / where`` to
    decide how to display.  Schema duplicated literally in both
    blueprints' tests; if a key changes here, those tests catch it.
    """
    return [
        {"severity": i.severity, "message": i.message, "where": i.where}
        for i in issues
    ]


# --------------------------------------------------------------------- #
#  JSON <-> Structure  (canonical Modify body shape, also reusable for  #
#  any future endpoint that takes "xyz + per-atom metadata" arrays)     #
# --------------------------------------------------------------------- #


def struct_from_body(body: Dict[str, Any]) -> Structure:
    """Reconstruct a Structure from the canonical body shape::

        {
          "xyz":            "<xyz string>",
          "atom_names":     [...],   # optional; len == n_atoms
          "residue_ids":    [...],   # optional
          "residue_names":  [...],   # optional
          "chain_ids":      [...],   # optional
          "title":          "..."    # optional
        }

    A metadata list is honoured only when its length matches the atom
    count; otherwise the default from ``Structure.from_xyz``
    (atom_names = elements, residue_ids = [1]*n, residue_names =
    ["MOL"]*n, chain_ids = ["A"]*n) is kept so a malformed metadata
    array can't corrupt the result.

    Raises ``ValueError`` when the xyz field is missing or empty;
    callers turn that into an HTTP 400 with the standard error shape.

    Construction goes through a single ``Structure(...)`` call so all
    invariants in ``Structure.__post_init__`` (parallel-array length
    checks, dtype coercion of positions) fire on the final shape --
    NOT via post-construction ``setattr`` that bypasses the contract.
    """
    xyz = body.get("xyz") or ""
    if not isinstance(xyz, str) or not xyz.strip():
        raise ValueError("missing or empty 'xyz'")
    title = body.get("title") or None
    base = Structure.from_xyz(xyz, title=title)
    n = base.n_atoms
    # Pick the body-supplied list only if it's the right shape; else
    # keep the from_xyz default.  Then construct a fresh Structure
    # so __post_init__ validates the combined invariants.
    def _pick(attr, default):
        v = body.get(attr)
        if isinstance(v, list) and len(v) == n:
            return list(v)
        return default
    return Structure(
        elements      = list(base.elements),
        positions     = base.positions,
        atom_names    = _pick("atom_names",    list(base.atom_names)),
        residue_ids   = _pick("residue_ids",   list(base.residue_ids)),
        residue_names = _pick("residue_names", list(base.residue_names)),
        chain_ids     = _pick("chain_ids",     list(base.chain_ids)),
        title         = base.title,
    )


def structure_to_dict(struct: Structure) -> Dict[str, Any]:
    """Serialise a Structure into the per-atom-metadata-rich JSON shape
    used by ``/api/build/load`` and every ``/api/modify/*`` response.
    Callers wrap with ``jsonify`` and add their own extra keys.
    """
    return {
        "xyz":           struct.to_xyz(),
        "elements":      list(struct.elements),
        "atom_names":    list(struct.atom_names),
        "residue_ids":   list(struct.residue_ids),
        "residue_names": list(struct.residue_names),
        "chain_ids":     list(struct.chain_ids),
        "n_atoms":       struct.n_atoms,
        "n_residues":    struct.n_residues,
        "title":         struct.title or "",
    }


def ok_structure_response(struct: Structure):
    """Build a Flask jsonify response for an op result.

    Adds ``ok: True`` and runs ``validate_geometry`` to populate the
    issues array.  Used by every /api/modify/* endpoint; build.py's
    /api/build/load adds its own ``source_format`` / ``summary`` /
    ``pdb`` keys on top.
    """
    payload = {"ok": True, **structure_to_dict(struct),
               "issues": issues_to_json(validate_geometry(struct))}
    return jsonify(payload)


def err(msg: str, code: int = 400):
    """Standard error response shape for the modify routes."""
    return jsonify({"ok": False, "error": msg}), code


def finite_float(name: str, value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float or raise ``ValueError`` with
    a request-facing message.  Used by the /api/modify/* float fields
    so a JSON body that passes ``"nan"`` / ``"inf"`` (or a stringified
    huge number that parses but breaks downstream geometry) gets
    rejected at the boundary instead of silently producing a
    NaN-coordinate structure.

    Returns ``default`` when ``value`` is None or "" -- mirrors the
    ``body.get(field, default)`` pattern already in the route
    handlers.
    """
    if value is None or value == "":
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must be a finite number; got {value!r}") from exc
    if not math.isfinite(f):
        raise ValueError(f"{name!r} must be finite; got {value!r}")
    return f


# --------------------------------------------------------------------- #
#  JSON -> dataclass coercion (Build's SIESTA / PySCF config; reusable  #
#  for any future dataclass-driven Modify endpoint, e.g. M5 electrode)  #
# --------------------------------------------------------------------- #


def dataclass_to_form_schema(cls, id_prefix: str) -> Dict[str, Any]:
    """Build a JSON form-rendering schema from an L1 config dataclass.

    Closes the last Principle-#1 anti-pattern: the SIESTA + PySCF
    form fields in ``web/templates/index.html`` and the per-field
    parse logic in ``viewer.js`` used to duplicate the dataclass
    field set (~50 fields each side).  This generator walks
    ``dataclasses.fields(cls)`` ONCE and emits everything the JS
    renderer needs to construct the form -- so adding a new field
    is now a one-line metadata change on the dataclass.

    Schema shape::

        {
          "config":    "SiestaConfig",
          "id_prefix": "p",
          "sections":  [
            {"name": "System", "fields": [<field_schema>, ...]},
            ...
          ],
        }

    Per-field shape (subset; only the relevant keys for the field's
    inferred kind are present)::

        {
          "name":     "<dataclass field name>",         # canonical key
          "id":       "<id_prefix>-<id_suffix>",        # HTML id
          "label":    "<human label>",                  # from metadata.label
          "help":     "<help / tooltip>",
          "default":  <JSON-serialisable default>,
          "tier":     "basic" | "advanced",
          "kind":     "checkbox" | "int" | "number" | "text"
                      | "select" | "tri-select" | "int-triple",
          # number / int:
          "min": ..., "max": ..., "step": ...,
          # select / tri-select:
          "choices": [...],
          "null_option": True,
          "null_label":  "<label for the empty option>",
          # int-triple (kgrid):
          "labels": ["x", "y", "z"],
          # display:
          "unit": "Å" | "Ry" | ...,
          "pattern": "<HTML pattern attr>",
        }

    **Opt-in via ``section``**: only fields whose metadata declares a
    ``"section"`` key are exposed.  Fields without a section live on
    the dataclass for the Python API / CLI but stay off the web form
    (psml paths, write_forces always-on flags, MD-only knobs that
    only matter for relax_type=Verlet, etc.).

    **ID override via ``id_suffix``**: by default the HTML id is
    ``"{id_prefix}-{field_name.replace('_', '-')}"``.  A few fields
    have shorter legacy ids (e.g. ``p-temperature`` for
    ``electronic_temperature``); they declare ``"id_suffix"`` so the
    compatibility engine + sessionStorage list stay backwards-
    compatible.
    """
    hints = typing.get_type_hints(cls)
    sections_in_order: List[str] = []
    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for f in fields(cls):
        section = f.metadata.get("section")
        if not section:
            continue
        if section not in by_section:
            sections_in_order.append(section)
            by_section[section] = []
        by_section[section].append(_field_to_schema(f, hints, id_prefix))
    return {
        "config":    cls.__name__,
        "id_prefix": id_prefix,
        "sections": [
            {"name": s, "fields": by_section[s]}
            for s in sections_in_order
        ],
    }


def _field_to_schema(f: dataclasses.Field,
                     hints: Dict[str, Any],
                     id_prefix: str) -> Dict[str, Any]:
    """One dataclass field -> one schema entry.

    Pure inspection: no I/O, no side effects, only field.type +
    field.metadata + field.default.  Optional[X] unwraps to X with
    ``optional=True`` so the renderer knows to emit an empty/auto
    sentinel option.
    """
    ann = hints.get(f.name, f.type)
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    is_optional = (origin is typing.Union and type(None) in args)
    if is_optional:
        ann = next((a for a in args if a is not type(None)), str)
        origin = typing.get_origin(ann)
        args   = typing.get_args(ann)

    md = dict(f.metadata)
    id_suffix = md.get("id_suffix", f.name.replace("_", "-"))
    out: Dict[str, Any] = {
        "name":     f.name,
        "id":       f"{id_prefix}-{id_suffix}",
        "label":    md.get("label", f.name.replace("_", " ").capitalize()),
        "help":     md.get("help", ""),
        "default":  _serialize_default(f),
        "optional": is_optional,
        "tier":     md.get("tier", "basic"),
    }
    if "unit" in md:
        out["unit"] = md["unit"]

    choices = md.get("choices")
    if choices is not None:
        out["kind"] = "select"
        out["choices"] = list(choices)
        # An Optional[str] with explicit choices needs an empty
        # sentinel option in the UI (e.g. the dispersion select
        # whose "none" choice maps to None).
        if is_optional:
            out["null_option"] = True
            out["null_label"] = md.get("null_label", "(default)")
    elif ann is bool:
        if is_optional:
            # Optional[bool] -> tri-select (auto / true / false).
            # Today only parallel_over_k uses this pattern.
            out["kind"] = "tri-select"
            out["choices"] = ["auto", "true", "false"]
        else:
            out["kind"] = "checkbox"
    elif ann is int:
        out["kind"] = "int"
        rng = md.get("range")
        if rng is not None:
            out["min"], out["max"] = rng
        if is_optional:
            out["null_option"] = True
            out["null_label"] = md.get("null_label", "(auto)")
    elif ann is float:
        out["kind"] = "number"
        # step="any" is the HTML "accept any float"; widgets can
        # override with metadata["step"] when they want spinner steps.
        out["step"] = md.get("step", "any")
        rng = md.get("range")
        if rng is not None:
            out["min"], out["max"] = rng
        if is_optional:
            out["null_option"] = True
            out["null_label"] = md.get("null_label", "(auto)")
    elif origin is tuple and args:
        # Tuple[int, int, int] -- only kgrid today.  Renderer emits
        # three side-by-side number inputs with sub-ids
        # f"{id}-{labels[i]}".  We pass the labels through so the
        # k-grid UI's "kx / ky / kz" stays declaration-driven.
        out["kind"] = "int-triple"
        out["labels"] = list(md.get("triple_labels", ("x", "y", "z")))
    elif ann is str:
        out["kind"] = "text"
    else:
        # Sequence[str] (species_order) etc. -- not exposed in the
        # form today.  Fall back to text so the schema is at least
        # well-formed for tests, but the field shouldn't have a
        # section anyway.
        out["kind"] = "text"

    if "pattern" in md:
        out["pattern"] = md["pattern"]
    return out


def _serialize_default(f: dataclasses.Field) -> Any:
    """JSON-friendly default for the schema.

    dataclasses use MISSING when the field uses ``default_factory``;
    we don't expose those (no form field uses one today) but if a
    future one does, ``None`` is a safe placeholder.  Tuples become
    lists for JSON compatibility.
    """
    if f.default is dataclasses.MISSING:
        return None
    v = f.default
    if isinstance(v, tuple):
        return list(v)
    return v


def coerce_to_field_type(field: dataclasses.Field, value: Any,
                         resolved_hints: Dict[str, Any]) -> Any:
    """Convert a JSON-arriving value to the field's declared type.

    The form layer can deliver number-typed fields as strings ("300"
    rather than 300) when the request comes from a non-browser HTTP
    client (the in-tree JS frontend coerces with parseFloat/parseInt
    so the test path is fine).  Without coercion, the dataclass
    happily stores the string, downstream the validator's range check
    raises ``TypeError`` on ``string < int`` and the validator-pass
    swallows it as a "skip this validator", quietly losing the
    out-of-range warning.

    Coercion respects ``Optional[X]`` (the empty string and ``None``
    pass through as ``None``).  ``bool`` accepts the JSON literal True
    / False as well as the strings ``"true"`` / ``"false"`` / ``"1"`` /
    ``"0"`` (case-insensitive).  Tuple-typed fields like ``kgrid``
    fall through to per-element int coercion.

    Unknown / unhandled types pass through untouched -- the dataclass
    constructor sees what the caller sent.

    Coercion failures (TypeError / ValueError) propagate to the
    caller so the endpoint can surface them as an error-severity
    Issue rather than HTTP 400.
    """
    ann = resolved_hints.get(field.name, field.type)
    origin = typing.get_origin(ann)
    args   = typing.get_args(ann)
    is_optional = (origin is typing.Union and type(None) in args)
    if is_optional:
        if value is None or value == "":
            return None
        ann = next((a for a in args if a is not type(None)), str)
        origin = typing.get_origin(ann)
        args   = typing.get_args(ann)

    if ann is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)
    if ann is int:
        return int(value)
    if ann is float:
        return float(value)
    if ann is str:
        return str(value)
    # Tuple[int, int, int] (kgrid is the only such field today).
    if origin is tuple and args:
        if not isinstance(value, (list, tuple)):
            return value
        elem_t = args[0]
        return tuple(elem_t(v) for v in value)
    # Sequence[str] (species_order in SiestaConfig) -- accept either
    # a comma-string or an already-list value.
    if origin in (list, tuple) and args and args[0] is str:
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value
    # Anything else: pass through.
    return value


def config_from_params(cls, params: Dict[str, Any],
                       hints: Dict[str, Any],
                       none_sentinels: Tuple[str, ...] = ()):
    """Build a dataclass instance from a JSON-style params dict.

    Walks dataclass fields, picks the matching key from ``params``,
    coerces to the field's declared type via ``coerce_to_field_type``,
    and constructs the dataclass.

    ``none_sentinels``: per-field rule for "this string means None"
    (e.g. ``("solvent", "auxbasis", "dispersion")`` for PySCFConfig
    where the form sends an empty string for "leave default").
    """
    by_name = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for k, v in params.items():
        f = by_name.get(k)
        if f is None:
            continue
        # Form-sentinel "empty string -> None / drop" handling for
        # specific Optional fields the JS deliberately blanks out.
        if k in none_sentinels and (v == "" or v is None):
            kwargs[k] = None
            continue
        # Backwards-compat: JS sometimes sends "none" for "no
        # dispersion".  Same treatment as the empty-string case.
        if k == "dispersion" and isinstance(v, str) and v.strip().lower() == "none":
            kwargs[k] = None
            continue
        # net_charge: empty string from the form means "auto-detect"
        # (don't pass the kwarg so the dataclass default of None
        # kicks in and render_fdf falls back to the phosphate
        # heuristic).
        if k == "net_charge" and (v == "" or v is None):
            continue
        # Coercion failures (TypeError / ValueError) propagate to the
        # endpoint, which surfaces them as an error-severity Issue
        # rather than HTTP 400 -- so the UI renders the same panel
        # for parse-failure as for validator-failure.
        kwargs[k] = coerce_to_field_type(f, v, hints)
    return cls(**kwargs)


__all__ = [
    "issues_to_json",
    "struct_from_body",
    "structure_to_dict",
    "ok_structure_response",
    "err",
    "finite_float",
    "coerce_to_field_type",
    "config_from_params",
]
