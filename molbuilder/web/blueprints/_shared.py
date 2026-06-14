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
from typing import Any, Dict, List, Optional, Tuple

from flask import jsonify

from molbuilder.structure import Structure
from molbuilder.validation import validate_geometry


# --------------------------------------------------------------------- #
#  Issues                                                                #
# --------------------------------------------------------------------- #


def resolve_workflow_group(where: str, cfg) -> Optional[str]:
    """Return the workflow-group binding for an Issue ``where`` field.

    Per docs/protocols/web-ui-coherence.md Rule 2, validator findings
    should attach to the workflow-group card whose fields they
    concern: a ``config.mesh_cutoff`` finding belongs in the Stage
    card; ``config.spin_polarized`` belongs in the Run profile card;
    ``config.max_scf_iter`` belongs in the Compute & budget card.
    The mapping is the SAME single source of truth that drives the
    form-schema render: each dataclass field's
    ``metadata["workflow_group"]``.

    Returns ``None`` for:
      * Issues that don't target a config field (where prefix isn't
        ``config.``) — geometry / cell / polymer findings render in
        a residual panel below the cards.
      * Issues whose dotted name doesn't resolve to a real dataclass
        field (typo, future-proofing).
      * Issues whose field has no ``workflow_group`` metadata (legacy
        untagged fields; rendered in residual panel).
    """
    if not where or not where.startswith("config."):
        return None
    if not dataclasses.is_dataclass(cfg):
        return None
    # Strip "config." prefix and any further dotted sub-fields:
    # "config.net_charge.makov_payne" → "net_charge".
    tail = where.split(".", 1)[1]
    field_name = tail.split(".", 1)[0]
    for f in fields(cfg):
        if f.name == field_name:
            return f.metadata.get("workflow_group")
    return None


def issues_to_json(issues, cfg=None):
    """Serialise List[Issue] for the JSON wire.

    The web client reads ``issues[].severity / message / where /
    workflow_group`` to decide how to display.  Schema duplicated
    literally in both blueprints' tests; if a key changes here,
    those tests catch it.

    ``cfg`` is the engine config dataclass — when provided, each
    issue's ``workflow_group`` is resolved (per
    :func:`resolve_workflow_group`) so the frontend can attach
    findings to their workflow-group card per web-ui-coherence
    Rule 2.  When ``cfg`` is None, the ``workflow_group`` key is
    omitted from the dict (the Issue's own ``workflow_group`` field
    is still honoured if pre-tagged).
    """
    out = []
    for i in issues:
        d = {"severity": i.severity, "message": i.message,
             "where": i.where}
        # An Issue may pre-tag its workflow_group; if not, derive
        # from the where field via the config dataclass metadata.
        group = i.workflow_group
        if group is None and cfg is not None:
            group = resolve_workflow_group(i.where, cfg)
        if group is not None:
            d["workflow_group"] = group
        out.append(d)
    return out


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


def atoms_list(struct: Structure) -> List[Dict[str, Any]]:
    """Build the per-atom payload list — the same shape
    ``/api/selection/atoms`` returns.

    Used by every response that carries a Structure so the front-end's
    selection store stays in sync with the in-memory geometry without
    a separate fetch.  Pre-2026-06-07, modifier-op responses lacked
    this and the selection panel went stale after every Delete / Add /
    Orient / etc — the disk hadn't changed yet so the
    ``/api/selection/atoms`` re-fetch returned pre-op atoms.

    Each row:

        {
            "index":         int,
            "element":       "C" | "H" | ...,
            "regions":       [str, ...],     # from struct.regions
            "is_frozen":     bool,           # from struct.frozen_atoms
            "atom_name":     "CA" | ...,     # optional, PDB-derived
            "residue_name":  "ALA" | ...,    # optional
            "chain_id":      "A"   | ...,    # optional
        }
    """
    n = len(struct.elements)
    atom_to_regions: Dict[int, list] = {}
    regions = getattr(struct, "regions", {}) or {}
    for label, idxs in regions.items():
        for idx in idxs:
            atom_to_regions.setdefault(idx, []).append(label)

    frozen_set = set(getattr(struct, "frozen_atoms", []) or [])

    atom_names    = struct.atom_names    or []
    residue_names = struct.residue_names or []
    chain_ids     = struct.chain_ids     or []

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {
            "index":     i,
            "element":   struct.elements[i],
            "regions":   atom_to_regions.get(i, []),
            "is_frozen": i in frozen_set,
        }
        if i < len(atom_names)    and atom_names[i]:
            row["atom_name"]    = atom_names[i]
        if i < len(residue_names) and residue_names[i]:
            row["residue_name"] = residue_names[i]
        if i < len(chain_ids)     and chain_ids[i]:
            row["chain_id"]     = chain_ids[i]
        rows.append(row)
    return rows


def workspace_payload(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The canonical wire shape for every endpoint that returns a
    ``Structure``.

    Per :doc:`protocols/workspace-state` § 4.4, this is the single
    serialiser that supersedes the four hand-rolled jsonify blobs
    that historically lived inside ``/api/build/load``,
    ``/api/build/molecule``, ``/api/modify/*``, and
    ``/api/selection/atoms``.  Adding a new field that every
    consumer should see (``bonds``, ``dipole``, …) is a one-line
    change here, applied to every endpoint at once.

    The shape (always present):

    .. code-block:: python

        {
            "text":          "<xyz/pdb bytes>",
            "source_format": "xyz" | "pdb",
            "title":         "<title or empty string>",
            "n_atoms":       <int>,
            "atoms":         [<per-atom row>, ...],   # per atoms_list
            "lattice":       [[...3...], [...3...], [...3...]] | None,
            "issues":        [<Issue JSON>, ...],
            "extra":         { ... endpoint-specific add-ons ... },
        }

    Endpoint-specific add-ons (``backend_used``,
    ``add_hydrogens_mode``, ``pdb``, ``summary``,
    ``selection_remap``, …) belong in ``extra``.  The dispatcher
    on the client reads the canonical keys and treats ``extra`` as
    opaque metadata — extending ``extra`` does not break any
    consumer.

    Notes
    -----
    * ``source_format`` defaults to ``"xyz"`` because
      :class:`molbuilder.structure.Structure` round-trips through
      :meth:`Structure.to_xyz`.  PDB-emitting endpoints set
      ``source_format="pdb"`` via ``extra`` plus a ``"text"``
      override at the callsite (see Phase 2 migration in
      :doc:`protocols/workspace-state` § 6).
    * ``lattice`` is currently always ``None``: the L1 ``Structure``
      dataclass carries geometry only; lattice belongs to
      :class:`molbuilder.frame.Frame` / :class:`Trajectory`.  When
      Structure grows a periodic-cell field, this helper is the
      one place to add it.
    * ``issues`` is populated via :func:`validate_geometry` — the
      same set the modify-tab response array already exposed.
      Callers that don't want the validation pass (e.g. a
      throughput-sensitive path that already validated) pass
      ``extra={"issues_skipped": True}`` and override ``issues``
      via ``extra`` if needed.
    """
    return {
        "text":          struct.to_xyz(),
        "source_format": "xyz",
        "title":         struct.title or "",
        "n_atoms":       struct.n_atoms,
        "atoms":         atoms_list(struct),
        "lattice":       None,
        "issues":        issues_to_json(validate_geometry(struct)),
        "extra":         dict(extra) if extra else {},
    }


def structure_to_dict(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The canonical-plus-legacy serialised shape for any
    Structure-returning endpoint.

    Routes through :func:`workspace_payload` for the canonical
    keys (``text``, ``source_format``, ``title``, ``n_atoms``,
    ``atoms``, ``lattice``, ``issues``, ``extra``) and adds the
    legacy aliases that the existing modify-tab front-end's
    ``applyStructure(r)`` reads directly (``xyz``, ``elements``,
    flat per-atom columns, ``n_residues``).

    The optional ``extra`` dict (Phase 2 addition, 2026-06-07)
    threads endpoint-specific keys (``pdb``, ``summary``,
    ``backend_used``, ``add_hydrogens_mode``, ``selection_remap``)
    into BOTH places at once:

    * At the top level of the returned dict, for back-compat with
      every existing JS consumer that reads them off the response
      root (Phase 1-3 clients).
    * In the canonical ``extra`` sub-dict, where the Phase 4+
      workspace dispatcher will read them after the client
      migration completes.

    Top-level ``extra`` keys override the canonical defaults — a
    caller emitting ``source_format="pdb"`` from a PDB-parsing
    endpoint replaces the canonical default of ``"xyz"`` at both
    the root level and inside ``extra``.

    One :func:`validate_geometry` pass per call (issues are read
    from the workspace payload; not recomputed in this helper).
    """
    extras = dict(extra) if extra else {}
    base = workspace_payload(struct, extra=extras)
    return {
        # Canonical keys (forward-compat with workspace dispatcher).
        "text":          base["text"],
        "source_format": base["source_format"],
        "title":         base["title"],
        "n_atoms":       base["n_atoms"],
        "atoms":         base["atoms"],
        "lattice":       base["lattice"],
        "issues":        base["issues"],
        "extra":         base["extra"],
        # Legacy aliases for existing modify-tab consumers
        # (applyStructure reads these directly).
        "xyz":           base["text"],
        "elements":      list(struct.elements),
        "atom_names":    list(struct.atom_names),
        "residue_ids":   list(struct.residue_ids),
        "residue_names": list(struct.residue_names),
        "chain_ids":     list(struct.chain_ids),
        "n_residues":    struct.n_residues,
        # Endpoint-specific keys at the top level for back-compat
        # with existing JS consumers.  Phase 4+ readers go through
        # ``extra`` instead.
        **extras,
    }


def ok_structure_response(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
):
    """Build a Flask jsonify response for any Structure-returning
    endpoint.

    Phase 2 of the workspace-state migration (2026-06-07) —
    ``/api/build/load`` + ``/api/build/molecule`` + every
    ``/api/modify/*`` route through this helper instead of
    hand-rolling their own jsonify blob.  The optional ``extra``
    dict carries per-endpoint add-ons:

    * ``/api/build/load``: ``{"pdb", "summary", "source_format"}``
      (``source_format`` overrides the canonical XYZ default with
      the actually-parsed format).
    * ``/api/build/molecule``: ``{"pdb", "summary",
      "backend_used", "add_hydrogens_mode"}``.
    * ``/api/modify/<op>``: empty today; Phase 3 adds
      ``selection_remap`` for delete/add_atom.

    Wraps :func:`structure_to_dict` (which routes through
    :func:`workspace_payload`) in ``{"ok": True, ...}``.  No
    extra validate-pass — issues come from the workspace payload
    via structure_to_dict.
    """
    return jsonify({"ok": True, **structure_to_dict(struct, extra=extra)})


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

    **Section ordering**: by default sections appear in the order
    their first field is declared in the dataclass.  When the class
    declares an ``_form_section_order`` class attribute (a tuple /
    list of section names), sections appear in that order instead;
    any sections present in field metadata but missing from the
    explicit order get appended after the explicit ones in
    declaration order.  This lets a dataclass control form layout
    without reorganising the (often natural) field declaration
    order.
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

    # Explicit section ordering via _form_section_order on the class.
    # Names not present in that list keep their declaration-order
    # position (appended after the explicit names).
    declared_order = getattr(cls, "_form_section_order", None)
    if declared_order:
        seen = set()
        ordered: List[str] = []
        for name in declared_order:
            if name in by_section and name not in seen:
                ordered.append(name)
                seen.add(name)
        for name in sections_in_order:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        sections_in_order = ordered

    # Per-section descriptions (optional class attribute).  When a
    # class declares ``_form_section_descriptions: Dict[str, str]``,
    # each section's schema entry picks up its blurb and the form
    # renderer surfaces it below the legend.  Sections missing from
    # the dict get no description (omitted from the output) so this
    # is opt-in per class.
    descriptions = getattr(cls, "_form_section_descriptions", {}) or {}

    def _section_entry(s: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"name": s, "fields": by_section[s]}
        desc = descriptions.get(s)
        if desc:
            entry["description"] = desc
        return entry

    return {
        "config":    cls.__name__,
        "id_prefix": id_prefix,
        "sections": [_section_entry(s) for s in sections_in_order],
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
        "default":  _serialize_default(f, ann),
        "optional": is_optional,
        "tier":     md.get("tier", "basic"),
    }
    if "unit" in md:
        out["unit"] = md["unit"]
    # Source-of-truth tag: the actual engine keyword this UI field
    # writes into the generated input.  Surfaced next to the form label
    # so the user can map UI -> generated script -> engine manual /
    # error messages without guessing.  Optional metadata key; fields
    # without an obvious 1:1 keyword mapping (kgrid as a block, etc.)
    # leave it unset and the UI shows no tag.
    if "engine_key" in md:
        out["engine_key"] = md["engine_key"]
    # Workflow-group tag (2026-06-13): one of "system" / "stage" /
    # "budget".  Drives the .workflow-group--<kind> card wrappers
    # in form-schema.js + STAGE_PRESETS' restricted write surface
    # in viewer.js.  Fields without this tag render bare (outside
    # any workflow-group wrapper) and STAGE_PRESETS never touches
    # them.  See docs/protocols/results-tab.md § 4.6 + the
    # .workflow-group framework in lib/form-schema.css.
    if "workflow_group" in md:
        out["workflow_group"] = md["workflow_group"]

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
        # Tuple[int, int, int] -- kgrid + Transport's k_mesh_transverse.
        # Renderer emits three side-by-side number inputs with sub-ids
        # f"{id}-{labels[i]}".  We pass the labels through so the
        # k-grid UI's "kx / ky / kz" stays declaration-driven.
        out["kind"] = "int-triple"
        out["labels"] = list(md.get("triple_labels", ("x", "y", "z")))
    elif origin in (list, tuple) and args and args[0] is float:
        # Variable-length List[float] -- Transport's bias_voltages_v.
        # No fixed-arity widget makes sense; render as text and let
        # the user enter a comma-separated list.  ``coerce_to_field_type``
        # parses it back into List[float] before the dataclass sees it.
        out["kind"] = "comma-floats"
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


def _serialize_default(f: dataclasses.Field, ann: Any = None) -> Any:
    """JSON-friendly default for the schema.

    Tuples become lists for JSON compatibility.  When the field uses
    ``default_factory`` (Transport's ``bias_voltages_v: List[float]``
    is the first such case), call the factory so the form shows the
    actual default — without this, the form opens with a blank input
    and the user can't see what the production-tuned default is.
    ``ann`` is the already-resolved type hint (per ``hints`` in
    ``_field_to_schema``); used to decide whether a list default
    should serialize as a comma-string (``List[float]`` for the
    comma-floats text input) or stay a list (``List[int]`` for the
    int-triple renderer).
    """
    if f.default is not dataclasses.MISSING:
        v = f.default
    elif f.default_factory is not dataclasses.MISSING:    # type: ignore[misc]
        try:
            v = f.default_factory()                       # type: ignore[misc]
        except Exception:
            return None
    else:
        return None
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, list):
        args = typing.get_args(ann) if ann is not None else ()
        origin = typing.get_origin(ann) if ann is not None else None
        # List[float] -> comma-string (comma-floats text input
        # pre-populates from this).  List[int] -> keep as list (no
        # variable-length int field exposed in the form today, but
        # the contract stays JSON-friendly).
        if origin in (list, tuple) and args and args[0] is float:
            return ", ".join(repr(x) for x in v)
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
    # Sequence[int] (frozen_indices / es_explicit_indices) -- accept
    # comma-separated indices with optional range syntax
    # "0-35, 100, 150-200" -> [0,1,...,35, 100, 150,...,200].  Used by
    # the Spectra tab's frozen-atom + L4 explicit-mode lists.
    if origin in (list, tuple) and args and args[0] is int:
        if isinstance(value, str):
            return _parse_int_list_with_ranges(value)
        if isinstance(value, (list, tuple)):
            # Already a sequence; coerce each element to int.  Reject
            # element-wise rather than silently truncating floats so
            # bad input surfaces.
            return [int(v) for v in value]
        return value
    # Sequence[float] (Transport's bias_voltages_v) -- accept a comma-
    # separated string ("0.0, 0.5, 1.0") or an already-list value.
    # Without this branch the form layer's text-input string passes
    # through unchanged and the dataclass stores it as a str, which
    # then fails downstream when the engine slices ``bias[0]``.
    if origin in (list, tuple) and args and args[0] is float:
        if isinstance(value, str):
            return [float(s.strip()) for s in value.split(",") if s.strip()]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return value
    # Anything else: pass through.
    return value


def _parse_int_list_with_ranges(s: str):
    """Parse ``"0-35, 100, 150-200"`` -> ``[0, 1, ..., 35, 100, 150, ..., 200]``.

    Each comma-separated token is either a bare integer or
    ``<lo>-<hi>`` (inclusive on both ends).  Whitespace around
    commas / hyphens is tolerated.  Empty tokens (trailing comma)
    are skipped.

    Raises ``ValueError`` with the offending token so the caller can
    surface it as a typed error to the user.
    """
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            # Negative-prefix support would be ambiguous with the
            # range separator; indices are 0-based so negatives don't
            # need to be supported here.
            lo_s, _, hi_s = tok.partition("-")
            try:
                lo = int(lo_s.strip())
                hi = int(hi_s.strip())
            except ValueError:
                raise ValueError(
                    f"could not parse index range {tok!r}; "
                    f"expected '<int>-<int>'"
                )
            if hi < lo:
                raise ValueError(
                    f"index range {tok!r} is empty (hi < lo)"
                )
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(tok))
            except ValueError:
                raise ValueError(
                    f"could not parse index {tok!r}; "
                    f"expected an integer"
                )
    return out


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


def apply_sidecar_if_possible(struct, structure_path):
    """Best-effort .molstruct.json sidecar application.

    Sets ``struct.frozen_atoms`` + ``struct.regions`` from the
    sidecar next to ``structure_path`` so engine preflight + emitters
    see the real boundary-condition data.  Cross-blueprint helper:
    used by /api/build/fdf, /api/build/pyscf, and
    /api/spectra/render to honor the contract that /modify's
    boundary-conditions edits flow through to the generators
    (design.md "Sidecar-driven boundary conditions").

    Returns
    -------
    None
      Success (sidecar applied) OR clean no-op (no sidecar, or
      structure_path empty).
    str
      User-facing notice when the sidecar EXISTS but couldn't be
      applied (path rejected, malformed JSON, atom-count mismatch).
      Caller surfaces as a preflight warn-severity Issue so the
      user knows their sidecar didn't take effect.

    Failure to apply is non-fatal because the caller might be
    running against a re-cropped or hand-pasted structure -- the
    form's own freeze/region fields stay authoritative.

    Moved 2026-05-26 from web/blueprints/spectra.py to _shared.py
    because Build's /api/build/fdf + /api/build/pyscf endpoints also
    need to apply the sidecar; importing from spectra blueprint
    created an asymmetric cross-blueprint dependency.
    """
    from .files import _resolve_within_roots, _PickerError
    from molbuilder.parsers import molstruct_json as _molstruct_json
    if not structure_path:
        return None
    try:
        resolved = _resolve_within_roots(structure_path)
    except _PickerError as exc:
        return (f"sidecar lookup skipped: structure_path rejected "
                f"({exc.message}); the form's freeze rules are the "
                f"sole boundary condition for this run.")
    if not resolved.exists():
        return (f"sidecar lookup skipped: {resolved.name!s} not on "
                f"disk; the form's freeze rules are the sole "
                f"boundary condition for this run.")
    if resolved.suffix.lower() not in (".xyz", ".pdb"):
        return None
    sidecar_path = _molstruct_json.sidecar_path_for(resolved)
    if not sidecar_path.exists():
        return None
    try:
        sidecar_data = _molstruct_json.load(sidecar_path)
        _molstruct_json.apply_to_structure(struct, sidecar_data)
    except _molstruct_json.MolstructJsonError as exc:
        return (f"sidecar at {sidecar_path.name} could not be "
                f"applied ({exc}); the form's freeze rules are "
                f"the sole boundary condition for this run.  "
                f"Re-export the sidecar from the Structure tab to "
                f"re-enable sidecar-driven divergence checks.")
    return None


def regions_pattern_b_notice(struct, engine_label: str):
    """Three-stage Pattern B (sidecar-contract.md § 6 B): every
    engine that DOESN'T consume the structure's region labels MUST
    explicitly notice them so the user can re-direct to Transport
    when a junction structure was meant.  Pre-task-#308 this block
    was inlined verbatim in /api/build/fdf and /api/build/pyscf;
    each new engine that joins the cohort (Spectra, future
    transport-but-not-NEGF) would re-implement it or — worse —
    silently absorb the labels.

    Returns
    -------
    None
      ``struct.regions`` is empty; nothing to notice.
    Issue
      ``info``-severity, ``where='config.regions'``, message
      enumerates the labels + names the engine + points at the
      Transport tab.  Caller appends to its issues list and
      returns alongside the rendered script / fdf.

    ``engine_label`` is what the user sees in the message — e.g.
    ``"the .fdf"``, ``"the PySCF script"``, ``"the spectra deck"``.
    Keep the phrasing engine-specific so the notice reads
    naturally.

    Added 2026-06-09 (task #308) as the dedupe of the two
    copy-pasted blocks in build.py.  Test coverage:
    tests/test_web.py::test_{fdf,pyscf}_surfaces_info_when_structure
    _carries_regions.
    """
    from molbuilder.issues import Issue
    if not struct.regions:
        return None
    region_labels = sorted(struct.regions.keys())
    return Issue(
        "info",
        ("Structure carries region labels ("
         + ", ".join(region_labels)
         + ") but a Structure-Optimization run does not consume "
         "them — they are reserved for Transport.  Generating "
         + engine_label + " here is OK if you only want "
         "SCF / relaxation / spectroscopy; for a transport "
         "calculation, switch to the Transport tab."),
        "config.regions",
    )


__all__ = [
    "atoms_list",
    "issues_to_json",
    "struct_from_body",
    "structure_to_dict",
    "ok_structure_response",
    "workspace_payload",
    "err",
    "finite_float",
    "coerce_to_field_type",
    "config_from_params",
    "apply_sidecar_if_possible",
    "regions_pattern_b_notice",
]
