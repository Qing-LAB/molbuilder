"""Helpers shared across the blueprints.

This module is the SINGLE source of truth for:

* the ``Issue`` -> JSON wire shape
* JSON -> Structure body parsing (xyz + per-atom metadata lists)
* Structure -> JSON response body construction
* JSON -> dataclass coercion (used by Build for SiestaConfig /
  PySCFConfig form values)

If a helper is genuinely blueprint-specific (e.g. Build's
``/api/build/load`` accepts both multipart and JSON, Modify's body
parsing always carries the canonical state bundle), it stays in
the calling blueprint.  Promote here when at least two callers
need the same behaviour and drift would silently break wire
contracts.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import typing
from dataclasses import fields
from typing import Any, Dict, List, Optional

from flask import jsonify

from molbuilder.structure import (
    Structure,
)
from molbuilder.validation import validate_geometry
from molbuilder.cell import resolve_and_check
from molbuilder.periodicity_gate import (notices_for_report,
                                         validate_periodicity)



# --------------------------------------------------------------------- #
#  Issues                                                                #
# --------------------------------------------------------------------- #


def resolve_workflow_group(where: str, cfg) -> Optional[str]:
    """Return the workflow-group binding for an Issue ``where`` field.

    Per docs/web/ui-contract.md Rule 2, validator findings
    should attach to the workflow-group card whose fields they
    concern: a ``config.mesh_cutoff`` finding belongs in the Stage
    card; ``config.spin_treatment`` belongs in the Run profile card;
    ``config.max_scf_iter`` belongs in the Budget card.  The mapping is
    each dataclass field's ``metadata["workflow_group"]`` -- the class's
    copy of the catalogue's ``group``, which the form draws its cards by;
    `test_every_mirrored_fact_agrees` keeps the two equal
    (`web/form-schema.md` § 1a).

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
        d = i.to_json()          # THE key set lives on `Issue`
        # An Issue may pre-tag its workflow_group; if not, derive
        # from the where field via the config dataclass metadata.
        group = i.workflow_group
        if group is None and cfg is not None:
            group = resolve_workflow_group(i.where, cfg)
        if group is not None:
            d["workflow_group"] = group
        # The stage label rides BESIDE ``where``, never inside it
        # (engines/stages.md § 4 R2): the same check produces the same id
        # whether it fired for a single run or for a stage, and the UI binds
        # behaviour to the id.  Omitted when absent.
        if i.stage is not None:
            d["stage"] = i.stage
        out.append(d)
    return out


# --------------------------------------------------------------------- #
#  JSON <-> Structure  (canonical Modify body shape, also reusable for  #
#  any future endpoint that takes "xyz + per-atom metadata" arrays)     #
# --------------------------------------------------------------------- #


def _struct_from_envelope(env: Dict[str, Any]) -> Structure:
    """The inverse, and the same rule: ``Structure.from_dict`` is the ONE
    deserialiser, and it validates through the same ``__post_init__`` a freshly
    built Structure runs -- so a malformed envelope is refused here rather than
    becoming a half-built structure downstream.

    ``source_index`` -- present when the envelope describes a SUBSET of a larger
    structure -- is the CALLER's bookkeeping for mapping an answer back onto the
    structure the subset came from.  The receiver answers about the atoms it was
    given and has no use for it, so it is ignored rather than rejected.
    """
    if not isinstance(env, dict):
        raise ValueError("'structure' must be an object")
    # WHAT THE ENVELOPE IS, checked by membership.  A key outside this set is a
    # fact the sender believes it transmitted -- refused rather than dropped.
    known = {"title", "elements", "positions", "atom_names", "residue_ids",
             "residue_names", "chain_ids", "metadata",
             "info",              # the free-form NON-structural store
                                  # (from_dict reads it)
             "source_index",      # the CALLER's map back onto a larger structure
             "document"}          # outbound only; a request's is ignored
    stray = sorted(k for k in env if k not in known)
    if stray:
        raise ValueError(
            f"structure carries {stray!r}, which the envelope does not define "
            f"(known: {sorted(known)!r}).  Metadata belongs under 'metadata'.")
    # AN EMPTY STRUCTURE IS A STRUCTURE: the slab op places from ABSOLUTE
    # coordinates -- dx, dy and start_z from the world origin -- so it needs
    # no atoms to build onto.
    #
    # `elements` MISSING is still refused: that is a malformed envelope, and
    # telling it apart from a deliberately empty one is the whole distinction.
    if env.get("elements") is None:
        raise ValueError("structure.elements is required (may be an empty list)")
    if not isinstance(env["elements"], list):
        raise ValueError("structure.elements must be a list")
    if not isinstance(env.get("positions"), list):
        raise ValueError("structure.positions must be a list of [x, y, z]")

    return Structure.from_dict(env)


def struct_from_body(body: Dict[str, Any], key: str = "structure") -> Structure:
    """Reconstruct a Structure from THE ENVELOPE, which is the only shape
    (`web-api.md` § 1, "The request envelope")::

        {"structure": {"elements": [...], "positions": [[x, y, z], …],
                       "metadata": {"regions": {...}, "cell": [...], ...}}}

    FLAT, with the per-atom facts beside the atoms and everything else under
    ``metadata`` -- the atoms as NUMBERS, and the facts beside them, so a
    caller holding coordinates never has to write a coordinate document to ask
    a question about them.  It is exactly what ``molview``'s
    ``structureForServer`` emits, which is the only shape any caller sends.

    **A body carries ``structure``, or it is refused** -- by name, naming the
    shape it wanted.

    ``key`` names WHICH envelope in the body to read, and exists for the one
    route that takes two: ``/api/modify/append`` is handed the structure being
    built in AND the one being added to it.  It is the same envelope shape read
    by the same reader either way -- a second parameter, not a second door --
    because two envelopes in a body must not become two ideas of what an
    envelope is.
    """
    envelope = body.get(key)
    if not isinstance(envelope, dict):
        raise ValueError(
            f"no {key!r} provided: a structure crosses in the envelope "
            "(web-api.md § 1) -- {elements, positions, metadata}")
    return _struct_from_envelope(envelope)


def atoms_list(struct: Structure) -> List[Dict[str, Any]]:
    """Build the per-atom payload list.

    Used by every response that carries a Structure so the front-end's
    selection store stays in sync with the in-memory geometry without
    a separate fetch.

    Each row:

        {
            "index":         int,
            "element":       "C" | "H" | ...,
            "x": float, "y": float, "z": float,   # COORDS -- the atom carries its
                                                  # own geometry (web-api.md § 1: the
                                                  # atoms as NUMBERS); no re-parse
            "regions":       [str, ...],     # EVERY label the atom carries,
                                             # reserved ones (`frozen`) included --
                                             # one representation, so the panel
                                             # cannot render the same fact twice
            # NO IDENTITY COLUMNS.  atom_names / residue_names / chain_ids
            # travel at the TOP level of the payload, beside `metadata` --
            # `structure.py::IDENTITY_FIELDS`.
        }
    """
    n = len(struct.elements)
    atom_to_regions: Dict[int, list] = {}
    regions = getattr(struct, "regions", {}) or {}
    for label, idxs in regions.items():
        for idx in idxs:
            atom_to_regions.setdefault(idx, []).append(label)

    positions = struct.positions

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        # Coordinates ride ON the atom (`web-api.md` § 1 -- the atoms as
        # numbers, so nobody has to write a coordinate document to ask a
        # question about them; the atom is the geometric truth, not a
        # re-parsed xyz string).
        pos = positions[i]
        row: Dict[str, Any] = {
            "index":     i,
            "element":   struct.elements[i],
            "x":         float(pos[0]),
            "y":         float(pos[1]),
            "z":         float(pos[2]),
            "regions":   atom_to_regions.get(i, []),
        }
        rows.append(row)
    return rows


def workspace_payload(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The canonical wire shape for every endpoint that returns a
    ``Structure``.

    Adding a new field that every
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
    ``add_hydrogens_mode``, ``pdb``, ``summary``, …) belong in
    ``extra``.  The dispatcher
    on the client reads the canonical keys and treats ``extra`` as
    opaque metadata — extending ``extra`` does not break any
    consumer.

    Notes
    -----
    * ``source_format`` defaults to ``"xyz"`` because
      :class:`molbuilder.structure.Structure` round-trips through
      :meth:`Structure.to_xyz`.  PDB-emitting endpoints set
      ``source_format="pdb"`` via ``extra`` plus a ``"text"``
      override at the callsite.
    * ``lattice`` is always ``None`` here, and NOT because the structure
      has no cell -- it has one: ``cell`` / ``engine_offset`` /
      ``axis_kind`` / ``vacuum`` travel in the ``periodicity`` block that
      :func:`structure_to_dict` takes from ``struct.to_wire()``, together
      with the resolved views -- one block, so a cell cannot half-arrive.
      ``lattice`` is a single-field spelling that no consumer reads; it
      stays for the wire shape's sake.
    * ``issues`` is populated via :func:`validate_geometry`.
    """
    return {
        "text":          struct.to_xyz(),
        "source_format": "xyz",
        "title":         struct.title or "",
        "n_atoms":       struct.n_atoms,
        "atoms":         atoms_list(struct),
        "lattice":       None,
        # ``cfg=None`` explicit (not implicit default) so the missing-cfg case is documented at the
        # call site.  ``validate_geometry`` emits only ``where=
        # "struct.*"`` issues -- no engine config field is in
        # scope here -- so workflow_group enrichment correctly
        # short-circuits to None.  A future refactor that moves
        # engine-config validation upstream of this helper MUST
        # pass cfg= or the per-card fan-out (web-ui-coherence
        # Rule 2) silently drops engine issues into the residual
        # panel.
        "issues":        issues_to_json(
            validate_geometry(struct), cfg=None),
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

    The optional ``extra`` dict threads endpoint-specific keys (``pdb``, ``summary``,
    ``backend_used``, ``add_hydrogens_mode``) into BOTH places at
    once:

    * At the top level of the returned dict, for every JS consumer
      that reads them off the response root.
    * In the canonical ``extra`` sub-dict.

    Top-level ``extra`` keys override the canonical defaults — a
    caller emitting ``source_format="pdb"`` from a PDB-parsing
    endpoint replaces the canonical default of ``"xyz"`` at both
    the root level and inside ``extra``.

    One :func:`validate_geometry` pass per call (issues are read
    from the workspace payload; not recomputed in this helper).
    """
    extras = dict(extra) if extra else {}
    base = workspace_payload(struct, extra=extras)
    # The drift-prone block -- the full `periodicity` (the raw cell and offset
    # + the server-resolved cell and box corner) + `annotations` + the identity
    # columns -- is assembled by the Structure itself (`struct.to_wire()`,
    # `model/structure.md` § 2.1 + § 4).  This helper hand-lists no metadata
    # field and re-runs no resolver, so a field added to `metadata_to_dict`
    # rides onto every endpoint automatically.  The web layer only adds its
    # OWN concerns (render `atoms`, `issues`, `text`/`xyz`, `extra`).
    wire = struct.to_wire()
    return {
        # THE ENVELOPE (web-api.md § 1) is the structure's OWN canonical dict --
        # not a wire shape assembled here. `to_dict` is the one serialiser the
        # sidecar, the persistence layer and the CLI already round-trip through,
        # and its rule is that nobody outside the class picks a structure apart,
        # because that is where a field goes missing. So a
        # field added to the structure reaches the wire with no edit in this file.
        #
        # It sits BESIDE the keys below rather than replacing them, and both are
        # derived from this one Structure, so they cannot come to disagree. The
        # legacy keys go when nothing reads them -- a question the code can answer.
        "structure":     struct.to_dict(),
        # Canonical keys (forward-compat with workspace dispatcher).
        "text":          base["text"],
        "source_format": base["source_format"],
        "title":         base["title"],
        "n_atoms":       base["n_atoms"],
        "atoms":         base["atoms"],
        "lattice":       base["lattice"],
        # Structure-owned: full periodicity (incl. resolved_cell, box_corner) +
        # annotations ride with the geometry into the store so a captured
        # electrode cell survives the modify op (`web-api.md` § 1, what
        # the envelope must be able to carry).
        "periodicity":   wire["periodicity"],
        "annotations":   wire["annotations"],
        "issues":        base["issues"],
        "extra":         base["extra"],
        # Legacy aliases for existing modify-tab consumers (identity columns
        # also sourced from the ONE view so they can't diverge).
        #
        # They stay: the envelope was "added not swapped" on purpose
        # (tests/test_structure_envelope_protocol.py), and
        # `test_a_response_carries_the_envelope_beside_todays_keys` guards each
        # one by name.
        "xyz":           base["text"],
        "elements":      wire["elements"],
        "atom_names":    wire["atom_names"],
        "residue_ids":   wire["residue_ids"],
        "residue_names": wire["residue_names"],
        "chain_ids":     wire["chain_ids"],
        "n_residues":    wire["n_residues"],
        # Endpoint-specific keys at the top level for the JS consumers
        # that read them off the response root.
        **extras,
    }


def ok_structure_response(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
):
    """Build a Flask jsonify response for any Structure-returning
    endpoint.

    ``/api/build/load`` + ``/api/build/molecule`` + every
    ``/api/modify/*`` op route through this helper.  The optional ``extra``
    dict carries per-endpoint add-ons:

    * ``/api/build/load``: ``{"pdb", "summary", "source_format"}``
      (``source_format`` overrides the canonical XYZ default with
      the actually-parsed format).
    * ``/api/build/molecule``: ``{"pdb", "summary",
      "backend_used", "add_hydrogens_mode"}``.
    * ``/api/modify/<op>``: no ``extra`` keys — the client CLEARS
      the selection on any atom-count change (molview.md § 11.1,
      "Effect on atom count"), so no per-op selection remap is emitted.

    Wraps :func:`structure_to_dict` (which routes through
    :func:`workspace_payload`) in ``{"ok": True, ...}``.

    EVERY STRUCTURE LEAVING FOR THE BROWSER IS CHECKED HERE, and this is the
    only place it can be done once (structure-periodicity.md § 8.1).

    Note what does NOT need doing here.  In the DERIVED regime the cell is a
    computed view -- ``resolve_cell`` builds it from the bounding box and the
    vacuum on every read -- so it follows the atoms by construction and there is
    nothing to regenerate.  An EXPLICIT cell is returned verbatim, which is the
    whole point of it, and is exactly the case where moving atoms can put them
    outside.  So the op runs, the cell answers for itself, and the check reports.

    ``extra["notices"]`` is the RECEIPTS slot, and it is kept ahead of what is
    computed here for the order the cell door already uses: what the edit did
    first, what is now true after it (molview.md § 6.8).  The merge is not
    decoration: without it, the assignment below would drop a caller's receipts
    without a word, which is the failure this whole helper exists to make
    impossible.
    """
    said = list((extra or {}).get("notices") or [])
    # THE ONE LINE (cell-plan.md § 6a): resolve once, check once, report.
    #
    # No try/except, because there is nothing to catch: these are the
    # loading/modifying doors, and § 8.2 says they REPORT a bad box rather than
    # refusing it -- so they ask the checker directly instead of calling the
    # raising gate and reconstructing a notice from the exception.
    _rc, issues = resolve_and_check(struct)
    said.extend(notices_for_report(issues))
    merged = dict(extra or {})
    if said:
        merged["notices"] = said
    return jsonify({"ok": True, **structure_to_dict(struct, extra=merged or None)})


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
#  The form schema, built from the CATALOGUE                            #
#                                                                       #
#  `web/form-schema.md` § 1: the catalogue is the source of truth.  The  #
#  presentation does not change -- the JS renderer already takes         #
#  whatever schema it is handed -- so this is one function pointed at a  #
#  different source, not a new UI.                                       #
# --------------------------------------------------------------------- #

#: How a template `type` becomes a control (§ 1.1's derived column).
_CONTROL_FOR_TYPE = {
    "bool":    "checkbox",
    "int":     "int",
    "pow2":    "int",          # an int with a constraint the validator holds
    "float":   "number",
    "str":     "text",
    "text":    "text",
    "int3":    "int-triple",
    "float3":  "float-triple",
    # A list renders as a TEXT input holding a comma-separated value, which
    # ``coerce_to_field_type`` parses back.  No new control kind: the
    # renderer has one for this shape already.
    "strlist": "text",
    "intlist": "text",
}



def _control_for(item) -> str:
    """Which widget renders this item.  `choices` wins: an enum is a select
    whatever its underlying type, and a tri-select is an OPTIONAL bool."""
    if item.choices:
        return "select"
    if item.type == "bool" and item.optional:
        return "tri-select"
    return _CONTROL_FOR_TYPE.get(item.type, "text")


def catalogue_to_form_schema(engine: str, id_prefix: str = "p",
                             calculation: str = "optimization",
                             surface: Optional[str] = None,
                             rung: Optional[str] = None,
                             template=None,
                             ) -> Dict[str, Any]:
    """The Build form's schema for *engine*, from the catalogue.

    ``template`` is the calculation's own (a parsed `Template`), when the
    surface is drawn from one: each field then carries the item's ``value``
    and its ``source`` (`form-schema.md` § 1.1, `engines/template.md` § 6.6
    obligation 2).  A value nobody chose is not carried -- the field is blank,
    its ``default`` the hint.

    ``rung`` narrows the ``"rung"`` surface to ONE rung's tab
    (`engines/transport.md` § 3.8.2a): the items whose `stages` declaration
    names that rung, plus the items declaring no rung, which any rung may
    set for itself.  Every field carries its ``stages`` so a surface can
    tell the rung's own items from the ones it merely may set.

    ``surface`` is a composite kind's second axis (`engines/transport.md`
    § 3.8.2): ``"shared"`` is the panel that edits the TEMPLATE -- every item
    the catalogue marks `shared` for this kind, except the `setup` group,
    whose two members (the identity, the pseudopotential directory) the
    description and the citation answer themselves -- and ``"rung"`` is the
    per-rung form, which never offers a shared item.  ``None`` is every
    surface at once, the ordinary kinds' one form.

    **The two grouping axes** (`form-schema.md` § 1.3), both carried by every
    item and answering different questions:

    * ``group`` -- *when do I set this?* -- is the OUTER card.
    * ``category`` -- *what question about the calculation is this?* -- is the
      legend INSIDE the card.  The six are shared, so SIESTA and PySCF show the
      same inner headings.

    Sections come out in § 6.2's reading order, which is the order the closed
    vocabulary is declared in -- not alphabetical, and not the order the items
    happen to sit in the file.
    """
    from molbuilder import template as _T

    parsed = _T.catalogue()
    items = _T.select(parsed, engine=engine)

    # ``staging`` is a PANEL THIS SURFACE DOES NOT HAVE.  The stage token is a
    # real parameter -- it reaches the generated script, and the template
    # carries it -- but it is answered by the staging surface, not typed here
    # (user, 2026-08-15: *"no staging related setup at all"*).  Filtered by the
    # item's own declaration rather than by a name this file would have to
    # keep: a second such parameter needs no edit here.
    items = [it for it in items if it.group != "staging"]
    # The form serves ONE calculation kind (`template.md` § 6.3's sibling
    # rule): an item another kind owns stays out by its own declaration.
    # The vibration form is the same renderer over the same catalogue.
    items = [it for it in items
             if not it.calculations or calculation in it.calculations]
    # THE KIND'S OWN RECOMMENDATION stands in for the general default on a
    # form built for that kind (template.md § 6.3a): the hint a blank field
    # shows, and what it is written at when nobody chose (form-schema.md
    # § 1.1).
    items = [_T.with_recommended(it, calculation) for it in items]
    # A ROLE ITEM IS NOT A FORM FIELD (`template.md` § 6.4, ruled
    # 2026-09-16): "offering any of these presents a choice with exactly one
    # correct answer, and a person who changed it would not be tuning the
    # run -- they would be stopping it being the run it is."
    #
    # AND IT IS PER KIND, like `calculations` and for the same reason.
    # `solution_method` is the RUNG's business for transport -- a device
    # solves with NEGF and a bulk lead does not -- and an ordinary choice
    # for an optimization, where nothing else decides it.  So the test is
    # membership of THIS calculation, never "has a role at all": filtering
    # on the latter would take a legitimate control off the Build form.
    #
    # ...BUT IT IS SHOWN (§ 6.6 obligation 3, plan § 5w K7): read-only, at
    # the answer the rung gives and why.  A form for one rung echoes that
    # rung's answers; a form for the whole calculation the ones every rung
    # gives alike -- an item answered rung by rung has no one answer there.
    # The shared panel carries the shared values alone.
    if surface == "shared":
        echoed: Dict[str, Any] = {}
    elif rung is not None:
        echoed = _T.role_answers(engine, calculation, rung)
    else:
        echoed = _T.fixed_on_every_rung(engine, calculation)
    items = [it for it in items
             if calculation not in it.role or it.name in echoed]
    if surface == "shared":
        items = [it for it in items
                 if calculation in it.shared and it.group != "setup"]
    elif surface == "rung":
        items = [it for it in items if calculation not in it.shared]
    elif surface is not None:
        raise ValueError(f"surface must be 'shared', 'rung' or None, "
                         f"not {surface!r}")
    if rung is not None:
        if surface != "rung":
            raise ValueError("a rung narrows the 'rung' surface only")
        # THE RUNG'S TAB: its own items, and the ones any rung may set --
        # never another rung's (those are on that rung's tab) and never a
        # shared one (the panel above).
        items = [it for it in items if not it.stages or rung in it.stages]
    held = ({h.name: h for h in template.items}
            if template is not None else {})
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        panel = it.category[0] if it.category else "procedure"
        field = _item_to_field(it, id_prefix, calculation)
        # THE TEMPLATE'S VALUE, AND WHOSE IT IS (form-schema.md § 1.1): a
        # value somebody gave -- a cited run, the structure's record, the
        # person -- or one written before sources were recorded.  A value
        # nobody chose stays off: the field is blank, *not chosen*.
        h = held.get(it.name)
        if h is not None and h.is_set and h.source != "default":
            field["value"] = _jsonable(h.value)
            field["source"] = h.source
        if it.name in echoed:
            field["locked"] = {"value": (None if it.name in _T.PER_POINT
                                         else _jsonable(echoed[it.name])),
                               "why": _T.why_role(it.name)}
        # WHICH RUNGS OWN IT (template.md § 6.4's `stages`; empty = any):
        # the fact a rung's tab folds its cards by, and the stage table
        # disables cells by.
        field["stages"] = list(it.stages)
        # WHAT THIS KIND CAN TAKE ON THIS ENGINE, and nothing else (the
        # catalogue's `offered`, `template.md` § 6.3a): the one set `resolve`
        # and the settings gate refuse by, so the form never offers what prep
        # would refuse -- no dynamics for a vibration's relaxation, no
        # restricted-open on SIESTA, no floating moment on PySCF.
        if it.choices:
            field["choices"] = list(_T.offered(it, engine, calculation))
        by_category.setdefault(panel, []).append(field)

    sections = [{"name": cat, "title": cat.capitalize(),
                 "fields": by_category[cat]}
                for cat in _T.CATEGORIES if cat in by_category]

    # A stage ladder is NOT here, and that is correct: `stages.md` § 1.1 makes
    # it the user's decision about what VARIES, and it lives in ``task.json``.
    # The catalogue carries parameters; how a ladder is set up is its own
    # design conversation (user, 2026-08-14).
    # The words a field's source is said in -- one vocabulary, the
    # template's (`template.SOURCE_WORDS`), so no surface coins its own.
    return {"config": engine, "id_prefix": id_prefix, "sections": sections,
            "source_words": dict(_T.SOURCE_WORDS)}


def _jsonable(v: Any) -> Any:
    """A template value as JSON carries it -- a tuple as a list."""
    return list(v) if isinstance(v, tuple) else v


def engine_key_for(item) -> str:
    """How this item is spelled for the engine — the string a SURFACE shows.

    **One writer for every surface.**  An ``anchor`` is DERIVED —
    ``_bare_anchor`` takes the leading token of ``engine_key`` and nothing
    checks that the token is a keyword — so for an item whose ``engine_key``
    leads with a VALUE, the anchor is that value.

    The order is the honest one: the full spelling if the item has it, then
    the keywords a ``deck`` item expands to, and the bare anchor only when
    there is nothing better (`template.md` § 5).
    """
    if getattr(item, "engine_key", ""):
        return item.engine_key
    if getattr(item, "expands", ()):
        return " + ".join(item.expands)
    return getattr(item, "anchor", "") or ""


def _item_to_field(item, id_prefix: str,
                   calculation: str = "optimization") -> Dict[str, Any]:
    """One catalogue item as one form field (`form-schema.md` § 1.1), for a
    ``calculation`` of that kind -- which components of a triple the kind
    fixes is a fact about the kind (`engines/siesta.md` § 6.1)."""
    out: Dict[str, Any] = {
        "name":     item.name,
        "id":       f"{id_prefix}-{item.name.replace('_', '-')}",
        "label":    item.label or item.name.replace("_", " ").capitalize(),
        "help":     item.help,
        "default":  (list(item.default) if isinstance(item.default, tuple)
                     else item.default),
        "optional": item.optional,
        "tier":     item.tier or "basic",
        "kind":     _control_for(item),
    }
    if item.unit:
        out["unit"] = item.unit
    if item.pattern:
        out["pattern"] = item.pattern
    if item.group:
        out["workflow_group"] = item.group
    # The engine-keyword badge.  **The FULL spelling, not the anchor**: an
    # anchor is the bare leading keyword (`template.md` § 5), and an
    # engine_key that is a molbuilder note is the only way a reader learns the
    # setting never reaches the deck, which `web/form-schema.md` § 1a requires
    # always be present.  `expands` is the fallback for a `deck` item whose
    # several keywords are the honest answer.
    spelled = engine_key_for(item)
    if spelled:
        out["engine_key"] = spelled
    if item.choices:
        out["choices"] = list(item.choices)
    elif out["kind"] == "tri-select":
        # An Optional[bool] has three states and the renderer walks
        # ``f.choices`` to build them; they are the CONTROL's vocabulary, not
        # the item's, so the catalogue does not carry them (§ 5's `choices` is
        # an enum's members).
        out["choices"] = ["auto", "true", "false"]
    if item.range:
        out["min"], out["max"] = item.range
    if out["kind"] in ("int", "number"):
        out["step"] = "1" if item.type in ("int", "pow2") else "any"
    if out["kind"] in ("int-triple", "float-triple"):
        # THE K-POINT MESH'S AXIS NAMES -- the ones `fixed` below is keyed
        # by, so a lock cannot miss its cell (`kmesh.AXES`).
        from molbuilder.kmesh import AXES
        out["labels"] = list(AXES)
        # A triple gets a step too: a bound or a step chosen in the renderer
        # is a second place for the rule to live.  `min`/`max` are already set above from `item.range` and
        # apply PER COMPONENT for a triple: `kgrid` bounds each axis count,
        # not their product.
        out["step"] = "1" if item.type == "int3" else "any"
        # THE COMPONENTS THIS KIND FIXES, drawn locked with their reason
        # (`kmesh.fixed`, `engines/siesta.md` § 6.1): a transport
        # calculation's third k component, which no rung reads.  The value
        # still travels, so the template states it; every door refuses
        # another one (`template.why_not`).
        from molbuilder.kmesh import fixed
        held = fixed(item.name, calculation)
        if held:
            out["fixed"] = {AXES[i]: {"value": v, "why": why}
                            for i, (v, why) in held.items()}
    if item.optional:
        out["null_option"] = True
        out["null_label"] = item.null_label or "(auto)"
    # DRAWN REQUIRED ON THE KINDS THAT DECLARE IT (`template.Item.required`,
    # plan § 5w K20): the value may still be blank, and the Send refuses
    # until it is answered -- the pseudopotential folder on SIESTA.
    if calculation in item.required:
        out["required"] = True
    if item.refs:
        # RESOLVED here, server-side, from the one bibliography
        # (molbuilder/references.py) -- the form shows a real title and
        # DOI, never a bare key.  An unknown key is silently omitted
        # HERE because the test suite is where it must fail
        # (tests/test_catalogue_refs.py); the form is not the CI.
        from molbuilder.references import citation_for
        cites = [c for c in (citation_for(k) for k in item.refs) if c]
        if cites:
            out["refs"] = cites
    return out


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
    ``"0"`` (case-insensitive).  A Tuple-typed field like ``kgrid`` takes
    a list OR the comma text a person types (``"4,4,1"``, ``"4x4x1"``,
    ``"4 4 1"`` -- the spellings ``--kgrid`` itself takes), and coerces
    per component.

    Unknown / unhandled types pass through untouched -- the dataclass
    constructor sees what the caller sent.

    A value it cannot make is refused with a ``ValueError`` that names the
    field -- a fractional count among them (:func:`_as_number`).
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

    # AN ENUM'S VALUE IS ONE OF ITS MEMBERS, WITH THE MEMBER'S TYPE
    # (`engines/template.md` § 5): `unpaired_electrons` is 0-10 or "free",
    # and a client that sends the text "2" means the member 2.  Matched on
    # the member's own spelling, so "free" stays a word and "2" becomes a
    # number; a value naming no member is refused here, by name.
    choices = field.metadata.get("choices")
    if choices:
        from molbuilder.template import is_member
        if is_member(value, choices):
            return value
        for c in choices:
            if not isinstance(value, bool) and str(c) == str(value).strip():
                return c
        raise ValueError(f"{field.name} = {value!r} is not one of "
                         f"{', '.join(map(repr, choices))}")

    if ann is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)
    if ann in (int, float):
        return _as_number(field.name, value, ann)
    if ann is str:
        return str(value)
    # Tuple[int, int, int] (kgrid, tbt_k_grid) and Tuple[float, float, float]
    # (kgrid_displacement).
    #
    # A COMMA STRING PARSES, for the same reason the Sequence[*] branches
    # just below accept one: a non-browser client sends the text a person
    # would type.
    #
    # A value that is neither a string nor a sequence is REFUSED here rather
    # than passed through, with a ValueError naming the field -- what this
    # function promises the caller for a value it cannot make.  The LENGTH
    # is not checked -- that is `_validate_kgrid`'s sentence to pass, and it
    # says it better.
    if origin is tuple and args:
        elem_t = args[0]
        if isinstance(value, str):
            value = [s for s in re.split(r"[,\sx]+", value.strip()) if s]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field.name} = {value!r} is not a list of "
                             f"values")
        return tuple(_as_number(field.name, v, elem_t)
                     if elem_t in (int, float) else elem_t(v)
                     for v in value)
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
            # Already a sequence; each element read as a whole number,
            # refused element-wise.
            return [_as_number(field.name, v, int) for v in value]
        return value
    # Anything else: pass through.
    return value


def _as_number(name: str, value: Any, typ: type) -> Any:
    """``value`` read as ``typ`` -- ``int`` or ``float`` -- or refused,
    naming the field (`web/form-schema.md` § 1.1).  A count with a fraction
    is not a count, and is never rounded: ``int(4.5)`` gave 4, a k-point
    count of 4.5 a mesh of four (the K3 review)."""
    try:
        if isinstance(value, bool):
            raise ValueError
        x = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} = {value!r} is not a number") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} = {value!r} is not a finite number")
    if typ is int:
        if not x.is_integer():
            raise ValueError(f"{name} = {value!r} is not a whole number -- "
                             f"a count is never rounded")
        return int(x)
    return x


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
                       calculation: str = ""):
    """The config a form's payload describes (`web/form-schema.md` § 1.1):
    the kind's own recommendations (`template.apply_recommended`), with the
    values the payload holds laid over them, each coerced to its field's
    declared type.

    **A blank is not chosen**, for every field: ``None`` or ``""`` leaves the
    field at what lies under it -- the kind's recommendation, else the class
    default, which for an optional field is its own blank.

    **A value that will not read as its type is refused, naming its
    field** (``coerce_to_field_type``): the caller says it as the refusal.
    """
    by_name = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for k, v in params.items():
        f = by_name.get(k)
        if f is None or v is None or v == "":
            continue
        kwargs[k] = coerce_to_field_type(f, v, hints)
    base = cls()
    if calculation:
        from molbuilder.template import apply_recommended
        base = apply_recommended(base, calculation)
    return dataclasses.replace(base, **kwargs)


def siesta_config_from_params(params: Dict[str, Any],
                              calculation: str = ""):
    """A ``SiestaConfig`` from a form's payload, keyed by catalogue names,
    each value coerced to its field's declared type (R5), over the kind's
    recommendations (:func:`config_from_params`).  ONE home for the doors
    that take a SIESTA form -- the hand-over, the preflight, the chemistry
    card and the transport describe door's shared panel -- so a blank, a
    comma-typed tuple and a number typed as text mean the same thing on
    every tab."""
    from molbuilder.config.siesta import SiestaConfig
    return config_from_params(SiestaConfig, params, _siesta_hints(),
                              calculation)


@functools.lru_cache(maxsize=1)
def _siesta_hints() -> Dict[str, Any]:
    from molbuilder.config.siesta import SiestaConfig
    return typing.get_type_hints(SiestaConfig)


class PeriodicityRefused(Exception):
    """The gate REFUSED the periodicity a request carried.

    Not a warning about a box: a state that cannot be represented at all -- a
    left-handed cell, or one too small to hold the structure whatever origin it
    is given (periodicity_gate, "Errors vs notices").  The user has to change
    something before the request can be answered, which is what a 400 means.

    It exists so that answer cannot be forgotten.  ``validate_periodicity``
    raises ``ValueError``, and every door that runs it on the way IN would
    have to remember a try/except.  Raising a type ONE handler in
    ``web/app.py`` knows about means a door inherits the right answer instead
    of inheriting the omission.  (Same reasoning, same
    file, as the 413 handler beside it.)
    """


def checked_periodicity(struct):
    """Run the gate and let a refusal become the door's 400.

    The one wrapper every entry path uses -- ``periodicity_checked_for_emit``
    below among them -- so none owns a copy of the translation.
    Returns the gate's ``(struct, notices)`` unchanged.
    """
    try:
        return validate_periodicity(struct)
    except ValueError as exc:
        raise PeriodicityRefused(str(exc)) from exc


def periodicity_checked_for_emit(struct):
    """REFUSE a bad box -- the emitting doors.  Checks; applies nothing.

    Returns the CHECKED structure -- the same object the gate was given, since
    the gate corrects nothing (clause 1: a resolved value is never written
    back).  Callers rebind so this stays the one seam every emitted structure
    passes through rather than an optional check.  A refusable cell raises
    :class:`PeriodicityRefused`, which the app turns into the door's 400.

    THE BOX ARRIVES IN THE ENVELOPE AND NOWHERE ELSE: a structure crosses
    once (web-api.md § 1), so a second place to say what the box is would be
    two sources silently ranked.
    """
    checked, _conditions = checked_periodicity(struct)
    return checked


__all__ = [
    "atoms_list",
    "issues_to_json",
    "struct_from_body",
    "structure_to_dict",
    "ok_structure_response",
    "workspace_payload",
    "err",
    "finite_float",
    "catalogue_to_form_schema",
    "coerce_to_field_type",
    "config_from_params",
    "PeriodicityRefused",
    "checked_periodicity",
]
