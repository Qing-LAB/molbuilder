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
import math
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence

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


def card_of_finding(where: str, cfg, calculation: str) -> Optional[str]:
    """The card a finding about ``where`` is shown on -- the card its field
    is drawn in on a ``calculation``'s form (`template.card_of`,
    `engines/template.md` § 6.2a): the field's result card where it names
    one for the kind, else its group's.  One rule for the form and for the
    finding, so a warning always sits beside its field
    (`web/ui-contract.md` § 5.1).

    ``None`` for a finding not about a config field (``where`` outside
    ``config.`` -- geometry, cell, polymer: the residual panel), and for a
    name no catalogue item of the config's engine holds.
    """
    if not where or not where.startswith("config."):
        return None
    if not dataclasses.is_dataclass(cfg):
        return None
    # "config.net_charge.makov_payne" -> "net_charge".
    field_name = where.split(".", 1)[1].split(".", 1)[0]
    from molbuilder.template import card_of, items_of
    item = items_of(cfg).get(field_name)
    return card_of(item, calculation) if item is not None else None


def issues_to_json(issues, cfg=None, *, calculation: str = ""):
    """Serialise List[Issue] for the JSON wire.

    The web client reads ``issues[].severity / message / where / card /
    stage`` to decide how to display (`science/validation.md` R1).

    ``cfg`` is the engine config dataclass and ``calculation`` the kind the
    form is drawn for: given both, each finding carries the ``card`` its
    field is drawn in (:func:`card_of_finding`), so the page puts it beside
    the field.  A config without its kind is refused -- the card depends on
    it -- and with no config the key is omitted: the finding goes to the
    residual panel.
    """
    if cfg is not None and not calculation:
        raise ValueError("issues_to_json: a config's findings are placed by "
                         "the kind's cards -- state the calculation")
    out = []
    for i in issues:
        d = i.to_json()          # THE key set lives on `Issue`
        card = (card_of_finding(i.where, cfg, calculation)
                if cfg is not None else None)
        if card is not None:
            d["card"] = card
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
             "frames",            # a frame set's coordinates, IN PLACE OF
                                  # `positions` (model/structure.md § 2.1)
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
    # THE COORDINATES: `positions`, or for a frame set `frames` -- one or the
    # other, as `to_dict` writes them (`Structure.from_dict` refuses both).
    if env.get("frames") is not None:
        if not isinstance(env["frames"], list):
            raise ValueError("structure.frames must be a list of frames, each "
                             "a list of [x, y, z]")
    elif not isinstance(env.get("positions"), list):
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
        # ``cfg=None`` explicit: ``validate_geometry`` emits only
        # ``where="struct.*"`` findings, which no card holds -- the
        # residual panel's.
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
        # The order the atoms are in, in words, worked out by the structure
        # like the resolved box (`model/overview.md` § 2.2a): what a viewer
        # holding an engine's order marks itself with; none in the user order.
        "atom_order_said": wire["atom_order_said"],
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
    # THE ONE LINE (archive/2026-08-20-cell-plan.md § 6a): resolve once, check once, report.
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


def sent_export(got):
    """An export (`ncdata.Exported`) as one attachment -- its data file, or
    its files as one zip -- what it holds without said in the response's
    ``X-Molbuilder-Said`` header: every export route's answer (the
    transport report's and the spectrum viewer's)."""
    import io
    from flask import send_file
    body, name, mime = got.download()
    out = send_file(io.BytesIO(body), mimetype=mime, as_attachment=True,
                    download_name=name)
    if got.said:
        out.headers["X-Molbuilder-Said"] = got.said
    return out


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
    # the setting door's reader parses back (`template.read_text`).  No new
    # control kind: the
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


def catalogue_to_form_schema(engine: Optional[str], id_prefix: str = "p",
                             calculation: str = "optimization",
                             surface: Optional[str] = None,
                             rung: Optional[str] = None,
                             template=None,
                             asks: Sequence[str] = (),
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

    ``surface`` says which part of the kind's settings this form draws, on
    a page that draws them in parts.  Across a composite kind's rungs
    (`engines/transport.md` § 3.8.2): ``"shared"`` is the panel that edits
    the TEMPLATE -- every item the catalogue marks `shared` for this kind,
    except the `setup` group, whose two members (the identity, the
    pseudopotential directory) the description and the citation answer
    themselves -- save ``asks``, what the citation asks of the person
    (`citation_defaults.asked_of_the_person`: a pair's pseudopotential
    directory) -- and ``"rung"`` is the per-rung form, which never offers a
    shared item.  Across the engines a kind runs on (`engines/template.md`
    § 6.2a): ``"state"`` and ``"common"`` -- ``engine`` ``None`` -- the
    settings one row serves on every engine of the kind
    (`template.shared_by_engines`), each offering every choice any of them
    takes: ``"state"`` their electronic state (`template.STATE_ITEMS`), the
    Spectrum tab's chemistry card's, ``"common"`` the rest; and ``"own"``
    one engine's settings without them, its name before every card title.
    ``None`` is every surface at once, the ordinary kinds' one form.

    **The grouping axes** (`form-schema.md` § 1.3), answering different
    questions:

    * the CARD -- `template.card_of`: the item's result for the kind, else
      its ``group``, *when do I set this?* -- is the OUTER card;
    * ``category`` -- *what question about the calculation is this?* -- is
      the legend INSIDE a group's card.  The six are shared, so SIESTA and
      PySCF show the same inner headings.

    Sections come out in § 6.2's reading order, which is the order the closed
    vocabulary is declared in -- not alphabetical, and not the order the items
    happen to sit in the file.
    """
    from molbuilder import template as _T
    from molbuilder.kind_table import kind as _kind

    parsed = _T.catalogue()
    if (surface in ("state", "common")) != (engine is None):
        raise ValueError("the 'state' and 'common' surfaces are every "
                         "engine's, and name none; every other surface names "
                         "its engine")
    # THE ENGINES' SHARED ROWS (§ 6.2a): drawn once, before the engine is
    # chosen -- their electronic state in its own part, beside the answer
    # it is worked into -- and left out of each engine's own part.
    common = _T.shared_by_engines(calculation)
    engines = _kind(calculation).engines if engine is None else (engine,)
    items = _T.select(parsed, engine=engine)
    if surface == "state":
        items = [it for it in items
                 if it.name in common and it.name in _T.STATE_ITEMS]
    elif surface == "common":
        items = [it for it in items
                 if it.name in common and it.name not in _T.STATE_ITEMS]
    elif surface == "own":
        items = [it for it in items if it.name not in common]

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
    if surface in ("shared", "state", "common"):
        echoed: Dict[str, Any] = {}
    elif rung is not None:
        echoed = _T.role_answers(engine, calculation, rung)
    else:
        echoed = _T.fixed_on_every_rung(engine, calculation)
    items = [it for it in items
             if calculation not in it.role or it.name in echoed]
    if surface == "shared":
        items = [it for it in items
                 if calculation in it.shared
                 and (it.group != "setup" or it.name in asks)]
    elif surface == "rung":
        # A RUNG'S TAB OFFERS NO RUN SETTING: those are the rung's run card's
        # (`engines/stages.md` § 6.8d, `template.run_settings`, the marker);
        # a locked echo of one stays, as every echo does (§ 6.6 obligation
        # 3).  Asked here, so no blueprint keeps a second filter (2026-10-18).
        _run = _T.run_settings(engine)
        items = [it for it in items if calculation not in it.shared
                 and (it.name not in _run or it.name in echoed)]
    elif surface not in (None, "state", "common", "own"):
        raise ValueError(f"surface must be 'shared', 'rung', 'state', "
                         f"'common', 'own' or None, not {surface!r}")
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
        # WHAT THE CITATION ASKS OF THE PERSON is drawn required, as the
        # kinds that declare an item are: shown and required by one answer
        # (`citation_defaults.asked_of_the_person`; `engines/transport.md`
        # § 3.1, "required for a pair").
        if it.name in asks:
            field["required"] = True
        # THE TEMPLATE'S VALUE, AND WHOSE IT IS (form-schema.md § 1.1): a
        # value somebody gave -- a cited run, the structure's record, the
        # person -- or one written before sources were recorded.  A value
        # nobody chose stays off: the field is blank, *not chosen*, its hint
        # the value the template holds -- what runs.
        h = held.get(it.name)
        if h is not None and h.is_set and h.source != "default":
            field["value"] = _jsonable(h.value)
            field["source"] = h.source
        elif h is not None and h.is_set:
            field["default"] = _jsonable(h.value)
        if it.name in echoed:
            field["locked"] = {"value": (None if it.name in _T.PER_POINT
                                         else _jsonable(echoed[it.name])),
                               "why": _T.why_role(it.name)}
            # ANSWERED POINT BY POINT (`template.PER_POINT`): no one value to
            # draw, so the page shows the points it holds (`showPoints`).
            if it.name in _T.PER_POINT:
                field["locked"]["per_point"] = True
        # WHICH RUNGS OWN IT (template.md § 6.4's `stages`; empty = any):
        # the fact a rung's tab folds its cards by, and the stage table
        # disables cells by.
        field["stages"] = list(it.stages)
        # ...AND ON THE SHARED PANEL, WHICH RUNGS ONE SHARED VALUE IS FOR,
        # in the template's words (`template.shared_scope_words`, TD11).
        if surface == "shared" and it.stages:
            field["scope"] = _T.shared_scope_words(it)
        # WHAT THIS KIND CAN TAKE ON THIS ENGINE, and nothing else (the
        # catalogue's `offered`, `template.md` § 6.3a): the one set `resolve`
        # and the settings gate refuse by, so the form never offers what prep
        # would refuse -- no dynamics for a vibration's relaxation, no
        # restricted-open on SIESTA, no floating moment on PySCF.
        if it.choices:
            field["choices"] = list(_T.offered_by_any(it, engines, calculation))
        # THE CARD IT IS DRAWN IN (`template.card_of`, § 6.2a) -- the rule
        # a finding about it is placed by too.
        field["card"] = _T.card_of(it, calculation)
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
    # AND THE CARDS, in drawing order with their words, from their one home
    # (`template.cards`: GROUP_WORDS and RESULTS, form-schema.md § 1.3); a
    # form of one rung's own values names the rung and says once whose its
    # values are (`template.stage_words`).
    # A CARD SAYS WHAT ITS FIELDS ON THIS FORM ARE (`template.card_line`):
    # the Setup card's by its fields, the Run profile's by whether it holds
    # the electronic state; a form showing none of a card's draws no card.
    shown_in: Dict[str, List[str]] = {}
    for sec in sections:
        for f in sec["fields"]:
            shown_in.setdefault(f["card"], []).append(f["name"])
    out = {"config": engine or "common", "id_prefix": id_prefix,
           "sections": sections,
           "source_words": dict(_T.SOURCE_WORDS),
           # A LIST, in drawing order (`template.cards`): the order IS the
           # cards' order, and a JSON object's keys are sorted on the wire.
           # Each card's line true of the fields this form draws in it
           # (`template.card_line`).
           "cards": [{"card": c.key, "title": c.title,
                      "subtitle": (_T.card_line(c.key, shown_in[c.key])
                                   if c.key in shown_in else c.subtitle),
                      "legends": c.legends}
                     for c in _T.cards(calculation)]}
    # A FORM THAT NAMES ITS CARDS -- one rung's own values, one engine's own
    # settings -- says whose they are before every card title.
    if rung is not None:
        out["card_prefix"] = rung
        out["stage_words"] = _T.stage_words(rung)
    elif surface == "own":
        from molbuilder.engine_table import engine as _engine
        out["card_prefix"] = _engine(engine).words
    return out


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
    # WHAT GREYS IT OUT (`engines/template.md` § 5): the renderer follows
    # the named fields of the same form, its parts drawn together
    # (`form-schema.md` § 1.3).
    if item.active_when:
        out["active_when"] = {k: list(v) for k, v in item.active_when}
        # ...and what it waits on, in words, shown beside it while greyed.
        from molbuilder.template import active_words
        words = active_words(item)
        out["active_words"] = words[:1].upper() + words[1:] + "."
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
        # ...each with the words a person reads (`choice_labels`).
        if item.choice_labels:
            out["choice_labels"] = {str(c): w for c, w in item.choice_labels}
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


def config_from_params(cls, params: Dict[str, Any],
                       calculation: str = ""):
    """The config a form's payload describes (`web/form-schema.md` § 1.1):
    the kind's own recommendations (`template.apply_recommended`), with the
    values the payload holds laid over them, each read as its item's
    declared type by the setting door's reader (`template.read_text`,
    `engines/template.md` § 5.4) -- the reader every page asks.

    **A blank is not chosen**, for every field: ``None`` or ``""`` leaves the
    field at what lies under it -- the kind's recommendation, else the class
    default, which for an optional field is its own blank.

    **A value that will not read as its type is refused, naming its
    field**: the caller says it as the refusal.
    """
    by_name = {f.name for f in fields(cls)}
    from molbuilder.template import items_of, read_text
    items = items_of(cls)
    kwargs: Dict[str, Any] = {}
    for k, v in params.items():
        if k not in by_name:
            continue
        value, why = read_text(items[k], v)
        if why:
            raise ValueError(f"{k} = {v!r}{why}")
        if value is not None:
            kwargs[k] = value
    base = cls()
    if calculation:
        from molbuilder.template import apply_recommended
        base = apply_recommended(base, calculation)
    return dataclasses.replace(base, **kwargs)


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


def for_writing(struct: Structure, *, judged_as=None):
    """``(struct, notices)``: a structure the browser sent, as a web door
    writes it for a person -- IN THE USER ORDER (``model/overview.md`` § 2.2a,
    I1): one in an engine's order is put back (``Structure.in_user_order``)
    and said in a ``structure.restored`` notice naming the order in its
    words -- and ITS BOX JUDGED (:func:`checked_periodicity`), a refusal the
    door's 400, the verdicts said after the restore.  THE ONE SEQUENCE the
    save, the export and the hand-over run, so a save, a download and a
    calculation's structure cannot disagree about what is written or
    refused.  ``judged_as`` -- the hand-over's -- is what the gate judges in
    the structure's place: the box as prep will place it (`cell.entering`);
    what is written stays as given."""
    struct, said = struct.in_user_order("the structure")
    restored = []
    if said is not None:
        from molbuilder.workingcopy_structure import RESTORED, read_notice
        restored = notices_for_report([read_notice(RESTORED, said)])
    _judged, notices = checked_periodicity(
        judged_as(struct) if judged_as is not None else struct)
    return struct, restored + notices


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
    "sent_export",
    "finite_float",
    "catalogue_to_form_schema",
    "config_from_params",
    "PeriodicityRefused",
    "checked_periodicity",
    "for_writing",
]
