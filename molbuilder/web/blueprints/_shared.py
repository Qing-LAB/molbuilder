"""Helpers shared across the blueprints.

This module is the SINGLE source of truth for:

* the ``Issue`` -> JSON wire shape
* JSON -> Structure body parsing (the structure envelope, ``to_dict``'s shape)
* Structure -> JSON response body construction (the structure sent once)
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
from molbuilder.cell import resolve_and_check
from molbuilder.validation import validate_geometry
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
            f"no {key!r} provided: a structure crosses in the envelope"
            " -- {elements, positions, metadata}")
    return _struct_from_envelope(envelope)


def structure_answer(struct: Structure) -> Dict[str, Any]:
    """THE ONE SHAPE A STRUCTURE IS ANSWERED IN (`web-api.md` § 1; the user,
    2026-11-10: *one shape, sent once*) -- the load, the builds, every edit
    and the cell door alike:

    * ``structure`` -- the structure's own dict (``Structure.to_dict``), the
      one serialiser the sidecar, the persistence layer and the CLI round-trip
      through, so a field added to the structure reaches the wire with no edit
      here, and it crosses once;
    * what the server works out about it that the dict does not carry
      (``Structure.to_wire``): ``resolved_box``, the box as it will be used,
      and ``atom_order_said``, the order its atoms are in, in words;
    * ``labels`` -- the labels a calculation kind reads on a structure's
      atoms, each with what it is for (`kind_table.labels`), the label
      palette's, so MolView spells none of them (`web/molview.md` § 6.2).

    A door adds what the call says beside it -- its notices, and a fact of
    the call no structure holds (the frames a file holds, the backend that
    built it)."""
    from molbuilder.kind_table import labels as _kind_labels
    return {
        "structure": struct.to_dict(),
        **struct.to_wire(),
        "labels":    [{"name": name, "description": about}
                      for name, about in _kind_labels()],
    }


def ok_structure_response(
    struct: Structure,
    *,
    extra: Optional[Dict[str, Any]] = None,
    found: Sequence[Dict[str, Any]] = (),
):
    """Build a Flask jsonify response for any Structure-returning
    endpoint: ``{"ok": True}``, the structure in its one shape
    (:func:`structure_answer`), and what the call says beside it.

    ``/api/build/load`` + ``/api/build/molecule`` + every ``/api/modify/*`` op
    + the cell door route through this helper.  ``found``: what the door finds
    about the result beyond the box's conditions, said after them -- the cell
    door's verdict of what the boundary does to the crystal (``science/
    junction-cell.md`` § 6.1).  ``extra`` carries what the call adds: its
    receipts under ``notices``, and a fact of the call no structure holds --
    ``n_frames`` (``/api/build/load``: the frames the file holds, when one
    was taken) and ``backend_used`` (``/api/build/molecule``).  The
    ``/api/modify/<op>`` routes add none -- the client CLEARS the selection on
    any atom-count change (molview.md § 11.1, "Effect on atom count"), so no
    per-op selection remap is emitted.

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
    said.extend(found)
    # THE GEOMETRY'S ADVICE, WHILE EDITING (`science/overview.md` § 3; the
    # user, 2026-11-10: "show them while editing") -- atoms closer than they
    # should be, an unusual hydrogen count -- said beside every structure a
    # person loads, builds or edits, never blocking; generation enforces.
    said.extend(notices_for_report(validate_geometry(struct)))
    merged = dict(extra or {})
    merged.pop("notices", None)
    return jsonify({"ok": True, **structure_answer(struct), **merged,
                    **({"notices": said} if said else {})})


def err(msg: str, code: int = 400, **fields):
    """THE ONE FAILURE ANSWER every route gives (`web-api.md` § 1):
    ``{ok: false, error: msg, ...fields}`` at ``code`` -- the sentence is the
    refusal, ``fields`` what a page acts on beside it (``findings``,
    ``needsOverwrite`` and ``replaces``, ``actual_mtime`` ...)."""
    return jsonify({"ok": False, "error": msg, **fields}), code


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


def control_for(item) -> str:
    """Which control edits this item -- THE ONE RULE (`form-schema.md`
    § 4): every Build form field is tagged by it (its ``kind``), and Task
    setup's two routes carry it (their ``control``) for the page's cells,
    run card rows and Measure adder (`task-setup.md` § 5.2).  A fixed set of
    choices is a select whatever the setting's type; an optional switch a
    three-way select; anything else by its type."""
    if item.choices:
        return "select"
    if item.type == "bool" and item.optional:
        return "tri-select"
    return _CONTROL_FOR_TYPE.get(item.type, "text")


def item_record(item, *, engine: Optional[str],
                calculation: str) -> Dict[str, Any]:
    """A setting AS EVERY PAGE NAMES AND OFFERS IT, for a ``calculation`` of
    that kind on ``engine`` -- ``None`` on a part every engine of the kind
    shares -- each fact from its one door: the record a form field
    (`form-schema.md` § 1.1) and Task setup's run cards, stage columns and
    shared values (`task-setup.md` § 3, § 5.2, § 6.2b) each start from, and
    add to.  Every key on every record: what an item has none of is said
    ``None``, an empty list or no words."""
    from molbuilder import template as _T
    from molbuilder.kind_table import kind as _kind
    engines = _kind(calculation).engines if engine is None else (engine,)
    return {
        "name":  item.name,
        # ITS LABEL, as every form names it -- its name where it has none
        # (`form-schema.md` § 1a, `task-setup.md` § 5).
        "label": item.label or item.name,
        "help":  item.help,
        "unit":  item.unit or None,
        # THE ONE DEFAULT for this kind (`template.default_of`), and where it
        # comes from in the words a blank leads it with (`default_words`).
        "default":       _jsonable(_T.default_of(item, calculation)),
        "default_words": _T.default_words(item, engine=engine,
                                          calculation=calculation),
        # WHICH CONTROL EDITS IT, the one rule (`control_for`).
        "control": control_for(item),
        # WHAT THIS KIND TAKES on these engines, and nothing else (the
        # catalogue's `offered`, `template.md` § 6.3a): the one set `resolve`
        # and the settings gate refuse by -- every choice either engine
        # takes on a part they share (`offered_by_any`).
        "choices": (list(_T.offered_by_any(item, engines, calculation))
                    if item.choices else None),
        # ...EACH VALUE IN ITS WORDS, a switch *on* / *off*
        # (`template.value_labels`).
        "choice_labels": _T.value_labels(item),
        # WHAT A BLANK IS, in the item's own words, on an optional one
        # (`template.md` § 5).
        "null_label": item.null_label,
        # WHICH RUNGS READ IT (`template.md` § 6.4; empty: every one).
        "stages": list(item.stages),
    }


def catalogue_to_form_schema(engine: Optional[str], id_prefix: str,
                             calculation: str,
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
    (`engines/transport.md` § 3.8.2a): what that rung's overrides may hold
    (`template.why_not_in` -- an item it reads, its own or one any rung may
    set for itself), and the answers it fixes, echoed.  Every field carries
    its ``stages`` so a surface can tell the rung's own items from the ones
    it merely may set.

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

    parsed = _T.catalogue()
    if (surface in ("state", "common")) != (engine is None):
        raise ValueError("the 'state' and 'common' surfaces are every "
                         "engine's, and name none; every other surface names "
                         "its engine")
    # THE ENGINES' SHARED ROWS (§ 6.2a): drawn once, before the engine is
    # chosen -- their electronic state in its own part, beside the answer
    # it is worked into -- and left out of each engine's own part.
    common = _T.shared_by_engines(calculation)
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
    items = [it for it in items if it.carried_by(calculation)]
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
    fixed = _T.fixed_by_role(engine, calculation)
    items = [it for it in items if it.name not in fixed or it.name in echoed]
    if surface == "shared":
        binding = _T.shared_by_every_stage(engine, calculation)
        items = [it for it in items
                 if it.name in binding
                 and (it.group != "setup" or it.name in asks)]
    elif surface == "rung":
        # A RUNG'S TAB OFFERS WHAT A STAGE'S OVERRIDES MAY HOLD -- the one
        # rule of where a value lives (`template.why_not_in`, `template.md`
        # § 5.4): never a run setting (the rung's run card's), never what
        # every stage shares (the panel above), never what the rung fixes,
        # and on a rung's own tab only what that rung reads -- never another
        # rung's.  A locked echo stays, as every echo does (§ 6.6
        # obligation 3).
        items = [it for it in items
                 if it.name in echoed or _T.why_not_in(
                     it, "override", engine=engine, kind=calculation,
                     stage=rung) is None]
    elif surface not in (None, "state", "common", "own"):
        raise ValueError(f"surface must be 'shared', 'rung', 'state', "
                         f"'common', 'own' or None, not {surface!r}")
    if rung is not None and surface != "rung":
        raise ValueError("a rung narrows the 'rung' surface only")
    held = ({h.name: h for h in template.items}
            if template is not None else {})
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        panel = it.category[0] if it.category else "procedure"
        field = _item_to_field(it, id_prefix, calculation, engine=engine)
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
            field["locked"] = {"value": (None if it.per_point_from
                                         else _jsonable(echoed[it.name])),
                               "why": _T.why_role(it.name)}
            # ANSWERED POINT BY POINT (the item's `per_point_from`): no one
            # value to draw, so the page shows the points it holds
            # (`showPoints`).
            if it.per_point_from:
                field["locked"]["per_point"] = True
        # ...AND ON THE SHARED PANEL, WHICH RUNGS ONE SHARED VALUE IS FOR,
        # in the template's words (`template.shared_scope_words`, TD11).
        if surface == "shared" and it.stages:
            field["scope"] = _T.shared_scope_words(it)
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


def _item_to_field(item, id_prefix: str, calculation: str, *,
                   engine: Optional[str] = None) -> Dict[str, Any]:
    """One catalogue item as one form field (`form-schema.md` § 1.1), for a
    ``calculation`` of that kind -- which components of a triple the kind
    fixes is a fact about the kind (`engines/siesta.md` § 6.1); ``engine``
    the form's, whose own default a default may be (``default_words``)."""
    # THE SETTING'S ONE RECORD (`item_record`), its control under the name
    # the renderer draws by, ``kind``, and the form's own keys.
    out: Dict[str, Any] = item_record(item, engine=engine,
                                      calculation=calculation)
    out["kind"] = out.pop("control")
    out.update({
        "id":       f"{id_prefix}-{item.name.replace('_', '-')}",
        "optional": item.optional,
        "tier":     item.tier or "basic",
    })
    if item.pattern:
        out["pattern"] = item.pattern
    # WHAT GREYS IT OUT (`engines/template.md` § 5): the renderer follows
    # the named fields of the same form, its parts drawn together
    # (`form-schema.md` § 1.3).
    if item.active_when:
        from molbuilder.template import active_when_of, active_words
        out["active_when"] = active_when_of(item)
        # ...and what it waits on, in words, shown beside it while greyed.
        words = active_words(item)
        out["active_words"] = words[:1].upper() + words[1:] + "."
    # The engine-keyword badge.  **The FULL spelling, not the anchor**: an
    # anchor is the bare leading keyword (`template.md` § 5), and an
    # engine_key that is a molbuilder note is the only way a reader learns the
    # setting never reaches the deck, which `web/form-schema.md` § 1a requires
    # always be present.  `expands` is the fallback for a `deck` item whose
    # several keywords are the honest answer.
    out["engine_key"] = engine_key_for(item)
    if out["kind"] == "tri-select":
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


def stated_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """WHAT A FORM'S PAYLOAD STATES (`web/form-schema.md` § 1.1): each of
    ``cls``'s fields the payload names, read as its item's declared type by
    the setting door's reader (`template.read_text`, `engines/template.md`
    § 5.4) -- the reader every page asks -- a blank left out: it is not
    chosen.  **A value that will not read as its type is refused, naming
    its field** (``ValueError``): the caller says it as the refusal."""
    by_name = {f.name for f in fields(cls)}
    from molbuilder.template import items_of, read_text
    items = items_of(cls)
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k not in by_name:
            continue
        value, why = read_text(items[k], v)
        if why:
            raise ValueError(f"{k} = {v!r}{why}")
        if value is not None:
            out[k] = value
    return out


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


def for_writing(struct: Structure, *, judged_as=None, engine=None):
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
    what is written stays as given.  ``engine`` -- the hand-over's too --
    the engine its findings are said to (`cell.box_findings_for`)."""
    struct, said = struct.in_user_order("the structure")
    restored = []
    if said is not None:
        from molbuilder.workingcopy_structure import RESTORED, read_notice
        restored = notices_for_report([read_notice(RESTORED, said)])
    _judged, notices = checked_periodicity(
        judged_as(struct) if judged_as is not None else struct, engine=engine)
    return struct, restored + notices


def checked_periodicity(struct, engine=None):
    """Run the gate and let a refusal become the door's 400.

    The one wrapper every entry path uses -- ``periodicity_checked_for_emit``
    below among them -- so none owns a copy of the translation.
    Returns the gate's ``(struct, notices)`` unchanged.
    """
    try:
        return validate_periodicity(struct, engine=engine)
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


def structure_from_text(text: str, filename: str = ""):
    """A STRUCTURE READ FROM A FILE'S TEXT, as an import reads it
    (`model/structure.md` § 2.3) -- the load door's pasted or picked text and
    the files layer's upload: its format by the codec's rule, the name's
    else the text's own (`StructureCodec.format_of_text`), read by the
    codec's one text read
    (`StructureCodec.read_document`: every frame, the title the file's own,
    never an engine record's words); atoms only, an XYZ text a lone file,
    said (`structure.lone_file`).  Returns ``(whole, said)``; a text that
    does not read raises ``ValueError`` in the reader's words."""
    from molbuilder.workingcopy_structure import (LONE_FILE, StructureCodec,
                                                  lone_xyz_said, read_notice)
    fmt = StructureCodec.format_of_text(text, filename)
    try:
        whole = StructureCodec.read_document(text, fmt)
    except Exception as exc:                      # noqa: BLE001 -- the reader's words
        raise ValueError(f"could not parse {fmt}: {exc}") from exc
    said = (notices_for_report([read_notice(
                LONE_FILE, lone_xyz_said(filename or "The pasted text", text))])
            if fmt == "xyz" else [])
    return whole, said


def write_structure(struct: Structure, path, *, overwrite: bool, said=()):
    """THE ONE WAY A WEB DOOR PUTS A STRUCTURE ON DISK (`model/structure.md`
    § 2.4) -- the save, and the upload of a structure file: written as the
    export writes it (:func:`for_writing`: the user order, the box judged; a
    refusal is the door's 400, before anyone is asked to overwrite -- the
    codec's name rule among them, `StructureCodec.refused_name`), then the
    one "already here" answer (`files.exists_answer`), then the codec's pair,
    both or neither.  ``said``: what the read said of a structure read from
    a file's text, answered before the gate's verdicts.  Returns the
    response, ``{ok, path, notices}``."""
    from pathlib import Path
    from molbuilder.workingcopy_structure import StructureCodec
    from .files import _writable, exists_answer
    # THE ONE WRITE-PATH CHECK first: below the root, its folder there, an
    # upload's name, not a folder, not a symlink.
    path = _writable(Path(path))
    refused = StructureCodec.refused_name(path)
    if refused is not None:
        return err(refused, 400)
    struct, notices = for_writing(struct)
    taken = exists_answer(path, overwrite)
    if taken is not None:
        return taken
    try:
        StructureCodec().write(struct, path)
    except ValueError as exc:          # the codec's refusal, in its words
        return err(str(exc), 400)
    except Exception as exc:  # noqa: BLE001 -- disk / permission -> 500
        return err(f"could not write {path.name}: {exc}", 500)
    return jsonify({"ok": True, "path": str(path),
                    "notices": list(said) + list(notices)})

__all__ = [
    "issues_to_json",
    "struct_from_body",
    "ok_structure_response",
    "structure_answer",
    "err",
    "sent_export",
    "finite_float",
    "catalogue_to_form_schema",
    "stated_params",
    "PeriodicityRefused",
    "checked_periodicity",
    "for_writing",
    "structure_from_text",
    "write_structure",
]
