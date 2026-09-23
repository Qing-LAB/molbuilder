"""Transport-calculation blueprint.

The ``/transport-calculation`` tab's server half.  The tab is the
composite's WHOLE describe surface (user ruling 2026-08-29: no
hand-over — nothing is awaiting, so the tab selects and decides):

    GET  /api/transport/schema            form-rendering schema for
                                          the TransportConfig dataclass
    GET  /api/transport/describe_attempt  the slot picker's describe
                                          seam: one line from a cited
                                          attempt's own .fdf, plus the
                                          server-spelled citation and
                                          the calculation's source file
    POST /api/transport/describe          the FINISHED task.json text —
                                          the web spelling of `jobset
                                          init --calculation transport`;
                                          the browser writes it where
                                          the user chose
    POST /api/transport/render            validate + render a device
                                          .fdf from a posted structure
                                          (the engine registry's
                                          validation surface; the
                                          composite renders through
                                          prep, never through this)

Mirrors the contract of Build's ``/api/build/schema/<engine>`` and
Spectra's ``/api/spectra/render``: the dataclass field metadata
is the single source of truth; the JS renders the form directly
from the returned schema; the render endpoint is thin and
dispatches all engine-specific logic through the registry.
"""

from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ._shared import (
    dataclass_to_form_schema as _dataclass_to_form_schema,
    issues_to_json as _issues_to_json,
)

from molbuilder.config.transport import TransportConfig
from molbuilder.units import UnknownUnit
from molbuilder.validation import validate as _validate


bp = Blueprint("transport", __name__)


def _fence(raw: str):
    """A tree-relative path → ``(citation, directory, root, refusal)``.

    ONE fence for both citation doors: contain the path to the projects
    tree, require a directory, and compute the citation string the rest
    of the composite speaks in.  ``refusal`` is a ready answer when
    either test fails and ``None`` when neither did.

    The two doors diverge only AFTER this, and deliberately: describing
    an unciteable directory answers 200 with the condition as its
    summary (that is how a person learns what a citation needs), while
    renaming one is simply a bad request.  Sharing the fence keeps that
    difference visible instead of hiding it inside two copies of the
    same three lines.
    """
    from pathlib import Path

    from molbuilder.projects import OutsideRoot, contain, projects_root
    root = Path(projects_root()).resolve()
    try:
        cite_dir = contain(root / raw, root)
    except OutsideRoot as exc:
        return None, None, root, (
            jsonify({"ok": False, "error": str(exc)}), 400)
    if not cite_dir.is_dir():
        return None, None, root, (
            jsonify({"ok": False,
                     "error": f"{raw} is not a directory"}), 404)
    return str(cite_dir.relative_to(root)), cite_dir, root, None


# ===================================================================== #
# /api/transport/describe_attempt  --  the slot picker's describe seam  #
# ===================================================================== #


@bp.route("/api/transport/describe_attempt", methods=["GET"])
def api_transport_describe_attempt() -> Any:
    """Classify a picked directory against the § 4.1b citation
    condition and describe what it provides — the `describe` seam the
    shared tree-picker feeds on when the Transport tab picks the
    junction slot (P7; reworked 2026-08-29, second user ruling: the
    condition is FILES, never layout).

    ``?path=`` is tree-relative.  The answer names the form the
    directory satisfies ("relaxation" | "structure"), or — when it
    satisfies neither — ``form: null`` with the refusal as the summary,
    naming exactly which file is missing (the condition stated).  For a
    relaxation it is honest about convergence (CONCLUDED / not / no
    record at all), and it says whether the electronic contract is the
    citation's ("cited") or the description's own ("open").
    ``structure`` carries the cited junction's labeled structure for
    the viewer (the /api/build/load ``{structure}`` envelope), so the
    tab shows the citation whatever the form — **and whether or not it
    composes**: a directory that classifies but cannot be built into a
    calculation still answers with its junction, and the refusal is
    appended to the summary above it.  ``fix`` is a word the tab can
    act on (today only ``"swap_electrodes"``), never prose to match.
    """
    from molbuilder.transport.compose import (ComposeError,
                                              classify_citation,
                                              compose_junction,
                                              labeled_citation_structure,
                                              recorded_contract_of)
    from molbuilder.parse.fdf import parse_fdf_params
    from molbuilder.transport.sort import (ORDER_INVERTED,
                                           electrode_orientation)

    raw = str(request.args.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    citation, cite_dir, root, refusal = _fence(raw)
    if refusal is not None:
        return refusal
    try:
        cited = classify_citation(cite_dir)
    except ComposeError as exc:
        # Not citable -- the refusal IS the answer (it names the
        # missing file and states the whole condition).
        return jsonify({"ok": True, "citation": None, "form": None,
                        "summary": str(exc)}), 200

    if cited.form == "structure":
        recorded = recorded_contract_of(cited)
        if recorded is not None:
            # 4.1b's third shade: the pair carries the finished run's
            # own contract, so the lane is CITED, same as a deck.
            contract = "cited"
            status = ("labeled structure · contract RECORDED from the "
                      f"{recorded.get('engine', '?')} deck "
                      f"({recorded.get('source', '?')})")
            # AND WHAT WAS EDITED SINCE.  Two flags, because they
            # invalidate different things (`molview.md` § 8.4a): a geometry
            # or cell op leaves the inherited mesh cutoff and k-mesh
            # converged for a cell that is gone, while a label write leaves
            # the settings standing and moves the electrode/device partition
            # this calculation sorts on.  Said HERE because this line is what
            # a person reads at the moment of choosing; `compose` warns again
            # on the prep path, for a citation nobody picked in a browser.
            edited = []
            if recorded.get("structure_modified"):
                edited.append("geometry/cell EDITED SINCE — the mesh cutoff "
                              "and k-mesh below were converged for a cell "
                              "that is no longer there")
            if recorded.get("labels_modified"):
                edited.append("labels EDITED SINCE — the settings stand, but "
                              "check the electrode and device regions are "
                              "still the partition that was relaxed")
            if edited:
                status += " · " + " · ".join(edited)
        else:
            contract = "open"
            status = "labeled structure (taken as given)"
        params_out = None
        concluded = None
    else:
        deck_text = cited.deck.read_text()
        try:
            p = parse_fdf_params(deck_text)
        except UnknownUnit as exc:
            # DESCRIBED, NOT REFUSED -- the same shape as the refusal
            # above: this route answers with the junction it can see and
            # appends what it could not read.  A deck stating a unit this
            # build cannot convert is a card with one line missing, not a
            # 500 on the tab.
            return jsonify({"ok": True, "citation": citation,
                            "form": cited.form,
                            "summary": f"the deck cannot be read: {exc}"}), 200
        bits = []
        if p.basis_size:
            bits.append(str(p.basis_size))
        if p.mesh_cutoff_ry:
            bits.append(f"{p.mesh_cutoff_ry:g} Ry")
        if p.xc:
            bits.append(str(p.xc))
        if p.kgrid:
            bits.append("k " + "x".join(str(k) for k in p.kgrid))
        if p.n_atoms:
            bits.append(f"{p.n_atoms} atoms")
        if cited.concluded is not None:
            state = f"CONCLUDED ({cited.concluded.strip()})"
        elif cited.has_record:
            state = ("NOT CONCLUDED -- still running, or force-stopped "
                     "(the two look identical on disk)")
        else:
            state = ("no run record -- the .XV is taken as the final "
                     "geometry (convergence unverified)")
        status = state + (" · " + " · ".join(bits) if bits else "")
        params_out = {
            "basis_size": p.basis_size,
            "mesh_cutoff_ry": p.mesh_cutoff_ry,
            "xc": p.xc,
            "kgrid": list(p.kgrid) if p.kgrid else None,
            "n_atoms": p.n_atoms,
        }
        contract = "cited"
        concluded = bool(cited.concluded)

    # TWO SEPARATE QUESTIONS, and the card needs both: *what is this
    # junction* (always answerable from the citation's own files) and
    # *can it be composed into a calculation* (a refusal, sometimes).
    # They used to be one call, so every reason a junction cannot be
    # BUILT -- labels missing, an electrode that moved, a mid-run
    # record, blocks that interleave -- also blanked the viewer, and
    # the refusal was read over an empty card instead of over the thing
    # it is about.
    #
    # The composition answers both when it succeeds (`relaxed` IS the
    # labeled citation structure), so the happy path reads the .XV once
    # and only the refusal path pays for a second look.
    structure_wire = None
    fix = None
    try:
        composed = compose_junction(citation, tree_root=root)
        rel_struct = (composed.relaxed
                      if composed.relaxed is not None
                      else composed.sorted.structure)
        structure_wire = rel_struct.to_dict()
        # THE CONVENTION IS CHECKED AND REPORTED, NEVER ENFORCED (user
        # ruling, 2026-08-29).  An inverted junction composes and runs
        # -- it biases the other end -- so the tab gets the observation
        # (with the numbers, for the meta line) plus `fix` as a WORD it
        # can act on without matching prose.
        for note in composed.sorted.notes:
            status = status + "  ⚠ " + note
        # AND THE LEADS' OWN MEASUREMENTS, which reached nothing until
        # 2026-09-20.  `extract_electrode_model` measures the periodic
        # seam and the principal-layer condition and writes both to
        # `ElectrodeModel.notes` -- and every reader stopped at
        # `sorted.notes`, so the one place that says whether a lead is
        # really bulk was computed and dropped.  Same rule as the line
        # above: checked and reported, never enforced.  Labelled,
        # because "the seam is ECLIPSED" is useless without which end.
        for model in (composed.electrode_left, composed.electrode_right):
            for note in (model.notes if model is not None else ()):
                status = status + f"  ⚠ {model.label}: " + note
        if (electrode_orientation(composed.sorted.structure)
                == ORDER_INVERTED):
            fix = "swap_electrodes"
    except Exception as exc:  # noqa: BLE001 -- surfaced, never fatal
        status = status + "  !! " + str(exc)
        # SHOW IT ANYWAY.  The labels may also be the wrong way round,
        # and the rename is still worth offering on a junction whose
        # refusal is about something else entirely.
        try:
            cited_struct, _src = labeled_citation_structure(cited)
            structure_wire = cited_struct.to_dict()
            if electrode_orientation(cited_struct) == ORDER_INVERTED:
                fix = "swap_electrodes"
        except (ComposeError, OSError, ValueError):
            # Labels that cannot be read leave nothing to draw and
            # nothing to offer.  NARROW on purpose: a blanket except
            # here would swallow a programming error into a silently
            # empty card.
            pass

    return jsonify({
        "ok": True,
        "citation": citation,
        "form": cited.form,
        "contract": contract,
        "concluded": concluded,
        "summary": status,
        "structure": structure_wire,
        "params": params_out,
        "fix": fix,
    })


@bp.route("/api/transport/swap_electrodes", methods=["POST"])
def swap_electrodes():
    """Swap ``L-electrode`` / ``R-electrode`` on a cited junction --
    the fix the person AGREED to after describe offered it.

    It edits their finished run's label block and nothing else (two
    arrays of indices in molbuilder's own metadata; no coordinate, no
    keyword, no result), which is why relabeling does not invalidate
    the relaxation.  Fixed at the source, so every later citation of
    that directory is right too.
    """
    from molbuilder.transport.compose import (ComposeError,
                                              resolve_citation,
                                              swap_electrode_labels)

    body = request.get_json(silent=True) or {}
    raw = str(body.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    citation, _cite_dir, root, refusal = _fence(raw)
    if refusal is not None:
        return refusal
    # RESOLVE AND CLASSIFY THROUGH THE ONE DOOR prep composes through
    # (`resolve_citation`), not a hand-rolled repeat of its three steps
    # with its own wording.  Unlike `describe_attempt` -- which must
    # answer 200 with the refusal as its summary, because describing an
    # unciteable directory is how a person LEARNS the condition -- a
    # rename asked of a directory that is not a citation is simply a
    # bad request.
    try:
        _dir, cited = resolve_citation(citation, root)
        changed = swap_electrode_labels(cited)
    except ComposeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({
        "ok": True,
        "changed": changed,
        "message": (f"Swapped L-electrode and R-electrode in {changed} "
                    f"-- labels only; no coordinate, keyword or result "
                    f"was touched."),
    })


# ===================================================================== #
# /api/transport/describe  --  the tab writes the DESCRIPTION itself   #
# ===================================================================== #


@bp.route("/api/transport/describe", methods=["POST"])
def api_transport_describe() -> Any:
    """Render the transport calculation's COMPLETE ``task.json`` text.

    There is no hand-over for the composite (user ruling 2026-08-29):
    the other kinds hand to Task setup because that tab owns questions
    they cannot answer — shape, stages, what varies.  Transport has
    none open: the five stages and the hierarchical shape are fixed by
    design, the identity derives from the citation, the knobs ride the
    stages' override bags.  So this door answers with the finished
    description, ONE file, and the browser writes it where the user
    chose through the content-blind file layer (`web/projects.md` § 1 —
    the same division of labour as every other tab's writes).

    Validation is the shipped codec's: the ``Task`` construction below
    is the same gate `read_task` and the CLI's ``jobset init`` run, and
    the citation resolves through the same door prep composes through.
    """
    from molbuilder.persist import json_text
    from molbuilder.projects import projects_root
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import Task, derive_run
    from molbuilder.transport.compose import ComposeError, resolve_citation
    from molbuilder import template as _T
    from molbuilder.transport.citation_defaults import (
        siesta_config_from_citation)
    from molbuilder.transport.stages import (CONTRACT_FIELDS,
                                             SEALED_ALWAYS,
                                             TRANSPORT_STAGES,
                                             resolvable_override_names,
                                             stages_for_transport)
    _RESOLVABLE = resolvable_override_names()

    body = request.get_json(silent=True) or {}
    engine = str(body.get("engine") or "siesta").lower()
    if engine != "siesta":
        return jsonify({"ok": False,
                        "error": "transport is SIESTA-first "
                                 "(TranSIESTA)"}), 400
    citation = str(body.get("junction") or "")
    if not citation:
        return jsonify({"ok": False,
                        "error": "no junction citation -- pick the "
                                 "relaxed junction's attempt first"}), 400
    bias_raw = body.get("bias") or [0.0]
    try:
        bias = tuple(float(v) for v in bias_raw)
    except (TypeError, ValueError):
        return jsonify({"ok": False,
                        "error": f"bias must be a list of volts, "
                                 f"got {bias_raw!r}"}), 400
    overrides = body.get("overrides") or {}
    if not isinstance(overrides, dict):
        return jsonify({"ok": False,
                        "error": "overrides must be an object"}), 400
    typed = str(body.get("name") or "") or "transport"

    try:
        _, cited = resolve_citation(citation, projects_root())
    except ComposeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Refused HERE, not at prep on the cluster: an unknown knob or a
    # sealed one names itself while changing it is still free.  Same
    # sets, same conditions as config_for (4.1b: the contract fields
    # are the citation's ONLY when the citation carries a deck).
    import dataclasses as _dc

    # THE VOCABULARY IS WHAT PREP CAN RESOLVE, and it is asked rather than
    # listed (`stages.resolvable_override_names`).  This door checked
    # `TransportConfig` alone, which refused a person's own lead k-density
    # (`electrode_kz` is a catalogue row); widening it to the UNION with
    # `SiestaConfig` fixed that and overshot in the other direction, letting
    # through three names prep then refuses -- so "Described" succeeded and
    # every later prep failed, naming a field the person never typed.
    for _name in overrides:
        if _name not in _RESOLVABLE:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is not a parameter this "
                                     f"calculation can carry: `prep` "
                                     f"resolves each rung against the "
                                     f"SIESTA schema, and either no field "
                                     f"of it has that name or it is a "
                                     f"machine fact the description must "
                                     f"never carry (engines/template.md "
                                     f"7)."}), 400
        if _name in SEALED_ALWAYS:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is the description's "
                                     f"own field (identity, bias) -- "
                                     f"it is never an override"}), 400
        if _name in CONTRACT_FIELDS:
            # SHARED, therefore not a per-stage override -- and that is the
            # reason, not "the citation owns it".
            #
            # This said *"cite a relaxation that ran with the values you
            # want"* until 2026-09-16, which was the SEALED reading and by
            # then actively misleading advice: it told a person to redo a
            # relaxation when they could edit one line of the template.
            # `engines/transport.md` § 2a.7 ruled that the cited run
            # DEFAULTS these values; what remains true is that they are
            # shared by every rung, so giving ONE rung its own would let the
            # device disagree with its own leads -- the single thing that
            # must be impossible.
            return jsonify({"ok": False,
                            "error": f"{_name!r} is shared by every stage "
                                     f"of this calculation, so it cannot "
                                     f"be a per-stage override: the "
                                     f"electrode and the device must not "
                                     f"be able to disagree about it.  "
                                     f"Change it in the calculation's "
                                     f"template ({_name} there applies to "
                                     f"all five rungs at once) -- it was "
                                     f"filled in from the run you cited, "
                                     f"and it is yours to change."}), 400
    try:
        task = Task(
            engine="siesta", shape="hierarchical",
            run=derive_run(typed, citation,
                           stage_names=TRANSPORT_STAGES),
            structure=None, calculation="transport",
            slots={"junction": citation}, bias=bias,
            # the stages.md 6.2 rule holds here too: an override names
            # a PROMOTED field, and `varies` is the promotion
            varies=tuple(sorted(overrides)),
            # ROUTED TO THE RUNG THAT OWNS EACH ONE (`engines/template.md`
            # § 6.4's `stages` declaration).  Every override went onto the
            # `device` rung until 2026-09-16, whatever it was -- so a
            # person's T(E) window was written into the deck `siesta` runs,
            # where the keyword is inert, and not into the deck `tbtrans`
            # runs, which is the one that computes T(E).  Silently: you
            # asked for ±3 eV and got the default.
            stages=tuple(stages_for_transport(overrides)))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({
        "ok": True,
        "label": task.label,
        # TWO FILES, and the second was missing until 2026-09-16.
        #
        # A transport description is `task.json` AND a template, like every
        # other kind's (TR1).  The CLI's `jobset init` wrote both; this door
        # returned only the first, so a description made in the browser had
        # no shared electronic description and `prep` refused it by name.
        # The regression was mine and the tests did not catch it because
        # they build a description through the fixture rather than through
        # this endpoint -- the door a person actually uses.
        #
        # Its values are DEFAULTED FROM THE CITED RUN (§ 2a.7, ruling 1) and
        # are the person's to change afterwards.
        "files": [{"name": TASK_FILENAME,
                   "text": json_text(task.to_dict())},
                  {"name": _T.template_filename(task.label),
                   "text": _T.template_with_values(
                       siesta_config_from_citation(cited.path,
                                                   label=task.label),
                       engine="siesta", calculation="transport")}],
        "notices": [],
    })


# ===================================================================== #
# /api/transport/schema  --  form schema endpoint                       #
# ===================================================================== #


@bp.route("/api/transport/schema", methods=["GET"])
def api_transport_schema() -> Any:
    """Return the transport TAB's form schema: the transport-only knobs.

    The electronic contract (engine, basis, XC, mesh, temperature, the
    transverse k, the bias, the label) is the CITATION's to say — it
    arrives from the cited junction's own deck at prep, and the
    describe door refuses those fields BY NAME.  A form field the door
    is guaranteed to refuse is a trap, not a control (found rendered
    2026-08-29: ten sealed fields sat as editable inputs, and the bias
    was asked twice), so the sealed set is filtered HERE, from the same
    one constant the two refusing doors read.  What remains IS the
    override lane: Transmission / NEGF / Runtime knobs that ride the
    device stage's bag (stages.md § 6.2).  The bias is card 4's own
    input — a describe-level fact beside the citation, not a config
    override.

    Section order still follows ``TransportConfig._form_section_order``;
    sections the filter empties (System, Electrodes) are dropped whole.
    """
    from molbuilder.transport.stages import (CONTRACT_FIELDS,
                                             SEALED_ALWAYS,
                                             UNRESOLVED_FIELDS,
                                             resolvable_override_names)
    # HIDDEN IN BOTH LANES, and the `?contract=` argument no longer
    # changes anything here.
    #
    # It used to: `cited` hid the contract fields and `open` offered them,
    # because a form-B citation (a labeled pair, no deck) had nowhere else
    # to state a basis.  On 2026-09-16 the describe door stopped making
    # that distinction -- `engines/transport.md` § 2a.7 ruled the cited run
    # DEFAULTS these values into the calculation's TEMPLATE, which every
    # form now gets, so they are never a per-stage override for anyone.
    #
    # The filter was not updated with the door, and for a few hours the
    # `open` lane rendered seven controls the door was guaranteed to
    # refuse -- precisely the trap this docstring says the filter exists
    # to prevent.  The argument is kept only so an older page that still
    # sends it is not a 400; it selects nothing.
    # ⚠ THIS FILTER IS MEASURED DEAD IN SEVEN OF ITS TEN BRANCHES, and the
    # swap that replaced it was REVERTED because it opened a worse hole.
    # Both halves are recorded in `plans/plan.md` W28; do not re-attempt
    # either without reading it.
    #
    # DEAD: `SEALED_ALWAYS`'s three names and four of `CONTRACT_FIELDS`'
    # seven are `TransportConfig` spellings, and `resolvable_override_names`
    # one line below rejects them first -- so those branches cannot fire.
    # LIVE HOLE, the same coin: the catalogue's own spellings for four of
    # those values (`mesh_cutoff`, `kgrid`, `pao_energy_shift`,
    # `electronic_temperature`) pass BOTH guards here and are refused by
    # `prep`, which is the "Described succeeded, every later prep failed"
    # trap this docstring says the filter exists to prevent.
    #
    # WHY THE CATALOGUE SWAP WAS REVERTED (2026-09-23): the catalogue's
    # narrowing offers `system_label`, `species_order`, `spin_treatment`
    # and `spin_total`, none of which carries the `citation` marker, so a
    # `skip_shared` filter built on that marker let them through as
    # PER-RUNG overrides.  `route_overrides` sends them to the device,
    # where a `system_label` breaks the ladder's file handover and a
    # `species_order` gives the device one orbital ordering and the leads
    # another -- the disagreement `model/chemistry.md` § 3a exists to make
    # impossible.  The root is that there is NO MARKER FOR "SHARED":
    # `citation` means "a cited run answers this" and Class A (§ 2a.13) is
    # larger than that.  W28 is where that is decided.
    hidden = set(SEALED_ALWAYS) | UNRESOLVED_FIELDS | set(CONTRACT_FIELDS)
    # AND EVERY CONTROL PREP CANNOT RESOLVE.  The three sets above are the
    # SEALED question -- what a person may not change.  This is the different
    # one: what the description is able to CARRY.  `num_threads`, `log_level`
    # and `max_memory_mb` are `TransportConfig` fields the schema has no row
    # for, so the door accepted them and every later prep refused the whole
    # calculation.  A control the door is guaranteed to reject is not a
    # control, and it is the same trap this filter already exists to close.
    import dataclasses as _dcs
    _resolvable = resolvable_override_names()
    hidden |= {f.name for f in _dcs.fields(TransportConfig)
               if f.name not in _resolvable}
    schema = _dataclass_to_form_schema(TransportConfig, "t")
    kept = []
    for sec in schema.get("sections", []):
        fields_left = [f for f in sec.get("fields", [])
                       if f.get("name") not in hidden]
        if fields_left:
            sec = dict(sec)
            sec["fields"] = fields_left
            kept.append(sec)
    schema = dict(schema)
    schema["sections"] = kept
    response: Dict[str, Any] = {"ok": True, "schema": schema}
    return jsonify(response)


# `POST /api/transport/render` DELETED 2026-09-17, and with it
# `_transport_config_from_params`, which built the `TransportConfig` the
# route coerced its form values into and had no other caller.
#
# It rendered a device deck through `TransiestaEngine.render_script` and
# handed the text back as JSON.  No browser called it: `lib/transport/core.js`
# stopped POSTing here on 2026-08-29 and the tab has fetched `/describe`,
# `/schema`, `/describe_attempt` and `/swap_electrodes` ever since.  It was
# the last caller of that renderer, and the renderer was a second writer of a
# deck the framework already writes -- one that read a different config class,
# so the pole-energy correction of 2026-09-16 never reached it and a deck from
# this route stopped SIESTA before the SCF loop.
#
# A BROWSER RENDERS NO DECK (`tabs.md`): a deck is rendered by `jobset prep`
# from a description, which is why `/api/build/fdf` and `/api/build/pyscf`
# went the same way on 2026-08-17.  The preflight this route also ran is not
# lost -- `validation` reaches it through the engine registry, which is the
# path the Generate button already used.
