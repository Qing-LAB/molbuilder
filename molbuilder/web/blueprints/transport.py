"""Transport-calculation blueprint.

The ``/transport-calculation`` tab's server half.  The tab is the
composite's WHOLE describe surface (user ruling 2026-08-29: no
hand-over — nothing is awaiting, so the tab selects and decides):

    GET  /api/transport/schema            the tab's two surfaces, the
                                          shared panel and a rung's form,
                                          from the catalogue narrowed to
                                          the kind
    GET  /api/transport/describe_attempt  the slot picker's describe
                                          seam: one line from what the
                                          citation records -- a run or a
                                          pair -- what its
                                          composition measured as
                                          findings, the server-spelled
                                          citation, the junction for the
                                          viewer and `fix`
    GET  /api/transport/record            a calculation's transport
                                          record, COMPOSED ON READ
                                          (`engines/transport.md`
                                          § 2a.12): every rung's state
                                          and every point's, as they are
                                          now
    GET  /api/transport/pdos              the PDOS of selected atoms at
                                          one point -- a frame and a
                                          voltage -- by orbital type
    POST /api/transport/describe          writes the FINISHED description
                                          into the chosen folder -- the
                                          web spelling of `jobset init
                                          --calculation transport`

Mirrors Build's ``/api/build/schema/<engine>``: the form is drawn from the
catalogue (`catalogue_to_form_schema`), and every deck renders through
`prep`, never through a route here.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ._shared import (
    catalogue_to_form_schema,
    issues_to_json as _issues_to_json,
)



bp = Blueprint("transport", __name__)


def _fence(raw: str):
    """A tree-relative path → ``(citation, path, root, refusal)``: the path
    contained to the projects tree, required to exist, and the citation
    string the rest of the composite speaks in.  ``refusal`` is a ready
    answer when either test fails and ``None`` when neither did -- a bad
    request, where a path that exists and is not citable answers 200 with
    the condition as its summary (that is how a person learns what a
    citation needs)."""
    from pathlib import Path

    from molbuilder.projects import OutsideRoot, contain, projects_root
    root = Path(projects_root()).resolve()
    try:
        cited = contain(root / raw, root)
    except OutsideRoot as exc:
        return None, None, root, (
            jsonify({"ok": False, "error": str(exc)}), 400)
    if not cited.exists():
        return None, None, root, (
            jsonify({"ok": False, "error": f"{raw} does not exist"}), 404)
    return str(cited.relative_to(root)), cited, root, None


# ===================================================================== #
# /api/transport/describe_attempt  --  the slot picker's describe seam  #
# ===================================================================== #


@bp.route("/api/transport/describe_attempt", methods=["GET"])
def api_transport_describe_attempt() -> Any:
    """Classify a picked path -- a run's folder or a pair's ``.xyz`` --
    against `engines/transport.md` § 3.1's citation condition and describe
    what it provides — the `describe`
    seam the shared tree-picker feeds on when the Transport tab picks
    the junction slot (user ruling: the condition is what the path holds,
    never where it sits).

    ``?path=`` is tree-relative: a finished relaxation run's folder, or a
    structure pair's ``.xyz`` (§ 3.1).  A path that is neither answers
    ``citation: null`` with the refusal as the summary, naming what is
    missing (the condition stated).  A citation's ``summary`` is one
    line: how a run ended (CONCLUDED / not / exit code), or a pair's
    frames and the deck its record names, then what its recorded
    contract states.  What its composition
    MEASURED rides ``findings`` -- rows of the one finding vocabulary
    (`science/validation.md` § 4.1 R2): each lead's seam and
    principal-layer measurements as ``info``, the electrode orientation
    as ``warn``, a composition refusal as ``error`` -- so the tab
    renders them through the one renderer, never a run-on string.
    ``structure`` carries the cited junction's labeled structure for
    the viewer (the /api/build/load ``{structure}`` envelope), so the
    tab shows the citation **whether or not it composes**: a directory
    that classifies but cannot be built into a calculation still
    answers with its junction, and the refusal is a finding beside it.
    ``fix`` is a word the tab can act on (today only
    ``"swap_electrodes"``), never prose to match.
    """
    from molbuilder.transport.compose import (CitedPair, ComposeError,
                                              classify_citation,
                                              compose_junction,
                                              labeled_citation_structure)
    from molbuilder.frame_average import answer as frames_answer
    from molbuilder.issues import Issue
    from molbuilder.transport.sort import (ORDER_INVERTED,
                                           electrode_orientation)

    raw = str(request.args.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    # THE RENAME THE TAB HOLDS (`engines/transport.md` § 4): described WITH
    # it, so the viewer and the meta line show the junction as this
    # calculation will compose it; the cited run is read, never written.
    swap = str(request.args.get("swap_electrodes") or "").lower() in (
        "1", "true", "yes")
    citation, cited_path, root, refusal = _fence(raw)
    if refusal is not None:
        return refusal
    try:
        cited = classify_citation(cited_path)
    except ComposeError as exc:
        # Not citable -- the refusal IS the answer (it names what is
        # missing and states the whole condition).
        return jsonify({"ok": True, "citation": None,
                        "summary": str(exc), "findings": []}), 200

    # WHAT ITS RECORD SAYS (§ 3.1): a cited pair's `info.calculation`; a
    # cited run's, its view as the Results tab reads it (`runs.view_of`) --
    # and what is not there is said, never filled in.
    if isinstance(cited, CitedPair):
        record = cited.calculation
        n_atoms = cited.structure.n_atoms
        state = (f"STRUCTURE PAIR · {cited.n_frames} frame"
                 f"{'' if cited.n_frames == 1 else 's'} · optimized with "
                 f"{record.get('source') or 'a deck its record does not name'}")
        concluded = True
    else:
        from molbuilder.runs import declared, view_of
        said = declared(cited.run)
        record = (view_of(cited.run).info or {}).get("calculation")
        n_atoms = (said.atom_metadata or {}).get("n_atoms_total")
        # HOW IT ENDED AND WHAT IT CONVERGED (`engines/transport.md` § 3.1):
        # said where a person chooses, so a geometry that did not converge
        # is cited knowingly -- what it hands over is then the last geometry
        # SIESTA wrote.
        if cited.concluded is None:
            state = ("NOT CONCLUDED -- still running, or force-stopped "
                     "(the two look identical on disk)")
        elif cited.exit_code != 0:
            state = (f"ENDED WITH EXIT CODE {cited.exit_code} "
                     f"({cited.concluded.strip()}) -- not citable until the "
                     f"relaxation runs to its end")
        else:
            state = f"CONCLUDED ({cited.concluded.strip()})"
            if cited.converged:
                state += " · " + cited.converged
                if cited.verdict == ("geometry", False):
                    state += (" -- its last geometry is the last one SIESTA "
                              "wrote, not a converged minimum")
        if record is None:
            state += (f" · its deck {said.deck.name if said.deck else ''} "
                      f"states no settings this build reads")
        concluded = bool(cited.concluded)
    c = (record or {}).get("contract") or {}
    bits = [str(c["basis_size"])] if c.get("basis_size") else []
    if c.get("mesh_cutoff"):
        bits.append(f"{c['mesh_cutoff']:g} Ry")
    if c.get("xc_functional") and c.get("xc_authors"):
        bits.append(f"{c['xc_functional']}/{c['xc_authors']}".upper())
    if c.get("kgrid"):
        shift = c.get("kgrid_displacement") or ()
        bits.append("k " + "x".join(str(k) for k in c["kgrid"])
                    + (" shifted " + " ".join(f"{s:g}" for s in shift)
                       if any(shift) else ""))
    if n_atoms:
        bits.append(f"{n_atoms} atoms")
    status = state + (" · " + " · ".join(bits) if bits else "")

# TWO SEPARATE QUESTIONS, and the card needs both: *what is this
# junction* (always answerable from the citation's own files) and
# *can it be composed into a calculation* (a refusal, sometimes).
# Kept apart, so a reason a junction cannot be BUILT -- labels
# missing, an electrode that moved, a mid-run record, blocks that
# interleave -- does not blank the viewer: the refusal is read over
# the thing it is about.
#
    # The composition answers both when it succeeds (`relaxed` IS the
    # labeled citation structure), so the happy path reads the .XV once
    # and only the refusal path pays for a second look.
    structure_wire = None
    fix = None
    # WHAT THE COMPOSITION MEASURED, as findings with the severity each
    # has (`science/validation.md` § 4.1 R4): a measurement is `info`, the
    # one convention warning `warn`, a refusal `error` -- so the page
    # tells them apart instead of reading six confirmations as warnings.
    findings: list = []
    try:
        composed = compose_junction(citation, tree_root=root,
                                    swap_electrodes=swap)
        structure_wire = composed.relaxed.to_dict()
        # THE CONVENTION IS CHECKED AND REPORTED, NEVER ENFORCED (user
        # ruling, 2026-08-29).  An inverted junction composes and runs
        # -- it biases the other end -- so the tab gets the observation
        # (the sort's one note, with the numbers) plus `fix` as a WORD
        # it can act on without matching prose.
        for note in composed.notes:
            findings.append(Issue("warn", note))
        # AND THE LEADS' OWN MEASUREMENTS.  `extract_electrode_model` and
        # the compose gate measure the periodic seam and the
        # principal-layer condition into `ElectrodeModel.notes`, the one
        # place that says whether a lead is really bulk; a condition that
        # FAILS is the gate's refusal, so what reaches here is measured
        # and reported.  Labelled, because "the seam is ECLIPSED" is
        # useless without which end.
        for model in (composed.electrode_left, composed.electrode_right):
            for note in (model.notes if model is not None else ()):
                findings.append(Issue("info", f"{model.label}: {note}"))
        if (electrode_orientation(composed.junction)
                == ORDER_INVERTED):
            fix = "swap_electrodes"
    except Exception as exc:  # noqa: BLE001 -- surfaced, never fatal
        findings.append(Issue("error", str(exc)))
        # SHOW IT ANYWAY.  The labels may also be the wrong way round,
        # and the rename is still worth offering on a junction whose
        # refusal is about something else entirely.
        try:
            from molbuilder.cell import entering
            cited_struct, _src = labeled_citation_structure(cited)
            # As this calculation will take it (`cell.entering`).
            structure_wire = entering(cited_struct).to_dict()
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
        "concluded": concluded,
        "summary": status,
        "findings": _issues_to_json(findings),
        "structure": structure_wire,
        "fix": fix,
        # THE RENAME THIS ANSWER WAS COMPOSED WITH, so the tab shows the
        # choice it holds and offers to withdraw it (§ 4).
        "swap_electrodes": swap,
        # WHAT ITS FRAMES ARE (§ 3.8.1): a cited pair's count and its rows,
        # the frames' post-process's answer; a cited run is one structure.
        "frame_set": frames_answer(
            cited.structure if isinstance(cited, CitedPair) else None),
    })


@bp.route("/api/transport/record", methods=["GET"])
def api_transport_record() -> Any:
    """The transport record of the calculation ``?path=`` names -- its
    the calculation's ``task.json`` (the report's handle), or its folder itself, so a
    ladder nothing has summarized yet has its report -- composed from its
    rungs as they are now
    (`transport.record.collect_record`, ``partial``: a ladder in progress
    has a report), never the copy `summarize task` wrote, which stays the
    command line's deliverable.  The bench summary's door, for a transport
    calculation (`/api/bench/summary`).

    Errors: 400 for a path outside the projects tree, or one that is not a
    transport calculation's; 404 for nothing there.
    """
    from pathlib import Path
    from molbuilder.runs import place_of
    from molbuilder.transport.record import RecordError, collect_record
    from .files import _PickerError, _resolve_within_roots

    raw = str(request.args.get("path") or "")
    try:
        path = _resolve_within_roots(raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not path.exists():
        return jsonify({"ok": False, "error": f"nothing at {raw}"}), 404
    base = Path(path) if path.is_dir() else Path(path).parent
    # THE FOLDER'S CALCULATION through its one door (`runs.place_of`).
    place = place_of(base)
    if place.task is None:
        return jsonify({"ok": False, "error": (
            f"{path.name}: " + (place.problem or
                                "no calculation of ours claims it"))}), 400
    task = place.task
    if task.calculation != "transport":
        return jsonify({"ok": False, "error": (
            f"{path.name}: {base.name} is a {task.calculation} "
            f"calculation, not a transport one")}), 400
    try:
        record = collect_record(base, task, partial=True)
    except (RecordError, OSError, ValueError) as exc:
        return jsonify({"ok": False, "error": (
            f"the record could not be composed: {exc}")}), 400
    return jsonify({"ok": True, "record": record})


@bp.route("/api/transport/pdos", methods=["GET"])
def api_transport_pdos() -> Any:
    """The PDOS of the atoms a person selected, at one point -- a frame
    and a voltage (`web/results.md` § 2.5): ``?path=`` the calculation's
    ``task.json`` or its folder, ``point=`` the bias in volts, ``frame=``
    the frame's index for a frame set (absent for one structure), ``atoms=``
    0-based indices, comma-separated, ``orbitals=`` ``all`` (default) or
    one of `tbtnc.ORBITAL_TYPES`.  Computed on request
    (`transport.record.selection_pdos`) -> ``{ok, energy_ev, pdos, atoms,
    outside_device, orbitals}``.  400 names what is missing."""
    from pathlib import Path
    from molbuilder.runs import place_of
    from molbuilder.transport.record import RecordError, selection_pdos
    from .files import _PickerError, _resolve_within_roots

    raw = str(request.args.get("path") or "")
    try:
        path = _resolve_within_roots(raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    base = Path(path).parent if Path(path).is_file() else Path(path)
    place = place_of(base)
    if place.task is None:
        return jsonify({"ok": False, "error": (
            f"{path.name}: " + (place.problem or
                                "no calculation of ours claims it"))}), 400
    task = place.task
    try:
        point = float(request.args.get("point", ""))
        raw_frame = request.args.get("frame")
        frame = None if raw_frame in (None, "") else int(raw_frame)
        atoms = [int(a) for a in
                 str(request.args.get("atoms") or "").split(",") if a]
    except Exception as exc:                                  # noqa: BLE001
        return jsonify({"ok": False, "error": (
            f"the request does not read: {exc}")}), 400
    try:
        got = selection_pdos(base, task, point, atoms,
                             str(request.args.get("orbitals") or "all"),
                             frame=frame)
    except RecordError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, **got})


# ===================================================================== #
# /api/transport/describe  --  the tab writes the DESCRIPTION itself   #
# ===================================================================== #


@bp.route("/api/transport/describe", methods=["POST"])
def api_transport_describe() -> Any:
    """Write the transport calculation's COMPLETE description -- the
    template and ``task.json`` -- into the folder the person chose.

    There is no hand-over for the composite (user ruling 2026-08-29):
    the other kinds hand to Task setup because that tab owns questions
    they cannot answer — shape, stages, what varies.  Transport has
    none open: the ladder and the hierarchical shape are fixed by the
    kind (`task.fixed_by_kind`), the identity derives from the citation,
    the knobs ride the stages' override bags.  So this door writes the
    finished description in one act, through the writer `jobset init`
    takes (`describe.write_description`), into a folder that holds none
    -- a description is written by the server's door that validated it
    (`web/handover-procedure.md` § 4).  Once written it is Task setup's
    to edit, as every kind's is (`engines/transport.md` § 3.8.2a).

    Validation is the codec's and the description's own check: the
    ``Task`` construction below is the gate `read_task` runs, and the
    task preflight (gate ③) is the one `jobset init`
    and the Task-setup save run -- its errors refuse, its warnings ride
    ``notices``.  The citation resolves through the same door prep
    composes through.
    """
    from molbuilder.projects import projects_root
    from molbuilder.task import Task, derive_run, fixed_by_kind
    from molbuilder.transport.compose import ComposeError, resolve_citation
    from molbuilder import template as _T
    from molbuilder.transport.stages import (TRANSPORT_STAGES,
                                             resolvable_override_names,
                                             stages_for_transport)
    _RESOLVABLE = resolvable_override_names()

    body = request.get_json(silent=True) or {}
    # THE ENGINE IS THE PANEL THE PERSON DESCRIBED ON (`engines/transport.md`
    # § 3.8.8), named by the page and never defaulted; one that runs no
    # transport is refused, saying why.
    from molbuilder.transport.stages import why_no_transport
    engine = str(body.get("engine") or "")
    if not engine:
        return jsonify({"ok": False,
                        "error": "engine must name the engine the "
                                 "calculation is described on -- the "
                                 "transport tab's engine strip names it"}), 400
    no = why_no_transport(engine)
    if no:
        return jsonify({"ok": False, "error": f"engine {engine!r}: {no}"}), 400
    citation = str(body.get("junction") or "")
    if not citation:
        return jsonify({"ok": False,
                        "error": "no junction citation -- pick the "
                                 "relaxed junction first: a finished "
                                 "relaxation run, or a structure pair "
                                 "saved from one"}), 400
    # THE LOW-BIAS APPROXIMATION, as the person chose it -- the
    # description's check below refuses a list of several without it, and
    # one with a single voltage (`engines/transport.md` § 2a.10).
    approx = body.get("low_bias_approximation")
    if approx is not None and not isinstance(approx, bool):
        return jsonify({"ok": False,
                        "error": "low_bias_approximation is true or false"}), 400
    # THE ELECTRODE RENAME, as the person chose it on the tab -- this
    # calculation's own (`engines/transport.md` § 4): the description
    # records it and compose applies it to the calculation's copy.
    swap = body.get("swap_electrodes", False)
    if not isinstance(swap, bool):
        return jsonify({"ok": False,
                        "error": "swap_electrodes is true or false"}), 400
    # PER-RUNG BAGS, the shape `task.stages` carries (`engines/transport.md`
    # § 3.8.2a): a rung's tab writes that rung's bag, so the rung is the
    # person's answer and nothing here routes.
    bags = body.get("stages") or {}
    if (not isinstance(bags, dict)
            or not all(isinstance(v, dict) for v in bags.values())):
        return jsonify({"ok": False,
                        "error": "stages must be an object of "
                                 "rung -> {parameter: value}"}), 400
    _unknown_rungs = sorted(set(bags) - set(TRANSPORT_STAGES))
    if _unknown_rungs:
        return jsonify({"ok": False,
                        "error": f"no such rung "
                                 f"{', '.join(map(repr, _unknown_rungs))}: "
                                 f"a transport ladder's rungs are "
                                 f"{', '.join(TRANSPORT_STAGES)}"}), 400
    shared_chosen = body.get("shared") or {}
    if not isinstance(shared_chosen, dict):
        return jsonify({"ok": False,
                        "error": "shared must be an object"}), 400
    typed = str(body.get("name") or "") or "transport"

    # THE FOLDER, fenced before anything else (`web/web-api.md` § 2.1).
    from molbuilder.projects import OutsideRoot, contain
    from pathlib import Path
    root = Path(projects_root()).resolve()
    dest_raw = str(body.get("dest") or "")
    if not dest_raw:
        return jsonify({"ok": False,
                        "error": "no folder to describe into -- select the "
                                 "calculation's folder in the projects "
                                 "sidebar"}), 400
    try:
        dest = contain(Path(dest_raw), root)
    except OutsideRoot as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    try:
        _, cited = resolve_citation(citation, projects_root())
    except ComposeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # THE BIAS, STATED: `[0.0]` for a single bias -- the one spelling both
    # roads write (`engines/transport.md` § 2a.10); nothing defaults it.
    # After both fences: a path outside the tree is refused first.
    bias_raw = body.get("bias")
    if not bias_raw:
        return jsonify({"ok": False,
                        "error": "bias: state the voltages -- [0.0] for a "
                                 "single bias (engines/transport.md 2a.10)"}), 400
    try:
        bias = tuple(float(v) for v in bias_raw)
    except (TypeError, ValueError):
        return jsonify({"ok": False,
                        "error": f"bias must be a list of volts, "
                                 f"got {bias_raw!r}"}), 400

    # THE FOLDER'S two refusals every door that describes asks
    # (`engines/transport.md` § 3.8.2a; user, 2026-10-11): one calculation
    # per folder -- Describe is one act, and a described calculation is
    # changed on Task setup -- and never inside its citation (§ 3.1).
    from molbuilder.describe import (Description, DescribeError,
                                     refuse_described, write_description)
    from molbuilder.transport.compose import refuse_inside_citation
    if not dest.is_dir():
        return jsonify({"ok": False,
                        "error": f"{dest_raw} is not a folder"}), 400
    shown = str(dest.relative_to(root))
    try:
        refuse_described(dest, shown=shown)
        refuse_inside_citation(dest, cited, shown=shown)
    except (DescribeError, ComposeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409

    # Refused HERE, not at prep on the cluster: an unknown knob, a shared
    # value or a role-fixed one names itself while changing it is still
    # free.  Three questions, three markers, one catalogue
    # (`engines/transport.md` § 3.8.2; `engines/template.md` § 6.4).
    # The one rule for "binds every stage" (`template.shared_by_every_stage`),
    # which `resolve` refuses at prep for every kind -- asked here too, so
    # the tab refuses while changing it is still free.
    _shared_names = _T.shared_by_every_stage(engine, "transport")
    # ...and the one rule for "the rung fixes it", which `resolve` refuses
    # at prep for every kind, with the same reason.
    _role_names = _T.fixed_by_role(engine, "transport")
    # THE VOCABULARY IS WHAT PREP CAN RESOLVE, and it is asked rather than
    # listed (`stages.resolvable_override_names`).
    for _rung, _name in [(r, n) for r, b in bags.items() for n in b]:
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
        if _name in _role_names:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is fixed by the rung, so "
                                     f"it cannot be a per-stage override: "
                                     f"{_T.why_role(_name)}."}), 400
        if _name in _shared_names:
            # SHARED, therefore not a per-stage override -- and that is the
            # reason, not "the citation owns it" (`engines/transport.md`
            # § 2a.7: the cited run DEFAULTS these values; what stays true
            # is that every rung shares them).
            return jsonify({"ok": False,
                            "error": f"{_name!r} is shared by "
                                     f"{_T.shared_by(_name)}, so it cannot be "
                                     f"a per-stage override: "
                                     f"{_T.why_shared(_name)}.  Change it on "
                                     f"the shared panel, which edits the "
                                     f"calculation's template -- there one "
                                     f"value applies to every rung that "
                                     f"reads it."
                            }), 400
    # A RUNG'S VALUE ON A RUNG THAT DOES NOT READ IT is the description's
    # own check's to refuse, below, as at every describe and at prep
    # (`template.unread_overrides`, plan § 5w K4).
    for _name in shared_chosen:
        if _name not in _shared_names:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is not a shared value of "
                                     f"this calculation; the shared panel "
                                     f"carries the items the catalogue "
                                     f"marks `shared` for transport"}), 400
    try:
        task = Task(
            engine=engine, shape=fixed_by_kind("transport")["shape"],
            run=derive_run(typed, citation,
                           stage_names=TRANSPORT_STAGES),
            structure=None, calculation="transport",
            slots={"junction": citation}, bias=bias,
            low_bias_approximation=approx,
            swap_electrodes=swap,
            # the stages.md 6.2 rule holds here too: an override names
            # a PROMOTED field, and `varies` is the promotion
            varies=tuple(sorted({n for b in bags.values() for n in b})),
            stages=tuple(stages_for_transport(bags)))
    except ValueError as exc:
        # THE CODEC'S REFUSAL, as a finding BESIDE THE CONTROL that wrote
        # it when the codec names where (`task.DescriptionError`: the
        # bias list, its treatment) -- the tab maps `task.bias` to its
        # rows (`science/validation.md` § 4.1 R2); the error line says it
        # too, as every refusal does.
        from molbuilder.issues import Issue
        where = getattr(exc, "where", "")
        return jsonify({
            "ok": False, "error": str(exc),
            "findings": _issues_to_json([Issue(
                "error", str(exc), where=f"task.{where}" if where else "")]),
        }), 400
    # The template's text through the one door the schema route draws both
    # surfaces from (`_panel_template`) -- a value its field cannot take, or
    # a citation it refuses (a cited run that carried a net charge, ES7),
    # said here by name rather than failing the whole response.
    from molbuilder.transport.citation_defaults import check_pseudopotentials
    try:
        check_pseudopotentials(cited, shared_chosen.get("psml_lib"))
        _tmpl_text = _panel_template(cited.path, shared_chosen,
                                     label=task.label)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # THE DESCRIPTION'S OWN CHECK (gate ③, plan § 5w K3).  The same
    # function the Task-setup save and `jobset init` run: its errors
    # refuse in the save's own words, its warnings ride `notices`.
    from molbuilder.validation.task import (preflight as _task_preflight,
                                            config_class_for as _cfg_cls_for)
    _pf = _task_preflight(task, template_text=_tmpl_text)
    # ...AND THE K-POINT MESHES EACH RUNG WILL WRITE, judged by the one
    # check prep's gate runs (`kmesh.check`; `engines/transport.md` § 0.3b)
    # on the junction's own axis kinds -- so one point along a direction
    # the electrodes repeat in is said here, beside its control, and not
    # first at prep.
    from molbuilder.transport.compose import junction_axis_kind
    from molbuilder.transport.deck import ladder_mesh_findings
    try:
        _kinds = junction_axis_kind(cited)
    except ComposeError:
        _kinds = None              # compose refuses it, by name, at prep
    if _kinds is not None:
        _pf = list(_pf) + ladder_mesh_findings(_tmpl_text, task, _kinds)
    _pf_errs = [i for i in _pf if i.severity == "error"]
    if _pf_errs:
        return jsonify({
            "ok": False,
            "error": "the description fails its own preflight "
                     "(engines/stages.md § 6.6):\n  - "
                     + "\n  - ".join(i.message for i in _pf_errs),
            "findings": _issues_to_json(_pf, cfg=_cfg_cls_for(task)),
        }), 400

    # WRITTEN HERE, every file or none, through the writer `jobset init`
    # takes (`describe.write_description`; `web/handover-procedure.md` § 4:
    # a description is written by the server's door that validated it).
    # TWO FILES: a transport description is `task.json` AND a template,
    # like every other kind's (TR1).  Its values are DEFAULTED FROM THE
    # CITED RUN (§ 2a.7, ruling 1) and are the person's to change
    # afterwards; a `citation` row nobody answered stays VALUELESS
    # (§ 3.8.3).
    try:
        written = write_description(Description(task=task,
                                                template=_tmpl_text), dest)
    except (DescribeError, OSError) as exc:
        return jsonify({"ok": False,
                        "error": f"nothing was written: {exc}"}), 500
    return jsonify({
        "ok": True,
        "label": task.label,
        "written": [p.name for p in written],
        "notices": _issues_to_json(_pf, cfg=_cfg_cls_for(task)),
    })


# ===================================================================== #
# /api/transport/schema  --  form schema endpoint                       #
# ===================================================================== #


@bp.route("/api/transport/schema", methods=["GET"])
def api_transport_schema() -> Any:
    """The transport tab's TWO surfaces, both from the catalogue narrowed to
    the kind (`engines/transport.md` § 3.8.2; the markers decide which
    value lands where, `catalogue_to_form_schema(surface=)`).

    ``?surface=rung`` (the default) is the per-rung form -- every transport
    item that is not `shared`, `role`, `allocation` or staging; its values
    are override bags, routed to the rung that owns each by the `stages`
    marker.  ``?surface=shared`` is the panel that edits the TEMPLATE:
    every item marked `shared` for transport, outside the `setup` group --
    save what the citation asks of the person, drawn required
    (`citation_defaults.asked_of_the_person`: a pair's pseudopotential
    directory).
    With ``&junction=<citation>`` both are drawn from the template this
    calculation's describe will write (`_panel_template`, plan § 5w K7):
    each field carries its value and where it came from (`form-schema.md`
    § 1.1) -- the shared panel holding the citation's answers (a
    deck's, a record's, or none; a `citation` row it does not answer is
    not chosen, § 3.8.3), and a rung's tab showing what the rung runs
    unless it sets its own.  A rung's tab also takes ``&shared=<json>``,
    what the shared panel holds, since a rung's value can follow a shared
    one (the transmission grid starts at the SCF's).  The answer names the
    citation's source so the page can say where the numbers came from.
    """
    from molbuilder.projects import projects_root
    from molbuilder.transport.citation_defaults import (asked_of_the_person,
                                                        citation_answers)
    from molbuilder.transport.compose import ComposeError, resolve_citation
    from molbuilder.transport.stages import (RUNG_NOTES, TRANSPORT_STAGES,
                                             why_no_transport)
    from molbuilder import template as _T

    surface = str(request.args.get("surface") or "rung")
    if surface not in ("rung", "shared"):
        return jsonify({"ok": False,
                        "error": f"surface must be 'rung' or 'shared', "
                                 f"not {surface!r}"}), 400
    # ONE ANSWER PER ENGINE (`engines/transport.md` § 3.8.8): the engine is
    # the panel the form is drawn on, named by the page, never defaulted --
    # and one that runs no transport is refused, saying why.
    engine = str(request.args.get("engine") or "")
    if not engine:
        return jsonify({"ok": False,
                        "error": "engine must name the engine whose form "
                                 "this is -- the transport tab's engine "
                                 "strip names it"}), 400
    no = why_no_transport(engine)
    if no:
        return jsonify({"ok": False, "error": f"engine {engine!r}: {no}"}), 400
    # ONE RUNG'S TAB (§ 3.8.2a): `&rung=<name>` narrows the rung surface to
    # the items that rung owns plus the ones any rung may set.  Without it
    # the answer is every rung item at once, and it carries the rung list
    # -- name, ladder index, one-line note -- the tab strip is built from.
    rung = str(request.args.get("rung") or "") or None
    if rung is not None and (surface != "rung"
                             or rung not in TRANSPORT_STAGES):
        return jsonify({"ok": False,
                        "error": f"rung must name one of "
                                 f"{', '.join(TRANSPORT_STAGES)} on the "
                                 f"rung surface, not {rung!r}"}), 400
    citation = str(request.args.get("junction") or "")
    source: Dict[str, Any] = {"kind": "none", "name": ""}
    template = None
    # WHAT THE CITATION ASKS OF THE PERSON joins the shared panel -- a pair's
    # pseudopotential directory (§ 3.8.1) -- by the rule the describe door's
    # check asks too.
    asks: tuple = ()
    if citation:
        try:
            _, cited = resolve_citation(citation, projects_root())
        except ComposeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        try:
            shared = json.loads(request.args.get("shared") or "{}")
        except ValueError:
            shared = None
        if not isinstance(shared, dict) or surface != "rung" and shared:
            return jsonify({"ok": False,
                            "error": "shared must be a JSON object of what "
                                     "the shared panel holds, asked by a "
                                     "rung's tab"}), 400
        try:
            template = _T.read_template(
                _panel_template(cited.path, shared, label="transport"))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        try:
            answers = citation_answers(cited.path)
        except ValueError as exc:
            # A CITED VALUE THIS BUILD CANNOT READ, refused by name
            # (`citation_defaults`; a 500 until 2026-10-19).
            return jsonify({"ok": False, "error": str(exc)}), 400
        source = {"kind": answers.source, "name": answers.source_name}
        asks = asked_of_the_person(cited)
    # ONE ID PER CONTROL ON THE PAGE: the five rung tabs render the same
    # items, so a rung's controls carry the rung in their ids
    # (`t-device-max-scf-iter`), and a finding's `stage` picks its own
    # tab's (`validation-findings.js`); the shared panel's are `t-…`.
    schema = catalogue_to_form_schema(engine,
                                      f"t-{rung}" if rung else "t",
                                      calculation="transport",
                                      surface=surface, rung=rung,
                                      template=template, asks=asks)
    response: Dict[str, Any] = {"ok": True, "surface": surface,
                                "engine": engine, "source": source}
    if surface == "rung":
        response["rung"] = rung
        response["rungs"] = [{"name": n, "index": i, "note": RUNG_NOTES[n]}
                             for i, n in enumerate(TRANSPORT_STAGES, start=1)]
        # NO SECOND FILTER HERE: which fields a rung's tab offers is the
        # schema builder's, by the catalogue's markers (`transport.md`
        # § 3.8.2; the run settings dropped by `surface="rung"`, 2026-10-18).
    response["schema"] = schema
    return jsonify(response)


def _panel_template(cite_dir, shared: Dict[str, Any], *, label: str) -> str:
    """The template a transport describe writes, from what the shared panel
    holds -- the one text the describe door writes and the schema route
    draws both surfaces from, so the tab cannot show one calculation and
    describe another (plan § 5w K7).

    The panel's values go through the door every SIESTA form goes through
    (`_shared.siesta_config_from_params`): coerced to each field's declared
    type, a value one cannot take refused naming it.  A BLANK IS NOT CHOSEN
    (`form-schema.md` § 1.1): the panel is drawn holding the citation's
    answers, so a field it sends blank was emptied and the citation's value
    is not applied -- a `citation` row is written valueless (§ 3.8.3), and
    a blank electronic state is "work it out" on the whole junction at
    prep, which the chemistry card beside the panel shows.  Never written as
    an empty string, which `prep` would refuse by name.

    Raises ``ValueError`` -- the value, or a citation the template's door
    refuses (a cited run that carried a net charge, ES7).
    """
    from ._shared import siesta_config_from_params
    from molbuilder.transport.citation_defaults import transport_template_text
    typed = {k: v for k, v in shared.items() if v not in (None, "")}
    cfg = siesta_config_from_params(typed, "transport")
    chosen = {k: getattr(cfg, k) for k in typed
              if getattr(cfg, k, None) is not None}
    blank = sorted(k for k, v in shared.items() if v in (None, ""))
    return transport_template_text(cite_dir, label=label, blank=blank,
                                   **chosen)


