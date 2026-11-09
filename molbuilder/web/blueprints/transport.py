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
    GET  /api/transport/fit               the transmission window of a
                                          calculation that reuses a
                                          finished one, fitted to the
                                          finished grid
    GET  /api/transport/digest            what is kept of each point's data
                                          file reading, and how far a
                                          reading under way is
    POST /api/transport/digest            read them, on the person's word
    GET  /api/transport/export            every result as it stands, as
                                          the data file or its tables
    POST /api/transport/describe          writes the FINISHED description
                                          into the chosen folder -- the
                                          web spelling of `molbuilder describe
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
    sent_export,
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


def _engine_refused(engine: str, what: str):
    """THE ENGINE A TRANSPORT DOOR IS ASKED ON (`engines/transport.md`
    § 3.8.8): named by the page, never defaulted -- the refusal of none
    named, or of one that runs no transport in the kind table's sentence --
    or ``None``, the engine stands.  ``what`` says what the engine is."""
    from molbuilder.kind_table import kind
    if not engine:
        return jsonify({"ok": False,
                        "error": f"engine must name the engine {what} -- "
                                 f"the transport tab's engine strip names "
                                 f"it"}), 400
    no = kind("transport").why_not_on(engine)
    if no:
        return jsonify({"ok": False, "error": f"engine {engine!r}: {no}"}), 400
    return None


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
    (`science/validation.md` § 4.1 R2): each lead's period and seam at
    their weight (`wizard.seam_finding`), the electrode orientation
    as ``warn``, a composition refusal as ``error`` -- so the tab
    renders them through the one renderer, never a run-on string.
    ``structure`` carries the cited junction's labeled structure for
    the viewer (the /api/build/load ``{structure}`` envelope), so the
    tab shows the citation **whether or not it composes**: a directory
    that classifies but cannot be built into a calculation still
    answers with its junction, and the refusal is a finding beside it.
    ``fix`` is a word the tab can act on (today only
    ``"swap_electrodes"``), never prose to match.

    ``&reuses=<folder>`` asks whether the path is a frame set of that
    transport calculation's junction (`engines/transport.md` § 2a.16, *A
    frame set of the same junction*): described with that calculation's
    rename and checked by the one rule (`transport.reuse.frame_set_of`), its
    summary opening with what it is; one whose frame 0 is another junction
    answers ``citation: null`` with the refusal as the summary.
    """
    from molbuilder.transport.compose import (CitedPair, ComposeError,
                                              classify_citation,
                                              compose_junction,
                                              labeled_citation_structure)
    from molbuilder.frame_average import answer_said
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
    # A FRAME SET OF A FINISHED CALCULATION'S JUNCTION, asked beside it
    # (`engines/transport.md` § 2a.16): described with that calculation's
    # rename, its own (§ 4), and checked below by the one rule.
    finished = None
    reuses_raw = str(request.args.get("reuses") or "")
    if reuses_raw:
        from molbuilder.transport.reuse import ReuseError, reused
        try:
            finished, _rel = reused(reuses_raw, tree_root=root)
        except ReuseError as exc:
            return jsonify({"ok": True, "citation": None,
                            "summary": str(exc), "findings": []}), 200
        swap = finished.task.swap_electrodes
    # A TRANSPORT CALCULATION'S FOLDER, to reuse it (`engines/transport.md`
    # § 2a.16): its junction, its description and its template fill the
    # cards, and Describe writes what `molbuilder describe --reuses` writes.
    # A description that does not read is said by that door; another
    # kind's folder is no citation, as any path that is not one.
    from molbuilder import task as _task
    from molbuilder.calcdirs import described
    from molbuilder.kind_table import kind as _kind
    if finished is None and described(cited_path):
        # A CALCULATION OF A KIND WHOSE DESCRIPTION MAY EXTEND A FINISHED ONE
        # (the kind table's ``description.carries``, ``reuses``) is cited to
        # be extended; one whose description does not read is said by that
        # door.
        try:
            extends = ("reuses" in
                       _kind(_task.of(cited_path).calculation)
                       .description.carries)
        except ValueError:
            extends = True
        if extends:
            return _describe_reused(citation, root)
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
    # WHAT THE CITED PAIR'S READ SAID, first: a pair that states no atom
    # order, taken as the user order (`model/overview.md` § 2.2b, F4).
    if isinstance(cited, CitedPair):
        findings.extend(cited.said)
    # THE ENGINE THE JUNCTION IS COMPOSED FOR, asked where it is needed: its
    # decks' atom order is the pair's (`kind_table.atom_order`).
    engine = str(request.args.get("engine") or "")
    refusal = _engine_refused(engine, "the junction is composed for")
    if refusal is not None:
        return refusal
    composed = None
    try:
        composed = compose_junction(citation, tree_root=root, engine=engine,
                                    swap_electrodes=swap)
        structure_wire = composed.relaxed.to_dict()
        # THE CONVENTION IS CHECKED AND REPORTED, NEVER ENFORCED (user
        # ruling, 2026-08-29).  An inverted junction composes and runs
        # -- it biases the other end -- so the tab gets the observation
        # (the sort's one note, with the numbers) plus `fix` as a WORD
        # it can act on without matching prose.
        # ...BUT NOT WHILE EXTENDING, where the rename is the finished
        # calculation's own and not offered (§ 2a.16).
        for note in (composed.notes if finished is None else ()):
            findings.append(Issue("warn", note))
        # AND THE LEADS' OWN MEASUREMENTS.  `extract_electrode_model`
        # measures each lead's period and periodic seam into
        # `ElectrodeModel.notes`, each at its weight; a lead that cannot
        # be one is the gate's refusal, so what reaches here is measured
        # and reported.  Labelled, because "the seam is ECLIPSED" is
        # useless without which end.
        for model in (composed.electrode_left, composed.electrode_right):
            for note in (model.notes if model is not None else ()):
                findings.append(Issue(note.severity,
                                      f"{model.label}: {note.message}",
                                      where=note.where))
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
            from molbuilder.transport.compose import with_swapped_leads
            cited_struct, _src = labeled_citation_structure(cited)
            # WITH THE RENAME HELD, as compose takes it (§ 4): the viewer
            # and the orientation question show the junction as it will be
            # composed.
            if swap:
                cited_struct = with_swapped_leads(cited_struct,
                                                   cited.path.name)
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

    # WHAT ITS FRAMES ARE (§ 3.8.1): a cited pair's count and its rows, the
    # frames' post-process's answer; a cited run is one structure.  Rows the
    # vocabulary refuses are said where the frames would be, and as a
    # refusal beside the junction, as every refusal here is (§ 2a.9).
    frame_set = answer_said(
        cited.structure if isinstance(cited, CitedPair) else None)
    if frame_set.get("why"):
        findings.append(Issue("error", str(frame_set["why"])))
    if finished is not None:
        # ITS FRAME 0 IS THE FINISHED JUNCTION, or it is refused here, where
        # it is picked, the junction the tab holds staying held; the rename
        # is the finished calculation's, never offered.
        from molbuilder.transport.reuse import ReuseError, frame_set_of
        try:
            frame_set_of(finished, citation, tree_root=root,
                         composed=composed)
        except ReuseError as exc:
            return jsonify({"ok": True, "citation": None,
                            "summary": str(exc), "findings": []}), 200
        status = (f"FRAME 0 IS THE JUNCTION {finished.base.name} RAN · "
                  + status)
        fix = None
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
        "frame_set": frame_set,
    })


def _describe_reused(folder: str, root) -> Any:
    """The describe door's answer for a transport calculation's folder,
    picked to reuse it (`engines/transport.md` § 2a.16, *On the page*): its
    meta line -- what it computed: its junction's citation, its treatment,
    its voltages, its window, where each group stands -- the junction its
    record holds, for the viewer, in the user order; the citation the cards are
    drawn for; ``reuses``, the folder; and its bias list and treatment.  A
    folder that is not a transport calculation's answers ``citation: null``
    with the refusal as its summary, as any path that is not citable."""
    from molbuilder.frame_average import answer_said
    from molbuilder.issues import Issue
    from molbuilder.jobset.rung import citation_of
    from molbuilder.jobset.runstatus import jobset_status
    from molbuilder.transport.compose import ComposeError, recorded_junction
    from molbuilder.transport.reuse import Extension, ReuseError, reused
    try:
        calc, rel = reused(folder, tree_root=root)
    except ReuseError as exc:
        return jsonify({"ok": True, "citation": None, "summary": str(exc),
                        "findings": []}), 200
    from molbuilder.jobset.ladder import stage_config
    from molbuilder.transport.window import window_rung
    task = calc.task
    unread = None
    try:
        recorded = recorded_junction(calc.base)
    except ComposeError as exc:
        recorded, unread = None, str(exc)
    # THE JUNCTION IN THE USER ORDER, once -- the structure the viewer draws
    # and the frames' rows and masses beside it, one list
    # (`model/overview.md` § 2.2a) -- or, a junction whose order is not
    # recorded (I9), as TranSIESTA lists it, marked and said (§ 2.2b, F4).
    junction = (recorded.structure.restored() if recorded is not None
                else None)
    unstated = recorded.order_unstated if recorded is not None else None
    if unstated:
        junction = junction.replace(order_unstated=True)
    # THE WINDOW ITS TRANSMISSION COMPUTED: the template's, or the rung's
    # own values over it (`ladder.stage_config`).
    cfg = stage_config(task, window_rung(), calc.template_text)
    window = (f"{cfg.transmission_emin_ev:g}…{cfg.transmission_emax_ev:g} "
              f"eV at {cfg.transmission_n_points} points")
    steps = jobset_status(calc).steps
    stands = ("every group finished"
              if steps and all(s.state == "finished" for s in steps) else
              "; ".join(f"group {s.group} {s.words.rstrip('.')}"
                        for s in steps))
    summary = " · ".join([
        "TRANSPORT CALCULATION", Extension(rel, task).name,
        f"its junction, cited from {citation_of(task)}",
        str(task.treatment),
        ", ".join(str(float(v)) for v in task.bias) + " V", window,
        stands] + ([f"its junction's record does not read: {unread}"]
                   if unread else [])
        + ([unstated] if unstated else []))
    # ITS FRAMES, as any citation's are said (§ 3.8.1, § 2a.9).
    frame_set = answer_said(junction if junction is not None
                            and junction.n_frames > 1 else None)
    return jsonify({
        "ok": True, "citation": citation_of(task), "reuses": rel,
        "concluded": True, "summary": summary,
        "findings": _issues_to_json(
            [Issue("error", str(frame_set["why"]))]
            if frame_set.get("why") else []),
        "structure": junction.to_dict() if junction is not None else None,
        "fix": None, "swap_electrodes": task.swap_electrodes,
        "frame_set": frame_set,
        "bias": [float(v) for v in task.bias],
        "low_bias_approximation": task.low_bias_approximation,
        # THE WINDOW ITS TRANSMISSION COMPUTED, in the meta line's words --
        # what the page's EXTENDING line says it describes.
        "window": window,
        # WHAT AN EXTENSION OF IT MAY CHANGE, AND WHAT IS FIXED
        # (`engines/transport.md` § 2a.16, *Extending it*): the rungs it
        # keeps, what each control says and the rung whose window a fit is
        # written into -- drawn as they are.
        "extend": Extension(rel, task).to_wire(),
    })


@bp.route("/api/transport/fit", methods=["GET"])
def api_transport_fit() -> Any:
    """THE TRANSMISSION WINDOW FITTED TO A FINISHED GRID
    (`engines/transport.md` § 2a.16): ``?reuses=<folder>&lo=&hi=[&finer=]``
    -- the range asked, fitted to the grid the finished calculation's
    transmission computed (`transport.window.fitted`): its ends, its count
    and the words said of it, which the page's *use these* writes into the
    transmission's fields.  A range or a factor that cannot be fitted, or a
    finished calculation holding no transmission point done, is refused in
    its words."""
    from pathlib import Path
    from molbuilder.projects import projects_root
    from molbuilder.transport.describe import fit as _fit
    try:
        try:
            lo = float(request.args.get("lo", ""))
            hi = float(request.args.get("hi", ""))
            finer = int(request.args.get("finer") or 1)
        except ValueError:
            raise ValueError("the window asked needs two energies in eV and "
                             "a whole number for times finer")
        fit = _fit(str(request.args.get("reuses") or ""), lo, hi, finer,
                   tree_root=Path(projects_root()).resolve())
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "words": fit.words(),
                    "values": fit.values()})


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
    from molbuilder.transport.record import RecordError, collect_record
    calc, refusal = _transport_calc(str(request.args.get("path") or ""))
    if refusal is not None:
        return refusal
    try:
        record = collect_record(calc, partial=True)
    except (RecordError, OSError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "record": record})


def _transport_calc(raw: str):
    """``(calc, refusal)`` -- the transport calculation a path names: its
    ``task.json``, its folder, or a folder below it, through the one door
    for that (`calculation.at`, A16); or the refusal: 400 for a path outside
    the projects tree or not a transport calculation's, 404 for nothing
    there."""
    from molbuilder.calculation import NotACalculation, at
    from .files import _resolve_within_roots
    path = _resolve_within_roots(raw)
    try:
        calc = at(path)
    except FileNotFoundError:
        return None, (jsonify({"ok": False, "error": f"nothing at {raw}"}),
                      404)
    except (NotACalculation, OSError, ValueError) as exc:
        return None, (jsonify({"ok": False, "error": str(exc)}), 400)
    from molbuilder.transport.record import not_transport
    other = not_transport(calc)
    if other:
        return None, (jsonify({"ok": False,
                               "error": f"{path.name}: {other}"}), 400)
    return calc, None


def _readings_answer(calc) -> Dict[str, Any]:
    """Each transmission point's readings, as the digest routes answer
    them: its point, in words, its folder, and each data file's reading --
    its path from the calculation, its state, size and claim."""
    from molbuilder.transport.tbtnc import point_readings
    base = calc.base
    return {"ok": True, "points": [
        {"point": e["point"].to_wire(), "words": e["point"].words(),
         "folder": str(e["folder"].relative_to(base)),
         "readings": [k.to_wire(base) for k in e["readings"]]}
        for e in point_readings(calc)]}


@bp.route("/api/transport/digest", methods=["GET"])
def api_transport_digest() -> Any:
    """WHAT IS KEPT OF EACH TRANSMISSION POINT'S DATA FILE READING
    (`model/parse.md` § 5e; `engines/transport.md` § 2a.12) -- ``?path=``
    the calculation's ``task.json`` or folder: per point, each data file's
    reading, its own and those in its ``shared/`` -- kept, missing, out of
    date or being read, with its size and the claim of a reading under way,
    how far it is.  Nothing is read: the page asks it while a reading
    runs."""
    calc, refusal = _transport_calc(str(request.args.get("path") or ""))
    if refusal is not None:
        return refusal
    return jsonify(_readings_answer(calc))


@bp.route("/api/transport/digest", methods=["POST"])
def api_transport_digest_read() -> Any:
    """READ, ON THE PERSON'S WORD, each point's data files whose reading is
    not kept -- every one with ``fresh`` (*Read again*) -- in this request,
    one at a time, each claimed so no other reading repeats it
    (`tbtnc.read_points`, `model/parse.md` § 5e); ``take_over`` removes
    another's claim (*Take over*).  Body ``{path, fresh, take_over}``.
    Answers once every file is read or found being read: each file's
    outcome, and the readings as they are then."""
    body = request.get_json(silent=True) or {}
    calc, refusal = _transport_calc(str(body.get("path") or ""))
    if refusal is not None:
        return refusal
    from molbuilder.transport.tbtnc import read_points
    outcomes = read_points(calc, fresh=bool(body.get("fresh")),
                           take_over=bool(body.get("take_over")))
    return jsonify({**_readings_answer(calc), "outcomes": outcomes})


@bp.route("/api/transport/export", methods=["GET"])
def api_transport_export() -> Any:
    """EVERY RESULT OF A TRANSPORT CALCULATION, AS IT STANDS
    (`engines/transport.md` § 2a.12, *Export*) -- ``?path=`` the
    calculation's ``task.json`` or folder, ``format=`` ``netcdf`` or
    ``csv`` -- an attachment: its data file, or its tables and README as one
    zip; composed on request from what is there (`transport.export`), no
    data file read and nothing written into the calculation's folder; the
    points it holds without their DOS by region said in the response's
    ``X-Molbuilder-Said`` header.  400 names what is refused."""
    from molbuilder.ncdata import ExportError
    from molbuilder.transport.export import export
    calc, refusal = _transport_calc(str(request.args.get("path") or ""))
    if refusal is not None:
        return refusal
    try:
        got = export(calc, str(request.args.get("format") or ""))
    except ExportError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return sent_export(got)


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
    from molbuilder.transport.record import RecordError, selection_pdos
    # THE CALCULATION, through its one door (`calculation.open`, A16), as
    # the record, the digest and the export routes open it.
    calc, refusal = _transport_calc(str(request.args.get("path") or ""))
    if refusal is not None:
        return refusal
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
        got = selection_pdos(calc, point, atoms,
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
    template, ``task.json`` and its job request -- into the folder the
    person chose.

    There is no hand-over for the composite (user ruling 2026-08-29):
    the other kinds hand to Task setup because that tab owns questions
    they cannot answer — stages, what varies.  Transport has none
    open: the ladder is fixed by the kind (the kind table), the
    identity derives from the citation,
    the knobs ride the stages' override bags.  So this door writes the
    finished description in one act, through the kind's one describe
    (`transport.describe`, which `molbuilder describe` takes too), into a
    folder that holds none -- a description is written by the server's door
    that validated it (`web/handover-procedure.md` § 4).  Once written it
    is Task setup's to edit, as every kind's is (`engines/transport.md`
    § 3.8.2a).  This door reads the body, fences the folder in the file
    layer's words, and answers what the describe answers -- its refusals
    with their findings, its notes as ``notices``.

    With ``reuses`` -- a transport calculation's folder, picked to reuse it
    (`engines/transport.md` § 2a.16) -- it writes what `molbuilder describe
    --reuses` writes: that calculation's description and its label, its
    junction's citation -- its own, or a frame set of it, ``junction``
    naming which -- as the cards change its voltages, its rungs' values and
    its shared ones, an extension's limits refused here (the tab's to
    hold); its template; its job request.  The junction itself and the
    pseudopotentials are its first prep's (§ 2a.16, *Describing it*).
    """
    from molbuilder.projects import projects_root

    body = request.get_json(silent=True) or {}
    # THE ENGINE IS THE PANEL THE PERSON DESCRIBED ON (`engines/transport.md`
    # § 3.8.8), named by the page and never defaulted; one that runs no
    # transport is refused, saying why.
    engine = str(body.get("engine") or "")
    refusal = _engine_refused(engine, "the calculation is described on")
    if refusal is not None:
        return refusal
    citation = str(body.get("junction") or "") or None
    # THE LOW-BIAS APPROXIMATION, as the person chose it -- the codec
    # refuses a list of several without it, and one with a single voltage
    # (`engines/transport.md` § 2a.10).
    approx = body.get("low_bias_approximation")
    if approx is not None and not isinstance(approx, bool):
        return jsonify({"ok": False,
                        "error": "low_bias_approximation is true or false"}), 400
    # THE ELECTRODE RENAME, as the person chose it on the tab -- this
    # calculation's own (`engines/transport.md` § 4): the description
    # records it and compose applies it to the calculation's copy.
    swap = body.get("swap_electrodes")
    if swap is not None and not isinstance(swap, bool):
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
                                 "stage -> {parameter: value}"}), 400
    shared_chosen = body.get("shared") or {}
    if not isinstance(shared_chosen, dict):
        return jsonify({"ok": False,
                        "error": "shared must be an object"}), 400
    typed = str(body.get("name") or "")
    # THE PLAN ALONE (`engines/transport.md` § 2a.16, *On the page*): the
    # same checks, then what the description would copy and run, written
    # nowhere the person sees -- no folder needed.
    plan_only = body.get("plan", False)
    if not isinstance(plan_only, bool):
        return jsonify({"ok": False, "error": "plan is true or false"}), 400

    # THE FOLDER, fenced before anything else (`web/web-api.md` § 2.1).
    from molbuilder.projects import OutsideRoot, contain
    from pathlib import Path
    root = Path(projects_root()).resolve()
    dest_raw = str(body.get("dest") or "")
    dest = None
    if not plan_only:
        if not dest_raw:
            return jsonify({"ok": False,
                            "error": "no folder to describe into -- select "
                                     "the calculation's folder in the "
                                     "projects sidebar"}), 400
        try:
            dest = contain(Path(dest_raw), root)
        except OutsideRoot as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        # A CALCULATION LIVES UNDER A TOPIC, as Build's door refuses it.
        from .files import calculation_folder_refusal
        too_high = calculation_folder_refusal(dest)
        if too_high:
            return jsonify({"ok": False, "error": too_high}), 400

    # EVERYTHING ELSE IS THE KIND'S ONE DESCRIBE (`transport.describe`), the
    # one `molbuilder describe` takes too: the citation or the calculation
    # it reuses, the folder's refusals, the voltages, each rung's values and
    # the shared ones, the task, an extension's limits -- held here, the
    # tab's -- the template, the description's own check, the files or the
    # plan alone, and an extension's manifest.
    bias_raw = body.get("bias")
    bias = None
    if bias_raw is not None:
        try:
            bias = tuple(float(v) for v in bias_raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False,
                            "error": f"bias must be a list of volts, "
                                     f"got {bias_raw!r}"}), 400
    if dest is not None and not dest.is_dir():
        return jsonify({"ok": False,
                        "error": f"{dest_raw} is not a folder"}), 400
    from molbuilder.describe import DescribeError
    from molbuilder.transport.describe import NotDescribed, describe
    from molbuilder.validation.task import config_class_for as _cfg_cls_for
    try:
        made = describe(
            calculation="transport", engine=engine, junction=citation,
            reuses=str(body.get("reuses") or "") or None, name=typed,
            dest=dest, bias=bias, low_bias_approximation=approx,
            swap_electrodes=swap, stages=bags, shared=shared_chosen,
            refuse_limits=True, tree_root=root)
    except NotDescribed as exc:
        answer: Dict[str, Any] = {"ok": False, "error": str(exc)}
        if exc.findings:
            answer["findings"] = _issues_to_json(
                exc.findings,
                cfg=_cfg_cls_for(exc.task) if exc.task is not None else None,
                calculation="transport")
        return jsonify(answer), 409 if exc.folder else 400
    except (DescribeError, OSError) as exc:
        # THE PLAN'S HIDDEN FOLDER REFUSED by the tree, said as the plan;
        # a write that failed, as nothing written.
        return jsonify({"ok": False,
                        "error": (f"the plan could not be worked out: {exc}"
                                  if plan_only
                                  else f"nothing was written: {exc}")}), 500
    notices = _issues_to_json(list(made.notes), cfg=_cfg_cls_for(made.task),
                              calculation="transport")
    manifest = list(made.manifest) if made.manifest is not None else None
    if plan_only:
        return jsonify({"ok": True, "plan": True, "manifest": manifest,
                        "notices": notices})
    return jsonify({
        "ok": True,
        "label": made.task.label,
        "written": [p.name for p in made.written],
        # A CALCULATION THAT REUSES A FINISHED ONE: what it will copy, and
        # what runs, in the manifest's words; null otherwise.
        "manifest": manifest,
        # WHAT IS SAID BESIDE IT, each at its weight -- the description's
        # own notes, a finished calculation carrying no composed junction,
        # a frame set cited in its junction's place (`issues.Issue`).
        "notices": notices,
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
    from molbuilder.jobset.rung import citation_of
    from molbuilder.transport.compose import ComposeError, resolve_citation
    from molbuilder.transport.stages import TRANSPORT_STAGES, rung_notes
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
    refusal = _engine_refused(engine, "whose form this is")
    if refusal is not None:
        return refusal
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
    reuses = str(request.args.get("reuses") or "")
    source: Dict[str, Any] = {"kind": "none", "name": ""}
    template = None
    held = None
    # WHAT THE CITATION ASKS OF THE PERSON joins the shared panel -- a pair's
    # pseudopotential directory (§ 3.8.1) -- by the rule the describe door's
    # check asks too.
    asks: tuple = ()
    try:
        shared = json.loads(request.args.get("shared") or "{}")
    except ValueError:
        shared = None
    if not isinstance(shared, dict) or surface != "rung" and shared:
        return jsonify({"ok": False,
                        "error": "shared must be a JSON object of what "
                                 "the shared panel holds, asked by a "
                                 "stage's tab"}), 400
    if reuses:
        # A CALCULATION THAT REUSES A FINISHED ONE (`engines/transport.md`
        # § 2a.16): both surfaces drawn from that calculation's template,
        # its shared values fixed, and a rung's tab holding its own
        # overrides.
        from molbuilder.transport.reuse import ReuseError, reused
        try:
            finished, rel = reused(reuses, tree_root=projects_root())
            template = _T.read_template(finished.template_text)
        except (ReuseError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        from molbuilder.transport.reuse import Extension
        extension = Extension(rel, finished.task)
        source = {"kind": "reuses", "name": extension.name}
        if rung is not None:
            held = extension.own(rung)
        # ...AND WHAT ITS CITATION ASKS OF THE PERSON, by the one rule the
        # citation's own panel asks (`asked_of_the_person`): a pair's
        # pseudopotential directory is drawn, kept as its value, and
        # carried by the way out as the person's.  A citation that no
        # longer resolves asks nothing here; its describe says why.
        try:
            _, cited = resolve_citation(citation_of(finished.task),
                                        projects_root())
            asks = asked_of_the_person(cited)
        except ComposeError:
            asks = ()
    elif citation:
        try:
            _, cited = resolve_citation(citation, projects_root())
        except ComposeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        try:
            # THE CITATION READ ONCE, for the template and its source.
            answers = citation_answers(cited)
        except ValueError as exc:
            # A CITED VALUE THIS BUILD CANNOT READ, refused by name
            # (`citation_defaults`; a 500 until 2026-10-19).
            return jsonify({"ok": False, "error": str(exc)}), 400
        try:
            template = _T.read_template(
                _panel_template(answers, shared, label="transport",
                                engine=engine))
        except ValueError as exc:
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
    if reuses and extension.fixes(surface=surface, rung=rung):
        # AN EXTENSION'S KEPT VALUES (`engines/transport.md` § 2a.16,
        # *Extending it*): the shared panel and the rungs that run once,
        # every field drawn at the value the finished calculation ran with,
        # never a control and never collected -- the one door's lock
        # (`transport.reuse.Extension.locked`).
        for sec in schema["sections"]:
            for f in sec["fields"]:
                if not f.get("locked"):
                    f["locked"] = extension.locked(f, rung=rung)
    response: Dict[str, Any] = {"ok": True, "surface": surface,
                                "engine": engine, "source": source}
    if surface == "rung":
        response["rung"] = rung
        notes = rung_notes()
        response["rungs"] = [{"name": n, "index": i, "note": notes[n]}
                             for i, n in enumerate(TRANSPORT_STAGES, start=1)]
        if held is not None:
            # WHAT THE RUNG'S TAB HOLDS of the calculation it reuses -- its
            # own overrides, which the tab starts from.
            response["held"] = held
        # NO SECOND FILTER HERE: which fields a rung's tab offers is the
        # schema builder's, by the catalogue's markers (`transport.md`
        # § 3.8.2; the run settings dropped by `surface="rung"`, 2026-10-18).
    response["schema"] = schema
    return jsonify(response)


def _panel_template(answers, shared: Dict[str, Any], *, label: str,
                    engine: str) -> str:
    """The template a transport describe writes, from what the shared panel
    holds and the citation's ``answers`` -- the one text the describe
    writes (`transport.describe`: the values typed by `describe.shared_of`,
    the text by `citation_defaults.transport_template_text`), which the
    schema route draws both surfaces from, so the tab cannot show one
    calculation and describe another (plan § 5w K7).  A blank is written
    valueless: a `citation` row (§ 3.8.3), and a blank electronic state is
    "work it out" on the whole junction at prep, which the chemistry card
    beside the panel shows.

    Raises ``ValueError`` -- the value, or a citation the template's door
    refuses (a cited run that carried a net charge, ES7).
    """
    from molbuilder.transport.citation_defaults import transport_template_text
    from molbuilder.transport.describe import shared_of
    chosen, blank = shared_of(shared, engine=engine, calculation="transport")
    return transport_template_text(answers, label=label, blank=blank,
                                   **chosen)


