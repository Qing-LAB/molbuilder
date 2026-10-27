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

import dataclasses
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
    if finished is None and described(cited_path):
        try:
            kind = _task.of(cited_path).calculation
        except ValueError:
            kind = "transport"
        if kind == "transport":
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
    composed = None
    try:
        composed = compose_junction(citation, tree_root=root,
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
    from molbuilder.jobset.runstatus import jobset_status
    from molbuilder.transport.compose import ComposeError, recorded_junction
    from molbuilder.transport.reuse import Extension, ReuseError, reused
    try:
        calc, rel = reused(folder, tree_root=root)
    except ReuseError as exc:
        return jsonify({"ok": True, "citation": None, "summary": str(exc),
                        "findings": []}), 200
    from molbuilder.jobset.ladder import stage_config
    from molbuilder.jobset.reuse import SHARING_RUNG
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
    cfg = stage_config(task, SHARING_RUNG, calc.template_text)
    window = (f"{cfg.transmission_emin_ev:g}…{cfg.transmission_emax_ev:g} "
              f"eV at {cfg.transmission_n_points} points")
    steps = jobset_status(calc).steps
    stands = ("every group finished"
              if steps and all(s.state == "finished" for s in steps) else
              "; ".join(f"group {s.group} {s.words.rstrip('.')}"
                        for s in steps))
    summary = " · ".join([
        "TRANSPORT CALCULATION", Extension(rel, task).name,
        f"its junction, cited from {task.slots['junction']}",
        str(task.treatment),
        ", ".join(str(float(v)) for v in task.bias) + " V", window,
        stands] + ([f"its junction's record does not read: {unread}"]
                   if unread else [])
        + ([unstated] if unstated else []))
    # ITS FRAMES, as any citation's are said (§ 3.8.1, § 2a.9).
    frame_set = answer_said(junction if junction is not None
                            and junction.n_frames > 1 else None)
    return jsonify({
        "ok": True, "citation": task.slots["junction"], "reuses": rel,
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
        # keeps, what it offers, and what each control says -- drawn as
        # they are.
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
    from molbuilder.jobset.reuse import finished_window
    from molbuilder.projects import projects_root
    from molbuilder.transport.reuse import ReuseError, reused
    from molbuilder.transport.window import fitted
    try:
        finished, _rel = reused(str(request.args.get("reuses") or ""),
                                tree_root=Path(projects_root()).resolve())
        lo = float(request.args.get("lo", ""))
        hi = float(request.args.get("hi", ""))
        fit = fitted(finished_window(finished), lo, hi,
                     int(request.args.get("finer") or 1))
    except (ReuseError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "words": fit.words(), "emin": fit.emin,
                    "emax": fit.emax, "points": fit.points})


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
    from .files import _PickerError, _resolve_within_roots
    try:
        path = _resolve_within_roots(raw)
    except _PickerError as exc:
        return None, (jsonify({"ok": False, "error": exc.message}),
                      exc.status)
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
    template and ``task.json`` -- into the folder the person chose.

    There is no hand-over for the composite (user ruling 2026-08-29):
    the other kinds hand to Task setup because that tab owns questions
    they cannot answer — shape, stages, what varies.  Transport has
    none open: the ladder and the hierarchical shape are fixed by the
    kind (`task.fixed_by_kind`), the identity derives from the citation,
    the knobs ride the stages' override bags.  So this door writes the
    finished description in one act, through the writer `molbuilder describe`
    takes (`describe.write_description`), into a folder that holds none
    -- a description is written by the server's door that validated it
    (`web/handover-procedure.md` § 4).  Once written it is Task setup's
    to edit, as every kind's is (`engines/transport.md` § 3.8.2a).

    Validation is the codec's and the description's own check: the
    ``Task`` construction below is the gate `task.of` runs, and the
    task preflight (gate ③) is the one `molbuilder describe`
    and the Task-setup save run -- its errors refuse, its warnings ride
    ``notices``.  The citation resolves through the same door prep
    composes through.

    With ``reuses`` -- a transport calculation's folder, picked to reuse it
    (`engines/transport.md` § 2a.16) -- it writes what `molbuilder describe
    --reuses` writes, through the same writer (`transport.reuse.
    write_reusing`): that calculation's description and its label, its
    junction -- its own, or a frame set of it, ``junction`` naming which --
    as the cards change its voltages, its rungs' values and its shared
    ones; its template, each value the panel changed said to be the
    person's; its job request.
    """
    from molbuilder.projects import projects_root
    from molbuilder.task import Task, derive_run, fixed_by_kind
    from molbuilder.transport.compose import (ComposeError,
                                             refuse_failed_citation,
                                             resolve_citation)
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

    # THE CALCULATION IT REUSES, or THE CITATION -- what the description
    # starts from (`engines/transport.md` § 2a.16, § 3.1).
    finished = cited = None
    reuses_raw = str(body.get("reuses") or "")
    if reuses_raw:
        from molbuilder.transport.reuse import ReuseError, reused
        try:
            finished, reuses_rel = reused(reuses_raw, tree_root=root)
        except ReuseError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
    else:
        try:
            _, cited = resolve_citation(citation, projects_root())
            # A RUN THAT FAILED, refused before anything is written
            # (`engines/transport.md` § 3.1).
            refuse_failed_citation(cited, citation)
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
    if dest is not None and not dest.is_dir():
        return jsonify({"ok": False,
                        "error": f"{dest_raw} is not a folder"}), 400
    try:
        if dest is not None:
            shown = str(dest.relative_to(root))
            refuse_described(dest, shown=shown)
            refuse_inside_citation(
                dest, finished.base if finished is not None else cited.path,
                shown=shown, what=("the finished calculation"
                                   if finished is not None
                                   else "the cited junction"))
    except (DescribeError, ComposeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409

    # AN EXTENSION'S KEPT RUNGS hold the finished calculation's own values
    # where the page sent none (it never sends a locked field); what it
    # does not take is refused once the description is built, below.
    ext = None
    if finished is not None:
        from molbuilder.transport.reuse import Extension
        ext = Extension(reuses_rel, finished.task)
        for _r in ext.fixed:
            bags[_r] = bags.get(_r) or ext.own(_r)

    # Refused HERE, not at prep on the cluster: an unknown knob names
    # itself while changing it is still free.  THE VOCABULARY IS WHAT PREP
    # CAN RESOLVE, asked rather than listed
    # (`stages.resolvable_override_names`).  Whether a rung's override may
    # hold a known one -- shared, fixed by the rung, a run setting, read by
    # that rung -- and its value's own rules are the description's own
    # check's, below: the setting door, asked with the rung, as at prep.
    _shared_names = _T.shared_by_every_stage(engine, "transport")
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
    for _name in shared_chosen:
        if _name not in _shared_names:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is not a shared value of "
                                     f"this calculation; the shared panel "
                                     f"carries the items the catalogue "
                                     f"marks `shared` for transport"}), 400
    try:
        # THE CALCULATION IT REUSES, as the cards change it -- its label
        # and its rename its own, its junction the finished one or a frame
        # set of it, which the writer checks (§ 2a.16, `reuse.citing`); or
        # a new one.
        from molbuilder.transport.reuse import citing
        task = (citing(dataclasses.replace(
            finished.task, reuses=reuses_rel, bias=bias,
            low_bias_approximation=approx,
            varies=tuple(sorted({n for b in bags.values() for n in b})),
            stages=tuple(stages_for_transport(bags))), citation)
            if finished is not None else Task(
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
            stages=tuple(stages_for_transport(bags))))
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
    # AN EXTENSION KEEPS WHAT THE FINISHED CALCULATION COMPUTED ONCE FOR
    # EVERY VOLTAGE (`engines/transport.md` § 2a.16, *Extending it*): the
    # rungs it keeps, the shared values, its voltages, its rename and its
    # label -- a change of any is another calculation, started from the
    # junction (the way out the page offers), refused BESIDE THE CONTROL IT
    # NAMES, nothing written (`transport.reuse.Extension.refusal`); its
    # junction, the finished one or a frame set of it, is the writer's to
    # check (`write_reusing`, below).
    if ext is not None:
        from molbuilder.issues import Issue
        no = ext.refusal(task, shared=shared_chosen)
        if no is not None:
            return jsonify({"ok": False, "error": no.message,
                            "findings": _issues_to_json([Issue(
                                "error", no.message,
                                where=f"task.{no.where}" if no.where
                                else "")])}), 400
    # The template's text through the one door the schema route draws both
    # surfaces from (`_panel_template`) -- a value its field cannot take, or
    # a citation it refuses (a cited run that carried a net charge, ES7),
    # said here by name rather than failing the whole response.
    from molbuilder.transport.citation_defaults import check_pseudopotentials
    try:
        if finished is not None:
            _tmpl_text = _reused_template(finished, shared_chosen)
        else:
            check_pseudopotentials(cited, shared_chosen.get("psml_lib"))
            _tmpl_text = _panel_template(cited.path, shared_chosen,
                                         label=task.label)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # THE DESCRIPTION'S OWN CHECK (gate ③, plan § 5w K3).  The same
    # function the Task-setup save and `molbuilder describe` run: its errors
    # refuse in the save's own words, its warnings ride `notices`.
    from molbuilder.validation.task import (preflight as _task_preflight,
                                            config_class_for as _cfg_cls_for)
    # THE CALCULATION'S START, described with its job request: each rung's
    # group by the one numbering rule (`describe.Description`).
    _desc = Description(task=task, template=_tmpl_text,
                        request=(finished.request if finished is not None
                                 else None))
    _pf = _task_preflight(task, template_text=_tmpl_text,
                          request=_desc.request)
    # ...AND THE K-POINT MESHES EACH RUNG WILL WRITE, judged by the one
    # check prep's gate runs (`kmesh.check`; `engines/transport.md` § 0.3b)
    # on the junction's own axis kinds -- so one point along a direction
    # the electrodes repeat in is said here, beside its control, and not
    # first at prep.
    from molbuilder.transport.compose import (junction_axis_kind,
                                              recorded_junction)
    from molbuilder.transport.deck import ladder_mesh_findings
    try:
        _kinds = (junction_axis_kind(cited) if cited is not None else
                  recorded_junction(finished.base).structure.axis_kind)
    except (ComposeError, AttributeError):
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

    # WRITTEN HERE, every file or none, through the writer `molbuilder describe`
    # takes (`describe.write_description`; `web/handover-procedure.md` § 4:
    # a description is written by the server's door that validated it).
    # THREE FILES: a transport calculation is `task.json`, its template and
    # its job request, like every other kind's (TR1).  Its values are
    # DEFAULTED FROM THE
    # CITED RUN (§ 2a.7, ruling 1) and are the person's to change
    # afterwards; a `citation` row nobody answered stays VALUELESS
    # (§ 3.8.3).
    from molbuilder.issues import Issue
    if plan_only:
        return _plan_of(_desc, finished, root=root, task=task,
                        notices=list(_pf), cfg=_cfg_cls_for(task))
    notes: list = []
    manifest = None
    try:
        if finished is not None:
            from molbuilder.transport.reuse import ReuseError, write_reusing
            try:
                written, notes = write_reusing(_desc, dest, finished,
                                               tree_root=root)
            except ReuseError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            # WHAT IT WILL REUSE, AND WHAT RUNS -- the one answer prep
            # composes (`jobset.reuse.decide`), said once it is described
            # (§ 2a.16, *Extending it*); nothing written.
            from molbuilder.calculation import open as _open_calc
            from molbuilder.jobset.errors import PrepError
            from molbuilder.jobset.reuse import decide
            try:
                _made = _open_calc(dest)
                _reuse = decide(_made, _made.request.numbers)
                manifest = (_reuse.lines(dest) if _reuse is not None
                            else None)
            except PrepError as exc:
                notes = list(notes) + [Issue(
                    "warn", f"what it will reuse is not known yet -- {exc}")]
        else:
            written = write_description(_desc, dest)
    except (DescribeError, OSError) as exc:
        return jsonify({"ok": False,
                        "error": f"nothing was written: {exc}"}), 500
    return jsonify({
        "ok": True,
        "label": task.label,
        "written": [p.name for p in written],
        # A CALCULATION THAT REUSES A FINISHED ONE: what it will copy, and
        # what runs, in the manifest's words; null otherwise.
        "manifest": manifest,
        # WHAT THE REUSE SAYS OF ITSELF, each at its weight -- a finished
        # calculation carrying no composed junction, a frame set written in
        # its junction's place -- beside the description's own notes
        # (`issues.Issue`, the one finding).
        "notices": _issues_to_json(list(_pf) + list(notes),
                                   cfg=_cfg_cls_for(task)),
    })


def _plan_of(desc, finished, *, root, task, notices, cfg) -> Any:
    """THE PLAN ALONE (`engines/transport.md` § 2a.16, *On the page*): what
    the description ``desc`` would copy from ``finished`` and run -- the one
    answer prep composes (`jobset.reuse.decide`) -- worked out in a hidden
    folder of its own under the projects tree's root, which the reuse needs
    to find the finished calculation by its path, removed whatever happens.
    A new calculation copies nothing: no plan."""
    import shutil
    import tempfile
    from pathlib import Path
    from molbuilder.issues import Issue
    if finished is None:
        return jsonify({"ok": True, "plan": True, "manifest": None,
                        "notices": _issues_to_json(notices, cfg=cfg)})
    from molbuilder.calculation import open as _open_calc
    from molbuilder.describe import DescribeError
    from molbuilder.jobset.errors import PrepError
    from molbuilder.jobset.reuse import decide
    from molbuilder.transport.reuse import ReuseError, write_reusing
    staging = None
    try:
        staging = Path(tempfile.mkdtemp(dir=root, prefix=".molbuilder-plan-"))
        where = staging / task.label
        try:
            _written, notes = write_reusing(desc, where, finished,
                                            tree_root=root)
        except (ReuseError, DescribeError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        try:
            made = _open_calc(where)
            reuse = decide(made, made.request.numbers)
            manifest = reuse.lines(where) if reuse is not None else None
        except PrepError as exc:
            manifest = None
            notes = list(notes) + [Issue(
                "warn", f"what it will reuse is not known yet -- {exc}")]
    except OSError as exc:
        # THE TREE REFUSING THE HIDDEN FOLDER, said as Describe says it.
        return jsonify({"ok": False,
                        "error": f"the plan could not be worked out: "
                                 f"{exc}"}), 500
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return jsonify({"ok": True, "plan": True, "manifest": manifest,
                    "notices": _issues_to_json(notices + list(notes),
                                               cfg=cfg)})


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
                                 "rung's tab"}), 400
    if reuses:
        # A CALCULATION THAT REUSES A FINISHED ONE (`engines/transport.md`
        # § 2a.16): both surfaces drawn from that calculation's template as
        # the panel changes it, and a rung's tab holding its own overrides.
        from molbuilder.transport.reuse import ReuseError, reused
        try:
            finished, rel = reused(reuses, tree_root=projects_root())
            template = _T.read_template(_reused_template(finished, shared))
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
            _, cited = resolve_citation(finished.task.slots["junction"],
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
        response["rungs"] = [{"name": n, "index": i, "note": RUNG_NOTES[n]}
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


def _reused_template(finished, shared: Dict[str, Any]) -> str:
    """The template a calculation that reuses ``finished`` writes, from what
    the shared panel holds (`engines/transport.md` § 2a.16): the finished
    calculation's, each value the panel changed said to be the person's,
    through the one door (`template.with_values_set`) -- the one text the
    describe door writes and the schema route draws from, as
    :func:`_panel_template` is for a citation.  A BLANK IS NOT CHOSEN
    (`form-schema.md` § 1.1): a field the panel drew holding a value
    somebody gave, sent blank, was emptied -- written valueless, not
    chosen; one it drew blank, a value nobody chose, is as it was."""
    from ._shared import siesta_config_from_params
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.template import (config_from_template, read_template,
                                     with_values_set)
    text = finished.template_text
    was = config_from_template(text, SiestaConfig)
    drawn = {it.name for it in read_template(text).items
             if it.is_set and it.source != "default"}
    typed = {k: v for k, v in shared.items() if v not in (None, "")}
    cfg = siesta_config_from_params(typed, "transport")
    changed = {k: getattr(cfg, k) for k in typed
               if getattr(cfg, k) != getattr(was, k, None)}
    emptied = {k: None for k, v in shared.items()
               if v in (None, "") and k in drawn}
    for values, source in ((changed, "person"), (emptied, "default")):
        if values:
            text = with_values_set(text, SiestaConfig,
                                   calculation="transport", values=values,
                                   source=source)
    return text


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


