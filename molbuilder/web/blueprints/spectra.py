"""Spectra blueprint -- the page and the ARTIFACT reader.

Routes:

    GET  /spectrum-calculation          the page (rendered template).
    POST /api/spectra/load              read the <label>.spectra.json the
                                        Results tab's dropdown picked into
                                        the typed SpectraResults shape
    POST /api/spectra/frameset          a mode of the result shown, as a
                                        frame set

The tab's compute half renders the CATALOGUE's vibration form (``GET /api/build/schema/pyscf?calculation=
vibration``, build.py) and hands the description to Task setup
(``POST /api/task-setup/handover``); the deck is written by ``prep``
on the machine that runs it (docs/execution/script-preparation.md).

Response shapes mirror Build's: every endpoint returns
``{"ok": True, ...}`` on success or ``{"ok": False, "error":
"<msg>"}`` on failure.

The load path is the file the Results tab's dropdown picked, ``{"path":
...}`` -- the one road a result reaches the page by (`web/spectra.md`
§ 7).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, render_template, request
from ._shared import err


from molbuilder.sidecars.spectra import (
    SpectraJsonError,
    SpectraJsonFieldError,
    SpectraJsonMalformedError,
    SpectraJsonNotFoundError,
    SpectraJsonSchemaError,
    parse_spectra_json,
)
from ...spectra.methods import with_ir_route


bp = Blueprint("spectra", __name__)




# ===================================================================== #
# Page route                                                            #
# ===================================================================== #


@bp.route("/spectrum-calculation")
def spectrum_calculation_page():
    """Render the Spectrum calculation tab.

    Heavy lifting is done by API routes elsewhere; this just hands
    back the template.  On load the tab's JS renders the CATALOGUE's
    vibration form (``/api/build/schema/pyscf?calculation=vibration``,
    build.py) through the shared form renderer.
    """
    return render_template("spectra.html")


# ===================================================================== #
# /api/spectra/load  -- parse a .spectra.json into typed results        #
# ===================================================================== #


def _page_constants():
    """The physical constants the spectrum page converts the file's numbers
    with, from their one home (`molbuilder.constants`) -- served with every
    load rather than copied into the browser (`architecture.md` § 3; user,
    2026-09-27: one source per fact).  Named by what each converts.  And
    each engine as a person reads its name, from the engine table, which the
    run summary's Engine row says a result's engine by."""
    from molbuilder import constants as C
    from molbuilder.engine_table import engine, engines
    from molbuilder.vibration_frames import OFFERED_NODES
    from molbuilder.spectra import activity as A
    return {
        # EACH ENGINE'S NAME, from its one home (the engine table), as the
        # build door serves the cards theirs (`build._analyze_response`).
        "engine_words": {name: engine(name).words for name in engines()},
        "hartree_ev":          C.HARTREE_EV,               # 1 Eh in eV
        "hartree_kcal_mol":    C.HARTREE_EV / C.KCAL_MOL_EV,  # 1 Eh in kcal/mol
        # each channel's band floor: a lane whose strongest real value is at
        # or below it has no band (`engines/vibration.md` § 6.6)
        "ir_present_floor_km_mol":    A.CHANNEL_PRESENT_FLOOR_IR_KM_MOL,
        "raman_present_floor_a4_amu": A.CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU,
        # the frame counts the doors offer, the default first
        # (`vibration_frames.OFFERED_NODES`, `web/spectra.md` § 4.3)
        "frameset_nodes": list(OFFERED_NODES),
    }


def _temperature_k(body):
    """The temperature the tab's animation asks its thermal spreads at --
    ``temperature_k`` in the JSON body, as a float, ``None`` when the
    request names none; a value that is not one is refused with the reason
    (a 400, `web-api.md` § 2), by the rule every door shares
    (`vibration.checked_temperature_k`)."""
    raw = (body or {}).get("temperature_k")
    if raw is None or raw == "":
        return None
    from molbuilder.vibration import checked_temperature_k
    return checked_temperature_k(raw)


def _framesets(results, path, asked: Optional[Dict[str, Any]]):
    """EACH MODE'S READOUT FOR THE DIALOG'S STATE (`web/spectra.md` § 4.3)
    -- ``(state, {mode number: block})``: at ``asked``, else at the thermal
    state at the record's temperature, the box the dialog starts at; ``None``
    where neither is stated or the result holds no geometry yet.  Each block
    is the motion toolkit's (`vibration_frames.readout`) on the result's own
    modes; a state the table refuses is a ``ValueError`` (a 400).  A
    result with no geometry yet is asked as such (``atoms_weighted``, what
    its `vibration` is refused on), never a caught refusal."""
    from molbuilder.vibration_frames import readout
    if results.atoms_weighted() is None:
        return None
    v = results.vibration(read_path=str(path))
    state = asked
    if state is None:
        if v.record_temperature_k is None:
            return None
        state = {"temperature_k": float(v.record_temperature_k)}
    return state, {m.index_1based: readout(m, state) for m in v.modes}


def _view_asked(body, key: str, read_ask) -> Optional[Dict[str, Any]]:
    """What the tab asks of a view of the result (`web/spectra.md` § 3.2,
    § 3.3, § 4.3) -- ``key`` in the JSON body, read by the one reading every
    door of that view shares (``read_ask``: `spectra.infrared.read_ask` for
    ``infrared``, `spectra.raman.read_ask` for ``raman``,
    `vibration_frames.read_state_ask` for ``frameset_state``).  ``None``
    when not asked; a malformed ask is refused with the reason (a 400)."""
    raw = (body or {}).get(key)
    if raw is None:
        return None
    return read_ask(raw)


def _loaded(results, run: Optional[Dict[str, Any]] = None,
            temperature_k: Optional[float] = None,
            infrared: Optional[Dict[str, Any]] = None,
            framesets=None,
            raman: Optional[Dict[str, Any]] = None, *, path):
    """The one success reply for /api/spectra/load — the typed results as a
    dict, plus what only the server can work out.

    ``temperature_k`` is the animation's temperature: each mode is served
    its ``thermal_spread_amu12_ang`` at it, from the one derivation
    (`vibration.thermal_spread_of`, `web/spectra.md` § 4.1); asked for
    none, no mode carries it.  ``framesets`` is the frame-set dialog's
    readout (:func:`_framesets`): each mode is served its ``frameset`` --
    its mean number of quanta, its spread and the outermost move for each
    frame count (`vibration_frames.readout`, § 4.3) -- at the state the dialog
    asked, else at the record's temperature, ``results.frameset_state``
    the state said back; the tab holds no constant and derives nothing.

    ``infrared`` is what the tab asks of the result's infrared -- the
    light's direction and the partition, along z by Hirshfeld charges when
    not asked: on a SIESTA result stating its charges, the free atoms'
    view, with every load; on a PySCF result holding atoms, the stored
    dipole change along the light's direction -- ``results.infrared``, from
    the one derivation (`spectra.infrared`, `web/spectra.md` § 3.2).  A view
    the result cannot give -- a partition it does not state, no stored
    vector -- is ``results.infrared_said``, the refusal in its words, and
    the result still loads: the tab sends the person's remembered choice
    with every load, and another result need not hold it.

    ``raman`` is what the tab asks of a SIESTA result's optical calls --
    the laser and the width (`engines/vibration.md` § 5.11 § 10):
    ``results.raman``, the view from the one derivation
    (`spectra.raman.raman_view_of`), over the optical calls' data file
    beside the result at ``path``; a view the result cannot give -- no
    optical route, a result opened without its folder, a width below the
    run's -- ``results.raman_said``, and the result still loads.

    WHY THE PAYLOAD IS NOT SIMPLY ``to_dict()``.  ``to_dict`` is the ON-DISK
    format: ``from_dict(to_dict(x)) == x``, byte-equal modulo float
    formatting, and the writer writes it -- the derived values its schema
    declares included (`engines/vibration.md` § 6.6).  A view is not one of
    them, so this wraps it: the file's own fields, unchanged, plus a computed
    one.

    WHAT IS ADDED, and why it cannot be computed in the browser.
    ``motion_share_by_element`` says which atoms a mode belongs to -- 91%
    carbon for a ring stretch -- and that needs atomic masses, because
    hydrogen has the largest displacement in almost every mode of an organic
    molecule while carrying almost none of the motion.  The browser has no
    masses.  Each free atom's share (``motion_share``, weighted by the masses
    the modes were weighted by) is the result's own: stored from schema 15,
    checked on read against the one derivation and worked out again by it
    (``to_dict``, `engines/vibration.md` § 6.6) -- for an older result,
    worked out on read.  This sums them by each atom's element, its species
    label's name (``SpectraResults.atoms_weighted``): a view of the panel's,
    so made here, where the request is.

    A mode that states no shares -- a result before schema 7, which states
    no masses, whose label names no element the mass table knows -- is
    served without the field: the panel drops one clause, and the spectrum
    still opens.
    """
    payload = results.to_dict()
    # THE PAGE'S CONSTANTS RIDE WITH THE RESULTS: the reply the page already
    # fetches, so serving them costs no round trip (`_page_constants`).
    payload["constants"] = _page_constants()

    # THE METHODS PARAGRAPH GAINS THE ONE FACT IT COULD NOT CARRY.  It is
    # composed in the deck composer, before the job runs, and which
    # dmu/dR route was taken is only settled inside the job -- so the
    # stored text is route-neutral by necessity.  Here the results ARE in
    # hand, so the sentence can be added, from the same home the rest of
    # the prose comes from.  Computed at load for the reason the element
    # shares are: every result already on disk gains it on open.
    if payload.get("methods_text"):
        payload["methods_text"] = with_ir_route(
            payload["methods_text"],
            payload.get("ir_route") or "",
            payload.get("ir_fd_step_ang"))

    # THE ELEMENT SHARES, a view over each mode's per-atom shares as the
    # result states them (`motion_share`, checked on read and worked out
    # again, `engines/vibration.md` § 6.6): each free atom's share
    # summed by its element, its species label's name, as the result reads
    # it (`SpectraResults.atoms_weighted`).
    from molbuilder.vibration import shares_by
    sharing = [mode for mode in payload.get("modes") or []
               if mode.get("motion_share") is not None]
    if sharing:
        free_elements = results.atoms_weighted(payload["free_atom_idxs"])[0]
        for mode in sharing:
            mode["motion_share_by_element"] = shares_by(free_elements,
                                                        mode["motion_share"])
    if temperature_k is not None:
        # the temperature the animation's spreads are at, said with them
        payload["temperature_k"] = temperature_k
        from molbuilder.vibration import thermal_spread_of
        for mode in payload.get("modes") or []:
            mode["thermal_spread_amu12_ang"] = thermal_spread_of(
                mode, temperature_k)
    # THE FRAME SET'S READOUT (`web/spectra.md` § 4.3): at the state the
    # dialog asked, else at the record's temperature; neither, no mode's
    # frameset.  The state said with the numbers.
    if framesets is not None:
        state, blocks = framesets
        payload["frameset_state"] = dict(state)
        for mode in payload.get("modes") or []:
            mode["frameset"] = blocks.get(mode.get("index_1based"))
    # THE FREE ATOMS' INFRARED, where the result states its charges
    # (`web/spectra.md` § 3.2); and, for a result that carries none, why,
    # in its own facts.
    from molbuilder.spectra.infrared import (SOURCE_DIPOLE, ORIENTATION_FREE,
                                             InfraredError, infrared_absent,
                                             infrared_view_of, orientation_of,
                                             read_ask, source_of)
    from molbuilder.spectra.infrared import asked as infrared_asked
    # WHETHER THE RUN ASKED FOR ITS INFRARED, read once from the result's
    # own facts, older files included (`infrared.asked`): the page's
    # column, CSV, sentence and summary read this, never `phase_ir`.
    payload["infrared_asked"] = infrared_asked(results)
    from molbuilder.spectra.methods import with_infrared_view
    # WHERE ITS INFRARED COMES FROM, and held or free -- whether the light's
    # direction is asked at all (`web/spectra.md` § 3.2).
    source = source_of(results)
    if source is not None:
        payload["infrared_source"] = source
        payload["infrared_orientation"] = orientation_of(results)
    # ...A RESULT THAT CARRIES NONE says why, once, from its own facts
    # (`infrared.infrared_absent`): not computed yet, Compute IR off,
    # written before charges were kept, or recorded none.
    absent = infrared_absent(results)
    if absent is not None:
        payload["infrared_absent"] = absent
    # WHAT IS ASKED -- the defaults when nothing is -- answered by the one
    # door (`infrared.infrared_view_of`) for every result with an infrared,
    # but one: a free result whose infrared is the engine's own shows the
    # file's column, the orientation average, and no choice (§ 3.2).
    asked = infrared if infrared is not None else read_ask({})
    view = None
    if source is not None and not (
            source == SOURCE_DIPOLE
            and payload["infrared_orientation"] == ORIENTATION_FREE):
        try:
            view = infrared_view_of(results, **asked)
        except InfraredError as exc:
            payload["infrared_said"] = str(exc)
    if view is not None:
        payload["infrared"] = view.to_wire()
        # THE WRITE-UP NAMES WHAT IS SHOWN (`web/spectra.md` § 9b.3).
        if payload.get("methods_text"):
            payload["methods_text"] = with_infrared_view(
                payload["methods_text"], payload["infrared"])
    # THE RAMAN AT THE LASER ASKED, from the optical calls' data file beside
    # the result -- one opened from its folder (`engines/vibration.md`
    # § 5.11 § 10).
    if raman is not None:
        from molbuilder.spectra.methods import with_raman_view
        from molbuilder.spectra.raman import (raman_view_of,
                                              require_optical_route)
        try:
            require_optical_route(results)
            payload["raman"] = raman_view_of(path, result=results,
                                             **raman).to_wire()
        except (ValueError, OSError) as exc:
            payload["raman_said"] = str(exc)
        else:
            if payload.get("methods_text"):
                # THE KEYS ARE THE PARAGRAPH'S, its new citation with them.
                from molbuilder.spectra.methods import extract_citation_keys
                payload["methods_text"] = with_raman_view(
                    payload["methods_text"], payload["raman"])
                payload["bibliography_keys"] = extract_citation_keys(
                    payload["methods_text"])
    # HOW THE RUN THE FILE BELONGS TO IS DOING -- the one door's answer, which
    # the viewer follows; ``None`` for a file no run of ours holds
    # (`web/results.md` § 4.1).
    return jsonify({"ok": True, "results": payload, "run": run})


@bp.route("/api/spectra/load", methods=["POST"])
def api_spectra_load():
    """Read the ``<label>.spectra.json`` the Results tab's dropdown picked --
    ``{"path": ...}``, the one road a result reaches the page by
    (`web/spectra.md` § 7) -- into the typed :class:`SpectraResults`,
    returned as a JSON-safe dict the UI consumes directly.

    Returns on success::

        {"ok": True, "results": <SpectraResults.to_dict() and what only
         the server works out>, "run": <runstatus.run_answer(path), or None>}

    On failure the response carries a structured error that maps
    1-to-1 to the parser's exception hierarchy so the UI can
    decide how to react:

      * SpectraJsonNotFoundError  -> 404
      * SpectraJsonSchemaError    -> 422 with ``expected`` / ``actual``
                                     so the UI can render an "update
                                     molbuilder" hint
      * SpectraJsonMalformedError -> 400
      * SpectraJsonFieldError     -> 400
    """
    body = request.get_json(silent=True) or {}
    # WHAT THE TAB ASKS -- the temperatures, the views -- each by the one
    # reading its doors share.
    try:
        temperature_k = _temperature_k(body)
        from molbuilder.vibration_frames import read_state_ask
        frameset_state = _view_asked(body, "frameset_state", read_state_ask)
        from molbuilder.spectra import infrared as _ir, raman as _raman
        infrared = _view_asked(body, "infrared", _ir.read_ask)
        raman = _view_asked(body, "raman", _raman.read_ask)
    except ValueError as exc:
        return err(str(exc), 400, kind="field")
    path = body.get("path")
    if not path:
        return err(
            "no path -- the result is the file the "
            "Results tab's dropdown picked",
            400)
    # THE FENCE (web-api.md § 2.1).  The parser -- correctly, being a
    # generic reader -- does a bare open(), so without it any path the
    # server process could read would have its EXISTENCE readable from
    # here, which is filesystem enumeration.  The rate limiter's
    # attack-string screen reads the URL, and this path arrives in a
    # JSON body, so nothing else stands here.
    from .files import _resolve_within_roots
    path = _resolve_within_roots(path)
    # THE RUN'S END IS READ BEFORE THE FILE'S LAST READ (`web/results.md`
    # § 4.1): the run this file belongs to is asked after the file is
    # read, and when it is no longer live the file is read again if it
    # changed in between -- so the viewer stops on the file's last state.
    # THE FILE'S VERSION by the ending reader's rule (`_run_ending.version_of`).
    from molbuilder.jobset.runstatus import run_answer
    from molbuilder.parse.engines._run_ending import version_of
    try:
        before = version_of(path)
        results = parse_spectra_json(path)
        run = run_answer(path)
        if not (run and run["live"]) and version_of(path) != before:
            results = parse_spectra_json(path)
    except SpectraJsonError as exc:
        return _err_load(exc)
    try:
        framesets = _framesets(results, path, frameset_state)
    except ValueError as exc:
        return err(str(exc), 400, kind="field")
    return _loaded(results, run=run, temperature_k=temperature_k,
                   infrared=infrared, raman=raman, path=path,
                   framesets=framesets)


# ===================================================================== #
# /api/spectra/frameset                                                 #
# ===================================================================== #


#: The write door's own keys, beside the state's (`web-api.md`'s row).
_WRITE_KEYS = ("path", "mode", "nodes", "out")


@bp.route("/api/spectra/frameset", methods=["POST"])
def api_spectra_frameset():
    """A MODE'S FRAME SET, written by the doors' frame set
    (`vibration_frames.frameset_from_mode`: the run door, the state's own
    rule and the motion toolkit's one writer; `engines/vibration-frames.md`
    § 4, § 5, § 6; `web/spectra.md` § 4.3): ``{path, mode, nodes, out}``
    and the state's arguments (``temperature_k``, ``quanta``,
    ``mean_quanta``, as the dialog asks them) -- the
    vibration result the page shows, whose folder is its run's, the frame
    count (the doors' default when unsaid), and the pair's ``.xyz``, each
    path inside the tree (the fence, `web-api.md` § 2.1), the ``.xyz``
    where a file may be written (`files._writable`).  Answers ``{ok, path,
    frames, said}``, or the writer's refusal in its words (400)."""
    from molbuilder.persist import unknown_keys
    from molbuilder.vibration_frames import (STATE_ASK_KEYS, FrameSetError,
                                             frameset_from_mode)
    from molbuilder.projects import projects_root
    from .files import _resolve_within_roots, _writable
    body = request.get_json(silent=True) or {}
    # A KEY THE DOOR DOES NOT TAKE IS REFUSED, never ignored -- a state
    # misspelled would write another state than the one asked.
    said = unknown_keys(body, _WRITE_KEYS + STATE_ASK_KEYS)
    if said:
        return err(said, 400)
    # THE RUN IS THE RESULT'S FOLDER, derived here as the one export
    # route's is (`GET /api/export`).
    run = _resolve_within_roots(str(body.get("path") or "")).parent
    # THE PAIR'S .xyz, WHERE A FILE MAY BE WRITTEN (the files layer's one
    # write-path check).
    out = _writable(_resolve_within_roots(str(body.get("out") or "")))
    try:
        mode = int(body.get("mode"))
        rule = ({} if body.get("nodes") is None
                else {"nodes": int(body["nodes"])})
    except (TypeError, ValueError):
        return err(
            "mode and nodes are whole "
            "numbers",
            400)
    root = projects_root()
    try:
        got = frameset_from_mode(
            run, mode=mode, out=out, **rule,
            **{k: body.get(k) for k in STATE_ASK_KEYS})
    except FrameSetError as exc:
        return err(str(exc), 400)
    return jsonify({"ok": True, **got.to_wire(root)})


# ===================================================================== #
# Helpers                                                               #
# ===================================================================== #


def _err_load(exc: SpectraJsonError):
    """Map a parser exception class to the HTTP response shape.

    Returns (Flask response, status_code) -- the dispatcher decides
    HTTP code based on exception type so the UI can decide how to
    react without parsing the message string.  Schema mismatches
    carry their numeric versions so the UI can render an "update
    molbuilder" hint directly.
    """
    if isinstance(exc, SpectraJsonNotFoundError):
        return err(str(exc), 404, kind="not_found")
    if isinstance(exc, SpectraJsonSchemaError):
        return err(
            str(exc),
            422,
            kind="schema_mismatch",
            expected_version=exc.expected,
            actual_version=exc.actual)
    if isinstance(exc, SpectraJsonMalformedError):
        return err(str(exc), 400, kind="malformed")
    if isinstance(exc, SpectraJsonFieldError):
        return err(str(exc), 400, kind="field")
    # Generic SpectraJsonError catch-all.
    return err(str(exc), 400, kind="parse_error")


__all__ = ["bp"]
