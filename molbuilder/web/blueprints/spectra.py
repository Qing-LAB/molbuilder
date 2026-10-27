"""Spectra blueprint -- the page and the ARTIFACT reader.

Routes:

    GET  /spectrum-calculation          the page (rendered template).
    POST /api/spectra/load              parse an uploaded <job>.spectra.json
                                        (or read JSON body) into the typed
                                        SpectraResults shape
    GET  /api/spectra/export            a vibration result's data files and
                                        its Raman view at a laser, as
                                        NetCDF or as tables -- one zip

The tab's compute half renders the CATALOGUE's vibration form (``GET /api/build/schema/pyscf?calculation=
vibration``, build.py) and hands the description to Task setup
(``POST /api/task-setup/handover``); the deck is written by ``prep``
on the machine that runs it (docs/execution/script-preparation.md).

Response shapes mirror Build's: every endpoint returns
``{"ok": True, ...}`` on success or ``{"ok": False, "error":
"<msg>"}`` on failure.

The load path:

  * Multipart: ``file=`` form-data field with the .spectra.json
    bytes.
  * JSON: ``{"path": "..."}`` to read from a local path (useful
    when the user runs the deck and wants to load results from
    a known job dir without manually uploading).
  * JSON: ``{"json": {...}}`` to parse a dict in-memory (for
    programmatic callers; matches parse_spectra_json_dict).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request


from molbuilder.sidecars.spectra import (
    SpectraJsonError,
    SpectraJsonFieldError,
    SpectraJsonMalformedError,
    SpectraJsonNotFoundError,
    SpectraJsonSchemaError,
    parse_spectra_json,
    parse_spectra_json_dict,
)
from molbuilder.spectra.results import motion_share_by_element
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
    2026-09-27: one source per fact).  Named by what each converts."""
    from molbuilder import constants as C
    from molbuilder.spectra import activity as A
    return {
        "hartree_ev":          C.HARTREE_EV,               # 1 Eh in eV
        "hartree_kcal_mol":    C.HARTREE_EV / C.KCAL_MOL_EV,  # 1 Eh in kcal/mol
        # each channel's band floor: a lane whose strongest real value is at
        # or below it has no band (`engines/vibration.md` § 6.6)
        "ir_present_floor_km_mol":    A.CHANNEL_PRESENT_FLOOR_IR_KM_MOL,
        "raman_present_floor_a4_amu": A.CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU,
    }


def _temperature_k(body, form, key: str = "temperature_k"):
    """A temperature the tab asks at -- ``temperature_k``, the animation's
    thermal spreads', or ``reference_temperature_k``, the frame-set
    dialog's reference's -- in the JSON body or the upload's form, as a
    float, ``None`` when the request names none; a value that is not one is
    refused with the reason (a 400, `web-api.md` § 2), by the rule every
    door shares (`spectra.derived.checked_temperature_k`)."""
    raw = (body or {}).get(key)
    if raw is None and form is not None:
        raw = form.get(key)
    if raw is None or raw == "":
        return None
    from molbuilder.spectra.derived import checked_temperature_k
    return checked_temperature_k(raw)


def _view_asked(body, form, key: str, read_ask) -> Optional[Dict[str, Any]]:
    """What the tab asks of a view of the result (`web/spectra.md` § 3.2,
    § 3.3) -- ``key`` in the JSON body, or as JSON text in the upload's
    form, read by the one reading every door of that view shares
    (``read_ask``: `spectra.infrared.read_ask` for ``infrared``,
    `spectra.raman.read_ask` for ``raman``).  ``None`` when not asked; a
    malformed ask is refused with the reason (a 400)."""
    raw = (body or {}).get(key)
    if raw is None and form is not None and form.get(key):
        try:
            raw = json.loads(form.get(key))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} is a JSON object: {exc}") from None
    if raw is None:
        return None
    return read_ask(raw)


def _loaded(results, run: Optional[Dict[str, Any]] = None,
            temperature_k: Optional[float] = None,
            infrared: Optional[Dict[str, Any]] = None,
            reference_temperature_k: Optional[float] = None,
            raman: Optional[Dict[str, Any]] = None, path=None):
    """The one success reply for /api/spectra/load — the typed results as a
    dict, plus what only the server can work out.

    ``temperature_k`` is the animation's temperature: each mode is served
    its ``thermal_spread_amu12_ang`` at it, from the one derivation
    (`spectra.derived.thermal_spread_of`, `web/spectra.md` § 4.1); asked for
    none, no mode carries it.  ``reference_temperature_k`` is the frame-set
    dialog's: each mode is served its ``reference_max_displacement_ang`` for
    three frames and five at it (`mode_frames.reference_max_displacement`,
    § 4.3), the run's own thermochemistry temperature when the dialog has
    named none -- the tab holds no constant and derives nothing.

    ``infrared`` is what the tab asks of the result's infrared: on a SIESTA
    result stating its charges, the named set's view; on a PySCF result
    holding atoms, the stored dipole change along the light's direction --
    ``results.infrared``, from the one derivation (`spectra.infrared`,
    `web/spectra.md` § 3.2).  A view the result cannot give -- a label its
    structure does not hold, a partition it does not state, no charges, no
    stored vector -- is ``results.infrared_said``, the refusal in its words,
    and the result still loads: the tab sends the person's remembered
    choice with every load, and another result need not hold it.

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
    masses.  A schema-7 result states its own (``equilibrium.masses_amu``,
    the masses its modes were weighted by), and the shares are weighted by
    them; an older one states none, and ``chemistry.atomic_mass`` -- the
    table every deck's masses are stated from -- weights them.  Shipping a
    periodic table into JavaScript would be a second copy of that table, so
    the server answers.

    COMPUTED AT LOAD, NOT STORED.  Every result already written -- including
    the ones on disk right now -- gains the field the moment it is opened; a
    schema bump would have left them showing nothing until re-run.  It costs
    one pass over each eigenvector.

    A mode whose shares cannot be worked out (an element ASE does not know)
    is served without the field rather than failing the load: the panel drops
    one clause, and the spectrum still opens.
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

    # THE ELEMENTS, from the structure the result states whole (v9; the
    # reader serves that shape for every file, `engines/vibration.md` § 6.10).
    from molbuilder.structure import Structure as _Structure
    elements = (list(_Structure.from_dict(payload["structure"]).elements)
                if payload.get("structure") else [])
    masses = (payload.get("equilibrium") or {}).get("masses_amu")
    # An empty free-atom list is "not recorded", not "no atom is free" -- a
    # result that never tracked the partition has one eigenvector row per atom,
    # which is what passing None means downstream.
    free = payload.get("free_atom_idxs") or None
    for mode in payload.get("modes") or []:
        rows = mode.get("eigenvector_canonical") or mode.get("eigenvector_display")
        if not rows or not elements:
            continue
        try:
            mode["motion_share_by_element"] = motion_share_by_element(
                elements, rows, free, masses_amu=masses)
        except (ValueError, KeyError):
            pass
    if temperature_k is not None:
        # the temperature the animation's spreads are at, said with them
        payload["temperature_k"] = temperature_k
        from molbuilder.spectra.derived import thermal_spread_of
        for mode in payload.get("modes") or []:
            mode["thermal_spread_amu12_ang"] = thermal_spread_of(
                mode, temperature_k)
    # THE REFERENCE DISPLACEMENT'S TEMPERATURE (`web/spectra.md` § 4.3): the
    # one the dialog asked, else the run's own thermochemistry temperature;
    # neither, no reference.  Said with the numbers.
    ref_t = reference_temperature_k
    if ref_t is None:
        ref_t = (payload.get("thermo") or {}).get("temperature_K")
    if ref_t is not None:
        from molbuilder.mode_frames import NODES, reference_max_displacement
        payload["reference_temperature_k"] = float(ref_t)
        for mode in payload.get("modes") or []:
            mode["reference_max_displacement_ang"] = {
                str(n): reference_max_displacement(mode, float(ref_t), n)
                for n in NODES}
    # THE LABELS A SET MAY BE NAMED BY, where the result states its charges
    # (`web/spectra.md` § 3.2) -- the structure's, read through its own
    # door, so the page picks no structure apart; and, for a SIESTA result
    # that states none, why, in its own facts.
    from molbuilder.spectra.infrared import (SOURCE_CHARGES, SOURCE_DIPOLE,
                                             ORIENTATION_HELD, InfraredError,
                                             atoms_of, dipole_view,
                                             infrared_view, no_charges_reason,
                                             orientation_of, sets_of,
                                             source_of)
    from molbuilder.spectra.methods import with_infrared_view
    # WHERE ITS INFRARED COMES FROM, and held or free -- whether the light's
    # direction is asked at all (`web/spectra.md` § 3.2).
    source = source_of(results)
    if source is not None:
        payload["infrared_source"] = source
        payload["infrared_orientation"] = orientation_of(results)
    if source == SOURCE_CHARGES and results.structure_stated():
        payload["infrared_sets"] = sets_of(results)
    elif source is None and str(results.engine).lower() == "siesta":
        payload["infrared_absent"] = (f"This result states no charges, so "
                                      f"it carries no infrared: "
                                      f"{no_charges_reason(results)}.")
    # WHAT IS ASKED, answered where it applies: a charges result's named
    # set; a held engine-dipole result's direction; a set named of a result
    # with no charges, the reason.  A free result's column is the file's.
    view = None
    try:
        if infrared is None:
            pass
        elif source == SOURCE_CHARGES and infrared["atoms"] is not None:
            atoms, name = atoms_of(results, infrared["atoms"])
            view = infrared_view(results, atoms=atoms,
                                 direction=infrared["direction"],
                                 partition=infrared["partition"],
                                 atoms_label=name)
        elif source == SOURCE_DIPOLE \
                and payload["infrared_orientation"] == ORIENTATION_HELD:
            view = dipole_view(results, direction=infrared["direction"])
        elif source is None and infrared["atoms"] is not None:
            payload["infrared_said"] = (f"the result states no charges: "
                                        f"{no_charges_reason(results)}")
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
        from molbuilder.spectra.raman import (RamanError, raman_view_of,
                                              require_optical_route)
        try:
            require_optical_route(results)
            if path is None:
                raise RamanError(
                    "this result was opened without its folder, so the "
                    "optical calls' data file beside it cannot be read -- "
                    "open it from its run")
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
    # the viewer follows; ``None`` for an upload or an inline result, which
    # belong to no run (`web/results.md` § 4.1).
    return jsonify({"ok": True, "results": payload, "run": run})


@bp.route("/api/spectra/load", methods=["POST"])
def api_spectra_load():
    """Parse a spectra.json (file upload / on-disk path / inline JSON)
    into the typed :class:`SpectraResults`, returned as a JSON-safe
    dict the UI consumes directly.

    Three input modes (mutually exclusive; pick whichever fits):

      1. ``file=``                multipart upload of the JSON bytes.
      2. ``{"path": "..."}``      read from a local path.  Useful for
                                  the post-run "I have results on
                                  disk" case where the user just
                                  enters the job dir.
      3. ``{"json": {...}}``      parse an in-memory dict (for
                                  programmatic callers, e.g. the
                                  live-watch poller that already has
                                  the JSON in hand).

    Returns on success::

        {"ok": True, "results": <SpectraResults.to_dict()>}

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
    # THE TEMPERATURE THE TAB ASKS ITS THERMAL SPREADS AT, whichever way the
    # result arrives (`_temperature_k`).
    try:
        temperature_k = _temperature_k(request.get_json(silent=True),
                                       request.form)
        reference_temperature_k = _temperature_k(
            request.get_json(silent=True), request.form,
            "reference_temperature_k")
        from molbuilder.spectra import infrared as _ir, raman as _raman
        infrared = _view_asked(request.get_json(silent=True), request.form,
                               "infrared", _ir.read_ask)
        raman = _view_asked(request.get_json(silent=True), request.form,
                            "raman", _raman.read_ask)
    except ValueError as exc:
        return jsonify({"ok": False, "kind": "field", "error": str(exc)}), 400
    # ---- multipart upload ---------------------------------------- #
    if request.files.get("file"):
        upload = request.files["file"]
        try:
            raw = upload.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            return _err_load(
                SpectraJsonMalformedError(
                    f"uploaded file is not UTF-8: {exc.reason} "
                    f"at byte {exc.start}"
                )
            )
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as exc:
            return _err_load(SpectraJsonMalformedError(
                f"uploaded file is not valid JSON: {exc}"
            ))
        try:
            results = parse_spectra_json_dict(d)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return _loaded(results, temperature_k=temperature_k,
                       infrared=infrared, raman=raman,
                       reference_temperature_k=reference_temperature_k)

    # ---- JSON body (path or inline JSON) ------------------------- #
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    inline = body.get("json")
    if path:
        # THE FENCE (web-api.md § 2.1).  The parser -- correctly, being a
        # generic reader -- does a bare open(), so without it any path the
        # server process could read would have its EXISTENCE readable from
        # here, which is filesystem enumeration.  The rate limiter's
        # attack-string screen reads the URL, and this path arrives in a
        # JSON body, so nothing else stands here.
        from .files import _PickerError, _resolve_within_roots
        try:
            path = _resolve_within_roots(path)
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status
        # THE RUN'S END IS READ BEFORE THE FILE'S LAST READ (`web/results.md`
        # § 4.1): the run this file belongs to is asked after the file is
        # read, and when it is no longer live the file is read again if it
        # changed in between -- so the viewer stops on the file's last state.
        from molbuilder.runs import run_answer
        try:
            before = _version(path)
            results = parse_spectra_json(path)
            run = run_answer(path)
            if not (run and run["live"]) and _version(path) != before:
                results = parse_spectra_json(path)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return _loaded(results, run=run, temperature_k=temperature_k,
                       infrared=infrared, raman=raman, path=path,
                       reference_temperature_k=reference_temperature_k)
    if inline is not None:
        if not isinstance(inline, dict):
            return _err_load(SpectraJsonMalformedError(
                f"'json' field must be an object, got {type(inline).__name__}"
            ))
        try:
            results = parse_spectra_json_dict(inline)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return _loaded(results, temperature_k=temperature_k,
                       infrared=infrared, raman=raman,
                       reference_temperature_k=reference_temperature_k)

    return jsonify({
        "ok":    False,
        "error": (
            "no input -- send a multipart 'file' field, "
            "or JSON {'path': '<...>'} or {'json': {...}}"
        ),
    }), 400


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _version(path) -> Optional[Tuple[int, int]]:
    """The file's size and modification time -- ``None`` when it is gone,
    which the read that follows reports in its own words."""
    import os
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


@bp.route("/api/spectra/frameset", methods=["POST"])
def api_spectra_frameset():
    """A MODE'S FRAME SET, written through the one generator
    (`engines/vibration.md` § 5.10 ③; `web/spectra.md` § 4.3):
    ``{run, mode, max_displacement_ang, nodes, out}`` -- the vibration
    run's folder and the pair's ``.xyz``, each inside the tree (the fence,
    `web-api.md` § 2.1).  Answers ``{ok, path, frames, said}``, or the
    generator's refusal in its words (400)."""
    from molbuilder.mode_frames import FrameSetError, frameset_from_mode
    from molbuilder.projects import projects_root
    from .files import _PickerError, _resolve_within_roots
    body = request.get_json(silent=True) or {}
    try:
        run = _resolve_within_roots(str(body.get("run") or ""))
        out = _resolve_within_roots(str(body.get("out") or ""))
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    try:
        mode = int(body.get("mode"))
        nodes = int(body.get("nodes") or 3)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "mode and nodes are whole "
                                              "numbers"}), 400
    root = projects_root()
    try:
        got = frameset_from_mode(
            run, mode=mode,
            max_displacement_ang=body.get("max_displacement_ang"),
            nodes=nodes, out=out, root=root)
    except FrameSetError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    try:
        wire = got.to_wire(root)
    except ValueError:
        wire = got.to_wire()
    return jsonify({"ok": True, **wire})


@bp.route("/api/spectra/export", methods=["GET"])
def api_spectra_export():
    """A VIBRATION RESULT'S DATA FILES AND ITS RAMAN VIEW, for the person to
    keep (`engines/vibration.md` § 5.11 § 10) -- ``?path=`` the result the
    viewer shows, its folder, or the calculation's folder; ``format=``
    ``netcdf`` or ``csv``; ``laser_nm=`` and ``broaden_ev=`` for the view --
    one zip, composed on request (`spectra.export`), nothing written into
    the calculation; a view not in it said in ``X-Molbuilder-Said``.  400
    names what is refused."""
    from molbuilder.calculation import NotACalculation, at
    from molbuilder.ncdata import ExportError
    from molbuilder.spectra.export import export
    from molbuilder.spectra.raman import ASK_KEYS
    from ._shared import finite_float, sent_export
    from .files import _PickerError, _resolve_within_roots
    raw_path = str(request.args.get("path") or "")
    try:
        path = _resolve_within_roots(raw_path)
        calc = at(path)
        asked = {key: finite_float(key, request.args.get(key))
                 for key in ASK_KEYS if request.args.get(key) not in (None, "")}
        got = export(calc, str(request.args.get("format") or ""),
                     folder=path if path.is_dir() else path.parent, **asked)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    except FileNotFoundError:
        return jsonify({"ok": False, "error": f"nothing at {raw_path}"}), 404
    except (NotACalculation, ExportError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return sent_export(got)


def _err_load(exc: SpectraJsonError):
    """Map a parser exception class to the HTTP response shape.

    Returns (Flask response, status_code) -- the dispatcher decides
    HTTP code based on exception type so the UI can decide how to
    react without parsing the message string.  Schema mismatches
    carry their numeric versions so the UI can render an "update
    molbuilder" hint directly.
    """
    if isinstance(exc, SpectraJsonNotFoundError):
        return jsonify({
            "ok":    False,
            "error": str(exc),
            "kind":  "not_found",
        }), 404
    if isinstance(exc, SpectraJsonSchemaError):
        return jsonify({
            "ok":               False,
            "error":            str(exc),
            "kind":             "schema_mismatch",
            "expected_version": exc.expected,
            "actual_version":   exc.actual,
        }), 422
    if isinstance(exc, SpectraJsonMalformedError):
        return jsonify({
            "ok":    False,
            "error": str(exc),
            "kind":  "malformed",
        }), 400
    if isinstance(exc, SpectraJsonFieldError):
        return jsonify({
            "ok":    False,
            "error": str(exc),
            "kind":  "field",
        }), 400
    # Generic SpectraJsonError catch-all.
    return jsonify({
        "ok":    False,
        "error": str(exc),
        "kind":  "parse_error",
    }), 400


__all__ = ["bp"]
