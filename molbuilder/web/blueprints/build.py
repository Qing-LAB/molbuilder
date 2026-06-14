"""Build blueprint -- structure-construction + emitter routes.

Routes (registered with no url_prefix; each carries its own full path):

    POST /api/build/molecule        build a Structure from sequence/SMILES/name
    POST /api/build/load            load an existing XYZ / PDB into a Structure
    POST /api/build/fdf             render a SIESTA .fdf for a Structure + params
    POST /api/build/pyscf           render a PySCF script for a Structure + params
    POST /api/build/preflight       fast validate-only path (no rendering)
    GET  /api/build/schema/<engine> form-rendering schema for the SIESTA /
                                    PySCF Build panel (engine ∈ {siesta, pyscf}).
                                    The Build tab's JS renders the form
                                    directly from this schema so the
                                    dataclass is the SINGLE source of truth
                                    for the field set + per-field UI hints.

The four endpoints share a single Flask app instance with the watch
blueprint at ``molbuilder/web/blueprints/watch.py``.  Two top-level
routes stay on the app itself rather than on this blueprint:

    GET  /                     the page (tabbed UI shell)
    GET  /api/health           liveness
    GET  /api/backends         available builder backends (consumed by
                               both Build and Watch tabs' pickers)

JSON shape:

  /api/build/molecule -- body: {"kind": "peptide|dna|rna|smiles|name",
                                "input": "<sequence-or-smiles-or-name>",
                                ...optional kind-specific knobs}
                         returns: {"ok": True, "xyz": "...", "pdb": "...",
                                   "n_atoms": N, "summary": "...",
                                   "title": "...", "elements": [...]}

      DNA / RNA tri-state add_hydrogens semantics:
        "auto"  (default) -- backend-aware: 3DNA gets H, AmberTools/3DNA-fiber
                             keeps the backend's existing H placement.
        "on"              -- always invoke chemistry.add_hydrogens.
        "off"              -- skip H addition entirely.
        true              -- back-compat alias for "auto" (NOT "on").
        false             -- back-compat alias for "off".

  /api/build/load     -- body: multipart with "file" field
                         OR JSON {"text": "...", "format": "auto"|"xyz"|"pdb",
                                  "filename": "<optional>"}
                         returns: same shape as /api/build/molecule
                                  plus "source_format": "xyz"|"pdb"

  /api/build/fdf      -- body: {"xyz": "<xyz>", "params": {<SiestaConfig dict>}}
                         returns: {"ok": True, "fdf": "<text>",
                                   "system_label": "..."}

  /api/build/pyscf    -- body: {"xyz": "<xyz>", "params": {<PySCFConfig dict>}}
                         returns: {"ok": True, "script": "<text>",
                                   "job_name": "..."}
"""

from __future__ import annotations

import typing
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ._shared import (
    config_from_params as _config_from_params,
    dataclass_to_form_schema as _dataclass_to_form_schema,
    issues_to_json as _issues_to_json,
    ok_structure_response,
)

from molbuilder import (
    build_dna, build_from_name, build_from_smiles,
    build_peptide, build_rna,
)
from molbuilder.config.pyscf  import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf  import render_script
from molbuilder.siesta import render_fdf
from molbuilder.structure import Structure
from molbuilder.validation import validate


bp = Blueprint("build", __name__)


# Map kind -> builder.  Keeps the dispatch tight; per-kind URL paths
# would be one route each (an internal refactor option for later --
# the dispatch table here makes that mechanical when wanted).
_BUILDERS = {
    "peptide": build_peptide,
    "dna":     build_dna,
    "rna":     build_rna,
    "smiles":  build_from_smiles,
    "name":    build_from_name,
}


def _resolve_path_within_roots(raw_path: str, *, must_exist: bool = True,
                                require: str = "file"):
    """Wrapper around files._resolve_within_roots so the two new
    endpoints below share the same picker-root validation as
    /api/selection/* and /api/files/*.  Without this the new
    endpoints would accept ANY path -- including ``/etc/passwd`` --
    a path-traversal / arbitrary-read security bug (caught in the
    2026-05-23 code-review pass).

    ``require``: "file" or "dir".  Returns the resolved Path on
    success.  Raises ``_PickerError`` on any rejection (caller wraps
    into a 400 JSON error).
    """
    from .files import _resolve_within_roots, _PickerError
    resolved = _resolve_within_roots(raw_path)
    if must_exist:
        if require == "file" and not resolved.is_file():
            raise _PickerError(400, f"path is not a file: {resolved}")
        if require == "dir" and not resolved.is_dir():
            raise _PickerError(400, f"path is not a directory: {resolved}")
    return resolved


def _sniff_structure_format(text: str) -> str:
    """Return ``"xyz"`` or ``"pdb"`` for raw structure text.

    The earlier sniff scanned only ``text[:120]`` for ``"ATOM "``,
    which missed real PDB files: their HEADER / TITLE / REMARK lines
    push the first ATOM record well past byte 120, so the file was
    misclassified as XYZ and ``Structure.from_xyz`` raised on the
    header lines.  Fix: rely on the format's own first-line rule
    instead of a byte-window scan.

    Rule: XYZ's first non-blank line is an atom count (positive int).
    Anything else (PDB headers, plain text, empty) is treated as PDB.
    Caller still wraps ``Structure.from_pdb`` in try/except, so a
    misclassified blob fails with a clear "could not parse" 400.
    """
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return "xyz" if int(line) > 0 else "pdb"
        except ValueError:
            return "pdb"
    return "pdb"


@bp.route("/api/run/install-wrapper", methods=["POST"])
def api_run_install_wrapper():
    """Drop ``<basename>.run.sh`` next to an emitted script so the
    user can ``bash <basename>.run.sh`` instead of remembering the
    threading + env exports every time.

    The wrapper handles three things that are otherwise a
    per-run-in-shell ritual:
      * BLAS / OpenMP pinning (OPENBLAS_NUM_THREADS=1,
        MKL_NUM_THREADS=1, OMP_NUM_THREADS auto-resolved)
      * ulimit -v for the memory cap (when ``max_memory_mb`` set)
      * Conda env activation via the three-path hybrid (idempotent
        if already active / source+activate if conda on PATH / clear
        error otherwise) for the right env per backend
        (env_for_category("siesta") | env_for_category("pyscf"))

    Body (JSON)::

      {
        "script_path":     "/abs/path/to/<basename>.{fdf,py}",
        "mpi_np":          4,           # optional, SIESTA-only
        "omp_threads":     null,        # null=auto: physical_cores // mpi_np
        "max_memory_mb":   4000,        # optional, emits ulimit -v
        "env":             null         # null=auto from extension
      }

    Returns::

      {
        "ok":             True,
        "wrapper_path":   "/abs/path/to/<basename>.run.sh",
        "wrapper_name":   "<basename>.run.sh",
        "overwritten":    False              # True when a prior .run.sh
                                             # was clobbered (so UI can
                                             # surface as amber notice)
      }

    Path validation: script_path must be under the configured picker
    roots (same gate as /api/files/write).  Wrapper is written next
    to the script with executable bits (0o755).
    """
    from .files import _PickerError
    body = request.get_json(silent=True) or {}
    script_path_raw = (body.get("script_path") or "").strip()
    if not script_path_raw:
        return jsonify({"ok": False,
                        "error": "script_path is required"}), 400
    try:
        script_path = _resolve_path_within_roots(
            script_path_raw, require="file",
        )
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    mpi_np        = body.get("mpi_np")
    omp_threads   = body.get("omp_threads")
    max_memory_mb = body.get("max_memory_mb")
    env_override  = body.get("env")
    # Coerce to None on falsy / zero so the helper's defaults kick in.
    mpi_np        = int(mpi_np) if mpi_np else None
    omp_threads   = int(omp_threads) if omp_threads else None
    max_memory_mb = int(max_memory_mb) if max_memory_mb else None

    # Track whether we OVERWROTE an existing wrapper so the UI can
    # surface that (the user may have hand-edited the .run.sh with
    # extra exports / a custom srun command etc.; silent clobber
    # would lose the work).  write_run_wrapper itself does
    # ``write_text`` which is silent overwrite -- check before
    # calling so we can report the prior state.
    wrapper_dest = script_path.parent / (script_path.stem + ".run.sh")
    pre_existed  = wrapper_dest.exists()

    from molbuilder.runwrap import write_run_wrapper, WrapperError
    try:
        wrapper = write_run_wrapper(
            script_path,
            env=env_override,
            mpi_np=mpi_np,
            omp_threads=omp_threads,
            max_memory_mb=max_memory_mb,
        )
    except WrapperError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:    # noqa: BLE001 -- surface any failure
        return jsonify({"ok": False,
                        "error": f"wrapper write failed: {exc}"}), 500

    return jsonify({
        "ok":           True,
        "wrapper_path": str(wrapper),
        "wrapper_name": wrapper.name,
        "overwritten":  pre_existed,
    })


@bp.route("/api/siesta/install-pseudos", methods=["POST"])
def api_siesta_install_pseudos():
    """Copy .psml files from cfg.psml_lib into the target directory
    so SIESTA finds them at run time.

    SIESTA discovers pseudopotentials by looking for ``<element>.psml``
    in the CURRENT WORKING DIRECTORY where the .fdf is being read from
    -- there's no "pseudopotential search path" directive in SIESTA's
    .fdf grammar.  So after writing the .fdf via /api/files/write, the
    JS calls this endpoint to copy the matching .psml files into the
    same directory.  Without this extra hop SIESTA fails at startup
    with ``pseudo_read: ERROR: Pseudopotential file not found``.

    Only ``.psml`` is installed (case-insensitive).  SIESTA also reads
    the legacy ``.psf`` / ``.vps`` formats, but our parser + validation
    pipeline (``molbuilder.pseudos``) is PSML-only, so we don't install
    pseudos we can't pre-validate.  Users on legacy formats stage
    those files manually.

    Body (JSON)::

      {
        "psml_lib":     "/abs/path/to/dir/of/psml/files",
        "dest_dir":     "/abs/path/to/where/the/fdf/is",
        "structure_path": "/abs/path/to/<.xyz|.pdb>",   # OR
        "structure_text": "<XYZ or PDB text>"
      }

    Returns::

      {
        "ok":           True,
        "copied":       ["C.psml", "Fe.psml", ...],     # new file written
        "overwritten":  ["S.psml", ...],                # replaced an
                                                        # existing file
        "skipped":      [{"file": "H.psml",
                          "reason": "already present (same file)"}, ...],
        "missing":      ["S"],                          # no .psml in lib
        "dest_dir":     "<resolved abs path>"
      }

    ``skipped`` only fires when ``dest_dir/<el>.psml`` already resolves
    to the same path as ``psml_lib/<el>.psml`` (i.e. the destination is
    a symlink back into the lib).  After a real shutil.copyfile the
    destination is a distinct file, so subsequent installs land in
    ``overwritten`` (silent clobber, surfaced to the UI).

    ``missing`` is the list of elements for which no .psml was found
    in ``psml_lib``.  Hard fail at SIESTA run time -- caller should
    surface as an error notice in the UI.
    """
    from .files import _PickerError
    body = request.get_json(silent=True) or {}
    psml_lib_raw = (body.get("psml_lib") or "").strip()
    dest_dir_raw = (body.get("dest_dir") or "").strip()
    if not psml_lib_raw or not dest_dir_raw:
        return jsonify({"ok": False,
                        "error": "both psml_lib and dest_dir are required"}), 400
    # Resolve dest_dir first so we can use it as the relative-path
    # anchor for psml_lib (the form persists dest-relative paths
    # after a successful Save -- see viewer.js#save-fdf handler).
    try:
        dest_dir = _resolve_path_within_roots(dest_dir_raw, require="dir")
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    # Two-stage resolution (see pseudos.resolve_psml_lib): first try
    # dest-relative (the portable form the Save handler persists),
    # then projects/-relative (the documented convention).  Lets a
    # single form value work across both Generate-only previews AND
    # a previously-saved project that holds the dest-relative form.
    from molbuilder.pseudos import resolve_psml_lib
    psml_lib_resolved = str(resolve_psml_lib(psml_lib_raw,
                                              dest_dir=dest_dir))
    try:
        psml_dir = _resolve_path_within_roots(psml_lib_resolved, require="dir")
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    # Resolve structure (elements decide which pseudos to copy).
    text_in = body.get("structure_text")
    path_in = body.get("structure_path")
    if path_in:
        try:
            p = _resolve_path_within_roots(path_in, require="file")
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status
        text_in = p.read_text()
        ext = p.suffix.lower()
    else:
        ext = "." + _sniff_structure_format(text_in or "")
    if not text_in:
        return jsonify({"ok": False,
                        "error": "structure_path or structure_text is required"}), 400

    from molbuilder.structure import Structure
    try:
        if ext == ".pdb":
            struct = Structure.from_pdb(text_in)
        else:
            struct = Structure.from_xyz(text_in)
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse structure: {exc}"}), 400

    # Walk the unique element set + copy each .psml from psml_lib
    # to dest_dir.  Use shutil.copyfile (no metadata bits, predictable
    # destination perms inherited from dest_dir).  Skip files only
    # when dst already resolves to the same path as src (the symlink
    # case); real prior copies are distinct inodes and land in the
    # overwritten[] bucket.
    import shutil
    seen: set = set()
    copied: list = []        # new files (no prior version in dest_dir)
    overwritten: list = []   # files that REPLACED an existing one
    skipped: list = []       # dst is a symlink back to src -- no-op
    missing: list = []
    for raw_el in struct.elements:
        el = raw_el.capitalize()
        if el in seen:
            continue
        seen.add(el)
        src = psml_dir / f"{el}.psml"
        if not src.is_file():
            # Fallback: case-insensitive match on the WHOLE basename so
            # we still find `fe.psml` / `FE.PSML` etc.  Older SIESTA
            # tutorials + community-built psml libs are inconsistent
            # about element-name case; SIESTA itself reads whatever the
            # %block ChemicalSpeciesLabel asks for, so we always write
            # ``<Element>.psml`` capitalized -- only the search of the
            # user's lib is loosened.
            el_lc = el.lower()
            cand = [p for p in psml_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() == ".psml"
                    and p.stem.lower() == el_lc]
            if cand:
                src = cand[0]
            else:
                missing.append(el)
                continue
        dst = dest_dir / f"{el}.psml"
        try:
            existed_before = dst.exists()
            if existed_before and src.resolve() == dst.resolve():
                skipped.append({"file": dst.name,
                                "reason": "already present (same file)"})
                continue
            shutil.copyfile(src, dst)
            if existed_before:
                # Surface CLOBBER explicitly -- the user may have
                # hand-edited a .psml (or pulled a different
                # functional family by accident); silent overwrite
                # would lose that.  The UI can highlight the
                # overwritten[] entries amber.
                overwritten.append(dst.name)
            else:
                copied.append(dst.name)
        except OSError as exc:
            return jsonify({"ok": False,
                            "error": f"copy failed for {el}.psml: {exc}"}), 500

    return jsonify({
        "ok":          True,
        "copied":      copied,
        "overwritten": overwritten,
        "skipped":     skipped,
        "missing":     missing,
        "dest_dir":    str(dest_dir),
    })


@bp.route("/api/siesta/check-pseudos", methods=["POST"])
def api_siesta_check_pseudos():
    """Validate a SIESTA pseudopotential directory against the elements
    a structure needs.

    Body (JSON)::

      {
        "psml_lib":        "/path/to/dir/of/psml/files",
        "structure_path":  "/abs/path/to/<.xyz|.pdb>",   # OR
        "structure_text":  "<XYZ or PDB text>",
        "xc_authors":      "PBE"                          # optional
      }

    Returns::

      {
        "ok":         True,
        "entries":  [{"element": "Fe", "status": "ok"|"missing"|"xc_mismatch"
                                                  |"relativistic_mismatch"
                                                  |"parse_warning",
                       "message": "...",
                       "path":    "/abs/path/to/Fe.psml" | null}, ...],
        "n_ok":     <int>,
        "n_missing":<int>,
        "n_mismatch": <int>
      }

    Surfaces per-element status so the UI can show a one-row badge
    per element next to the SIESTA basis section.  All status codes
    are advisory except ``missing`` which is a hard fail (SIESTA
    won't start without a pseudo for every element).
    """
    from .files import _PickerError
    body = request.get_json(silent=True) or {}
    psml_lib_raw = (body.get("psml_lib") or "").strip()
    if not psml_lib_raw:
        return jsonify({"ok": False, "error": "psml_lib path is required"}), 400
    try:
        psml_dir = _resolve_path_within_roots(psml_lib_raw, require="dir")
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    # Resolve structure: same picker-root gate as /api/selection/atoms.
    text_in = body.get("structure_text")
    path_in = body.get("structure_path")
    if path_in:
        try:
            p = _resolve_path_within_roots(path_in, require="file")
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status
        text_in = p.read_text()
        ext = p.suffix.lower()
    else:
        ext = "." + _sniff_structure_format(text_in or "")
    if not text_in:
        return jsonify({"ok": False,
                        "error": "structure_path or structure_text is required"}), 400

    from molbuilder.structure import Structure
    try:
        if ext == ".pdb":
            struct = Structure.from_pdb(text_in)
        else:
            struct = Structure.from_xyz(text_in)
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse structure: {exc}"}), 400

    from molbuilder.pseudos import check_coverage
    expected_xc_authors = (body.get("xc_authors") or "").strip() or None
    # Derive XC family from authors when given (PBE / PBEsol / BLYP / revPBE -> GGA;
    # CA / PZ / PW -> LDA; DRSLL / LMKLL -> VDW).
    GGA = {"pbe", "pbesol", "blyp", "revpbe", "rpbe"}
    LDA = {"ca", "pz", "pw"}
    VDW = {"drsll", "lmkll"}
    expected_xc_family = None
    if expected_xc_authors:
        a = expected_xc_authors.lower()
        if a in GGA:
            expected_xc_family = "GGA"
        elif a in LDA:
            expected_xc_family = "LDA"
        elif a in VDW:
            expected_xc_family = "VDW"
    entries = check_coverage(
        struct.elements, psml_dir,
        expected_xc_family=expected_xc_family,
        expected_xc_authors=expected_xc_authors,
    )
    by_status = {}
    for e in entries:
        by_status[e.status] = by_status.get(e.status, 0) + 1
    return jsonify({
        "ok":         True,
        "entries":    [{"element": e.element, "status": e.status,
                         "message": e.message,
                         "path":   str(e.path) if e.path else None}
                        for e in entries],
        "n_ok":       by_status.get("ok", 0),
        "n_missing":  by_status.get("missing", 0),
        "n_mismatch": (by_status.get("xc_mismatch", 0)
                       + by_status.get("relativistic_mismatch", 0)),
    })


@bp.route("/api/structure/analyze", methods=["POST"])
def api_structure_analyze():
    """Engine-agnostic chemistry analysis of a structure.

    Body (JSON)::

      {
        "structure_path":  "/abs/path/to/<.xyz|.pdb>",   # OR
        "structure_text":  "<XYZ or PDB text>"
      }

    Returns the ``ChemistryAnalysis`` dataclass serialised plus a
    ``suggested.<engine>`` block built by iterating every registered
    parameter adapter.  See ``docs/protocols/scientific-validation.md``
    § 5.1 for the response-shape contract and § 4 for the adapter
    Protocol.

    The endpoint is deliberately thin (~20 LoC of logic) — the
    chemistry analyzer (``molbuilder.chemistry.analyze_structure``)
    holds every chemistry rule; the per-engine adapters
    (``molbuilder.<engine>.auto_defaults``) hold every engine
    translation.  Adding a new engine = drop an adapter file +
    import it in ``web/blueprints/__init__.py``; this endpoint
    needs no change.

    The same ``ChemistryAnalysis`` instance backs the pre-emission
    validation pass (``validation.check_open_shell_metal``) —
    auto-detect and validate cannot disagree by construction.
    """
    from dataclasses import asdict
    from .files import _PickerError
    from molbuilder.chemistry import analyze_structure, registered_adapters
    from molbuilder.structure import Structure

    body = request.get_json(silent=True) or {}
    text_in = body.get("structure_text")
    path_in = body.get("structure_path")
    if path_in:
        try:
            p = _resolve_path_within_roots(path_in, require="file")
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status
        text_in = p.read_text()
        ext = p.suffix.lower()
    else:
        ext = "." + _sniff_structure_format(text_in or "")
    if not text_in:
        return jsonify({"ok": False,
                        "error": "structure_path or structure_text is required"}), 400

    try:
        if ext == ".pdb":
            struct = Structure.from_pdb(text_in)
        else:
            struct = Structure.from_xyz(text_in)
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse structure: {exc}"}), 400

    # analyze_structure raises KeyError on an unknown element symbol
    # (typos, bad PDB column fallback) via total_electrons.  Catch -> 400
    # with the parser's clear message; without this it would surface
    # as a 500 Internal Server Error.
    try:
        analysis = analyze_structure(struct)
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc).strip("'")}), 400

    return jsonify({
        "ok":                  True,
        "n_atoms":             analysis.n_atoms,
        "elements":            analysis.elements,
        "n_electrons_neutral": analysis.n_electrons_neutral,
        "metals":              analysis.metals,
        "metal_hints":         [asdict(h) for h in analysis.metal_hints],
        "suggested":           {
            name: asdict(cls.to_params(analysis))
            for name, cls in registered_adapters().items()
        },
        "warnings":            list(analysis.warnings),
    })


@bp.route("/api/build/molecule", methods=["POST"])
def api_build_molecule():
    body = request.get_json(silent=True) or {}
    kind = (body.get("kind") or "").strip().lower()
    text = (body.get("input") or "").strip()
    if kind not in _BUILDERS:
        return jsonify({"ok": False,
                        "error": f"Unknown kind {kind!r}; "
                                 f"valid: {sorted(_BUILDERS)}"}), 400
    if not text:
        return jsonify({"ok": False, "error": "empty input"}), 400
    backend_used: str | None = None
    h_mode_used: str | None = None
    try:
        # DNA / RNA accept extra knobs (backend / form / terminal).
        if kind in ("dna", "rna"):
            requested = body.get("backend", "auto")
            # add_hydrogens is tri-state: auto / on / off.  The web
            # form sends a string ("auto" by default).  We accept
            # bool too for back-compat with older client code.
            h_mode_raw = body.get("add_hydrogens", "auto")
            if isinstance(h_mode_raw, bool):
                h_mode_used = "auto" if h_mode_raw else "off"
            else:
                h_mode_used = str(h_mode_raw).lower()
                if h_mode_used not in ("auto", "on", "off"):
                    return jsonify({
                        "ok": False,
                        "error": (
                            f"add_hydrogens must be 'auto'/'on'/'off' "
                            f"(or bool); got {h_mode_raw!r}"
                        ),
                    }), 400
            kwargs = {
                "backend":  requested,
                "form":     body.get("form",     "B" if kind == "dna" else "A"),
                "terminal": body.get("terminal", "OH"),
                "add_hydrogens": h_mode_used,
                "protonate_phosphates":
                    bool(body.get("protonate_phosphates", True)),
            }
            # Resolve "auto" before the build so the UI can display
            # which backend actually ran -- this matches dispatch()'s
            # selection logic exactly (see auto_backend_name docstring).
            if requested == "auto":
                # Use the canonical path (the back-compat shim at
                # molbuilder.backends is for external callers; in-tree
                # code goes direct to builders.backends).
                from molbuilder.builders.backends import auto_backend_name
                backend_used = auto_backend_name()
            else:
                backend_used = requested
            struct = _BUILDERS[kind](text, **kwargs)
        else:
            struct = _BUILDERS[kind](text)
    except ImportError as exc:
        return jsonify({"ok": False,
                        "error": f"missing dependency: {exc}"}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Workspace-state Phase 2 migration (2026-06-07): route through
    # the canonical ``ok_structure_response`` helper.  Endpoint-
    # specific keys (pdb, summary, backend_used, add_hydrogens_mode)
    # land BOTH at the top level (back-compat with every existing
    # JS consumer that reads them off the response root) AND in the
    # canonical ``extra`` sub-dict (Phase 4+ workspace-dispatcher
    # consumers read them from there).  Issues + canonical atoms
    # come from the helper — one validate_geometry pass.
    return ok_structure_response(struct, extra={
        # build/molecule's legacy contract: title defaults to the
        # build kind ("smiles" / "dna" / …) when the Structure
        # itself carries no title (most builders don't set one).
        # Override via extra rather than mutating struct.title so
        # downstream code that reuses the Structure sees the
        # canonical (empty) title.
        "title":             struct.title or kind,
        "pdb":               struct.to_pdb(),
        "summary":           struct.summary(),
        "backend_used":      backend_used,
        # Tri-state H-add decision actually used (echoes the
        # request, or "auto" when not explicitly requested).  None
        # for non-nucleic builds (peptide/SMILES/name) where the
        # kwarg doesn't apply.
        "add_hydrogens_mode": h_mode_used,
    })


@bp.route("/api/build/load", methods=["POST"])
def api_build_load():
    """Accept either:
      * multipart/form-data with a single file field "file", or
      * JSON {"text": "...", "format": "xyz"|"pdb"|"auto",
              "filename": "<optional>"}
    Returns the same JSON shape as /api/build/molecule so the front
    end can treat the result identically.
    """
    text: str = ""
    fmt: str = "auto"
    filename: str = ""
    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename or ""
        text = f.read().decode("utf-8", errors="replace")
    else:
        body = request.get_json(silent=True) or {}
        text = body.get("text") or ""
        fmt = (body.get("format") or "auto").lower()
        filename = body.get("filename") or ""

    if not text.strip():
        return jsonify({"ok": False, "error": "empty input"}), 400

    if fmt == "auto":
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("xyz", "pdb"):
            fmt = ext
        else:
            # Sniff: PDB lines start with ATOM/HETATM/HEADER/TITLE,
            # XYZ first line is an atom count.
            first = text.lstrip().splitlines()[0] if text.strip() else ""
            fmt = "xyz" if first.strip().isdigit() else "pdb"

    try:
        if fmt == "xyz":
            struct = Structure.from_xyz(text, title=filename or None)
        elif fmt == "pdb":
            struct = Structure.from_pdb(text, title=filename or None)
        else:
            return jsonify({"ok": False,
                            "error": f"unknown format {fmt!r}; "
                                     "expected xyz or pdb"}), 400
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse {fmt}: {exc}"}), 400

    # Workspace-state Phase 2 migration (2026-06-07): route through
    # the canonical ``ok_structure_response`` helper.  Per-atom
    # payload, legacy aliases, validate-pass issues, and the
    # forward-compat ``extra`` sub-dict all come from the helper
    # in a single call.  Endpoint extras (pdb, summary, the
    # actual parsed format, title fallback) override the
    # canonical defaults at both the top level and the canonical
    # ``extra`` sub-dict — same threading rule for every key.
    return ok_structure_response(struct, extra={
        # /api/build/load's legacy contract: title defaults to the
        # filename (or format name) when the input carries none.
        # Override via extra so downstream code that reuses the
        # Structure sees the canonical (possibly empty) title.
        "title":         struct.title or (filename or fmt),
        "pdb":           struct.to_pdb(),
        "summary":       struct.summary(),
        # Override the canonical XYZ default with the actually-
        # parsed format; the helper threads this through to both
        # the top level and the ``extra`` sub-dict.
        "source_format": fmt,
    })


@bp.route("/api/build/fdf", methods=["POST"])
def api_build_fdf():
    body = request.get_json(silent=True) or {}
    xyz_text = body.get("xyz")
    params: Dict[str, Any] = body.get("params") or {}
    if not xyz_text:
        return jsonify({"ok": False, "error": "no xyz provided"}), 400

    # Parse XYZ -> Structure (skip ASE round-trip; we wrote it ourselves)
    try:
        struct = _xyz_to_structure(xyz_text)
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse xyz: {exc}"}), 400

    # Three-stage contract carrier: apply the .molstruct.json sidecar
    # if the JS passed structure_path.  Without this hop, the user's
    # /modify freeze list never reaches render_fdf -- a silent boundary-
    # conditions drop (caught by the 2026-05-25 review).  Mirrors what
    # /api/spectra/render does; the helper is owned by spectra blueprint.
    sidecar_notice = None
    structure_path = (body.get("structure_path") or "").strip()
    if structure_path:
        from ._shared import apply_sidecar_if_possible
        sidecar_notice = apply_sidecar_if_possible(struct, structure_path)

    try:
        cfg = _siesta_config_from_params(params)
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"bad parameters: {exc}"}), 400

    # Optional dest_dir hint: the JS sends the Projects sidebar's
    # current dir so the validator can resolve dest-relative
    # ``cfg.psml_lib`` paths correctly (the portable form the Save
    # handler persists).  Validates against the picker-root allowlist
    # before trusting it -- otherwise a hostile client could pass a
    # path outside projects/ to influence resolution.
    dest_dir = None
    dest_dir_raw = (body.get("dest_dir") or "").strip()
    if dest_dir_raw:
        try:
            dest_dir = _resolve_path_within_roots(dest_dir_raw, require="dir")
        except Exception:
            # Bad / outside-root hint: ignore silently.  The validator
            # falls back to projects/-relative resolution.
            dest_dir = None

    # Validate before render so the web layer gets a structured copy
    # of the issues.  render_fdf will validate again and write warnings
    # to stderr / raise on errors -- we keep that for CLI/library
    # callers; here we want the issues as JSON for the UI.
    issues = validate(struct, cfg, dest_dir=dest_dir)
    if sidecar_notice:
        issues.append(Issue("warn", sidecar_notice, "config.frozen_atoms"))
    # Three-stage Pattern B: the SIESTA SCF/relaxation deck does
    # NOT consume electrode regions (L-electrode / R-electrode /
    # bridge) — those drive the Transport tab.  See
    # _shared.regions_pattern_b_notice for the canonical issue.
    from ._shared import regions_pattern_b_notice
    regions_notice = regions_pattern_b_notice(struct, "the .fdf")
    if regions_notice is not None:
        issues.append(regions_notice)
    try:
        fdf = render_fdf(struct, cfg)
    except ValidationError as exc:
        # Preserve the sidecar notice (and any other pre-validate
        # issues we appended above) in the error response.  Previously
        # we returned only ``exc.issues`` -- the user lost the
        # "sidecar at <file> could not be applied" warn on render
        # failure, which is exactly when they need it (the missing
        # sidecar may be the reason validation rejected the run).
        # Caught by the 2026-05-26 review.
        merged_issues = _issues_to_json(exc.issues, cfg=cfg)
        # Add any pre-render issues NOT already in exc.issues (de-dup
        # by (severity, where, message) tuple).
        exc_keys = {(d["severity"], d.get("where", ""), d["message"])
                    for d in merged_issues}
        for i in _issues_to_json(issues, cfg=cfg):
            if (i["severity"], i.get("where", ""), i["message"]) not in exc_keys:
                merged_issues.append(i)
        return jsonify({
            "ok":     False,
            "error":  str(exc),
            "issues": merged_issues,
        }), 400
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"render failed: {exc}"}), 500

    return jsonify({
        "ok": True,
        "fdf": fdf,
        "system_label": cfg.system_label,
        "issues": _issues_to_json(issues, cfg=cfg),
    })


@bp.route("/api/build/pyscf", methods=["POST"])
def api_build_pyscf():
    body = request.get_json(silent=True) or {}
    xyz_text = body.get("xyz")
    params: Dict[str, Any] = body.get("params") or {}
    if not xyz_text:
        return jsonify({"ok": False, "error": "no xyz provided"}), 400

    try:
        struct = _xyz_to_structure(xyz_text)
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse xyz: {exc}"}), 400

    # Three-stage contract carrier: apply the .molstruct.json sidecar
    # so /modify's frozen_atoms reach render_script.  Mirrors the
    # /api/build/fdf hop added in the same commit; without it the
    # PySCF emitter never sees struct.frozen_atoms even though the
    # emission code is ready to honor it.
    sidecar_notice = None
    structure_path = (body.get("structure_path") or "").strip()
    if structure_path:
        from ._shared import apply_sidecar_if_possible
        sidecar_notice = apply_sidecar_if_possible(struct, structure_path)

    try:
        cfg = _pyscf_config_from_params(params)
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"bad parameters: {exc}"}), 400

    issues = validate(struct, cfg)
    if sidecar_notice:
        issues.append(Issue("warn", sidecar_notice, "config.frozen_atoms"))
    # Three-stage Pattern B (mirrors /api/build/fdf above).
    from ._shared import regions_pattern_b_notice
    regions_notice = regions_pattern_b_notice(struct, "the PySCF script")
    if regions_notice is not None:
        issues.append(regions_notice)
    try:
        script = render_script(struct, cfg)
    except ValidationError as exc:
        # Same merge as /api/build/fdf above: keep the sidecar notice
        # + any pre-render issues in the error response so the user
        # sees the full picture instead of just render_script's
        # internal validation issues.
        merged_issues = _issues_to_json(exc.issues, cfg=cfg)
        exc_keys = {(d["severity"], d.get("where", ""), d["message"])
                    for d in merged_issues}
        for i in _issues_to_json(issues, cfg=cfg):
            if (i["severity"], i.get("where", ""), i["message"]) not in exc_keys:
                merged_issues.append(i)
        return jsonify({
            "ok":     False,
            "error":  str(exc),
            "issues": merged_issues,
        }), 400
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"render failed: {exc}"}), 500

    return jsonify({
        "ok": True,
        "script": script,
        "job_name": cfg.job_name,
        "issues": _issues_to_json(issues, cfg=cfg),
    })


@bp.route("/api/build/schema/<engine>", methods=["GET"])
def api_build_schema(engine: str):
    """Form-rendering schema for the SIESTA or PySCF Build panel.

    Returns the JSON-friendly shape produced by
    ``_shared.dataclass_to_form_schema()`` -- see the helper docstring
    for the exact field/section layout.  The Build tab's JS calls
    this once on page load and renders the form panel directly from
    the returned schema; no static HTML field declarations are
    duplicated.

    ``engine`` is constrained to {"siesta", "pyscf"} so a typo
    surfaces as a clean 404 instead of leaking a default response.
    """
    engine = (engine or "").strip().lower()
    cls_map = {
        "siesta": (SiestaConfig, "p"),
        "pyscf":  (PySCFConfig,  "py"),
    }
    if engine not in cls_map:
        return jsonify({
            "ok": False,
            "error": (
                f"unknown engine {engine!r}; "
                f"expected one of {sorted(cls_map)}"
            ),
        }), 404
    cls, id_prefix = cls_map[engine]
    return jsonify({
        "ok": True,
        "schema": _dataclass_to_form_schema(cls, id_prefix),
    })


@bp.route("/api/build/preflight", methods=["POST"])
def api_build_preflight():
    """Cheap validation-only endpoint for the live UI hint panel.

    Body: ``{"xyz": "<text>", "engine": "siesta"|"pyscf",
             "params": {<config dict>}}``

    Returns ``{"ok": True, "issues": [{"severity", "message",
    "where"}, ...]}``; on bad input returns ``{"ok": False, "error":
    ...}`` with HTTP 400.

    Rationale: the build form has many knobs whose interactions
    matter (k-grid vs vacuum padding, hybrid functional vs grid
    level, charged peptide without explicit charge override, ...).
    Pre-existing UX surfaced these only after the user clicked
    Generate, jamming them into a single status line.  This endpoint
    runs ``validate(struct, cfg)`` without rendering FDF / PySCF
    text -- much cheaper -- so the UI can call it on debounced form
    input and update a structured issues panel live.
    """
    body = request.get_json(silent=True) or {}
    xyz_text = body.get("xyz")
    engine = (body.get("engine") or "").strip().lower()
    params: Dict[str, Any] = body.get("params") or {}

    if not xyz_text:
        return jsonify({"ok": False, "error": "no xyz provided"}), 400
    if engine not in ("siesta", "pyscf"):
        return jsonify({
            "ok": False,
            "error": f"engine must be 'siesta' or 'pyscf'; got {engine!r}",
        }), 400

    try:
        struct = _xyz_to_structure(xyz_text)
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse xyz: {exc}"}), 400

    try:
        if engine == "siesta":
            cfg = _siesta_config_from_params(params)
        else:
            cfg = _pyscf_config_from_params(params)
    except Exception as exc:
        # Bad params surface as a single error-severity issue so the
        # UI can display the same panel for the "your config doesn't
        # parse" case without a separate code path.
        return jsonify({
            "ok": True,
            "issues": [{"severity": "error",
                        "message": f"bad parameters: {exc}",
                        "where":   "config"}],
        })

    return jsonify({
        "ok": True,
        "issues": _issues_to_json(validate(struct, cfg), cfg=cfg),
    })


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _xyz_to_structure(xyz_text: str) -> Structure:
    """Thin wrapper that delegates to Structure.from_xyz so the web
    layer doesn't carry its own parser."""
    return Structure.from_xyz(xyz_text, title="from-browser")


_SIESTA_HINTS = typing.get_type_hints(SiestaConfig)
_PYSCF_HINTS  = typing.get_type_hints(PySCFConfig)


def _siesta_config_from_params(params: Dict[str, Any]) -> SiestaConfig:
    """Build a SiestaConfig from a JSON params dict, with per-field
    type coercion (R5)."""
    return _config_from_params(SiestaConfig, params, _SIESTA_HINTS)


def _pyscf_config_from_params(params: Dict[str, Any]) -> PySCFConfig:
    """Build a PySCFConfig from a JSON params dict, with per-field
    type coercion (R5).  Empty-string sentinels for solvent /
    auxbasis / dispersion are normalised to None so the form's "leave
    default" UI gesture round-trips correctly."""
    return _config_from_params(
        PySCFConfig, params, _PYSCF_HINTS,
        none_sentinels=("solvent", "auxbasis", "dispersion"),
    )
