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

  /api/build/fdf      -- body: {"structure": {<envelope>},
                                "params": {<SiestaConfig dict>}}
                         returns: {"ok": True, "fdf": "<text>",
                                   "system_label": "..."}

  /api/build/pyscf    -- body: {"structure": {<envelope>},
                                "params": {<PySCFConfig dict>}}
                         returns: {"ok": True, "script": "<text>",
                                   "job_name": "..."}

  The three emitting doors (fdf / pyscf / preflight) read the structure
  through ``_shared.struct_from_body`` -- the atoms as NUMBERS with their
  facts beside them, which is what the browser holds and what every other
  structure door already takes.  A legacy ``{"xyz": "<text>"}`` body still
  works; the helper accepts either and the envelope wins when both appear.
"""

from __future__ import annotations

import typing
import json
import pathlib
from datetime import datetime
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ._shared import (
    config_from_params as _config_from_params,
    catalogue_to_form_schema as _catalogue_to_form_schema,
    issues_to_json as _issues_to_json,
    ok_structure_response,
    struct_from_body as _struct_from_body,
)

from molbuilder import (
    build_dna, build_from_name, build_from_smiles,
    build_peptide, build_rna,
)
from molbuilder.config.pyscf  import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.runtime_config import RuntimeConfigError
from molbuilder.structure import Structure
from molbuilder.validation import validate
from .files import _resolve_within_roots, _PickerError


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
        "env":             null,        # null=auto from extension
        "continue_retries": null        # optional, SIESTA-only, 1..5:
                                        # bake the warm-retry budget
                                        # (auto --continue on SCF-abort /
                                        # geometry step-cap; see
                                        # running-a-job.md § 3.5)
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

    mpi_np           = body.get("mpi_np")
    omp_threads      = body.get("omp_threads")
    max_memory_mb    = body.get("max_memory_mb")
    env_override     = body.get("env")
    continue_retries = body.get("continue_retries")
    # Coerce to None on falsy / zero so the helper's defaults kick in.
    # A non-numeric value is the CALLER's error -> 400, not a 500.
    try:
        mpi_np           = int(mpi_np) if mpi_np else None
        omp_threads      = int(omp_threads) if omp_threads else None
        max_memory_mb    = int(max_memory_mb) if max_memory_mb else None
        continue_retries = int(continue_retries) if continue_retries else None
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error":
                        "mpi_np / omp_threads / max_memory_mb / "
                        "continue_retries must be integers"}), 400
    # Mirror the shared schema's bound (SiestaConfig.continue_retries,
    # range 1..5) -- the wrapper bakes this number into a retry loop, so an
    # uncapped request would render an unbounded one.
    if continue_retries is not None and not (1 <= continue_retries <= 5):
        return jsonify({"ok": False, "error":
                        "continue_retries must be between 1 and 5"}), 400

    # Track whether we OVERWROTE an existing wrapper so the UI can
    # surface that (the user may have hand-edited the .run.sh with
    # extra exports / a custom srun command etc.; silent clobber
    # would lose the work).  write_run_wrapper itself does
    # ``write_text`` which is silent overwrite -- check before
    # calling so we can report the prior state.
    wrapper_dest = script_path.parent / (script_path.stem + ".run.sh")
    pre_existed  = wrapper_dest.exists()

    from molbuilder.runwrap import write_run_wrapper, WrapperError
    from molbuilder.jobset.model import Resources
    # The ALLOCATION this tab is asking for, assembled once (architecture.md
    # § 3.1, rule A8).  This call passed four loose values until 2026-08-17 and
    # named no `cpus_per_task`, so the `.sbatch` it emits alongside carried no
    # `-c` at all while the `.run.sh` beside it baked the right OMP default --
    # the mirror image of what `jobset/prep.py` got wrong, and for the same
    # reason: two callers, one door, and each choosing its own subset.
    #
    # `omp_threads` is the tab's word for it and `cpus_per_task` the
    # scheduler's (job-contracts.md § 6.2); the translation happens HERE,
    # at the boundary, which is the rule that section already states.
    resources = Resources(
        mpi_np=mpi_np,
        cpus_per_task=omp_threads,
        max_memory_mb=max_memory_mb,
        continue_retries=continue_retries,
    )
    try:
        wrapper = write_run_wrapper(
            script_path,
            resources=resources,
            env=env_override,
        )
    except WrapperError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except RuntimeConfigError as exc:
        # Operator-config problem (e.g. script_generation.activation is
        # not set -- the v2 generator refuses to emit).  The message is
        # the actionable fix text; 400 per the four-bucket contract, not
        # a 500 that reads like an internal fault.  This exact case hid
        # the qlabsrv missing-.run.sh regression.
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

    # WHICH ELEMENTS ARE IN IT -- the only thing this route wants from the
    # structure.  It copies one `<element>.psml` per element and reads nothing
    # else: not the positions, not the labels, not the cell.
    #
    # IT TOOK `structure_text` / `structure_path` AND ITS CALLER STOPPED SENDING
    # EITHER.  `7447d7d` moved the Save flow onto the envelope --
    # `structure: _structureForRequest()` -- and did not bring this route with
    # it, so every SIESTA save since has answered 400 here: "structure_path or
    # structure_text is required".  The .fdf was written, the pseudos were not
    # installed, and the wrapper was then skipped on purpose ("pseudos
    # incomplete"), leaving a deck that cannot run.  Reported from a real save,
    # 2026-08-04.
    #
    # The envelope is what the one caller sends, so the envelope is what this
    # takes -- through `struct_from_body`, the one deserialiser, like every
    # other door that receives a structure.
    if not isinstance(body.get("structure"), dict):
        return jsonify({
            "ok": False,
            "error": "no 'structure' provided (its elements decide which "
                     "pseudopotentials to copy)",
        }), 400
    from ._shared import struct_from_body
    try:
        struct = struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not read structure: {exc}"}), 400

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
    parameter adapter.  See ``docs/science/validation.md``
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
    build_warnings: list[str] = []
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
            # relax_clashes (DNA explicit-duplex): opt-in force-field relief of a
            # mismatched pair's steric overlap.  build_dna ignores it for ss / RNA.
            if kind == "dna":
                kwargs["relax_clashes"] = bool(body.get("relax_clashes", False))
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
            # Capture builder RuntimeWarnings (e.g. a mismatched-duplex steric
            # CLASH, or the amber "extended polymer" note) so the UI can surface
            # them -- otherwise they'd only reach the server log.
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as _caught:
                _warnings.simplefilter("always")
                struct = _BUILDERS[kind](text, **kwargs)
            build_warnings = [str(w.message) for w in _caught]
        elif kind in ("smiles", "name"):
            # RDKit-first, OpenBabel-fallback (Name lookup resolves to SMILES then
            # builds, so it rides the same chain): surface WHICH engine produced
            # the geometry so the user knows when they're on the lower-fidelity path.
            struct, backend_used = _BUILDERS[kind](text, return_backend=True)
        else:
            struct = _BUILDERS[kind](text)
    except ImportError as exc:
        return jsonify({"ok": False,
                        "error": f"missing dependency: {exc}"}), 500
    except Exception as exc:
        # web-api.md § 1.6 (d): an unhandled exception from the
        # builder dispatch is server fault, not protocol error.
        # The user's input passed shape validation (kind + input)
        # before reaching here; whatever went wrong is on us.
        return jsonify({"ok": False, "error": str(exc)}), 500

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
        # Builder warnings (e.g. a mismatched-duplex steric clash) for the UI to
        # surface; empty list when the build was clean.
        "build_warnings":     build_warnings,
    })


@bp.route("/api/structure/periodicity", methods=["POST"])
def api_periodicity():
    """The unified periodicity door (structure-periodicity.md § 6.2): ONE
    entry point for the FOUR Cell-page edits — ``vacuum`` / ``axis_kind`` /
    ``cell`` / ``cell_origin`` (``periodicity_gate.OPS``) — through the
    frame-contract gate.  There is deliberately NO ``calibrate`` op: moving
    atoms is not a periodicity edit, it lives with the Modify ops
    (``/api/modify/calibrate``), and emission translates to the engine frame
    implicitly.

    Body: ``{"structure": <envelope>, "op": <one of OPS>, "payload": ...}`` --
    THE ENVELOPE every other structure door takes (web-api.md § 1), so a caller
    holding coordinates as numbers never writes a coordinate document to ask a
    question about them (molview.md § 11.7).  ``payload`` is required (may be
    ``null``) for ``cell`` / ``cell_origin``, where ``null`` means "clear it" --
    omitting the key is an error rather than a silent clear.

    This door used to take a ``{"data": {xyz, sidecar}}`` blob, which the one
    caller that exists -- MolView's ``commitPeriodicityOp``, the ONE door the
    cell changes through (molview.md § 6.2) -- could not produce, because the
    browser writes no coordinate document.  So it answered 400 to every request
    ever made of it and the cell door had never once succeeded.

    Response: ``{ok, periodicity, notices}`` -- ``periodicity`` is the cell block
    exactly as ``/api/build/load`` sends it (``cell`` / ``cell_origin`` /
    ``axis_kind`` / ``vacuum`` plus the ``resolved_*`` views beside them), so the
    client adopts it verbatim through the same path a load takes and there is one
    shape for the block rather than two.  ``notices`` is a list of
    ``{level, message}`` for the Cell page (molview.md § 6.8).  There is no
    "the gate changed this" marker and there should not be: clause 1 forbids the
    gate writing a resolved value back, so nothing is ever changed to mark.

    400 on: an unknown op, a missing payload for ``cell`` / ``cell_origin``, a
    malformed envelope, and every contract violation the gate raises (a
    left-handed cell, a cell no origin could make fit, a degenerate derived
    box, a periodic axis with no explicit cell)."""
    from molbuilder.periodicity_gate import apply_edit, OPS, validate_periodicity
    body = request.get_json(silent=True) or {}
    op = body.get("op")
    if op not in OPS:
        return jsonify({"ok": False,
                        "error": f"'op' must be one of {list(OPS)}"}), 400
    if not isinstance(body.get("structure"), dict):
        return jsonify({"ok": False,
                        "error": "missing or invalid 'structure' envelope "
                                 "(need {elements, positions, metadata})"}), 400
    if op in ("cell", "cell_origin") and "payload" not in body:
        # For these ops a null payload is a DESTRUCTIVE action (clear /
        # reset) -- a dropped key must not be indistinguishable from an
        # explicit clear.
        return jsonify({"ok": False,
                        "error": f"op '{op}' requires an explicit "
                                 f"'payload' (use null to clear/reset)"}), 400
    try:
        struct = _struct_from_body(body)
        # The frame-contract gate every other structure door runs, so what the
        # edit is applied to is what a load would have produced.  Its notices
        # describe the state that ARRIVED and are deliberately dropped: this
        # answer describes the state the edit PRODUCED, and a user who has just
        # corrected a box must not be told it is still wrong (molview.md § 6.8).
        struct, _incoming = validate_periodicity(struct)
        new_struct, receipts = apply_edit(struct, op, body.get("payload"))
        # The CONDITIONS are re-derived on the RESULT, so "the box does not
        # contain the structure" is answered about the box that now exists.
        new_struct, conditions = validate_periodicity(new_struct)
        # Receipts first (what the edit did), then conditions (what is now true).
        notices = list(receipts) + list(conditions)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 -- malformed envelope -> 400
        return jsonify({"ok": False, "error": str(exc)}), 400
    # The block the structure itself assembles -- raw values and the resolved
    # views together -- so a field added to it reaches this door with no edit
    # here, and the client cannot be handed a block missing the resolved half.
    return jsonify({
        "ok": True,
        "periodicity": new_struct.to_wire()["periodicity"],
        "notices": notices,
    })


@bp.route("/api/structure/save", methods=["POST"])
def api_structure_save():
    """FILE-ONLY save through the ONE authority (structure-authority.md § 3.3), the
    symmetric inverse of the file-only load below.  The browser hands the SETTLED model
    as the structure envelope (web-api.md § 1); the SERVER writes the ``<stem>.xyz`` + ``<stem>.molstruct.json``
    pair via ``StructureCodec.write``.  Python owns the pairing, the write order/atomicity,
    AND the sidecar schema -- ``write`` stamps ``schema_version`` + a real ``structure_hash``
    (``molstruct.to_dict``), so the pair the load door reads back is VALID.  The browser
    never authors the sidecar envelope (the drift that made a browser-written sidecar
    unloadable).  Body: ``{"path": "<project-relative .xyz>", "structure": {...},
    "overwrite": bool}``.  Returns ``{ok:true, path}`` | ``{ok:false, needsOverwrite:true}``
    (409, drives the tab's overwrite dialog) | ``{ok:false, error}``."""
    from molbuilder.web.blueprints.files import _resolve_within_roots, _PickerError
    from molbuilder.workingcopy_structure import StructureCodec
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    overwrite = bool(body.get("overwrite"))
    if not isinstance(path, str) or not path:
        return jsonify({"ok": False, "error": "missing or invalid 'path'"}), 400
    # THE STRUCTURE, not a document the browser wrote.  A `{xyz, sidecar}` blob
    # was the old shape, and taking it is what left the browser writing the
    # `.xyz` half -- the one-path rule (molview.md § 11.7) cannot be true while a
    # door accepts bytes.  The structure arrives as the envelope every other door
    # takes (web-api.md § 1), and the SERVER writes both files from it.
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    try:
        resolved = _resolve_within_roots(path)   # save-as target need not exist yet
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    # Overwrite gate: the GEOMETRY file's existence drives the tab's overwrite dialog
    # (mirrors the /api/files/write 409 contract) -- refuse unless the caller confirmed.
    if resolved.exists() and not overwrite:
        return jsonify({"ok": False, "needsOverwrite": True,
                        "error": f"file already exists: {path}"}), 409
    frames = body.get("frames")
    if frames is not None and not isinstance(frames, list):
        return jsonify({"ok": False,
                        "error": "'frames' must be a list of coordinate lists"}), 400
    try:
        StructureCodec().write(struct, resolved, frames=frames)
    except ValueError as exc:          # a frame that does not carry these atoms
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 -- disk / permission -> 500
        return jsonify({"ok": False, "error": f"could not save {path}: {exc}"}), 500
    return jsonify({"ok": True, "path": path})


@bp.route("/api/structure/export", methods=["POST"])
def api_structure_export():
    """The pair a save would write -- NAMED, and returned instead of written.

    Same generator, different destination: :func:`api_structure_save` puts
    ``StructureCodec.pair`` on disk and this hands it back through
    ``StructureCodec.files``, so a structure saved into a project and the same
    structure downloaded are byte-identical BY CONSTRUCTION rather than by two
    code paths agreeing.

    That division exists because the browser cannot produce the pair itself. The
    sidecar's envelope -- ``schema_version``, and the ``structure_hash`` pinning
    it to its geometry -- is the codec's, and a browser-authored one shipped
    without the version key once: the load door then refused the pair on the next
    open and every label in it was silently dropped.

    Body: the ENVELOPE (web-api.md § 1) plus two optional keys --
    ``{"structure": {...}, "name": "<stem>", "frames": [...]}``.

    WHO NAMES WHAT.  ``name`` is a STEM and nothing else (``wire_frame40-120``,
    no extension), because only the caller knows what an export IS: which
    structure, which frames, chosen at which moment.  The SUFFIX is the server's,
    because it follows from the format and the format follows from the frame
    count, which ``pair()`` already decided -- a caller that appends its own is
    answering a question that has an answer.  The caller that did appended
    ``.xyz`` to a multi-frame export, so a download arrived named ``.xyz`` with
    extended-XYZ ``Lattice=`` lines inside it, at the extension every trajectory
    reader dispatches on.  A missing / empty / path-shaped ``name`` falls back to
    ``structure``; only the last path component is ever used, and nothing here
    touches the filesystem.

    Returns ``{ok, files: [{name, text}], frames, notices}`` -- each entry is a
    file as it would exist on disk, under the name it would exist as.  One entry
    means the structure carries no metadata worth keeping, which is exactly when
    a save writes no ``.json`` either (``no .json == empty metadata``)."""
    from molbuilder.workingcopy_structure import StructureCodec
    body = request.get_json(silent=True) or {}
    if not isinstance(body.get("structure"), dict):
        return jsonify({"ok": False,
                        "error": "missing or invalid 'structure' envelope "
                                 "(need {geometry: {elements, positions}, metadata})"}), 400
    try:
        struct = _struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # The same gate every other structure door runs, so what leaves here is
    # judged by the same rules a save is judged by -- and a cell it refuses
    # leaves as a 400 carrying the gate's sentence, not as a 500.
    from ._shared import checked_periodicity
    struct, notices = checked_periodicity(struct)
    # THE FRAMES, when a range was asked for (molview.md § 11.3).  They ride
    # BESIDE the envelope rather than inside it -- the same shape
    # ``/api/build/load`` takes on the way in: one structure carrying the
    # identity and the metadata, plus the coordinates of the frames wanted.
    # Absent, this is the single-frame export it always was.
    frames = body.get("frames")
    if frames is not None and not isinstance(frames, list):
        return jsonify({"ok": False,
                        "error": "'frames' must be a list of coordinate lists"}), 400
    # THE STEM, reduced to its last component.  This never reaches the
    # filesystem -- ``files()`` builds names in memory -- but it does reach the
    # browser as a download name, so a path-shaped one is flattened rather than
    # passed on.
    raw_name = str(body.get("name") or "").replace("\\", "/")
    stem = raw_name.rsplit("/", 1)[-1].strip()
    if not stem or stem in (".", ".."):
        stem = "structure"
    try:
        made = StructureCodec().files(struct, stem, frames=frames)
    except ValueError as exc:          # a frame that does not carry these atoms
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True,
                    "files": [{"name": path.name,
                               "text": blob.decode("utf-8")}
                              for path, blob in made],
                    "frames": len(frames) if frames else 1,
                    "notices": notices})


@bp.route("/api/build/load", methods=["POST"])
def api_build_load():
    """Accept either:
      * multipart/form-data with a single file field "file", or
      * JSON {"path": "<project-relative structure file>"} -- the
        FILE-ONLY load: the SERVER reads the .xyz(/.pdb) + its paired
        .molstruct.json through StructureCodec.read (the ONE authority
        owns the file access AND the pairing).  This is how a project
        file is opened -- no raw text, no browser-side sidecar path.
      * JSON {"text": "...", "format": "xyz"|"pdb"|"auto",
              "filename": "<optional>"} -- raw-geometry IMPORT (a paste /
        upload with no persisted file yet); metadata-less.
    Returns the same JSON shape as /api/build/molecule so the front
    end can treat the result identically.
    """
    # FILE-ONLY load through the ONE authority (structure-authority.md): a project
    # ``path`` means the SERVER reads the .xyz + paired .molstruct.json via
    # StructureCodec.read -- Python owns the file access + the .xyz<->.molstruct
    # pairing, so there is NO raw-text hand-crafting and NO browser-side sidecar
    # derivation.  ``to_wire`` (via ok_structure_response) emits the enriched atoms
    # + periodicity + annotations in one response.
    _pbody = request.get_json(silent=True) or {}
    _path = _pbody.get("path")
    if _path:
        from molbuilder.web.blueprints.files import (
            _resolve_within_roots, _PickerError)
        from molbuilder.workingcopy_structure import StructureCodec
        try:
            _resolved = _resolve_within_roots(_path)
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status
        if not _resolved.exists():
            return jsonify({"ok": False, "error": f"no such file: {_path}"}), 404
        try:
            struct = StructureCodec().read(_resolved)
        except Exception as exc:  # noqa: BLE001 -- parse/sidecar error -> 400
            return jsonify(
                {"ok": False, "error": f"could not load {_path}: {exc}"}), 400
        # No `notices` passed: `ok_structure_response` validates every structure
        # it sends, so the conditions for THIS one are produced on the way out.
        # The codec used to hand its own copy of them up as well, and the load
        # door answered with the same sentence twice.
        return ok_structure_response(struct, extra={
            "source_format": ("pdb" if str(_resolved).lower().endswith(".pdb")
                              else "xyz"),
            "title": struct.title or _resolved.name,
        })

    # A STRUCTURE PUT BACK, with no file and no text behind it.  A tab that
    # showed a structure before the page was left hands the SAME envelope every
    # edit posts -- atoms as numbers, the facts beside them -- and gets the same
    # answer a file load gives, so a restored viewer is indistinguishable from a
    # freshly-loaded one.  Nothing is parsed: the one deserialiser rebuilds it,
    # and refuses a malformed envelope rather than half-building a structure.
    if isinstance(_pbody.get("structure"), dict):
        from ._shared import struct_from_body
        try:
            struct = struct_from_body(_pbody)
        except (ValueError, TypeError) as exc:
            return jsonify({"ok": False,
                            "error": f"could not restore structure: {exc}"}), 400
        # THE BOX CAME IN THE ENVELOPE, like everything else about these atoms,
        # and `from_dict` applied it.  Nothing more to apply.
        #
        # This ran `apply_periodicity_only(struct, _pbody)` here, which reads a
        # TOP-LEVEL `periodicity` and writes it over what the envelope just set
        # -- a second source for the cell on a route that had already taken a
        # first (2026-08-04, the same shape the labels wore in #41).  It could
        # not fire from the shipped client: `requestBodyFor` RETURNS on the
        # structure branch, so a restore body cannot carry both keys.  A reader
        # nobody can currently reach is still a reader; that is exactly how the
        # label version stayed invisible for months.
        #
        # It is still applied on the TEXT branch below, and is right there: that
        # body has no envelope, so a stated block is the only way to say what
        # the box is.  A load APPLIES rather than refuses either way -- a bad
        # box is reported with the answer (`ok_structure_response`) so the user
        # can open the structure and fix it in the Cell page.  Refusing would
        # make a structure with a bad box unopenable, and so unfixable.
        return ok_structure_response(struct, extra={
            "source_format": "xyz",
            "title": struct.title or "restored structure",
        })

    text: str = ""
    fmt: str = "auto"
    filename: str = ""
    # The paired .molstruct.json CONTENT (raw JSON string), read by the browser
    # through the concealed projects file package (``projects.readFile``) and
    # handed in so this ONE parse seam applies the sidecar -- regions / frozen /
    # cell / axis_kind / vacuum / annotations -- onto the parsed Structure.
    # None (or the multipart upload path) -> a plain geometry load, no sidecar.
    sidecar_text: str = ""
    # The TRUSTED per-atom metadata block (regions / frozen / annotations)
    # a results-side caller recovered from a run's input script's
    # ATOM-METADATA block (parse.scripts.atom_metadata) -- NOT a standalone
    # .molstruct.json file.  Distinct from ``sidecar`` because it is
    # molbuilder's own emit and by design omits the sidecar-file envelope
    # (structure_hash), so it is applied via ``apply_to_structure`` (lenient,
    # atom-count-only), never validated through ``load_text``.  Carries only
    # atom-scoped keys, so the parsed geometry / cell above stay intact.
    atom_metadata_text: str = ""
    # Bound on BOTH branches: a multipart upload carries no JSON, and the
    # periodicity seam below reads this for every path through the route.
    body: Dict[str, Any] = {}
    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename or ""
        text = f.read().decode("utf-8", errors="replace")
    else:
        body = request.get_json(silent=True) or {}
        text = body.get("text") or ""
        fmt = (body.get("format") or "auto").lower()
        filename = body.get("filename") or ""
        sidecar_text = body.get("sidecar") or ""
        atom_metadata_text = body.get("atom_metadata") or ""

    if not text.strip():
        return jsonify({"ok": False, "error": "empty input"}), 400

    if fmt == "auto":
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("xyz", "pdb"):
            fmt = ext
        else:
            # Sniff by content -- the ONE shared rule (XYZ's first non-blank line is
            # a POSITIVE atom count; anything else is PDB).  Delegate to the same
            # helper the /api/build/molecule path uses so the two never disagree on an
            # edge case (e.g. a leading "0" line: int("0")>0 is False -> pdb, whereas
            # the old inline `"0".isdigit()` said xyz).
            fmt = _sniff_structure_format(text)

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

    # Apply the paired .molstruct.json (if the caller handed its content):
    # regions / frozen / cell / axis_kind / vacuum / annotations land on the
    # parsed Structure so ``ok_structure_response`` emits the ENRICHED atoms +
    # periodicity + annotations in ONE response.  This is the parse seam
    # ``molview.data.openMolecule`` calls after reading BOTH files through the
    # projects file package -- the sidecar schema lives in one place
    # (sidecars/molstruct), never in the file layer or the browser.
    if sidecar_text.strip():
        from molbuilder.sidecars import molstruct as _molstruct
        try:
            _molstruct.apply_to_structure(
                struct, _molstruct.load_text(sidecar_text))
        except _molstruct.MolstructJsonError as exc:
            # A malformed / atom-count-mismatched sidecar is a client error, not
            # a 500 -- surface the schema module's precise message.
            return jsonify({"ok": False,
                            "error": f"sidecar: {exc}"}), 400

    # Trusted per-atom metadata block (see ``atom_metadata_text`` above):
    # apply the SAME regions / frozen / annotations fields onto the parsed
    # Structure via the lenient seam.  ``apply_to_structure`` re-runs its
    # own atom-count + index validation and raises MolstructJsonError on a
    # mismatch (surfaced as a 400, same as the sidecar path).  The block is
    # trusted JSON, so it is parsed directly -- NOT through ``load_text``,
    # which would reject it for lacking the untrusted-file envelope.
    if atom_metadata_text.strip():
        import json as _json
        from molbuilder.sidecars import molstruct as _molstruct
        try:
            _molstruct.apply_to_structure(
                struct, _json.loads(atom_metadata_text))
        except _json.JSONDecodeError as exc:
            return jsonify({"ok": False,
                            "error": f"atom_metadata: not valid JSON: {exc}"}), 400
        except _molstruct.MolstructJsonError as exc:
            return jsonify({"ok": False,
                            "error": f"atom_metadata: {exc}"}), 400

    # THE PERIODICITY THE CALLER STATED: ``body["periodicity"]`` =
    # {cell, cell_origin, axis_kind, vacuum}, applied verbatim.
    #
    # APPLIED, NOT JUDGED.  This is a LOAD, so a bad box is REPORTED with the
    # answer rather than refused -- the emitting doors (fdf / pyscf / preflight
    # / spectra / transport / export) are the ones that refuse, because what
    # they produce is a calculation somebody runs.  A load that refused would
    # leave a structure with a bad box unopenable, and so unfixable: the user
    # could not even get it on screen to correct it.
    #
    # WHY IT IS A FIELD OF ITS OWN and not folded into the metadata block above.
    # A run has no `.molstruct.json`: the Results tab recovers its labels from
    # the input script and its lattice from the output logs -- two facts from
    # two places.  Folding the lattice into the labels document meant the
    # browser opening a document the server wrote, and re-stamping the atom
    # count that guards it.
    #
    # AFTER the sidecar / atom_metadata application, never before:
    # ``apply_metadata_dict`` is full-REPLACE, so a block applied second would
    # reset the cell this just set.
    from ._shared import apply_periodicity_only
    struct = apply_periodicity_only(struct, body)

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


# ---------------------------------------------------------------------- #
#  DELETED 2026-08-17 -- ``/api/build/fdf`` and ``/api/build/pyscf``.      #
#                                                                         #
#  The two deck-emitting doors.  Script generation left the               #
#  structure-optimization tab on 2026-08-15 (user: the tab collects        #
#  parameters, the staging surface owns the rest), and nothing replaced    #
#  the callers: `/api/build/fdf` had ZERO references in any JS or HTML     #
#  outside a comment saying it was orphaned, and `/api/build/pyscf` had    #
#  none at all.  Both were reachable, both rendered a deck, and only       #
#  tests called them.                                                     #
#                                                                         #
#  A reachable door with no caller is not free.  It is a second way to     #
#  render a deck -- the thing `prep` owns (`generator.md` § 7) -- kept     #
#  alive by its own tests, which is how a "still works" argument gets      #
#  made for a path no user can take.                                      #
#                                                                         #
#  A browser renders no deck.  `jobset prep` does, on the machine that     #
#  will run it (`project-layout.md` § 2.2).                               #
# ---------------------------------------------------------------------- #


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
    _cls, id_prefix = cls_map[engine]
    # Built from the CATALOGUE (`web/form-schema.md` § 1), not from the config
    # class: a parameter is defined in molbuilder/data/catalogue.template.toml,
    # and the class is a translator on the way OUT to an engine.  The renderer
    # is unchanged -- it takes whatever schema it is handed.
    return jsonify({
        "ok": True,
        "schema": _catalogue_to_form_schema(engine, id_prefix),
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
    engine = (body.get("engine") or "").strip().lower()
    params: Dict[str, Any] = body.get("params") or {}

    if engine not in ("siesta", "pyscf"):
        return jsonify({
            "ok": False,
            "error": f"engine must be 'siesta' or 'pyscf'; got {engine!r}",
        }), 400

    # THE STRUCTURE ARRIVES AS DATA, through the one reader every structure
    # door shares: the atoms as numbers and the facts beside them, with the
    # legacy `xyz` text still accepted for a caller that has only text.
    #
    # THIS DOOR READ `xyz` AND NOTHING ELSE, and the browser stopped sending it
    # -- the tab posts the envelope, like every other emitting door already
    # takes.  So Generate FDF, Generate PySCF and the live preflight on
    # /structure-optimization all answered `400 no xyz provided` for the exact
    # body the tab sends.  Found by driving the page: the boot test caught the
    # console error only once a restored structure made the preflight fire
    # before anybody clicked anything.
    try:
        struct = _struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # Preflight must see exactly what Generate sees (labels + the
    # model's periodicity truth) -- it validated a phantom before.
    from ._shared import periodicity_checked_for_emit
    struct = periodicity_checked_for_emit(struct)

    try:
        if engine == "siesta":
            cfg = _siesta_config_from_params(params)
        else:
            cfg = _pyscf_config_from_params(params)
    except Exception as exc:
        # L3 R4-A fix 2026-06-14: the "bad params" branch now
        # returns ``ok: False`` to match the wire-api.md envelope
        # contract -- pre-fix it returned ``ok: True`` even though
        # the config didn't parse, which the UI (viewer.js:211)
        # silently ignored because it gates issue-rendering on
        # ``r.ok``.  Switching to ``ok: False`` + the same issue
        # makes the failure surface uniformly with
        # ``/api/build/fdf``'s 400 path; the UI's existing
        # ``!body.ok`` gate then renders the issue + the user sees
        # the parse error in the issues panel.
        return jsonify({
            "ok":     False,
            "error":  f"bad parameters: {exc}",
            "issues": [{"severity": "error",
                        "message": f"bad parameters: {exc}",
                        "where":   "config"}],
        }), 400

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


# --------------------------------------------------------------------- #
#  Hand-over to Task setup                                              #
# --------------------------------------------------------------------- #

#: The hand-over file's own schema.  **Not** ``molbuilder/task@1``, and the
#: difference is the point: this file is deliberately INCOMPLETE -- it carries
#: what the parameter tab knows and cannot carry ``shape``, which is required
#: with no default because inferring it "would hand somebody a directory tree
#: they never asked for" (`engines/stages.md` § 6.7).  A file claiming
#: ``molbuilder/task@1`` while failing its own reader is worse than one that
#: says what it is; ``check_schema`` refuses a wrong artifact BY NAME, so this
#: cannot be mistaken for a description anywhere.
TASK_HANDOVER_SCHEMA = "molbuilder/task-handover@1"

#: What the hand-over is called on disk.  The extension is LAST on purpose --
#: `task.1st.json`, not `task.json.1st` -- so the editor's suffix map gives it
#: JSON highlighting (`lib/codemirror-load.js`), and so nothing looking for
#: `task.json` finds it.  That second half matters more than it looks:
#: `checkpoint.py::_BUNDLE_DESCRIPTORS` treats a `task.json` as the marker that
#: a folder "declares itself the root of one multi-directory unit of work", so
#: writing a premature one would make the folder claim to be a calculation root
#: before it is one (`checkpointing.md` L1).
TASK_HANDOVER_NAME = "task.1st.json"


@bp.route("/api/task-setup/handover", methods=["POST"])
def api_task_setup_handover():
    """RENDER the parameter tab's work, for the browser to write.

    **This writes nothing.**  `web/projects.md` § 1 puts raw bytes in the
    content-blind file layer that *"every tab can use"* -- `writeFile` /
    `safeSave` / `deleteEntry` -- and a tab that opens files itself bypasses the
    roots guard, the lock, the uniform `{ok, ...}` envelope and the sidebar
    re-list that come with it.  So this returns the two TEXTS and the caller
    puts them where the user chose, through `projects.safeSave`.

    What is genuinely server-side is the render: only Python can turn a config
    into `<label>.template.toml`, because `template_with_values` narrows the
    catalogue and fills in the answers.

    Four files, and none of them is a runnable anything:

      * ``<label>.template.toml`` -- every parameter with the value in force.
        This is the file the parameter tab's work has been going into a void
        for: the tab collects the physics and produces no artifact, so without
        this there is no path from the form to a calculation at all.
      * ``task.1st.json`` -- what the tab knows about the calculation ITSELF:
        the engine, the structure it is of, and what it is called.
      * ``<label>.xyz`` + ``<label>.molstruct.json`` -- THE STRUCTURE, from
        ``StructureCodec``, the same generator ``/api/structure/export`` uses.
        ``molview.md`` § 11.7: the server writes every file, so the pair a
        person downloads and the pair that lands here cannot differ.

    **This is a hand-over, not a description.**  `tabs.md` forbids an in-memory
    "send to tab" hand-off, and this obeys it -- the transfer goes through disk,
    so the receiving tab reads files like every other reader and nothing depends
    on state you cannot see in the folder.  Task setup finishes the job: it asks
    for the shape, takes the stages, and on a successful save writes the real
    ``task.json`` and removes this file.
    """
    body = request.get_json(silent=True) or {}
    engine = str(body.get("engine") or "siesta").lower()
    if engine not in ("siesta", "pyscf"):
        return jsonify({"ok": False,
                        "error": f"unknown engine {engine!r}"}), 400

    try:
        struct = _struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    params: Dict[str, Any] = body.get("params") or {}
    try:
        cfg = (_siesta_config_from_params(params) if engine == "siesta"
               else _pyscf_config_from_params(params))
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    from molbuilder.identity import normalise_id, run_id
    from molbuilder.template import (template_filename as _template_filename,
                                     template_with_values)

    # WHAT THIS CALCULATION IS CALLED — the identity the person typed, and the
    # destination folder only when they did not.
    #
    # `run-identity.md` § 4: *"The label is the SystemLabel / JOB literal.
    # There is no second name."*  This took the folder's name unconditionally,
    # so the identity typed on the parameter tab stayed in the template while
    # `task.json` carried another — two names for one calculation, and both
    # persisted.  Downstream the engine wrote `<system_label>.XV` while every
    # file molbuilder named was stemmed on the task label, so `prep --from`
    # refused a carry from a stage that HAD run and had produced exactly those
    # files: *"that attempt holds none of the files this stage would continue
    # from. Did it run?"*
    #
    # Which field carries the identity is the ENGINE's to say, and it says so
    # (`RestartGroup.field`) — no `if engine ==` here.  The folder name is
    # still the answer when the field is untouched, because the schema default
    # is a placeholder (`siesta`, `pyscf_relax`) and naming a calculation after
    # a placeholder is worse than naming it after the folder somebody chose.
    from molbuilder.config.pyscf import PYSCF_RESTART_GROUP
    from molbuilder.config.siesta import SIESTA_RESTART_GROUP
    _group = SIESTA_RESTART_GROUP if engine == "siesta" else PYSCF_RESTART_GROUP
    _identity = str(getattr(cfg, _group.field, "") or "")
    _placeholder = str(
        type(cfg).__dataclass_fields__[_group.field].default or "")
    typed = (_identity if _identity and _identity != _placeholder
             else (str(body.get("name") or "") or "calculation"))
    label = normalise_id(typed)
    formula = str(getattr(struct, "formula", "") or "")

    # AND THE TEMPLATE CARRIES THE SAME ONE.  Choosing the label above is only
    # half of "there is no second name": the template's identity field is what
    # the ENGINE writes its files under, so if it kept the placeholder while
    # `task.json` took the folder's name, the split would simply reappear from
    # the other side.  Normalisation happens once and the result is stored
    # (§ 3 rule 1) — this is the storing.
    import dataclasses as _dc
    cfg = _dc.replace(cfg, **{_group.field: label})

    try:
        template_text = template_with_values(cfg, engine=engine)
    except Exception as exc:                      # a bad value, named
        return jsonify({"ok": False, "error": str(exc)}), 400

    # THE STRUCTURE ITSELF, from the one generator.  `molview.md` § 11.7: the
    # server writes every file, because a browser-authored pair drifts from the
    # server's -- it shipped once without the sidecar's `schema_version` and
    # every label in it was dropped silently on the next open.  So this asks
    # `StructureCodec` for the pair exactly as `/api/structure/export` does, and
    # the two are byte-identical by construction rather than by agreement.
    #
    # The STEM is ours (the calculation's label); the SUFFIXES are the codec's,
    # because the format follows from the frame count and the pairing rule has
    # one home (`model/structure.md` § 2.4).  A caller appending `.xyz` here
    # would be answering a question that already has an answer.
    from molbuilder.workingcopy_structure import StructureCodec
    from ._shared import checked_periodicity
    struct, struct_notices = checked_periodicity(struct)
    try:
        made = StructureCodec().files(struct, label)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    structure_files = [{"name": path.name, "text": blob.decode("utf-8")}
                       for path, blob in made]
    # `source` names the COORDINATE document only.  The sidecar beside it is
    # found by the pairing rule, which has one home (`model/structure.md`
    # § 2.4) and is the codec's -- naming it here would be a second copy of a
    # rule this file does not own, and § 11.7 says that is how the `.extxyz`
    # round trip came to not close.
    geometry_name = next((f["name"] for f in structure_files
                          if not f["name"].endswith(".json")), "")

    handover = {
        "schema":    TASK_HANDOVER_SCHEMA,
        # JSON has no comments, so the file carries a line that says what it
        # is.  It IS read by a person -- Task setup shows it in the editor --
        # and a file whose whole job is to be handed between two surfaces
        # should not need a document open beside it to be understood.
        "_what":     "A hand-over from the Structure-optimization tab, not a "
                     "description. It carries the parameters (in the .template.toml "
                     "beside it) plus what this calculation is OF. It is missing "
                     "`shape` and `stages` on purpose -- Task setup asks for those, "
                     "and on a successful save writes the real task.json and deletes "
                     "this file. Nothing runs from it. The structure it is OF is the "
                     "file named under `structure.files` in this same folder, written "
                     "by the server's one codec: `structure.source` names the .xyz, "
                     "which carries the coordinates and the cell, and the "
                     ".molstruct.json beside it carries the region labels and frozen "
                     "atoms.",
        "engine":    {"name": engine},
        "run":       {"name": typed,
                      "id": run_id(typed, formula),
                      "created": datetime.now().astimezone().isoformat(timespec="seconds")},
        # WHAT THIS IS OF -- by NAME, pointing at files in this same folder.
        # It used to record `structure_path`, which was the projects sidebar's
        # selected file: a second fact read at a second moment, which
        # `molview.md` § 9.3a forbids for exactly the reason it went wrong --
        # the cursor sat on a `.template.toml`, so the hand-over claimed a
        # calculation was OF its own parameter file.  These names come from the
        # structure that was sent, so they cannot disagree with it.
        "structure": {"source":  geometry_name,
                      "formula": formula,
                      "atoms":   len(getattr(struct, "elements", []) or [])},
        # No `shape`, no `stages` -- Task setup asks.  Stated rather than
        # omitted so a reader of the file knows it is waiting on them.
        "awaiting":  ["shape", "stages"],
    }

    return jsonify({
        "ok":            True,
        "label":         label,
        # THE door (`template.template_filename`), not a literal suffix --
        # this was the seventh site forming this name, and the one the
        # 2026-08-17 sweep missed because it spelled `.template.toml`
        # rather than joining SUFFIX.
        "template_name": _template_filename(label),
        "template_text": template_text,
        "handover_name": TASK_HANDOVER_NAME,
        "handover_text": json.dumps(handover, indent=2) + "\n",
        # Each entry is a file as it would exist on disk, under the name it
        # would exist as -- nothing is left for the browser to work out.
        "structure_files": structure_files,
        "notices": struct_notices,
    })


@bp.route("/api/task-setup/save", methods=["POST"])
def api_task_setup_save():
    """Write the description a person has been reading, and resolve a hand-over.

    **The BUFFER is the source.**  The editor is where a description is checked
    and corrected before it is written (`web/task-setup.md` § 9a), so this takes
    the text as edited -- never a re-serialisation of a parsed model, which
    would silently discard whatever was typed in the editor.

    **Refused rather than repaired.**  The text goes through the shipped reader
    (`task.read_task`), so a description that does not parse, names a field the
    schema does not know, or carries no stage is refused with the reason -- the
    same answer the CLI gives, from the same code.  A browser that "fixed" a
    description would be the second, drifting writer this design exists to
    avoid.

    **The hand-over resolves in one direction.**  On success `task.json` exists
    and `task.1st.json` is removed, so the next visit finds one description and
    no ambiguity about which file is current (`engines/stages.md` § 6.5a).
    Removed only AFTER the write succeeds: the reverse order loses the
    parameters if the write fails.
    """
    body = request.get_json(silent=True) or {}
    dest_raw = str(body.get("dest") or "")
    text     = body.get("text")
    if not dest_raw:
        return jsonify({"ok": False, "error": "no destination folder given"}), 400
    if not isinstance(text, str) or not text.strip():
        return jsonify({"ok": False, "error": "nothing to save"}), 400

    try:
        dest = _resolve_within_roots(dest_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not dest.is_dir():
        return jsonify({"ok": False, "error": f"not a directory: {dest_raw}"}), 400

    import tempfile
    # FILENAME comes from `task.py` too: the description's NAME is that
    # module's to spell, like its bytes (`task-description.md` § 6.4 --
    # one reader, so the two surfaces cannot produce different files).
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import read_task, write_task

    # Validate by READING it, in a scratch file, so nothing lands in the
    # calculation folder unless it is a description the rest of the system
    # can open.  `read_task` is the same door `prep` uses.
    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp) / TASK_FILENAME
        probe.write_text(text, encoding="utf-8")
        try:
            task = read_task(probe)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    # ONE JOB PER FOLDER (`job-contracts.md` § 2.1 Rule 1).  A folder already
    # describing a DIFFERENT calculation is not a folder this save may land in:
    # the ids say they are different calculations, and overwriting one with the
    # other orphans every warm file and output already keyed to it.
    #
    # Compared by RUN ID, not by path: the id is `<label>_<formula>` and is the
    # one thing that says which calculation a folder is
    # (`run-identity.md` § 2.0a).  Re-saving the SAME calculation is the
    # ordinary case and must stay free.
    existing = dest / TASK_FILENAME
    if existing.is_file():
        try:
            prior = read_task(existing)
        except Exception:
            prior = None                      # unreadable: let the write fix it
        if prior is not None and prior.run.id != task.run.id:
            return jsonify({
                "ok": False,
                "error": f"this folder already describes a different "
                         f"calculation ({prior.run.id!r}); saving {task.run.id!r} "
                         f"here would orphan its results. One job per folder — "
                         f"pick or make another.",
            }), 409

    try:
        write_task(dest / TASK_FILENAME, task)      # atomic (persist.write_json)
    except OSError as exc:
        return jsonify({"ok": False, "error": f"could not write: {exc}"}), 500

    # The hand-over's REMOVAL is the browser's, through
    # `projects.deleteEntry` -- moving bytes is the content-blind layer's job
    # (`projects.md` § 1), and unlinking here would bypass its guard and the
    # sidebar re-list.  Reported so the caller knows whether to.
    return jsonify({
        "ok":            True,
        "wrote":         TASK_FILENAME,
        "handover_name": TASK_HANDOVER_NAME,
        "handover_here": (dest / TASK_HANDOVER_NAME).is_file(),
        "stages":        [st.name for st in task.stages],
    })


@bp.route("/api/task-setup/sweepable", methods=["GET"])
def api_task_setup_sweepable():
    """The parameters a benchmark may sweep, for the Task-setup picker.

    **Not the form schema, and the difference is the rule.**
    ``catalogue_to_form_schema`` filters the ``staging`` group out — a
    parameter form does not ask how many ranks the scheduler granted — but
    those are exactly the knobs a benchmark measures.  So this reads the
    catalogue directly and applies § 6.8's rule instead:

      > A key must name a field the engine already declares sweepable — the
        ``execution`` category, which `template.md` § 6.2 defines as *"knobs
        that change speed and not the answer"*.

    Sweeping anything outside it means each point silently measures a
    DIFFERENT calculation, and the comparison is meaningless.

    Each item says whether the machine answers it: an ``allocation`` resolver
    means a description may never carry a value for it (`template.md` § 6.4),
    so the picker can show it as measurable-only rather than as a choice.
    """
    engine = str(request.args.get("engine") or "siesta").lower()
    if engine not in ("siesta", "pyscf"):
        return jsonify({"ok": False, "error": f"unknown engine {engine!r}"}), 400

    from molbuilder import template as _T
    parsed = _T.catalogue()
    out = []
    for it in _T.select(parsed, engine=engine):
        if "execution" not in (it.category or ()):
            continue
        out.append({
            "name":            it.name,
            "label":           it.label or it.name,
            "help":            it.help or "",
            "machine_answers": it.allocation,
        })
    return jsonify({"ok": True, "engine": engine, "items": out})


@bp.route("/api/task-setup/columns", methods=["GET"])
def api_task_setup_columns():
    """Which parameters may become a column of the stage table.

    `engines/stages.md` § 6.2: *"Any setting the description is allowed to hold
    may become a column. The ones it is not allowed to hold may not."*  There is
    no separate list — § 1.2 already says a stage may name any field of the
    shared schema, and `template.md` § 7 already forbids the description to hold
    the settings the machine answers.  Those two rules give the set with nothing
    left to decide, and it is the same membership `prep` applies when it accepts
    or refuses an override: a column offered here is a column `prep` will take.

    **Why this is not `/api/build/schema`**, which is what the tab read until
    2026-08-18.  That is the PARAMETER FORM's schema, and it filters the whole
    `staging` group out on purpose — a form does not ask a person how many ranks
    the scheduler granted (`form-schema.md` § 1.3).  Filtering a panel and
    limiting a table are different jobs, and borrowing the answer to the first
    for the second cost the table its most important column: `restart`, the
    field that decides whether a ladder is a ladder, sits in `staging` and so
    could never be added.  Every ladder built anywhere but
    `jobset describe --stage-strategy` therefore ran every stage `clean`.

    `group` rides along because it is still the right answer to a different
    question — which columns the table STARTS with (§ 1.3).
    """
    engine = str(request.args.get("engine") or "siesta").lower()
    if engine not in ("siesta", "pyscf"):
        return jsonify({"ok": False, "error": f"unknown engine {engine!r}"}), 400

    from molbuilder import template as _T
    parsed = _T.catalogue()
    out = []
    for it in _T.select(parsed, engine=engine):
        # THE membership rule, asked of the item rather than restated here.
        if it.allocation:
            continue
        out.append({
            "name":    it.name,
            "label":   it.label or it.name,
            "help":    it.help or "",
            "unit":    it.unit or "",
            "default": it.default,
            "group":   it.group or "",
            "engine_key": it.anchor or "",
        })
    return jsonify({"ok": True, "engine": engine, "items": out})


@bp.route("/api/task-setup/template-values", methods=["GET"])
def api_task_setup_template_values():
    """What the folder's own template answers -- the baseline a stage inherits.

    `engines/stages.md` § 6.2: a stage that sets nothing "uses the template's
    value".  THE TEMPLATE'S -- not the catalogue's.  A tab that shows a
    catalogue default in an empty cell is naming a number the job will not run
    whenever the sender changed that parameter, which is the whole point of the
    hand-over: the k-grid a person chose in a parameter tab lands HERE.

    The server reads it because TOML is a format, and `projects.md` § 3 keeps a
    format's correctness on this side of the wire -- `read_template` is the
    same parser `prep` opens the file with, so the browser cannot become a
    second reader that disagrees about what a value is.
    """
    dir_raw = str(request.args.get("dir") or "")
    if not dir_raw:
        return jsonify({"ok": False, "error": "no folder given"}), 400
    try:
        folder = _resolve_within_roots(dir_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not folder.is_dir():
        return jsonify({"ok": False, "error": f"not a directory: {dir_raw}"}), 400

    # THE door, not a glob (`template.find_template`).  This took
    # ``sorted(glob(...))[0]`` until 2026-08-17 -- so a folder holding two
    # templates had this tab reading one file and `prep` reading the other,
    # which is precisely the split this endpoint's own docstring argues
    # against one layer down: it shared `prep`'s PARSER and not its PATH.
    from molbuilder.template import find_template, read_template, select
    try:
        found = find_template(folder)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if found is None:
        return jsonify({"ok": True, "name": None, "values": {}})

    try:
        tmpl = read_template(found.read_text())
    except Exception as exc:
        # A template that does not parse is the user's to fix, and saying which
        # file beats an empty table that looks like "nothing was sent".
        return jsonify({"ok": False, "name": found.name,
                        "error": f"{found.name}: {exc}"}), 400

    # THE one read API (`engines/template.md` § 8.0), not a comprehension over
    # `.items` -- which is what this was, and which re-implemented the
    # `Template.values()` deleted on 2026-08-17 as one of four second readers.
    values = {it.name: it.value for it in select(tmpl) if it.is_set}
    return jsonify({"ok": True, "name": found.name, "values": values})


@bp.route("/api/task-setup/presets", methods=["GET"])
def api_task_setup_presets():
    """The shipped tier presets, for filling a stage's row.

    These are the SAME table `default_siesta_stages` builds the shipped ladder
    from, so a stage filled here and a stage of the default ladder cannot drift
    -- `engines/tuning.md` § 4 is the authority for what number each tier
    carries, and this serves it rather than restating it.
    """
    engine = str(request.args.get("engine") or "siesta").lower()
    out = []
    if engine == "siesta":
        from molbuilder.config.siesta import (SIESTA_STAGE_NAMES,
                                              SIESTA_STAGE_PRESETS)
        for tier in sorted(SIESTA_STAGE_PRESETS):
            out.append({"tier": tier,
                        "name": SIESTA_STAGE_NAMES.get(tier, f"stage{tier}"),
                        "values": dict(SIESTA_STAGE_PRESETS[tier])})
    elif engine == "pyscf":
        # Same source as the shipped PySCF ladder, for the same reason the
        # SIESTA arm above reads SIESTA's: a stage filled from a preset here
        # and a stage of the default ladder must not be able to disagree.
        # ``restart`` is dropped -- it is a rung's POSITION, not its tier
        # (`run-identity.md` § 4 rule 3), so it is not a value to fill a row
        # with.  Every tier is offered whatever the strategy enables; the
        # enable-mask is the ladder's business, not this menu's.
        from molbuilder.pyscf.stages import default_pyscf_stages
        for i, st in enumerate(default_pyscf_stages(), start=1):
            out.append({"tier": i, "name": st.name,
                        "values": {k: v for k, v in st.overrides.items()
                                   if k != "restart"}})
    else:
        return jsonify({"ok": False, "error": f"unknown engine {engine!r}"}), 400
    return jsonify({"ok": True, "engine": engine, "presets": out})
