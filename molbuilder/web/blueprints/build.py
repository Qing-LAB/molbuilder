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
)

from molbuilder import (
    build_dna, build_from_name, build_from_smiles,
    build_peptide, build_rna,
)
from molbuilder.config.pyscf  import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.issues import ValidationError
from molbuilder.pyscf  import render_script
from molbuilder.siesta import render_fdf
from molbuilder.structure import Structure
from molbuilder.validation import validate, validate_geometry


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


@bp.route("/api/siesta/install-pseudos", methods=["POST"])
def api_siesta_install_pseudos():
    """Copy .psml files from cfg.psml_lib into the target directory
    so SIESTA finds them at run time.

    SIESTA discovers pseudopotentials by looking for ``<element>.psml``
    (or ``.psf`` / ``.vps``) in the CURRENT WORKING DIRECTORY where
    the .fdf is being read from -- there's no "pseudopotential search
    path" directive in SIESTA's .fdf grammar.  So after writing the
    .fdf via /api/files/write, the JS calls this endpoint to copy
    the matching .psml files into the same directory.  Without this
    extra hop SIESTA fails at startup with
    ``pseudo_read: ERROR: Pseudopotential file not found``.

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
        "copied":       ["C.psml", "Fe.psml", ...],
        "skipped":      [{"file": "H.psml", "reason": "already present"}, ...],
        "missing":      ["S"],   # elements with no .psml in lib
        "dest_dir":     "<resolved abs path>"
      }

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
    try:
        psml_dir = _resolve_path_within_roots(psml_lib_raw, require="dir")
        dest_dir = _resolve_path_within_roots(dest_dir_raw, require="dir")
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
        ext = ".pdb" if (text_in and "ATOM " in text_in[:120]) else ".xyz"
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
    # destination perms inherited from dest_dir).  Skip files that
    # are already in dest_dir (same inode = already copied).
    import shutil
    seen: set = set()
    copied: list = []
    skipped: list = []
    missing: list = []
    for raw_el in struct.elements:
        el = raw_el.capitalize()
        if el in seen:
            continue
        seen.add(el)
        src = psml_dir / f"{el}.psml"
        if not src.is_file():
            # Try a case-insensitive search before giving up.
            cand = list(psml_dir.glob(f"{el}.[pP][sS][mM][lL]"))
            if cand:
                src = cand[0]
            else:
                missing.append(el)
                continue
        dst = dest_dir / f"{el}.psml"
        try:
            if dst.exists() and src.resolve() == dst.resolve():
                skipped.append({"file": dst.name,
                                "reason": "already present (same file)"})
                continue
            shutil.copyfile(src, dst)
            copied.append(dst.name)
        except OSError as exc:
            return jsonify({"ok": False,
                            "error": f"copy failed for {el}.psml: {exc}"}), 500

    return jsonify({
        "ok":       True,
        "copied":   copied,
        "skipped":  skipped,
        "missing":  missing,
        "dest_dir": str(dest_dir),
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
        ext = ".pdb" if (text_in and "ATOM " in text_in[:120]) else ".xyz"
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
    """Analyse a structure and return suggested defaults for charge /
    spin / method, plus detected open-shell metals.

    Body (JSON)::

      {
        "structure_path":  "/abs/path/to/<.xyz|.pdb>",   # OR
        "structure_text":  "<XYZ or PDB text>"
      }

    Returns::

      {
        "ok":           True,
        "n_atoms":      <int>,
        "elements":     ["C", "Fe", ...]   # unique, sorted
        "metals":       ["Fe"]              # open-shell only; empty for organics
        "metal_hints":  [{"element": "Fe",
                          "common_oxidation_states": [...],
                          "common_spins": [{"spin": 0, "label": "Fe(II) low-spin (CO/CN)"},
                                           {"spin": 2, "label": "Fe(II) intermediate"},
                                           {"spin": 4, "label": "Fe(II) high-spin"}, ...]}],
        "suggested": {
          "pyscf":  {"charge": 0, "spin": 2, "method": "UKS",
                     "rationale": "..."},
          "siesta": {"net_charge": 0, "spin_polarized": True,
                     "spin_total": 2.0,
                     "rationale": "..."}
        },
        "warnings":  ["...", ...]   # always-applicable advisories
      }

    The UI uses this for an "Auto-detect" button that pre-fills the
    form with sensible (not necessarily correct -- always echoes the
    rationale so the user sanity-checks).
    """
    from .files import _PickerError
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
        ext = ".pdb" if (text_in and "ATOM " in text_in[:120]) else ".xyz"
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

    from molbuilder.chemistry import (detect_open_shell_metals,
                                       explain_metal_spin,
                                       total_electrons)
    # total_electrons raises KeyError on an unknown element symbol
    # (typos, bad PDB column fallback).  Catch -> 400 with the
    # parser's clear message; the previous behaviour propagated to
    # a 500 Internal Server Error with a stack trace, which leaked
    # internals to the client.
    metals = detect_open_shell_metals(struct)
    try:
        n_e = total_electrons(struct, 0)
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Build per-metal hints: enumerate the common spin values from
    # explain_metal_spin's mapping (we walk 0..6 and collect those
    # that have a hint registered).
    metal_hints = []
    for m in metals:
        candidates = []
        for s in range(0, 7):
            hint = explain_metal_spin(m, s)
            if hint:
                candidates.append({"spin": s, "label": hint})
        metal_hints.append({"element": m, "common_spins": candidates})

    # Suggested defaults.  Strategy:
    #   * Organic / no open-shell metal -> charge=0, spin=0, RKS.
    #   * Open-shell metal present -> pick the FIRST metal's most
    #     common spin state (intermediate for Fe, doublet for Cu, etc.)
    #     as the suggested default.  User can override.
    warnings: list = []
    if metals:
        # Pick a reasonable spin per metal: Fe -> 2 (intermediate, FeTPP-style);
        # Mn -> 5 (Mn(II) HS); Co -> 1 (Co(II) LS as a safe pick);
        # Cu -> 1 (Cu(II)); Ni -> 0 (Ni(II) square-planar LS); Cr -> 3
        # (Cr(III)).  Conservative -- the user MUST verify against
        # experimental data; the rationale text says so.
        DEFAULT_SPIN = {"Fe": 2, "Mn": 5, "Co": 1, "Ni": 0, "Cu": 1, "Cr": 3,
                        "V": 3, "Ti": 2, "Sc": 1}
        sug_spin = DEFAULT_SPIN.get(metals[0], 2)
        # Parity fix: if (n_e - charge) parity doesn't match sug_spin, bump.
        if (n_e % 2) != (sug_spin % 2):
            sug_spin = sug_spin + 1 if sug_spin == 0 else sug_spin - 1
            warnings.append(
                f"Adjusted suggested spin from default to {sug_spin} "
                f"to match electron-count parity (sum(Z)={n_e}, charge=0)."
            )
        sug_method = "UKS"
        rationale_py = (
            f"Detected open-shell metal {', '.join(metals)}.  "
            f"Suggesting spin={sug_spin} ({explain_metal_spin(metals[0], sug_spin) or '?'}) "
            f"with method=UKS.  Verify against your experimental data "
            f"(Mössbauer / UV-Vis / EPR) -- the right spin depends on "
            f"axial coordination, not just element identity."
        )
        rationale_si = rationale_py
        siesta_spin_total = float(sug_spin)
        siesta_polarized = True
    else:
        sug_spin = 0 if (n_e % 2 == 0) else 1
        sug_method = "RKS" if sug_spin == 0 else "UKS"
        rationale_py = (
            f"No open-shell metals detected; suggesting closed-shell "
            f"{'singlet' if sug_spin == 0 else 'doublet'} "
            f"(spin={sug_spin}, {sug_method})."
        )
        rationale_si = rationale_py
        siesta_spin_total = float(sug_spin)
        siesta_polarized = (sug_spin > 0)

    return jsonify({
        "ok":           True,
        "n_atoms":      struct.n_atoms,
        "elements":     sorted(set(e.capitalize() for e in struct.elements)),
        "metals":       metals,
        "metal_hints":  metal_hints,
        "n_electrons_neutral": n_e,
        "suggested":    {
            "pyscf":  {
                "charge":   0,
                "spin":     sug_spin,
                "method":   sug_method,
                "rationale": rationale_py,
            },
            "siesta": {
                "net_charge":     0,
                "spin_polarized": siesta_polarized,
                "spin_total":     siesta_spin_total,
                "rationale":      rationale_si,
            },
        },
        "warnings":     warnings,
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

    # Geometry-only validation at build time -- no cfg yet (the user
    # hasn't picked SIESTA vs PySCF).  Surfaces the H/heavy-ratio warn
    # for a heavy-atom skeleton before they even click Generate.
    issues = validate_geometry(struct)

    return jsonify({
        "ok": True,
        "xyz": struct.to_xyz(),
        "pdb": struct.to_pdb(),
        "n_atoms": struct.n_atoms,
        "n_residues": struct.n_residues,
        "summary": struct.summary(),
        "title": struct.title or kind,
        "elements": list(struct.elements),
        "backend_used": backend_used,
        # Tri-state H-add decision actually used (echoes the request,
        # or "auto" when not explicitly requested).  None for non-
        # nucleic builds (peptide/SMILES/name) where the kwarg
        # doesn't apply.
        "add_hydrogens_mode": h_mode_used,
        "issues": _issues_to_json(issues),
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

    return jsonify({
        "ok": True,
        "xyz": struct.to_xyz(),
        "pdb": struct.to_pdb(),
        "n_atoms": struct.n_atoms,
        "n_residues": struct.n_residues,
        "summary": struct.summary(),
        "title": struct.title or (filename or fmt),
        "elements": list(struct.elements),
        # Per-atom PDB metadata for downstream consumers (e.g. the
        # Modify tab's atom list, M2).  The fields are always present
        # on Structure (defaults filled in __post_init__) so JSON
        # callers don't need null-check branches.
        "atom_names":    list(struct.atom_names),
        "residue_ids":   list(struct.residue_ids),
        "residue_names": list(struct.residue_names),
        "chain_ids":     list(struct.chain_ids),
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

    try:
        cfg = _siesta_config_from_params(params)
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"bad parameters: {exc}"}), 400

    # Validate before render so the web layer gets a structured copy
    # of the issues.  render_fdf will validate again and write warnings
    # to stderr / raise on errors -- we keep that for CLI/library
    # callers; here we want the issues as JSON for the UI.
    issues = validate(struct, cfg)
    try:
        fdf = render_fdf(struct, cfg)
    except ValidationError as exc:
        return jsonify({
            "ok": False,
            "error": str(exc),
            "issues": _issues_to_json(exc.issues),
        }), 400
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"render failed: {exc}"}), 500

    return jsonify({
        "ok": True,
        "fdf": fdf,
        "system_label": cfg.system_label,
        "issues": _issues_to_json(issues),
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

    try:
        cfg = _pyscf_config_from_params(params)
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"bad parameters: {exc}"}), 400

    issues = validate(struct, cfg)
    try:
        script = render_script(struct, cfg)
    except ValidationError as exc:
        return jsonify({
            "ok": False,
            "error": str(exc),
            "issues": _issues_to_json(exc.issues),
        }), 400
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"render failed: {exc}"}), 500

    return jsonify({
        "ok": True,
        "script": script,
        "job_name": cfg.job_name,
        "issues": _issues_to_json(issues),
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
        "issues": _issues_to_json(validate(struct, cfg)),
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
