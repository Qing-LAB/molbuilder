"""Build blueprint -- structure-construction + emitter routes.

Routes (registered with no url_prefix; each carries its own full path):

    POST /api/build/molecule   build a Structure from sequence/SMILES/name
    POST /api/build/load       load an existing XYZ / PDB into a Structure
    POST /api/build/fdf        render a SIESTA .fdf for a Structure + params
    POST /api/build/pyscf      render a PySCF script for a Structure + params

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

from dataclasses import fields
from typing import Any, Dict

from flask import Blueprint, jsonify, request

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
from molbuilder.validation import validate, validate_geometry


def _issues_to_json(issues):
    """Serialise List[Issue] for the JSON wire.  The web client reads
    `issues[].severity / message / where` to decide how to display."""
    return [{"severity": i.severity, "message": i.message, "where": i.where}
            for i in issues]


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
                from molbuilder.backends import auto_backend_name
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
    except Exception as exc:
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
    except Exception as exc:
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


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _xyz_to_structure(xyz_text: str) -> Structure:
    """Thin wrapper that delegates to Structure.from_xyz so the web
    layer doesn't carry its own parser."""
    return Structure.from_xyz(xyz_text, title="from-browser")


def _siesta_config_from_params(params: Dict[str, Any]) -> SiestaConfig:
    """Build a SiestaConfig from a dict, picking only fields it knows."""
    valid = {f.name for f in fields(SiestaConfig)}
    kwargs: Dict[str, Any] = {}
    for k, v in params.items():
        if k not in valid:
            continue
        # kgrid arrives as [a, b, c]
        if k == "kgrid" and isinstance(v, (list, tuple)) and len(v) == 3:
            kwargs[k] = (int(v[0]), int(v[1]), int(v[2]))
        # net_charge: empty string from the form means "auto-detect"
        elif k == "net_charge" and (v == "" or v is None):
            continue
        else:
            kwargs[k] = v
    return SiestaConfig(**kwargs)


def _pyscf_config_from_params(params: Dict[str, Any]) -> PySCFConfig:
    """Build a PySCFConfig from a dict, picking only fields it knows."""
    valid = {f.name for f in fields(PySCFConfig)}
    kwargs: Dict[str, Any] = {}
    for k, v in params.items():
        if k not in valid:
            continue
        # Empty string from the form means "leave default / None".
        if v == "" and k in ("solvent", "auxbasis", "dispersion"):
            kwargs[k] = None
            continue
        # JS sends "none" for "no dispersion"
        if k == "dispersion" and isinstance(v, str) and v.lower() == "none":
            kwargs[k] = None
            continue
        kwargs[k] = v
    return PySCFConfig(**kwargs)
