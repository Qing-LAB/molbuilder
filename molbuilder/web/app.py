"""Flask UI for molbuilder.

Two JSON endpoints:

    POST /api/build  body: {"kind": "peptide|dna|rna|smiles|name",
                             "input": "<sequence-or-smiles-or-name>"}
                     returns: {"ok": True, "xyz": "...", "n_atoms": N,
                               "summary": "..."}
                     The XYZ field is what the browser feeds to 3Dmol.

    POST /api/fdf    body: {"xyz": "<xyz-text>", "params": {<Config-dict>}}
                     returns: {"ok": True, "fdf": "<fdf-text>",
                               "species": [...]}
                     The browser triggers a download via a Blob.

The page itself lives in templates/index.html and static/viewer.js.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from .. import (
    build_dna, build_from_name, build_from_smiles,
    build_peptide, build_rna,
)
from ..siesta import SiestaConfig, render_fdf
from ..pyscf import PySCFConfig, render_script
from ..structure import Structure


# Cap multipart-upload size for /api/load.  Build side only needs ~10 MB
# (a 10k-atom PDB is ~1 MB at 80 bytes/line); the watch side accepts
# trajectory log uploads up to 50 MB.  Flask's MAX_CONTENT_LENGTH is a
# single global cap on the app, so we use the larger of the two.
_MAX_UPLOAD_MB = 50


# Handle re-export differences between flask versions for ASGI/WSGI users.
_BUILDERS = {
    "peptide": build_peptide,
    "dna":     build_dna,
    "rna":     build_rna,
    "smiles":  build_from_smiles,
    "name":    build_from_name,
}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024

    # Watch routes live on a Blueprint so the watch and build halves of
    # the merged UI share one Flask instance.  See
    # web/blueprints/watch.py for the full route map (/watch page +
    # /api/watch/{formats,load,data}).
    from .blueprints.watch import bp as watch_bp
    app.register_blueprint(watch_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    # ----------------------------------------------------------------
    # Build a structure from a sequence / SMILES / name
    # ----------------------------------------------------------------
    @app.route("/api/build", methods=["POST"])
    def api_build():
        body = request.get_json(silent=True) or {}
        kind = (body.get("kind") or "").strip().lower()
        text = (body.get("input") or "").strip()
        if kind not in _BUILDERS:
            return jsonify({"ok": False,
                            "error": f"Unknown kind {kind!r}; "
                                     f"valid: {sorted(_BUILDERS)}"}), 400
        if not text:
            return jsonify({"ok": False, "error": "empty input"}), 400
        try:
            # DNA / RNA accept extra knobs (backend / form / terminal).
            if kind in ("dna", "rna"):
                kwargs = {
                    "backend":  body.get("backend",  "auto"),
                    "form":     body.get("form",     "B" if kind == "dna" else "A"),
                    "terminal": body.get("terminal", "OH"),
                    "protonate_phosphates":
                        bool(body.get("protonate_phosphates", True)),
                }
                struct = _BUILDERS[kind](text, **kwargs)
            else:
                struct = _BUILDERS[kind](text)
        except ImportError as exc:
            return jsonify({"ok": False,
                            "error": f"missing dependency: {exc}"}), 500
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return jsonify({
            "ok": True,
            "xyz": struct.to_xyz(),
            "pdb": struct.to_pdb(),
            "n_atoms": struct.n_atoms,
            "n_residues": struct.n_residues,
            "summary": struct.summary(),
            "title": struct.title or kind,
            "elements": list(struct.elements),
        })

    # ----------------------------------------------------------------
    # Render an FDF from a structure (sent as XYZ) + parameters
    # ----------------------------------------------------------------
    @app.route("/api/fdf", methods=["POST"])
    def api_fdf():
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

        # Build a Config from whatever subset of fields the browser sent.
        try:
            cfg = _config_from_params(params)
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"bad parameters: {exc}"}), 400

        try:
            fdf = render_fdf(struct, cfg)
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"render failed: {exc}"}), 500

        return jsonify({
            "ok": True,
            "fdf": fdf,
            "system_label": cfg.system_label,
        })

    # ----------------------------------------------------------------
    # Load an existing XYZ / PDB into the viewer
    # ----------------------------------------------------------------
    @app.route("/api/load", methods=["POST"])
    def api_load():
        """Accept either:
          * multipart/form-data with a single file field "file", or
          * JSON {"text": "...", "format": "xyz"|"pdb"|"auto",
                  "filename": "<optional>"}
        Returns the same JSON shape as /api/build so the front end can
        treat the result identically.
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

    # ----------------------------------------------------------------
    # Render a PySCF script from a structure (sent as XYZ) + parameters
    # ----------------------------------------------------------------
    @app.route("/api/pyscf", methods=["POST"])
    def api_pyscf():
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

        try:
            script = render_script(struct, cfg)
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"render failed: {exc}"}), 500

        return jsonify({
            "ok": True,
            "script": script,
            "job_name": cfg.job_name,
        })

    @app.route("/api/health")
    def api_health():
        return jsonify({"ok": True, "version": _molbuilder_version()})

    @app.route("/api/backends")
    def api_backends():
        from ..backends import available_backends
        return jsonify({"ok": True, "available": available_backends()})

    return app


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _molbuilder_version() -> str:
    from .. import __version__
    return __version__


def _xyz_to_structure(xyz_text: str) -> Structure:
    """Thin wrapper that delegates to Structure.from_xyz so the web
    layer doesn't carry its own parser.  Kept as a function for
    backwards compatibility with code that imported it."""
    return Structure.from_xyz(xyz_text, title="from-browser")


def _config_from_params(params: Dict[str, Any]) -> SiestaConfig:
    """Build a SiestaConfig from a dict, picking only fields it knows."""
    valid = {f.name for f in fields(SiestaConfig)}
    kwargs = {}
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
    kwargs = {}
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
