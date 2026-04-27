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

import io
from dataclasses import asdict, fields
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from .. import (
    build_dna, build_from_name, build_from_smiles,
    build_peptide, build_rna,
)
from ..siesta import Config, render_fdf
from ..structure import Structure


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
    """Lightweight XYZ parser; we wrote the XYZ ourselves so the format
    is well-defined."""
    lines = xyz_text.splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ too short")
    n = int(lines[0].strip())
    atoms = []
    for raw in lines[2:2 + n]:
        parts = raw.split()
        if len(parts) < 4:
            continue
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    if len(atoms) != n:
        raise ValueError(f"expected {n} atoms, got {len(atoms)}")

    import numpy as np
    return Structure(
        elements=[a[0] for a in atoms],
        positions=np.array([[a[1], a[2], a[3]] for a in atoms]),
        title=(lines[1].strip() or "from-browser"),
    )


def _config_from_params(params: Dict[str, Any]) -> Config:
    """Build a Config from a dict, picking only fields it knows about."""
    valid = {f.name for f in fields(Config)}
    kwargs = {}
    for k, v in params.items():
        if k not in valid:
            continue
        # kgrid arrives as [a, b, c]
        if k == "kgrid" and isinstance(v, (list, tuple)) and len(v) == 3:
            kwargs[k] = (int(v[0]), int(v[1]), int(v[2]))
        else:
            kwargs[k] = v
    return Config(**kwargs)
