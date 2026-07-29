"""Third-party browser assets must retain notices in source and wheels."""
from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "molbuilder" / "web" / "static" / "vendor"


def test_vendor_inventory_lists_every_shipped_library():
    inventory = (VENDOR / "README.md").read_text(encoding="utf-8")
    for name in (
        "3Dmol.js", "gif.js", "CodeMirror", "DOMPurify", "GitGraph",
        "Marked", "Mermaid", "Plotly.js",
    ):
        assert name in inventory


def test_vendor_notices_include_complete_license_texts():
    notices = {
        "LICENSE-3Dmol.txt": "Redistribution and use in source and binary forms",
        "gif.min.js.LICENSE.txt": "Permission is hereby granted",
        "codemirror/LICENSE": "Permission is hereby granted",
        "dompurify/LICENSE": "Apache License",
        "gitgraph/LICENSE": "Permission is hereby granted",
        "marked/LICENSE": "Permission is hereby granted",
        "mermaid/LICENSE": "Permission is hereby granted",
        "LICENSE-plotly.txt": "Permission is hereby granted",
    }
    for rel, required_text in notices.items():
        assert required_text in (VENDOR / rel).read_text(encoding="utf-8")


def test_vendor_assets_and_notices_are_packaged():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = data["tool"]["setuptools"]["package-data"]["molbuilder"]
    assert "web/static/vendor/*" in patterns
    assert "web/static/vendor/*/*" in patterns
