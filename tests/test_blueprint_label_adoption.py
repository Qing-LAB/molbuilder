"""Every blueprint that reconstructs a Structure from a request body
must apply the body's frozen_atoms+regions labels.  Closes the class
of bug where /api/modify/* silently stripped boundary conditions
before fix landed.
"""
from pathlib import Path

_BLUEPRINTS_DIR = Path(__file__).resolve().parent.parent / "molbuilder" / "web" / "blueprints"


def test_struct_from_body_callers_apply_labels():
    offenders = []
    for path in _BLUEPRINTS_DIR.glob("*.py"):
        if path.name == "_shared.py":
            continue
        text = path.read_text()
        if "struct_from_body(" not in text:
            continue
        if "apply_labels_to_struct(" in text:
            continue
        offenders.append(path.name)
    assert not offenders, (
        f"blueprints call struct_from_body without applying labels: "
        f"{offenders}. Three-stage contract violation (sidecar-contract.md). "
        f"Add `apply_labels_to_struct(struct, body)` after struct_from_body "
        f"or document why the route legitimately discards labels."
    )
