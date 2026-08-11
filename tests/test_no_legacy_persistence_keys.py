"""Drift-guard: web/workspace.md § 9 names ``molbuilder.workspace.v1``
as the sole sessionStorage persistence key.  Pre-Phase-10 the modify
viewer wrote ``modify-state`` and canvas-state wrote
``molbuilder.structure_canvas`` as parallel mirrors; both retired.

A future contributor reintroducing either key would silently violate
the single-mirror invariant the dispatcher's restore path depends on.
"""
from pathlib import Path

_STATIC = Path(__file__).resolve().parent.parent / "molbuilder" / "web" / "static"
_DISPATCHER_DIR = _STATIC / "lib" / "workspace"

# The dispatcher is the only legitimate consumer of the canonical key;
# it owns the write + the read.  Every other module reads through ws.*.
_LEGACY_KEYS = ("modify-state", "molbuilder.structure_canvas")


def test_no_legacy_persistence_key_writes():
    offenders = []
    for path in _STATIC.rglob("*.js"):
        text = path.read_text()
        for key in _LEGACY_KEYS:
            if f'"{key}"' not in text and f"'{key}'" not in text:
                continue
            offenders.append((path.relative_to(_STATIC).as_posix(), key))
    assert not offenders, (
        f"legacy sessionStorage keys still referenced: {offenders}. "
        f"web/workspace.md § 9 names molbuilder.workspace.v1 as "
        f"the sole persistence key. Read/write goes through the workspace "
        f"dispatcher (lib/workspace/dispatcher.js); other modules must "
        f"call it through the tag-naming doors the dispatcher exposes "
        f"(persist / readState / pruneStatesAbove -- workspace.md § 5)."
    )
