"""L2 Node test: the Slab panel's "Orthogonal cell" box follows the surface.

`science/junction-cell.md` § 2b — the cell shape is a real choice on fcc(111)
alone.  ASE cannot build a non-orthogonal (100) or (110) cell at all, so a
free checkbox on those surfaces offers a slab that does not exist: the box
starts unchecked, and unchecked is the one setting a (100) slab cannot be
built with, so the default request came back a 400.

`renderOrthogonalChoice` in `modify/slab-panel.js` sets and disables the box
where the surface allows one shape.  Which shapes exist is `/api/modify/meta`'s
`orthogonal_choices`, and `tests/test_fcc_cell_shapes.py` holds that table to
the ASE it describes; the panel is driven here with what the route answers,
on each plane it offers (`fcc_planes`).  A panel carrying its own copy of the
table that agrees with the server today passes too: that second copy is a
review finding (`process/code-audit.md` § 1c), not something a run can see.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import lifted, run_node

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/slab-panel.js"


def _run(plane: str, meta: dict) -> dict:
    """`renderOrthogonalChoice` on ``plane`` with the server's ``meta``,
    the box as the panel opens it -- unchecked -- and what it left."""
    fn = lifted(MODULE, "function renderOrthogonalChoice(",
                "/* A seam only continues")
    return run_node([], f"""
        const dom = {{
            "slab-orthogonal": {{ checked: false, disabled: false }},
            "slab-orthogonal-note": {{ hidden: true, textContent: "",
                classList: {{ remove() {{}}, add() {{}}, toggle() {{}} }} }},
        }};
        function $(id) {{ return dom[id] || null; }}
        function picked() {{ return {json.dumps(plane)}; }}
        const meta = {json.dumps(meta)};

        {fn}

        renderOrthogonalChoice();
        console.log(JSON.stringify({{
            checked:  dom["slab-orthogonal"].checked,
            disabled: dom["slab-orthogonal"].disabled,
            note:     dom["slab-orthogonal-note"].textContent,
            note_hidden: dom["slab-orthogonal-note"].hidden,
        }}));
    """)


@pytest.fixture(scope="module")
def meta():
    """What `/api/modify/meta` answers -- the panel's one source for the
    planes and the shapes each allows."""
    from molbuilder.web.app import create_app
    r = create_app(config={}).test_client().get("/api/modify/meta")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def test_each_offered_surface_gets_the_shapes_the_server_allows(meta):
    """Each plane the panel offers, as the server answers it: one shape --
    set, locked, and said in a note naming the plane, since leaving the box
    unchecked on fcc(100) is the 400; two -- the choice kept, no note."""
    planes = meta["fcc_planes"]
    shapes = {p: meta["orthogonal_choices"].get(p) for p in planes}
    assert any(len(v) == 1 for v in shapes.values() if v), shapes
    assert any(len(v) == 2 for v in shapes.values() if v), shapes
    for plane, allowed in shapes.items():
        out = _run(plane, meta)
        if allowed and len(allowed) == 1:
            assert out["checked"] is allowed[0], (plane, out)
            assert out["disabled"] is True, (
                f"fcc({plane}): there is no choice, so do not offer one")
            assert out["note_hidden"] is False and plane in out["note"]
        else:
            assert out["checked"] is False and out["disabled"] is False, (
                f"fcc({plane}): the box left as the person has it", out)
            assert out["note_hidden"] is True, (
                f"fcc({plane}): no note where there is a real choice")
