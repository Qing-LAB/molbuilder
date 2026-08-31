"""L2 Node test: the Slab panel's "Orthogonal cell" box follows the surface.

`science/junction-cell.md` § 2b — the cell shape is a real choice on fcc(111)
alone.  ASE cannot build a non-orthogonal (100) or (110) cell at all, so a
free checkbox on those surfaces offers a slab that does not exist: the box
starts unchecked, and unchecked is the one setting a (100) slab cannot be
built with, so the default request came back a 400.

`renderOrthogonalChoice` in `modify/slab-panel.js` sets and disables the box
where the surface allows one shape.  What it must NOT do is carry its own copy
of the rule — which shapes exist is `/api/modify/meta`'s `orthogonal_choices`,
and `tests/test_fcc_cell_shapes.py` holds that table to the ASE it describes.
This test therefore drives the function with a *stated* meta rather than the
real one, so a panel that ignored the server and hardcoded the planes would
still fail here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/slab-panel.js"


def _extract_fn_source(name: str) -> str:
    src = MODULE.read_text(encoding="utf-8")
    start = src.find(f"function {name}(")
    if start < 0:
        pytest.fail(
            f"Could not find ``function {name}(`` in "
            f"{MODULE.relative_to(ROOT)}.  Either the function was renamed "
            f"or this test's parser needs updating."
        )
    depth, i = 0, src.find("{", start)
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    pytest.fail(f"Unbalanced braces in function {name}")


def _run(plane: str, choices, *, box_checked: bool = False) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    fn = _extract_fn_source("renderOrthogonalChoice")
    bootstrap = f"""
        const dom = {{
            "slab-orthogonal": {{ checked: {json.dumps(box_checked)},
                                  disabled: false }},
            "slab-orthogonal-note": {{ hidden: true, textContent: "",
                classList: {{ remove() {{}}, add() {{}}, toggle() {{}} }} }},
        }};
        function $(id) {{ return dom[id] || null; }}
        function picked() {{ return {json.dumps(plane)}; }}
        const meta = {{ orthogonal_choices: {json.dumps(choices)} }};

        {fn}

        renderOrthogonalChoice();
        console.log(JSON.stringify({{
            checked:  dom["slab-orthogonal"].checked,
            disabled: dom["slab-orthogonal"].disabled,
            note:     dom["slab-orthogonal-note"].textContent,
            note_hidden: dom["slab-orthogonal-note"].hidden,
        }}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", bootstrap],
                          capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_REAL = {"100": [True], "110": [True], "111": [False, True]}


@pytest.mark.parametrize("plane", ["100", "110"])
def test_a_surface_with_one_shape_gets_it_set_and_locked(plane):
    out = _run(plane, _REAL)
    assert out["checked"] is True, (
        f"fcc({plane}) can only be built orthogonal -- leaving the box "
        f"unchecked is the 400")
    assert out["disabled"] is True, "there is no choice, so do not offer one"
    assert out["note_hidden"] is False and plane in out["note"]


def test_the_one_surface_with_a_choice_keeps_it():
    out = _run("111", _REAL)
    assert out["disabled"] is False
    assert out["note_hidden"] is True, "no note where there is a real choice"


def test_it_reads_the_server_and_does_not_hardcode_the_planes():
    """The mutation that matters: a panel with its own copy of the rule.

    Told (111) has one shape and (100) has two -- the opposite of reality --
    a panel reading the server follows; one with a hardcoded table does not.
    """
    inverted = {"100": [False, True], "111": [True]}
    assert _run("111", inverted)["disabled"] is True
    assert _run("100", inverted)["disabled"] is False


def test_an_unknown_surface_is_left_alone():
    """Says nothing rather than guessing -- the panel's habit elsewhere."""
    out = _run("311", _REAL, box_checked=True)
    assert out["disabled"] is False
    assert out["checked"] is True, "an unknown plane must not flip the box"
    assert out["note_hidden"] is True
