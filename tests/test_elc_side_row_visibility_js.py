"""L2 Node test: electrode-panel #elc-side-row visibility on mode change.

The electrode-builder panel on /molbuilder has a select
``#elc-mode`` with two values:

  * ``symmetric`` (alias: ``pair``) — builds both electrodes
    around a selected pair of anchor atoms.  The side picker is
    irrelevant; the panel hides it.
  * ``single`` — builds one electrode on a chosen side of a single
    anchor.  The side picker (``#elc-side-row``) reveals so the
    user can pick ``+z`` / ``-z``.

The contract lives in ``modify/viewer.js::refreshElcReadouts``
(line 948).  It runs on every elc-* change and gates
``#elc-side-row.hidden`` on the mode value.  No prior test pins
this contract; the visibility-on-state-change pattern is the
class flagged by Agent B's audit as a generalizable gap.

Same extraction approach as ``test_hide_frozen_visibility_transition_js
.py``: pull the function source via brace-depth walk, evaluate
under Node with stubbed DOM.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/viewer.js"


def _extract_fn_source(name: str) -> str:
    src = MODULE.read_text(encoding="utf-8")
    start = src.find(f"function {name}(")
    if start < 0:
        pytest.fail(
            f"Could not find ``function {name}(`` in "
            f"{MODULE.relative_to(ROOT)}.  Either the function "
            f"was renamed or this test's parser needs updating."
        )
    open_brace = src.find("{", start)
    depth = 0
    i = open_brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    pytest.fail(f"Unbalanced braces in function {name}")


def _run_with_mode(mode_value: str) -> dict[str, object]:
    """Stub the DOM elements ``refreshElcReadouts`` touches; set
    ``#elc-mode``'s value; call the function; report what changed.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    fn = _extract_fn_source("refreshElcReadouts")

    bootstrap = f"""
        // ===== DOM stubs =====
        // Each element keeps an explicit ``hidden`` /
        // ``textContent`` / ``value`` slot — assignments mutate
        // these and we read them back at the end.
        const dom = {{
            "elc-mode":      {{ value: {json.dumps(mode_value)} }},
            "elc-gap":       {{ value: "8.0" }},
            "elc-dx":        {{ value: "0" }},
            "elc-dy":        {{ value: "0" }},
            "elc-gap-val":   {{ textContent: "" }},
            "elc-dx-val":    {{ textContent: "" }},
            "elc-dy-val":    {{ textContent: "" }},
            "elc-gap-label": {{ textContent: "" }},
            "elc-side-row":  {{ hidden: false, _id: "elc-side-row" }},
        }};
        function $(id) {{ return dom[id] || null; }}

        // ===== Extracted function =====
        {fn}

        // ===== Drive + report =====
        refreshElcReadouts();
        console.log(JSON.stringify({{
            side_row_hidden: dom["elc-side-row"].hidden,
            gap_label:       dom["elc-gap-label"].textContent,
        }}));
    """

    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- #
#  Single mode → side picker visible                                     #
# --------------------------------------------------------------------- #


def test_single_mode_reveals_side_row():
    out = _run_with_mode("single")
    assert out["side_row_hidden"] is False, (
        "single-mode electrode needs the +z/-z picker visible"
    )


def test_single_mode_gap_label_says_contact():
    """Single-mode gap is anchor-to-closest-layer; the label
    swaps to ``contact`` so the user knows the unit changed."""
    out = _run_with_mode("single")
    assert out["gap_label"] == "contact"


# --------------------------------------------------------------------- #
#  Pair / symmetric modes → side picker hidden                           #
# --------------------------------------------------------------------- #


def test_symmetric_mode_hides_side_row():
    """Symmetric mode (the default) builds both electrodes
    automatically — no per-electrode side to pick."""
    out = _run_with_mode("symmetric")
    assert out["side_row_hidden"] is True


def test_pair_mode_hides_side_row():
    """Older alias accepted by the same form: ``pair`` behaves
    like ``symmetric``.  The function gates on ``=== 'single'``
    so anything not 'single' hides the row."""
    out = _run_with_mode("pair")
    assert out["side_row_hidden"] is True


def test_pair_mode_gap_label_says_gap():
    out = _run_with_mode("pair")
    assert out["gap_label"] == "gap"


# --------------------------------------------------------------------- #
#  Defensive: unknown mode value                                         #
# --------------------------------------------------------------------- #


def test_unknown_mode_defaults_to_hide_side_row():
    """Any unknown value (e.g., the select was reset to its
    placeholder) routes through the ``!= 'single'`` branch and
    keeps the row hidden — safer than revealing a control that
    won't act sensibly."""
    out = _run_with_mode("garbage")
    assert out["side_row_hidden"] is True
