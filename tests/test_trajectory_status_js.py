"""Unit tests for the status-label classifier in ``lib/trajectory/
core.js`` (the ``_classifyStopReason`` helper).

The classifier translates the parser's ``error_message`` strings
(SCF_NOT_CONV, propor: ERROR, ABNORMAL_TERMINATION, ...) into short
human-readable reason tags shown on the run-state info panel
("Reason: SCF non-convergence" etc.).

Tests run the JS module under Node in a minimal stub environment +
poke ``_classifyStopReason`` directly.  The module wraps the
helper in an IIFE closure; we expose it via a small ``module.exports``
shim appended below the source so node can ``require()``-style
access it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _run_node(classifier_call: str) -> str:
    """Execute the classifier under Node and return its result.

    ``classifier_call`` is the JS expression to evaluate inside the
    module's closure (e.g. ``_classifyStopReason('SCF_NOT_CONV: ...')``).
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    # Stub the bits of the module's runtime that try to look at the
    # DOM / Plotly on load.  We don't need any of them -- the
    # classifier is pure -- so a few no-op shims are enough.
    bootstrap = """
        global.window  = global;
        global.document = {
            getElementById: () => null,
            querySelector:  () => null,
            createElement:  () => ({ appendChild: () => {} }),
            addEventListener: () => {},
        };
        global.Plotly  = { newPlot: () => {}, react: () => {} };
        global.$3Dmol  = { createViewer: () => ({ /* unused here */ }) };
        global.AbortController = function () {
            this.signal = {};
            this.abort = () => {};
        };
    """
    # Inject an export shim into the IIFE so we can call the helper
    # from outside.  The module is IIFE-wrapped:
    #   (function (root) { ...; function _classifyStopReason(...) {...} })(window);
    # By appending a final line BEFORE ``})(...)`` we'd need to edit
    # the source.  Simpler: append after the IIFE a wrapper that uses
    # the same source-level trick the molbuilder JS tests use --
    # eval the module's text inside a function that lets us export
    # internal helpers via a sentinel global.  We rely on the fact
    # that the IIFE attaches everything it wants to expose to
    # ``window.molbuilder``; ``_classifyStopReason`` is NOT exposed,
    # so we have to read it from the source as text + run it directly.
    #
    # Hacky but works: we grep the function out of the source, run
    # IT alone, and call it.  The classifier has no closure deps
    # (it's a pure function over its argument).
    src = MODULE.read_text()
    # Extract the _classifyStopReason function body.  The function
    # is defined inside an IIFE, indented 4 spaces.  We slice from
    # ``function _classifyStopReason`` to the next top-level ``function``
    # definition (the one that follows -- ``function setStatus``).
    ix = src.index("function _classifyStopReason")
    end = src.index("function setStatus", ix)
    fn_source = src[ix:end].rstrip()
    full = bootstrap + "\n" + fn_source + "\n" + (
        "console.log(JSON.stringify(" + classifier_call + "));"
    )
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", full],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}\n"
            f"--- extracted fn source: ---\n{fn_source}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestClassifyStopReason:

    def test_case_insensitive(self):
        """Real .out files vary in capitalisation; classifier should
        not depend on exact case."""
        for msg in (
            "scf_not_conv",
            "SCF_NOT_CONV",
            "Scf_Not_Conv",
        ):
            out = _run_node(f"_classifyStopReason({msg!r})")
            assert out == "SCF non-convergence", f"msg={msg!r}"

    def test_empty_returns_null(self):
        out = _run_node("_classifyStopReason('')")
        assert out is None

    def test_unknown_message_falls_back_to_raw(self):
        """An unfamiliar message is shown verbatim (trimmed of leading
        / trailing whitespace) so the user still sees what the
        parser flagged."""
        out = _run_node(
            "_classifyStopReason('  some unrecognised diagnostic  ')"
        )
        assert out == "some unrecognised diagnostic"

    def test_long_unknown_truncated(self):
        """Very long unfamiliar messages get truncated with an
        ellipsis so the badge doesn't overflow."""
        long_msg = "X" * 200
        out = _run_node(f"_classifyStopReason({long_msg!r})")
        assert out.endswith("...")
        assert len(out) <= 120

    def test_priority_scf_over_stopping(self):
        """When a single line could match multiple classifiers
        (e.g. 'SCF_NOT_CONV...' AND elsewhere 'Stopping Program'),
        the more specific match wins per the order in the function
        body.  Pin the priority so a future re-order is intentional."""
        out = _run_node(
            "_classifyStopReason('SCF_NOT_CONV: ... Stopping Program from Node')"
        )
        # SCF is checked first -> wins.
        assert out == "SCF non-convergence"
