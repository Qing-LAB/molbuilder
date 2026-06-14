"""Embed handle method-set contract — L2 source-text invariant.

Demotes ``TestHandleSurface::test_handle_has_exact_documented_method_set``
out of the Playwright e2e tier
(``tests/test_mol_viewer_embed_e2e.py``).  The e2e version spun up
Chromium (~5 s with 3Dmol bootstrapping) just to enumerate
``Object.keys(handle)``.  That's a SHAPE check: which methods the
embed exports.  The shape is fixed at the source-code level — the
``return { … }`` block at the end of ``create(target, opts)`` lists
every exported method, in source order.

A source-text test on that return block delivers the same
contract pin with no browser:

  * The L2 test extracts every ``name: <symbol>,`` line from the
    handle return block.
  * Compares against the documented EXPECTED_METHODS set.
  * Catches the D1 drift the e2e test was written to catch
    (documented method silently absent) AND the inverse drift
    (an export sneaks in without being documented).

Per docs/protocols/test-strategy.md § 5 (source-text invariants):
this is the canonical use case.  The handle's shape is a code-
level contract; verifying it via grep over the source is the right
level of abstraction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module   # L2 — source-text invariant

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/mol-viewer-embed.js"


# The handle's documented method set, drawn from
# docs/protocols/embedded-viewer.md § 3.2.  Adding or removing a
# documented method requires updating BOTH the doc and this list
# (intentional friction).
EXPECTED_METHODS = sorted([
    # Data setters
    "setStructure", "appendFrames",
    # Style + overlays + projection
    "setStyle", "setAxes", "setCell", "setLabels",
    "setArrows", "setPick", "setBackground",
    "setOverlays", "setAtomStyle", "setProjection",
    # Camera
    "getCamera", "setCamera",
    # Knob bar
    "setKnobs",
    # Animation control
    "setAnimation", "playAnimation", "pauseAnimation",
    "isAnimationPlaying", "setAnimationFrame", "getAnimationFrame",
    # Read accessors
    "getAtomCount", "getElements", "getAtomCoords",
    "getPickedIndices", "setPickedIndices",
    "getStructureText",
    # Declarative-state getters (D3 symmetry — round-trip with setX)
    "getStyle", "getAxes", "getCell", "getLabels",
    "getOverlays", "getPick", "getKnobs", "getArrows",
    "getAnimation", "getBackground", "getLattice",
    # Ordered batch runner (D4)
    "applyState",
    # Output / export
    "screenshot", "exportData",
    "captureFrames", "exportAnimation",
    # Lifecycle
    "refit", "setPivot", "render", "dispose",
    # Escape hatch
    "_viewer3dmol",
])
# ``_test`` is an OBJECT (the per-instance test handle), not a
# function — present in the return block but not in the function-
# methods contract.  Listed here for completeness; not asserted in
# the function-set check below.
EXPECTED_NON_FUNCTIONS = sorted(["_test"])


def _extract_handle_keys() -> tuple[list[str], list[str]]:
    """Parse the embed module's handle return block + return
    (function-keys, non-function-keys).

    The return block lives at the end of the ``create(target, opts)``
    function; it's a literal ``return { name: value, ... }`` so we
    can extract every ``key:`` line via a tight regex.  We
    distinguish function values (``key: setName,``) from non-
    function values (``key: _buildTestHandle(state),``) by checking
    whether the value reference appears to be a function name.

    The convention for this module is that every documented method
    is a function declared in the enclosing scope and listed
    verbatim on the right-hand side.  Any non-trivial expression on
    the right (e.g. ``_buildTestHandle(state)``) signals a non-
    function export (an instance-built object).
    """
    src = MODULE.read_text()
    # Find the handle return block — it starts with ``setStructure:
    # setStructure,`` (the first entry per the embed's source) and
    # ends with the closing ``};``.
    start_match = re.search(
        r"^\s+return\s+\{\s*\n\s+setStructure:\s+setStructure,",
        src, re.MULTILINE,
    )
    if start_match is None:
        pytest.fail(
            "Could not locate the embed module's handle return block. "
            "If the format changed, update this test's parser.")
    start_idx = start_match.end()
    # The block ends at the next ``};`` after the start position.
    end_match = re.search(r"\n\s+\};", src[start_idx:])
    if end_match is None:
        pytest.fail(
            "Could not locate the embed module's handle return block "
            "terminator.")
    block = src[start_match.start():start_idx + end_match.start()]
    # Each entry: ``    name: <rhs>,`` — extract name + rhs.
    fns: list[str] = []
    nons: list[str] = []
    for m in re.finditer(
            r"^\s+([a-zA-Z_][a-zA-Z_0-9]*):\s+([^,\n]+),\s*$",
            block, re.MULTILINE):
        name, rhs = m.group(1), m.group(2).strip()
        # Function value iff the rhs is a bare identifier matching
        # the key name (the embed's convention) OR matches another
        # bare identifier (no parens, brackets, or operators).  An
        # expression like ``_buildTestHandle(state)`` is a non-fn.
        if re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", rhs):
            fns.append(name)
        else:
            nons.append(name)
    return sorted(fns), sorted(nons)


class TestHandleSurface:
    """Pure source-text contract — no browser, no embed mount."""

    def test_handle_exposes_every_documented_method(self):
        """The handle's return block must export every documented
        method.  Catches D1-class drift: a documented method
        silently removed from the export."""
        fns, _ = _extract_handle_keys()
        missing = [m for m in EXPECTED_METHODS if m not in fns]
        assert not missing, (
            f"Handle is missing documented methods: {missing}\n"
            f"Present functions: {fns}\n"
            f"This is doc-vs-code drift; either implement the "
            f"missing methods or update § 3.2 of the contract."
        )

    def test_handle_exports_no_undocumented_methods(self):
        """The handle MUST NOT export functions that aren't in the
        documented set.  Catches D1-class drift: an internal helper
        accidentally leaking into the public surface."""
        fns, _ = _extract_handle_keys()
        extras = [m for m in fns if m not in EXPECTED_METHODS]
        assert not extras, (
            f"Handle exports undocumented functions: {extras}.\n"
            f"Either remove them or document them in § 3.2."
        )

    def test_handle_non_function_exports_match_documented_set(self):
        """The ``_test`` export is an object, not a function — the
        per-instance test handle.  Catches D1 drift if another non-
        function export sneaks in."""
        _, nons = _extract_handle_keys()
        assert nons == EXPECTED_NON_FUNCTIONS, (
            f"Non-function exports drifted from documented set.\n"
            f"  Expected: {EXPECTED_NON_FUNCTIONS}\n"
            f"  Got:      {nons}\n"
            f"Document the new non-function exports in § 3.2 or "
            f"remove them.")
