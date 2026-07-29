"""Module-init contract — L2 source-text invariant.

Demotes ``test_runtime_modules_registered_on_modify`` from the
Playwright e2e tier (``tests/test_molbuilder_e2e.py``).  The e2e
test loaded ``/molbuilder`` in Chromium just to read
``window.molbuilder.runtime.listRegistered()`` and assert that
each named module is present.  That's a pure shape check; the
underlying contract is that each module's source file contains a
``runtime.register("name", api)`` call at top-level scope.

A source-text grep over ``lib/**/*.js`` catches the same drift —
if a module is renamed, the matching call disappears from the
source.  No browser, no script load, no async wait.

Per docs/process/testing.md (source-text invariants):
this is the canonical use case.  The module-init contract is
expressed in code as ``runtime.register(NAME, ...)``; verifying
its presence by reading the source is the right level of
abstraction.

Phase 9 (2026-06-13) note: ``"selection.store"`` and
``"structure.canvas"`` registrations were both deleted (their
dispatcher-internal singletons no longer go through the runtime
registry; the dispatcher owns them directly).  This test pins
the survivors.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
# Scan the entire script tree, not just lib/.  Per-tab modules
# (e.g. modify/viewer.js, modify/selection-bootstrap.js) live
# alongside lib/ and also call runtime.register().  Track B retired
# ``modify.handle`` (the concealed MolView module now owns the embed
# handle via ``molview.data.attachViewHandle`` and registers
# ``molview.data``); consumers whenReady("molview.data") instead.
STATIC = ROOT / "molbuilder/web/static"


# Matches ``runtime.register("name", ...)`` across single-line
# and multi-line forms.  ``\s*`` after the opening paren absorbs
# the line break + indent in the common multi-line form:
#
#     root.molbuilder.runtime.register(
#         "selection.panel", panel);
_REGISTER_CALL_RE = re.compile(
    r'runtime\.register\(\s*["\']([^"\']+)["\']',
)


# Per design.md "Module init contract" — the module names every
# consumer may ``whenReady("X")`` on.  Updating this list requires
# editing design.md too (intentional friction so a rename surfaces
# at review time).
# Track B retired ``modify.handle``: the concealed MolView module now owns the
# embed handle via ``molview.data.attachViewHandle`` (a method, not a runtime
# registration), so there is no ``.handle`` registration to require here.  The
# data model itself IS registered (``_runtime().register("molview.data")`` in
# data-model.js) but through the ``_runtime()`` helper form this scanner's regex
# doesn't match; it isn't a whenReady dependency of any consumer, so it's not in
# this required set.
_REQUIRED_REGISTRATIONS = sorted([
    "projects",
    "selection.panel",
    "selection.viewerAdapter",
])


@pytest.fixture(scope="module")
def discovered_registrations() -> set[str]:
    """Scan every .js file under ``static/`` and return the set of
    names passed to ``runtime.register(...)``."""
    names: set[str] = set()
    for path in STATIC.rglob("*.js"):
        # Skip vendored / minified bundles.
        rel = path.relative_to(STATIC).as_posix()
        if rel.startswith("vendor/") or rel.endswith(".min.js"):
            continue
        src = path.read_text()
        for match in _REGISTER_CALL_RE.finditer(src):
            names.add(match.group(1))
    return names


@pytest.mark.parametrize("required", _REQUIRED_REGISTRATIONS)
def test_required_module_calls_runtime_register(
    discovered_registrations, required,
):
    """The named module MUST call ``runtime.register(name, api)``
    at its top-level scope.  Catches a rename that drops the
    registration (e.g., the module gets refactored and someone
    forgets to wire ``whenReady``).

    Per design.md "Module init contract"."""
    assert required in discovered_registrations, (
        f"No file under static/ calls "
        f"runtime.register({required!r}, ...) — the module "
        f"either renamed itself without updating design.md or "
        f"dropped the registration entirely.  Discovered "
        f"registrations: {sorted(discovered_registrations)}"
    )
