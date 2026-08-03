"""A module owns its CSS the way it owns its JavaScript.

MolView is sealed at ``index.js``; ``molview.css`` is that same seal in another
language, and ``molviewer-*`` is its private vocabulary.  A page stylesheet
reaching into it is the same category error as a tab reaching past ``mount`` —
and it fails the same way, late and far from its cause: the module renames a
part, and a page nobody was thinking about goes visually wrong.

This guard is the boundary from ``docs/web/css-system-plan.md`` § 1, as an
assert.  It runs in one direction only, on purpose: it checks that PAGE sheets do
not name MODULE classes.  It does not read the module sheets to judge them,
because that is exactly the reach it exists to forbid.

WHAT IT WOULD HAVE CAUGHT, and does not yet: the inventory of 2026-08-02 found
20 rules in ``results/style.css`` styling ``.inspector-card*`` — elements built
entirely in JavaScript by ``lib/inspectors/``.  That is a module whose appearance
lives in a page's sheet, so mounting an inspector anywhere else renders it
unstyled (the module loads on /results, /molbuilder AND /spectra; only /results
loads the sheet).  It is a real defect, its fix is a MODULE change — repatriating
those rules to ``lib/inspectors/`` — and until that lands the prefix stays on the
KNOWN list below with its reason.  A guard that fails on a known, recorded defect
teaches people to skip the suite; one that records it keeps the number honest and
falls to zero when the work is done.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "molbuilder" / "web" / "static"

#: Class prefixes that belong to an ESM module.  A page sheet may not name one.
MODULE_PREFIXES = (
    "molviewer-",      # MolView (lib/molview/)
    "mol-viewer-",     # the retired embed — must never come back
    "projects-",       # the projects sidebar
    "trajectory-",     # the trajectory inspector
    "sysload-",        # the system-load monitor
)

#: Prefixes a page sheet DOES still name, each with the reason and the fix.
#: Shrink this list; never grow it without recording why here.
KNOWN: dict[str, str] = {
    "inspector-": (
        "results/style.css holds the inspectors module's entire appearance — "
        ".inspector-card/-header/-title/-note/-actions/-link are built in JS by "
        "lib/inspectors/ and styled by no other sheet.  Fix is a MODULE change "
        "(repatriate to lib/inspectors/, linked by every page that mounts one); "
        "see css-system-plan.md § 3.2 + § 7."
    ),
}

#: Sheets owned by a module.  Listed so the test knows what is NOT a page sheet;
#: never read.
MODULE_SHEETS = {
    "lib/molview/molview.css",
    "lib/projects/projects-sidebar.css",
    "lib/trajectory/trajectory-inspector.css",
    "lib/inspectors/spectra.css",
    "lib/inspectors/markdown.css",
    "lib/results/bundle-handoff.css",
    "lib/system-load-monitor.css",
    "lib/app-notifications.css",
}

EXCLUDE = ("vendor", "codemirror")


def _page_sheets() -> list[Path]:
    return sorted(
        p for p in STATIC.rglob("*.css")
        if not any(x in str(p) for x in EXCLUDE)
        and str(p.relative_to(STATIC)) not in MODULE_SHEETS
    )


def _selectors(css: str) -> list[str]:
    """Top-level selector strings, comments stripped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out, depth, buf = [], 0, ""
    for ch in css:
        if ch == "{":
            if depth == 0:
                out.append(buf.strip())
            depth += 1
            buf = ""
        elif ch == "}":
            depth -= 1
            buf = ""
        else:
            buf += ch
    return [s for s in out if s and not s.startswith("@")]


@pytest.mark.parametrize("sheet", _page_sheets(), ids=lambda p: str(p.name))
def test_a_page_sheet_does_not_name_a_module_class(sheet: Path) -> None:
    """A page may not style a module's parts.

    If a page needs a module to look different, that is a mount option or a
    change made INSIDE the module — never a rule out here reaching at its names.
    """
    offenders: list[str] = []
    for sel in _selectors(sheet.read_text()):
        for name in re.findall(r"\.([A-Za-z][A-Za-z0-9_-]*)", sel):
            if name.startswith(tuple(KNOWN)):
                continue
            if name.startswith(MODULE_PREFIXES):
                offenders.append(f"{sel}   (.{name})")
    assert not offenders, (
        f"{sheet.relative_to(STATIC)} styles classes owned by an ESM module:\n  "
        + "\n  ".join(offenders)
        + "\n\nA module owns its CSS the way it owns its JS "
          "(css-system-plan.md § 1).  Delete the reach; if the module must look "
          "different, change the module."
    )


def test_the_retired_embed_stylesheet_stays_deleted() -> None:
    """``lib/viewer/mol-viewer-embed.css`` was 722 lines styling a viewer MolView
    replaced, loaded by no template and no script.  Deleted 2026-08-02.  It is
    named here so a merge cannot quietly restore it."""
    assert not (STATIC / "lib" / "viewer" / "mol-viewer-embed.css").exists()


def test_the_known_list_is_shrinking_not_growing() -> None:
    """Every entry is a recorded defect with a named fix, not a permanent carve-out.

    One entry today.  If this number goes up, the boundary is being renegotiated
    by whoever is in a hurry — which is how it was lost the first time.
    """
    assert len(KNOWN) <= 1, (
        "the module-boundary exception list grew: "
        + ", ".join(sorted(KNOWN))
        + " — each entry needs a reason and a fix recorded beside it."
    )
