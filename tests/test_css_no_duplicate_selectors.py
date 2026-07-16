"""CSS Phase 1 invariant: no full selector appears in 2+ CSS files.

Why this test exists.  The 2026-06-12 spectrum-tab checkbox-as-bullet
bug had a simple root cause:
``.schema-field.is-advanced::before { content: "•" }`` was defined in
BOTH ``static/style.css`` AND ``static/lib/form-schema.css``.  The two
load in cascade order (form-schema first, style.css second), so the
style.css copy silently won at the browser layer.  When the fix was
made in form-schema.css alone, nothing changed in the page — the
style.css copy was still overriding it.  Wasted debugging time, and
the bug rode in for days before the user caught it.

The disease isn't checkboxes; it's the structural pattern.  Across
the codebase, **12 full selectors** were defined identically in 2+
files at the time this test landed.  Any of them could produce the
same silent-override surprise during a refactor.

The cure: a one-rule-one-home invariant.  This test parses every
``.css`` file under ``molbuilder/web/static/`` (excluding vendored
code) and asserts that no normalized selector string appears as a
rule target in more than one file.

What counts as "the same selector"?  After comment removal and
whitespace collapse, two rules with identical selector strings —
e.g. both ``.sr-only`` or both ``button.primary`` — are the same.
Per-tab overrides that scope with a parent selector (e.g.
``.modify-tab .card`` vs ``.spectra-tab .card``) are distinct strings
and are allowed.

Allowlist.  A small set of selectors (``.card``, ``.status``,
``.tagline``, etc.) are intentionally defined in multiple files
because the per-tab style sheets carry distinct visual variants
without scoping by parent class.  Each allowlist entry is a
``# TODO`` — Phase 2 (token hygiene) and Phase 3 (per-component
file reorg) will pull these into proper homes.  Adding to the
allowlist requires a TODO link to the fix plan.

Adding to or growing the allowlist without a plan is a code smell;
shrink it instead.

Phase 1 reference: docs/protocols/playwright-tests.md § A10 + the
2026-06-12 audit follow-up entry in design.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


STATIC_ROOT = (Path(__file__).resolve().parent.parent
               / "molbuilder" / "web" / "static")

# Files we don't own — third-party or generated.
EXCLUDE_PATH_SUBSTRS = ("vendor", "codemirror")

# Selectors that are CURRENTLY defined in 2+ files because the
# per-tab CSS organization predates the one-rule-one-home rule.
# Each entry documents WHY and the migration plan.
#
# DO NOT add to this list without a follow-up plan recorded next
# to the entry.  Phase 2 / Phase 3 of the CSS reorganization will
# eliminate the per-tab variants by scoping to parent selectors.
ALLOWLIST: dict[str, str] = {
    # Tab-level surface primitives — page-shell.css has the base;
    # modify/spectra each redefine with their own padding / border
    # variants.  Phase 3 will scope these as `.modify-tab .card`,
    # `.spectra-tab .card`, etc. and let page-shell's base win
    # everywhere else.
    ".card":           "phase 3: scope per-tab .card overrides under .modify-tab / .spectra-tab parent",
    ".status":         "phase 3: scope per-tab .status overrides",
    ".status.error":   "phase 3: scope per-tab status variants",
    ".status.ok":      "phase 3: scope per-tab status variants",
    ".status.warn":    "phase 3: scope per-tab status variants",
    "header .tagline": "phase 3: scope per-tab .tagline overrides",
    # (`.viewer-controls` allowlist entry removed: it's no longer a cross-file
    #  duplicate -- only spectra/style.css defines it as a rule now.)
    # Issues panel uses different DOM contracts on /spectra
    # (`.issue` children) vs the optimization tab
    # (`.issues-panel .issue-item` children).  Renaming the spectra
    # variant to `.spectra-issues-panel` is the planned fix but
    # touches the form-error rendering path — deferred to Phase 3.
    ".issues-panel":          "phase 3: rename spectra variant to .spectra-issues-panel (different DOM contract)",
    ".issues-panel[hidden]":  "phase 3: ditto",
}


def _iter_css_files() -> list[Path]:
    return sorted(
        p for p in STATIC_ROOT.rglob("*.css")
        if not any(x in str(p) for x in EXCLUDE_PATH_SUBSTRS)
    )


def _strip_comments(body: str) -> str:
    return re.sub(r"/\*.*?\*/", "", body, flags=re.S)


_RULE_RE = re.compile(r"([^{}]+?)\s*\{", re.S)
_AT_RULES = ("@media", "@supports", "@keyframes", "@font-face",
             "@import", "@charset", "@page", "@property", "@layer")


def _normalize_selector(sel: str) -> str:
    """Collapse whitespace and strip — that's enough to detect
    'these two rules target the exact same selector'."""
    return re.sub(r"\s+", " ", sel).strip()


def _collect_selector_homes() -> dict[str, list[str]]:
    """Walk every CSS file, return ``{selector: [files...]}`` for
    every normalized selector that appears as a rule target."""
    home: dict[str, set[str]] = {}
    for css in _iter_css_files():
        rel = str(css.relative_to(STATIC_ROOT))
        body = _strip_comments(css.read_text())
        for m in _RULE_RE.finditer(body):
            sel_list = m.group(1).strip()
            if not sel_list:
                continue
            # Skip @-rules (e.g., @media wraps a block whose
            # internals will be picked up by their own rule iterations).
            if sel_list.startswith(_AT_RULES):
                continue
            for sel in sel_list.split(","):
                norm = _normalize_selector(sel)
                if not norm:
                    continue
                # Must reference at least one class — element-only
                # selectors (``button``, ``input``) routinely repeat
                # across files for legitimate cascade reasons; the
                # bug class we care about is class-selector level.
                if "." not in norm:
                    continue
                home.setdefault(norm, set()).add(rel)
    return {k: sorted(v) for k, v in home.items()}


def test_no_duplicate_full_selectors():
    """The Phase 1 invariant: every CSS selector that targets at
    least one class lives in exactly one file (unless allowlisted).

    If this test fails, you have two options:
      1. Move the rule to a single canonical home and delete the
         duplicates.  This is almost always the right answer.
      2. If the duplicate is intentional (per-tab override with no
         parent scoping yet), add it to ALLOWLIST above with a
         brief WHY + the migration plan.

    Adding to ALLOWLIST without a plan is a code smell."""
    homes = _collect_selector_homes()
    dupes = {sel: files for sel, files in homes.items() if len(files) > 1}
    new_dupes = {
        sel: files for sel, files in dupes.items()
        if sel not in ALLOWLIST
    }
    if new_dupes:
        lines = [
            "Found CSS selectors defined in 2+ files (not in ALLOWLIST).",
            "Each one is a load-order-fragile silent-override hazard —",
            "the same bug class that caused the 2026-06-12 spectrum",
            "checkbox-as-bullet bug.",
            "",
            "For each entry below, either:",
            "  (a) pick the right home, delete the duplicates, OR",
            "  (b) add the selector to ALLOWLIST with a migration plan.",
            "",
        ]
        for sel, files in sorted(new_dupes.items()):
            lines.append(f"  {sel!r}")
            for f in files:
                lines.append(f"    {f}")
            lines.append("")
        pytest.fail("\n".join(lines))


def test_allowlist_entries_still_have_duplicates():
    """The ALLOWLIST is a punch list of known-duplicate selectors
    waiting for a follow-up cleanup.  If an entry no longer has
    duplicates in the codebase, it should be removed from the
    allowlist (the cleanup is done).

    This protects against bit-rot — the allowlist staying populated
    long after the dupes are gone."""
    homes = _collect_selector_homes()
    stale = []
    for sel in ALLOWLIST:
        files = homes.get(sel, [])
        if len(files) <= 1:
            stale.append((sel, files))
    if stale:
        lines = [
            "ALLOWLIST contains selectors that no longer have",
            "duplicates in the codebase (cleanup is done — remove",
            "them from ALLOWLIST):",
            "",
        ]
        for sel, files in stale:
            lines.append(f"  {sel!r}  (currently in: {files or ['none']})")
        pytest.fail("\n".join(lines))
