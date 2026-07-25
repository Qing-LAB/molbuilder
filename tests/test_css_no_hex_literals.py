"""CSS Phase 2 invariant: no raw hex literals outside the token
layer.

Why this test exists.  Phase 1 killed duplicate selectors and
prevented the silent-override bug class.  Phase 2 tackles the
companion drift: every time an author writes ``color: #4a9eff``
instead of ``color: var(--accent)``, two things go wrong:

  1. The exact value drifts from the canonical token (``--accent``
     is ``#6ba6ff``, not ``#4a9eff`` — that one's already off by 7
     points per channel).
  2. A theme change requires touching N files instead of one
     ``tokens.css`` edit.

The invariant: every color in non-token CSS files goes through a
token reference (``var(--accent)``, ``var(--bg-card)``, ...).  Raw
hex literals are allowed ONLY in:

  * ``lib/tokens.css`` — that file IS the token home.
  * ``:root { --foo: #hex; }`` blocks — those DEFINE per-file
    namespaced tokens (e.g. projects-sidebar's ``--ps-*`` palette).
  * Fallback values inside ``var(--token, #fallback)`` — those are
    defensive only-fires-if-token-undefined values.
  * Inside ``/* */`` comments — for documentation.
  * Explicitly opt-out lines with a ``/* exempt: <reason> */``
    marker on the same line — for cases like the 3Dmol canvas
    background that genuinely needs ``#ffffff`` to match the
    library's hardcoded canvas color, or a cohesive per-panel
    pastel theme that would split if pulled into tokens.

Budget model.  Phase 2 cleanup of the small files (style.css,
modify/style.css, results/style.css, lib/page-shell.css, etc.)
ships with this commit.  The two big component files —
``lib/molview/selection/selection-panel.css`` (97 raw hex) and
``lib/projects/projects-sidebar.css`` (66 raw hex) — need a deeper Phase 3
migration that introduces ``--sp-*`` / promotes ``--ps-*`` into
tokens.css.  Until that lands, this test caps each allowlisted
file at its current count: adding NEW raw hex makes the test fail.

To extend an allowlist budget, you must first explain why the
file truly needs more raw hex than it has now.  The bias is to
shrink the budget over time, not grow it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


STATIC_ROOT = (Path(__file__).resolve().parent.parent
               / "molbuilder" / "web" / "static")

EXCLUDE_PATH_SUBSTRS = ("vendor", "codemirror")

# Files where raw hex is allowed up to a documented cap.  Each
# entry is (max_allowed, why + phase-N-plan).  Shrink the cap
# whenever a cleanup commit reduces the count; do NOT grow without
# a plan.  Goal: drive every entry to zero, then promote to
# STRICT_FILES below.
HEX_BUDGET: dict[str, tuple[int, str]] = {}   # empty 2026-07 -- every sheet is STRICT

# Files allowed ZERO raw hex (anything outside protected regions
# OR not on an /* exempt: ... */ line is a bug).  Phase 3
# (2026-06-13) moved selection-panel, projects-sidebar, tabs,
# mol-viewer-embed, measurement-chip, and spectra from BUDGET into
# STRICT after promoting their palettes (--ps-*, --sp-*) into
# tokens.css and replacing raw hex with token references.
STRICT_FILES: set[str] = {
    # ("style.css" removed 2026-07: the file was deleted in the #58 split.)
    "results/style.css",
    "modify/style.css",
    "transport/style.css",
    "structure-optimization/style.css",
    "spectra/style.css",
    "lib/page-shell.css",
    "lib/form-schema.css",
    "lib/form-components.css",
    "lib/projects/projects-sidebar.css",
    "lib/molview/selection/selection-panel.css",
    "lib/tabs.css",
    "lib/viewer/mol-viewer-embed.css",
    "lib/molview/selection/measurement-chip.css",
    "lib/molview/fused-layout.css",
    "lib/inspectors/markdown.css",
    "lib/inspectors/spectra.css",
    "lib/trajectory/trajectory-inspector.css",
    "lib/system-load-monitor.css",
    "lib/results/bundle-handoff.css",
    "lib/app-notifications.css",
}

# tokens.css is its own thing — that file IS the home for hex.
TOKEN_FILE = "lib/tokens.css"


# Protected-region patterns
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_ROOT_BLOCK = re.compile(r":root\s*\{[^}]*\}", re.S)
_FALLBACK = re.compile(r"var\(\s*--[\w-]+\s*,\s*#[0-9a-fA-F]{3,8}\s*\)")
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# A line containing `/* exempt: ... */` opts every hex on that line out.
_EXEMPT_LINE = re.compile(r"/\*\s*exempt[:\s]")


def _iter_css_files() -> list[Path]:
    return sorted(
        p for p in STATIC_ROOT.rglob("*.css")
        if not any(x in str(p) for x in EXCLUDE_PATH_SUBSTRS)
    )


def _count_raw_hex(path: Path) -> tuple[int, list[tuple[int, str]]]:
    """Return (count, sample_locations) of raw hex literals — those
    NOT inside a comment, a :root token-definition block, a
    var(--x, #fallback) fallback, or on a line marked
    `/* exempt: ... */`."""
    body = path.read_text()
    n = len(body)
    masked = bytearray(n)
    for pat in (_COMMENT, _ROOT_BLOCK, _FALLBACK):
        for m in pat.finditer(body):
            for i in range(m.start(), m.end()):
                masked[i] = 1

    # Determine exempt lines
    lines = body.splitlines(keepends=True)
    line_starts = [0]
    for ln in lines[:-1]:
        line_starts.append(line_starts[-1] + len(ln))
    exempt_lines: set[int] = set()
    for i, ln in enumerate(lines, 1):
        if _EXEMPT_LINE.search(ln):
            exempt_lines.add(i)

    def line_of(pos: int) -> int:
        # Binary search would be faster but list is short
        for i, start in enumerate(line_starts):
            if start > pos:
                return i
        return len(line_starts)

    raw: list[tuple[int, str]] = []
    for m in _HEX.finditer(body):
        if masked[m.start()] == 1:
            continue
        line_no = line_of(m.start())
        if line_no in exempt_lines:
            continue
        raw.append((line_no, m.group(0)))
    return len(raw), raw[:5]


def test_strict_files_have_no_raw_hex():
    """Files in STRICT_FILES must have ZERO raw hex.  These are
    Phase 2 successes; adding a raw hex here is a regression."""
    failures: list[str] = []
    for css in _iter_css_files():
        rel = str(css.relative_to(STATIC_ROOT))
        if rel not in STRICT_FILES:
            continue
        count, samples = _count_raw_hex(css)
        if count > 0:
            failures.append(
                f"{rel}: {count} raw hex literal(s) — first samples: "
                + ", ".join(f"line {ln} {h}" for ln, h in samples)
            )
    if failures:
        pytest.fail(
            "Raw hex in strict file(s) — replace with token references "
            "(var(--accent), var(--bg-card), etc.) or wrap with a "
            "`/* exempt: <reason> */` comment on the same line.\n\n"
            + "\n".join("  " + f for f in failures)
        )


def test_budget_files_within_cap():
    """Files in HEX_BUDGET have a documented cap.  Each cleanup
    commit should LOWER the cap; new code may NOT push it higher."""
    failures: list[str] = []
    for css in _iter_css_files():
        rel = str(css.relative_to(STATIC_ROOT))
        if rel not in HEX_BUDGET:
            continue
        cap, plan = HEX_BUDGET[rel]
        count, samples = _count_raw_hex(css)
        if count > cap:
            failures.append(
                f"{rel}: {count} raw hex (cap = {cap}). Plan: {plan}\n"
                f"      First samples: "
                + ", ".join(f"line {ln} {h}" for ln, h in samples)
            )
    if failures:
        pytest.fail(
            "Raw hex count exceeded budget in CSS file(s).  Either:\n"
            "  (a) replace the new hex with a token reference, OR\n"
            "  (b) lower the cap if your edit reduced the count, OR\n"
            "  (c) if you genuinely need more hex than the cap allows,\n"
            "      explain why in the HEX_BUDGET entry — and consider\n"
            "      whether the new hex belongs in tokens.css instead.\n\n"
            + "\n".join("  " + f for f in failures)
        )


def test_budget_files_at_cap_or_below():
    """The HEX_BUDGET caps should track the actual count and shrink
    over time.  If a file has fewer raw hex than its budget cap,
    lower the cap so the next regression fails immediately instead
    of slipping under the old ceiling."""
    too_loose: list[str] = []
    for css in _iter_css_files():
        rel = str(css.relative_to(STATIC_ROOT))
        if rel not in HEX_BUDGET:
            continue
        cap, _ = HEX_BUDGET[rel]
        count, _ = _count_raw_hex(css)
        # Allow exactly the cap or one below (small drift OK).  But
        # if the actual count is way below, the cap should shrink.
        if cap > count + 2:
            too_loose.append(
                f"{rel}: cap = {cap}, actual = {count}.  Lower the "
                f"cap to {count} in tests/test_css_no_hex_literals.py"
            )
    if too_loose:
        pytest.fail(
            "HEX_BUDGET cap is looser than necessary.  Cleanup work "
            "lowered the count — lower the cap so future regressions "
            "are caught immediately.\n\n"
            + "\n".join("  " + f for f in too_loose)
        )


def test_token_file_is_unique_home_for_token_definitions():
    """tokens.css owns the canonical global tokens (--bg-page,
    --accent, --text-primary, etc.).  Per-file namespaced
    extensions are fine — projects-sidebar defines its own
    --ps-* palette in its :root, and that's a legitimate
    component-scoped token system.

    This test guards against the specific bug where a global
    token name gets REDEFINED outside tokens.css, fragmenting the
    design palette."""
    # Read the canonical token names from tokens.css itself.
    token_path = STATIC_ROOT / TOKEN_FILE
    canonical_names = set(
        re.findall(r"--([a-zA-Z][\w-]*)\s*:", token_path.read_text())
    )
    if not canonical_names:
        pytest.fail("Could not parse any tokens from lib/tokens.css — "
                    "the regex needs an update.")

    bad: list[tuple[str, str]] = []
    for css in _iter_css_files():
        rel = str(css.relative_to(STATIC_ROOT))
        if rel == TOKEN_FILE:
            continue
        body = _COMMENT.sub("", css.read_text())
        # Find every --token: in :root blocks of OTHER files
        for root_m in _ROOT_BLOCK.finditer(body):
            for tok_m in re.finditer(r"--([a-zA-Z][\w-]*)\s*:",
                                     root_m.group(0)):
                name = tok_m.group(1)
                if name in canonical_names:
                    bad.append((rel, name))
    if bad:
        pytest.fail(
            "Canonical token name(s) redefined outside lib/tokens.css.  "
            "Move the definition into tokens.css so there's one source "
            "of truth.  Per-file namespaced tokens (e.g. --ps-* in "
            "projects-sidebar.css) are fine — only canonical names "
            "trigger this check.\n\n"
            + "\n".join(f"  {rel}: --{name}" for rel, name in bad)
        )
