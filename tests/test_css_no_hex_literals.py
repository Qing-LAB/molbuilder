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
  * A custom-property DEFINITION — ``--foo: #hex`` — wherever it is
    declared.  It used to be masked only inside ``:root``, but a
    module that can be dropped into a foreign host declares its
    palette on its own class instead (``.spectrumchart { --…: #… }``,
    ``.docs-render { --dr-…: #… }``) precisely so the component still
    looks right with no page stylesheet at all — the embed-safety
    pattern of ui-contract.md § 2.  Masking only ``:root`` made those
    files unlistable, which is how three of them ended up checked by
    nothing.
  * Fallback values inside ``var(--token, #fallback)`` — those are
    defensive only-fires-if-token-undefined values.
  * Inside ``/* */`` comments — for documentation.
  * Explicitly opt-out lines with a ``/* exempt: <reason> */``
    marker on the same line — for cases like the 3Dmol canvas
    background that genuinely needs ``#ffffff`` to match the
    library's hardcoded canvas color, or a cohesive per-panel
    pastel theme that would split if pulled into tokens.

Coverage (corrected 2026-08-25).  This ran off two hand-maintained
lists — ``STRICT_FILES`` and a ``HEX_BUDGET`` of per-file caps — and
checked **nothing else**.  Phase 3 emptied the budget, but the strict
list was still enumerated by hand, so seven stylesheets were on
neither list and went unchecked; a sheet added tomorrow would have
been exempt by default, which is the opposite of an invariant.

Both lists are gone.  **Every** stylesheet under ``static/`` is
checked, ``lib/tokens.css`` excepted, so a new file is covered the day
it lands and there is no list to remember to update.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


STATIC_ROOT = (Path(__file__).resolve().parent.parent
               / "molbuilder" / "web" / "static")

EXCLUDE_PATH_SUBSTRS = ("vendor", "codemirror")

# tokens.css is its own thing — that file IS the home for hex.
TOKEN_FILE = "lib/tokens.css"


# Protected-region patterns
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_ROOT_BLOCK = re.compile(r":root\s*\{[^}]*\}", re.S)
_FALLBACK = re.compile(r"var\(\s*--[\w-]+\s*,\s*#[0-9a-fA-F]{3,8}\s*\)")
# A custom-property DEFINITION is a token definition wherever it sits --
# `:root`, or a class an embeddable declares its own palette on.  _ROOT_BLOCK
# stays for the whole-block case (it also covers non-hex noise inside).
_CUSTOM_PROP = re.compile(r"--[\w-]+\s*:[^;}]*[;}]")
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
    for pat in (_COMMENT, _ROOT_BLOCK, _FALLBACK, _CUSTOM_PROP):
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


def test_no_stylesheet_has_raw_hex():
    """EVERY stylesheet under static/ (tokens.css excepted) must have zero
    raw hex.  No allowlist: this used to consult two hand-maintained lists
    and skip anything on neither, which left seven files unchecked and made
    "add a new stylesheet" an unnoticed way out of the invariant."""
    failures: list[str] = []
    for css in _iter_css_files():
        rel = str(css.relative_to(STATIC_ROOT))
        if rel == TOKEN_FILE:
            continue
        count, samples = _count_raw_hex(css)
        if count > 0:
            failures.append(
                f"{rel}: {count} raw hex literal(s) — first samples: "
                + ", ".join(f"line {ln} {h}" for ln, h in samples)
            )
    if failures:
        pytest.fail(
            "Raw hex outside the token layer — replace with token references "
            "(var(--accent), var(--bg-card), ...), declare it as a custom "
            "property the component then reads, or mark the line "
            "`/* exempt: <reason> */`.\n\n"
            + "\n".join("  " + f for f in failures)
        )


def test_every_stylesheet_is_actually_reached():
    """The scan must SEE the files.  A path filter that silently matched
    everything would make the test above vacuously green."""
    seen = {str(p.relative_to(STATIC_ROOT)) for p in _iter_css_files()}
    assert TOKEN_FILE in seen
    for expected in ("modify/style.css", "task-setup/style.css",
                     "documents/docs-render.css", "lib/spectrumchart/_style.css"):
        assert expected in seen, f"{expected} is not being scanned"
    assert len(seen) >= 20, f"only {len(seen)} stylesheets scanned"


def test_token_file_is_the_ONLY_home_for_token_definitions():
    """**`lib/tokens.css` names every token once** (`ui-contract.md` § 2),
    module-private ones included: *"these live in the same one file,
    promoted out of scattered per-file blocks."*

    **The rule is not "no redefinition", it is "no second home"**, and the
    difference is what this test used to miss.  It exempted any name not
    already canonical, on the docstring's reasoning that *"per-file
    namespaced extensions are fine -- projects-sidebar defines its own
    --ps-* palette in its :root"*.  That state had not existed since
    2026-06-13, when those 37 tokens were promoted; what the exemption
    protected in 2026-09-02's tree was three tokens with NO module prefix
    at all -- `--page-max-width` and `--focus-ring` in `modify/style.css`,
    `--shadow-soft` in the trajectory inspector, one of them labelled
    *"inspector-only token additions"*.

    **`:root` is not component-scoped.**  It matches the document wherever
    the sheet is loaded, so an unprefixed name set by one component is set
    for every page loading it -- and two components could set it
    differently, with the last sheet in the `<head>` winning.  A palette
    that depends on link order is the fragmentation § 2 exists to prevent,
    arriving through the door marked "scoped".

    So: no `:root` token definition outside `tokens.css`, prefixed or not.
    A module-private token is welcome -- in the one file, under its
    prefix, beside the others."""
    # Read the canonical names anyway -- not to exempt anything, but so the
    # failure can say whether this is a REDEFINITION of a name that already
    # has a home, or a NEW token in the wrong place.  The two are different
    # to fix and the message should not make the reader work it out.
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
                bad.append((rel, tok_m.group(1)))
    if bad:
        pytest.fail(
            "Token(s) defined in a `:root` outside lib/tokens.css.  A "
            "`:root` block is global wherever the sheet loads, so this is "
            "a second home for a name, not a component-scoped one — and "
            "which value wins comes down to <head> order.  Move the "
            "definition into tokens.css; a module-private token belongs "
            "there too, under its prefix (ui-contract.md 2).\n\n"
            + "\n".join(
                f"  {rel}: --{name}"
                + ("   (REDEFINES the one in tokens.css)"
                   if name in canonical_names else "   (new token, wrong file)")
                for rel, name in bad)
        )
