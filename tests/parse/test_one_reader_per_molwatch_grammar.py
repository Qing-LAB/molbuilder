"""A line format has ONE reader, in the module that owns the format.

**The drift this prevents, which already happened.** `engines/pyscf.py` kept
a private copy of the `# convergence.<key>:` grammar "to avoid coupling". The
copy was letter-first and flat-only, so a staged header (`01_coarse.<leaf>`)
read as **EMPTY** on the trajectory-view path while the molwatch-view path,
once fixed, read the same file fine (2026-08-19). The fix was
`molwatch.parse_convergence_line` — *"THE one reader… coupling to the
format's owner is the point."*

**And then it was not guarded.** Nothing asserted the copy stayed gone, so
its three neighbours in the same file — `# error:`, `# concluded:`, and the
`==== molwatch step N ====` marker — kept their private copies for another
three weeks, and `engines/siesta.py` kept a fourth copy of the
`# runtime.<key>:` grammar plus a character-identical ten-line coercion.
All four were unified on 2026-09-05; this is what keeps them that way.

**The FORMAT being shared is deliberate** — `runtime_info` owns the write
side and `siesta.py` notes that the spectra and Build writers emit
*"IDENTICAL line format"*. What must not be duplicated is the READER.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ENGINES = Path(__file__).resolve().parents[2] / "molbuilder" / "parse" / "engines"
OWNER = "molwatch.py"

#: Fragments that only appear in a regex written to match a molwatch line.
#: Keyed by the grammar, so a failure names which format was re-spelled.
GRAMMARS = {
    "# runtime.<key>:":       r"runtime\.",
    "# convergence.<key>:":   r"convergence\.",
    "# error:":               r"error:",
    "# concluded:":           r"concluded:",
    "==== molwatch step N ====": r"molwatch\s+step",
}


def _regex_literals(path: Path):
    """Every string handed to `re.compile` in one module, with its line.

    Collects string constants from ANYWHERE inside the first argument, not
    just a bare literal: `molwatch.py` builds the convergence pattern by
    concatenation (`r"...convergence\\.(" + _CONV_KEY + r"):..."`) so it
    would be invisible to a literal-only reader — and the anti-vacuity test
    below caught exactly that when this file was first written.
    """
    out = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "compile"
                and node.args):
            parts = [c.value for c in ast.walk(node.args[0])
                     if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            if parts:
                out.append((node.lineno, "".join(parts)))
    return out


def test_the_owner_still_defines_every_molwatch_grammar():
    """Anti-vacuity: the scan must find the real thing before it can police it.

    If `molwatch.py` stopped compiling these — a rename, a move — the check
    below would pass by scanning for fragments that no longer exist anywhere,
    which is the failure mode this whole file exists to prevent one level up.
    """
    owner_src = "\n".join(r for _, r in _regex_literals(ENGINES / OWNER))
    missing = [g for g, frag in GRAMMARS.items() if frag not in owner_src]
    assert not missing, (
        f"{OWNER} no longer defines the grammar(s) {missing} — either they "
        "moved (repoint this test) or the owner lost them (a defect)")


def test_no_engine_parser_re_spells_a_molwatch_grammar():
    """One reader per format, and the owner is `molwatch.py`."""
    offenders = []
    scanned = 0
    for path in sorted(ENGINES.glob("*.py")):
        if path.name == OWNER:
            continue
        scanned += 1
        for lineno, pattern in _regex_literals(path):
            for grammar, fragment in GRAMMARS.items():
                if fragment in pattern:
                    offenders.append(
                        f"  {path.name}:{lineno} re-spells {grammar!r}\n"
                        f"      {pattern!r}")

    assert scanned >= 3, (
        f"only {scanned} engine modules scanned — the glob is blind, so the "
        "assertion below would pass vacuously")
    assert not offenders, (
        "these modules compile their own regex for a line format that "
        f"`{OWNER}` owns. Two readers of one grammar drift — that is not a "
        "hypothetical here, it is the 2026-08-19 convergence-header bug. Call "
        "the owner's reader instead (`parse_convergence_line`, "
        "`parse_runtime_line`, `parse_conclusion_line`) or import its "
        "compiled pattern.\n\n" + "\n".join(offenders))


def test_the_footer_reader_puts_error_above_concluded():
    """The precedence rule lives with the grammar, not at each call site.

    A log is appended across attempts, so a `# concluded:` can follow a
    `# error:` from an earlier one. Both readers used to spell this rule out
    separately, free to disagree about which marker wins.
    """
    from molbuilder.parse.engines.molwatch import parse_conclusion_line

    out = {}
    for line in ("# error: attempt one died", "# concluded: success"):
        parse_conclusion_line(line, out)
    assert out["run_state"] == "stopped", (
        f"a later `# concluded:` un-failed a run that reported an error: {out}")
    assert out["error_message"] == "attempt one died"

    # ...and the LAST error wins, so a second attempt's message is the one
    # a reader sees.
    out2 = {}
    for line in ("# error: first", "# error: second"):
        parse_conclusion_line(line, out2)
    assert out2["error_message"] == "second", out2

    # A clean run still ends cleanly, or "error wins" would just mean
    # "nothing ever ends".
    out3 = {}
    parse_conclusion_line("# concluded: success", out3)
    assert out3["run_state"] == "ended", out3
