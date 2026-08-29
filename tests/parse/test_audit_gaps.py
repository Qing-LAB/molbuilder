"""Half-Phase-H gap closures from the pre-Phase-H audit.

Pins:
  * Public-surface coverage: every symbol exported from
    molbuilder.parse.__init__ has at least one test reference.
  * ParseWarning is constructible + carries the documented fields
    (audit found 0 test refs to it despite being exposed publicly).
  * parse_text() is exercised end-to-end via the umbrella
    ScriptSourceTextParser (audit found parse_text in __all__
    but no direct smoke test).
  * Forbidden patterns per parse-module.md § 9:
      - P2 (TextParser no I/O) — formal lint test on parse/scripts/*
      - P3 (FileParser no subprocess) — lint test on every
        FileParser source file
      - P6 (engine-specific code only in engines/coords) — lint
        test that parse/types.py + parse/base.py + parse/registry.py
        contain NO engine name strings (siesta / pyscf / molwatch)
  * Top-level re-exports: ScriptSourceTextParser
    are importable from molbuilder.parse for the canonical "I want
    the umbrella" / "I want the bundle assembler" usage pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import molbuilder.parse as parse_module
from molbuilder.parse import (
    ParseWarning,
    ScriptResult,
    ScriptSourceTextParser,
    parse_text,
)


REPO = Path(__file__).resolve().parents[2]


# ---- Public surface coverage ------------------------------------ #


def test_every_public_export_is_importable():
    """Every symbol declared in parse.__all__ resolves to something
    not None on the module — catches accidental __all__ drift
    where a name is listed but the corresponding import was
    forgotten / typo'd."""
    for name in parse_module.__all__:
        assert hasattr(parse_module, name), (
            f"parse.__all__ declares {name!r} but the module has no "
            f"such attribute")
        assert getattr(parse_module, name) is not None, (
            f"parse.{name} resolved to None")


def test_top_level_re_exports_match_subpackage_classes():
    """ScriptSourceTextParser at parse.* is the same class as its
    sub-package canonical home.  (BundleDirParser's row retired
    2026-08-29 with the bundle parser -- calculation-to-calculation
    passing is gone; the composite CITES.)"""
    from molbuilder.parse.scripts.source import (
        ScriptSourceTextParser as _SubScript,
    )
    assert ScriptSourceTextParser is _SubScript


# ---- ParseWarning coverage -------------------------------------- #


def test_parsewarning_constructible_with_documented_fields():
    """ParseWarning is a frozen dataclass per parse-module.md § 3."""
    w = ParseWarning(
        source="/path/to/file.out",
        line_no=42,
        snippet="suspect line",
        error="parse failure",
        category="scf-format",
    )
    assert w.source == "/path/to/file.out"
    assert w.line_no == 42
    assert w.snippet == "suspect line"
    assert w.error == "parse failure"
    assert w.category == "scf-format"


def test_parsewarning_is_frozen():
    """ParseWarning is frozen — mutation raises."""
    from dataclasses import FrozenInstanceError
    w = ParseWarning(source="x", line_no=None, snippet=None,
                     error="x", category="x")
    with pytest.raises(FrozenInstanceError):
        w.error = "tampered"   # noqa


def test_parsewarning_allows_none_for_line_no_and_snippet():
    """line_no and snippet are Optional per the dataclass — None
    must be accepted (e.g., parser-level errors without a specific
    source line)."""
    w = ParseWarning(source="<text>", line_no=None, snippet=None,
                     error="malformed payload", category="json")
    assert w.line_no is None
    assert w.snippet is None


# ---- parse_text smoke test -------------------------------------- #


def test_parse_text_dispatches_to_text_parser_class():
    """parse_text(text, parser=ScriptSourceTextParser) routes to
    the class's .parse() method.  Smoke-test for the public API."""
    fdf_minimal = "SystemLabel test\nNumberOfAtoms 0\n"
    result = parse_text(fdf_minimal, parser=ScriptSourceTextParser)
    assert isinstance(result, ScriptResult)
    assert result.parser_name == "fdf-script-source"
    # No script-contract blocks → all per-block fields None.
    assert result.header is None
    assert result.atom_metadata is None


# ---- Forbidden-pattern lint tests (parse-module.md § 9) -------- #


_PARSE_DIR = REPO / "molbuilder" / "parse"


def _glob_py(rel_subdir: str) -> list[Path]:
    """Walk a parse/ sub-directory returning every .py except
    __init__/__pycache__/_helpers."""
    out: list[Path] = []
    for p in (_PARSE_DIR / rel_subdir).rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        out.append(p)
    return out


def test_forbidden_p2_textparsers_do_no_io():
    """parse-module.md § 9 forbidden #2: TextParsers do NO I/O.
    They take a string in memory; reading the file is the caller's
    job.  Lint every parse/scripts/*.py for I/O tokens."""
    forbidden = (
        "read_text", "read_bytes", "open(", ".read()",
        "Path(", "with open",
    )
    for p in _glob_py("scripts"):
        if p.name in ("__init__.py", "_helpers.py", "markers.py"):
            continue
        text = p.read_text()
        for token in forbidden:
            assert token not in text, (
                f"{p.relative_to(_PARSE_DIR)} contains forbidden "
                f"I/O token {token!r} — TextParsers operate on "
                f"text in memory only (parse-module.md § 9 #2)")


def test_forbidden_p3_fileparsers_no_subprocess_or_network():
    """parse-module.md § 9 forbidden #3: FileParsers do NO
    subprocess / network / threads.  Lint parse/engines/* +
    parse/sidecars/* + parse/coords/* for the forbidden tokens."""
    forbidden = (
        "import subprocess", "from subprocess",
        "import socket", "from socket",
        "import urllib", "from urllib",
        "import requests", "from requests",
        "import threading", "from threading",
        "Thread(", "Popen(", ".system(",
    )
    file_parser_dirs = ("engines", "sidecars", "coords")
    for subdir in file_parser_dirs:
        for p in _glob_py(subdir):
            if p.name in ("__init__.py", "_helpers.py"):
                continue
            text = p.read_text()
            for token in forbidden:
                assert token not in text, (
                    f"{p.relative_to(_PARSE_DIR)} contains forbidden "
                    f"token {token!r} — FileParsers must do no "
                    f"subprocess/network/threads (parse-module.md § 9 #3)")


def test_forbidden_p6_no_engine_specific_code_in_core():
    """parse-module.md § 9 forbidden #7 (was #6 in earlier drafts):
    engine-specific code only in parse/engines/ and parse/coords/.
    The core layer (parse/types.py, parse/base.py, parse/registry.py,
    parse/errors.py) must NOT mention engine names — that's a sign
    an engine concept leaked into the framework."""
    core_files = ("types.py", "base.py", "registry.py", "errors.py")
    # Lower-case word boundaries; matches engine names regardless
    # of case.  Allowed in DOCSTRINGS (which mention examples) so
    # we scan for the names appearing in CODE — heuristic: any
    # occurrence outside a triple-quoted string or # comment.
    import re as _re
    engine_names = (r"\bsiesta\b", r"\bpyscf\b", r"\bmolwatch\b")
    for fname in core_files:
        p = _PARSE_DIR / fname
        text = p.read_text()
        # Strip both kinds of strings + comments cheaply to lower
        # false-positive rate.  This is a heuristic; if it ever
        # flags a docstring legitimately mentioning an engine,
        # rewrite the docstring.
        stripped = _re.sub(r'""".*?"""', "", text, flags=_re.S)
        stripped = _re.sub(r"'''.*?'''", "", stripped, flags=_re.S)
        stripped = _re.sub(r"#.*$", "", stripped, flags=_re.M)
        for pat in engine_names:
            assert not _re.search(pat, stripped, _re.I), (
                f"parse/{fname} contains engine name matching "
                f"{pat} OUTSIDE docstrings/comments — engine-specific "
                f"concepts must live in parse/engines/ or "
                f"parse/coords/ (parse-module.md § 9 #7)")
