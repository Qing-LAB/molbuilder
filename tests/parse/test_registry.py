"""L2 tests for molbuilder.parse — the unified parse module.

# Retired 2026-09-10: `@dataclass(frozen=True)` is the enforcement, and
# CPython refuses a non-frozen subclass of a frozen one on its own.
# A test that mutates an instance to watch Python raise tests Python.

Pins docs/model/parse.md (registry + dispatch) and
§ 3 (ParseResult discriminators) on real fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import (
    AmbiguousFormatError,
    DirParser,
    FileParser,
    ParseResult,
    TrajectoryResult,
    UnknownFormatError,
    detect,
    parse,
    parse_dir,
    register,
)
from molbuilder.parse.engines import (
    MolwatchLogFileParser,
    PySCFOutFileParser,
    SiestaOutFileParser,
)
from molbuilder.parse.registry import (
    _registered_dir_parsers,
    _registered_file_parsers,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIESTA_FIXTURE = REPO_ROOT / "tests" / "watch" / "fixtures" / "siesta_frozen" \
    / "hemeC-stage2-run3-finished-42fr.out"
# BUILT, NOT FOUND.  The fixture was a real run under projects/, behind a
# `pytest.skip("fixture absent")` -- so it read the user's scientific record
# on this machine and SKIPPED (green, proving nothing) anywhere else.


def _need(p: Path) -> Path:
    """Assert the fixture is there.

    This used to ``pytest.skip`` on a missing file.  Every fixture it guards is
    COMMITTED under tests/ -- so absence means a broken checkout or a deleted
    file, and skipping turned that into a green run that proved nothing.  A
    missing committed fixture is a failure, loudly.
    """
    assert p.exists(), (
        f"committed fixture missing: {p}.  It is versioned with these tests; "
        f"a checkout without it is broken, not a reason to skip.")
    return p


# Registration --------------------------------------------------------- #


def test_engine_parsers_registered():
    """All three Phase-C engine wrappers land in the FileParser
    registry at import time."""
    file_parsers = _registered_file_parsers()
    names = {p.name for p in file_parsers}
    assert "siesta" in names
    assert "pyscf" in names
    assert "molwatch" in names


# `test_job_dir_parser_registered` stood here until 2026-09-04.
# `JobDirParser` was the only registered DirParser and it retired
# with the eleven-field summary it produced; `parse_dir` and the
# `DirParser` ABC remain for the next composer that needs them
# (`parse/dirs/__init__.py`).


# Detection + dispatch ------------------------------------------------- #


def test_detect_siesta_out_file():
    """detect() on a SIESTA .out picks SiestaOutFileParser."""
    cls = detect(_need(SIESTA_FIXTURE))
    assert cls is SiestaOutFileParser


def test_detect_unknown_extension_raises():
    """detect() on a non-engine file raises UnknownFormatError with
    the supported-formats list."""
    nonsense = REPO_ROOT / "README.md"
    if not nonsense.exists():
        pytest.skip("README.md absent")
    with pytest.raises(UnknownFormatError) as ei:
        detect(nonsense)
    msg = str(ei.value)
    # Hint list mentions each registered parser by label.
    assert "Supported" in msg


def test_parse_siesta_out_returns_trajectoryresult():
    """parse() on a SIESTA .out returns TrajectoryResult, not the
    legacy Trajectory dataclass."""
    result = parse(_need(SIESTA_FIXTURE))
    assert isinstance(result, TrajectoryResult)
    assert result.result_kind == "trajectory"
    assert result.source_format == "siesta"
    assert result.parser_name == "siesta"
    # Frames carry over from the legacy parser.
    assert len(result.frames) > 0


def test_parse_dir_on_non_directory_raises():
    """parse_dir refuses non-directories cleanly."""
    with pytest.raises(UnknownFormatError):
        parse_dir(_need(SIESTA_FIXTURE))


# Result discriminators ----------------------------------------------- #




# Frozen invariant ---------------------------------------------------- #
