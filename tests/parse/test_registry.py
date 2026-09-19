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


# `JobDirParser` retired 2026-09-04 with the eleven-field summary it
# produced, leaving the DirParser registry empty -- so `parse_dir` could
# only raise, which is why six functions across three modules were called
# by name instead.  A `JobDirParser` is registered again since 2026-09-18
# (`plans/plan.md` § 5c) answering the four fields § 5.0 names, each with
# a reader.  That it ANSWERS is held in `dirs/test_rundir.py`, beside the
# parser, rather than by a registration assertion here: a membership
# check on the registry list passes on a parser that raises.


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


# ---- a parser must not claim what it cannot read ---------------------- #


def test_the_xyz_sniffer_does_not_claim_siestas_forces_file(tmp_path):
    """SIESTA's `.FA` has the SHAPE of an XYZ and is not one.

    An XYZ is `N` / comment / `ELEMENT x y z`.  SIESTA's forces file is
    `N` / `index fx fy fz` -- so the count matches, the three floats match,
    and the only thing that differs is the FIRST COLUMN: an element symbol
    against an integer index.  The sniffer's own docstring says it verifies
    *"N atom lines of `element x y z` form"*; it checked the `x y z` and
    never the element, so it claimed the file.

    Measured 2026-09-18 across `projects/`: **67 real files** were claimed
    this way -- 26 `.FA`, 21 `.FAC`, 10 `.KP`, 10 extensionless -- and
    opening one answered **HTTP 500**, because the error `parse()` then
    raised is a `RunFileError`, which is not a `ParseError` and so escaped
    the route's handler.

    Safe by measurement, not by hope: all 114 real `.xyz` files in the tree
    carry an element symbol in that column and none carries a number.

    MUTATION THIS MUST FAIL AGAINST: drop the first-column test.
    """
    from molbuilder.parse import detect
    from molbuilder.parse.errors import UnknownFormatError

    # SIX rows, not three: the sniffer eats row 1 as the "comment" and then
    # samples three more, so a 3-atom stand-in runs out of lines and is
    # refused for the WRONG reason.  The real file has 444.
    fa = tmp_path / "siesta.FA"
    fa.write_text("   6\n" + "".join(
        f"     {i}  -0.181996701E+00  -0.160907730E+00   0.365946288E+00\n"
        for i in range(1, 7)), encoding="utf-8")
    with pytest.raises(UnknownFormatError):
        detect(str(fa))

    # ...and a real trajectory in the same shape is still claimed.
    xyz = tmp_path / "job_geom_optim.xyz"
    xyz.write_text("3\nIteration 0 Energy -188.30066236\n"
                   "O       -1.4049990000   -0.0013780000    0.0000000000\n"
                   "C        0.0000000000    0.0000000000    0.0000000000\n"
                   "O        1.4049990000    0.0013780000    0.0000000000\n",
                   encoding="utf-8")
    assert detect(str(xyz)).name == "pyscf"


def test_a_parser_raises_ParseError_even_when_the_name_is_not_ours(tmp_path):
    """A PARSER RAISES `ParseError`.  Anything else escapes the web layer.

    `PySCFOutFileParser` reads companions -- the geomeTRIC log, the molwatch
    log, the stdout -- by COMPOSING their names from this file's stem, and
    `runfiles` refuses a stem that is not a legal label (§ 2.1: a dotted
    label cannot be read back out of a filename).  So `my.job_geom_optim.xyz`
    -- a perfectly good XYZ the parser CLAIMS -- raised `RunFileError`, which
    is a `ValueError`, not a `ParseError`, and `/api/watch/load`'s handler
    let it through as an HTTP 500.

    The promise is kept at the BOUNDARY, not at each compose site: patching
    the first site moved the raise from `_resolve_job_token` to
    `_read_scf_history`, and there are more.

    MUTATION THIS MUST FAIL AGAINST: drop the wrap in
    `PySCFOutFileParser.parse`.
    """
    from molbuilder.parse import detect
    from molbuilder.parse.errors import ParseError

    p = tmp_path / "my.job_geom_optim.xyz"
    p.write_text("3\nIteration 0 Energy -1.0\n"
                 "O 0.0 0.0 0.0\nC 1.0 0.0 0.0\nO 2.0 0.0 0.0\n",
                 encoding="utf-8")
    assert detect(str(p)).name == "pyscf", "still claimed -- it IS an XYZ"
    with pytest.raises(ParseError):
        detect(str(p)).parse(p)
