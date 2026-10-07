"""L2 tests for molbuilder.parse — the unified parse module.

# Retired 2026-09-10: `@dataclass(frozen=True)` is the enforcement, and
# CPython refuses a non-frozen subclass of a frozen one on its own.
# A test that mutates an instance to watch Python raise tests Python.

Pins docs/model/parse.md (registry + dispatch) and
§ 3 (ParseResult discriminators).  A SIESTA output's detection and parse are
read off a run made on the road, `tests/test_siesta_flat_run_e2e.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import UnknownFormatError, detect
from molbuilder.parse.registry import _registered_file_parsers


REPO_ROOT = Path(__file__).resolve().parents[2]
# The SIESTA output's detection and parse moved to the e2e tier 2026-10-06:
# they read a run made on the road with the real SIESTA,
# `tests/test_siesta_flat_run_e2e.py` (`process/testing.md` § 6).


# Registration --------------------------------------------------------- #


def test_engine_parsers_registered():
    """All three Phase-C engine wrappers land in the FileParser
    registry at import time."""
    file_parsers = _registered_file_parsers()
    names = {p.name for p in file_parsers}
    assert "siesta" in names
    assert "pyscf" in names
    assert "molwatch" in names


# A directory is not a file: the registry holds FileParsers only, and a run
# folder is the run door's (`molbuilder.runs.folder_answer`, `model/parse.md`
# § 5).  `JobDirParser` and `parse_dir` stood here until 2026-10-04: the
# directory door read the description, which this floor must not (plan B11,
# B14).


# Detection + dispatch ------------------------------------------------- #


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


def test_detect_refuses_a_directory_by_name(tmp_path):
    """A run folder is the run door's (`model/parse.md` § 3): the registry
    says so rather than trying a file parser on a directory."""
    with pytest.raises(UnknownFormatError, match="run door"):
        detect(tmp_path)


# Result discriminators ----------------------------------------------- #


# Frozen invariant ---------------------------------------------------- #


# ---- a parser must not claim what it cannot read ---------------------- #


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 2 tests here detected a SIESTA .FA or a geomeTRIC trajectory
# typed by hand (`process/testing.md` § 6).


def test_nothing_shipped_beside_a_job_is_claimed_as_its_output(tmp_path):
    """Every attempt holds the monitor's one file, `mb_monitor.pyz` -- a zip of
    the framework modules it runs on (`runwrap.MONITOR_BUNDLE`) -- beside the
    engine's own files, and those modules QUOTE the engines' lines:
    `siesta_reader.py` names ``Begin Broyden opt. move`` in its docstring, a
    SIESTA content marker.  When the modules stood beside the deck as files
    (until 2026-09-26) the registry claimed that one as a SIESTA output and
    the Results tab listed it openable.  Neither the bundle nor any Python
    file is a run's output -- a module unpacked from it, or a person's own
    script quoting a line.

    API-level over the registry, on the real shipped bytes and sources: the
    Results listing asks exactly this of each file.

    MUTATION THIS MUST FAIL AGAINST: drop the SIESTA sniffer's ``.py`` guard.
    """
    from molbuilder.parse import detect
    from molbuilder.parse.errors import UnknownFormatError
    from molbuilder.runwrap import (MONITOR_BUNDLE, MONITOR_COMPANIONS,
                                    companion_source, monitor_bundle)

    (tmp_path / MONITOR_BUNDLE).write_bytes(monitor_bundle())
    with pytest.raises(UnknownFormatError):
        detect(str(tmp_path / MONITOR_BUNDLE))
    claimed = {}
    for name in MONITOR_COMPANIONS:
        (tmp_path / name).write_text(companion_source(name), encoding="utf-8")
        try:
            claimed[name] = detect(str(tmp_path / name)).name
        except UnknownFormatError:
            pass
    assert not claimed, claimed
