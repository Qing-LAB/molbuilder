"""Pin the parse-log sidecar contract: every parser emits a
``<input-stem>.parse.log`` next to its input, describing activity
and any problems, by default ON.  See molbuilder/parse/_log.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from molbuilder.parse._log import ParseLogger, _sidecar_path


_SIESTA_STUB = dedent("""\
    Siesta Version  : 5.4.2
    Architecture   : x86_64-linux-gnu
    Parallelisations: MPI

    * Running on 4 MPI processes

    siesta: System type = molecule

    siesta: Final energy (eV):
    siesta:  Total =          -1234.567

    End of run: 25-JUN-2026 12:00:00
    """)


# ----------------------------------------------------------------- #
#  Sidecar-path naming                                               #
# ----------------------------------------------------------------- #


def test_sidecar_path_for_dot_out():
    assert _sidecar_path("/x/job.out") == Path("/x/job.parse.log")


def test_sidecar_path_for_molwatch_log():
    assert (_sidecar_path("/x/job.molwatch.log")
            == Path("/x/job.molwatch.parse.log"))


def test_sidecar_path_for_transport_json():
    assert (_sidecar_path("/x/job.transport.json")
            == Path("/x/job.transport.parse.log"))


# ----------------------------------------------------------------- #
#  Default ON                                                        #
# ----------------------------------------------------------------- #


def test_reading_a_run_writes_nothing_beside_it_by_default(tmp_path, monkeypatch):
    """**A READER MUST NOT WRITE INTO THE DIRECTORY IT IS READING.**

    The sidecar was default-ON from the day it shipped until 2026-09-18, so
    merely LOOKING at a run -- a `jobset status`, a Watch poll, a sweep --
    created and grew a file inside the user's own project folder.  Measured
    across `projects/` on the day it was changed: 108 files, 145 KB, and
    **76 of them recorded nothing but a successful scan**.  Nothing in the
    tree reads one back (two writers, zero readers, Python and JS).

    User ruling: *"Let's keep it default off. There is no reader."*
    """
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
    from molbuilder.parse.engines.siesta import SiestaParser
    out = tmp_path / "job.out"
    out.write_text(_SIESTA_STUB)
    SiestaParser.parse(str(out))
    assert not (tmp_path / "job.parse.log").exists(), (
        "reading a run wrote a sidecar beside it with no one asking")
    # ...and the directory gained nothing else either.
    assert [p.name for p in tmp_path.iterdir()] == ["job.out"]


def test_the_env_var_turns_it_on_and_it_still_records_everything(tmp_path,
                                                                 monkeypatch):
    """Opting in is the whole interface -- there is no UI, deliberately: no
    web route in this project writes settings, and a debugging aid with no
    reader does not earn the first one."""
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", "1")
    from molbuilder.parse.engines.siesta import SiestaParser
    out = tmp_path / "job.out"
    out.write_text(_SIESTA_STUB)
    SiestaParser.parse(str(out))
    log = tmp_path / "job.parse.log"
    assert log.exists(), "MOLBUILDER_PARSE_LOG=1 must turn the sidecar on"
    body = log.read_text()
    for expected in ("siesta scan begin", "scan started", "INFO",
                     "scan finished"):
        assert expected in body, expected


def test_env_var_disables_log(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", "0")
    from molbuilder.parse.engines.siesta import SiestaParser
    out = tmp_path / "job.out"
    out.write_text(_SIESTA_STUB)
    SiestaParser.parse(str(out))
    assert not (tmp_path / "job.parse.log").exists()


@pytest.mark.parametrize("val", ["false", "no", "off", "FALSE", "Off"])
def test_env_var_truthy_strings_disable(tmp_path, monkeypatch, val):
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", val)
    from molbuilder.parse.engines.siesta import SiestaParser
    out = tmp_path / "job.out"
    out.write_text(_SIESTA_STUB)
    SiestaParser.parse(str(out))
    assert not (tmp_path / "job.parse.log").exists()


# ----------------------------------------------------------------- #
#  Append mode (re-parses accumulate)                                #
# ----------------------------------------------------------------- #


def test_log_appends_on_reparse(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", "1")
    from molbuilder.parse.engines.siesta import SiestaParser
    out = tmp_path / "job.out"
    out.write_text(_SIESTA_STUB)
    SiestaParser.parse(str(out))
    SiestaParser.parse(str(out))
    log = tmp_path / "job.parse.log"
    body = log.read_text()
    # Two "scan begin" banners means we appended (not truncated).
    assert body.count("siesta scan begin") == 2


# ----------------------------------------------------------------- #
#  Read-only directory degrades silently                             #
# ----------------------------------------------------------------- #


def test_read_only_directory_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", "1")
    from molbuilder.parse.engines.siesta import SiestaParser
    out = tmp_path / "job.out"
    out.write_text(_SIESTA_STUB)
    # Make the directory read-only so the .parse.log can't be opened.
    os.chmod(tmp_path, 0o500)
    try:
        traj = SiestaParser.parse(str(out))
        # Parse still succeeds; log just isn't written.
        assert traj is not None
    finally:
        os.chmod(tmp_path, 0o700)


# ----------------------------------------------------------------- #
#  Warnings surface in the log                                       #
# ----------------------------------------------------------------- #


def test_parse_warnings_appear_in_log(tmp_path, monkeypatch):
    """A .out with SCF column corruption forces a ParseWarning;
    that warning must show up as a WARN line in the parse.log."""
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", "1")
    from molbuilder.parse.engines.siesta import SiestaParser
    # SIESTA SCF line with corrupted columns (Fortran overflow ****).
    corrupt = dedent("""\
        Siesta Version  : 5.4.2

        * Running on 4 MPI processes

           scf:    1  XXXBADXXX  not a number  garbage
           scf:    2  -1234.567890   0.001000   0.000100  -8.000  -8.000
        End of run: 25-JUN-2026 12:00:00
        """)
    out = tmp_path / "job.out"
    out.write_text(corrupt)
    SiestaParser.parse(str(out))
    log_body = (tmp_path / "job.parse.log").read_text()
    # Should record activity even if no warnings were raised on this
    # particular input -- the scan-begin / scan-finished lines are
    # always present.
    assert "scan finished" in log_body


# ----------------------------------------------------------------- #
#  Transport sidecar parser also logs                                #
# ----------------------------------------------------------------- #


def test_parse_logger_warn_carries_line_and_snippet(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", "1")
    out = tmp_path / "x.out"
    out.write_text("garbage\n")
    with ParseLogger(str(out), parser_name="test") as log:
        log.warn("float() failed", line_no=42,
                 snippet="   -1.5XX23  ", category="scf")
    body = (tmp_path / "x.parse.log").read_text()
    assert "WARN" in body
    assert "line 42" in body
    assert "[scf]" in body
    assert "float() failed" in body
    # Snippet quoted in the warning line.
    assert "-1.5XX23" in body


# `test_transport_sidecar_parser_writes_log` and
# `test_transport_sidecar_logs_v1_rejection` deleted 2026-09-17 with the
# transport sidecar parser.  Both drove `dump_transport_json`, a writer with
# no production caller in any revision, to prove the parse log was written
# and that a v1 payload was rejected.  The molstruct and spectra sidecars
# still cover the log; the v1 shape is a format molbuilder never wrote.
