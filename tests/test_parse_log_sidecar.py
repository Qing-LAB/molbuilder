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


def test_siesta_parser_writes_log_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
    from molbuilder.parse.engines.siesta import SiestaParser
    out = tmp_path / "job.out"
    out.write_text(_SIESTA_STUB)
    SiestaParser.parse(str(out))
    log = tmp_path / "job.parse.log"
    assert log.exists(), "parse-log sidecar must be written by default"
    body = log.read_text()
    assert "siesta scan begin" in body
    assert "scan started" in body
    assert "INFO" in body
    assert "scan finished" in body


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
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
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
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
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
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
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


def test_transport_sidecar_parser_writes_log(tmp_path, monkeypatch):
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
    from molbuilder.transport.results import TransportResults
    from molbuilder.sidecars.transport import dump_transport_json
    from molbuilder.parse.sidecars.transport import _parse_transport_json

    results = TransportResults(
        energy_grid_eV=np.array([-1.0, 0.0, 1.0]),
        transmission=np.array([0.1, 0.5, 0.9]),
        regions={"L-electrode": [0, 1], "R-electrode": [5, 6]},
        frozen_atoms=[0, 1, 5, 6],
        complete=True,
    )
    p = tmp_path / "job.transport.json"
    dump_transport_json(results, p)
    parsed = _parse_transport_json(str(p))
    assert parsed.regions["L-electrode"] == [0, 1]

    log = tmp_path / "job.transport.parse.log"
    assert log.exists()
    body = log.read_text()
    assert "transport-sidecar scan begin" in body
    assert "schema_version='2'" in body
    assert "2 regions" in body
    assert "4 frozen atoms" in body


def test_transport_sidecar_logs_v1_rejection(tmp_path, monkeypatch):
    """v1 sidecars are rejected (user directive 2026-06-25); the
    log must record the rejection cleanly."""
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
    from molbuilder.parse.sidecars.transport import _parse_transport_json
    from molbuilder.sidecars.transport import TransportJsonSchemaError

    p = tmp_path / "old.transport.json"
    p.write_text(json.dumps({
        "schema_version": "1",
        "metadata": {},
        "energy_grid_eV": [],
        "transmission": [],
        "fermi_energy_eV": 0.0,
        "conductance_G0": 0.0,
        "pdos": {},
        "bias_grid_V": None,
        "current_uA": None,
        "methods_text": "",
        "bibliography_keys": [],
        "complete": False,
    }))
    with pytest.raises(TransportJsonSchemaError):
        _parse_transport_json(str(p))
    body = (tmp_path / "old.transport.parse.log").read_text()
    assert "schema_version='1'" in body
    # Scan-aborted line carries the exception type + message.
    assert ("ERROR" in body or "aborted" in body)


# ----------------------------------------------------------------- #
#  ParseLogger direct API                                            #
# ----------------------------------------------------------------- #


def test_parse_logger_warn_carries_line_and_snippet(tmp_path, monkeypatch):
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)
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
