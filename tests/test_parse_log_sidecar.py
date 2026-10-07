"""Pin the parse-log sidecar contract: every parser emits a
``<input-stem>.parse.log`` next to its input, describing activity
and any problems, when ``MOLBUILDER_PARSE_LOG`` turns it on.  See
molbuilder/parse/_log.py.
"""
from __future__ import annotations

from pathlib import Path


from molbuilder.parse._log import ParseLogger, _sidecar_path


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
#  ParseLogger.warn                                                  #
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
