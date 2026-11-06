"""Pin the parse-log sidecar contract: a parser reading a run's output
writes its log beside it -- ``<base>.parse.log``, or
``<base>.molwatch.parse.log`` for a progress log (`project-layout.md`
§ 5.3) -- describing activity and any problems, when
``MOLBUILDER_PARSE_LOG`` turns it on.  See molbuilder/parse/_log.py.
"""
from __future__ import annotations

from molbuilder.parse._log import ParseLogger


# ----------------------------------------------------------------- #
#  Sidecar-path naming                                               #
# ----------------------------------------------------------------- #


def test_a_progress_logs_parse_log_is_its_own_row(tmp_path, monkeypatch):
    """`project-layout.md` § 5.3 lists two parse logs: ``<base>.parse.log``,
    and the progress log's own, ``<base>.molwatch.parse.log`` -- the one
    name the parse log's writer decides (the rest are `named_beside`'s).
    THE SILENT FAILURE: a progress log's parse log named as an output's,
    ``job_01_relax.parse.log``, beside the output's own."""
    monkeypatch.setenv("MOLBUILDER_PARSE_LOG", "1")
    read = tmp_path / "job_01_relax.molwatch.log"
    read.write_text("x\n")
    with ParseLogger(str(read), parser_name="test") as log:
        log.warn("it was read", line_no=1)
    assert (tmp_path / "job_01_relax.molwatch.parse.log").is_file()
    assert not (tmp_path / "job_01_relax.parse.log").exists()


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
