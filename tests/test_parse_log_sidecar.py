"""Pin the parse-log sidecar contract: a parser reading a run's output
writes its log beside it -- ``<base>.parse.log``, or
``<base>.molwatch.parse.log`` for a progress log (`project-layout.md`
§ 5.3) -- describing activity and any problems, when
``MOLBUILDER_PARSE_LOG`` turns it on.  See molbuilder/parse/_log.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse._log import ParseLogger, _sidecar_path


# ----------------------------------------------------------------- #
#  Sidecar-path naming                                               #
# ----------------------------------------------------------------- #


@pytest.mark.parametrize("read,log", [
    ("job.out", "job.parse.log"),
    ("job_01_coarse.out", "job_01_coarse.parse.log"),
    ("job.molwatch.log", "job.molwatch.parse.log"),
    ("notes.txt", None),
], ids=["an-output", "a-stages-output", "a-progress-log",
        "a-name-no-row-reads-back"])
def test_the_parse_log_is_named_by_the_layouts_rows(read, log):
    """`project-layout.md` § 5.3 lists two parse logs, ``<base>.parse.log``
    and ``<base>.molwatch.parse.log``, on the stem the file read leaves; a
    file whose name says no role gets none.  THE SILENT FAILURE: a log named
    by cutting the last dot -- ``notes.parse.log``,
    ``job.transport.parse.log`` -- a name the layout does not list, written
    into the folder being read."""
    assert _sidecar_path(f"/x/{read}") == (
        None if log is None else Path(f"/x/{log}"))


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
