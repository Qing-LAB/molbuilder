"""The ``molbuilder runtime-info`` CLI refuses a file no parser reads.  See
cli.py::cmd_runtime_info.  Its output tests read a measured output copied
under another name and were retired 2026-10-04 (`process/testing.md` § 6).
"""
from __future__ import annotations


from click.testing import CliRunner

from molbuilder.cli import cli


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 6 tests here read a measured SIESTA output copied under another
# name (one with constraint lines added by hand) (`process/testing.md` § 6).


def test_unknown_format_exits_nonzero(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("not a SIESTA or PySCF output\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p)])
    assert result.exit_code != 0
    assert "Error" in result.output or "Error" in (result.stderr or "")
