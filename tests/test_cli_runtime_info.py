"""The ``molbuilder runtime-info`` CLI refuses a file no parser reads.  See
cli.py::cmd_runtime_info.
"""
from __future__ import annotations


from click.testing import CliRunner

from molbuilder.cli import cli


def test_unknown_format_exits_nonzero(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("not a SIESTA or PySCF output\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p)])
    assert result.exit_code != 0
    assert "Error" in result.output or "Error" in (result.stderr or "")
