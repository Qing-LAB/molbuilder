"""The ``molbuilder runtime-info`` CLI: the file it reads, and the name it
writes beside it.  See cli.py::cmd_runtime_info.
"""
from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from molbuilder.cli import cli
from molbuilder.trajectory_log.emitter import MolwatchEmitter


def test_unknown_format_exits_nonzero(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("not a SIESTA or PySCF output\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p)])
    assert result.exit_code != 0
    assert "Error" in result.output or "Error" in (result.stderr or "")


class _Mol:
    """What the progress log's writer asks of a molecule: two atoms."""
    natm = 2

    def atom_coords(self, unit="Ang"):
        return np.zeros((2, 3))

    def atom_symbol(self, i):
        return "O"


@pytest.mark.parametrize("read,wrote", [
    ("job.molwatch.log", "job.runtime_info.json"),
    ("notes.txt", None),
], ids=["a-file-on-its-label", "a-name-no-role-reads-back"])
def test_the_runtime_info_is_named_beside_the_file_on_its_stem(
        read, wrote, tmp_path):
    """`project-layout.md` § 5.3, ``<base>.runtime_info.json`` -- one per
    run: named on the stem the file's own role leaves, and refused, naming
    ``--out``, for a file whose name says no role.  The progress log is
    the run's own writer's (`trajectory_log.emitter`).  THE SILENT FAILURE:
    a name cut at the last dot -- ``job.molwatch.runtime_info.json`` beside
    ``job.molwatch.log`` -- which no reader of the run reads back."""
    MolwatchEmitter(str(tmp_path / read), "job", _Mol(),
                    runtime_info={"mpi_np": 2})
    result = CliRunner().invoke(cli, ["runtime-info", str(tmp_path / read)])
    beside = sorted(p.name for p in tmp_path.iterdir() if p.name != read)
    if wrote is None:
        assert result.exit_code != 0
        assert "--out" in result.output, result.output
        assert beside == [], beside
    else:
        assert result.exit_code == 0, result.output
        assert beside == [wrote], beside
