"""The ``molbuilder runtime-info`` CLI: the file it reads, and the name it
writes beside it.  See cli.py::cmd_runtime_info.
"""
from __future__ import annotations

import numpy as np
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


def test_the_runtime_info_is_named_beside_the_file_on_its_stem(tmp_path):
    """`project-layout.md` § 5.3, ``<base>.runtime_info.json`` -- one per
    run: named on the stem the file's own role leaves.  The progress log is
    the run's own writer's (`trajectory_log.emitter`), named as a stage's
    run names it.  THE SILENT FAILURE: a name cut at the last dot --
    ``job_01_relax.molwatch.runtime_info.json`` beside the progress log --
    which no reader of the run reads back."""
    read = "job_01_relax.molwatch.log"
    MolwatchEmitter(str(tmp_path / read), "job", _Mol(),
                    runtime_info={"mpi_np": 2})
    result = CliRunner().invoke(cli, ["runtime-info", str(tmp_path / read)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != read) == [
        "job_01_relax.runtime_info.json"]
