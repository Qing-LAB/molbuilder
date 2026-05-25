"""End-to-end coverage of the ``molbuilder run`` subcommand.

Uses ``CliRunner`` to invoke the click command surface; touches the
real filesystem (the wrapper-emit step IS the unit being tested).
``set_capabilities`` injects a synthetic Capabilities so no real
``conda env list`` runs during these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from molbuilder import cli, diagnostics
from molbuilder.diagnostics import Capabilities


@pytest.fixture(autouse=True)
def _inject_caps():
    """Bind a synthetic Capabilities for every test in this file.

    Avoids running the real ``detect()`` (which spawns ``conda env
    list``) on every test.  The conftest-level autouse fixture resets
    the singleton afterwards.
    """
    diagnostics.set_capabilities(Capabilities(
        runtime_config = {},
        conda_binary   = "/usr/bin/conda",
        conda_envs     = frozenset({"molbuilder-siesta", "molbuilder-pySCF"}),
    ))
    yield


# --------------------------------------------------------------------- #
#  Happy paths                                                          #
# --------------------------------------------------------------------- #


def test_run_emits_wrapper_for_fdf(tmp_path):
    fdf = tmp_path / "my-job.fdf"
    fdf.write_text("# fake fdf\n")
    res = CliRunner().invoke(cli.cli, ["run", str(fdf)])
    assert res.exit_code == 0, res.output
    wrapper = tmp_path / "my-job.run.sh"
    assert wrapper.is_file()
    txt = wrapper.read_text()
    # 2026-05-24: launcher is probe-resolved at runtime
    # (``exec $_launch_cmd my-job.fdf > my-job.out``).
    assert "my-job.fdf > my-job.out" in txt
    # 2026-05-23: wrapper switched from ``conda run -n`` to the
    # source+activate hybrid; the env name still appears in the
    # ``conda activate`` line (see molbuilder/runwrap.py).
    assert "conda activate molbuilder-siesta" in txt


def test_run_emits_wrapper_for_py(tmp_path):
    py = tmp_path / "my-job.py"
    py.write_text("# fake py\n")
    res = CliRunner().invoke(cli.cli, ["run", str(py)])
    assert res.exit_code == 0
    txt = (tmp_path / "my-job.run.sh").read_text()
    assert "python my-job.py" in txt
    assert "conda activate molbuilder-pySCF" in txt


def test_run_passes_mpi_np_for_siesta(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("# fake\n")
    res = CliRunner().invoke(cli.cli, ["run", str(fdf), "--np", "8"])
    assert res.exit_code == 0
    # The probe block's MPI branch sets _launch_cmd to mpirun -np 8 siesta.
    assert '_launch_cmd="mpirun -np 8 siesta"' in (tmp_path / "x.run.sh").read_text()


def test_run_env_override(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("# fake\n")
    res = CliRunner().invoke(cli.cli,
                              ["run", str(fdf), "--env", "siesta-stable"])
    assert res.exit_code == 0
    assert "conda activate siesta-stable" in (tmp_path / "x.run.sh").read_text()


def test_run_prints_run_command_hint(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("# fake\n")
    res = CliRunner().invoke(cli.cli, ["run", str(fdf)])
    assert "Wrote" in res.output
    assert "Run:" in res.output
    assert "x.run.sh" in res.output


# --------------------------------------------------------------------- #
#  Error paths                                                          #
# --------------------------------------------------------------------- #


def test_run_unknown_extension_errors(tmp_path):
    txt = tmp_path / "x.txt"
    txt.write_text("not a script\n")
    res = CliRunner().invoke(cli.cli, ["run", str(txt)])
    assert res.exit_code != 0
    # Click's UsageError prefixes "Error:".
    assert "unsupported script extension" in (res.stderr + res.output).lower()


def test_run_nonexistent_path_errors():
    res = CliRunner().invoke(cli.cli, ["run", "/tmp/no-such-file.fdf"])
    # Click's exists=True validates before our handler runs.
    assert res.exit_code != 0


def test_run_help():
    res = CliRunner().invoke(cli.cli, ["run", "--help"])
    assert res.exit_code == 0
    assert "--env" in res.output
    assert "--np"  in res.output
    assert ".fdf" in res.output
