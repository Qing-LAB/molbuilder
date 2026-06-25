"""Test the ``molbuilder runtime-info`` CLI -- offline JSON sidecar
for SIESTA / PySCF output files.  See cli.py::cmd_runtime_info."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from molbuilder.cli import cli


_STUB_OUT = dedent("""\
    Siesta Version  : 5.4.2
    Architecture   : x86_64-linux-gnu
    Compiler version: GNU Fortran 13.3.0
    Parallelisations: MPI, OPENMP

    NetCDF support
    ELPA support

    * Running on 4 MPI processes

    redata: Diag.Algorithm = ELPA-1STAGE
    redata: Diag.ELPA.GPU = T

    ELPA: NVIDIA GPU detected: NVIDIA A100-SXM4-40GB (sm_80)

    siesta: System type = molecule

    siesta: Final energy (eV):
    siesta:  Total =          -1234.567

    End of run: 25-JUN-2026 12:00:00
    """)


def _write_stub(dir_: Path, name: str = "job.out") -> Path:
    p = dir_ / name
    p.write_text(_STUB_OUT)
    return p


def test_default_path_writes_sidecar_next_to_input(tmp_path):
    p = _write_stub(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p)])
    assert result.exit_code == 0, result.output
    sidecar = tmp_path / "job.runtime_info.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["siesta_build"]["version"] == "5.4.2"
    assert data["siesta_diag"]["algorithm"] == "ELPA-1STAGE"
    assert data["siesta_diag"]["elpa_gpu"] is True


def test_explicit_out_path(tmp_path):
    p = _write_stub(tmp_path)
    target = tmp_path / "custom.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p),
                                 "--out", str(target)])
    assert result.exit_code == 0
    assert target.exists()
    data = json.loads(target.read_text())
    assert data["siesta_build"]["parallelisations"] == ["MPI", "OPENMP"]


def test_stdout_mode_emits_json(tmp_path):
    p = _write_stub(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p), "--out", "-"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["siesta_diag"]["gpu_device"] == "NVIDIA A100-SXM4-40GB"
    # No sidecar file created in stdout mode.
    assert not (tmp_path / "job.runtime_info.json").exists()


def test_pretty_default_indents(tmp_path):
    p = _write_stub(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p), "--out", "-"])
    # --pretty defaults to True; output should have multiple lines.
    assert "\n" in result.output.strip()


def test_no_pretty_collapses(tmp_path):
    p = _write_stub(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p), "--out", "-",
                                 "--no-pretty"])
    assert result.exit_code == 0
    # Compact form -- single JSON line, no indentation.
    body = result.output.strip()
    assert body.count("\n") == 0


def test_frozen_atoms_set_serialised_as_sorted_list(tmp_path):
    """runtime_info["frozen_atoms"] is a Python set in-memory.  The
    sidecar emitter must convert it to a sorted list (JSON-native +
    deterministic ordering) -- not bail with TypeError."""
    out_text = _STUB_OUT + dedent("""\

        siesta: Constraints applied in the following order:
        siesta: Constraint (3): pos
          [ 5 -- 7 ]
        """)
    p = tmp_path / "constrained.out"
    p.write_text(out_text)
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p), "--out", "-"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    if "frozen_atoms" in data:
        # Must be a list, sorted ascending.  (Set serialisation
        # would have failed json.dumps with TypeError.)
        fa = data["frozen_atoms"]
        assert isinstance(fa, list)
        assert fa == sorted(fa)


def test_unknown_format_exits_nonzero(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("not a SIESTA or PySCF output\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p)])
    assert result.exit_code != 0
    assert "Error" in result.output or "Error" in (result.stderr or "")
