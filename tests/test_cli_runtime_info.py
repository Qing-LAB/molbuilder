"""Test the ``molbuilder runtime-info`` CLI -- offline JSON sidecar
for SIESTA / PySCF output files.  See cli.py::cmd_runtime_info."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from molbuilder.cli import cli


#: A REAL SIESTA 5.4.2 run (``tests/watch/fixtures/siesta_frozen``).  A
#: hand-written stub stood here until 2026-09-26, its header lines ones SIESTA
#: never prints.
_REAL = (Path(__file__).parent / "watch" / "fixtures" / "siesta_frozen"
         / "hemeC-stage2-run3-finished-42fr.out")


def _write_stub(dir_: Path, name: str = "job.out") -> Path:
    p = dir_ / name
    p.write_text(_REAL.read_text(errors="replace"))
    return p


def test_default_path_writes_sidecar_next_to_input(tmp_path):
    p = _write_stub(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p)])
    assert result.exit_code == 0, result.output
    sidecar = tmp_path / "job.runtime_info.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["siesta_build"]["version"].startswith("5.4.2")
    assert data["siesta_diag"]["algorithm"] == "D&C"


def test_explicit_out_path(tmp_path):
    p = _write_stub(tmp_path)
    target = tmp_path / "custom.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p),
                                 "--out", str(target)])
    assert result.exit_code == 0
    assert target.exists()
    data = json.loads(target.read_text())
    assert data["siesta_build"]["parallelisations"] == ["MPI"]


def test_stdout_mode_emits_json(tmp_path):
    p = _write_stub(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p), "--out", "-"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["siesta_diag"]["distribution"] == "2 x 4"
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
    out_text = _REAL.read_text(errors="replace") + dedent("""\

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
    # ASSERTED, not guarded on.  This whole body sat behind
    # `if "frozen_atoms" in data:`, so the key going missing -- the regression
    # the test is named for -- passed silently (2026-09-10).
    assert "frozen_atoms" in data, (
        f"the constrained atoms did not reach runtime_info: {sorted(data)}")
    fa = data["frozen_atoms"]
    # A set would have failed json.dumps with TypeError; sorted is what makes
    # the sidecar deterministic.
    assert fa == sorted(fa) and fa == [4, 5, 6], fa


def test_unknown_format_exits_nonzero(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("not a SIESTA or PySCF output\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["runtime-info", str(p)])
    assert result.exit_code != 0
    assert "Error" in result.output or "Error" in (result.stderr or "")
