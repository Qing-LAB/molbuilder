"""Tests for the per-stage convergence-target extension of the
molwatch log format (#542 / C1.4).

Pins:
  * ``write_initial_preview`` accepts ``stage_name`` + ``convergence_
    targets`` kwargs and emits them as ``# stage:`` /
    ``# convergence.<key>:`` header lines.
  * Backwards-compatible: omitting both kwargs produces the same
    output as before the C1.4 change (no spurious headers).
  * Invalid convergence-target keys (whitespace) raise ValueError at
    write time rather than producing a malformed log that the
    inspector would mis-parse.
  * The SIESTA multi-stage CLI emits one ``<basename>-<stage>.
    molwatch.log`` per enabled stage with the right per-stage target.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from molbuilder.cli import cli
from molbuilder.structure import Structure
from molbuilder.trajectory_log import write_initial_preview


# A 3-D (non-linear) molecule so the derived vacuum cell isn't degenerate at
# the default vacuum=0 (structure-periodicity.md).  These CLI tests exercise
# stage/log mechanics, not geometry, so methane is fine.
_XYZ = textwrap.dedent("""\
    5

    C  0.000  0.000  0.000
    H  0.629  0.629  0.629
    H -0.629 -0.629  0.629
    H -0.629  0.629 -0.629
    H  0.629 -0.629 -0.629
""")


@pytest.fixture
def xyz(tmp_path):
    p = tmp_path / "h2.xyz"
    p.write_text(_XYZ)
    return p


@pytest.fixture
def h2():
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        vacuum=(12.0, 12.0, 12.0),   # non-degenerate cell for this linear molecule
    )


# --------------------------------------------------------------------- #
#  write_initial_preview kwargs                                          #
# --------------------------------------------------------------------- #


def test_stage_name_emits_stage_header(h2, tmp_path):
    p = tmp_path / "JOB-stage1.molwatch.log"
    write_initial_preview(h2, p, job="JOB", engine="siesta",
                            stage_name="stage1")
    text = p.read_text()
    assert "# stage: stage1" in text


def test_convergence_targets_emit_namespaced_headers(h2, tmp_path):
    p = tmp_path / "JOB-stage1.molwatch.log"
    write_initial_preview(h2, p, job="JOB", engine="siesta",
                            convergence_targets={
                                "max_force_ev_per_ang": 0.05,
                                "max_steps":            600,
                            })
    text = p.read_text()
    assert "# convergence.max_force_ev_per_ang: 0.05" in text
    assert "# convergence.max_steps: 600" in text


def test_backwards_compat_no_kwargs_means_no_stage_or_convergence_headers(
        h2, tmp_path):
    """The pre-C1.4 callers don't pass stage_name / convergence_targets.
    Their output must be byte-equivalent for the new header subset:
    no ``# stage:``, no ``# convergence.``."""
    p = tmp_path / "JOB.molwatch.log"
    write_initial_preview(h2, p, job="JOB", engine="siesta")
    text = p.read_text()
    assert "# stage:" not in text
    assert "# convergence." not in text


def test_convergence_targets_with_whitespace_key_raises(h2, tmp_path):
    """A key with internal whitespace would break the line-oriented
    parser the inspector uses.  Reject at write time."""
    p = tmp_path / "bad.molwatch.log"
    with pytest.raises(ValueError, match="whitespace"):
        write_initial_preview(h2, p, job="JOB", engine="siesta",
                                convergence_targets={
                                    "max force": 0.05,  # space in key
                                })


def test_convergence_targets_empty_dict_emits_no_headers(h2, tmp_path):
    """An empty dict is treated like None (no header lines)."""
    p = tmp_path / "JOB.molwatch.log"
    write_initial_preview(h2, p, job="JOB", engine="siesta",
                            convergence_targets={})
    text = p.read_text()
    assert "# convergence." not in text


def test_stage_name_and_convergence_can_be_combined(h2, tmp_path):
    p = tmp_path / "JOB-stage2.molwatch.log"
    write_initial_preview(h2, p, job="JOB", engine="siesta",
                            stage_name="stage2",
                            convergence_targets={
                                "max_force_ev_per_ang": 0.04,
                            })
    text = p.read_text()
    assert "# stage: stage2" in text
    assert "# convergence.max_force_ev_per_ang: 0.04" in text
    # Headers appear before the step block.
    stage_ix = text.find("# stage: stage2")
    step_ix = text.find("==== molwatch step 0 begin ====")
    assert 0 < stage_ix < step_ix


def test_step_block_unchanged_by_new_kwargs(h2, tmp_path, monkeypatch):
    """The step-0 preview body (coords, energy=None, scf_history)
    must be byte-equivalent whether or not the new kwargs are set --
    they only add HEADER lines, not body content.

    Mock ``time.time`` so the two writes get the same wall_time
    (without the mock the two calls land in different milliseconds
    and the body's wall_time line diverges by microseconds when the
    test runs as part of a larger suite)."""
    monkeypatch.setattr(
        "molbuilder.trajectory_log.format.time.time",
        lambda: 1735000000.123,
    )
    p1 = tmp_path / "without.log"
    p2 = tmp_path / "with.log"
    write_initial_preview(h2, p1, job="J", engine="siesta")
    write_initial_preview(h2, p2, job="J", engine="siesta",
                            stage_name="stage1",
                            convergence_targets={"max_force_ev_per_ang": 0.05})
    body_marker = "==== molwatch step 0 begin ===="
    body1 = p1.read_text().split(body_marker, 1)[1]
    body2 = p2.read_text().split(body_marker, 1)[1]
    assert body1 == body2


# --------------------------------------------------------------------- #
#  CLI: multi-stage SIESTA emits per-stage molwatch logs                #
# --------------------------------------------------------------------- #


def _invoke(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def test_multi_stage_cli_emits_per_stage_molwatch_logs(xyz, tmp_path):
    """``molbuilder fdf ... --stage-strategy vib-quality`` produces
    one ``<basename>-<stage>.molwatch.log`` per enabled stage."""
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "vib-quality")
    assert r.exit_code == 0, r.output
    logs = sorted(p.name for p in tmp_path.glob("JOB-stage*.molwatch.log"))
    assert logs == ["JOB-stage1.molwatch.log",
                    "JOB-stage2.molwatch.log",
                    "JOB-stage3.molwatch.log"]


def test_per_stage_molwatch_log_carries_stage_target(xyz, tmp_path):
    """Each per-stage log carries the stage's own
    ``max_force_ev_per_ang`` -- so the watch-tab horizontal threshold
    matches the stage that's currently running.

    Default ladder (siesta/stages.py::default_siesta_stages) per stage:
      stage1: 0.05 (loose preopt)
      stage2: 0.04 (publishable)
      stage3: 0.01 (crystal-tight)
    """
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "vib-quality")
    assert r.exit_code == 0, r.output
    text1 = (tmp_path / "JOB-stage1.molwatch.log").read_text()
    text2 = (tmp_path / "JOB-stage2.molwatch.log").read_text()
    text3 = (tmp_path / "JOB-stage3.molwatch.log").read_text()
    assert "# convergence.max_force_ev_per_ang: 0.05" in text1
    assert "# convergence.max_force_ev_per_ang: 0.04" in text2
    assert "# convergence.max_force_ev_per_ang: 0.01" in text3
    # And each carries its own stage label.
    assert "# stage: stage1" in text1
    assert "# stage: stage2" in text2
    assert "# stage: stage3" in text3


def test_per_stage_molwatch_log_carries_max_steps(xyz, tmp_path):
    """Each per-stage log carries its own ``max_steps`` so the
    inspector can render the right "progress through the stage"
    indicator.  Defaults: stage1=600, stage2=200, stage3=100."""
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "vib-quality")
    assert r.exit_code == 0, r.output
    text1 = (tmp_path / "JOB-stage1.molwatch.log").read_text()
    text2 = (tmp_path / "JOB-stage2.molwatch.log").read_text()
    text3 = (tmp_path / "JOB-stage3.molwatch.log").read_text()
    assert "# convergence.max_steps: 600" in text1
    assert "# convergence.max_steps: 200" in text2
    assert "# convergence.max_steps: 100" in text3


def test_two_stage_strategy_emits_only_two_logs(xyz, tmp_path):
    """``--stage-strategy publishable`` (default: stage1+stage2)
    produces TWO logs, not three -- stage3 is disabled."""
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    logs = sorted(p.name for p in tmp_path.glob("JOB-stage*.molwatch.log"))
    assert logs == ["JOB-stage1.molwatch.log",
                    "JOB-stage2.molwatch.log"]


def test_single_stage_path_unchanged_by_c14(xyz, tmp_path):
    """The single-stage (non-multi-stage) CLI path predates C1.4 and
    must not regress -- one ``.molwatch.log`` (or none, depending on
    cfg.write_molwatch_log) and no ``-stage`` suffix in the filename."""
    fdf = tmp_path / "JOB.fdf"
    # No --stage-strategy / --stages-json -> single-stage branch.
    r = _invoke("fdf", str(xyz), str(fdf))
    assert r.exit_code == 0, r.output
    # No -stageN logs.
    assert not any(tmp_path.glob("JOB-stage*.molwatch.log"))
