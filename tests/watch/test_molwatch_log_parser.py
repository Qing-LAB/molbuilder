"""Sanity check for the unified ``.molwatch.log`` parser.

Builds a tiny synthetic log that contains the same markers as the
real thing (two complete steps + a torn third step where the closing
``end`` marker hasn't been written yet) and verifies that the parser
keeps two complete steps with correct numbers, drops the torn one,
and round-trips through strict JSON.

Format spec lives in ``docs/types/parsers.md``.  These tests are
spec-derived: they don't peek at parser internals, they just assert
the documented invariants.
"""

from __future__ import annotations

import json
import math

import pytest

from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict
from molbuilder.parse.engines.molwatch import (
    MolwatchLogFileParser,
    MolwatchLogParser,
)


# A small, hand-written log with two complete blocks + one torn one.
# The torn third block has a 'begin' but no matching 'end' marker --
# that's exactly the state of the file when molwatch tails a still-
# running job mid-write.
SAMPLE = """\
# molwatch trajectory log v1
# generator: molbuilder/pyscf_input
# engine: pyscf
# job: water_relax
# units: energy=eV, force=eV/Ang, coords=Ang
# created: 2026-04-25T11:00:00

==== molwatch step 0 begin ====
step_index: 0
n_atoms: 3
coordinates (Ang):
   O      0.00000000      0.00000000      0.00000000
   H      0.95700000      0.00000000      0.00000000
   H     -0.23900000      0.92700000      0.00000000
energy (eV): -76.12345600
forces (eV/Ang):
   O     -0.00100000     -0.00200000      0.00000000
   H      0.00050000      0.00100000      0.00000000
   H      0.00050000      0.00100000      0.00000000
max_force (eV/Ang): 0.00240000
scf_history begin
#  cycle      energy(eV)         delta_E(eV)        gnorm(eV/Ang)            ddm
       1     -76.00000000        0.00000000      5.00000000e-02   1.00000000e-01
       2     -76.10000000       -0.10000000      5.00000000e-03   1.00000000e-02
       3     -76.12345600       -0.02345600      1.00000000e-04   1.00000000e-04
scf_history end
==== molwatch step 0 end ====

==== molwatch step 1 begin ====
step_index: 1
n_atoms: 3
coordinates (Ang):
   O      0.01000000      0.00000000      0.00000000
   H      0.96700000      0.00000000      0.00000000
   H     -0.22900000      0.92700000      0.00000000
energy (eV): -76.20000000
forces (eV/Ang):
   O      0.00010000      0.00020000      0.00000000
   H     -0.00005000     -0.00010000      0.00000000
   H     -0.00005000     -0.00010000      0.00000000
max_force (eV/Ang): 0.00022400
scf_history begin
#  cycle      energy(eV)         delta_E(eV)        gnorm(eV/Ang)            ddm
       1     -76.20000000        0.00000000      1.00000000e-04   1.00000000e-04
       2     -76.20000000        0.00000000              None             None
scf_history end
==== molwatch step 1 end ====

==== molwatch step 2 begin ====
step_index: 2
n_atoms: 3
coordinates (Ang):
   O      0.02000000      0.00000000      0.00000000
"""


@pytest.fixture
def mw_path(tmp_path):
    p = tmp_path / "water_relax.molwatch.log"
    p.write_text(SAMPLE)
    return str(p)


def test_can_parse(mw_path):
    assert MolwatchLogParser.can_parse(mw_path) is True


def test_can_parse_rejects_non_molwatch(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("just some text\nnot a molwatch log\n")
    assert MolwatchLogParser.can_parse(str(p)) is False


def test_torn_final_block_dropped(mw_path):
    """A block with `begin` but no `end` is dropped silently --
    so molwatch never shows a half-written final step."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert len(result["frames"]) == 2


def test_frame_coordinates(mw_path):
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert result["frames"][0] == [
        ["O",  0.0,     0.0,   0.0],
        ["H",  0.957,   0.0,   0.0],
        ["H", -0.239,   0.927, 0.0],
    ]
    assert result["frames"][1][0] == ["O", 0.01, 0.0, 0.0]


def test_energies(mw_path):
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert math.isclose(result["energies"][0], -76.123456)
    assert math.isclose(result["energies"][1], -76.20)


def test_max_forces(mw_path):
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert math.isclose(result["max_forces"][0], 0.00240)
    assert math.isclose(result["max_forces"][1], 0.000224)


def test_per_atom_forces(mw_path):
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert result["forces"][0] == [
        [-0.001, -0.002, 0.0],
        [ 0.0005, 0.001, 0.0],
        [ 0.0005, 0.001, 0.0],
    ]


def test_iterations(mw_path):
    """`iterations` mirrors the per-block `step_index` from the
    `==== molwatch step <N> ====` markers, in encounter order."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert result["iterations"] == [0, 1]


def test_lattice_is_none(mw_path):
    """molwatch logs are for molecules; lattice is always None."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert result["lattice"] is None


def test_source_format_from_engine_header(mw_path):
    """The `# engine: pyscf` header maps into result["source_format"]."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    assert result["source_format"] == "pyscf"


def test_scf_history_per_step(mw_path):
    """Two step blocks; each carries its own scf_history list."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    runs = result["scf_history"]
    assert len(runs) == 2
    assert len(runs[0]) == 3
    assert len(runs[1]) == 2


def test_scf_cycle_keys(mw_path):
    """Every per-cycle entry has the unified key set.

    ``wall_time`` (2026-06-20) is included unconditionally — None
    for pre-feature .molwatch.log files (5-column SCF rows),
    epoch-seconds float for new files (6-column SCF rows).
    """
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    expected = {"cycle", "energy", "delta_E", "gnorm", "ddm", "wall_time"}
    for run in result["scf_history"]:
        for entry in run:
            assert set(entry.keys()) == expected


def test_scf_none_residuals_round_trip(mw_path):
    """A residual written as the literal 'None' becomes JSON null --
    not a string, not NaN."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    last_cycle = result["scf_history"][1][1]
    assert last_cycle["gnorm"] is None
    assert last_cycle["ddm"] is None


def test_json_strict_safe(mw_path):
    """Result must serialise with allow_nan=False -- the molwatch /api/data
    endpoint uses strict JSON so a NaN slipping through is a contract bug."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    json.dumps(result, allow_nan=False)


def test_index_aligned_arrays(mw_path):
    """Per-step lists must be index-aligned with frames -- the front-end
    walks them in lockstep via the slider, so a length mismatch is a
    spec violation."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    n = len(result["frames"])
    assert n == len(result["energies"])
    assert n == len(result["max_forces"])
    assert n == len(result["forces"])
    assert n == len(result["iterations"])
    assert n == len(result["scf_history"])


def test_engine_default_when_header_missing(tmp_path):
    """If the `# engine:` header line is absent, source_format defaults
    to 'molwatch' (so the UI still has a string, not None)."""
    sample_no_engine = (
        "# molwatch trajectory log v1\n"
        "# generator: hand-rolled\n"
        "# units: energy=eV, force=eV/Ang, coords=Ang\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H      0.00000000      0.00000000      0.00000000\n"
        "energy (eV): -1.0\n"
        "forces (eV/Ang):\n"
        "   H      0.00000000      0.00000000      0.00000000\n"
        "max_force (eV/Ang): 0.00000000\n"
        "scf_history begin\n"
        "scf_history end\n"
        "==== molwatch step 0 end ====\n"
    )
    p = tmp_path / "noeng.molwatch.log"
    p.write_text(sample_no_engine)
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(p)))
    assert result["source_format"] == "molwatch"
    assert len(result["frames"]) == 1
    # An empty scf_history is allowed (no cycles in this synthetic block).
    assert result["scf_history"][0] == []


def test_registry_dispatches_to_molwatch_parser(mw_path):
    """The registry must pick MolwatchLogParser for `.molwatch.log` files
    -- not the SIESTA or PySCF parser, which would either reject or
    misread our format."""
    from molbuilder.parse import detect as detect_parser
    assert detect_parser(mw_path) is MolwatchLogFileParser


def test_wall_time_field_round_trip(tmp_path):
    """When step blocks carry `wall_time: <unix epoch>`, the parser
    must surface it on Frame.wall_time and the legacy dict must expose
    it as `wall_times[i]` index-aligned with frames.  Used by the
    Watch UI to render "Started 2h 15m ago, last step 30s ago"."""
    sample = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "# created: 2026-04-25T11:00:00\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "kind: initial_preview\n"
        "wall_time: 1777864027.425\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H  0.0  0.0  0.0\n"
        "energy (eV): None\n"
        "forces (eV/Ang):\n"
        "max_force (eV/Ang): None\n"
        "scf_history begin\n"
        "scf_history end\n"
        "==== molwatch step 0 end ====\n"
        "\n"
        "==== molwatch step 1 begin ====\n"
        "step_index: 1\n"
        "wall_time: 1777864041.373\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H  0.01  0.0  0.0\n"
        "energy (eV): -0.5\n"
        "forces (eV/Ang):\n"
        "   H  0.0  0.0  0.0\n"
        "max_force (eV/Ang): 0.0\n"
        "scf_history begin\n"
        "scf_history end\n"
        "==== molwatch step 1 end ====\n"
    )
    p = tmp_path / "wt.molwatch.log"
    p.write_text(sample)
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(p)))
    wt = result["wall_times"]
    assert len(wt) == 2 and len(wt) == len(result["frames"])
    assert wt[0] == pytest.approx(1777864027.425)
    assert wt[1] == pytest.approx(1777864041.373)
    # 13.948 seconds of "simulation" in the synthetic file
    assert wt[1] - wt[0] == pytest.approx(13.948, abs=0.001)


def test_run_state_finished_when_concluded_marker_present(tmp_path):
    """When the writer emits ``# concluded: <ISO>`` (atexit hook on
    normal script exit), the parser must surface
    Trajectory.run_state == "finished".  This is the user-facing
    "is the run done?" signal -- authoritative, not heuristic."""
    sample = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "# created: 2026-04-25T11:00:00\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H  0.0  0.0  0.0\n"
        "energy (eV): -0.5\n"
        "forces (eV/Ang):\n"
        "max_force (eV/Ang): 0.0\n"
        "scf_history begin\n"
        "scf_history end\n"
        "==== molwatch step 0 end ====\n"
        "\n"
        "# concluded: 2026-04-25T11:23:45\n"
    )
    p = tmp_path / "done.molwatch.log"
    p.write_text(sample)
    traj = MolwatchLogParser.parse(str(p))
    assert traj.run_state == "finished"
    assert traj.error_message is None


def test_run_state_error_when_error_marker_present(tmp_path):
    """``# error: <msg>`` written by the excepthook means the script
    raised an uncaught exception.  Error has priority over the
    co-emitted ``# concluded:`` line (atexit fires after excepthook,
    so both appear; the parser keeps the error)."""
    sample = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "# created: 2026-04-25T11:00:00\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H  0.0  0.0  0.0\n"
        "energy (eV): -0.5\n"
        "forces (eV/Ang):\n"
        "max_force (eV/Ang): 0.0\n"
        "scf_history begin\n"
        "scf_history end\n"
        "==== molwatch step 0 end ====\n"
        "\n"
        "# error: ValueError: bad geometry\n"
        "# concluded: 2026-04-25T11:23:45\n"
    )
    p = tmp_path / "err.molwatch.log"
    p.write_text(sample)
    traj = MolwatchLogParser.parse(str(p))
    assert traj.run_state == "error"
    assert "ValueError" in traj.error_message
    assert "bad geometry" in traj.error_message


def test_run_state_ongoing_when_no_marker(mw_path):
    """No ``# concluded:`` and no ``# error:`` -> run is treated as
    ongoing.  This is the SAFE default: don't claim "finished" when
    we can't see the marker (the writer might have been SIGKILL'd)."""
    traj = MolwatchLogParser.parse(mw_path)
    assert traj.run_state == "ongoing"
    assert traj.error_message is None


def test_run_state_propagates_to_legacy_dict(tmp_path):
    """The watch web layer reads `run_state` and `error_message` out
    of the legacy dict to drive the UI badge.  Pin the round-trip."""
    sample = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H  0.0  0.0  0.0\n"
        "energy (eV): None\n"
        "forces (eV/Ang):\n"
        "max_force (eV/Ang): None\n"
        "scf_history begin\n"
        "scf_history end\n"
        "==== molwatch step 0 end ====\n"
        "# error: RuntimeError: SCF did not converge\n"
        "# concluded: 2026-04-25T11:23:45\n"
    )
    p = tmp_path / "rt.molwatch.log"
    p.write_text(sample)
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(p)))
    assert result["run_state"] == "error"
    assert "RuntimeError" in result["error_message"]


def test_wall_time_absent_in_old_logs(mw_path):
    """The SAMPLE log has no `wall_time:` lines (it's older than the
    field).  Parser must fall back to None for each frame and the
    legacy dict's `wall_times` list must be all-None index-aligned
    with frames -- otherwise the watch UI's optional-elapsed code path
    breaks."""
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(mw_path))
    wt = result["wall_times"]
    assert len(wt) == len(result["frames"])
    assert all(v is None for v in wt)


def test_scf_cycle_wall_time_round_trip(tmp_path):
    """6-column SCF rows (added 2026-06-20) round-trip through the
    parser as a 'wall_time' field on each per-cycle dict.  Per-cycle
    time delta is then ``scf[i+1]['wall_time'] - scf[i]['wall_time']``
    without any client-side stitching."""
    sample = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "# created: 2026-06-20T10:00:00\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H  0.0  0.0  0.0\n"
        "energy (eV): -0.5\n"
        "forces (eV/Ang):\n"
        "max_force (eV/Ang): 0.0\n"
        "scf_history begin\n"
        "#  cycle  energy(eV)  delta_E(eV)  gnorm(eV/Ang)  ddm  wall_time(s)\n"
        "       1     -0.40000000      0.00000000     1.00000000e-01"
        "   2.00000000e-01    1718500000.100\n"
        "       2     -0.49000000     -0.09000000     5.00000000e-02"
        "   1.00000000e-01    1718500001.350\n"
        "       3     -0.50000000     -0.01000000     1.00000000e-03"
        "   1.00000000e-03    1718500002.500\n"
        "scf_history end\n"
        "==== molwatch step 0 end ====\n"
    )
    p = tmp_path / "scfwt.molwatch.log"
    p.write_text(sample)
    traj = MolwatchLogParser.parse(str(p))
    assert len(traj.frames) == 1
    scf = traj.frames[0].scf_history
    assert scf is not None
    assert len(scf) == 3
    # Every cycle dict carries wall_time as a float epoch.
    for c in scf:
        assert "wall_time" in c
        assert isinstance(c["wall_time"], float)
    # Deltas: 1.25 s, then 1.15 s -- direct subtraction works.
    assert scf[1]["wall_time"] - scf[0]["wall_time"] == pytest.approx(1.25)
    assert scf[2]["wall_time"] - scf[1]["wall_time"] == pytest.approx(1.15)


def test_scf_cycle_wall_time_optional_in_old_logs(tmp_path):
    """Pre-feature .molwatch.log files (5-column SCF rows) must
    parse fine; the 'wall_time' field surfaces as None on each cycle
    dict so consumers can branch on presence without crashing."""
    sample = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "# created: 2026-04-25T11:00:00\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H  0.0  0.0  0.0\n"
        "energy (eV): -0.5\n"
        "forces (eV/Ang):\n"
        "max_force (eV/Ang): 0.0\n"
        "scf_history begin\n"
        "#  cycle  energy(eV)  delta_E(eV)  gnorm(eV/Ang)  ddm\n"
        "       1     -0.40000000      0.00000000     1.00000000e-01"
        "   2.00000000e-01\n"
        "       2     -0.50000000     -0.10000000     1.00000000e-03"
        "   1.00000000e-03\n"
        "scf_history end\n"
        "==== molwatch step 0 end ====\n"
    )
    p = tmp_path / "oldscf.molwatch.log"
    p.write_text(sample)
    traj = MolwatchLogParser.parse(str(p))
    scf = traj.frames[0].scf_history
    assert scf is not None
    for c in scf:
        assert "wall_time" in c
        assert c["wall_time"] is None
