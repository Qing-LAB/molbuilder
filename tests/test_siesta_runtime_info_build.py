"""Pin the SIESTA build + diagonalizer probes that populate
``runtime_info["siesta_build"]`` + ``runtime_info["siesta_diag"]``.

User contract (2026-06-25): the watcher / results tab must surface
WHAT THE BINARY ACTUALLY RAN WITH -- version, parallelisation, ELPA
linkage, requested diagonalizer, GPU device -- so users reading the
results can verify the run matched their intent.  The data is parsed
out of the SIESTA .out file (the binary's self-report at startup +
the redata: echo of input fdf), so no extra sidecar / probe is
needed.  See molbuilder/parse/engines/siesta.py near the regex
constants for the line-format references.
"""
from __future__ import annotations

from textwrap import dedent

import pytest

from molbuilder.parse.engines.siesta import SiestaParser


# A minimal but realistic SIESTA 5.4.2 header + redata block.  Real
# .out files have hundreds of lines between the header and the SCF
# table; the parser's _scan_runtime_info is invoked on every non-
# section line so the in-between content doesn't matter for these
# probes.  We trim to what the parser actually consumes for clarity.
_SAMPLE_HEADER_GPU = """\
Siesta Version  : 5.4.2
Architecture   : x86_64-linux-gnu
Compiler version: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
PP flags       : -DMPI -DCDF -DGRID_DP
Parallelisations: MPI, OPENMP

NetCDF support
NetCDF-4 support
Lua support
ELPA support

* Running on 4 MPI processes
* Running on host: workstation-01

redata: Diagonalization algorithm                  = ELPA-1STAGE
redata: Diag.ELPA.GPU                              = T

ELPA: NVIDIA GPU detected: NVIDIA A100-SXM4-40GB (sm_80)

siesta: System type = molecule

"""

_SAMPLE_HEADER_CPU = """\
Version : 5.4.2
Architecture : x86_64-linux-gnu
Compiler version: GNU Fortran 13.3.0
Parallelisations: MPI

NetCDF support

* Running on 8 MPI processes

redata: Diagonalization algorithm = divide-and-conquer

siesta: System type = molecule

"""


def _parse_minimal(tmp_path, header_body: str):
    """Write a stub .out with the header followed by trivial body and
    return the SiestaParser's Trajectory.  The body has just enough
    structure for the parser not to bail early."""
    out = tmp_path / "stub.out"
    # SIESTA's End of run marker terminates the parser cleanly.
    body = header_body + dedent("""\

        siesta: Final energy (eV):
        siesta:  Total =          -1234.567

        End of run: 25-JUN-2026 12:00:00
        """)
    out.write_text(body)
    return SiestaParser.parse(str(out))


# ----------------------------------------------------------------- #
#  Build-header capture                                              #
# ----------------------------------------------------------------- #


def test_version_captured_from_siesta_prefix(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    assert traj.runtime_info["siesta_build"]["version"] == "5.4.2"


def test_version_captured_from_bare_prefix(tmp_path):
    """Older SIESTA builds emit ``Version :`` without the ``Siesta``
    prefix.  Both shapes must populate the same key."""
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_CPU)
    assert traj.runtime_info["siesta_build"]["version"] == "5.4.2"


def test_architecture_captured(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    assert (traj.runtime_info["siesta_build"]["architecture"]
            == "x86_64-linux-gnu")


def test_compiler_captured(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    assert "13.3.0" in traj.runtime_info["siesta_build"]["compiler"]


def test_parallelisations_tokenised(tmp_path):
    """Comma-separated list of parallelisation tokens, upper-cased."""
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    assert (traj.runtime_info["siesta_build"]["parallelisations"]
            == ["MPI", "OPENMP"])


def test_parallelisations_single_token(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_CPU)
    assert (traj.runtime_info["siesta_build"]["parallelisations"]
            == ["MPI"])


def test_feature_presence_lines_become_booleans(tmp_path):
    """Bare-name presence lines (``NetCDF support``, ``ELPA support``)
    map to ``True`` on the matching key.  Absent features are
    simply absent from the dict (NOT explicit ``False``) -- the
    inspector tolerates absence."""
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    build = traj.runtime_info["siesta_build"]
    assert build.get("netcdf")  is True
    assert build.get("netcdf4") is True
    assert build.get("lua")     is True
    assert build.get("elpa")    is True
    # Not in the sample header -> absent key.
    assert "elsi"  not in build
    assert "pexsi" not in build


def test_cpu_only_header_omits_elpa(tmp_path):
    """A CPU-only build whose header doesn't print ``ELPA support``
    must NOT have ``elpa: True`` in the captured build dict.  This
    is the inverse of the GPU case: the difference between
    ``cpu_elpa: True`` and absence of the key is load-bearing for
    the inspector's "this run used ELPA" badge."""
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_CPU)
    build = traj.runtime_info["siesta_build"]
    assert "elpa" not in build


# ----------------------------------------------------------------- #
#  Diagonalizer capture                                              #
# ----------------------------------------------------------------- #


def test_diag_algorithm_captured(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    assert (traj.runtime_info["siesta_diag"]["algorithm"]
            == "ELPA-1STAGE")


def test_diag_algorithm_cpu_default(tmp_path):
    """SIESTA's default ScaLAPACK path echoes its algorithm name."""
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_CPU)
    assert (traj.runtime_info["siesta_diag"]["algorithm"]
            == "DIVIDE-AND-CONQUER")


def test_diag_elpa_gpu_flag_captured_true(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    assert traj.runtime_info["siesta_diag"]["elpa_gpu"] is True


def test_diag_elpa_gpu_flag_absent_on_cpu_run(tmp_path):
    """CPU runs don't echo Diag.ELPA.GPU at all; the key stays
    absent rather than defaulting to False."""
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_CPU)
    diag = traj.runtime_info.get("siesta_diag", {})
    assert "elpa_gpu" not in diag


def test_gpu_device_captured_with_compute_capability(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_GPU)
    diag = traj.runtime_info["siesta_diag"]
    assert diag["gpu_device"] == "NVIDIA A100-SXM4-40GB"
    assert diag["gpu_compute_capability"] == "sm_80"


def test_gpu_keys_absent_on_cpu_run(tmp_path):
    traj = _parse_minimal(tmp_path, _SAMPLE_HEADER_CPU)
    diag = traj.runtime_info.get("siesta_diag", {})
    assert "gpu_device" not in diag
    assert "gpu_compute_capability" not in diag


# ----------------------------------------------------------------- #
#  Defensive: header is optional                                     #
# ----------------------------------------------------------------- #


def test_no_header_yields_no_build_dict(tmp_path):
    """A truncated .out with no header lines doesn't crash the
    parser AND doesn't synthesise a fake ``siesta_build`` dict --
    the key just stays absent."""
    body = dedent("""\
        siesta: System type = molecule

        End of run: 25-JUN-2026 12:00:00
        """)
    out = tmp_path / "headerless.out"
    out.write_text(body)
    traj = SiestaParser.parse(str(out))
    assert "siesta_build" not in traj.runtime_info
    assert "siesta_diag"  not in traj.runtime_info


# ----------------------------------------------------------------- #
#  Loose-capture probe validation (audit-2026-06-26 T1 BLOCKER 3)   #
# ----------------------------------------------------------------- #
#  The parallelisations + diag-algorithm probes capture free-form
#  tokens.  An unrecognised token MUST still be recorded (so the
#  Results tab shows what SIESTA printed) but ALSO raise a
#  ParseWarning -- a future SIESTA shape we don't understand must
#  surface rather than silently masquerade as ground truth.


def _runtime_warnings(traj):
    return [w for w in traj.parse_warnings
            if w.category == "runtime_info"]


def test_known_parallelisations_and_algo_emit_no_warning(tmp_path):
    """The realistic GPU + CPU headers use only known tokens
    (MPI/OPENMP, ELPA-1STAGE, divide-and-conquer) -- no runtime_info
    warning should fire for them."""
    for header in (_SAMPLE_HEADER_GPU, _SAMPLE_HEADER_CPU):
        traj = _parse_minimal(tmp_path, header)
        assert _runtime_warnings(traj) == []


def test_unknown_parallelisation_token_warns_but_records(tmp_path):
    """A future ``Parallelisations: hybrid(MPI:8)`` shape is recorded
    verbatim AND flagged -- the audit's headline example."""
    header = (
        "Version : 5.9.0\n"
        "Parallelisations: HYBRID(MPI:8)\n"
        "\n"
        "siesta: System type = molecule\n"
    )
    traj = _parse_minimal(tmp_path, header)
    # Recorded as-is (not dropped, not mangled away).
    assert (traj.runtime_info["siesta_build"]["parallelisations"]
            == ["HYBRID(MPI:8)"])
    # And flagged.
    warns = _runtime_warnings(traj)
    assert warns, "unknown parallelisation token must raise a warning"
    assert "parallelisation" in warns[0].error.lower()
    assert "HYBRID(MPI:8)" in warns[0].error


def test_unknown_diag_algorithm_warns_but_records(tmp_path):
    """A diag-algorithm token outside SIESTA's vocabulary is recorded
    AND flagged (a real SIESTA die()s on unknown algos, so this only
    happens when a newer SIESTA adds an alias we don't track)."""
    header = (
        "Version : 5.9.0\n"
        "Parallelisations: MPI\n"
        "redata: Diagonalization algorithm = quantum-magic\n"
        "\n"
        "siesta: System type = molecule\n"
    )
    traj = _parse_minimal(tmp_path, header)
    assert (traj.runtime_info["siesta_diag"]["algorithm"]
            == "QUANTUM-MAGIC")
    warns = _runtime_warnings(traj)
    assert warns, "unknown diag algorithm must raise a warning"
    assert "diag.algorithm" in warns[0].error.lower()
    assert "QUANTUM-MAGIC" in warns[0].error


def test_uncommon_but_valid_algo_alias_does_not_warn(tmp_path):
    """A legitimate-but-unusual SIESTA alias (e.g. ``vd`` for
    divide-and-conquer, ``vr`` for MRRR) must NOT warn -- the
    vocabulary is the full alias set from diag_option.F90, not just
    the canonical names."""
    for alias in ("vd", "RRR", "elpa-1", "noexpert"):
        header = (
            "Version : 5.4.2\n"
            "Parallelisations: MPI\n"
            f"redata: Diagonalization algorithm = {alias}\n"
            "\n"
            "siesta: System type = molecule\n"
        )
        traj = _parse_minimal(tmp_path, header)
        assert _runtime_warnings(traj) == [], (
            f"valid alias {alias!r} should not warn")
