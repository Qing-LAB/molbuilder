"""Unit tests for molbuilder.bench."""
from __future__ import annotations

from pathlib import Path
import pytest

from molbuilder import bench
from molbuilder.bench import (
    DEFAULT_POINTS, DIAG_1STAGE, DIAG_2STAGE, FieldDecl, Point,
    disable_md, force_max_scf_iters,
    override_field_value, parse_bench_marks,
)


# --------------------------------------------------------------------- #
#  Point parsing + slug                                                  #
# --------------------------------------------------------------------- #


def test_point_parse_plain_triplet():
    p = Point.parse("4,2,64")
    assert p.np == 4 and p.omp == 2 and p.bs == 64
    assert p.diag == DIAG_1STAGE
    assert p.pin is True
    assert p.slug == "np4_omp2_bz64_1s_p1"


def test_point_parse_keyed_triplet():
    p = Point.parse("np=10,omp=1,bs=64")
    assert p.np == 10 and p.omp == 1 and p.bs == 64
    assert p.diag == DIAG_1STAGE
    assert p.pin is True


def test_point_parse_plain_quadruplet_diag_alias():
    p = Point.parse("4,2,64,2s")
    assert p.diag == DIAG_2STAGE
    assert p.pin is True
    assert p.slug == "np4_omp2_bz64_2s_p1"


def test_point_parse_keyed_quadruplet():
    p = Point.parse("np=4,omp=2,bs=64,diag=ELPA-2STAGE")
    assert p.diag == DIAG_2STAGE
    assert p.pin is True


def test_point_parse_plain_quintuplet_pin_alias():
    p = Point.parse("20,1,64,1s,nopin")
    assert p.pin is False
    assert p.slug == "np20_omp1_bz64_1s_p0"


def test_point_parse_keyed_quintuplet():
    p = Point.parse("np=10,omp=2,bs=64,diag=2s,pin=false")
    assert p.np == 10 and p.omp == 2 and p.pin is False


def test_point_parse_rejects_bad_arity():
    with pytest.raises(ValueError):
        Point.parse("4,2")


def test_point_parse_rejects_bad_pin():
    with pytest.raises(ValueError):
        Point.parse("4,2,64,1s,bogus")


def test_default_points_match_design():
    """Pin the 18-point default sweep (9 base shapes × ELPA-1/2STAGE).

    Run-order: most aggressive first (all-cores, biggest blocks).
    """
    quints = [(p.np, p.omp, p.bs, p.diag, p.pin) for p in DEFAULT_POINTS]
    base_shapes = [
        # all-physical-cores cross-socket (pin=False)
        (20, 1,  64, False),
        (10, 2,  64, False),
        # max single-socket, biggest blocks (pin=True)
        (10, 1, 256, True),
        (10, 1, 128, True),
        (10, 1,  64, True),
        # 4-ranks-per-GPU anchor, big-to-small block
        (4, 2,  128, True),
        (4, 2,   64, True),
        (4, 2,   32, True),
        # host-bound stress test
        (2, 5,   64, True),
    ]
    expected = [
        (np, omp, bs, diag, pin)
        for (np, omp, bs, pin) in base_shapes
        for diag in (DIAG_1STAGE, DIAG_2STAGE)
    ]
    assert quints == expected
    # 9 shapes × 2 diag = 18.
    assert len(DEFAULT_POINTS) == 18


# --------------------------------------------------------------------- #
#  parse_bench_marks                                                    #
# --------------------------------------------------------------------- #


_SAMPLE_FDF = """\
# === molbuilder bench-marks BEGIN ===
#   version v1
#   n_atoms             212
#   n_orbitals_est      2700
#   gpu_mode            true
#
#   field BlockSize         anchor=BlockSize         type=pow2  range=[16,256]  default=256
#   field MaxSCFIterations  anchor=MaxSCFIterations  type=int   default=500
#   field MD.NumCGsteps     anchor=MD.NumCGsteps     type=int   default=200
#   field MeshCutoff        anchor=MeshCutoff        type=float  unit=Ry  default=400.0
# === molbuilder bench-marks END ===

BlockSize          256
MaxSCFIterations   500
MD.NumCGsteps      200
MeshCutoff         400.0 Ry
"""


def test_parse_bench_marks_extracts_version_and_metadata():
    marks = parse_bench_marks(_SAMPLE_FDF)
    assert marks is not None
    assert marks["version"] == "v1"
    assert marks["metadata"]["n_atoms"] == "212"
    assert marks["metadata"]["gpu_mode"] == "true"


def test_parse_bench_marks_extracts_field_decls():
    marks = parse_bench_marks(_SAMPLE_FDF)
    by_name = {f.name: f for f in marks["fields"]}
    assert by_name["BlockSize"].anchor == "BlockSize"
    assert by_name["BlockSize"].type_ == "pow2"
    assert by_name["BlockSize"].range_ == (16.0, 256.0)
    assert by_name["BlockSize"].default == "256"
    assert by_name["MeshCutoff"].unit == "Ry"
    assert by_name["MeshCutoff"].default == "400.0"


def test_parse_bench_marks_returns_none_when_block_missing():
    assert parse_bench_marks("BlockSize 64\nSystemName foo\n") is None


# --------------------------------------------------------------------- #
#  Engine-body edits                                                    #
# --------------------------------------------------------------------- #


def test_override_field_value_replaces_only_first_match():
    text = (
        "# comment line BlockSize\n"
        "BlockSize          256\n"
        "MeshCutoff         400.0 Ry\n"
    )
    out = override_field_value(text, "BlockSize", "32")
    assert "BlockSize          32" in out
    assert "256" not in out
    # comment line untouched
    assert "# comment line BlockSize\n" in out


def test_force_max_scf_iters_replaces_existing_line():
    text = "MaxSCFIterations  500\nBlockSize 64\n"
    out = force_max_scf_iters(text, 5)
    assert "MaxSCFIterations          5" in out
    assert "MaxSCFIterations  500" not in out


def test_force_max_scf_iters_appends_when_missing():
    text = "BlockSize 64\n"
    out = force_max_scf_iters(text, 5)
    assert "MaxSCFIterations  5" in out


def test_disable_md_zeros_cg_steps():
    text = "MD.NumCGsteps  200\nBlockSize 64\n"
    out = disable_md(text)
    assert "MD.NumCGsteps          0" in out


def test_disable_md_handles_alternate_relax_keyword():
    text = "MD.NumBroydenSteps  100\nBlockSize 64\n"
    out = disable_md(text)
    assert "MD.NumBroydenSteps          0" in out


def test_strip_numa_pin_clobbers_exact_literal():
    """Match the exact assignment line runwrap.py writes today."""
    text = (
        '_n_sockets=$(grep -c "^physical id" /proc/cpuinfo)\n'
        'if [ "$_gpu_numa" != "unknown" ] && [ "$_n_sockets" -ge 2 ]; then\n'
        '    _numa_wrap_gpu="numactl --cpunodebind=$_gpu_numa --membind=$_gpu_numa"\n'
        'fi\n'
        '$_numa_wrap_gpu mpirun -np 4 siesta\n'
    )
    out = bench._strip_numa_pin(text)
    assert 'numactl --cpunodebind' not in out
    assert '_numa_wrap_gpu=""' in out
    # mpirun launch line preserved unchanged.
    assert '$_numa_wrap_gpu mpirun -np 4 siesta' in out


def test_strip_numa_pin_is_idempotent_when_already_empty():
    """When the wrapper already has no numactl pin, leave it alone."""
    text = (
        '_numa_wrap_gpu=""\n'
        '$_numa_wrap_gpu mpirun -np 4 siesta\n'
    )
    out = bench._strip_numa_pin(text)
    assert out == text


# --------------------------------------------------------------------- #
#  Parse SIESTA .out                                                    #
# --------------------------------------------------------------------- #


_SAMPLE_OUT = """\
some pre-banner text
* ProcessorY, Blocksize:    2  64
molbuilder: chosen 4 ranks × 2 threads = 8 of 19 budget cores
diag: Algorithm                                = ELPA-2stage
some setup
   scf:    1  -100.0  -101.0  -101.0  1.0 -5.0 100.0
timer: Routine,Calls,Time,% = IterSCF        1      45.225  37.28
   scf:    2  -102.0  -103.0  -103.0  0.5 -5.0  20.0
   scf:    3  -103.0  -104.0  -104.0  0.1 -5.0   5.0
"""


def test_parse_point_out_extracts_effective_values(tmp_path):
    out = tmp_path / "test.out"
    out.write_text(_SAMPLE_OUT, encoding="utf-8")
    parsed = bench.parse_point_out(out)
    assert parsed["iters_done"] == 3
    assert parsed["first_iter_s"] == 45.225
    assert parsed["effective_np"] == 4
    assert parsed["effective_omp"] == 2
    assert parsed["effective_bs"] == 64
    assert parsed["effective_diag"] == "ELPA-2stage"


def test_parse_point_out_missing_file_returns_zero(tmp_path):
    parsed = bench.parse_point_out(tmp_path / "does-not-exist.out")
    assert parsed == {"iters_done": 0}
