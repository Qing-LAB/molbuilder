"""L3 tests for molbuilder.parse.dirs.job.JobDirParser.

Pins docs/protocols/job-decoder.md § 2-7 + docs/protocols/parse-
module.md § 7 against real project fixtures:

* TJ-BDT-Au111 — single-stage, script-contract block present
  (post-2026-06-16 .fdf).  Exercises the authoritative job_type
  classification path.
* BDT-withAuJunction — multi-stage (stage1 + stage3), 9 .out
  files, no script-contract block (pre-2026-06-16 .fdf).
  Exercises the sniff-fallback + multi-source consolidation paths.

Phase B of parse-module migration: JobDirParser returns a frozen
JobResult dataclass, not a dict.  Tests use attribute access
(decoded.plots) rather than dict access (decoded["plots"]).
"""

from __future__ import annotations

import pytest
from pathlib import Path

from molbuilder.parse import JobResult, parse_dir
from molbuilder.parse.dirs.job import (
    SCHEMA_VERSION,
    ENGINE_BODY_KEYS,
    JobTypeAmbiguousError,
    JobDirParser,
    decode_run_dir,
)


# Fixture dirs ---------------------------------------------------------- #

PROJECTS_ROOT = Path("/home/qqing/molbuilder/projects/BDT/optimization")
TJ_DIR  = PROJECTS_ROOT / "TJ-BDT-Au111"
BDT_DIR = PROJECTS_ROOT / "BDT-withAuJunction"


def _need_dir(p: Path) -> Path:
    if not p.is_dir():
        pytest.skip(f"fixture dir absent: {p}")
    return p


# Top-level smoke ------------------------------------------------------- #


def test_decode_returns_typed_jobresult():
    """JobDirParser.parse and decode_run_dir both return JobResult,
    not dict."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    assert isinstance(decoded, JobResult)
    assert decoded.result_kind == "job"


def test_parse_dir_dispatches_to_jobdirparser():
    """The public parse_dir() registry call routes to JobDirParser
    for a job directory."""
    decoded = parse_dir(_need_dir(TJ_DIR))
    assert isinstance(decoded, JobResult)
    assert decoded.parser_name == "job-dir"


def test_decode_tj_bdt_au111_smoke():
    """JobResult on TJ-BDT-Au111 carries every required attribute
    per § 2."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    assert decoded.schema_version == SCHEMA_VERSION
    assert decoded.engine == "siesta"
    assert decoded.system_label == "siesta-BDT-Au111-TJ"
    # Top-level ParseResult fields
    assert decoded.parser_name == "job-dir"
    assert decoded.parsed_at.endswith("Z")
    assert decoded.source.endswith("TJ-BDT-Au111")


def test_decode_bdt_withaujunction_smoke():
    """BDT-withAuJunction (multi-stage, no script-contract block) is
    a JobResult and classifies as optimization via sniff fallback."""
    decoded = decode_run_dir(_need_dir(BDT_DIR))
    assert isinstance(decoded, JobResult)
    assert decoded.schema_version == SCHEMA_VERSION
    assert decoded.engine == "siesta"
    assert decoded.job_type == "optimization"


# job_type classification ---------------------------------------------- #


def test_job_type_classification_script_contract():
    """TJ-BDT-Au111 has the BENCH-MARKS block with MD.NumCGsteps —
    job_type derives as 'optimization' via the script-contract path."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    assert decoded.job_type == "optimization"


def test_job_type_classification_sniff_fallback():
    """BDT-withAuJunction's .fdf predates the script-contract;
    classification falls through to the engine-body sniff and still
    lands on 'optimization'."""
    decoded = decode_run_dir(_need_dir(BDT_DIR))
    assert decoded.job_type == "optimization"


def test_job_type_ambiguous_raises(tmp_path):
    """A synthetic .fdf with BOTH MD.NumCGsteps > 0 AND a TS.Elec.*
    block raises JobTypeAmbiguousError."""
    from molbuilder.parse.dirs.job import _classify_job_type
    text = (
        "SystemLabel test\n"
        "MD.TypeOfRun CG\n"
        "MD.NumCGsteps 200\n"
        "%block TS.Elec.L\n"
        "  HS.files left.TSHS\n"
        "%endblock TS.Elec.L\n"
    )
    with pytest.raises(JobTypeAmbiguousError):
        _classify_job_type(text)


# engine_input envelope ------------------------------------------------ #


def test_engine_input_envelope_per_stage_for_tj_bdt():
    """TJ-BDT-Au111 has one stage1 .fdf; engine_input_by_stage carries
    its envelope keyed by filename."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    ei_map = decoded.engine_input_by_stage
    assert len(ei_map) >= 1
    key = next(k for k in ei_map if "stage1" in k)
    ei = ei_map[key]
    for fld in (
        "schema_version", "engine", "source_fdf",
        "header", "provenance", "bench_marks",
        "atom_metadata", "user_custom_verbatim",
        "engine_body_summary",
    ):
        assert fld in ei, f"engine_input missing {fld!r}"
    assert ei["engine"] == "siesta"
    assert ei["atom_metadata"]["present"] is True
    assert ei["bench_marks"]["present"] is True
    assert ei["provenance"]["present"] is True


def test_engine_input_envelope_no_script_contract_for_bdt():
    """BDT-withAuJunction's .fdf has no script-contract block.  The
    envelope still parses; the *present* flags are False."""
    decoded = decode_run_dir(_need_dir(BDT_DIR))
    ei_map = decoded.engine_input_by_stage
    assert len(ei_map) >= 1
    sample = next(iter(ei_map.values()))
    assert sample["atom_metadata"]["present"] is False
    assert sample["bench_marks"]["present"] is False
    assert sample["provenance"]["present"] is False
    assert isinstance(sample["engine_body_summary"], dict)


def test_engine_body_summary_curated_keys_are_exact():
    """The engine_body_summary contains EXACTLY the curated key list
    — no more, no less."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    ei_map = decoded.engine_input_by_stage
    summary = next(iter(ei_map.values()))["engine_body_summary"]
    assert set(summary.keys()) == set(ENGINE_BODY_KEYS)


def test_engine_body_summary_raw_string_values():
    """Values are raw strings (with units, dots, etc.) — no
    interpretation."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    ei_map = decoded.engine_input_by_stage
    summary = next(iter(ei_map.values()))["engine_body_summary"]
    assert summary["XC.functional"] == "GGA"
    assert summary["XC.authors"] == "PBE"
    assert summary["SolutionMethod"] == "diagon"
    assert summary["Diag.Algorithm"] == "ELPA-2STAGE"
    assert summary["MeshCutoff"] and "Ry" in summary["MeshCutoff"]


def test_engine_body_summary_kgrid_block_extraction():
    """kgrid_Monkhorst_Pack from a %block reduces to 'AxBxC' form."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    ei_map = decoded.engine_input_by_stage
    summary = next(iter(ei_map.values()))["engine_body_summary"]
    assert summary["kgrid_Monkhorst_Pack"] == "1x1x1"


# multi-source consolidation ------------------------------------------- #


def test_multistage_plot_buckets_keyed_by_out_filename():
    """BDT-withAuJunction has multiple .out files.  Each appears as
    its own plot bucket keyed by literal filename."""
    decoded = decode_run_dir(_need_dir(BDT_DIR))
    etot = decoded.plots["etot_per_cg"]
    assert len(etot) >= 1
    for key in etot.keys():
        assert key.endswith(".out")
        assert (BDT_DIR / key).is_file()


def test_source_files_index_has_fdf_and_out():
    """source_files lists every .fdf and .out with stage / mtime /
    size."""
    decoded = decode_run_dir(_need_dir(BDT_DIR))
    kinds = {entry["kind"] for entry in decoded.source_files}
    assert "fdf" in kinds
    assert "out" in kinds
    for entry in decoded.source_files:
        for fld in ("path", "kind", "stage", "mtime", "size_bytes"):
            assert fld in entry, f"source_files entry missing {fld!r}"


# geometry ------------------------------------------------------------- #


def test_geometry_carries_xyz_and_cell():
    """TJ-BDT-Au111 has a .XV — geometry reads from it and carries
    xyz + cell + regions + frozen_atoms."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    geom = decoded.geometry
    assert geom["n_atoms"] == 444
    assert len(geom["xyz"]) == 444
    assert geom["cell"] is not None
    assert isinstance(geom["regions"], dict)
    assert len(geom["regions"]) >= 3
    assert "L-electrode" in geom["regions"]
    assert geom["coords_state"] in ("converged", "initial")


# status + progress ---------------------------------------------------- #


def test_status_shape():
    decoded = decode_run_dir(_need_dir(BDT_DIR))
    s = decoded.status
    assert s["state"] in ("running", "stale", "finished", "failed")
    for fld in ("state", "detail", "last_change_at", "active_source"):
        assert fld in s


def test_status_failed_state_from_error_run(tmp_path):
    """A run whose active .out carries a SIESTA fatal marker decodes
    to status.state == "failed".

    Regression (2026-07-27): _build_status branched on run_state ==
    "failed", a value NO engine parser emits (their vocabulary is
    ongoing|finished|error|unknown), so the failed state was
    unreachable and a crashed run reported "stale"/"running"
    forever.  The parser "error" state must map to the envelope
    "failed" state."""
    (tmp_path / "crash.fdf").write_text(
        "SystemLabel crash\n"
        "MD.TypeOfRun CG\n"
        "MD.NumCGsteps 100\n"
    )
    # Minimal but REAL SIESTA .out: the banner is a strong sniff
    # marker (SiestaOutFileParser.can_parse), the last line is one
    # of the registered fatal markers (engines/siesta.py) that sets
    # run_state="error".
    (tmp_path / "crash.out").write_text(
        "                           Welcome to SIESTA\n"
        "reinit: System Label: crash\n"
        "siesta: ERROR: out of memory in dense solver\n"
    )
    decoded = decode_run_dir(tmp_path)
    assert decoded.status["state"] == "failed"
    assert decoded.status["active_source"] == "crash.out"


def test_progress_carries_cg_step_count():
    decoded = decode_run_dir(_need_dir(BDT_DIR))
    prog = decoded.progress
    for fld in (
        "current_cg_step", "target_cg_steps", "current_scf_iter_global",
        "last_iter_wall_s", "mean_iter_wall_s",
        "estimated_remaining_s", "stages_completed",
        "stages_total_known",
    ):
        assert fld in prog


# Frozen-dataclass invariant ------------------------------------------ #


def test_jobresult_is_frozen():
    """ParseResult subclasses (incl. JobResult) are frozen — mutation
    raises FrozenInstanceError per parse-module.md § 9 forbidden #4."""
    decoded = decode_run_dir(_need_dir(TJ_DIR))
    with pytest.raises(Exception):       # dataclasses.FrozenInstanceError
        decoded.job_type = "spectrum"    # noqa


# Forbidden-pattern lint test ----------------------------------------- #


def test_no_direct_out_grep_in_decoder():
    """The JobDirParser module must NOT contain regex/grep calls
    against .out content.  Enforces job-decoder.md § 9 forbidden #2
    mechanically.

    Allowed: filesystem-level checks (mtime / size / glob).
    Forbidden: re.search against .out body — must go through
    detect_parser().parse() instead.
    """
    src = (
        Path(__file__).resolve().parents[3]
        / "molbuilder" / "parse" / "dirs" / "job.py"
    ).read_text()
    assert "out_path.read_text(" not in src, (
        "job.py opens .out content directly; must go through "
        "detect_parser().parse()"
    )
    assert src.count("read_text") < 8, (
        "job.py has too many read_text calls; should only read "
        ".fdf bodies, not other engine outputs"
    )
