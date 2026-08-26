"""L3 tests for molbuilder.parse.dirs.job.JobDirParser.

Pins docs/execution/running-a-job.md § 4 + docs/model/parse.md
against real project fixtures:

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

# BUILT, NOT FOUND (2026-08-03).
#
# These pointed at two real run directories under projects/ -- the user's
# scientific record -- behind a `pytest.skip("fixture dir absent")`.  Three
# problems in one: the relevance of those runs was never confirmed, they change
# meaning the day they are regenerated, and on any machine without them the
# tests SKIP and the suite still reads green.
#
# The input side is now rendered from a junction defined in source.  The output
# side is a FROZEN SIESTA log from tests/watch/fixtures/siesta_frozen/ -- real
# engine output, checked in beside the tests and reviewed when it changes.  A
# hand-written .out would test a guess at SIESTA's format instead of SIESTA's.


def _single_stage(tmp_path):
    """One .out in the directory."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from support.junction import job_run_dir
    return job_run_dir(tmp_path)


def _multi_stage(tmp_path):
    """Several .out files -- one plot bucket each."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from support.junction import job_run_dir
    return job_run_dir(tmp_path, out_names=(
        "hemeC-stage1-scf_not_conv-5fr.out",
        "hemeC-stage2-run3-finished-42fr.out",
    ))





# Top-level smoke ------------------------------------------------------- #


def test_decode_returns_typed_jobresult(tmp_path):
    """JobDirParser.parse and decode_run_dir both return JobResult,
    not dict."""
    decoded = decode_run_dir(_single_stage(tmp_path))
    assert isinstance(decoded, JobResult)
    assert decoded.result_kind == "job"


def test_parse_dir_dispatches_to_jobdirparser(tmp_path):
    """The public parse_dir() registry call routes to JobDirParser
    for a job directory."""
    decoded = parse_dir(_single_stage(tmp_path))
    assert isinstance(decoded, JobResult)
    assert decoded.parser_name == "job-dir"


def test_decode_carries_every_required_attribute(tmp_path):
    """A JobResult carries every field § 2 requires.

    The label and source assertions used to name a specific captured run
    ("siesta-BDT-Au111-TJ"); they now name what the fixture was BUILT with,
    which is the same assertion without the dependency on somebody's data."""
    run = _single_stage(tmp_path)
    decoded = decode_run_dir(run)
    assert decoded.schema_version == SCHEMA_VERSION
    assert decoded.engine == "siesta"
    assert decoded.system_label == "junction"        # what we rendered it as
    assert decoded.parser_name == "job-dir"
    assert decoded.parsed_at.endswith("Z")
    assert decoded.source.endswith(run.name)


def test_decode_bdt_withaujunction_smoke(tmp_path):
    """BDT-withAuJunction (multi-stage, no script-contract block) is
    a JobResult and classifies as optimization via sniff fallback."""
    decoded = decode_run_dir(_multi_stage(tmp_path))
    assert isinstance(decoded, JobResult)
    assert decoded.schema_version == SCHEMA_VERSION
    assert decoded.engine == "siesta"
    assert decoded.job_type == "optimization"


# job_type classification ---------------------------------------------- #


def test_job_type_classification_script_contract(tmp_path):
    """TJ-BDT-Au111 has the BENCH-MARKS block with MD.NumCGsteps —
    job_type derives as 'optimization' via the script-contract path."""
    decoded = decode_run_dir(_single_stage(tmp_path))
    assert decoded.job_type == "optimization"


def test_job_type_classification_sniff_fallback(tmp_path):
    """BDT-withAuJunction's .fdf predates the script-contract;
    classification falls through to the engine-body sniff and still
    lands on 'optimization'."""
    decoded = decode_run_dir(_multi_stage(tmp_path))
    assert decoded.job_type == "optimization"


def test_job_type_ambiguous_raises(tmp_path):
    """A synthetic .fdf with BOTH MD.Steps > 0 AND a TS.Elec.*
    block raises JobTypeAmbiguousError."""
    from molbuilder.parse.dirs.job import _classify_job_type
    text = (
        "SystemLabel test\n"
        "MD.TypeOfRun CG\n"
        "MD.Steps 200\n"
        "%block TS.Elec.L\n"
        "  HS.files left.TSHS\n"
        "%endblock TS.Elec.L\n"
    )
    with pytest.raises(JobTypeAmbiguousError):
        _classify_job_type(text)


# engine_input envelope ------------------------------------------------ #


def test_engine_input_envelope_per_stage_for_tj_bdt(tmp_path):
    """TJ-BDT-Au111 has one stage1 .fdf; engine_input_by_stage carries
    its envelope keyed by filename."""
    decoded = decode_run_dir(_single_stage(tmp_path))
    ei_map = decoded.engine_input_by_stage
    assert len(ei_map) >= 1
    # Keyed by FILENAME.  This looked for "stage1" -- a property of the
    # captured directory's naming, not of the envelope contract.
    ei = next(iter(ei_map.values()))
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


def test_engine_input_envelope_without_the_script_contract_blocks(tmp_path):
    """A .fdf carrying no script-contract blocks still yields an envelope; the
    *present* flags are just False.

    The blocks are STRIPPED from a generated script.  This used to lean on a
    captured directory happening to predate them -- "a file without a feature"
    sourced from someone's records, which stops being true the day it is
    regenerated."""
    run = _multi_stage(tmp_path)
    fdf = next(run.glob("*.fdf"))
    text = fdf.read_text(encoding="utf-8")
    for begin, end in (("# === molbuilder atom-metadata BEGIN ===",
                        "# === molbuilder atom-metadata END ==="),
                       ("# === molbuilder bench-marks BEGIN ===",
                        "# === molbuilder bench-marks END ==="),
                       ("# === molbuilder provenance BEGIN ===",
                        "# === molbuilder provenance END ===")):
        while begin in text:
            a = text.index(begin)
            b = text.index(end) + len(end)
            text = text[:a] + text[b:]
    fdf.write_text(text, encoding="utf-8")

    decoded = decode_run_dir(run)
    ei_map = decoded.engine_input_by_stage
    assert len(ei_map) >= 1
    sample = next(iter(ei_map.values()))
    assert sample["atom_metadata"]["present"] is False
    assert sample["bench_marks"]["present"] is False
    assert sample["provenance"]["present"] is False
    assert isinstance(sample["engine_body_summary"], dict)


def test_engine_body_summary_curated_keys_are_exact(tmp_path):
    """The engine_body_summary contains EXACTLY the curated key list
    — no more, no less."""
    decoded = decode_run_dir(_single_stage(tmp_path))
    ei_map = decoded.engine_input_by_stage
    summary = next(iter(ei_map.values()))["engine_body_summary"]
    assert set(summary.keys()) == set(ENGINE_BODY_KEYS)


def test_engine_body_summary_raw_string_values(tmp_path):
    """Values are raw strings (with units, dots, etc.) — no
    interpretation."""
    decoded = decode_run_dir(_single_stage(tmp_path))
    ei_map = decoded.engine_input_by_stage
    summary = next(iter(ei_map.values()))["engine_body_summary"]
    assert summary["XC.functional"] == "GGA"
    assert summary["XC.authors"] == "PBE"
    assert summary["SolutionMethod"] == "diagon"
    assert summary["Diag.Algorithm"] == "ELPA-2STAGE"
    assert summary["MeshCutoff"] and "Ry" in summary["MeshCutoff"]


def test_engine_body_summary_kgrid_block_extraction(tmp_path):
    """kgrid_Monkhorst_Pack from a %block reduces to 'AxBxC' form."""
    decoded = decode_run_dir(_single_stage(tmp_path))
    ei_map = decoded.engine_input_by_stage
    summary = next(iter(ei_map.values()))["engine_body_summary"]
    assert summary["kgrid_Monkhorst_Pack"] == "1x1x1"


# multi-source consolidation ------------------------------------------- #


def test_multistage_plot_buckets_keyed_by_out_filename(tmp_path):
    """BDT-withAuJunction has multiple .out files.  Each appears as
    its own plot bucket keyed by literal filename."""
    decoded = decode_run_dir(_multi_stage(tmp_path))
    etot = decoded.plots["etot_per_cg"]
    assert len(etot) >= 1
    for key in etot.keys():
        assert key.endswith(".out")
        assert (_multi_stage(tmp_path) / key).is_file()


def test_source_files_index_has_fdf_and_out(tmp_path):
    """source_files lists every .fdf and .out with stage / mtime /
    size."""
    decoded = decode_run_dir(_multi_stage(tmp_path))
    kinds = {entry["kind"] for entry in decoded.source_files}
    assert "fdf" in kinds
    assert "out" in kinds
    for entry in decoded.source_files:
        for fld in ("path", "kind", "stage", "mtime", "size_bytes"):
            assert fld in entry, f"source_files entry missing {fld!r}"


# geometry ------------------------------------------------------------- #


def test_geometry_carries_xyz_and_cell(tmp_path):
    """TJ-BDT-Au111 has a .XV — geometry reads from it and carries xyz + cell.

    THE LABELS ARE NO LONGER ASSERTED HERE, and that is the strict-version
    policy showing up rather than a regression. This directory's `.fdf` was
    generated before v7, so its ATOM-METADATA block is refused rather than read
    (structure-molstruct.md § 2): an older block keeps the same facts in
    different places, and reading it hands back labels that look complete and
    are missing the frozen set.

    The emit/parse contract for labels is covered where it belongs, against a
    run directory BUILT from a known structure rather than a captured one:
    `tests/parse/dirs/test_bundle.py` and `tests/test_fdf_generator_roundtrip.py`.
    This test keeps what a captured directory can still honestly prove -- that
    a real `.XV` is found and read.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from support.junction import N_ATOMS

    decoded = decode_run_dir(_single_stage(tmp_path))
    geom = decoded.geometry
    assert geom["n_atoms"] == N_ATOMS
    assert len(geom["xyz"]) == N_ATOMS
    assert geom["cell"] is not None
    assert isinstance(geom["regions"], dict)
    assert geom["coords_state"] in ("converged", "initial")


# status + progress ---------------------------------------------------- #


def test_status_shape(tmp_path):
    decoded = decode_run_dir(_multi_stage(tmp_path))
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
    # run_state="stopped".
    (tmp_path / "crash.out").write_text(
        "                           Welcome to SIESTA\n"
        "reinit: System Label: crash\n"
        "siesta: ERROR: out of memory in dense solver\n"
    )
    decoded = decode_run_dir(tmp_path)
    assert decoded.status["state"] == "failed"
    assert decoded.status["active_source"] == "crash.out"


def test_progress_carries_cg_step_count(tmp_path):
    decoded = decode_run_dir(_multi_stage(tmp_path))
    prog = decoded.progress
    for fld in (
        "current_cg_step", "target_cg_steps", "current_scf_iter_global",
        "last_iter_wall_s", "mean_iter_wall_s",
        "estimated_remaining_s", "stages_completed",
        "stages_total_known",
    ):
        assert fld in prog


# Frozen-dataclass invariant ------------------------------------------ #


def test_jobresult_is_frozen(tmp_path):
    """ParseResult subclasses (incl. JobResult) are frozen — mutation
    raises FrozenInstanceError per parse-module.md § 9 forbidden #4."""
    decoded = decode_run_dir(_single_stage(tmp_path))
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


# --------------------------------------------------------------------- #
#  Result files are engine-neutral: the molwatch conclusion counts      #
# --------------------------------------------------------------------- #
# running-a-job.md § 4: state is derived over the directory's RESULT
# files -- every .out, plus each molwatch log whose footer concludes the
# run.  A PySCF attempt has no .out at all, so the molwatch conclusion
# is its ONLY end-of-run marker (found 2026-08-19: a finished PySCF
# attempt read "running -- no .out file yet" forever).


def _mw_log(dirpath, name, *, concluded):
    body = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "# job: w\n"
        "# units: energy=eV, force=eV/Ang, coords=Ang\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "kind: initial_preview\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H       0.0 0.0 0.0\n"
        "==== molwatch step 0 end ====\n"
    )
    if concluded:
        body += "\n# concluded: 2026-08-19T12:00:00\n"
    p = Path(dirpath) / name
    p.write_text(body)
    return p


def test_a_concluded_molwatch_log_finishes_an_attempt_with_no_out(tmp_path):
    """A PySCF attempt: no .out ever exists, the concluded molwatch log
    is the result file, and the verb's one job is to say finished."""
    _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=True)
    res = parse_dir(tmp_path)
    assert res.status["state"] == "finished"
    assert res.status["detail"] == "job_completed"


def test_a_seed_molwatch_log_is_a_live_view_not_a_result(tmp_path):
    """The prep-time seed has no conclusion footer, so it contributes
    nothing: the attempt reads as running with no result yet -- never as
    finished, and never as a state the seed's mtime could steer."""
    _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=False)
    res = parse_dir(tmp_path)
    assert res.status["state"] == "running"
    assert res.status["detail"] == "no result file yet"
