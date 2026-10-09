"""L2 tests for the sidecar FileParsers.

Pins:
  * Each sidecar parser is registered + claims its filename
    suffix.
  * parse() returns a SidecarResult with the right schema tag.
  * detect() routes to the right sidecar parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import SidecarResult, detect, parse
from molbuilder.parse.sidecars import (
    MolstructSidecarFileParser,
    SpectraSidecarFileParser,
)
from molbuilder.parse.registry import _registered_file_parsers


REPO = Path(__file__).resolve().parents[2]
MOLSTRUCT_FX = REPO / "tests" / "data" / "au_bdt_au.molstruct.json"


def _need(p: Path) -> Path:
    """Assert the fixture is there.

    Every fixture it guards is COMMITTED under tests/ -- so absence means a
    broken checkout or a deleted file, and a missing committed fixture is a
    failure, loudly.
    """
    assert p.exists(), (
        f"committed fixture missing: {p}.  It is versioned with these tests; "
        f"a checkout without it is broken, not a reason to skip.")
    return p


# Registration --------------------------------------------------------- #


def test_sidecar_parsers_registered():
    names = {p.name for p in _registered_file_parsers()}
    assert "molstruct-json" in names
    assert "spectra-json"   in names


def test_molstruct_parser_claims_suffix():
    assert MolstructSidecarFileParser.can_parse(_need(MOLSTRUCT_FX))
    assert not MolstructSidecarFileParser.can_parse(REPO / "README.md")


def test_spectra_parser_claims_suffix(tmp_path):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from support.junction import spectra_sidecar
    assert SpectraSidecarFileParser.can_parse(
        spectra_sidecar(tmp_path / "built.spectra.json"))


def test_detect_routes_to_molstruct_parser():
    cls = detect(_need(MOLSTRUCT_FX))
    assert cls is MolstructSidecarFileParser


# Parse + payload + schema -------------------------------------------- #


def test_parse_molstruct_returns_sidecarresult():
    result = parse(_need(MOLSTRUCT_FX))
    assert isinstance(result, SidecarResult)
    assert result.result_kind == "sidecar"
    assert result.schema.startswith("molstruct/v")
    assert result.parser_name == "molstruct-json"
    assert isinstance(result.payload, dict)
    # The schema tag matches what the payload declares
    sv_in_payload = result.payload.get("schema_version")
    if sv_in_payload is not None:
        assert result.schema == f"molstruct/v{sv_in_payload}"


def test_parse_spectra_returns_sidecarresult_with_payload(tmp_path):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from support.junction import spectra_sidecar
    result = parse(spectra_sidecar(tmp_path / "built.spectra.json"))
    assert isinstance(result, SidecarResult)
    assert result.schema.startswith("spectra/v")
    assert "schema_version" in result.payload


def test_sidecar_result_is_frozen():
    result = parse(_need(MOLSTRUCT_FX))
    with pytest.raises(Exception):
        result.schema = "tampered/v0"   # noqa


# --------------------------------------------------------------------- #
#  A benchmark sweep's plan                                             #
# --------------------------------------------------------------------- #

def _job_set(tmp_path, kind):
    """A real `job-set.json` of either kind, through `JobSet.write`.

    Same principle as `_record` above: the fixture goes through the
    WRITER, so the parser cannot be tested against a shape nothing emits.
    """
    from molbuilder.jobset.model import Job, JobSet, Resources

    js = JobSet(name="JOB", engine="siesta", kind=kind, shared=[],
                jobs=[Job(name="p1", script="p1.run.sh",
                          resources=Resources(mpi_np=4))])
    return js.write(tmp_path / "job-set.json")


def test_a_sweep_is_read_by_a_parser_so_the_door_can_offer_it(tmp_path):
    """The bench viewer was unreachable from the Results tab for a day.

    `/api/results/dir` sends each file the registry's verdict and the
    picker drops anything with `parser: null` -- so when nothing claimed
    `job-set.json`, a sweep directory listed as *"no result files yet"*
    and `bench-summary.js` was never consulted.  Measured 2026-09-19 on
    `projects/AuSlab/.../01_coarse/bench`, whose only two files are the
    plan and `STAGE-PLAN.md`.

    The e2e suite could not catch it: its `_mount` helper calls
    `inspectors.pick(path)` directly, which is the one path that skips
    the picker's gate.  This asserts the gate.
    """
    kind = detect(str(_job_set(tmp_path, "sweep")))
    assert kind.name == "job-set-sweep"
    got = kind.parse(_job_set(tmp_path, "sweep"))
    assert got.schema == "job-set/v1"
    assert got.payload["kind"] == "sweep"
    assert got.payload["jobs"][0]["name"] == "p1"


def _task(tmp_path, calculation):
    """A real `task.json` of the given kind, through `write_task` -- the
    writer, so the parser is never tested against a shape nothing emits."""
    from molbuilder.task import (Stage, StructureRef, Task, derive_run,
                                 write_task)
    transport = calculation == "transport"
    stages = (("seed", "electrode_L", "electrode_R", "device", "transmission")
              if transport else ("coarse",))
    dest = tmp_path / calculation
    dest.mkdir(exist_ok=True)
    write_task(dest / "task.json", Task(
        engine="siesta", shape="hierarchical",
        run=derive_run("T", "J/structure/junc" if transport else "H2",
                       stage_names=stages),
        structure=None if transport
        else StructureRef(source="J/structure/h2.xyz", formula="H2", atoms=2),
        calculation=calculation,
        slots={"junction": "J/structure/junc"} if calculation == "transport"
        else {},
        bias=(0.0, 0.2) if transport else (),
        low_bias_approximation=False if transport else None,
        varies=(), execution={"mpi_np": 4, "omp_threads": 1},
        stages=tuple(Stage(name=n, overrides={}) for n in stages)))
    return dest / "task.json"


def test_a_transport_description_is_read_by_a_parser_so_the_root_opens_its_report(tmp_path):
    """`web/results.md` § 0.1, § 2.5: a transport calculation's report is
    composed on read, so the root opens it before any `summarize task` --
    through its description, the one file there from `jobset init` on.
    The picker drops a file no parser claims, so `task.json` needs the
    registry's word; and only a TRANSPORT calculation's, read off the
    description's own `calculation`, never the name -- a relaxation's root
    opens no report.

    Silent before this: a transport ladder in progress listed as *no result
    files yet* until its first summarize, against § 2.5's rule.
    """
    from molbuilder.parse.errors import UnknownFormatError

    kind = detect(str(_task(tmp_path, "transport")))
    assert kind.name == "transport-task"
    got = kind.parse(_task(tmp_path, "transport"))
    assert got.schema == "transport-task/v1"
    assert got.payload["calculation"] == "transport"
    with pytest.raises(UnknownFormatError):
        detect(str(_task(tmp_path, "optimization")))


def test_an_ordinary_ladder_is_refused_though_the_filename_matches(tmp_path):
    """The DISCRIMINATOR, not the name.

    A calculation's stage ladder is also called `job-set.json`, and its
    `/api/bench/summary` answers 400 because there is no sweep to
    summarise.  Offering it would mount the bench viewer on a refusal.
    It is declined for the reason stated on disk (`kind: ladder`) rather
    than by the whole name being excluded, which would take the sweep with
    it.

    MUTATION THIS MUST FAIL AGAINST: drop the `kind` test from
    `_load_sweep`, keeping only the schema check.
    """
    from molbuilder.parse.errors import UnknownFormatError

    with pytest.raises(UnknownFormatError):
        detect(str(_job_set(tmp_path, "ladder")))
