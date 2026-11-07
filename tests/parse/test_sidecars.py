"""L2 tests for the sidecar FileParsers: what molbuilder's writers write, the
registry claims, under the right parser and schema tag.  A vibration's
`.spectra.json` is asked in `tests/test_spectrum_results_e2e.py`, on a real run's.
"""

from __future__ import annotations

import pytest

from molbuilder.parse import detect


def test_a_structures_sidecar_is_read_by_its_parser(tmp_path):
    """A pair as the writer writes it: its `.molstruct.json` is claimed by the
    sidecar parser and read at the version this build writes.  THE SILENT
    FAILURE: the parser dropped from the registry, so the file is listed with
    `parser: null` and the picker drops it.  `model/parse.md` § 3, § 6.
    """
    from molbuilder.sidecars import molstruct
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    f = tmp_path / "h2.xyz"
    StructureCodec().write(Structure(elements=["H", "H"],
                                     positions=[[0, 0, 0], [0, 0, 0.74]]), f)
    side = molstruct.sidecar_path_for(f)
    kind = detect(str(side))
    assert kind.name == "molstruct-json"
    assert kind.parse(side).schema == f"molstruct/v{molstruct.SCHEMA_VERSION}"


# --------------------------------------------------------------------- #
#  A benchmark sweep's plan                                             #
# --------------------------------------------------------------------- #

def _job_set(tmp_path, kind):
    """A real `job-set.json` of either kind, through `JobSet.write`.

    The fixture goes through the WRITER, so the parser cannot be tested
    against a shape nothing emits.
    """
    from molbuilder.jobset.model import Job, JobSet, Resources

    js = JobSet(name="JOB", engine="siesta", kind=kind, shared=[],
                jobs=[Job(name="p1", script="p1.run.sh",
                          resources=Resources(mpi_np=4))])
    return js.write(tmp_path)


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
    write_task(dest, Task(
        engine="siesta", run=derive_run("T", "J/structure/junc" if transport else "H2",
                       stage_names=stages),
        structure=None if transport
        else StructureRef(source="J/structure/h2.xyz", formula="H2", atoms=2),
        calculation=calculation,
        slots={"junction": "J/structure/junc"} if calculation == "transport"
        else {},
        bias=(0.0, 0.2) if transport else (),
        low_bias_approximation=False if transport else None,
        varies=(), stages=tuple(Stage(name=n, overrides={}) for n in stages)))
    return dest / "task.json"


def test_a_transport_description_is_read_by_a_parser_so_the_root_opens_its_report(tmp_path):
    """`web/results.md` § 0.1, § 2.5: a transport calculation's report is
    composed on read, so the root opens it before any `summarize task` --
    through its description, the one file there from `molbuilder describe` on.
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
    assert got.schema == "transport-task/v2"   # the file's own version
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
