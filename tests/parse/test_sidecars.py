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

def _plans(tmp_path, monkeypatch):
    """The road's H2 (`support.road`), its benchmark prepared and its first
    group: ``(the benchmark's plan, in its stage's `bench/` folder; the
    calculation's ladder, `job-set.json` at its root)`` -- as `prep bench`
    and `prep task` write them, so the parser is never tested against a
    file nothing writes, nor a place nothing puts one."""
    from support.road import (declare_bench, described, jobset, prep_groups,
                              road_tree)
    calc = described(road_tree(tmp_path, monkeypatch))
    declare_bench(calc, {"mpi_np": [2], "omp_threads": [1]})
    r = jobset("prep", "bench", "coarse", "--bundle", calc, "--target",
               "this")
    assert r.exit_code == 0, r.output
    prep_groups(calc, 1)
    return calc / "01_coarse" / "bench" / "job-set.json", calc / "job-set.json"


def test_a_sweep_is_read_by_a_parser_so_the_door_can_offer_it(tmp_path,
                                                              monkeypatch):
    """The bench viewer was unreachable from the Results tab for a day.

    `/api/results/dir` sends each file the registry's verdict and the
    picker drops anything with `parser: null` -- so when nothing claimed
    `job-set.json`, a sweep directory listed as *"no result files yet"*
    and `bench-summary.js` was never consulted.  Measured 2026-09-19 on
    `projects/AuSlab/.../01_coarse/bench`, whose only two files are the
    plan and `STAGE-PLAN.md`.

    The e2e suite could not catch it: its `_mount` helper calls
    `inspectors.pick(path)` directly, which is the one path that skips
    the picker's gate.  This asserts the gate, on the plan `prep bench`
    writes in its benchmark's folder (`model/parse.md` § 6: claimed by its
    name and its place).
    """
    plan, _ladder = _plans(tmp_path, monkeypatch)
    kind = detect(str(plan))
    assert kind.name == "job-set-sweep"
    got = kind.parse(plan)
    assert got.schema == "job-set/v1"
    assert got.payload["kind"] == "sweep"
    assert got.payload["jobs"], got.payload


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


def test_an_ordinary_ladder_is_refused_though_the_filename_matches(
        tmp_path, monkeypatch):
    """The PLACE, not the name alone (`model/parse.md` § 6).

    A calculation's stage ladder is also called `job-set.json`, and its
    `/api/bench/summary` answers 400 because there is no sweep to
    summarise.  Offering it would mount the bench viewer on a refusal.
    It is declined because it lies at the calculation's root, no
    benchmark's folder (`calcdirs.where`) -- never by its content, which
    would hide a benchmark's plan of another version as no parser's.

    MUTATION THIS MUST FAIL AGAINST: drop the place test from
    `JobSetSweepFileParser.can_parse`, keeping only the name.
    """
    from molbuilder.parse.errors import UnknownFormatError

    _plan, ladder = _plans(tmp_path, monkeypatch)
    assert ladder.is_file(), ladder
    with pytest.raises(UnknownFormatError):
        detect(str(ladder))
