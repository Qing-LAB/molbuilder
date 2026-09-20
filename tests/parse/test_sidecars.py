"""L2 tests for Phase D sidecar FileParsers.

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
# NO PATH INTO projects/.  The spectra fixture was
# projects/BDT/spectrum/BDT-only/spectra.spectra.json -- the user's scientific
# record -- behind a `pytest.skip("fixture absent")`, which is the dangerous
# half: on a machine without that run the test SKIPS and the suite still reads
# green.  It is now WRITTEN by the application's own `dump_spectra_json`, so
# the document is valid by construction and cannot go stale.


def _need(p: Path) -> Path:
    """Assert the fixture is there.

    This used to ``pytest.skip`` on a missing file.  Every fixture it guards is
    COMMITTED under tests/ -- so absence means a broken checkout or a deleted
    file, and skipping turned that into a green run that proved nothing.  A
    missing committed fixture is a failure, loudly.
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


# ---- the transport record, and the predecessor it is not --------------- #


def _record(tmp_path, **over):
    """A record written by the LIVE writer, not hand-built."""
    from molbuilder.transport.record import write_record, TRANSPORT_RESULT_SCHEMA
    rec = {"schema": TRANSPORT_RESULT_SCHEMA, "label": "junction",
           "energies_relative_to_ef": True, "stages": [],
           "treatment": "single-bias", "points": [],
           "iv": {"voltages_v": [0.0], "current_a": [1e-9]},
           "provenance": {"slot": None,
                          "atom_permutation": "atom-permutation.json"}}
    rec.update(over)
    return write_record(tmp_path, rec)


def test_a_transport_record_is_read_by_a_parser_not_by_the_browser(tmp_path):
    """`<label>.transport.json` was the one result kind no Python could read.

    `parse/sidecars/transport.py` was deleted 2026-09-17 and rightly: it
    claimed a file only when the payload carried a top-level
    `schema_version`, while the live writer (`transport/record.py`) emits
    `schema` -- so it sat in the registry unable to claim the one file
    molbuilder writes, and its own writer had zero production callers in
    every revision.

    What changed is that the reader now has a consumer.  `/api/results/dir`
    asks the registry what reads each file, and the answer here was
    *nothing* -- so the Results tab parsed it in the BROWSER
    (`lib/inspectors/transport.js`: `JSON.parse(body.text)`), the only
    result kind whose format was understood only in JavaScript.

    The fixture goes through `write_record`, so this cannot drift from the
    shape actually written -- which is exactly how the predecessor failed.
    """
    from molbuilder.parse import detect

    p = _record(tmp_path)
    kind = detect(str(p))
    assert kind.name == "transport-json"
    got = kind.parse(p)
    assert got.schema == "transport/v1"
    assert got.payload["label"] == "junction"
    assert got.payload["treatment"] == "single-bias"
    assert got.payload["iv"]["voltages_v"] == [0.0]


def test_it_refuses_a_json_that_is_not_a_transport_record(tmp_path):
    """The schema is checked, not the suffix -- which is the predecessor's
    lesson pointing the other way.  `check_schema` compares the NAME as well
    as the major, so another `@1` artifact cannot sail through.

    MUTATION THIS MUST FAIL AGAINST: drop the `check_schema` call.
    """
    import json

    from molbuilder.parse import detect
    from molbuilder.parse.errors import UnknownFormatError

    foreign = tmp_path / "other.transport.json"
    foreign.write_text(json.dumps({"schema": "molbuilder/task@1",
                                   "label": "x"}), encoding="utf-8")
    with pytest.raises(UnknownFormatError):
        detect(str(foreign))


def test_the_catalogue_row_lets_the_door_offer_it(tmp_path):
    """`result_roles("transport")` must NAME the record, or the door can
    never offer a transport calculation its own deliverable.

    The row gained `staged=False` on 2026-09-18 and that half was inert:
    `calculation` was still None, so `result_roles("transport")` answered
    `('.molwatch.log',)` -- the progress channel of a run, for a
    CALCULATION-level result.  `engines/transport.md` § 2a.12: the
    transmission stage's output is the deliverable and everything else in
    the tree exists to make it trustworthy.
    """
    from molbuilder.runfiles import result_roles
    assert result_roles("transport")[0] == ".transport.json"


# --------------------------------------------------------------------- #
#  A benchmark sweep's plan -- the same gap as the transport record,     #
#  found two days later (2026-09-19)                                     #
# --------------------------------------------------------------------- #

def _job_set(tmp_path, kind):
    """A real `job-set.json` of either kind, through `JobSet.write`.

    Same principle as `_record` above: the fixture goes through the
    WRITER, so the parser cannot be tested against a shape nothing emits
    -- the failure the deleted transport parser is the monument to.
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


def test_an_ordinary_ladder_is_refused_though_the_filename_matches(tmp_path):
    """The DISCRIMINATOR, not the name -- and this is the half that was
    used to justify dropping the file entirely.

    A calculation's stage ladder is also called `job-set.json`, and its
    `/api/bench/summary` answers 400 because there is no sweep to
    summarise.  Offering it would mount the bench viewer on a refusal.
    It is declined for the reason stated on disk (`kind: ladder`) rather
    than by the whole name being excluded, which took the sweep with it.

    MUTATION THIS MUST FAIL AGAINST: drop the `kind` test from
    `_load_sweep`, keeping only the schema check.
    """
    from molbuilder.parse.errors import UnknownFormatError

    with pytest.raises(UnknownFormatError):
        detect(str(_job_set(tmp_path, "ladder")))


def test_a_damaged_sweep_says_damaged_not_ladder(tmp_path):
    """A refusal has to name its OWN cause.

    `_load_sweep` answered `None` for every failure and `parse` turned
    all of them into one sentence about `kind: ladder`.  So a plan
    truncated by a killed write reported as a healthy ladder, the bench
    directory listed as "no result files yet" -- the symptom this parser
    exists to remove -- and nothing said the file was damaged.

    ASKED THROUGH `parse`, NOT `detect`, and the difference is the point.
    `detect` fans a boolean `can_parse` over every registered parser, so
    it cannot attribute a refusal to one of them and answers its own
    generic "no registered file parser knows how to handle ...".  That
    is by construction and this test does not pretend otherwise: what it
    pins is that the parser ITSELF, asked directly, says which of its
    seven failure modes it hit.  The picker path still shows only an
    absence -- see the note in `job_set.py`.

    MUTATION THIS MUST FAIL AGAINST: collapse `_load_sweep`'s raises
    back into a single `return None`.
    """
    from molbuilder.parse.errors import UnknownFormatError
    from molbuilder.parse.sidecars import JobSetSweepFileParser

    f = tmp_path / "job-set.json"
    f.write_text('{"schema": "molbuilder/job-set@1", "kind": "swe',
                 encoding="utf-8")
    assert JobSetSweepFileParser.can_parse(f) is False
    with pytest.raises(UnknownFormatError) as e:
        JobSetSweepFileParser.parse(f)
    said = str(e.value)
    assert "JSON" in said, said
    assert "ladder" not in said, (
        "a damaged file is reported as an ordinary ladder: " + said)
