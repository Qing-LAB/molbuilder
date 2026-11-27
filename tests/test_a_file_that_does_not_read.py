"""A FILE A READER CANNOT READ IS REFUSED BY NAME, IN ONE KIND --
`execution/job-contracts.md` § 6.1, from *"A file a reader cannot read is
refused by name, in one kind"* through the family table.

Each reader reads the versions it lists and refuses every other file it
cannot read in words -- what it found, what this molbuilder reads, and the
way on, its family's; the refusal carries one mark, `persist.FileRefused`,
and the command line and the web server say it at their edge.  Each rule
here is a table of rows, each row a file as a person meets it on the road:
the calculation `molbuilder describe` writes, prepared by `jobset prep`, its
file then given another version or a content its reader does not read.
Every version a row states is one a molbuilder wrote -- an older one a
release before this one wrote, or the next major, a later molbuilder's.

THE SILENT FAILURES: a Python traceback, or a server error, where a person
was owed the refusal's words; a file a later molbuilder wrote told to be
made again by an older one's way; `jobset migrate` calling a calculation
*current* whose file it cannot read; words naming the wrong thing -- a
record that does not read said as absent, a file named twice; a page that
opens a calculation refusing it whole for one file a gentler door says.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from support.road import (declare_bench, described, job_lines, jobset,
                          left_in_the_background, molbuilder, prep_groups,
                          prepared_here, road_tree)


def _set_schema(path: Path, schema: str) -> None:
    """``path``'s schema set to ``schema``, the rest as its writer wrote it
    -- a JSON record's ``schema`` key, a TOML file's ``schema`` line."""
    if path.suffix == ".json":
        path.write_text(json.dumps({**json.loads(path.read_text()),
                                    "schema": schema}, indent=2))
        return
    text, n = re.subn(r'(?m)^schema\s*=\s*"[^"]*"', f'schema = "{schema}"',
                      path.read_text())
    assert n == 1, path
    path.write_text(text)


def _set_sidecar_version(pair: Path, version: int) -> Path:
    """The structure pair ``pair``'s sidecar made to state ``version``, the
    rest as its writer wrote it; returns the sidecar."""
    from molbuilder.sidecars.molstruct import sidecar_path_for
    side = sidecar_path_for(pair)
    assert side.is_file(), side
    side.write_text(json.dumps({**json.loads(side.read_text()),
                                "schema_version": version}, indent=2))
    return side


def _said(r) -> str:
    """What a command said, its lines joined as one sentence."""
    return " ".join(r.output.split())


def _served():
    """The web server, on the tree the test set up -- made after it, as a
    person starts it over their tree."""
    from molbuilder.web.app import create_app
    return create_app(config={"rate_limit": {"enabled": False}}).test_client()


def _transport(tmp_path, monkeypatch) -> Path:
    """A transport calculation on the road, its first group prepared: a
    frame-less junction cited as a pair (`support.junction.bridged_chain`)
    stating the optimization it came from -- its junction composed and its
    composition recorded beside `task.json`."""
    from support.junction import bridged_chain
    junction = {**bridged_chain(), "info": {"calculation": {"contract": {}}}}
    tree = road_tree(tmp_path, monkeypatch, structure=junction)
    calc = described(tree, calculation="transport", name="T")
    prep_groups(calc, 1)
    return calc


#: THE WORDS OF A VERSION, by which way it differs (§ 6.1, *The versions'
#: words*): an older one belongs to a calculation described before
#: 2026-11-04, a later one to a later molbuilder.
_OLDER = ("described before 2026-11-04", "describe the calculation again")
_LATER = ("written by a later molbuilder than this one",
          "open it with the molbuilder that wrote it")

#: EACH FAMILY'S WAY ON FOR A FILE OF AN OLDER VERSION, OR ONE THAT DOES NOT
#: READ, in its words (§ 6.1's table) -- none of which a later molbuilder's
#: file is told: its way on is the molbuilder that wrote it.
_OLDER_WAYS = ("describe the calculation again", "writes it again",
               "probe --write", "jobset migrate", "Fix it by hand",
               "give it back", "move it aside", "run the vibration again")


# --------------------------------------------------------------------- #
#  The command line's edge                                               #
# --------------------------------------------------------------------- #

#: A VERB THAT READS A FILE ON ITS WAY, a row each: what is typed, the
#: calculation's benchmark prepared and its plan then a later molbuilder's.
_BENCH_VERBS = {
    "launch bench, no stage named": ("launch", "bench", "--mode", "direct"),
    "summarize bench, no stage named": ("summarize", "bench"),
}


@pytest.mark.parametrize("verb", list(_BENCH_VERBS))
def test_a_verb_says_a_file_it_cannot_read_at_the_command_lines_edge(
        verb, tmp_path, monkeypatch):
    """§ 6.1: the command line says a file it cannot read at its edge --
    ``Error: <the words>``, exit 1 -- a verb that never asked for it
    included: the benchmark's plan (`<NN>_<stage>/bench/job-set.json`) a
    later molbuilder wrote, met by `launch bench` and `summarize bench`
    looking for the stage with a benchmark, said in the later words alone.
    THE SILENT FAILURE: the Python traceback both died of (2.86.0)."""
    calc = described(road_tree(tmp_path, monkeypatch))
    declare_bench(calc, {"mpi_np": [2], "omp_threads": [1]})
    r = jobset("prep", "bench", "coarse", "--bundle", calc, "--target",
               "this")
    assert r.exit_code == 0, r.output
    _set_schema(calc / "01_coarse" / "bench" / "job-set.json",
                "molbuilder/job-set@2")
    r = jobset(*_BENCH_VERBS[verb], "--bundle", calc)
    assert r.exit_code == 1, r.output
    assert isinstance(r.exception, SystemExit), repr(r.exception)
    said = _said(r)
    for words in ("Error:", "it says 'molbuilder/job-set@2'",
                  "this molbuilder reads molbuilder/job-set@1", *_LATER):
        assert words in said, (words, said)
    for way in _OLDER_WAYS:
        assert way not in said, (way, said)


# --------------------------------------------------------------------- #
#  Calculation setup: each version its reader does not read              #
# --------------------------------------------------------------------- #

#: A CALCULATION'S SETUP, a row each: the file, the version it is given,
#: and the version's words -- the calculation described; its plan prepared
#: first; its own warm-file list the engine's copied beside it first.  The
#: older row is the template of 2026-08-11 to 08-13 (`template@1`); every
#: other file of the setup has had one major, so its row is a later
#: molbuilder's.
_SETUP_VERSIONS = {
    "a description a later molbuilder wrote": (
        "task.json", "molbuilder/task@3", _LATER),
    "a job request a later molbuilder wrote": (
        "job-request.json", "molbuilder/job-request@2", _LATER),
    "a template of an older major": (
        "H2.template.toml", "molbuilder/template@1", _OLDER),
    "a template a later molbuilder wrote": (
        "H2.template.toml", "molbuilder/template@3", _LATER),
    "a plan a later molbuilder wrote": (
        "job-set.json", "molbuilder/job-set@2", _LATER),
    "its own warm-file list a later molbuilder wrote": (
        "warm-files.toml", "molbuilder/warm-files@2", _LATER),
}

#: WHERE A PERSON MEETS IT: the verb that reads the file next -- prep, or
#: for a prepared plan its launch, a dry run -- and `jobset migrate`.
_SETUP_DOORS = ("its reader", "jobset migrate")


@pytest.mark.parametrize("door", _SETUP_DOORS)
@pytest.mark.parametrize("row", list(_SETUP_VERSIONS))
def test_a_calculations_setup_of_another_version_is_refused_with_its_way_on(
        row, door, tmp_path, monkeypatch):
    """§ 6.1, the family table's *calculation setup*: a file of a version
    `jobset migrate` does not translate is refused by name -- an older
    major belongs to a calculation described before 2026-11-04, its way on
    the calculation described again; a later one says a later molbuilder
    wrote it, to be opened with that molbuilder -- at the verb that reads
    it and at `jobset migrate`, which says it does not translate it and
    the same way on.  THE SILENT FAILURES: a later file told the older
    one's way; migrate calling the calculation *current* (a plan, a
    warm-file list) or saying a rewrite *would still be refused* (a
    description, a job request) for a file it never translates; the
    warm-file list's path said twice."""
    from molbuilder.warmfiles import warm_list
    name, schema, words = _SETUP_VERSIONS[row]
    calc = described(road_tree(tmp_path, monkeypatch))
    if name == "job-set.json":
        prep_groups(calc, 1)
    if name == "warm-files.toml":
        (calc / name).write_text(
            Path(warm_list("siesta").path).read_text())
    _set_schema(calc / name, schema)
    if door == "jobset migrate":
        r = jobset("migrate", "--bundle", calc)
    elif name == "job-set.json":
        r = jobset("launch", "task", "--mode", "direct", "--dry-run",
                   "--bundle", calc)
    else:
        r = jobset("prep", "task", "--group", "1", "--bundle", calc,
                   "--target", "this")
    assert r.exit_code == 1, r.output
    assert isinstance(r.exception, SystemExit), repr(r.exception)
    said = _said(r)
    for w in (f"it says '{schema}'", *words):
        assert w in said, (w, said)
    for w in ({_OLDER: _LATER, _LATER: _OLDER}[words]
              + ("nothing to migrate", "would still be refused")):
        assert w not in said, (w, said)
    if door == "jobset migrate":
        assert "`jobset migrate` does not translate" in said, said
    assert said.count(str(calc / name)) <= 1, said


# --------------------------------------------------------------------- #
#  The files `jobset migrate` reads and does not translate               #
# --------------------------------------------------------------------- #

def _another_version(calc: Path, which: str) -> Path:
    """``calc``'s file ``which`` made of a version this molbuilder does not
    read, the rest as its writer wrote it; returns the file."""
    from molbuilder.runfiles import (JUNCTION_FILE, ROLE_SOURCE,
                                     SLOT_PROVENANCE_FILE, compose)
    from molbuilder.scheduler.record import calculation_record
    if which == "source":
        return _set_sidecar_version(calc / compose("H2", ROLE_SOURCE), 16)
    if which == "junction":
        return _set_sidecar_version(calc / JUNCTION_FILE, 16)
    if which == "later junction":
        from molbuilder.sidecars.molstruct import SCHEMA_VERSION
        return _set_sidecar_version(calc / JUNCTION_FILE, SCHEMA_VERSION + 1)
    if which == "provenance":
        _set_schema(calc / SLOT_PROVENANCE_FILE,
                    "molbuilder/slot-provenance@1")
        return calc / SLOT_PROVENANCE_FILE
    _set_schema(calculation_record(calc), "molbuilder/environment@4")
    return calculation_record(calc)


#: A FILE `jobset migrate` READS AND DOES NOT TRANSLATE, a row each (§ 6.1's
#: family table; `job-request.md` § 8): the calculation it lies in, the file
#: made of a version this molbuilder does not read, and its reader's words.
_NOT_TRANSLATED = {
    "the copy of its structure, saved before the one version read": (
        "optimization", "source",
        ("version 16", "described before 2026-11-04",
         "Describe it again from the structure it was made from")),
    "its composition record, composed before 2.10.0": (
        "transport", "provenance",
        ("'molbuilder/slot-provenance@1'",
         "a transport calculation composed before 2.10.0")),
    "its composed junction, a later molbuilder's": (
        "transport", "later junction",
        ("is version", "written by a later molbuilder than this one",
         "open it with the molbuilder that wrote it")),
    "its machine's record, a later molbuilder's": (
        "optimization", "machine",
        ("it says 'molbuilder/environment@4'", *_LATER)),
}


@pytest.mark.parametrize("row", list(_NOT_TRANSLATED))
def test_migrate_refuses_a_file_it_reads_and_does_not_translate(
        row, tmp_path, monkeypatch):
    """§ 6.1 and `job-request.md` § 8: a file of the calculation `jobset
    migrate` reads and does not translate -- the copy of its structure, the
    composition record, the calculation's copy of its machine's record, of
    a version this molbuilder does not read, and the composed junction a
    later molbuilder wrote -- is refused by name, nothing written, in its
    reader's words and its family's way on.  THE SILENT FAILURE: *nothing
    to migrate* -- the calculation called current -- of a calculation its
    next prep refuses (2.87.0's first part; a later machine copy and a
    later junction, 2.86.0 too)."""
    kind, which, words = _NOT_TRANSLATED[row]
    if kind == "transport":
        calc = _transport(tmp_path, monkeypatch)
    else:
        calc = described(road_tree(tmp_path, monkeypatch))
        prep_groups(calc, 1)
    path = _another_version(calc, which)
    before = path.read_bytes()
    r = jobset("migrate", "--bundle", calc)
    assert r.exit_code == 1, r.output
    assert isinstance(r.exception, SystemExit), repr(r.exception)
    said = _said(r)
    for w in (f"`jobset migrate` does not translate {path.name}", *words):
        assert w in said, (w, said)
    assert "nothing to migrate" not in said, said
    assert path.read_bytes() == before


def _transport_launched(tmp_path, monkeypatch) -> Path:
    """A transport calculation on the road (:func:`_transport`) whose first
    group was launched -- left going in its activation, the machine's
    record saying a shell enters its environment with ``sleep 60;``, as
    `support.road.prepared_here` lays it, then killed outright: a stage
    launched, its runs on the junction recorded.  No engine runs."""
    from conftest import write_machine_record
    from support.road import SHELL_CARRIED
    for var in SHELL_CARRIED:
        monkeypatch.delenv(var, raising=False)
    write_machine_record(env_init={"activation": "sleep 60;",
                                   "preamble": "true"})
    calc = _transport(tmp_path, monkeypatch)
    with left_in_the_background("task", "--group", "1", "--bundle", calc):
        pass
    return calc


def _older(calc: Path) -> Path:
    """The composed junction's sidecar made an older molbuilder's (version
    16, saved before 2026-11-04)."""
    from molbuilder.runfiles import JUNCTION_FILE
    return _set_sidecar_version(calc / JUNCTION_FILE, 16)


def _edited(calc: Path) -> Path:
    """The composed junction's first atom moved by hand in its `.xyz` -- a
    record prep composed, edited since."""
    from molbuilder.runfiles import JUNCTION_FILE
    path = calc / JUNCTION_FILE
    lines = path.read_text().splitlines()
    first = lines[2].split()
    first[1] = f"{float(first[1]) + 0.25:.6f}"
    lines[2] = "  ".join(first)
    path.write_text("\n".join(lines) + "\n")
    return path


#: A COMPOSED JUNCTION THAT DOES NOT READ, by the calculation's state: how
#: the calculation is made, how its junction does not read, `migrate`'s
#: exit, and its words.
_JUNCTION_BY_STATE = {
    "an older molbuilder's, none of its stages launched": (
        _transport, _older, 0,
        ("junction.molstruct.json", "composed before 2026-11-04",
         "its next prep composes the junction again")),
    "an older molbuilder's, a stage launched": (
        _transport_launched, _older, 1,
        ("`jobset migrate` does not translate junction.molstruct.json",
         "is not composed again", "was composed before 2026-11-04",
         "ran on the junction recorded", "never replaced")),
    "edited since prep, none of its stages launched": (
        _transport, _edited, 0,
        ("junction.xyz does not read", "its next prep composes the junction "
         "again")),
}


@pytest.mark.parametrize("state", list(_JUNCTION_BY_STATE))
def test_migrate_answers_a_composed_junction_that_does_not_read_by_the_calculations_state(
        state, tmp_path, monkeypatch):
    """§ 6.1's family table (*a transport calculation's composed
    junction*), `engines/transport.md` § 1.1 and `job-request.md` § 8: a
    composed junction that does not read -- an older molbuilder's, or one
    edited since prep -- is answered by `jobset migrate` as the next prep
    answers it, by the calculation's state
    (`transport.gather.no_usable_record`): none of its stages launched, its
    next prep composes the junction again -- said, written in the ledger,
    nothing written and the calculation never called *current*; one
    launched, refused in the junction's words and the way back to a state
    before that launch.  THE
    SILENT FAILURES: *current* of a calculation whose junction does not
    read (2.86.0, an edited one; 2.87.0's first part, an older one too); a
    calculation its next prep composes again told to be described anew, or
    refused."""
    from molbuilder.jobset.ledger import read as ledger
    make, spoil, exit_code, words = _JUNCTION_BY_STATE[state]
    calc = make(tmp_path, monkeypatch)
    spoiled = spoil(calc)
    before = spoiled.read_bytes()
    r = jobset("migrate", "--bundle", calc)
    assert r.exit_code == exit_code, r.output
    said = _said(r)
    for w in words:
        assert w in said, (w, said)
    not_words = ("nothing to migrate",) + (
        ("does not translate", "Describe a new transport calculation")
        if exit_code == 0 else ("its next prep composes",))
    for w in not_words:
        assert w not in said, (w, said)
    assert spoiled.read_bytes() == before
    if exit_code == 0:
        logged = " ".join(" ".join(map(str, e.get("changes") or []))
                          for e in ledger(calc) if e.get("verb") == "migrate")
        assert "composes the junction again" in logged, logged


# --------------------------------------------------------------------- #
#  A later molbuilder's file: refused, never made again                  #
# --------------------------------------------------------------------- #

def test_a_junction_a_later_molbuilder_composed_is_refused_never_composed_again(
        tmp_path, monkeypatch):
    """§ 6.1, the family table's *composed junction*, and `engines/
    transport.md` § 1.1: a transport calculation's composed junction whose
    sidecar a later molbuilder wrote is refused at the next prep in the
    later words -- never taken as no record and composed again over it, as
    one of an older version is while none of its stages is launched.  THE
    SILENT FAILURE: the later molbuilder's junction quietly replaced by
    this one's composition (2.86.0 and 2.87.0's first part both)."""
    from molbuilder.runfiles import JUNCTION_FILE
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION
    calc = _transport(tmp_path, monkeypatch)
    side = _set_sidecar_version(calc / JUNCTION_FILE, SCHEMA_VERSION + 1)
    before = side.read_bytes()
    r = jobset("prep", "task", "--group", "1", "--bundle", calc, "--target",
               "this")
    assert r.exit_code == 1, r.output
    assert isinstance(r.exception, SystemExit), repr(r.exception)
    said = _said(r)
    for w in (JUNCTION_FILE, f"version {SCHEMA_VERSION + 1}", *_LATER):
        assert w in said, (w, said)
    assert side.read_bytes() == before


# --------------------------------------------------------------------- #
#  A results file of another version, listed and opened                  #
# --------------------------------------------------------------------- #

def _loose(name: str, body: dict):
    """A results file ``name`` holding ``body``, in a folder of the tree no
    calculation claims -- as a person copies one out."""
    def made(tmp_path, monkeypatch) -> Path:
        from conftest import point_the_one_door_at
        folder = point_the_one_door_at(tmp_path, monkeypatch.setenv) / "loose"
        folder.mkdir(parents=True)
        (folder / name).write_text(json.dumps(body))
        return folder / name
    return made


def _benchmark_plan(schema: str):
    """A benchmark's plan, `<NN>_<stage>/bench/job-set.json`, prepared on
    the road and then of ``schema``."""
    def made(tmp_path, monkeypatch) -> Path:
        calc = described(road_tree(tmp_path, monkeypatch))
        declare_bench(calc, {"mpi_np": [2], "omp_threads": [1]})
        r = jobset("prep", "bench", "coarse", "--bundle", calc, "--target",
                   "this")
        assert r.exit_code == 0, r.output
        plan = calc / "01_coarse" / "bench" / "job-set.json"
        _set_schema(plan, schema)
        return plan
    return made


def _transport_description(schema: str):
    """A transport calculation's description, `task.json` at its root, as
    `molbuilder describe` writes it on the road, then of ``schema``."""
    def made(tmp_path, monkeypatch) -> Path:
        from support.junction import bridged_chain
        junction = {**bridged_chain(),
                    "info": {"calculation": {"contract": {}}}}
        tree = road_tree(tmp_path, monkeypatch, structure=junction)
        calc = described(tree, calculation="transport", name="T")
        _set_schema(calc / "task.json", schema)
        return calc / "task.json"
    return made


#: A FILE THE RESULTS TAB LISTS, a row each: how it is made -- a version
#: this molbuilder does not read -- the parser that claims it, and the
#: words its reader refuses it in.
_RESULTS_OF_ANOTHER_VERSION = {
    "a transport result": (
        _loose("T.transport.json",
               {"schema": "molbuilder/transport-result@18"}),
        "transport-json",
        ("it says 'molbuilder/transport-result@18'",
         "this molbuilder reads molbuilder/transport-result@19",
         "jobset summarize task writes it again")),
    "a displacement sweep": (
        _loose("V.fc-sweep.json",
               {"schema": "molbuilder/fc-displacement-sweep@1"}),
        "fc-sweep-json",
        ("it says 'molbuilder/fc-displacement-sweep@1'",
         "this molbuilder reads molbuilder/fc-displacement-sweep@2",
         "jobset summarize task writes it again")),
    "a run of points' index a later molbuilder wrote": (
        _loose("V_01_freq.points.json", {"schema": "molbuilder/run-points@2"}),
        "run-points-json",
        ("it says 'molbuilder/run-points@2'",
         "this molbuilder reads molbuilder/run-points@1", *_LATER)),
    "a run's notification record a later molbuilder wrote": (
        _loose("notified.json",
               {"schema": "molbuilder/run-notified@2", "sent": []}),
        "run-notified-json",
        ("it says 'molbuilder/run-notified@2'",
         "this molbuilder reads molbuilder/run-notified@1", *_LATER)),
    "a benchmark's plan a later molbuilder wrote": (
        _benchmark_plan("molbuilder/job-set@2"),
        "job-set-sweep",
        ("it says 'molbuilder/job-set@2'",
         "this molbuilder reads molbuilder/job-set@1", *_LATER)),
    "a transport calculation's description a later molbuilder wrote": (
        _transport_description("molbuilder/task@3"),
        "transport-task",
        ("it says 'molbuilder/task@3'",
         "this molbuilder reads molbuilder/task@2", *_LATER)),
}


@pytest.mark.parametrize("row", list(_RESULTS_OF_ANOTHER_VERSION))
def test_a_results_file_of_another_version_is_listed_and_says_why_it_does_not_open(
        row, tmp_path, monkeypatch):
    """§ 6.1 (*the Results tab still lists a file*) and `model/parse.md`
    § 6 (a parser claims its file by its name -- a benchmark's plan by its
    name and its place, a transport calculation's description by its name
    and its kind, never its version -- and refuses on parse): a results file of a
    version this molbuilder does not read is listed with the parser that
    reads its kind (`/api/results/dir`), and opened it says its reader's
    words -- what it found, what this molbuilder reads, the way on, a later
    molbuilder's file the later one alone (`postprocess.parse`).  THE
    SILENT FAILURE: the file listed as no parser's, and opened *No
    registered file parser knows how to handle* it (2.86.0 and 2.87.0's
    first part both)."""
    from molbuilder.persist import FileRefused
    from molbuilder.postprocess import parse
    made, parser, words = _RESULTS_OF_ANOTHER_VERSION[row]
    path = made(tmp_path, monkeypatch)
    got = _served().get("/api/results/dir",
                        query_string={"path": path.parent})
    assert got.status_code == 200, got.get_json()
    listed = {f["name"]: f for f in got.get_json()["files"]}
    assert listed[path.name]["parser"] == parser, listed[path.name]
    with pytest.raises(FileRefused) as refused:
        parse(path)
    said = str(refused.value)
    assert "No registered file parser" not in said, said
    for w in words:
        assert w in said, (w, said)
    if _LATER[0] in words:
        for way in _OLDER_WAYS:
            assert way not in said, (way, said)


def test_the_sweep_page_says_a_record_it_cannot_read_in_its_readers_words(
        tmp_path, monkeypatch):
    """`web/results.md` § 0.2 and § 6.1: the Results tab's displacement-
    sweep page reads its record through the server's reader
    (`/api/spectra/sweep`, the registry's `fc-sweep-json`), so a record of
    another version is the page's refusal in its reader's words -- a 400,
    what it found, what this molbuilder reads, the way on.  THE SILENT
    FAILURE: the page reading the file's text itself and drawing an older
    record as current (2.87.0's first part)."""
    path = _loose("V.fc-sweep.json",
                  {"schema": "molbuilder/fc-displacement-sweep@1"})(
        tmp_path, monkeypatch)
    got = _served().get("/api/spectra/sweep", query_string={"path": path})
    body = got.get_json()
    assert got.status_code == 400, body
    for w in ("it says 'molbuilder/fc-displacement-sweep@1'",
              "this molbuilder reads molbuilder/fc-displacement-sweep@2",
              "jobset summarize task writes it again"):
        assert w in body["error"], (w, body)


# --------------------------------------------------------------------- #
#  The machine's records and prep's: each told its own way on            #
# --------------------------------------------------------------------- #

def _this_machines(schema: str, *, another: bool, target=None):
    """THIS MACHINE'S RECORD of ``schema``, another machine's on file when
    ``another`` -- probed there, copied here -- and the calculation's first
    prep, naming ``target`` or none."""
    def asked(tmp_path, monkeypatch):
        from molbuilder.config_dir import ensure_private_dir
        from molbuilder.scheduler import machine_scope_path
        from molbuilder.scheduler.record import named_environment_path
        calc = described(road_tree(tmp_path, monkeypatch))
        here = Path(machine_scope_path())
        if another or target:
            there = named_environment_path("sol")
            ensure_private_dir(there.parent)
            there.write_bytes(here.read_bytes())
        _set_schema(named_environment_path(target) if target else here,
                    schema)
        return jobset("prep", "task", "--group", "1", "--bundle", calc,
                      *(("--target", target) if target else ()))
    return asked


def _named_machine(schema: str, verb: str):
    """ANOTHER MACHINE'S RECORD, `sol`'s -- probed there, copied here -- of
    ``schema``, this machine's own reading; asked by ``verb``: `jobset
    machines`, or the calculation's first prep naming no machine."""
    def asked(tmp_path, monkeypatch):
        from molbuilder.config_dir import ensure_private_dir
        from molbuilder.scheduler import machine_scope_path
        from molbuilder.scheduler.record import named_environment_path
        calc = described(road_tree(tmp_path, monkeypatch))
        there = named_environment_path("sol")
        ensure_private_dir(there.parent)
        there.write_bytes(Path(machine_scope_path()).read_bytes())
        _set_schema(there, schema)
        if verb == "machines":
            return jobset("machines")
        return jobset("prep", "task", "--group", "1", "--bundle", calc)
    return asked


def _at_envs_init(schema: str):
    """THIS MACHINE'S RECORD of ``schema``, met by `molbuilder envs
    init-config`, which keeps a record that is there."""
    def asked(tmp_path, monkeypatch):
        from molbuilder.scheduler import machine_scope_path
        road_tree(tmp_path, monkeypatch)
        _set_schema(Path(machine_scope_path()), schema)
        return molbuilder("envs", "init-config", "--activation",
                          "conda activate", "--yes")
    return asked


def _the_calculations(name: str, held: "str | None"):
    """THE CALCULATION'S OWN RECORD ``name`` -- its copy of its machine's
    record, or prep's record of what each stage was prepared with -- of
    the schema ``held``, or text that is not JSON (``None``), after its
    first prep; asked by `jobset status`."""
    def asked(tmp_path, monkeypatch):
        calc = described(road_tree(tmp_path, monkeypatch))
        prep_groups(calc, 1)
        if held is None:
            (calc / name).write_text("{ not JSON")
        else:
            _set_schema(calc / name, held)
        return jobset("status", "--bundle", calc)
    return asked


#: A RECORD OF THE MACHINE OR OF PREP THAT DOES NOT READ, a row each (§ 6.1's
#: family table; `configuration.md` § 5): how it is met, the words it is
#: said in, and the words it is not -- an older molbuilder's told its
#: family's way on, a later one's the later words alone.
_RECORDS = {
    "this machine's, an older molbuilder's, at the question of machines": (
        _this_machines("molbuilder/environment@2", another=True),
        ("this machine's record does not read",
         "an older molbuilder wrote it (molbuilder/environment@2",
         "probe --write"),
        ("This machine has no record yet", *_LATER)),
    "this machine's, a later molbuilder's, at the question of machines": (
        _this_machines("molbuilder/environment@4", another=True),
        ("this machine's record does not read",
         "it says 'molbuilder/environment@4'", *_LATER),
        ("This machine has no record yet", "probe --write")),
    "this machine's, a later molbuilder's, the only one": (
        _this_machines("molbuilder/environment@4", another=False),
        ("it says 'molbuilder/environment@4'", *_LATER),
        ("probe --write",)),
    "this machine's, a later molbuilder's, at envs init-config": (
        _at_envs_init("molbuilder/environment@4"),
        ("it says 'molbuilder/environment@4'", *_LATER),
        ("probe --write",)),
    "a named machine's, a later molbuilder's, in the machines listing": (
        _named_machine("molbuilder/environment@4", "machines"),
        ("cannot be read", "it says 'molbuilder/environment@4'", *_LATER),
        ("write it again",)),
    "a named machine's, a later molbuilder's, at the question of machines": (
        _named_machine("molbuilder/environment@4", "prep"),
        ("sol's record does not read", "it says 'molbuilder/environment@4'",
         *_LATER),
        ("probe --write --name sol",)),
    "a named machine's, a later molbuilder's": (
        _this_machines("molbuilder/environment@4", another=True,
                       target="sol"),
        ("--target 'sol' names a record that cannot be read",
         "it says 'molbuilder/environment@4'", *_LATER),
        ("probe --write",)),
    "the calculation's copy, an older molbuilder's": (
        _the_calculations("environment.json", "molbuilder/environment@2"),
        ("this calculation's record of its machine cannot be read",
         "an older molbuilder wrote it (molbuilder/environment@2",
         "jobset migrate --bundle"),
        ("ready to launch", "Fix it by hand", *_LATER)),
    "the calculation's copy, a later molbuilder's": (
        _the_calculations("environment.json", "molbuilder/environment@4"),
        ("this calculation's record of its machine cannot be read",
         "it says 'molbuilder/environment@4'", *_LATER),
        ("ready to launch", "Fix it by hand", "jobset migrate")),
    "the calculation's copy, not JSON": (
        _the_calculations("environment.json", None),
        ("this calculation's record of its machine cannot be read",
         "invalid JSON", "Fix it by hand"),
        ("ready to launch",)),
    "prep's record of each stage, a later molbuilder's": (
        _the_calculations("prepared-with.json",
                          "molbuilder/prepared-with@2"),
        ("prepared-with.json", "it says 'molbuilder/prepared-with@2'",
         *_LATER),
        ("give it back", "prepare every group again")),
}


@pytest.mark.parametrize("row", list(_RECORDS))
def test_a_record_of_the_machine_or_of_prep_that_does_not_read_is_told_its_way_on(
        row, tmp_path, monkeypatch):
    """§ 6.1 (*The versions' words*, the family table's *settings and
    machine*) and `configuration.md` § 5: a machine's record, or prep's
    record of what each stage was prepared with, that does not read is said
    in its reader's words where it is met -- the question of which machine,
    prep, `jobset status`, `jobset machines`, `envs init-config` -- with its
    family's way on: this machine's or a
    named one's written again by `jobset probe --write`, the calculation's
    copy of an older molbuilder rewritten by `jobset migrate`, one that
    does not read put right by hand or given back; one a later molbuilder
    wrote told the later words alone.  THE SILENT FAILURES: *This machine
    has no record yet* said of a record that is there, and *ready to
    launch* of a calculation whose machine cannot be read (2.86.0); a later
    molbuilder's record told to be probed, migrated, mended or prepared
    again by this one (2.86.0 and 2.87.0's first part both)."""
    asked, words, not_words = _RECORDS[row]
    said = _said(asked(tmp_path, monkeypatch))
    for w in words:
        assert w in said, (w, said)
    for w in not_words:
        assert w not in said, (w, said)


# --------------------------------------------------------------------- #
#  A page that opens a calculation opens it, saying what does not read   #
# --------------------------------------------------------------------- #

#: A CALCULATION'S COPY OF ITS MACHINE'S RECORD THAT DOES NOT READ, a row
#: each: what it holds -- a schema of another version, or text that is not
#: JSON -- and its reader's words.
_MACHINE_COPIES = {
    "an older molbuilder's": (
        "molbuilder/environment@2",
        ("an older molbuilder wrote it (molbuilder/environment@2",)),
    "a later molbuilder's": (
        "molbuilder/environment@4",
        ("it says 'molbuilder/environment@4'", *_LATER)),
    "not JSON": (None, ("invalid JSON",)),
}

def _prepared_h2(tmp_path, monkeypatch) -> Path:
    """The road's H2, its first group prepared."""
    calc = described(road_tree(tmp_path, monkeypatch))
    prep_groups(calc, 1)
    return calc


#: A PAGE THAT OPENS A CALCULATION: the calculation, the route the page
#: asks and how it names the calculation -- its folder, or its address in
#: the tree -- and where its answer says what does not read
#: (`web/results.md` § 2.4, `web/task-setup.md` § 11, `engines/transport.md`
#: § 2a.16: the ladder, or the pick's summary).
_CALCULATION_PAGES = {
    "the Results tab": (_prepared_h2, "/api/results/dir", "path",
                        lambda body: body["ladder"]["unread"]),
    "Task setup": (_prepared_h2, "/api/task-setup/folder", "dir",
                   lambda body: body["ladder"]["error"]),
    "the Transport tab's extension pick": (
        _transport, "/api/transport/describe_attempt", "address",
        lambda body: body["summary"]),
    "the Results tab's transport report": (
        _transport, "/api/transport/record", "path",
        lambda body: " ".join(str(st.get("detail") or "")
                              for st in body["record"]["stages"])),
}


@pytest.mark.parametrize("copy", list(_MACHINE_COPIES))
@pytest.mark.parametrize("page", list(_CALCULATION_PAGES))
def test_a_page_opens_a_calculation_whose_machine_copy_does_not_read(
        page, copy, tmp_path, monkeypatch):
    """§ 6.1 (*a door answers more gently by design*: Task setup still opens
    a folder, the Results tab still lists a file, the Transport tab's pick
    says what it cannot read): a calculation whose copy of its machine's
    record does not read opens on the Results tab, in Task setup, at the
    Transport tab's pick and as the Results tab's transport report -- 200
    -- each saying the copy's words where what it reads would be: the
    ladder, the pick's status line, each rung's state.  THE SILENT FAILURE: each page refusing the whole
    calculation (400) for its machine's record, where 2.86.0 opened it
    (2.87.0's first part)."""
    from molbuilder.projects import address
    from molbuilder.scheduler.record import calculation_record
    make, route, named_by, said_of = _CALCULATION_PAGES[page]
    held, words = _MACHINE_COPIES[copy]
    calc = make(tmp_path, monkeypatch)
    if held is None:
        calculation_record(calc).write_text("{ not JSON")
    else:
        _set_schema(calculation_record(calc), held)
    query = ({"path": address(calc)} if named_by == "address"
             else {named_by: calc})
    got = _served().get(route, query_string=query)
    body = got.get_json()
    assert got.status_code == 200, body
    said = " ".join(str(said_of(body)).split())
    for w in ("this calculation's record of its machine cannot be read",
              *words):
        assert w in said, (w, said)


# --------------------------------------------------------------------- #
#  Words that named the wrong thing                                      #
# --------------------------------------------------------------------- #

def test_stopping_a_job_whose_file_does_not_read_says_so(tmp_path):
    """`job-system.md` § 6.0a: `jobs stop <id>` of a job whose address is on
    file and does not read -- a later molbuilder's -- says the file's words,
    the later ones.  THE SILENT FAILURE: *no job <id> is on file* said of a
    file that is there (2.86.0)."""
    from molbuilder.config_dir import jobs_dir
    jobs_dir().mkdir(parents=True, exist_ok=True)
    (jobs_dir() / "host-123.json").write_text(json.dumps(
        {"schema": "molbuilder/job@2", "id": "host-123"}))
    r = molbuilder("jobs", "stop", "host-123", "--yes")
    assert r.exit_code == 1, r.output
    said = _said(r)
    assert "no job host-123 is on file" not in said, said
    for w in ("it says 'molbuilder/job@2'", "host-123.json", *_LATER):
        assert w in said, (w, said)


#: A LAUNCHED JOB'S RUN THAT DOES NOT READ, a row each: the file that does
#: not read -- its calculation's description, the run's own record -- made
#: so, and the words `jobs status` says of the run in.
_A_JOBS_RUN = {
    "its calculation's description": (
        lambda calc, run: (calc / "task.json").write_text("{ not JSON"),
        ("unreadable", "its calculation does not read", "invalid JSON")),
    "the run's own record": (
        lambda calc, run: _set_schema(run / "calcdir.json",
                                      "molbuilder/calcdir@2"),
        ("unreadable", "calcdir.json", "it says 'molbuilder/calcdir@2'",
         *_LATER)),
}


@pytest.mark.parametrize("row", list(_A_JOBS_RUN))
def test_jobs_status_says_a_run_that_does_not_read_in_its_words(
        row, tmp_path, monkeypatch):
    """`job-system.md` § 6.0a: `molbuilder jobs status` says a launched job
    whose run cannot be read -- its calculation's description, or the run's
    own record, does not read -- as unreadable, in the file's words.  The
    job is the road's (`support.road.prepared_here`), launched and left
    going in its activation.  THE SILENT FAILURE: *its launch record names
    no run of ours* said of the run its launch made."""
    make, words = _A_JOBS_RUN[row]
    calc, run, _names = prepared_here(tmp_path, monkeypatch, "sleep 60;")
    with left_in_the_background("task", "--stage", "coarse", "--bundle",
                                calc) as job:
        make(calc, run)
        r = molbuilder("jobs", "status")
    said = " ".join(" ".join(job_lines(r.output, job.id)).split())
    for w in words:
        assert w in said, (w, said)
    assert "names no run of ours" not in said, said


def _results_tab_said(tmp_path, monkeypatch, run_of_calc):
    """What the Results tab says of a run whose own record does not read:
    the folder answer's trail and its place's words (`/api/results/dir`)."""
    calc = described(road_tree(tmp_path, monkeypatch))
    prep_groups(calc, 1)
    run = calc / "01_coarse" / "run-0"
    run_of_calc(run)
    got = _served().get("/api/results/dir", query_string={"path": run})
    body = got.get_json()
    assert got.status_code == 200, body
    return [" ".join(body["attempts"]), str(body["place"]["says"])]


def _frameset_said(tmp_path, monkeypatch, run_of_calc):
    """What `molbuilder frameset from-mode` says of a vibration's run whose
    own record does not read (`engines/vibration-frames.md` § 4)."""
    calc = described(road_tree(tmp_path, monkeypatch, engine="pyscf"),
                     calculation="vibration", engine="pyscf")
    prep_groups(calc, 1)
    run = calc / "01_freq" / "run-0"
    run_of_calc(run)
    r = molbuilder("frameset", "from-mode", run, "--mode", "1",
                   "--temperature", "300", "--out", calc.parent / "m.xyz")
    assert r.exit_code == 1, r.output
    assert isinstance(r.exception, SystemExit), repr(r.exception)
    return [_said(r)]


#: WHERE A RUN'S FOLDER IS SAID: the Results tab, and the frame-set door.
_A_RUNS_FOLDER = {
    "the Results tab": _results_tab_said,
    "frameset from-mode": _frameset_said,
}


@pytest.mark.parametrize("door", list(_A_RUNS_FOLDER))
def test_a_folder_whose_record_does_not_read_is_said_so(
        door, tmp_path, monkeypatch):
    """`project-layout.md` § 1.4a and `web/results.md` § 0.1: a folder whose
    own record, its `calcdir.json`, does not read -- a later molbuilder's
    -- is said as that, in its reader's words, wherever the folder is
    asked what it is: the Results tab's trail and the words its page shows
    for the folder (`place.says`), and the frame-set door.  THE SILENT
    FAILURE: *not marked as part of a calculation* said of a run whose
    record is there (2.86.0 and 2.87.0's first part both, the page's note
    among them)."""
    from molbuilder.runs import NOT_MARKED
    sentences = _A_RUNS_FOLDER[door](
        tmp_path, monkeypatch,
        lambda run: _set_schema(run / "calcdir.json",
                                "molbuilder/calcdir@2"))
    for said in sentences:
        assert NOT_MARKED not in said, said
        for w in ("calcdir.json", "does not read",
                  "it says 'molbuilder/calcdir@2'"):
            assert w in said, (w, said)


#: A HAND-OVER THAT DOES NOT READ, a row each: what its file holds, the
#: words Task setup's folder says it in, and the words it does not.
_HAND_OVERS = {
    "not JSON": ("{ not JSON", ("task.1st.json", "invalid JSON",
                                "hand it over again from the tab that made "
                                "it"), ()),
    "a later molbuilder's": (
        None, ("task.1st.json", "it says 'molbuilder/task-handover@2'",
               *_LATER),
        ("hand it over again",)),
}


@pytest.mark.parametrize("row", list(_HAND_OVERS))
def test_a_hand_over_that_does_not_read_is_said_with_its_way_on(
        row, isolated_projects_root):
    """§ 6.1's family table (*a hand-over, `task.1st.json`*) and
    `web/handover-procedure.md` § 2: a hand-over that does not read is said
    by Task setup's folder in its reader's words with its way on -- handed
    over again from its tab, which replaces it whole -- one a later
    molbuilder wrote with the later words alone.  The hand-over is the
    road's (`support.road.handed_over`, as the page sends it).  THE SILENT
    FAILURE: the reader's words with no way on, where the family table
    states one (2.86.0 and 2.87.0's first part both)."""
    from support.envelope import envelope
    from support.road import fresh_calc_dir, handed_over
    text, words, not_words = _HAND_OVERS[row]
    folder = fresh_calc_dir(isolated_projects_root, species=["H"])
    client = _served()
    handed_over(client, folder, dict(
        structure=envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]),
        engine="siesta", calculation="optimization", name="probe",
        params={"system_label": "probe"}))
    if text is None:
        _set_schema(folder / "task.1st.json", "molbuilder/task-handover@2")
    else:
        (folder / "task.1st.json").write_text(text)
    got = client.get("/api/task-setup/folder", query_string={"dir": folder})
    body = got.get_json()
    assert got.status_code == 200, body
    said = " ".join(str((body["handover"] or {}).get("error")).split())
    for w in words:
        assert w in said, (w, said)
    for w in not_words:
        assert w not in said, (w, said)


#: THE PERSON'S REPORT-CARD LAYOUT THAT DOES NOT READ, a row each: what it
#: holds, the words it is refused in, and the words it is not.
_REPORT_CARD_COPIES = {
    "not JSON": (
        "{ not JSON", ("invalid JSON", "move it aside"), ()),
    "a later molbuilder's": (
        json.dumps({"schema": "molbuilder/report-cards@2"}),
        ("it says 'molbuilder/report-cards@2'", *_LATER),
        ("move it aside", "put it right")),
}


@pytest.mark.parametrize("copy", list(_REPORT_CARD_COPIES))
def test_a_report_card_layout_that_does_not_read_says_its_way_on(
        copy, tmp_path):
    """`configuration.md` § 3.1 and `run-reports.md` § 4.1c: the person's
    `report-cards.json` that does not read is refused with its way on -- it
    replaces the shipped layout and is optional -- one a later molbuilder
    wrote with the later words alone, and the page asking for the cards
    hears it as a refusal, never a server's fault.  THE SILENT FAILURES: the
    path and the reason with no way on, answered 500 (2.86.0); a later
    molbuilder's layout told to be put right by this one (2.87.0's first
    part)."""
    from molbuilder.report_cards import layout_path
    from molbuilder.web.blueprints.notify import reports_path
    text, words, not_words = _REPORT_CARD_COPIES[copy]
    layout_path().parent.mkdir(parents=True, exist_ok=True)
    layout_path().write_text(text)
    # THE LISTENER HOLDS A REPORT OF HERS, so the page's cards are laid out.
    reports_path("alice").parent.mkdir(parents=True, exist_ok=True)
    reports_path("alice").write_text("")
    got = _served().get("/api/notify/reports",
                         query_string={"user": "alice"})
    body = got.get_json()
    assert got.status_code == 400, body
    said = body["error"]
    for w in ("report-cards.json", *words):
        assert w in said, (w, said)
    for w in not_words:
        assert w not in said, (w, said)


# --------------------------------------------------------------------- #
#  The doors that said a class name, or the file twice                   #
# --------------------------------------------------------------------- #

def test_a_vibration_result_that_does_not_read_is_named_once(
        tmp_path, monkeypatch):
    """`engines/vibration.md` § 6.1: a vibration result of another version
    is refused in its reader's words, the file named once, at `frameset
    from-mode`, `/api/spectra/frameset` and `jobset export`.  THE SILENT
    FAILURE: *V.spectra.json: V.spectra.json: …* (2.86.0)."""
    from molbuilder.runfiles import ROLE_SPECTRA, compose
    from molbuilder.runs import run_of
    calc = described(road_tree(tmp_path, monkeypatch, engine="pyscf"),
                     calculation="vibration", engine="pyscf")
    prep_groups(calc, 1)
    run = calc / "01_freq" / "run-0"
    result = run / compose(run_of(run).label, ROLE_SPECTRA)
    result.write_text(json.dumps({"schema_version": 14}))
    out = calc.parent / "mode1.xyz"
    sentences = []
    r = molbuilder("frameset", "from-mode", run, "--mode", "1",
                   "--temperature", "300", "--out", out)
    assert r.exit_code == 1, r.output
    sentences.append(_said(r))
    # THE PAGE SENDS THE RESULT IT SHOWS, whose folder is its run's.
    got = _served().post("/api/spectra/frameset", json={
        "path": str(result), "mode": 1, "temperature_k": 300,
        "out": str(out)})
    assert got.status_code == 400, got.get_json()
    sentences.append(got.get_json()["error"])
    r = jobset("export", "--format", "netcdf", "--to", tmp_path / "out",
               "--bundle", calc)
    assert r.exit_code == 1, r.output
    sentences.append(_said(r))
    for said in sentences:
        assert said.count(result.name) == 1, said
        assert "the result is schema 14" in said, said


def test_the_bench_summary_says_a_plan_of_another_version_in_its_readers_words(
        tmp_path, monkeypatch):
    """§ 6.1, the web server's edge: the bench summary
    (`/api/bench/summary`) of a plan a later molbuilder wrote answers a 400
    in its reader's words, no Python class name before them.  The route's
    catch-all re-raises a refused file to the edge; were it caught there,
    its own 400 would carry the same words behind its prefix, so this row
    holds the words and not the re-raise -- and Task setup's prep, whose
    catch-all re-raises too, meets every setup file in the calculation's
    door or as prep's own refusal first: no file on the road reaches that
    one.  THE SILENT FAILURE: *ValueError:* printed to a person (2.86.0)."""
    calc = described(road_tree(tmp_path, monkeypatch))
    declare_bench(calc, {"mpi_np": [2], "omp_threads": [1]})
    r = jobset("prep", "bench", "coarse", "--bundle", calc, "--target",
               "this")
    assert r.exit_code == 0, r.output
    plan = calc / "01_coarse" / "bench" / "job-set.json"
    _set_schema(plan, "molbuilder/job-set@2")
    got = _served().get("/api/bench/summary", query_string={"path": plan})
    body = got.get_json()
    assert got.status_code == 400, body
    said = body["error"]
    for name in ("Error:", "ValueError", "SchemaMismatch",
                 "RecordUnreadable"):
        assert name not in said, (name, said)
    for w in ("schema mismatch: it says 'molbuilder/job-set@2'", *_LATER):
        assert w in said, (w, said)


def test_a_run_deck_whose_record_block_does_not_read_is_refused_at_the_web_servers_edge(
        tmp_path, monkeypatch):
    """§ 6.1, the web server's edge, and `job-contracts.md` § 3.1: a run's
    deck whose ATOM-METADATA block does not read -- its END fence cut, as a
    hand edit leaves it -- is refused where the run is opened, `POST
    /api/watch/load` of the run's folder, as a 400 in the deck's reader's
    words: the deck named, the block, the way back.  The edge's one handler
    says it (`web/app.py`), the deck's own handler folded into it.  THE
    SILENT FAILURE: a server error where the person was owed the deck's
    words."""
    from molbuilder.deck_record import BLOCK_ATOM_METADATA, fence
    from molbuilder.runs import run_of
    calc = described(road_tree(tmp_path, monkeypatch))
    prep_groups(calc, 1)
    run = calc / "01_coarse" / "run-0"
    deck = run_of(run).deck
    lines = deck.read_text().splitlines()
    kept = [ln for ln in lines if fence(ln) != (BLOCK_ATOM_METADATA, "END")]
    assert len(kept) == len(lines) - 1, deck
    deck.write_text("\n".join(kept) + "\n")
    got = _served().post("/api/watch/load", json={"path": str(run)})
    body = got.get_json()
    assert got.status_code == 400, body
    for w in (deck.name, BLOCK_ATOM_METADATA, "no BEGIN/END pair",
              "checkpoint restore"):
        assert w in body["error"], (w, body)


def test_the_results_tab_says_a_transport_description_that_does_not_read_in_its_words(
        tmp_path, monkeypatch):
    """`model/parse.md` § 6: the Results tab's reader of a transport
    calculation's description refuses one that does not read -- a later
    molbuilder's -- in the description's reader's words, with no Python
    class name before them.  THE SILENT FAILURE: *ValueError: …* printed to
    a person (2.86.0)."""
    from molbuilder.persist import FileRefused
    from molbuilder.task_reader import TransportTaskFileParser
    from support.junction import bridged_chain
    junction = {**bridged_chain(), "info": {"calculation": {"contract": {}}}}
    tree = road_tree(tmp_path, monkeypatch, structure=junction)
    transport = described(tree, calculation="transport", name="T")
    _set_schema(transport / "task.json", "molbuilder/task@3")
    with pytest.raises(FileRefused) as refused:
        TransportTaskFileParser.parse(transport / "task.json")
    said = str(refused.value)
    assert "ValueError" not in said and "SchemaMismatch" not in said, said
    for w in ("it says 'molbuilder/task@3'", *_LATER):
        assert w in said, (w, said)
