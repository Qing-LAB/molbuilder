"""The road a test drives -- `molbuilder describe`, `prep`, `launch` -- and the one
thing a test may put beside it: a machine whose record names queues, with a
scheduler that queues nothing.  A run is made on the road or not at all: an
output copied into a stage's folder, or a conclusion written by hand, is a
run that never happened (user, 2026-10-03).

WHY THIS FILE EXISTS.  A test drives the designed workflow (user, 2026-09-23:
*"tests should be using our established jobset workflow, unless you have a
strong reason to focus on api"*).  Several files grew their own copy of the
same steps -- an `init` of H2, a stub `sbatch` -- and a copy is free to drift
from the road it imitates.  The steps live here once.

* :func:`jobset` -- the verbs, as a person types them;
* :func:`each_is_taken` -- every command an output prints, typed back as
  printed (`job-system.md` § 5.3: what molbuilder prints, you can type);
* :func:`describe_calculation` -- `molbuilder describe` of a held H2 in a box, or
  the structure a row gives, SIESTA, the shipped `publishable` ladder
  (coarse, medium);
* :func:`ladder_edited` -- a description's ladder edited through Task
  setup's Save, as the page edits it;
* :func:`set_by_the_person` -- a calculation's template items changed as a
  person changes them in the file;
* :func:`a_machine_with_queues` -- a machine record naming queues, and a
  SLURM submit command where the record says it is, off PATH, that writes
  down every call -- where it was made and what it said -- and refuses it:
  the basic tests send nothing to a scheduler and assume nothing of its
  answers (user, 2026-10-06);
* :func:`sbatch_line` -- the `sbatch` line a launch showed, as its words;
* :func:`prepared_here` -- a stage prepared for this machine, its run
  script entering only what the record says (:data:`SHELL_CARRIED`);
* :func:`left_in_the_background` -- a launch left in the background, its
  run killed outright whatever the test met.
"""
from __future__ import annotations

import os
import shlex
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from molbuilder.calculation import open as open_calculation
from support.machines import SLURM


def molbuilder(*args, input=None):
    """`molbuilder <args>`, as typed -- ``input`` what the person types at
    its questions."""
    from click.testing import CliRunner

    from molbuilder.cli import cli
    return CliRunner().invoke(cli, [str(a) for a in args], input=input)


def job_lines(output: str, job: str) -> list:
    """The lines `molbuilder jobs status` says of ``job``: its own line and
    the lines under it, up to the next job's (`job-system.md` § 6.0a)."""
    lines = output.splitlines()
    at = next((i for i, ln in enumerate(lines)
               if ln.startswith(f"{job}  ")), None)
    if at is None:
        return []
    end = next((i for i in range(at + 1, len(lines))
                if lines[i] and not lines[i].startswith(" ")), len(lines))
    return lines[at:end]


def jobset(*args, input=None):
    """`molbuilder jobset <args>`, as typed -- with ``input`` as what the
    person types at its questions, when it asks any."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args],
                              input=input)


def printed_commands(output: str):
    """Every ``molbuilder jobset`` command line an output prints, as a shell
    splits it -- a ``#`` comment cut off, as bash does -- without the
    program's two words."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("molbuilder jobset "):
            yield shlex.split(line, comments=True)[2:]


def each_is_taken(output: str) -> int:
    """Type back every command ``output`` prints, from where the test
    stands, and assert each is taken -- a launch only planned
    (``--dry-run``), so nothing is sent.  Returns how many there were."""
    printed = list(printed_commands(output))
    for words in printed:
        got = jobset(*words, *(["--dry-run"] if words[0] == "launch" else []))
        assert got.exit_code == 0, f"{words}: {_one_line(got)}"
    return len(printed)


#: Where the road writes its structure, under the projects tree.
STRUCTURE = "P/structure/structure.xyz"

#: THE ROAD'S STRUCTURE when a row names none: H2, its first atom held, in a
#: 10 Å box -- the smallest calculation that shows a mechanism.
H2 = {"elements": ["H", "H"],
      "positions": [[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]],
      "regions": {"frozen_atoms": [0]},
      "cell": [10.0, 10.0, 10.0]}

#: THE ROAD'S CO2 (plan T-CO2): listed O O C, the carbon held, so the
#: held-first order a SIESTA vibration's decks are written in is a 3-cycle
#: and a restore in the wrong direction gives another answer.  Linear, in a
#: 10 Å box; one C=O a little long, so a relaxation moves.
CO2 = {"elements": ["O", "O", "C"],
       "positions": [[5.0, 5.0, 6.20], [5.0, 5.0, 3.84], [5.0, 5.0, 5.0]],
       "regions": {"frozen_atoms": [2]},
       "cell": [10.0, 10.0, 10.0]}

#: ITS HELD-FIRST ORDER RECORD -- the rule of `model/overview.md` § 2.2's
#: second key applied to it (the held carbon first, then the oxygens in their
#: own order): C O O, [2, 0, 1] one way and [1, 2, 0] the other.  What every
#: test of a deck, a run or a pair in that order expects.
CO2_HELD_FIRST = {"key": "held-first", "sorted_to_original": [2, 0, 1],
                  "original_to_sorted": [1, 2, 0]}
CO2_ENGINE_ELEMENTS = [CO2["elements"][i]
                       for i in CO2_HELD_FIRST["sorted_to_original"]]


#: Water, closed shell and free, in its box.
WATER = {"elements": ["O", "H", "H"],
         "positions": [[5.0, 5.0, 5.0], [5.96, 5.0, 5.0], [4.76, 5.93, 5.0]],
         "cell": [10.0, 10.0, 10.0]}


def road_structure(s):
    """A road structure -- :data:`H2`, :data:`CO2`, a row's -- as a
    ``Structure``: its ``elements``, ``positions`` (Å), ``regions``
    (``frozen_atoms`` among them), its box's three edges or three vectors,
    ``cell`` (Å), each axis's ``axis_kind`` (isolated unless given) and the
    origin a person assigned, ``engine_offset`` (none unless given).
    THE ONE BUILDER of the road's structures -- one a test already holds as
    a ``Structure`` (a box its vacuum sizes, an origin the Cell page's own op
    assigned), as it stands; a pair on disk, as its read door opens it."""
    from molbuilder.structure import Structure
    if isinstance(s, Structure):
        return s
    if isinstance(s, Path):
        from molbuilder.workingcopy_structure import StructureCodec
        return StructureCodec().load(s)
    return Structure(
        elements=list(s["elements"]),
        positions=np.array(s["positions"], dtype=float),
        regions={k: list(v) for k, v in s.get("regions", {}).items()},
        cell=(np.asarray(s["cell"], dtype=float) if np.ndim(s["cell"]) == 2
              else np.diag([float(a) for a in s["cell"]])),
        axis_kind=tuple(s.get("axis_kind", ["isolated"] * 3)),
        engine_offset=s.get("engine_offset"))


def co2_in_engine_order():
    """The road's CO2 as a SIESTA vibration's decks hold it: sorted by its
    decks' order (`transport.sort.sort_by`, `kind_table.atom_order`), its
    order record on it."""
    from molbuilder.kind_table import atom_order
    from molbuilder.transport.sort import sort_by
    return sort_by(road_structure(CO2),
                   atom_order("vibration", "siesta")).structure


def describe_calculation(tmp_path, monkeypatch, *, name: str = "H2",
                         calculation: str = "optimization",
                         engine: str = "siesta",
                         structure=None,
                         stage_strategy: "str | None" = None,
                         bundle: "str | None" = None,
                         bias: str = "0", frames: int = 1,
                         refused: bool = False) -> Path:
    """`molbuilder describe` on ``structure`` -- :data:`H2` unless given
    (:func:`road_structure`: its ``elements``, ``positions`` (Å), ``regions``
    (``frozen_atoms`` among them), its box's three edges, ``cell`` (Å), and
    each axis's ``axis_kind`` (isolated unless given)) -- the bundle, at
    ``<projects>/P/<calculation>/<name>`` unless ``bundle`` names another
    folder from the projects root (a second calculation in one tree takes a
    name or a folder of its own): a SIESTA optimization's shipped
    `publishable` ladder, unless ``stage_strategy`` names another (``""``
    for the one stage the template alone describes), or a vibration's own
    (`freq`).

    A TRANSPORT CALCULATION CITES ``structure``, the pair a real relaxation
    was saved as (`support.real_runs`, `engines/transport.md` § 3.1), at
    ``bias`` (volts, as `--bias` takes them; several say
    `--no-low-bias-approximation`), ``frames`` frames, each moving its bridge
    as a frame set's must.

    ITS RUN CARD STATES THE LAUNCH SHAPE -- two ranks and one thread for
    SIESTA, one thread for PySCF -- in its job request, as a described
    calculation does: a run whose shape is stated nowhere is refused at prep
    (`architecture.md` § 5.2).  A test about that refusal takes the card
    away.

    ``refused`` expects describe to refuse, and returns what it said."""
    from conftest import write_pseudos
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.workingcopy_structure import StructureCodec
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    (tree / "pseudopotential").mkdir(exist_ok=True)
    transport = calculation == "transport"
    s = road_structure(H2 if structure is None else structure)
    siesta = engine == "siesta"
    if siesta:
        write_pseudos(tree / "pseudopotential", sorted(set(s.elements)))
    if transport:
        from molbuilder.transport.sort import REGION_BRIDGE
        if frames > 1:
            moved = [s.positions.copy() for _ in range(frames)]
            for i, at in enumerate(moved):
                at[s.regions[REGION_BRIDGE]] += 0.01 * i
            s = s.with_frames(moved)
    StructureCodec().write(s, tree / STRUCTURE)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    strategy = (stage_strategy if stage_strategy is not None
                else "publishable" if calculation == "optimization" and siesta
                else "")
    where = bundle or f"P/{calculation}/{name}"
    told = (("--slot", f"junction={STRUCTURE}", "--bias", bias,
             *(("--no-low-bias-approximation",) if "," in bias else ()))
            if transport else ("--structure", STRUCTURE))
    r = molbuilder("describe", *told,
               "--bundle", where, "--engine", engine,
               "--name", name, "--calculation", calculation,
               *(("--stage-strategy", strategy) if strategy else ()),
               *(("--psml-lib", "pseudopotential") if siesta else ()))
    if refused:
        assert r.exit_code != 0, r.output
        return r.output
    assert r.exit_code == 0, r.output
    described = tree / where
    request = read_request(described)
    request.setdefault("execution", {"mpi_np": 2, "omp_threads": 1} if siesta
                       else {"threads": 1})
    write_request(described, request)
    return described


def prep_groups(calculation: Path, *groups) -> None:
    """`jobset prep task` of each of ``groups``, in turn, for this machine."""
    for group in groups:
        r = jobset("prep", "task", "--group", group, "--bundle", calculation,
                   "--target", "this")
        assert r.exit_code == 0, r.output


def a_new_folder(tmp_path, under: str) -> str:
    """A calculation's folder nothing holds yet, from the projects root: each
    calculation one of its own, in a tree a test describes in twice."""
    held = (tmp_path / "projects" / "P" / under).glob("JOB*")
    return f"P/{under}/JOB{1 + sum(1 for _ in held)}"


def described_and_prepared(tmp_path, monkeypatch, struct, engine, *,
                           calculation="optimization", template=None,
                           overrides=None, hand_edit=None, refused=False):
    """`molbuilder describe` of ``struct``, then `jobset prep task` of its
    first rung: ``(the calculation, that stage, the deck's text)``.  Each call
    is a calculation of its own, in a folder of its own.  An optimization's
    ladder is its engine's shipped one, a vibration's its own; ``template`` is
    what the person set in its template, ``overrides`` what its first stage
    says over that, through Task setup's Save, ``hand_edit(calculation)`` a
    file changed by hand.

    ``refused`` expects the step that refuses -- Save, for ``overrides``, else
    prep -- and returns what it said in place of the deck."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    # AS A PAIR, through the codec: the cell, the vacuum and the frozen set
    # live in the sidecar, and a bare `.xyz` would hand prep a molecule in the
    # default box -- a different structure from the one described.
    dest = describe_calculation(
        tmp_path, monkeypatch, name="JOB", structure=struct, engine=engine,
        calculation=calculation,
        bundle=a_new_folder(tmp_path, calculation),
        stage_strategy="publishable" if calculation == "optimization" else None)
    stage = task_of(dest).stages[0].name
    if template:
        set_by_the_person(dest, template)
    if hand_edit is not None:
        hand_edit(dest)
    if overrides:
        saved = save_both(dest,
                          _description_with_overrides(dest, {stage: overrides}),
                          read_request(dest))
        if refused:
            assert saved.status_code == 400, saved.get_json()
            return dest, stage, saved.get_json()["error"]
        assert saved.status_code == 200, saved.get_json()
    r = jobset("prep", "task", *group_words(dest, stage), "--bundle", dest,
               "--target", "this")
    if refused:
        assert r.exit_code != 0, r.output
        return dest, stage, r.output
    assert r.exit_code == 0, r.output
    # THE DECK BY ITS NAME, through the catalogue -- not the first file with
    # the suffix: a stage directory may hold other `.py` files than the deck
    # (a person's own scripts may).
    home = stage_home(dest, task_of(dest), stage)
    deck = home.dir / home.names.name(".fdf" if engine == "siesta" else ".py")
    if engine == "siesta" and calculation == "vibration":
        # ONE RUN IN PHASES: each phase's deck in its folder below the
        # stage (`engines/vibration.md` § 5.11 § 2) -- the force-constant
        # phase's is the one the offset rule speaks of.
        from molbuilder.siesta.vibration_deck import FC_PART
        deck = home.dir / FC_PART / deck.name
    if engine == "siesta":
        from molbuilder.parse.fdf import read_deck
        return dest, stage, read_deck(deck)
    return dest, stage, deck.read_text()


def until(holds, *, within: float, what: str) -> None:
    """``holds()``, within ``within`` seconds -- a run's own files read as
    it goes."""
    import time
    end = time.monotonic() + within
    while not holds():
        assert time.monotonic() < end, f"{what}: not within {within} s"
        time.sleep(0.5)


#: WHAT THE TEST'S OWN SHELL WOULD SLIP INTO A RUN SCRIPT AHEAD OF ITS
#: RECORD: a script every non-interactive bash reads first (``BASH_ENV``),
#: and an environment already active (``CONDA_PREFIX``, which the job's
#: python is looked up under, `runwrap.env_python_check`).  A run script a
#: test starts runs without them, so it enters only what its record says.
SHELL_CARRIED = ("BASH_ENV", "CONDA_PREFIX")


def prepared_here(tmp_path, monkeypatch, activation: str,
                  name: str = "H2here", calculation: str = "optimization",
                  stage: str = "coarse", engine: str = "siesta"):
    """An H2 calculation on ``engine`` prepared for this machine
    (:func:`describe_calculation`), whose record says a shell enters its
    environment with ``activation`` -- the line its run script runs once
    its session log is open, before anything else -- its ``stage``'s group
    prepared: ``(the calculation, the stage's run's folder, the stage's
    names)``.  No engine runs on this road."""
    from conftest import write_machine_record
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    for var in SHELL_CARRIED:
        monkeypatch.delenv(var, raising=False)
    write_machine_record(env_init={"activation": activation,
                                   "preamble": "true"})
    bundle = describe_calculation(tmp_path, monkeypatch, engine=engine,
                                  calculation=calculation, name=name)
    r = jobset("prep", "task", *group_words(bundle, stage),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    home = stage_home(bundle, task_of(bundle), stage)
    return bundle, latest_attempt(home.dir), home.names


@contextmanager
def left_in_the_background(*words):
    """`jobset launch <words> --mode direct --yes`, left in the background
    as a person leaves it: its one job's address, the run's process group
    killed outright and reaped on the way out, whatever the test met -- no
    run outlives its test."""
    import signal
    from molbuilder.joblist import every
    r = jobset("launch", *words, "--mode", "direct", "--yes")
    jobs = every()
    try:
        assert r.exit_code == 0 and len(jobs) == 1, (r.output, jobs)
        yield jobs[0]
    finally:
        for job in jobs:
            try:
                os.killpg(job.pid, signal.SIGKILL)
                os.waitpid(job.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass


def killed_outright(job) -> None:
    """A run here's process group sent SIGKILL -- which no shell can catch
    -- and the run gone, as `molbuilder jobs` asks it."""
    import signal
    from molbuilder.joblist import going_here
    os.killpg(job.pid, signal.SIGKILL)
    until(lambda: going_here(job) is False, within=15,
          what="its process gone")


def read_request(bundle: Path) -> dict:
    """The calculation's ``job-request.json``, as it lies."""
    import json
    return json.loads((bundle / "job-request.json").read_text())


def write_request(bundle: Path, request: dict) -> None:
    """The calculation's ``job-request.json``, written as a person edits it."""
    import json
    (bundle / "job-request.json").write_text(json.dumps(request, indent=2))


def one_rank(bundle: Path, **card) -> None:
    """The calculation's run card: one rank, one thread -- the shape a run
    states (`architecture.md` § 5.2) -- and what else ``card`` states."""
    request = read_request(bundle)
    request["execution"] = {**request.get("execution", {}), "mpi_np": 1,
                            "omp_threads": 1, **card}
    write_request(bundle, request)


def read_as(bundle: Path, client) -> dict:
    """What each record was when the page read the folder -- the folder
    answer's ``read_as``, which Save compares (`web/task-setup.md` § 8)."""
    r = client.get("/api/task-setup/folder", query_string={"dir": str(bundle)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["read_as"]


def declare_bench(calculation: Path, axes: dict) -> None:
    """``axes`` as the benchmark the job request declares
    (`execution/generator.md` § 4.3a)."""
    request = read_request(calculation)
    request["bench"] = axes
    write_request(calculation, request)


def save_both(bundle: Path, task: dict, request: dict, client=None):
    """``task`` and ``request`` through Task setup's Save -- the page's
    door, which writes both (`execution/job-request.md` § 7), naming what
    each record was when the page read the folder."""
    import json
    if client is None:
        from molbuilder.web.app import create_app
        client = create_app(config={}).test_client()
    return client.post("/api/task-setup/save",
                       json={"dest": str(bundle), "text": json.dumps(task),
                             "request_text": json.dumps(request),
                             "read_as": read_as(bundle, client)})


def a_machine_with_queues(tmp_path, monkeypatch, domains,
                          **record) -> Path:
    """This machine's record names ``domains`` (`scheduler.Domain` rows) --
    and ``record``'s other fields, when given -- and a SLURM scheduler whose
    submit command, where the record says it is, writes down each call and
    REFUSES it, in molbuilder's words.  Returns the file each call is
    written to, one line each: ``<where it was run> | <its arguments>``.

    THE COMMAND IS THE RECORD'S, AND IT IS NOT ON PATH (`configuration.md`
    M-2, `scheduler.md` § 6a): the `sbatch` first on PATH stays the suite's
    own, which refuses in other words and writes nothing down -- so a launch
    that ran whatever PATH finds instead of what the machine's record names
    sends no call here, and says the suite's words.

    NO SCHEDULER TEXT (user, 2026-10-06: *"there should be no assumption
    what so ever about the text returned by slurm ... the whole default test
    set should never be based on fabricated text"*).  What a
    scheduler answers is read where one answers -- the field tier
    (`tests/field/`, `testing.md` § 0).  Refusing also keeps the basic suite
    from sending a real job when it runs on a cluster's login node."""
    import dataclasses
    from conftest import write_machine_record
    bin_dir = tmp_path / "scheduler-bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch-calls.log"
    f = bin_dir / "sbatch"
    f.write_text(
        "#!/bin/sh\n"
        f'echo "$(pwd) | $*" >> "{calls}"\n'
        'echo "the basic tests send nothing to a scheduler -- what one '
        'answers is the field tier\'s (testing.md 0)" >&2\n'
        "exit 1\n")
    f.chmod(0o755)
    write_machine_record(
        scheduler=dataclasses.replace(SLURM, submit=str(f)),
        domains=list(domains), **record)
    return calls


def _submits(word: str) -> bool:
    """Whether ``word`` is the SLURM submit command a road record names --
    `sbatch`, by its bare name or its full path, as the road's records name
    it."""
    return Path(word).name == "sbatch"


def sbatch_line(output: str):
    """The `sbatch` line a launch showed -- in its question, or as a dry
    run's ``WOULD run`` -- as its arguments, from its command on."""
    for ln in output.splitlines():
        words = ln.split()
        at = next((i for i, w in enumerate(words) if _submits(w)), None)
        if at is not None:
            return [w for w in words[at:] if not w.startswith("[")]
    raise AssertionError(f"no sbatch line shown:\n{output}")


def calls_made(calls: Path):
    """The `sbatch` calls written down so far, as ``(where, argv)``."""
    if not calls.is_file():
        return []
    out = []
    for line in calls.read_text().splitlines():
        where, _sep, argv = line.partition(" | ")
        out.append((Path(where), argv.split()))
    return out


# --------------------------------------------------------------------- #
#  ONE RUNNER FOR A CONTRACT'S CASE TABLE                               #
# --------------------------------------------------------------------- #
#
# A contract's cases are rows of a TOML table (`tests/data/<contract>.toml`),
# and every row runs down the road a person runs -- `molbuilder describe`, the
# description and the target's record written as the row says, `prep`,
# `launch --dry-run` -- checking the layers it names (`docs/process/testing.md`
# § 6):
#
#   1. ALLOWED OR REFUSED, in its words -- at prep (`refused`, a sentence or a
#      list of them; `said`) or at
#      launch (`launch_refused`, the same; `listing`) -- a dry run, unless
#      `launch_sends` sends it (answered `--yes`), which a row does only to
#      be refused: a launch that runs starts an engine, and a test that
#      reads what an engine wrote is an end-to-end test (plan § 5y); the
#      basic tests send nothing to a scheduler either, whose `sbatch`
#      refuses (W57 T1) -- what molbuilder put to the record's scheduler
#      (`scheduler_called`: each call its submit command was given, in order,
#      as the words each carries) -- and what the launch left
#      (`launch_writes_nothing`: every file under the calculation as it was;
#      `after_launch`: the after-prep checks below, asked again);
#   2. WHAT IS PRODUCED -- the `.sbatch` header (`header`, or
#      `header_absent`), the deck (`deck`; `deck_says`, what the engine
#      reads for each keyword it names -- a SIESTA deck through the one fdf
#      reader, `parse.fdf.read_fdf`, its comments skipped, a PySCF deck's
#      assignments at the top level as Python runs them, the last one
#      standing; and `deck_reads`, what the one fdf reader's
#      `parse_fdf_params`, the reader every consumer of a deck asks, reads
#      from it, field by field), the run script
#      (`run_sh`), each
#      with a `_lacks` twin -- and the first three with an `_order` twin
#      (lines standing in that order) and a `_once` twin (lines standing
#      once each); the `sbatch` line launch shows (`line`,
#      `line_lacks`); a benchmark's trials (`bench_gres`, their headers'
#      `bench_header` and `bench_header_lacks`).
#
# INPUTS: `engine` (siesta | pyscf); `bias` -- a transport calculation's, as
# `--bias` takes it, its `structure` cited; `groups` -- each stage's group, in
# order, over the ones `molbuilder describe` gave; `stage_run` -- a stage's
# own run card, `{name = {...}}`; `run` -- the calculation's run card, the
# job request's `execution`, over the table's `run_base` for the engine (a
# row with `bench`: its own `run` alone); `allocation` -- the job request's
# `allocation`, over the table's `allocation_base` where the
# machine has a scheduler; `run_unset` / `allocation_unset` -- base keys a row
# takes away; `template` -- values in the template, the person's
# (`set_by_the_person`; SIESTA's over the table's `siesta_template`), a
# run setting among them onto the run card; `bench` -- the job request's `bench`, then
# `prep bench`; `notify` -- the job request's `notify`, when the calculation
# speaks up;
# `machine_config` -- THIS machine's `molbuilder.json`; `prep` / `launch` --
# the flags typed (launch
# with `--mode submit`, unless `launch_mode` names another -- "" for none, so
# the config's `launch.mode` decides; of the stage `launch_stage` names,
# `coarse` unless it names another -- "" for none, the stage left out);
# `machine` -- "this" (this machine IS
# the target, its record
# listing `queues`), "named" (this machine is a workstation; the target is a
# record named `sol` listing `queues`), "workstation" (no queues at all) or
# "unprobed" (this machine never probed: no record at all), the table's own
# `machine` unless the row names one;
# `record` -- more fields of THIS machine's record; `queues` -- replaces the
# table's menu;
# `probe` -- the record made as a person makes it instead, by `jobset probe
# --write --yes`, once per list of flags, in order, after `machine_config`
# and over `record` when the row gives them; a `--name` among the flags names
# the target.  A probe given as a table, `{flags = [...], answers = "n\n"}`,
# runs WITHOUT `--yes` and is typed `answers` at its questions ("" is EOF);
# `probe_refused` -- the last probe is
# refused, saying each of these, and the row ends there.  A probe row's
# machine holds the `molbuilder.json` `envs init-config` leaves -- its
# `env_init` -- unless `machine_config` gives the file;
# `saved_first` -- the folder's state saved before anything else, as a person
# saves it (`molbuilder checkpoint init`); `before` -- the verbs a person typed
# first, each a list of words (`["prep", "task", "--group", "1"]`), the calculation
# and the target named as the row's own prep names them; `changed` -- the
# records then changed: `template`, `{name = value}`, in the template's file as a person
# changes it (no run setting); and through Task setup's Save, blocks of the
# job request (`{execution = {...}}`), and `overrides`,
# `{stage = {name = value}}`, into the description (`_road_changed`); `removed` / `added` -- the
# stages then removed, or added (`{name, at}`, `at` the place, the end when
# absent), through the same Save; `regrouped` -- each stage's group set
# anew, one number a stage, through the same Save, and `regroup_refused` --
# what Save's refusal of it says, the row ending there, `regroup_status` its
# status (409 unless the row says: a job request its preflight refuses is
# 400); `change_refused` --
# what Save's refusal of the row's `changed` or `removed` says, nothing
# saved or written, the row ending there (`_saved_or_refused`), its way back
# restoring the state whose note ends with `restores_state`;
# `source_edited` --
# `{atom, line}`: the calculation's structure file edited by hand after it was
# described, its `atom`-th atom line replaced by `line`, its sidecar
# untouched; `held_edited` -- its held atoms set anew after it was
# described, written back through the codec; `own_warm_files` -- the calculation's own restart-file
# list, the engine's
# copied beside `task.json` with `withhold` / `add` / `resumes`
# (`_road_own_warm_files`),
# before anything is prepared; `stage` -- the
# stage (or a list of stages) the row's prep names, `coarse` unless given --
# typed as the group each is in;
# `calculation` -- the kind `molbuilder describe` describes, an optimization unless
# given; `structure` -- the structure it describes, H2 unless given
# (`describe_calculation`: `elements`, `positions`, `regions`, `cell`,
# `axis_kind`, `engine_offset`); `stage_strategy` -- the ladder `molbuilder describe` is told,
# `publishable` for a SIESTA optimization unless given, `""` for the one
# stage the template alone describes.
#
# AFTER PREP, whatever it answered: the folders it left (`made`, paths
# under the calculation that exist; `made_lacks`, ones that do not;
# `made_says`, `{path: lines}` -- a file it made and the lines it holds,
# read as `run_sh`'s are), the
# files it left as they were (`kept`: paths under the calculation whose
# bytes and write time are the same before and after the row's prep), every
# file the Task setup card names for a stage on disk as it names it
# (`card_written`: `{stage, moments}` each -- `_road_card_written`), the
# atoms each of its progress logs states the run holds, read back by
# the log's own reader (`progress_log_holds` -- `_road_progress_log_holds`),
# the targets they hand the Watch tab (`progress_log_targets`),
# whether each stage runs as a run of points before any prep, and the words the
# run card says it in (`points_before_prep`: `{stage = words}`, `""` for a stage
# that runs once -- `_road_points_before_prep`),
# what a stage's points are compared by since its prep (`points_compared`, the
# stage -- `_road_points_compared`),
# what a person's script reads of each log's run, which has written no step
# (`steps_read`: `{elements, listed_by}` -- `_road_steps_read`),
# the files each stage's job declares it carries from a run it continues,
# as the plan states it and its reader reads it -- the declaration, which
# the launch's `warm_carry` then judges pair by pair (`plan_carries`,
# `{stage = [file, ...]}` -- `_road_plan_carries`), the
# folder's saved states, newest first
# (`saved_states`, their notes -- `{stamp}` standing for the time a note
# leads with, `2026-10-03 14:05:12`), what `status` says of the calculation
# (`status_says`, and what it does not, `status_lacks`), and the decisions
# its ledger holds (`ledger_holds` -- a decision, or a verb's, `"launch
# continues"`, or a table naming one with its facts, `{decision = "launch
# launched", time = "3h"}`);
# a prep that was not refused says nothing of `said_lacks`, and each of
# `said_once` once.  `both_doors`: the page's prep door on the same folder as it
# stood serves the lines the terminal printed (`_road_both_doors`:
# `preview_said`, `preview_lacks`, `preview_group_header`, `page_found_once`,
# `page_stage_found`).
# THEN, refused or not: the description saved through Task setup's Save with
# `saved`'s fields changed (`{shape = "flat"}`) -- refused, with
# `save_refused`'s words, with `save_status`, 409 unless the row says (400:
# the description's own reader refused it) -- or taken.
#
# 0 · WHAT THE PROBE RECORDS, checked before anything else: what it says
# (`probe_said`, lines any of the probes printed) and what the record it
# wrote holds (`record_says`, a table of the record's fields, nested as the
# file nests them; `record_lacks`, the fields it leaves out;
# `record_declared`, the facts its `source` says were declared -- `flag`
# among the note's parts, `configuration.md` M-1;
# `record_stamped`, its `detected_at` is the last probe's own -- M-6: the
# stamp follows the probe whatever survives; `record_found_on_path`, each
# scheduler command the record names where PATH finds it by the name the
# contract gives it -- `{submit = "sbatch"}`: the full path `shutil.which`
# answers where the probe ran).  `probe_path_lacks` -- the
# commands the probe finds NOT on PATH where it runs (`["sbatch"]`: a
# workstation, or a node whose scheduler is elsewhere), the only place PATH
# is read: discovery (`scheduler.md` § 6a).  A row about the record alone
# ends there: `ends_at = "probe"`.


def _road_target(table, case, tmp_path, monkeypatch) -> str:
    """The machine the case preps for, recorded as a probe records one --
    and the name `--target` takes for it."""
    from conftest import write_machine_record
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      write_environment)
    if "probe" in case:
        # A MACHINE SET UP THE WAY MOLBUILDER SETS ONE UP: `envs init-config`
        # leaves `env_init` in its molbuilder.json, which the probe requires
        # (`configuration.md` § 4) -- unless the row gives the file itself.
        _write_machine_config({"machine_config": {"env_init": {
            "activation": "conda activate", "preamble": "true"}}, **case})
        if "record" in case:
            write_machine_record(**case["record"])
        if case.get("probe_path_lacks"):
            gone = case["probe_path_lacks"]
            monkeypatch.setenv("PATH", os.pathsep.join(
                d for d in os.environ["PATH"].split(os.pathsep)
                if d and not any(os.path.exists(os.path.join(d, c))
                                 for c in gone)))
        from datetime import datetime, timezone
        said, named = [], []
        for step in case["probe"]:
            flags, answers = ((step["flags"], step.get("answers", ""))
                              if isinstance(step, dict) else (step, None))
            # The stamp is to the second, as the probe writes it.
            began = datetime.now(timezone.utc).replace(microsecond=0)
            r = jobset("probe", "--write",
                       *(["--yes"] if answers is None else []), *flags,
                       input=answers)
            if "probe_refused" in case and step is case["probe"][-1]:
                assert r.exit_code != 0, _one_line(r)
                for words in case["probe_refused"]:
                    assert words in r.output, _one_line(r)
                return "this"
            assert r.exit_code == 0, _one_line(r)
            said.append(r.output)
            if "--name" in flags:
                named.append(flags[flags.index("--name") + 1])
        target = named[-1] if named else "this"
        _road_probe_layer(case, said, target, began)
        return target
    queues = [Domain.from_row(q)
              for q in case.get("queues", table.get("queues", []))]
    record = dict(case.get("record", {}))
    where = case.get("machine", table.get("machine", "this"))
    if where == "unprobed":
        # NEVER PROBED: the record the suite writes for every test, gone.
        from molbuilder.scheduler import machine_scope_path
        Path(machine_scope_path()).unlink(missing_ok=True)
        return "this"
    if where == "this":
        a_machine_with_queues(tmp_path, monkeypatch, queues, **record)
        return "this"
    write_machine_record(**record)          # this machine: no queue at all
    if where == "workstation":
        return "this"
    from molbuilder.config_dir import ensure_private_dir
    from molbuilder.scheduler import environments_dir, named_environment_path
    ensure_private_dir(environments_dir())
    fields = dict(scheduler=SLURM, domains=queues,
                  topology=Topology(sockets=2, cores_per_socket=24),
                  env_init={"activation": "conda activate",
                            "preamble": "true"})
    write_environment(Environment(**fields), named_environment_path("sol"))
    return "sol"


def _road_probe_layer(case, said, target, began) -> None:
    """0 · what the probe said, and what the record it wrote holds --
    ``began``, when its last probe started."""
    import json
    from datetime import datetime
    for words in case.get("probe_said", []):
        assert any(words in out for out in said), \
            f"probe_said: {words!r} in none of: {_one_line(said[-1])}"
    from molbuilder.scheduler import machine_scope_path, named_environment_path
    path = (machine_scope_path() if target == "this"
            else named_environment_path(target))
    record = json.loads(path.read_text())
    if "record_says" in case:
        _road_holds(record, case["record_says"], "record")
    for key in case.get("record_lacks", []):
        assert key not in record, f"record.{key} is {record[key]!r}"
    for role, name in case.get("record_found_on_path", {}).items():
        import shutil
        found = shutil.which(name)
        assert found and record["scheduler"].get(role) == found, \
            (f"record.scheduler.{role} is {record['scheduler'].get(role)!r}, "
             f"not {name!r} where PATH finds it ({found!r})")
    for fact in case.get("record_declared", []):
        note = record["source"].get(fact, "")
        assert "flag" in note.split("+"), \
            f"record.source.{fact} is {note!r}: it does not say declared"
    if case.get("record_stamped"):
        stamp = record["detected_at"]
        assert stamp and datetime.fromisoformat(stamp) >= began, \
            f"record.detected_at is {stamp!r}, older than its probe ({began})"


def ladder_edited(case, bundle) -> bool:
    """The stages a person removes (``removed``) or adds (``added`` --
    ``{name, at, overrides}``, ``at`` the place, the end when absent;
    ``overrides`` what the stage sets of its own, which the description then
    varies, `engines/stages.md` § 6.2), written through Task setup's Save as
    the page writes them -- taken, or refused (:func:`_saved_or_refused`)."""
    import json
    task = json.loads((bundle / "task.json").read_text())
    request = read_request(bundle)
    groups = {n: e["group"] for n, e in request["stages"].items()}
    gone = set(case.get("removed", ()))
    task["stages"] = [st for st in task["stages"] if st["name"] not in gone]
    for new in case.get("added", ()):
        # A NEW STAGE, A NEW GROUP, every stage below it moved one on -- as
        # the page adds one (`task-setup.md` § 5.3).
        at = new.get("at", len(task["stages"]))
        g = groups[task["stages"][at - 1]["name"]] + 1 if at else 1
        for st in task["stages"][at:]:
            groups[st["name"]] += 1
        groups[new["name"]] = g
        task["stages"].insert(at, {"name": new["name"],
                                   **({"overrides": new["overrides"]}
                                      if new.get("overrides") else {})})
        task["varies"] = sorted(set(task.get("varies") or [])
                                | set(new.get("overrides") or {}))
    # THE NUMBERS AGAIN FROM 1, each stage keeping who it shares a job with.
    seen = []
    for st in task["stages"]:
        if groups[st["name"]] not in seen:
            seen.append(groups[st["name"]])
    entries = request["stages"]
    request["stages"] = {
        st["name"]: {**{k: v for k, v in entries.get(st["name"], {}).items()
                        if k != "group"},
                     "group": seen.index(groups[st["name"]]) + 1}
        for st in task["stages"]}
    return _saved_or_refused(case, bundle, task, request)


def _saved_or_refused(case, bundle, task, request) -> bool:
    """Task setup's Save of ``task`` and ``request``: taken -- ``True`` --
    or, the row's ``change_refused``, refused (`web/task-setup.md` § 8):
    `409`, saying each of its words, the way back restoring the state the
    row's ``restores_state`` names, and no state saved and no record
    written -- ``False``, the row ending there."""
    from molbuilder.checkpoint import Repo
    records = ("task.json", "job-request.json")
    was = {f: _as_written(bundle / f) for f in records}
    # A folder saves its first state at its first prep: a Save before that
    # has none to count.
    repo = Repo(str(bundle))
    states = len(repo.states()) if repo.initialized else 0
    r = save_both(bundle, task, request)
    said = (r.get_json() or {}).get("error") or ""
    if "change_refused" not in case:
        assert r.status_code == 200, said
        return True
    assert r.status_code == 409, said
    for words in case["change_refused"]:
        assert words in said, said
    if "restores_state" in case:
        _restores_by_id(case, bundle, said)
    assert {f: _as_written(bundle / f) for f in records} == was, \
        "a refused Save wrote a record"
    assert (len(repo.states()) if repo.initialized else 0) == states, \
        "a refused Save saved a state"
    return False


def _restores_by_id(case, bundle, text) -> None:
    """THE WAY BACK RESTORES THE STATE ITS NOTE NAMES, by its id: a
    `checkpoint restore` line of ``text`` ends with the id of the state
    whose note ends with the row's ``restores_state``."""
    from molbuilder.checkpoint import Repo
    sid = next(st.short for st in Repo(str(bundle)).states()
               if st.note.endswith(case["restores_state"]))
    assert any("molbuilder checkpoint restore" in ln
               and ln.rstrip().endswith(sid)
               for ln in text.splitlines()), text


def _road_source_edited(edit, bundle) -> None:
    """The calculation's structure file edited by hand after it was
    described: its ``atom``-th atom line (0-based) replaced by ``line``,
    the sidecar beside it untouched (`structure-molstruct.md` § 3)."""
    from molbuilder.task import of as task_of
    src = bundle / Path(task_of(bundle).structure.source).name
    lines = src.read_text().split("\n")
    lines[2 + edit["atom"]] = edit["line"]
    src.write_text("\n".join(lines))


def _road_held_edited(held, bundle) -> None:
    """The calculation's structure, its held atoms set anew (``held``) after
    it was described, written back through the codec -- its coordinates and
    its pin unchanged."""
    from molbuilder.task import of as task_of
    from molbuilder.workingcopy_structure import StructureCodec
    src = bundle / Path(task_of(bundle).structure.source).name
    mine = StructureCodec().load(src)
    StructureCodec().write(
        mine.replace(regions={**mine.regions, "frozen_atoms": list(held)}),
        src)


def _road_regrouped(groups, bundle, refused=None, status=409) -> bool:
    """Each stage's group set anew -- ``groups``, one number a stage, in
    order -- through Task setup's Save, as the page's Group column writes
    them: taken, or refused saying each of ``refused`` (the row's
    ``regroup_refused``) with ``status`` (its ``regroup_status``: 409 for
    groups at odds with the folder's runs, `web/task-setup.md` § 8; 400 for
    a job request its own preflight refuses, `execution/job-request.md`
    § 3.3, the Save's answer to a record it refuses) -- ``False`` then, the
    row ending there."""
    import json
    task = json.loads((bundle / "task.json").read_text())
    request = read_request(bundle)
    for st, g in zip(task["stages"], groups):
        request["stages"][st["name"]]["group"] = g
    r = save_both(bundle, task, request)
    if refused:
        assert r.status_code == status, r.get_json()
        for words in refused:
            assert words in r.get_json()["error"], r.get_json()["error"]
        return False
    assert r.status_code == 200, (r.get_json() or {}).get("error")
    return True


def _description_with_overrides(bundle, overrides) -> dict:
    """``bundle``'s description as Task setup's Save would be handed it with
    ``overrides``, ``{stage: {name: value}}``, each name joining `varies`
    (`engines/stages.md` § 6.2)."""
    import json
    task = json.loads((bundle / "task.json").read_text())
    for stage, values in overrides.items():
        st = next(s for s in task["stages"] if s["name"] == stage)
        st["overrides"] = {**st.get("overrides", {}), **values}
        task["varies"] = sorted(set(task["varies"]) | set(values))
    return task


def _road_changed(case, bundle) -> bool:
    """The records changed before the row's prep: ``changed``'s
    ``template``, ``{name: value}``, in the template's file as a person
    changes it (:func:`set_by_the_person`); then, through Task setup's Save,
    its blocks of how it runs into the job request, and its ``overrides``,
    ``{stage: {name: value}}``, into the description, each name joining
    `varies` -- taken, or refused (:func:`_saved_or_refused`)."""
    if "changed" not in case:
        return True
    saved = {k: v for k, v in case["changed"].items() if k != "template"}
    if "template" in case["changed"]:
        from molbuilder.template import run_settings
        on_card = set(case["changed"]["template"]) & set(
            run_settings(case.get("engine", "siesta")))
        # A RUN SETTING IS THE RUN CARD'S (`job-request.md` § 2): a row
        # changes it under ``changed``'s blocks, never the template's.
        assert not on_card, f"run settings under changed.template: {on_card}"
        set_by_the_person(bundle, case["changed"]["template"])
    if not saved:
        return True
    task = _description_with_overrides(bundle, saved.pop("overrides", {}))
    request = read_request(bundle)
    request.update(saved)
    return _saved_or_refused(case, bundle, task, request)


def _road_both_doors(case, kind, named, target, twin, bundle, printed) -> None:
    """The page's prep door on ``twin``, the folder as the terminal's prep
    found it: its preview said as what it would do (``preview_said``,
    ``preview_lacks``), then every line it serves is one the terminal printed,
    in order, none between them -- a state's id and its time apart, which are
    each prep's own (`job-system.md`, *What prep answers*); a finding of
    ``page_found_once`` is drawn once, above the stages or on one, and one of
    ``page_stage_found`` under its stage; a preview's last member holds its
    group's header lines, ``preview_group_header``, what the page shows of the
    group line for line."""
    import re
    from molbuilder.web.app import create_app
    client = create_app(config={}).test_client()
    pick = ({"stages": list(named)} if kind == "bench"
            else {"groups": [int(n) for n in named if n != "--group"]})
    body = dict(pick, dest=str(twin), kind=kind, target=target)

    def lines_of(answer):
        return (answer["above"]["lines"]
                + [line for a in answer["answers"] for line in a["lines"]])

    preview = client.post("/api/task-setup/prep",
                          json=dict(body, plan=True)).get_json()
    assert preview["ok"], preview
    said = "\n".join(lines_of(preview))
    for words in case.get("preview_said", []):
        assert words in said, said
    for words in case.get("preview_lacks", []):
        assert words not in said, said
    for words in case.get("preview_group_header", []):
        assert words in preview["answers"][-1]["group"]["launch"]["header"], \
            preview["answers"][-1]
    done = client.post("/api/task-setup/prep",
                       json=dict(body, plan_id=preview["plan_id"])).get_json()
    assert done["ok"], done

    def plain(line):
        return re.sub(r"(saved first: )\S+ -- \S+ \S+", r"\1",
                      line.replace(str(twin), str(bundle)))

    served = [plain(line) for line in lines_of(done)]
    terminal = [plain(line) for line in printed.splitlines()]
    # ...FROM THE STATE IT SAVED TO THE NEXT STEP: nothing the terminal says of
    # the prep left out of what the page is served.
    start = next((i for i, line in enumerate(terminal)
                  if line.startswith("the folder's state was")), None)
    end = next((i for i, line in enumerate(terminal) if i > (start or 0)
                and line.startswith(("next --", "each group, and what is next"))),
               None)
    assert start is not None and end is not None and (
        terminal[start:end] == served), (served, terminal)
    found = [f["message"] for f in done["above"]["findings"]]
    for a in done["answers"]:
        found += [f["message"] for f in a["findings"] + a["deck_findings"]]
    for words in case.get("page_found_once", []):
        assert sum(words in message for message in found) == 1, found
    for stage, want in case.get("page_stage_found", {}).items():
        (one,) = [a for a in done["answers"] if a["stage"] == stage]
        own = [f["message"] for f in one["findings"] + one["deck_findings"]]
        for words in want:
            assert any(words in message for message in own), (stage, own)


def _road_saved(case, bundle) -> None:
    """The description saved through Task setup's Save -- the page's door,
    which no `jobset` verb reaches -- with ``saved``'s fields changed:
    refused, saying each of ``save_refused``, or taken."""
    if "saved" not in case:
        return
    import json
    from molbuilder.web.app import create_app
    task = json.loads((bundle / "task.json").read_text())
    request = read_request(bundle)
    # EACH CHANGED KEY INTO ITS RECORD: how it runs into the job request,
    # what is computed into the description (`execution/job-request.md` § 1).
    for key, value in case["saved"].items():
        (request if key in ("allocation", "execution", "bench", "notify")
         else task)[key] = value
    client = create_app(config={}).test_client()
    r = save_both(bundle, task, request, client)
    said = (r.get_json() or {}).get("error") or ""
    if "save_refused" not in case:
        assert r.status_code == 200, said
        return
    assert r.status_code == case.get("save_status", 409), (r.status_code,
                                                          said)
    for words in case["save_refused"]:
        assert words in said, said


def _road_own_warm_files(case, bundle) -> None:
    """The calculation's OWN restart-file list (``own_warm_files``), made as
    the shipped file says to make it (`job-contracts.md` § 4.2a): the
    engine's file copied beside ``task.json`` and edited -- ``withhold``
    takes the ``carry`` off those suffixes' rows, ``add`` appends a row
    known and not carried for each, ``resumes`` states its one
    section-level fact for every kind, in ``[base]``."""
    import re
    from molbuilder.task import of as task_of
    from molbuilder.warmfiles import FILENAME, warm_list
    own = case["own_warm_files"]
    engine = str(task_of(bundle).engine)
    text = Path(warm_list(engine).path).read_text()
    for suffix in own.get("withhold", []):
        row = re.compile(r'(suffix\s*=\s*"' + re.escape(suffix)
                         + r'"[^\n]*\n)carry\s*=\s*"when-continuing"[^\n]*\n')
        text, n = row.subn(r"\1", text)
        assert n == 1, f"no carried row for {suffix} to withhold"
    for suffix in own.get("add", []):
        text += f'\n[[base.file]]\nsuffix = "{suffix}"\n'
    if "resumes" in own:
        first = text.index("[[base.file]]")
        text = (text[:first]
                + f"[base]\nresumes = {str(own['resumes']).lower()}\n\n"
                + text[first:])
    (bundle / FILENAME).write_text(text)


def _road_plan_carries(want, bundle) -> None:
    """The files each stage's job declares it carries from a run it
    continues -- its ``warm`` files, in order, as the plan prep wrote holds
    them and the launch reads them (`jobset.model.JobSet.load`):
    ``{stage: [file, ...]}``.  Which of them a launch takes is
    `warm_carry`'s, asked at the launch (`job-contracts.md` § 4.2a)."""
    from molbuilder.jobset.model import JobSet
    plan = JobSet.of(bundle)
    jobs = {j.name: j for j in plan.jobs}
    for stage, files in want.items():
        assert stage in jobs, f"the plan holds no job {stage!r}: {sorted(jobs)}"
        assert [w.name for w in jobs[stage].warm] == files, \
            (stage, [w.name for w in jobs[stage].warm])


def _road_progress_log_holds(want, bundle) -> None:
    """The atoms each progress log the prep wrote states the run holds,
    as its own reader reads them (`model/parse.md` § 5.3)."""
    from molbuilder.parse import detect
    logs = _road_runs(bundle, "*.molwatch.log")
    assert logs, "the prep wrote no progress log"
    for log in logs:
        got = detect(log).parse(str(log)).runtime_info.get("frozen_atoms")
        assert got == list(want), (log.relative_to(bundle), got)


def _road_steps_read(want, bundle) -> None:
    """Each progress log of a run the prep opened, as a person's script reads
    it (`model/overview.md` § 2.2b).  The prep wrote no step: `read_steps`
    refuses the log and the run's folder, where taking the log's input geometry
    for a result would answer a queued run with a last geometry and no energy.
    `parse` hands back the file's own order and says whose, and each structure
    on the answer answers for the run's order: its index, and its atoms put
    back as the person listed them.  A folder that is no run and a file that is
    not there are refused, naming what to do."""
    import numpy as np
    from molbuilder.postprocess import parse, read_steps
    from molbuilder.runs import RunStepsError, run_of
    logs = [p for p in _road_runs(bundle, "*.molwatch.log")
            if run_of(p) is not None]
    assert logs, "the prep opened no run with a progress log"
    for log in logs:
        for asked in (log, log.parent):
            with pytest.raises(RunStepsError,
                               match="holds no step its run finished"):
                read_steps(asked)
        said = parse(log)
        atoms = said.atom_index_map
        assert atoms.listed_by == want["listed_by"]
        assert atoms.stated_by == str(log)
        assert atoms.user != atoms.engine, (
            "this row is to read a run that sorted its atoms")
        struct = said.frames[0].structure
        every = list(range(atoms.n_atoms))
        assert struct.map_to_user_index(every) == atoms.map_to_user_index(every)
        mine = struct.restored()
        assert list(mine.elements) == want["elements"]
        assert np.allclose(atoms.map_rows_to_user_order(struct.positions)[0],
                           mine.positions)
        with pytest.raises(RunStepsError, match="is not there"):
            read_steps(log.parent / "no_such_file.out")
    with pytest.raises(RunStepsError, match="name one of its runs"):
        read_steps(bundle)


def _road_points_before_prep(want, bundle) -> None:
    """Whether each stage runs as a run of points, asked of a calculation
    described and not yet prepared (`jobset.points.runs_points`), and the words
    the run card says it in (`prep.walk_words`), so its one wall is told while
    the wall is typed, never only after the first prep."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.points import runs_points
    from molbuilder.jobset.prep import walk_words
    calc = open_calculation(bundle)
    for stage, words in want.items():
        assert runs_points(calc, stage) is bool(words), stage
        if words:
            assert walk_words(calc, stage) == words


def _road_points_compared(stage, bundle) -> None:
    """What the stage's prep recorded against the records as they read now
    (`job-system.md`, *Changed since prepared*): a point worded otherwise is no
    difference, as a release may reword one, and a point in another folder is."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset import prepared_with as pw
    was = pw.read(bundle)[stage]
    now = pw.values_of(open_calculation(bundle), stage)
    assert was["point_folders"] and pw.differences(was, now) == []
    assert pw.differences({**was, "points": ["said otherwise"]}, now) == []
    moved = pw.differences({**was, "point_folders": ["elsewhere"]}, now)
    assert [d.kind for d in moved] == [pw.POINTS], moved


def _road_progress_log_targets(want, bundle) -> None:
    """The targets each progress log the prep wrote hands the Watch tab, as
    its own reader reads them (`web/trajectory.md` § 3)."""
    from molbuilder.parse import detect
    logs = _road_runs(bundle, "*.molwatch.log")
    assert logs, "the prep wrote no progress log"
    for log in logs:
        got = detect(log).parse(str(log)).runtime_info.get(
            "convergence_targets") or {}
        assert got == want, (log.relative_to(bundle), got)


def _road_after_prep(case, bundle) -> None:
    """What the row's prep left, refused or not -- or, given a row's
    `after_launch`, what its launch left: the folder's saved states, newest
    first (`saved_states`), what `status` says of the calculation
    (`status_says`, and what it does not, `status_lacks`), and the decisions
    its ledger holds (`ledger_holds`)."""
    if "saved_states" in case:
        import re
        from molbuilder.checkpoint import Repo
        repo = Repo(str(bundle))
        got = [st.note for st in repo.states()] if repo.initialized else []
        want = [re.compile(re.escape(w).replace(
                    re.escape("{stamp}"),
                    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}") + "$")
                for w in case["saved_states"]]
        assert len(got) == len(want) and all(
            w.match(g) for w, g in zip(want, got)), f"saved states: {got}"
    for where in case.get("made", []):
        assert (bundle / where).exists(), \
            f"{where} was not made: {sorted(p.name for p in bundle.iterdir())}"
    for where in case.get("made_lacks", []):
        assert not (bundle / where).exists(), f"{where} was made"
    for where, lines in case.get("made_says", {}).items():
        _road_lines({"made_says": lines}, "made_says",
                    (bundle / where).read_text())
    if "card_written" in case:
        _road_card_written(case["card_written"], bundle)
    if "progress_log_holds" in case:
        _road_progress_log_holds(case["progress_log_holds"], bundle)
    if "steps_read" in case:
        _road_steps_read(case["steps_read"], bundle)
    if "progress_log_targets" in case:
        _road_progress_log_targets(case["progress_log_targets"], bundle)
    if "plan_carries" in case:
        _road_plan_carries(case["plan_carries"], bundle)
    if "points_compared" in case:
        _road_points_compared(case["points_compared"], bundle)
    if "status_says" in case or "status_lacks" in case:
        st = jobset("status", "--bundle", bundle)
        assert st.exit_code == 0, _one_line(st)
        for words in case.get("status_says", []):
            assert words in st.output, _one_line(st)
        for words in case.get("status_lacks", []):
            assert words not in st.output, _one_line(st)
    if "ledger_holds" in case:
        from molbuilder.jobset import ledger
        lines = ledger.read(bundle)
        # A DECISION, or a verb's decision (``launch continues``): prep and
        # launch both record what a stage continues from.
        decided = ({e["decision"] for e in lines}
                   | {f"{e['verb']} {e['decision']}" for e in lines})
        for decision in case["ledger_holds"]:
            if isinstance(decision, dict):
                # A DECISION AND ITS FACTS: some line of it holds each.
                want = dict(decision)
                name = want.pop("decision")
                assert any(name in (e["decision"],
                                    f"{e['verb']} {e['decision']}")
                           and all(e.get(k) == v for k, v in want.items())
                           for e in lines), \
                    f"no {name} line holds {want}: {lines}"
                continue
            assert decision in decided, f"the ledger holds: {decided}"


def _road_card_written(specs, bundle) -> None:
    """Every name the Task setup card gives a stage -- the catalogue's
    `manifest`, for the moments asked -- is a file in that stage's folder or
    its runs; a file the catalogue places at the calculation's level --
    the stage's pipeline log -- in the calculation's folder.  A name written
    only sometimes (`only`) is not promised; a field (`<stamp>`) stands for
    any value."""
    import fnmatch
    import re
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runfiles import manifest
    from molbuilder.task import of as task_of
    task = task_of(bundle)
    for spec in specs:
        home = stage_home(bundle, task, spec["stage"])
        at_root = [p.name for p in bundle.iterdir() if p.is_file()]
        held = [p.name for p in home.dir.rglob("*") if p.is_file()]
        for row in manifest(task.label, home.token,
                            engine=task.engine, when=tuple(spec["moments"]),
                            calculation=task.calculation):
            if row["only"]:
                continue
            pattern = re.sub(r"<[a-z_]+>", "*", row["name"])
            where = at_root if row["level"] == "calculation" else held
            assert any(fnmatch.fnmatchcase(n, pattern) for n in where), (
                f"the card names {row['name']} for {spec['stage']}, and "
                f"no such file is there: {sorted(held)}")


def history_is_the_runs_own(out: Path, label: str) -> None:
    """A SIESTA run's history is its own (`job-contracts.md` § 2.2): SIESTA's
    ``<label>.MD.nc``, under SIESTA's name in the run's own folder -- the one
    its output ``out`` lies in -- as the run door finds it
    (`Run.on_label`), and the one the output's reader pairs the output with
    and upgrades its frames from (`siesta_mdnc.sibling_md_nc`, by the label
    the output prints); and it holds this run's steps alone, each row a
    step of that output in its order -- but the last, a move the run
    stopped before evaluating (`model/parse.md` § 5a: row k is the geometry
    after move k+1) -- never an earlier run's, since no hand-over copies a
    history into a new run.  A run that made no move -- relaxed at its
    first step -- writes none, and its output is paired with nothing."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import align_to_reference
    from molbuilder.runs import run_of
    own = run_of(out.parent).on_label(".MD.nc")
    got = parse(str(out))
    if own is None:
        assert "mdnc_source" not in got.runtime_info, got.runtime_info
        return
    assert own == out.parent / f"{label}.MD.nc", (
        f"the run's history is {own}, not SIESTA's own name in the run's "
        f"folder")
    assert got.runtime_info.get("mdnc_source") == own.name, got.runtime_info
    rows = parse(str(own)).frames
    matched = [j for j in align_to_reference(got.frames, rows)
               if j is not None]
    assert matched in (list(range(len(rows))), list(range(len(rows) - 1))), (
        f"{own.name}: {len(rows)} rows, and {out.name}'s frames are "
        f"{matched} of them -- rows that are not this run's steps")


def _as_written(path: Path):
    """A file as it stands: its bytes and its write time to the
    nanosecond -- a file written again with the same bytes is still written
    again, which is what `kept` rows are about."""
    assert path.is_file(), f"{path} is not there to keep"
    return path.read_bytes(), path.stat().st_mtime_ns


def _all_written(bundle: Path) -> dict:
    """Every file under the calculation, by its size and write time -- what a
    step that writes nothing leaves as it was."""
    return {str(p.relative_to(bundle)): (p.stat().st_size,
                                         p.stat().st_mtime_ns)
            for p in sorted(bundle.rglob("*")) if p.is_file()}


def _road_holds(got, want, where) -> None:
    """``want`` -- a table -- is in ``got``, key by key, nested."""
    for key, value in want.items():
        assert isinstance(got, dict) and key in got, \
            f"{where}.{key} missing from {got!r}"
        if isinstance(value, dict):
            _road_holds(got[key], value, f"{where}.{key}")
        else:
            assert got[key] == value, \
                f"{where}.{key} is {got[key]!r}, not {value!r}"


def _over_base(base, mine, unset):
    """A row's block over the table's base, less what the row takes away."""
    out = dict(base or {})
    out.update(mine or {})
    for key in unset or ():
        out.pop(key, None)
    return out


def _road_describe(table, case, tmp_path, monkeypatch) -> Path:
    """`molbuilder describe` of H2 or the case's own structure, then the
    case's template values, its job request's blocks and this machine's
    `molbuilder.json` -- a run setting among the table's or the case's
    template values stated on the calculation's run card, where it lives
    (`execution/job-request.md` § 2), and a stage's own `restart` in its
    overrides, where what it builds on is said."""
    import json
    engine = case.get("engine", "siesta")
    calculation = case.get("calculation", "optimization")
    bundle = describe_calculation(tmp_path, monkeypatch, engine=engine,
                                  calculation=calculation,
                                  structure=case.get("structure"),
                                  stage_strategy=case.get("stage_strategy"),
                                  bias=case.get("bias", "0"))
    from molbuilder.template import run_settings
    on_card = run_settings(engine)
    values = (dict(table.get("siesta_template", {}))
              if engine == "siesta" else {})
    values.update(case.get("template", {}))
    card_values = {k: v for k, v in values.items() if k in on_card}
    values = {k: v for k, v in values.items() if k not in on_card}
    if values:
        set_by_the_person(bundle, values)
    task = json.loads((bundle / "task.json").read_text())
    request = read_request(bundle)
    # THE GROUPS THE ROW STATES, one number a stage, in order -- and each
    # stage's own run card, by name; what says what a stage computes
    # (`restart`) into its overrides.
    for st, g in zip(task["stages"], case.get("groups", ())):
        request["stages"][st["name"]]["group"] = g
    for st in task["stages"]:
        mine = dict(case.get("stage_run", {}).get(st["name"], {}))
        own = {k: mine.pop(k) for k in list(mine) if k not in on_card
               and k not in ("time", "domain")}
        if own:
            st["overrides"] = {**st.get("overrides", {}), **own}
            task["varies"] = sorted(set(task.get("varies") or []) | set(own))
        if mine:
            entry = request["stages"][st["name"]]
            entry["execution"] = {**entry.get("execution", {}), **mine}
    scheduled = case.get("machine", table.get("machine", "this")) \
        != "workstation"
    run = (_over_base(table.get("run_base", {}).get(engine),
                      case.get("run"), case.get("run_unset"))
           if "bench" not in case else case.get("run"))
    blocks = {
        "execution": ({**card_values, **(run or {})} or None),
        "allocation": _over_base(
            table.get("allocation_base") if scheduled else None,
            case.get("allocation"), case.get("allocation_unset")),
        "bench": case.get("bench"),
        "notify": case.get("notify"),
    }
    for block, value in blocks.items():
        if value:
            request[block] = value
        else:
            request.pop(block, None)   # the row states none of it
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    write_request(bundle, request)
    if "probe" not in case:
        _write_machine_config(case)
    return bundle


def _write_machine_config(case) -> None:
    """THIS machine's `molbuilder.json`, as the row gives it -- written as
    it stands, so a row may hand it a section the reader refuses."""
    if "machine_config" in case:
        import json
        from molbuilder.runtime_config import machine_config_path
        from molbuilder.config_dir import ensure_private_dir
        mine = machine_config_path()
        ensure_private_dir(mine.parent)
        mine.write_text(json.dumps(case["machine_config"]))


def _deck_says(engine, text):
    """What the engine reads from a deck, keyword by keyword --
    ``({key: value}, the key a keyword is filed under)``: SIESTA's through
    the one fdf reader (`parse.fdf.read_fdf`), each value its words joined
    by one space; a PySCF deck's names and attributes assigned at the top
    level, as Python runs them -- the last assignment standing, its value a
    literal or its source."""
    if engine == "siesta":
        from molbuilder.parse.fdf import fdf_key, read_fdf
        scalars, _ = read_fdf(text)
        return {k: " ".join(v) for k, v in scalars.items()}, fdf_key
    import ast
    said = {}

    def top(body):
        for node in body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, (ast.Name, ast.Attribute)):
                        try:
                            value = ast.literal_eval(node.value)
                        except ValueError:
                            value = ast.unparse(node.value)
                        said[ast.unparse(target)] = value
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For)):
                for part in ("body", "orelse", "finalbody"):
                    top(getattr(node, part, []))
                for handler in getattr(node, "handlers", []):
                    top(handler.body)
    top(ast.parse(text).body)
    return said, (lambda keyword: keyword)


def _road_lines(case, key, text):
    """``key``'s lines are in ``text``; ``key_lacks``'s are not;
    ``key_order``'s stand in that order -- every one of a line before the
    first of the next; ``key_once``'s stand once each -- read with runs of
    blanks as one, since a deck aligns its values in columns."""
    import re
    flat = re.sub(r"[ \t]+", " ", text)
    for line in case.get(key, []):
        assert line in flat, \
            f"{key}: {line!r} missing from: {_one_line(text)}"
    for line in case.get(f"{key}_lacks", []):
        assert line not in flat, \
            f"{key}: {line!r} present in: {_one_line(text)}"
    order = case.get(f"{key}_order", [])
    for line in order:
        assert line in flat, \
            f"{key}: {line!r} missing from: {_one_line(text)}"
    for before, after in zip(order, order[1:]):
        assert flat.rindex(before) < flat.index(after), \
            f"{key}: {before!r} stands after {after!r}"
    for line in case.get(f"{key}_once", []):
        assert flat.count(line) == 1, \
            f"{key}: {line!r} stands {flat.count(line)} times"


def _road_runs(bundle: Path, pattern: str) -> "list[Path]":
    """The files of this pattern the run's prep wrote, outside a bench."""
    return [p for p in bundle.rglob(pattern)
            if "bench" not in p.relative_to(bundle).parts]


def _the_runs(bundle: Path, pattern: str) -> Path:
    """The file of this name the run's prep wrote -- in the stage's
    folder and its first run, one text in both."""
    found = _road_runs(bundle, pattern)
    assert found and len({p.read_text() for p in found}) == 1, found
    return found[-1]


def shelf_headers(bundle: Path, stage: str, *, side=None) -> dict:
    """The shelves of ``stage``'s prepared benchmark as its launch would
    send them -- planned by the launch verb's own planner
    (`submit.plan_launch`), nothing sent and nothing written -- each with
    the request its line sends, the header prep wrote read back by the
    header's reader (`Submission.request`): ``{shelf: its #SBATCH lines}``,
    in the order the shelves are sent; ``side`` is `--only`'s."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset.submit import plan_launch
    from molbuilder.paths import bench_container
    from molbuilder.task import of as task_of
    task = task_of(bundle)
    container = bundle / bench_container(stage_home(bundle, task,
                                                    stage).token)
    plan = plan_launch(open_calculation(bundle),
                       bench=(JobSet.of(container), container),
                       mode="submit", side=side, record=False,
                       told=dict(kind="bench", stage=stage, trial=None,
                                 mode="submit", flags=[]))
    return {s.name: list(s.request()) for s in plan.submissions}


def stage_file(bundle: Path, stage: str, role: str) -> Path:
    """The file ``stage``'s prep wrote for ``role`` -- ``.sbatch``,
    ``.run.sh``, the deck's suffix -- by the one composer's names
    (`runfiles.RunNames`): on the calculation's label, its token."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    names = stage_home(bundle, task_of(bundle), stage).names
    return _the_runs(bundle, names.name(role))


def _one_line(text) -> str:
    """A failure's text on ONE line: `tools/testrun.py` reports
    ``--tb=line``, which shows a message's first line and nothing else --
    and a command's first line is its config banner, never its answer.
    Handed a command's RESULT, it adds the traceback of an exception the
    command did not turn into a refusal, which its output never shows."""
    if not isinstance(text, str):
        import traceback
        r, text = text, text.output
        if r.exception is not None and not isinstance(r.exception,
                                                      SystemExit):
            text += "".join(traceback.format_exception(*r.exc_info)[-12:])
    return " ⏎ ".join(ln.strip() for ln in text.splitlines() if ln.strip())


def _named(kind: str, stage, bundle, *, verb: str = "prep") -> tuple:
    """How a row's stage is typed (`job-system.md` § 5.3): a benchmark's by
    position; a task's prep by the group each stage is in, as the
    calculation's own description numbers it (``--group``); a task's launch
    by ``--stage`` for one stage, by its group for several.  A name the
    description does not hold is typed as named, for its refusal."""
    if not stage:
        return ()
    stages = stage if isinstance(stage, list) else [stage]
    if kind == "bench":
        return tuple(stages)
    if verb == "launch" and len(stages) == 1:
        return ("--stage", stages[0])
    # THE STAGE AS EVERY VERB RESOLVES IT -- a name in any case, or `#N`,
    # the number its folder carries (`identity.resolve_stage_ref`).
    from molbuilder.identity import StageRef, resolve_stage_ref
    from molbuilder.jobset.naming import ladder_homes
    from molbuilder.task import of as task_of
    task = task_of(Path(bundle))
    refs = [StageRef(h.seq, h.name) for h in ladder_homes(Path(bundle), task)]
    groups = []
    for s in stages:
        try:
            name = resolve_stage_ref(refs, s).name
        except ValueError:
            return tuple(x for s in stages for x in ("--stage", s))
        g = read_request(Path(bundle))["stages"][name]["group"]
        if g not in groups:
            groups.append(g)
    return tuple(x for g in groups for x in ("--group", str(g)))


def group_words(bundle, *stages) -> tuple:
    """``--group N`` for the groups ``stages`` are in, as the calculation's
    own job request numbers them -- how a person names them to `prep task`
    (`job-system.md` § 5.3)."""
    return _named("task", list(stages), bundle)


def run_road_case(table, case, tmp_path, monkeypatch, *, ran=None) -> None:
    """ONE ROW of a contract's case table, down the road, every layer it
    names checked -- on the calculation it describes, or on ``ran``, a copy
    of one a real run made (`support.real_runs.copy_of`), which it starts
    from as it stands."""
    import json
    target = _road_target(table, case, tmp_path, monkeypatch)
    if case.get("ends_at") == "probe":
        return
    bundle = (ran if ran is not None
              else _road_describe(table, case, tmp_path, monkeypatch))
    if "points_before_prep" in case:
        _road_points_before_prep(case["points_before_prep"], bundle)
    if "own_warm_files" in case:
        _road_own_warm_files(case, bundle)
    if case.get("saved_first"):
        from click.testing import CliRunner
        from molbuilder.cli import cli
        got = CliRunner().invoke(cli, ["checkpoint", "init", "-p",
                                       str(bundle), "-m", "set up"])
        assert got.exit_code == 0, _one_line(got)
    for words in case.get("before", []):
        # A machine is named at prep; the other verbs read the one prep set.
        got = jobset(*words, "--bundle", bundle,
                     *(("--target", target) if words[0] == "prep" else ()))
        assert got.exit_code == 0, f"{words}: {_one_line(got)}"
    if not _road_changed(case, bundle):
        return
    if ("removed" in case or "added" in case) and not ladder_edited(
            case, bundle):
        return
    if "source_edited" in case:
        _road_source_edited(case["source_edited"], bundle)
    if "held_edited" in case:
        _road_held_edited(case["held_edited"], bundle)
    if "regrouped" in case and not _road_regrouped(
            case["regrouped"], bundle, case.get("regroup_refused"),
            case.get("regroup_status", 409)):
        return
    kind = "bench" if "bench" in case else "task"
    kept = {where: _as_written(bundle / where)
            for where in case.get("kept", [])}
    named = _named(kind, case.get("stage", "coarse"), bundle)
    # THE SAME FOLDER FOR THE PAGE'S DOOR, as it stands before this prep.
    twin = None
    if case.get("both_doors"):
        import shutil
        twin = bundle.with_name(bundle.name + "-page")
        shutil.copytree(bundle, twin)
    r = jobset("prep", kind, *named, "--bundle", bundle,
               "--target", target, *case.get("prep", []))
    for where, was in kept.items():
        assert _as_written(bundle / where) == was, \
            f"{where} was written again by the row's prep"

    # 1 · ALLOWED OR REFUSED -- at prep, in its words
    if "refused" in case:
        # One sentence, or several of one refusal -- a row is one input, so
        # what its refusal must say is listed on it, never a second row.
        said = case["refused"]
        assert r.exit_code != 0, _one_line(r)
        for words in ([said] if isinstance(said, str) else said):
            assert words in r.output, _one_line(r)
        if "restores_state" in case:
            _restores_by_id(case, bundle, r.output)
        _road_after_prep(case, bundle)
        _road_saved(case, bundle)
        return
    assert r.exit_code == 0, _one_line(r)
    for words in case.get("said", []):
        assert words in r.output, _one_line(r)
    for words in case.get("said_lacks", []):
        assert words not in r.output, _one_line(r)
    for words in case.get("said_once", []):
        assert r.output.count(words) == 1, (words, _one_line(r))
    if twin is not None:
        _road_both_doors(case, kind, named, target, twin, bundle, r.output)
    _road_after_prep(case, bundle)
    _road_saved(case, bundle)

    # 2 · WHAT IS PRODUCED -- the run's header, deck and run script
    if case.get("header_absent"):
        assert not _road_runs(bundle, "*.sbatch"), _road_runs(bundle,
                                                              "*.sbatch")
    deck = ".py" if case.get("engine") == "pyscf" else ".fdf"
    for key, role in (("header", ".sbatch"), ("deck", deck),
                      ("run_sh", ".run.sh")):
        if any(k in case for k in (key, f"{key}_lacks", f"{key}_order",
                                   f"{key}_once")):
            _road_lines(case, key, stage_file(
                bundle, case.get("stage", "coarse"), role).read_text())
    # ...and the deck as its engine reads it, keyword by keyword
    if "deck_says" in case:
        text = stage_file(bundle, case.get("stage", "coarse"),
                          deck).read_text()
        said, key = _deck_says(case.get("engine", "siesta"), text)
        for keyword, want in case["deck_says"].items():
            assert key(keyword) in said, \
                f"deck_says: {keyword} is not read from: {_one_line(text)}"
            have = said[key(keyword)]
            assert (have == pytest.approx(want)
                    if isinstance(want, float) else have == want), \
                (keyword, have, want)
    # ...and the deck as its readers read it: through the one fdf reader,
    # field by field, never by its lines
    if "deck_reads" in case:
        from molbuilder.parse.fdf import parse_fdf_params
        got = parse_fdf_params(stage_file(
            bundle, case.get("stage", "coarse"), ".fdf").read_text())
        for field, want in case["deck_reads"].items():
            have = getattr(got, field)
            assert (have == want if isinstance(want, (bool, str))
                    else have == pytest.approx(want)), (field, have, want)
    # ...a benchmark's trials, when the row names them
    if kind == "bench" and any(k in case for k in (
            "bench_gres", "bench_header", "bench_header_lacks")):
        plan = next(bundle.rglob("bench/job-set.json"))
        asked = sorted({j["resources"]["gres"]
                        for j in json.loads(plan.read_text())["jobs"]})
        assert asked == sorted(case.get("bench_gres", asked)), asked
        headers = sorted(plan.parent.rglob("*.sbatch"))
        assert headers or not ({"bench_header", "bench_header_lacks"}
                               & set(case)), \
            "bench_header: the benchmark wrote no .sbatch"
        for header in headers:
            _road_lines({"h": case.get("bench_header", []),
                         "h_lacks": case.get("bench_header_lacks", [])}, "h",
                        header.read_text())
    # ...and the `sbatch` line(s) launch shows -- one per shelf of a
    # benchmark -- or its refusal
    if "launch" in case:
        mode = case.get("launch_mode", "submit")
        was = _all_written(bundle)
        stage_typed = case.get("launch_stage", "coarse")
        r = jobset("launch", kind, *_named(kind, stage_typed or None, bundle,
                                           verb="launch"),
                   "--bundle", bundle, *(("--mode", mode) if mode else ()),
                   *(() if case.get("launch_sends") else ("--dry-run",)),
                   "--yes", *case["launch"])
        if "launch_refused" in case:
            said = case["launch_refused"]
            assert r.exit_code != 0, _one_line(r)
            for words in ([said] if isinstance(said, str) else said):
                assert words in r.output, _one_line(r)
        else:
            assert r.exit_code == 0, _one_line(r)
            sent = [ln for ln in r.output.splitlines()
                    if any(_submits(w) for w in ln.split())]
            # A LINE TO SHOW, unless the launch runs here or the machine's
            # record names no scheduler to send it to.
            assert sent or mode == "direct" or case.get(
                "machine", table.get("machine", "this")) == "workstation", \
                _one_line(r)
            for ln in sent:
                _road_lines(case, "line", " ".join(sbatch_line(ln)))
        for words in case.get("listing", []):
            assert words in r.output, _one_line(r)
        if "scheduler_called" in case:
            # WHAT MOLBUILDER PUT TO THE RECORD'S SCHEDULER, as its submit
            # command wrote each call down (`a_machine_with_queues`).
            made = [argv for _where, argv in
                    calls_made(tmp_path / "sbatch-calls.log")]
            want = case["scheduler_called"]
            assert len(made) == len(want) and all(
                all(w in argv for w in words)
                for argv, words in zip(made, want)), (made, want)
        if case.get("launch_writes_nothing"):
            now = _all_written(bundle)
            assert now == was, sorted(set(now.items()) ^ set(was.items()))
        if "after_launch" in case:
            _road_after_prep(case["after_launch"], bundle)


def set_by_the_person(calculation: Path, values: dict) -> None:
    """Items of ``calculation``'s template changed as a person changes them
    in the file: each value of ``values``, said to be the person's
    (`engines/template.md` § 6.6), written by `template.with_values_set` --
    the template found by its door (`calculation.open`), never named here."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.engine_table import config_class
    from molbuilder.template import one, read_template, with_values_set
    calc = open_calculation(calculation)
    path = calc.template.path
    path.write_text(with_values_set(
        calc.template_text, config_class(str(calc.task.engine)),
        calculation=str(calc.task.calculation), values=values,
        source="person"))
    read = read_template(path.read_text())
    assert all(one(read, k).source == "person" for k in values), values
