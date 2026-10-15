"""The road a test drives -- `jobset init`, `prep`, `launch` -- and the one
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
* :func:`describe_calculation` -- `jobset init` of a held H2 in a box, or
  the structure a row gives, SIESTA, the shipped `publishable` ladder
  (coarse, medium);
* :func:`ladder_of_freq_alone`, :func:`stage_added` -- a description's
  ladder edited as a person edits it;
* :func:`a_machine_with_queues` -- a machine record naming queues, and an
  `sbatch` on PATH that writes down every call -- where it was made and
  what it said -- and refuses it: the basic tests send nothing to a
  scheduler and assume nothing of its answers (user, 2026-10-06);
* :func:`sbatch_line` -- the `sbatch` line a launch showed, as its words.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

import numpy as np
import pytest


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


def road_structure(s: dict):
    """A road structure -- :data:`H2`, :data:`CO2`, a row's -- as a
    ``Structure``: its ``elements``, ``positions`` (Å), ``regions``
    (``frozen_atoms`` among them), its box's three edges or three vectors,
    ``cell`` (Å), and each axis's ``axis_kind`` (isolated unless given).
    THE ONE BUILDER of the road's structures."""
    from molbuilder.structure import Structure
    return Structure(
        elements=list(s["elements"]),
        positions=np.array(s["positions"], dtype=float),
        regions={k: list(v) for k, v in s.get("regions", {}).items()},
        cell=(np.asarray(s["cell"], dtype=float) if np.ndim(s["cell"]) == 2
              else np.diag([float(a) for a in s["cell"]])),
        axis_kind=tuple(s.get("axis_kind", ["isolated"] * 3)))


def co2_in_engine_order():
    """The road's CO2 as a SIESTA vibration's decks hold it: sorted by its
    decks' key (`transport.sort.sort_by`, `deck_sort_key`), its order record
    on it."""
    from molbuilder.transport.sort import deck_sort_key, sort_by
    return sort_by(road_structure(CO2),
                   deck_sort_key("siesta", "vibration")).structure


def describe_calculation(tmp_path, monkeypatch, *,
                         shape: str = "hierarchical", name: str = "H2",
                         calculation: str = "optimization",
                         engine: str = "siesta",
                         structure: "dict | None" = None,
                         stage_strategy: "str | None" = None) -> Path:
    """`jobset init` on ``structure`` -- :data:`H2` unless given: its
    ``elements``, ``positions`` (Å), ``regions`` (``frozen_atoms`` among
    them), its box's three edges, ``cell`` (Å), and each axis's
    ``axis_kind`` (isolated unless given) -- the bundle, at
    ``<projects>/P/<calculation>/<name>``: a SIESTA optimization's shipped
    `publishable` ladder, unless ``stage_strategy`` names another (``""``
    for the one stage the template alone describes), or a vibration's own
    (`relax`, `freq`).

    ITS RUN CARD STATES THE LAUNCH SHAPE -- two ranks and one thread for
    SIESTA, one thread for PySCF -- as a described calculation does: a run
    whose shape is stated nowhere is refused at prep (`architecture.md`
    § 5.2).  A test about that refusal takes the card away."""
    from conftest import write_pseudos
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.workingcopy_structure import StructureCodec
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    s = structure or H2
    write_pseudos(tree / "pseudopotential", sorted(set(s["elements"])))
    StructureCodec().write(road_structure(s), tree / STRUCTURE)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    siesta = engine == "siesta"
    strategy = (stage_strategy if stage_strategy is not None
                else "publishable" if calculation == "optimization" and siesta
                else "")
    r = jobset("init", "--structure", STRUCTURE,
               "--bundle", f"P/{calculation}/{name}", "--engine", engine,
               "--shape", shape, "--name", name,
               "--calculation", calculation,
               *(("--stage-strategy", strategy) if strategy else ()),
               *(("--psml-lib", "pseudopotential") if siesta else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / calculation / name
    import json
    task = json.loads((bundle / "task.json").read_text())
    task.setdefault("execution", {"mpi_np": 2, "omp_threads": 1} if siesta
                    else {"threads": 1})
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    return bundle


def ladder_of_freq_alone(bundle) -> None:
    """The vibration ladder cut to its `freq` stage alone, as a person edits
    the description: the one stage left is the ladder's first, so its group
    is 1 (`engines/stages.md` § 2)."""
    import json
    path = Path(bundle) / "task.json"
    task = json.loads(path.read_text())
    task["stages"] = [dict(s, group=1) for s in task["stages"]
                      if s["name"] == "freq"]
    path.write_text(json.dumps(task, indent=2))


def stage_added(bundle, stage: dict) -> None:
    """``stage`` -- its ``name``, ``group`` and ``overrides`` -- added to the
    description's ladder, the description varying what it overrides
    (`engines/stages.md` § 6.2), as a person edits it."""
    import json
    path = Path(bundle) / "task.json"
    task = json.loads(path.read_text())
    task["varies"] = sorted(set(task.get("varies") or [])
                            | set(stage["overrides"]))
    task["stages"].append({"enabled": True, **stage})
    path.write_text(json.dumps(task, indent=2))


def a_machine_with_queues(tmp_path, monkeypatch, domains,
                          **record) -> Path:
    """This machine's record names ``domains`` (`scheduler.Domain` rows) --
    and ``record``'s other fields, when given -- and the `sbatch` first on
    PATH writes down each call and REFUSES it, in molbuilder's words.
    Returns the file each call is written to, one line each: ``<where it
    was run> | <its arguments>``.

    NO SCHEDULER TEXT (user, 2026-10-06: *"there should be no assumption
    what so ever about the text returned by slurm ... the whole default test
    set should never be based on fabricated text"*).  What a
    scheduler answers is read where one answers -- the field tier
    (`tests/field/`, `testing.md` § 0).  Refusing also keeps the basic suite
    from sending a real job when it runs on a cluster's login node."""
    from conftest import write_machine_record
    write_machine_record(scheduler="slurm", domains=list(domains), **record)
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
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return calls


def sbatch_line(output: str):
    """The `sbatch` line a launch showed -- in its question, or as a dry
    run's ``WOULD run`` -- as its arguments."""
    for ln in output.splitlines():
        words = ln.split()
        if "sbatch" in words:
            words = words[words.index("sbatch"):]
            return [w for w in words if not w.startswith("[")]
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
# and every row runs down the road a person runs -- `jobset init`, the
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
#      refuses (W57 T1) -- and what the launch left
#      (`launch_writes_nothing`: every file under the calculation as it was;
#      `after_launch`: the after-prep checks below, asked again);
#   2. WHAT IS PRODUCED -- the `.sbatch` header (`header`, or
#      `header_absent`), the deck (`deck`; and `deck_reads`, what the one fdf
#      reader -- `parse.fdf.parse_fdf_params`, the reader every consumer of
#      a deck asks -- reads from it, field by field), the run script
#      (`run_sh`), each
#      with a `_lacks` twin -- and the first three with an `_order` twin
#      (lines standing in that order) and a `_once` twin (lines standing
#      once each); the `sbatch` line launch shows (`line`,
#      `line_lacks`); a benchmark's trials (`bench_gres`,
#      `bench_header_lacks`).
#
# INPUTS: `engine` (siesta | pyscf); `groups` -- each stage's group, in
# order, over the ones `jobset init` gave; `stage_run` -- a stage's own run
# card, `{name = {...}}`; `run` -- the run card, task.json
# `execution`, over the table's `run_base` for the engine (a row with
# `bench`: its own `run` alone); `allocation` --
# task.json `allocation`, over the table's `allocation_base` where the
# machine has a scheduler; `run_unset` / `allocation_unset` -- base keys a row
# takes away; `template` -- values in the template (SIESTA's over the
# table's `siesta_template`); `bench` -- task.json `bench`, then `prep bench`;
# `notify` -- task.json `notify`, when the calculation speaks up;
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
# `record` -- more fields of THIS machine's record; `named_record` -- more
# fields of the named target's; `queues` -- replaces the table's menu;
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
# and the target named as the row's own prep names them; `removed` / `added` -- the
# stages then removed, or added (`{name, at}`, `at` the place, the end when
# absent), through the same Save; `regrouped` -- each stage's group set
# anew, one number a stage, through the same Save; `source_edited` --
# `{atom, line}`: the calculation's structure file edited by hand after it was
# described, its `atom`-th atom line replaced by `line`, its sidecar
# untouched; `held_edited` -- its held atoms set anew after it was
# described, written back through the codec; `own_warm_files` -- the calculation's own restart-file
# list, the engine's
# copied beside `task.json` with `withhold` / `add` / `resumes`
# (`_road_own_warm_files`),
# before anything is prepared; `stage` -- the
# stage (or a list of stages) the row's prep names, `coarse` unless given --
# typed as the group each is in; `answers` -- what
# the person types at the row's prep's question ("" is EOF, no terminal);
# `calculation` -- the kind `jobset init` describes, an optimization unless
# given; `shape` -- the shape it describes, hierarchical unless given;
# `structure` -- the structure it describes, H2 unless given
# (`describe_calculation`: `elements`, `positions`, `regions`, `cell`,
# `axis_kind`); `stage_strategy` -- the ladder `jobset init` is told,
# `publishable` for a SIESTA optimization unless given, `""` for the one
# stage the template alone describes.
#
# AFTER PREP, whatever it answered: the folders it left (`made`, paths
# under the calculation that exist; `made_lacks`, ones that do not), the
# files it left as they were (`kept`: paths under the calculation whose
# bytes and write time are the same before and after the row's prep), every
# file the Task setup card names for a stage on disk as it names it
# (`card_written`: `{stage, moments}` each -- `_road_card_written`), the
# atoms each of its progress logs states the run holds, read back by
# the log's own reader (`progress_log_holds` -- `_road_progress_log_holds`),
# the
# folder's saved states, newest first
# (`saved_states`, their notes -- `{stamp}` standing for the time a note
# leads with, `2026-10-03 14:05:12`), what `status` says of the calculation
# (`status_says`, and what it does not, `status_lacks`), and the decisions
# its ledger holds and does not hold (`ledger_holds`, `ledger_lacks` -- a
# decision, or a verb's, `"launch continues"`; in `ledger_holds`, a table
# names one with its facts, `{decision = "launch launched", time = "3h"}`);
# a prep that was not refused says nothing of `said_lacks`.
# THEN, refused or not: the description saved through Task setup's Save with
# `saved`'s fields changed (`{shape = "flat"}`) -- refused, with
# `save_refused`'s words, the sentence the folder answer gave the shape card
# before (`shape_fixed`), or taken.  A REFUSED prep, its remedy done: this
# machine re-probed holding `reprobed` (its record's fields), the same prep
# is typed again and taken.
#
# 0 · WHAT THE PROBE RECORDS, checked before anything else: what it says
# (`probe_said`, lines any of the probes printed) and what the record it
# wrote holds (`record_says`, a table of the record's fields, nested as the
# file nests them; `record_lacks`, the fields it leaves out;
# `record_declared`, the facts its `source` says were declared -- `flag`
# among the note's parts, `configuration.md` M-1;
# `record_stamped`, its `detected_at` is the last probe's own -- M-6: the
# stamp follows the probe whatever survives).  A row about the record alone
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
    fields = dict(scheduler="slurm", domains=queues,
                  topology=Topology(sockets=2, cores_per_socket=24),
                  env_init={"activation": "conda activate",
                            "preamble": "true"})
    fields.update(case.get("named_record", {}))
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
    for fact in case.get("record_declared", []):
        note = record["source"].get(fact, "")
        assert "flag" in note.split("+"), \
            f"record.source.{fact} is {note!r}: it does not say declared"
    if case.get("record_stamped"):
        stamp = record["detected_at"]
        assert stamp and datetime.fromisoformat(stamp) >= began, \
            f"record.detected_at is {stamp!r}, older than its probe ({began})"


def _road_ladder_edited(case, bundle) -> None:
    """The stages a person removes (``removed``) or adds (``added`` --
    ``{name, at}``, ``at`` the place, the end when absent) after the steps
    before it, written through Task setup's Save as the page writes them."""
    import json
    from molbuilder.web.app import create_app
    task = json.loads((bundle / "task.json").read_text())
    gone = set(case.get("removed", ()))
    task["stages"] = [st for st in task["stages"] if st["name"] not in gone]
    for new in case.get("added", ()):
        # A NEW STAGE, A NEW GROUP, every stage below it moved one on -- as
        # the page adds one (`task-setup.md` § 5.3).
        at = new.get("at", len(task["stages"]))
        g = task["stages"][at - 1]["group"] + 1 if at else 1
        for st in task["stages"][at:]:
            st["group"] += 1
        task["stages"].insert(at, {"name": new["name"], "group": g})
    # THE NUMBERS AGAIN FROM 1, each stage keeping who it shares a job with.
    seen = []
    for st in task["stages"]:
        if st["group"] not in seen:
            seen.append(st["group"])
    for st in task["stages"]:
        st["group"] = seen.index(st["group"]) + 1
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    assert r.status_code == 200, (r.get_json() or {}).get("error")


def _road_source_edited(edit, bundle) -> None:
    """The calculation's structure file edited by hand after it was
    described: its ``atom``-th atom line (0-based) replaced by ``line``,
    the sidecar beside it untouched (`structure-molstruct.md` § 3)."""
    from molbuilder.task import read_task
    src = bundle / Path(read_task(bundle / "task.json").structure.source).name
    lines = src.read_text().split("\n")
    lines[2 + edit["atom"]] = edit["line"]
    src.write_text("\n".join(lines))


def _road_held_edited(held, bundle) -> None:
    """The calculation's structure, its held atoms set anew (``held``) after
    it was described, written back through the codec -- its coordinates and
    its pin unchanged."""
    from molbuilder.task import read_task
    from molbuilder.workingcopy_structure import StructureCodec
    src = bundle / Path(read_task(bundle / "task.json").structure.source).name
    mine = StructureCodec().load(src)
    StructureCodec().write(
        mine.replace(regions={**mine.regions, "frozen_atoms": list(held)}),
        src)


def _road_regrouped(groups, bundle) -> None:
    """Each stage's group set anew -- ``groups``, one number a stage, in
    order -- through Task setup's Save, as the page's Group column writes
    them."""
    import json
    from molbuilder.web.app import create_app
    task = json.loads((bundle / "task.json").read_text())
    for st, g in zip(task["stages"], groups):
        st["group"] = g
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    assert r.status_code == 200, (r.get_json() or {}).get("error")


def _road_saved(case, bundle) -> None:
    """The description saved through Task setup's Save -- the page's door,
    which no `jobset` verb reaches -- with ``saved``'s fields changed:
    refused, saying each of ``save_refused`` in the sentence the folder
    answer gave the shape card (`shape_fixed`), or taken."""
    if "saved" not in case:
        return
    import json
    from molbuilder.web.app import create_app
    task = json.loads((bundle / "task.json").read_text())
    shape = task.get("shape")
    task.update(case["saved"])
    client = create_app(config={}).test_client()
    # What the shape card is told before anything is typed: the same answer
    # the save refuses with (`task-setup.md` § 4).
    card = client.get("/api/task-setup/folder",
                      query_string={"dir": str(bundle)}).get_json()
    r = client.post("/api/task-setup/save",
                    json={"dest": str(bundle), "text": json.dumps(task)})
    said = (r.get_json() or {}).get("error") or ""
    if "save_refused" not in case:
        assert r.status_code == 200, said
        if "shape" in case["saved"]:          # another shape, taken: free
            assert card["shape_fixed"] is None, card["shape_fixed"]
        return
    assert r.status_code == 409, (r.status_code, said)
    for words in case["save_refused"]:
        assert words in said, said
    assert card["shape_fixed"] == {"shape": shape, "why": said}, (
        card["shape_fixed"])


def _reprobed(fields) -> None:
    """This machine's record written anew, with ``fields`` -- what `jobset
    probe --write` writes once the machine has changed."""
    import dataclasses
    from molbuilder.scheduler import (machine_scope_path, read_environment,
                                      write_environment)
    at = machine_scope_path()
    write_environment(dataclasses.replace(read_environment(at), **fields), at)


def _road_own_warm_files(case, bundle) -> None:
    """The calculation's OWN restart-file list (``own_warm_files``), made as
    the shipped file says to make it (`job-contracts.md` § 4.2a): the
    engine's file copied beside ``task.json`` and edited -- ``withhold``
    takes the ``carry`` off those suffixes' rows, ``add`` appends a row
    known and not carried for each, ``resumes`` states its one
    section-level fact for every kind, in ``[base]``."""
    import re
    from molbuilder.task import read_task
    from molbuilder.warmfiles import FILENAME, warm_list
    own = case["own_warm_files"]
    engine = str(read_task(bundle / "task.json").engine)
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


def _road_progress_log_holds(want, bundle) -> None:
    """The atoms each progress log the prep wrote states the run holds,
    as its own reader reads them (`model/parse.md` § 5.3)."""
    from molbuilder.parse import detect
    logs = _road_runs(bundle, "*.molwatch.log")
    assert logs, "the prep wrote no progress log"
    for log in logs:
        got = detect(log).parse(str(log)).runtime_info.get("frozen_atoms")
        assert got == list(want), (log.relative_to(bundle), got)


def _road_after_prep(case, bundle) -> None:
    """What the row's prep left, refused or not -- or, given a row's
    `after_launch`, what its launch left: the folder's saved states, newest
    first (`saved_states`), what `status` says of the calculation
    (`status_says`, and what it does not, `status_lacks`), and the decisions
    its ledger holds and does not hold (`ledger_holds`, `ledger_lacks`)."""
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
    if "card_written" in case:
        _road_card_written(case["card_written"], bundle)
    if "progress_log_holds" in case:
        _road_progress_log_holds(case["progress_log_holds"], bundle)
    if "status_says" in case or "status_lacks" in case:
        st = jobset("status", "--bundle", bundle)
        assert st.exit_code == 0, _one_line(st)
        for words in case.get("status_says", []):
            assert words in st.output, _one_line(st)
        for words in case.get("status_lacks", []):
            assert words not in st.output, _one_line(st)
    if "ledger_lacks" in case or "ledger_holds" in case:
        import json
        from molbuilder.jobset.ledger import LEDGER_FILE
        log = bundle / LEDGER_FILE
        lines = [json.loads(x) for x in
                 (log.read_text().splitlines() if log.is_file() else [])
                 if x.strip()]
        # A DECISION, or a verb's decision (``launch continues``): prep and
        # launch both record what a stage continues from.
        decided = ({e["decision"] for e in lines}
                   | {f"{e['verb']} {e['decision']}" for e in lines})
        for decision in case.get("ledger_lacks", []):
            assert decision not in decided, f"the ledger holds: {decided}"
        for decision in case.get("ledger_holds", []):
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
    `manifest`, in the calculation's shape, for the moments asked -- is a file
    in that stage's folder or its attempts (the hierarchy), or in the folder
    itself (flat); a file the catalogue places at the calculation's level --
    the stage's pipeline log -- in the calculation's folder, whatever the
    shape.  A name written only sometimes (`only`) is not promised; a field
    (`<stamp>`) stands for any value."""
    import fnmatch
    import re
    from molbuilder.jobset.materialize import stage_home
    from molbuilder.runfiles import manifest
    from molbuilder.task import read_task
    task = read_task(bundle / "task.json")
    for spec in specs:
        token = stage_home(bundle, task, spec["stage"]).token
        at_root = [p.name for p in bundle.iterdir() if p.is_file()]
        held = ([p.name for p in (bundle / token).rglob("*") if p.is_file()]
                if task.shape == "hierarchical" else at_root)
        for row in manifest(task.label, token, shape=task.shape,
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
    own name for it, ``<label>.MD.nc``, not left in the folder once the engine
    has exited; and the run's ``<label>_<stage>-run<N>.MD.nc`` beside its
    output ``out``, which the output's frames are upgraded from, each row a
    step of that output in its order -- but the last, a move the run stopped
    before evaluating (`model/parse.md` § 5a: row k is the geometry after
    move k+1).  A run that made no move -- relaxed at its first step --
    writes none, and its output is paired with nothing."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import align_to_reference
    assert not (out.parent / f"{label}.MD.nc").exists(), (
        f"SIESTA's own name for the history is still in {out.parent}")
    own = out.with_name(out.name[:-len(".out")] + ".MD.nc")
    got = parse(str(out))
    if not own.is_file():
        assert "mdnc_source" not in got.runtime_info, got.runtime_info
        return
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
    """`jobset init` of H2 or the case's own structure, then the case's
    template values, description blocks and this machine's
    `molbuilder.json`."""
    import dataclasses
    import json
    engine = case.get("engine", "siesta")
    calculation = case.get("calculation", "optimization")
    bundle = describe_calculation(tmp_path, monkeypatch, engine=engine,
                                  calculation=calculation,
                                  shape=case.get("shape", "hierarchical"),
                                  structure=case.get("structure"),
                                  stage_strategy=case.get("stage_strategy"))
    values = (dict(table.get("siesta_template", {}))
              if engine == "siesta" else {})
    values.update(case.get("template", {}))
    if values:
        from molbuilder.config.pyscf import PySCFConfig
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.template import (config_from_template, template_path,
                                         template_with_values)
        cls = SiestaConfig if engine == "siesta" else PySCFConfig
        path = template_path(bundle, "H2")
        cfg = dataclasses.replace(
            config_from_template(path.read_text(), cls), **values)
        path.write_text(template_with_values(cfg, engine=engine,
                                             calculation=calculation))
    task = json.loads((bundle / "task.json").read_text())
    # THE GROUPS THE ROW STATES, one number a stage, in order -- and each
    # stage's own run card, by name.
    for st, g in zip(task["stages"], case.get("groups", ())):
        st["group"] = g
    for st in task["stages"]:
        if st["name"] in case.get("stage_run", {}):
            st["execution"] = {**st.get("execution", {}),
                               **case["stage_run"][st["name"]]}
    scheduled = case.get("machine", table.get("machine", "this")) \
        != "workstation"
    blocks = {
        "execution": (_over_base(table.get("run_base", {}).get(engine),
                                 case.get("run"), case.get("run_unset"))
                      if "bench" not in case else case.get("run")),
        "allocation": _over_base(
            table.get("allocation_base") if scheduled else None,
            case.get("allocation"), case.get("allocation_unset")),
        "bench": case.get("bench"),
        "notify": case.get("notify"),
    }
    for block, value in blocks.items():
        if value:
            task[block] = value
        else:
            task.pop(block, None)      # the row states none of it
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
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


def _stage_names(bundle: Path, stage: str):
    """The names of ``stage``'s files -- the one composer's
    (`runfiles.RunNames`): on the calculation's label, its token, in its
    shape."""
    from molbuilder.jobset.materialize import stage_home
    from molbuilder.runfiles import RunNames
    from molbuilder.task import read_task
    task = read_task(bundle / "task.json")
    return RunNames.of(task.label, stage_home(bundle, task, stage).token,
                       task.shape)


def _the_runs(bundle: Path, pattern: str) -> Path:
    """The file of this name the run's prep wrote -- in the stage's
    folder and its attempt, one text in both."""
    found = _road_runs(bundle, pattern)
    assert found and len({p.read_text() for p in found}) == 1, found
    return found[-1]


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
    from molbuilder.jobset.materialize import ladder_homes
    from molbuilder.task import read_task
    task = read_task(Path(bundle) / "task.json")
    refs = [StageRef(h.seq, h.name) for h in ladder_homes(Path(bundle), task)]
    groups = []
    for s in stages:
        try:
            name = resolve_stage_ref(refs, s).name
        except ValueError:
            return tuple(x for s in stages for x in ("--stage", s))
        g = next(st.group for st in task.stages if st.name == name)
        if g not in groups:
            groups.append(g)
    return tuple(x for g in groups for x in ("--group", str(g)))


def group_words(bundle, *stages) -> tuple:
    """``--group N`` for the groups ``stages`` are in, as the calculation's
    own description numbers them -- how a person names them to `prep task`
    (`job-system.md` § 5.3)."""
    return _named("task", list(stages), bundle)


def run_road_case(table, case, tmp_path, monkeypatch) -> None:
    """ONE ROW of a contract's case table, down the road, every layer it
    names checked."""
    import json
    target = _road_target(table, case, tmp_path, monkeypatch)
    if case.get("ends_at") == "probe":
        return
    bundle = _road_describe(table, case, tmp_path, monkeypatch)
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
    if "removed" in case or "added" in case:
        _road_ladder_edited(case, bundle)
    if "source_edited" in case:
        _road_source_edited(case["source_edited"], bundle)
    if "held_edited" in case:
        _road_held_edited(case["held_edited"], bundle)
    if "regrouped" in case:
        _road_regrouped(case["regrouped"], bundle)
    kind = "bench" if "bench" in case else "task"
    kept = {where: _as_written(bundle / where)
            for where in case.get("kept", [])}
    named = _named(kind, case.get("stage", "coarse"), bundle)
    r = jobset("prep", kind, *named, "--bundle", bundle,
               "--target", target, *case.get("prep", []),
               input=case.get("answers"))
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
        _road_after_prep(case, bundle)
        _road_saved(case, bundle)
        if "reprobed" in case:
            # WHAT THE REFUSAL SAYS TO DO, DONE -- the machine probed again,
            # holding what it lacked -- and the same prep is taken.
            _reprobed(case["reprobed"])
            r = jobset("prep", kind, *named,
                       "--bundle", bundle,
                       "--target", target, *case.get("prep", []))
            assert r.exit_code == 0, _one_line(r)
        return
    assert r.exit_code == 0, _one_line(r)
    for words in case.get("said", []):
        assert words in r.output, _one_line(r)
    for words in case.get("said_lacks", []):
        assert words not in r.output, _one_line(r)
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
            names = _stage_names(bundle, case.get("stage", "coarse"))
            _road_lines(case, key,
                        _the_runs(bundle, names.name(role)).read_text())
    # ...and the deck as its readers read it: through the one fdf reader,
    # field by field, never by its lines
    if "deck_reads" in case:
        from molbuilder.parse.fdf import parse_fdf_params
        names = _stage_names(bundle, case.get("stage", "coarse"))
        got = parse_fdf_params(
            _the_runs(bundle, names.name(".fdf")).read_text())
        for field, want in case["deck_reads"].items():
            have = getattr(got, field)
            assert (have == want if isinstance(want, (bool, str))
                    else have == pytest.approx(want)), (field, have, want)
    # ...a benchmark's trials, when the row names them
    if kind == "bench" and ("bench_gres" in case
                            or "bench_header_lacks" in case):
        plan = next(bundle.rglob("bench/job-set.json"))
        asked = sorted({j["resources"]["gres"]
                        for j in json.loads(plan.read_text())["jobs"]})
        assert asked == sorted(case.get("bench_gres", asked)), asked
        headers = sorted(plan.parent.rglob("*.sbatch"))
        assert headers or "bench_header_lacks" not in case, \
            "bench_header_lacks: the benchmark wrote no .sbatch"
        for header in headers:
            _road_lines({"h_lacks": case.get("bench_header_lacks", [])}, "h",
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
            sent = [ln for ln in r.output.splitlines() if "sbatch" in ln.split()]
            assert sent or mode == "direct", _one_line(r)
            for ln in sent:
                _road_lines(case, "line", " ".join(sbatch_line(ln)))
        for words in case.get("listing", []):
            assert words in r.output, _one_line(r)
        if case.get("launch_writes_nothing"):
            now = _all_written(bundle)
            assert now == was, sorted(set(now.items()) ^ set(was.items()))
        if "after_launch" in case:
            _road_after_prep(case["after_launch"], bundle)
