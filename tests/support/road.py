"""The road a test drives -- `jobset init`, `prep`, `launch` -- and the one
thing a test may put beside it: a machine whose record names queues, with a
scheduler that queues nothing.  A run is made on the road or not at all: an
output copied into a stage's folder, or a conclusion written by hand, is a
run that never happened (user, 2026-10-03; `support.road.a_finished_run` and
every test built on it were retired that day).

WHY THIS FILE EXISTS.  A test drives the designed workflow (user, 2026-09-23:
*"tests should be using our established jobset workflow, unless you have a
strong reason to focus on api"*).  Several files grew their own copy of the
same steps -- an `init` of H2, a stub `sbatch` -- and a copy is free to drift
from the road it imitates.  The steps live here once.

* :func:`jobset` -- the verbs, as a person types them;
* :func:`each_is_taken` -- every command an output prints, typed back as
  printed (`job-system.md` § 5.3: what molbuilder prints, you can type);
* :func:`describe_h2` -- `jobset init` of a held H2 in a box, SIESTA, the
  shipped `publishable` ladder (coarse, medium; tight disabled) -- or
  PySCF's own;
* :func:`a_queue_that_answers` -- a machine record naming queues, and an
  `sbatch` on PATH that queues nothing: it writes down every call -- where
  it was made and what it said -- answers ``--test-only`` the way Sol's
  did, and otherwise gives a job id;
* :func:`sbatch_line` -- the `sbatch` line a launch showed, as its words;
* :func:`gpus_given` -- the GPUs a machine hands a job, with an
  `nvidia-smi` that knows them;
* :func:`strip_preamble_activation` -- a generated run script, runnable in
  a bare shell: its preamble and environment activation cut out.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

import numpy as np


#: Sol's own answer to `sbatch --test-only`, verbatim (2026-08-27).
SOL_PREDICTION = ("sbatch: Job 62266174 to start at 2026-08-27T11:22:03 a "
                  "using 4 processors on nodes sc078 in partition htc")


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


def describe_h2(tmp_path, monkeypatch, *, shape: str = "hierarchical",
                name: str = "H2", calculation: str = "optimization",
                engine: str = "siesta") -> Path:
    """`jobset init` on a held H2 in a box -- the bundle, at
    ``<projects>/P/<calculation>/<name>``: a SIESTA optimization's shipped
    `publishable` ladder, a vibration's own (`relax`, `freq`), or PySCF's
    own ladder.

    ITS RUN CARD STATES THE LAUNCH SHAPE -- two ranks and one thread for
    SIESTA, one thread for PySCF -- as a described calculation does: a run
    whose shape is stated nowhere is refused at prep (`architecture.md`
    § 5.2).  A test about that refusal takes the card away."""
    from conftest import write_pseudos
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    write_pseudos(tree / "pseudopotential", ["H"])
    StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]]),
                  regions={"frozen_atoms": [0]},
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3),
        tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    siesta = engine == "siesta"
    r = jobset("init", "--structure", "P/structure/h2.xyz",
               "--bundle", f"P/{calculation}/{name}", "--engine", engine,
               "--shape", shape, "--name", name,
               "--calculation", calculation,
               *(("--stage-strategy", "publishable")
                 if calculation == "optimization" and siesta else ()),
               *(("--psml-lib", "pseudopotential") if siesta else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / calculation / name
    import json
    task = json.loads((bundle / "task.json").read_text())
    task.setdefault("execution", {"mpi_np": 2, "omp_threads": 1} if siesta
                    else {"threads": 1})
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    return bundle


def a_queue_that_answers(tmp_path, monkeypatch, domains, **record) -> Path:
    """This machine's record names ``domains`` (`scheduler.Domain` rows) --
    and ``record``'s other fields, when given -- and the `sbatch` first on
    PATH queues nothing.  Returns the file each call is written to, one line
    each: ``<where it was run> | <its arguments>``."""
    from conftest import write_machine_record
    write_machine_record(scheduler="slurm", domains=list(domains), **record)
    bin_dir = tmp_path / "scheduler-bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch-calls.log"
    f = bin_dir / "sbatch"
    f.write_text(
        "#!/bin/sh\n"
        f'echo "$(pwd) | $*" >> "{calls}"\n'
        'case " $* " in\n'
        f'  *" --test-only "*) echo "{SOL_PREDICTION}" >&2; exit 0 ;;\n'
        "esac\n"
        'echo "Submitted batch job 4242"\n')
    f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return calls


def gpus_given(tmp_path, monkeypatch, visible: str) -> None:
    """The GPUs a machine hands a job: ``CUDA_VISIBLE_DEVICES`` as a
    scheduler sets it, and an `nvidia-smi` first on PATH that knows each of
    them -- listed by ``-L``, and a PCI address for ``--id`` that no real
    device has, so the NUMA lookup reads *unknown* on any box.  The run
    script then asks about its GPUs as it would on the node, wherever the
    test runs."""
    n = len([g for g in visible.split(",") if g])
    bin_dir = tmp_path / "gpu-bin"
    bin_dir.mkdir(exist_ok=True)
    f = bin_dir / "nvidia-smi"
    listing = "".join(f"GPU {i}: Stand-in GPU (UUID: GPU-stand-in-{i})\\n"
                      for i in range(n))
    f.write_text(
        "#!/bin/sh\n"
        'case " $* " in\n'
        f'  *" -L "*) printf "{listing}" ;;\n'
        '  *) echo "00000000:FE:1F.7" ;;\n'
        "esac\n")
    f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)


def strip_preamble_activation(text: str) -> str:
    """Remove the baked preamble + conda-activation block (script-
    execution blocks 3-4) from a rendered wrapper so the behaviour
    tests can EXECUTE it in a bare CI shell.  ``module load mamba`` /
    ``source activate`` exit 127 without an HPC module system or conda;
    under ``set -e`` that aborts the wrapper before the cold block ever
    runs.  ``_log`` is defined earlier (block 2) so the cold block's
    logging survives the strip -- the wrapper is RUN, here, as the
    person's machine would run it, short of entering an environment."""
    pre = text.find("# --- Baked preamble")
    assert pre >= 0, "baked-preamble marker not found in wrapper"
    # Since U10 the bootstrap AND the post-activation state dump each sit
    # inside a help guard (if [ "$_mb_help" = "0" ]); the cut must span
    # from the FIRST guard's opener through the SECOND guard's close, or
    # the truncated wrapper keeps an unopened fi.
    start = text.rfind('if [ "$_mb_help" = "0" ]; then', 0, pre)
    assert start >= 0, "help-guard opener not found before the preamble"
    em = text.find("which python:", pre)
    assert em >= 0, "activation conda-dump end marker not found"
    close = text.find("\nfi\n", em)
    assert close >= 0, "post-activation guard close not found"
    # ``set -u`` is restored explicitly: the real wrapper disables
    # nounset around the activation (NVCC_PREPEND_FLAGS) and re-enables
    # it INSIDE the region cut here, so without this line the stripped
    # harness runs everything after the preamble with nounset off --
    # which is how the unbraced-$_warm_label death (redo NEW-1) stayed
    # invisible to every executed test in this file.
    return (
        text[:start]
        + "# preamble + activation stripped for CI (no conda here).\n"
        + "set -u\n"
        + text[close + 4:]
    )


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
# `launch --dry-run`, and on a machine with no queue the run script's own
# dry run -- checking the layers it names (`docs/process/testing.md` § 6):
#
#   1. ALLOWED OR REFUSED, in its words -- at prep (`refused`, a sentence or a
#      list of them; `said`) or at
#      launch (`launch_refused`, the same; `listing`);
#   2. WHAT IS PRODUCED -- the `.sbatch` header (`header`, or
#      `header_absent`), the deck (`deck`), the run script (`run_sh`), each
#      with a `_lacks` twin; the `sbatch` line launch shows (`line`,
#      `line_lacks`); a benchmark's trials (`bench_gres`,
#      `bench_header_lacks`);
#   3. WHAT THE RUN SCRIPT DOES HERE -- its dry run, given the GPUs the
#      machine hands it (`given_gpus`, `run_args`, `runs`).
#
# INPUTS: `engine` (siesta | pyscf); `run` -- the run card, task.json
# `execution`, over the table's `run_base` for the engine; `allocation` --
# task.json `allocation`, over the table's `allocation_base` where the
# machine has a scheduler; `run_unset` / `allocation_unset` -- base keys a row
# takes away; `template` -- values in the template (SIESTA's over the
# table's `siesta_template`); `bench` -- task.json `bench`, then `prep bench`;
# `machine_config` -- THIS machine's `molbuilder.json`; `prep` / `launch` --
# the flags typed (launch
# with `--mode submit`, unless `launch_mode` names another -- "" for none, so
# the config's `launch.mode` decides; of the stage `launch_stage` names,
# `coarse` unless it names another); `machine` -- "this" (this machine IS
# the target, its record
# listing `queues`), "named" (this machine is a workstation; the target is a
# record named `sol` listing `queues`) or "workstation" (no queues at all),
# the table's own `machine` unless the row names one;
# `record` -- more fields of THIS machine's record; `named_record` -- more
# fields of the named target's; `queues` -- replaces the table's menu;
# `probe` -- the record made as a person makes it instead, by `jobset probe
# --write --yes`, once per list of flags, in order, after `machine_config`
# and over `record` when the row gives them; a `--name` among the flags names
# the target.  A probe given as a table, `{flags = [...], answers = "n\n"}`,
# runs WITHOUT `--yes` and is typed `answers` at its questions ("" is EOF);
# `record_text` -- a file already at the path the first probe writes, as
# text (a record that does not read); `probe_refused` -- the last probe is
# refused, saying each of these, and the row ends there.  A probe row's
# machine holds the `molbuilder.json` `envs init-config` leaves -- its
# `env_init` -- unless `machine_config` gives the file;
# `calculation_record` -- the calculation's own copy of its
# machine's record, there before prep (a table, or text for a broken file);
# `saved_first` -- the folder's state saved before anything else, as a person
# saves it (`molbuilder checkpoint init`); `before` -- the verbs a person typed
# first, each a list of words (`["prep", "run", "coarse"]`), the calculation
# and the target named as the row's own prep names them; `disabled` -- the
# stages then disabled through Task setup's Save; `removed` / `added` -- the
# stages then removed, or added (`{name, at}`, `at` the place, the end when
# absent), through the same Save; `stand_in` -- what the suite's stand-in
# engine does on the row's launches (`_road_stand_in`: `rc`,
# `leaves_restart`); `own_warm_files` -- the calculation's own restart-file
# list, the engine's
# copied beside `task.json` with `withhold` / `add` (`_road_own_warm_files`),
# before anything is prepped; `stage` -- the
# stage the
# row's prep names, `coarse` unless given; `answers` -- what
# the person types at the row's prep's question ("" is EOF, no terminal);
# `calculation` -- the kind `jobset init` describes, an optimization unless
# given.
#
# AFTER PREP, whatever it answered: the folders it left (`made`, paths
# under the calculation that exist; `made_lacks`, ones that do not), how the
# server answers a viewer about a stage's newest run (`run_answer`: `stage`,
# `state`, `live` -- `_road_run_answer`), the
# folder's saved states, newest first
# (`saved_states`, their notes -- `{stamp}` standing for the time a note
# leads with, `2026-10-03 14:05:12`), what `status` says of the calculation
# (`status_says`, and what it does not, `status_lacks`), and the decisions
# its ledger does not hold
# (`ledger_lacks`); a prep that was not refused says nothing of `said_lacks`.
# THEN, refused or not: the description saved through Task setup's Save with
# `saved`'s fields changed (`{shape = "flat"}`) -- refused, with
# `save_refused`'s words, or taken.  A REFUSED prep, its remedy done: this
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
        if "record_text" in case:
            from molbuilder.config_dir import ensure_private_dir
            from molbuilder.scheduler import (machine_scope_path,
                                              named_environment_path)
            first = case["probe"][0]
            first = first["flags"] if isinstance(first, dict) else first
            there = (named_environment_path(first[first.index("--name") + 1])
                     if "--name" in first else machine_scope_path())
            ensure_private_dir(there.parent)
            there.write_text(case["record_text"])
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
    if where == "this":
        a_queue_that_answers(tmp_path, monkeypatch, queues, **record)
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


def _road_disabled(names, bundle) -> None:
    """These stages disabled, as a person disables one -- the stage table's
    switch, written through Task setup's Save."""
    import json
    from molbuilder.web.app import create_app
    task = json.loads((bundle / "task.json").read_text())
    for st in task["stages"]:
        if st["name"] in names:
            st["enabled"] = False
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    assert r.status_code == 200, (r.get_json() or {}).get("error")


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
        at = new.get("at", len(task["stages"]))
        task["stages"].insert(at, {"name": new["name"]})
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    assert r.status_code == 200, (r.get_json() or {}).get("error")


def _road_saved(case, bundle) -> None:
    """The description saved through Task setup's Save -- the page's door,
    which no `jobset` verb reaches -- with ``saved``'s fields changed:
    refused, saying each of ``save_refused``, or taken."""
    if "saved" not in case:
        return
    import json
    from molbuilder.web.app import create_app
    task = json.loads((bundle / "task.json").read_text())
    task.update(case["saved"])
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    said = (r.get_json() or {}).get("error") or ""
    if "save_refused" not in case:
        assert r.status_code == 200, said
        return
    assert r.status_code == 409, (r.status_code, said)
    for words in case["save_refused"]:
        assert words in said, said


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
    known and not carried for each."""
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
    (bundle / FILENAME).write_text(text)


def _road_stand_in(asked, monkeypatch) -> None:
    """What the suite's stand-in engine does on this row's launches
    (`conftest._STUB_BODIES`): end with exit code ``rc``, and -- with
    ``leaves_restart`` -- leave the restart file SIESTA leaves, empty, so a
    run can be continued.  The run's records are our wrapper's, written as
    it concludes; nothing is laid by hand."""
    if "rc" in asked:
        monkeypatch.setenv("MB_STAND_IN_RC", str(asked["rc"]))
    if asked.get("leaves_restart"):
        monkeypatch.setenv("MB_STAND_IN_LEAVES_XV", "1")


def _road_run_answer(want, bundle) -> None:
    """How the server answers a viewer about a stage's newest run
    (`parse.dirs.run_answer`, the one door the watch and spectra loads
    ask): ``want`` holds the stage and the ``state`` and ``live`` expected,
    asked about the run's own deck where its prep put it."""
    from molbuilder.jobset.materialize import run_dir, stage_home
    from molbuilder.parse.dirs import run_answer
    from molbuilder.runfiles import stem
    from molbuilder.task import read_task
    task = read_task(bundle / "task.json")
    home = stage_home(bundle, task, want["stage"])
    deck = run_dir(home.dir) / (stem(task.label, home.token) + ".fdf")
    got = run_answer(str(deck))
    for key in ("state", "live"):
        if key in want:
            assert got.get(key) == want[key], f"{key}: {got}"


def _road_after_prep(case, bundle) -> None:
    """What the row's prep left, refused or not: the folder's saved states,
    newest first (`saved_states`), what `status` says of the calculation
    (`status_says`, and what it does not, `status_lacks`), and the decisions
    its ledger does not hold
    (`ledger_lacks`)."""
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
    if "run_answer" in case:
        _road_run_answer(case["run_answer"], bundle)
    if "status_says" in case or "status_lacks" in case:
        st = jobset("status", "--bundle", bundle)
        assert st.exit_code == 0, _one_line(st)
        for words in case.get("status_says", []):
            assert words in st.output, _one_line(st)
        for words in case.get("status_lacks", []):
            assert words not in st.output, _one_line(st)
    if "ledger_lacks" in case:
        import json
        from molbuilder.jobset.ledger import LEDGER_FILE
        log = bundle / LEDGER_FILE
        decided = [json.loads(x)["decision"] for x in
                   (log.read_text().splitlines() if log.is_file() else [])
                   if x.strip()]
        for decision in case["ledger_lacks"]:
            assert decision not in decided, f"the ledger holds: {decided}"


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
    """`jobset init` of H2, then the case's template values, description
    blocks and this machine's `molbuilder.json`."""
    import dataclasses
    import json
    engine = case.get("engine", "siesta")
    calculation = case.get("calculation", "optimization")
    bundle = describe_h2(tmp_path, monkeypatch, engine=engine,
                         calculation=calculation)
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
    """``key``'s lines are in ``text``; ``key_lacks``'s are not -- read with
    runs of blanks as one, since a deck aligns its values in columns."""
    import re
    flat = re.sub(r"[ \t]+", " ", text)
    for line in case.get(key, []):
        assert line in flat, \
            f"{key}: {line!r} missing from: {_one_line(text)}"
    for line in case.get(f"{key}_lacks", []):
        assert line not in flat, \
            f"{key}: {line!r} present in: {_one_line(text)}"


def _road_runs(bundle: Path, pattern: str) -> "list[Path]":
    """The files of this pattern the run's prep wrote, outside a bench."""
    return [p for p in bundle.rglob(pattern)
            if "bench" not in p.relative_to(bundle).parts]


def _the_runs(bundle: Path, pattern: str) -> Path:
    """The file of this pattern the run's prep wrote -- in the stage's
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


def run_road_case(table, case, tmp_path, monkeypatch) -> None:
    """ONE ROW of a contract's case table, down the road, every layer it
    names checked."""
    import json
    import subprocess
    target = _road_target(table, case, tmp_path, monkeypatch)
    if case.get("ends_at") == "probe":
        return
    if "given_gpus" in case:
        gpus_given(tmp_path, monkeypatch, case["given_gpus"])
    bundle = _road_describe(table, case, tmp_path, monkeypatch)
    if "calculation_record" in case:
        from molbuilder.scheduler.record import calculation_record
        given = case["calculation_record"]
        calculation_record(bundle).write_text(
            given if isinstance(given, str) else json.dumps(given))
    if "own_warm_files" in case:
        _road_own_warm_files(case, bundle)
    if case.get("saved_first"):
        from click.testing import CliRunner
        from molbuilder.cli import cli
        got = CliRunner().invoke(cli, ["checkpoint", "init", "-p",
                                       str(bundle), "-m", "set up"])
        assert got.exit_code == 0, _one_line(got)
    if "stand_in" in case:
        _road_stand_in(case["stand_in"], monkeypatch)
    for words in case.get("before", []):
        # A machine is named at prep; the other verbs read the one prep set.
        got = jobset(*words, "--bundle", bundle,
                     *(("--target", target) if words[0] == "prep" else ()))
        assert got.exit_code == 0, f"{words}: {_one_line(got)}"
    if "disabled" in case:
        _road_disabled(case["disabled"], bundle)
    if "removed" in case or "added" in case:
        _road_ladder_edited(case, bundle)
    kind = "bench" if "bench" in case else "run"
    r = jobset("prep", kind, case.get("stage", "coarse"), "--bundle", bundle,
               "--target", target, *case.get("prep", []),
               input=case.get("answers"))

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
            r = jobset("prep", kind, case.get("stage", "coarse"),
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
    deck = "*.py" if case.get("engine") == "pyscf" else "*.fdf"
    for key, pattern in (("header", "*.sbatch"), ("deck", deck),
                         ("run_sh", "*.run.sh")):
        if key in case or f"{key}_lacks" in case:
            _road_lines(case, key, _the_runs(bundle, pattern).read_text())
    # ...a benchmark's trials, when the row names them
    if kind == "bench" and ("bench_gres" in case
                            or "bench_header_lacks" in case):
        plan = next(bundle.rglob("bench/job-set.json"))
        asked = sorted({j["resources"]["gres"]
                        for j in json.loads(plan.read_text())["jobs"]})
        assert asked == sorted(case.get("bench_gres", asked)), asked
        for header in plan.parent.rglob("*.sbatch"):
            _road_lines({"h_lacks": case.get("bench_header_lacks", [])}, "h",
                        header.read_text())
    # ...and the `sbatch` line(s) launch shows -- one per shelf of a
    # benchmark -- or its refusal
    if "launch" in case:
        mode = case.get("launch_mode", "submit")
        r = jobset("launch", kind, case.get("launch_stage", "coarse"),
                   "--bundle", bundle, *(("--mode", mode) if mode else ()),
                   "--dry-run", "--yes", *case["launch"])
        if "launch_refused" in case:
            said = case["launch_refused"]
            assert r.exit_code != 0, _one_line(r)
            for words in ([said] if isinstance(said, str) else said):
                assert words in r.output, _one_line(r)
        else:
            assert r.exit_code == 0, _one_line(r)
            sent = [ln for ln in r.output.splitlines() if "sbatch" in ln.split()]
            assert sent, _one_line(r)
            for ln in sent:
                _road_lines(case, "line", " ".join(sbatch_line(ln)))
        for words in case.get("listing", []):
            assert words in r.output, _one_line(r)

    # 3 · WHAT THE RUN SCRIPT DOES HERE -- its dry run, given the case's GPUs
    if "given_gpus" in case:
        script = _the_runs(bundle, "*.run.sh")
        script.write_text(strip_preamble_activation(script.read_text()))
        # The rank and thread counts this shell may carry are scrubbed, so
        # what resolves is the script's own chain.
        env = {k: v for k, v in os.environ.items()
               if k not in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "MB_NP",
                            "SLURM_NTASKS", "SLURM_JOB_ID", "PBS_NP",
                            "MOLBUILDER_USE_MPS")}
        env.update(MB_LAUNCHED_BY="manual")
        done = subprocess.run(["bash", str(script), "--dry-run",
                               *case.get("run_args", [])],
                              cwd=script.parent, capture_output=True,
                              text=True, timeout=60, env=env)
        said = done.stdout + done.stderr
        assert done.returncode == 0, _one_line(said[-3000:])
        for words in case["runs"]:
            assert words in said, _one_line(said[-3000:])
