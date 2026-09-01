"""`prep`, entire — the five steps, on the machine that will run it.

Contract: ``docs/execution/project-layout.md`` § 2.3.1 (the five steps, and why
their order is forced) · ``docs/execution/generator.md`` § 5 (the element, and
why the deck writer reads the allocation) · ``docs/execution/architecture.md``
§ 4.1.

**This file is the migration's proof.** Steps 2 and 3 did not exist inside
`prep` until 2026-08-11: the decks were finished at `molbuilder fdf` time, on a
machine that could not know the rank count, and `prep` refused unless they were
already in the bundle. Everything here would have been impossible to write.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.jobset.model import Resources
from molbuilder.jobset.prep import PrepError, prep_calculation
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """The rest of the sandbox (B-9, 2026-08-13): the calc fixture's
    dotted .molbuilder.json fixed the bundle scope, but the file still
    ran with cwd = repo root, so the repo's SERVER-scope molbuilder.json
    (preamble concatenation, scheduler) kept folding into every wrapper
    under test."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture
def calc(tmp_path):
    """A described calculation, exactly as `jobset init` leaves it.

    The activation config is the **dotted, bundle-scoped**
    ``.molbuilder.json`` — the project scope the wrapper writer resolves
    from the script's own directory.  An undotted ``molbuilder.json`` here
    is INERT (that name is the cwd-first server scope), and until 2026-08-12
    this fixture wrote exactly that: the tests then silently resolved
    ``script_generation.activation`` from the developer's repo-root config —
    14 of 24 failed under an isolated cwd+HOME while green in-repo.
    """
    struct = Structure(elements=["S", "C", "C", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                           [1.21, 0.0, 2.48], [2.15, 0.0, 1.94]]),
                       vacuum=(10.0, 10.0, 10.0))
    src = tmp_path / "bdt.xyz"
    src.write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    desc = D.build_description(
        struct, SiestaConfig(system_label="calc", mesh_cutoff=300.0),
        default_siesta_stages("publishable"),
        engine="siesta", shape="hierarchical", name="calc", source=str(src))
    D.write_description(desc, dest)
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "source /opt/conda/etc/profile.d/conda.sh"}}))
    _pseudos_for(dest, ["S", "C", "H"])
    return dest


from conftest import write_pseudos as _pseudos_for



# --------------------------------------------------------------------- #
#  The migration — prep RENDERS the deck                                 #
# --------------------------------------------------------------------- #

def test_prep_renders_the_deck_it_used_to_demand(calc):
    """**§ 9.3's one real migration.**  `prep` refused unless the deck was
    already in the bundle root — *"render the inputs before prep"* — because
    the producer ran at *produce*.  A described calculation carries no deck at
    all, so this passing is the migration."""
    assert not list(calc.glob("*.fdf"))
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=32))
    assert (calc / "01_coarse" / "calc_01_coarse.fdf").is_file()


def test_the_five_steps_all_leave_their_mark(calc):
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    assert (calc / "environment.json").is_file()        # 1 resolve the machine
    assert (calc / "01_coarse" / "calc_01_coarse.fdf").is_file()      # 3 render the deck
    assert (calc / "01_coarse" / "calc_01_coarse.run.sh").is_file()   # 4 render the wrapper
    assert (calc / "01_coarse").is_dir()                # 5 build the directory


def test_floor_three_is_written_by_prep_not_read_by_it(calc):
    """`describe` writes floor 2 only.  Until `prep` derived floor 3 from it,
    there was nothing to derive it from and nothing that did."""
    assert not (calc / "job-set.json").exists()
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    js = json.loads((calc / "job-set.json").read_text())
    assert [j["name"] for j in js["jobs"]] == ["coarse"]
    assert js["jobs"][0]["script"] == "calc_01_coarse.fdf"


# --------------------------------------------------------------------- #
#  The deck and the launch agree BY CONSTRUCTION                         #
# --------------------------------------------------------------------- #

def test_the_deck_records_the_rank_count_it_was_rendered_for(calc):
    """**The `-np 14` class of failure, made unconstructible.**

    A deck derives values from the rank count — ``BlockSize`` above all — so a
    deck rendered without one says ``mpi_np auto`` and then gets launched at 32.
    Both now come from one resolved element, so they cannot disagree.
    """
    from molbuilder.parse.scripts.bench_marks import _extract_bench_marks_dict
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=32))
    marks = _extract_bench_marks_dict(
        (calc / "01_coarse" / "calc_01_coarse.fdf").read_text())
    assert marks.get("mpi_np") == 32


def test_the_launch_agreement_holds_for_a_deck_prep_just_made(calc):
    """`check_launch_matches_deck` refuses a deck rendered for another launch.
    A deck `prep` itself rendered must never trip it — and it did, on the first
    run of this path, because the deck was rendered from the values alone."""
    from molbuilder.jobset.agreement import launch_agreement
    from molbuilder.jobset._cli import _load
    from molbuilder.jobset.materialize import job_dir_names, shape_of
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=32))
    js, _ = _load(str(calc))
    # the deck lives in the JOB's directory (L1, roadmap 7.10) -- ask the
    # naming authority, exactly as the launch door does.
    d = calc / job_dir_names(js, shape_of(js, calc))[js.jobs[0].name]
    assert launch_agreement(d, js.jobs[0]).verdict == "agrees"


def test_the_allocation_reaches_the_jobs_resources(calc):
    prep_calculation(calc, "coarse",
                     allocation=Resources(mpi_np=16, cpus_per_task=4,
                                          max_memory_mb=4096))
    js = json.loads((calc / "job-set.json").read_text())
    r = js["jobs"][0]["resources"]
    assert (r["mpi_np"], r["cpus_per_task"], r["max_memory_mb"]) == (16, 4, 4096)


# --------------------------------------------------------------------- #
#  Step 2 really resolved — the stage's own science is in the deck       #
# --------------------------------------------------------------------- #

def test_the_named_stages_overrides_are_what_got_rendered(calc):
    """`coarse` is CG at a loose force tolerance and `medium` is Broyden at a
    tighter one.  Rendering the wrong rung would be silent, so this names the
    values rather than asserting a file exists."""
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    coarse = (calc / "01_coarse" / "calc_01_coarse.fdf").read_text()
    assert "MD.TypeOfRun CG" in coarse
    assert "MD.MaxForceTol 0.05" in coarse

    prep_calculation(calc, "medium", allocation=Resources(mpi_np=8))
    medium = (calc / "02_medium" / "calc_02_medium.fdf").read_text()
    assert "MD.TypeOfRun Broyden" in medium
    assert "MD.MaxForceTol 0.04" in medium


def test_the_deck_carries_its_stages_token_and_header(calc):
    """The stage's artifact token reaches the rendered deck: every filename
    molbuilder chooses picks it up, and the ``# Stage <token> --`` line names
    the science of the config being rendered, so the comment cannot drift from
    the keywords below it (decision 27).  Gated here because `prep` is the
    token's producer now -- ``molbuilder fdf --stage N`` was, until it went."""
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    coarse = (calc / "01_coarse" / "calc_01_coarse.fdf").read_text()
    assert "calc_01_coarse.out" in coarse
    assert "# Stage 01_coarse --" in coarse


def test_the_template_supplies_what_no_stage_varies(calc):
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    assert "MeshCutoff 300.0" in (calc / "01_coarse" / "calc_01_coarse.fdf").read_text()


# --------------------------------------------------------------------- #
#  Refusals                                                              #
# --------------------------------------------------------------------- #

def test_a_folder_with_no_description_is_refused_by_name(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PrepError, match=r"jobset init"):
        prep_calculation(tmp_path / "empty", "coarse",
                         allocation=Resources(mpi_np=8))


def test_a_structure_that_changed_since_describing_is_refused(calc, tmp_path):
    """§ 6.3's witness earning its place: the description records a formula and
    an atom count, so building a *different* calculation under the same id is
    caught rather than discovered in the results.

    The mutated file is the CALCULATION'S OWN copy — `describe` copies the
    structure in since 2026-08-12 (M9's walk found nothing made "beside the
    calculation first" true), and that copy is what `prep` reads."""
    (calc / "bdt.source.xyz").write_text(
        Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                  vacuum=(10.0, 10.0, 10.0)).to_xyz())
    with pytest.raises(PrepError, match=r"structure has changed"):
        prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))


def test_a_stage_that_is_not_in_the_ladder_is_refused(calc):
    with pytest.raises(PrepError, match=r"coarse"):
        prep_calculation(calc, "nonesuch", allocation=Resources(mpi_np=8))


# --------------------------------------------------------------------- #
#  The warm-retry budget — § 6.2's one no-flag row, on THIS route        #
# --------------------------------------------------------------------- #

def test_the_warm_retry_budget_travels_the_described_route(calc):
    """A-5 (final review, 2026-08-13): `job-contracts.md § 6.2`'s last row
    says ``continue_retries`` is *"translated at resolve.py — rides the
    element's Resources; prep bakes it into the wrapper"*.  Until the fix
    nothing performed that translation on this route: the web route hands
    the value straight to the wrapper writer, while the described route
    resolved an allocation that never carried it — so the wrapper rendered
    NO retry loop and `job-system.md § 4.1`'s "travels the whole way" was
    true of one road out of two.

    Pinned on the EMITTED TEXT because the defect class is a value that
    travels correctly and is dropped at the last hop.  Both precedences:
    the template's answer rides by default; an explicit allocation wins."""
    tpl = calc / "calc.template.toml"
    head, sep, tail = tpl.read_text().partition("[item.continue_retries]")
    assert sep, "the template lost its continue_retries item"
    body, nxt, rest = tail.partition("\n[item.")
    assert "value = 1" in body, body
    tpl.write_text(head + sep + body.replace("value = 1", "value = 4", 1)
                   + nxt + rest)
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    text = (calc / "01_coarse" / "calc_01_coarse.run.sh").read_text()
    assert "_siesta_retry_max=4" in text, (
        "the template's warm-retry budget never reached the wrapper -- "
        "the § 6.2 translation at resolve.py is broken again (A-5)")
    # an explicitly stated allocation wins over the template's answer
    prep_calculation(calc, "coarse",
                     allocation=Resources(mpi_np=8, continue_retries=2))
    text = (calc / "01_coarse" / "calc_01_coarse.run.sh").read_text()
    assert "_siesta_retry_max=2" in text


# --------------------------------------------------------------------- #
#  The reporting policy travels from the description to the wrapper      #
# --------------------------------------------------------------------- #

def test_the_descriptions_notify_block_reaches_the_wrapper(calc):
    """`task.json` says WHEN this calculation should speak up, and the
    monitor is what speaks -- so the policy has to survive the whole trip:
    description -> `Resources` -> the emitted `mb_monitor.py` line.

    **This is the link a wrapper-level test cannot see.**  Tests that build
    a `Resources` by hand and render from it pass whether or not `prep`
    ever reads the description: deleting the one line that applies
    `task.notify` left all ten of them green (mutation-tested 2026-08-26).
    The seam is only observable from a real description.
    """
    import json as _json
    from molbuilder.task import FILENAME as TASK_FILENAME

    task_file = calc / TASK_FILENAME
    obj = _json.loads(task_file.read_text())
    obj["notify"] = {"on_scf_converged": True, "every_hours": 4}
    task_file.write_text(_json.dumps(obj))

    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))

    wrappers = list(calc.rglob("*.run.sh"))
    assert wrappers, "prep wrote no wrapper"
    text = "\n".join(w.read_text() for w in wrappers)
    assert "--notify-on-scf" in text
    assert "--notify-every-hours 4" in text


def test_the_decision_log_prints_channel_names_readably():
    """The ledger's whole value is that a person can read it, which is what
    `_flat_resources`' own docstring says.

    Every field it renders was a scalar until `notify_channels` (2026-08-31),
    so the default `f"{v}"` put a Python repr into the file -- and a bare
    `()` for the answer that matters most, *send this calculation nowhere*.
    """
    from molbuilder.jobset.prep import _flat_resources

    assert "notify_channels=slack,lab" in _flat_resources(
        Resources(mpi_np=4, notify_channels=("slack", "lab")))
    assert "notify_channels=(none)" in _flat_resources(
        Resources(mpi_np=4, notify_channels=()))
    # absent stays absent: `None` means "not asked for" and is left out
    # rather than printed as a null.
    assert "notify_channels" not in _flat_resources(Resources(mpi_np=4))


def test_the_descriptions_run_values_reach_the_wrapper(calc):
    """§ 6.8a extended: the machine-answered values a person chose are an
    ASK, and they travel the same seam the queue and the wall already do —
    description -> `Resources` -> the rendered wrapper.

    **This is the link a Resources-level test cannot see.** A test that
    builds a `Resources` by hand passes whether or not `prep` ever reads
    `task.json`'s block; deleting the one line that applies it left ten of
    them green when the same thing happened to `notify` (2026-08-26).
    """
    import json as _json
    from molbuilder.task import FILENAME as TASK_FILENAME

    task_file = calc / TASK_FILENAME
    obj = _json.loads(task_file.read_text())
    obj["allocation"] = {"domain": "htc", "time": "1-00:00:00",
                         "mpi_np": 8}
    task_file.write_text(_json.dumps(obj))

    prep_calculation(calc, "coarse")
    text = "\n".join(w.read_text() for w in calc.rglob("*.run.sh"))
    # THE ASSIGNED DEFAULT, not a substring of the whole script: the
    # wrapper's own `--np` help line contains "-np 8" for any value, so a
    # loose match here would pass whatever the description said.
    assert "_mpi_np_default=8" in text, text[:400]


def test_a_flag_still_wins_over_the_descriptions_run_value(calc):
    """The precedence § 6.8a spells out, at its first seam: a person typing
    a number now is answering about now."""
    import json as _json
    from molbuilder.task import FILENAME as TASK_FILENAME

    task_file = calc / TASK_FILENAME
    obj = _json.loads(task_file.read_text())
    obj["allocation"] = {"mpi_np": 8}
    task_file.write_text(_json.dumps(obj))

    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=4))
    text = "\n".join(w.read_text() for w in calc.rglob("*.run.sh"))
    assert "_mpi_np_default=4" in text
    assert "_mpi_np_default=8" not in text


def test_a_rung_that_differs_gets_its_own_allocation(calc):
    """§ 6.8b, at the seam: the flat block is the default and the rung
    overrides only what it names."""
    import json as _json
    from molbuilder.task import FILENAME as TASK_FILENAME

    task_file = calc / TASK_FILENAME
    obj = _json.loads(task_file.read_text())
    obj["allocation"] = {"domain": "htc", "time": "0-04:00:00"}
    obj["stage_allocation"] = {"coarse": {"time": "2-00:00:00"}}
    task_file.write_text(_json.dumps(obj))

    from molbuilder.jobset.prep import _allocation_for
    from molbuilder.task import read_task
    task = read_task(task_file)
    merged = _allocation_for(task, "coarse")
    assert merged.domain == "htc" and merged.time == "2-00:00:00"
    # and a rung that says nothing gets the flat block untouched
    assert _allocation_for(task, "tight").time == "0-04:00:00"


def test_measuring_takes_the_benchs_own_wall(calc):
    """§ 6.8c: `prep bench` is `sweep`-bearing, and that is what selects the
    bench block.  Absent, it is the run's — unchanged."""
    import json as _json
    from molbuilder.jobset.prep import _allocation_for
    from molbuilder.task import FILENAME as TASK_FILENAME, read_task

    task_file = calc / TASK_FILENAME
    obj = _json.loads(task_file.read_text())
    obj["allocation"] = {"domain": "htc", "time": "2-00:00:00"}
    obj["bench_allocation"] = {"domain": "general", "time": "0-00:30:00"}
    task_file.write_text(_json.dumps(obj))
    task = read_task(task_file)

    measuring = _allocation_for(task, "coarse", measuring=True)
    assert measuring.time == "0-00:30:00" and measuring.domain == "general"
    running = _allocation_for(task, "coarse")
    assert running.time == "2-00:00:00" and running.domain == "htc"


def test_the_descriptions_channel_names_reach_the_wrapper(calc):
    """The same seam, for the field that says WHERE by name.  A test that
    builds a `Resources` by hand cannot see whether `prep` ever reads
    `task.notify.channels` at all."""
    import json as _json
    from molbuilder.task import FILENAME as TASK_FILENAME

    task_file = calc / TASK_FILENAME
    obj = _json.loads(task_file.read_text())
    obj["notify"] = {"on_scf_converged": True, "channels": ["slack", "lab"]}
    task_file.write_text(_json.dumps(obj))

    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    text = "\n".join(w.read_text() for w in calc.rglob("*.run.sh"))
    assert '--notify-channels "slack,lab"' in text


def test_an_EMPTY_channel_list_survives_prep(calc):
    """**A truthiness guard would drop it here**, and the job would then
    report to every channel on the machine -- from a description that says
    in as many words to report to none (`run-reports.md` § 3.0).  Every
    other rider IS off when falsy, which is exactly why this one is easy to
    get wrong on the way past.
    """
    import json as _json
    from molbuilder.task import FILENAME as TASK_FILENAME

    task_file = calc / TASK_FILENAME
    obj = _json.loads(task_file.read_text())
    obj["notify"] = {"on_scf_converged": True, "channels": []}
    task_file.write_text(_json.dumps(obj))

    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    text = "\n".join(w.read_text() for w in calc.rglob("*.run.sh"))
    assert '--notify-channels ""' in text


def test_a_description_without_notify_leaves_the_wrapper_alone(calc):
    """The other half: absent must stay absent all the way down, or every
    prepped bundle changes for people who never asked for this."""
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    text = "\n".join(w.read_text() for w in calc.rglob("*.run.sh"))
    assert "--notify-" not in text
