"""What molbuilder prints, you can type -- through the road: `molbuilder describe`,
`prep`, `launch`, `status`, and every command they print, typed back as
printed (`support.road.each_is_taken`).

PINS: ``docs/execution/job-system.md`` § 5.3 (*what molbuilder prints, you
can type*; every printed command composed in one place -- the calculation
named every time, a launch's mode stated unless the
calculation is launched on this machine and its config names one, a
refusal that asks for a stage offering the stages and
the command for the first, one command a line, any prose after `#`; a
benchmark's sweep answered with its own verbs) and § 7 (a calculation with
no benchmark says so); plan W52.

PREVENTS, each read in the code before 2026-10-01 (the W52 review): a next
step printing `--mode submit|direct`, which bash runs as a pipe; remedies
naming no calculation, so a line pasted from anywhere but inside it acted on
another or on none; a `<stage>` nobody can type; a stage-less launch offered
with no mode; a note on the command's own line; a bench verb offering a stage
with no benchmark, or a calculation that has none; a sweep's status worded as
a ladder's, its trials read from the wrong folder.

Each printed `molbuilder jobset ...` line is split as a shell would and run
from where the person stands -- the projects tree's parent, not inside the
calculation -- with `--dry-run` added to a launch, which plans and sends
nothing.  Nothing here launches an engine.
"""
from __future__ import annotations

from pathlib import Path

from support.road import (TYPED, described, each_is_taken, jobset,
                          printed_commands, road_tree, set_by_the_person)


def _described(tmp_path, monkeypatch, **blocks):
    """`support.road.described`'s calculation, its job request stating the wall
    and the memory too -- the two values a job handed to a scheduler states
    (`architecture.md` § 5.2) -- so the `--mode submit` line a next step
    prints is one that is taken, as it is for a person who stated them; and
    ``blocks``, the request's other blocks, as `described` takes them."""
    return described(road_tree(tmp_path, monkeypatch),
                     allocation={"time": "0-01:00:00", "mem": "8G"}, **blocks)


def test_the_lines_task_setup_shows_are_the_terminals_and_are_taken(
        tmp_path, monkeypatch):
    """W55 B4 / D11 / D12: Task setup's command lines are the terminal's own
    (`/api/task-setup/commands`, `jobset/commands.stage_lines`) -- the
    calculation named, a launch line per mode where this machine's config
    sets none -- and each, typed back from outside the calculation, is
    taken.  Prepared, the folder answer names the machine the calculation is
    set to; the group's lines are its launch alone (the ready door's
    *needs no prep*) -- until a value of it changes, never launched,
    when its prep leads them again (`job-system.md`, *Changed since
    prepared*); and the next group's prep line needs
    no `--target` -- the calculation's copy answers -- and carries no
    `--cold`, which is its launch's (`job-system.md` § 5.4).

    API-LEVEL for the page's two doors, which no `jobset` verb reaches; the
    lines themselves are typed down the road.

    MUTATIONS THIS MUST FAIL AGAINST: a launch line with no mode; a folder
    answer without the set machine; a
    prepared stage still offered its prep; a stage changed since its prep
    offered its launch alone, which sends what was prepared."""
    from molbuilder.web.app import create_app
    bundle = _described(tmp_path, monkeypatch)
    client = create_app(config={}).test_client()

    def lines(group, **choice):
        r = client.post("/api/task-setup/commands", json=dict(
            dest=str(bundle), kind="task", groups=[group], target="this",
            **choice))
        assert r.status_code == 200, r.get_json()
        return r.get_json()["lines"]

    coarse = lines(1)
    assert [l for l in coarse if " launch " in l] and all(
        "--mode " in l for l in coarse if " launch " in l), coarse
    assert each_is_taken("\n".join(coarse)) == len(coarse), coarse

    folder = client.get("/api/task-setup/folder?dir=" + str(bundle)).get_json()
    assert folder["set_to"] == "(this machine)", folder
    again = lines(1)
    assert again and all(" launch " in l for l in again), again
    # ...A VALUE OF IT CHANGED BY THE PERSON, never launched: its prep again.
    from molbuilder.calculation import open as open_calculation
    from molbuilder.template import one, read_template
    was = one(read_template(open_calculation(bundle).template_text),
              "mesh_cutoff").value
    set_by_the_person(bundle, {"mesh_cutoff": was + 50})
    changed = lines(1)
    assert changed[0].split(" --bundle ")[0] == \
        f"{TYPED} jobset prep task --group 1", changed
    assert each_is_taken("\n".join(changed)) == len(changed), changed
    medium = lines(2, cold=True)
    assert medium[0].split(" --bundle ")[0] == \
        f"{TYPED} jobset prep task --group 2", medium
    # The prep line needs no `--target` and carries no `--cold`, which is
    # its launch's -- and there IS a launch line to carry it.
    assert "--target" not in medium[0] and "--cold" not in medium[0], medium
    launches = [l for l in medium if " launch " in l]
    assert launches and all("--cold" in l for l in launches), medium


def test_every_command_a_benchmark_prints_is_taken(tmp_path, monkeypatch):
    """Medium's benchmark prepared, nothing else: its next steps; `launch
    bench` naming no stage, which offers medium -- the stage with a
    benchmark -- and its launch with the mode; `launch task --stage medium`, refused
    with the prep of the first stage and the benchmark's note apart from
    it; `status` in the bench folder, which reads its trials where they are
    and names the sweep's own verbs for the calculation.  Each line, typed
    back from outside the calculation, is taken.

    MUTATIONS THIS MUST FAIL AGAINST: the stage-less bench refusal offering
    every stage (coarse, never benched, first); the note on the command's
    line; a sweep's status read from its bench folder, or worded as a
    ladder's."""
    bundle = _described(tmp_path, monkeypatch,
                        bench={"mpi_np": [1, 2], "omp_threads": [1]})
    r = jobset("prep", "bench", "medium", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    assert each_is_taken(r.output), r.output

    for verb in (("launch", "bench", "--mode", "direct"),
                 ("launch", "task", "--stage", "medium", "--mode", "direct")):
        r = jobset(*verb, "--bundle", bundle)
        assert r.exit_code != 0, (verb, r.output)
        assert each_is_taken(r.output), r.output

    r = jobset("status", "--bundle", bundle / "02_medium" / "bench")
    assert r.exit_code == 0, r.output
    printed = list(printed_commands(r.output))
    # its trials read where they are -- prepared, not launched -- so its
    # launch comes first, then the read-back
    assert printed and printed[0][:3] == ["launch", "bench", "medium"], (
        r.output)
    assert all(w[1] == "bench" for w in printed), r.output
    assert each_is_taken(r.output), r.output


def test_a_calculation_with_no_benchmark_says_so_first(tmp_path,
                                                        monkeypatch):
    """PySCF has no benchmark lane: `prep bench` and `launch bench`, naming
    no stage, say so -- the entry's own words (`bench_refusal`) -- rather
    than offer a stage whose `prep bench` is refused in turn.

    MUTATION THIS MUST FAIL AGAINST: the stage-less refusal asked before
    the calculation's own (it offered `prep bench coarse`)."""
    bundle = described(road_tree(tmp_path, monkeypatch, engine="pyscf"),
                       engine="pyscf")
    for verb in (("prep", "bench", "--target", "this"),
                 ("launch", "bench", "--mode", "direct")):
        r = jobset(*verb, "--bundle", bundle)
        assert r.exit_code != 0, (verb, r.output)
        assert "Benchmarks measure SIESTA runs only" in r.output, (verb, r.output)
        each_is_taken(r.output)


def test_a_printed_line_runs_in_a_shell_with_the_host_env_active(
        tmp_path, monkeypatch):
    """§ 5.3, *the program is named as you type it* (the user, 2026-11-21:
    "Spell in full, one printer"): a command line molbuilder prints, typed
    into a shell where the host env is active -- its folder in
    ``CONDA_PREFIX`` -- in the calculation's own folder, runs molbuilder on
    that env's own Python, which finds this checkout by the path file its
    recipe writes (`ops/deployment.md` § 0; the user, 2026-11-22: "Host env
    points at the repo").  Asked for its ``--help``, so it answers and
    changes nothing.  THE SILENT FAILURES: a printed ``molbuilder jobset
    ...`` that a shell answered with *command not found* -- no install puts
    a ``molbuilder`` on PATH -- and a full line answered with *No module
    named molbuilder* outside the checkout.

    The host env is a scratch one -- a named need: the real host env is off
    limits to a test -- a venv whose own folder names the suite's packages
    (so no path file of the real env's is read), the host recipe's default
    steps after its packages run in it as its manager runs them
    (``CONDA_PREFIX`` its folder)."""
    import os
    import subprocess
    import sys
    import sysconfig
    from molbuilder.envs.recipes import recipe_by_name
    bundle = _described(tmp_path, monkeypatch)
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    line = next((ln.strip().split("   #")[0] for ln in r.output.splitlines()
                 if ln.strip().startswith(TYPED + " jobset ")), None)
    assert line, r.output
    host_env = tmp_path / "host-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                    str(host_env)], check=True, timeout=120)
    own = Path(subprocess.run(
        [str(host_env / "bin" / "python"), "-I", "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True).stdout.strip())
    (own / "the-suites-packages.pth").write_text(
        sysconfig.get_paths()["purelib"] + "\n")
    active = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    active["CONDA_PREFIX"] = str(host_env)
    for step in recipe_by_name("molbuilder").extra_set():
        subprocess.run(step.argv, env=active, cwd=tmp_path, check=True,
                       capture_output=True, timeout=60)
    done = subprocess.run(["bash", "-c", line + " --help"], env=active,
                          cwd=bundle, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, (line, done.stderr)
    assert done.stderr.startswith("molbuilder "), (line, done.stderr)
