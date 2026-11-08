"""The ``jobset`` model's own rules, and the road's two ends it stands on
(docs/execution/job-system.md).

WHAT PREP AND LAUNCH DO WITH A PLAN IS ASKED OF THE PLAN PREP WROTE, as rows
(`tests/data/prep_protocol.toml`: a stage's number and the plan that keeps it,
what a first prep says; `gpu_contract.toml`: a benchmark's plan;
`the_catalogue.toml`: where its progress log lies).  What is here: the rules a
``JobSet`` holds of itself -- in memory, which is correct input for its own
class -- the one resolver of a typed stage, a folder that is no calculation,
what a job declares surviving its plan, and the one run script on two machines.
"""

from __future__ import annotations

import pytest

from molbuilder.identity import StageRef, resolve_stage_ref
from molbuilder.jobset.model import Job, JobSet, WarmFile
from support.machines import SLURM, WORKSTATION
from support.road import (described, group_words, jobset, read_request,
                          road_tree, stage_file, write_request)


# --------------------------------------------------------------------- #
#  The model (job-set@1)                                                 #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("wrong, words", [
    (JobSet("n", "siesta", "ladder", jobs=[]), ["empty"]),
    (JobSet("n", "siesta", "bogus", jobs=[Job("a", "a.fdf")]), ["kind"]),
    (JobSet("JOB", "siesta", "ladder",
            jobs=[Job("s1", "JOB_01_s1.fdf",
                      warm=[WarmFile("JOB.CG", requires_same="optimizer")])]),
     ["optimizer", "traits"]),
], ids=["no job", "a kind no ladder has", "a condition it could never meet"])
def test_a_plan_no_producer_should_build_is_reported_by_name(wrong, words):
    """What the engines cannot recover from is a producer's bug, and each is
    reported naming it (`JobSet.validate`, which prep and the layout ask of the
    plan they build; `job-system.md` § 3).

    An empty job-set would prep and submit successfully while doing nothing,
    which reads to the person as a finished launch.  `kind` selects real
    behaviour downstream -- a ladder runs one stage at a time while a sweep's
    points are independent -- so an unknown value cannot be defaulted.  A
    `requires_same` naming a trait the job does not declare can never be
    satisfied, so the file would simply never be carried: "fail safe" here
    means starting cold while everything reports success."""
    (said,) = wrong.validate()
    for word in words:
        assert word in said, said


def test_what_a_job_declares_survives_its_plan(tmp_path, monkeypatch):
    """`prep` writes the plan and `launch` reads it, from the persisted file
    -- a declaration that does not round-trip is one the machine that runs the
    job never sees.  Asked of the plan prep wrote for the ladder a person gets:
    the tiers really do change optimizer between rung one and rung two, which
    is what makes `.CG` a live question (`job-contracts.md` § 4.2a)."""
    from molbuilder.persist import read_json
    bundle = described(road_tree(tmp_path, monkeypatch),
                       stage_strategy="vib-quality")
    r = jobset("prep", "task", *group_words(bundle, "coarse", "medium", "tight"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    plan = JobSet.of(bundle)
    # lowercase: the trait normalizes at its ONE producer so
    # "Broyden" vs "broyden" (one optimizer, two hands) compare equal
    assert [j.traits["optimizer"] for j in plan.jobs] == [
        "cg", "broyden", "broyden"]
    assert [(w.name, w.requires_same) for w in plan.job("medium").warm] == [
        ("H2.XV", None), ("H2.DM", None), ("H2.CG", "optimizer")]
    # ABSENT, not null, for an unconditional file (checkpointing.md S3).
    on_disk = read_json(JobSet.path_of(bundle))
    assert on_disk["jobs"][1]["warm"][0] == {"name": "H2.XV"}


# --------------------------------------------------------------------- #
#  A folder that is no calculation                                       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("verb, said", [
    (["status"], ["no job-set.json"]),
    (["prep", "bench"], ["not a described calculation", "molbuilder describe"]),
], ids=["status", "prep"])
def test_a_folder_that_is_no_calculation_is_refused_saying_so(
        isolated_projects_root, verb, said):
    """Pointing `--bundle` at the wrong directory is the ordinary mistake.
    Exiting 0 with an empty plan would read as "this bundle has no jobs" -- a
    different and much more alarming thing to be told about work you believe
    you prepared.  `describe` is floor 2's only writer, so prep names it as the
    next step (`job-system.md` § 5.1), and lays nothing out."""
    folder = isolated_projects_root / "P" / "not-a-calculation"
    folder.mkdir(parents=True)
    r = jobset(*verb, "--bundle", folder)
    assert r.exit_code != 0, r.output
    for words in said:
        assert words in r.output, r.output
    assert list(folder.iterdir()) == []


# --------------------------------------------------------------------- #
#  One resolver of a typed stage (`job-system.md` § 5.3)                 #
# --------------------------------------------------------------------- #

#: A ladder whose stage 2 was removed after its prep, a stage named ``3`` left
#: in it: stage names are ``[A-Za-z0-9_]+``, so one may legitimately be named
#: that -- which is exactly why the bare-number spelling died (user-settled
#: 2026-08-21).
_LADDER = [StageRef(1, "3"), StageRef(3, "tight")]
_SWEEP = [StageRef(None, "p1"), StageRef(None, "p2")]


@pytest.mark.parametrize("refs, typed, seq, name", [
    (_LADDER, "3", 1, "3"),            # ONLY ever the name ...
    (_LADDER, "#3", 3, "tight"),       # ... `#3` ONLY ever the assigned number
    (_LADDER, "#1", 1, "3"),
    (_LADDER, "tight", 3, "tight"),
    (_SWEEP, "p2", None, "p2"),        # one resolver serves both kinds
])
def test_a_stage_is_its_name_or_hash_its_number(refs, typed, seq, name):
    """``#`` cannot appear in a name, so the two can never collide; and `#N` is
    the number the stage's folder carries, never its row -- the two differ the
    moment a stage is removed (`engines/stages.md` R5)."""
    got = resolve_stage_ref(refs, typed)
    assert (got.seq, got.name) == (seq, name)


@pytest.mark.parametrize("refs, typed, says, lacks", [
    (_LADDER, "03_tight", "no stage named", None),   # tokens retired
    (_LADDER, "#2", "no stage numbered", None),      # the row of tight; not its seq
    (_SWEEP, "2", "p1, p2", "number"),               # no ordinals to offer
], ids=["a token", "a row", "a number a sweep cannot have"])
def test_a_stage_nothing_holds_is_refused_naming_the_ones_there_are(
        refs, typed, says, lacks):
    """One resolver serves both kinds, so its refusal names what the set holds
    and stops offering ordinals to a job-set that has none."""
    with pytest.raises(ValueError) as got:
        resolve_stage_ref(refs, typed)
    assert says in str(got.value), got.value
    assert lacks is None or lacks not in str(got.value), got.value


# --------------------------------------------------------------------- #
#  P12 unit 3 -- the two environments, pinned on the claim that matters  #
# --------------------------------------------------------------------- #

def test_the_inner_wrapper_is_byte_identical_on_both(tmp_path, monkeypatch):
    """**The claim the whole two-layer split rests on**, and it was asserted
    nowhere until 2026-08-10.

    `architecture.md` § 9: the outer `.sbatch` is a header whose body calls the
    inner `.run.sh`, and the inner one owns activation and launch.  If that is
    true then the SAME `.run.sh` runs in both places -- so a run you debugged
    on your laptop is the run the cluster performs.  If it ever stops being
    true, the laptop stops being a rehearsal and this test is how you find out.

    ON THE ROAD, two calculations -- so a function, not a row, which is one:
    one description by `molbuilder describe`, copied before anything was prepared, and
    each prepared by `jobset prep` -- for this machine, a workstation, and for
    `sol`, a cluster whose record names a queue, with the same way into an
    environment.
    """
    import shutil
    from conftest import write_machine_record
    from molbuilder.config_dir import ensure_private_dir
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      environments_dir,
                                      named_environment_path,
                                      write_environment)
    enters = {"activation": "conda activate",
              "preamble": "source /x/conda.sh"}
    write_machine_record(scheduler=WORKSTATION, env_init=enters)
    ensure_private_dir(environments_dir())
    write_environment(Environment(
        scheduler=SLURM, topology=Topology(sockets=2, cores_per_socket=64),
        domains=[Domain.from_row({
            "name": "cpu", "partition": "cpu", "qos": "public",
            "max_time": "1-00:00:00", "max_cores": 128,
            "node_types": [{"cores": 128, "nodes": 4}]})],
        env_init=enters), named_environment_path("sol"))
    ws = described(road_tree(tmp_path, monkeypatch))
    hpc = ws.parent / "H2-on-sol"
    shutil.copytree(ws, hpc)
    # A RUN ON A CLUSTER STATES ITS QUEUE, WALL AND MEMORY
    # (`architecture.md` § 5.2) -- in the job request, as a person states it.
    request = read_request(hpc)
    request["allocation"] = {"domain": "cpu", "time": "04:00:00", "mem": "8G"}
    write_request(hpc, request)
    for bundle, target in ((ws, "this"), (hpc, "sol")):
        r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
                   "--target", target)
        assert r.exit_code == 0, r.output
    assert stage_file(hpc, "coarse", ".sbatch"), "the cluster has no header"

    def _logic(text: str) -> str:
        """The script without its PROVENANCE timestamp.

        ``generated-at`` is wall clock at seconds precision, and the two preps
        above run in sequence -- so a byte-for-byte comparison fails whenever
        they straddle a second boundary, which has nothing to do with
        workstation versus cluster and everything to do with how busy the
        process was. The claim being tested is about the script's LOGIC; a
        generation timestamp is provenance by definition, so it is excluded
        rather than the claim being weakened.
        """
        return "\n".join(l for l in text.splitlines()
                          if not l.lstrip("# ").startswith("generated-at"))

    a = _logic(stage_file(ws, "coarse", ".run.sh").read_text())
    b = _logic(stage_file(hpc, "coarse", ".run.sh").read_text())
    assert a == b, (
        "the run script differs between a workstation and a cluster -- the "
        "inner wrapper is supposed to be the same file, so a laptop run is a "
        "rehearsal of the cluster run")
