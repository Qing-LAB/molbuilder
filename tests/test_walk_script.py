"""The one walk script (`submit._walk_script`), run as bash -- what a
group's job, a benchmark's shelf and a run of points execute in one
submission (`job-system.md` § 6.0; `engines/transport.md` § 2a.11)."""
from __future__ import annotations

import subprocess


def test_a_point_whose_start_cannot_be_copied_is_not_run(tmp_path):
    """`engines/transport.md` § 2a.11 -- each walked point starts from the
    point before it in its frame, and its record says so (`.continued-from`,
    written with the plan).  A point before it that ended without leaving
    that file leaves nothing to start from: the point is not run -- run from
    its deck, its record would name a start it never had -- it counts as
    failed, and the points after it in its chain are not run either -- and
    why the copy failed, the shell's own words, is in the walk's log, not on
    a terminal a walk left in the background no longer has (`job-system.md`
    § 6.0).  Run as bash, since what runs is the shell's answer, not the
    text's.

    API-LEVEL: the generated walk, run on stub members that only record
    that they ran."""
    from molbuilder.jobset.submit import _walk_script
    for p in ("p0", "p1", "p2"):
        (tmp_path / p).mkdir()
        (tmp_path / p / "run.sh").write_text("touch ran\n")
    steps = [("dev@p0", "p0", "run.sh", "", ()),
             ("dev@p1", "p1", "run.sh", "", (("p0/H2.TSDE", "p1"),)),
             ("dev@p2", "p2", "run.sh", "", (("p1/H2.TSDE", "p2"),))]
    script = _walk_script("walk", steps, where="a run of points",
                          log="walk.log", member_word="point",
                          needs=[None, 0, 1])
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path,
                       capture_output=True, text=True)
    ran = sorted(p for p in ("p0", "p1", "p2")
                 if (tmp_path / p / "ran").exists())
    log = (tmp_path / "walk.log").read_text()
    assert ran == ["p0"], (ran, log)
    assert r.returncode != 0, log
    assert "dev@p1 not run -- its start could not be copied" in log, log
    assert "dev@p2 not run -- what it builds on failed" in log, log
    assert "p0/H2.TSDE" in log and r.stderr == "", (log, r.stderr)


def test_a_walk_started_again_runs_no_point(tmp_path, monkeypatch):
    """`execution/runs-of-points.md` § 3 -- the walk starts once, as every
    run does (`project-layout.md` § 1.6.1): the walk a launch wrote,
    started again in its run -- as a job its scheduler requeued starts
    again -- refuses before it starts a point, since a point it walks holds
    a start, names the launch that opens the next run, and says so in its
    log.  THE SILENT FAILURE: a requeued job running, in the same run, the
    points its first start never reached.

    The road: H2's vibration launched here, its relaxation point ending in
    its activation, its force constants not run -- then its walk started
    again from the stage's folder, as its header would."""
    from support.road import jobset, prepared_here
    bundle, run, names = prepared_here(tmp_path, monkeypatch, "false",
                                       name="H2vib", calculation="vibration",
                                       stage="freq")
    r = jobset("launch", "task", "--bundle", bundle, "--mode", "direct",
               "--foreground", "--yes")
    assert r.exit_code == 1, r.output
    log = names.name(".runwrap.log")
    relax, fc = run / "relax", run / "fc"
    first = (relax / log).read_bytes()
    assert not (fc / log).exists()
    home = run.parent
    (walk,) = (home / "launch").glob("*.run.sh")
    again = subprocess.run(["bash", str(walk)], cwd=home,
                           capture_output=True, text=True)
    said = (home / "launch" / walk.name.replace(".run.sh", ".log")
            ).read_text()
    assert again.returncode == 2, said
    assert (f"refused -- this run already holds a start (run-0/relax/{log})"
            f": a run is one start; to continue it, launch the stage again"
            in said), said
    assert (relax / log).read_bytes() == first and not (fc / log).exists()


def test_a_warm_launch_with_nothing_to_take_is_refused_naming_cold(
        tmp_path, monkeypatch):
    """`job-system.md` R-L17, for a run of points as for a run of one: a
    stage launched again warm continues from its newest launched run, so a
    run that left nothing it takes -- no point done, no point's warm files
    -- is refused before anything is written, naming the cold launch.
    THE SILENT FAILURE: a new run recorded as continuing from one it took
    nothing from.  The road: H2's vibration launched here, its relaxation
    ending in its activation before it wrote anything, then launched again
    warm."""
    from support.road import jobset, prepared_here
    bundle, run, _names = prepared_here(tmp_path, monkeypatch, "false",
                                        name="H2vib",
                                        calculation="vibration",
                                        stage="freq")
    r = jobset("launch", "task", "--bundle", bundle, "--mode", "direct",
               "--foreground", "--yes")
    assert r.exit_code == 1, r.output
    again = jobset("launch", "task", "--stage", "freq", "--bundle", bundle,
                   "--mode", "direct", "--foreground", "--yes")
    assert again.exit_code != 0, again.output
    assert "which is impossible here" in again.output, again.output
    assert "--cold" in again.output, again.output
    assert not (run.parent / "run-1").exists()

