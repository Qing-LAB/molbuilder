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
    failed, and the points after it in its chain are not run either.  Run as
    bash, since what runs is the shell's answer, not the text's.

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
                          stop_on_failure=True, chains=[0, 0, 0])
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path,
                       capture_output=True, text=True)
    ran = sorted(p for p in ("p0", "p1", "p2")
                 if (tmp_path / p / "ran").exists())
    log = (tmp_path / "walk.log").read_text()
    assert ran == ["p0"], (ran, log)
    assert r.returncode != 0, log
    assert "dev@p1 not run -- its start could not be copied" in log, log
    assert "dev@p2 not run -- its chain stopped" in log, log
