"""The PROCESS half of a run's status — the wrapper's marker and the
engine's own.

`model/parse.md` § 5; the probe is `parse/dirs/job.py::_process_conclusion`.

**Every test here guards a defect that SHIPPED**, which is the only reason
they exist.  The probe was written and merged with no test at all, and the
adversarial review that found these ran a day later:

* the verdict compared the marker against the literal ``"rc=0"`` while the
  wrapper writes ``rc=<N> at <date>`` -- so **19 of the 20 markers in the
  checkout could not match**, and every content-silent successful run read
  ``failed``.  The one that passed was the hand-made fixture the change was
  "measured" on;
* ``0_NORMAL_EXIT`` carries no label, and in the FLAT shape one directory
  holds every rung -- so one rung's clean exit answered for all of them;
* the newest-attempt rule ranged over ``.concluded`` only, so a previous
  re-run's goodbye was read as this one's.
"""
from __future__ import annotations

import pathlib

from molbuilder.parse.dirs.job import (_process_conclusion, _rc_ok,
                                       run_status)

#: What the wrapper actually writes (`runwrap.py`: `printf "rc=%s at %s\n"`).
REAL = "rc=0 at Tue Sep 15 05:55:06 PM MST 2026"
RUNNING_OUT = "Siesta Version: 5.4.2\nsiesta: iscf\nscf:  1  -100.0\n"



def _dir(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / name.replace("__", ".")).write_text(text)
    return tmp_path


class TestTheMarkerIsParsedNotMatched:
    """`rc=` is a FIELD, and the wrapper writes a date after it."""

    def test_the_real_marker_shape_reads_as_success(self):
        assert _rc_ok(REAL) is True, (
            "the wrapper writes `rc=0 at <date>`; an exact test against "
            "'rc=0' matched 1 of the 20 markers in the checkout")

    def test_a_failure_reads_as_failure_in_the_same_shape(self):
        assert _rc_ok("rc=1 at Fri Sep 11 10:00:00 AM MST 2026") is False
        assert _rc_ok("rc=127 at Fri Sep 11") is False

    def test_the_engine_marker_is_success_and_a_sentinel_is_not(self):
        assert _rc_ok("0_NORMAL_EXIT") is True
        # `_process_conclusion` substitutes this for an unreadable marker.
        assert _rc_ok("rc=?") is False

    def test_nothing_to_read_is_not_success(self):
        for empty in ("", "\n", "rc=", "rc= at Tue"):
            assert _rc_ok(empty) is False, empty


class TestWhichMarkerSpeaks:

    def test_a_previous_attempts_goodbye_is_not_this_ones(self, tmp_path):
        """`materialize.attempt_concluded` states the rule: *"an earlier
        index's marker beside a newer unconcluded `.out` is a previous
        re-run's goodbye, not this one's."*  The probe shipped ranging over
        `.concluded` alone, so it read run0's marker while run1 was live."""
        _dir(tmp_path, **{"bdt__fdf": "SystemLabel bdt",
                          "bdt-run0__concluded": REAL,
                          "bdt-run1__out": RUNNING_OUT})
        assert _process_conclusion(tmp_path) is None

    def test_the_marker_counts_at_the_highest_index(self, tmp_path):
        _dir(tmp_path, **{"bdt__fdf": "x", "bdt-run0__out": RUNNING_OUT,
                          "bdt-run0__concluded": REAL})
        assert _process_conclusion(tmp_path) == REAL


class TestTheEngineMarkerCannotBeAttributedToARung:
    """`0_NORMAL_EXIT` is SIESTA's, written in its cwd, and carries no
    label.  Under the flat shape that cwd is shared by every rung."""

    def test_it_is_ignored_when_the_caller_narrowed_to_one_rung(self, tmp_path):
        _dir(tmp_path, **{"a_01_x__fdf": "x", "a_02_y__fdf": "x",
                          "0_NORMAL_EXIT": ""})
        assert _process_conclusion(tmp_path, "a_02_y*") is None, (
            "one rung's clean exit answered for another")

    def test_it_still_answers_when_nobody_narrowed(self, tmp_path):
        _dir(tmp_path, **{"a__fdf": "x", "0_NORMAL_EXIT": ""})
        assert _process_conclusion(tmp_path, "*") == "0_NORMAL_EXIT"


class TestTheStatusUsesIt:
    """The case the probe exists for: `attempt_concluded`'s docstring names
    it -- an engine that dies before printing leaves a marker and NO
    output.  Content is silent; only the marker can answer."""

    def test_a_marker_with_no_output_at_all_decides(self, tmp_path):
        _dir(tmp_path, **{"bdt__fdf": "x", "bdt-run0__concluded": REAL})
        st = run_status(tmp_path)
        assert st.state == "finished", st
        assert st.concluded == REAL

    def test_a_failed_conclusion_with_no_output_is_failed(self, tmp_path):
        _dir(tmp_path, **{"bdt__fdf": "x",
                          "bdt-run0__concluded": "rc=1 at Fri Sep 11"})
        assert run_status(tmp_path).state == "failed"

    def test_no_marker_and_no_growth_is_still_stale_not_failed(self, tmp_path):
        """The age rule survives: a killed job leaves no goodbye, and only
        the clock can say so."""
        import os, time
        _dir(tmp_path, **{"bdt__fdf": "x", "bdt-run0__out": RUNNING_OUT})
        old = time.time() - 3600
        os.utime(tmp_path / "bdt-run0.out", (old, old))
        assert run_status(tmp_path).state == "stale"


class TestTheEnginesOwnMarkerCountsForACitation:
    """SIESTA writes `0_NORMAL_EXIT` as its last act on a clean exit, so a
    run carrying it ran to its own end whatever launched it — or nothing
    did.  `engines/transport.md`: *"evidence is FILES, never a marker
    spelling of ours."*

    **This guards a regression that shipped and broke a citation.**
    `classify_citation` weighed this marker itself until the 2026-09-18
    migration onto `run_status` deleted the fallback; the revert restored
    the call and not the fallback, so `attempt_concluded` — which cannot
    see an unlabelled marker — answered `None` for every SIESTA-only run.
    Measured: 5 of 5 citable directories in the checkout refused to
    compose, and the Transport tab printed "NOT CONCLUDED — still running,
    or force-stopped" for a finished relaxation.
    """

    def test_a_siesta_only_relaxation_reads_as_concluded(self, tmp_path):
        from molbuilder.transport.compose import classify_citation
        (tmp_path / "relax.fdf").write_text("SystemLabel relax\n")
        (tmp_path / "relax.XV").write_text("x\n")
        (tmp_path / "0_NORMAL_EXIT").write_text("")
        cited = classify_citation(tmp_path)
        assert cited.concluded == "0_NORMAL_EXIT", (
            "SIESTA's own clean-exit marker did not count -- a finished "
            f"relaxation reads as still running: {cited}")

    def test_without_it_the_same_directory_is_not_concluded(self, tmp_path):
        """Anti-vacuity: the assertion above must not pass by calling
        everything concluded."""
        from molbuilder.transport.compose import classify_citation
        (tmp_path / "relax.fdf").write_text("SystemLabel relax\n")
        (tmp_path / "relax.XV").write_text("x\n")
        assert classify_citation(tmp_path).concluded is None


class TestTheEngineMarkerMayNotDecide:
    """`0_NORMAL_EXIT` is a bare filename: no label, no `-run<N>`.

    A `.concluded` carries both, so `_process_conclusion` can refuse a
    PREVIOUS attempt's goodbye.  The engine's marker cannot be attributed to
    an attempt at all, so a leftover promoted a silent, un-growing output to
    `finished`.  `_process_conclusion` already refuses it on the STAGE axis
    when the caller narrows; this is the ATTEMPT axis.
    """

    def test_a_leftover_marker_does_not_promote_a_silent_output(self,
                                                                tmp_path):
        import os
        import time
        _dir(tmp_path, **{"bdt__fdf": "x", "bdt-run0__out": RUNNING_OUT,
                          "0_NORMAL_EXIT": ""})
        old = time.time() - 3600
        os.utime(tmp_path / "bdt-run0.out", (old, old))
        st = run_status(tmp_path)
        assert st.state == "stale", (
            "a marker that cannot be attributed to an attempt decided the "
            f"state anyway: {st}")
        assert st.concluded == "0_NORMAL_EXIT", (
            "the evidence must still be REPORTED -- it just may not decide")

    def test_an_attributable_marker_still_decides_the_same_case(self,
                                                               tmp_path):
        """The narrowing is exactly one marker wide: a `.concluded` beside
        the same silent output still says finished."""
        import os
        import time
        _dir(tmp_path, **{"bdt__fdf": "x", "bdt-run0__out": RUNNING_OUT,
                          "bdt-run0__concluded": REAL})
        old = time.time() - 3600
        os.utime(tmp_path / "bdt-run0.out", (old, old))
        assert run_status(tmp_path).state == "finished"
