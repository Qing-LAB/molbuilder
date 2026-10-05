"""What the engine says it used.

`model/parse.md` § 5d.2-5d.3.  The run record's setup column is read from the
ENGINE's own account -- SIESTA's ``fdf.<timestamp>.log`` -- never echoed from
the deck.

API-level.  ``fdf.20260925T194936.695.log`` is the whole fdf log of the
diverged 42-pole device run of 2026-09-25 -- measured.  The per-phase timing
test stood on that run's output cut short and on epochs made up for it, and
was retired 2026-10-04 (`process/testing.md` § 6); the first device run
through the road times its phases (plan § 5t, P3).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import detect

_HERE = Path(__file__).parent / "fixtures" / "transiesta"


def test_the_engine_s_own_log_says_what_it_used_and_what_nobody_set():
    """Every key SIESTA read, with the value it read: a setting the deck
    carried reads as set, a label it did not carry reads as a default -- and
    a key read to two values keeps both, in order.  The diverged device's
    contour pole energy was that: nobody set it, TranSIESTA read it as 1.5 eV
    (0.1102 Ry) and then as the continued fraction's own 0.2507 Ry, and ran
    the second -- 42 poles, in no record until this log was read."""
    path = _HERE / "fdf.20260925T194936.695.log"
    res = detect(path).parse(path)
    assert res.result_kind == "engine-params"
    mesh = res.params["meshcutoff"]
    assert (mesh["number"], mesh["unit"], mesh["default"]) == (150.0, "Ry",
                                                               False)
    pole = res.params["tscontourseqpole"]
    assert "number" not in pole, "a key read to two values has no one value"
    assert [(r["number"], r["default"]) for r in pole["readings"]] == [
        (pytest.approx(0.1102479665), True),
        (pytest.approx(0.2507105650), True)]
    assert res.params["scfmixermethod"] == {"key": "SCF.Mixer.Method",
                                            "value": "Pulay",
                                            "default": True}
    assert res.params["paoenergyshift"]["original"].startswith("0.1000000000E-01")
    # Found by the DECK's spelling of the block: fdf's own label rule.
    from molbuilder.parse.fdf import _norm
    kgrid = res.blocks[_norm("kgrid_Monkhorst_Pack")]
    assert kgrid[2].split()[:3] == ["0", "0", "1"]


def test_one_setting_read_in_two_cases_is_one_reading():
    """TranSIESTA reads `MD.TypeOfRun` four times -- none, CG, none, cg --
    each `# default value`: two settings, not four, because fdf matches a
    value's words case-blind.  The fdf log's reader keeps each distinct
    reading once.

    On the 2026-09-25 device's own log: only a TranSIESTA run reads a key
    that way, and a device run takes hours, not a test's minutes.  Whether a
    key is the deck's is checked on a run the road makes
    (`test_siesta_stopped_run_e2e.py`).

    MUTATION THIS MUST FAIL AGAINST: readings compared by case.
    """
    path = _HERE / "fdf.20260925T194936.695.log"
    res = detect(path).parse(path)
    readings = res.params["mdtypeofrun"]["readings"]
    assert [r["value"] for r in readings] == ["none", "CG"]


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# `test_each_phase_is_timed_on_its_own` timed the device output cut to its
# first 1453 and last 35 lines, on epochs made up for it (`process/testing.md`
# § 6).
