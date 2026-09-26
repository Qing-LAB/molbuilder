"""What the engine says it used, and how long each phase took.

`model/parse.md` § 5d.2-5d.3.  The run record's setup column is read from the
ENGINE's own account -- SIESTA's ``fdf.<timestamp>.log`` -- never echoed from
the deck; and a device's seconds per iteration are its phases', not one
average of two.

API-level.  ``fdf.20260925T194936.695.log`` is the whole fdf log of the
diverged 42-pole device run of 2026-09-25 -- measured.  The timing test's ROWS
are that run's own (``device-diverging.out``); its EPOCHS are constructed, at
the rates that run measured, because no wrapper wrote a two-phase timing log
before the tee read both phases -- the first device run through the road
replaces them (plan § 5t, P3).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import detect
from molbuilder.parse.instruments.scf_timing import scf_timing_metrics

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


def test_each_phase_is_timed_on_its_own():
    """A device's periodic iterations and its NEGF iterations are timed
    separately, the step between them -- TranSIESTA's switch and the whole
    first NEGF iteration -- is timed as neither, and the headline is the
    NEGF loop's."""
    from molbuilder.parse.engines.siesta_grammar import PHASE_NEGF, scf_row
    rows = [(line, scf_row(line)) for line in
            (_HERE / "device-diverging.out").read_text().splitlines()]
    t, log = 1000.0, []
    for line, row in rows:
        if row is None:
            continue
        log.append(f"{t:.3f} {row.iscf} {line}")
        t += 27.5 if row.phase == PHASE_NEGF else 97.0
        if row.phase != PHASE_NEGF and row.iscf == 7:
            t += 600.0                         # the switch + NEGF iteration 1
    m = scf_timing_metrics("\n".join(log))
    assert m["s_per_iter_periodic"] == pytest.approx(97.0)
    assert m["s_per_iter_negf"] == pytest.approx(27.5)
    assert (m["rows_periodic"], m["rows_negf"]) == (7, 8)
    # 7 and 8 rows give 6 and 7 intervals WITHIN each phase, less the first
    # (warm-up) -- the cross-phase step is none of them.
    assert (m["iters_measured_periodic"], m["iters_measured_negf"]) == (5, 6)
    assert (m["s_per_iter"], m["iters_measured"]) == (pytest.approx(27.5), 6)
