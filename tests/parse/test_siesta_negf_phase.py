"""A TranSIESTA device's two SCF phases, read from its own output.

`model/parse.md` § 5d.5-5d.6.  A device runs SIESTA's periodic initialization
and then TranSIESTA's NEGF loop in one ``.out``; until 2026-09-26 the parser
matched only the first phase's ``scf:`` rows, so a device that diverged for
1000 NEGF iterations was reported with the initialization's energy and, while
live, as *converged*.

API-level on a MEASURED fixture: ``device-converging-live.out``, the whole
output of the 123-pole device run of 2026-09-26
(`projects/claude-w33/transport/au333bdt-t/04_device`), stopped at its ninth
NEGF iteration.  The tests on the 42-pole run's output cut to its first 1453
and last 35 lines were retired with it 2026-10-04 (`process/testing.md` § 6).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import detect

_HERE = Path(__file__).parent / "fixtures" / "transiesta"


def _parse(name):
    path = str(_HERE / name)
    return detect(path).parse(path)


def _negf(res):
    return [c for c in (res.frames[-1].scf_history or [])
            if c.get("phase") == "negf"]


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 4 tests here read the device output cut to its first 1453 and
# last 35 lines, or the live one with convergence lines added by hand (`process/testing.md` § 6).


def test_a_live_device_is_not_converged_by_its_initialization():
    """Mid-NEGF, the device has not converged: the periodic phase's "SCF cycle
    converged" line does not speak for it."""
    res = _parse("device-converging-live.out")
    assert res.scf_converged is None
    assert res.runtime_info["scf_phases"]["periodic"]["converged"] is True
    assert res.runtime_info["scf_phases"]["negf"]["converged"] is None
    assert res.frames[-1].energy == pytest.approx(-503070.814878)
    assert _negf(res)[-1]["dq"] == pytest.approx(0.0172)
