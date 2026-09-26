"""A TranSIESTA device's two SCF phases, read from its own output.

`model/parse.md` § 5d.5-5d.6.  A device runs SIESTA's periodic initialization
and then TranSIESTA's NEGF loop in one ``.out``; until 2026-09-26 the parser
matched only the first phase's ``scf:`` rows, so a device that diverged for
1000 NEGF iterations was reported with the initialization's energy and, while
live, as *converged*.

API-level on MEASURED fixtures: both files are trimmed from the real device
runs of 2026-09-25/26 (`projects/claude-w33/transport/au333bdt-t/04_device`):
``device-diverging.out`` is the 42-pole run -- its first 1453 lines and its
last 35 -- and ``device-converging-live.out`` the whole output of the 123-pole
run, stopped at its ninth NEGF iteration.
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


def test_both_phases_are_read_and_each_negf_iteration_carries_its_charge():
    """The periodic cycles stay beside the NEGF loop's, and every NEGF
    iteration carries the charges TranSIESTA reported for it -- which is what
    shows the diverged run losing 29 electrons on its FIRST step, before any
    mixing (§ 5d.6's set-up-fault symptom)."""
    res = _parse("device-diverging.out")
    phases = res.runtime_info["scf_phases"]
    assert phases["periodic"]["cycles"] == 7
    assert phases["negf"]["cycles"] == 8
    first, last = _negf(res)[0], _negf(res)[-1]
    assert first["dq"] == pytest.approx(-29.1)
    assert first["charges"]["C1"] < 0 and first["charges"]["C2"] < 0
    assert first["vha_ev"] == pytest.approx(-18.656448)
    assert last["cycle"] == 1000 and last["dq"] == pytest.approx(-584.0)


def test_a_device_energy_is_its_negf_phase():
    """The device reports the NEGF loop's energy -- the initialization's
    -437,029 eV was how a run 584 electrons short read as sound."""
    res = _parse("device-diverging.out")
    assert res.frames[-1].energy == pytest.approx(-205444.335258)
    assert res.scf_converged is False
    assert res.runtime_info["scf_phases"]["negf"]["converged"] is False


def test_a_live_device_is_not_converged_by_its_initialization():
    """Mid-NEGF, the device has not converged: the periodic phase's "SCF cycle
    converged" line does not speak for it."""
    res = _parse("device-converging-live.out")
    assert res.scf_converged is None
    assert res.runtime_info["scf_phases"]["periodic"]["converged"] is True
    assert res.runtime_info["scf_phases"]["negf"]["converged"] is None
    assert res.frames[-1].energy == pytest.approx(-503070.814878)
    assert _negf(res)[-1]["dq"] == pytest.approx(0.0172)


def test_a_converged_device_keeps_its_negf_energy_over_the_closing_line(
        tmp_path):
    """When the NEGF loop converges, SIESTA closes the step with
    ``siesta: E_KS(eV) =`` -- and that line prints SIESTA's own formula even
    in a TranSIESTA run (``Src/state_analysis.F``), 66 eV from the NEGF
    energy at step 1.  The device keeps the NEGF phase's number, and its
    convergence is the NEGF phase's.

    The live fixture, finished with the lines a converged run prints -- each
    in SIESTA's own format (``Src/scfconvergence_test.F``,
    ``Src/state_analysis.F``), the E_KS value the fixture's own SIESTA-formula
    ``Etot`` at the first NEGF step.
    """
    body = (_HERE / "device-converging-live.out").read_text()
    body += ("\nSCF Convergence by DM+H+dQ criterion\n"
             "max |DM_out - DM_in|         :     0.0000049800\n"
             "SCF cycle converged after 9 iterations\n"
             "\nsiesta: E_KS(eV) =          -437034.5303\n")
    f = tmp_path / "device-converged.out"
    f.write_text(body)
    res = detect(str(f)).parse(str(f))
    assert res.frames[-1].energy == pytest.approx(-503070.814878)
    assert res.scf_converged is True
    assert res.runtime_info["scf_phases"]["negf"]["converged"] is True


def test_transiesta_states_its_contour_and_its_starting_charge():
    """TranSIESTA's start-up echo is the only place the continued fraction's
    pole count exists, and the charge distribution at the switch is the
    baseline every NEGF iteration is judged against.  The echo lists one
    segment per chemical potential with the same labels
    (``Src/m_ts_contour_eq.f90``), so each is kept on its own."""
    div = _parse("device-diverging.out").runtime_info["transiesta"]
    live = _parse("device-converging-live.out").runtime_info["transiesta"]
    (seg,) = div["contours"]["Residual contour"]
    assert seg["Number of poles"] == "42"
    assert seg["Continued fraction chemical potential"] == "0.0000 eV"
    assert live["contours"]["Residual contour"][0]["Number of poles"] == "123"
    q = div["charge_at_switch"]
    assert q["target"] == pytest.approx(2092.0)
    assert q["C1"] == pytest.approx(3.59449)
    assert abs(q["dQ"]) < 1e-9
    assert div["electrodes"]["L"]["principal_cell"] == "perfect"
    assert div["electrodes"]["R"]["principal_cell"] == "perfect"
