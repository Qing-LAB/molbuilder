"""The transmission rung's own account, and the header it shares with SIESTA.

`model/parse.md` § 5d.2 and § 5d.5.  No reader claimed a TBtrans ``.out``
until 2026-09-26: the transmission rung reached no record, and the wrapper
labelled it SIESTA.

API-level: ``chain-tbtrans-v0.4.out`` is a MEASURED TBtrans 5.4.2 output
(the chain ladder's transmission rung, 2026-08-29) and
``device-converging-live.out`` a measured SIESTA 5.4.2 one.  The spin-channel
test names its files by TBtrans's own rule (``Util/TS/TBtrans/m_tbt_save.F90``
``name_save``) because no polarized transport run exists to measure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import detect
from molbuilder.parse.engines.tbtrans import (
    read_tbtrans_out, transmission_files,
)

_DATA = Path(__file__).parents[1] / "data"
_TS = Path(__file__).parent / "fixtures" / "transiesta"


def test_tbtrans_states_what_ran_and_what_flowed():
    """Its build, its ranks, its k-points, its time -- and the bias it
    applied with the current that flowed, which is the rung's result."""
    facts = read_tbtrans_out((_DATA / "chain-tbtrans-v0.4.out").read_text())
    assert (facts["build"]["executable"], facts["build"]["version"]) == (
        "tbtrans", "5.4.2")
    assert (facts["n_mpi_processes"], facts["k_points"]) == (4, 1)
    assert facts["completed_s"] == [pytest.approx(13.992)]  # one pass
    assert facts["run_start_local"] == "2026-08-29T02:26:53"
    (flow,) = facts["currents"]
    assert (flow["from"], flow["to"]) == ("L", "R")
    assert flow["voltage_v"] == pytest.approx(0.4)
    assert flow["current_a"] == pytest.approx(0.309835e-4)
    assert flow["power_w"] == pytest.approx(-0.619664e-5)


def test_siesta_s_header_and_launch_lines_are_read_by_the_same_reader(
        tmp_path):
    """SIESTA prints TBtrans's header and launch lines, and one reader reads
    both -- which is what records the packaged SIESTA's ELSI (5.4.2 writes
    ``ELSI support. Solvers:``, which the bare-name pattern it replaced read
    as absent) and a one-rank run's rank count (``* Running in serial mode``,
    which no reader took until 2026-09-26).  ``offsetH2.out`` is a measured
    one-rank SIESTA 5.4.2 run."""
    import shutil
    path = str(_TS / "device-converging-live.out")
    info = detect(path).parse(path).runtime_info
    assert (info["siesta_build"]["executable"],
            info["siesta_build"]["version"]) == ("siesta", "5.4.2")
    assert info["siesta_build"]["elsi"] is True
    assert info["n_mpi_processes"] == 10
    serial = tmp_path / "offsetH2.out"
    shutil.copy(Path(__file__).parent / "fixtures" / "siesta_mdnc"
                / "offsetH2.out", serial)
    info = detect(str(serial)).parse(str(serial)).runtime_info
    assert info["n_mpi_processes"] == 1


def test_a_polarized_run_s_two_channels_are_found(tmp_path):
    """A polarized TBtrans run writes ``.TBT_UP.`` and ``.TBT_DN.`` files and
    no ``.TBT.`` one -- a reader globbing ``.TBT.`` alone finds nothing."""
    for name in ("dev.TBT_UP.AVTRANS_L-R", "dev.TBT_DN.AVTRANS_L-R",
                 "other.TBT.AVTRANS_L-R"):
        (tmp_path / name).write_text("")
    found = transmission_files(tmp_path, "dev")
    assert sorted(found) == ["down", "up"]
    assert found["up"][0].name == "dev.TBT_UP.AVTRANS_L-R"
