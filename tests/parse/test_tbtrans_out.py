"""The transmission rung's files, found by TBtrans's own names.

`model/parse.md` § 5d.2 and § 5d.5.  The spin-channel test names its files by
TBtrans's own rule (``Util/TS/TBtrans/m_tbt_save.F90`` ``name_save``) because
no polarized transport run exists to measure.
"""
from __future__ import annotations


from molbuilder.parse.engines.tbtrans import transmission_files


def test_a_polarized_run_s_two_channels_are_found(tmp_path):
    """A polarized TBtrans run writes ``.TBT_UP.`` and ``.TBT_DN.`` files and
    no ``.TBT.`` one -- a reader globbing ``.TBT.`` alone finds nothing."""
    for name in ("dev.TBT_UP.AVTRANS_L-R", "dev.TBT_DN.AVTRANS_L-R",
                 "other.TBT.AVTRANS_L-R"):
        (tmp_path / name).write_text("")
    found = transmission_files(tmp_path, "dev")
    assert sorted(found) == ["down", "up"]
    assert found["up"][0].name == "dev.TBT_UP.AVTRANS_L-R"
