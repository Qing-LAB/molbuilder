"""The transmission rung's files, found by TBtrans's own names.

`model/parse.md` § 5d.2, § 5d.5 and § 5e.  The spin-channel test names its
files by TBtrans's own rule (``Util/TS/TBtrans/m_tbt_save.F90`` ``name_save``) because
no polarized transport run exists to measure.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.engines.tbtrans import averaged_files


@pytest.mark.parametrize("files,pair,leads,found", [
    (("dev.TBT_UP.AVTRANS_L-R", "dev.TBT_DN.AVTRANS_L-R",
      "other.TBT.AVTRANS_L-R"), "L-R", ("L", "R"),
     {"up": {"transmission": "dev.TBT_UP.AVTRANS_L-R"},
      "down": {"transmission": "dev.TBT_DN.AVTRANS_L-R"}}),
], ids=["a-polarized-run's-two-channels"])
def test_a_points_averaged_files_are_found_by_tbtrans_own_names(
        tmp_path, files, pair, leads, found):
    """TBtrans names its k-averaged files by the channel and by the leads
    the deck's ``TS.Elecs`` block names (``name_save``): a polarized run
    writes ``.TBT_UP.`` and ``.TBT_DN.`` files and no ``.TBT.`` one.  THE
    SILENT FAILURE: a reader globbing ``.TBT.`` alone, finding nothing of a
    polarized run."""
    for name in files:
        (tmp_path / name).write_text("")
    got = averaged_files(tmp_path, "dev", pair=pair, leads=leads)
    assert {ch: {q: ({k: p.name for k, p in v.items()} if isinstance(v, dict)
                     else v.name) for q, v in d.items()}
            for ch, d in got.items()} == found