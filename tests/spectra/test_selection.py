"""``select_modes``: which modes the per-mode probe runs on.

`engines/vibration.md` § 4.8 -- the three selectors and the frequency window
that filters `all`.  The PySCF vibration script imports this function from
`mb_pyscf.pyz` (`engines/pyscf.md` § 3), so these test the selector a run
uses.  The listed modes' TEXT is read by
`PySCFConfig.explicit_modes` before it gets here, and is tested where it is
read (`test_methods.py`, the end-to-end probe run).
"""

from __future__ import annotations

from molbuilder.spectra import select_modes

#: Six modes' frequencies, cm⁻¹, in mode order.
FREQS = [412.3, 745.0, 1023.4, 1612.0, 2956.0, 3656.0]


def test_skip_selects_nothing():
    assert select_modes(FREQS, "skip") == []


def test_all_selects_every_mode_inside_the_window():
    """Window [800, 2500] -> modes 3 (1023) and 4 (1612)."""
    assert select_modes(FREQS, "all") == [1, 2, 3, 4, 5, 6]
    assert select_modes(FREQS, "all", freq_min_cm1=800.0,
                        freq_max_cm1=2500.0) == [3, 4]


def test_explicit_selects_the_listed_modes_and_ignores_the_window():
    """Naming a mode is saying *that one*: the window, which would drop
    mode 3, does not apply.  A number that names no mode -- 9, of six -- is
    dropped (§ 4.8): nothing can be probed for it."""
    assert select_modes(FREQS, "explicit", explicit=[3, 5, 6],
                        freq_min_cm1=2000.0) == [3, 5, 6]
    assert select_modes(FREQS, "explicit", explicit=[3, 9]) == [3]
