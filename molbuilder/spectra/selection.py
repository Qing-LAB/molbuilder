"""Which modes the per-mode electronic-structure probe runs on.

One function, :func:`select_modes`, and the PySCF vibration script imports
it: this file travels beside the job inside ``mb_pyscf.pyz``
(``runwrap.PYSCF_COMPANIONS``, ``engines/pyscf.md`` § 3), where molbuilder is
not installed -- so it imports nothing.  Until 2026-10-05 the script carried
a hand-written copy of it, held equal to this one by a test.

The three selectors (`engines/vibration.md` § 4.8):

    skip      -> []  (no probe)
    all       -> every mode inside the frequency window
    explicit  -> exactly the listed modes; the window is IGNORED --
                 the person named specific modes, and the window does
                 not override that

(`top_n` and `threshold` ranked modes by Raman activity; retired by
decision 2026-09-23 and removed 2026-09-28, V1.6 -- the probe measures
d(eps)/dQ, which follows its own selection rule.)
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def select_modes(frequencies_cm1: Sequence[float], selection: str, *,
                 explicit: Sequence[int] = (),
                 freq_min_cm1: Optional[float] = None,
                 freq_max_cm1: Optional[float] = None) -> List[int]:
    """The 1-based numbers of the modes the probe runs on.

    ``frequencies_cm1`` holds every mode's frequency in its order, so mode
    ``n`` is ``frequencies_cm1[n - 1]``; an imaginary mode's is negative.
    ``explicit`` is the listed modes as the config's one reader gives them
    (`PySCFConfig.explicit_modes`: sorted, each once).  The window
    ``[freq_min_cm1, freq_max_cm1]`` is closed, and either bound ``None``
    leaves that side open -- a negative minimum opts imaginary modes in.

    A listed number with no mode is returned as listed: the mode count is
    known only once the Hessian is done, and the script skips such a number
    there, saying so.
    """
    if selection == "skip":
        return []
    if selection == "explicit":
        return list(explicit)
    if selection == "all":
        return [n for n, f in enumerate(frequencies_cm1, start=1)
                if (freq_min_cm1 is None or f >= freq_min_cm1)
                and (freq_max_cm1 is None or f <= freq_max_cm1)]
    raise ValueError(f"es_mode_selection {selection!r}: the selectors are "
                     f"'skip', 'all' and 'explicit'")


__all__ = ["select_modes"]
