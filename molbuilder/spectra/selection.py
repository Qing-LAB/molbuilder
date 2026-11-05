"""Which modes a per-mode step runs on -- PySCF's orbital probe, SIESTA's
optical calls.

One function, :func:`select_modes`, and both jobs import it: this file
travels beside the job inside ``mb_pyscf.pyz`` and ``mb_vibration.pyz``
(``engines/pyscf.md`` § 3, ``engines/vibration.md`` § 5.11), where
molbuilder is not installed -- so it imports nothing.

The selectors (`engines/vibration.md` § 3.1, § 4.8):

    skip      -> []  (the probe's: none)
    all       -> every mode inside the frequency window, when one is given
    range     -> every mode inside the frequency window (Raman for's)
    explicit  -> exactly the listed modes (the probe's); the window is
                 IGNORED -- the person named specific modes, and the
                 window does not override that
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def select_modes(frequencies_cm1: Sequence[float], selection: str, *,
                 explicit: Sequence[int] = (),
                 freq_min_cm1: Optional[float] = None,
                 freq_max_cm1: Optional[float] = None) -> List[int]:
    """The 1-based numbers of the modes a per-mode step runs on.

    ``frequencies_cm1`` holds every mode's frequency in its order, so mode
    ``n`` is ``frequencies_cm1[n - 1]``; an imaginary mode's is negative.
    ``explicit`` is the listed modes as the config's one reader gives them
    (`PySCFConfig.explicit_modes`: sorted, each once).  The window
    ``[freq_min_cm1, freq_max_cm1]`` is closed, and either bound ``None``
    leaves that side open -- a negative minimum opts imaginary modes in.

    A listed number that names no mode is dropped: nothing can be probed
    for it, and the selection is what the run records.  The script says
    which were dropped -- the mode count is known only once the Hessian is
    done, so nothing before the run could refuse them.
    """
    if selection == "skip":
        return []
    if selection == "explicit":
        return [n for n in explicit if 1 <= n <= len(frequencies_cm1)]
    if selection in ("all", "range"):
        return [n for n, f in enumerate(frequencies_cm1, start=1)
                if (freq_min_cm1 is None or f >= freq_min_cm1)
                and (freq_max_cm1 is None or f <= freq_max_cm1)]
    raise ValueError(f"mode selection {selection!r}: the selectors are "
                     f"'skip', 'all', 'range' and 'explicit'")


__all__ = ["select_modes"]
