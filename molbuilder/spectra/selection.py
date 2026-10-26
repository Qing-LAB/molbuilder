"""Which modes the per-mode electronic-structure probe runs on.

One function, :func:`select_modes`, and the PySCF vibration script imports
it: this file travels beside the job inside ``mb_pyscf.pyz``
(``runwrap.PYSCF_COMPANIONS``, ``engines/pyscf.md`` § 3), where molbuilder is
not installed -- so it imports nothing.

The three selectors (`engines/vibration.md` § 4.8):

    skip      -> []  (no probe)
    all       -> every mode inside the frequency window
    explicit  -> exactly the listed modes; the window is IGNORED --
                 the person named specific modes, and the window does
                 not override that
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence


class SelectionError(ValueError):
    """A rule is malformed, references unknown data (e.g. a region
    label that isn't on the Structure), or fails JSON validation."""


_RANGE_TOKEN_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$", re.ASCII)


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

    A listed number that names no mode is dropped: nothing can be probed
    for it, and the selection is what the run records.  The script says
    which were dropped -- the mode count is known only once the Hessian is
    done, so nothing before the run could refuse them.
    """
    if selection == "skip":
        return []
    if selection == "explicit":
        return [n for n in explicit if 1 <= n <= len(frequencies_cm1)]
    if selection == "all":
        return [n for n, f in enumerate(frequencies_cm1, start=1)
                if (freq_min_cm1 is None or f >= freq_min_cm1)
                and (freq_max_cm1 is None or f <= freq_max_cm1)]
    raise ValueError(f"es_mode_selection {selection!r}: the selectors are "
                     f"'skip', 'all' and 'explicit'")


def parse_index_list(expression: str, *, first: int = 0,
                     count: Optional[int] = None) -> List[int]:
    """Parse ``"0-35, 100, 150-200"`` into a sorted unique list of indices.

    The one reader of the index-list grammar: comma-separated tokens, each
    ``<int>`` or ``<lo>-<hi>`` (inclusive), whitespace tolerated; an empty
    expression is an empty list.  ``first`` is where the numbering starts --
    0 for atoms (``model/overview.md`` § 2), 1 for vibrational modes
    (``engines/vibration.md`` § 4.8) -- and ``count``, when the number of
    items is known, bounds it above.

    Raises :class:`SelectionError` on a malformed token (a stray comma
    included), a reversed range (``35-0``), or an index outside
    ``[first, first + count)``.
    """
    if not expression or not expression.strip():
        return []
    out: set = set()
    for tok in expression.split(","):
        m = _RANGE_TOKEN_RE.fullmatch(tok)
        if not m:
            raise SelectionError(
                f"invalid token {tok.strip()!r} in {expression!r}; "
                f"expected '<int>' or '<int>-<int>'"
            )
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise SelectionError(f"range {lo}-{hi} has hi < lo")
        if lo < first:
            raise SelectionError(
                f"{lo} is below {first}, where the numbering starts")
        if count is not None and hi >= first + count:
            raise SelectionError(
                f"range {lo}-{hi} out of [{first}, {first + count})")
        out.update(range(lo, hi + 1))
    return sorted(out)


__all__ = ["SelectionError", "parse_index_list", "select_modes"]
