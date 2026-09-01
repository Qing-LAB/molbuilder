"""``max_orbital_rc_ang`` — the basis reach an ``.ion`` file records.

The principal-layer gate (`transport/compose.py`, contract:
`archive/2026-09-01-transport-design.md` § 3) needs each element's largest orbital
cutoff radius.  That number is never guessed: SIESTA writes it into the
``<El>.ion`` file beside every run — each ``#orbital`` header is
followed by a ``npts, delta, cutoff`` line, cutoff in Bohr.  KB
projectors have their own section and are NOT orbitals; anchoring on
the ``#orbital`` marker keeps them out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from molbuilder.constants import BOHR_ANGSTROM as _BOHR_TO_ANG


def max_orbital_rc_ang(path) -> Optional[float]:
    """The largest orbital cutoff radius in *path*, in Å — or ``None``
    when the file is missing or holds no parseable orbital block."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    best: Optional[float] = None
    expect_data = False
    for line in lines:
        if expect_data:
            parts = line.split()
            try:
                rc = float(parts[2])
            except (IndexError, ValueError):
                pass            # not the data line after all
            else:
                if best is None or rc > best:
                    best = rc
        # Asked on every line, never skipped by the branch above: two
        # headers in a row (a block with no data line) must still arm
        # the second one rather than swallow it.
        expect_data = "#orbital" in line
    return None if best is None else best * _BOHR_TO_ANG
