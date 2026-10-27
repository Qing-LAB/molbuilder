"""SIESTA's ``<label>.EPSIMG`` -- the absorption spectrum an optical
calculation writes, read (`model/parse.md` § 5f; `engines/vibration.md`
§ 5.11 § 8).

MODULE  parse.engines.siesta_epsimg_reader (numpy and its siblings; it
        travels beside the job in ``mb_vibration.pyz``,
        `runwrap.VIBRATION_COMPANIONS`)
ROLE    the file's one reader, :func:`read_epsimg`: its header -- the energy
        range, the spin components, the f-sum rule and the Drude plasma
        frequency squared, per spin -- and its rows, the photon energy and
        the imaginary part of the dielectric function per spin
USED-BY the registered parser (`parse.engines.siesta_epsimg`); an optical
        call's record step beside the job (`spectra.siesta_vibration`)

THE FORMAT, as SIESTA 5.4.2 writes it (`Src/optical.F`, on node 0 once every
transition is summed): a header of ``##`` lines, each title followed by its
value line --

    ## Minimum and maximum energy in eV        ## <min> <max>
    ## Number of spin components               ## <nspin>
    ## f-sum rule                              ## <value>      (one spin)
    ## wp^2 in Ha^2 for Drude                  ## <value>

-- or, with two spin components, ``f-sum rule spin 1``, ``f-sum rule spin
2``, ``wp^2 in Ha^2 for Drude, spin 1``, ``..., spin 2``; then one row per
energy, list-directed: the photon energy ħω in eV, then ε₂ for each spin
component, dimensionless.  The energies are ``wmin + (i-1)·dw`` with
``dw = Optical.Broaden / 10`` (`Src/optical.F:287`); the header's range is
printed to four decimals (``f10.4``).

**A file cut short is refused**: SIESTA writes the rows after the header,
so a run stopped while writing leaves a file whose last energy falls short
of the header's maximum -- read whole, it would integrate a spectrum that
ends early, which no reader downstream could tell.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job.
try:                                        # inside molbuilder
    from . import siesta_grammar as _G
    from ..errors import ParseError
except ImportError:                         # beside a job, in mb_vibration.pyz
    import siesta_grammar as _G  # type: ignore[no-redef]
    from errors import ParseError  # type: ignore[no-redef]

#: THE HEADER'S TITLES, as SIESTA prints them (`Src/optical.F`), compared
#: lowercased -- each followed by its value line.
RANGE_TITLE = "minimum and maximum energy in ev"
SPIN_TITLE = "number of spin components"
#: The f-sum rule's and the Drude term's titles: bare with one spin
#: component, ``<title> spin N`` / ``<title>, spin N`` with two.
FSUM_TITLE = "f-sum rule"
DRUDE_TITLE = "wp^2 in ha^2 for drude"

#: THE SPIN CHANNELS a file's columns are, in the SIESTA family's names
#: (`siesta_grammar.TBT_CHANNELS`, the transport data file's `spin`): one
#: column unpolarized; two, up then down (`Src/optical.F`, ``ispin`` 1..2).
_NAMES = tuple(name for name, _tag in _G.TBT_CHANNELS)
CHANNELS: Dict[int, Tuple[str, ...]] = {1: _NAMES[:1], 2: _NAMES[1:]}

#: Half the header's printed decimal (``f10.4``): the rows' first and last
#: energies agree with the header's range within it.
_HEADER_HALF_DECIMAL_EV = 0.5e-4


@dataclass(frozen=True)
class EpsImg:
    """What an ``.EPSIMG`` says, its units as written."""
    path: str
    #: The photon energy ħω of each row, eV, ascending.
    energy_ev: np.ndarray
    #: ε₂ for each spin channel at each energy, ``(spin, energy)``,
    #: dimensionless.
    eps2: np.ndarray
    #: The spin channels the columns are (:data:`CHANNELS`).
    spin: Tuple[str, ...]
    #: The header's energy range, eV, as printed.
    energy_range_ev: Tuple[float, float]
    #: The f-sum rule per spin channel, as SIESTA states it: the integral
    #: of ε₂ against the electrons it should count -- 1 when every
    #: transition is in the range.
    f_sum: Tuple[float, ...]
    #: The Drude term's plasma frequency squared per spin channel, Ha²:
    #: the intraband part SIESTA computes beside ε₂ and leaves out of it.
    drude_wp2_ha2: Tuple[float, ...]


def _number(token: str, *, what: str, name: str) -> float:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise ParseError(f"{name}: {what} is {token!r}, not a number") from None


def read_epsimg(path) -> EpsImg:
    """The ``.EPSIMG`` at ``path``, read whole -- its header, then every
    row, each the photon energy and one ε₂ per spin channel the header
    states.  Refused by name (:class:`ParseError`): a header missing a title
    or a value, a row of another width, energies not ascending, and rows
    that end short of the header's range -- a file cut short."""
    p = Path(path)
    name = p.name
    header: Dict[str, List[str]] = {}
    title = None
    rows: List[List[float]] = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("##"):
                if rows:
                    raise ParseError(f"{name}: line {n}: a header line after "
                                     f"the rows began")
                body = text[2:].strip()
                if title is None:
                    title = body.lower()
                else:
                    header[title] = body.split()
                    title = None
                continue
            if title is not None:
                raise ParseError(f"{name}: the header's '{title}' has no "
                                 f"value line")
            rows.append([_number(t, what=f"line {n}", name=name)
                         for t in text.split()])
    for t in (RANGE_TITLE, SPIN_TITLE):
        if t not in header:
            raise ParseError(f"{name}: the header states no '{t}' -- not a "
                             f"SIESTA optical spectrum (Src/optical.F)")
    n_spin = int(_number(header[SPIN_TITLE][0], what="the spin count",
                         name=name))
    if n_spin not in CHANNELS:
        raise ParseError(f"{name}: {n_spin} spin components; SIESTA's "
                         f"optical spectrum has 1 or 2")
    lo, hi = (_number(t, what="the energy range", name=name)
              for t in header[RANGE_TITLE][:2])

    def per_spin(base: str, sep: str) -> Tuple[float, ...]:
        keys = ([base] if n_spin == 1 else
                [f"{base}{sep} spin {i}" for i in range(1, n_spin + 1)])
        missing = [k for k in keys if k not in header]
        if missing:
            raise ParseError(f"{name}: the header states no "
                             f"{', '.join(repr(k) for k in missing)}")
        return tuple(_number(header[k][0], what=k, name=name) for k in keys)

    f_sum = per_spin(FSUM_TITLE, "")
    drude = per_spin(DRUDE_TITLE, ",")
    if not rows:
        raise ParseError(f"{name}: the header holds no rows below it -- the "
                         f"spectrum was not written")
    widths = {len(r) for r in rows}
    if widths != {1 + n_spin}:
        raise ParseError(
            f"{name}: rows of {sorted(widths)} numbers; the header's "
            f"{n_spin} spin component(s) make each row {1 + n_spin} -- the "
            f"energy, then ε₂ per spin")
    table = np.asarray(rows, dtype=float)
    energy = table[:, 0]
    if energy.size > 1 and not np.all(np.diff(energy) > 0):
        raise ParseError(f"{name}: the energies are not ascending")
    if (abs(energy[0] - lo) > _HEADER_HALF_DECIMAL_EV
            or abs(energy[-1] - hi) > _HEADER_HALF_DECIMAL_EV):
        raise ParseError(
            f"{name}: the rows run from {energy[0]:.4f} to {energy[-1]:.4f} "
            f"eV and the header states {lo:.4f} to {hi:.4f} -- the file was "
            f"cut short (the call stopped while writing it)")
    return EpsImg(path=str(p), energy_ev=energy,
                  eps2=np.ascontiguousarray(table[:, 1:].T),
                  spin=CHANNELS[n_spin], energy_range_ev=(lo, hi),
                  f_sum=f_sum, drude_wp2_ha2=drude)


def is_epsimg(path) -> bool:
    """Whether ``path`` begins as an ``.EPSIMG`` does -- its first line the
    range's title -- read from its first line alone.  SIESTA's
    ``.contrib.EPSIMG`` beside it, the transitions one by one, begins
    otherwise (``# ispin =``)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            first = fh.readline().strip()
    except OSError:
        return False
    return first.startswith("##") and first[2:].strip().lower() == RANGE_TITLE


__all__ = ["EpsImg", "read_epsimg", "is_epsimg", "CHANNELS",
           "RANGE_TITLE", "SPIN_TITLE", "FSUM_TITLE", "DRUDE_TITLE"]
