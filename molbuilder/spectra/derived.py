"""A mode's DERIVED values -- one derivation, for every writer of a result
file and for the reader's check (`engines/vibration.md` § 6.6, I16).

MODULE  spectra.derived (L1; the standard library and its siblings --
        nothing that needs the package installed)
ROLE    from what a mode STATES -- its frequency, whether it is imaginary, its
        canonical eigenvector and its two strengths -- the values derived from
        them: the activity classes (`spectra.activity`, a decision against the
        whole run), the zero-point amplitude ``Q_zp = sqrt(hbar / 2 omega)``
        and every free atom's displacement at it, ``Q_zp * L_canonical``; and
        a mode's thermal spread at a temperature,
        ``sigma = Q_zp * sqrt(coth(hbar omega / 2 k_B T))``
        (`science/vibrational-averaging.md` § 2)
USED-BY the one writer of a result file, every engine's
        (`sidecars.spectra.write_spectra_payload`); `SpectraResults.to_dict`;
        the reader's check (`SpectraResults.from_dict`); a mode's frame set's
        check (`frameset`)
TRAVELS in both bundles -- `runwrap.PYSCF_COMPANIONS` and
        `VIBRATION_COMPANIONS` -- so it imports its siblings two ways and
        nothing else of molbuilder

**Written beside what they come from, and checked on read** (user,
2026-10-10: "explicit is always better than implicit in data science"): a
result file states every derived value, so a person reads it without
molbuilder, and a reader computes each again with this module and refuses a
file whose stored value disagrees -- a file never carries a number its own
inputs contradict.  The activity class is a decision, so its rule
(`spectra.activity`) is part of the file's definition: changing it is a new
schema version.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from ..constants import CM1_KELVIN, ZERO_POINT_Q2_AMU_ANG2_CM1
    from .activity import classify_modes
except ImportError:                         # beside a job, in a bundle
    from constants import CM1_KELVIN, ZERO_POINT_Q2_AMU_ANG2_CM1
    from activity import classify_modes

#: THE KEYS A MODE'S ROW CARRIES THAT ARE DERIVED from its other keys
#: (`engines/vibration.md` § 6.3) -- what every writer states and the reader
#: checks.
DERIVED_MODE_KEYS = ("ir_active", "raman_active", "activity_class",
                     "zero_point_amplitude_amu12_ang",
                     "zero_point_displacement_ang")

#: How far a stored derived number may be from its derivation, relative
#: (`engines/vibration.md` § 6.7): a number written at full precision and
#: read back is the derivation's own to the last bits.
DERIVED_RTOL = 1e-9


def zero_point_amplitude_amu12_ang(frequency_cm1: float,
                                   has_imag: bool) -> Optional[float]:
    """``Q_zp = sqrt(hbar / 2 omega)`` in amu^1/2.A,
    ``sqrt(ZERO_POINT_Q2_AMU_ANG2_CM1 / nu)`` -- or ``None`` for an imaginary
    mode, which has no amplitude."""
    nu = float(frequency_cm1)
    if has_imag or not nu > 0.0:
        return None
    return math.sqrt(ZERO_POINT_Q2_AMU_ANG2_CM1 / nu)


def thermal_spread_amu12_ang(zero_point_amplitude_amu12_ang: float,
                             frequency_cm1: float,
                             temperature_k: float) -> float:
    """The spread of a harmonic mode's coordinate at ``temperature_k``,
    ``sigma = Q_zp * sqrt(coth(hbar omega / 2 k_B T))`` in amu^1/2.A --
    exactly the Gaussian's width at every temperature, ``Q_zp`` at 0 K, its
    limit (`science/vibrational-averaging.md` § 2).  ``hbar omega / k_B T``
    is ``nu * CM1_KELVIN / T``."""
    if float(temperature_k) == 0.0:
        return float(zero_point_amplitude_amu12_ang)
    x = float(frequency_cm1) * CM1_KELVIN / (2.0 * float(temperature_k))
    return float(zero_point_amplitude_amu12_ang) * math.sqrt(
        1.0 / math.tanh(x))


def mode_derivations(modes: Sequence[Mapping[str, Any]]
                     ) -> List[Dict[str, Any]]:
    """Each mode's derived values, ``{key: value}`` over
    :data:`DERIVED_MODE_KEYS`, from the mode rows as a result file states
    them (``frequency_cm1``, ``has_imag``, ``eigenvector_canonical``,
    ``ir_intensity_km_mol``, ``raman_activity_a4_amu``), in mode order."""
    if not modes:
        return []
    flags = classify_modes([m.get("ir_intensity_km_mol") for m in modes],
                           [m.get("raman_activity_a4_amu") for m in modes])
    out: List[Dict[str, Any]] = []
    for m, flag in zip(modes, flags):
        q = zero_point_amplitude_amu12_ang(m["frequency_cm1"],
                                           bool(m.get("has_imag")))
        rows = m.get("eigenvector_canonical")
        out.append({
            **flag,
            "zero_point_amplitude_amu12_ang": q,
            "zero_point_displacement_ang": (
                None if q is None or rows is None else
                [[None if x is None else q * float(x) for x in row]
                 for row in rows]),
        })
    return out


def with_derived(modes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The mode rows, each with its derived values stated beside the rest --
    what a writer writes."""
    return [{**dict(m), **d} for m, d in zip(modes, mode_derivations(modes))]


def disagreements(modes: Sequence[Mapping[str, Any]], *,
                  rtol: float = DERIVED_RTOL) -> List[str]:
    """What a file's stored derived values say that their derivation does
    not -- one sentence per mode and key, a key not stated among them; ``[]``
    when every stored value is the derivation's."""
    out: List[str] = []
    for m, d in zip(modes, mode_derivations(modes)):
        for key in DERIVED_MODE_KEYS:
            if key not in m:
                out.append(f"mode {m.get('index_1based')}: {key} is not "
                           f"stated")
            elif not _same(m[key], d[key], rtol):
                out.append(f"mode {m.get('index_1based')}: {key} is "
                           f"{m[key]!r}, its derivation {d[key]!r}")
    return out


def _same(a: Any, b: Any, rtol: float) -> bool:
    """``a`` and ``b`` the same value: equal when not numbers (``None``,
    true/false, text), within ``rtol`` when numbers, element by element when
    lists."""
    if isinstance(a, list) or isinstance(b, list):
        return (isinstance(a, list) and isinstance(b, list)
                and len(a) == len(b)
                and all(_same(x, y, rtol) for x, y in zip(a, b)))
    if (isinstance(a, bool) or isinstance(b, bool) or a is None or b is None
            or not isinstance(a, (int, float))
            or not isinstance(b, (int, float))):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=0.0)
