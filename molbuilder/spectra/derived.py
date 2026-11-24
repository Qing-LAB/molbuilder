"""A mode's DERIVED values -- one derivation, for every writer of a result
file and for the reader's check (`engines/vibration.md` § 6.6, § 6.11 § 3,
I16).

MODULE  spectra.derived (L2 by the index, `architecture.md`; it imports
        the activity classes, the mode's mechanics (`vibration`) and the
        modes' temperatures (`spectra.normal_modes`), numpy -- nothing that
        needs the package installed)
ROLE    from what a mode STATES -- its frequency, whether it is imaginary, its
        canonical eigenvector and its two strengths -- and the masses the
        modes were weighted by, the values a result states derived from them:
        the activity classes (`spectra.activity`, a decision against the
        whole run), the zero-point amplitude and every free atom's
        displacement at it (v7); each free atom's share of the motion, the
        largest zero-point move, the reduced mass and force constant, and
        the block of each mode's spread and mean quanta at the modes' own
        temperatures (v15) -- each by the mode's mechanics' one home
        (`vibration`), composed there; this module states them
USED-BY the one writer of a result file, every engine's
        (`sidecars.spectra.write_spectra_payload`); `SpectraResults.to_dict`;
        the reader's check (`SpectraResults.from_dict`)
TRAVELS in both bundles -- the PySCF script's and a SIESTA vibration's
        (`runwrap.members`) -- so it imports its siblings two ways and
        nothing else of molbuilder

**Written beside what they come from, and checked on read** (user,
2026-10-10: "explicit is always better than implicit in data science"): a
result file states every derived value, so a person reads it without
molbuilder, and a reader computes each again with this module and refuses a
file whose stored value disagrees -- a file never carries a number its own
inputs contradict.  The activity class is a decision, so its rule
(`spectra.activity`) is part of the file's definition: changing it is a new
schema version.  The modes' temperatures are a choice too, and the file's
own: the writer states them, the reader checks each row at the temperatures
the file states and never rebuilds the list (§ 6.6), so a later list -- the
thermochemistry's grid is a display choice -- changes new files only; and
the check's tolerance stands above what a constant's last digits move, so a
later correction of the constants changes new files only too (§ 6.7).
"""
from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from ..vibration import (THERMAL_ROW_KEYS, force_constant_mdyn_per_ang,
                             largest_zero_point_move_ang, motion_shares_of,
                             reduced_mass_amu, thermal_row,
                             zero_point_amplitude_amu12_ang,
                             zero_point_displacement_ang)
    from .activity import classify_modes
    from .normal_modes import mode_temperatures
except ImportError:                         # beside a job, in a bundle
    from vibration import (  # type: ignore[no-redef]
        THERMAL_ROW_KEYS, force_constant_mdyn_per_ang,
        largest_zero_point_move_ang, motion_shares_of, reduced_mass_amu,
        thermal_row, zero_point_amplitude_amu12_ang,
        zero_point_displacement_ang)
    from activity import classify_modes  # type: ignore[no-redef]
    from normal_modes import mode_temperatures  # type: ignore[no-redef]

#: THE KEYS A MODE'S ROW CARRIES THAT ARE DERIVED from its other keys
#: (`engines/vibration.md` § 6.3), by the schema version that first states
#: them: what every writer states and the reader checks.
DERIVED_MODE_KEYS_V7 = ("ir_active", "raman_active", "activity_class",
                        "zero_point_amplitude_amu12_ang",
                        "zero_point_displacement_ang")
DERIVED_MODE_KEYS_V15 = ("motion_share", "largest_zero_point_move_ang",
                         "reduced_mass_amu", "force_constant_mdyn_per_ang")
DERIVED_MODE_KEYS = DERIVED_MODE_KEYS_V7 + DERIVED_MODE_KEYS_V15

#: THE RESULT'S BLOCK OF EACH MODE'S SPREAD AND MEAN QUANTA (v15,
#: `engines/vibration.md` § 6.2), its keys a mode's thermal row's
#: (`vibration.THERMAL_ROW_KEYS`: ``temperature_k``,
#: ``thermal_spread_amu12_ang``, ``mean_quanta``).
THERMAL_KEY = "modes_thermal"

#: How far a stored derived number may be from its derivation, relative
#: (`engines/vibration.md` § 6.7): far below any physical meaning, so a hand
#: edit, a wrong formula or a wrong unit is caught; far above what a
#: constant's last digits move -- a mode's mean quanta in the cold moves by
#: ``x = h c nu / k_B T`` times a constant's relative change, ``x`` up to
#: 745 -- or another machine's rounding, so a later correction of the
#: constants changes new files only.
DERIVED_RTOL = 1e-6
#: ...and the distance under which two numbers are the same whatever their
#: ratio: the smallest normal double.  Below it a number's last bit is no
#: longer a small part of it, and ``exp`` is not rounded alike on every
#: machine -- the job's node writes a mode's mean quanta in the cold, the
#: server reads it (§ 6.7).  No physical derived value comes near it.
DERIVED_FLOOR = sys.float_info.min


def mode_derivations(modes: Sequence[Mapping[str, Any]],
                     masses_free_amu: Optional[Sequence[float]]
                     ) -> List[Dict[str, Any]]:
    """Each mode's derived values, ``{key: value}`` over
    :data:`DERIVED_MODE_KEYS`, from the mode rows as a result file states
    them (``frequency_cm1``, ``has_imag``, ``eigenvector_canonical``,
    ``ir_intensity_km_mol``, ``raman_activity_a4_amu``) and the free atoms'
    masses in the free list's order (``None`` where a file states none: no
    shares), in mode order."""
    if not modes:
        return []
    flags = classify_modes([m.get("ir_intensity_km_mol") for m in modes],
                           [m.get("raman_activity_a4_amu") for m in modes])
    out: List[Dict[str, Any]] = []
    for m, flag in zip(modes, flags):
        frequency, imaginary = m["frequency_cm1"], bool(m.get("has_imag"))
        q = zero_point_amplitude_amu12_ang(frequency, imaginary)
        rows = m.get("eigenvector_canonical")
        mu = None if rows is None else reduced_mass_amu(rows)
        moves = (None if rows is None else
                 zero_point_displacement_ang(frequency, imaginary, rows))
        out.append({
            **flag,
            "zero_point_amplitude_amu12_ang": q,
            "zero_point_displacement_ang": (
                None if moves is None else moves.tolist()),
            "motion_share": (
                None if rows is None or masses_free_amu is None else
                motion_shares_of(rows, masses_free_amu).tolist()),
            "largest_zero_point_move_ang": (
                None if rows is None else
                largest_zero_point_move_ang(frequency, imaginary, rows)),
            "reduced_mass_amu": mu,
            "force_constant_mdyn_per_ang": force_constant_mdyn_per_ang(
                frequency, mu),
        })
    return out


def thermal_block(modes: Sequence[Mapping[str, Any]],
                  temperatures_k: Sequence[float]
                  ) -> Optional[Dict[str, Any]]:
    """The result's ``modes_thermal`` (`engines/vibration.md` § 6.2): the
    temperatures ``temperatures_k`` and each mode's spread and mean quanta
    at each, in mode order -- an imaginary mode's ``None``; ``None`` where
    the result has no modes yet, stated so from the run's first write
    (§ 6.5)."""
    if not modes:
        return None
    temps = [float(t) for t in temperatures_k]
    rows = [thermal_row(m["frequency_cm1"], bool(m.get("has_imag")), temps)
            for m in modes]
    return dict(zip(THERMAL_ROW_KEYS,
                    (temps, [spreads for spreads, _ in rows],
                     [quanta for _, quanta in rows])))


def stated_temperatures(block: Any) -> Optional[Any]:
    """The modes' temperatures a result's ``modes_thermal`` states -- its
    ``temperature_k`` as written, ``None`` where it states none (no block,
    or a block without them) -- for the writer and the reader alike.  A
    block that is not one, or states a key outside
    :data:`THERMAL_ROW_KEYS`, is refused by name: the reader's rule
    (`engines/vibration.md` § 6.7), a file states its block whole."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{THERMAL_KEY} is a block of "
                         f"{list(THERMAL_ROW_KEYS)}; got {block!r}")
    unknown = sorted(set(block) - set(THERMAL_ROW_KEYS))
    if unknown:
        raise ValueError(f"{THERMAL_KEY} states {unknown}, none of its keys "
                         f"{list(THERMAL_ROW_KEYS)}")
    return block.get(THERMAL_ROW_KEYS[0])


def with_derived(modes: Sequence[Mapping[str, Any]],
                 masses_free_amu: Optional[Sequence[float]]
                 ) -> List[Dict[str, Any]]:
    """The mode rows, each with its derived values stated beside the rest --
    what a writer writes; ``masses_free_amu`` as :func:`mode_derivations`
    takes them."""
    return [{**dict(m), **d}
            for m, d in zip(modes, mode_derivations(modes, masses_free_amu))]


def free_masses_of(payload: Mapping[str, Any]) -> List[float]:
    """The free atoms' masses a result's payload states -- its
    ``equilibrium.masses_amu`` at its ``free_atom_idxs`` -- refused by name
    where the payload states none, or none for a free atom: the modes were
    weighted by masses a result states beside its geometry, every atom's
    (§ 6.2, v7)."""
    masses = (payload.get("equilibrium") or {}).get("masses_amu")
    if masses is None:
        raise ValueError("the result has modes and states no masses "
                         "(equilibrium.masses_amu): its derived values need "
                         "the masses the modes were weighted by")
    free = [int(i) for i in payload.get("free_atom_idxs") or ()]
    if any(not 0 <= i < len(masses) for i in free):
        raise ValueError(
            f"the result states {len(masses)} masses "
            f"(equilibrium.masses_amu) and a free atom at index {max(free)}: "
            f"every atom's mass is stated, in the user's order")
    return [float(masses[i]) for i in free]


def payload_with_derived(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """A result's payload with every derived value stated -- each mode's, and
    the ``modes_thermal`` block at the modes' temperatures (the payload's
    own where it states them, :func:`stated_temperatures`, else today's,
    `normal_modes.mode_temperatures`), ``None`` before the modes exist --
    what the one writer writes, whichever engine built the payload."""
    modes = payload.get("modes") or []
    if not modes:
        return {**payload, THERMAL_KEY: None}
    temperatures = mode_temperatures(
        (payload.get("thermo") or {}).get("temperature_K"),
        stated_temperatures(payload.get(THERMAL_KEY)))
    return {**payload,
            "modes": with_derived(modes, free_masses_of(payload)),
            THERMAL_KEY: thermal_block(modes, temperatures)}


def disagreements(payload: Mapping[str, Any], version: int,
                  masses_free_amu: Optional[Sequence[float]]) -> List[str]:
    """What a file's stored derived values say that their derivation does
    not -- one sentence per mode and key, a key not stated among them, and
    from v15 the ``modes_thermal`` block, each row checked at the
    temperatures the file states (the list is the file's own, never
    rebuilt); ``[]`` when every stored value is the derivation's.
    ``version`` the file's schema version: a key is checked from the version
    that first states it; ``masses_free_amu`` the free atoms' masses the
    file states (:func:`free_masses_of`), ``None`` where it states none."""
    keys = (DERIVED_MODE_KEYS if int(version) >= 15
            else DERIVED_MODE_KEYS_V7)
    modes = payload.get("modes") or []
    out: List[str] = []
    for m, d in zip(modes, mode_derivations(modes, masses_free_amu)):
        for key in keys:
            if key not in m:
                out.append(f"mode {m.get('index_1based')}: {key} is not "
                           f"stated")
                continue
            found = _first_difference(m[key], d[key])
            if found is not None:
                at, stated, derived = found
                where = "".join(f"[{i}]" for i in at)
                out.append(f"mode {m.get('index_1based')}: {key}{where} is "
                           f"{stated!r}, its derivation {derived!r}")
    if int(version) >= 15 and modes:
        block = payload.get(THERMAL_KEY)
        if block is None:
            return out + [f"{THERMAL_KEY} is not stated"]
        try:
            stated_k = stated_temperatures(block)
            missing = [k for k in THERMAL_ROW_KEYS if block.get(k) is None]
            if missing:
                return out + [f"{THERMAL_KEY}.{k} is not stated"
                              for k in missing]
            # AT THE FILE'S OWN TEMPERATURES (§ 6.6), never rebuilt.
            temps = mode_temperatures(None, stated_k)
        except ValueError as e:
            return out + [str(e)]
        derived = thermal_block(modes, temps)
        for key in THERMAL_ROW_KEYS[1:]:
            found = _first_difference(block[key], derived[key])
            if found is None:
                continue
            at, stated, value = found
            said = f"{THERMAL_KEY}.{key}"
            if len(at) >= 1:
                said += f" of mode {modes[at[0]].get('index_1based')}"
            if len(at) >= 2:
                said += f" at {temps[at[1]]:g} K"
            out.append(f"{said} is {stated!r}, its derivation {value!r}"
                       if len(at) >= 2 else f"{said} is not its derivation")
    return out


def _first_difference(a: Any, b: Any, at: tuple = ()):
    """Where ``a`` and ``b`` first differ (:func:`_same`, list by list) --
    ``(indices, a's value, b's value)`` -- or ``None`` where they are the
    same: what a refusal names, not two whole lists."""
    if (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)):
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, at + (i,))
            if found is not None:
                return found
        return None
    return None if _same(a, b, DERIVED_RTOL) else (at, a, b)


def _same(a: Any, b: Any, rtol: float) -> bool:
    """``a`` and ``b`` the same value: equal when not numbers (``None``,
    true/false, text), within ``rtol`` when numbers -- or closer than
    :data:`DERIVED_FLOOR` -- element by element when lists."""
    if isinstance(a, list) or isinstance(b, list):
        return (isinstance(a, list) and isinstance(b, list)
                and len(a) == len(b)
                and all(_same(x, y, rtol) for x, y in zip(a, b)))
    if (isinstance(a, bool) or isinstance(b, bool) or a is None or b is None
            or not isinstance(a, (int, float))
            or not isinstance(b, (int, float))):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=rtol,
                        abs_tol=DERIVED_FLOOR)


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.spectra.activity",
                "molbuilder.vibration",
                "molbuilder.spectra.normal_modes")
