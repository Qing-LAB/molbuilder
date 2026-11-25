"""THE model molecule a vibration test computes with, built as the result's
reader builds a `vibration.Vibration` (`engines/vibration-frames.md` § 1).

Three atoms, three modes at 120, 700 and 1600 cm-1 orthonormal with the
masses, ``sum_A m_A L_k,A . L_l,A = delta_kl`` -- so the closed forms of
`science/vibrational-averaging.md` § 2b.5 and § 2c hold for it -- at the
record's temperature given, the structure it states beside it; with
``held``, a fourth atom held still, every mode zero on it.  Its geometry
carries digits beyond the 10^-6 Å a pair holds, as a relaxation's does, so
a set read back holds frame 0 rounded.  One builder for the mechanics'
tests and the frame sets' (`tests/test_vibration_mechanics.py`,
`tests/test_vibration_frames.py`).

WHY TYPED (`docs/process/testing.md` § 6, the user's choice of 2026-11-24):
the closed forms need modes known on paper, and the writer's and reader's
fast rows stand on the same model; a real computed vibration's road is the
end-to-end's (`tests/test_vibration_mechanics_e2e.py`).
"""
from __future__ import annotations

import numpy as np


def model_vibration(record_temperature_k=300.0, *, held: bool = False):
    from molbuilder import vibration as V
    from molbuilder.structure import Structure
    masses = np.array([12.0, 1.008, 15.999])
    u, _ = np.linalg.qr(np.random.default_rng(20261123).standard_normal(
        (9, 3)))
    rows = u.T.reshape(3, 3, 3) / np.sqrt(masses)[None, :, None]
    positions = [[0.0, 0.0, 0.0], [1.0912345678, 0.0, 0.0],
                 [0.0, 1.2034567891, 0.0]]
    xyz = "".join(f"{el} {x!r} {y!r} {z!r}\n" for el, (x, y, z)
                  in zip("CHO", positions))
    elements = ["C", "H", "O"]
    if held:
        masses = np.append(masses, 1.008)
        xyz += "H 3 3 3\n"
        positions.append([3.0, 3.0, 3.0])
        elements.append("H")
    modes = tuple(V.Mode(index_1based=k + 1, frequency_cm1=nu,
                         has_imag=False,
                         eigenvector_canonical_per_amu12=rows[k])
                  for k, nu in enumerate((120.0, 700.0, 1600.0)))
    n = len(positions)
    return V.Vibration(
        modes=modes, free_atom_idxs=(0, 1, 2), masses_amu=masses,
        element_symbols=tuple(elements), positions_ang=np.array(positions),
        record_temperature_k=record_temperature_k,
        structure=Structure.from_xyz(f"{n}\nmodel\n{xyz}"))
