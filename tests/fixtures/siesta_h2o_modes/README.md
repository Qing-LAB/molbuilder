# Three modes of free water — a SIESTA road result

`H2O.spectra.json` is the spectrum the SIESTA vibration road wrote,
unedited: H₂O with nothing held, PBE/DZP, a 300 Ry mesh, relaxed by the
ladder's `relax` stage and then `FC.Displacement 0.04 Bohr` over all three
atoms (`FC.First 1`, `FC.Last 3`), derived by the job's own finish
(`engines/vibration.md` § 5.5).  Measured 2026-09-28 on this workstation
(SIESTA 5.4.2, two ranks; `claude-validate/spectrum/h2o-siesta-free`):
1574.7, 3634.0 and 3807.0 cm⁻¹, six whole-body motions removed, the
reference forces 0.0008 eV/Å against the 0.01 eV/Å criterion.

Why it is a fixture: three modes over free atoms of **unequal mass** (O and
two H), so a matching that drops the mass weighting, or matches by rank,
gives a different answer on it (`tests/test_displacement_sweep.py`).  The
road's own sweep test runs H₂ with one free atom, whose single mode can
neither swap nor mix, and a water with its oxygen held frees only the two
hydrogens, whose equal masses make the weighting invisible.
