# Post-processing a vibration

**Role:** guide
**Domain:** engines
**Companions:** [`engines/vibration.md`](?doc=engines/vibration.md) — the
calculation these results come from, and its result file's contract (§ 6);
[`model/datafiles.md`](?doc=model/datafiles.md) — the data files these calls
read; [`web/results.md`](?doc=web/results.md) § 1.1 — the Export that hands
them over;
[`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md)
and [`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md)
— the same for an optimization and a transport calculation.

What a finished vibration holds, which call answers what, and worked examples in
Python: the modes atom by atom, the infrared and the Raman, the thermochemistry,
the charges and the optical calls. Every example below, save § 2a's (whose
tests § 2a names), ran on a real calculation before it was written here: on
SIESTA, a molecule of 4 free atoms held between 48 gold atoms, whose engine
order differs from its user order; on PySCF, an H2, for what § 2, § 3, § 4 and
§ 7 say of that engine.

## 1. One import

```python
from molbuilder.postprocess import (read_spectra_data, read_optical_data,
                                    infrared_view, raman_view, outermost_move,
                                    read_steps, atom_index_map, vibration)
import xarray as xr
```

| name | answers |
|---|---|
| `read_spectra_data(path)` | `<label>.spectra.nc`: the whole result as data, and its atom index map |
| `vibration(path)` | the result's modes to compute with: each atom's move for a coordinate and back, a mode's states and their frames, the mode at another frequency (§ 2a) |
| `infrared_view(path, direction=, partition=)` | the infrared the spectrum viewer draws |
| `raman_view(path, laser_nm=, broaden_ev=)` | a SIESTA vibration's Raman at a laser |
| `read_optical_data(path)` | `<label>.optical.nc`: the optical calls, each frame's ε₂ |
| `outermost_move(path, mode=, temperature_k=)` | how far a frame set at that temperature moves the atom moving most |
| `read_steps(path)` | the relaxation's or the force constants' steps, in your atoms' order (§ 7) |

The result sits in the run's folder: `01_freq/run-0/<label>.spectra.nc`, with its
JSON beside it. The Results tab's Export, in its top line
([`web/results.md`](?doc=web/results.md) § 1.1), hands over, as *NetCDF*,
the same files as one zip — `<label>.spectra.nc`, the optical calls'
`<label>.optical.nc` where the run made them, and, with a laser applied in
the Raman view, `<label>.raman.nc`, the view at that laser, read with
`postprocess.read_raman_data` — and `<label>.steps.nc`, its runs' every step
and each step's SCF cycles (its relaxation, its force constants, its optical
calls; `postprocess.read_steps_data`,
[`model/datafiles.md`](?doc=model/datafiles.md) § 9.5) — and, as *CSV*, their
tables. **Every row of the result is in your atoms' order**, listed by user
index, each atom's engine index beside it.

## 2. The modes, atom by atom

```python
data = read_spectra_data("01_freq/run-0/v-molecule.spectra.nc")
data["atom_index_map"].listed_by         # 'user_index'
elements = data["structure"]["elements"] # every atom, your order
free = data["free_atom_idxs"]            # the atoms the vibration moved, your index
masses = data["equilibrium"]["masses_amu"]

mode = data["modes"][0]
mode["index_1based"], mode["frequency_cm1"], mode["has_imag"]
mode["ir_intensity_km_mol"], mode["raman_activity_a4_amu"]
```

**A mode's rows are the free atoms, in `free`'s order**: row `k` is atom `free[k]`,
and a held atom has none.

```python
import numpy as np
vector = np.asarray(mode["eigenvector_canonical"])   # (free atom, xyz), amu^-1/2
of_atom = dict(zip(free, vector))                    # your atom's row, by its index
```

The Cartesian eigenvector is normalised with each atom's mass — the sum over
atoms of m|L|² is 1, m in amu — so two modes are orthogonal with each atom's
mass, which is a check that every row stands on its atom:

```python
E = np.array([np.asarray(m["eigenvector_canonical"]).ravel() for m in data["modes"]])
w = np.repeat(np.asarray(masses)[free], 3)
np.round((E * w) @ E.T, 6)               # the identity
```

`mode["eigenvector_display"]` is the same vector rescaled so its largest component
is 1: what a viewer draws, with no meaning in its size. Intensities and amplitudes
pair with the canonical one ([`engines/vibration.md`](?doc=engines/vibration.md)
§ 6).

The same, as labelled arrays — the file states every variable's unit and meaning:

```python
ds = xr.load_dataset("01_freq/run-0/v-molecule.spectra.nc", group="modes")
ds.eigenvector_canonical_per_amu12.sel(mode=7, free_atom=free[0])   # the result's eigenvector_canonical, by its name in the file
ds.frequency_cm1.attrs["units"], ds.frequency_cm1.attrs["long_name"]
```

## 2a. Each mode's mechanics — `vibration`

`vibration(path)` reads the result — its JSON or its `.nc`, one answer — as
modes to compute with, every atom in your order, the theory
[`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2b ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.11 § 6). The
end-to-end test (`tests/test_vibration_mechanics_e2e.py`) makes the move, the
projection back, the zero-point state and the shares on a real PySCF H₂; the
rest are `tests/test_vibration_mechanics.py`'s, on the science's own modes:

```python
v = vibration("01_freq/run-0/v-h2.spectra.nc")    # or the .spectra.json
m = v.modes[0]                                    # mode 1
m.frequency_cm1, m.zero_point_amplitude_amu12_ang # Q_zp = sqrt(hbar / 2 omega)
m.element_shares                                  # {'H': 1.0}

u = m.displacement(zero_point_widths=1.0)         # each atom's move at Q_zp, Å (x, y, z)
m.coordinate(u)                                   # back: sum_A m_A L_A . u_A = Q_zp
m.displaced(0.05)                                 # every atom at R + Q L, Q = 0.05 amu^1/2.Å
m.with_frequency(0.96 * m.frequency_cm1)          # the same pattern at a scaled frequency
```

**Where the molecule sits along the mode** is a state — at a temperature,
at a mean number of quanta, with `quanta` exactly, a steady swing of
`swing_quanta` over a bath, or the classical limits (`classical=True`):

```python
s = m.state(temperature_k=0.0)        # the zero-point motion: spread Q_zp
s.spread_amu12_ang, s.mean_quanta
s.atom_rms_ang                        # each atom's r.m.s. move, Å
s.pdf(0.0), s.cdf(0.05), s.moment(4)  # density, distribution, <Q^4>
f = m.state(quanta=1).frames(nodes=3) # three frames, exact to degree 5
f.nodes_amu12_ang, f.weights          # where along the mode, and each one's weight
f.positions_ang                       # each frame's geometry, every atom at R + Q L
```

A thermal or classical state, or a swing's bath, given neither a temperature
nor mean quanta takes the one the result's thermochemistry states, and says so —
`s.temperature_k`, `s.temperature_from` (`"given"` or `"the record's"`); a
result stating none asks for one. `quanta`, `mean_quanta` and a classical swing
alone take no temperature. An imaginary mode has
moves for a coordinate given and no state.

## 3. The infrared and the Raman

```python
ir = infrared_view("01_freq/run-0/v-molecule.spectra.json", direction="z")
ir.source, ir.partition, ir.atoms_label
ir.modes[0]

raman = raman_view("01_freq/run-0/v-molecule.spectra.json", laser_nm=532)
raman.photon_energy_ev, raman.broaden_ev, raman.derived_from
raman.modes[0]
```

**The infrared differs by engine.** A SIESTA result's is the free atoms', from their
charges (`source` `charges`, `partition` `hirshfeld` or `voronoi`), and says why
where the run computed none. A PySCF result's is the whole system's, from the dipole
change it stored (`source` `dipole`, no partition). `raman_view` is a SIESTA
vibration's, from its optical calls.

## 4. The thermochemistry and the orbitals

```python
thermo = data["thermo"]
thermo["zpe_eh"], thermo["g_eh"], thermo["temperature_K"], thermo["regime"]
thermo["grid"]["temperatures_K"][:3], thermo["grid"]["g_eh"][:3]
data["equilibrium"]["scf_energy_eh"]
```

All in hartree, as the file's units say. `data["equilibrium"]` holds
`scf_energy_eh`, `mo_energies_eh` and `homo_idx` where the engine reported them: a
PySCF run's. On SIESTA all three are `None`, and the relaxation's energy is
`read_steps`' (§ 7).

## 5. The charges, on your atoms

A SIESTA force-constant run prints each atom's charge at every displacement, where
the deck asked for them, *Compute IR* on. The file then holds this group; a PySCF
result, and a SIESTA one with *Compute IR* off, hold none:

```python
pc = xr.load_dataset("01_freq/run-0/v-molecule.spectra.nc", group="partial_charges")
pc.attrs["atoms_listed_by"]              # 'user_index'
pc.hirshfeld_charge_e.sel(fc_step=0, atom=0)   # your atom 0, at the reference
int(pc.displaced_atom.sel(fc_step=1)), str(pc.displaced_axis.sel(fc_step=1).values)
```

## 6. The optical calls

```python
optical = read_optical_data("01_freq/run-0/v-molecule.optical.nc")
optical["photon_energy_ev"][:3], optical["spin"]
call = optical["frames"][1]
call["mode"], call["side"], call["dq_amu12_ang"], call["fermi_level_ev"]
reference = np.asarray(optical["frames"][0]["position_ang"])   # every atom, your order
moved = np.asarray(call["position_ang"]) - reference           # side × dq × the eigenvector
```

## 7. The relaxation and the force constants

A SIESTA vibration runs each phase as a SIESTA call of its own — a point of the
one run, in its own folder (`relax`, `fc`) — on a copy sorted held atoms first,
so **their files are in the engine's order**. `read_steps` hands back your order:

```python
relax = read_steps("01_freq/run-0/relax")
relax.atom_index_map.listed_by, relax.elements[:4], relax.held_atoms[:4]
relax.positions_ang[-1, 0]                # your atom 0, Å
relax.relaxation["converged"]             # its largest force within its tolerance
fc = read_steps("01_freq/run-0/fc")       # one step per displacement
```

A PySCF vibration is one run, whose folder's result is the spectrum; its folder
gives the relaxation's steps from the log beside it, as naming the log does:

```python
relax = read_steps("01_freq/run-0")       # or "01_freq/run-0/J_01_freq.molwatch.log"
```

What a step is, the force figures, the energy and where the coordinates stand are
[`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md) § 3.

## 8. Atom indices

Every atom index here counts from **0**; a mode's number from 1, as the file's
`mode` axis says. `atom_index_map(path)` answers any file, and
[`model/overview.md`](?doc=model/overview.md) § 2.2b states the rule.
