# Post-processing a vibration

What a finished vibration holds, which call answers what, and worked examples in
Python: the modes atom by atom, the infrared and the Raman, the thermochemistry,
the charges and the optical calls. Every example below ran on a real calculation
before it was written here: on SIESTA, a molecule of 4 free atoms held between 48
gold atoms, whose engine order differs from its user order; on PySCF, an H2, for
what § 2, § 3, § 4 and § 7 say of that engine.

## 1. One import

```python
from molbuilder.postprocess import (read_spectra_data, read_optical_data,
                                    infrared_view, raman_view, outermost_move,
                                    read_steps, atom_index_map)
import xarray as xr
```

| name | answers |
|---|---|
| `read_spectra_data(path)` | `<label>.spectra.nc`: the whole result as data, and its atom index map |
| `infrared_view(path, direction=, partition=)` | the infrared the spectrum viewer draws |
| `raman_view(path, laser_nm=, broaden_ev=)` | a SIESTA vibration's Raman at a laser |
| `read_optical_data(path)` | `<label>.optical.nc`: the optical calls, each frame's ε₂ |
| `outermost_move(path, mode=, temperature_k=)` | how far a frame set at that temperature moves the atom moving most |
| `read_steps(path)` | the relaxation's or the force constants' steps, in your atoms' order (§ 7) |

The result sits in the run's folder: `01_freq/run-0/<label>.spectra.nc`, with its
JSON beside it. **Every row of the result is in your atoms' order**, listed by user
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

The Cartesian eigenvector is mass-weighted, so two modes are orthogonal with each
atom's mass, which is a check that every row stands on its atom:

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
ds.eigenvector_canonical.sel(mode=7, free_atom=free[0])
ds.frequency_cm1.attrs["units"], ds.frequency_cm1.attrs["long_name"]
```

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
PySCF run's. On SIESTA they are `None` and empty, and the relaxation's energy is
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

A SIESTA vibration runs its phases as runs of their own, its held atoms first, so
**their files are in the engine's order**. `read_steps` hands back your order:

```python
relax = read_steps("01_freq/run-0/relax")
relax.atom_index_map.listed_by, relax.elements[:4], relax.held_atoms[:4]
relax.positions_ang[-1, 0]                # your atom 0, Å
relax.relaxation["converged"]             # its largest force within its tolerance
fc = read_steps("01_freq/run-0/fc")       # one step per displacement
```

A PySCF vibration is one run, whose folder's result is the spectrum, so name the log
holding its relaxation:

```python
relax = read_steps("01_freq/run-0/J_01_freq.molwatch.log")
```

What a step is, the force figures, the energy and where the coordinates stand are
[`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md) § 3.

## 8. Atom indices

Every atom index here counts from **0**; a mode's number from 1, as the file's
`mode` axis says. `atom_index_map(path)` answers any file, and
[`model/overview.md`](?doc=model/overview.md) § 2.2b states the rule.
