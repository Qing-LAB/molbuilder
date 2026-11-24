# Post-processing a vibration

**Role:** guide
**Domain:** engines
**Companions:** [`engines/vibration.md`](?doc=engines/vibration.md) — the
calculation these results come from, and its result file's contract (§ 6);
[`model/datafiles.md`](?doc=model/datafiles.md) — the data files these calls
read; [`web/results.md`](?doc=web/results.md) § 1.1 — the Export that hands
them over; [`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)
— the motion toolkit, frames from the modes (§ 2b);
[`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md)
and [`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md)
— the same for an optimization and a transport calculation.

What a finished vibration holds, which call answers what, and worked examples in
Python: the modes atom by atom, the infrared and the Raman, the thermochemistry,
the charges and the optical calls. Every example below, save § 2a's (whose
tests § 2a names) and § 2b's (to build), ran on a real calculation before it
was written here: on
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
| `vibration(path)` | the result's modes to compute with: the values the record stores of each — the zero-point amplitude and displacement, the largest zero-point move, each free atom's share of the motion and its sums by element, label or region, Gaussian's reduced mass and force constant, the spread and mean quanta at the file's temperatures — each atom's move for a coordinate and back, a mode's states and their frames, the mode at another frequency (§ 2a) |
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
m.motion_shares(by="element")                     # {'H': 1.0}; by="label", by="region"
m.reduced_mass_amu, m.force_constant_mdyn_per_ang # Gaussian's convention, as the record states them
m.motion_share                                    # each free atom's share, the record's
m.thermal                                         # spread and mean quanta at the temperatures the file states

u = m.displacement(zero_point_widths=1.0)         # each atom's move at Q_zp, Å (x, y, z)
m.coordinate(u)                                   # back: sum_A m_A L_A . u_A = Q_zp
m.displaced(0.05)                                 # every atom at R + Q L, Q = 0.05 amu^1/2.Å
m.with_frequency(0.96 * m.frequency_cm1)          # the same pattern at a scaled frequency
```

**Where the molecule sits along the mode** is a state — at a temperature,
at a mean number of quanta, with `quanta` exactly, driven in a steady swing of
`swing_quanta` (its coherent state from R3d2b2 — in 2.78.0 a swing alone
takes the record's temperature for its bath; with a temperature, a warm mode
driven), or the classical limits (`classical=True`) — the table of states is
[`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 2:

```python
s = m.state(temperature_k=0.0)        # the zero-point motion: spread Q_zp
s.spread_amu12_ang, s.mean_quanta
s.atom_rms_ang                        # each atom's r.m.s. move, Å
s.pdf(0.0), s.cdf(0.05), s.moment(4)  # density, distribution, <Q^4>
f = m.state(quanta=1).frames(nodes=3) # three frames, exact to degree 5
f.nodes_amu12_ang, f.weights          # where along the mode (f.q_amu12_ang from R3d2b2), each one's weight
f.positions_ang                       # each frame's geometry, every atom at R + Q L
```

Which state takes the record's temperature when none is given, and says so
(`s.temperature_k`, `s.temperature_from`), is one table,
[`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 2 —
with what changes for a swing named by `swing_quanta` alone in R3d2b2 (§ 9
there). An imaginary mode has moves for a coordinate given and no state.

## 2b. Frames from the modes — the motion toolkit

*(To build — [`plans/plan.md`](?doc=plans/plan.md) § 2; the design and every
choice is [`engines/vibration-frames.md`](?doc=engines/vibration-frames.md),
the theory [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 2c, § 5.)* Frames for a calculation over them — transport's, any other —
each written as a frame set the calculation reads by its weights alone.
Paths come from the projects tree's doors, which check each name, so they
hold wherever the script or notebook runs; `P` the project:

```python
from molbuilder.postprocess import vibration, write_frame_set, structure_dir, topic_dir
v = vibration(structure_dir("P", "spectrum", "V") / "01_freq/run-0")   # the run's folder, or its result file
out = topic_dir("P", "structure")                                     # where a transport citation reads sets

# THE WHOLE MOLECULE: every mode thermal at 300 K, mode 7 driven, warm at 300 K too
motion = v.motion(temperature_k=300.0).with_mode(7, swing_quanta=4, temperature_k=300.0)
snaps = motion.sample(64, seed=20261123, antithetic=True)   # seed and pairing: yours, stated
snaps.mode_index_1based            # the modes, one column each
snaps.atom_moves_ang.shape         # (65, n_atoms, 3): frame 0 (the relaxed geometry), then the displacement map
snaps.said                         # the largest move over the set, the mode moving it
write_frame_set(out / "v-300K-drive7.xyz", snaps)   # frame 0 relaxed at weight 0, then the 64

baseline = v.motion(temperature_k=300.0).sample(64, seed=20261123, antithetic=True)
write_frame_set(out / "v-300K.xyz", baseline)       # the same temperature, every mode but 7 drawn alike

# TWO MODES ON ONE LASER: one phase per snapshot, each mode's lag yours (the science says how)
motion2 = v.motion(temperature_k=300.0).with_drive({9: 2.0, 11: 1.0}, lags_rad={9: 1.57, 11: 0.12})

# ONE MODE ON ITS OWN: three frames of mode 7 at 300 K
f = v.mode(7).state(temperature_k=300.0).frames(nodes=3)
f.q_amu12_ang[:, 0], f.weights     # 0, ±sqrt(3) sigma; 2/3, 1/6, 1/6
write_frame_set(out / "v-m7-300K.xyz", f)
```

A model of your own — a two-mode grid to see a cross term, frames from any
rule — is built from the geometry (`v.displaced({...})`) and the writer's
general form, worked in [`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)
§ 3.4. **How many frames** a small effect needs, and how much a 64-frame set
can see, is the science's § 5.5: the error falls as $1/\sqrt N$, and a
single mode's own effect is cheapest from that mode's own two sets.

The driven mode is compared with its baseline **at one temperature** — the
warm driven state against the thermal one — so the two differ by the drive
alone; the cold coherent default against a warm baseline measures the drive
less the mode's own thermal quanta, its exact baseline that one mode at 0 K
beside the warm others, `with_mode(7, temperature_k=0.0)`
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 6). After a transport calculation over a snapshot set, its average and that
average's error are read from its data file, the pairs' means giving the
error ([`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md)
§ 3.2). The example notebook, `docs/engines/vibration-frames.ipynb`, runs
each of these on a vibration result and checks each against the science
([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 6.1).

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
