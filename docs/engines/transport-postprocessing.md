# Post-processing a transport calculation

**Role:** guide
**Domain:** engines
**Companions:** [`engines/transport.md`](?doc=engines/transport.md) — the
calculation these results come from;
[`model/datafiles.md`](?doc=model/datafiles.md) — `<label>.transport.nc` and
what every data file shares;
[`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) —
a frame set's average; [`web/jupyter.md`](?doc=web/jupyter.md) § 7 — the
notebook that imports this molbuilder;
[`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md)
— a run's steps (§ 3).

*(The user, 2026-10-23: "the best way to do this is to give an example code
… if user has already exported a data or want to read the data of the
transmission or IV or other data, what does the Python script look like …
This can be part of the document separately, like as a post-processing
transmission document, and then the web UI and the result tab can refer to
that document.")*

What a finished — or a running — transport calculation holds, which file to
read for what, and worked examples in Python: T(E), the conductance and the
I–V, the DOS and its regions, a selection's PDOS, the convergence, and how
an atom's index is mapped between your structure and the engine's files.
Every example below ran on a real calculation before it was written here —
a junction of 22 atoms whose device holds 4, its points at 0 and 0.2 V.

## 1. One import

A notebook on the Notebook tab imports the molbuilder its server runs
([`web/jupyter.md`](?doc=web/jupyter.md) § 7); a script on the machine does
the same from its checkout:

```python
from molbuilder.postprocess import (parse, atom_index_map, read_structure,
                                    read_transport_data, read_steps, read_xv)
```

| name | answers |
|---|---|
| `read_transport_data(path)` | `<label>.transport.nc` — every result with its words, the attributes that hold records as data ([`model/datafiles.md`](?doc=model/datafiles.md) § 2) — and its atom index map |
| `atom_index_map(path)` | which index a file's atoms are listed by, and both indices of each (§ 4) |
| `read_structure(path)` | a structure pair — the calculation's `junction.xyz` among them — in the user order, every frame: `(structure, atoms, said)`, `atoms` the returned structure's atom index map (listed by user index, each atom's engine index beside) and `said` what its read says, a line each |
| `read_steps(run)` | the steps a run finished — geometry, forces, energy — **in your atoms' order**, from its folder: `01_seed/run-0`, `04_device/run-0/v0.2` (§ 4). A transmission point has none, and is refused |
| `read_xv(path)` | SIESTA's own `.XV`, **in the engine's order** — row `i` is engine index `i` (§ 4) |
| `project_dir`, `list_projects`, `list_topics`, `list_structures`, … | where a calculation is, by name, in the tree `projects_root()` answers |
| `parse(path)` | a file a run wrote that molbuilder's parsers read, **in the file's own order**, `result.atom_index_map` saying whose — an engine's output, a TBtrans data file (`.TBT.nc`), which answers its kept reading, reading it first where none is kept |

Everything else is the libraries you know: xarray and netCDF4 for the data
files, pandas for the CSV export, sisl for TBtrans's own files.

## 2. The files, and what each is for

How the framework writes and reads each is
[`model/datafiles.md`](?doc=model/datafiles.md) § 7; here, where each is and
what to read it with.

| file | where | holds | read it with |
|---|---|---|---|
| `<label>.transport.nc` | the calculation's folder, written by `jobset summarize task` (and by an export, anywhere you choose) | **every result**, in groups — the junction whole (its cell, each atom's region and labels), each stage, each point and its device run, and a summary across the points: T(E) per point and spin, G, the currents and the I–V, the DOS and its regions, the eigenchannels, the frames' average, every run's state, SCF history and engine version — each variable with its units and definition | xarray (§ 3) |
| `<label>.transport-csv/` | an export's CSV form | the same, as long tables — one per set of axes — and `README.txt` | pandas (§ 3.4) |
| `<label>.TBT.digest.nc` | beside each transmission point's `.TBT.nc` | the **DOS per orbital**, k-averaged, once read — on the DOS tab's *Read the DOS per orbital*, or by `summarize task` | xarray (§ 5); `parse` of the `.TBT.nc` beside it answers the same reading |
| `<label>.TBT.nc` | each transmission point's folder | TBtrans's own data: everything per k-point, energy and orbital — gigabytes on a large junction | sisl (§ 6) |
| `<label>.TBT.AV*` | beside it | TBtrans's own k-averaged text files: T(E), the DOS, the leads' DOS, the eigenchannels | any text reader — whitespace columns, `#` comments (`numpy.loadtxt`) |

**Counting**: every atom index in a file molbuilder writes in its own
formats counts from 0 (`atom_index_base` says so); a deck and an engine's
input count as their engine does — a SIESTA `.fdf`'s `elec-pos`,
`TS.Atoms.Buffer` and `Geometry.Constraints` from 1 — and MolView numbers an
atom its index + 1 on screen.

## 3. The data file

The calculation as groups ([`engines/transport.md`](?doc=engines/transport.md)
§ 2a.12): `/structure` the junction, `/stages/<stage>` each stage that runs
once, `/points/v0.2` each point — a frame set's `/points/f003/v0.2` — with
its `device` run, and `/summary` across the points. **The summary and each
point are one composition**: a value both hold is one number. Each group is
read alone, its axes its own:

```python
import xarray as xr
root = xr.load_dataset("T.transport.nc")                     # the file's own words
s = xr.load_dataset("T.transport.nc", group="summary")        # across the points
p = xr.load_dataset("T.transport.nc", group="points/v0.2")     # one calculation whole
tree = xr.open_datatree("T.transport.nc")                     # every group (close it, § 7)
sorted(tree.groups)
tree.close()
```

### 3.1 T(E), the conductance, the I–V

```python
# T(E) at 0.2 V, frame 0 -- the axes are coordinates, selected by value
s.transmission.sel(frame=0, bias_v=0.2).plot(x="energy_ev")

# G / G0 at E_F, every point
s.conductance_g0.to_series()

# the I-V, as the record computed it (iv_computed says how)
s.iv_current_a.sel(frame=0).plot(x="iv_bias_v")
root.attrs["iv_computed"]

# the same point taken whole: its current is the summary's
float(p.current_a) == float(s.current_a.sel(frame=0, bias_v=0.2))
```

### 3.2 A frame set: along the mode, the average, its error

A mode's own set, read along the mode — its frames' coordinate is a row of
the frame set's rows, which the file carries in the structure it cites
(`structure_json`, [`engines/transport.md`](?doc=engines/transport.md)
§ 2a.12), read through the structure's door by `frame_rows` and set beside
`frame` in one line; each frame's `weight`, the one row the calculation
reads, is a label beside `frame` already:

```python
from molbuilder.postprocess import frame_rows
s = xr.load_dataset("F.transport.nc", group="summary")       # read whole, no handle held (§ 7)
rows = frame_rows("F.transport.nc")                           # the set's rows and each frame's
s = s.assign_coords(q_amu12_ang=("frame", [rows.value("q_amu12_ang", f) for f in range(len(rows.frames))]))
along_q = s.swap_dims(frame="q_amu12_ang").sortby("q_amu12_ang")
along_q.conductance_g0.differentiate("q_amu12_ang")
s.average_transmission.sel(bias_v=0.0).plot(x="energy_ev")
s.average_why.to_series()           # why an average is none, where it is
```

**Averaged at another state** — a mode's own set. Each frame is its
position along the mode; its weight is the motion's. The same frames
averaged over a bell curve at another temperature, at $n$ quanta, classical
— any chain or distribution ([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 2) — the weights the motion toolkit's for the set's rows
([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 7;
[`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.4), the averaging transport's own (`frame_average.averaged_at`) —
nothing computed again, nothing written:

```python
from molbuilder.postprocess import reaveraged, states
r = reaveraged("F.transport.nc", states.thermal(temperature_k=77.0))   # the bell curve at 77 K
r["weights"].weights              # each frame's weight for this state
r["weights"].beyond               # the share of the state's motion beyond the outermost frames
r["weights"].exact_to_degree      # the degree the weights reproduce the state's averages to
r["conductance_change_percent"]   # <ΔG>/G at each voltage
r["said"]                         # the state taken, the share beyond, a weight below zero, ...
r["assumptions"]                  # transport's own, then the set's, for the state it was written for
reaveraged("F.transport.nc", states.level(1))                       # one quantum exactly
reaveraged("F.transport.nc", states.thermal().add_quantum())         # one photon on the set's warm mode
```

Its rules — the binding, the temperature an open start takes when none is
given, what it says beside the weights and what it refuses — are the
toolkit's, stated once ([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)
§ 2's table, § 7; the weights alone, `postprocess.weights_for_set(frame_rows(path),
...)`): the set re-weighted to its own state is not said to reach short
or to be blind (science § 5.4's items 1 and 2), frames placed for another
motion are; a weight below zero is said of any frames. **Frames re-weighted are
exact only to degree $m-1$** ($m$ for an odd number of symmetric frames and
a symmetric state): one mode's change of state read from one set
re-weighted needs five frames at least, or two sets, one per state
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 6).

**A snapshot set's average, and its error.** Every frame of a snapshot set
weighs $1/N$ but frame 0, the relaxed geometry at weight 0. The record's
average is the frames' average; its statistical error is read from the
frames ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.5) — with mirrored pairs (the set's `antithetic` row 1: frames
$2j-1$ and $2j$ a pair), the spread of the pair means over $\sqrt{N/2}$:

```python
from molbuilder.postprocess import sampling
g = s.conductance_g0.sel(bias_v=0.0).values       # frame 0 first, the relaxed geometry
means = sampling.pair_means(g[1:])                # frame 0 left out; values 1,2 a pair, 3,4 the next:
                                                  # (g[1]+g[2])/2, (g[3]+g[4])/2, ... -- grouped two by two by the function
mean, error = sampling.mean_with_error(means)     # the average and its error, from the pair means
# independent snapshots (antithetic 0): sampling.mean_with_error(g[1:])
```

The motion toolkit gives the same from the data file's values whole, one
per frame, frame 0 first — the set read back leaving frame 0 out itself
(`read_frame_set(...).standard_error(g)`,
an `Estimate`, and `parts` for the even and odd shares, [`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)
§ 3.3, § 7). **More frames** are the same seed drawn further, or another seed,
written as a set of their own: the two calculations' results put end to end,
each set's frame 0 left out, are averaged with their error from all their
pairs together (`sampling.mean_with_error` over `sampling.pair_means`,
[`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 3.3,
§ 9.1). **How many frames**: the error falls as $1/\sqrt N$, so a small
effect needs many — how many, independent or paired, and the worked count,
are [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.5's, from the error a person's own frames give; one mode's effect alone is
cheapest from that mode's own two sets — three frames each, the excited
state and the thermal one ([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)
§ 4).

**A set against its variation in one mode.** Two transport calculations,
over two snapshot sets drawn with the same seed **at the same temperature** —
the second with one mode set apart in another state, a chain bound to it —
one photon on it, say (`with_mode(k, states.thermal().add_quantum())`) — so they differ by that
mode alone. Each frame of the one and the same
frame of the other differ by that mode's draw alone ([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)
§ 3.3), so the difference is taken frame by frame, and its error from the
spread of the differences — the other modes' part cancelling exactly only
where the response adds over the modes, otherwise in part, as the spread
says ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.5):

```python
d = xr.load_dataset("D.transport.nc", group="summary").conductance_g0.sel(bias_v=0.0).values
b = xr.load_dataset("B.transport.nc", group="summary").conductance_g0.sel(bias_v=0.0).values
diff = sampling.pair_means(d[1:] - b[1:])     # frame 0 left out; pairs, as above
change, error = sampling.mean_with_error(diff)
```

### 3.3 The convergence

```python
d = xr.load_dataset("T.transport.nc", group="points/v0.2/device")
n = int(d.scf_cycles)                                 # a point that ran one; xarray reads -1, none ran, as NaN
d.scf_ddmax.isel(scf_cycle=slice(0, n)).plot()
d.scf_phase.isel(scf_cycle=slice(0, n)).values       # periodic / negf
seed = xr.load_dataset("T.transport.nc", group="stages/seed")
float(seed.fermi_level_ev), seed.attrs["engine_version"]  # its E_F, the SIESTA that ran it
```

### 3.4 The CSV export

```python
import pandas as pd
t = pd.read_csv("T.transport-csv/summary/frame_bias_v_energy_ev.csv")
t[(t.frame == 0) & (t.bias_v == 0.2)].plot(x="energy_ev", y="transmission")
# a group's tables under its path (points/v0.2/..., stages/seed/...);
# README.txt says what every column is, in the variables' own words
```

### 3.5 The junction, and which atoms a DOS covers

```python
j = xr.load_dataset("T.transport.nc", group="structure")
j.cell_ang.values                   # the lattice vectors as rows, angstrom
j.axis_kind.values                  # isolated / periodic / transport, per vector
j.atom_region.to_series()           # L-electrode, bridge, R-electrode -- or empty
p.dos_atom.sum()                    # how many atoms this point's DOS covers

import json
from molbuilder.structure import Structure
junction = Structure.from_dict(json.loads(j.attrs["structure_json"]))
junction.regions                    # every label and its atoms, yours among them
```

`/structure` is laid out as the framework lays out a structure
([`model/datafiles.md`](?doc=model/datafiles.md) § 3a): a vibration result's
`.spectra.nc` holds the same variables, its `position_ang` one geometry's
(`atom`, `xyz`) where this file's runs over its frames.

`dos_region_per_ev` is the DOS per orbital summed over the atoms of a region that
the point's DOS covers (`atom_region`, `dos_atom`); a sum over atoms you
choose is yours to form, from the point's digest (§ 5).

### 3.6 What is missing, and why

```python
# a point whose DOS per orbital is not kept has no DOS by region
p.dos_per_orbital_reading.values
p.why.values                        # a point not done: why
```

## 4. Atom indices — user index and engine index

A transport calculation's seed, device and transmission runs list the
atoms in **TranSIESTA's order** — the junction sorted into contiguous
blocks along the transport axis: a buffer, the lower lead, the bridge, the
upper lead, a buffer ([`engines/transport.md`](?doc=engines/transport.md)
§ 4) — the **engine index**; each electrode rung runs a cell of its own,
listed by its own engine index alone. Your structure, the data
file, the transport report's drawing and the Molbuilder tab use **the user
order** — the **user index**; the Results tab's viewer of a sorted run's own
file shows the engine's order, and says so. Every file says which it is
listed by, and carries both where both are known
([`model/overview.md`](?doc=model/overview.md) § 2.2b):

```python
atoms = atom_index_map("T.transport.nc")
atoms.listed_by                     # 'user_index'
atoms.map_to_engine_index([0, 5])   # user atoms 0 and 5, as engine indices
atoms.table()                       # every atom: its user index and engine index

# the same in the data file itself, its junction's group
xr.load_dataset("T.transport.nc", group="structure").engine_atom_index.sel(
    atom=[0, 5]).values

# the junction, in the user order, with the map of the structure returned
junction, junction_atoms, said = read_structure("junction.xyz")
junction_atoms.listed_by            # 'user_index', as the structure now is
```

**A file SIESTA itself wrote is the engine's**: `read_xv` of a run's `.XV` lists
its atoms by engine index, as the file does. Row 0 of a transport run's `.XV` is
engine atom 0, which is not your atom 0:

```python
struct = read_xv("01_seed/run-0/T.XV")        # engine order, as SIESTA wrote it
atoms = atom_index_map("01_seed/run-0/T.XV")  # that run's map, from its deck
mine, _ = atoms.map_rows_to_user_order(struct.positions)   # row i: your atom i
```

**A run's steps, whichever it sorted**: `read_steps` crosses every row by the
run's map, so a position or a force is already your atom's; an electrode's cell is
a structure of its own, its rows its own engine index, and `said` says so. A device
point's forces are all NaN, TranSIESTA computing none in such a run, and `said` says
that too.

```python
seed = read_steps("01_seed/run-0")
seed.positions_ang[-1, 0]       # your atom 0, Å
seed.forces_ev_per_ang[-1, 0]   # the force on it, eV/Å
seed.final_energy_ev            # the last energy of a finished step, eV
seed.last                       # a Structure: your order, its labels, its box
seed.applied_offset_ang         # what prep moved the structure it was handed by
```

What a step is, the force figures, the energy and where the coordinates stand are
[`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md) § 3.

On the command line: `molbuilder atom-index <file or run folder>` prints the
table; `--to-engine 0,5` (user indices to engine indices) or `--to-user 0,5`
(engine indices to user indices) maps indices.

## 5. The DOS per orbital — a point's digest

TBtrans computes the DOS of the **device region** only: an atom of a lead
outside it has none. The digest says which atoms the device holds
(`device_atom`) and which atom each device orbital — one DOS column — is on
(`device_orbital_atom`) — both by engine index.

```python
point = "05_transmission/run-0/v0.2/"
dg = xr.load_dataset(point + "T.TBT.digest.nc")
dg.attrs["atoms_listed_by"]                          # 'engine_index'
atoms = atom_index_map(point + "T.TBT.digest.nc")

# the device's atoms, by user index -- the ones that have a DOS
mine = atoms.map_to_user_index(dg.device_atom.values.tolist())

# the PDOS of two of them: their engine indices, their orbitals
wanted = atoms.map_to_engine_index(mine[:2])
theirs = dg.device_orbital_atom.isin(wanted).values
pdos = dg.orbital_dos_per_ev.isel(device_orbital=theirs).sum("device_orbital")  # per energy
pdos.plot(x="energy_ev")
```

### 5.1 The same from the data file: `postprocess.pdos`

The PDOS of atoms you choose — by index, by label, by region, mixed, their
union — summed as the report sums it, from the data file and each point's
digest beside it ([`engines/transport.md`](?doc=engines/transport.md)
§ 2a.12, *A PDOS of the atoms you choose, from the files*). In the
calculation's folder, or in the folder of the Export's *NetCDF with DOS per
orbital* — no `.TBT.nc` needed:

```python
from molbuilder.postprocess import pdos
r = pdos("F.transport.nc", "f001/v0.2",
         regions=["bridge"], labels=["R-electrode"], atoms=[0])
r["energy_ev"], r["pdos"]           # 1/eV, on the point's energies
r["atoms"]                          # the atoms summed -- your user indices
r["outside_device"]                 # named, outside TBtrans's device region: no DOS
r["digests"], r["said"]             # what was read, and what the sum is
pdos("F.transport.nc", "f001/v0.2", regions=["bridge"], orbitals="s")
```

The selection, and so the sum, is yours. A point whose readings were not
all kept when the file was written, a digest read again since, or an orbital
type the basis does not hold is refused by name.

## 6. TBtrans's own file, with sisl

The same PDOS from the point's `.TBT.nc` — every k-point read, the cost the
digest saves — agrees with the digest's sum to its single precision, once
both are in one energy order: sisl answers in the file's own order, which a
calculation that reuses a finished one joins from two grids, while the
digest's runs from low to high:

```python
import numpy as np
import sisl
tbt = sisl.get_sile(point + "T.TBT.nc")
run = atom_index_map(point)                  # the point's run, from its deck
pdos = tbt.DOS(atoms=run.map_to_engine_index(mine[:2]))   # sisl counts from 0
energy = tbt.E                               # the file's own energy order
tbt.close()                                  # one handle at a time (§ 7)
order = np.argsort(energy)                   # low to high, as the digest
pdos, energy = pdos[order], energy[order]
```

sisl asks its own questions too — `tbt.transmission()`, `tbt.ADOS(...)`,
`tbt.orbital_current(...)` — always by engine index, counted from 0.

## 7. One handle on a file at a time, in your notebook too

molbuilder's readers open a file, read what they read and close it before
they return — never two handles on one file
([`model/datafiles.md`](?doc=model/datafiles.md) § 4, *one door, one
access, one handle*). Your notebook is your own process: the NetCDF libraries installed
here (netCDF-C 4.10.1, HDF5 2.2.0) crash a process that opens a file again
after a second handle on it read a text variable and closed, while another
handle held it — so keep one handle on a file at a time. The examples
here read with `xr.load_dataset(path)`, which reads the file whole and holds
no handle after it returns; an `xr.open_dataset` — lazy, its handle held —
is closed (`ds.close()`) before the same file is read with molbuilder's
readers or opened again, and a sisl sile is closed when done. Separate
programs — the server, another notebook — never meet it.
