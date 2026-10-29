# Post-processing a transport calculation

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
                                    read_transport_data)
```

| name | answers |
|---|---|
| `read_transport_data(path)` | `<label>.transport.nc` — every result with its words — and its atom index map |
| `atom_index_map(path)` | which index a file's atoms are listed by, and both indices of each (§ 4) |
| `read_structure(path)` | a structure pair — the calculation's `junction.xyz` among them — in the user order, every frame: `(structure, atoms, said)`, `atoms` the returned structure's atom index map (listed by user index, each atom's engine index beside) and `said` what its read says, a line each |
| `parse(path)` | a file a run wrote that molbuilder's parsers read — an engine's output, a TBtrans data file (`.TBT.nc`), which answers its kept reading, reading it first where none is kept |

Everything else is the libraries you know: xarray and netCDF4 for the data
files, pandas for the CSV export, sisl for TBtrans's own files.

## 2. The files, and what each is for

| file | where | holds | read it with |
|---|---|---|---|
| `<label>.transport.nc` | the calculation's folder, written by `jobset summarize task` (and by an export, anywhere you choose) | **every result** on its grid: T(E) per point and spin, G, the currents and the I–V, the DOS and its regions, the eigenchannels, the frames' average, every run's state and SCF history — each variable with its units and definition | xarray (§ 3) |
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

```python
import xarray as xr
ds = xr.load_dataset("T.transport.nc")   # read whole, the file closed (§ 7)
```

### 3.1 T(E), the conductance, the I–V

```python
# T(E) at 0.2 V, frame 0 -- the axes are coordinates, selected by value
ds.transmission.sel(frame=0, bias_v=0.2).plot(x="energy_ev")

# G / G0 at E_F, every point
ds.conductance_g0.to_series()

# the I-V, as the record computed it (iv_computed says how)
ds.iv_current_a.sel(frame=0).plot(x="iv_bias_v")
ds.attrs["iv_computed"]
```

### 3.2 A frame set: along the mode, and the average

```python
along_q = ds.swap_dims(frame="q_amu12_ang").sortby("q_amu12_ang")
along_q.conductance_g0.differentiate("q_amu12_ang")
ds.average_transmission.sel(bias_v=0.0).plot(x="energy_ev")
ds.average_why.to_series()          # why an average is none, where it is
```

**Averaged over another motion** *(contract written 2026-10-29, to be
built)*. Each frame is its position along the mode; its weight is the
motion's. The same frames averaged over a bell curve at another temperature,
or with a swing added by a push in step with the mode
([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5.4) — nothing computed again, nothing written:

```python
from molbuilder.postprocess import reaveraged
r = reaveraged("T.transport.nc", temperature_k=300, swing_quanta=1.0)
r["motion"]                       # the quanta, the spreads, the swing's turning points
r["weights"]                      # each frame's weight for this motion
r["conductance_change_percent"]   # <ΔG>/G at each voltage
r["said"]                         # the frames' reach, a weight below zero, ...
```

A swing whose turning points lie beyond the outermost frame is refused, and
with three frames the number is said to be blind to the shape (§ 5.4
there).

### 3.3 The convergence

```python
pt = ds.sel(frame=0, bias_v=0.2)
n = int(pt.device_scf_cycles)
pt.device_scf_ddmax.isel(scf_cycle=slice(0, n)).plot()
pt.device_scf_phase.isel(scf_cycle=slice(0, n)).values   # periodic / negf
```

### 3.4 The CSV export

```python
import pandas as pd
t = pd.read_csv("T.transport-csv/frame_bias_v_energy_ev.csv")
t[(t.frame == 0) & (t.bias_v == 0.2)].plot(x="energy_ev", y="transmission")
# README.txt says what every column is, in the variables' own words
```

### 3.5 What is missing, and why

```python
# a point whose DOS per orbital is not kept has no DOS by region
ds.dos_per_orbital_reading.to_series()
ds.point_why.to_series()            # a point not done: why
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

# the same in the data file itself
ds.engine_atom_index.sel(atom=[0, 5]).values

# the junction, in the user order, with the map of the structure returned
junction, junction_atoms, said = read_structure("junction.xyz")
junction_atoms.listed_by            # 'user_index', as the structure now is
```

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
pdos = dg.DOS.isel(device_orbital=theirs).sum("device_orbital")  # per energy
pdos.plot(x="energy_ev")
```

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
