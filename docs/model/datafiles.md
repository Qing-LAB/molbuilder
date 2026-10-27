# Data files — one framework for every file molbuilder keeps its results in

**Role:** contract
**Domain:** model
**Module:** `molbuilder/ncdata.py` (the variable record, the one writer, the
one door, the tables) · `molbuilder/persist.py` (the atomic write) ·
`molbuilder/atom_permutation.py` (the atom index map) ·
`molbuilder/postprocess.py` (the one import).
**Companions:** [`model/overview.md`](?doc=model/overview.md) § 2.2b (which
index a file's atoms are listed by) · each file's own chapter, which lists its
variables and nothing of the framework: [`engines/transport.md`](?doc=engines/transport.md)
§ 2a.12 (the transport data file), [`engines/vibration.md`](?doc=engines/vibration.md)
§ 6.10.5 (the vibration data file), [`model/parse.md`](?doc=model/parse.md)
§ 5e (a TBtrans data file's kept reading) · the worked examples a person
starts from, [`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md).

*(The user, 2026-10-23: "make it a general data file framework doc … and add
the specific case for transport, so it is clear what it serves"; before it,
"we need to have a framework level agreement rather than … hacking solution
… the correct packaging of … file access within molbuilder".)*

## 0. What it serves

A calculation's results are read long after it ran, by readers molbuilder
does not control: its own pages and commands, a person's notebook (xarray,
netCDF4, sisl, pandas), a script on another machine a year later. A **data
file** is a NetCDF-4 file molbuilder writes so that every one of those
readers gets the same answer from the file alone. The framework promises
six things, each kept by one piece of code:

| promise | kept by |
|---|---|
| **every value says what it is** — its units, its meaning, how it was obtained, the file it came from — readable with no molbuilder installed | the variable record, `ncdata.Var` (§ 2) |
| **every file says which atoms its rows are** — listed by user index or engine index, each atom's index in both lists, counting from 0 | the atom index map, given to the writer as the atom axis itself (§ 3; `model/overview.md` § 2.2b) |
| **one shape for each kind** — one structure at one voltage is a 1 × 1 grid, never a different layout; a value not computed is the stated fill, never left out | each file's variable table and the one writer (§ 3) |
| **a file is written whole or not at all, and read as one version** | the one writer's atomic write; a long reading's version check at every step (§ 3, § 4) |
| **reading never crashes the process or blocks a page** — the NetCDF libraries are not thread-safe, and crash a process that holds two handles on one file | the one door, `ncdata.reading` (§ 4) |
| **the same content as tables**, for a spreadsheet, from the same bytes | `ncdata.as_tables` (§ 5) |

A data file is composed from the calculation's one composition — the
transport record, the vibration result — never read back from another file
by a second reader: it is that composition on its grid, for analysis.

## 1. The files

| file | what it is | written by | schema | its chapter |
|---|---|---|---|---|
| `<label>.transport.nc` | every result of a transport calculation on its grid | `jobset summarize task`; an export composes the same bytes | `molbuilder/transport-data@5` | `engines/transport.md` § 2a.12 |
| `<label>.spectra.nc` | a vibration result, beside its JSON — the same payload, tied by the JSON's sha256 | the result's one writer, at every write on both engines | `molbuilder/spectra-data@7` (`@1`–`@6` read as they are) | `engines/vibration.md` § 6.10.5 |
| `<label>.TBT.digest.nc` (`.TBT_UP` / `.TBT_DN`) | a TBtrans data file's DOS per orbital, k-averaged once and kept | the reading of the `.TBT.nc` beside it | `molbuilder/tbt-digest@3` | `model/parse.md` § 5e |
| `<label>.optical.nc` | a SIESTA vibration's optical calls — each frame's ε₂ on the photon energies, its mode, side and geometry, the settings the calls read: the record of the Raman phase so far | the run's own record steps — the force constants', every frame pending, then each optical call's — and its finish | `molbuilder/optical-data@1` | `engines/vibration.md` § 5.11 § 8 |
| `<label>.raman.nc` | a SIESTA vibration's Raman view at the laser and width a person named — per picked mode `∂α/∂Q` and the strength, at the laser and static, and the equilibrium `α`; the two data files it was derived from by name and sha256 | an export alone (`spectra.export`), never into the calculation | `molbuilder/raman-view@1` | `engines/vibration.md` § 5.11 § 10 |

**An engine's NetCDF files molbuilder reads** come through the same door
(§ 4): TBtrans's `<label>.TBT.nc` (through sisl, `sile=True`) and SIESTA's
`<label>.MD.nc` history (`model/parse.md` § 5d). **Not data files**: the
JSON records a calculation keeps (written whole by `persist.write_bytes`, the
atom index map under `atom_index_map` where they hold atoms — `model/overview.md`
§ 2.2b), the text logs, the decks.

## 2. A variable states itself — `ncdata.Var`

Each file's chapter is a table of `Var` rows — the file's whole content,
named once, read by the writer and the reader alike:

| field | what it says |
|---|---|
| name, dimensions | the variable, over its axes — every axis a row of § 2a and a coordinate of its own name, so a reader selects by value (`transmission.sel(bias_v=0.2)`) |
| `kind` | `given` (an input as stated), `raw` (as the engine printed it), `measured` (read off the structures — a frame set's rows, [`structure.md`](?doc=model/structure.md) § 2.2f), `derived` (molbuilder's computation), `state` (what happened: a run's state, why a value is missing), `coordinate` |
| `units` · `long_name` · `definition` | one spelling a unit (`angstrom`, `A`, `eV`); what it is; how it is obtained — the formula for a derived one |
| `source` | for a raw one, the file it was read from |
| `auxiliary` | a label beside an axis (CF auxiliary coordinate, named in each variable's `coordinates`): an atom's element, a frame's folder |
| `atom_index` | its values are atom indices, counting by the file's atom index map (`free_atom`, a digest's `device_orbital_atom`) |

**Encoding, so a round trip is exact**: a number's fill is NaN, a text's the
empty string, an integer's its stated fill (`empty`); a boolean an `int8`
with `flag_values 0 1` (`flag`); an empty list a dimension of length 0; a
nested record JSON text (`json_attr`). **The file says what it is**: its
`schema`, and each chapter's attributes. A result with parts is written in
groups (the vibration result's `/structure`, `/equilibrium`, `/modes`).

## 2a. The axes — one name for one concept, in every file

*(The user, 2026-10-24: "i need a document/contract with existing known axis
and what to do in the future when more axis that is not in the document
appears. name consistency should be explicit and documented.")*

**Every axis of every data file is a row of the table below** — its name,
what its points are, the values its coordinate holds, its units and the
files that use it. The table is the one source: `ncdata.AXES` holds the same
rows, and the one writer (§ 3) refuses a dimension that is not one of them,
naming this section — in every file, every group, the root's included.

**The rules:**

1. **One concept, one name; one name, one concept** — in every file. The
   energy relative to the leads' Fermi level is `energy_ev` in the transport
   data file and in the digest alike; `xyz` is a Cartesian component wherever
   it appears, and never a cell vector's.
2. **How a name is made**: lowercase, words joined by `_`, singular. The
   axis of a physical quantity names its unit, as a variable does
   (`energy_ev`, `bias_v`, `temperature_k`); the axis of counted things names
   the thing (`atom`, `mode`, `frame`); a subset of a counted thing is
   `<subset>_<thing>` (`free_atom`, `device_atom`, `device_orbital`), its
   coordinate the thing's own index.
3. **Every axis has a coordinate of its own name**, so a reader selects by
   value (`ds.sel(bias_v=0.2)`, `ds.sel(mode=7)`): a quantity's values; a
   counted thing's index, its base said in the coordinate's `definition`;
   labels as text (`x`, `y`, `z`). A second label of the same points (an
   atom's element, a frame's folder) is an auxiliary coordinate beside it.
   An atom index counts by the file's atom index map (§ 3; `atom_index`).
4. **A new axis**: in the same change that first writes it, its row is added
   here and to `ncdata.AXES` — its name made by rule 2 and checked against
   every row for a concept already named. Until then the writer refuses it.
5. **A row is never renamed in place.** Renaming an axis is a new `schema`
   of every file that uses it; each file's reader keeps reading the old name
   as the new one (the vibration data file's `_EARLIER`), so a file written
   before still reads — but for a kept reading, which is out of date at
   another version and made again from its source (the digest,
   `model/parse.md` § 5e).

| axis | its points | its coordinate's values | units (— labels) | files |
|---|---|---|---|---|
| `atom` | the atoms | each atom's index by the file's atom index map — the user index in the transport, vibration and optical files, the engine index in the digest (`atoms_listed_by`), from 0 | 1 | transport, vibration (`/structure`), optical, digest |
| `free_atom` | the atoms a vibration moves | their atom indices, from 0 | 1 | vibration |
| `device_atom` | the atoms of TranSIESTA's device region | their atom indices, from 0 | 1 | digest |
| `atom_bound` | each atom's first orbital, and one past the last | from 0 to the atom count | 1 | digest |
| `device_orbital` | the device region's orbitals, one DOS column each | each orbital's index in the geometry, from 0; its atom beside it (`device_orbital_atom`) | 1 | digest |
| `xyz` | a Cartesian component | `x`, `y`, `z` | — | transport, vibration, optical, digest |
| `cell_vector` | the cell's three vectors | `a`, `b`, `c` | — | vibration, optical (`/structure`), digest |
| `reciprocal_vector` | the reciprocal cell's three vectors — a k-point's reduced components | `b1`, `b2`, `b3` | — | digest |
| `kpoint` | the k-points a reading averaged over | from 0 | 1 | digest |
| `frame` | a frame set's frames — a transport calculation's structures, a vibration's optical calls (the reference geometry, then each mode's two sides) | from 0, as MolView's API counts | 1 | transport, optical |
| `bias_v` | the voltages computed | each voltage | V | transport |
| `iv_bias_v` | the I–V's voltages | each voltage | V | transport |
| `energy_ev` | energy relative to the leads' Fermi level | each energy, ascending | eV | transport, digest |
| `spin` | spin channels | `up` and `down` — or, unpolarized, the one channel by the spin its deck states | — | transport, optical |
| `lead` | the two leads | their names as TBtrans names them | — | transport, digest |
| `region` | molbuilder's regions of the junction | `L-electrode`, `bridge`, `R-electrode` | — | transport |
| `eigenchannel` | transmission eigenchannels | from 0, as TBtrans orders them | 1 | transport, digest |
| `rung` | the rungs that run once | their stage names | — | transport |
| `scf_cycle` | a run's SCF rows | from 0 | 1 | transport |
| `charge_part` | TranSIESTA's charge columns | `D`, `E1`, `C1`, … as its `ts-q:` header names them | — | transport |
| `mode` | vibrational modes | 1-based, ascending frequency | 1 | vibration, Raman view |
| `removed` | the rigid motions removed | from 0 | 1 | vibration |
| `mo` | molecular orbitals | from 0, ascending energy | 1 | vibration |
| `es` | the modes the electronic-structure probe ran on | each mode's 1-based index | 1 | vibration |
| `window` | an orbital's place in the probe's window | from 0, ascending energy | 1 | vibration |
| `temperature_k` | the thermochemistry's temperatures | each temperature | K | vibration |
| `fc_step` | a SIESTA force-constant run's steps | from 0, step 0 the reference geometry, then each displacement in SIESTA's order | 1 | vibration (`/partial_charges`) |
| `photon_energy_ev` | the light's energy, ħω — a concept apart from `energy_ev`, which is an electron's energy relative to the leads' Fermi level | each photon energy as SIESTA wrote it, ascending | eV | optical |

**Files written before these names** read as rule 5 says: the vibration
data file's reader takes `@1`–`@3`'s names for the new ones; a digest before
`@3` is out of date, read again from its `.TBT.nc`.

## 3. Writing — one writer

`ncdata.write_group(group, dims, variables, values)` writes a group's
dimensions and variables from the table, each with its attributes and its
`coordinates`. **Every axis it creates or its variables use is a row of
§ 2a** (`ncdata.AXES`), with a coordinate of its own name in the row's units
— else it refuses, naming § 2a (`ncdata.check_axes`); a group states its own
axes, never one its parent created. **The atom axis is given as its atom index map, never a
count** (`dims={"atom": atoms, …}`, an `AtomIndexMap`): the writer states
`atoms_listed_by`, `atom_index_base` and, where a column is missing,
`atom_index_why` as the group's attributes, and `user_atom_index` /
`engine_atom_index` as variables on the axis — each where it is known — and
refuses a bare number, or a map over another count. **The file is built in
memory** (`in_memory`, `as_bytes`) and handed whole to the package's one
atomic writer (`persist.write_bytes`): a reader sees the old file or the new
one, never half of either. Every NetCDF call — a write included — holds the
process's one lock (§ 4).

## 4. Reading — one door, one access, one handle

**One door for every NetCDF file molbuilder opens** (`ncdata.reading`):
`with reading(path) as ds:` holds the process's one lock (`ncdata.NC_LOCK`),
opens the file read-only — a `netCDF4.Dataset`, or with `sile=True` a sisl
sile for an engine's own file sisl reads, a TBtrans `.TBT.nc` — and closes it
before the lock is released. Two measured failures make the rule:

| measured | so |
|---|---|
| 2026-10-22: two request threads inside netCDF-C at once — two readings and two pages asking states, in one process — segfaulted in 2 of 3 runs; the library is not thread-safe | **one access at a time in the process**: every open, read and write holds the one lock |
| 2026-10-23, netCDF-C 4.10.1 and HDF5 2.2.0, every env here: two handles on one file in one process share one open file (one descriptor); after the second reads a text variable and closes, the next open of that file segfaults — while the handle still held reads on, and numbers alone never do; two separate processes, each with its own handle, never | **no handle outlives its access**: molbuilder's process never holds two handles on one file, so it never reaches that sequence |

**A long reading is a series of accesses**, so the page's other requests run
between its steps and never meet a held handle; each step checks the file is
the version the reading began with (its size and time), and refuses one that
changed rather than mix two. **A file opened again inside its own access is
refused by name** (`ncdata.OpenedTwice`) — the one mistake the door can see —
and no reader's own "this file does not read" swallows it. **Nothing is read
whole into memory** but what molbuilder composed there itself (an export's
bytes, laid out as tables). **The door is the only opener** of a NetCDF file
in molbuilder's code — each data file's reader (`transport.datafile.
read_data_file`, `spectra.datafile.read_spectra_data`, the digest's), the
TBtrans parser, the `.MD.nc` reader, the atom index map's reader — kept so by
review; sisl reads an engine's other files (`.TS.HSX`, `.ORB_INDX`), which
are not NetCDF, through its own doors. **Every reader gives the map back with
the data**: `read_variables(group)` and `read_atom_index_map(group)` — the
map's one reader, `AtomIndexMap.from_dict`, checked as it is read.

**Outside molbuilder's process** — a notebook, a person's script —
molbuilder's readers go through the same door; a handle the person opens is
theirs, and the post-processing document says the libraries' limit: read
with `xr.load_dataset` (whole, no handle kept), or close an
`xr.open_dataset` before the same file is read again.

## 5. As tables — `ncdata.as_tables`

Any data file molbuilder writes, laid out for a spreadsheet from its bytes:
**one long table per set of axes** in each group, named by them
(`frame_bias_v_energy_ev.csv`), a group's tables keeping its enclosing
groups' axes; each axis a column under its own name, each variable a column
named with its units, a value not stated an empty cell; and a `README.txt`
in the variables' own words — every variable's definition and source, the
file's attributes, the atom axis's `atoms_listed_by` and `atom_index_base`.
The atom axis's own table (`atom.csv`) carries both index columns.

## 6. One import — `molbuilder.postprocess`

A person's script reads every data file with the functions molbuilder's own
doors ask (`model/parse.md` § 3, *One layer*): `read_transport_data`,
`read_spectra_data` — each with its atom index map — `atom_index_map(path)`,
`read_structure(path)` for a structure pair in the user order with the map
of the structure it returns, `parse(path)` for an engine's file. A notebook's kernel imports the molbuilder its server
runs ([`web/jupyter.md`](?doc=web/jupyter.md) § 7).

## 7. The transport calculation — what the framework serves, file by file

A finished transport calculation of the 22-atom hydrogen junction this
contract's tests build — its atoms sorted into TranSIESTA's blocks, engine
atom 0 user atom 13 — holds these files. Each is one of the framework's
cases, and the chain between them is the framework's point: a person asks
about **their** atoms, in the user order, and every file answers about the
same atoms.

| file | where | holds | written / read | the framework at work |
|---|---|---|---|---|
| `T.TBT.nc` | each transmission point's folder | TBtrans's own data: everything per k-point, energy and orbital — gigabytes on a large junction | written by TBtrans; read **on a person's word only**, through the door with `sile=True`, **one k-point an access** (113 opens on Sol's AuBDTAu point: 1.4 s held, 5.6 s reopened), its version checked each step | the one door (§ 4); its atoms are its run's engine order, answered by the run's deck (`model/overview.md` § 2.2b, F3) |
| `T.TBT.digest.nc` | beside it | the DOS per orbital, k-averaged once, and the facts the report needs (energies, the device's atoms, each column's atom) | written by that reading through `write_group`, **listed by engine index** with each atom's user index (the point's run's map, handed to the parser by the caller that knows the run); read by the report's DOS by region and a selection's PDOS, its map **checked against the composed junction's record** (F7) | the one writer, the atom axis as its map, `device_orbital_atom` / `device_atom` marked `atom_index` (§ 2, § 3) |
| `T.transport.nc` | the calculation's folder | every result on one grid — `frame` × `bias_v` × `energy_ev` (× `spin`, `region`, `lead`, `eigenchannel`, `scf_cycle`, `charge_part`), each run's state and SCF history, the calculation's own records — one shape for one structure at one voltage too | written by `summarize task` from the record's one composition (`record.collect_record`), the atom axis **listed by user index** with each atom's engine index from the junction's order record; read by `read_data_file`, `postprocess.read_transport_data`, and the export | the variable record (§ 2), the one writer (§ 3), the one door (§ 4) |
| `T.transport-csv/` | where an export is asked to write it | the same as long tables, `atom.csv` with both index columns, `README.txt` | `ncdata.as_tables` of the data file's bytes, composed on request — the export reads no data file (`engines/transport.md` § 2a.12, *Export*) | the tables (§ 5) |

**The chain, for one selection.** A person selects the bridge's first atom,
**user index 0**, on the report's Device view (it numbers it *#1*). The data
file states it: `engine_atom_index[0]` is **9** — the device run's order.
The digest is listed by engine index, so its columns on atom 9 are that
atom's orbitals; their DOS summed is the atom's PDOS, the same the report
draws (`record.selection_pdos`, in through `Structure.map_from_user_index`,
back through `map_to_user_index`). A notebook asks sisl the same question of
`T.TBT.nc` with the same **9**, from the data file's own column — and gets
the same curve to the digest's single precision. No step translates by hand:
each crossing is the map's (`model/overview.md` § 2.2b), and each file
states the map it is listed by.

**What the framework prevents, in this case**: a page that reads a
gigabyte `.TBT.nc` while another request reads its digest — one access at a
time, no handle held between k-points; a notebook holding the data file
open while molbuilder reads it — the document's `xr.load_dataset`; a
selection of atom 0 answered with engine atom 0, a lead atom — the map in
every file; an export that is half one version and half another — composed
whole, written atomically.

## 8. Adding a data file

A new kind of result is a data file when a reader outside molbuilder needs
its values on a grid. It is added as the three above were: a `Var` table in
its own module, every variable with its units and words, every axis a row of
§ 2a — an axis already named there used by its name, a new one added there
first; one writer through
`write_group`, its atom axis — if it holds atoms — given as the map from the
door that knows the atoms' columns (`model/overview.md` § 2.2b, F2); a
`schema` string, versioned when a variable changes; its reader through
`reading`, refusing another schema by name; its catalogue row
(`runfiles.WRITTEN`: its role, its writer, its door, `listed_by`); and its
chapter, which lists its variables and cites this page for the rest.
