# Data files — one framework for every file molbuilder keeps its results in

**Role:** contract
**Domain:** model
**Module:** `molbuilder/ncdata.py` (the variable record, the one writer, the
one door, the tables) · `molbuilder/persist.py` (the atomic write) ·
`molbuilder/atom_permutation.py` (the atom index map) ·
`molbuilder/postprocess.py` (the one import).
**Companions:** [`model/overview.md`](?doc=model/overview.md) § 2.2b (which
index a file's atoms are listed by) · **every name a data file holds — each
file, group, variable and attribute, where it is and what it is — is in this
chapter's catalogue, § 9** *(the user, 2026-11-16: "a unified place to find
where it is and what it is")*; each file's own chapter explains its science
and links to its entry: [`engines/transport.md`](?doc=engines/transport.md)
§ 2a.12 (the transport data file), [`engines/vibration.md`](?doc=engines/vibration.md)
§ 5.11 and § 6.10 (the optical calls, the Raman view, the vibration data
file), [`model/parse.md`](?doc=model/parse.md) § 5e (a TBtrans data file's
kept reading) · the worked examples a person
starts from, [`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md),
[`engines/vibration-postprocessing.md`](?doc=engines/vibration-postprocessing.md),
[`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md).

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
seven things, each kept by one piece of code:

| promise | kept by |
|---|---|
| **every value says what it is** — its units, its meaning, how it was obtained, the file it came from — readable with no molbuilder installed | the variable record, `ncdata.Var` (§ 2) |
| **every file and every group says what it is** — what it holds, which schema, which molbuilder wrote it, how its data were made — and each unit it uses, in words | the root's statement and each group's title, required by the one writer (§ 2b), one spelling for each unit (§ 2c) |
| **every file says which atoms its rows are** — listed by user index or engine index, each atom's index in both lists, counting from 0 | the atom index map, given to the writer as the atom axis itself (§ 3; `model/overview.md` § 2.2b) |
| **one shape for each kind** — one structure at one voltage is a 1 × 1 grid, never a different layout; a value not computed is the stated fill, never left out | each file's variable table and the one writer (§ 3) |
| **a file is written whole or not at all, and read as one version** | the one writer's atomic write; a long reading's version check at every step (§ 3, § 4) |
| **reading never crashes the process or blocks a page** — the NetCDF libraries are not thread-safe, and crash a process that holds two handles on one file | the one door, `ncdata.reading` (§ 4) |
| **the same content as tables**, for a spreadsheet, from the same bytes | `ncdata.as_tables` (§ 5) |

A data file is composed from the calculation's one composition — the
transport record, the vibration result — never read back from another file
by a second reader: it is that composition on its grid, for analysis.

## 1. The files

| file | what it is | written by | schema | its chapter · its names |
|---|---|---|---|---|
| `<label>.transport.nc` | every result of a transport calculation, in groups: the junction, each stage, each point and its device run, and the summary across the points | `jobset summarize task`; an export composes the same file | `molbuilder/transport-data@12` | `engines/transport.md` § 2a.12 · § 9.1 |
| `<label>.spectra.nc` | a vibration result, beside its JSON — the same payload, tied by the JSON's sha256 | the result's one writer, at every write on both engines | `molbuilder/spectra-data@10` (`@1`–`@9` read as they are) | `engines/vibration.md` § 6.10, *The data file* · § 9.2 |
| `<label>.TBT.digest.nc` (`.TBT_UP` / `.TBT_DN`) | a TBtrans data file's DOS per orbital, k-averaged once and kept | the reading of the `.TBT.nc` beside it | `molbuilder/tbt-digest@5` | `model/parse.md` § 5e · § 9.6 |
| `<label>.optical.nc` | a SIESTA vibration's optical calls — each frame's ε₂ on the photon energies, its mode, side and geometry, the settings the calls read: the record of the Raman phase so far | the run's own record steps — the force constants', every frame pending, then each optical call's — and its finish | `molbuilder/optical-data@5` (`@1`–`@4` read as they are) | `engines/vibration.md` § 5.11 § 8 · § 9.3 |
| `<label>.optimization.nc` | an optimization's every step on one grid — each atom's position and force, its charges where the run printed them, the energy, the stress, the largest force — and each run's own record: whether it converged, how it ended | an export alone (`optimization.export`), never into the calculation | `molbuilder/optimization-data@3` | `engines/optimization-postprocessing.md` § 6 · § 9.5 |
| `<label>.raman.nc` | a SIESTA vibration's Raman view at the laser and width a person named — per picked mode `∂α/∂Q` and the strength, at the laser and static, and the equilibrium `α`; what it was derived from by name and sha256 — the result, `<label>.spectra.json`, and the optical calls' data file | an export alone (`spectra.export`), never into the calculation | `molbuilder/raman-view@4` | `engines/vibration.md` § 5.11 § 10 · § 9.4 |

**An engine's NetCDF files molbuilder reads** come through the same door
(§ 4): TBtrans's `<label>.TBT.nc` (through sisl, `sile=True`) and SIESTA's
`<label>.MD.nc` history (`model/parse.md` § 5a). **Not data files**: the
JSON records a calculation keeps (written whole by `persist.write_bytes`, the
atom index map under `atom_index_map` where they hold atoms — `model/overview.md`
§ 2.2b), the text logs, the decks.

## 2. A variable states itself — `ncdata.Var`

Each file's writer holds a table of `Var` rows — the file's whole
content, named once, read by the writer and the reader alike; the catalogue
(§ 9) lists every row with what it means:

| field | what it says |
|---|---|
| name, dimensions | the variable, over its axes — every axis a row of § 2a and a coordinate of its own name, so a reader selects by value (`transmission.sel(bias_v=0.2)`) |
| `kind` | `given` (an input as stated), `raw` (as the engine printed it), `measured` (read off the structures — a frame set's rows, [`structure.md`](?doc=model/structure.md) § 2.2f), `derived` (molbuilder's computation), `state` (what happened: a run's state, why a value is missing), `coordinate` |
| `units` · `long_name` · `definition` | one spelling a unit (`angstrom`, `A`, `eV`); what it is; how it is obtained — the formula for a derived one. **A value cited from another record keeps what it means**: a variable holding a row a structure states (a frame set's, [`structure.md`](?doc=model/structure.md) § 2.2f) takes its `definition` from the row's meaning as written (`note`), so a number carried into a file still says what it is and how it was obtained |
| `source` | for a raw one, the file it was read from |
| `auxiliary` | a label beside an axis (CF auxiliary coordinate, named in each variable's `coordinates`): an atom's element, a frame's folder — **of the kind its value is** (`given`, `raw`), never `coordinate`, which is an axis's own |
| `atom_index` | its values are atom indices, counting by the file's atom index map (`free_atom`, a digest's `device_orbital_atom`) |

**Every variable says what it is**: its long name and its definition,
never empty; a number its unit in one spelling of § 2c — `1` where it has
none; a text no unit. **A quantity is a variable**, over its axes or over
none (a variable of no dimension: a step, a window's ends, a temperature):
never an attribute, which holds text — a sentence, a name, a record as JSON
— a count or a flag. The one writer refuses a variable that does not say
what it is, and a real number kept as an attribute (§ 3); a whole number it
cannot tell from a count, so a quantity kept as one is the writer's own
error to avoid.

**Encoding, so a round trip is exact**: a number's fill is NaN, a text's the
empty string, an integer's its stated fill (`empty`); a boolean an `int8`
with `flag_values 0 1` (`flag`); an empty list a dimension of length 0; a
nested record JSON text (`json_attr`), read back as the record by `unjson_attr`.
A result with parts is written in groups (the vibration result's `/structure`,
`/equilibrium`, `/modes`; the transport data file's `/structure`,
`/stages/<stage>`, `/points/<point>` and `/summary`, every group read alone
— its axes its own coordinates — and a value two groups hold written once
from one record entry, `engines/transport.md` § 2a.12).

**Which attributes are records is each file's stated list** — the transport
data file's `transport.datafile.JSON_ATTRS` and its frames' part's
`frame_average.JSON_ATTRS`, the vibration result's block tables' `json`
entries, `structure_json` the framework's — each marked **record** in § 9.
A name on the list is a record the file promises,
and its reader hands it over as data; any other attribute is text, as written,
and promises nothing *(the user, 2026-11-04)*.

## 2b. The file and each group say what they are

*(Proposed 2026-10-27 — "one rule in the shared writer: every file and group
says what it is, plus the versions and the units of numbers stored as
attributes; then each file's own gaps get filled" — and the user's yes: "go
ahead with the NetCDF self-description".)*

Every file states, on its root, its `title`, `schema`, `molbuilder_version`,
`source`, `Conventions` and `unit_meanings`, and every group its `title` —
what each says: § 9.0.1.

The root's are written by the file's own statement (`ncdata.state`), a
group's title with its variables (`write_group(…, title=)`), and
`unit_meanings` by the writer from the units the variables state; the one
writer refuses a file whose root or any group lacks one (`ncdata.as_bytes`).
**The writer states no time of its own**: a file composed again from the
same records holds the same groups, variables, values and attributes — an
export is the data file `summarize` writes (§ 1) — and when its data were
made is its sources' own record (each run's launch and end, `run.json`); a
time a record holds — the vibration result's `timestamp` — is that record's,
carried as it is. **Its bytes are not the same**: the HDF5 library under
NetCDF-4 stamps the file's root header with the time it was built (four
times and their checksum), so two compositions a second apart differ in
those bytes alone; a file is compared by what it holds, never by its
sha256.

## 2c. Units — one spelling each

**Every unit a data file states a number in is a row of the table below**,
spelled one way in every file; `ncdata.UNITS` holds the same rows, and the
one writer refuses a number in another spelling. **A name carries its unit
as the last column spells it** — `force_ev_per_ang`, `dos_total_per_ev`,
`raman_activity_ang4_per_amu` — and a number with no unit (`1`) carries none
*(the user, 2026-11-16: "fix all, style too")*. A new unit is a row here and
there, in the change that first writes it.

| spelling | means | in a name |
|---|---|---|
| `1` | no unit — a count, an index, a fraction or a flag | — |
| `%` | per cent | `percent` |
| `s` | second | `s` |
| `K` | kelvin | `k` |
| `V` | volt | `v` |
| `A` | ampere — never ångström | `a` |
| `e` | the elementary charge, 1.602176634×10⁻¹⁹ C | `e` |
| `eV` | electronvolt, 1.602176634e-19 J | `ev` |
| `eV/angstrom` | electronvolt per ångström | `ev_per_ang` |
| `kbar` | kilobar, 10⁸ Pa | `kbar` |
| `1/eV` | per electronvolt | `per_ev` |
| `Eh` | hartree, the atomic unit of energy, 27.211386 eV | `eh` |
| `Eh^2` | hartree squared | `eh2` |
| `Eh/K` | hartree per kelvin | `eh_per_k` |
| `Eh/bohr` | hartree per bohr, the atomic unit of force (1 bohr = 0.529177 Å) | `eh_per_bohr` |
| `atm` | standard atmosphere, 101325 Pa | `atm` |
| `G0` | the conductance quantum, 2e²/h = 7.748092×10⁻⁵ S | `g0` |
| `nm` | nanometre | `nm` |
| `cm-1` | wavenumber, per centimetre | `cm1` |
| `km/mol` | kilometre per mole | `km_per_mol` |
| `angstrom` | ångström, 10⁻¹⁰ m | `ang` |
| `angstrom^3` | cubic ångström | `ang3` |
| `amu` | the unified atomic mass unit (dalton), 1.66053907e-27 kg | `amu` |
| `amu^1/2 angstrom` | square-root amu times ångström | `amu12_ang` |
| `amu^-1/2` | per square-root amu | `per_amu12` |
| `angstrom^2/amu^1/2` | square ångström per square-root amu | `ang2_per_amu12` |
| `D/(angstrom amu^1/2)` | debye (3.33564×10⁻³⁰ C m) per ångström per square-root amu | `d_per_ang_amu12` |
| `angstrom^4/amu` | ångström to the fourth per amu | `ang4_per_amu` |

**CF-1.8 and its units**: CF reads a unit through UDUNITS, which has no
name for the hartree, the conductance quantum or a square-root mass; those
are spelled as the table says, and every file states each spelling it uses
in words (`unit_meanings`, § 2b), so a reader without molbuilder reads what
a number is in.

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
   energy relative to the device's Fermi level is `energy_ev` in the transport
   data file and in the digest alike; `xyz` is a Cartesian component wherever
   it appears, and never a cell vector's. **In a file of groups, no
   variable takes the name of an axis an enclosing group defines**: a tree
   reader (xarray's DataTree) hands the inherited axis in its place, the
   value lost without an error — the one writer refuses it
   (`ncdata.write_group`), and the transport data file's stage and device
   energies are `total_energy_ev` for it — the concept's one name.
2. **How a name is made**: lowercase, words joined by `_`; an axis and a
   variable named for one point (singular), an attribute that lists
   several plural (`spins`, `partitions`); a word whole (`index`, never
   `idx`) but for an abbreviation its field writes itself, used one way in
   every file and said in its variable's words — `scf` (self-consistent
   field), `dos` (density of states), `iv`, `ir`, `mo` (molecular orbital),
   `homo`, `zpe` (zero-point energy), `vib`, `fc` (force constant), `es`
   (electronic structure), `eq` (equilibrium), `neq` (non-equilibrium),
   `negf`, `tbt` (TBtrans), `vha`, `kt` (k_B T), `ddmax` / `dhmax`
   (SIESTA's own), `imag` (imaginary), `fd` (finite difference), `iscf`,
   `max`, `mtime`, `n_` (the number of). A count says what it counts
   (`scf_cycles`, `n_modes`, `removed_motions_count`). An axis or a variable of a physical
   quantity ends with its unit, spelled as § 2c's last column says
   (`energy_ev`, `bias_v`, `temperature_k`, `dos_total_per_ev`); the axis of
   counted things names the thing (`atom`, `mode`, `frame`); a subset of a
   counted thing is `<subset>_<thing>` (`free_atom`, `device_atom`,
   `device_orbital`), its coordinate the thing's own index. **One concept
   keeps one name in every file**: the converged total energy is
   `total_energy_<unit>` — E_KS of its last SCF cycle, converged where its
   `scf_converged` says so (an SCF row's own, `scf_energy_ev`); the Fermi
   level `fermi_level_ev`; a run's SCF count `scf_cycles`; a stage
   `stage`; the one spin channel of a calculation that is not
   spin-polarized `unpolarized`. A name a group's prefix scopes keeps it
   where values of several owners sit together (the transport summary's
   `device_…` and `point_state`, the optimization file's `run_state`). **A
   record a file keeps whole as JSON text** (`current_means`,
   `fermi_frames`) keeps the keys its own contract gives it; every other
   name — variables, attributes, axes — is the file's and follows this
   rule, a value whose record key differs stating that key
   (`ncdata.Var.key`; § 9 gives it). The one writer refuses a number whose
   name does not end with its unit (`ncdata.UNIT_IN_NAME`).
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
   of every file that uses it. A reader that reads its older files as they
   are (the vibration result's, the optical calls' — § 1) keeps reading
   the old name as the new one, each variable stating the names older
   files wrote it under (`ncdata.Var.earlier`, read by `ncdata.named_in`);
   a file refused by its version (§ 3a) is written again by its command; a
   kept reading is out of date at another version and made again from its
   source (the digest, `model/parse.md` § 5e).

| axis | its points | its coordinate's values | units (— labels) | files |
|---|---|---|---|---|
| `atom` | the atoms | each atom's index by the file's atom index map — the user index in the transport, vibration, optical and optimization files, the engine index in the digest (`atoms_listed_by`), from 0 | 1 | transport, vibration (`/structure`), optical, optimization, digest |
| `free_atom` | the atoms a vibration moves | their atom indices, from 0 | 1 | vibration |
| `device_atom` | the atoms of TranSIESTA's device region | their atom indices, from 0 | 1 | digest |
| `atom_bound` | each atom's first orbital, and one past the last | from 0 to the atom count | 1 | digest |
| `device_orbital` | the device region's orbitals, one DOS column each | each orbital's index in the geometry, from 0; its atom beside it (`device_orbital_atom`) | 1 | digest |
| `orbital` | the junction's basis orbitals, every atom's, in the device run's order | each orbital's index in the geometry, from 0 — the digest's `device_orbital` names a subset of them; its atom, quantum numbers and type beside it (`orbital_atom` …) | 1 | transport (`/structure`) |
| `xyz` | a Cartesian component | `x`, `y`, `z` | — | transport, vibration, optical, optimization, digest |
| `cell_vector` | the cell's three vectors | `a`, `b`, `c` | — | transport, vibration, optical (`/structure`), optimization, digest |
| `reciprocal_vector` | the reciprocal cell's three vectors — a k-point's reduced components | `b1`, `b2`, `b3` | — | digest |
| `kpoint` | the k-points a reading averaged over | from 0 | 1 | digest |
| `frame` | a frame set's frames — a transport calculation's structures, a vibration's optical calls (the reference geometry, then each mode's two sides) | from 0, as MolView's API counts | 1 | transport, optical |
| `bias_v` | the voltages computed | each voltage | V | transport |
| `iv_bias_v` | the I–V's voltages | each voltage | V | transport |
| `energy_ev` | energy relative to the device's Fermi level, the zero TBtrans measures from — the leads' common Fermi level at 0 V; under a bias V the left lead's chemical potential sits at +V/2 on it and the right's at −V/2 | each energy, ascending | eV | transport, digest |
| `spin` | spin channels | `up` and `down`, or `unpolarized` — the one channel of a calculation that is not spin-polarized | — | transport, optical |
| `lead` | the two leads | their names as TBtrans names them | — | transport, digest |
| `region` | molbuilder's regions of the junction | `L-electrode`, `bridge`, `R-electrode` | — | transport |
| `eigenchannel` | transmission eigenchannels | from 0, as TBtrans orders them | 1 | transport, digest |
| `scf_cycle` | a run's SCF rows | from 0 | 1 | transport |
| `charge_part` | TranSIESTA's charge columns | `D`, `E1`, `C1`, … as its `ts-q:` header names them | — | transport |
| `mode` | vibrational modes | 1-based, ascending frequency | 1 | vibration, Raman view |
| `removed` | the rigid motions removed | from 0 | 1 | vibration |
| `mo` | molecular orbitals | from 0, ascending energy | 1 | vibration |
| `probed_mode` | the modes the electronic-structure probe ran on | each mode's 1-based index | 1 | vibration |
| `window` | an orbital's place in the probe's window | from 0, ascending energy | 1 | vibration |
| `temperature_k` | the thermochemistry's temperatures | each temperature | K | vibration |
| `fc_step` | a SIESTA force-constant run's steps | from 0, step 0 the reference geometry, then each displacement in SIESTA's order | 1 | vibration (`/partial_charges`) |
| `photon_energy_ev` | the light's energy, ħω — a concept apart from `energy_ev`, which is an electron's energy relative to the device's Fermi level at 0 V | each photon energy as SIESTA wrote it, ascending | eV | optical |
| `run` | a calculation's runs, stage by stage and each stage's starts in turn | each run's folder below the calculation (`01_coarse/run-0`) | — | optimization |
| `step` | those runs' steps, run by run | from 0; the run each is of beside it (`run_of_step`) | 1 | optimization |
| `voigt` | a stress tensor's six components | `xx`, `yy`, `zz`, `yz`, `xz`, `xy`, SIESTA's order | — | optimization |

**Files written before these names** read as rule 5 says: the vibration
data file's reader takes `@1`–`@3`'s names for the new ones, and `@1`–`@9`'s
for the names 2.72.3 gave by rule 2; the optical calls' reader `@1`–`@4`'s —
each by the variable's `earlier` names; a digest before `@5` is out of
date, read again from its `.TBT.nc`.

## 3. Writing — one writer

`ncdata.write_group(group, dims, variables, values, title=)` writes a group's
title, dimensions and variables from the table, each with its attributes and its
`coordinates`. **Every axis it creates or its variables use is a row of
§ 2a** (`ncdata.AXES`), with a coordinate of its own name in the row's units
— else it refuses, naming § 2a (`ncdata.check_axes`); a group states its own
axes, never one its parent created. **Every variable says what it is**
(`ncdata.check_said`, § 2): a long name, a definition, a number's unit
from § 2c, a raw one its source. **The file states itself** (`ncdata.state`, § 2b) before it is
closed; `ncdata.as_bytes` refuses a root or a group that does not, and an
attribute holding a real number. **The atom axis is given as its atom index map, never a
count** (`dims={"atom": atoms, …}`, an `AtomIndexMap`): the writer states
`atoms_listed_by`, `atom_index_base` and, where a column is missing,
`atom_index_why` as the group's attributes, and `user_atom_index` /
`engine_atom_index` as variables on the axis — each where it is known — and
refuses a bare number, or a map over another count. **The file is built in
memory** (`in_memory`, `as_bytes`) and handed whole to the package's one
atomic writer (`persist.write_bytes`): a reader sees the old file or the new
one, never half of either. Every NetCDF call — a write included — holds the
process's one lock (§ 4).

## 3a. One framework for what every file shares

*(The user, 2026-11-16: "make sure the code discipline of unified api and
framework level implementation is honored with no duplicated/redundant code
or design"; a review across the kinds found each of these written two to
seven ways.)*

**A structure, one layout in every file.** The geometry a file is about is
laid out by one door, `ncdata.structure_vars` with its values
`structure_values` and its attribute `structure_attrs` — written whole for a
file of one geometry by `write_structure_group`, its title saying which
geometry it is (the vibration result, the optical calls), and read back by
`read_structure_group`: in the group `/structure` (every variable and what
it is: § 9.0.5), the atom axis listed by the file's atom index map; `xyz`, `cell_vector`; each atom's `element` (a
label beside the atom axis), `mass_amu` (a value; the fill where the
structure states none) and `held` (the `frozen_atoms` label, a flag);
`position_ang` — over `atom`, `xyz` for one geometry, over `frame`, `atom`,
`xyz` in a file whose kind runs over frames (the transport data file, one
frame for one structure); `cell_ang` over `cell_vector`, `xyz`; `axis_kind`
over **`cell_vector`** — each lattice vector's periodicity, never a Cartesian
component's (§ 2a rule 1); and the attribute `structure_json`,
`Structure.to_dict()` whole, **its labels among it — never a second encoding
of them by hand**, read back as its dict by the framework. A kind adds its
own variables to the same group in the same write (the transport file: each
atom's region, the basis orbitals, each frame's rows); a reader asks the
structure's labels of `Structure.from_dict(structure_json)`. The
optimization file is not laid out so: its geometry is a trajectory, each
step's positions and each run's held atoms and cell, stated beside its steps
(`engines/optimization-postprocessing.md` § 6).

**A record held as text says so.** A variable whose text is a record — a
JSON list or object (a frame's rows, a run's phases, a point's digests) — is
declared so (`Var(…, json=True)`, written with the attribute `content`
`json`): its writer hands the records to the one writer, which writes each
as its text (`ncdata.write_group`, a record not stated the fill, the empty
text), and the one reader of variables (`ncdata.read_variables`) hands it
back as the record, the fill as `None`; no caller encodes or decodes one by
hand. The tables (§ 5) keep the text as the file holds it.

**An axis more than one file runs over has its coordinate from its row.**
Its row of `ncdata.AXES` carries its coordinate's meaning, once — the atoms
by the file's map; `xyz`, `cell_vector`, `reciprocal_vector` with their fixed
labels; `frame`, `energy_ev`, `spin`, `lead`, `eigenchannel`, `mode` — and
`ncdata.coordinate(axis, more=, source=)` makes the coordinate variable, a
file adding only where its points come from and, for a raw one, its source;
`ncdata.axis_values` gives a fixed axis's values. An axis one kind alone
runs over (a vibration's `fc_step`, a transport run's `scf_cycle`) has its
coordinate declared once, in that kind's writer.

**The engine, said once.** `ncdata.state(…, engine=, engine_version=,
computed_in=)` writes `source` in one sentence — *computed by SIESTA 5.4.2
in …*, *its version not stated by its output* where none is — and the
attribute `engine_version`, the version read by the one reader of an engine
output's build header (`siesta_reader.engine_version`); a file derived from
others states `derived_from=` — and, a view derived from files an engine
computed (the Raman view), both, in one sentence. One name,
`engine_version`, wherever a file states an engine's version (the optical
calls and the Raman view named it `siesta_version` until 2.72.2); a file
that states
only what it was derived from (the optimization file, a TBtrans digest)
states none.

**A file read back by one door.** Every reader opens a data file through
`ncdata.checked` — the one door (§ 4) and the schema checked by
`persist.check_schema`, the one door of the schema convention for every
record and every data file: a schema it does not read is refused by name,
the schemas it reads listed, and the way on where a command writes the file
again (the transport file's `summarize task`, the optimization file's and
the Raman view's export). A TBtrans digest is not refused: one of another
version is out of date and read again (`model/parse.md` § 5e). A file's
groups are walked by one walk, `ncdata.groups`, a group after its parent —
the tables (§ 5), the atom index door, the file's own statement. A file read
back as variables alone — the transport data file, the optimization file —
is `ncdata.read_file(path, schemas=)`: `{attrs, groups: {path: {attrs,
variables}}}`, the root under `""`, its records decoded and its flags as the
file stores them — `1` yes, `0` no, `-1` not stated (xarray masks the last
as NaN, a table leaves its cell empty): **test `== 1`**, which means yes on
every road. A file whose reader rebuilds a record — the vibration result's
payload, the optical calls' frames, the Raman view's modes — does that from
the same doors.

Each file moves its schema with this, and its chapter says which older ones
it still reads: the vibration result's and the optical calls', which a run
wrote and nothing writes again, are read as they are; the transport and
optimization files and the Raman view, which an export or `summarize`
writes, are refused by version, their writer's command writing them again.

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
whole into memory** but an export's bytes — what it composed, laid out as
tables, and the data files it copies, each read once and opened from those
same bytes (`ncdata.reading(path, data=)`), so what it copies is what was
read. **The door is the only opener** of a NetCDF file
in molbuilder's code — each data file's reader (`transport.datafile.
read_data_file`, `spectra.datafile.read_spectra_data`,
`spectra.optical_data.read_optical_data`, `spectra.raman.read_raman_data`,
`optimization.datafile.read_data_file`, the digest's), the
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
(`frame_bias_v_energy_ev.csv`), the variables of no dimension in its
`scalars.csv`, one row; a group's tables keeping its enclosing
groups' axes; each axis a column under its own name, each variable a column
named with its units, a value not stated an empty cell; and a `README.txt`
in the variables' own words — every variable's definition and source, the
file's attributes, the atom axis's `atoms_listed_by` and `atom_index_base`.
The atom axis's own table (`atom.csv`; a group's under its path, the transport file's `structure/atom.csv`) carries both index columns.

## 5a. An export — one door per kind, one route, one command

*(The user, 2026-11-16: "One /api/export"; "make sure the code discipline of
unified api and framework level implementation is honored with no
duplicated/redundant code or design".)*

**An export** is a calculation's results as they stand, as NetCDF — its data
files — or as CSV — their tables (§ 5) — composed on request in one shape
(`ncdata.Exported`, `ncdata.exported`, named by `ncdata.export_name`) and
written outside the calculation, never into it. **A kind that exports names
its module in the kind table** (`export`, `kind_table.export_of`), and the
module provides two doors: `export(calc, fmt, *, folder=None, **asks)`, the
composition — `folder` the folder it was asked at, which a vibration's
export reads as the result the viewer shows and the other kinds' do not
need — and `name_of(calc, fmt, **asks)`, its name, known before anything
is composed.

**What a kind's export asks beyond the form is declared data, in the kind
table** — a row each, `[<kind>.export_asks.<name>]` with its `type`
(`number`, `flag`) and its `words` (`kind_table.ExportAsk`): the transport
calculation's `dos_per_orbital`, the vibration's `laser_nm` and
`broaden_ev`. The table refuses, as it loads, a row named as a surface's
own (`path`, `format`, `to`, `bundle`, `folder`); and, as the export
resolves (`kind_table.export_of`), a row its `export` does not take by
name. **The command line and the page's route read those
rows**: `jobset export` offers an option for each (`--dos-per-orbital`,
`--laser-nm`, `--broaden-ev`), its help the row's words; the one route, `GET
/api/export?path=&format=&<asks>`, takes each row's name. Both read a value
through one door (`kind_table.export_asks_of`): **a number** as a finite number
— its meaning the kind's to judge (a laser's wavelength positive,
`spectra.raman.read_ask`) — **a flag** set by `1`, not by `0`; an ask with
no value left out. Both refuse, first, together and as the person wrote
them, the asks the calculation's kind does not take, naming the kinds that
take them; and a kind with no export (`kind_table.export_of`). The answer is the route's one
attachment (`_shared.sent_export`), what the export holds without said
beside it. *(The pages name their asks themselves until one export bar
draws them from the rows.)*

## 6. One import — `molbuilder.postprocess`

A person's script reads every data file with the functions molbuilder's own
doors ask (`model/parse.md` § 3, *One layer*): `read_transport_data`,
`read_spectra_data`, `read_optical_data` — each with its atom index map —
`read_optimization_data` for an exported optimization, also with its map,
`read_raman_data` for an exported Raman view, `infrared_view` and
`raman_view` for the spectrum viewer's two derivations, `atom_index_map(path)`,
`read_structure(path)` for a structure pair in the user order with the map
of the structure it returns, `parse(path)` for an engine's file in its own order
with the map on the answer, `read_steps(path)` for a run's steps in the user
order, `list_runs(calculation)` for its runs, `read_xv(path)` for SIESTA's own `.XV`, in the engine's order. `pdos(data_file, point, atoms=, labels=, regions=, orbitals=)` for the PDOS of a mixed selection at a transport point, from the data file and the point's digests beside it, summed as the report sums (`engines/transport.md` § 2a.12). A notebook's kernel imports the molbuilder its server
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
| `T.transport.nc` | the calculation's folder | every result in groups — `/structure` the junction whole (the structure layout every file shares, § 3a, with each atom's region), `/stages/<stage>` each once-run stage, `/points/<point>` each point's transmission and its `device` run, `/summary` what is compared across the points (the I–V, G and T(E) over the voltages) — each run's state and SCF history, the calculation's own records — one shape for one structure at one voltage too | written by `summarize task` from the record's one composition (`record.collect_record`), the atom axis **listed by user index** with each atom's engine index from the junction's order record; read by `read_data_file`, `postprocess.read_transport_data`, and the export | the variable record (§ 2), the one writer (§ 3), the one door (§ 4) |
| `T.transport-csv/` | where an export is asked to write it | the same as long tables, a group's under its path — `structure/atom.csv` with both index columns — and `README.txt` | `ncdata.as_tables` of the data file's bytes, composed on request — the export reads no data file (`engines/transport.md` § 2a.12, *Export*) | the tables (§ 5) |
| `T.transport-netcdf/` | where an export is asked to write it, with the DOS per orbital | the data file and, at the paths its points name, each kept digest as it is | the export's bytes copied, never read; read by `postprocess.pdos` from the folder, no `.TBT.nc` beside (`engines/transport.md` § 2a.12, *A PDOS of the atoms you choose, from the files*) | the one door's version rule, the reading a data file names (§ 4) |

**The chain, for one selection.** A person selects the bridge's first atom,
**user index 0**, on the report's Device view (it numbers it *#1*). The data
file states it: `/structure`'s `engine_atom_index[0]` is **9** — the device run's order.
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
its values on a grid. It is added as the files above were: a `Var` table in
its own module, every variable with its units and words, every axis a row of
§ 2a — an axis already named there used by its name, a new one added there
first — every unit a row of § 2c; its root's statement and each group's
title (§ 2b); one writer through
`write_group`, its atom axis — if it holds atoms — given as the map from the
door that knows the atoms' columns (`model/overview.md` § 2.2b, F2); a
`schema` string, versioned when a variable changes; its reader through
`reading`, refusing another schema by name; its row in the run files'
catalogue (`runfiles.WRITTEN`: its role, its writer, its door, `listed_by`)
-- a file written only by an export, outside every calculation
(`<label>.raman.nc`), has none, since no run's folder holds it; **its entry
in § 9, every group, variable and attribute with what it is, in the same
change**; and its chapter, which explains its science and links to that
entry. A name follows § 2a rule 2 and § 2c's spellings from its first
release.

## 9. The catalogue — every name a data file holds, where it is and what it is

*(The user, 2026-11-16: "check and update the documents about all the
metadata, the name of the axes or netcdf file items or categories, so that the
meaning of it is very clear. And we have a unified place to find where it is
and what it is.")*

**One place to look a name up.** Every file molbuilder writes (§ 1) is listed
here group by group, and every variable and attribute it holds is said once —
where it is and what it is. Each variable's words here are the file's own
(`long_name`, `definition`, written from its writer's `ncdata.Var` table),
said plainer where they can be, never differently: a name changes here, in
its writer's table and in the file's `schema` in one change (§ 2a rule 5). The
engine chapters keep the science and link here.

### 9.0 How to read it, and what every file shares

**A file** (9.1–9.6) opens with what it is, who writes it, its `schema` and the
chapter that explains its science. **Its groups** follow by path — `/` the
root, `/structure` a group below it, `<point>` a name that varies. **A group's
variables** are a table:

| column | says |
|---|---|
| `name` | the variable as the file names it |
| `over` | the axes it runs over (§ 2a), in order; — for a single value |
| `kind` | its `kind` attribute (9.0.2) |
| `units` | its `units` attribute, spelled as § 2c spells it; — for a text, which has none |
| `what it is` | what the value is and how it was obtained. **Record** — a JSON text the file's reader hands over decoded (`content` `json`). **Flag** — `1` yes, `0` no, `−1` not stated (test `== 1`, § 3a). **Atom indices** — its values count atoms by the file's atom index map (9.0.3). *From …* — the file a raw value was read from (its `source`). **Label** — a label beside an axis (9.0.4) |

**A name and its record key.** Where a file names a value otherwise than
the record it is written from keys it — a unit spelled as a name spells it
(§ 2c), a word made whole — the row gives the record's key: *the result's
key* `scf_energy_eh`. The readers of `molbuilder.postprocess` hand a value
back under that key — `read_spectra_data` the result's, `read_optical_data`
the calls' record's, `read_raman_data` the view's — and xarray or netCDF4
under the file's name.

**A group's attributes** follow: `attribute` | `what it says`, **record**
marking a JSON text its reader hands over decoded and *JSON text* one it
hands over as text. What every file, group or variable carries is said once
in 9.0 and only named where it appears.

**Axes and labels.** Every group holds the coordinate of each axis it runs
over, and the labels beside it, so a group read alone is whole (§ 2a rule 3).
A group lists them on one line — *Axes* and *Labels* — and describes them in
the group that is their home; the coordinates more than one file shares are
9.0.4.

#### 9.0.1 The file's own statement — every file's root (§ 2b, § 3a)

| attribute | on | what it says |
|---|---|---|
| `title` | the root, and every group | one sentence: what the file — the calculation and the result — or the group holds |
| `schema` | the root | which file this is, at which version: `molbuilder/<file>@<n>`; a reader refuses a version it does not read, by name |
| `molbuilder_version` | the root | the molbuilder that wrote the file — for a file a job writes (the vibration result, the optical calls), the one that ran the job |
| `source` | the root | how its data were made, one sentence: *computed by* the engine and its version *in* where it ran; *derived from* the files it was read from; or both, for a view derived from files an engine computed. *Its version not stated by its output* where the engine's output states none |
| `engine_version` | the root, where an engine computed the data; each run's group in the transport file | the engine's version, as its output's build header states it (`5.4.2` for SIESTA, `2.14.0` for PySCF); empty where its output states none. A file that states only what it was derived from (9.5, 9.6) has none |
| `Conventions` | the root | `CF-1.8` |
| `unit_meanings` | the root | *JSON text*: each unit spelling the file's variables use, and what it means in words (§ 2c) |
| `label` | the root, every file but the digest | the calculation's label, the `<label>` of its files' names |

#### 9.0.2 What every variable carries (§ 2, § 3a)

| attribute | on | what it says |
|---|---|---|
| `kind` | every variable | how the value was obtained: `given` — an input, as stated (a setting, a structure's value, a value cited from another record); `raw` — as the engine printed or wrote it; `measured` — read off the structures' coordinates (a frame set's rows, [`model/structure.md`](?doc=model/structure.md) § 2.2f); `derived` — molbuilder's computation, its formula in `definition`; `state` — what happened: a run's state, why a value is missing; `coordinate` — an axis's own values; a label beside an axis is of the kind its value is (§ 2) |
| `units` | every number | one spelling of § 2c — `1` for a count, an index, a fraction or a flag; a text has none |
| `long_name` | every variable | what it is, in a few words |
| `definition` | every variable | how it is obtained — the formula for a derived one — and what its fill value means there; a value cited from another record keeps that record's meaning (§ 2) |
| `source` | every raw variable, and any other read from a file | the file it was read from — `<label>` standing for the calculation's label, `<lead>`, `<L>-<R>` for a lead's name and the lead pair |
| `flag_values` · `flag_meanings` | every flag | `0 1` and `no yes`: an `int8`, `1` yes, `0` no; its fill, `−1`, is *not stated* |
| `_FillValue` | every number, and an integer that can be not stated | the value a cell holds where nothing is stated: NaN for a number; `−1` for a flag, and for a count or an index that can be not stated. A text has none: not stated is the empty text |
| `atom_index` | a variable whose values are atom indices | `1`: its values count atoms by the file's atom index map (9.0.3), from 0 |
| `content` | a text variable whose every value is a record | `json`: each value a JSON list or object, read back decoded; the empty text where none is stated |
| `coordinates` | a variable over an axis that has labels | the names of those labels — CF auxiliary coordinates — so xarray and netCDF4 open them beside it |

#### 9.0.3 The atom index map — which atoms a group's rows are (§ 3; [`model/overview.md`](?doc=model/overview.md) § 2.2b)

Every group with an `atom` axis states it — the attributes on the group, the
two indices as labels beside `atom`:

| attribute | what it says |
|---|---|
| `atoms_listed_by` | which index the `atom` axis lists its atoms by: `user_index` — row *i* is the atom of index *i* in the structure the user built or cited; `engine_index` — row *i* is atom *i* of the deck the engine ran; `not_stated` — the order is not known, read as by user index |
| `atom_index_base` | `0`: every atom index in the file counts from 0 |
| `atom_index_why` | where one of the two variables below is not written, why that index is not known — *a structure of its own (the electrode cell)*, *no run of ours holds this file*; absent where both are written |

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `user_atom_index` | atom | given | 1 | **label**: the atom's user index — its index in the structure the user built or cited, from 0; that structure's view numbers it index + 1 |
| `engine_atom_index` | atom | given | 1 | **label**: the atom's engine index — its index in the order the engine ran, its deck's, from 0 |

Where the engine ran the user order the two are equal, and both are written.
A variable marked **atom indices** counts by this map.

#### 9.0.4 The coordinates more than one file shares (§ 2a, § 3a)

Every dimension is a row of § 2a's axes, with a coordinate of its own name;
the coordinates of `atom`, `xyz`, `cell_vector`, `reciprocal_vector`, `frame`,
`energy_ev`, `spin`, `lead`, `eigenchannel` and `mode` are made from their row
(`ncdata.coordinate`), the same words in every file, and § 2a says what each
is. A file adds, in the coordinate's `definition`, where its points come
from — said in its home group. An axis one file alone runs over is described
there.

#### 9.0.5 A structure, one layout — `/structure` (§ 3a)

The geometry a file is about, in the vibration result (9.2), the optical calls
(9.3) and the transport file (9.1). The optimization file states its geometry
as a trajectory instead (9.5), the digest as its data file states it (9.6).

*Axes* `atom`, `xyz`, `cell_vector` (9.0.4) · *Labels* `element`,
`user_atom_index`, `engine_atom_index` (9.0.3)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `element` | atom | given | — | **label**: the atom's chemical symbol, in the file's atom order |
| `position_ang` | atom, xyz — in the transport file frame, atom, xyz | given | angstrom | each atom's position, in the file's atom order: the structure's coordinates — in the transport file every frame's |
| `mass_amu` | atom | given | amu | the atom's mass, the one the calculation used, as its result or its structure states it; NaN where neither does |
| `held` | atom | given | 1 | **flag** (`1` or `0`): 1 for an atom the structure holds fixed — its `frozen_atoms` label — 0 for a free one |
| `cell_ang` | cell_vector, xyz | given | angstrom | the cell: its three lattice vectors as rows; NaN where the structure has none |
| `axis_kind` | cell_vector | given | — | each lattice vector's periodicity — `periodic`, `isolated` or `transport` — as the structure states it ([`model/structure-periodicity.md`](?doc=model/structure-periodicity.md)) |

| attribute | what it says |
|---|---|
| `title` | which geometry this is (the file's section names it) |
| `atoms_listed_by` · `atom_index_base` · `atom_index_why` | 9.0.3 |
| `structure_json` | **record**: the structure whole, `Structure.to_dict()` — its elements, positions, cell, its labels (the held atoms among them as `frozen_atoms`), its identity columns and `info` — read back by `Structure.from_dict`; a label is read from here, never encoded a second time |

---

### 9.1 `<label>.transport.nc` — a transport calculation's results

Every result of a transport calculation, in groups: the junction, each stage
that runs once, each point and its device run, and a summary across the
points. Written by `jobset summarize task` beside `<label>.transport.json`,
and composed the same way by an export; schema `molbuilder/transport-data@12`
— another is refused, `summarize task` writing it again. The science — the
ladder, the bias treatment, how each number is computed — is
[`engines/transport.md`](?doc=engines/transport.md) § 2a.12; reading it in a
notebook, [`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md).

```text
/                              the calculation's records -- attributes alone
/structure                     the junction: its atoms, cell, basis orbitals, every frame and its rows
/stages/<stage>                seed, electrode_L, electrode_R -- each stage that runs once
/points/<point>                v0, v0.2, v-0.2 ... -- each point the description asks for, done or not
/points/<point>/device         the device run that point's transmission gathered
/summary                       what is compared across the points, over frame x bias_v
a frame set's:
/points/<frame>                f000, f001 ... -- one group per frame, no variable
/points/<frame>/<point>        f000/v0.2 ... -- each point, as above, and its device
```

**One table, cut into groups.** A value over `frame` × `bias_v` sits in each
point's group at that point, and `/summary` gathers some of them over the whole
grid from the same array — `/summary`'s `current_a` at 0.2 V and
`/points/v0.2`'s `current_a` are one number. The summary, where every
point's and every device's values sit together, names a value by its owner
where its group does not — `/summary`'s `point_state` is a point's `state`,
its `device_fermi_level_ev` the device group's `fermi_level_ev`; a stage's
values are its group's, the stage's name the group's. A point not done holds every variable at its fill, `done` `0`.

**A frame set** — a calculation citing several structures, a mode's frames
([`model/structure.md`](?doc=model/structure.md) § 2.2f) — adds the level
`/points/<frame>` (title *the points of frame f000*, no variable), `frame`
longer than 1, and *the frames' part*: each frame's rows and the set's numbers
in `/structure`, the frames' average in `/summary`, and the set's records on
`/` — every one of its variables written for one structure too, at its fill.

#### `/` — the calculation's records

No variable.

| attribute | what it says |
|---|---|
| `title` · `schema` · `molbuilder_version` · `source` · `Conventions` · `unit_meanings` · `label` | 9.0.1 — `source` *computed by SIESTA <version> in the calculation's runs — its TranSIESTA and TBtrans*; each run named in `provenance_chain` and in a point's `folder` |
| `engine_version` | 9.0.1: the SIESTA version every run's output states — the versions joined by *and* where runs state more than one |
| `treatment` | how the voltages were computed ([`engines/transport.md`](?doc=engines/transport.md) § 2a.10): `single-bias`, `self-consistent` — the device converged at every voltage — or `low-bias-approximation` |
| `treatment_note` | what that treatment means, one sentence |
| `low_bias_approximation` | `1` or `0`: whether the I–V was computed from the 0 V transmission alone |
| `iv_computed` | how `iv_current_a` was computed: `tbtrans` — each point's own current — or `low-bias-approximation` — the integral over each frame's 0 V transmission; on `/summary` too |
| `iv_notes` | **record**: the I–V's notes, each with the `frame` and the `bias_v` it is about — a voltage whose window the 0 V transmission does not reach; a frame's 0 V point not run (`bias_v` null); empty under `tbtrans`. On `/summary` too |
| `current_means` | **record**: what the current is, in words, for each spin option the points ran with, keyed as the calculation's record keys it (`non-polarized`, `polarized` — the spin option, not a channel's name — and `joined` for a point whose current is integrated over a shared grid) — the total of both spin channels |
| `energies_relative_to_fermi_level` | `1`: T(E)'s energies are relative to the device's Fermi level by TBtrans's own convention — asserted, not measured |
| `caveat` | the DFT-NEGF caveat: a plain GGA functional places the molecule's levels too close to E_F and overestimates a molecular junction's conductance — read T(E) for where the levels are and how they couple |
| `n_frames` | how many frames the cited structure has: 1 for one structure |
| `fermi_frames` | **record**: the Fermi levels, each in its run's own frame — the device's (`device_ef_ev`), the seed's (`seed_ef_ev`), each lead's (`lead_ef_ev`) — the record's own keys; the `ts-Vha` of the device's periodic start (`vha_ev`, printed in `vha_from`) that relates the device's zero to the seed's; the device's E_F in the seed's frame and its difference from the seed's; and the same in words (`note`) ([`engines/transport.md`](?doc=engines/transport.md) § 2a.12, *The Fermi levels*) |
| `scf_criteria` | **record**: each stage's SCF tolerances by phase — each criterion (`dDmax`, `dHmax`, `EDM`, `Eharris`, `FreeEng`), its tolerance, its unit and whether it is required |
| `provenance_chain` | **record**: each stage's run (`attempt`) and which file it gathered from which run; for the device, each point's run and the run it started from |
| `citation` | the structure the junction was composed from, as the description cites it: a saved structure's path, or a run's folder |
| `citation_kind` | which `citation` is: `pair` — a saved structure — or `run` |
| `description` | *JSON text*: the calculation's description, its `task.json`, as read when the file was written |
| `template` | the calculation's `<label>.template.toml`, as text |
| `job_request` | *JSON text*: the calculation's `job-request.json` |
| *the frames' part*, where the cited structure states them: | |
| `sampling_rule` | the rule, by name, that placed the frames along the mode and gave their weights (`gauss-hermite`) |
| `mass_source` | which masses `/structure`'s `mass_amu` holds (`standard atomic weights`) — the masses the position along the mode is measured with |
| `vibration_run` · `vibration_result_sha256` | the vibration result the frames were made from: its run's folder, and its JSON's sha256 |
| `structure_rows` | **record**: every row the frame set's structure states, as written — its name, value, unit and meaning (`note`) |
| `frame_warnings` | **record**: what is said of the frames' rows — their weights' sum off 1, a row written without its meaning; empty where nothing is |
| `average_assumptions` | **record**: what a harmonic mode's average assumes ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 7), where every frame states a weight |

#### `/structure` — the junction

The layout of 9.0.5, `position_ang` over `frame`, `atom`, `xyz` — every
frame's coordinates, the composed junction restored to the user order — and
`cell_ang` the composed junction's, the same for every frame. Listed by user
index; `engine_atom_index` is the device run's order, TranSIESTA's, so a point's
`.TBT.nc` is asked about through sisl with it. The junction adds its regions,
its basis and its frames:

*Axes* `atom`, `xyz`, `cell_vector` (9.0.4), `frame`, `orbital` · *Labels*
`element`, `atom_region`, `user_atom_index`, `engine_atom_index` (atom);
`frame_token`, `frame_rows` and each frame's rows (frame)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `frame` | frame | coordinate | 1 | a frame of the set, by its index from 0; molbuilder's pages and messages number it from 1 ([`model/structure.md`](?doc=model/structure.md) § 2.2e). 1 long for one structure |
| `frame_token` | frame | given | — | **label**: the frame's folder name — `f000` for frame 0 ([`engines/transport.md`](?doc=engines/transport.md) § 2a.11); empty for one structure, which has no frame folder |
| `atom_region` | atom | given | — | **label**: the region molbuilder owns that the atom lies in — `L-electrode`, `bridge`, `R-electrode` — from the composed junction's labels; empty for an atom in none. A point's `dos_region_per_ev` sums by it |
| `orbital` | orbital | coordinate | 1 | an orbital of the junction's basis — every atom's, in the device run's order — by its index in the geometry, from 0; a digest's `device_orbital` (9.6) names a subset of these. 0 long where `orbitals_why` says why |
| `orbital_atom` | orbital | given | 1 | **atom indices**: the atom the orbital sits on, by user index |
| `orbital_n` · `orbital_l` · `orbital_m` · `orbital_zeta` | orbital | raw | 1 | the orbital's principal quantum number, angular momentum, magnetic number and zeta, as the device run's basis states them (n, l, m, z) — *from the device run's `<label>.ORB_INDX`* |
| `orbital_type` | orbital | derived | — | its type from l and m, in the words a selection's orbital menu uses — `s`, `px`, `py`, `pz`, `d…`; empty for an l the menu does not offer |

*The frames' part* — each frame's rows, labels beside `frame` (in `/summary`
too), and the set's numbers. Each is NaN where the frame, or the set, states
none; its `definition` is the meaning the frame set writes for it (its
`note`), the vocabulary's of [`model/structure.md`](?doc=model/structure.md)
§ 2.2f where it writes none:

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `frame_rows` | frame | given | — | **label**, **record**: every row the frame states, as written — name, value, unit and meaning |
| `displacement_ang` | frame | measured | angstrom | **label**: the frame's plain displacement from the first frame, √(Σ_A \|R_A − R0_A\|²) over every atom — no masses |
| `max_atom_displacement_ang` | frame | measured | angstrom | **label**: the largest single atom's displacement from the first frame, max_A \|R_A − R0_A\| |
| `q_amu12_ang` | frame | derived | amu^1/2 angstrom | **label**: the frame's position along the mode, the mass-weighted normal coordinate q — every atom displaced by q L_A, Σ_A m_A \|L_A\|² = 1 over the `mass_amu` masses; \|q\| = √(Σ_A m_A \|R_A − R0_A\|²), its sign the side of the first frame it sits on |
| `node_sigma` | frame | derived | 1 | **label**: the same position in units of the spread, q / σ |
| `weight` | frame | given | 1 | **label**: the frame's share of the mode's average — the sampling rule's weight at this frame's node, the weights summing to 1 |
| `mode_index_1based` | — | given | 1 | which mode of the vibration result the frames sample, 1-based as the result counts them |
| `frequency_cm1` | — | given | cm-1 | that mode's wavenumber, as the vibration result states it |
| `temperature_k` | — | given | K | the temperature of the motion the frames sample — the junction's, or for a mode pumped out of equilibrium its effective temperature; the input that sets the spread |
| `mean_quanta` | — | derived | 1 | the mode's mean number of vibrational quanta at that temperature, n = 1 / (exp(ħω/k_BT) − 1) |
| `zero_point_amplitude_amu12_ang` | — | given | amu^1/2 angstrom | the mode's zero-point amplitude Q_zp = √(ħ/2ω) in q, as the vibration result states it |
| `sigma_amu12_ang` | — | derived | amu^1/2 angstrom | the spread σ of the Gaussian in q the frames sample: σ = Q_zp √(2n + 1) |
| `sampling_nodes` | — | given | 1 | m, the rule's number of nodes, one frame at each (the frame generator writes 3, 5 or 7) |
| `max_displacement_ang` | — | measured | angstrom | the outermost move: the largest single atom's move at the outermost frames — the check against a bond length |

| attribute | what it says |
|---|---|
| `title` | *the junction of <label> as the calculation composed it, listed by user index …* |
| `atoms_listed_by` · `atom_index_base` | 9.0.3: `user_index`, `0` |
| `structure_json` | 9.0.5 — the junction whole, every label it carries among it: the regions and the person's own |
| `orbitals_why` | why the orbital table is empty — no junction composed yet, no device run holding its `.ORB_INDX` yet, or what did not read; empty where the table is there |

#### `/stages/<stage>` — each stage that runs once

`/stages` holds a group per stage that runs once — `seed`, `electrode_L`,
`electrode_R` — and only its `title`.

*Axes* `scf_cycle`

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `scf_cycle` | scf_cycle | coordinate | 1 | an SCF row's index within its run, from 0 — every row of the run's last step, a device's periodic start and its NEGF loop alike; a run of fewer rows padded with the fill |
| `state` | — | state | — | the stage's run state, as molbuilder words a run's state (`finished`, `failed`, `not run`, `pending` …) |
| `detail` | — | state | — | molbuilder's sentence for that state |
| `scf_converged` | — | state | 1 | **flag**: whether the stage's SCF converged |
| `total_energy_ev` | — | raw | eV | the stage's total energy: the Kohn–Sham energy E_KS of its run's last SCF cycle — converged where `scf_converged` says so — *from the stage's SIESTA output* |
| `fermi_level_ev` | — | raw | eV | the stage's Fermi level at its last SCF cycle, in its own run's frame — how the frames relate is `fermi_frames` (`/`) — *from the stage's SIESTA output* |
| `scf_cycles` | — | state | 1 | how many SCF rows the run holds — the rows beyond are the fill; `−1` where none ran (`0` is a count) |
| `scf_iscf` | scf_cycle | raw | 1 | SIESTA's own cycle number (iscf), from 1 in each phase; `−1` fill |
| `scf_energy_ev` | scf_cycle | raw | eV | the cycle's Kohn–Sham energy E_KS |
| `scf_delta_e_ev` | scf_cycle | derived | eV | E_KS less the cycle before's, in the same phase; 0 at a phase's first cycle |
| `scf_ddmax` | scf_cycle | raw | 1 | dDmax — the density matrix's largest change |
| `scf_dhmax_ev` | scf_cycle | raw | eV | dHmax — the Hamiltonian's largest change |
| `scf_fermi_level_ev` | scf_cycle | raw | eV | the cycle's Fermi level Ef |
| `scf_phases` | — | state | — | **record**: the run's phases (`periodic`; a device's `negf` too) — for each, what it had to reach and how it ended: its cycles and whether it converged |
| `copied_from` | — | state | — | the run this stage's newest run is a copy of, in a calculation that reuses a finished one ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16); empty otherwise |

The `scf_*` rows over `scf_cycle` are *from the run's SIESTA output*.

| attribute | what it says |
|---|---|
| `title` | *the stage seed: its run's state, energy, Fermi level and SCF history* |
| `engine_version` | 9.0.1 — that run's SIESTA |

#### `/points/<point>` — a point: its transmission, its DOS, its current

A group per point the description asks for, named as its folder is — `v0`,
`v0.2`; in a frame set below its frame, `f003/v0.2`.

*Axes* `energy_ev`, `spin` (home `/summary`), `region`, `lead`,
`eigenchannel`, `atom` · *Labels* `element`, `atom_region`,
`user_atom_index`, `engine_atom_index` (atom)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `region` | region | coordinate | — | the regions molbuilder owns: `L-electrode`, `bridge`, `R-electrode` |
| `lead` | lead | coordinate | — | a lead, by its name: `L`, `R` (§ 2a) |
| `eigenchannel` | eigenchannel | coordinate | 1 | 9.0.4 — as many as the point's files hold, those TBtrans was asked for (`TBT.T.Eig`); none where no file is read |
| `frame` | — | given | 1 | the point's frame, from 0 — `/summary`'s `frame` at this point (0 for one structure) |
| `bias_v` | — | given | V | the point's voltage — `/summary`'s `bias_v` at this point |
| `done` | — | state | 1 | **flag** (`1` or `0`): 1 when the point has its transmission; 0 otherwise, its transmission's values the fill — a gap, never filled in. Its device group holds its device run's values wherever that ran |
| `state` | — | state | — | the transmission run's state, as molbuilder words a run's state (`finished`, `failed`, `not run`, `pending` …) |
| `folder` | — | state | — | the transmission point's folder, relative to the calculation's: where its raw values were read — a point that shares a finished point's energies holds those in its `shared/` folder; for a point not done, where its attempt is |
| `why` | — | state | — | for a point not done, why, in its run's words — pending, or failed and what stopped it |
| `progress` | — | state | — | for a point not done, how far its TBtrans loop got — running, or where it stopped — in its run record's sentence ([`model/parse.md`](?doc=model/parse.md) § 5d.1b) |
| `transmission_by_spin` | spin, energy_ev | raw | 1 | T(E), k-averaged, each spin channel's — *from `<label>.TBT.AVTRANS_<L>-<R>` (`TBT_UP` / `TBT_DN` polarized)* |
| `transmission` | energy_ev | derived | 1 | the transmission G is computed from: a non-polarized point's one channel, a polarized point's mean of the two, (T_up + T_down) / 2 |
| `conductance_g0` | — | derived | G0 | G / G0 = T at E − E_F = 0, linearly interpolated, G0 = 2e²/h; under a bias, T(0) of that bias's transmission — neither I/V nor dI/dV |
| `current_by_spin_a` | spin | raw | A | the current TBtrans printed, each channel's own — a non-polarized point's one channel, half the junction's total; the fill for a point whose energies two runs computed — *from the transmission point's TBtrans output* |
| `current_a` | — | derived | A | the junction's total current, as `current_from` says: `tbtrans` — twice the printed figure unpolarized, the channels' sum polarized; `joined` — the record's integral over the point's joined grid |
| `current_from` | — | state | — | which current `current_a` is: `tbtrans` or `joined` ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) |
| `dos_total_per_ev` | energy_ev | raw | 1/eV | the device's Green-function DOS, k-averaged: a non-polarized point's one spin's, a polarized point's up + down — *from `<label>.TBT.AVDOS`* |
| `dos_total_by_spin_per_ev` | spin, energy_ev | raw | 1/eV | the same DOS, each spin channel's own — *from `<label>.TBT.AVDOS` (`TBT_UP` / `TBT_DN` polarized)* |
| `dos_region_per_ev` | region, energy_ev | raw | 1/eV | the device's DOS by region: the point's DOS per orbital summed over the orbitals of the region's atoms its DOS covers (`atom_region`, `dos_atom` 1); spin as `dos_total_per_ev`; the fill where the reading is not kept — *from the point's `<label>.TBT.digest.nc` (9.6)* |
| `lead_spectral_dos_per_ev` | lead, energy_ev | raw | 1/eV | each lead's spectral DOS: the device's DOS of the states coming in from that lead, k-averaged by TBtrans. TBtrans writes it for every lead but the last; the last lead's row is the fill (`TBT.DOS.A.All` not set) — *from `<label>.TBT.AVADOS_<lead>`* |
| `lead_bulk_dos_per_ev` | lead, energy_ev | raw | 1/eV | a lead's bulk DOS (`TBT.DOS.Elecs`), k-averaged; spin as `dos_total_per_ev` — *from `<label>.TBT.AVBDOS_<lead>`* |
| `eigenchannel_transmission` | spin, eigenchannel, energy_ev | raw | 1 | each eigenchannel's transmission (`TBT.T.Eig`), k-averaged, each spin channel's own — *from `<label>.TBT.AVTEIG_<L>-<R>`* |
| `dos_atom` | atom | state | 1 | **flag**: 1 for each atom of TBtrans's device region — the only atoms it computes a DOS for — 0 for every other, from the point's kept reading; `−1` where none is kept |
| `dos_per_orbital_files` | — | state | — | **record**: the point's digests — each one's path from the calculation's folder, its spin channel, and the version of its `.TBT.nc` it states, [size in bytes, modification time in ns] — the reading `dos_region_per_ev` was summed from, which `postprocess.pdos` reads back; empty unless every reading of the point is kept |
| `dos_per_orbital_reading` | — | state | — | the state of the point's DOS-per-orbital reading: `kept`, or each data file's state in words — not read yet, out of date, being read by …, or a reading just tried that failed ([`model/parse.md`](?doc=model/parse.md) § 5e); `dos_region_per_ev` is the fill where it is not kept |
| `dos_why` | — | state | — | for a done point with no DOS, why, in its reader's words; its DOS variables the fill |
| `tbt_kpoints` | — | raw | 1 | how many transverse k-points the transmission used; `−1` where not stated — *from the transmission point's TBtrans output* |
| `tbt_kpoint_method` | — | raw | — | how those k-points were chosen, as TBtrans states it — *from its output* |
| `loop_elapsed_s` | — | raw | s | the transmission's loop time: its passes' k-point × energy loops summed, as TBtrans measured them — *from its output* |
| `copied_from` | — | state | — | the run this point is a copy of, in a calculation that reuses a finished one ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16); empty otherwise |
| `shared_from` | — | state | — | the finished point whose energies this point shares, by its folder ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16); empty for a point that shares none |
| `shared_held` | — | state | 1 | how many energies that finished point holds; `−1` where the point shares none |
| `shared_computed` | — | state | 1 | how many energies this point's own run computes beside them; `−1` where it shares none |

| attribute | what it says |
|---|---|
| `title` | *the point 0 V: its transmission, its DOS, its current and its state* |
| `atoms_listed_by` · `atom_index_base` | 9.0.3 |
| `engine_version` | 9.0.1 — the transmission run's TBtrans |

#### `/points/<point>/device` — the point's device run

The device run the point's transmission gathered — wherever it ran, the
transmission done or not.

*Axes* `charge_part`, `scf_cycle` (home `/stages/<stage>`)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `charge_part` | charge_part | coordinate | — | a column of TranSIESTA's `ts-q:` line, as its header names it: `D` the bridge — TranSIESTA's device region; `E1`, `E2` each electrode block; `C1`, `C2` the overlap across each electrode's boundary with the bridge; `O` the rest; `B` the buffer atoms, where there are some ([`engines/transport.md`](?doc=engines/transport.md) § 2a.12, *The device's charges*) |
| `state` · `detail` | — | state | — | the device run's state, as molbuilder words a run's state (`not run` among them), and molbuilder's sentence for it |
| `scf_converged` | — | state | 1 | **flag**: whether the device run's SCF converged |
| `fermi_level_ev` | — | raw | eV | the device's Fermi level at its NEGF phase's last cycle, in TranSIESTA's frame |
| `vha_ev` | — | raw | eV | the boundary Hartree potential `ts-Vha` TranSIESTA printed at its NEGF phase's last cycle — a value, not the shift between the device's zero and the seed's, which is the periodic start's (`fermi_frames`) |
| `total_energy_ev` | — | raw | eV | the NEGF phase's total energy: E_KS of its last SCF cycle, converged where `scf_converged` says so — never compared across points or calculations ([`engines/transport.md`](?doc=engines/transport.md) § 1.1) |
| `negf_cycles` | — | raw | 1 | how many NEGF cycles ran — the NEGF rows of its SCF history, after the periodic start; `−1` fill |
| `dq_e` | — | raw | e | the last NEGF cycle's `dQ`: every column of its `ts-q:` line summed, less the neutral count — the charge the open boundaries did not conserve |
| `charge_e` | charge_part | raw | e | the last NEGF cycle's Mulliken populations, column by column as TranSIESTA printed them: a net population for a region, an overlap population across a boundary — electrons counted, positive, not a net charge |
| `d_change_from_0v_e` | — | derived | e | D(V) − D(0 V): the bridge's net Mulliken population (`charge_e` at `D`) at this voltage less the same frame's at 0 V — positive where the bridge gained electrons; `0` at 0 V itself; the fill where the frame has no finished 0 V point |
| `contour_poles` | — | raw | 1 | the equilibrium contour's pole count, as TranSIESTA echoed it |
| `neq_emin_ev` · `neq_emax_ev` | — | raw | eV | under a bias, the non-equilibrium line's lower and upper ends |
| `neq_points` | — | raw | 1 | under a bias, the non-equilibrium line's points |
| `scf_cycles` · `scf_iscf` · `scf_energy_ev` · `scf_delta_e_ev` · `scf_ddmax` · `scf_dhmax_ev` · `scf_fermi_level_ev` · `scf_phases` | | | | the run's SCF history and phases, as a stage's (`/stages/<stage>`) |
| `scf_dq_e` | scf_cycle | raw | e | on a NEGF row, its `ts-q:` line's `dQ` — the charge the open boundaries did not conserve, not a change between cycles |
| `scf_vha_ev` | scf_cycle | raw | eV | on a NEGF row, the boundary Hartree potential `ts-Vha` |
| `scf_phase` | scf_cycle | state | — | the row's phase: `periodic` — the start — or `negf` |
| `scf_charge_e` | scf_cycle, charge_part | raw | e | every NEGF row's Mulliken populations, column by column as TranSIESTA printed them |

Every raw variable here is *from the device point's SIESTA output*.

| attribute | what it says |
|---|---|
| `title` | *the device run of the point 0 V — the one its transmission gathered: its NEGF figures, charges and SCF history* |
| `engine_version` | 9.0.1 — that run's SIESTA |

#### `/summary` — what is compared across the points

*Axes* `frame` (home `/structure`), `bias_v`, `energy_ev`, `spin`,
`iv_bias_v` · *Labels* `frame_token`, `frame_rows` and each frame's rows
(`/structure`)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `bias_v` | bias_v | coordinate | V | the voltages the device and the transmission ran at — the description's list, in its order: [0.0] for a single bias, a description stating none refused |
| `energy_ev` | energy_ev | coordinate | eV | 9.0.4 — the transmission's grid, at the full precision of each point's `.TBT.nc`, ascending; TBtrans's text files, printed to 5 decimals, put on it within half that decimal; a point that shares energies, its joined grid |
| `spin` | spin | coordinate | — | `unpolarized` — the one channel of a calculation that is not spin-polarized — or `up` and `down` (§ 2a) |
| `iv_bias_v` | iv_bias_v | coordinate | V | the I–V's voltages — the description's voltage list, in its order; under the low-bias approximation, more than ran |
| *gathered from the points*, at every frame × voltage: | | | | |
| `done` · `point_state` | frame, bias_v | state | 1 · — | a point's `done` and `state` |
| `conductance_g0` | frame, bias_v | derived | G0 | a point's |
| `current_a` · `current_from` | frame, bias_v | derived · state | A · — | a point's |
| `current_by_spin_a` | frame, bias_v, spin | raw | A | a point's |
| `transmission` | frame, bias_v, energy_ev | derived | 1 | a point's — T(E, V) over every frame |
| `device_fermi_level_ev` · `device_vha_ev` · `device_dq_e` | frame, bias_v | raw | eV · eV · e | its device group's `fermi_level_ev`, `vha_ev`, `dq_e` |
| `device_d_change_from_0v_e` | frame, bias_v | derived | e | its device group's `d_change_from_0v_e` |
| `device_scf_converged` | frame, bias_v | state | 1 | **flag**: its device group's `scf_converged` |
| *the I–V*: | | | | |
| `iv_current_a` | frame, iv_bias_v | derived | A | the I–V, as `iv_computed` says: `tbtrans` — each point's own `current_a`; `low-bias-approximation` — I(V) = G0 ∫ T(E, 0) [f(E − V/2) − f(E + V/2)] dE over that frame's 0 V transmission, E in eV, f the Fermi function at `iv_kt_ev`; the fill at a voltage whose window and Fermi tails reach past the transmission's energies (`iv_notes`) |
| `iv_current_by_spin_a` | frame, iv_bias_v, spin | derived | A | each spin's I–V current, a polarized calculation's, as the record states it; the fill for a non-polarized one, whose two spins carry half the total each |
| `iv_kt_ev` | — | given | eV | k_BT of the leads' Fermi functions in the low-bias integral — the device run's electronic temperature; NaN where the I–V is TBtrans's own |
| `iv_window_lowest_ev` · `iv_window_highest_ev` | — | derived | eV | the low-bias integral's range: the 0 V transmission's lowest and highest energy, E − E_F; NaN where the I–V is TBtrans's own |
| *the frames' average* — NaN, or the empty text, where there is none: | | | | |
| `average_transmission` | bias_v, energy_ev | derived | 1 | ⟨T(E)⟩ = Σ_f weight_f T_f(E) over every frame at its stated weight — the frame set's `sampling_rule` at its `sampling_nodes` frames |
| `average_delta_transmission` | bias_v, energy_ev | derived | 1 | ⟨T(E)⟩ − T_0(E): the average's change from frame 0's |
| `average_conductance_g0` | bias_v | derived | G0 | ⟨G⟩/G0 = Σ_f weight_f G_f/G0, at the same weights |
| `average_conductance_change_percent` | bias_v | derived | % | 100 (⟨G⟩ − G_f0)/G_f0, G_f0 frame 0's conductance |
| `average_why` | bias_v | state | — | why there is no average at this voltage: one structure, or frames that state no weights — the same at every voltage — or the frames whose point at this voltage is not done yet |
| `average_conductance_why` | bias_v | state | — | why the averaged conductance or its change is none: the window does not straddle E_F, or frame 0's conductance is zero |
| `weight_sum` | — | derived | 1 | Σ_f weight_f, checked against 1 within `weight_tolerance`; NaN where the frames state no weights |
| `weight_tolerance` | — | given | 1 | how far from 1 the weights' sum may be; NaN where the frames state no weights |

| attribute | what it says |
|---|---|
| `title` | *what is compared across the points …* |
| `iv_computed` · `iv_notes` | as the root's (`/`), kept here so the I–V reads alone |

---

### 9.2 `<label>.spectra.nc` — a vibration result

The vibration result as data — the same payload as `<label>.spectra.json`
beside it, tied to it by that JSON's sha256 (`json_sha256`). Written by the
job at every write of the result, on both engines (the result's one writer,
`write_spectra_payload`); schema `molbuilder/spectra-data@10` — which older
ones it reads, § 1. The science is [`engines/vibration.md`](?doc=engines/vibration.md)
§ 4 (PySCF) and § 5 (SIESTA); the result's keys, § 6.2–6.6.

```text
/                              the result's scalar blocks; four numbers
/structure                     the geometry the vibration is about (9.0.5)
/equilibrium                   the equilibrium SCF -- PySCF's; NaN and -1 on SIESTA
/modes                         the normal modes
/modes/electronic_structure    the probe -- always there, a row per mode PySCF's probe ran on
/thermo                        the thermochemistry at one temperature
/thermo/grid                   the same over temperature -- no temperature where none is drawn
/partial_charges               the force-constant run's charges -- SIESTA, where its deck asked for them
```

#### `/`

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `ir_fd_step_ang` | — | given | angstrom | the infrared's finite-difference step: each free atom moved by it either way, where the dipole's derivatives were taken by difference ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.6); NaN otherwise |
| `raman_fd_step_ang` | — | given | angstrom | the Raman's finite-difference step — the step the polarizability's derivatives were taken over, where they were taken by difference (on SIESTA the optical calls' δ); NaN otherwise |
| `relaxation_max_force_eh_per_bohr` | — | raw | Eh/bohr | the largest absolute Cartesian force component on a free atom at the geometry the Hessian is taken at — the force the relaxation is judged by ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.3) — *from the engine's forces at that geometry* · *the record's key* `max_force_eh_bohr` |
| `relaxation_max_force_all_atoms_eh_per_bohr` | — | raw | Eh/bohr | the same over every atom, held atoms included — *from the engine's forces at that geometry* · *the record's key* `max_force_all_atoms_eh_bohr` |

A key is written where the result states it.

| attribute | what it says |
|---|---|
| `title` · `schema` · `molbuilder_version` · `source` · `engine_version` · `Conventions` · `unit_meanings` · `label` | 9.0.1 — `source` *computed by pyscf 2.14.0* or *computed by siesta 5.4.2*: the engine as the result keys it (`siesta`, `pyscf`), where the other files write *SIESTA* |
| `engine` | the engine that computed it: `pyscf` or `siesta` |
| `json_sha256` | the sha256 of the `<label>.spectra.json` written with this file — the pair tied at every rewrite |
| `spectra_schema_version` | the schema version of that JSON result (`14`) — not this file's, which is `schema` |
| `timestamp` | when the result was written, UTC — the result's own time, carried as it is |
| `n_atoms_total` | how many atoms the structure has |
| `methods_text` | the Methods paragraph composed for the result ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.10) |
| `bibliography_keys` | **record**: the citation keys `methods_text` cites |
| `config` | **record**: how it was computed — PySCF's description config; on SIESTA the engine, the kind, the stage and the force-constant deck: its name and sha256, every keyword and block it states but the geometry, and each species' pseudopotential by name and sha256 ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.2) |
| `engine_metadata` | **record**: the engine's own facts — `{}` on PySCF; on SIESTA `fc_file`, `fc_displacement_ang`, `fc_range_engine_1based` (SIESTA's numbering, from 1), `fc_asymmetry_max_ev_ang2`, `reference_force_criterion_ev_ang` ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.2) |
| `runtime_info` | **record**: PySCF's CPU, threads, GPU and host |
| `optical` | **record**: the optical calls' citation — their data file (9.3) and its sha256, the light's direction, δ, the selection asked and applied, frame 0's state and each mode's frequency, δQ and its two sides' states ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.2); null on PySCF and where no optical call was planned |
| `phase_frequencies` · `phase_relaxation` · `phase_raman` · `phase_ir` · `phase_es` | each phase's state: `empty`, `running`, `complete` or `not requested` ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.9) |
| `ir_route` | the route the IR strengths came by: `analytic`, `finite-difference`, `charges` (SIESTA: from the stored charges) or `none` |
| `raman_route` | the route the Raman came by: `analytic`, `finite-difference`, `optical` (SIESTA: its optical calls, the strength a view, 9.4) or `none` |
| `hessian_scope` | `free` — second derivatives for the free atoms only — or `all` |
| `n_atoms_in_hessian` | how many atoms the Hessian covers; `−1` not stated |
| `hessian_density_fit` | `1`, `0`, `−1`: whether the Hessian itself was density-fitted; `−1` not stated (SIESTA) |
| `selected_mode_index_1based` | **record**: the modes, 1-based, the electronic-structure probe ran on (PySCF) · *the record's key* `selected_mode_idxs_1based` |
| `removed_motions_count` | how many whole-body motions were removed before diagonalising — `/modes`' `removed` |
| `relaxation_enabled` | `1`, `0`, `−1`: whether a relaxation ran before the frequencies — on SIESTA, the run's `relax` call |
| `relaxation_already_relaxed` | `1`, `0`, `−1`: the person's statement that the geometry was already relaxed |
| `relaxation_n_steps` | how many relaxation steps ran |
| `relaxation_converged` | `1`, `0`, `−1`: whether the judged force (`relaxation_max_force_eh_per_bohr`) is within the criterion |
| `relaxation_warning` | the warning shown when the geometry is not a stationary point; empty where none |
| `relaxation_record` | **record**: the relaxing stage's record whole — on SIESTA the `relax` call's; null where no call relaxed |

#### `/structure` — the geometry the vibration is about

The layout of 9.0.5: the equilibrium structure, the geometry the Hessian is
taken at, in the user order; `mass_amu` the masses the analysis weighted by;
`held` the result's held atoms. Listed by user index, `engine_atom_index` the
stages' deck order — held atoms first on SIESTA, the user order on PySCF.

#### `/equilibrium` — the equilibrium SCF

Always written; NaN and `−1` on SIESTA, a periodic engine with a Fermi level
and no HOMO.

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `mo` | mo | coordinate | 1 | a molecular orbital, by its index from 0, in ascending energy |
| `total_energy_eh` | — | raw | Eh | the SCF energy E_SCF at the equilibrium geometry; NaN where the engine reports none, and the orbital energies with it — *from the equilibrium SCF* · *the record's key* `scf_energy_eh` |
| `mo_energy_eh` | mo | raw | Eh | every molecular orbital's energy at the equilibrium SCF, ascending — *from the equilibrium SCF* |

| attribute | what it says |
|---|---|
| `title` | *the equilibrium SCF …* |
| `homo_index` | the HOMO's index on `mo`, from 0; `−1` where not stated · *the record's key* `homo_idx` |

#### `/modes` — the normal modes

*Axes* `mode`, `xyz` (9.0.4), `free_atom`, `removed`

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `free_atom` | free_atom | coordinate | 1 | **atom indices**: which atom each eigenvector row is — the free atoms' user indices, from 0, counting by `/structure`'s map |
| `removed` | removed | coordinate | 1 | a whole-body motion removed before diagonalising, from 0 |
| `frequency_cm1` | mode | derived | cm-1 | the harmonic frequency, as a wavenumber; negative for an imaginary mode |
| `has_imag` | mode | derived | 1 | **flag**: 1 for an imaginary frequency |
| `eigenvector_canonical_per_amu12` | mode, free_atom, xyz | derived | amu^-1/2 | the Cartesian eigenvector L, normalised so that Σ_atoms m\|L\|² = 1, m in amu — the form intensities and amplitudes pair with ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.3) · *the record's key* `eigenvector_canonical` |
| `eigenvector_display` | mode, free_atom, xyz | derived | 1 | the same, scaled so its largest absolute Cartesian component is 1 — for drawing only |
| `zero_point_amplitude_amu12_ang` | mode | derived | amu^1/2 angstrom | the zero-point amplitude Q_zp = √(ħ/2ω) in mass-weighted coordinates ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.6); NaN for an imaginary mode, which has none |
| `zero_point_displacement_ang` | mode, free_atom, xyz | derived | angstrom | each free atom's displacement at the zero-point amplitude, Q_zp L_canonical; NaN for an imaginary mode |
| `ir_intensity_km_per_mol` | mode | derived | km/mol | the IR intensity: the orientation average of \|dμ/dQ\|² ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.6); NaN where not computed — every SIESTA mode; 0 is a measured absence · *the record's key* `ir_intensity_km_mol` |
| `raman_activity_ang4_per_amu` | mode | derived | angstrom^4/amu | Placzek's orientation-averaged Raman activity 45a'² + 7γ'², from the polarizability's derivatives along the mode ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.6); NaN where not computed — every SIESTA mode, whose Raman is a view (9.4) · *the record's key* `raman_activity_a4_amu` |
| `dipole_derivative_d_per_ang_amu12` | mode, xyz | derived | D/(angstrom amu^1/2) | the dipole change along the mode, dμ/dQ: the engine's dμ/dR projected on the canonical eigenvector — the vector the IR intensity is the orientation average of; NaN where none was kept (SIESTA) |
| `ir_active` · `raman_active` | mode | derived | 1 | **flag**: the activity class's verdict for IR, for Raman ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.6); `−1` where the channel was not computed |
| `activity_class` | mode | derived | — | the mode's class, judged over the whole run ([`engines/vibration.md`](?doc=engines/vibration.md) § 6.6): `both`, `ir-only`, `raman-only`, `silent`, or `partial` — a channel not computed |
| `removed_motion` | removed, free_atom, xyz | derived | 1 | a removed whole-body motion: an orthonormal Cartesian pattern over the free atoms |

| attribute | what it says |
|---|---|
| `title` | *the normal modes …* |

#### `/modes/electronic_structure` — the electronic-structure probe ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.8)

Always written; a row per mode PySCF's probe ran on, none otherwise.

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `probed_mode` | probed_mode | coordinate | 1 | a probed mode, by its 1-based index in the result |
| `window` | window | coordinate | 1 | an orbital's place in the probe's window, from 0, ascending energy |
| `homo_index_in_window` | probed_mode | given | 1 | the HOMO's index among the window's orbitals, from 0 |
| `amplitude_ang` | probed_mode | given | angstrom | the push A each way along the mode: the mode's display form's largest Cartesian component moves by it |
| `total_energy_eq_eh` · `total_energy_minus_eh` · `total_energy_plus_eh` | probed_mode | raw | Eh | the total energy at equilibrium, displaced by −A, displaced by +A along the mode · *the record's keys* `scf_energy_eq_eh` · `scf_energy_minus_eh` · `scf_energy_plus_eh` |
| `mo_energy_eq_eh` · `mo_energy_minus_eh` · `mo_energy_plus_eh` | probed_mode, window | raw | Eh | the window's orbital energies, ascending, at equilibrium, at −A, at +A · *the record's keys* `mo_energies_eq_eh` · `mo_energies_minus_eh` · `mo_energies_plus_eh` |

Every raw variable here is *from the probe's SCFs*.

#### `/thermo` — the thermochemistry at one temperature ([`engines/vibration.md`](?doc=engines/vibration.md) § 4.7)

Its values where the result states the block.

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `temperature_k` | — | given | K | the temperature the values here are at · *the record's key* `temperature_K` |
| `pressure_atm` | — | given | atm | the pressure of the gas-phase translational term; NaN where the regime has none |
| `zpe_eh` | — | derived | Eh | the harmonic zero-point energy |
| `h_eh` | — | derived | Eh | the enthalpy as the regime sums it: `rrho` H at the temperature; `vibrational-only` the vibrational sums above the electronic energy, no kT — not an enthalpy |
| `s_eh_per_k` | — | derived | Eh/K | the entropy, as the regime defines it · *the record's key* `s_eh_k` |
| `g_eh` | — | derived | Eh | the free energy, G = H − TS |

| attribute | what it says |
|---|---|
| `title` | *the harmonic thermochemistry at one temperature …* |
| `regime` | `rrho` — the free molecule's rigid-rotor harmonic-oscillator thermochemistry as an ideal gas — or `vibrational-only` — the vibrational sums alone (SIESTA's) |
| `note` | the regime in words |
| `n_modes` | how many modes the result reports |
| `n_imag_excluded` | how many imaginary modes the sums leave out — they have no partition function |
| `n_rigid_removed` | how many whole-body motions were removed before diagonalising |

#### `/thermo/grid` — the thermochemistry over temperature

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `temperature_k` | temperature_k | coordinate | K | each temperature the curves are drawn at; none where the result draws no curves · *the record's key* `temperatures_K` |
| `zpe_eh` · `h_eh` · `s_eh_per_k` · `g_eh` | temperature_k | derived | Eh · Eh · Eh/K · Eh | as `/thermo`'s, at each temperature |
| `u_vib_eh` | temperature_k | derived | Eh | the vibrational thermal energy U_vib at each temperature, the zero-point energy excluded |

| attribute | what it says |
|---|---|
| `title` | *the thermochemistry drawn over temperature …* |
| `curves` | **record**: which of the curves the result holds; absent where it draws none |

#### `/partial_charges` — the force-constant run's charges (SIESTA)

Written where the result states them: a SIESTA run whose deck asked for the
charges (*Compute IR*); absent on PySCF.

*Axes* `fc_step`, `atom` (9.0.4) · *Labels* `user_atom_index`,
`engine_atom_index` (atom)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `fc_step` | fc_step | coordinate | 1 | a step of the force-constant run, from 0: 0 the reference geometry, then each displacement in SIESTA's order ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.4) |
| `fc_displacement_ang` | — | given | angstrom | the force-constant run's displacement: each step but 0 moves its atom by it, either way, along its axis · *the record's key* `displacement_ang` |
| `displaced_atom` | fc_step | given | 1 | **atom indices**: the atom the step displaces, by user index; `−1` at step 0 |
| `displaced_axis` | fc_step | given | — | `x`, `y` or `z`, the axis it is displaced along; empty at step 0 |
| `displaced_side` | fc_step | given | 1 | −1 or +1, the way the step moves it along the axis; 0 at step 0 |
| `hirshfeld_charge_e` · `voronoi_charge_e` | fc_step, atom | raw | e | each atom's Hirshfeld, Voronoi charge at each step: its valence charge less the electrons the partition gives it — positive where electrons left it — *from the run's output, its Hirshfeld / Voronoi Atomic Populations blocks*; one variable per partition `partitions` names |

| attribute | what it says |
|---|---|
| `title` | *the force-constant run's atomic charges at each of its steps …* |
| `atoms_listed_by` · `atom_index_base` | 9.0.3 |
| `partitions` | **record**: the partitions it holds — `hirshfeld`, `voronoi` |

---

### 9.3 `<label>.optical.nc` — a SIESTA vibration's optical calls

The record of a SIESTA vibration's Raman phase so far: every optical call's ε₂
on the photon energies, its mode, side and geometry, its state and SCF facts,
and the settings the calls read. Written at the run's root by the run's own
record steps — the force constants' (every frame pending), each optical
call's, and the finish — and rewritten as each call ends; cited by the result's
`optical` (9.2). Schema `molbuilder/optical-data@5` — which older ones it reads, § 1. The science: [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11
§ 8, [`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md) § 3.

#### `/`

No variable.

| attribute | what it says |
|---|---|
| `title` · `schema` · `molbuilder_version` · `source` · `engine_version` · `Conventions` · `unit_meanings` · `label` | 9.0.1 — `title` the calculation in words: ε₂ for a field along the direction `/settings` states, first-order perturbation over the self-consistent states with no local field, at the reference geometry and ±dQ along each selected mode; `source` *computed by SIESTA <version> in <run>* |
| `run` | the run whose calls the file records, its folder below the calculation (`01_freq/run-0`) |

#### `/settings` — what the calls were asked

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `delta_ang` | — | given | angstrom | δ: how far each displaced frame moves the mode's most-moving atom — dQ = δ / max_k \|L_k\|, max_k \|L_k\| the length of the largest atom's displacement vector in the mode ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 3) |

| attribute | what it says |
|---|---|
| `title` | *what the optical calls were asked …* |
| `keywords` | **record**: every optical keyword (`OpticalCalculation`, `Optical.*`) as the deck wrote it |
| `direction` | **record**: the light's polarization direction, three Cartesian components (`Optical.Vector`) |
| `selection` | **record**: the modes asked (`asked`: `all`, or a frequency `range` and its bounds) and applied (`applied`: their 1-based indices) |
| `deck_name` · `deck_text` · `deck_sha256` | the optical deck's name, its whole text and its sha256 |

#### `/structure` — the geometry the calls displace

The layout of 9.0.5: the force-constant run's reference geometry, in the user
order, with the result's masses.

#### `/frames` — one row per optical call

*Axes* `frame`, `atom`, `xyz` (9.0.4), `photon_energy_ev`, `spin` · *Labels*
`call` (frame); `user_atom_index`, `engine_atom_index` (atom)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `frame` | frame | coordinate | 1 | 9.0.4 — an optical call: frame 0 the reference geometry, then each selected mode's +dQ and −dQ sides, in the order the calls were planned |
| `photon_energy_ev` | photon_energy_ev | coordinate | eV | the light's energy ħω of each row, as SIESTA wrote it, `Optical.Broaden` / 10 apart — *from `<label>.EPSIMG`* |
| `spin` | spin | coordinate | — | 9.0.4 — the channels the optical deck's spin computes, in the SIESTA family's names (`unpolarized` for the one channel) |
| `call` | frame | given | — | **label**: the call's folder below the run: `optical/f000`, `optical/f001` … |
| `mode` | frame | given | 1 | the mode the frame is displaced along, by its 1-based index in the result; `−1` at frame 0 |
| `frequency_cm1` | frame | given | cm-1 | that mode's harmonic frequency; NaN at frame 0 |
| `side` | frame | given | 1 | +1 or −1, the side of the displacement along the mode; 0 at frame 0 |
| `dq_amu12_ang` | frame | given | amu^1/2 angstrom | the displacement along the mode in mass-weighted coordinates, dQ = δ / max_k \|L_k\| — the most-moving atom moves by δ; NaN at frame 0 |
| `position_ang` | frame, atom, xyz | raw | angstrom | the geometry the call computed at, in the user order; NaN for a call that wrote none — *from the call's output, `<label>_<stage>.out`* |
| `state` | frame | state | — | the call's state by the run's one done rule: `finished`; `not done` — it ran to its end and is not done; `failed` — it ran and ended with an error; `not run` — its run's walk ended before starting it; `pending` — not run yet, its row the fill |
| `computed_in` | frame | state | — | the run that computed the call, its folder below the calculation — this run's, or the one a warm launch took it over from; empty while pending |
| `scf_cycles` | frame | raw | 1 | how many cycles the call's SCF ran; `−1` where it wrote none — *from its output* · *the record's key* `scf_iterations` |
| `total_energy_ev` | frame | raw | eV | the Kohn–Sham energy E_KS of the call's SCF — *from its output* |
| `fermi_level_ev` | frame, spin | raw | eV | each channel's Fermi level at the call's last SCF cycle: its own (Ef_up, Ef_dn) under `Spin.Fix`, else the one Ef every channel shares — *from its output* |
| `engine_elapsed_s` | frame | raw | s | SIESTA's wall time, launch to exit, as the call's session log measured it — *from its benchmark line* |
| `f_sum` | frame, spin | raw | 1 | SIESTA's f-sum (Thomas–Reiche–Kuhn) check: V ∫ ω ε₂(ω) dω / (2π² N_e) over the range — near 1 when the interband transitions are in the range; the Drude term is left out of ε₂ — *from `<label>.EPSIMG`, its header* |
| `drude_wp2_eh2` | frame, spin | raw | Eh^2 | the Drude plasma frequency squared — the intraband term SIESTA computes beside ε₂ and leaves out of it — *from `<label>.EPSIMG`, its header* · *the record's key* `drude_wp2_ha2` |
| `eps2` | frame, photon_energy_ev, spin | raw | 1 | ε₂, the imaginary part of the dielectric function, for a field along `/settings`' `direction`, each transition spread as a Gaussian of `Optical.Broaden`; the width a person reads is applied when it is presented (9.4) — *from `<label>.EPSIMG`* |

| attribute | what it says |
|---|---|
| `title` | *one row per optical call …* |
| `atoms_listed_by` · `atom_index_base` | 9.0.3 |

---

### 9.4 `<label>.raman.nc` — a SIESTA vibration's Raman view

The Raman strengths of a SIESTA vibration's picked modes at a laser and width a
person named — per mode the polarizability's change ∂α/∂Q and the strength
R = \|∂α/∂Q\|², at the laser and static, and frame 0's α — derived from the
optical calls' ε₂ (9.3), independent states, no local field; a strength, not
an intensity (no laser-frequency or temperature factor). The `title` says the
derivation step by step. Written only by an export (`spectra.export`,
`jobset export … --laser-nm`), never into the calculation; schema
`molbuilder/raman-view@4`, any other refused — export the view again. The
science: [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 10,
[`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md) § 3.2a.

#### `/`

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `laser_nm` | — | given | nm | the laser's wavelength, as asked |
| `photon_energy_ev` | — | derived | eV | the laser's photon energy, hc / λ — where the polarizability is read |
| `broaden_ev` | — | given | eV | the width the spectrum was broadened to: s of SIESTA's Gaussian exp(−(E/s)²), full width at half height 2√(ln 2) s; the run's own when not asked, never below it |
| `run_broaden_ev` | — | given | eV | the run's own width, the optical deck's `Optical.Broaden` — the spectrum as SIESTA computed it |
| `delta_ang` | — | given | angstrom | δ, as the optical calls' (9.3 `/settings`) |
| `cell_volume_ang3` | — | derived | angstrom^3 | V, the reference structure's cell volume, in α = V/4π (ε − 1) · *the record's key* `cell_volume_a3` |
| `band_floor_ang4_per_amu` | — | given | angstrom^4/amu | the Raman channel's band floor: where the strongest mode's strength is at or below it there is no band, and `relative` is NaN for every mode — set on the Placzek activity's scale, which runs above R · *the record's key* `band_floor_a4_amu` |

| attribute | what it says |
|---|---|
| `title` · `schema` · `molbuilder_version` · `source` · `engine_version` · `Conventions` · `unit_meanings` · `label` | 9.0.1 — `source` *derived from `<label>.spectra.json`, `<label>.optical.nc`; computed by SIESTA <version> in <run>* |
| `run` | the run whose optical calls the view was derived from |
| `direction` | **record**: the light's direction, three Cartesian components |
| `spins` | **record**: the spin channels summed |
| `derived_from` | **record**: each file it was derived from, by name and sha256 |

#### `/equilibrium` — frame 0's polarizability

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `alpha_static_ang3` | — | derived | angstrom^3 | frame 0's polarizability, static: V/4π (ε₁ − 1) at zero frequency for light along `direction`, ε₁ by Kramers–Kronig over the stored spectrum, cut at its highest energy · *the record's key* `alpha_static_a3` |
| `alpha_laser_real_ang3` | — | derived | angstrom^3 | its real part at the laser: V/4π (ε₁ − 1) at the laser's frequency · *the record's key* `alpha_laser_real_a3` |
| `alpha_laser_imag_ang3` | — | derived | angstrom^3 | its imaginary part at the laser: V/4π ε₂ — zero off any electronic transition · *the record's key* `alpha_laser_imag_a3` |

#### `/modes` — each picked mode

*Axes* `mode` (9.0.4 — the result's numbering)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `frequency_cm1` | mode | given | cm-1 | the mode's harmonic frequency |
| `dq_amu12_ang` | mode | given | amu^1/2 angstrom | dQ, the displacement along the mode (9.3) |
| `dalpha_dq_static_ang2_per_amu12` | mode | derived | angstrom^2/amu^1/2 | ∂α/∂Q, static: [α(+dQ) − α(−dQ)] / 2dQ at zero frequency · *the record's key* `dalpha_dq_static_a2_per_amu12` |
| `dalpha_dq_laser_real_ang2_per_amu12` | mode | derived | angstrom^2/amu^1/2 | its real part at the laser: [α₁(+dQ) − α₁(−dQ)] / 2dQ, α₁ = V/4π (ε₁ − 1) · *the record's key* `dalpha_dq_laser_real_a2_per_amu12` |
| `dalpha_dq_laser_imag_ang2_per_amu12` | mode | derived | angstrom^2/amu^1/2 | the polarizability's change along the mode at the laser, its imaginary part: [α₂(+δQ) − α₂(−δQ)] / 2δQ, α₂ = V/4π ε₂ — zero off any electronic transition · *the record's key* `dalpha_dq_laser_imag_a2_per_amu12` |
| `strength_static_ang4_per_amu` | mode | derived | angstrom^4/amu | R = (∂α/∂Q)² at zero frequency, light along `direction` · *the record's key* `strength_static_a4_amu` |
| `strength_laser_ang4_per_amu` | mode | derived | angstrom^4/amu | R = \|∂α/∂Q\|² = real² + imaginary² at the laser · *the record's key* `strength_laser_a4_amu` |
| `relative` | mode | derived | 1 | the strength at the laser over the strongest such strength among the modes whose frequency is real (not imaginary); NaN for every mode when that strongest is at or below `band_floor_ang4_per_amu` |
| `state` | mode | state | — | `finished`, or — a side not done — `not ended`, `failed`, `not run` or `not done`; its numbers then the fill |
| `why` | mode | state | — | the side not done and its state (*the +δQ side failed*); empty when finished |

---

### 9.5 `<label>.optimization.nc` — an optimization's steps

Every step an optimization's runs finished, on one grid, and each run's own
record. Written only by an export (`jobset export`, `optimization.export`),
never into the calculation; schema `molbuilder/optimization-data@3`, any other
refused — export again. One group, `/`, and no `/structure`: the geometry is a
trajectory, each step's positions and each run's held atoms and cell (§ 3a).
The file holds no times. Its charges are what the runs printed: SIESTA prints
them where its deck asks, which an optimization's deck does not of itself — NaN
then.
The science and examples: [`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md)
§ 3, § 6.

#### `/`

*Axes* `atom`, `xyz`, `cell_vector` (9.0.4), `run`, `step`, `voigt` · *Labels*
`element`, `user_atom_index`, `engine_atom_index` (atom); `run_of_step`
(step); `stage_of_run` (run)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `run` | run | coordinate | — | a run, by its folder below the calculation (`01_coarse/run-0`): stage by stage and each stage's starts in turn — the order of the calculation's stages, not always the order they ran in; a run that gave no step is listed too, `said` saying why |
| `step` | step | coordinate | 1 | a step, from 0, run after run: each geometry step a run finished, and the final geometry a SIESTA relaxation ends on |
| `voigt` | voigt | coordinate | — | a stress component: `xx`, `yy`, `zz`, `yz`, `xz`, `xy`, SIESTA's order |
| `element` | atom | given | — | **label**: the atom's chemical symbol, in the user order |
| `run_of_step` | step | given | — | **label**: the run the step is of, as `run` names it |
| `stage_of_run` | run | given | — | **label**: the stage the run is of, by its name |
| *each step*: | | | | |
| `position_ang` | step, atom, xyz | raw | angstrom | each atom's position, the coordinates the engine ran, in the user order — they hold the move `applied_offset_ang` states |
| `force_ev_per_ang` | step, atom, xyz | raw | eV/angstrom | the force on each atom, in the user order; NaN where the step states none |
| `total_energy_ev` | step | raw | eV | the step's total energy: SIESTA's E_KS — its output prints it to 0.0001 eV, its `.MD.nc` history holds the steps between in full — or PySCF's SCF total energy; NaN where the step states none, as a SIESTA relaxation's last row, the final geometry |
| `max_force_ev_per_ang` | step | derived | eV/angstrom | the largest absolute Cartesian force component on any atom; NaN where the step states no force |
| `max_force_free_ev_per_ang` | step | derived | eV/angstrom | the same over the atoms the run moved — what a relaxation's tolerance tests; NaN where the step states no force or the run holds every atom |
| `hirshfeld_charge_e` · `voronoi_charge_e` | step, atom | raw | e | each atom's Hirshfeld, Voronoi charge: its valence charge less the electrons the partition gives it — positive where electrons left it — in the user order; NaN where the run printed none (SIESTA prints them where its deck asks) |
| `stress_kbar` | step, voigt | raw | kbar | the step's stress tensor, as the run printed it; NaN where it printed none |
| *each run*: | | | | |
| `continued_from` | run | given | — | the run it continued from, as `run` names it, by the run's own record; empty for a run that started from the structure |
| `output_file` | run | state | — | the file its steps were read from — the run's own output |
| `final_total_energy_ev` | run | derived | eV | the last energy of a step it finished, as `total_energy_ev` states it; NaN where none states one |
| `force_tolerance_ev_per_ang` | run | raw | eV/angstrom | the force tolerance the run's relaxation was to reach, as the run states it; NaN for a run that relaxed nothing |
| `force_within_tolerance` | run | derived | 1 | **flag**: whether the run's largest force component on the atoms it moved, at the last step stating one, is within its tolerance — SIESTA's own test; geomeTRIC tests each atom's force length beside other criteria. A run still going moves on: read it with `run_state`. `−1` for a run that relaxed nothing |
| `scf_converged` | run | state | 1 | **flag**: whether its last SCF converged, as the run says it; `−1` where it cannot say |
| `run_state` | run | state | — | how the run ended, as its file says: `ended`, `stopped`, `out_of_memory`, or `running` — the file shows no ending, which a slow run and one that died with no word both are |
| `said` | run | state | — | what the run says beside its numbers: why it holds no step here; a step still being computed or stopped inside, which is not among the steps; forces it does not compute |
| `held` | run, atom | raw | 1 | **flag**: whether the run's output says it holds the atom fixed |
| `applied_offset_ang` | run, xyz | given | angstrom | what prep moved the structure it was handed by: that structure plus this is where a first stage starts, and the deck already holds it; a later stage starts where the one before ended; NaN where the deck records none |
| `cell_ang` | run, cell_vector, xyz | given | angstrom | the cell the run ended in, a row per lattice vector: the output's own, else the box its deck placed the atoms in, which on an isolated axis is no lattice; NaN for none |
| `axis_kind` | run, cell_vector | given | — | each lattice vector's periodicity — `periodic`, `isolated` or `transport` — as the run's structure states it |

Every raw variable is *from the run's own output, named by `output_file`, and
for SIESTA's coordinates and energies the `.MD.nc` history beside it, for the
steps it holds*.

| attribute | what it says |
|---|---|
| `title` · `schema` · `molbuilder_version` · `source` · `Conventions` · `unit_meanings` · `label` | 9.0.1 — no `engine_version`; `source` *derived from <engine>'s output of each run, named by output_file* |
| `engine` | the engine the runs ran: `siesta` or `pyscf` |
| `atoms_listed_by` · `atom_index_base` | 9.0.3: `user_index`, `0` |

---

### 9.6 `<label>.TBT.digest.nc` — a TBtrans data file's kept reading

A transmission point's TBtrans data file `<label>.TBT.nc`, read once and kept
beside it: the device's DOS per orbital averaged over the k-points, what the
data file states of its geometry and its electrodes, and TBtrans's own
k-averaged quantities from its text files — one spin channel's, the data
file's. Written by the SIESTA family's parser of that file (`tbtrans-nc`) when
its DOS per orbital is read — on a person's word, or by `summarize task`; a
polarized point's two are `.TBT_UP.digest.nc` and `.TBT_DN.digest.nc`. Schema
`molbuilder/tbt-digest@5`; one of another version, or of another version of
its data file, is out of date and read again. **Listed by engine index** —
TBtrans's geometry, TranSIESTA's order. The science:
[`model/parse.md`](?doc=model/parse.md) § 5e; what the transport file takes
from it, 9.1's `dos_region_per_ev` and `dos_per_orbital_files`.

#### `/`

*Axes* `energy_ev`, `device_orbital`, `atom`, `atom_bound`, `xyz`,
`cell_vector`, `reciprocal_vector` (9.0.4), `kpoint`, `lead`, `device_atom`,
`eigenchannel` · *Labels* `device_orbital_atom` (device_orbital); `element`,
`user_atom_index`, `engine_atom_index` (atom)

| name | over | kind | units | what it is |
|---|---|---|---|---|
| `energy_ev` | energy_ev | coordinate | eV | 9.0.4 — TBtrans's energies at the data file's full precision, ascending — *from `<label>.TBT.nc`* |
| `device_orbital` | device_orbital | coordinate | 1 | an orbital of TBtrans's device region, by its index in the geometry, from 0 — one DOS column each, in sisl's sorted order |
| `device_orbital_atom` | device_orbital | raw | 1 | **label**, **atom indices**: the atom each device orbital sits on, by engine index |
| `atom_bound` | atom_bound | coordinate | 1 | the axis `first_orbital` runs over: one entry per atom and one past the last — atom *i*'s orbitals are `first_orbital[i]` up to `first_orbital[i+1]` − 1 |
| `first_orbital` | atom_bound | raw | 1 | the geometry's first orbital of each atom, from 0, and one past the last orbital — *from `<label>.TBT.nc`* |
| `device_atom` | device_atom | coordinate | 1 | **atom indices**: the atoms of TBtrans's device region, by engine index — *from `<label>.TBT.nc`* |
| `kpoint` | kpoint | coordinate | 1 | a k-point the reading averaged over, from 0, in the data file's order |
| `kpoint_reduced` | kpoint, reciprocal_vector | raw | 1 | each k-point, as fractions of the reciprocal vectors b1, b2, b3 — *from `<label>.TBT.nc`* |
| `kpoint_weight` | kpoint | raw | 1 | each k-point's weight, the data file's own; they sum to 1 — *from `<label>.TBT.nc`* |
| `lead` | lead | coordinate | — | 9.0.4 — as the deck's `TS.Elecs` block names them — *from `<label>.TBT.nc`* |
| `lead_mu_ev` | lead | raw | eV | each lead's chemical potential, on `energy_ev`'s scale — *from `<label>.TBT.nc`* |
| `lead_kt_ev` | lead | raw | eV | each lead's electronic temperature, k_BT of its Fermi distribution — *from `<label>.TBT.nc`* |
| `device_eta_ev` | — | raw | eV | η, the imaginary part added to the energy in the device's Green's function — *from `<label>.TBT.nc`* |
| `position_ang` | atom, xyz | raw | angstrom | each atom's position — *from `<label>.TBT.nc`* |
| `cell_ang` | cell_vector, xyz | raw | angstrom | the cell's three lattice vectors, one a row — *from `<label>.TBT.nc`* |
| `element` | atom | raw | — | **label**: each atom's element — which the data file does not state — read from the Hamiltonian TBtrans read, where it is there and is this geometry; absent otherwise (a point that shares a finished point's energies holds that point's in its `shared/` folder, which has no Hamiltonian) — *from `<label>.TS.HSX`* |
| `orbital_dos_per_ev` | energy_ev, device_orbital | derived | 1/eV | the device's DOS per device orbital, averaged over the k-points: each k-point's DOS times its weight `kpoint_weight`, summed in double precision, kept in TBtrans's single; absent where the data file holds no DOS — *from `<label>.TBT.nc`* |
| *TBtrans's own k-averages, each where its text file is there*: | | | | |
| `transmission` | energy_ev | raw | 1 | the k-averaged transmission, as TBtrans printed it — *from `<label>.TBT.AVTRANS_L-R`* |
| `dos_total_per_ev` | energy_ev | raw | 1/eV | the device's k-averaged DOS — *from `<label>.TBT.AVDOS`* |
| `eigenchannel` | eigenchannel | coordinate | 1 | 9.0.4 |
| `eigenchannel_transmission` | eigenchannel, energy_ev | raw | 1 | each eigenchannel's k-averaged transmission — *from `<label>.TBT.AVTEIG_L-R`* |
| `lead_spectral_dos_per_ev` | lead, energy_ev | raw | 1/eV | each lead's spectral DOS: the device's DOS of the states coming in from that lead, k-averaged by TBtrans. TBtrans writes it for every lead but the last; the last lead's row is the fill (`TBT.DOS.A.All` not set) — *from `<label>.TBT.AVADOS_<lead>`* |
| `lead_bulk_dos_per_ev` | lead, energy_ev | raw | 1/eV | each lead's k-averaged bulk DOS — *from `<label>.TBT.AVBDOS_<lead>`* |

Each value is the data file's channel's (`channel`), with no spin factor
(`spin`); the text files' names carry the channel's tag (`TBT`, `TBT_UP`,
`TBT_DN`).

| attribute | what it says |
|---|---|
| `title` · `schema` · `molbuilder_version` · `source` · `Conventions` · `unit_meanings` | 9.0.1 — no `engine_version`, no `label`; `source` *derived from* each file its variables were read from |
| `atoms_listed_by` · `atom_index_base` · `atom_index_why` | 9.0.3: `engine_index`, `0`; a data file no run of ours holds has no user index, and `atom_index_why` says so |
| `data_file` | the TBtrans data file read (`<label>.TBT.nc`) |
| `data_file_size` · `data_file_mtime_ns` | the version of the data file this is a reading of: its size in bytes and its modification time in ns, as text |
| `data_file_<attribute>` — `data_file_source` · `data_file_date` · `data_file_gamma` | each attribute the data file states of itself, copied with the prefix: TBtrans writes `source` (the program, `TBtrans`), `date` (the day it wrote the file) and `Gamma` (`true` / `false`, TBtrans's own); molbuilder reads none of them |
| `channel` | the data file's spin channel: `unpolarized`, or `up` / `down` for a polarized point's `TBT_UP` / `TBT_DN` |
| `spin` | in words: TBtrans's T and DOS carry no spin factor — a non-polarized run's are each spin's, up = down; a polarized run's are this file's channel alone |
| `energies_held` | in words: the energies this data file holds are its run's own; a point that shares energies with a finished one holds the others in its `shared/` folder, beside their own digest |
| `orbital_types` | in words: the data file does not state each orbital's type — the device run's `<label>.ORB_INDX` names its l and m, in the same order |
