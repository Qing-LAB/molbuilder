# The post-processing API, evaluated on real runs (2026-11-05)

*(the user: "evaluate the completeness and usefulness of the api designed specifically to support user to extract data from different aspects and correctly interpret data, and assure the correct index etc conventions will not cause user to get the wrong data"; "understand fully the netcdf data set we generated and how post process we've already done in the monitor/script of our own so you do not reinvent wheels")*

Read first: `model/datafiles.md`, `model/overview.md` § 2–2.2b, `model/parse.md` § 3, `engines/transport-postprocessing.md`, `web/jupyter.md` § 7, `postprocess.py`, `runs.py`, the commands. Every check below ran the layer on finished real runs whose engine order differs from the user's, with hand-run scripts importing `molbuilder.postprocess` as a notebook does.

## What exists, so nothing is built twice

| question | what answers it today |
|---|---|
| every transport result | `<label>.transport.nc`, `read_transport_data`, `jobset export` (NetCDF or CSV) |
| DOS per orbital, PDOS | `<label>.TBT.digest.nc`, sisl on `.TBT.nc`, kept reading through `parse` with its `atom_index_map` field |
| every vibration result | `<label>.spectra.nc` and `.json`, `read_spectra_data`, `jobset export` |
| infrared, Raman at a laser, a frame set's reach | `infrared_view`, `raman_view`, `read_optical_data`, `read_raman_data`, `outermost_move` |
| a transport set under another motion | `reaveraged` |
| which atom is which | `atom_index_map`, `AtomIndexMap` (`map_to_user_index`, `map_to_engine_index`, `map_rows_to_user_order`, `map_rows_to_engine_order`, `table`), `molbuilder atom-index` |
| a structure in the user order | `read_structure`; `xv2xyz --from-run`; `runs.steps_of` (not in the layer) |
| one run's steps as JSON, with the map | `molbuilder watch parse`, `runtime-info` |
| what the run script and monitor already produce | the trajectory log with its map, `_initial` / `_optimized` pairs, the vibration result, `optical.nc`, the Makov-Payne script, `.monitor.log`, `.scf-timing.log`, `.util.csv` |

## What the real runs showed

| claim tested | how | result |
|---|---|---|
| Transport data file: atoms in the user order | positions and elements against the junction in the user order | exact |
| The file's engine column | against the junction's order record | same |
| DOS by region is the region's atoms | `dos_region[bridge]` against the digest's columns of those atoms through the map | largest difference 7e-9 of 0.15 |
| sisl agrees | `.TBT.nc` geometry at mapped engine indices against the user's atoms | exact |
| Vibration rows on their atoms | mass-weighted overlap of the modes (S, C, C, S, four free atoms) | 1 on the diagonal, 0 off; 0.72 off with the masses on the wrong rows |
| Modes and optical calls agree | each call's displacement against ±dq × eigenvector, 18 calls over two runs | within 1e-8 Å; 0.042 Å with the rows on the wrong atoms |
| Charges | the result's Hirshfeld row against SIESTA's own print at the mapped index | exact; 0.31 e off without the map |
| Held atoms | against the person's structure, three runs | same, positions unmoved |
| Refusals | an index past the end; an electrode's atom asked for a user index; a whole calculation folder | each refused, naming why |

## What is wrong or missing

**G1. Reading a sorted run's own files hands back the engine's order with nothing on the answer that says so.** `parse("01_seed/run-0/T.XV").structure.positions[0]` is the user's atom 13, not atom 0: atom for atom against the user's structure the largest mismatch is 22 Å. `parse(T_01_seed.out).frames[i]`, its positions and forces, the same. `structure.atom_order` is `None`, and only the trajectory log carries the map. `atom_index_map(path)` then gives the exact answer, but a person must know to ask, against `overview.md` § 2.2b F6 (*the statement travels with the data*). The transmission reading already does it right: `parse(.TBT.nc)` answers an `atom_index_map` field.
Measured on the seed; the device, transmission, and a SIESTA vibration's relax, force-constant and optical runs state non-identity maps too. Not affected: a SIESTA optimization (a run with held atoms states no order record), PySCF (sorts nothing).

**G2. An optimization has no data file, no export and no worked example.** `jobset export` takes transport and vibration only. And `parse.md`'s own example prints `last.energy` of `frames[-1]`, which is `None` on a real SIESTA run: the last frame is the final geometry with no energy, the energy being the frame before.

**G3. Nothing in the layer goes from a calculation to its files.** `atom_index_map` rightly refuses a calculation folder and names its runs, but a notebook has no call for "this calculation's stages, newest runs and result files", or for "the calculations in this project". The project functions 2.65.17's draft removed were the second half of that.

**G4. Worked examples exist for transport only.** Vibration has one import line; optimization has `parse.md`'s general one.

**G5. Not tried:** engine files `parse` does not read (`.EIG`, `.FA`, `.ORB_INDX`, `.BONDS`, PySCF's `.chk` and log). `web/jupyter.md` § 7 gives them to sisl and ase, and `atom_index_map` answers them. Also not tried: a spin-polarized run, and `reaveraged` on a true mode's frame set (the run tried is a family of structures, and was refused by name).
