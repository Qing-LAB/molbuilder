# Post-processing an optimization

What a finished — or a running — optimization holds, which call answers what,
and worked examples in Python: the energy and the forces step by step, the final
geometry, the SCF, the settings the run read, and how to find a calculation.
Every example below ran on a real calculation before it was written here, on
both engines where it says so.

## 1. One import

A notebook on the Notebook tab imports the molbuilder its server runs
([`web/jupyter.md`](?doc=web/jupyter.md) § 7); a script on the machine does the
same from its checkout.

```python
from molbuilder.postprocess import (parse, read_steps, read_structure, read_xv,
                                    atom_index_map, list_projects, list_topics,
                                    list_structures, structure_dir)
```

| name | answers |
|---|---|
| `read_steps(path)` | the steps a run finished, **in your atoms' order**, from a run's folder or its `.out` or `.molwatch.log`: the call to start from (§ 3) |
| `read_structure(path)` | a structure pair, in your atoms' order: `(structure, atoms, said)` |
| `parse(path)` | one file as it stands, **in the file's own order**, `result.atom_index_map` saying whose (§ 5) |
| `read_xv(path)` | SIESTA's own `.XV`, **in the engine's order** (§ 4) |
| `atom_index_map(path)` | which index a file's atoms are listed by, and both indices of each (§ 5) |
| `list_projects()`, `list_topics(p)`, `list_structures(p, t)`, `structure_dir(p, t, s)` | where a calculation is, by name (§ 2) |

## 2. Finding the calculation

```python
list_projects()                          # your projects, by name
list_topics("BDT")                       # 'structure', 'optimization', ...
list_structures("BDT", "optimization")   # its calculations
calc = structure_dir("BDT", "optimization", "relaxed")   # the folder, names checked
```

A calculation holds a folder per stage (`01_coarse`, `02_fine`, …), each stage a
folder per start (`run-0`, `run-1`, …): a run is one of those. The newest start
of the last stage is where a ladder finished; `molbuilder jobset status --bundle
<calculation>` says how each stands.

## 3. The steps, in your atoms' order

```python
steps = read_steps("01_coarse/run-0")     # a run's folder, or its output in it
steps.atom_index_map.listed_by            # 'user_index'
steps.elements[0]                         # your atom 0's chemical symbol
steps.energy_ev                           # by step, eV; NaN where a step states none
steps.final_energy_ev                     # the last energy of a finished step, eV
steps.positions_ang[-1, 0]                # your atom 0 at the last step, Å
steps.forces_ev_per_ang[0, 0]             # the force on it at the first step, eV/Å
steps.held_atoms                          # the atoms the run says it held, by your index
steps.said                                # what it says beside the numbers
```

**A step is what the engine finished.** The input geometry a progress log opens
with is none, and neither is a step still being computed or one the run stopped
inside, whose energy is no result: `said` says so, and `parse` shows it as it stands
(§ 5). A run that has finished none is refused by name. **A SIESTA relaxation's last
row is the final geometry it wrote, which states no energy of its own**:
`energy_ev[-1]` is NaN there, and `final_energy_ev` answers the step before it. A
PySCF run's last step states its energy.

**Did it converge?** The run's own record, on both engines, says so by the run's
own tolerance:

```python
steps.relaxation["converged"]
steps.relaxation["force_tolerance_ev_ang"]
steps.relaxation["max_force_free_ev_ang"] # what that tolerance tests: the atoms it moved
steps.relaxation["max_force_ev_ang"]      # with the held atoms in
steps.max_force_free_ev_per_ang           # the same figure, step by step, eV/Å
steps.max_force_ev_per_ang                # with the held atoms in, step by step
```

A held atom feels a force the run never relaxes, so the figure with the held atoms
in can stay large on a run that converged. Each figure is the largest Cartesian
component on any atom, as SIESTA's `Max` line prints it and its tolerance tests it,
on PySCF too. `relaxation` is `None` for a run that relaxed nothing.

**Which energy.** On SIESTA a step's energy is its `E_KS`, and on a transport device
point the last row of its TranSIESTA loop, which is not the `E_KS` the output prints
after it. On PySCF it is the SCF's total energy, in eV here.

**Where the coordinates stand.** They are the ones the engine ran: the run's input,
moved as one by what prep placed it in the box by.

```python
steps.applied_offset_ang                  # the run's input + this = the first step's coordinates
steps.positions_ang[0] - steps.applied_offset_ang   # the run's input
```

For a run that starts from your structure, that input is your structure. A later
rung and a SIESTA vibration's force constants start from where the step before
them ended.

**How it ended.** `steps.run_state` is what the file says, `ended`, `stopped`,
`out_of_memory` or `running`, and `running` means only that the file shows no
ending: the file alone cannot tell a slow run from one that died with no word
([`model/parse.md`](?doc=model/parse.md) § 2b). `molbuilder jobset status --bundle
<calculation>` asks about the job.

## 4. The final geometry

```python
final = steps.last                        # a Structure: your order, labels, box
final.positions, final.elements, final.cell
sorted(final.regions), final.frozen_atoms
start = steps.first                       # where it started, the same way
```

To keep it, `molbuilder xv2xyz --from-run run-0/job.XV final.xyz` writes the pair in
your order. **PySCF** writes the pair itself; read it in your order:

```python
final, atoms, said = read_structure("01_coarse/run-0/J_optimized.xyz")
atoms.listed_by                           # 'user_index'
```

**SIESTA's `.XV` read alone** is the engine's: `read_xv` says so, since the file
says nothing of it. An optimization keeps your order, so here the two agree; a
transport calculation's or a SIESTA vibration's run does not, and row 0 is not
your atom 0 ([`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md) § 4).

```python
struct = read_xv("01_coarse/run-0/J.XV")       # engine order, as SIESTA wrote it
atoms = atom_index_map("01_coarse/run-0/J.XV") # that run's map, from its deck
mine, _ = atoms.map_rows_to_user_order(struct.positions)   # row i: your atom i
```

## 5. One file as it stands

`parse` reads one file and hands it back in **its own order**; `atom_index_map` on
the answer says whose that is.

```python
r = parse("01_coarse/run-0/J_01_coarse.out")
r.atom_index_map.listed_by                # 'engine_index'
r.atom_index_map.table()[:2]              # each atom's user index and engine index
r.run_state, r.scf_converged, r.cause
r.frames[0].scf_history[-1]               # the first step's last SCF row
r.runtime_info["scf_criteria"]            # what the SCF had to reach
r.frames[-1].in_progress                  # the step still being computed, as it stands
```

Each `Structure` on the answer knows the order of the run or pair that holds the
file, so its own index answers are about your atoms:

```python
xv = parse("01_coarse/run-0/J.XV")
xv.structure.map_to_user_index([0])       # your index of this file's row 0
xv.structure.restored()                   # its atoms put back in your order
```

| file | reads as |
|---|---|
| `<label>_<stage>.out` (SIESTA), `.molwatch.log` | a trajectory: `frames`, `run_state`, `scf_converged`, `runtime_info`. A PySCF run writes its steps into its `.molwatch.log`; a SIESTA run's holds only the input geometry prep put there |
| `<label>.XV`, a PySCF run's `_optimized.xyz` and `_initial.xyz`, a `.pdb` | a structure: `structure`, `cell` |
| any other `.xyz`: a pair's, geomeTRIC's `_geom_optim.xyz` | a trajectory: `frames`. `read_structure` reads a pair whole |
| `fdf.*.log` | what SIESTA read: `params`, `blocks` |
| `.scf-timing.log`, `.monitor.log`, `.util.csv` | what the wrapper measured: `metrics` |

**SIESTA's and PySCF's other files** — `.EIG`, `.ORB_INDX`, `.DM`, PySCF's `.chk` —
are the libraries': `sisl` reads SIESTA's, in the notebook's env and here alike, and
`pyscf` its own, in `molbuilder-pySCF` only, since the notebook's env does not hold
it. Their atoms are the engine's, and `atom_index_map(<the file>)` answers them as
any other.

## 6. Atom indices

Every atom index here counts from **0**; MolView numbers an atom its index + 1 on
screen. A SIESTA `.fdf` and geomeTRIC's constraints count from 1, as their engines
do. The rule and every file's statement:
[`model/overview.md`](?doc=model/overview.md) § 2–2.2b.
