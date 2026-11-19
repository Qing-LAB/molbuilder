*Verdict: acted on in fix 2b.* Every way `jobset prep` could refuse a calculation, read from the code at 2.77.5 for the user's rule of 2026-11-18 ("None of these things should stop any calculation being constructed or being run"). Decided from it: the 13 scientific judgments (S) became warnings; TranSIESTA's own stops among the (c) rows - a Fermi cutoff below 2.5 kT, non-collinear or spin-orbit spin on transport, a stripe junction's k-points - became information (the user, 2026-11-02); the PySCF solvent row (#46) became a warning naming what the results leave out. Kept as errors: nothing to compute (a), a file or input the engine cannot open (b), what a deck or our records cannot express (c), bookkeeping and conflicts (d) - among them the user's two rulings on a failed citation (#141) and the walk's bias list from 0 V (#90).

# Every refusal on the road description → decks (prep, describe, hand-over)

Read-only inventory, the checkout at `af5c653c9` (2.77.5), 2026-11-18.

**How a finding refuses.** An `Issue("error", …)` refuses wherever its list reaches
`validation.report` (raises `ValidationError`; prep's `user_error_as_prep` turns it into
`PrepError`) or `validation.task.refuse_on_error`, or a door that filters `severity == "error"`
(`prep.checked` 2072, `transport/rung.py` 210, `periodicity_gate` 125/194). The settings gate
`validation.validate` runs at **every deck render** (`script_emit.render_deck` 1232/1248), on the web
preflight panel (`web/blueprints/build.py` 703, shown, not refused) and through the description's
preflight only for the rows marked *preflight*. `molbuilder describe` and the Task-setup hand-over run
`validation.task.preflight` (+ SIESTA's pseudopotential check, `siesta/seam.py` 59) — not the
structure/geometry gate. The loading/editing doors report `cell.check` and `validate_geometry`
with errors downgraded to `warn` (`periodicity_gate.notices_for_report` 144, `_shared.py` 266).

**Category key.** (a) nothing would be computed · (b) a file / input the engine cannot open or read ·
(c) the engine or molbuilder's deck cannot do it (capability) · (d) bookkeeping / integrity ·
**(S)** a scientific judgment of ours about a computable calculation. **†** = unsure, discussed below.
*Traced*: R = the code at that line read this session (catalogue rows: the TOML row read);
R* = read with its caller chain.

## 1 · The settings gate (`validation.validate`, every deck)

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 1 | validation/geometry.py:43-49 | closest atom pair < 0.3 Å (any engine; 0.3–0.7 Å is a separate `warn`) | "closest atom pair is {d} Å apart -- atoms are effectively on top of each other; SCF will diverge" | **(S)** | the engine computes it; "SCF will diverge" is a forecast of the engine's own behaviour. MEASURED below | R* |
| 2 | cell.py:372 (resolve 185-189) | a periodic axis with no explicit lattice | `cell.unresolvable`: resolve_cell's own sentence (no lattice for a periodic axis) | (d) | a value stated nowhere | R |
| 3 | cell.py:383-390 | det(cell) < 0 (mirrored) | "This cell is mirrored: its three vectors turn the wrong way round (left-handed…" | (d)† | representability rule; refused on PySCF too (`box_findings_for` 743-755 keeps every box error for a free-space engine) | R |
| 4 | cell.py:396-403 | structure's fractional span > 1 along a non-periodic axis | "The molecule is longer than the cell along {a}, so it cannot fit wherever you put the box" | (d)† | the declared axis kind vs a box shorter than the atoms; on an isolated axis SIESTA would still compute (overlapping images); PySCF uses no box | R |
| 5 | cell.py:405-406 | box volume below tolerance (flat side / coplanar rows) | "This box is flat (… Å), so there is no space for the calculation to happen in." | (a) | no box to place atoms in; note: a stated all-zero vacuum on a planar molecule also refuses a PySCF run (no box used) | R |
| 6 | cell.py:647-658 (`require_placed`, from script_emit.py:1319) | after placement an atom lies outside the cell along a non-periodic axis (a stated origin) | "atoms lie outside the cell after placement -- {…}. Along a non-periodic axis the engine would get an atom outside its box" | (c) on transport / (d)† elsewhere | TranSIESTA refuses atoms outside (c); on an isolated axis or PySCF it is our placement contract | R |
| 7 | cell.py:635-641 | a computed frame reached the engine not centred by `to_engine` | "the atoms handed to the engine still carry an offset of …" (ValueError) | (d) | programmer integrity | R |
| 8 | validation/chemistry.py:65-73 | a species label names no element | "species label(s) {…} name no element, so {engine} has no atomic number to run with" | (b) | engine cannot read the species | R |
| 9 | kind_table.py:468 (called validation/__init__.py:237) | the kind is not run on this engine | KindTableError naming the engines the kind runs on | (c) | no backend | R |
| 10 | validation/metadata.py:117-126 | a field's `validate=` callable raised | "Field metadata for ``{f}`` has a ``validate`` callable that raised …" | (d) | programmer bug | R |
| 11 | config/siesta.py:95-102 (system_label; config/pyscf.py:119 job_name) | identity not `[A-Za-z0-9_-]+` | "{label}={value!r} is not a valid job basename. Must match […]+" | (d) | a name that is not a name | R |
| 12 | validation/metadata.py:170-176 | a config holds another value for an item every rung fixes (render skipped `resolve`) | "``{name}`` is {have!r}, and the stage fixes it at {answer!r}: …" | (d) | integrity | R |
| 13 | validation/metadata.py:200 → template.py:2447 (kmesh.py:76-80) | transport: `kgrid`/`tbt_k_grid` z ≠ 1 or `kgrid_displacement` z ≠ 0 | "{item} = [...], whose z component a transport calculation fixes at 1: one k-point along z…" | (c) | both engines force one k-point, no offset, along the open axis | R |
| 14 | metadata.py:200 → template.py:2453; catalogue.template.toml:181-183 | `solution_method = transiesta` on optimization / vibration | "…which an optimization does not offer: transiesta is a transport device's solver: an open-boundary run needs the electrodes…" | (c) | no electrodes outside transport | R |
| 15 | metadata.py:200 → template.py:2453; catalogue:508-510 | `relax_type = none` on a vibration | "…none relaxes nothing: relax_type names the relaxer a vibration's relax stage uses, and whether that stage runs…" | (d) | `already_relaxed` is the switch; two statements | R |
| 16 | template.py:2609 (fallback of 14/15) | a choice no `not_offered` row explains | "…{kind} on {engine} does not run it" | (c) | capability | R |
| 17 | metadata.py:200 → template.py:2461; catalogue:445, 2421 | `kgrid` / `tbt_k_grid` component ≤ 0 | "…must be greater than 0 -- each number is the k-point count along its axis… 0 would sample nothing" | (a) | nothing sampled | R |
| 18 | template.py:2461; catalogue:623 | `fc_displacement` ≤ 0 | "…must be greater than 0 -- SIESTA divides each force difference by it (ofc.f90)" | (a) | division by zero, nothing computed | R |
| 19 | template.py:2461; catalogue:854, 880, 940, 1709 | `mpi_np` / `gpu_count` / `omp_threads` / `threads` ≤ 0 | "…must be greater than 0 -- a count of MPI processes: 0 runs nothing" (each its own why) | (a) | nothing runs | R |
| 20 | template.py:2461; catalogue:899 | `block_size` ≤ 0 | "…must be greater than 0 -- it is a count of orbitals per rank; leave it unset…" | (d) | not a count | R |
| 21 | template.py:2461; catalogue:962 | `max_memory_mb` ≤ 0 | "…a cap holds that many MB; no cap is the row left blank, never 0" | (d) | 0 is not a cap | R |
| 22 | template.py:2461; catalogue:1830 | `temperature_K` ≤ 0 | "…PySCF's ideal-gas thermochemistry divides by kT and takes logarithms… 0 writes inf and nan" | (c) | engine writes inf/nan | R |
| 23 | template.py:2461; catalogue:1855 | `pressure_atm` ≤ 0 | "…PySCF's ideal-gas entropy takes a logarithm of kT/P… so 0 writes inf" | (c) | engine writes inf | R |
| 24 | template.py:2461; catalogue:3151 | `optical_broaden` ≤ 0 | "…SIESTA spreads each transition as a Gaussian of this width on an energy grid a tenth of it" | (c) | no grid to write on | R |
| 25 | template.py:2466; catalogue:2880 | `negf_neq_fermi_cutoff_kt` < 2.5 | "…must be at least 2.5 -- TranSIESTA stops a run below 2.5" | (c)† | the engine's own start stop — elsewhere such stops are `info` (poles < 20, T < 10 K) | R |
| 26 | template.py:2461; catalogue:3063 | `electrode_kz` ≤ 1 | "…must be greater than 1 -- a lead is periodic bulk metal along z… a wrong transmission with no runtime error" | **(S)** | TranSIESTA runs kz = 1; the refusal is our verdict that the result is wrong | R |
| 27 | validation/chemistry.py:251-256 (ES7) | `net_charge` stated ≠ 0 on transport | "net_charge = {q}, and a transport calculation carries +0 -- … net charge and gating are deferred by ruling" | (c) | molbuilder's transport decks write no charge (kind_table.toml:235) | R |
| 28 | chemistry.py:277-283 (ES4); catalogue:1164-1167 | spin `restricted-open` on SIESTA | "spin_treatment = … cannot run here: … SIESTA has no restricted open-shell formalism" | (c) | engine lacks it | R |
| 29 | chemistry.py:277; catalogue:1169-1177 | `non-collinear` / `spin-orbit` on PySCF | "…PySCF's molecular SCF here is collinear" / "…has no spin-orbit coupling" | (c) | engine lacks it | R |
| 30 | chemistry.py:277; catalogue:1179-1182 | `non-collinear` / `spin-orbit` on transport | "…TranSIESTA stops on more than two spin components (m_transiesta.F90…)" | (c)† | TranSIESTA does not implement it — but phrased as an engine stop (see 25) | R |
| 31 | chemistry.py:277; catalogue:1184-1187 | `restricted-open` on a PySCF vibration | "…PySCF has no analytic ROHF/ROKS Hessian, and the vibration deck takes the analytic Hessian" | (c) | deck route lacks it | R |
| 32 | chemistry.py:277; catalogue:1189-1192 | `unrestricted` on a PySCF vibration (resolved, incl. detected) | "…the spectrum record carries one spin channel… its reader refuses a second" | (c) | our record format | R |
| 33 | chemistry.py:277; catalogue:1232-1235 | `unpaired_electrons = free` on PySCF | "…PySCF fixes the moment: it occupies exactly N-up and N-down from mol.spin" | (c) | engine lacks it | R |
| 34 | chemistry.py:277; catalogue:1237-1239 | count ≠ 0 on a PySCF vibration | "…a PySCF vibration is restricted: every electron paired" | (c) | follows 32 | R |
| 35 | chemistry.py:311-317 (ES5) | `restricted` with `unpaired_electrons` ≠ 0 | "restricted means every electron paired, and unpaired_electrons = {n} asks for…" | (d) | two values conflict | R |
| 36 | validation/siesta.py:78-92 | no `psml_lib` and the calculation folder lacks a species (folder known; `warn` without one) | "No pseudopotential folder is set: SIESTA needs a .psml file for each of …, and will not start without them." | (b) | file missing | R |
| 37 | validation/siesta.py:99 | `psml_lib` spelling the anchor rule refuses (`PsmlLibError`) | the rule's own sentence (outside the tree / dotted) | (d) | not a path the rule answers | R |
| 38 | validation/siesta.py:116-128 | `psml_lib` folder missing (absolute, or relative with the folder known) | "the pseudopotential folder {p} does not exist or is not a directory, and the .psml files…" | (b) | file missing | R |
| 39 | siesta.py:142 → pseudos.py:669, 709 (`missing`) | no `<label>.psml` where SIESTA opens it / label not nameable | "no .psml file for {X} in {d} -- SIESTA opens <label>.psml… so it will refuse to start" | (b) | engine cannot open | R |
| 40 | siesta.py:142 → pseudos.py:695 (`misnamed`) | the element's file exists under another name | "{f} declares element {X}, but SIESTA opens {label}.psml… the run will refuse to start" | (b) | engine never opens it | R |
| 41 | siesta.py:142 → pseudos.py:724 (`misnamed`) | `<label>.psml` declares another element | "{f} declares element {Y}, but species {X} is {X} -- SIESTA… would read it as {X}'s" | (d)† | two statements conflict; whether SIESTA itself stops not read | R |
| 42 | validation/siesta.py:513-538 | transport device under bias: spacing leaves the bias window no point | "negf_neq_spacing_ev = {δ} eV leaves the bias window at {V} V (… wide…) no point" | (a) | nothing on the line | R |
| 43 | kmesh.py:326-341 | transport SCF grid samples an isolated axis > 1 while the other in-plane axis is periodic | "{grid} along {x} = {n} on an isolated axis of a junction… TranSIESTA stops the device on it" | (c)† | TranSIESTA's own stop (`ts_electrode.F90`), after seed + leads ran; "refused for now" (user 2026-09-30) | R |
| 44 | validation/pyscf.py:380-388 (+ pyscf/scf_setup.py:205 at render) | solvent name not in the PCM dielectric table | "unknown solvent {s!r}; this deck's PCM dielectric table knows: …" | (c) | no dielectric to write | R |
| 45 | validation/spectra.py:291-298 | PySCF vibration: a held index outside the structure | "the frozen atoms name {…}, outside this structure's {n} atoms" | (d) | index names no atom | R |
| 46 | validation/spectra.py:375-397 | PySCF vibration, solvent set AND (held atoms OR Raman OR IR) | "PCM solvation ({s}) reaches one route: the frequencies of a structure with no atoms held, with IR and Raman off…" | (c)† | our deck routes omit the PCM Hessian term — MEASURED below; a candidate the user named | R* |
| 47 | validation/spectra.py:421-437 | PySCF vibration `symmetry = true` with in-deck relaxation | "symmetry = true with in-deck relaxation: a geomeTRIC step leaves the point group, and PySCF's re-symmetrization would reorient…" | (c)† | probed deck behaviour (frame reoriented under the optimizer), not physics; whether the run fails or mis-relaxes not measured | R |
| 48 | validation/spectra.py:483-491 | SIESTA vibration: a held index outside the structure | "the frozen set names {…}, outside this structure's {n} atoms" | (d) | index names no atom | R |
| 49 | validation/spectra.py:492-499 (+ siesta/vibration_deck.py:126) | SIESTA vibration: every atom held | "every atom is held: a force-constant run needs at least one free atom to nudge." | (a) | nothing to compute | R |
| 50 | validation/spectra.py:605-614 | probe selector `all`: freq_min > freq_max | "Probe modes from, {lo} cm⁻¹, is above Probe modes up to… the window's two ends conflict" | (d) | conflict | R |
| 51 | validation/spectra.py:620-628 | probe selector `explicit`: list unreadable | "The explicit mode list {t!r} can't be read: …" | (b) | unreadable input | R |
| 52 | validation/spectra.py:655-662 | Raman `range` with neither end | "Raman for is {range} and neither Raman from nor Raman up to is given…" | (d) | stated nowhere | R |
| 53 | validation/spectra.py:663-670 | Raman min > max | "Raman from, {lo} cm⁻¹, is above Raman up to… the range's two ends conflict" | (d) | conflict | R |
| 54 | validation/spectra.py:720 | vibration kind on an engine with no checks written | "the vibration kind has no science for engine {e!r}…" (TypeError) | (d) | programmer gap | R |
| 55 | validation/stages.py:133-143 (preflight + every ladder check) | a rung that reads `relax_steps` has 0 and a moving `relax_type` | "stage {s!r} relaxes ({type}) in relax_steps = 0 steps -- MD.Steps 0 is a single point… relaxes nothing" | (S)† | SIESTA runs a single point; "relaxes nothing" is our reading of intent — or (d) relax_type vs 0 steps | R |
| 56 | validation/transport.py:138 → sidecar.py:165-176 | every transport rung: room at the z boundary > SEAM_VACUUM_FACTOR × the lead's layer spacing | "the cell leaves {r} Å at the transport boundary -- … it is vacuum, and the lead is a surface rather than a lead" | **(S)** | computable; our verdict on the physics (threshold factor) | R |
| 57 | validation/transport.py:138 → sidecar.py:177-194 | every transport rung: room < spacing / factor | "the cell leaves {r} Å at the transport boundary -- … the two leads' end layers meet through the image -- a collision" | **(S)** | computable on a fused cell; our verdict | R |
| 58 | validation/transport.py:180-192 | transport: a declared-periodic transverse axis where the lead's nearest image is > factor × its bond | "axis {a} is declared periodic, but across its boundary the nearest atom of {lead} is {r} Å away…" | **(S)**† | measured geometry vs a threshold; could be read as (d) declaration vs geometry | R |
| 59 | validation/transport.py:253-271 (transmission rung; also preflight via 232-250) | the sampled window does not reach ±(abs(V)/2 + 5 kT) | "the transmission's sampled window [lo, hi] eV does not reach ±{r} eV for {V} V…; widen it" | **(S)** | TBtrans computes T(E); the current is cut short | R |

## 2 · The written deck read back (`script_emit.check_deck`, engine `check_rules`)

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 60 | script_emit.py:1464 | the deck file is not there after writing | "the deck was not written: {p}" | (d) | integrity | R |
| 61 | script_emit.py:1480-1490 | USER-CUSTOM / PROVENANCE marker count ≠ 1 (a marker pasted into the reader's section) | "{label}'s {BEGIN} marker appears {n} times in {deck}, expected once" | (d) | ambiguous boundary | R |
| 62 | script_emit.py:1509-1515 | a line the parameters step wrote is not in the file | "{deck}: the parameters step wrote {line!r} and the file does not contain that line" | (d) | writer bug | R |
| 63 | siesta/layout.py:490-510 | a keyword / block written twice with different values (incl. the reader's own lines) | "{word} is written twice with different values… libfdf takes the first" | (d) | two values conflict | R |
| 64 | siesta/layout.py:513-519 | SystemLabel ≠ the stamped identity | "the deck's SystemLabel is not the identity it was written for…" | (d) | identity | R |
| 65 | siesta/layout.py:524-530 | NumberOfAtoms ≠ coordinate rows | "NumberOfAtoms says {n} but the coordinate block has {m} rows" | (d) | integrity | R |
| 66 | siesta/layout.py:541-547 | a species index the coordinates use is undeclared | "the coordinate block uses species {k}, which ChemicalSpeciesLabel does not declare" | (b) | engine cannot read | R |
| 67 | pyscf/layout.py:262-269 | the script does not compile | "the generated script does not parse: {msg} (line {n})" | (b) | engine cannot read | R |
| 68 | pyscf/layout.py:271-273 | no `gto.M(` | "the script builds no molecule (no gto.M)" | (d) | integrity | R |
| 69 | pyscf/layout.py:278-285 | JOB literal ≠ stamped identity | "the script's JOB literal is not the identity it was written for…" | (d) | identity | R |
| 70 | script_emit.py:162, 708, 766, 1313, 1400; siesta/layout.py:387, 425; siesta/input.py:133, 440, 590, 600, 618, 623; siesta/vibration_deck.py:130; pyscf/input.py:1491; electronic_state.py:196 | writer / catalogue / caller integrity (undeclared type, no `not_set`, no engine frame, unknown render argument, no rank count for BlockSize, unset solver, free atoms not a trailing run, module not shipped…) | e.g. "the {engine} deck carries no engine frame, so nothing says where its atoms were placed…" | (d) | programmer / integrity | R |
| 71 | siesta/input.py:497 | SIESTA `relax_type` outside CG/Broyden/FIRE at the relaxation block | "relax_type {t!r} is not a relaxation molbuilder runs -- its step keyword is not known here" | (c) | no step keyword to bound the loop | R |
| 72 | siesta/vibration_deck.py:56-58 | `optical_direction` is the zero vector | "optical_direction is the zero vector: the light's electric field needs a direction" | (a) | nothing to compute | R |

## 3 · The setting door (`template.accepts` / `why_not_in`) and the description's preflight (`validation/task.py`)

These run at `molbuilder describe` (describe.py:208-215), the Transport describe (transport/describe.py:385), Task-setup Save, and prep's checkpoint 3 (`prep.checked`, prep.py:2066-2077); `resolve` asks the same door again (resolve.py:564, 438). Every `why_not` row above (13-26) is reachable through each of them.

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 73 | validation/task.py:82-87 | the engine has no generator here | "engine {e!r} has no generator in this backend. Available: …" | (c) | no backend | R |
| 74 | validation/task.py:101-110 | a stage the pair retired is still a stage | "stage {s!r}: since {v}, {why} -- and this description still holds the {role} as a stage…" | (d) | migrate | R |
| 75 | validation/task.py:166-170 → jobset/group.py:189-198 | a stage shares a group with a stage it builds on | "`{s}` is in group {n} with {u}, which it builds on: one job runs its members… nothing handed between them" | (c)† | the group job hands no files between members; words add a workflow judgment ("before you have looked") | R |
| 76 | validation/task.py:248-251 | a run-card name is no setting | "{where}: {why_not_on_card sentence}." | (d) | not a name | R |
| 77 | validation/task.py:253-259 | a run-card value the setting door refuses (home, null, type, why_not) | "{where} sets {name} = {v}{clause}." | (d) | + inherits 13-26 | R |
| 78 | validation/task.py:288-292 | a bench row names no setting | "bench: {unknown_keys sentence}. A benchmark row sweeps a run setting…" | (d) | not a name | R |
| 79 | validation/task.py:294-297 → template.py:2750, 2755 | a bench row names a non-run setting / an allocation item that is not a grid axis | "…which {engine} does not declare as an `execution` parameter -- so sweeping it would measure a different calculation…" | (d)† | the picked point is applied as a launch shape; words carry a judgment | R |
| 80 | validation/task.py:300-304 → template.py:2834 | a bench point null / of the wrong type / why_not | "bench declares {name} = {v}{clause}." | (d) | + inherits | R |
| 81 | validation/task.py:307-311 | a bench row repeats a point | "bench declares {name} = [...]; a repeated point would measure one configuration twice." | (d) | duplicate | R |
| 82 | validation/task.py:370-374 | `varies` names no field | "'varies': {why_not_items sentence}." | (d) | not a name | R |
| 83 | validation/task.py:375-381 → template.py:2758-2765 | `varies` names a run setting / a role-fixed / a shared item | "'varies' names {n!r}, which the stage fixes / which is shared by …" | (d) | wrong home | R |
| 84 | validation/task.py:382-390 | an override names no field | "stage {s!r} overrides: {sentence}." | (d) | not a name | R |
| 85 | validation/task.py:413-423 → template.py:2738-2770 | an override in a wrong home: not carried / run setting / role-fixed / shared / a rung that does not read it | "stage {s!r} sets {k} = {v}, which the stage fixes / … which only the device stage … reads, not {s}…" | (d) | wrong home | R |
| 86 | template.py:2836-2839 (accepts) | `null` for an item whose unset is no answer | "…but this setting has no unset -- null is taken only by a setting whose unset is an answer" | (d) | stated nowhere | R |
| 87 | template.py:2785-2804 (accepts) | a value not of the catalogue's type / not an enum member / a fractional count | "…which is not a whole number -- this setting counts things…" / "…which is not one of …" | (b) | unreadable as its type | R |

## 4 · The description codec (`task.Task`) — every road reads a description through it

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 88 | task.py:390-394 | a transport description states no voltages | "{a transport calculation}'s description states its voltages -- [0.0] for a single bias…" | (d) | stated nowhere | R |
| 89 | task.py:396-400 | a voltage that is not a number | "bias voltages must be numbers (got …)" | (b) | unreadable | R |
| 90 | task.py:401-406 | the bias list does not start at 0.0 (a single 0.5 V point included) | "the bias list must start at 0.0 (got …) -- a list starts from equilibrium: the low-bias approximation converges…" | (c)† | the walk's design (each point starts from the one before); TranSIESTA could run 0.5 V cold — candidate | R |
| 91 | task.py:407-426 | a repeated voltage / two voltages one folder | "the bias list repeats {v} V -- each point is one device run in its own folder" | (d) | duplicate | R |
| 92 | task.py:430-438 | several voltages, low-bias approximation unanswered | "Several voltages: choose yes or no under "the low-bias approximation?"…" | (d) | stated nowhere | R |
| 93 | task.py:439-444 | low-bias answered for ≤ 1 voltage | "bias.low_bias_approximation answers a list of several voltages; this one has {n} point" | (d) | answers nothing | R |
| 94 | task.py:446-449; describe.py:158-163 | no stages | "'stages' is present but empty. A job has at least one stage…" | (a) | nothing to run | R |
| 95 | task.py:455-466 | duplicate stage names (case-insensitive) | "duplicate stage name(s) {…} (compared case-insensitively)…" | (d) | identity | R |

## 5 · `resolve` (prep step 2)

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 96 | resolve.py:294 | a benchmark point asks more than this prep's ask | "this sweep asks for {axis}={v}, which exceeds this prep's ask of {axis}={c}…" | (d) | two statements conflict | R |
| 97 | resolve.py:337 | the template states a role-fixed item another value | "the template sets {name!r} to {v!r}, which the stage fixes…: {why}. Remove the value…" | (d) | wrong home | R |
| 98 | resolve.py:438 | a resolved value `why_not` refuses | "{who} sets {name} = {v}{clause}." | (d) | inherits 13-26 | R |
| 99 | resolve.py:501 (template.py config_from_template 3192-3248) | the template has an unknown / retired item, a value of the wrong type or choice | the template reader's sentence ("template's {k} = …{why}") | (d) | unreadable / migrate | R |
| 100 | resolve.py:571 | a 0 written before its meaning changed | why_zeros_written_before (names `jobset migrate`) | (d) | migrate | R |
| 101 | resolve.py:580, 599 | override names no field / run-setting default unappliable | "{stage}: overrides: {sentence}." | (d) | integrity | R |
| 102 | resolve.py:607, 628, 631, 639 | a pin names a machine fact / nothing / a fixed item; a sweep axis names a fixed item | "pin(s) … name machine fact(s)…" / "sweep axis(es) … name what the stage fixes" | (d) | wrong home | R |
| 103 | resolve.py:668, 680, 689 | sweep axis names nothing / partial translation axes / translation returns non-Resources | "sweep axis(es) … name neither a template item…, an allocation field, nor a declared translation axis" | (d) | integrity | R |
| 104 | resolve.py:783 | two sweep points render one label | "two sweep points render the same label {l!r}…" | (d) | identity | R |
| 105 | resolve.py:851, 857 | stage not named / no such stage | "this description has a ladder, so a stage has to be named…" | (d) | stated nowhere | R |
| 106 | resolve.py:979 | label longer than the engine keeps whole | "the label {l!r} is {n} characters; {SIESTA} keeps a label whole… only up to {m}" | (c) | the engine cuts file names | R |
| 107 | resolve.py:1038 | an axis token outside [A-Za-z0-9_] | "axis {k}={v!r} renders as {part!r}, which leaves the label charset…" | (d) | not a name | R |
| 108 | resolve.py:1092, 1097 (card_values) | run card: name no setting / value refused | "the run card: …" / "the run card sets {n} = {v}{why}." | (d) | backstop of 76-77 | R |

## 6 · prep's own refusals (`jobset/prep.py` and what it calls)

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 109 | jobset/prep.py:2066-2077 | the description / job request fails its preflight (sections 3-4) | "the description or its job request fails its own preflight:\n - …" | (d) | aggregate of 73-95 | R |
| 110 | jobset/machine.py:113, 142, 151 (prep.py:2079-2080) | no machine record / activation unstated | the machine record's refusal (names `jobset probe`) | (d) | stated nowhere | R |
| 111 | jobset/placement.py:34-147 (prep.py:2129-2135, 2190) | ranks / threads / queue / wall / memory unstated | "stage {s!r} states no {…} -- and nothing fills one in. State each: …" | (d) | stated nowhere | R |
| 112 | jobset/placement.py:80-87 | a rank count for a one-process engine | "states a rank count (--np) for {e}, which runs one process -- a rank count names nothing it runs" | (d) | names nothing | R |
| 113 | jobset/placement.py:94-105, 152-162 | queue not in the record / record lists none | "names the queue {q!r}, which the target's record does not list…" | (d) | stated nowhere | R |
| 114 | jobset/placement.py:202-205 (prep.py:2198-2202) | `Unplaceable` (a queue statement missing) — never for the fit | "{stage}: {Unplaceable sentence}" | (d) | the fit is shown, not a veto | R |
| 115 | jobset/prep.py:2180-2184 | GPU request disagrees with itself | "stage {s!r}: {GpuRequestError}" | (d) | conflict | R |
| 116 | jobset/prep_inputs.py:284-309 (431; prep.py:2401-2404) | kind / engine has no benchmark | "a transport calculation has no benchmark sweep: …" / "Benchmarks measure SIESTA runs only…" | (c) | no bench lane | R |
| 117 | jobset/prep_inputs.py:80-84 | bench shape errors (78-81) | first message | (d) | backstop | R |
| 118 | jobset/prep_inputs.py:453, 533, 553 | bench on a scheduler with no queues / grid unstated / gpu_count on a CPU-only bench | "this benchmark does not state …: a benchmark's grid is yours to state…" | (d) | stated nowhere / conflict | R |
| 119 | jobset/prep_inputs.py:657, 669 | no (GPU) cell left within this prep's ask | "no bench cell is left -- see the cells left out. Raise this prep's ask…" | (a) | nothing to run | R |
| 120 | jobset/prep.py:134, 172 | invalid JobSet / deck missing in its job folder | "cannot prep an invalid JobSet: …" | (d) | integrity | R |
| 121 | jobset/rung.py:98, 102, 113, 122 | structure pair edited / missing / fingerprint or formula changed since described | "the structure has changed since this calculation was described…" | (d) | one calculation = one structure | R |
| 122 | jobset/prep.py:978-996, 3098-3110 (2935) | a launched stage's points / deck now compute something else | "stage {s!r} was launched, and its deck rendered now is not the calculation its runs ran…" | (d) | launched stage changed | R |
| 123 | jobset/prep.py:2404-2426, 2437, 2455, 2755-2848, 2897, 3213, 3229; 1040 (ShelfError), 2612, 2641 (GroupError) | stage/group not named, prepared already, plan changed since preview, folder state not saved, group membership / sequence / held-by-sequence, groups that cannot share one job | e.g. "`prep task` names the groups it prepares…" | (d) | bookkeeping | R |

## 7 · The transport kind's describe / compose / rung (`transport/*`)

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 124 | transport/describe.py:136, 149 | shared-panel item not shared for the kind / value unreadable | "{item!r} is not a shared value of this calculation…" | (d)/(b) | wrong home / unreadable | R |
| 125 | transport/describe.py:215, 267 → describe.py:255-313 (also the hand-over, build.py:946-955) | the folder is too high / a file / an unmakeable name / already a calculation / inside or holding another | "{f} already holds a calculation (task.json) -- one calculation per folder…" | (d) | layout | R |
| 126 | transport/describe.py:225, 410, 414 → reuse.py:82-94, 155-170, 223, 720, 837 | the reused calculation not found / not transport / its record unreadable / frame 0 not its junction | "{raw}: it is not a transport calculation…" | (d) | integrity | R |
| 127 | transport/describe.py:228 | engine differs from the finished calculation's | "engine {e!r}: {f} ran on {r}, and a calculation that reuses it runs on the engine it ran…" | (d) | conflict | R |
| 128 | transport/describe.py:235 | engine not named | "engine: name the engine the calculation runs on -- never assumed or inferred" | (d) | stated nowhere | R |
| 129 | transport/describe.py:239 | kind not on the engine | "engine {e!r}: no {…} transport backend is…" | (c) | no backend | R |
| 130 | transport/describe.py:248 | no junction citation | "no junction citation -- pick the relaxed junction first…" | (d) | stated nowhere | R |
| 131 | transport/describe.py:274 | an empty bias list | "bias: state the voltages -- [0.0] for a single bias" | (a) | nothing to compute | R |
| 132 | transport/describe.py:293 | no name | "name: the calculation has no name…" | (d) | stated nowhere | R |
| 133 | transport/describe.py:313 → window.py:90, 95 | fitted window not a range / even "finer" factor | "the window asked, {lo}…{hi} eV, is not a range…" / "…must be an odd whole number" | (d) | finished energies must stay grid points | R |
| 134 | transport/describe.py:354 → reuse.py:449-476 | an extension changes its rename / label / a kept rung's values / a shared value | "{what} is fixed by {name}: an extension copies in the runs every voltage stands on…" | (d) | conflict | R |
| 135 | transport/citation_defaults.py:64-71 | the cited record's k-point grid unreadable | "the cited record states a k-point grid this build cannot read…" | (b) | unreadable | R |
| 136 | transport/citation_defaults.py:185-193 | the cited relaxation was run at net charge ≠ 0 | "the citation's record states a net charge of {q}… Relax the junction neutral and cite that run" | **(S)** | the neutral junction at a charged geometry computes; a vibration's level-of-theory mismatch is only warned | R |
| 137 | transport/citation_defaults.py:228-235 | a cited pair and no pseudopotential folder | "{pair} is a structure pair, and a pair brings no pseudopotentials: name the directory…" | (b) | files missing | R |
| 138 | transport/citation_defaults.py:236-239 | a pseudopotential folder named for a cited run | "a pseudopotential directory does not apply to a cited run -- its pseudopotentials travel with it." | (d)† | a value that enters nothing — elsewhere warned (`check_inactive`) | R |
| 139 | transport/compose.py:247, 258, 267, 277, 282 | the cited folder holds no readable, launched run of ours | "{d} holds no run of molbuilder's: {where}…" | (d)/(b) | integrity | R |
| 140 | transport/compose.py:289 | the cited run relaxed no geometry (vibration / transport rung) | "…which relaxed no geometry -- its output reports a self-consistent field alone…" | (d) | the citation is a relaxation | R |
| 141 | transport/compose.py:420-445 (refuse_failed_citation; describe and prep) | the cited relaxation ended with an error / killed / never concluded | "the cited relaxation {c!r} {how}. It left no relaxed junction, and a calculation citing it could never be prepared" | (d)† | user ruled refuse twice (2026-10-25, 2026-11-11); its last geometry may exist — candidate | R |
| 142 | transport/compose.py:813 | the cited relaxation is still going | "the cited relaxation {c!r} has not concluded -- …this is composed once it does." | (d) | nothing to cite yet | R |
| 143 | transport/compose.py:320, 338 | cited pair has no sidecar / no info.calculation | "{p} carries no info.calculation -- …nothing says which settings its geometry was optimized with" | (d) | stated nowhere | R |
| 144 | transport/compose.py:333, 969, 488, 514, 537, 552, 558 | pair / composed junction / deck record / order / output unreadable; a cell transport cannot use | "{p} does not read as a structure pair ({exc})…" | (b) | unreadable | R |
| 145 | transport/compose.py:355-380 | a frame of a frame set moves an electrode atom > 1e-3 Å from frame 0 | "{f}: frame {k} moves {atom} by {d} A -- every frame of a set shares the leads' calculation…" | **(S)**† | every frame computes; our verdict that frame 0's lead calculation no longer fits (tolerance) | R |
| 146 | transport/compose.py:408, 412 | citation outside the tree / names nothing | "the junction citation {c!r} leaves the projects tree…" | (d) | layout | R |
| 147 | transport/compose.py:462 | recorded axis kinds across transport not periodic/isolated | "{where} records the axis kinds {k!r}; across the transport axis a junction is periodic or isolated" | (d) | integrity | R |
| 148 | transport/compose.py:565 | cited relaxation carries no region labels | "the cited relaxation in {p} carries no region labels for its {n} atoms…" | (d) | stated nowhere | R |
| 149 | transport/compose.py:637 | swap_electrodes with one lead label | "{s} does not carry {lab}, so there is no pair to swap…" | (d) | conflict | R |
| 150 | transport/compose.py:676-692 | centred junction still has atoms outside the cell along z | "the junction, centred in its cell, has {atoms} outside it along z…" (+ the collision words of 57) | (c) | TranSIESTA requires every atom inside | R |
| 151 | transport/compose.py:693-704 | an assigned origin puts atoms outside along z | "the cited junction has {atoms} outside its cell along z, … where TranSIESTA requires every atom inside" | (c) | as 150 | R |
| 152 | transport/compose.py:794, 828 | the cited structure has no usable lattice | "the cited pair {p} {bad} -- a junction needs its lattice…" | (d) | stated nowhere | R |
| 153 | transport/compose.py:876, 894 | composed record incomplete / the cited pair changed since composed | "the cited pair {c!r} has changed since this calculation was composed from it…" | (d) | integrity | R |
| 154 | transport/wizard.py:263-276 (→ compose.py:615/619) | an electrode-labelled atom is not in `frozen_atoms` | "{n} atom(s) in {lab!r} are NOT FROZEN: … A lead is the pristine bulk… must be held still" | **(S)**† | a lead block is still computable; the warning at relaxation (sidecar.py:100) has other words | R |
| 155 | transport/wizard.py:281-293 | electrode atoms not held by the relaxation per its record | "{n} atom(s) in {lab!r} were NOT HELD by the relaxation that left this geometry…" | **(S)**† | as 154 | R |
| 156 | transport/wizard.py:299-304 | prior geometry has another atom count | "cannot check {lab!r} against the geometry the relaxation started from…" | (d) | integrity | R |
| 157 | transport/wizard.py:306-322 | an electrode atom moved > 1e-3 Å (FROZEN_TOL_ANG) during the relaxation | "{n} atom(s) in {lab!r} MOVED during the cited relaxation: … Frozen means unmoved" | **(S)**† | tolerance verdict; computable | R |
| 158 | transport/wizard.py:363-368 | region not an electrode region | "region {lab!r} is not an electrode region; found …" | (d) | integrity | R |
| 159 | cell.py:836-839 (via wizard.py:249) | a lead block of one atomic layer | "cannot derive a bulk z-period from a single atomic layer…" | (c) | no period to build the lead cell | R |
| 160 | cell.py:843-853 (via wizard.py:249) | lead layers' spacings spread > UNIFORM_SPACING_TOL_ANG | "the layers are not evenly spaced: … so repeating this block does not reproduce a bulk lead" | **(S)**† | a lead cell could be built (z_span + first spacing); verdict on bulk-ness — or (c) | R |
| 161 | transport/sort.py:225, 239 | an atom with no partition label / with two | "{n} atom(s) carry no partition label: … Every atom must be exactly one of…" | (d) | integrity | R |
| 162 | transport/sort.py:273 | partition lacks L/R electrode or bridge | "the partition is missing {…} -- a 2-terminal junction needs all three" | (c) | no two-terminal deck | R |
| 163 | transport/sort.py:283 | the two electrode blocks interleave along z | "the L-electrode and R-electrode blocks INTERLEAVE along the transport axis…" | (c) | no order can be built | R |
| 164 | transport/sort.py:332 | buffer atoms at or inside the electrode blocks | "{n} buffer atom(s) sit AT or INSIDE the electrode blocks along the transport axis…" | (c) | TranSIESTA's order: buffer outside | R |
| 165 | transport/sort.py:368, 371, 409 | no such sort key / already remapped / not a bijection | "internal error: the sort is not a bijection…" | (d) | programmer | R |
| 166 | transport/rung.py:70 | pseudopotentials neither in the calculation nor the citation | "this transport calculation needs {files} and there is none in {f}/… or in the cited directory" | (b) | files missing | R |
| 167 | transport/rung.py:205-212 | the junction's electronic state refused (rows 27-35) | first error message | (d) | backstop of 27-35 | R |
| 168 | transport/rung.py:124, 291, 350, 377, 410 | resolve / compose record / reuse refusals translated; compose record does not read back | "the composed junction was written and could not be read back: …" | (d) | integrity | R |

## 8 · The Cell page's edit gate (`periodicity_gate`) — on the hand-over's `for_writing`

| # | file:line | condition | refusal's words (start) | Cat | why | traced |
|---|---|---|---|---|---|---|
| 169 | periodicity_gate.py:186-196 (hand-over `for_writing`), 115-126 (edits) | any `cell.check` error (rows 2-5) | the checker's own sentence | (d)/(a) | same rows as 2-5 | R |
| 170 | periodicity_gate.py:415-528 | periodic axis with no cell / clearing the cell while periodic / origin on a derived box / malformed payload | "a periodic axis needs an explicit commensurate cell first…" | (d) | stated nowhere / malformed | R |

---

## Counts (170 rows; a mixed row counted once, by the first category written)

| category | rows | of which † |
|---|---|---|
| (a) nothing computed | 10 — #5, 17, 18, 19, 42, 49, 72, 94, 119, 131 | 0 |
| (b) cannot open / read | 14 — #8, 36, 38, 39, 40, 51, 66, 67, 87, 89, 135, 137, 144, 166 | 0 |
| (c) capability | 34 | 8 — #6, 25, 30, 43, 46, 47, 75, 90 |
| (d) bookkeeping / integrity | 99 | 6 — #3, 4, 41, 79, 138, 141 |
| **(S)** scientific judgment | 13 — #1, 26, 55, 56, 57, 58, 59, 136, 145, 154, 155, 157, 160 | 7 — #55, 58, 145, 154, 155, 157, 160 |

## The (S) rows — is a warning of the same words already somewhere?

1. **#1 atoms < 0.3 Å** (geometry.py:45). MEASURED: the one site (no other close-atom refusal in the tree); the settings gate runs it for every engine (the cell-free half of `validate_geometry`), so it refuses at prep (render_deck → report → PrepError) and shows as an error on the web preflight. **The same words already reach people as a warning**: the loading/editing doors pass it through `notices_for_report` (error → warn, `_shared.py:266`) and `molbuilder build` prints it unraised (`cli.py:187`). Not asked by `molbuilder describe` or the hand-over. Turning it is a severity change at line 45; "SCF will diverge" forecasts the engine's behaviour and may want rewording. (0.3–0.7 Å is already a `warn` in other words.)
2. **#26 `electrode_kz` ≤ 1** (catalogue:3063, `above`). No warning of those words; values 2–49 already warn through the generic range words ([50, 200]). Turning = dropping `above` from the row (its `why` then has no home) — a catalogue change, not a severity flag.
3. **#56 transport boundary vacuum** (sidecar.py:168 via validation/transport.py:138 `severity="error"`). Same function and words; at a junction's relaxation the vacuum half is not said (`vacuum=False`), so no warning of these words exists. One argument at transport.py:138 turns both 56 and 57.
4. **#57 transport collision** (sidecar.py:182). **Same words already warn** at a labelled junction's relaxation (`validate` → `check_junction_boundary(…, vacuum=False)`, severity `warn`). compose.py:684 also raises them when the centred junction does not fit along z — that one is (c), row 150.
5. **#58 transverse vacuum** (validation/transport.py:182). No warning of these words (only the one-atom "cannot be measured" warn). Unsure: also readable as (d), the declared kind vs the geometry.
6. **#59 transmission window short of the bias window** (validation/transport.py:269; also the description preflight via `check_description`). The words (`energies.window_short_of`) are already the record's own note on an un-integrated voltage (`record.py:308-313`, current left `None`) — a note, not a warning Issue. No `warn` Issue carries them.
7. **#136 cited relaxation at a non-zero charge** (citation_defaults.py:188). No warning of these words; the analogous vibration case (geometry relaxed at another level of theory, charge included) is `warn`/`info` in `check_relaxation_record` (other words).
8. **#145 frame set moves electrode atoms** (compose.py:374). No warning. †
9. **#154 lead atoms not frozen** (wizard.py:268). A warning in other words exists at the junction's relaxation (`structure.electrode_frozen`, sidecar.py:100), whose docstring calls the compose refusal "what makes a bad junction impossible". †
10. **#155 lead atoms not held per the relaxation record** (wizard.py:288). No warning. †
11. **#157 lead atom moved > 1e-3 Å** (wizard.py:314). No warning. †
12. **#160 lead layers unevenly spaced** (cell.py:846). No warning. † (could be (c): the lead cell takes one spacing.)
13. **#55 `relax_steps = 0` on a relaxing rung** (validation/stages.py:139). No warning. † (or (d): `relax_type` says relax, 0 steps says not; SIESTA runs a single point.)

## Unsure rows, and the second measured one

* **#46 PySCF vibration with a solvent and IR on** (validation/spectra.py:375-397; condition: a known solvent AND held atoms or Raman or IR). MEASURED, by reading the deck and the installed PySCF (`molbuilder-pySCF`, pyscf 2.x): every SCF of the deck is solvated (`_mb_apply_solvent`, vibration_emitters.py:631 equilibrium, :1009 each displaced SCF), so the run computes. But PySCF adds the PCM Hessian term only in `WithSolventHess.kernel` (`pyscf/solvent/hessian/pcm.py:1099-1111`: `de = de_solute + with_solvent.hess(dm)`), and two of the deck's routes do not call `kernel`: the **analytic IR route** (IR on, Raman off, nothing held → `_WANT_ANALYTIC_IR`, emitters.py:808) takes the Hessian from upstream `proc_hessian_` = `hess_elec + hess_nuc` (`pyscf/prop/infrared/rhf.py:32-50`; `spectra/pyscf_vibration.py:186-204`), and the **held-atom route** sums `hess_elec(atmlst) + hess_nuc` (pyscf_vibration.py:156-163). With a solvent, turning IR on therefore silently drops the solvent's second-derivative term from the frequencies. With Raman on and nothing held, the Hessian is `mf.Hessian().kernel()` (solvent included); whether the polarizability's response carries the solvent was not measured. So this is a gap in our deck's routes — category (c) as written, fixable by adding `with_solvent.hess(dm)` on those two routes, after which nothing is left to refuse; turned to a warning as-is, the run's frequencies would knowingly miss that term. No warning of these words exists (an `info` covers the one consistent route).
* **#25, #30, #43 — the engine's own stops refused**: `negf_neq_fermi_cutoff_kt < 2.5`, non-collinear / spin-orbit on transport, the stripe k-grid. Other engine stops on the same road are `info`, never refused (TranSIESTA's 20 poles and 10 K, transport.py:71-101; a fixed count under non-collinear, ES6; OMM spin hint; PySCF parity, chemistry.py:349-352) — per the 2026-11-02 rule. Category (c) here; the two treatments disagree.
* **#47 symmetry + in-deck relaxation** (PySCF vibration): a probed deck behaviour (frame reoriented under geomeTRIC); whether the run fails or mis-relaxes was not measured. (c)†.
* **#90 bias list must start at 0.0**: the walk's design; a single 0.5 V point is refused. (c)† — candidate.
* **#141 cited relaxation ended without finishing**: the user ruled refuse (2026-10-25, 2026-11-11); its last geometry may still exist, and a finished-but-unconverged geometry is already cited knowingly. (d)† — candidate.
* **#3, #4, #6 (non-transport axes), #5 on PySCF**: box representability errors bind a free-space PySCF run too (`cell.box_findings_for`), though PySCF uses no box.
* **#41** `<label>.psml` declares another element: whether SIESTA itself stops on it was not read.
* **#75** stage grouped with its upstream: capability (nothing handed between members of one job), but the words add "before you have looked at what … produced".
* **#79** bench sweeping a non-run setting, **#138** a psml folder named for a cited run: bookkeeping, but #79's words carry a judgment and #138 refuses a value that enters nothing, which `check_inactive` only warns about.

## Modules read

Whole: `validation/__init__.py`, `geometry.py`, `metadata.py`, `chemistry.py`, `stages.py`, `task.py`, `transport.py`, `siesta.py`, `pyscf.py`; `kmesh.py`. Nearly whole: `validation/spectra.py` (1-60, 240-728 read; 60-240 searched — warn/info only). Searched for every refusal and each read in context: `validation/sidecar.py`, `validation/builders.py` (warn only, not on the road), `cell.py`, `template.py` (why_not, why_not_in, accepts, type door, why_not_offered, offered, reads/why_unread, fixed_*, config_from_template), `resolve.py`, `script_emit.py`, `siesta/layout.py`, `pyscf/layout.py`, `electronic_state.py`, `periodicity_gate.py`, `jobset/prep.py`, `jobset/placement.py`, `jobset/prep_inputs.py`, `jobset/machine.py`, `jobset/rung.py`, `jobset/group.py`, `transport/describe.py`, `compose.py`, `reuse.py`, `wizard.py`, `sort.py`, `rung.py`, `citation_defaults.py`, `deck.py`, `window.py`, `stages.py`, `energies.py`, `record.py` (low_bias_iv), `task.py` (codec), `describe.py`, `config/siesta.py`, `pseudos.py` (statuses), `kind_table.py` (on), the deck writers' raise sites (`siesta/input.py`, `siesta/vibration_deck.py`, `pyscf/input.py`, `pyscf/scf_setup.py`), `web/blueprints/build.py` (preflight + hand-over refusals), and the catalogue's every `above`/`at_least`/`offered`/`not_offered`/`role` row. Upstream PySCF (read-only): `solvent/hessian/pcm.py` `WithSolventHess`, `prop/infrared/rhf.py` `proc_hessian_`.
