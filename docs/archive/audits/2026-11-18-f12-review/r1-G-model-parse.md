# F12 review, round 1 — slice G: the structure model, the readers of engine output, the data files

Checkout `the checkout` at f578482db (2.77.2). Read-only; nothing in the checkout was written.
Python was run only with `-B`, to check claims, and it wrote nothing.

**Coverage.** I read these chapters end to end: `model/overview.md`, `structure.md`,
`structure-molstruct.md`, `structure-periodicity.md`, `structure-annotations.md` and `chemistry.md`.
For `datafiles.md` I read §§ 0–8, § 9.0 and § 9.5 in full; I checked § 9.5 row by row against
`stepsdata.py`. I read these modules whole: `structure.py`, `cell.py`, `workingcopy_structure.py`,
`structure_pair.py`, `periodicity_gate.py`, `atom_permutation.py`, `atom_index.py`,
`engine_atom_index.py`, `annotations_fdf.py`, `frame.py`, `kmesh.py`, `selection.py`, `units.py`,
`constants.py`, `stepsdata.py` and `sidecars/molstruct.py` (the codec `structure-molstruct.md`
governs). Of `chemistry.py` I read every function `chemistry.md` covers; of `electronic_state.py`, the
charge step. Of `ncdata.py` I read `write_group`, `check_said`, `check_axes` and the unit tables.

`parse.md` and every module under `parse/` were reviewed separately; their findings are the PARSE
section below.

---

## STRUCTURE / CHEMISTRY / DATA FILES — defects

### G-D1 · Pressing Apply on the Cell page without changing anything marks the structure's recorded calculation as outdated, and its recorded charge and spin stop being used

- **Where:** `molbuilder/periodicity_gate.py:368` (`apply_edit`), reached from
  `web/static/modify/periodicity.js:578-622` (`applyCell`) and
  `web/static/lib/molview/model-jobs.js:959-990` (`commitPeriodicityOp`).
- **Rule:** `model/structure.md` § 2.2a: *"An edit is meant to OUTDATE the record
  (`structure_modified` / `labels_modified`), which is a MARK ON IT"*. The mark's own docstring
  (`Structure.mark_contract_outdated`) says *"`structure_modified` says the geometry or the cell
  moved"*.
- **What the code does:** `apply_edit` marks the copy before it looks at the op:
  ```python
  s = struct.copy()
  ...
  s.mark_contract_outdated()
  notices: List[dict] = []
  s = _edit(s, op, payload, notices)
  ```
  The browser sends every Apply, whatever is staged. `applyCell` builds the whole block from the
  boxes and calls `commitOp("block", payload)`. `refresh()` disables `pv-apply` only when the canvas
  is empty (`b.disabled = !has`). `commitPeriodicityOp` posts the block unconditionally. Nothing
  compares the block with the box the structure already has.
- **What reads the mark:** `electronic_state._recorded` (`electronic_state.py:238`):
  `if not isinstance(contract, dict) or cal.get("structure_modified"): return None`. The page's
  warning also reads it (*"converged for a cell that is no longer there"*).
- **Failure scenario:**
  1. A person saves a structure out of a finished SIESTA run whose deck stated `net_charge = -1`.
     Its `info.calculation` contract records that charge.
  2. They load it on the Molbuilder tab, open the Cell page to look at the box, and press Apply
     without changing anything.
  3. The answer comes back with `info.calculation.structure_modified = true`.
  4. In a new calculation with `net_charge` left blank, the charge step skips the recorded −1
     (ES7) and falls to the phosphate rule, giving 0. The deck is written with the wrong electron
     count.
  5. The structure is also shown as edited, and its settings as converged for a cell that is gone.
     Neither is true.
- **How sure:** read and traced, Python and JS. The one thing not run is the browser itself.

### G-D2 · Two chemistry helpers compare species labels as bare text, against the chapter's one rule

- **Where:** `molbuilder/chemistry.py:787-793` (`_adjacency`, used by
  `formal_charge_from_phosphates` and `protonate_phosphate_oxygens`) and `chemistry.py:1418-1427`
  (`estimate_partial_charges`, used by `estimate_dipole_moment_debye`).
- **Rule:** `model/chemistry.md` § 3: *"`is_atom(label, symbol)` → bool | **is this atom that
  element?** … The one way to ask — a bare `el == "P"` says a labelled species is not phosphorus"*.
- **What the code does:**
  ```python
  cutoff = _HX_CUT if ("H" in (ei, ej)) else _XX_CUT
  ...
  if ej == "H": nb_h[i].append(j)
  else:         nb_heavy[i].append(j)
  ```
  and `en_i = _PAULING_EN.get(ei, _DEFAULT_EN)` / `if "H" in (ei, ej):`.
  The phosphorus and oxygen tests in the same functions do use `is_atom`.
- **Failure scenarios:**
  - **Charge:** a hydrogen labelled `H1` (its own basis) is taken as a heavy atom, with the 1.95 Å
     cutoff and counted in `nb_heavy`. A bare phosphate oxygen with an `H1` hydrogen-bonded to it
     1.9 Å away then has two "heavy" neighbours and is not counted as non-bridging. The charge step
     (`formal_charge_from_phosphates`, used when `net_charge` is blank) undercounts −1 for it.
  - **Dipole:** an oxygen labelled `O1` gets the default electronegativity 2.20 instead of 3.44, so
    the dipole estimate behind the polar-molecule advice says the molecule is barely polar.
- **How sure:** read and traced. The failure needs labelled species; I did not run a case.

## STRUCTURE / CHEMISTRY / DATA FILES — untrue sentences

### G-U1 · An isolated axis sampled above 1 can be refused
- **Chapter:** `model/structure-periodicity.md` § 2, the axis-kind table, isolated row, k column:
  *"Γ — above 1 is warned, never refused"*.
- **Code:** `molbuilder/kmesh.py:328-340` (`check`). On a transport calculation's SCF grid, an
  isolated in-plane axis sampled more than once while the other in-plane axis is periodic (`_a_stripe`)
  is `Issue("error", …, "… such a junction is refused for now")`. Only the case after it,
  `elif axis.role == "gamma" and axis.count > 1:`, is a warning.
- **How sure:** read.

### G-U2 · Empty XYZ text is not answered as "XYZ is empty"
- **Chapter:** `model/structure.md` § 2.3, *`from_xyz` requirements*: *"an empty one *XYZ is
  empty: …*"*.
- **Code:** `molbuilder/structure.py:2130-2139`. Empty or whitespace-only text reaches
  `if not images: raise ValueError("XYZ holds no frames")`. The `StopIteration` branch that says
  "XYZ is empty" is not reached.
- **Measured:** `Structure.from_xyz('')` and `from_xyz(' ')` both raise `ValueError: XYZ holds no
  frames`.
- **How sure:** run.

### G-U3 · The species order a person sets is not used for the description's list of pseudopotentials
- **Chapter:** `model/chemistry.md` § 3a, site table: *"`describe.pseudo_species`, prep's
  pseudopotential copy ✅ | so the pseudos a description names are listed in deck order"*. Also
  *"The override is resolved inside the one entry, not at each call site, so no emitter can honour
  it alone."*
- **Code:** `molbuilder/jobset/_cli.py:721`: `pseudo_species=species_order(struct.elements)`.
  `cfg.species_order` is never passed, although `cfg` is in hand there. Likewise
  `siesta/seam.py:141`, `transport/rung.py:48` and `transport/reuse.py:293`. Only
  `siesta/input.py:635` and `transport/transiesta.py:249` pass the person's order. So the order is
  resolved at each call site, not in the one entry.
- **Scenario:** a person sets `species_order = ["Au","S","C","H"]` and runs `molbuilder describe`.
  The deck lists Au, S, C, H; the description lists C, H, S, Au.
- **Effect:** only the order of a list. The copies themselves are by set, so nothing is wrong in a
  run.
- **How sure:** read.

### G-U4 · The transport data file's embedded structure keeps the view's field
- **Chapter:** `model/structure.md` § 2.2, *A field of the view*: *"a result file's embedded
  structure, `to_dict(view=False)`, leaves it out too"*.
- **Code:** `molbuilder/transport/datafile.py:951` writes `structure_json` from
  `junction.to_dict()`, and `:790` passes `structure_values(junction.to_dict() …)`. Both use the
  default `view=True`, so `/structure`'s `structure_json` carries `"order_unstated": false`.
- **Effect:** always `false` on that road, so nothing reads wrong. The sentence is still untrue for
  this file.
- **How sure:** read.

### G-U5 · The transport rungs write coordinates to 10 decimals, not 8
- **Chapter:** `model/structure-periodicity.md` § 6.0, *The record* (line 515): *"to 8 decimals
  (PySCF and the transport rungs write their coordinates to 8, SIESTA to 10: the record is inside
  every tolerance either way)"*.
- **Code:** `transport/transiesta.py:268-270` writes a rung's coordinates through
  `siesta.deck_lines.coordinates_rows`, which is `f"{x:.10f} {y:.10f} {z:.10f} …"`
  (`siesta/deck_lines.py:52`). Only PySCF writes 8 (`pyscf/input.py:164`). `chemistry.md` § 3a says
  the same thing as the code: *"Both write their species, lattice and coordinate rows through one
  writer, `siesta/deck_lines.py`"*.
- **Effect:** none on a run. The sentence is untrue.
- **How sure:** read.

### Checked and found as the chapters say
- **The codec** (`StructureCodec.pair/files/write/write_moved/load`):
  - both-or-neither write order;
  - the pin over atom lines, cell and axis kinds;
  - the version set {7…17}; retired keys at all three gates; species labels;
  - the atom-index map checked against `atom_order`;
  - F4's `not_stated` / `a_copy`; the engine-record comment line.
- **Placement and checks:** the placement rule and its checks (`cell.engine_offset`, `to_engine`,
  `require_placed`, `entering`, `check` / table B); the periodicity door's ops and their receipts.
- **Structure object:**
  - frame sets: `frame_at` offset rule, `with_frames`, `take`, `restored`;
  - `concat`'s owners table; the `customized` doors;
  - PDB TER and TITLE handling;
  - `to_wire`'s `box_corner`.
- **Order records:** `atom_permutation` (`Permutation`, `AtomIndexMap` makers, readers and
  relisting).
- **Chemistry and units:** the charge step's order (transport rule, stated, recorded, phosphate);
  the phosphate arithmetic; the species-order rule's three cases; the constants.
- **Steps file:** every name, kind and unit in `datafiles.md` § 9.5 against `stepsdata.RUN_VARS`,
  `STEP_VARS` and the SCF group.

## PARSE — defects and untrue sentences

This part was reviewed separately: `parse.md` end to end and every module under `parse/` whole.
I then checked each cited line myself; where its line numbers were off, the right ones are given
here. The chapter's worked examples were checked against `SiestaReader`, `_run_ending` and the
TranSIESTA device example, and all of them matched.

### P-D1 · The SIESTA reader ignores the unit SIESTA prints on its coordinates heading
- **Where:** `parse/engines/siesta_grammar.py:527` (`COORDS_BEGIN = "outcoor:"`),
  `parse/engines/siesta_reader.py:486-510` (`_on_coords_start` tests only the
  relaxed/unrelaxed marker; `_consume_coords` keeps `[parts[-1], x, y, z]`), and
  `parse/engines/siesta.py:129-130`:
  ```python
  elements=[row[0] for row in coords],
  positions=np.array([row[1:4] for row in coords], dtype=float)
  ```
  No unit check is made anywhere on this road.
- **Rule:** `parse.md` § 7 rule 9: *"A token the parser does not recognise is recorded AND flagged
  … never quietly accepted as what the run did."* § 5.2: a folder no calculation claims has its
  files listed, each as the registry reads it.
- **What SIESTA does** (`Src/outcoor.f`):
  - lines 64-66: the default output format is `NotScaledCartesianBohr`, set by `AtomCoorFormatOut`;
  - lines 96-112: the heading says `(Bohr):`, `(scaled):` or `(fractional):`, and the numbers are in
    that unit;
  - lines 142-143: the last column is the species label, not the element.
- **Failure scenario:** a deck run by hand with no `AtomicCoordinatesFormat` (SIESTA then uses Bohr)
  or with `Fractional`. Its `.out` is opened from the Results picker, `molbuilder watch parse` or
  `postprocess.parse`.
  - Every frame is 1.89× too large, or holds fractions read as Å.
  - The `.MD.nc` upgrade matches no frame to 1e-4 Å, so it cannot fix them.
  - A label such as `Au_s` becomes the element, and `Structure` accepts it.
- **Who is affected:** only outputs molbuilder did not write. molbuilder's own decks always write
  `AtomicCoordinatesFormat Ang` (`siesta/deck_lines.py:82-84`), and their labels are element names.
- **How sure:** read and traced against SIESTA's source.

### P-D2 · An open-shell PySCF relaxation's geomeTRIC trajectory gets each frame's SCF from the wrong evaluation
- **Where:** `parse/engines/pyscf.py:139` (`_read_scf_history`) and `:224`:
  `scf_for_step = (scf_history[i] …)`. The log's SCF runs are paired with geomeTRIC's frames by
  position, i with i.
- **Rule:** `parse.md` § 5a: PySCF's own log supplies *"each step's SCF history"*. Rule 1: the sibling
  file *"only swaps values into frames that already exist"*.
- **Why it shifts:** with `spin_treatment = "unrestricted"`, the deck runs a stability block first
  (`pyscf/input.py:621-624`). Its kernel and up to three restarts (lines 963, 996) print `cycle=`
  lines into the same `<stem>.log`. So the log holds one or more extra SCF runs before geomeTRIC's
  first evaluation.
- **Failure scenario:** the person opens `<stem>_geom_optim.xyz` from the Results picker or
  `postprocess.parse`.
  - Frame 0's SCF chart is the stability SCF; frame k shows evaluation k−1's, or earlier after
    restarts.
  - Each frame's SCF ends at another energy than the frame's own.
  - `pyscf_log_scf_upgraded` still says every frame was upgraded.
- **Not affected:** the `.molwatch.log`, the default view of a PySCF optimization.
- **How sure:** traced through the deck emitter, PySCF and geomeTRIC sources, and the reader. Not
  run on a real log.

### P-D3 · One report line with a non-number `sent_at` makes the This machine tab's report list fail with a 500
- **Where:** `parse/instruments/reports.py:50` and `:52`:
  `rows.sort(key=lambda d: -float(d.get("sent_at") or 0))`, and the same `float(...)` in the
  `since` filter.
- **Rule:** the module's own role, *"a line that does not read skipped and counted"*. `parse.md` § 5c
  gives the answer as *"reports (newest first), total, skipped"*.
- **Why it can happen:** the listener's `_clean` (`web/blueprints/notify.py:235-250`) keeps any
  string as `sent_at`. That module calls this text attacker-controlled once a key is known.
- **Failure scenario:** one validly signed report carries `"sent_at": "x"`. From then on
  `GET /api/notify/reports` raises `ValueError` and answers 500, so the tab lists nothing until that
  log rotates away.
- **How sure:** read and traced. It needs a crafted or buggy report; molbuilder's own monitor
  always sends a number.

### P-U1 · The recorded contract carries three electronic-state keys the deck may not state
- **Chapter:** `parse.md` § 5b (line 1215): *"`contract.contract_of(deck)` reads back … the engine, the
  fields the deck states (and only those), the deck's name, and its `sha256`."* `contract.py:31`
  says the same.
- **Code:** `parse/fdf.py:239-241` defaults `net_charge = 0`, `spin_treatment = "restricted"` and
  `unpaired_electrons = 0`. `parse/engines/siesta_contract.py:52-54` copies them, and `:62` drops only
  `None`. So all three are in every SIESTA contract.
- **Effect:** a deck that never mentions `Spin` or `NetCharge` records them in `info.calculation`
  as if the deck stated them; they are SIESTA's defaults. The code's comment says this is deliberate,
  so it is the sentence that is untrue.
- **How sure:** read.

---

## Totals

**4 defects and 9 untrue sentences.**

- **Structure, chemistry and data files:** 2 defects (G-D1, G-D2) and 5 untrue sentences (G-U1 to
  G-U5).
- **Parse:** 3 defects (P-D1 to P-D3) and 1 untrue sentence (P-U1).

The most severe:
1. **G-D1:** pressing Apply on the Cell page with nothing changed marks the recorded calculation
   outdated. A blank charge then stops using the recorded one, so the deck can get the wrong
   electron count.
2. **P-D1:** a SIESTA `.out` printing Bohr or fractional coordinates is read as Å, and its species
   labels as elements. Only outputs molbuilder did not write.
3. **P-D2:** an open-shell PySCF relaxation's geomeTRIC trajectory shows each frame the previous
   evaluation's SCF.
4. **P-D3:** one malformed `sent_at` in a report makes the report list answer 500.
5. **G-D2:** the phosphate charge and the dipole estimate compare species labels as bare text, so
   labelled species (`H1`, `O1`) are read wrong.

**Coverage limits:**
- For `datafiles.md`, I checked § 9.0, the steps file (§ 9.5) row by row, and the digest (§ 9.6) and
  the optical, Raman and chart files (§§ 9.3, 9.4, 9.7) by name, kind and unit.
- § 9.1 (the transport file) and § 9.2 (the vibration result) were not checked row by row: their
  writers (`transport/datafile.py`, `spectra/datafile.py`) build names through per-group prefixes I
  did not trace.
