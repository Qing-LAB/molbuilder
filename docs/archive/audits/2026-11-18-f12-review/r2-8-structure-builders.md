# F12 review, round 2 — R2-8: the structure model and the builders

Checkout `the checkout` at f578482db (2.77.2). Read only. Nothing was
run that writes: the measurements below are in-memory Python calls in the
`molbuilder` env (`PYTHONDONTWRITEBYTECODE=1` after the first one), and
`git status` is unchanged from the start of the session.

**Totals: 35 findings — 25 in the code (C-1 … C-25), 10 in the tests (T-1 … T-10).**

Round 1's builder findings (H-D1 … H-D8) are not repeated. Two findings here
are the duplicate behind one of them: C-9 is the cause of H-D5, and T-1 is why
H-D7 passes its test.

## What was read

**Code, whole:** `structure.py`, `cell.py`, `workingcopy_structure.py`,
`periodicity_gate.py`, `chemistry.py`, `atom_permutation.py`, `atom_index.py`,
`engine_atom_index.py`, `selection.py`, `annotations_fdf.py`, `modify.py`,
`nucleic.py`, `peptide.py`, `smiles.py`, `pubchem.py`, `residues.py`,
`builders/__init__.py`, `builders/backends/{__init__,_common,_amber,_rdkit,_threedna}.py`.

**Read for the other side of a finding:** `issues.py` 40-100,
`validation/geometry.py` 60-95 and 215-275, `validation/siesta.py` 440-470,
`validation/builders.py` (grep), `web/blueprints/build.py` 200-262,
`web/blueprints/_shared.py` 70-100 and 240-270, `web/blueprints/modify.py`
(grep), `siesta/input.py` 1095-1117, `cli.py` 295-320, and pubchempy 1.0.5's
`Compound.__init__` and `canonical_smiles`.

**Tests, whole:** `test_structure.py`, `test_reserved_label_one_store.py`,
`test_chemistry.py`, `test_chemistry_analyzer.py`, `validation/test_chemistry.py`,
`validation/_helpers.py`, `test_engine_atom_index.py`, `test_fcc_lattice_table.py`,
`test_species_order_is_one_rule.py`, `test_backends.py`, `test_residues.py`,
`test_peptide.py`, `test_smiles_fallback.py`, `test_fcc_cell_shapes.py`,
`test_build_findings.py`.

**Tests, in part** (an outline of every test, plus the ranges named):
- `test_periodicity_gate.py`: 1-60, 960-972, 1517-1640
- `test_cell.py`: 455-520
- `test_seam.py`: 30-82
- `test_add_slab.py`: 170-190
- `test_modify.py`: 1205-1349
- `test_structure_replace_door.py`: 1-60
- `test_structure_pair_one_generator.py`: 230-300
- `test_dna_double_strand.py`: 1-100
- `test_nucleic.py`: 90-190 and 420-439
- `test_pubchem.py`: 1-72
- `test_atom_selection.py`: 1-66
- `conftest.py`: 796-846
- `support/road.py`: 130-205

**Tests, not read:** `test_structure_authority_roundtrip.py`,
`test_structure_periodicity.py`, `test_atom_annotations.py`, `test_frame_sets.py`,
`test_fcc_measure.py`, `test_backend_unavailable_is_not_a_crash.py` (outline
only), `test_one_door_reads_a_structure.py`, `test_load.py`, `test_pdb_ter.py`,
`test_atom_identity_end_to_end.py`, `test_geometry_pin.py`,
`test_smiles_and_siesta.py`, `test_electrode_pair_seam.py`,
`test_molstruct_json.py`, `test_sidecar_annotations.py`, `test_held_atoms.py`,
`test_frame_set_meanings.py`, and the roughly 60 further files that only import
`Structure`. 85 test files import `structure`, so reading every one whole was
not possible in one pass.

---

## CODE

### C-1 — The helix-form and terminal-state vocabularies have no home; every door types them, and the build door checks neither (medium)
- **The one home that exists:** `builders/backends/__init__.py:53-54`, `DEFAULT_FORM = {"dna": "B", "rna": "A"}` and `DEFAULT_TERMINAL = "OH"`. It holds the defaults only, not the set of allowed words.
- **Where the words are typed instead:**
  - `cli.py:305`: `type=click.Choice(["OH", "5P", "3P", "PP"])`
  - `cli.py:308`: `type=click.Choice(["B", "A", "Z"])`
  - `nucleic.py:182`: `elif (form or DEFAULT_FORM["dna"]).upper() not in ("B", "A", "Z"):`
  - `_rdkit.py:87`: `f"valid: 'OH', '5P', '3P', 'PP'"`
  - The set is also implied, separately, by the keys of `_threedna._TERMINAL_BUILT` (`:223`), `_threedna._FIBER_FLAGS` (`:213-218`) and `_rdkit._FLAVOR` (`:34-39`).
- **What goes wrong.** Neither `builders.build` nor the route refuses an unknown word. Measured with `backends.built(b, "dna", "B", "XX")`:
  - threedna: raw `KeyError 'XX'`;
  - rdkit: raw `KeyError ('dna', 'XX')`;
  - amber: answers `terminal='OH'`, so the after-build check would say *"The terminal XX … was built OH"*.
  - Read-and-traced (not run): through `/api/build/molecule` the KeyError reaches the route's last `except Exception` → 500.
  - A form added to `_FIBER_FLAGS` (say a new 3DNA form) stays refused by `nucleic.py:182` and by the CLI.
- **Owner:** `builders/backends/__init__.py`, beside the defaults.
- **Smallest fix.**
  - Declare `FORMS` per kind and `TERMINALS` there, next to the defaults.
  - `builders.build` refuses a word outside them, naming them.
  - The CLI `Choice`s, `nucleic.py:182` and `_rdkit`'s message read them.
  - Each backend table is keyed over them.
- **Sureness:** read both sides; the two KeyErrors measured.

### C-2 — "A tool this machine lacks" is raised as four different exception types; the route answers one as advice and the rest as faults (medium)
- **The four types:**
  - `BackendUnavailable`, said to be *"the door every builder already uses for a dependency this install lacks"*:
    - `chemistry.py:1077-1084`, when `add_hydrogens` finds neither OpenBabel nor RDKit;
    - `nucleic.py:168-174`;
    - every backend.
  - `RuntimeError`: `chemistry.py:1310-1315`, when `relieve_clashes` finds no OpenBabel: `raise RuntimeError("relax_clashes needs OpenBabel …")`.
  - `ImportError`:
    - `peptide.py:41-45` (PeptideBuilder);
    - `smiles.py:34-38` (RDKit);
    - `pubchem.py:220-224` (pubchempy).
  - `ValueError`: `smiles.py:111-118`, *"… and the OpenBabel fallback is not installed."*
- **What goes wrong.** `web/blueprints/build.py:226-257` maps:
  - `BackendUnavailable` → 200 advisory;
  - `ImportError` → 500 *"missing dependency"*;
  - `ValueError` → 400;
  - anything else → 500.

  So the same missing OpenBabel is three answers:
  - advice on an H-add;
  - a server fault on a duplex with `relax_clashes`;
  - a bad request on a SMILES fallback.

  A missing RDKit on the SMILES panel is a 500 *"crash"*, which the route's own comment says a missing tool must never be.
- **Owner:** `chemistry.BackendUnavailable` (its docstring names RDKit and OpenBabel).
- **Smallest fix.** Each of the five raises becomes `BackendUnavailable(..., missing=…)`.
- **Sureness:** read every raise and the route's handlers.

### C-3 — Atomic number → symbol is a hand-typed table beside ASE's, while symbol → atomic number reads ASE's (medium)
- **Places:**
  - `chemistry.py:83-102`, `PERIODIC_TABLE: List[str] = ["", "H", "He", … "Lr"]`, read by `symbol_for_z` (`:104-118`). Its readers are `parse/coords/siesta_xv.py:114` and `parse/engines/siesta_mdnc.py:194`.
  - `chemistry.py:319-325`, `atomic_number` → `from ase.data import atomic_numbers`. Also `atomic_mass` (`:494`) and `Structure.from_pdb` (`structure.py:2318`).
- **Owner:** ASE's table. `atomic_mass`'s own docstring (`:478-481`) says *"Typing 118 masses into this file would be a second source of truth"*, and the same argument applies to the symbols.
- **What drifts (measured).**
  - `PERIODIC_TABLE[1:]` equals `ase.data.chemical_symbols[1:104]` today, but stops at Z = 103.
  - `atomic_number("Rf") == 104`, while `symbol_for_z(104)` raises *"outside the table this build carries"*.
  - So a structure labelled `Rf` passes every check, yet a run's `.XV` of the same atoms cannot be read back.
  - The readers do not travel beside a job: both import `molbuilder.chemistry`, which already reaches ASE lazily.
- **Smallest fix.** `symbol_for_z` reads `ase.data.chemical_symbols` (Z ≥ 1), and the list goes.
- **Sureness:** read both; drift measured.

### C-4 — Bonded neighbours are worked out three times, with the two cut-offs typed three times, each comparing raw labels instead of `is_atom` (medium)
- **Places:**
  - `chemistry.py:772-794`: `_HX_CUT = 1.30`, `_XX_CUT = 1.95`, and `_adjacency` with `if ej == "H": nb_h[i].append(j)`.
  - `chemistry.py:1379-1426`, `estimate_partial_charges(…, bond_cutoff: float = 1.95, hx_cutoff: float = 1.30)`. It runs its own O(n²) loop with `if "H" in (ei, ej)`.
  - `builders/backends/_amber.py:196-216`, `_fix_methylene_hydrogens`: `HX_CUT = 1.30`, `XX_CUT = 1.95`, and the same loop again with `if ej == "H"`.
- **Owner:** `chemistry._adjacency`. Element comparisons belong to `chemistry.is_atom` (`:295-316`): *"**The one way to ask.** `el == "P"` is the wrong question the moment a user labels a species"*. `test_chemistry.py:488` claims *"Every element comparison in `molbuilder/` goes through this"*, which these three loops contradict.
- **What goes wrong (measured).**
  - Water labelled `O, H1, H2`: `_adjacency` returns every atom as a heavy neighbour of the others, and no hydrogens. `H1`–`H2` (1.5 Å) are "bonded" under the 1.95 Å heavy cut-off.
  - A labelled hydroxyl hydrogen is therefore invisible to `formal_charge_from_phosphates` and `protonate_phosphate_oxygens`.
  - Changing one cut-off (say `_XX_CUT` for a Si–Si bond) silently leaves the other two.
- **Smallest fix.**
  - One neighbour function in `chemistry`, asking `is_atom(…, "H")`.
  - `estimate_partial_charges` and `_fix_methylene_hydrogens` take its lists.
  - The cut-off parameters go.
- **Sureness:** read all three; the label effect measured.

### C-5 — "Is residue r joined to r−1 by an O3′–P bridge" is computed twice, keyed differently, with 1.8 Å typed twice (medium)
- **Places:**
  - `builders/backends/_common.py:58-91`, `verify_backbone_connectivity(struct, max_O3_P: float = 1.80)`. Keyed by `(chain_ids[i], residue_ids[i])`; bridged when `d <= max_O3_P`.
  - `validation/geometry.py:226-249`, `_check_polymer_orientation`. Keyed by `residue_ids[i]` alone; `if d < 1.8:  # bridged`.
  - A third reader of the same bridge, by atom name: `chemistry.relieve_clashes` keeps `{"O3'", "P"}, {"O3*", "P"}` bonds (`chemistry.py:1337`).
- **What goes wrong.**
  - The backends' check knows a duplex's strands are separate chains. The validator's does not, and treats any atom named `P` as a backbone P. That is the mechanism of round 1's H-D2: phospho-Ser → a false *"disconnected backbone"* warning.
  - A threshold change in one place is not in the other: a builder could pass its own check and get the validator's warning, or the reverse.
- **Owner:** none yet. It is a structure fact, so it belongs in L1 `chemistry`, answering the bridged pairs per chain. Both callers then ask it.
- **Smallest fix.** One `backbone_bridges(struct)` in `chemistry`, keyed by chain, with the threshold named once. The validator and the backends read it.
- **Sureness:** read both sides.

### C-6 — The Cell page's door parses each field twice: once for the `block` op, once for the single-field op (medium)
All in `periodicity_gate.py`:
- **vacuum:**
  - `:300-311` (`_apply_block`);
  - `:393-407` (`_edit`).

  Both say *"vacuum must be 3 non-negative floats (Å), or null to clear it"*.
- **axis_kind:**
  - `:262-264` and `:425-428`, both `raise ValueError("axis_kind must be 3 of isolated|transport|periodic")`. That is a hand-typed copy of `structure.AXIS_KINDS` (`structure.py:117`).
  - `Structure.__post_init__` (`structure.py:965-970`) refuses the same thing in its own words.
- **cell:** `:266-273` and `:464-468`.
- **origin (box_corner):** `:283-298` and `:496-505`. The messages already differ: *"or null for Automatic"* appears in one only.
- **"a periodic axis needs an explicit cell"** is said four times in four sentences: `:278-281`, `:408-412`, `:433-436` and `:451-454`.
- **"an origin is assigned on a typed cell only"**: `:295-298` and `:496-499`.
- **"The first error the one checker found → ValueError"** is written twice: `_refuse_on_error` (`:122-126`) and `validate_periodicity` (`:185-196`).
- **What drifts.** A rule changed for one op, such as allowing a zero vacuum on a transport axis, is still enforced by the other. The two refusal messages for the same bad value already differ.
- **Owner:** `periodicity_gate`.
- **Smallest fix.**
  - One reader per field (vacuum, kinds from `AXIS_KINDS`, cell, corner), used by both `_apply_block` and `_edit`.
  - Or express each single-field op as a `block` built from the structure's current state.
- **Sureness:** read whole.

### C-7 — One finding (`Issue`) goes out on the wire in two shapes, each claiming to be the only one (medium)
- **Shape 1:** `issues.py:77-92`, `Issue.to_json`. Its docstring: *"THE key set, in one place"*: `severity, message, where[, stage]`. Used by `_shared.issues_to_json` (`_shared.py:84`, adding `card`), the build route (`:1312`) and checkpoint (`:92`).
- **Shape 2:** `periodicity_gate.py:150-158`, `_wire`. Its docstring: *"THE ONLY PLACE the wire shape is written"*:
  ```python
  return {"severity": i.severity, "message": i.message,
          "where": i.where, "about": i.where.split(".")[0]}
  ```
  It is used through `notices_for_report` for:
  - the builder's findings (`build.py:290`);
  - the read notices (`build.py:572-579`, `_shared.py:783`, `:836`);
  - the geometry findings (`_shared.py:266`);
  - the watch page (`watch.py:245`).

  So it is not only the cell's.
- **What drifts.** A key added to `Issue.to_json`, such as a stage on a geometry finding, never reaches a notice. `_wire` drops `stage` today.
- **Owner:** `Issue.to_json`.
- **Smallest fix.** `_wire` starts from `i.to_json()` and adds `about`. The error→warn softening stays in `notices_for_report`.
- **Sureness:** read both sides and their callers.

### C-8 — The duplex rules are enforced in both `nucleic` and the backends, with two exception types for one condition (medium)
- **"B-form only" for two strands:** `nucleic.py:177-181` and `_threedna.py:548-552`, the same sentence twice (*"two-strand (arbitrary / mismatched) duplexes are B-form only …"*).
- **"Equal length":** `nucleic.py:195-199` and `_threedna.py:553-558`.
- **"A duplex needs 3DNA":**
  - `nucleic.py:160-174` raises `BackendUnavailable`, and its comment says *"one condition wants one type"*.
  - `_amber.py:61-64` and `_rdkit.py:67-70` raise `ValueError` for the same condition.
- **"The DNA forms":** `nucleic.py:182` types `("B", "A", "Z")`, which is what `_threedna._FIBER_FLAGS` holds (C-1).
- **What drifts.**
  - Teach `rebuild` A-form, and `nucleic` still refuses it.
  - Call `dispatch(..., backend="amber", double_strand=True)` directly (a script, the Python API), and the person gets a `ValueError` (400) where the build door gives advice.
- **Owner:** the backend that has the limit. `_threedna` for B-only and equal length; the duplex capability as a declared field of each backend module.
- **Smallest fix.**
  - Each backend states `duplex: bool`; `nucleic` asks it.
  - The B-only and equal-length checks live once, in `_threedna`, raised before it runs anything.
- **Sureness:** read all of them.

### C-9 — A DNA sequence is pre-parsed and letter-filtered in five places before (and beside) the one grammar — the duplicate behind H-D5 (medium)
- **The one grammar:** `residues._parse`, reached by `parse_dna_sequence` / `parse_rna_sequence` (`residues.py:284-311`), which strip the 5′/3′ labels with `_strip_directionality`.
- **The second parser:** `nucleic._strip_direction` (`nucleic.py:38-40`) strips the labels first, then:
  ```python
  core = "".join(c for c in body if c.isalpha()).upper()
  ```
  `build_dna` then calls `parse_dna_sequence(seq)` (`:189`), which strips labels again, now a no-op.
- **Three more letter filters, each re-cleaning what the grammar already checked:**
  - `_threedna._located` (`:384`) and `_build_arbitrary_duplex` (`:553`);
  - `_amber.build` (`:79`);
  - `_rdkit.build` (`:93`): `one_letter = "".join(c for c in sequence.upper() if c.isalpha())`.
- **A list re-keyed by hand:** `nucleic.py:190`, `cleaned = "".join(c[1] for c in codes)`. It turns `DA` back into `A` by string position, where `residues.DNA_THREE_TO_ONE` (`residues.py:133`) is the table. RNA relies on its code being one letter (`:236`).
- **What goes wrong:** round 1's H-D5. The first filter drops `-`, digits and brackets, so `"AT3GC"` builds ATGC and `"[DA]TGC"` becomes `DATGC`. RNA's single path refuses the same input.
- **Owner:** `residues.parse_dna_sequence`.
- **Smallest fix.**
  - `parse_dna_notation` splits only the `ds,` prefix and the comma, and hands each strand to `parse_dna_sequence`.
  - The backends take the one-letter string through `DNA_THREE_TO_ONE` / `RNA_THREE_TO_ONE`.
  - Their filters go.
- **Sureness:** read-and-traced; the behaviour was measured in round 1.

### C-10 — The `frozen_atoms` door onto the label store has its own index rule, looser than the store's (medium)
- **Places:**
  - `structure.py:1701-1715`, the setter: `kept = sorted({int(i) for i in indices})`, then `self.regions[FROZEN_LABEL] = kept`.
  - The store's one validator: `structure.py:181-192` (`_atom_index`, which refuses `"2"`, `True` and `1.7` and checks the range), called by `_validate_regions` (`:1532-1534`).
- **What goes wrong (measured).**
  - `Structure(..., frozen_atoms=["2", True, 1.7])` → `{'frozen_atoms': [1, 2]}`.
  - The same list through `regions={FROZEN_LABEL: ["2"]}` is refused (*"atom index must be an int; got str"*).
  - After construction, `s.frozen_atoms = [99]` on a 3-atom structure is stored without a check. It is refused only at the next `copy()`, far from where it was written.
  - The same fact (which atoms carry the reserved label) is therefore accepted by two rules. `test_reserved_label_one_store.py:196` claims *"One validator, because one store"*.
- **Owner:** `_validate_regions` / `_atom_index`.
- **Smallest fix.** The setter writes the raw list into the store and runs `_validate_regions(self.n_atoms)` (after `regions` exists). No `int()` of its own.
- **Sureness:** measured.

### C-11 — A second RDKit-molecule → `Structure` converter, and the ETKDG embed recipe written twice (medium)
- **Converter:**
  - `smiles.py:148-162`, `_struct_from_rdkit`: `atom_names.append(f"{sym}{pdb_numbering(i)}")`.
  - `builders/backends/_common.py:13-55`, `rdkit_mol_to_structure`. For a molecule with no PDB residue info (every SMILES molecule) it writes the same `f"{atom.GetSymbol()}{pdb_numbering(i)}"` names. Its residue columns are 1 / `MOL` / `A`, which `Structure` fills anyway.
  - `_common`'s version also refuses a molecule with no conformer.
- **Embed recipe:**
  - `smiles.py:122-129`, `_rdkit_embed`: ETKDGv3, `randomSeed = seed` (default `0xF00D`, `:58`), retried with `useRandomCoords = True`.
  - `_rdkit.py:104-112`: the same steps again, with `0xF00D` typed again.
- **What drifts.** A seed or embedding change (say ETKDGv3 → v4, or a srETKDG for macrocycles) lands in one builder only. SMILES and nucleic RDKit structures then diverge in reproducibility.
- **Owner:** `_common.rdkit_mol_to_structure` for the conversion, and `smiles._rdkit_embed` for the embed (or move it to `_common`).
- **Smallest fix.** `smiles` calls `rdkit_mol_to_structure`; `_rdkit.build` calls the one embed function with the one seed.
- **Sureness:** read both pairs.

### C-12 — "Is this an atom of the structure" is checked by five hand-written loops beside `checked_indices`, "the one bounds check" (low-medium)
- **The owner:** `atom_permutation.py:275-291`, `checked_indices`. It refuses a bool, a non-int and an out-of-range index, naming them; `AtomIndexError` is a `ValueError`. `structure.py` already reaches it (`_checked_indices`, `:239-247`).
- **The same check, re-written:**
  - `structure.py:181-192`, `_atom_index`: the same three tests, its own words (*"out of range [0, n)"*).
  - `modify.py:134-138`, `add_atom`: raises `IndexError(f"anchor_index {anchor_index} out of range …")`.
  - `modify.py:345-349`, `orient_along_axis`: `IndexError`.
  - `modify.py:255-261`, `_moved_subset`: `ValueError` with `atom_words`.
  - `selection.py:302-308`, `ByClick`: `SelectionError`.
- **What drifts.**
  - Three exception types and four wordings for one refusal. A route that catches `ValueError` (C-2's table) answers 400 for one index edit and 500 for another (`IndexError`).
  - None of the four modify/selection loops refuses `True` or `"3"`.
- **Smallest fix.** Each asks `checked_indices(idx, n, what=…)`. `selection` wraps the refusal in its own `SelectionError` if it must.
- **Sureness:** read all; the exception mapping read in `build.py`. The modify routes were not traced.

### C-13 — The fcc surface set is kept in four lists, and the electrode metals in a list beside the table that holds them (low-medium)
- **Planes:**
  - `modify.py:566`: `SUPPORTED_FCC_PLANES = ("100", "110", "111")`
  - `modify.py:586-590`: `FCC_ORTHOGONAL_CHOICES` keys
  - `modify.py:637`: `builders = {"100": fcc100, "110": fcc110, "111": fcc111}`, with its own message (*"expected '100', '110', or '111'"*)
  - `cell.py:1312`: `STACKING_PERIOD = {"111": 3, "100": 2, "110": 2}`
  - The route checks `plane not in SUPPORTED_FCC_PLANES` (`web/blueprints/modify.py:177`, `:628`), while `add_slab` checks the builder dict (`modify.py:854`). `add_slab` even carries a `# pragma: no cover` branch for when the lists disagree (`modify.py:858-865`: *"the two lists are separate facts and this is the message if they ever disagree"*).
- **Metals:** `modify.py:564`, `SUPPORTED_FCC_ELEMENTS = ("Au", "Ag", "Cu", "Ni", "Pt", "Pd")`, beside `data/fcc_lattice.json`'s `metals` (measured: equal today). The comment calls adding a metal *"a two-step change"*.
- **What drifts.**
  - A plane added to `SUPPORTED_FCC_PLANES` passes the route and is refused by the builder.
  - A metal added to the JSON is served in the lattice table (`/api/modify/meta`) but refused by `_check_fcc_element`.
- **Owner:**
  - Planes: one table keyed by plane, carrying ASE's builder, the orthogonal choices and the stacking period (the last can stay in `cell` and be keyed off it).
  - Metals: the JSON. Its `system` field already says fcc. The import-time worry the comment gives is met by the existing lazy cache.
- **Smallest fix.** Derive `SUPPORTED_FCC_PLANES` from the one plane table, and `SUPPORTED_FCC_ELEMENTS` from the loaded table.
- **Sureness:** read all; equality measured.

### C-14 — Each engine's atom-numbering base is stated twice — in a named function and in the dispatch table — and the binding test the module promises does not exist (low-medium)
- **Places:**
  - `engine_atom_index.py:36-43`: `def siesta_numbering(i): return i + 1`.
  - `:57-65`: `geometric_numbering` → `i + 1`.
  - `:68-76`: `pdb_numbering` → `i + 1`.
  - `:79-87`: `pyscf_numbering` → `i`.
  - The same facts again in `:102-107`, `_ENGINE_INDEX_BASE = {"siesta": 1, "geometric": 1, "pyscf": 0, "pdb": 1}`.
  - `:98-99` says *"A test binds the named functions above to `convert_to_engine_numbering` so the two surfaces cannot drift"*. `tests/test_engine_atom_index.py` instead pins each side to its own literal (`:9-10` `== i + 1`; `:21-23`). No test anywhere compares the two (grep).
- **Also:** `pyscf_numbering` has no product caller (grep: only its test). Its docstring keeps it *"so any future mol.atom index emission routes through here"*.
- **What drifts.** Change the table's `"siesta"` base and every `siesta_numbering` caller (the constraints block, `siesta/input.py:1097`) keeps the old one. The parser's inverse then disagrees with the deck.
- **Owner:** `_ENGINE_INDEX_BASE`.
- **Smallest fix.** The named functions are `convert_to_engine_numbering(i, "<engine>")`, and the unused one goes.
- **Sureness:** read both; grep for callers.

### C-15 — "Which residue names are amino acids" is a hand-kept list of what `residues` already holds (low-medium)
- **Places:**
  - `chemistry.py:139-144`, `_AMINO_ACID_RESIDUE_NAMES = frozenset({"ALA", … "VAL", "SEP", "TPO", "PTR", "MLY", "M3L", "ALY"})` (*"Modified residues we support"*).
  - `residues.py:34-39` (`AA_THREE_TO_ONE`) and `:56-125` (`MODIFIED_RESIDUES`).
  - Measured: equal today.
- **What drifts.** A modified residue registered in `MODIFIED_RESIDUES` (the module's own extension point) is not counted by `expected_pH7_peptide_charge`. A peptide built only of new residues reads as "not a peptide" and loses its charge hint.
- **Owner:** `residues` (`chemistry` already imports it, at `:1116`).
- **Smallest fix.** `set(AA_THREE_TO_ONE) | set(MODIFIED_RESIDUES)`.
- **Sureness:** read both; equality measured.

### C-16 — The default-vacuum notice states SIESTA's "enough vacuum" number as its own literal (low-medium)
- **Places:**
  - `cell.py:442-446`, `cell.vacuum_defaulted`: *"… usually too little for an accurate result: 8 Å per side or more is typical."*
  - `validation/siesta.py:444-445`: `_VACUUM_MIN_NEUTRAL = 8.0` and `_VACUUM_MIN_CHARGED = 25.0`, the threshold `cell.vacuum_thin` actually applies.
  - `structure.py:496` and `:1111` restate both numbers in comments.
- **What drifts.** Raise the validator's neutral threshold and the default-gap notice still tells the person 8 Å is typical, while the validator warns about the 8 Å they then type.
- **Owner:** the validator's value. It has to move to L1 so `cell` can read it: `cell` cannot import `validation`.
- **Smallest fix.** Put the two thresholds in `cell` (or `constants`); `validation/siesta.py` imports them and `cell.check` interpolates.
- **Sureness:** read both.

### C-17 — The same builder option, `add_hydrogens`, is a bool on the peptide and a three-word string on DNA/RNA (low-medium)
- **Places:**
  - `peptide.py:52`: `add_hydrogens: bool = True`.
  - `nucleic.py:85` and `:226`: `add_hydrogens: str = "auto"`, read by `normalise_h_mode` (`:245-251`), which refuses booleans.
  - `tests/test_nucleic.py:166-172` explains why: *"`True` once meant `auto` here and *always* for a peptide: one word, two meanings, so a person states `auto`, `on` or `off`"*. The peptide side kept the bool.
- **What goes wrong.** A script doing `build(kind, …, add_hydrogens=x)` over kinds must pass a bool to one and a word to the other. `build_peptide(seq, add_hydrogens="off")` adds hydrogens, because a non-empty string is true.
- **Smallest fix.** `build_peptide` takes the same three words through `normalise_h_mode`. `"auto"` for a peptide means "on": PeptideBuilder writes heavy atoms only.
- **Sureness:** read both.

### C-18 — The modified-residue patch removes and appends atoms by rebuilding every per-atom column by hand (low)
- **Places:**
  - `peptide.py:166-222`, `_patch_residue`: it builds a `keep_mask`, then filters six per-atom lists one by one, appends, and constructs a new `Structure`.
  - The structure's own doors already do this: `Structure.without` (`structure.py:1835-1848`) and `_of_atoms` (`:1920-1940`) move every per-atom field. `Structure.replace`'s docstring warns that *"enumerating fields to copy them is how fields come to be forgotten"*.
- **What drifts.** A per-atom column added to `Structure` later is silently dropped by every peptide build with a modified residue. Today the patch's removal finds nothing to remove anyway (round 1 H-D1: every `remove_atoms` name is a hydrogen).
- **Smallest fix.** `struct.without(removed)`, then append the new atoms through the same append door modify uses (C-19).
- **Sureness:** read both.

### C-19 — Three hand-written ways to append atoms to a structure (low)
- **Places:**
  - `modify.py:184-195`, `add_atom`: `Structure(elements=struct.elements + [element], …, **struct._carry_nonatom())`.
  - `modify.py:743-754`, `_finish_slab`: the same hand-assembled constructor.
  - `modify.py:1104`, `append_structure`: `Structure.concat`.
  - "A fresh residue id" is typed twice: `max(struct.residue_ids) … + 1` at `modify.py:177` and `:691`.
- **What drifts.** A rule for joining atoms (e.g. how chains are numbered, or a new per-atom column) has to be written three times. The two hand paths already differ from `concat` in residue renumbering.
- **Smallest fix.** Build the new atoms as a `Structure` and join through `Structure.concat` (it already carries labels, channels and the non-atom block). Then set what the op decides, such as the slab's box.
- **Sureness:** read all; plausible that `concat`'s residue renumbering fits both ops.

### C-20 — "Which atoms form one layer" has two rules (low)
- **Places:**
  - `modify.py:888`, `zs = sorted({round(float(z), 6) for z in all_pos[:, 2]})`.
  - `cell.py:769-789`, `LAYER_TOL_ANG = 0.5` and `detect_layers`, single-linkage.
  - `tests/test_seam.py:37` copies the first rule.
  - `cell.py:799-801`'s comment acknowledges the round-6 rule.
- **What drifts.** An ASE slab whose layer z wobbles by 1e-6 Å (a different `a`, a rotated cell) splits a layer under the round-6 rule but not under `detect_layers`. The registry window `add_slab` takes then counts the wrong layers.
- **Smallest fix.** `add_slab` takes `detect_layers` (it returns centroids; keep each atom within `LAYER_TOL_ANG` of the window's).
- **Sureness:** read both; the drift case is plausible.

### C-21 — The Amber residue-code grammar is written by the backend and read by `residues`, each by hand (low)
- **Places:**
  - `_amber.py:296-310`: `prefix = "D" if kind == "dna" else ""`, `suffix = "5"` / `"3"`, `f"{prefix}{letter}{suffix}"`.
  - `residues.py:132` (`DNA_ONE_TO_THREE`, `"A": "DA"`) already holds the prefix.
  - `residues.nucleobase` (`:167-170`) reads the suffix back off: `if name[-1:] in ("5", "3"): name = name[:-1]`.
- **What drifts.** A terminal residue code with a different suffix (`DA5MP`-style, which `_amber.py:276-279` names as the 5′-P route) is written by one and not recognised by the other. Its base's bond orders are then never set before OpenBabel adds hydrogens.
- **Smallest fix.** The backend composes from `DNA_ONE_TO_THREE` / `RNA_ONE_TO_THREE`. The terminal suffixes are named once in `residues`, beside `nucleobase`.
- **Sureness:** read both.

### C-22 — fcc's nearest-neighbour ratio, and the lattice-constant check, each written twice in `cell.py` (low)
- **Places:**
  - `cell.py:981`: `a=float(np.sqrt(2.0) * d_nn)` in `measure_fcc`.
  - `cell.py:1332-1336`: `_NEAREST_NEIGHBOUR = {"fcc": _math.sqrt(2.0) / 2.0, …}`, the same fact the other way round.
  - The *"lattice constant must be a positive length"* check is typed twice: `:1365-1369` and `:1394-1398`.
- **Smallest fix.** `a = d_nn / _NEAREST_NEIGHBOUR["fcc"]`, and one `_lattice_constant(a)` reader.
- **Sureness:** read.

### C-23 — "An index in this structure → its user index" is implemented twice (low)
- **Places:**
  - `structure.py:1874-1884`, `Structure.map_to_user_index`: `if self.atom_order is None: return _checked_indices()(indices, self.n_atoms)`, else `self.atom_order.map_to_user_index(indices)`.
  - `atom_permutation.py:260-267`, the module function `map_to_user_index(record, engine_indices, n_atoms)`: the same two branches.
  - `map_from_user_index` mirrors it.
  - `atom_permutation` also has two checks of "a list naming every atom once": `_two_lists.indices` (`:76-82`) and `_column` (`:294-306`).
- **Smallest fix.** The structure's method calls the module function with `self.atom_order`. `_two_lists` checks each list with `_column`, then the inverse.
- **Sureness:** read both.

### C-24 — Small restatements inside `structure.py` (low)
- **XYZ precision.** `geometry_lines` rounds to a hard-typed six:
  ```python
  f"{round(float(v), 6) + 0.0:.6f}"  # :2017
  ```
  Its docstring ties that to *"a pair written by the codec (six)"*, which is `XYZ_DECIMALS` (`:123`, used at `:2401-2402`). Change `XYZ_DECIMALS` and the fingerprint lines no longer match the pair's text.
- **The vacuum's length.** `_vacuum_from_stored` (`:543-547`) converts and refuses a non-triple. `__post_init__` (`:995-998`), which runs right after it in `apply_metadata_dict`, does both again with the same message.
- **Fix.** Use `XYZ_DECIMALS`; let `_vacuum_from_stored` keep only the `None` / triple read and leave the check to `__post_init__`.
- **Sureness:** read.

### C-25 — The H-to-heavy ratio is computed two ways (low, plausible)
- **Places:**
  - `nucleic.py:289-292` (the `"auto"` gate): `n_heavy = len(struct.elements) - n_h`. Every non-hydrogen counts, metals and unreadable labels included.
  - `validation/geometry.py:76-79` (the `geometry.h_ratio` note): heavy atoms counted against `chemistry.NONMETALS` only.
- **What goes wrong.** For one built structure, the note a person reads and the gate that decided whether hydrogens were added count different atoms. A metal-containing nucleic structure can sit above 0.5 for one and below it for the other.
- **Smallest fix.** One `h_ratio(struct)` in `chemistry`, unless the two purposes are meant to differ. If they are, each should say so.
- **Sureness:** read both. Whether the difference is intended is not stated anywhere I read.

---

## TESTS

### T-1 — The PubChem fake is a hand-typed stand-in whose `canonical_smiles` carries stereochemistry the real library's never does — why H-D7 passes (medium)
- **Places:**
  - `tests/test_pubchem.py:26-31`:
    ```python
    class _FakeCompound:
        def __init__(self, canonical=None, isomeric=None):
            self.canonical_smiles = canonical
    ```
  - `:56-71`: *"If canonical_smiles is empty / None but isomeric_smiles is set (rare but happens for stereo-rich molecules)"*.
  - The real entry: pubchempy 1.0.5's `Compound(record)` takes a record dict offline, and its `canonical_smiles` is `connectivity_smiles`, *"with no stereochemistry information"* (read in the installed package).
- **What goes wrong.** The tests pin `pubchem.smiles_for_name` against a model of the library that is not the library. The stereo loss (H-D7) cannot fail here, and the "fallback" case the second test describes never happens with the real class.
- **Fix.** Build the compounds through `pubchempy.Compound(record)` with a record carrying both SMILES. Keep the network stubbed at `get_compounds` only; that is the named reason for a fake.
- **Sureness:** read both.

### T-2 — The phosphate diester is typed three times; the shared fixture already exists (low-medium)
- **Places:**
  - `tests/conftest.py:821-843`, `deprotonated_diester`: C5′ at (−2.5, 0, 0) … C3′ at (2.5, 0, 0).
  - `tests/test_chemistry.py:25-52`, `_diester`: the same seven atoms and coordinates, plus optional H.
  - `tests/test_chemistry.py:120-126`, `test_terminal_phosphate_dianion`: the coordinates again.
- **Fix.** One builder in `tests/support/`, taking the protonation, used by `conftest` and both tests.
- **Sureness:** read all three.

### T-3 — `tests/validation/_helpers.py` is a second helper home beside `tests/support/`, and its peptide is hand-made where the road builds one (low-medium)
- **Places:**
  - `tests/validation/_helpers.py:29-44`, `_peptide_struct`: *"Synthetic peptide: one CA atom per residue, no actual chemistry"*.
  - `validation/test_chemistry.py:22-55` uses it.
  - The road is `peptide.build_peptide` (no network, installed in the host env).
- **What goes wrong.** The protonation-hint rule is tested on a structure no builder can produce. A change to how the builder names residues would not fail these tests.
- **Fix.** Use `build_peptide(...)` (or skip as `test_peptide.py` does). Move `_vacuum_cell` into `tests/support/` or drop it for `np.eye(3) * L`.
- **Sureness:** read.

### T-4 — The label store's mechanisms are each pinned in two or three files (low)
- **`copy()` independence:**
  - `test_structure.py:230-238`
  - `test_modify.py:1212-1232`
- **Deletion remaps the labels:**
  - `test_modify.py:1255-1277`
  - `test_reserved_label_one_store.py:158-172`, which also pins the ordinary `"keep"` label's remap, not only the reserved one.
- **`concat` offsets the labels:**
  - `test_structure.py:252-263`
  - `test_reserved_label_one_store.py:175-185`
- **The region index range check:**
  - `test_structure.py:149-159`, `:186-190`
  - `test_reserved_label_one_store.py:196-203`, `:378-398`
- **Fix.** Keep each mechanism in one file, as a table of rows where it is several cases. `test_modify.py:1301-1307`'s own comment already assigns the reserved label to `test_reserved_label_one_store.py`.
- **Sureness:** read.

### T-5 — DNA direction labels are pinned four times across three files, because DNA has two parsers (C-9) (low)
- **Places:**
  - `test_residues.py:76-98` and `:101-113`
  - `test_nucleic.py:427-439`
  - `test_dna_double_strand.py:30-34`
  - The four bad strings `"5'-ATGC"`, `"ATGC-3'"`, `"5'-ATGC-5'"`, `"3'-ATGC-3'"` appear in both `test_residues.py:88-93` and `test_nucleic.py:427`.
- **Fix.** After C-9, keep the grammar's cases in `test_residues.py`; `parse_dna_notation`'s tests keep only the `ds,` / comma split.
- **Sureness:** read.

### T-6 — A test restates the stacking-period table as a literal, while another measures it from real slabs (low)
- **Places:**
  - `test_cell.py:459-465`: `assert cellmod.STACKING_PERIOD == {"111": 3, "100": 2, "110": 2}`.
  - `test_seam.py:43-55` checks the same numbers as an outcome: a seam continues exactly when the layer count is a whole number of `STACKING_PERIOD[plane]`, and `verdict.period == period` on ASE-built slabs.
- **Fix.** Drop the literal pin.
- **Sureness:** read both.

### T-7 — The numbering test pins each copy to its own literal instead of to each other (C-14) (low)
- **Place:** `tests/test_engine_atom_index.py:7-26`: `siesta_numbering(i) == i + 1`; `convert_from_engine_numbering(5, "siesta") == 4`.
- **What goes wrong.** Mutate the table to 0 for siesta: the named function is unchanged and the literal-based assert catches only the dispatch half. The module's promise of a binding test is unmet.
- **Fix.** After C-14, one row per engine: its base, both directions.
- **Sureness:** read.

### T-8 — Tests copy the product's recipes to set up their preconditions (low)
- **Places:**
  - `test_smiles_fallback.py:55-60`: ETKDGv3, `randomSeed = 0xF00D`, a `useRandomCoords` retry. That is the third copy of the embed (C-11), where `smiles._rdkit_embed(AllChem, m, 0xF00D)` is the code's own door.
  - `test_seam.py:37`: `zs = sorted({round(float(z), 6) …})`, `add_slab`'s layer rule (C-20).
- **What goes wrong.** If the product changes its recipe, the precondition still passes on the old one and no longer says what the product does.
- **Fix.** Call the product's helper.
- **Sureness:** read.

### T-9 — A `sys.path` prelude in the middle of a test file, where the suite already imports `support` (low)
- **Place:** `tests/test_periodicity_gate.py:967-969`:
  ```python
  import sys as _sys, pathlib as _pl
  _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
  from support.envelope import from_xyz_with_periodicity as _env_per
  ```
- **The rest of the suite:** `test_backends.py:25` and others do `from support import tooling` without it (`pyproject.toml:245-246`; `conftest.py`).
- **Fix.** Delete the two lines; move the import to the top.
- **Sureness:** read; grep found the same insert in 6 other test files (not in this slice).

### T-10 — Three tests pin one walk of the `auto` order, each with its own fake registry (low)
- **Place:** `test_backends.py:57-96` and `:216-238`: `auto_prefers_amber_then_rdkit`, `auto_falls_through_when_amber_unavailable`, `auto_falls_through_threedna_silently`.
- **What goes wrong.** Each monkeypatches `_load_backends` and `available_backends` by hand. The first one's docstring states the order as *"amber > rdkit"*, though the product's order (`_AUTO_ORDER`) is threedna > amber > rdkit.
- **Fix.** One table of rows (availability → backend picked) over the real `_AUTO_ORDER`, one fake registry.
- **Sureness:** read.

---

## Checked, not counted
- **`annotations_fdf.unregistered_channels` has no caller,** and no strategy is registered anywhere. The module was brought back by the user's decision (`docs/archive/audits/2026-11-05-2.65.17-removals.md`, *"i agree to your #1"*), so this is not a finding.
- **`cell.PLACED_TOL_ANG` (1e-6 Å) and `structure.SAME_PLACE_ANG` (1e-6 Å):** two names for two questions (an atom past a face; two geometries the same). Each is justified from its own side.
- **`StructureCodec.format_of_text` and the parse registry's PDB detection** answer different questions (a text with no name; a file a parser may read).
- **`Structure.to_pyscf` writes labels (`Au1`) while the file writers write names:** PySCF reads labelled atoms, so it looks chosen.
- **The many `Structure(...)` literals in tests** are inputs to model-level rules, not stand-ins for a writer's output, except where named above (T-2, T-3).
