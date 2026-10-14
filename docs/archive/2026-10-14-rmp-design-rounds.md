# RMP's design rounds — 2026-10-14

**Role:** archive — plan § 5ab as it stood through the day's rounds (the rule as first decided, the reading- and writing-side survey, the revision plan, its validation round and corrections), verbatim, before it was rewritten as one model: the invariants and the one API, now the contract in [`model/overview.md`](?doc=model/overview.md) § 2.2a, and the build in the plan's § 5ab. Kept for the evidence each round read in the code; what it decided is in the model, and what it superseded is said there.

*(user, 2026-10-14: "check this logically, and consolidate the document of design/contract, and plan")*

---

## § 5ab, as it stood — The remapped state — the mapping contract stated at every output *(RMP; decided with the user 2026-10-14)*

> *(user, 2026-10-14)*: *"there is a mapping contract, all things will return
> to original index from input, no mapping is exposed at the output"*; *"it
> should (1) clearly mark the output result as "remapped", and (2) if the
> result finished, the generate the "restored" mapping result from that as the
> final result. (3) if the prep does not see that final result as "restored"
> it either should error, or warn for a conversion first"*; *"i would make do
> warning and convert (if mapping index record is there, otherwise error)"*;
> *"in the modify/molbuild tab, when loading a data set with "remapped" meta
> state, we should get a warning and if the mapping index is present we can
> allow a restore operation"*; *"that means parser should be reviewed and
> revised accordingly?"*; *"maybe we should enforce that remapping index to be
> saved into the .json if the meta data shows "remapped" state"*.

**The contract it completes** — [`model/overview.md`](?doc=model/overview.md)
§ 2.2: the program sorts a COPY at prep, records the permutation both ways,
and maps every number that comes back through the inverse before a person
sees it; the sorted order never leaks.

**What is true today** *(read 2026-10-14)*. Two kinds prepare their decks from
a reordered copy: a **SIESTA vibration** — every stage, its `relax` stage
included, because the force-constant stage starts from `relax`'s density
(`DM.UseSaveDM`, `engines/vibration.md` § 5.3), stored atom by atom in the
deck's order (`jobset/prep.py`, the SIESTA vibration rung; key `held-first`) —
and **transport** — every rung from the composed junction sorted for
TranSIESTA (`transport/compose.py`; key `transport`). A standalone structure
optimization is never reordered. Two readers apply the record: the vibration
finish (the spectrum is written in the input order) and the transport
composer reading its own record. **Nothing else does**: the parsers, the run
door (`runs.declared`, `runs.view_of`), the Results tab (a reordered run's
frames and labels shown in the engine's order, unmarked, while the same
calculation's spectrum is in the input order), a structure exported from such
a run, `xv2xyz --from-run` — and the force-constant prep, which copies
`relax`'s final coordinates row by row (V1.41: right only while both preps
held the same atoms).

**The rule, decided:**

1. **Stated, never silent.** A structure or per-atom result in an engine's
   reordered order carries the state **remapped**, its index with it. The run
   door marks what it builds from a reordered run; the parsers stay the
   engine's file view, and nothing is reordered silently while it is read.
2. **"Remapped" is never stated without its index.** A sidecar in the remapped
   state carries the full index — the reordering's key and both directions,
   in the shape of `atom-permutation.json`, read by its one reader
   (`atom_permutation`): one data structure, not a second format. Sidecar v14,
   the key optional (absent: the person's own order). A remapped state
   without a complete, self-consistent index is refused on read, by name — only
   a hand edit makes one.
3. **A run on a reordered copy leaves its restored result** when it finishes:
   its final structure in the input order, with the person's labels, as a
   pair (the name to settle; proposed `<label>_<stage>-run<N>.restored.xyz` and
   its sidecar). A force-constant run's restored spectrum already is one
   (`spectra.json`).
4. **A door that builds from a reordered run's output** — prep (V1.41's
   force-constant stage among them), a transport citation, the Molbuilder
   tab's append — **restores** through the index and says so; a remapped state
   with no index cannot exist (rule 2). The Results export and
   `xv2xyz --from-run` build nothing: they carry the state and its index.
5. **The Molbuilder tab**, loading a remapped structure, **warns and offers
   restore**: the atoms back in the input order, every per-atom field with
   them.

**The survey** *(2026-10-14, two read-only agents at `8c5b6be1`, each
finding re-read before it is used here)*. Every reader of a reordered run
shows the engine's order unmarked — the SIESTA parsers (`.out`, `.MD.nc`,
`.XV`, the molwatch log), the run door (`runs.declared`, `runs.view_of`), the
Results load and export, `xv2xyz --from-run`, the transport report and its
PDOS selections, the transport citation of a reordered run (any finished run
of ours may be cited — a vibration's `relax` included, its order then sorted a
second time, never composed), and the force-constant prep (V1.41, which also
imports a parser class directly, against `model/parse.md` § 3). The labels are
not wrong: the deck's ATOM-METADATA block is written from the sorted copy, so
labels and coordinates agree — in the engine's order, unmarked. Handled today:
the vibration finish (through the record's one reader), the transport
composer (reading the record, then mapping by hand in `wizard._atoms_named`
and `compose.py`, against § 2.2's *never by hand*). **Where the index lives
today does not hold**: `atom-permutation.json` is written at the
calculation's root at every prep — in the flat layout every run's own folder,
so each prep overwrites the one before it — and the shared package copies it
into every attempt, the transport electrode rungs and benchmark trials
included, whose atoms it does not describe. The browser rebuilds the metadata
it sends field by field (`model-jobs.js`), so a new field is dropped on every
round trip unless both directions carry it. A pair cited for transport must
carry `info.calculation`, which a job cannot compose. Stray engine indices in
otherwise restored results: `info.relaxation.held_atom_idxs` (V1.42) and
`engine_metadata.fc_range_1based`.

**The revision plan — approved by the user 2026-10-14** (*"yes to all, go with your recommendations"*).

*The constraints every part meets* *(the user, 2026-10-14: "your design need
to guarantee a unified api access, no handcrafted code, the json and xyz
travel, read and write together, no split handling")*:

- **The pair is one unit.** The remapped state is a field of the structure,
  so it is read and written only with its atoms, by the codec, in the
  `.xyz` and its `.molstruct.json` together; no index travels in a file of
  its own (`atom-permutation.json` retires), and the restored result is a
  pair the codec writes.
- **One door for each act.** A structure from a run comes through the run
  door (`runs.view_of`) — never a parser class imported for it (V1.41's
  `SiestaParser`); a restore is `Structure.take` through the record's one
  reader — never an index mapped by hand (`wizard._atoms_named`,
  `compose.py`'s `start[list(...)]`); every write is the codec's.
- **No field listed by hand.** The two places that list a structure's
  metadata field by field are fixed, not extended: the deck's ATOM-METADATA
  block is written from the structure's own serialiser
  (`metadata_to_dict`), not the hand list in `script_emit.emit_atom_metadata`
  / `apply_atom_metadata`; the browser carries the metadata block whole
  through one function each way, not the key-by-key rebuild in
  `model-jobs.js` (`structureFromServer`, `structureForServer`) — so a field
  the structure gains reaches every road with no edit.

*Where the index lives — one home for each thing:*

- **A structure** carries it as one field, `atom_order` — the key and both
  directions, in the record's shape, checked by the record's one reader (split
  into a reader of the dict that the file and the field share). **Present
  means remapped**: the state cannot exist without its index. It rides
  everything the structure's fields ride — `replace`, `to_dict`, the sidecar
  (v14), the codec — and the browser's two directions.
- **A run** states it in its own deck, in the ATOM-METADATA block (the
  sidecar's shape, so the same field), read through `runs.declared`: each run
  its own index — the flat layout's overwrite and the electrode rungs' false
  copy end with it.
- **`atom-permutation.json` retires**: the vibration finish reads the index
  from its deck; the transport record's junction carries it in its sidecar;
  `.transport.nc` carries the index on its atom axis, not a file's name.
- **The identity order is not remapped** — nothing moved, no state.

*Who says it, and who restores:*

- **The run door marks**: `view_of` — the one builder of a structure from a
  run (the Results load, `xv2xyz --from-run`, the citation) — sets the field
  from the run's deck. The parsers stay the engine's file view; the bare file
  readers (`watch parse`, `watch tail`, `runtime-info`, bare `xv2xyz`) read a
  file, not a run, and the contract says so.
- **The Results tab** shows a reordered run in the engine's order, marked
  (the key named); an export carries the state and its index (v14).
- **The Molbuilder tab** warns while a structure is remapped (a condition of
  every answer, not a load-only notice, since notices clear on an edit) and
  offers **restore** — a server operation, `Structure.take` through the
  index, the field cleared, the selection cleared with it. An edit that adds
  or removes atoms on a remapped structure is refused, naming restore (as a
  frame set's is).
- **A door that builds** restores with a warning — the force-constant prep
  (V1.41: `relax`'s restored result, or its final coordinates through its
  deck's index, read through the run door, not a parser class), a calculation
  whose structure is a remapped pair (`_structure_for`), the transport
  citation (restored before the transport sort, so the record is one
  permutation of the person's order — and a run that is not a relaxation, a
  force-constant stage, refused as `CITATION_CONDITION` already says),
  `xv2xyz --from-run`. A remapped state with no index cannot exist (the field
  is the index).

*What a run leaves:*

- **A SIESTA vibration's `relax` stage writes its restored result** when it
  finishes: `<label>_<stage>-run<N>.restored.xyz` and its sidecar, the
  relaxed geometry in the input order with the person's labels and box — its
  job's finish, through `StructureCodec.write_moved`, the payload made by
  prep, the final coordinates from the `.out` by the travelling reader. The
  run records (`info.calculation`, `info.relaxation`) are composed on this
  machine by the door that hands the pair on, as an export does today. A
  restore that fails is said in the session log and does not change how the
  run ended — the doors restore with a warning. The force-constant stage's
  restored result is its spectrum, as today; **transport rungs** are single
  points whose structure is the record's junction — no restored pair, the
  junction stated remapped with its index; the electrode rungs carry none.
- **Stray engine indices**: `held_atom_idxs` cut (V1.42);
  `fc_range_1based` stated as the engine's, under `engine_metadata` where it is.

*The validation round* *(2026-10-14: two fresh agents read the approved plan
against every reader and writer at `03b09e7c`; each finding re-read before it
is used here)*. What it changed — ◆ marks a choice the round forced, for the
user's word with R0:

- **Reordering a remapped structure composes** ◆: `Structure.take` composes
  its order into the field's index (the new `sorted_to_original[j]` is the
  old `[order[j]]`) and clears the field when the composition is the
  identity — so a restore is `take` through the field and a second sort is
  one composed permutation (§ 2.2's *one copy, one permutation*), never a
  stale index under a new order. `SortResult`'s tuples become a view of the
  sorted structure's field — one source from R2 on.
- **One restore, one sentence** ◆: `Structure.restored()` — `take` through the
  field — is called by every door (the route, prep, the citation, the
  append, the CLI's `modify`); an edit that adds or removes atoms on a
  remapped structure is refused by one constant naming it, as a frame set's
  is (`_carry_nonatom`); `concat` (the Molbuilder tab's append, reached by a
  load into an open tab) restores a remapped addition and says so.
- **The warning is the structure's state, not a notice** ◆: the browser's
  notices clear at every settle (`model.js`), a label edit or a Results poll
  among them; the Molbuilder tab and the Results tab show the state from the
  structure's field, read at every settle.
- **Restore clears the selection**: an `OPERATIONS` column for a reorder (a
  count kept, every index moved), and `molview.md` § 11.1's table gains it.
- **The browser's third hand list**: `getStructure()` (`model.js`) lists the
  master copy field by field too; one function each way, all three.
- **The index rides inside the deck's ATOM-METADATA block**, the block
  written from `metadata_to_dict` and read wherever it is read
  (`emit_atom_metadata`'s gate, `apply_atom_metadata`'s early return,
  `Declared.atom_metadata_for`, `render_deck`'s note) — never a `RunView`
  attribute of its own (`watch._frame0_structure` rebuilds the view from the
  block); a field refusal surfaces (`watch.py`'s blanket `except` goes).
- **The codec's engine-own branch is a run door**: SIESTA's own `<label>.xyz`
  in a reordered run (`WriteCoorXmol`, on by default) opens through the run's
  view, marked.
- **A run made before RMP** ◆: its index is in `atom-permutation.json` and its
  deck states none. A door that builds refuses it by name — *prepared before
  this version recorded its atoms' order; prepare the stage again* — as the
  vibration finish already refuses an old deck; a viewer says it and shows
  the engine's order; an old transport record is not loaded
  (`load_compose_record` answers why). No reader of the old file remains.
- **One source for a door that holds the run** ◆: prep and the citation
  restore through the run's deck index, always; the restored pair is the
  person's copy, for the Results tab and the Molbuilder tab — so V1.41's
  deck bytes never depend on which of two sources exists, and R3 does not
  wait on R4.
- **The citation restores both geometries**: `_run_geometry`'s relaxed
  structure and its starting geometry (the leads' *unmoved* gate compares
  the two), and the cited-pair branch too; **a run that is not a geometry
  relaxation is refused** through the verdict `_classify_run` already reads
  (`CitedRun.verdict`), as `CITATION_CONDITION` states.
- **The `relax` finish** ◆: keyed on the deck written from a remapped
  structure; each finish's failure policy a column of `_FINISH_BUNDLES` — the
  force constants' failure the job's, as now, the restore's said in the
  session log and never stopping the relaxation, its load check a warning;
  the payload in a deck block read by `deck_record`, built by
  `pyscf/input._sidecar_for`'s rule — the **run's box**, the engine's origin
  stated, never the person's design cell; the final rows permuted through the
  deck's index before `write_moved`; the catalogue row kept out of the run's
  result roles (`runs.about`), so the run door never puts the deck's sorted
  labels on it.
- **The density handed from `relax` to `freq`** ◆: the `.DM` is ordered as
  `relax`'s deck; when the two decks' indices differ (the held set grew
  between the preps), `freq` starts its reference SCF cold
  (`DM.UseSaveDM .false.`) and says so — the force constants are unchanged.
- **Stray engine indices**: `held_atom_idxs` cut in R1 (it outlives a
  restore otherwise); `fc_range_1based` is the engine's numbering wherever it
  is written — `spectra.json`'s `engine_metadata` and `fc-sweep.json`'s stage
  rows, where the sweep reads `.FC` with it — and is named so.
- **Stages prepared before the upgrade** and not yet launched are refused at
  launch (`submit._as_described` compares the re-planned deck); the refusal
  names the version that prepared them, not *the template changed*.
- **Placement**: `atom_order` in `METADATA_FIELDS` — a v14 sidecar writes
  `null` for the person's order, so "absent" reads "null or absent"; the
  field's key one of `SORT_KEYS`, required; the identity read as no index,
  the split reader answering it.
- **Untrue in the survey, corrected**: transport writes its record's index
  when it composes, not at every prep.

*The order, each phase closed as a milestone (§ 0a):*

| phase | what | tests |
|---|---|---|
| **R0** | the contracts, for the user's review before code — `model/overview.md` § 2.2; `model/structure.md` § 2.1–2.2e (and § 2.2 step 4's *two translators*); `model/structure-molstruct.md` (v14); `model/structure-annotations.md` § 4a; `model/parse.md` § 3, § 5.3, § 5b.1; `execution/job-contracts.md` § 3.1, § 3.4, § 6.1, § 6.3; `execution/project-layout.md` § 1.6.3, § 5.1; `execution/architecture.md` § 3.2; `execution/job-system.md` § 3; `engines/vibration.md` § 5.2, § 5.2a, § 5.5, § 6.5, § 6.9, § 7 (I7), § 8, § 10; `engines/transport.md` § 2a.12, § 3.1; `science/normal-modes.md`'s order note; `web/web-api.md`; `web/molview.md` § 6.2, § 6.8, § 11.1; `web/results.md`; `configuration.md`'s bundle rule | — |
| **R1** | the model: `atom_order` and the record's dict reader; `take` composing; `restored()` and the one refusal; `concat` restoring; sidecar v14; the codec; `held_atom_idxs` cut; the browser's three hand lists made one function each way; the state shown from the field; `/api/modify/restore` with its `OPERATIONS` row and the CLI's | `geometry_pin.toml` (v14 rows, an incomplete index refused), `frame_sets.toml` (restore over a set), the model and browser round trips — every order a **3-cycle** (O O C with the carbon held: `[2, 0, 1]`), so the two directions differ |
| **R2** | said at the source: `sort_by` setting the field, `SortResult` a view of it; the deck's block from `metadata_to_dict`; `declared` and `view_of` marking; the codec's engine-own branch; the Results tab's state; the export and `xv2xyz --from-run` carrying it; a pre-RMP run said or refused; launch naming the version | the Results route's rows |
| **R3** | restored where it builds: V1.41 through `relax`'s deck index (the run door, not a parser class); `_structure_for`; the citation (both geometries, the pair branch, the kind refused); the by-hand maps through the record's reader | the SIESTA vibration e2e on T-CO2's ladder (O O C, a held set grown between the preps) — not `prep_protocol.toml`, whose road runs no engine |
| **R4** | what a run leaves: the `relax` finish (its policy, its payload block, the run's box), the catalogue row, the cold `freq` start when the orders differ | the same e2e |
| **R5** | `atom-permutation.json` retired: the vibration finish reading its deck, the transport record's junction carrying the field (`record.py`'s provenance key and its schema, `compose.record_files`, `.transport.nc`'s atom axis), no reader of the old file left | the vibration and transport e2e — the transport junction's input order **permuted**, so its sort is not the identity |
| **R6** | the transport report: a server door restoring the junction for drawing (`junction.xyz` stays sorted on disk — the device's regions and the leads are read from it); the PDOS selections mapped in, and `outside_device` back out, through the record | a partial PDOS selection on the permuted junction |

**Decided with it** *(the user, 2026-10-14)*: the field's name and home as
above; `atom-permutation.json` retired; no restored pair for transport rungs;
a failed restore not failing the relaxation; the identity order not
remapped; the Results tab marking (not restoring) a reordered run's view;
R6 drawing the junction in the person's order. RMP is new functionality (a restore operation, sidecar v14): its own middle
number, after 2.9.6 is pushed — before V1.39, whose structures it reads.

