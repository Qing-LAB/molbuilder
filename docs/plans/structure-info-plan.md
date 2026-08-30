# Structure info — free metadata that travels with the pair

**Role:** plan (the board below is the live status; rows flip to Done as
they land)
**Domain:** model · web · execution
**Started:** 2026-08-29 (user rulings, same day: the § 1 store, the
Metadata pane, and the recorded electronic contract as its first
consumer)
**Companions:** [`web/molview.md`](?doc=web/molview.md) (the pane + the
doors) · [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md)
(persistence) · [`plans/transport-design.md`](?doc=plans/transport-design.md)
§ 4.1b (the consumer that motivated it)

## 1. What is being built, in one paragraph

A structure gains a **free-form, non-structural metadata store** —
`info`, a JSON dict of key → value — supplied by host tabs through ONE
MolView API (`viewer.data.info.set / remove / get`), displayed in a new
**Metadata** pane beside Selection and Cell, and persisted in the
`.xyz + .molstruct.json` pair so it travels wherever the structure
goes.  Its first consumer closes a real gap: when the Results tab
exports a structure from a finished run, the engine's own parser
records the **electronic contract** (basis, XC, mesh, k, temperature)
into `info.calculation` — and a transport citation of that pair then
runs with the relaxation's contract **sealed**, exactly as if the deck
itself had been cited (`transport-design.md` § 4.1b gains the third
shade: pair-with-recorded-contract).

Design stances (user, 2026-08-29):

* **The contract block's keys are one agreed vocabulary** — exactly
  `TransportConfig`'s `CONTRACT_FIELDS` names (`transport/stages.py`),
  never a spelling invented at a call site.  The extractor is pinned to
  the set by test (`tests/parse/test_contract.py`), the sealed fill
  reads only names in it, and any new field joins by joining the
  constant — so the recorder, the pane, the pair and the seal cannot
  drift apart on a name.

* `info` is **not structural**: it never enters `structure_hash`, the
  frozen/region machinery, or any deck emission — it *describes* the
  structure.  That is also why the read-only gate does not apply to it
  (`molview.md` § 9.4's one question — "does this change the structure
  the calculation ran on?" — answers no), which is what lets the
  read-only Results viewer attach the contract before export.
* The pane **displays**; the API **mutates**.  A person reads the
  metadata in the viewer; which keys exist is the host tabs' business.
* A recorded contract **seals** the transport fields (fdf-is-truth
  transfers to the recorded copy); a hand-made pair with no
  `info.calculation` stays the open lane.  Consistency with the
  relaxation is the point of the contract, and the refusal already
  names the way out (re-export, or cite something else).

## 2. The board

| # | leg | steps | proof | status |
|---|---|---|---|---|
| **I1** | **The contract text** | `molview.md`: the Metadata pane, the `data.info` doors, the § 9.4 reasoning, install/export ride.  `model/structure.md` + `structure-molstruct.md`: the `info` field, sidecar schema 9 (additive; 8 stays readable), hash invariance.  `transport-design.md` § 4.1b: the recorded-contract shade. | doc guards green (`test_docs_structure`, `test_doc_claims`) | **Done 2026-08-29** — doc guards green |
| **I2** | **Server core** | `Structure.info: dict` (JSON-refusing setter path at the dict doors), `to_dict`/`from_dict` top-level `info`, `structure_hash` untouched by it; sidecar `SCHEMA_VERSION` 9 — write `info` when non-empty, read 8 and 9, stray-key rule updated. | round-trip tests: pair→load→`info` intact; a v8 file loads with empty `info`; hash equal with/without `info` | **Done 2026-08-29** — `TestInfoBlock` pins the five invariants (ride-whole, empty-writes-no-key, hash invariance, v8 full-replace, JSON refusal at write) |
| **I3** | **The wire** | `/api/build/load` answers `info` (path + `{structure}` envelope branches); `struct_from_body` carries it; the save door writes it through the codec. | endpoint round-trip test | **Done 2026-08-29** — `to_wire` + the `{structure}` envelope gate carry it; the web battery green |
| **I4** | **MolView** | model `info` store + announce; `viewer.data.info.set/remove/get` (ungated, no unsaved badge, rides the editable draft); `structureFromServer`/`exportFile` carry it; `ui.js` third panel page `Metadata` (read-only key/value rendering, honest empty state). | js pins (doors exposed, third tab present); browser check: set a key → pane shows it → export → sidecar holds it | **Done 2026-08-29** — `test_molview_info_js` pins the ungated doors, the page, the two-way wire, the sheet; browser-proven (the demo: `info.set` → the Metadata tab renders the contract JSON) |
| **I5** | **The recorded contract (producer)** | one `contract_of(run_dir) -> dict \| None` interface with per-engine extractors (SIESTA `parse_fdf_params`; PySCF its deck parser); the Results-tab structure load attaches `info.calculation` (engine, contract, source deck + sha) at the same server-side compose that already attaches periodicity. | load a finished run in Results → pane shows the contract → export → pair carries it | **Done 2026-08-29** — `parse/contract.py::contract_of` (SIESTA real; a PySCF deck answers None until its extractor lands), `/api/results/contract`, the structure inspector records through the `info` door; the Export menu's indicator + the tab's presence dot shipped with it.  Residue: the TRAJECTORY inspector's export does not record yet (its load rides /api/watch, not the structure door) — § 3 |
| **I6** | **The sealed pair (consumer)** | `transport/compose.py`: a form-B citation whose sidecar carries `info.calculation` fills the config from it and **seals** `CONTRACT_FIELDS` (same refusal as form A, wording names the recorded deck); `describe_attempt` answers `contract: "cited"` with "contract recorded from the <engine> deck" in the summary; the tab's lane logic already follows `contract`. | compose + describe tests both lanes; **mutation:** drop the seal → the stays-sealed test fails; browser: cite an exported pair → contract fields hidden, meta line says recorded | **Done 2026-08-29** — `recorded_contract_of` (one reader for compose + both web doors), the fill forces kz=1, the seal's refusal names the record, the travel copy keeps it (`TestTheRecordedContract`, seal mutation-killed); live: the recorded pair answers 'contract RECORDED from the siesta deck', and `prep run seed` renders SZ/150 Ry from the record, not the defaults |
| **I7** | **Close-out** | full battery; browser walk of the whole chain (Results export → cite → describe → prep); this board flipped to Done; README/toc rows already indexed. | battery green (the 9 § 3 pre-existing failures excepted); walk screenshots | open |

Order is the dependency order; nothing in I5–I6 starts before I2–I4
hold, because the store must exist before anything records into it.

## 3. The parked backlog (so one file answers "what is open")

* ~~Nine pre-existing test failures on main~~ — **diagnosed and fixed
  2026-08-29**, each read whole and classified:
  `test_admin_reload` ×2 = test staleness (the probes invoked the
  retired `serve --port` spelling; `serve` became a group — all four
  probes retargeted at `serve foreground`);
  `test_catalogue_agreement` = REAL DRIFT, the catalogue was the wrong
  side (it still taught the retired `./` psml_lib spelling the code
  refuses — aligned to the code's truth, said here as the test
  demands);
  `test_docs_tab` toc dedupe = passes on current toc (the duplicate it
  saw is gone);
  `test_http_status_contract` = stale doc header (route count 85 → 88,
  the contract endpoint's row added);
  `test_launch_ask_mode` = test-logic bug (a first-occurrence slice
  between two anchors inverted to empty when the transport chain added
  an earlier `if mode == "direct"`; the code itself was correct — the
  end anchor is now searched after the start);
  `test_negative_body_assert_lint` = a missing status guard in a test
  body (added);
  `test_molview_mount` ×2 (plus two more the same cause) = stale pins
  from two legitimate features (the `buffer` label joined the
  predefined set with the transport composite; the Metadata page made
  the panel three pages).
* **CSS polish batch** from the 2026-08-29 fresh-eyes review (wrong
  `var()` fallbacks documenting stale values; raw sizes in
  `molview.css` against its own tokens; the inspector vocabulary
  living in `results/style.css`; `modify.html`'s three hand-written
  `molviewer-*` names; `.hint` restyled by three page sheets;
  `rec-diff`/`btn`/`secondary` template classes with no rules).  Some
  entries sit on `web/ui-contract.md`'s recorded keep-list
  (2026-08-25) — touching those reopens decisions, deliberately not
  done in the review round.
* **Two caller-less endpoints** (`/api/checkpoint/config`,
  `/api/docs/list`) and `/api/selection/atoms` (test-pinned as a
  vocabulary contract — confirm the role before deleting).
* ~~The transport tab did not work on a REAL junction~~ — **fixed
  2026-08-29** (user: "nothing works in the transport tab"; walked on
  `Au-BDT-Au/optimization/AuBDTAuRelax/01_coarse/bench/…`).  Four
  rules were wrong, each fixed where the rule lives, not at the
  symptom:
  1. **Conclusion evidence was our marker's spelling**, so a real
     SIESTA run that exited cleanly under someone else's launcher read
     as "still running".  Evidence is FILES: `0_NORMAL_EXIT` counts
     (§ 4.1b's table now says so).
  2. **The sort refused a junction whose author named the top block
     `L-electrode`** — an assumption about how the user builds
     structures, the same class § 4.1b banned for layouts.  Settled by
     the user's ruling (2026-08-29): **check z, warn, the author
     decides.**  The atom order and the semi-infinite directions now
     follow the GEOMETRY (lower block first — that is the `-A3` lead,
     not a preference), the chemical potential follows the NAME, and an
     inverted pair gets a note from the sort, a `warn` from the engine
     preflight and a rename offer in the tab.  One sentence for all
     three (`sort.py::inverted_note`).  The only refusal left is
     INTERLEAVING blocks, which no rename can help.
  3. **The principal-layer gate refused on a fixed ~12 Å floor** —
     a guessed number that rejected a valid 3-layer Au lead.  It now
     READS the orbital ranges from the citation's own `.ion` files
     (`parse/ion.py`) and compares 2·rc_max against the next-nearest
     cell gap, **2·period − span** (§ 3); unreadable → honestly
     UNVERIFIED, never a refusal.
  4. **Describe wrote into the cited directory** (the sidebar
     selection lingers there), dropping `task.json` inside the
     relaxation it cites.  Refused now, and the success line names the
     folder it wrote.
  5. **The chempot was bound by geometry, not by the label**, so a
     junction whose author put L on top would have rendered an I–V
     whose sign contradicted its own labels — silently.  The binding is
     the NAME now, and the deck STATES the outcome: which region is the
     low-z `-A3` lead, which carries µ = +V/2, and — when those differ
     — that the high-z lead is the positively biased one.  Verified
     against TranSIESTA's own convention, not memory: the author's
     reference inputs (`zerothi/ts-tbt-sisl-tutorial` TS_01/TS_02) pair
     `Left` with `electrode-position 1` + `-a1` + `mu V/2`, while
     SIESTA's shipped `Tests/16.TranSiesta/ts_chain.fdf` names the same
     two leads `high`/`low` — proving the names are free and only
     first-atoms ↔ `-a3` is fixed.
  Proven end-to-end on the user's own junction: cite → describe →
  `prep run` for all five stages (444 atoms, DZP/300 Ry/PBE/2×2×1 read
  from the deck, pseudos pulled from the cited dir, 108-atom leads with
  a 7.20 Å period); device and transmission correctly wait on their
  upstream conclusions.
* ~~The download button was unreadable and carried history~~ —
  **reworked 2026-08-29** (user).  It names its target, says *Zipping…*
  in its own label while the server compresses, and refuses further
  clicks until the save starts — which needed the door split in two
  (`POST /api/files/zip_prepare` builds and answers what it built;
  `GET /api/files/download_zip?token=` streams it), because a plain
  navigation reports neither end of a build that takes minutes.  The
  archive is the **pure execution directory**: the three storage
  subtrees stay out (`.molbuilder_workspace`, `.git`, `.binsnapshots`,
  each named by its owning module's constant) and the walk PRUNES
  them rather than filtering.  Current results always ride — the
  exclusion is by directory, never by size or suffix, so the live
  `.DM`/`.TSDE` in a run folder travel while the snapshot copies of
  the same paths do not.  *(Not `git archive`: a checkpoint keeps big
  files out of git deliberately, so a git export would drop exactly
  the heavy outputs the far machine needs.)*  Measured on the user's
  relaxation: 233 files / 137 MB out of 503 MB on disk.
* **Flat-shape citation ergonomics**: a flat calculation folder holds
  several stage decks and one shared `.XV`, so citing it refuses as
  ambiguous (never guess which deck is the contract).  Fine per
  § 4.1b; a future refinement could disambiguate by the concluded
  marker's stage.
* **Transport viewer default orientation**: every junction's transport
  axis is z, and the default camera looks down z — a 1-D chain reads
  as a single ball until rotated.  A host camera-orientation door is a
  `molview.md` contract question, not a patch.
* **The trajectory inspector does not record the contract yet** — its
  structure arrives through `/api/watch/load`, not the structure door,
  so an export from the trajectory view carries no `info.calculation`;
  the recording call it needs is the same one the structure inspector
  makes (I5).
* **The Results-tab transmission inspector** (reads the shipped
  `<label>.transport.json`) and `TransiestaEngine.parse_output`
  (raises by design, pointing at the record) — roadmap § 2 items 3–4.

## 4. Static-review residue — 2026-08-29, all addressed 2026-08-30

A three-pass read of the transport + download work (no tests; the suite
was green through every one of these). Nineteen findings; the ones the
convention rework already closed are struck, the rest are batched below
so each batch is one coherent change with one place to verify it.
**All of them are now done** — each fixed at the rule that produced it,
recorded batch by batch below.

**Closed by the convention rework** (same day): the swap button's
missing error handler; the ordering gate blaming the labels and
advising a prep run that cannot help; the deck's µ = +V/2 comment
emitted only when the block names already said it; the preflight's
hoisted `pos_a`/`zc`; `sort.py`'s stale imports in `compose.py`; and
the doc contradictions in `engines/transport.md`, in
[`transport-design.md`](?doc=plans/transport-design.md) § 4.1a, and in
this board's own items 2, 3 and 5.

**Withdrawn**: "the swap corrupts `selection_rules`". Nothing in
molbuilder ever writes that field — it is an empty pass-through slot in
the sidecar format, and no `.molstruct.json` in the projects tree
carries one. The shape inconsistency was real; the consequence was not.

### Batch A — what the composed record claims about itself

* **A1. The label file is missing from the provenance.** When a form-A
  citation's labels come from a `.molstruct.json` beside the deck
  rather than from the deck's own block, `classify_citation` leaves
  `cited.sidecar = None`, so `compose.py`'s `provenance["files"]`
  hashes the deck and the `.XV` only. The file that decided which
  atoms are electrodes — the one the rename endpoint now rewrites — is
  unrecorded. *Fix:* have `labeled_citation_structure` report its
  source into the provenance.
* **A2. The saved record does not require the file that labels it.**
  `write_compose_record` writes `junction.xyz` + its sidecar but
  returns a list naming only the first (and the caller discards the
  list). `load_compose_record`'s completeness check omits the sidecar
  too, so a record whose sidecar was deleted loads an *unlabeled*
  structure and dies inside the lead gates instead of answering `None`
  = "incomplete, recompose". *Fix:* require the sidecar; drop the dead
  return or make it name every file written.
* **A3. The lead gates fire in the wrong order.** A block whose label
  boundary cuts a partial layer has a meaningless interlayer spacing,
  hence a meaningless period — and the orbital-range check runs first
  and refuses on those numbers. *Fix:* run the tiling check first, so
  the refusal names the actual defect.

### Batch B — the sort's buffer rule

* **B1. Buffer atoms are assigned by the global z midpoint**, while the
  docstring says "the end it is nearest". They differ when one end's
  padding is taller than the rest of the structure, and then a
  correctly labeled junction is refused for "buffer inside the
  electrode". *Fix:* nearest end, as documented.

### Batch C — the download door

* **C1. Four numbers are computed, returned, and shown nowhere.**
  `zip_prepare` answers `files`, `bytes`, `skipped`, `excluded`;
  `mutation-bar.js` reads only `ok`, `error`, `token`. So the archive
  never reports what it produced (the "233 files, 137 MB" the split
  door existed to make sayable) and never reports what it left out —
  `skipped` counts symlinks that point outside the tree. *Fix:* say
  both on the button's own status line.
* **C2. The busy reset re-enables the button unconditionally.**
  Navigate to a projects root mid-build and the 1.5 s reset leaves it
  enabled where it must be disabled, clicking to nothing. *Fix:* hand
  control back to `_updateButtonEnablement`.
* **C3. The tooltip has two homes** — hardcoded in
  `_projects_sidebar.html` and overwritten by JS on first paint.
* **C4. `_zip_target_refusals`' docstring** claims it is shared by both
  halves of the door so they cannot disagree; it has one caller.

### Batch D — dead weight

* **D1.** `compose_junction` imports `apply_to_structure`,
  `load_sidecar` and `apply_inbody_atom_metadata` and uses none of them
  (the label reading moved into `labeled_citation_structure`).
* **D2.** The cited deck is parsed twice per compose — `params` at the
  top of the form-A branch, then `parse_fdf_params(deck_text)` again in
  the return.
* **D3.** `POST /api/transport/swap_electrodes` re-implements
  `contain` + `is_dir` + `classify_citation` instead of calling
  `resolve_citation`, which exists and is documented as the door the
  web hand-over and prep share.
* **D4.** `wizard.electrode_wizard` spells `12.0` instead of
  `MIN_ELECTRODE_THICKNESS_ANG`, three definitions below the constant.

### Batch E — the tab shows what it is talking about

* **E1. `describe_attempt` answers `structure: null` on any compose
  failure**, so a junction that classifies but cannot build shows an
  empty viewer — including, until the rework, the one being offered a
  rename. The comment above that block claims it "still answers".
  `labeled_citation_structure` returns exactly the structure needed and
  is already imported on that path. *Fix:* fill the viewer from it, so
  the refusal is shown *on* the junction it is about.

### Batch F — the last doc disagreement

* **F1.** § 3 still quotes the retired refusal wording ("exceeds the
  L-electrode block's period Y Å") three lines above the
  `2·period − span` rule that replaced it.
* **F2.** `task-handover.js`'s comment says a calculation "never lives
  inside its citation", but the guard tests equality only — a folder
  *underneath* the citation passes.

### What was actually changed, 2026-08-30

Each batch was fixed at the rule, not at the symptom:

* **A1/A2 — one answer to "what is this record".**
  `compose.py::record_files(form)` names the travelling copy's file set
  and both the write and the load ask it; the write now refuses to
  report a record it did not put down. The label file joined the set,
  which is what closes the load-as-unlabeled crash. The provenance
  gained the label SOURCE: `labeled_citation_structure` already
  answered which file it read, and the caller was throwing that answer
  away.
* **A3 — the two lead gates are ordered by their dependency**, tiling
  first, with the reason stated in both the code and § 3: everything
  the second computes is derived from what the first validates.
* **B1 — a buffer atom's side and its legality are ONE question.**
  They were two rules (side from the whole structure's midpoint,
  legality from the electrode blocks) that agree on ordinary junctions
  and part company on lopsided ones. Now: below the lower block is the
  low end, above the upper block is the high end, anything else is the
  refusal — one rule cannot disagree with itself.
* **C — one writer owns the Download button.** `zipBusy` and the last
  archive's report are INPUTS to the enablement pass rather than a
  second writer, which removes both the re-enable-mid-build bug and the
  lit-but-inert-at-the-root bug by construction. The prepare call's
  count/size/skipped are reported (tooltip; an alert when files were
  left out — a silently dropped file is not acceptable in something
  being carried to another machine). The template's duplicate tooltip
  is gone.
* **D — the deck is parsed once per compose**, `compose_junction`'s
  three orphaned imports are gone, the rename endpoint goes through
  `resolve_citation` (with both citation doors sharing one `_fence`,
  and the deliberate difference between them stated where they part),
  and the wizard's `12.0` literal is the constant.
* **E — the viewer's structure comes from the label door**, not from
  the compose, so a junction that cannot be BUILT is still shown while
  its refusal is read. The two questions — "what is this?" and "can it
  be composed?" — are now asked separately, which is what they always
  were.
* **F — § 3's quoted refusal matches the rule that replaced it**, and
  the hand-over guard tests containment, not equality: a folder
  *inside* the citation buries the calculation exactly as thoroughly.

## 5. The metadata bridge — settled and BUILT 2026-08-30

**The question that started it** (user): an export from the Results
*trajectory* view carries no `info.calculation`, so a transport citation
of that pair lands in the open lane instead of sealed. Reproduced live
2026-08-30: the export menu's "Includes metadata" line stays hidden and
the Metadata pane is empty for any SIESTA relaxation (relaxations open
in the trajectory inspector, not the structure inspector).

### 5.1 The rule (user, 2026-08-30)

> The tab the viewer resides in is in charge of finding the metadata in
> the run's deck (`.fdf`, or PySCF's, per engine) and handing it to
> MolView. MolView has no control over what metadata comes; it is
> always the host that provides it.

And: the store is **persistent** — it survives the reloads a tab makes
while showing one run — and is **refreshed by the tab** when it moves
in or out of a run.

### 5.2 What the code does — one claim here was WRONG

* **`info` IS the one additional-metadata API.** A free dict of
  key → value. `atomMetadata` and `periodicity` earn their own load
  parameters because they BECOME the structure; `info` only
  *describes* it, so every future category is a new KEY inside it and
  costs nothing anywhere else. **This held.**
* **It rides an export out**: `createWriteOut` writes `structure.info`
  into the envelope, `StructureCodec` writes it into the
  `.molstruct.json` (schema 9). **This held** — measured 2026-08-30.
* **It rides a load in — IT DID NOT.**  This section said
  `structureFromServer` "copies `payload.info` onto the new structure",
  and it did read exactly that: a FLAT `payload.info`.  **No route has
  ever sent one.**  A `Structure`'s fields arrive inside the canonical
  `payload.structure` envelope — which is where `to_dict` puts `info`,
  and where this same reader already takes `title` from, with a comment
  explaining why the flat key was wrong *there*.

  So every structure loaded since the store shipped arrived with an
  empty one, at HTTP 200. **Three consequences, all live:** a saved
  pair lost its recorded contract the moment it was re-opened; a modify
  op wiped a Results tab's contract (`applyOp` installs the server's
  answer through the same reader); and with the store gone,
  `markContractOutdated` had nothing left to flag, so an edit could
  never mark a contract stale.

  **How it survived**: `test_the_wire_carries_info_both_ways` asserted
  the string `payload.info` appeared in the module — which the broken
  line satisfied perfectly. A pin that asks whether a name is mentioned
  cannot tell a working path from a dead one. Retired; replaced by
  `tests/test_structure_info_bridge.py`, which walks the chain.
* **`molview.md` § 8.4a** was right about the *design* and ahead of the
  code. It now also states the re-supply rule, which side finds the
  metadata, and the envelope the store arrives in.

### 5.3 The gaps — there were three, not one

1. **The trajectory page never asked.** Its `installMolecule({text,
   frames, forces, atomMetadata, periodicity})` call passed **no
   `info`**, and it rebuilds on every poll and every filter change, so
   anything a host set once was gone by the next rebuild. Contrast the
   structure inspector (`lib/inspectors/structure.js`), which gets away
   with a single `data.info.set("calculation", …)` after its load
   **because it loads once and never rebuilds**.
2. **The load door could not have taken it anyway.** `requestBodyFor`
   sent no `info`, and `/api/build/load`'s text branch read none — the
   one shape with no document behind it, and so the one that needs a
   caller to state the store. (`path` reads it off the sidecar and
   `{structure}` carries it in its envelope; both already worked.)
3. **And the answer would have been dropped on arrival** — § 5.2's
   flat-key read. This is the one that also broke saved pairs and
   modify ops, quite apart from trajectories.

Fixing only #1 would have changed nothing observable. That is what the
plan's own "one field" estimate missed, and why the code was measured
before more of it was written.

### 5.4 The build, in order — DONE 2026-08-30

Landed as: `parse/dirs/run_info.py` (new) · `watch.py::_run_metadata` ·
`build.py` `info` seam · `results.py` thin caller ·
`model-jobs.js` (both directions) · `trajectory/core.js` (held + passed) ·
`tests/test_structure_info_bridge.py` (14 pins, 11 mutations caught).

1. **Server — one composer, not a third parallel line.**
   `/api/watch/load` has THREE response builders (multi-log,
   single-file, upload) plus the `/api/watch/data` poll. Two builders
   answer `atom_metadata` + `periodicity` from a directory, using two
   DIFFERENT directory rules (`resolved_from_dir` vs
   `resolved_from_dir or dirname(path)`); the upload builder answers
   neither; the poll omits them **by design** (the browser's APPLY rule
   is `!== undefined` → keep, so an always-present bundle would clear
   metadata on every poll).
   Give the run's metadata ONE composer keyed by the search directory,
   returning the block every builder spreads — upload included, which
   then answers "nothing available" deliberately rather than by
   omission. `contract_of(directory)` (`molbuilder/parse/contract.py`)
   is the extractor; `/api/results/contract` becomes a thin caller of
   the same composer.
2. **Page — hold it like its two neighbours.** Set from the load
   response; clear at the two moments the page ALREADY has (the
   `transition("LOADING")` reset and the go-idle reset, which today
   null `atomMetadata`/`periodicity` side by side).
3. **Page — pass `info` in the `installMolecule` call.** One field.
   Every future category is a key inside it.
4. **Then the chain closes**: a trajectory export carries the
   contract → a transport citation of that pair seals its contract
   fields (`compose.py::recorded_contract_of` already reads
   `info.calculation`) → **I7's browser walk can complete**.

### 5.5 Holistic check to run with it

* **Export**: the "Includes metadata: …" line reads the store live off
  the model (`molview/ui.js::drawInfoNote`, subscribed), so it needs no
  change — it lights up the moment the store is on the structure.
  ✔ chain pinned; still to see on screen in the browser walk.
* **Transport tab**: `compose.py::recorded_contract_of` reads
  `info.calculation` out of the cited pair's sidecar, and the sidecar
  carries the store (measured). ✔ chain pinned; the *"contract RECORDED
  from the siesta deck"* wording and the sealed fields are for the walk.
* **The two consequences nobody was looking for**, both now live and
  both worth watching in the walk: re-opening a saved pair shows its
  recorded contract in the Metadata pane, and a modify op no longer
  wipes it — which means `structure_modified` starts appearing on
  edited structures, as § 8.4a always said it should.
* **Fourth consumer — answered.** The spectra inspector mounts a
  viewer and exports from it, and it opens project pairs through
  `projects.parser.openMolecule` → `installMolecule({path})` → the same
  load door. So do Modify and the transport tab. **Every one of them
  was losing a pair's store on open, and all four are fixed by the one
  reader** — none needs a change of its own. Only a host reading
  metadata out of a RUN (a deck, a log) has to state it, which today is
  the trajectory page alone.

### 5.5a One question the fix surfaced — NOT decided, not built

`markContractOutdated` can now actually fire, so an edited structure
carries `info.calculation.structure_modified: true` into its saved pair.
**Nothing on the Python side reads that flag** (checked 2026-08-30: no
reader in `molbuilder/`). So a transport citation of an edited pair
seals the deck's contract onto atoms the deck never saw, and says
nothing about it.

Whether that deserves a warning — and at which door, the citation or the
prep gate — is a design call, not a bug to patch. Recorded here; the
flag itself is doing exactly what § 8.4a asks of it.

### 5.6 Order of everything still open

```
 0  MODIFY FUNCTIONS (Molbuilder tab)  — user calls it higher priority;
    not yet described.  Blocks nothing technically; do it first on ask.
 1  § 5.4 steps 1-3          → unblocks 2
 2  I7 close-out: the browser walk (Results export → cite → describe → prep)
 3  independent backlog, any order:
      · caller-less endpoints — `/api/docs/list` is a second, unused
        answer to "what docs exist" beside `/api/docs/toc` (which the
        tab actually calls); `/api/checkpoint/config` is the read half
        of a route whose write half was deliberately removed;
        `/api/selection/atoms` is pinned by five test files — decide
        the ROLE, do not just delete
      · transport viewer default orientation (a MolView camera-door
        contract question)
      · flat-shape citation ergonomics
      · CSS polish batch (part sits on `web/ui-contract.md`'s recorded
        keep-list — touching those reopens decisions)
      · Results transmission inspector (the record exists; the reader
        does not)
```
