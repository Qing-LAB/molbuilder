# F12 round 2 — R2-12: MolView (sealed package, judged on its own terms)

## What I read

- **Code, whole:** every JS file under `molbuilder/web/static/lib/molview/`:
  `index.js`, `_atom.js`, `stores.js`, `history.js`, `ui-context.js`, `mount.js`,
  `model.js`, `model-jobs.js`, `render-engine.js`, `3dmol-embed.js`, `ui.js`, `demo.js`.
- **`molview.css`, in part:** the header and the whole token block (lines 1–160), plus a
  grep of every rule after it for literal colours. Nothing outside the token block uses a
  literal. That part is clean.
- **The chapter `docs/web/molview.md`, in part:** § 6.2, § 6.6, § 9.5, the contact-reference
  part of § 11.6, and § 13–15 whole. The other sections were not read end to end this
  round. Round 1 (`r1-K-molview.md`) read the whole chapter.
- **Owners outside the slice, read only to confirm a duplicate:**
  - `structure.py`: `FROZEN_LABEL`, `BOX_FIELDS`, `_customized_row`, `normalise_customized`, `resolve_cell`, `to_wire`
  - `web/blueprints/selection.py`, whole
  - `selection.py`: the rule `KIND`s
  - `kind_table.labels()`
  - `data/contact_distance.json`, `data/README.md`
  - `docs/web/web-api.md`: the `selection/eval` row
- **Tests, whole:** `support/molview_host.mjs`, `support/molview_page.py`,
  `support/workspace_page.py`, `test_molview_reserved_label_js.py`.
- **Tests, in part** (the preludes, fixtures, stand-ins and the specific tests named
  below; the other test bodies were not read):
  - `test_molview_stores_history.py` 1–430
  - `test_molview_render_engine.py` 1–135
  - `test_molview_e2e.py` 1–100, plus where its stand-in is used
  - `test_molview_mount.py` 1–250, 1021–1081, 2451–2589
  - `test_molview_model.py` 155–192, 1767–1810
  - `test_molview_measurement.py` 1–72, 230–330
  - `test_molview_isolate_picking.py` 1–60
  - `test_molview_data_shapes.py` 1–100, 380–460
  - `test_molview_ui_context.py` 80–122
  - `test_molview_3dmol_embed.py` 36–60
  - `test_molview_refusal.py` 44–120
  - `test_reserved_label_one_store.py`: header and outline
  - the headers of the two stand-ins
- **Tests not read:** `test_molview_files_door_js.py`. It tests `projects/molview-doors.js`,
  which is outside this slice.

Nothing below compares MolView with VibrationView. Round 1's correctness findings
(D1–D11, U1–U6) are not repeated. Where a duplicate is the cause of one, that is said.

**Totals: 18 code findings and 9 test findings.**

---

## A. Code

### A1. The contact-distance table is a hand copy of a JSON file that no code reads. (high; read both sides)

- `ui.js:1055-1062`:
  ```js
   * ONE HOME, MIRRORED.  The numbers and their literature sources live in
   * `molbuilder/data/contact_distance.json`; this is a copy, because a fetch per
   * measurement for six constants is not worth a round trip.
  const CONTACT_REFERENCE = {
      "Au|S": 2.40, "Ag|S": 2.50, "Cu|S": 2.30,
      "Ni|S": 2.20, "Pt|N": 2.05, "Pd|S": 2.30,
  };
  ```
- `molbuilder/data/contact_distance.json` (`"metals": {"Au": {"d": 2.40, "anchor": "S"} …}`)
  is the declared home. `data/README.md:56` says it is read by `molbuilder.modify`. That is
  not true: no Python file opens it. `modify.py:818` only says `contact_distance` is not
  taken, and `cli.py:800` is an unrelated key. The chapter, § 11.6 near line 4272, says the
  table "has one job left": this readout.
- **Owner:** the JSON file, which is the documented home and carries the sources. But the
  only live consumer reads the copy.
- **What drifts:** someone corrects Pt–N in the JSON, which is where the sources and the
  README say the value lives. The readout keeps showing 2.05 Å. No test notices: the test
  `test_molview_mount.py:1077` asserts the hand-typed `"Au–S reference: 2.40 Å"`, and
  `test_contact_distance_reference.py` no longer exists (only its `.pyc` remains).
- **Smallest fix:** make the JSON the one source the readout reads. Have the server serve it
  once (for example, beside the structure answer's `labels`), or import the file as a module
  asset at load. That is one read, not one per measurement. Then delete the JS copy. The
  README's "read by" line needs correcting either way.

### A2. The filter channels in `_atom.js` are dead code, and the filter row kinds are two hand-kept lists instead. (high; read the code and grepped every caller)

- `_atom.js:159-222` defines `KIND`, `atomChannels` and `channelKinds`. Its header says
  "the enumerated list and the carried list MOVE TOGETHER". **No production file imports any
  of them.** `grep channelKinds|atomChannels|KIND.` over `web/static` finds only `_atom.js`
  itself, and no current test imports them either (only a stale `.pyc` of
  `test_molview_data_shapes` does).
- The row kinds actually offered are typed out twice:
  - `ui.js:2710-2713`:
    `[["by_element","By element"],["by_index","By atom number"],["by_residue","By residue"],["by_label","By label"]]`
  - `stores.js:359-381`, `rowToRule`'s
    `case "by_element" / "by_residue" / "by_label" / "by_index"`, with `default: return null`.
  - `stores.js:224` also writes the default kind: `{ kind: "by_element", value: "" }`.
- `ui.js:2707-2709` says: "WHICH RULES EXIST is the server's vocabulary, and these are its
  names". They are not. The server's names are `by_index_range`, `by_residue_name` and
  `by_region` (`selection.py:118-171`). `rowToRule` translates.
- **Owners:** the rule vocabulary belongs to `selection.py`. The list of row kinds should
  have one home in MolView: a table of rows `{kind, label, toRule}`.
- **What drifts:** add a fifth row (say `by_chain`) to the `ui.js` list. `rowToRule` returns
  `null` for it, so the row silently constrains nothing. The person filters by chain and gets
  every atom.
- The chapter still states the dead rule. § 6.2 says "Those three — element, labels,
  residue — are exactly what the filter enumerates from an atom … the same list", and § 13.3
  has the row "§ 6.2 — the data holds what the filter enumerates … neither list can grow
  without the other". Meanwhile § 9.5 says "The four kinds are always offered".
- **Smallest fix:** delete `KIND`, `atomChannels` and `channelKinds`, and the § 6.2 sentence
  and § 13.3 row that rest on them. Make one row table in `stores.js` that the panel reads
  for its options and `rowToRule` reads for its translation.

### A3. The switch list is typed out three times in production code; the table is consulted only once. (high; read all three places)

- **The owner**, `stores.js:49-56`:
  `const SWITCH_DEFAULTS = { isolate, showIndex, showForces, showCell, showAxis, forceScale }`.
- **Copy 1**, `stores.js:152-157`, the panel's snapshot:
  `isolate: switches.isolate, showIndex: switches.showIndex, … forceScale: switches.forceScale,`
- **Copy 2**, `model.js:199-202`, what decides whether the drawing redraws:
  `JSON.stringify([state.selection, state.isolate, state.showIndex, state.showForces, state.showCell, state.showAxis, state.forceScale])`
- **What drifts:** add a switch row to `SWITCH_DEFAULTS` (say `showBonds`). `setSwitch`
  accepts it and the view lane saves it (`ui-context.js:77` reads `switches()`, which is
  whole). But the snapshot never carries it, so the rail's button never lights. And
  `drawnFrom` never changes, so the drawing never redraws. The test that should catch this
  pins the copy instead (see T6).
- **Smallest fix:** build the snapshot with `Object.assign({...the editor facts}, switches)`.
  Have `model.js` compare `[state.selection, selection.switches()]`. The table then becomes
  the only list.

### A4. The render engine writes out the isolate rule a second time, beside "the one definition". (medium; read both)

- `render-engine.js:98-100`:
  `export function isolateInEffect(switches, selection) { return !!(switches || {}).isolate && (selection || []).length > 0; }`.
  Its comment says this is "the one definition (§ 6.3)", with three readers.
- `render-engine.js:445-450`, `currentDrawnKey()`:
  `(sw.isolate && selection().length) ? "isolate:" + selection().slice().sort(...).join(",") : "all"`.
  This is the same rule inline. It also keys on the raw selection, where `sourceIndexFor`
  drops out-of-range picks and duplicates.
- **What drifts:** change what "isolate in effect" means in `isolateInEffect`, for example
  "with no selected atom drawn". The cost decision in `switchesChanged`
  (`currentDrawnKey() !== drawnKey`) keeps the old rule and skips or forces rebuilds that
  `processFrame` disagrees with.
- **Smallest fix:** derive the key from `sourceIndexFor(selection, switches, count)`, or at
  least from `isolateInEffect`.

### A5. The frozen label's name, its description and a branch on that name are kept in MolView, though the server owns them. (medium; read both)

- `model-jobs.js:47`: `export const FROZEN_LABEL = "frozen_atoms";`. Its comment says
  "the same name the server stores it under".
- The server's constant is `structure.py:114`: `FROZEN_LABEL = "frozen_atoms"`.
- `model-jobs.js:83-84` also keeps its meaning:
  `{ name: FROZEN_LABEL, description: "these atoms are held still by the calculation" }`.
- `model-jobs.js:104` and `ui.js:2551` branch on the name:
  `tone: entry.name === FROZEN_LABEL ? "warn" : …` and `tag.title = name === FROZEN_LABEL ? …`.
- **Owner:** `structure.py`. The chapter (§ 6.6) says the meaning of `frozen_atoms` "lives …
  in the structure model", and that the kind table's labels come with the structure "so a
  rename happens in the table, once".
- **What drifts:** a rename on the server leaves MolView offering, warning about and
  `getFrozen()`-ing a name the calculation no longer acts on. The test that claims to cover
  "the label's one name, on both sides" types the name itself (T7), so it still passes.
- **Smallest fix:** serve the frozen label beside the kind's labels, with its sentence and a
  flag such as "changes the run". Then `offeredLabels` reads the flag instead of comparing
  names, and `FROZEN_LABEL` in JS becomes the served name.

### A6. `BOX_METADATA_KEYS` is a hand copy of Python's `BOX_FIELDS`. (medium; read both lists. That the receiver refuses an unknown metadata key is plausible from `structure.py`'s `check_keys` / `RETIRED_METADATA_KEYS` handling, not traced)

- `model-jobs.js:125-130`:
  ```js
  /* The box half of that list: Python's `structure.BOX_FIELDS` …
  const BOX_METADATA_KEYS = ["cell", "engine_offset", "offset_by", "axis_kind", "vacuum"];
  ```
- The owner, `structure.py:176`:
  `BOX_FIELDS = ("cell", "engine_offset", "offset_by", "axis_kind", "vacuum")`.
- **What drifts:**
  - A key added in Python rides `otherMetadata`. That is harmless, but it never reaches the
    Cell page.
  - A key renamed or retired in Python is worse. The outbound translator
    (`model-jobs.js:492-493`, `Object.fromEntries(BOX_METADATA_KEYS.map(key => [key, clone(per[key])]))`)
    writes the old key, as `null`, into `metadata` on every edit, save and cell commit.
- **Smallest fix:** the server already sends `resolved_box` beside the structure. Send the
  box key list with it too (one field, read once), or have `structureFromServer` take the
  box keys from a served list. Then delete the JS literal.

### A7. The named-value row rule and the "a name is the structure's or a frame's, never both" rule are both copied from Python. (medium; read both)

- The row rule:
  - `model-jobs.js:529-553`, `customizedRow`: "the same rule `Structure`'s `_customized_row`
    applies on the server". It checks the name, the value type, and that unit and note are
    text.
  - The owner is `structure.py:274-311`, `_customized_row`. It also refuses non-finite
    floats and unknown keys, and coerces numpy types.
- The "never both" rule:
  - `model.js:946-954`:
    `const elsewhere = f == null ? (rowsPerFrame || []).some(...) : namesOf(structure.customized).has(name); if (elsewhere) throw …`
  - The owner is `structure.py:366-371` (`normalise_customized`, "a name is the
    structure's or the frames', never both").
- **What drifts:** the server relaxes or tightens a row (for example, it allows `null` to
  clear a value). The Metadata page then refuses what the server accepts, or accepts a row
  the next edit's round trip refuses with a different sentence. Every edit sends the
  structure through `Structure.from_dict`, so the server already judges every row.
- **Smallest fix:** keep only the type coercion the form needs (`custValue`). Let the row go
  into the structure and leave refusing it to the next round trip's answer. Or, if the rule
  must answer at once, ask a server door for it. Do not keep a second rule.

### A8. The filter route takes the structure's atoms in a second, hand-made shape. (medium; read both sides. `web-api.md:124` documents this as a deliberate exception, so the fix is a design change that needs a yes)

- `model-jobs.js:1010-1024`, `resolveFilter`:
  `{ element, labels, residueName: facts.residue || null, atomName: facts.name || null, chainId: facts.chain || null }`
  for each atom, posted as `{atoms, rule}`.
- `web/blueprints/selection.py:51-89`, `_struct_from_atoms`, rebuilds a `Structure` from that
  record:
  - it accepts two spellings of each key: `a.get("residueName") or a.get("residue_name")`,
    and `a.get("labels") or a.get("regions")`;
  - it types its own defaults: `"X"`, `"MOL"`, `"A"`, and the element symbol for the atom name.
- **Owners:** every other door MolView calls takes the one envelope (`structureForServer` to
  `Structure.from_dict`). `web-api.md:133-134` states the rule: "A structure crosses in one
  envelope, in both directions, at every door". `Structure.__post_init__` owns those defaults.
- Inside MolView, the same per-atom fact also already has two spellings:
  `getAtoms()` uses `residue` (`model.js:799`), while `resolveFilter` uses `residueName`.
- **What drifts:**
  - Each new identity column, or each `selection.py` rule that matches on one, must be added
    in two parsers and two places in the browser.
  - The server's placeholders and the filter's placeholders are two sources: change
    `Structure`'s default chain and `by_chain_id` matches a different set on the filter route.
- **Smallest fix:** post `{structure: readData(), rule}` and build the structure with
  `Structure.from_dict`. That would carry the positions the rule ignores, a few kilobytes.
  Then retire `_struct_from_atoms` and its double spellings.

### A9. The sealed layer keeps its own drawing defaults beside the view store's. `mount.js` calls this the forbidden second home. (low-medium; read all sides)

- **The owner**, `stores.js:63-68`:
  `VIEW_DEFAULTS = { style: "stick", radius: 1, background: null, orthographic: false }`.
- **The seal's own copies**, in `3dmol-embed.js`:
  - `:270-271`: `view: Object.assign({ rep: "stick", radiusScale: 1, background: null }, opts.view || {})`
  - `:139-140`: `const rep = view.rep || "stick"; const scale = view.radiusScale || 1.0;`
  - `:362`: `(state.view.rep || "stick") === "stick"`
- `mount.js:182-187` says: "Seeded once with the store's current value so the drawing starts
  on the store's defaults rather than the seal's own — two sets of defaults that merely
  happen to agree is the second home § 7 forbids."
- **What drifts:** change the default style to `"ball-and-stick"` in `VIEW_DEFAULTS`. Any
  path that reaches the seal before the seed, or a `setStyle` carrying a partial view,
  silently falls back to sticks. That includes the bondless-marker branch at `:362`.
- **Smallest fix:** give the seal no defaults. `create` should take the view it is handed
  (mount already seeds it), and `styleSpec` should refuse a missing `rep` rather than
  supplying one.

### A10. `mount.js` decides which failure happened by matching the sealed layer's error text. (low-medium; read both)

- `3dmol-embed.js:252`: `throw new Error("3dmol-embed: 3Dmol-min.js must be loaded first");`
- `mount.js:144`: `if (/must be loaded first/i.test(why)) { said = "…its drawing library did not load. That is a SERVER problem…" }`
- **What drifts:** reword the seal's sentence. Every missing-script failure is then reported
  as a WebGL / browser problem: the user is told to check `chrome://gpu` for a deployment fault.
- **Smallest fix:** the seal throws an error carrying a reason field (for example
  `e.reason = "library-missing"`). The mount branches on the field, and the sentence stays
  the seal's.

### A11. The handle's selection surface is a hand-written list of 17 doors. 13 of them have no outside caller, and its own comment says they do not leave. (low-medium; read `mount.js` and grepped every outside caller)

- `mount.js:332-350` lists
  `get, getState, subscribe, add, remove, clear, all, invert, adopt, setSwitch, setIsolate, writeLabel, setEditor, addFilter, updateFilter, removeFilter, setCombinator`.
- `mount.js:313-316` says: "the only writes that leave are the ones a consumer was found to
  need. Everything else -- adding, removing, inverting, filtering, labelling, adopting,
  picking -- is the panel's".
- Outside MolView, the only uses are `data.selection.get`, `.subscribe` and `.setSwitch`
  (`lib/trajectory/core.js:767` and `:2242`, `structure-card.js`).
- The list also renames the label door's parameters. It forwards `writeLabel: (n, a, v) =>`,
  but the store's order is `(name, verb, atoms)` (`stores.js:311`). A host reading the
  surface's names would pass atoms where the verb goes.
- **What drifts:** the list grows by hand as the store grows. The reason it exists ("only
  what a consumer needs") is already false for 13 of its 17 entries.
- **Smallest fix:** narrow it to the doors a consumer uses (`get`, `getState`, `subscribe`,
  `setSwitch`), and drive tests through the panel's own controls. Alternatively, keep a
  `"minus toggle"` rule and stop re-listing the doors.

### A12. The draft and point envelope version `1` is written five times in `history.js`. (low; read)

- The writes are `history.js:142`, `:143` and `:193` (`{ v: 1, state: bytes … }`). The reads
  are `:295` and `:460` (`saved.v !== 1`).
- The same idea in `ui-context.js:22` has one constant, `const VERSION = 1`, read at `:74`
  and `:109`.
- **What drifts:** bump the writer at one site and the readers refuse that site's own files,
  or the reverse.
- **Smallest fix:** one `const VERSION = 1` in `history.js`.

### A13. The package has three implementations of the listener list. (low; read all three)

- `stores.js:71-87`, `subscribable()`, with `add` returning a disposer and `fire` wrapped in
  try/catch.
- `model.js:154-155`, `:350-360`, `:1083-1089` and `:1100-1106`, which hold
  `structureListeners` and `frameListeners` as arrays, each with its own announce loop and
  disposer.
- `mount.js:238-261`, which holds `playbackHeard` as a `Set`, with its own announce and disposer.
- **What drifts:** a fix to one, for example letting a listener's error reach the console,
  or guarding against unsubscribing during a fire, does not reach the others.
- **Smallest fix:** export `subscribable` (from `stores.js`, or a small leaf like `_atom.js`)
  and use it in `model.js` and `mount.js`.

### A14. Two deep-copy helpers and several inline copies do the same job. (low; read)

- `model.js:383`: `const copy = (v) => (v == null ? v : JSON.parse(JSON.stringify(v)));`
- `model-jobs.js:113-115`: `function clone(v) { return v == null ? null : JSON.parse(JSON.stringify(v)); }`.
  It differs only in what it returns for `undefined`.
- Inline round trips: `model.js:847`, `:851`, `:868`, `:886`, and `ui.js:1924`.
- **Smallest fix:** keep one helper and import it everywhere.

### A15. Two colours in the drawing duplicate stylesheet tokens, and one already disagrees with it. (low; read both)

- The ruler's colour:
  - `3dmol-embed.js:75-78` says "Cool blue in both, the hue the ruler already is elsewhere:
    the chip's border takes `--molviewer-color-accent`", then sets
    `MEASURE_MARK = { color: "#5ad2ff" … }` and `MEASURE_ARROW = { color: "#5ad2ff" … }`.
  - `molview.css:93` has `--molviewer-color-accent: var(--accent, #6ba6ff);`. So the 3D marks
    and the chip are already different blues. The seal reads other scene colours from CSS
    tokens (`readCssVar`, `:120-130`); this one is a literal.
- The "Dark" background preset:
  - `ui.js:604-605` has `["#1d2128", "Dark"]`, described as "the card's own dark".
  - That is the fallback of `--molviewer-color-surface` (`molview.css:81`). The stylesheet's
    header (`:29`) says no rule "ever says `#1d2128`".
  - A host that themes `--bg-card` changes the card but not the swatch named after it.
- **Smallest fix:** declare a `--molviewer-scene-measure-color` token and read it the way the
  cell colour is read. Make the Dark swatch's value the resolved surface token.

### A16. The sealed layer has a dead label rule that disagrees with the live one, and a "two callers" primitive whose second caller re-implements it. (low; read)

- `3dmol-embed.js:564-566` has a fallback for an arrow with no `labelEnd`:
  `[a.end[0] * 1.05, …]`. The only arrows with labels are the triads, which always carry
  `labelEnd` (`render-engine.js:361-363`, `LABEL_PAST_TIP = 1.15` at `:279`). So the branch
  is dead and states a different "how far past the tip" from the one that is live.
- `3dmol-embed.js:406` says "ONE GLOW PRIMITIVE, two callers", but `redrawGlow` has one
  caller (`:610`). `redrawMeasurement`'s `mark` (`:459-468`) builds the same sphere by hand.
- **Smallest fix:** delete the 1.05 fallback. Have `mark` call `redrawGlow`'s sphere builder.

### A17. The Cell page works out "vacuum is not used" and "(default)" from raw keys, re-stating part of the server's cell resolver. (low-medium; the server rule was read, and the drift is plausible)

- `ui.js:2893-2897`: `const manual = !!rawCell; cellRegime.textContent = manual ? "The box is an explicit cell — vacuum is not used." : …`.
- `ui.js:2929`: `["Vacuum", vector(cell.vacuum), !rawVacuum, manual]`.
- `ui.js:2925`: Axes is marked default when `!rawAxes || isolatedEverywhere(rawAxes)`.
- **Owner:** `structure.py:1068-1087`, `resolve_cell`. An explicit cell wins verbatim, and
  with no cell the vacuum pads isolated axes only (`transport` ignores it). The browser marks
  only the first case. With no cell and a transport axis, the page shows the vacuum as used
  on that axis, which the resolver ignores.
- **Smallest fix:** add to `to_wire`'s `resolved_box` whether each axis's vacuum reaches the
  box. That is one boolean per axis, from the resolver. The page then reads it instead of
  guessing.

### A18. The demo's file door reads a `frames` field that MolView never sends. (low; read both)

- `demo.js:124`: `frames: (payload && payload.frames) || undefined,`
- MolView's one caller, `ui.js:815-818`, sends `files.save(destination, stem, { structure: file.structure })`.
  The frames already ride inside the envelope.
- **Smallest fix:** drop the field.

---

## B. Tests

### T1. The history tests use a hand-typed workspace even though the real one runs in node. (medium; read both)

- `test_molview_stores_history.py:50-80` defines `makeStore()`, a stand-in for the
  workspace, used about 19 times.
- `support/workspace_page.py` serves the real workspace (`workspace/dispatcher.js` on the
  app's routes) in node. `test_molview_mount.py` and `test_molview_model.py` already use it
  (`molview_page.run(..., workspace=True)`).
- § 13.1 forbids "a stand-in for what is real and reachable". The one need named for it is
  `failWrite()`, a refused write. A refusal can be caused for real: a full quota
  (`_node_esm.TO_THE_BRIM`), or a server refusal.
- **What drifts:** the dispatcher's door changes, for example `persist` answering a promise
  of the server's verdict, which `watchAnswer` depends on. The history suite stays green on
  its own fake.
- **Fix:** drive `createHistory` with `ws` from `workspace_page`.

### T2. The end-to-end test copies the demo page's workspace stand-in verbatim. (medium; read both)

- `test_molview_e2e.py:42-63`, `_MEMORY_WORKSPACE`, says "the demo page's stand-in, as a
  JavaScript expression". It is used five times (`:200`, `:608`, `:683`, `:744`, `:839`).
- The original is `demo.js:67-89`, `memoryWorkspace()`.
- Change one and the other does not follow. This test runs on the real server, where the
  real workspace door is reachable.
- **Fix:** mount the extra viewers with the page's real workspace, or one shared stand-in file
  under `tests/support/` that the demo does not carry.

### T3. The render-engine suite stands in for the sealed layer and hand-builds the model's data source in an old shape. (medium; read both)

- `test_molview_render_engine.py:38-81` defines `EMBED`, a stand-in for `3dmol-embed.js`.
  The real sealed layer runs in node over `support/molview_3dmol_standin.js`; the mount tests
  do exactly that. Its one named need is `__setShortBy` (the heal test).
- `test_molview_render_engine.py:86-110`, `makeSource`, hand-types the structure as
  `{elements, annotations, cell: null}`. The model's structure has no `cell` field: the box
  is `periodicity` (§ 6.2, `sceneFor(s.periodicity)`). It also types the switch list a fifth
  time: `{ isolate, showIndex, showForces, showCell, showAxis, forceScale }`.
- **What drifts:** the engine reads the model's real source, which this suite never builds.
  A field the model renames passes here.
- **Fix:** build the source from `createModel` (`molview_host.mjs`) with a loaded road
  structure, over the real sealed layer and the 3Dmol stand-in. Keep a short-drawing stand-in
  only for the heal test.

### T4. The view-lane tests copy the mount's wiring rule and stand in for the engine. (low-medium; read both)

- `test_molview_ui_context.py:87-112`, `session()`, wires the lane "as `mount.js` attaches
  it". It copies `hasTruthLane: !(opts.model.mode === "readonly")` from `mount.js:290` and
  uses a hand engine, `makeEngine`. The reason given is "since the real mount needs a DOM",
  but the DOM stand-in exists (`support/molview_dom_standin.js`), and the mount tests mount
  for real with it.
- **What drifts:** change how `mount` attaches the lane (the rule, `playback`, `canvasEl`) and
  these tests still pass on their copy.
- **Fix:** mount through `molview_host.mount` with the two stand-ins, as
  `test_molview_mount.py` does.

### T5. "The rule vocabulary is the server's" is checked only against a hand-typed list. (medium; read both)

- `test_molview_stores_history.py:245-281` asserts hand-typed rule shapes:
  `{"op": "by_residue_name", …}`, `{"op": "by_region", …}`, `{"op": "by_index_range", …}`.
  It never asks `selection.py`. Its own docstring names the failure it cannot see: "a panel
  that builds rules nothing can evaluate — which no unit test of the panel alone would ever
  notice".
- **Fix:** post each built rule to `/api/selection/eval` (the app's routes run in node) on a
  road structure, and assert the matched atoms.

### T6. The snapshot test pins the copied key list, so it cannot catch A3. (low-medium; read)

- `test_molview_stores_history.py:357-361` asserts
  `keys == ["combinator", "filterOutcome", "filters", "forceScale", "isolate", "mode", "selection", "showAxis", "showCell", "showForces", "showIndex"]`.
  That is a transcription of `snapshot()`, the kind of list § 13.1 rules out.
- The rule it claims to guard is "a fact the store keeps but does not hand over does not
  exist". It would hold if the test asserted that every key of `sel.switches()` is in the
  snapshot. As written, a switch added to `SWITCH_DEFAULTS` passes.

### T7. The reserved-label test types the name itself instead of reading the owner's constant. (low-medium; read)

- `test_molview_reserved_label_js.py:33` has `FROZEN = "frozen_atoms"  # the label's one name, on both sides`.
  The owner, `structure.FROZEN_LABEL`, is importable in the test.
- As written, a server rename leaves the JS name, the test's pair and the test agreeing with
  each other, while the server acts on another name. A5's drift goes unseen.
- **Fix:** `from molbuilder.structure import FROZEN_LABEL`, and save the pair through
  `Structure.frozen_atoms` so the name comes from the server.

### T8. Facts with an owner are hand-typed in the mount tests. (low; read)

- `test_molview_mount.py:1823`, `:2017` and `:2587` type the kind table's labels:
  `["L-electrode", "R-electrode", "bridge", "buffer"]`. The owner is `kind_table.labels()`.
  § 6.6 says those names are examples, "not as a list this document keeps".
- `test_molview_mount.py:1077-1081` types `2.40 Å` (see A1).
- **Fix:** read `kind_table.labels()` and the contact-distance JSON in Python, and pass them
  as `given=`.

### T9. Small overlaps and a source lift. (low)

- `test_molview_model.py:1767` and `test_molview_reserved_label_js.py:170` both pin that
  `translate` goes to `/api/modify/translate`. The one-mutation-in-flight test
  (`test_molview_reserved_label_js.py:191`) is about § 11.1, not reserved labels, and sits in
  the wrong file.
- `support/molview_page.py:43` reads the source text for the lane delay:
  `LANE_DELAY = lifted(UICTX, "const WRITE_DELAY_MS = ", ";")`. Exporting the constant from
  `ui-context.js` would let the harness import it instead of reading the file.

---

## The five that matter most

1. **A1:** the contact-distance readout reads a hand copy, and its declared home
   (`data/contact_distance.json`) has no reader at all.
2. **A2:** `_atom.js`'s filter channels have no caller. The row kinds are two hand-kept lists
   (`ui.js`, `stores.js`) that can silently disagree, and the chapter still states the dead rule.
3. **A3:** the switch list is copied into the snapshot and the redraw check. A new switch would
   never light its button or redraw, and T6 pins the copy instead.
4. **A5, A6, A7:** three server rules are copied into the browser: the frozen label's
   name and meaning, the box key list, and the named-value row rules.
5. **T1–T4:** tests stand in for things that are real and reachable in node: the workspace
   (twice), the sealed layer, and the mount's lane wiring. They can stay green while the real
   doors change.
