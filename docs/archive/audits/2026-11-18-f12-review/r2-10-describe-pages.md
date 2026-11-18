# F12 review, round 2 — R2-10: the describe pages' scripts

Checkout `the checkout` at f578482db (2.77.2), read as it stands on
disk: the uncommitted change to `task-setup/viewer.js` (the structure-pair names
written either way, ~1743) is in what I read. Nothing in the checkout was
touched; the only file written is this report.

Paths are under `molbuilder/web/static/` unless they start with `molbuilder/`,
`tests/` or `docs/`.

## What I read

**Read whole, every script of the slice:**
- `task-setup/viewer.js` (all 5,478 lines).
- `lib/*.js`, all 33 files directly in `lib/`.
- `lib/projects/*.js` (11 files).
- `lib/workspace/dispatcher.js`.
- `modify/viewer.js`, `selection-bootstrap.js`, `periodicity.js`, `slab-panel.js`.
- `modify/structure/*.js` (8 files).
- `structure-optimization/viewer.js`, `this-machine/page.js`, `documents/page.js`, `jupyternb/index.js`.

I did not read the slice's `.css` files, because the brief asks about scripts.

**Python I read to check the other side of each finding:**
- `web/blueprints/_shared.py` (item_record, `_item_to_field`), `files.py` (`_entry_dict`, read and range limits, zip_prepare), `notify_setup.py` (`_row`, `_state`), `checkpoint.py`, `modify.py` blueprint `meta`, `workspace_storage.py`.
- `molbuilder/units.py` (`size_words`), `cell.py`, `structure.py` (`AXIS_KINDS`), `modify.py` (the slab), `residues.py`, `nucleic.py`, `peptide.py`, `pseudos.py` (`resolve_psml_lib`), `job_request.py` (notify), `scheduler/record.py` (the machines listing), `template.py` (`value_labels`), `checkpoint.py` (`State.short`), `config_dir.py`, `cli.py` (notify-token), `envs/recipes.py` (`_JUPYTER`), `kmesh.py`.

**Tests:**
- **Read whole:** `tests/test_task_setup_save_bar_js.py`, `tests/test_slab_orthogonal_choice_js.py`.
- **Read in part (the harness, and the tests named below):**
  - `test_structure_page_js.py` (1–200, 295–330)
  - `test_structure_save_js.py` (1–110)
  - `test_structure_smiles_js.py` (1–65, 275–339)
  - `test_structure_dna_js.py` (1–65)
  - `test_structure_peptide_js.py` (30–70)
  - `test_task_setup_notify_js.py` (1–200)
  - `test_checkpoint_sensor_js.py` (590–693)
  - `tests/support/molview_page.py` (head)
  - `tests/support/task_setup_page.py` (head)
- **Scanned only with grep** (for source or doc text read, and for per-file node runners): the other JS tests that drive the slice — `test_projects_public_surface_js`, `test_projects_api_envelope_js`, `test_projects_render_sidebar_js`, `test_page_busy_js`, `test_tab_strip_js`, `test_one_stop_list_js`, `test_validation_findings_js`, `test_molview_files_door_js`, `test_chemistry_module_js`, `test_form_schema_*`, and the rest. None of them reads source text as an assertion.
- **e2e:** the 18 files that open these pages were scanned with grep only, and none was read whole. Their file reads are outcome reads: the file the page wrote.
- **Not read:** the rest of those files.

**Round 1** (`r1-J1-describe-pages.md`) is not repeated. F18 names the duplicate behind round 1's P-D6.

---

## Totals: 33 findings (28 in the code, 5 in the tests)

## The five that matter most

1. **F1:** the builder panels keep their own copy of the builders' input rules. Today the peptide panel refuses `AR[SEP]C`, which the server builds.
2. **F2:** the Modify tab's axis kinds and slab choices are typed in JS, though `/api/modify/meta` exists to serve exactly these.
3. **F4:** a setting's value is put into words by four renderers. `scf_must_converge` reads "yes" in its dropdown and "on" everywhere else.
4. **F8:** Task setup's rule is "the page spells neither file name", yet it types `task.json` / `job-request.json` into eight sentences.
5. **F20:** five loaders inject a vendored script or stylesheet. Only CodeMirror's is the exported one.

---

## A. One door per fact: a page computes or types what the server answers

### F1 · The builder panels re-implement the builders' input rules; the peptide panel already refuses what the server builds
High. Read both sides; drift verified by running both.

**Page:**
- `modify/structure/peptide.js:36`: `var VALID_AA = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;`. Applied at `:61-67`, which refuses before any request: "Sequence must use one-letter amino-acid codes".
- `modify/structure/dna.js`:
  - `:28-30` `VALID_DNA = /^[ACGT]+$/i`, `VALID_FORMS = ["B","A","Z"]`, `VALID_BACKENDS = ["auto","threedna","amber","rdkit"]`.
  - `:39-66` `_parseDnaNotation`, whose own comment says it "mirrors the server's nucleic.parse_dna_notation".
  - `:88-112` the two-strand rules, including "overhangs aren't supported yet".
  - `:153-171` the Z-DNA rule, commented "Matches the server-side `_is_alternating_gc` predicate exactly".
- `modify/structure/rna.js:25-27` (the same three lists) and `:47`.
- `modify/structure/panel-deps.js:174-185` `strandCore`, which strips the 5'/3' labels.

**Owner:**
- `molbuilder/residues.py:1-18` (the grammar: `bracketed = "[" 3-or-4-letter-code "]"`), `AA_ONE_TO_THREE`, `DNA_/RNA_ONE_TO_THREE`, and `_strip_directionality`.
- `molbuilder/nucleic.py:43-76` `parse_dna_notation`.
- `molbuilder/builders/backends/__init__.py:120` `_AUTO_ORDER` and `:53` `DEFAULT_FORM`.
- `builders/backends/_threedna.py:244`.
- The arch table routes a build through `builders.build`, its findings through `validation/builders.py`.

**What goes wrong now:**
- `parse_peptide_sequence("AR[SEP]C")` gives `['ALA','ARG','SEP','CYS']`; phosphoserine is a registered modified residue.
- The panel's regex answers `false` for the same text (both run here). A person typing a modified residue is refused with a sentence that says it cannot be built.
- The DNA and RNA panels refuse bracketed codes the same way: `strandCore` keeps the letters, so `[5MC]` becomes `MC` and fails ACGT.

**What drifts next:**
- The day the server takes overhangs, a new backend, or a new form, the page still refuses them in its own words.

**Smallest fix:**
- Drop the client-side letter, notation, form and backend checks. Send what was typed; the chain already shows the server's refusal (`panel-deps.js:82`).
- If the panel needs the forms and backends for its menus, serve them from the builders' tables.

### F2 · The Modify tab's dropdown vocabularies are typed in JS beside the door built to serve them
Medium-high. Read both sides.

**Page:**
- `modify/periodicity.js:25`: `var AXIS = ["isolated", "periodic", "transport"];`
- `modify/slab-panel.js:24`: `const GROWS = [["+z", "+z (up)"], ["-z", "-z (down)"]];`
- `modify/slab-panel.js:28`: `const SEQUENCES = ["ABC", "ACB"];`

**Owner:**
- `molbuilder/structure.py:117`: `AXIS_KINDS = ("periodic", "isolated", "transport")`, which `periodicity_gate.py:263, 426` checks against.
- `molbuilder/modify.py:843-847`: `if grow not in ("+z", "-z")` … `if sequence not in ("ABC", "ACB")`. These are inline literals with no name.
- **The door:** `web/blueprints/modify.py:83-127` (`/api/modify/meta`), whose docstring says "Anti-pattern dodged: parallel field-metadata tables in the web layer that drift from the Python tuples". The slab panel already fetches it (`slab-panel.js:117`).

**Drift:**
- Add an axis kind to `AXIS_KINDS`: the gate takes it, but the Cell panel never offers it.
- A structure that carries it is shown with `sel.value` set to a value the select has no option for. Apply (`periodicity.js:580-582`) then sends whatever the select shows, overwriting the stored kind.

**Smallest fix:**
- Serve `axis_kinds` (from `AXIS_KINDS`) and the grow and sequence choices in `/api/modify/meta`.
- Name the two tuples in `modify.py`.
- Fill the selects from the answer.

### F3 · The Cell panel recomputes the cell gate's verdicts in JS
Medium. Read both sides. Whether a dry-run door is the right fix is plausible, not verified.

**Page:**
- `modify/periodicity.js:191`: `var DET_QUIET = 1e-6;`. This is the value of `cell.py:61 ZERO_VOLUME_TOL = 1e-6` under another name.
- `:197-216` computes the determinant and words its own refusal: "These three vectors are left-handed … and the server will refuse them".
- `:321-351` `spanAlong` computes the atoms' extent along a lattice vector.

**Owner:**
- `cell.check` (`cell.py:379-390`, `cell.left_handed`: "This cell is mirrored…").
- `ResolvedCell.clearances` / `empty_span` (`cell.py:111-128`).
- The arch table, § 3.2: "what is true of a box — its volume and handedness … the length the atoms leave empty along each lattice vector — `cell.resolve`". § 2's "Do NOT … compute a box in JS" names the same thing.

**Drift:**
- Change `ZERO_VOLUME_TOL` or the left-handed wording: the panel predicts by its own constant and says its own sentence. One finding then reads two ways on one page.

**Caveat:**
- The panel works on the staged, unsent cell, which no route answers today.

**Smallest fix:**
- A read-only check of the staged `block` (the gate's resolve and check without the commit), asked as the person types. The note then shows the gate's sentence and the span the server measured.

### F4 · A setting's value in words: four renderers, and the tri-select already disagrees
Medium. Read both sides; a concrete case found in the served schema.

**Page:**
- `lib/form-schema.js:191-199` `show()`: `if (v === true) return "on"; if (v === false) return "off";`, before it looks at `choice_labels`.
- `lib/form-schema.js:272-285` `makeTriSelect`: `const said = { auto: f.null_label, true: "yes", false: "no" };`
- `task-setup/viewer.js:2514-2545` `renderValue` / `valueWords` / `wordsOf`, which read `choice_labels`.
- `structure-optimization/viewer.js:220-227` `mountRecommended`'s `show()`: on/off and "(empty)", with no `choice_labels`.

**Owner:**
- `template.value_labels` / `choice_words` (`molbuilder/template.py:2512-2532`), served on every field as `choice_labels` (`web/blueprints/_shared.py:393-395`).

**What goes wrong now:**
- The SIESTA optimization form serves `scf_must_converge` with `choice_labels {"true": "on", "false": "off"}`.
- Its dropdown offers "yes" / "no". The same field's caption, Task setup's stage cell and the Recommended panel say "on" / "off".
- A choice with words of its own (a vibration's mode probe, `all` meaning "All modes") shows raw in the Recommended panel.

**Smallest fix:**
- One `valueWords(field, v)` in `form-schema.js`, reading `choice_labels` and the unit.
- The tri-select's options, `show()`, the Recommended panel and Task setup all ask it.

### F5 · The sidebar writes file sizes in words of its own
Medium. Read both sides.

**Page:**
- `lib/projects/list.js:105-110` `_humanSize`: `" K"` (no decimal), `" M"`, `" G"`.
- `lib/projects/preview.js:603-606`: `sizeMB` / `capMB`, "MB" over binary units.

**Owner:**
- `molbuilder/units.py:134-141` `size_words` ("49.2 GiB"). The arch table § 3.2 makes it the door for "a size in bytes, in words — a file's, a folder's".
- It is already served beside zip_prepare (`web/blueprints/files.py:765, 838`), and the Download confirm draws it (`lib/projects/mutation-bar.js:280`).

**What goes wrong:**
- One folder's file reads "1.2 G" in its sidebar row and "1.2 GiB" in the Download confirm.
- A 17.4 KiB file reads "17 K".

**Smallest fix:**
- `_entry_dict` (`files.py:391-426`) serves `size_words` beside `size`, and the row draws it.
- The preview says the stat answer's words.

### F6 · One machine's memory, spelled twice on one card
Medium. Read both sides.

**Page:** `task-setup/viewer.js:4171-4176`, used at `:4217`:
```js
function _fmtGB(gb) { if (gb >= 1024) return (Math.floor(gb / 102.4) / 10) + " TB"; return Math.floor(gb) + " GB"; }
```

**Server:**
- `molbuilder/scheduler/record.py:1103-1105`: `bits.append(f"{float(_mem):g} GB")`, in the `summary` the same card shows (`viewer.js:4076-4079`).
- Raw `mem_total_gb` sent at `:1145`.
- A queue's memory is the server's `mem_range` (`viewer.js:4154, 4214`).
- **Owner:** the scheduler's quantities (arch § 3.2: "a memory request is the scheduler's quantity, said in its own terms").

**What goes wrong:**
- A 1031.6 GB node reads "1031.6 GB" in the machine line and "1 TB" under the memory field. That is one fact from two writers.

**Smallest fix:** the machines answer carries the node's memory in the scheduler's words, and `_fmtGB` goes.

### F7 · Task setup re-implements the notify block's grammar
Medium-low. Read both sides.

**Page:**
- `task-setup/viewer.js:4443-4471` `notifyValues` writes `channels: chans === null ? ["*"] : chans`.
- `:4476-4479` `_everyOrThese` reads `["*"]` back. Its comment calls it "the one reading of both lists (task.py's `_every_or_these`)", but the function lives in `job_request.py`, so the citation is stale too.
- `:4451-4454` the hours rule.

**Owner:** `molbuilder/job_request.py:567-589` `_every_or_these` ("`["*"]` every (`None`), `[]` none").

**Drift:**
- Change how "every" is spelled in the record: the page goes on writing and reading `["*"]`, Save refuses, and the card reads a stored list wrongly.

**Smallest fix:**
- The folder answer serves the notify block as its reader read it (channels and report as every / none / names), and the card paints from that.
- The page's write is the card's three states, which the save door spells.

### F8 · Task setup's rule says it spells no file name, then spells both in eight sentences
Medium. Read both sides.

**The rule:** `task-setup/viewer.js:38-42`: "The description's and the hand-over's names are the folder answer's (`task_name`, `handover_name`): the page spells neither."

**Where it follows the rule:** `:1688`, `:1694`, `:1762`, `:1782` (`said.task_name`, `said.request_name`, `said.handover_name`).

**Where it spells them anyway:**
- `:1822` "This folder carries no task.json and no hand-over."
- `:1840` "task.json is here and its reader refuses it"
- `:1849` "job-request.json is here and its reader …"
- `:2586` "in task.json:"
- `:2713` "this page edits task.json"
- `:4793-4796` the save bar
- `:4904` "Saving task.json and job-request.json…"
- `lib/task-handover.js:75` "writes task.json when you save"

**Owner:** `runfiles.WRITTEN`, as the folder answer serves it.

**Drift:**
- Rename either record: the file rows and the editor's tabs follow, while the save bar, the refusals and the busy cover say the old names.
- Within `readFolder` itself, `:1762` names the hand-over by the answer while `:1840` names the description by a literal.

**Smallest fix:** keep `task_name` / `request_name` in `_fs` and compose those sentences from them.

### F9 · The measure card keeps its own list of the machine axes, and branches on `use_gpu`
Medium-low. Read the page and the comment that names the server's flag.

**Page:**
- `task-setup/viewer.js:3121-3124`: "what it tries is stated in What to measure: the ranks, the cores per rank and, on the GPU, the GPU counts."
- `:1257-1268`: `name === "use_gpu" ? (… "the cpu-vs-gpu axis …") : (…)`.

**Owner:**
- The catalogue's `allocation` flag, served per item as `machine_answers` by `/api/task-setup/sweepable`.
- The page already reads it at `:88-103`.

**Drift:** add or drop a machine-answered item and the sentence lists the old three. The branch keys a sentence on one setting's name in shared code.

**Smallest fix:** list the `machine_answers` items by their labels. If a per-item note is needed, it is a catalogue word, not a branch.

### F10 · Server limits typed on the page
Low-medium. Read both sides.

**Read limits:**
- `lib/projects/preview.js:59`: `BULK_READ_MAX_BYTES = 16 * 1024 * 1024`, a copy of `web/blueprints/files.py:71 _MAX_READ_BYTES`.
- `:64`: `PAGE_BYTES = 256 * 1024`, a copy of `files.py:901 _DEFAULT_RANGE_BYTES`. A third copy is `lib/inspectors/source.js:16`.

**A refusal matched by its words:**
- `preview.js:537`: `if (/not valid UTF-8/.test(reason))` matches the sentence at `files.py:583`. A rewording of that sentence silently drops the binary notice.

**The workspace window:**
- `modify/viewer.js:349-350`: "A tab keeps its last 30 saves." is a copy of `web/blueprints/workspace_storage.py:62 _STATE_WINDOW = 30`.

**Smallest fix:**
- The stat answer carries the single-read ceiling.
- The range read relies on the server's own default.
- The read refusal carries a flag (binary) rather than words to match.
- The window's sentence comes from the server, or loses its number.

### F11 · Two pages each compose a channel's test state, in different words
Low-medium. Read both sides.

**Page:**
- `this-machine/page.js:69-77`: "reached " + ago, "failed " + ago, "never tested".
- `task-setup/viewer.js:5383-5390`: `known.kind + ", reached when last tested"`, ", failed when last tested", ", never tested".

**Owner:** `notify_setup._row` (`web/blueprints/notify_setup.py:173-183`) serves `tested_ok` and `tested_at`. Both pages work their sentence out from those.

**Smallest fix:** `_row` serves the words (`tested_words`), and both pages draw them.

### F12 · This machine judges the channels file's mode against its own literal
Low-medium. Read both sides.

**Page:**
- `this-machine/page.js:127-130`: `d.mode && d.mode !== "0o600" ? "  — warning: that file is " + d.mode + ", not 0600" : ""`.

**Owner:**
- `molbuilder/config_dir.py:58 PRIVATE_FILE_MODE = 0o600`.
- The server sends the raw `oct()` (`notify_setup.py:214-222`).

**Drift:** a stricter `0o400` file is warned as "not 0600". Any change in how the mode is spelled warns on every load.

**Smallest fix:** `_state` answers the warning, or a `mode_ok` flag, judged against `PRIVATE_FILE_MODE`.

### F13 · The channels-file recipe is composed twice
Low-medium. Read both sides.

**Page:** `this-machine/page.js:311-316`: `JSON.stringify({ channels: { molbuilder: { url: d.url, key: d.key } } }, …)`, with the channel name fixed.

**CLI:** `molbuilder/cli.py:2213-2215`: `{"channels": {channel: {"url": …, "key": token}}}`, with the channel name from `--channel`.

**Shared and not shared:** `auth_setup.issue_notify_key` is shared by both callers (`auth_setup.py:89-111`, "One door, because there are two of them"). The recipe is not.

**Smallest fix:** the issue route answers the recipe, composed by one function the CLI calls too.

### F14 · The Jupyter tab lists the notebook env's packages by hand, twice, and the list is already short
Low-medium. Read both sides.

**Page:** `jupyternb/index.js:276-277` and `:384-386`: "the notebook env's own python — numpy, scipy, pandas and matplotlib".

**Owner:** `molbuilder/envs/recipes.py` `_JUPYTER.conda_packages` (`~2636-2668`). It also carries `ase`, `sisl`, `netcdf4` and `xarray`.

**Smallest fix:** the status answer carries the recipe's own words (its description, or its analysis packages).

### F15 · The psml_lib caption restates the resolver, and disagrees with it
Low-medium. Read both sides.

**Page:**
- `structure-optimization/viewer.js:398-415`: an absolute or `~` path is shown as typed; a relative one as "→ <v>, in the projects folder".
- Its comment (`:411-413`) says "The Save step also tries dest-relative first".

**Owner:** `molbuilder/pseudos.py:148-187` `resolve_psml_lib`:
- A relative path is measured from the tree root walked up from `dest`.
- `./` and `../` are refused (`:180-186`).
- `describe_psml_anchor` is its words.

**What goes wrong:** "./pseudopotential" is captioned "in the projects folder", while the server refuses that spelling.

**Smallest fix:** drop the caption, since the live preflight already says it, or have the preflight answer carry the resolved folder.

### F16 · The history view shortens a state id itself
Low. Read both sides.

**Page:** `lib/projects/checkpoint.js:348` and `:541`: `state.parent.slice(0, 7)`; `:827`: `sha.slice(0, 7)`.

**Owner:**
- `molbuilder/checkpoint.py:290-292`: `def short(self): return self.id[:7]`.
- Served as `short` (`web/blueprints/checkpoint.py:112`).

**Drift:** change the short length and a row's "from abc1234" no longer matches the parent row's own id.

**Smallest fix:** serve `parent_short`, or look the parent up among the listed states and use its `short`.

### F17 · Structure suffixes in words, typed beside the served rule
Low. Read both sides.

**Page:**
- `lib/structure-card.js:134-135`: "(not loadable — .xyz / .pdb only)" and "Pick a .xyz / .pdb in the Projects sidebar."
- `:192`: "(this tab loads .xyz and .pdb files)".
- `lib/projects/list.js:484-487`: ".molstruct.json" in the rename hint.

**Owner:**
- `isStructureFile` / `pairedSuffix` (`lib/projects/state.js:336-365`), read from the header the server fills from `sidecars/molstruct.py:209 GEOMETRY_SUFFIXES`.

**Smallest fix:** compose the words from `_pairing().read` and `.paired`.

### F18 · The name rule typed three ways on the page (behind round 1's P-D6)
Low. Read the page side; the server's pattern located.

**Page:**
- `lib/projects/dialogs.js:22-30` `_nameRefusal` refuses only `/`, `\`, `.` and `..`.
- `lib/projects/mutation-bar.js:337, 362`: placeholder "letters / digits / _ / -".
- `lib/projects/checkpoint.js:687-688`: "Letters, digits, - and _ only".

**Owner:** `web/blueprints/files.py:1163 _UPLOAD_FILENAME_RE` / `_writable`, and `validate_folder_name`.

**Smallest fix:** serve the rule's pattern and words once (with the roots answer), and have the dialogs' `validate` and the placeholders ask it.

### F19 · The Download tooltip types what the archive leaves out
Low. Read both sides.

**Page:** `lib/projects/mutation-bar.js:254-257`: "no checkpoint history (.git, .binsnapshots), no workspace store and no engine scratch".

**Server:** zip_prepare answers `left_out` (`files.py:841`), which the confirm already draws (`mutation-bar.js:288-292`).

**Smallest fix:** the tooltip says nothing it cannot read, or says the last `left_out` it was told.

---

## B. Duplicated or redundant code

### F20 · Five loaders inject a vendored script or stylesheet
Medium. Read all of them.

- `lib/codemirror-load.js:147-169`: `injectStylesheet` and `injectScript`, the only exported ones.
- `lib/markdown-render.js:23-31` `_loadScript`, and `:129-135` its own KaTeX `<link>`.
- `lib/media-export.js:27-43`: the gif loader inline. Its comment says "the same lazy shape as the checkpoint graph's vendor load".
- `lib/projects/checkpoint.js:412-434` `_loadGitGraph`.
- Outside the slice, the same again: `lib/inspectors/markdown.js:31-37` `loadScript`.

**Drift:** they differ already. Only `injectScript` sets `async=false`, and only CodeMirror's checks that the stylesheet loaded. A CSP or base-path change touches five places.

**Smallest fix:** one vendored-asset loader (once per source, with the global checked) in `lib/`, asked by all five.

### F21 · Five element builders
Low. Read all of them.

- `lib/dom.js:25-30` `el(tag, cls, text)`, whose header says "use this one copy".
- `lib/dialog.js:91-96` `_node`: identical.
- `lib/tree-picker.js:35-40` `_el`: identical.
- Two attribute builders with different rules:
  - `lib/form-schema.js:74-115` sets DOM properties and refuses `on*` keys.
  - `task-setup/viewer.js:143-153` always uses `setAttribute` and refuses nothing.

**Smallest fix:** dialog and tree-picker take `dom.el` (it is loaded by the header on every page). One attribute variant lives beside it.

### F22 · Three `CSS.escape` stand-ins
Low. Read all three.

- `lib/form-schema.js:994-999` `cssEsc`.
- `lib/projects/list.js:801-804` `_cssEscape`.
- `lib/validation-findings.js:188`, inline.

Each has a different fallback.

**Smallest fix:** every browser this app supports has `CSS.escape`; use it, or keep one helper.

### F23 · The five builder panels copy their wiring
Medium-low. Read all five.

- `modify/structure/smiles.js:63-122` (wirePanel) and `:124-154` (the boot block).
- `name.js:53-105` and `:107-135`. Its header says "Identical shape to `smiles.js`".
- `peptide.js:75-122` and `:124-153`.
- `dna.js:186-283`.
- `rna.js:79-173`.

**What is copied:** `setStatus`, the click handler, "Generated N atoms from …", "Kept existing workspace.", the Enter key, and the configure / DOMContentLoaded / runtime.register block. The wording has already drifted (smiles names the backend, name does not).

**The door that exists:** `panel-deps.js` already owns `generate` and `generateText`, the "one generate chain".

**Smallest fix:** `panel-deps` grows a `wire(ids, generate, words)` and a `boot(name, api)`, and each panel keeps only its request.

### F24 · The sidebar-pick-to-Load readout written twice, in different words
Low-medium. Read both sides.

**The two copies:**
- `lib/structure-card.js:120-136` `refreshLoadButton`: "Selected: x" / "Loaded: x" / "(not loadable — …)".
- `modify/selection-bootstrap.js:131-165` (a candidate listener machine with one subscriber) and `:238-257` `_refreshLoadUI`: "Picked: x" / "Loaded: x".

**What goes wrong:** the same state is "Selected" on Structure optimization and Spectrum, and "Picked" on Molbuilder.

**Smallest fix:** the readout is one helper both use. The Molbuilder tab's load path may stay its own.

### F25 · Dead members and dead fallbacks
Low. Read; callers searched across `molbuilder/` and `tests/`.

- `modify/structure/page.js:333-335`, `:422` `markDirtyAfterModification` is a no-op ("Nothing to do"). Only `tests/test_structure_page_js.py:315` calls it.
- `page.js:407-412`, `:430` `onCanvasChange` has no caller anywhere.
- `lib/form-schema.js:896` and `:1031-1034`: the fallback `["x","y","z"]` for a triple's labels.
  - The server always sends `labels = list(kmesh.AXES)` for a triple (`_shared.py:693`).
  - `makeTriple` (`:333`) reads `f.labels` with no fallback.
  - So the fallback is unreachable, and a copy of `kmesh.AXES`.

**Smallest fix:** remove all three, and the test row that pins the no-op (T5).

### F26 · Token values with hard-coded fallbacks
Low; the cost of the drift is plausible.

**Page:**
- `lib/projects/checkpoint.js:466-491`: `cssVar("--ps-accent", "#4f7ab8")` and four more colours, plus a font.
- `lib/system-load-monitor.js:83`: `if (!stroke) stroke = "#3a9";`.

**The rule:** `lib/plot-theme.js:266-273`: "a token that is not there is said, never covered by a second copy of its value". Also `ui-contract.md` § 2, one palette file.

**Smallest fix:** read the token and say when it is missing, as plot-theme does.

### F27 · The keyword badge classifies an item by its text
Low; plausible.

**Page:** `lib/form-schema.js:149`: `if (key.startsWith("(molbuilder"))` marks the badge "molbuilder-only knob".

**Source:**
- The marker is free text inside the catalogue's `engine_key` (32 items in `molbuilder/data/catalogue.template.toml`).
- The catalogue carries a `kind` per item (`engine` / `deck` / `produce` / `wrapper`).

**Drift:** an `engine_key` worded without the leading "(molbuilder" is badged as an engine keyword.

**Smallest fix:** the field carries the flag (`_shared._item_to_field`), and the page reads it.

### F28 · Task setup words the notify policy twice
Low. Read both sides.

**Two composers in one file:**
- `task-setup/viewer.js:4538-4566` `paintNotifyNote`: "Reports each SCF convergence, every N h …".
- `:4853-4861` `paintHeads`: "each SCF converged · every N h · every channel".

The two wordings already differ.

**Smallest fix:** one function, which the head shortens.

---

## C. Tests

### T1 · Hand-typed viewer, page and door stand-ins beside the one MolView node door
Medium. Read both harnesses and the support door.

**The stand-ins:**
- `tests/test_structure_page_js.py:31-133`: `_mkFakeCanvas` and `_mkFakeModal`, with handcrafted structures such as `{title: "eth", elements: ["C"], positions: [[0, 0, 0]]}`.
- `tests/test_structure_save_js.py:41-96`: `fakeModel`, `fakeDoor` and `fakePage`.
  - `fakePage` re-implements `page.js`'s `markSavedTo` and `getCanvasSnapshot`.
  - `fakeDoor` re-implements `molview-doors.js`'s answer.
- Both start Node themselves (`:140-180`, `:33-110`).

**The door:** `tests/support/molview_page.py:1-22`, "ONE door for every MolView test under node". It runs the real `createModel` and `mount`, answered by the served routes.

**Silent failure:** change `getCanvasSnapshot`'s fields or the door's answer, and these tests pass against their copies.

**Known:** `docs/process/testing.md` (*It is the harness to use…*) lists both among the seven files that still start Node themselves. The hand-typed stand-ins are the part not named there.

**Smallest fix:** the page and the save panel are driven on a real model through `molview_page`, with the real `page.js` and doors loaded.

### T2 · The builder-panel harness is copied per file
Medium-low. Read all three.

- `tests/test_structure_dna_js.py:37-61` (`WIRED` and `_run`) and `tests/test_structure_peptide_js.py:33-57` are the same text, the variable name and the element id aside.
- `tests/test_structure_smiles_js.py:42-53` `_run(…, panel=…)` is already generic over the panel.

**Smallest fix:** one panel harness, the generic `_run` with the wiring prelude, in `tests/support/`, used by all three.

### T3 · Source surgery by hand beside the shared lifter, on handcrafted records
Medium-low. Read whole.

**The file:** `tests/test_task_setup_save_bar_js.py`:
- `:32-40` cuts `viewer.js` by two comment strings (`src.index("/** The records the editor holds")`, `src.index("/* ---------- loading a folder ---------- */")`) and rewrites code with `body.replace("await ensureEditor()", "_cm")`.
- `:53-75` stands in `$`, `refreshSave`, `saidSave`, `_cm` and `_docs`, and starts its own Node.
- `:92-108` feeds `'{"a":1}'` / `'{"r":1}'` as the two records.

**The shared door:** `tests/_node_esm.py:56` `lifted(module, start, end)`, which `test_task_setup_notify_js.py:30-80` uses with records from `support.road.described` (`:186-201`).

**Smallest fix:** `lifted` and `run_node`, with buffers holding a described calculation's records.

### T4 · Tests pin the panels' copies of the builders' rules (F1)
Low. Read in part.

- `tests/test_structure_peptide_js.py` ("ACBDEF" refused before any request).
- `tests/test_structure_dna_js.py` (its docstring: "what it refuses before any request (… a letter not ACGT, Z-DNA …)").
- `tests/test_structure_smiles_js.py:334-339` (thymine in the RNA panel: `body is None`).

These hold the duplicate in place. With F1 fixed they become rows of what the server refuses, its words shown.

### T5 · A test pins a no-op
Low. Read.

- `tests/test_structure_page_js.py:303-325` calls `page.markDirtyAfterModification()` (F25) and asserts that it touched nothing.
- The row tests the absence of behaviour in a member no production code calls.
- **Fix:** remove it along with the member.

---

## Not findings (checked and left out)

- `lib/workspace/dispatcher.js` names its own files (`ws-<tab>-<tag>`, `-draft`). That is the module's own fact (`workspace.md` § 4), not a second answer.
- `lib/task-handover.js` `HANDOVER.needs` checks that a structure is present before the send. It is the door's own guard and says its own words.
- `modify/slab-panel.js`'s stacking-period modulo and walk labels read the served `stacking_period`. The arithmetic is the page's own, and its comment says so.
- `documents/page.js` rewrites root-README links. This is a reader-side presentation of the docs tree, and round 1 already covered the page.
- MolView and VibrationView internals were left out, as sealed.
