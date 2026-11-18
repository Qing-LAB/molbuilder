# F12 review, round 1 — slice J1: the describe pages

Checkout `the checkout` at f578482db (2.77.2). Read-only; nothing in the checkout was touched.

**Chapters read end to end:** `docs/web/tabs.md`, `task-setup.md`, `projects.md`, `workspace.md`, `this-machine.md`, `ui-contract.md`.

**Code read whole:**
- This machine, Documents and JupyterLab, with their templates and server routes (`notify_setup.py`, `docs.py`).
- `lib/*.js`: `page-busy`, `dialog`, `warning-modal`, `server`, `status`, `dom`, `time-ago`, `page-prefs`, `tab-strip`, `one-stop-list`, `codemirror-load`, `markdown-render`, `app-notifications`, `app-reload`, `molbuilder-runtime`.
- `_page_head.html`, `_app_header.html`, `tabs.py`.

The rest of the slice went to three parallel readers, each reading its code whole against the same chapters:
- Task setup;
- the projects sidebar with the workspace;
- the Molbuilder tab with Structure optimization.

Their headline findings were checked again in the code before they went in here:
- the symlink delete;
- the engine-list cache;
- the late template values;
- the empty slab box;
- the save readout.

The known item at `task-setup/viewer.js` ~1743 (structure-pair names left from the last folder) is not repeated.

Paths below are under `molbuilder/web/static/` unless they start with `molbuilder/` or `docs/`.

**Totals: 24 defects, 15 untrue sentences.**

---

## Most severe five

1. **Deleting a symlink in the sidebar deletes what it points to.** P-D1.
2. **A PySCF hand-over can be offered SIESTA's settings.** This happens when it is opened after a folder of another engine or kind. TS-D1.
3. **Task setup can show another folder's template values.** A late folder answer overwrites the template values of the folder now shown. TS-D2.
4. **Structure optimization keeps only frame 1 of a frame set across a reload.** After the reload, Send writes frame 1, not the frame the person chose. SO-D1.
5. **This machine says "nothing is being sent" when only one channel is skipped.** The other channels are still sent, and the skipped row is not marked. TM-D1.

---

## A. This machine

### TM-D1 · One skipped channel is reported as the whole file being unusable, and its row is not marked
Defect. Read-and-traced.

- **Where:** `this-machine/page.js:117-120` (`paintChannels`) and `:44-98` (`rowFor`).
- **Rule:** `this-machine.md` § 3: "A channel already in the file that fails the rule is listed, marked as one the monitor skips, with its fault."
- **What the server sends:**
  - Each row carries `usable` (`molbuilder/web/blueprints/notify_setup.py:187`).
  - `problem` is filled for a partially skipped file too: `"the monitor will skip 'x' — <fault>.  Nothing is sent to a skipped channel…"` (`notify_setup.py:249-255`).
- **What the page does:**
  - `if (d.problem) { say("tm-file", "There is a file at " + d.path + " but it cannot be used: " + d.problem + " — nothing is being sent.", "error"); }`
  - `rowFor` never reads `c.usable`.
- **Failure:**
  - **Input:** a file holds `slack` (a valid `https://hooks.slack.com/…`) and `lab` (an `http://` address with kind `slack`).
  - **Shown:** in red, "There is a file at … but it cannot be used: the monitor will skip 'lab' — … — nothing is being sent."
  - **Actually:** `slack` receives every report, and the `lab` row looks like a working one.

### TM-D2 · The server's note about the reports list is overwritten by "Nothing received yet."
Defect. Read-and-traced.

- **Where:** `this-machine/page.js:488-491` then `:466-467`.
- **Rule:**
  - `this-machine.md` § 4.1: "the reports this server received for you … Read through one door, `GET /api/notify/reports`".
  - `ui-contract.md` § 12.1: every place a page shows a sentence of the server's draws it.
- **What the code does:**
  - `loadReports` says `d.note || "Nothing received yet."`.
  - It then calls `paintReports()`, which, with no reports, says `"Nothing received yet."` again.
  - So the note is replaced at once. The server's notes are `notify_setup.py:584-594`:
    - "no user to read reports for: sign in, or name one with ?user=<name>"
    - "this server holds no key for 'alice@x' and no reports of theirs; the users with a key: alice"
- **Failure:**
  - **Case 1:** with sign-in disabled, the card always says "Nothing received yet.", even when reports exist for keyed users.
  - **Case 2:** a person signed in as `alice@x.edu` whose key was issued to `alice` sees "Nothing received yet." instead of the sentence naming the users with a key.

### TM-D3 · The filter count stays after the filter is cleared
Defect, low. Read-and-traced.

- **Where:** `this-machine/page.js:466-470`.
- **Rule:** `this-machine.md` § 4.1: "A filter narrows by calculation, machine or event".
- **What the code does:** `if (!reports.length) {…} else if (q) { say(…, shown + " of " + reports.length + " shown.") }`. There is no branch for an empty filter with reports present.
- **Failure:** type "relax" and the line says "2 of 40 shown." Clear the box and all 40 are listed, but the line still says "2 of 40 shown." until the next 30-s poll.

### TM-U1 · Untrue: "the kind already says it: a channel with a key is a listener and one without is a webhook"
Read-and-traced.

- **Sentence:**
  - `this-machine.md` § 3: "*Whether a key is stored* is not a separate column, because the kind already says it: a channel with a key is a listener and one without is a webhook".
  - § 2: "§ 3's answer is derived from whether a key is stored".
- **Code:** `monitor.channel_kind` (`molbuilder/monitor.py:1595-1605`) takes the declared `kind`, else the host. A key plays no part.
  - The page never shows the `has_key` field the server sends (`notify_setup.py:179`).
  - `save_channel` stores a `molbuilder`-kind channel with no key (`notify_setup.py:359-361`).
- **Consequence:** a new listener channel saved with the key box empty shows as kind "molbuilder", with nothing saying no key is stored. Every report then goes unsigned and is refused (404) until someone presses Test.

### TM-U2 · Untrue: "The key is issued to the email you sign in with"
Low. Read-and-traced.

- **Sentence:** `this-machine.md` § 3.
- **Code:**
  - The Issue form takes any id typed (`this-machine/page.js:268`, `notify_setup.py:617-634`).
  - Only the placeholder suggests the email; nothing fills or checks it.
- **Consequence:** a key issued to another spelling is not the name the reports list reads by default. See TM-D2.

### UI-U1 · Untrue: "the token rule bans inline colors"
Low. Read-and-traced.

- **Sentence:** `ui-contract.md` § 7. § 2 also says "never write a raw palette color".
- **Code:** `this-machine/page.js:420`: `bar.style.background = "#" + (… c.color …)`. This is the report card's state colour, written inline.
- **Note:** this is the only inline colour write in the slice. `this-machine.md` § 4.1 asks for "the state's colour", so one of the two chapters needs to name it as an exception.

---

## B. Documents

### DOC-D1 · A slow document answer replaces the one opened after it; the highlighted link and the URL disagree with what is shown
Defect. Read-and-traced; the race needs a slow answer.

- **Where:** `documents/page.js:98-166` (`openDoc`).
- **Rule:** `tabs.md` § 5.4: "Opening a doc writes `?doc=...` back into the URL".
- **What the code does:**
  - There is no sequence guard and no abort.
  - Each call sets the link's highlight at once, renders whatever answer arrives, and then calls `replaceState` with its own path.
- **Failure:**
  - **Steps:** click *Routes* under Indexes (composed from the code on each read, so it is slow), then at once click `web/tabs.md`.
  - **Order of answers:** `tabs.md` renders first, then the Routes answer lands and replaces it.
  - **Result:** the tree highlights `tabs.md`, the pane shows Routes, and the URL says `?doc=index/routes.md`.

### DOC-D2 · The load that discovers a new sub-document hides the parent's other sub-documents
Defect, low. Read-and-traced.

- **Where:** `molbuilder/web/blueprints/docs.py:284-290`.
- **Rule:** `tabs.md` § 5.3: "A new file named `model/structure-symmetry.md` would be discovered and nested under `model/structure.md`".
- **What the code does:**
  ```python
  if stem in pending_subs:
      c = dict(c); c["children"] = pending_subs.pop(stem)
  ```
  This replaces the parent's listed children rather than adding to them. The write-back at `:355-360` appends correctly.
- **Failure:** add `docs/model/structure-symmetry.md` and open Documents. Under `model/structure.md` only `structure-symmetry.md` shows; `structure-periodicity`, `-annotations` and `-molstruct` are gone. They come back on the next load.

### DOC-D3 · A URL `?doc=../LICENSE` opens nothing
Defect, low. Read-and-traced.

- **Where:** `documents/page.js:279`: `if (initial && _linksByPath[initial])`.
- **Rule:** `tabs.md` § 5.4: "loading a URL with `?doc=...` opens that doc".
- **What the code does:** it opens only paths that are links in the tree. LICENSE is not in the tree, yet opening it from the README writes `?doc=..%2FLICENSE` into the URL (`:161-165`).
- **Failure:** open LICENSE from the README, then reload or share the link. The empty Documents page shows.

### DOC-U1 · Untrue: "The Archive folder's children are left empty — the server fills them in"
Read-and-traced; confirmed on disk.

- **Sentence:** `tabs.md` § 5.2: "The Archive folder's children are left empty — the server fills them in automatically by scanning the `archive/` directory". The JSON example shows `"children": []`.
- **Code:**
  - `docs.py:337-339` assigns the scanned list into the toc node itself (`node["children"] = archive_children`).
  - `:365-375` writes `toc.json` whenever it changed.
- **Seen on disk:** `docs/toc.json`'s Archive holds 85 entries, each with `label` and `subtitle`.

---

## C. Task setup

### TS-D1 · A hand-over opened after a folder of another engine or kind is offered the previous folder's settings
Defect, medium. Read-and-traced.

- **Where:**
  - `task-setup/viewer.js:2247-2260` (columns), `:2321-2326` (sweepable settings), `:2602-2604` (presets).
  - The hand-over path reaches them through `:1791`, then `:4704-4707`.
- **Rule:** `task-setup.md` § 6.1: "One fetch per engine key at a time: a second caller for the same key awaits the fetch in flight rather than reading the previous folder's list. So a PySCF hand-over is never offered SIESTA's settings".
- **What the code does:**
  ```js
  if (_cols && _colsKey === key) { _fillMeta(_cols); return _cols; }
  if (_colsInflight[key]) return _colsInflight[key];
  _colsKey = key;
  ```
  - The key is set when the fetch starts, but `_cols` still holds the old key's list. A second caller for the new key passes the cache check and gets the old list.
  - `_sweep` and `_presets` work the same way.
  - Line 1791's `refreshPickers()` is not awaited. `proposeHandover` then asks again (4705-4706) and draws the run cards and the stage table from the old lists.
  - The late fresh answer redraws nothing: it is out of sequence and returns at 2432.
- **Failure:** open a SIESTA optimization folder, then a PySCF hand-over folder. The run cards show MPI ranks, OMP threads and Diagonalizer, and the column picker lists SIESTA's columns, until a later edit redraws the page.

### TS-D2 · A late folder answer overwrites the template values of the folder now shown
Defect, medium. Read-and-traced.

- **Where:** `task-setup/viewer.js:2497-2508` (`loadFolderAnswer`), compared with the check at `:1643-1647`.
- **Rule:**
  - `task-setup.md` § 2.1: "an answer that lands for a folder the page has left is discarded".
  - § 5.1: an empty cell shows "the folder's template value".
- **What the code does:** `if (t.ok) _tmpl = {name: t.name, values: …}` runs inside `loadFolderAnswer`, before `readFolder`'s `if (dir !== _dir) return;`.
- **Failure:**
  1. Click folder A, whose answer is slow (a calculation with runs).
  2. Click folder B at once.
  3. B lands, then A lands and sets `_tmpl` to A's template.
- **Result:** B's empty cells, the picker's "now: …" and the hovers ("This job (A.template.toml): …") show A's values until the next read.

### TS-D3 · When a folder answer fails, the rail's "What gets written" keeps the last folder's files
Defect, low to medium. Read-and-traced.

- **Where:** `task-setup/viewer.js:1651-1653` → `refuseFolder` (`:2488-2494`). Only the no-folder branch (`:1624-1634`) resets the `#ts-file-list` rows.
- **Rule:** `task-setup.md` § 2.1: "a folder opened anew — … a refused one — draws every card from it, and a card with nothing to show for that folder is hidden".
- **Failure:** with folder A open, pick a folder whose answer fails (404, the server unreachable, or signed out).
  - The state box says the folder could not be read.
  - The rail card, open by default, still lists A's files: "already here — saving would update it", "here — 50 atoms", and "5 of the 5 listed".
- **Note:** this is related to the known ~1743 item but is a different code path.

### TS-D4 · A settings list that failed to load is drawn as an empty list and never asked for again
Defect, low to medium. Read-and-traced.

- **Where:** `task-setup/viewer.js:2266-2272`, `:2338-2346` and `:2611-2614` set `items = []` on failure. `fillPicker` (`:2388-2391`) draws it.
- **Rule:** `ui-contract.md` § 12: "never paints a failed answer as data: a refusal is not … an empty list, or a menu left empty".
- **What the user sees after one failed fetch:**
  - The column picker says "every parameter is already a column".
  - "+ Add setting" says "every setting is already listed".
  - The preset menu disappears.
- **Why it lasts:** `[]` counts as true, so the cache checks keep returning it for that engine and kind until the page is reloaded.

### TS-D5 · Two failures are dropped without a word
Defect, low. Read-and-traced.

- **Continue from**, `task-setup/viewer.js:3166-3169`:
  - The code: `if (cf && !cf.error) { … }`. The server sends a per-stage `continue_from: {error}` on purpose (`molbuilder/web/blueprints/build.py:2180-2182`), and the page never shows it.
  - The rule: `task-setup.md` § 11: "or, while the launch would refuse it, why".
- **Memory need**, `:2784`:
  - The code: `if (!answer.ok) return;`. Every stage's memory line stays blank.
  - The rule: § 7.1 and § 6.2b: "A check that cannot run says so … never a blank".

### TS-U1 · Untrue: "the loaders' own SIESTA fallback is reachable only on a page with neither"
`task-setup.md` § 6.1.

- There is no SIESTA fallback. With no engine or kind, the loaders return `[]` (`viewer.js:2245`, `:2319`, `:2600`).

### TS-U2 · Untrue: the run card's "+ Add setting" is not as § 6.2b describes
`task-setup.md` § 6.2b.

- **Sentence:** "The first point is typed before the row exists … *+ Add setting* with an empty box is refused … Each row's class — *chosen* … *machine* … *measured* — is the grid door's word".
- **Code:**
  - The run card's adder (`viewer.js:1140-1161`) has no value box; it adds an empty row. The same section says so later: "a row added on the card and not yet filled has one [×]".
  - Run-card rows are classed chosen, inherited or unstated by the page itself (`:1121-1124`).
- **What the sentence actually describes:** the *What to measure* card's adder (`:5085-5117`) and its row classes (`:1200`, `_rowKinds`).

### TS-U3 · Untrue: the identity facts include "the label every emitted file is stemmed on"
`task-setup.md` § 3.

- `renderCameOver` (`viewer.js:2110-2155`) draws no label row.
- The label is derived (`task.py:481`), not stored in `task.json`. It shows only inside the template's file name in the rail.

### TS-U4 · Untrue: a card's change is refused while the editor holds text that does not parse
`task-setup.md` § 9a: "while the editor holds something that is not valid JSON … a card's change is refused, said at the top of the page".

- `patchDoc` (`viewer.js:4300-4316`) is used by the queue and notify cards. It parses only the `job-request.json` buffer.
- So with `task.json`'s text mid-edit and not valid, a change to the queue, wall, memory or notify card is still written into `job-request.json`, with no refusal.
- Only `syncFromModel` (`:1985`) checks both files.

---

## D. Projects sidebar and workspace

**Workspace:** `lib/workspace/dispatcher.js` matches `workspace.md` as read; there is no finding.

### P-D1 · Delete, Rename, Move and Copy on a symlink act on its target — data loss
Defect, high. Read-and-traced.

- **Where:**
  - Server: `molbuilder/web/blueprints/files.py:1764` (delete) and `:1498` (rename). Move and copy go the same way.
  - Every path passes `_resolve_within_roots` → `projects.contain`, which calls `Path(raw).expanduser().resolve()` (`molbuilder/projects.py:288`).
  - Listing: `files.py:398-416`. Sidebar: `lib/projects/list.js:548-559`.
- **Rule:**
  - `projects.md` § 4: "one row per item, with folders, symlinks, and files marked differently".
  - § 4.1: "The fence resolves every path it is given, a link included".
  - Nothing says deleting a link removes what it points to.
- **What the code does:**
  - `_entry_dict` uses `p.stat()` and `p.is_dir()`, which follow the link. A link to a folder is listed as `"directory"`.
  - The sidebar asks "Delete 'calc'? This cannot be undone." and sends `recursive=true`.
  - The server runs `shutil.rmtree(resolved)` on the resolved target.
- **Failure:**
  - **Setup:** `projects/P/user/calc -> projects/Q/optimization/calc1`.
  - **Delete "calc":** it removes `Q/optimization/calc1` and all its contents, and leaves the link dangling.
  - **Rename:** renames the target. **Move** on a link to an `.xyz`: moves the real pair.
  - **A broken link** (listed as `"other"`) cannot be deleted at all: the answer is 404.

### P-D2 · The run-history button is not re-read when its own folder is announced while the history view is closed
Defect, medium. Read-and-traced.

- **Where:** `lib/projects/checkpoint.js:931-936`:
  ```js
  if (dir && dir === …folderKey(_state.currentDir) && _state.open) _refresh();
  ```
  The other path, `onDirectoryChange`, returns early for the same folder (`:184`).
- **Rule:**
  - `projects.md` § 2: "Whoever shows a folder follows it: … the run history reads it again".
  - § 4: the button's colour is honest.
- **Failure:**
  - A hand-over folder has no history, so the button is hidden. Task setup Save writes `task.json`, takes a first state and announces the folder. The button stays hidden until the folder is left and entered again.
  - An amber "N changed" button stays amber after a Save or a Prep has taken a state.

### P-D3 · A failed history read on entering a folder shows nothing
Defect, low to medium. Read-and-traced.

- **Where:** `lib/projects/checkpoint.js:189-190` (`_hide(); _refresh();`), `:608-611` and `:292-297`.
- **Rule:** `projects.md` § 4: "red = error".
- **What the code does:** `_renderError` paints `data-state="error"` on a button that `_hide()` left hidden, and writes its words into the hidden `#ps-checkpoint`.
- **Failure:** a calculation folder with a damaged `.git` makes `/api/checkpoint/state` answer 500. No button and no message appear, so the folder looks like it is not a calculation.

### P-D4 · The sidebar's own ⋯ → View opens nothing when the browser will not keep the selection
Defect, low. Read-and-traced.

- **Where:** `lib/projects/list.js:449-456`: `setShared(currentPath, fullPath); … return showPreview();`. No path is passed.
- **Rule:** `projects.md` § 5, `showPreview`: "passes `sel.file`: `setShared` publishes its payload even when the browser refuses to keep it, so the pick can be empty".
- **Failure:** with site data blocked, `getCurrentFile()` is "", so `preview.js:414-419` only warns in the console. View does nothing.

### P-D5 · A preview that loads late shows the earlier file's text under the new file's title
Defect, low. Read-and-traced; needs a timing.

- **Where:** `lib/projects/preview.js:528-582` (`_loadBulk`). It does not check `_state.path === path` after its await. The chunked reader does check (`:648`).
- **Rule:** `projects.md` § 4: the preview is the view of the file opened.
- **Failure:**
  1. Open a large file and press Esc while it reads.
  2. Open a small one.
  3. The large file's text lands last and fills the editor under the small file's title.
- **Limit of the harm:** a Save is refused by the mtime check unless the two mtimes match.

### P-D6 · The name questions accept names the server then refuses
Defect, low. Read-and-traced.

- **Where:** `lib/projects/dialogs.js:22-30` (`_nameRefusal`), used by `chooseSavePath`, New project and New folder.
- **Rule:** `ui-contract.md` § 11: `askFields` has "the answer's button enabled only while the answer can stand".
- **What the code does:** it refuses only `/`, `\`, `.` and `..`. The server refuses more:
  - `_writable`, by `^[A-Za-z0-9][A-Za-z0-9._-]*$`;
  - `validate_folder_name`.
- **Failure:** in Save to project, type "water dimer". Both questions are answered, then `/api/structure/save` refuses "contains unsupported characters", and the person starts again.

### P-U1 · Untrue: "folders, symlinks, and files marked differently"
`projects.md` § 4.

- `files.py:408-416` classifies a link by its target, and `list.js:614-618` draws "→" only for `kind === "symlink"`.
- So a link to a file reads as a file, a link to a folder as a folder, and a broken link is `"other"` drawn with the file glyph.

### P-U2 · Untrue: the Download confirm shows the archive's "exact size"
`projects.md` § 4: "the archive's name, its file count and its exact size".

- `lib/projects/mutation-bar.js:107` shows `size_words`, which is one decimal in binary units ("49.2 GiB").
- The exact `bytes` the route serves is not shown.

### P-U3 · Untrue: every file call answers the refusal's fields
`projects.md` § 5: every file call answers `{ok: false, error, …the refusal's fields}` (`actual_mtime`, `needsOverwrite`, `replaces`).

- `writeFile` (`lib/projects/state.js:483-485`, `:509-512`) keeps only `error`, `actual_mtime` and `aborted`. It drops `needsOverwrite`, `replaces`, `status` and `signedOut`.
- `readCurrentFile` (`:400`) answers `{ok:false, error}` only.
- No current caller depends on the dropped fields.

### P-U4 · Untrue: the upload notification is "a warning when there are notices"
`projects.md` § 5, the `upload` row.

- `mutation-bar.js:221-224` takes the most severe notice's tone (`validationFindings.outcome`), so info-only notices stay info.
- This matches `tabs.md` § 6 and validation R4, so the `projects.md` sentence is the wrong one.

---

## E. Molbuilder and Structure optimization

### SO-D1 · A frame set survives a reload as frame 1 alone; Send then writes frame 1
Defect, medium. Read-and-traced.

- **Where:** `lib/structure-card.js:151-157` (`keep`), with `:171-176`, `:195` and `:221-222`.
- **Rule:** `tabs.md` § 3: "A reload keeps the structure … the structure the viewer shows as the tab's own draft … saved when the tab loads one". The same section adds: "the structure the tab sends is the one its readout names".
- **What the code does:**
  - `keep()` saves `data().exportFile().structure`. With no range, that is the displayed frame only (`lib/molview/model-jobs.js:723-724`).
  - It runs only right after the load, when frame 0 is shown.
- **Failure:**
  - **Steps:** load `traj.xyz` (9 frames), scrub to frame 7, reload.
  - **On screen:** "Restored traj.xyz — N atoms." with frame 1 only. A double-click on the same file does nothing, because `f === shown && !again` returns null.
  - **Send:** before the reload it wrote frame 7 and said "— the structure at frame 7 of 9". After the reload it writes frame 1 and names no frame.

### SO-D2 · After a failed load, the readout names the failed file while the canvas and Send keep the earlier structure
Defect, low. Read-and-traced.

- **Where:** `lib/structure-card.js:113-118` (`failed`) and `:217-219`.
- **Rule:** `tabs.md` § 3: "the structure the tab sends is the one its readout names".
- **What the code does:** `failed()` writes "load failed: b.xyz" and "—" for atoms and formula. `installMolecule` throws before `put` (`model-jobs.js:576-579`), so the viewer keeps `a.xyz`, and `card.structure()` returns it.
- **Failure:** load `a.xyz`, then `b.xyz` fails. A Send writes `a.xyz` as `<label>.source.xyz`, and the chemistry card and the live check go on describing `a.xyz`.

### MB-D1 · Add slab with an empty lattice-constant box still sends a request
Defect, low to medium. Read-and-traced.

- **Where:** `modify/slab-panel.js:102-106` (`num`) and `:434-444`.
- **Rule:** `tabs.md` § 2: "**Add slab with the box empty sends nothing** and says what to do — type the constant or pick a reference".
- **What the code does:** `const v = el ? Number(el.value) : NaN; return Number.isFinite(v) ? v : dflt;`. The box is `<input type="number" id="slab-a">` (`molbuilder/web/templates/modify.html:646`), and `Number("") === 0`. So `if (!Number.isFinite(a))` never fires and `lattice_constant: 0` is posted.
- **Failure:** clear the box and press Add slab. The server refuses "'lattice_constant' must be > 0 Å" on the edit line. The panel's own sentence ("The lattice constant box is empty … Nothing was built.") never appears.

### MB-D2 · A Load whose add fails says nothing
Defect, low to medium. Read-and-traced.

- **Where:** `modify/selection-bootstrap.js:353-356`, `:385` and `:217-219`.
- **Rule:** `ui-contract.md` § 12: a failed answer is said, never painted as data or nothing. The load gate's answer in `tabs.md` § 2 is `{ok:false, error}`.
- **What the code does:**
  - `if (!gate || gate.ok !== true) return;`.
  - `applyOp("append")` returns null while another edit runs (`model-jobs.js:856`), and throws on a server refusal or a lost connection.
  - The callers (`_loadBtn` click, `onCommit`) have no catch.
- **Failure:** with a structure open, a Load whose append is refused, or meets a dropped connection, changes nothing on screen. The status line keeps its old text, and the error ends as an unhandled rejection.

### MB-D3 · The save readout says "Not saved yet" for a structure loaded from a file, and never "Unsaved" after an edit
Defect, low. Read-and-traced.

- **Where:** `modify/structure/save.js:208-221` and `modify/structure/page.js:371-386`.
- **Rule:** `tabs.md` § 6: "*Not saved yet* for a structure that has never been written"; "*Unsaved — Target: …* once you edit again".
- **What the code does:**
  - `markLoadedFrom(path)` sets `_lastSavedTo = null`.
  - The readout reads only `lastSaveTo`: `if (path) … else if (hasContent) readout.textContent = "Not saved yet"`.
  - It never reads `loadedFrom` or `isDirty` in that branch.
- **Failure:** load `water.xyz` unchanged and the readout says "Not saved yet". After an edit it still says the same.

### MB-D4 · The Cell op-tab reached by the keyboard does not turn on measuring
Defect, low. Read-and-traced.

- **Where:** `modify/periodicity.js:776-782`: a `click` listener on `#optab-btn-cell`. The arrow, Home and End keys select through `tab-strip.select()` (`lib/tab-strip.js:99-110`) and fire no click.
- **Rule:** `ui-contract.md` § 4.1: "Every action a click performs has a keyboard path".
- **Failure:** reach Cell with the arrow keys. Measuring stays off, and the pick buttons stay disabled. The Transform tab uses `onChange` and does not have this problem.

### SO-D3 · The first live check can describe default parameters rather than the restored forms
Defect, low. Plausible: the race was not measured.

- **Where:** `structure-optimization/viewer.js:113-118` and `:349-350`.
- **Rule:** `tabs.md` § 3: "Live validation runs against `/api/build/preflight` as you edit".
- **What the code does:** if the restored structure lands before the schemas, `collectParams` returns `{}` and the preflight checks defaults. After the forms are drawn and restored it is not asked again, because `wirePreflightListeners` is attached after `_forms.restore()`.
- **Effect:** the findings, and the live findings Send keeps beside its own, describe defaults until the person edits a field.
- **What is missing:** whether the schema fetch loses that race in practice.

### MB-U1 · Untrue: "It is the one controller that fetches a route directly: `GET /api/modify/meta`"
`tabs.md` § 2.

- `slab-panel.js` also asks `/api/modify/spacings` (`:290`) and `/api/modify/lattice-from-run` (`:382`).
- `modify/selection-bootstrap.js:312` asks `/api/build/load`.

### MB-U2 · Untrue: `structure-optimization/viewer.js` is a hybrid that imports MolView's `mount`
`tabs.md` § 8 lists it as "hybrid — top-level `import`s (MolView's `mount`)", one of "the page controllers that mount MolView".

- `structure-optimization/viewer.js:25-26` imports only `attachStructureCard` and `ask`.
- `lib/structure-card.js:84` mounts the viewer.

### MB-U3 · Untrue: the load gate receives "the record of what made it"
`tabs.md` § 2: "then the page's load gate (`structurePage.loadIntoCanvas`, with the record of what made it)".

- The panels pass `{kind, generator_input}` (`modify/structure/panel-deps.js:134-135`, `dna.js:181-183`).
- `loadIntoCanvas` reads only `source.file` (`modify/structure/page.js:113`) and drops the rest.

---

## Checked and consistent (not findings)

**The shell**
- `TABS` (nine tabs), the `/` redirect, and which pages carry the sidebar (tabs.md § 1).
- `_page_head.html` / `_app_header.html` load order and contents (ui-contract.md § 1).

**Shared components**
- The page fence: claim and throw, `run` composing in one claim, Cancel, the keyboard fence, focus given back (§ 10).
- The dialog component and the one confirm (§ 11). No `alert`, `confirm` or `prompt`, and no stray HTML writes (§ 7).
- The one reader's table, and only `lib/server.js` calling `fetch` in the slice (§ 12). `codeWords` (§ 12.1).
- The tab strip and the one-stop list (§ 4.1). `timeAgo` / `timeSpan` (§ 4).
- The CodeMirror suffix map and the vendored modes (task-setup.md § 9a).

**Documents**
- The `..` refusal (tabs.md § 5.4).
- The Indexes folder (§ 5.6).
- The theme preference (§ 5.5).

**This machine**
- Masking, merge on save, the retired shape cleared, the key cleared after save, the issued key shown once, arm-then-remove.

**JupyterLab**
- The state table matches `jupyter.md` § 4.0.

**Workspace and Molbuilder**
- The workspace's tag spelling, per-tab ids, the two ordering rules, and failures reported through `onPersistError`.
- Molbuilder's add-or-clear question, the frame chooser, the generator chain under the fence, the `modify:panel` note, and restore-first (workspace.md § 7).

**Task setup**
- Save held until the answer lands, `read_as`, 409 said.
- Foreign changes kept and said.
- Preview gating Prep, and `--target` dropped for a set machine.
- Column removal arms on the first click and acts on the second; the preset adds its columns first.
- The fence on every card edit.
- Folding as a checklist, one rail card open at a time.
- The notify block `["*"]` / `[]`.
