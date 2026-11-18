# Verdicts — r1-J1 (the describe pages)

| id | verdict | class | road | one-line reason |
|---|---|---|---|---|
| TM-D1 | CONFIRMED | M | hand-edited channels file only: `save_channel` refuses any address `load_channels` would skip; the contract allows hand-kept files | `paintChannels` prints "cannot be used … — nothing is being sent" in red for any `problem`, which `_file_note` also sets when only one channel is skipped. `rowFor` never reads `c.usable`. |
| TM-D2 | CONFIRMED | M | own road: sign-in off, or a key issued to an id other than the sign-in email | `loadReports` writes `d.note`, then `paintReports()` with no reports overwrites it at once with "Nothing received yet." |
| TM-D3 | CONFIRMED | M | own road | `paintReports` has no branch for an empty filter with reports present, so "2 of 40 shown." stays until the next 30-s `loadReports`. Low. |
| TM-U1 | CONFIRMED | T | own road: a listener saved with the key box empty | `channel_kind` reads the declared kind, else the host; a key plays no part. `_row`'s own docstring says `has_key` is a separate question. Fix the § 2 and § 3 sentences. The page should also show `has_key` for a molbuilder channel, since a key-less listener looks like a working one. |
| TM-U2 | CONFIRMED | T | own road | The Issue form takes any id typed; only the placeholder says "the email you sign in with". § 4 also has keys issued for colleagues. The sentence should say the id typed is the name the report log reads. Low. |
| UI-U1 | CONFIRMED | T | n/a | `this-machine/page.js:420` writes the report card's state colour inline. The colour is data from the server's layout, not palette, and the code is right. § 2/§ 7 should name "a colour the data carries" as an exception. Low. |
| DOC-D1 | CONFIRMED | N | own road, a race: a slow index answer, then a quick second click | `openDoc` has no sequence check. A late answer renders and calls `replaceState`, so the highlight is on tabs.md while the pane and the URL show Routes. |
| DOC-D2 | CONFIRMED | N | a developer adding a docs file | Reproduced on a scratch copy. On the first load only `structure-symmetry.md` is listed under `structure.md`, because `pending_subs.pop` replaces the toc children. The next load lists all four. |
| DOC-D3 | CONFIRMED | N | own road: README → LICENSE, then reload | `init` opens `?doc=` only if `_linksByPath` has it. `wireDocLink` never registers `../LICENSE`, so the reload shows an empty pane. |
| DOC-U1 | CONFIRMED | T | n/a | `_build_toc_tree` assigns the scanned list into the toc node and writes `toc.json`. On disk the Archive holds 85 entries. The docstring says this is intended, so the chapter is the side to fix. |
| TS-D1 | CONFIRMED | M | own road: a SIESTA folder, then a PySCF hand-over | `loadColumnChoices`/`loadSweepChoices`/`loadPresets` set `_colsKey` etc. before the fetch, but leave the old `_cols`. `proposeHandover`'s call a few microtasks later passes `_cols && _colsKey === key`, and gets SIESTA's list for the run cards (`standingRows`), the presets and the second `refreshPickers`. The first refresh is then out of sequence. Save's preflight would refuse a SIESTA field picked there. |
| TS-D2 | CONFIRMED | M | own road: two quick folder clicks, the first slower | `loadFolderAnswer` sets `_tmpl` after its await, with no folder check. A's late answer replaces B's template values in `defaultText`/`helpText` until the next read. |
| TS-D3 | CONFIRMED | M | own road plus a failed folder answer (signed out, server unreachable, folder gone) | `refuseFolder` and `hideDescriptionCards` leave the `#ts-file-list` rows alone. Only the no-folder branch resets them, so `paintHeads` counts A's files. Low. |
| TS-D4 | CONFIRMED | M | own road plus one failed fetch | On failure `items = []` is cached, and `[]` is truthy. `fillPicker` then says "every parameter is already a column", and the cache keeps it for that engine:kind. The machine card does note "raw setting names", but the pickers paint the failure as data. |
| TS-D5 | CONFIRMED | M | rare: `continue_from_choices` raising, or the need ask failing after the plan ask succeeded | `if (cf && !cf.error)` drops the server's `{error}` without a word, and `refreshNeed` returns on `!answer.ok`, leaving the memory line blank. Low. |
| TS-U1 | CONFIRMED | T | n/a | There is no SIESTA fallback: with no engine or kind each loader returns `[]` and asks nothing. |
| TS-U2 | CONFIRMED | T | n/a | The run card's *+ Add setting* (`addSetting`) adds an empty row with no value box, and its rows are classed chosen/inherited/unstated by the page. "The first point is typed before the row exists" and "chosen/machine/measured … the grid door's word" describe the What-to-measure card. |
| TS-U3 | CONFIRMED | T | n/a | `renderCameOver` draws no label row; the label shows only inside the run id and the file names. Low. |
| TS-U4 | CONFIRMED | T | n/a | `patchDoc` (queue and notify cards) parses only `job-request.json` and writes it while `task.json` is mid-edit. Nothing unparsed is overwritten, so the code is harmless; the sentence over-claims and should say "a change to a record whose text does not parse". |
| P-D1 | CONFIRMED | W | hand-made symlink inside `projects/` (molbuilder writes none); the contract lists links as tree items | Reproduced in scratch: `contain` resolves the link to its target and `_entry_dict` lists it as "directory". Delete then `rmtree`s the target, and rename/move act on the target. A broken link gets a 404. Duplicates r1-I D1. |
| P-D2 | CONFIRMED | M | own road: Save or Prep with the history view closed | The `onFolderChanged` handler refreshes only when `_state.open`, and `onDirectoryChange` returns early for the same folder. The button stays hidden or amber although `_readInto` says its colour must be honest with the view closed. |
| P-D3 | CONFIRMED | M | rare: a damaged `.git`, git failing, a server fault | `_hide()` then `_renderError` paints `data-state=error` on the hidden toggle and the advisory inside the hidden `#ps-checkpoint`. Nothing is seen. |
| P-D4 | CONFIRMED | N | a browser refusing site storage (private window, blocked data) | The View item calls `showPreview()` without the path. `getCurrentFile()` reads the refused slot as "", so the modal only warns in the console. `preview.js`'s own comment names this case. |
| P-D5 | CONFIRMED | N | own road, a race: Esc during a multi-MB read, then another file | `_loadBulk` has no `_state.path === path` check after its await (the chunk reader has one). In practice a Save is blocked, because a slow file is over 1 MB and its late answer sets `editable = false`. |
| P-D6 | CONFIRMED | N | own road | `_nameRefusal` refuses only separators and `.`/`..`. The server's `_UPLOAD_FILENAME_RE` and the folder rule refuse more, e.g. "water dimer", after both questions. The refusal is said, just late. |
| P-U1 | CONFIRMED | T | hand-made symlink | `_entry_dict` classifies a link by its target and `list.js` draws "→" only for `kind === "symlink"`, which a link to a file or folder never gets. The code side should change (classify by `lstat`): this hides P-D1's danger. |
| P-U2 | CONFIRMED | T | n/a | The confirm shows `size_words` (one decimal, binary units), not the exact byte count. Low. |
| P-U3 | CONFIRMED | T | n/a | `writeFile` keeps only `error`/`actual_mtime`/`aborted`, the Blob path only `error`/`aborted`, and `readCurrentFile` only `error`. No current caller needs the dropped fields. Low. |
| P-U4 | CONFIRMED | T | n/a | `outcome` → `toneOf` takes the most severe notice's tone, so info-only notices stay info. This matches `tabs.md` § 6 and R4; fix `projects.md`. |
| SO-D1 | CONFIRMED | W | own road: a multi-frame `.xyz`, scrub, reload | `keep()` runs only in `loadNow`, and saves `exportFile()` with no range, which is the frame on screen (frame 1). After a reload one frame comes back, the double-click is a no-op (`f === shown`), `frame()` is null, and Send writes frame 1. |
| SO-D2 | CONFIRMED | M | a file that fails to load (foreign or broken), after one that loaded | `failed()` rewrites the readout to "load failed: b.xyz" while the viewer keeps `a.xyz`. Send, the chemistry card and the live check all read `card.structure()` = `a.xyz`. Low. |
| SO-D3 | PLAUSIBLE | M | own road, timing | `collectParams` returns `{}` before the schemas land, and the preflight is not asked again after `_forms.restore()`; the chemistry card is. Not shown: whether a restore plus the 250 ms debounce beats the schema fetch, which needs MolView mounted first. |
| MB-D1 | CONFIRMED | N | own road | `Number("") === 0`, so `num("slab-a", NaN)` returns 0 and `lattice_constant: 0` is posted. The server's refusal is said instead of the panel's sentence, and nothing is built. |
| MB-D2 | CONFIRMED | C | own road plus a dropped connection or server error while adding into an open structure | `postJson` throws, and `_append` → `loadIntoCanvas` → `await` in `_commitFile` has no catch, nor do its click and `onCommit` callers. There is no global handler: an unhandled rejection, nothing on screen. |
| MB-D3 | CONFIRMED | M | own road | `markLoadedFrom` clears `_lastSavedTo`. `refreshState` reads only `lastSaveTo`, so a loaded file reads "Not saved yet" and an edit never shows "Unsaved". Low. |
| MB-D4 | CONFIRMED | N | own road (keyboard) | The measuring switch-on is a `click` listener. Arrow, Home and End go through `tabStrip.select`, which fires no click; Enter on the focused tab still works. |
| MB-U1 | CONFIRMED | T | n/a | `slab-panel.js` also asks `/api/modify/spacings` and `/api/modify/lattice-from-run`; `selection-bootstrap.js` asks `/api/build/load`; `panel-deps.js` asks the build route. |
| MB-U2 | CONFIRMED | T | n/a | `structure-optimization/viewer.js` imports `attachStructureCard` and `ask`; `lib/structure-card.js` mounts MolView. |
| MB-U3 | REFUTED | — | — | `generate` does call `loadIntoCanvas({structure}, generator)` with `{kind, generator_input}`, so the sentence is literally true. That the gate reads only `source.file` is a dead-argument code note, not an untrue sentence. |

**P-D1 (W).** A person who made a symlink inside `projects/` loses data by using the sidebar on the link:
- **Delete** asks "Delete 'calc'? This cannot be undone." and sends `recursive=true`. The server fences the path with `projects.contain`, which follows the link, so it `shutil.rmtree`s the target folder and leaves the link dangling.
- **Rename** and **Move** move the target, its sidecar too.
- **The tree** lists the link as a plain folder or file (P-U1), so nothing warns the person.
- **`_writable`'s own refusal** says "delete the link and retry", which, done in the sidebar, deletes what the link points to.

Smallest fix: the delete, rename and move routes check `Path(raw).is_symlink()` (an `lstat`) on the path as given, before resolving it. A link is unlinked or renamed as itself, and `_entry_dict` reports it as `symlink` from `lstat`.

**SO-D1 (W).** On Structure optimization, and on the Spectrum tab that shares the card, a person loads a multi-frame `.xyz`, scrubs to frame 7 and reloads.
- **What comes back:** a single structure, frame 1, "Restored traj.xyz — N atoms.". The frame set is gone, and double-clicking the same file does nothing.
- **What Send writes:** frame 1 as the hand-over's structure, with no frame named. Before the reload it wrote frame 7 of 9.
- **Why:** `keep()` saves only the displayed frame, at load time.

Smallest fix: `keep()` saves the whole set (`exportFile({from: 0, to: count - 1})`) and the frame on screen, and `restore()` puts both back. Alternatively, `keep()` is also run when the frame changes.

**MB-D2 (C).** On the Molbuilder tab, with a structure open, a person loads a file and answers "Add to existing view". If the append is refused, the server errors, or the connection drops, the page says nothing:
- **Mechanism:** `applyOp("append")` throws, and nothing on the Load path catches it.
- **What the person sees:** the status line keeps its old text, and the error ends as an unhandled promise rejection in the console.
- **A rarer case:** when `applyOp` returns null (another edit in flight), the gate's `{ok:false}` also returns silently.

Smallest fix: `_commitFile` wraps the `loadIntoCanvas` call in `try/catch` and says the error on the status line, as its own `load()` does for `/api/build/load`. A `{ok:false}` gate answer is said too.
