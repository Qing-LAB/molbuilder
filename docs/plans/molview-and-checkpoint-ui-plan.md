# The checkpoint view, the view context, and a working Export — plan

**Role:** plan (open)
**Domain:** web
**Started:** 2026-08-19 (user, four items, one batch; reviewed and amended same day — filter-disable rule, transport init-restore, read-only selection ownership, the two re-scoped contract sentences)
**Companions:** [`web/projects.md`](?doc=web/projects.md) — the sidebar the
checkpoint view folds into; [`web/molview.md`](?doc=web/molview.md) § 9.6 ·
§ 11.2 · § 11.2a · § 11.3 · § 11.4 — the contracts items 2–4 implement or
amend; [`web/workspace.md`](?doc=web/workspace.md) — the tag doctrine the new
lane rides on; [`execution/checkpointing.md`](?doc=execution/checkpointing.md)
— the invariants the panel serves, untouched here.

The user's four asks, 2026-08-19, quoted tight:

1. the checkpoint panel is "integrated inside the project sidebar … in a very
   inconsistent way: on the right side, no button to fold it or move it" —
   let it **share the file-list space**, switched by "a small button (that has
   color to indicate the checkpoint status such as error, clean, or changes
   detected) … next to the filter input".
2. "molview in each tab probably did not correctly store its persistency of
   its structural data with meta data … molview can go a bit further to save
   its toggle button state, camera setting and selection state too".
3. camera/toggle recording is "**not considered as 'change of data'** but just
   for the UI persistency".
4. Export "focus[es] on data export, and neither save to project dir nor
   download button actually does anything"; missing image/trajectory export —
   "two categories: data or image … a dialog to specify the starting frame and
   ending frame … a single png … or webm or gif … resolution can also be an
   option".

**The standing verdicts this plan builds on, not around:** the docs already
specify most of this (Data|Image with a frame range — molview.md § 11.3; the
menu's owner — § 11.4; the tag-per-saver doctrine — workspace.md § 4). The
implementations fall short of them. Where an ask *reverses* a recorded
decision (§ 9.6's camera), the contract is amended first, openly.

---

## Part 1 — the checkpoint panel becomes a view of the file-list space

### Facts (verified 2026-08-19)

- The panel is **not inside** the projects `<aside>`: it is a sibling
  `.dock-panel` in the shared `.sidebar-rail`
  (`templates/_projects_sidebar.html:169`), rendered to the right at a fixed
  `--ck-w: 20rem` (`tokens.css:126`, `projects-sidebar.css:1308-1323`) with
  **no resize handle and no width-reclaiming fold** — its "collapse"
  (`checkpoint.js:562-577`) hides the children and keeps the 20rem column.
  A stale comment still claims it mounts "BELOW the file tree"
  (`projects-sidebar.css:1301`).
- The status pill already has the states the user wants on the button:
  `data-state` = `uninit` (gray) / `dirty` (amber) / `clean` (green) /
  `error` (red), tokens at `tokens.css:149-161`, written at
  `checkpoint.js:215-265`. (A CSS comment names a `running` blue state no JS
  ever sets — dead, remove.)
- List and graph views + the tablist toggle already exist
  (`checkpoint.js:130-133`, `:348`, `:396-491`; gitgraph lazy-loaded).
- Visibility rule: depth-3 run dir under the projects root
  (`checkpoint.js:38`, `:145-157`); refresh on dir change / manual / after
  mutations — **no polling** (verified: no timers in the file).
- The filter row is `.ps-filter-bar` — the input plus an overlay clear `×`,
  nothing else on the row (`_projects_sidebar.html:126-141`,
  `projects-sidebar.css:592-624`).
- **Bug:** the documented public API `projects.checkpoint.status(dir)`
  (projects.md § 5) GETs `/api/checkpoint/status` — **no such route exists**
  (blueprint has `/state`; `checkpoint.js:842-853`) and reads `initialised`
  where the server writes `initialized` (`checkpoint.py:158`). Every caller
  gets `{ok:false}` forever.

### Design (proposed)

- **One column.** The rail `<section id="ps-checkpoint">` is deleted —
  template, `--ck-w`, the width rules, its collapse chevron, the dead
  `running` comment; no compatibility shim. The checkpoint DOM moves inside
  the `<aside>` as `#ps-checkpoint-view`, a sibling of `#ps-list`, `hidden`
  by default. Inner ids keep their names, so `checkpoint.js` keeps its
  bindings; what changes is where the panel lives and who shows it.
- **The toggle button** `#ps-checkpoint-toggle` joins `.ps-filter-bar`,
  right of the input (the bar becomes flex; the clear `×` stays an overlay on
  the input). It renders the pill's `data-state` as its color — the four
  existing tokens — with the state text in `title`. It exists **only when
  the current dir is a run dir** (today's depth-3 rule, same fetch triggers,
  still no polling); otherwise it is hidden and the filter keeps the row.
- **Clicking swaps the list area**: files (`#ps-list` + filter semantics) ↔
  checkpoint view (status line, actions, list/graph tabs — today's internals
  restyled for the sidebar's width; the graph keeps its own scroll). The
  choice persists in sessionStorage beside today's view-mode key; navigating
  to a non-run dir returns to files and hides the button. While the
  checkpoint view is showing, the **filter input is disabled** — it filters
  files, and a control that acts on a hidden list must say so rather than
  act invisibly. One owner for the whole swap: `checkpoint.js` (the same
  module owns the panel, the button, and the rule that decides both).
- **Space result:** the sidebar's existing drag handle governs everything;
  the second column is gone.
- **The API fix rides along:** `status()` → GET `/api/checkpoint/state`,
  field `initialized`; projects.md § 5's row corrected to the server's
  spelling. While in the CSS: add the missing `.ps-checkpoint-row-parent`
  rule, delete the never-emitted `.ps-checkpoint-row-archive`.

### Contract edits (first)

`web/projects.md` § 4 (run-history bullet → the toggled view, the button, the
rule that its color is the status) and § 5 (the `status()` row); the stale
CSS comment becomes true prose.

### Done when

The rail has one panel; the button appears only in a run dir, colored by
live status; the swap is exclusive and survives a reload; `status()` returns
real answers; existing checkpoint behaviors (init/save/tag/restore,
list/graph, no polling) pass unchanged; a browser walk-through on the dev
server confirms the layout at narrow and wide sidebar widths.

---

## Part 2 — persistence: close the truth-lane gaps, add the view-context lane
*(items 2 + 3)*

### Facts (verified 2026-08-19)

Per-tab truth-lane wiring today (tag → behavior):

| Tab | tag | mode | restores? | persists? |
|---|---|---|---|---|
| Modify | `modify` (+ `modify:panel` note) | editable | **yes** (`load(0)`, `modify/viewer.js:1119`) | yes — correct |
| Results/structure | `results:structure` | read-only | no (deliberate — re-opens from file) | none (contract: read-only has no history) |
| Results/trajectory | `results:trajectory` | read-only | no | none |
| **Transport** | `transport` | **editable** | **NO** | **write-only**: anchors point 0 + drafts on label edits (`lib/transport/core.js:250`; `model.js:441`) and never reads them back — the user's electrode labels are saved and never restored |
| Spectra | `spectra` | read-only | no (re-opens from file) | none |
| Structure-opt | `structure-opt` | read-only | tab-level restore of `{structure, loadedFrom}` | tab-level persist — works, but writes to `{workspaceId("structure-opt"), state_index: 0}`, the exact key a history point 0 would use; safe only while read-only |
| Task-setup | — | — | — | has no MolView at all |

- The saved bytes (`model.js:168-172`): structure + all frames + forces +
  **selection as a bare index array** (`stores.js:163`). The switches
  (isolate, labels, cell, axes, force arrows, scale), view settings (style,
  radius, background, projection), camera, and displayed frame are **not**
  saved — § 11.2's deliberate "truth vs a view of the truth" line, and
  § 9.6 records rejecting camera persistence by name.
- **Model bug:** `restoreState` never sets the unit state to HOLDING
  (`model.js:173-185` vs `:430`), so after a session restore
  `addFrames`/`setForces` throw "nothing loaded" — live on the Modify tab.
- **Stale docs/comments:** workspace.md's two "Not yet true of the code"
  boxes (tasks #44, #47) describe code that has since been fixed
  (`dispatcher.js:284-290` — `persist(tag,…)`, no `useNamespace`;
  `modify/viewer.js:417-419` — retract honesty); the § 5 table still lists a
  deleted `STORAGE_KEY`; `modify/viewer.js:1086-1091` claims a browser-storage
  copy that no longer exists.
- The demo page's stand-in workspace still has the old 4-arg `persist`
  (`demo.js:77`), so the demo timeline is silently dead.

### Design (proposed)

**A. Truth-lane fixes (no new design — the contracts already rule):**

1. `model.js`: restore/adopt sets HOLDING when a structure landed — the
   § 11.2a state machine made true.
2. Transport restores at page init — **not** in its lazy mount, which only
   runs after a structure commit and therefore can never fire on a reload.
   The tab mounts the viewer at init and calls `load(0)`; a successful
   adoption shows the viewer region exactly as a commit would have, and a
   `null` leaves today's pre-commit UI (§ 11.2a "coming back to a session").
   Its labels come back.
3. Structure-opt's tab note moves off the history's key shape to its own tag
   (`structure-opt:panel`), the documented `modify:panel` pattern.
4. The demo stand-in gets the 3-arg signature (or the real doors from
   Part 3); the stale comments and the two satisfied "not yet true" boxes in
   workspace.md are deleted; `STORAGE_KEY` row removed.

**B. The view-context lane — the new piece (items 2+3):**

A second workspace slot per viewer, tag **`<owner>:ui`** (the § 4 multi-saver
doctrine — two tags, two slots), holding what § 11.2 deliberately keeps out
of the truth:

```
{ v: 1,
  match:    { n_atoms, n_frames },      // who this context belongs to
  view:     { style, radius, background, orthographic },
  switches: { isolate, showIndex, showForces, showCell, showAxis, forceScale },
  camera:   <the sealed layer's pose>,
  frame:    <displayed frame index>,
  selection: [atom indices] }        // read by viewers WITHOUT a truth lane
```

- **Written by MolView itself** (it owns both stores), debounced, on any
  view/switch/frame change and on camera interaction end (pointer-up / wheel
  on the canvas — pose *read at write time*, never tracked).
- **Item 3's guarantee, pinned by test:** a view-context write never raises
  the badge, never rewrites the draft, never lays a point — it is not an
  edit, and the truth lane does not know it exists.
- **Restored at mount** for every viewer, read-only included (the truth rule
  § 9.4 gates history, not looking), after the structure is in: switches and
  view always; **camera, frame and selection only when `match` equals the
  installed structure** — the § 9.6 trade honoured by guard: a stale pose can
  leave the molecule off-screen, so a mismatch falls back to
  fit-to-structure. Reset still re-fits and clears the stored pose.
  **Selection has one owner per viewer**: where a truth lane exists
  (editable — the draft already restores it), the ui lane's copy is never
  applied; only a viewer with no history (read-only) reads it back. Two
  lanes both restoring the same fact is the drift § 5.2 exists to prevent.
- **One implementation in MolView, zero per-tab code.** Every tab that mounts
  a viewer gets the lane by construction — the framework answer to "each tab
  should come back as you left it".
- **Camera mechanics need one sanctioned read**: the sealed layer already
  holds the pose (§ 9.9); it gains `getCamera()`/`setCamera(pose)`, exposed
  through the renderEngine as a passthrough. § 9.7's "commands only" gets the
  same carve-out § 11.3 already grants the image ("the drawing library
  already has the image, so it is asked for it").

### Contract edits (first)

- `molview.md` § 9.6 rewritten: the camera is still not truth and still never
  enters a saved state; it **is** recorded in the view-context lane
  (user-decided 2026-08-19), with the match-guard and the off-screen trade
  stated; § 9.7/§ 9.9 note the sanctioned pose read.
- `molview.md` new § 11.2b — *the view context*: the lane, its tag, its
  guarantee (never a data change), who writes and when.
- `workspace.md` § 4 example tags gain `<owner>:ui`; the satisfied
  "not yet true" boxes deleted.
- Two sentences that the lane would silently falsify are re-scoped to the
  truth lane: § 11.2's "neither the camera nor the switches nor the displayed
  frame is written anywhere", and § 11.2a's "nothing is written to storage"
  for read-only viewers. A contract sentence that becomes false the day a
  feature ships is a lie with a start date — both are amended in the same
  change that ships the lane.
- `modify-persistency-investigation.md` gains its closing note (rows now
  covered) or is archived per the finished-plan rule.

### Done when

Reload any tab and the viewer returns with the same structure (where the
truth lane applies), same switches, same style, same camera (same structure)
and same frame; transport labels survive a reload; toggling anything visual
raises no badge and writes no draft (mutation-tested); restore-then-addFrames
works on Modify; suite green.

---

## Part 3 — Export that actually exports *(item 4)*

### Facts (verified 2026-08-19)

- The menu has one "Structure" section with two rows — *Save to project* and
  *Download* — and both call `files.save(...)` behind
  `if (!file || !files || …) return;` (`ui.js:732-742`). **`files` is a mount
  option that no production mount site passes** (verified all seven) — so
  both rows are silent no-ops on every real page. Only the demo passes a
  door, whose "project" half is itself a stub ("would save …", `demo.js:152`).
  The mount test injects its own door (`test_molview_mount.py:944-991`),
  which is how this ships green.
- `model.exportFile(range)` already does the right thing: frame range
  (clamped, defaulting to the displayed frame), structure **with the sidecar
  fields**, refusal on inconsistency (`model-jobs.js:440-479`). No caller
  passes a range.
- The stem's range form (`wire_frame40-120`, § 11.4) exists only in prose;
  code has the single-frame form (`ui.js:759-764`).
- PNG capture exists **sealed and unreachable**: `3dmol-embed.js:717-746`
  (`capture(width,height)` → Blob, resized via `pngURI`); nothing above
  exposes it. No webm/gif machinery in MolView; the vendored gif encoder
  (`static/vendor/gif.min.js`) is consumed only by VibrationView's private
  `_export.js` (gif/webm/png-zip over MediaRecorder).
- Server: `/api/structure/save` accepts `{path, structure, frames?, overwrite}`
  and writes the pair (`build.py:747-794`); `/api/structure/export` returns
  `{files:[{name,text}…]}` for a download, stem from the caller, suffix from
  the server (`build.py:797ff`); `projects.writeFile(path, Blob)` exists for
  binary writes. **No new server routes needed.**

### Design (proposed)

**A. One real files door, built once.** A small module in the projects domain
(`lib/projects/molview-doors.js`, exported on the one door as
`projects.molviewFiles()`), passed as `opts.files` by each mount site — one
line per tab, one implementation, per § 11.2a's rule that doors are handed in
(the seal stays; MolView still holds no file route). It implements:

- `save("download", stem, payload)` → `/api/structure/export` → browser
  download of **every returned file** — the `.xyz` *and* the
  `.molstruct.json` (item 4's "including meta information").
- `save("project", stem, payload)` → the projects dialogs pick folder + name
  (the § 11.3 rule: the user names a project save), then
  `/api/structure/save` with the overwrite flow — the same pair, written by
  the same server codec the Save panel uses.
- `saveBinary(destination, filename, blob)` — the image half: download via
  the browser, project via `projects.writeFile`.

**B. The menu becomes § 11.3's menu.** Two sections, **Data** and **Image**,
each with *Save to project* and *Download*:

- **Data**: multi-frame structures get the frame-range dialog (opens on the
  displayed frame; one-frame structures never ask — both are § 11.3 verbatim);
  the range flows into `exportFile(range)`; one frame → plain `.xyz` + `.json`,
  more → extended-XYZ + `.json` (the server already does this); the stem
  gains its documented range form.
- **Image**: the dialog asks the frame range (same defaulting), a
  **resolution** choice (the user's addition — 1×/2×/4× of the canvas, shown
  with resulting pixels), and — for a range — **webm or gif**. One frame →
  `.png`. Rendered from the drawing with the view as set (§ 11.3): the
  renderEngine exposes `capture(...)` as a passthrough to the sealed layer's
  existing implementation; a movie steps frames, captures each, encodes.
- **The encoder is a shared module**, `lib/media-export.js` (gif via the
  existing vendored encoder, webm via MediaRecorder), consumed by MolView
  now. VibrationView keeps its private copy untouched in this change —
  MolView and VibrationView stay separate modules — and its migration onto
  the shared module is a follow-up, listed under deferrals.

**C. Honest tests replace the self-supplied door.** The mount test stops
injecting its own `files`; new tests pin: production mounts pass the door;
download yields both files; project save writes the pair through the route
(overwrite path included); the one-frame menu asks nothing; the range reaches
`exportFile`; a captured png honours the resolution; a range yields the
chosen movie format; Image reads the drawing (isolate/style visible in the
capture call, not recomputed).

### Contract edits (first)

`molview.md` § 11.3/§ 11.4: the resolution option added to the Image dialog's
spec; the movie-format choice (webm|gif) stated; the "Left:" note rewritten
when delivered; `web/projects.md` § 5 gains the door. web-api.md already
documents both routes.

### Done when

On the live server, from the Modify tab: Export → Data → Download hands the
browser `name.xyz` + `name.molstruct.json`; Save to project writes the pair
where the dialog said; Export → Image produces a real `.png` at the chosen
resolution, and a range produces a playing `.webm`/`.gif` of exactly those
frames with the on-screen style; the same menu works on a read-only viewer
(Results); suite green.

---

## Order, and what is deferred

| Phase | Scope | Status |
|---|---|---|
| **1** | Part 1 (checkpoint view + `status()` fix) | ✅ delivered 2026-08-19 (`e16eb043`) |
| **2** | Part 2A truth-lane fixes (HOLDING bug, transport restore, tag hardening, stale notes) | ✅ delivered 2026-08-19 (`bfb40920`) |
| **3** | Part 2B view-context lane (+ § 9.6 amendment) | ✅ delivered 2026-08-19 (`5f306ab8`) |
| **4** | Part 3 export (door → Data → Image) | ✅ delivered 2026-08-19 (`c49b9c07`) |

All four phases: contract first, executing tests, mutations red, page-boot
suite green.

**Live-test findings folded in the same day** (user, on the dev server):
the pair's download arrived one file short — not a server loss (the round
trip carries regions/cell/origin/axis/vacuum, probed) but the browser's
multiple-download policy swallowing the second programmatic click; a pair
now leaves as one `<stem>.zip` (store-zip extracted to `lib/zip-store.js`,
one home, VibrationView swapped onto it verbatim).  The save-where question
became the unified `projects.chooseSavePath` — named future consumer:
transport-input consolidation.  **Open before this plan archives:** the live browser
walk-throughs (the Claude-in-Chrome extension was not connected during the
build) — checkpoint swap at narrow/wide widths, a reload round-trip on each
tab, a real Data download + Image png/webm on the dev server.

Each phase: contract first, code, tests (mutation-checked), targeted batches,
then the browser walk-through on the dev server.

**Deferred, decided here:** migrating VibrationView onto the shared encoder;
any cross-tab live sync (projects.md § 6's known gap, untouched); the
checkpoint panel on mobile (today's panel is desktop-only — the toggled view
inherits the sidebar's existing drawer behavior and nothing more).

**Pending:** the Claude-in-Chrome extension was not connected during
planning; the live walk-throughs happen once it is. The header's reload
button opens a native `confirm()` that freezes the automation — it is never
clicked from the browser session (deployment.md § 4).
