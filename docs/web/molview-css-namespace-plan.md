# MolView — the stylesheet namespace plan

**Role:** plan
**Domain:** web
**Started:** 2026-08-01
**Companions:** [`molview.md`](?doc=web/molview.md) — the contract this serves, in
particular § 4 (a self-contained module) and § 8.1 (what the card is made of).
Retired when the last phase lands.

---

## 1. Why this exists

MolView's whole doctrine is concealment: one entry point, 3Dmol named in exactly
one file, nothing on `window`. The stylesheet obeys none of it. It publishes
**167 class names under 9 different prefixes** into a global stylesheet shared
with every other page in the application.

That is not a tidiness complaint. Measured:

| | |
|---|---|
| class names MolView defines | **167** |
| also defined *outside* MolView | **58 (35%)** |
| of those, shared with `lib/viewer/mol-viewer-embed.css` | **46** |

**Whichever stylesheet loads last wins.** Forty-six of MolView's names are
currently shared with the 3Dmol embed — the module MolView is supposed to be
independent of, and which task #104 exists to separate it from. So MolView's
card, menus, export dialog and busy cover are all styled by a sheet belonging to
somebody else, and a change over there silently restyles this.

The rest of the overlap is worse for being ordinary: `.card` is defined in
**nine** other stylesheets, `.is-active` in five, `.viewer` and `.viewer-wrap` in
three each, `.sr-only` in two. These are names any page might use, sitting in a
module that claims to be sealed.

> There is a live symptom already: `tests/test_css_no_duplicate_selectors.py`
> **fails**, and has been failing on this.
>
> After phase 1 removed the frozen sheet's noise, what it reports is exact:
> **82 duplicate selectors, every one of them between `lib/molview/molview.css`
> and `lib/viewer/mol-viewer-embed.css`.** A single pairing, no other file
> involved. That guard is therefore not a chore to satisfy at the end — **it is
> phase 3's acceptance criterion**, and it goes green when the embed-shared
> names are gone.

## 2. The naming scheme

**One prefix, spelled out: `molviewer-`.** Not `mv-`, not `mvf-`. A prefix that
has to be decoded is a prefix people guess at, and guessing is how `mol-viewer-`,
`molview-` and `viewer-` came to be three spellings of one owner.

The shape is **`molviewer-<area>-<part>`**, where the area names a place on the
card that § 8.1 already names, so a reader can find it without a legend:

| Area | What it covers |
|---|---|
| `card` | the card shell, its header, the stacked/broken states |
| `window` | the 3D window — stage, canvas, busy cover, info line |
| `rail` | the left-edge switch rail (§ 1.1's six toggles) |
| `menu` | the View menu — style, radius, background, projection |
| `export` | the Export menu and its dialog |
| `frames` | the frame bar — slider, transport, counter, speed, loop |
| `panel` | the panel container, its fold, and the page tabs |
| `selection` | the Selection page — modes, actions, status |
| `filter` | the filter rows |
| `atoms` | the atom table and its columns |
| `label` | the label chips (reserved tones included) |
| `regions` | the region-definitions help block |
| `cell` | the Cell page |
| `overlay` | badge, measurement readout, corner placements |
| `is` | state flags, namespaced — `molviewer-is-active` |

## 3. The mapping

Every current prefix, where it goes, and how many names move:

| Today | → | Count | Note |
|---|---|--:|---|
| `selection-filter-*` | `molviewer-filter-*` | 11 | the filter rows earn their own area |
| `selection-*` (rest) | `molviewer-selection-*` | 36 | |
| `selection-atom-table` | `molviewer-atoms-table` | 1 | |
| `selection-measurement-overlay` | `molviewer-overlay-measurement` | 1 | **collides with `results/style.css`** |
| `selection-header-tabs` | `molviewer-panel-tabs` | 1 | it is the panel's tab bar, not the selection's |
| `mol-viewer-export-*` | `molviewer-export-*` | 17 | **all collide with the embed** |
| `mol-viewer-menu*` | `molviewer-menu*` | 4 | **collide** |
| `mol-viewer-bg-*` | `molviewer-menu-background-*` | 3 | **collide** |
| `mol-viewer-rep-*` | `molviewer-menu-style-*` | 2 | `rep` is 3Dmol's word, not ours |
| `mol-viewer-radius-row` | `molviewer-menu-radius-row` | 1 | **collides** |
| `mol-viewer-quickbar` / `-quick` | `molviewer-rail` / `-rail-button` | 2 | "quick" named nothing |
| `mol-viewer-busy*` | `molviewer-window-busy*` | 2 | **collide** |
| `mol-viewer-canvas` / `-stage` | `molviewer-window-canvas` / `-stage` | 2 | **collide** |
| `mol-viewer-card*` | `molviewer-card*` | 3 | **collide** |
| `mol-viewer-*` (rest) | per area above | 9 | **collide** |
| `molview-card*` | `molviewer-card*` | 2 | |
| `molview-overlay*` | `molviewer-overlay*` | 7 | |
| `molview-fold-*` | `molviewer-panel-fold-*` | 2 | |
| `molview-*` (rest) | per area | 5 | |
| `mvf-*` | `molviewer-frames-*` | 8 | `mvf` is unreadable |
| `frame-counter` / `frame-slider` | `molviewer-frames-*` | 2 | **collide with the embed** |
| `cell-*` | `molviewer-cell-*` | 8 | |
| `region-defs-*` | `molviewer-regions-*` | 8 | |
| `tag-*` | `molviewer-label-*` | 8 | `tag` is ambiguous with HTML tags |
| `col-*` | `molviewer-atoms-column-*` | 6 | `col` could be anything |
| `panel-page*` | `molviewer-panel-tab*` | 3 | |
| `ctab-panel` | `molviewer-panel-page` | 1 | one orphan from an older scheme |
| `is-*` | `molviewer-is-*` | 4 | **`.is-active` collides with five sheets** |
| `viewer` / `viewer-wrap` / `viewer-toggle` | `molviewer-window` / `-window-wrap` / `-rail-toggle` | 3 | **collide** |
| `card` / `card-header` | `molviewer-card` / `-card-header` | 2 | **`.card` collides with NINE sheets** |
| `in-use` | `molviewer-is-in-use` | 1 | |
| `sr-only` | `molviewer-screen-reader-only` | 1 | **collides**; also unabbreviated |

### What the rename surfaces

Mapping the names revealed **five concepts spelled more than one way**, which is
the real cost of nine prefixes and worth fixing in the same pass:

| Concept | Spelled today as |
|---|---|
| the card | `card`, `mol-viewer-card`, `molview-card` |
| the 3D window | `viewer`, `molview-viewer` |
| the rail toggle | `viewer-toggle`, `mol-viewer-toggle` |
| the frame counter | `frame-counter`, `mvf-counter` |
| the frame slider | `frame-slider`, `mvf-slider` |

Each collapses to one name. That is a **behaviour change, not a rename** — two
selectors becoming one can change which rule wins — so each of the five gets
checked in the browser individually rather than as part of a bulk pass.

## 4. The plan

**The rule that shapes it: never blind-sed a namespace rename.** A stylesheet
rename is invisible to unit tests — every stub keeps passing while the UI is
broken, because nothing in a DOM stand-in has a computed style. So the safety has
to come from ordering and from the browser, not from the suite.

**Phase 1 — retire the dead sheet.** ✅ **Landed 2026-08-01.** Deleted
`lib/molview-old/molview.css` (2020 lines), referenced by nothing — not by a
template, not by the frozen tree's own JS, which points at `fused-layout.css`
instead.

It did **not** turn the guard green, and expecting it to was wrong: the guard was
already failing on the embed collisions underneath. What it did is worth more —
it left the report unambiguous. Before, the noise from a dead file sat on top of
the real problem; now the failure reads *82 selectors, one file pair*, which is
exactly the work phase 3 does and nothing else.

The rest of `molview-old/` stays until the MolView-users pass is done with it as
a reference. The file is in git history either way.

**Phase 2 — the areas nobody else touches**, in this order, one commit each:
`cell` → `regions` → `label` → `atoms` → `filter`. None of their names appear in
another stylesheet, so a mistake shows up only inside MolView, and each is small
enough to read the whole diff.

**Phase 3 — the areas that collide with the embed**: `export`, `menu`, `window`,
`rail`, `frames`, `card`. This is where the win is — 46 names stop being shared —
and where the risk is, because today's appearance may *depend* on the embed's
rules. Take one area per commit and screenshot before and after.

**Phase 4 — the global names**: `is-*`, `card`, `viewer*`, `sr-only`. Smallest
diff, largest blast radius: another page may be relying on MolView's copy of
`.card` or `.is-active` winning. Do it last, when everything else is settled.

**Phase 5 — the guard.** `test_css_no_duplicate_selectors` already goes green at
the end of phase 3; this adds the stronger one — every selector in
`lib/molview/molview.css` starts with `molviewer-`, so the next stray name cannot
drift back in. It lands last because added earlier it just fails for four phases.

### Per-phase method

1. `git grep` the old name across `.css`, `.js`, `.py` and `.html` — class names
   appear in `el("div", "…")` calls, in `classList` toggles, and in test
   selectors. **Fix every hit before running anything**, so a half-renamed state
   is never committed.
2. Run the JS suites and `tests/test_css_no_duplicate_selectors.py`.
3. **Open the page and look.** The suite cannot see a style regression; a
   browser can. Drive it with Claude in Chrome against a running
   `molbuilder serve`, and check the same circuit every time so the phases are
   comparable:

   | Step | What a regression looks like |
   |---|---|
   | load a structure | the card does not size, or the window is not square |
   | fold the panel, unfold it | the handle's arrow points the wrong way, or the panel does not collapse |
   | open the View menu | swatches unstyled, the active style not lit |
   | open the Export menu | rows unstyled or the dialog unpositioned |
   | switch to the Cell page and back | the tab does not light, the matrix loses its rows |
   | play a trajectory | the frame bar loses its slider or counter |
   | select atoms, add a label | chips lose their tone, the × is unplaced |

   Take a screenshot at the same point before and after each phase, and read the
   console: an unstyled element is silent, so the eye is the instrument here.
4. Commit that area alone, with the old and new names in the message so a later
   bisect can read what moved.

### What each phase is worth

| Phase | Names moved | Duplicate selectors left |
|---|--:|--:|
| 1 — retire the dead sheet ✅ | 0 | 82 (was: 82 + the frozen sheet's noise) |
| 2 — the private areas | 41 | 82 — none of these collide |
| 3 — the embed-shared areas | ~60 | **0** ← the guard goes green here |
| 4 — the global names | 8 | 0, and `.card` / `.is-active` stop being shared with nine and five other sheets |

Phase 3 is the one that matters. Phases 2 and 4 are cheap either side of it.
