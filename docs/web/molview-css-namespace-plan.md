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

## 4. Ownership — measured, and it redraws the plan

Before renaming anything, every class the stylesheet defines was traced to who
**creates** it (MolView's own JS), who else **references** it (a template or
another module's JS), and who else **defines** it (another stylesheet):

| Bucket | Count | What it means |
|---|--:|---|
| **MolView's own** | **72** | created only by `ui.js` / `mount.js`, styled only here — rename freely |
| **shared** with templates or other JS | **72** | the risky half: all the `mol-viewer-*` names the 3Dmol embed also uses, plus the global ones |
| **orphaned** — styled, created by nobody | **9** | ✅ cleared 2026-08-01. Was reported as 17; two were the speed box (since built) and **four were phantoms** — names that appear only inside comments recording their own earlier removal, which the first sweep counted as CSS |
| collides but is ours | 1 | `mol-viewer-toggle` |

**The phase-2 "private areas" label was unreliable and is now replaced by this
measurement.** It had already proved wrong once — `cell-*` turned out to belong
to a server-rendered partial — and guessing again would waste the same day twice.

### The 17 orphans are two different things

**Stale classes** — the feature exists under another name, so the rule is simply
dead: `selection-measurement-overlay` (the readout is built, under other
classes), `tag-frozen` (superseded by the `tag-reserved--N` tones),
`molview-overlay--top-left` / `--top-right` (two of four corners unused),
`mol-viewer-action`, `mol-viewer-bare`, `col-name`, `ctab-panel`,
`selection-actions-left`, `selection-add-btn-primary`, `selection-field-label`.

**Styled but never built** — the UI was designed, given CSS, and no control was
ever written. These are contract claims with nothing behind them:

| Orphan | What the contract promises |
|---|---|
| ~~`mvf-speed`, `mvf-speed-input`~~ | ✅ **Built 2026-08-01** (`ac41377`). § 1.1 had specified it exactly — *"speed box in milliseconds per frame (20–3000, default 150)"* — and the CSS was already waiting. No longer an orphan; it moves with the `frames` area in phase 2 |
| `selection-isolate-toggle` | § 8.5: isolate *"is the one switch with a control of its own ('Show selected only')"*. Only the rail switch exists; the panel control does not |
| `selection-loading`, `selection-error` | loading and error states for the panel, never rendered |

That is the answer to "is any UI design not linked to actual code": **yes, three
of them**, and the stylesheet is what gave them away — a rule with no element is
a design that was drawn and dropped.

---

## 5. The plan

**The rule that shapes it: never blind-sed a namespace rename.** A stylesheet
rename is invisible to unit tests — every stub keeps passing while the UI is
broken, because nothing in a DOM stand-in has a computed style. So the safety
comes from ordering and from the browser, not from the suite.

**Phase 1 — retire the dead sheet.** ✅ **Landed 2026-08-01** (`233f06c`).
Deleted `lib/molview-old/molview.css` (2020 lines), referenced by nothing. It did
not turn the duplicate-selector guard green — that was my wrong expectation, the
guard was already failing on the embed collisions underneath — but it left the
report unambiguous: 82 duplicates, one file pair.

**Phase 1b — one size for the panel's controls.** ✅ **Landed 2026-08-01**
(`979ea9c`), out of order because it is a defect rather than a rename. Three type
sizes, three font families and four control heights inside one card, because
browsers do not inherit fonts into `button` / `input` / `select`. Now one of each.

**Phase 1c — clear the orphans.** ✅ **Landed 2026-08-01.** Nine real rules
deleted (79 lines).

TWO THINGS THIS TAUGHT, both about measuring CSS with regexes:

1. **Four of the seventeen were phantoms.** `mol-viewer-action`,
   `mol-viewer-bare`, `selection-field-label` and `selection-isolate-toggle`
   appear in the file only inside comments that record their own earlier
   deletion. A sweep that does not strip comments first counts those as live
   rules. (Usefully, the `selection-isolate-toggle` comment says the control
   "moved to the viewer-controls bar" — so the rail switch IS the isolate
   control, and § 8.5's claim of a second one in the panel was already stale.)
2. **A deletion pass must strip comments before matching selectors.** The first
   attempt did not, so a rule preceded by a comment naming an orphan was deleted
   with it — `.mol-viewer-bg-row` (the View menu's background row) and
   `.selection-assign-select` (the label dropdown) both went. Caught by diffing
   the class list before and after, reverted, redone with offset-preserving
   comment masking: nine deleted, **zero collateral**.

The check that mattered was not the test suite — it stayed green through the
damage — but the before/after class diff and the browser. For the three **unbuilt** ones, the CSS
does not get deleted quietly — each is a decision: build the control, or drop it
from the contract too. The speed box is the one worth building; the contract even
specifies its range.

Doing this first shrinks phases 2–4 by 17 names and removes the risk of
carefully renaming rules that style nothing.

**Phase 2 — the 72 that are MolView's own.** No template and no other module
references them, so a mistake stays inside MolView. In order: `cell` →
`regions` → `label` → `atoms` → `filter` → `panel` → `frames` → `selection`.
One commit each.

**Phase 3 — the 72 shared names.** Where the win is (46 stop being shared with
the 3Dmol embed) and where the risk is, because today's appearance may *depend*
on the embed's rules. One area per commit, screenshot before and after. **The
duplicate-selector guard going green is this phase's acceptance criterion.**

**Phase 4 — the global names** (`card`, `is-*`, `viewer*`, `sr-only`). Smallest
diff, largest blast radius: another page may rely on MolView's `.card` winning.

**Phase 5 — the guard.** Every selector in `lib/molview/molview.css` starts with
`molviewer-`. Last, because added earlier it just fails for four phases.

### What must survive the rename

Two pieces of work landed **after** the mapping table in § 3 was generated, so
they are not in it. Renaming without carrying them is how a fix gets silently
undone by a later phase — the failure this whole plan exists to avoid.

**The control-sizing tokens and their rule (phase 1b).** These are what make the
panel's type sizes and heights consistent, and they are scoped by a class that
phase 2 renames:

| Today | → | Moves in |
|---|---|---|
| `--mv-control-font` / `-family` / `-line` / `-height` / `-pad-y` / `-pad-x` / `-box` | `--molviewer-control-*` | phase 2, with the `panel` area |
| `.molview-panel button, input, select, textarea` | `.molviewer-panel …` | phase 2 |
| `.molview-panel input[type="checkbox"] / [type="radio"]` | `.molviewer-panel …` | phase 2 |

The scoping selector is the fragile part: rename `.molview-panel` and forget
these three rules, and every control in the panel silently reverts to browser
defaults — three type sizes and four heights again, with nothing failing. **The
check after phase 2 is therefore not "does it still look right" but the
measurement that found it**: one font size, one family, one row height.

**The speed control's classes.** `mvf-speed` / `mvf-speed-input` are now live and
carry real styling (width, spinner removal). They rename with the rest of `mvf-*`
→ `molviewer-frames-*`, and the input's class is load-bearing — it was built
without one for an hour and the control appeared unstyled.

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
