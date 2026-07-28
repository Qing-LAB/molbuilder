# README screenshot manifest

**Role:** process
**Domain:** process
**Companions:** [`testing.md`](?doc=process/testing.md) ·
[`web/tabs.md`](?doc=web/tabs.md) (what each tab actually shows)

The README references 10 screenshots under `docs/img/`.  This doc is
the capture guide: for each filename, the URL to open, the state to
set, the zoom region, and what the image must communicate.  Update
this file whenever the README's image references change.

> **⚠ Known stale capture (2026-07-28).**  `tab-bar.png` — and the nav
> strip inside `hero-molbuilder.png` — show **five** tabs.  The shipped
> app has **six**: the **Documents** tab landed after these were taken
> ([`web/tabs.md § 1`](?doc=web/tabs.md)).  The README's alt text for
> `tab-bar.png` also still says "Five-tab nav strip".  Both need a
> re-capture + an alt-text fix (the re-capture rule below already
> mandates it: a new tab is exactly the trigger).

> **Demo data convention.**  Every screenshot uses the project at
> `projects/BDT/` so the README reads as one continuous Au–BDT–Au
> example — the SMILES → junction → optimisation → spectrum →
> transport → results loop in one project tree.  Stage-3 optimised
> geometry is at
> `projects/BDT/structure/BDT-AuJunction_siestaStage1_optimized.xyz`
> (despite the `siestaStage1_optimized` filename — it is the latest
> usable geometry today).

## Pre-capture setup (do once)

1. `molbuilder serve --port 8888` from the repo root.
2. Open `http://localhost:8888/` in Chrome / Firefox.
3. Resize the browser to **1440 × 900** (or 1600 × 1000) — gives a
   consistent pixel grid across all screenshots so they look like a
   family.  Wider than 1600 starts to introduce flex-grid wrap that
   the README's prose doesn't describe.
4. Light theme.  If your OS is set to dark, switch the browser
   appearance for these captures — the README's product-marketing
   tone reads more cleanly on light, and dark-mode contrast in the
   3Dmol viewer fights with the surrounding panels.
5. Open the `BDT` project in the sidebar so the cursor is set; some
   shots pre-fill from `getCurrentDir()`.

## How to take a clean shot

- **Chrome DevTools → Capture node screenshot** (right-click an
  element in the Elements panel → "Capture node screenshot"): zooms
  to the element's bounding box exactly, no surrounding chrome.
  Best tool for the panel-zoom shots below.
- **OS-level area-select** (macOS `⌘⇧4`, GNOME
  `gnome-screenshot -a`, Windows Snipping Tool): use for whole-tab
  shots when you want the browser frame cropped out manually.
- **No browser DevTools panel** in any screenshot.  No address-bar
  highlight.  No personal bookmarks in the bookmark strip — hide
  the bookmark strip first (`Ctrl-Shift-B`).
- Default 3Dmol camera (no user rotation) where the structure shape
  matters — Au-BDT-Au reads cleanly with the junction axis vertical.

## The 10 manifest entries

| # | Filename | Used in README § | URL | What to load / set | Zoom region | Communicates |
|---|---|---|---|---|---|---|
| 1 | `hero-molbuilder.png` | top of README | `/molbuilder` | Load `projects/BDT/structure/BDT-AuJunction_siestaStage1_optimized.xyz`; Junction panel open; default camera | Full content area (sidebar + main + the right-side commands stack) — exclude browser chrome | "This is molbuilder" — sidebar + tabs + viewer + commands in one frame |
| 2 | `tab-bar.png` | § Feature tour intro | `/molbuilder` | — | Zoom: just the top tab strip (**six** tabs — the order comes from the one `TABS` list in `tabs.py`).  ~1200 × 60 px | Names + order of every tab |
| 3 | `sidebar-projects.png` | § Workflow + § Documentation | `/molbuilder` | Expand `BDT/` → `structure/`; cursor on `BDT-AuJunction_siestaStage1_optimized.xyz` so the paired `.molstruct.json` shows as a sidecar | Zoom: just the projects sidebar column.  ~360 × 700 px | Tree shape + the structure↔sidecar pairing |
| 4 | `molbuilder-workspace.png` | § 1 Molbuilder tab | `/molbuilder` | Load the BDT junction; one atom selected in the viewer to show the amber **shape glow** (the only selection highlight — halos were removed, see [`web/molview.md`](?doc=web/molview.md)) + sync to the atom list on the left | Zoom: the `/molbuilder` content area (no sidebar, no top tab strip).  ~1080 × 760 px | The 3-panel layout: atom list + viewer + commands stack |
| 5 | `structure-optimization-form.png` | § 2 Structure optimization | `/structure-optimization` | Pick `BDT-AuJunction_siestaStage1_optimized.xyz` from the sidebar; engine = SIESTA; default profile.  Issues panel will show INFO notices about Au-thiol detection | Zoom: the `/structure-optimization` content area (form on left, viewer on right, issues panel docked below).  ~1100 × 760 px | Schema-driven form + workflow-group cards + inline detection chip + issues |
| 6 | `spectrum-form.png` | § 3 Spectrum calculation | `/spectrum-calculation` | Pick `projects/BDT/spectrum/BDT-only/spectra.spectra.py` *if* present — else use any small molecule. Default form values | Zoom: the form panel only.  ~520 × 700 px | The vertical workflow-group layout (Profile / Stage / Budget) |
| 7 | `transport-form.png` | § 4 Transport calculation | `/transport-calculation` | Pick the same BDT junction `.xyz`.  Default form (electrode-Au, 0 V) | Zoom: the form + a small slice of the viewer showing the region-labelled atoms.  ~1080 × 760 px | Electrode region labels flowing in from the sidecar |
| 8 | `results-trajectory.png` | § 5 Results | `/results` | Open the BDT multi-stage optimisation: cursor on `projects/BDT/optimization/BDT-withAuJunction/` (the directory itself); the inspector merges stages 1+2+3 into one trajectory | Zoom: the content area: viewer top, frame strip + scrub slider mid, energy / force / SCF-residual plots stacked on the right.  ~1100 × 800 px | Live multi-stage trajectory inspector |
| 9 | `results-spectra.png` | § 5 Results | `/results` | Open `projects/BDT/spectrum/BDT-only/spectra.spectra.json` | Zoom: the content area.  Show the Lorentzian-broadened spectrum chart + modes table + the 3-D viewer on a selected mode | The spectra inspector's two-panel layout |
| 10 | `results-bundle-card.png` | § 5 Results — "Bundle for next stage" | `/results` | Cursor on any file inside the BDT optimisation dir so the form pre-fills.  Don't click Bundle (we want the resting state).  Alternatively: capture once after a successful Bundle so the green result panel shows | Zoom: just the Bundle card.  ~800 × 360 px (resting) or ~800 × 540 px (with the green/amber result panel) | The new Step-3 workflow-handoff UI; if showing a result, the `final_coords_from` field is the load-bearing UX signal |

## Naming + path convention

- All PNGs land at `docs/img/<exact-filename>.png` as listed above.
- The manifest is the whole of `docs/img/`: exactly these 10 files,
  each referenced by the README.  A PNG nobody references doesn't
  belong there.

## Sizing rules

- **Aspect ratio**: keep each shot's natural aspect; don't crop to
  16:9 for the sake of it.  GitHub renders at full width by
  default; the README places no inline width constraint.
- **DPI**: capture at 2× / Retina if your machine supports it.
  GitHub down-samples on display but keeps the high-res for the
  "open image in new tab" path.
- **Compression**: `pngquant --quality 80-95 *.png` is fine for
  product screenshots; loss is invisible at GitHub's render scale.

## Re-capture cadence

- Re-shoot any image whose tab's "Feature tour" prose changes
  materially (new panel, renamed button, removed setting).
- Re-shoot all 10 on a major UI refactor (major-version bump).
- Smaller corrections (palette tweak, font change) don't need a
  re-capture unless the change is the headline of the README
  paragraph.
