# README screenshot manifest

**Role:** process
**Domain:** process
**Companions:** [`testing.md`](?doc=process/testing.md) ·
[`web/tabs.md`](?doc=web/tabs.md) (what each tab actually shows)

`docs/img/` holds 10 screenshots.  This doc is the capture guide: for
each filename, the URL to open, the state to set, the zoom region, and
what the image must communicate.  **Since the 2026-07-28 README
rewrite, the README embeds only `hero-molbuilder.png`** — the other
nine are kept current per this manifest for docs embedding and any
future README gallery.  Update this file whenever an embedding of one
of these images changes.

> **⚠ Known stale capture (2026-07-28, still stale 2026-09-01).**
> `tab-bar.png` — and the nav strip inside `hero-molbuilder.png` — show
> **five** tabs.  The shipped app has **eight**: **Documents** landed after
> these were taken, then **Task setup**, then **This machine**
> ([`web/tabs.md § 1`](?doc=web/tabs.md) is the roster and the only place
> the count is decided).  Both need a re-capture (the re-capture rule below
> already mandates it: a new tab is exactly the trigger); the README's hero
> alt text deliberately avoids naming a tab count until then.
>
> Three tabs behind is the argument for the rule, not against it: each one
> landed with a note like this and none triggered the capture, because a
> note is not a mechanism.  The count had read *five · six · seven*
> simultaneously across three documents, all of them restating `tabs.md`
> § 1.  The **owner** is pinned now —
> `test_doc_claims.py::test_the_tab_count_is_stated_in_prose_and_true`
> compares § 1's prose against `TABS` — so a new tab cannot land without at
> least one document telling the truth.  **The captures themselves are still
> only a rule**: nothing can look at a PNG and count tabs in it.

> **Demo data convention.**  Every screenshot uses the project at
> `projects/BDT/` so the README reads as one continuous Au–BDT–Au
> example — the SMILES → junction → optimisation → spectrum →
> transport → results loop in one project tree.  Stage-3 optimised
> geometry is at
> `projects/BDT/structure/BDT-AuJunction_siestaStage1_optimized.xyz`
> (despite the `siestaStage1_optimized` filename — it is the latest
> usable geometry today).

## Pre-capture setup (do once)

1. `molbuilder serve foreground --port 8888` from the repo root.
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
| 2 | `tab-bar.png` | § Feature tour intro | `/molbuilder` | — | Zoom: just the top tab strip (**eight** tabs — the order comes from the one `TABS` list in `tabs.py`).  ~1200 × 60 px | Names + order of every tab |
| 3 | `sidebar-projects.png` | § Workflow + § Documentation | `/molbuilder` | Expand `BDT/` → `structure/`; cursor on `BDT-AuJunction_siestaStage1_optimized.xyz` so the paired `.molstruct.json` shows as a sidecar | Zoom: just the projects sidebar column.  ~360 × 700 px | Tree shape + the structure↔sidecar pairing |
| 4 | `molbuilder-workspace.png` | § 1 Molbuilder tab | `/molbuilder` | Load the BDT junction; one atom selected in the viewer to show the amber **shape glow** (the only selection highlight — halos were removed, see [`web/molview.md`](?doc=web/molview.md)) + sync to the atom list on the left | Zoom: the `/molbuilder` content area (no sidebar, no top tab strip).  ~1080 × 760 px | The 3-panel layout: atom list + viewer + commands stack |
| 5 | `structure-optimization-form.png` | § 2 Structure optimization | `/structure-optimization` | Pick `BDT-AuJunction_siestaStage1_optimized.xyz` from the sidebar; engine = SIESTA; default profile.  Issues panel will show INFO notices about Au-thiol detection | Zoom: the `/structure-optimization` content area (form on left, viewer on right, issues panel docked below).  ~1100 × 760 px | Schema-driven form + workflow-group cards + inline detection chip + issues |
| 6 | `spectrum-form.png` | § 3 Spectrum calculation | `/spectrum-calculation` | Pick `projects/BDT/spectrum/BDT-only/spectra.spectra.py` *if* present — else use any small molecule. Default form values | Zoom: the form panel only.  ~520 × 700 px | The vertical workflow-group layout (Profile / Stage / Budget) |
| 7 | `transport-form.png` | § 4 Transport calculation | `/transport-calculation` | Cite a finished junction attempt through **Choose junction attempt…** (a `run-N` folder); the viewer and chemistry analysis follow the citation | Zoom: card 1 (citation + viewer) + a slice of card 3's override lane.  ~1080 × 760 px | The composite's describe surface: the citation drives everything |
| 8 | `results-trajectory.png` | § 5 Results | `/results` | Open ONE stage of the BDT optimisation: pick `BDT_02_medium.molwatch.log` from the file picker (stages are separate runs; the inspector shows the one you pick) | Zoom: the content area: viewer top, frame strip + scrub slider mid, energy / force / SCF-residual plots stacked on the right.  ~1100 × 800 px | Live trajectory inspector, one stage |
| 9 | `results-spectra.png` | § 5 Results | `/results` | Open `projects/BDT/spectrum/BDT-only/spectra.spectra.json` | Zoom: the content area.  Show the Lorentzian-broadened spectrum chart + modes table + the 3-D viewer on a selected mode | The spectra inspector's two-panel layout |

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
