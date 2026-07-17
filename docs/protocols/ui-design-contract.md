# UI design contract — layout, CSS & module boundaries

The **structural** counterpart to [`web-ui-coherence.md`](web-ui-coherence.md).
That doc governs *data* coherence (two surfaces reporting on the same structure
must agree). **This** doc governs *how UI is built* — the layout framework,
CSS architecture, responsive rules, **and the JS module boundaries** every tab
follows, so the web UI stays one coherent system instead of N one-off pages.

The through-line of every rule below is **one owner, one door**: one home per
visual pattern (§2.1), per token (§1), per API surface and per in-memory model
(§7). A second copy — a duplicated rule, a parallel token scheme, a raw `fetch`
beside the wrapper — is the drift these rules exist to catch.

If you are adding or changing any UI, read the **Checklist** (§9) first.

---

## 1. Design tokens are the only source of colour / spacing / type

`lib/tokens.css` defines every colour, and the spacing/gap scale, as CSS custom
properties (`--bg-card`, `--border-soft`, `--text-primary`, `--accent`,
`--success/-error/-warning`, `--gap`, …).

- **Never hardcode** a hex colour, or a raw spacing literal, in a component.
  Use `var(--token)`. A `#1d2128` in a component file is a bug.
- A component that needs a new colour/spacing adds a **token**, not a literal.
- Fallbacks are allowed for embed-outside-the-shell safety:
  `var(--accent, #6ba6ff)`.
- **One token vocabulary.** `tokens.css` names are canonical — `--bg-card`,
  `--text-secondary`, `--success`, `--radius`/`--radius-sm`. Do **not** introduce
  a parallel scheme for the same concept (`--surface-elevated` for a card bg,
  `--fg` for text, `--radius-md`, `--ok`) — that is value-duplication one level
  up, and the two schemes drift. An embeddable module may write
  `var(--token, fallback)` so it renders in a foreign host, **but the token must
  still be defined in `tokens.css`**. A `var(--x, …)` whose `--x` exists nowhere
  is a **phantom token**: it silently *always* uses the fallback, so the palette
  no longer controls it — and worse, if the *same* phantom name carries **different**
  fallbacks at different sites (as `--border-subtle` does: rgba-white at .06/.08/.12/
  .18; `--radius-md` at 8px vs 10px; `--text-tertiary` at #777 vs #7c7c7c), the sites
  only *look* coordinated — defining the token would snap them together and shift most
  of them.
  *Known debt (CSS-migration step 3, open):* the embed modules — `mol-viewer-embed`,
  `results/bundle-handoff`, `system-load-monitor`, `trajectory-inspector` — use a
  **translucent-overlay** token family (`--surface-*`, `--border-subtle`) plus an
  embed severity set (`--ok/--warn/--bad`), none defined in `tokens.css`. The overlay
  *approach* is legitimate for embeddables (a translucent tint adapts to any host
  background, like the `var(--token, fallback)` pattern itself) — the defects are that
  the tokens are **undefined** and **internally inconsistent**. Resolve by either
  promoting them into `tokens.css` as a documented **embeddable-overlay tier**
  (consolidated to one value each), or collapsing onto the canonical palette — a
  design decision, not a mechanical rename.

## 2. CSS lives in layers — put a rule in the lowest layer that fits

| Layer | File(s) | Owns |
|---|---|---|
| **Tokens** | `lib/tokens.css` | colour / spacing / type variables — nothing else |
| **Shell + primitives** | `lib/page-shell.css` | cross-tab chrome (`body`, `.card`) + **reusable layout primitives** (`.card-row`) |
| **Module** | `lib/<module>.css` (`selection-panel.css`, `molview/fused-layout.css`, `form-schema.css`) | one self-contained component; may use `@container`; exposes tuning **vars** |
| **Page** | `<tab>/style.css` | *composition only* — max-width/centring, which primitives a tab uses, per-instance var values. **No new primitives here.** |

Rule: a pattern used by **more than one tab** is a primitive in `page-shell.css`,
not copy-pasted per page. A pattern is a **module** if it's a component; a
**page** rule only wires modules/primitives together.

### 2.1 A repeated visual pattern is ONE shared class — never a duplicated rule

Tokens (§1) stop duplicated *values*; this stops duplicated *rules*. If two elements
should look the same — a caption, a pill, a mini-control — they share **one class**.
You never re-declare the same `font:` / `padding:` / `text-transform:` block under a
second selector. Two selectors with the same declarations is the drift this rule exists
to catch.

- **Captions / small labels** (uppercase 11px muted) → **`.selection-mini-label`**.
  Do NOT add `.foo-label { font: 600 11px…; text-transform: uppercase }`.
- **Before writing a rule, grep for the pattern** (`grep "600 11px"`,
  `grep "text-transform: uppercase"`, `grep "aspect-ratio"`). If a class already has
  it, reuse the class. If the block is already copied in ≥2 places, promote it to one
  shared class and delete the copies.
- A genuinely unique one-off gets its own class; a pattern that appears twice does not.

### 2.2 No page stylesheet is a disguised shared library

The layer table's "Page = composition only" has one notorious violation worth
naming so it isn't copied: `static/style.css` is headed *"Build page styles"* but
actually holds shared form-page **components** — `.app-grid`, `.auto-detect-*`,
`.viewer-*` / `.card-viewer`, `.issues-panel`, the form-control + `button` base,
`.status`, `.hint`. Spectra and Transport must link the **whole** index sheet —
dragging in inert index-only rules (`#generate-*`, `.tabs/.tab-btn`) — just to
reach those shared parts.

The target end-state: shared form-page components live in a **named shared sheet**
(shell/primitive or a `lib/form-components.css`), index-only rules stay in the page
sheet, and each page links only what it uses. Until that migration lands, the
holding rule is: **never add a new shared component to `style.css`** — put it in the
right module/shell layer (§2). A page sheet that a *different* page has to import is
exactly the smell this rule catches.

### 2.3 Namespaces — every class and private token has exactly one owner

Extensibility means adding a module without colliding with, or silently overriding,
another. This is already the norm for the newer modules — make it **universal**, and
the drift in §2.1/§2.2/§1 stops being possible by construction.

- **Module classes carry the module prefix.** `ps-*` (projects sidebar),
  `selection-*`, `molview-*` / `mvf-*`, `bundle-*`, `system-*`, `convergence-*`,
  `md-*` (markdown inspector), `schema-*` (form), `region-*`, `source-*`. A class
  with **no** prefix must be a *deliberate global primitive* (`.card`, `.card-row`)
  living in `page-shell.css` — never a component that merely happens to sit in a page
  sheet. The un-namespaced shared bits still in `style.css` (`.status`, `.hint`,
  `.issues-panel`, `.viewer-*`, `.auto-detect-*`, `.app-grid`) are **debt**: on
  migration each moves to either a `page-shell` primitive (if truly global) or a
  prefixed module class.
- **Two token tiers, no third.** The **global palette** in `tokens.css` is the single
  unprefixed vocabulary (`--bg-*`, `--text-*`, `--border-*`, `--accent-*`,
  `--radius-*`, `--space-*`, `--success` / `--error` / `--warn-*`). **Module-private**
  tokens are prefixed and defined by that module (`--ps-*`, `--sp-*`). A `--surface-*` /
  `--fg` / `--radius-md` scheme for a palette concept is not a third tier — it is drift
  (§1); map it onto the palette.
- **One class = one owner = one home.** A selector is defined in exactly one file —
  the module (or shell) that owns it. The four `.status` copies (`page-shell` + `style`
  + `modify` + `spectra`, with drifting modifier sets and *two* different `.error`
  tokens) are the anti-pattern this rule retires: consolidate to a single owner.
- **JS mirrors this.** Each module hangs off one namespace
  (`window.molbuilder.<module>`) and exposes one door (§7); no two modules reach into
  the same state.

## 3. Responsive layout is CONTENT-driven, never viewport-magic-numbers

The failure mode we keep hitting: a hardcoded `@media (max-width: 900px)` that
doesn't match the content's real minimum widths, leaving a band where cards
overlap or overflow.

- **Reflow from the content's own minimum widths**, so the layout adapts with
  no arbitrary breakpoint:
  - **Asymmetric card rows** → the `.card-row` primitive (§4).
  - **Equal-track grids** → `repeat(auto-fit, minmax(min(<N>px, 100%), 1fr))`
    (see [`mobile-layout.md`](mobile-layout.md)).
- **Embeddable components query their OWN width, not the viewport.** Use
  `container-type: inline-size` + `@container`, so the component is correct at
  full-page width AND inside a small card (e.g. the fused molview card adapts
  side-by-side ↔ stacked by its own width — the Results inspector reuses it at a
  fraction of the page width). Reserve `@media` for genuinely page-global chrome.
- If a numeric threshold is truly unavoidable, **derive it from the content
  minimums and name it** (a CSS var / a comment showing the sum) — never a bare
  magic number.

## 4. Layout primitives — the shared vocabulary

Reuse these; don't reinvent them per tab.

- **`.card`** (`page-shell.css`) — the canonical surface (bg, border, radius,
  padding). Every panel is a `.card`.
- **`.card-row`** (`page-shell.css`) — a responsive row of cards of *different*
  widths. Children set three vars; the row wraps a card to its own line the
  instant the minimums stop fitting (no media query, no overlap):

  ```css
  .card-row > * { flex: var(--card-weight,1) 1 var(--card-basis,320px);
                  min-width: var(--card-min,300px); }
  ```
  ```html
  <div class="card-row">…</div>
  ```
  ```css
  .primary-card   { --card-weight: 2.4; --card-basis: 640px; --card-min: 340px; }
  .secondary-card { --card-weight: 1;   --card-basis: 320px; --card-min: 300px; }
  ```
- **Equal grids** — `repeat(auto-fit, minmax(min(<N>px,100%), 1fr))`
  ([`mobile-layout.md`](mobile-layout.md)).
- **Fused card** (`molview/fused-layout.css`) — a self-adapting card holding a
  primary square viewer + a foldable side panel; switches side-by-side ↔ stacked
  by its own `@container` width. The reference implementation of §3 + §5.

## 5. Sizing rules that avoid the traps we've hit

- **A component tunes itself through named CSS vars, not literals baked into the
  layout rule.** `.card-row` reads `--card-*`; the molview viewer reads
  `--viewer-edge/--viewer-min`. The layout rule stays literal-free; a consumer
  overrides the var. This is what makes a primitive reusable.
- **Square media** (a 3D viewer, a plot) → `aspect-ratio: 1/1`, bounded by
  `min-width` (floor) and `max-height: min(60vh, <cap>)` (so it never fills the
  full width or blows up the height).
- **A card can never be squeezed below its `min-width`.** A neighbour’s track /
  flex-basis must respect it, or it overflows into the neighbour (the
  crush/overlap bug). `.card-row` enforces this by construction.
- **Don't pin a fixed `height` where content varies** (e.g. a viewer that's
  empty until a structure loads) — it strands controls off-screen. Cap with
  `max-height`, or let the primary element own the height.

## 6. Display vs. edit surfaces (bridges to the data contract)

- A **display** surface mirrors in-memory data through the module's read
  accessors and is **read-only** — no write path. It tags a never-set value
  `"(default)"` via the `{ value, isDefault }` accessor shape. (MolView Cell page:
  [`structure-periodicity.md`](structure-periodicity.md) § 3b.)
- **Editing is explicit and separate** — staged inputs committed by an Update
  action, never auto-committed from a display surface.
- In-memory data is reached **only** through the module's accessor API, which
  returns **defensive copies** ([`workspace-contract.md`](workspace-contract.md)
  § 1.2.1) — a UI surface can't mutate the store by holding a returned value.

## 7. Module & data boundaries — one door per concern (the JS side)

The CSS rules above keep the *look* one system; these keep the *code* one system —
the same "one owner, one door" principle applied to network, in-memory data, and
persistence. Each boundary is authoritative in its own doc; this is the
cross-cutting summary a UI change must respect (and the checklist, §9, enforces).

- **One fetch caller per API surface.** UI code never calls `fetch()` for a shared
  backend API directly — it goes through the module's HTTP wrapper, so
  error / abort / caching normalise in **one** place. `/api/files/*` +
  `/api/projects/*` → `lib/projects/api.js` (the uniform `{ok, error, aborted}`
  envelope + `cache:"no-store"`). A raw `fetch("/api/files/…")` in a consumer is the
  bug this catches ([`projects-sidebar.md`](projects-sidebar.md) Principle 6); a
  source guard (`test_projects_api_envelope_js.py`) pins it *within the `projects/`
  subtree*.
  *Known gap (CSS-migration step 4, open):* `api.js` is an **ES module**, so only
  ES-module code (`projects/*`) can `import` it. The classic-script consumers —
  `spectra/viewer.js`, `viewer.js` (old Build), `inspectors/{source,registry,markdown}`,
  `structure/sidecar-labels.js`, `molview/_selection-store-impl.js` — still raw-fetch
  `/api/files/read|read_range|write`, each hand-rolling its own error handling. The
  target is a global bridge (`window.molbuilder.filesApi = { read, write, readRange,
  stat }` exported from `api.js`) that classic scripts call, so **all** consumers route
  through the one envelope. Until then the rule holds only for module code.
- **In-memory data has one owner; reach it only through accessors.** The live
  structure / atoms / frames model is owned by `molview.data`
  ([`molview-module.md`](molview-module.md)). A UI surface reads it through
  accessors that return **defensive copies** (§6), never by holding a store
  reference. Two panels showing the same structure agree because they read the
  *same* door, not parallel copies.
- **Format-blind layers stay format-blind.** The workspace does exactly two things —
  session state + concealed file **bytes** access — and never interprets
  structure or format; the *consumer* owns data + structure + format
  ([`workspace-contract.md`](workspace-contract.md)). Persistence is **push-only**
  and **write-ordered**: a later state write for the same index can't be overtaken
  by a stale one (serialised write-chain).
- **A concealed module is a complete seal.** MolView, VibrationView, and the
  projects sidebar each own their DOM, CSS, and API end-to-end and drop in as one
  unit; a host *wires* them, never reaches inside. A sibling capability is a **new
  sealed module** (VibrationView is MolView's sibling, not a branch inside it), not
  a flag bolted onto an existing one.
- **The web UI is a thin wrapper over the Python API.** A tab *composes*
  subcommands / accessors; business logic and science live in the Python layer,
  not re-implemented in JS ([`architecture.md`](../architecture.md)).

## 8. Per-module UI systems — the spec each panel follows (don't drift)

Each module that owns a UI panel declares **here** the exact system it uses — its CSS
file, tokens, layout mechanism, and shared classes — so a change stays *inside* that
system instead of inventing a parallel one. **Adding a module with a panel? Add a
subsection.** Editing one? Read its subsection first.

### 8.1 MolView — the fused card · `lib/molview/fused-layout.css`

- **Layout:** `.molview-card` is `container-type: inline-size`, with ONE source of truth
  for size in three named vars — `--viewer-edge` (`min(60vh, 560px)`), `--viewer-min`
  (`320px`), `--panel-min` (`320px`). A single `@container (max-width: 664px)` query
  (= `viewer-min + handle + panel-min`, shown in the comment) flips side-by-side ↔
  stacked. **No `@media`, no other breakpoint, no second magic number.**
- **Viewer:** a 1:1 square sized to `--viewer-extent` — a derived value the `.molview-body`
  computes once (`max(--viewer-min, min(--viewer-edge, 100cqw − --panel-min − --fold-w))`)
  and BOTH the viewer and the panel key their height off, so the two bottom-align at every
  width with **no JS** and no fixed height. `100cqw` is the card's own content width (content-
  driven, §3), so the viewer never fills the card (leftover → margin/panel).
- **Stacked (`@container` under 664):** viewer + panel share one `--stack-extent`
  (`min(--viewer-edge, 100cqw)`) for a matching width; the fold handle becomes a **compact,
  centred grip** (hugs its chevron), never a full-width bar — so it can't read as wider than
  the columns. No `vh`/`px` magic; the panel height reuses `--viewer-edge` via `--stack-extent`.
- **Viewer-controls bar:** the view toggles (isolate / k-grid) are `.viewer-toggle`
  (**one shared class both molviews use**), mounted by `molview.mountViewControls`
  ([`molview-module.md`](molview-module.md)) — never re-styled per page.
- **Reuse:** Modify and every Results structure card mount the SAME card; a card change
  is one edit here, not per consumer.

### 8.2 Selection / Cell panel · `lib/selection-panel.css`

- **Tokens:** the panel's own `--ps-*` scale (`--ps-fg`, `--ps-fg-dim`, `--ps-bg-deep`,
  `--ps-border`, `--ps-hover`, `--ps-selected-*`). Panel rules use these — never raw hex.
- **Captions:** `.selection-mini-label` is THE small-caps caption (k-grid label,
  "Combine", "Target"). **Never re-declare the 600/11px/uppercase block** (§2.1).
- **Header tabs:** `[Selection|Cell]` = `.panel-page-switch` / `.panel-page-option`
  (radio-driven page swap); pages `#panel-page-selection` / `#panel-page-cell`.
- **Cell readout:** display-only (§6) — `#cell-*-value` spans + the `.cell-matrix` 3×3
  grid; filled by `renderCell` from the `ws.get*Info()` accessors.
- **Display-only:** the panel never edits periodicity, and the view TOGGLES are NOT here
  — they live in the viewer bar (§8.1). Editing periodicity is the Modify Cell op-tab.

## 9. Checklist — adding or changing a UI surface

1. **Tokens** — colours/spacing come from `var(--token)`; no literals.
2. **No duplicate rule** (§2.1) — grep the pattern first; reuse the shared class
   (`.selection-mini-label`, `.viewer-toggle`, `.card`) instead of re-declaring it.
3. **Reuse a primitive** — `.card`, `.card-row`, the auto-fit grid, the fused
   card. Don't hand-roll a layout another tab already has.
4. **Content-driven reflow** — wrap/stack from content min-widths; **no** new
   `@media (max-width: …)` for a card row.
5. **Embeddable? Use `@container`**, not `@media` — so it's correct at any width.
6. **Expose tuning as vars**, keep the layout rule literal-free.
7. **Right layer** (§2) — primitive → `page-shell.css`; component → module CSS;
   only composition in `<tab>/style.css`. **Never grow `style.css` with a new
   shared component** (§2.2).
8. **Follow the module's §8 subsection** — if the panel has one, stay in its system.
9. **Display surfaces are read-only** through accessors (§6); edits are explicit.
10. **One door for data + network** (§7) — no raw `fetch` for a shared API (go
    through `api.js`); reach in-memory data only via the owning module's accessors.
11. **Add a layout regression test** — e.g. sweep widths and assert no
    overlap/overflow (`test_workspace_cards_never_overlap`,
    `test_fused_no_overflow_when_squeezed`).

## 10. What this document does NOT cover

- **Data/logic coherence** (two surfaces agreeing about a structure) →
  [`web-ui-coherence.md`](web-ui-coherence.md).
- **The exact token palette** → `lib/tokens.css` is authoritative.
- **The form/validator workflow-card routing** → `web-ui-coherence.md` § Rule 2.

## References

- `lib/tokens.css` — the token palette.
- `lib/page-shell.css` — `.card`, `.card-row`.
- `lib/molview/fused-layout.css` — the fused-card `@container` reference.
- [`mobile-layout.md`](mobile-layout.md) — auto-fit grids + `.card-row`.
- [`web-ui-coherence.md`](web-ui-coherence.md) — the data-coherence companion.
- [`structure-periodicity.md`](structure-periodicity.md) § 3b — display-vs-edit.
- [`workspace-contract.md`](workspace-contract.md) § 1.2.1 — accessor / defensive-copy contract; push-only persistence.
- [`molview-module.md`](molview-module.md) — `molview.data` owns the in-memory model + accessor API.
- [`projects-sidebar.md`](projects-sidebar.md) Principle 6 — `api.js` is the sole fetch caller.
- [`../architecture.md`](../architecture.md) — module roles/layers; web UI as a thin wrapper over the Python API.
