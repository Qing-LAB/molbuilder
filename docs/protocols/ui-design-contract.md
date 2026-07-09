# UI design contract — layout & CSS framework

The **structural** counterpart to [`web-ui-coherence.md`](web-ui-coherence.md).
That doc governs *data* coherence (two surfaces reporting on the same structure
must agree). **This** doc governs *how UI is built* — the layout framework,
CSS architecture, and responsive rules every tab follows, so the web UI stays
one coherent system instead of N one-off pages.

If you are adding or changing any UI, read the **Checklist** (§7) first.

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

## 7. Per-module UI systems — the spec each panel follows (don't drift)

Each module that owns a UI panel declares **here** the exact system it uses — its CSS
file, tokens, layout mechanism, and shared classes — so a change stays *inside* that
system instead of inventing a parallel one. **Adding a module with a panel? Add a
subsection.** Editing one? Read its subsection first.

### 7.1 MolView — the fused card · `lib/molview/fused-layout.css`

- **Layout:** `.molview-card` is `container-type: inline-size`, with ONE source of truth
  for size in three named vars — `--viewer-edge` (`min(60vh, 560px)`), `--viewer-min`
  (`320px`), `--panel-min` (`320px`). A single `@container (max-width: 664px)` query
  (= `viewer-min + handle + panel-min`, shown in the comment) flips side-by-side ↔
  stacked. **No `@media`, no other breakpoint, no second magic number.**
- **Viewer:** a 1:1 square (`aspect-ratio`), floored by `--viewer-min`, capped by
  `--viewer-edge`; it never fills the card (leftover width → panel).
- **Viewer-controls bar:** the view toggles (isolate / k-grid) are `.viewer-toggle`
  (**one shared class both molviews use**), mounted by `molview.mountViewControls`
  ([`molview-module.md`](molview-module.md)) — never re-styled per page.
- **Reuse:** Modify and every Results structure card mount the SAME card; a card change
  is one edit here, not per consumer.

### 7.2 Selection / Cell panel · `lib/selection-panel.css`

- **Tokens:** the panel's own `--ps-*` scale (`--ps-fg`, `--ps-fg-dim`, `--ps-bg-deep`,
  `--ps-border`, `--ps-hover`, `--ps-selected-*`). Panel rules use these — never raw hex.
- **Captions:** `.selection-mini-label` is THE small-caps caption (k-grid label,
  "Combine", "Target"). **Never re-declare the 600/11px/uppercase block** (§2.1).
- **Header tabs:** `[Selection|Cell]` = `.panel-page-switch` / `.panel-page-option`
  (radio-driven page swap); pages `#panel-page-selection` / `#panel-page-cell`.
- **Cell readout:** display-only (§6) — `#cell-*-value` spans + the `.cell-matrix` 3×3
  grid; filled by `renderCell` from the `ws.get*Info()` accessors.
- **Display-only:** the panel never edits periodicity, and the view TOGGLES are NOT here
  — they live in the viewer bar (§7.1). Editing periodicity is the Modify Cell op-tab.

## 8. Checklist — adding or changing a UI surface

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
   only composition in `<tab>/style.css`.
8. **Follow the module's §7 subsection** — if the panel has one, stay in its system.
9. **Display surfaces are read-only** through accessors (§6); edits are explicit.
10. **Add a layout regression test** — e.g. sweep widths and assert no
    overlap/overflow (`test_workspace_cards_never_overlap`,
    `test_fused_no_overflow_when_squeezed`).

## 9. What this document does NOT cover

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
- [`workspace-contract.md`](workspace-contract.md) § 1.2.1 — accessor / defensive-copy contract.
