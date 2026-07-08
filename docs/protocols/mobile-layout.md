# Mobile layout — responsive-grid pattern

## Summary

Every `repeat(auto-fit, minmax(<N>px, 1fr))` in molbuilder CSS uses
the modern responsive-floor form

    grid-template-columns: repeat(auto-fit, minmax(min(<N>px, 100%), 1fr));

NOT the bare `minmax(<N>px, 1fr)`.  This is a hard rule for every
multi-column grid that has to survive a 360-px phone viewport
without producing a horizontal scrollbar.

Established 2026-06-14 by `tests/test_molbuilder_e2e.py::test_modify_layout_phone_width_no_horizontal_overflow` after the failure
class surfaced on `.workspace-grid` in `modify/style.css`.

---

## Why

CSS Grid's `minmax(<N>px, 1fr)` says "each track is AT LEAST `<N>`
pixels wide".  When the parent's content area is narrower than
`<N>` (e.g. a phone viewport whose body+main padding leaves
~316 px when `<N>` is 360), the track can't shrink — it forces the
parent to overflow and a horizontal scrollbar appears on every
phone load.

`minmax(min(<N>px, 100%), 1fr)` clamps the lower bound to whichever
is smaller — `<N>` OR the container's actual width.  When the
container is wider than `<N>`, behaviour is identical to the bare
form (track stays at `<N>` minimum).  When the container shrinks
below `<N>`, the track shrinks with it — no overflow.

This is supported in every browser that supports CSS Grid + CSS
Math Functions Level 1 (Chrome 79+, Firefox 75+, Safari 11.1+),
which we already require.

---

## Where the pattern is used

| File | Selector | Track `<N>` |
|---|---|---|
| `web/static/lib/form-schema.css` | `.param-grid` (default) | 340 px |
| `web/static/lib/form-schema.css` | `.param-grid` (≤1100 px) | 300 px |

If you add a new `repeat(auto-fit, minmax(...))` grid anywhere
under `molbuilder/web/static/`, use the `min()` floor.

---

## `.card-row` — asymmetric card rows (2026-07-08)

`repeat(auto-fit, minmax(...))` is for grids of **equal** tracks.  A row of
cards with **different** widths + grow-weights (e.g. a wide primary card beside
a narrower secondary one) uses the shared `.card-row` primitive in
`lib/page-shell.css` instead — a `flex-wrap` row, still **content-driven with no
media breakpoint**:

    .card-row { display: flex; flex-wrap: wrap; ... }
    .card-row > * { flex: var(--card-weight,1) 1 var(--card-basis,320px);
                    min-width: var(--card-min,300px); }

Each child sets three vars; the row keeps children side by side while their
`--card-min`s fit and **wraps** the overflow to a new row the instant they
don't — so a card can never be squeezed below its min-width and overflow INTO
its neighbour (the crush/overlap failure of a `grid-template` whose track min is
smaller than the card's content min).

Any tab can reuse it: add `class="card-row"` and set `--card-weight/-basis/-min`
per card.  Used by `.workspace-grid` (modify): molview card `2.4 / 640 / 340`,
op-controls card `1 / 320 / 300`.  Guarded by
`tests/…::test_workspace_cards_never_overlap`.

**Don't** re-introduce a fixed `grid-template-columns: A B` + a `@media
(max-width: …)` stack for a card row — the media number never matches the real
content minimums and leaves an overlap gap between them.

---

## Why not just a media query?

Media queries respond to the **viewport** width, not the container
width.  A workspace panel rendered into a narrow column inside a
desktop-width viewport can be 280 px wide even though the viewport
is 1440 px — the breakpoint never fires, but the grid still
overflows.  `min()` reads the container width directly, so it
works everywhere `auto-fit` does.

The 720-px breakpoint in `form-schema.css` that force-flattens
`.param-grid` to a single column on actual phones is still
load-bearing for visual hierarchy (label-above-input vs.
label-beside-input) — keep it.  The `min()` floor is a safety
net for the in-between cases media queries can't see.

---

## Anti-patterns

* **Don't** write `minmax(<N>px, 1fr)` without the `min()` floor.
  The next renderer to drop your grid into a narrow panel
  rediscovers the bug.
* **Don't** rely on `@media (max-width: …) { grid-template-columns:
  1fr }` alone — it doesn't fire when the container is narrow
  inside a wide viewport.
* **Don't** add `overflow-x: hidden` to mask the symptom — the
  underlying widths are still wrong and the user can still scroll
  via touch / trackpad.
