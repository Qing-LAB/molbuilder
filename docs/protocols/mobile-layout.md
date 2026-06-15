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
| `web/static/modify/style.css` | `.workspace-grid` | 360 px |
| `web/static/lib/form-schema.css` | `.param-grid` (default) | 340 px |
| `web/static/lib/form-schema.css` | `.param-grid` (≤1100 px) | 300 px |

If you add a new `repeat(auto-fit, minmax(...))` grid anywhere
under `molbuilder/web/static/`, use the `min()` floor.

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
