# T3 — CSS/UI holistic audit

**Date**: 2026-06-26
**Auditor**: Claude (general-purpose subagent, T3)
**Scope**: every `*.css` file under `molbuilder/web/static/` (excluding `vendor/`)

## Summary

- Total CSS files audited: **16** (+ 2 vendored CodeMirror files excluded)
- Total lines of CSS: **7 049** (excluding vendor)
- Token coverage estimate: ~80% color, ~55% spacing, ~5% font-family
- `!important` count: **2** (1 justified, 1 partially justified)
- `z-index` distinct values: **11** (1, 5, 6, 10, 50, 85, 90, 95, 100, 1000, 99999) — no documented stack-context plan beyond projects-sidebar's local commentary
- Naming conventions in play: 3 (kebab-case dominant, `is-*` state classes, namespaced tab/component prefixes); BEM/underscores effectively absent (<1%)
- Distinct transition durations: **10** values (39× `0.12s` + 36× `120ms` are the same thing in two notations → token candidate)

The token system itself (`lib/tokens.css`) is well-organised and documented. The drift sits in **(a)** the new run-history panel in `projects-sidebar.css` (color literals + raw `rem` values for 11 new components), **(b)** undefined tokens referenced via `var(--x, fallback)` in `system-load-monitor.css` and `mol-viewer-embed.css`, and **(c)** the ad-hoc monospace stacks repeated 25+ times across files instead of `var(--font-mono)`.

---

## Token system audit

### Tokens that exist (lib/tokens.css)

- **Surfaces**: `--bg-page`, `--bg-card`, `--bg-card-hover`, `--bg-input`, `--bg-input-focus`, `--bg-fieldset`, `--bg-code`
- **Borders**: `--border-strong`, `--border-soft`, `--border-dashed`
- **Text**: `--text-primary`, `--text-secondary`, `--text-muted`, `--text-mono`
- **Accents/states**: `--accent`, `--accent-hover`, `--accent-strong`, `--accent-bg`, `--accent-glow`, `--header-gradient-{top,bottom}`, `--success`, `--error`, `--warning`, `--warning-bg`, `--warn-soft`, `--text-on-accent`
- **Shadows**: `--shadow-card`, `--shadow-input`
- **Layout**: `--radius`, `--radius-sm`, `--radius-lg`, `--gap`
- **Spacing scale**: `--space-xs` (4px) … `--space-xl` (28px)
- **Typography scale**: `--font-mono`, `--text-xs` … `--text-xl`, `--line-*`, `--weight-*`
- **Component palettes**: `--ps-*` (sidebar, 17 tokens), `--sp-*` (selection panel, 14 tokens), workflow-group `--group-*-accent` (3 tokens)

### Token coverage gaps — undefined tokens silently fall back to literals

These `var(--x, fallback)` calls reference tokens that **do not exist** in `tokens.css`. The fallback path is the one actually rendered; the token name is aspirational. Each is a one-line fix to the token file but a real risk: a future theme retune of the named token does nothing.

| File:Line | `var(--name)` referenced | Defined? | Falls back to |
|---|---|---|---|
| `lib/system-load-monitor.css:34` | `--surface-elevated` | NO | `rgba(20,24,28,0.55)` |
| `lib/system-load-monitor.css:36, 94, 136, 193, 216` | `--border-subtle` | NO | `rgba(255,255,255,0.06–0.18)` (4 different alphas) |
| `lib/system-load-monitor.css:33` | `--space-md` fallback `12px` | YES (=14px) | inconsistent fallback |
| `lib/system-load-monitor.css:43` | `--space-md` fallback `16px` | YES (=14px) | also inconsistent (different from the file's own line 33) |
| `lib/system-load-monitor.css:127–129` | `--ok`, `--warn`, `--bad` | NO | `#4ad991`, `#f0c040`, `#ef5b5b`. Tokens `--success`, `--warning`, `--error` exist but are not aliased. |
| `lib/mol-viewer-embed.css:184, 256` | `--accent-on` | NO | `#14171c`. `--text-on-accent` IS defined with the same intent and value. Rename mismatch. |
| `lib/projects-sidebar.css:1511` | `--ps-bg-2` | NO | `#2a2a2a`. `--ps-bg-tile` (#1c1c1c) and `--ps-bg-deep` (#141414) are the actual sidebar bg-step tokens. |

**Verdict**: 7 undefined token references in production CSS. These are the highest-leverage cleanups — each is one line in tokens.css.

### Theme drift — color literals not tokenised at all

#### projects-sidebar.css run-history / checkpoint panel (lines 1340–1590, added today's commit)
The new run-history UI introduces **11 hardcoded hex colors** for state badges, ref-chips, and the advisory banner — none tokenised:

| File:Line | Literal | Role | Suggested |
|---|---|---|---|
| `projects-sidebar.css:1371–1373` | `#2d4a2d` / `#b8e0b8` / `#4a7a4a` | sensor "clean" | `--ps-state-clean-{bg,fg,border}` |
| `projects-sidebar.css:1376–1378` | `#4a3d2d` / `#e0c898` / `#7a6539` | sensor "dirty" / advisory banner (also duplicated at L1531–1533) | `--ps-state-dirty-*` |
| `projects-sidebar.css:1381–1383` | `#3a3a3a` / `#aaa` / `#555` | sensor "uninit" | `--ps-state-uninit-*` |
| `projects-sidebar.css:1386–1388` | `#4a2d2d` / `#e0a8a8` / `#7a4a4a` | sensor "error" | `--ps-state-error-*` |
| `projects-sidebar.css:1420, 1483, 1559` | `#fff` | button text on accent | `--ps-accent-fg` already defined (`#cfe0fb`) — but #fff is used instead, inconsistent with other accent-text in same file |
| `projects-sidebar.css:1463` | `#88aacc` | SHA color | `--ps-fg-sha` |
| `projects-sidebar.css:1488` | `#7a5b2d` | ref-chip kind=tag | `--ps-ref-tag-bg` |
| `projects-sidebar.css:1491` | `#2d4a5a` | ref-chip kind=branch | `--ps-ref-branch-bg` |
| `projects-sidebar.css:1523` | `#e0a868` | action=restore color (same hex as `.md-dirty-flag` at markdown.css:39 — should be tokenised together) | `--accent-restore` |

#### lib/inspectors/markdown.css (new today)
| File:Line | Literal | Should be |
|---|---|---|
| `markdown.css:39` | `#e0a868` | shared "dirty/restore amber" token (also at projects-sidebar.css:1523) |
| `markdown.css:46` | `#fff` | accent-fg token |
| `markdown.css:115` | `#e0c898` | inline-code fg — same hex as sensor-dirty color, coincidence not pattern |

#### projects-sidebar.css raw rgba scattered (older code)
| Line | Literal | Notes |
|---|---|---|
| 145 | `rgba(255,255,255,0.06)` | "neutral overlay" — recurs in 4 places |
| 177, 269, 329, 495, 802, 834, 1269, 1297, 1316 | various `rgba(0,0,0,0.25–0.6)` box-shadows / scrims | 10 distinct alphas; no `--shadow-*` for these |
| 575, 582, 971, 979 | `rgba(74,158,255,…)` and `rgba(79,122,184,…)` | accent at low alpha — should derive from `--ps-accent` via `color-mix()` or `--accent-bg`-style tokens |

#### lib/trajectory-inspector.css warn-pastel
Lines 650–698 mark `#fff7e0` / `#6b4a00` / `#fffaef` / `#8c6d20` / `#f5e8c2` as `/* exempt: warn-pastel theme */`. Comment-as-policy is fine here (intentional pastel for a "soft warning" review surface), but **the exemption isn't enforced**: there's no convention saying "if you want pastel, use these tokens." Suggest promoting to `--warn-pastel-{bg,fg,border}` so the policy is mechanical, not honour-based.

---

## Magic-number top 20

Raw counts of distinct `px`/`rem` literals appearing across all CSS (excluding vendor). The "tokenised" column is "Y" if a token of that value already exists in `tokens.css`.

| Rank | Literal | Count | Token exists? | Notes |
|---|---|---|---|---|
| 1 | `1px` | 232 | n/a | border width — universal, fine as literal |
| 2 | `0.4rem` | 79 | NO | new run-history panel + form-schema; ~6.4px. No token covers this. |
| 3 | `6px` | 76 | NO | between `--space-xs`(4) and `--space-sm`(8); ambiguous |
| 4 | `0.5rem` | 73 | partial | ~8px = `--space-sm`; but written as rem not token |
| 5 | `4px` | 71 | YES (`--space-xs`) | tokenisable; widespread literal use |
| 6 | `0.6rem` | 61 | NO | ~9.6px; no token |
| 7 | `3px` | 56 | NO | between xs and sm; ambiguous |
| 8 | `12px` | 50 | NO | between sm(8) and md(14); ambiguous |
| 9 | `8px` | 45 | YES (`--space-sm`) | tokenisable |
| 10 | `2px` | 44 | NO | border/outline; usually fine |
| 11 | `0.8rem` | 43 | NO | font-size ≈ 12.8px; no `--text-*` matches exactly |
| 12 | `0.3rem` | 42 | NO | ~4.8px |
| 13 | `0.85rem` | 41 | NO | font-size ≈ 13.6px |
| 14 | `1rem` | 29 | NO | spacing baseline; could alias `--space-md` |
| 15 | `14px` | 27 | YES (`--space-md`/`--gap`) | tokenisable |
| 16 | `0.7rem` | 25 | NO | |
| 17 | `10px` | 24 | NO | |
| 18 | `0.35rem` | 24 | NO | one-off micro-spacings |
| 19 | `0.25rem` | 24 | NO | |
| 20 | `0.75rem` | 20 | NO | |

**Pattern**: the codebase has two parallel spacing systems — `px` tokens in `tokens.css` (4/8/14/20/28) and ad-hoc `rem` literals in component CSS (0.25/0.3/0.35/0.4/0.5/0.6/0.7/0.75/0.8/0.85). The run-history panel and markdown inspector added today are pure-rem; the older form-schema/trajectory code is mostly px. There is no token rem-scale; nothing forces consistency.

---

## `!important` inventory

| File:Line | Selector | Justified? | Notes |
|---|---|---|---|
| `lib/projects-sidebar.css:215` | `.ps-collapsed-handle { display: none !important; }` inside `@media (max-width: 640px)` | partial | Comment above (L210–212) explains: narrow-viewport drawer takes over below 640, so the desktop handle must lose. Reasonable. Could equally be solved with `:not(.is-narrow-viewport)` on the L207 rule. |
| `results/style.css:505` | `.result-list-bar[hidden] { display: none !important; }` | YES | Inline comment L502–504 documents the reason: gated on >= 1 results; the bar is a flex child and a non-important `display:none` was being overridden by an enclosing `display: flex` cascade. Clean defensive use. |

**Verdict**: zero abuse. The codebase has discipline here.

---

## z-index map

| Value | File:Line | Selector | Stack-context purpose |
|---|---|---|---|
| 1 | `lib/selection-panel.css:311` | `.region-tag` | local elevation inside selection panel |
| 1 | `lib/trajectory-inspector.css:479` | `.chip` | chip overlay on inspector |
| 1 | `spectra/style.css:375` | `.spectrum-tooltip` | tooltip over plot |
| 5 | `lib/projects-sidebar.css:49` | `.projects-sidebar` | sidebar baseline |
| 5 | `lib/selection/measurement-chip.css:35` | `.measurement-chip` | chip above viewer canvas |
| 6 | `lib/projects-sidebar.css:167` | `.ps-collapsed-handle` | tab strip above sidebar |
| 10 | `lib/projects-sidebar.css:76` | `.ps-resize-handle` | resize handle on top of sidebar contents |
| 50 | `lib/projects-sidebar.css:490` | `.ps-entry-action-menu` | context menu inside sidebar |
| 85 | `lib/projects-sidebar.css:1315` | drawer scrim (narrow viewport) | below drawer, above page |
| 90 | `lib/projects-sidebar.css:1268` | drawer panel (narrow viewport) | the slide-in drawer |
| 95 | `lib/projects-sidebar.css:1286` | drawer close button | above drawer |
| 100 | `lib/projects-sidebar.css:784` | `.ps-file-preview-modal` | file preview modal (inline comment: "above sidebar (5) + everything else") |
| 1000 | `lib/mol-viewer-embed.css:201` | viewer toolbar floating button | embed-internal stack |
| 99999 | `lib/mol-viewer-embed.css:474` | `.mol-viewer-export-modal` | "top of everything, including host page" |

**Observations**:
1. The only **partially-documented** stacking plan is in `projects-sidebar.css` (commentary at L1209–1262); everything else is uncoordinated.
2. The jump from `100` (sidebar modal) to `1000` (embed toolbar) to `99999` (embed export modal) is intentional — the embed is mounted on third-party sites and must beat their z-indexes — but is not commented anywhere outside the embed itself. A short stacking-context plan (probably in `lib/page-shell.css` or a new `docs/protocols/z-index.md`) would prevent future drift.
3. `z-index: 1` appears in three files for three different reasons (selection tag, inspector chip, spectrum tooltip). Each is a local elevation; no risk, but `auto`/`positioned` would suffice for two of the three.

---

## Layout-primitive opportunities

- **122 `display: flex` rules** vs. **21 `display: grid`** across the codebase. No `.flex-row`/`.flex-col`/`.stack` utility exists. The most repeated pattern is `display: flex; align-items: center; gap: 0.4rem|0.5rem|0.6rem;` (a horizontal item row with leading icon + label). I count **at least 23 sites** of this pattern in `projects-sidebar.css` alone (e.g. `.ps-checkpoint-header`, `.ps-checkpoint-row-refs`, `.ps-checkpoint-actions`, `.ps-checkpoint-view-toggle`, and 19 older sidebar rules). A `.ps-row` or page-shell-level `.row-h` utility would absorb half the verbosity.
- **Repeated card-shape rule**: `padding + border + border-radius + background: var(--ps-bg-tile|--bg-card)` recurs in run-history, selection-panel, system-load-monitor, markdown inspector, bundle-handoff. `page-shell.css` already defines `.card` — but the new components do not opt into it. Either inherit `.card` or expose a `.card-sub` for nested-card use.
- **Two-column 1:1 split layout** (`md-split`, `.viewer + .controls` in trajectory) is hand-rolled in each file. A primitive would help.

---

## Naming convention catalog

| Convention | Where | Sample classes | Health |
|---|---|---|---|
| Tab-prefixed kebab-case | most files | `.ps-checkpoint-row-actions`, `.sp-tag-remove`, `.mv-toolbar`, `.md-render-pane`, `.schema-card` | dominant; consistent |
| Unprefixed kebab-case | `page-shell.css`, `style.css`, top of every tab CSS | `.card`, `.status`, `.tagline`, `.viewer-wrap`, `.modes-table` | global / shared chrome — fine |
| State-class `is-*` | scattered | `.is-active`, `.is-expanded`, `.is-dirty`, `.is-projects-sidebar-collapsed`, `.is-narrow-viewport` (25 uses across selectors) | consistent — clear state semantics |
| Snake/underscore | 3 sites total | (incidental, not a convention) | negligible |
| BEM `block__elem--mod` | 3 sites | (incidental in `.codemirror__…`) | not in use |
| `data-*` attribute selectors | several | `.ps-checkpoint-sensor[data-state="clean"]`, `.ps-checkpoint-ref-chip[data-kind="tag"]`, `.ps-checkpoint-row-action-btn[data-action="restore"]` | consistent for kinded variants — good pattern |

**Verdict**: the project has **one** consistent convention (tab-prefix + kebab + `is-` for state + `data-*` for kinded variants). No work to do here except to keep using it.

---

## Responsive breakpoints

| Breakpoint | Files | Tokenised? |
|---|---|---|
| `max-width: 480px` | system-load-monitor, mol-viewer-embed | NO |
| `max-width: 640px` | projects-sidebar (×3), style.css, trajectory-inspector | NO — used 5× |
| `max-width: 720px` | form-schema (×2) | NO |
| `max-width: 768px` | modify, markdown inspector | NO |
| `max-width: 980px` | trajectory-inspector | NO |
| `max-width: 1100px` | form-schema | NO |
| `prefers-reduced-motion: reduce` | mol-viewer-embed | n/a |

**Verdict**: 6 distinct width breakpoints (480/640/720/768/980/1100). No tokens, no convention. `640` (5 uses, "phone") and `768` (2 uses, "tablet") and `1100` (1 use, "narrow desktop") are tokenisable into `--bp-sm/md/lg`. CSS variables don't work directly in `@media`; the cleanest practice is `@custom-media` (PostCSS) or a short comment-block convention in `tokens.css` saying which breakpoints to use. Today there's neither.

---

## CSS↔JS coupling concerns

### `getComputedStyle` reading CSS tokens — principled

| File:Line | Reads | Use |
|---|---|---|
| `projects-sidebar.js:190,214` | `--ps-w` | sidebar resize-handle clamps user drag to the token's value as default |
| `projects/checkpoint.js:339` | (any var) | helper for sensor colour reads (run-history) |
| `system-load-monitor.js:64–68` | `--load-ok`/`--load-warn`/`--load-bad` | canvas stroke colour from CSS — **but these tokens are undefined**, see the token-gap table above. The canvas falls back to whatever string the empty fallback produces. |
| `trajectory/core.js:1470, 1689, 2047` | several | plotly traces' colours come from CSS |

These are mostly the right pattern (single source of truth for colour in CSS, read into canvas), **but** `system-load-monitor.js`'s reads land on undefined tokens. Either the JS is broken or it's silently using the canvas default — worth a runtime check (out of scope for T3; flag for T1/T5).

### Inline style misuse

- `lib/projects/list.js:772`: `li.style.cssText = "padding: 0.7rem; color: #e07a7a;";`
  - One-off error-row styling. Should be a class (`.ps-list-error` or `.ps-row.is-error`). Sneaks in both a magic spacing (`0.7rem`) and a magic color (`#e07a7a` happens to match `--ps-danger-fg` defined in tokens.css).
- `lib/projects-sidebar.js:163`: `document.documentElement.style.setProperty(...)` — sets `--ps-w` from saved width. Principled.

### JS computing pixel sizes CSS could do

No egregious cases found. The trajectory inspector and the viewer-embed do canvas-sizing math because canvas needs explicit pixels — that's a real constraint, not a JS-vs-CSS confusion.

---

## Findings (ranked)

### IMPORTANT 1: Undefined tokens silently fall back to literals

7 sites reference tokens that don't exist (`--surface-elevated`, `--border-subtle`, `--ok`/`--warn`/`--bad`, `--accent-on`, `--ps-bg-2`). A future theme retune of the **named** token will have no effect. One-line fixes in `lib/tokens.css`: alias them to existing tokens (e.g. `--accent-on: var(--text-on-accent)`, `--ok: var(--success)`, `--ps-bg-2: var(--ps-bg-tile)`) or rename the consumers to the canonical names.

### IMPORTANT 2: New run-history panel introduced 11 untokenised state colours

`projects-sidebar.css:1340–1590` adds 4 sensor states + 2 ref-chip kinds + advisory banner + restore-button accent as raw hex. The token system has `--ps-*` family precisely for this. Per the project's "no parallel state-color tables" rule in `web-ui-coherence.md` Rule 4 (cited in tokens.css comment at L168), these should be promoted to `--ps-state-{clean,dirty,uninit,error}-{bg,fg,border}` plus `--ps-ref-{tag,branch}-bg` plus an `--accent-restore` (which also fixes the duplicate `#e0a868` in markdown.css:39).

### IMPORTANT 3: Spacing scale only half-applied

The codebase has two parallel spacing systems: tokens in px (4/8/14/20/28) and ad-hoc rems (0.25/0.3/0.35/0.4/0.5/0.6/0.7/0.75/0.8/0.85) in component CSS. New code added today (run-history, markdown inspector) is **all rem-literal, zero token use** for padding/margin/gap. A rem-equivalent token scale (`--space-rem-xs … xl`) or a stricter convention ("use the px token even inside a rem-sized component") would absorb 79 + 73 + 71 + 61 + 56 + 50 + 45 + 43 + 42 ≈ 520 magic-number sites.

### IMPORTANT 4: `--font-mono` token exists but is used in ~5% of monospace stacks

Token defined as `ui-monospace, "SF Mono", Consolas, monospace`. Actual usage across the codebase shows **25 distinct sites** declaring `font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;` or `ui-monospace, SFMono-Regular, Menlo, monospace;` — three different stacks, none using the token. Pick one canonical stack (probably the JetBrains variant since it's most common) and put it on `--font-mono`, then sweep.

### NIT 1: Transition duration written as both `0.12s` and `120ms`

39 sites use `0.12s`, 36 use `120ms` — identical to the browser. A `--transition-fast: 120ms` token would unify them and let designers retune feel in one place. Same for `0.1s`/`100ms` (6 + 4 sites).

### NIT 2: No documented z-index plan

11 distinct z-index values, one short commentary block in `projects-sidebar.css`. A `docs/protocols/z-index.md` or top-of-`page-shell.css` enumeration (page-content < sidebar(5) < sidebar-handles(10) < sidebar-menu(50) < drawer-scrim(85) < drawer(90) < drawer-close(95) < modal(100) < embed-internal(1000) < embed-modal(99999)) would prevent future authors picking arbitrary values.

### NIT 3: Breakpoints unconventionalised

6 distinct max-widths; `640` is reused 5× without a token. Either pick 2–3 canonical breakpoints or document the existing six.

### NIT 4: `projects-sidebar.js:163` and `projects/list.js:772` inline-style sneaks

The `list.js:772` `style.cssText` should be a class. The `projects-sidebar.js:163` `setProperty('--ps-w', ...)` is fine (writing a CSS variable from JS persistence is principled).

### NIT 5: `trajectory-inspector.css` warn-pastel exemption is unenforced

5 hex literals marked `/* exempt: warn-pastel theme */`. Make the exemption mechanical: `--warn-pastel-bg/fg/border` tokens, then the comment becomes a rule.

---

## Cross-cutting recommendations

1. **One-line cleanup first**: alias the 7 undefined tokens in `tokens.css`. Highest leverage, zero risk.
2. **Run-history panel sweep**: promote 11 state colours into `--ps-*` tokens before the panel ships to more users; this is the bulk of today's drift.
3. **Decide the rem-scale story**: either token it (`--space-rem-*`) or document a strict "px token in component rem context" convention. Right now neither exists.
4. **Adopt `--font-mono` everywhere**: 25 monospace stacks → 1.
5. **Two utility primitives**: `.row-h` (flex horizontal with gap) and `.card-sub` (nested-card surface). Will absorb ~50 repeated rules.
6. **Add `docs/protocols/css-tokens.md` (or extend `web-ui-coherence.md`)** listing: token families, naming rules, the rem/px decision, breakpoint table, z-index ladder, transition-duration token, font stacks. The token file itself comments most of this but a doc index is missing.

---

## Out-of-scope flags (for sibling auditors)

- **T1 / T5**: `system-load-monitor.js:64–68` reads three undefined tokens — needs runtime check that the canvas-stroke colours actually render (vs. defaulting to black). Possibly a real bug.
- **T1 / T5**: `lib/projects/list.js:772` error-row inline style sneaks a hardcoded color (`#e07a7a`) that already exists as `--ps-danger-fg`; small but indicates the JS-side adopts CSS classes inconsistently.
