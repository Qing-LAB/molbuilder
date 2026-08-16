# The tab CSS system — a proposal

**Role:** plan (proposed, not started)
**Domain:** web
**Started:** 2026-08-02
**Companions:** [`tabs.md`](?doc=web/tabs.md) — the pages this covers.
[`molview.md`](?doc=web/molview.md) — the module whose sheet this deliberately does
**not** touch, and why.

---

## 1. The boundary, first

**A module owns its CSS exactly the way it owns its JavaScript.** MolView is sealed
at `index.js`; `molview.css` is the same seal in another language, and a page-CSS
cleanup reaching into it is the same category error as a tab reaching past
`mount`. The names it uses (`molviewer-*`) are its private vocabulary.

So the work is split by owner, and only one side is in scope:

| | Sheet | Owner | In scope |
|---|---|---|:--:|
| ESM | `lib/molview/molview.css` | MolView | **no** |
| ESM | `lib/projects/projects-sidebar.css` | projects | **no** |
| ESM | `lib/trajectory/trajectory-inspector.css` | trajectory | **no** |
| ESM | `lib/inspectors/spectra.css` · `markdown.css` | inspectors | **no** |
| ESM | `lib/results/bundle-handoff.css` | results bundle | **no** |
| ESM | `lib/system-load-monitor.css` · `lib/app-notifications.css` | their widgets | **no** |
| page | `lib/tokens.css` · `page-shell.css` · `tabs.css` | the app shell | yes |
| page | `lib/form-components.css` · `form-schema.css` | the form family | yes |
| page | `modify/` · `structure-optimization/` · `spectra/` · `transport/` · `results/` · `documents/` | each tab | yes |

**The one thing to say to a module sheet is "you are not the page's".** If a page
needs a module to look different, that is a mount option or a module change, made
inside the module — not a rule in a page sheet reaching at `molviewer-*`. Worth
checking for as part of the inventory (§ 3.1), and worth a guard (§ 5).

**Already found, and it is the cheapest win in the whole plan:**
`lib/viewer/mol-viewer-embed.css` — **722 lines, 79 rules, 80 hard-coded lengths,
23 raw colours — is loaded by no template at all.** It belongs to the embed MolView
replaced. Deleting it is the first commit.

---

## 2. Why this is worth doing, in measurements

Not an opinion — four numbers, each reproducible.

**Ownership.** **21 selectors are defined in two or more files.** The worst are not
components but the document itself:

| Selector | Defined in |
|---|---|
| `body` | page-shell + **modify + spectra + structure-optimization** |
| `.status` | page-shell + **modify + spectra + structure-optimization** |
| `html` | modify + spectra + structure-optimization |
| `*` · `header` · `header h1` | page-shell + modify + spectra |
| `.card` | page-shell + modify + spectra |
| `footer` · `footer a` · `footer a:hover` | modify + structure-optimization |
| `input[type=text]` · `textarea` · `textarea:focus` | page-shell + spectra |
| `fieldset input:disabled` | form-schema + structure-optimization |

Every one of these is decided by `<link>` order, and no page states that it means
to win. Three tabs each carry their own definition of what `body` is.

**The existing guard cannot see most of it.** `test_css_no_duplicate_selectors.py`
skips element-only selectors on purpose — *"element-only selectors routinely repeat
across files for legitimate cascade reasons"* — so `body`, `html`, `*`, `header`,
`footer`, `input`, `textarea` are invisible to it. That assumption is what let the
document tier drift across four sheets while the suite stayed green. It also
allowlists `.card` / `.status` / `header .tagline` with a recorded plan
(*"phase 3: scope per-tab overrides"*), which this document is the plan for.

**Scale.** Hard-coded lengths and raw colours per page sheet:

| Sheet | magic lengths | raw colours |
|---|---:|---:|
| `spectra/style.css` | 38 | 0 |
| `structure-optimization/style.css` | 31 | 0 |
| `results/style.css` | 15 | 5 |
| `documents/docs-render.css` | 14 | 15 |
| `transport/style.css` | 7 | 0 |
| **`modify/style.css`** | **7** | **0** |

`modify` is the low mark because it has just been through this pass (101
declarations moved onto the scale, three stale colour fallbacks removed). It is the
reference implementation, not the first job.

**Residue.** Rules whose elements no longer exist, markup carrying classes nothing
styles, and whole sheets loaded by nobody. On `modify` alone this pass found five
dead classes in the markup, one dead rule block, and a stale `header .tagline`
override. Unmeasured elsewhere.

---

## 3. A categorical scheme

The question this answers: *given any CSS rule, which file does it belong in, and
who decides?* Four tiers, and **every rule in the page layer is in exactly one**.

| Tier | What it is | Home | May contain |
|---|---|---|---|
| **T0 Tokens** | values with no selectors — the scales, the palette, the radii | `lib/tokens.css` | custom properties only |
| **T1 Document** | the bare document: `html`, `body`, `*`, base type, focus, link and form-element defaults | `lib/page-shell.css` | element + attribute selectors |
| **T2 Components** | things that look the same on every tab: card, status line, hint, button variants, form controls, the tab bar | `page-shell` · `form-components` · `form-schema` · `tabs` | one class name each, one home each |
| **T3 Page vocabulary** | a tab's own parts — op panels, cell editor, timeline, result panes | `<page>/style.css` | `<page>-*` classes only |

**The rule that makes it enforceable, and it is one sentence:**

> **A page sheet may contain only T3.** No element selectors, no bare
> shared-component class names, no `:root`.

And the two escapes, which are what make the rule liveable rather than a wish:

1. **A page needs a component to look different** → it scopes under its own root
   (`.modify-page .card`), which is a T3 selector because the specificity comes
   from the page's own name. The intent is stated, and link order stops mattering.
2. **Several pages need the same difference** → it is not a page variant. It is
   promoted to T2 as a named variant (`.card--flush`), with one home.

**Why this split and not another.** It is the split the app already has, drawn from
what the pages actually load: the three generator tabs share `form-schema` +
`form-components` and keep 3–12 classes of their own; `modify` loads no form schema
and keeps 52; `results` and `documents` are viewers. The tiers name that, they do
not impose it. The value is that the boundary becomes checkable.

### 3.1 What the investigation produces

One table, every rule in the page layer, with a tier and a verdict:

| Column | |
|---|---|
| selector | as written |
| file | where it is now |
| tier | T0–T3 by the definitions above |
| verdict | `keep` · `move to <file>` · `scope under <page>` · `promote to variant` · `delete (residue)` |
| evidence | the markup that uses it, or "none found" |

Plus three lists: sheets loaded by nobody, page rules reaching at module names
(`molviewer-*`, `projects-*`, …), and markup classes no sheet styles.

---

## 3.2 What the inventory found (step B, run 2026-08-02)

Built by a script over every page-layer sheet; module sheets counted and never
read (§ 1). **421 page-layer rules**: T0 1 · T1 63 · T2 127 · **T2? 91** · T3 137.

**T2? is the number that matters** — 91 rules in *page* sheets that name a class
the page does not own. That is the boundary, measured.

### The contamination, and it points the other way

**20 rules in `results/style.css` are the inspectors module's own appearance.**
`.inspector-card`, `-header`, `-title`, `-note`, `-actions`, `-link` are built
**entirely in JavaScript** by `lib/inspectors/` — `_partial_inspector_factory.js`
(the failed-to-mount card), `source.js` (`inspector-card source-card`) and
`structure.js` (`inspector-card structure-card`). No template writes them. And no
sheet styles them except a page's.

So this is not a page reaching into a module — it is **a module whose visual
identity lives in a page's stylesheet**, which is the same coupling seen from the
other end and is worse, because the module cannot be mounted anywhere else without
carrying a page sheet with it. The inspectors module is loaded by **results.html,
modify.html and spectra.html**; only results.html loads `results/style.css`. An
inspector's error card on `/spectra` renders unstyled.

**Not fixed here, because the fix is a module change**: the rules are repatriated
to a sheet under `lib/inspectors/`, which every page mounting an inspector links —
and § 1 says a module's CSS is changed inside the module, deliberately, not as a
side effect of tidying a page. Recorded as the first item of § 7.

### The document tier, measured pairwise

Comparing each page's copy against `page-shell.css`'s, property by property:

| Selector | Page copy vs the shared one |
|---|---|
| `*` | **byte-identical** in modify and spectra — **done, deleted** |
| `html, body` | differs in all three, on 8 properties each |
| `header` · `header h1` | differs in modify and spectra |
| `button` · `:hover` · `:disabled` | differs in spectra |
| `footer` · `footer a` · `:hover` | **page-shell has no rule at all**; modify and structure-optimization each invented one, and they differ (sopt's still carries `16px 28px` / `0.82rem`) |
| `textarea` | page-shell has no rule; spectra invented one |

**Only the `*` rules were provably safe** — identical bodies, so deleting them
cannot change a pixel. Everything else is a visual decision and waits for a
browser, which is the whole reason C is called the risky step.

---

## 4. Order of work

Ordered so each step is verifiable alone and the risky ones come last.

| | Step | Why here |
|---|---|---|
| **A** ✅ | ~~Delete `lib/viewer/mol-viewer-embed.css`~~ **done 2026-08-02** — and it turned out the whole `lib/viewer/` directory is orphaned: 307 KB of JS that no template loads and no module imports, MolView having been rebuilt on its own `3dmol-embed.js`. Only the stylesheet was deleted here; the JS is task #104's, and the `test_css_molview_namespace` skip that anticipated exactly this now fires | loaded by nothing; pure subtraction, no visual risk |
| **B** ✅ | ~~Inventory~~ **done 2026-08-02, § 3.2** | the whole plan is guesswork without it |
| **C** ◐ | **T1: one document tier.** The `*` reset is **done** — it was byte-identical in modify and spectra, so deleting it provably cannot change a pixel. The rest (`html, body`, `header`, `header h1`, `button`, `footer`, `textarea`) genuinely differs per page and is **blocked on a browser** | the deepest drift, and it changes every page — so it happens once, deliberately, with all six pages checked in a browser |
| **D** | **T2: one home per component.** `.card`, `.status`, `header .tagline` — the allowlist entries. Each page's variant either scopes under its own root or becomes a named variant | shrinks the guard's allowlist to zero |
| **E** | **T3 per page**, one page per commit: tokens → namespace → residue. `modify` is done and is the worked example; then `spectra`, `structure-optimization`, `transport`, `results`, `documents` | independent, and a browser check per page is cheap |
| **F** ◐ | Extend the guards (§ 5). **Guard 3 is done** — `tests/test_css_module_boundary.py`: no page sheet may name a module class, the retired embed sheet stays deleted, and the exception list may not grow. Guards 1 and 2 wait for C and D, because a guard that fails on the day it lands teaches people to skip the suite | mostly last, so it locks in a state that already holds — but the boundary guard could land now, since the boundary already holds except for one recorded item |

**C before E.** Converting a page's own vocabulary while the document tier is still
contested means measuring spacing against a base three files are fighting over.

---

## 5. Guards, so it cannot drift back

The existing `test_css_no_duplicate_selectors.py` becomes three:

1. **One home, including elements.** Drop the element-only exemption — that is the
   blind spot § 2 measured. Keep an allowlist only for `@media`-scoped repeats.
2. **A page sheet contains only T3.** No element selector, no `:root`, no bare
   shared-component class. This is the rule of § 3 as an assert, and it is what
   makes the tiers real rather than a document nobody re-reads.
3. **No page rule names a module's class.** A page sheet may not mention
   `molviewer-*`, `projects-*`, `trajectory-*` — the § 1 boundary, enforced.

Plus the two that already exist and should keep running: the `[hidden]`-precedence
audit and the duplicate-selector check.

---

## 6. What this is not

- **Not a redesign.** Nothing here changes what the app looks like on purpose. Every
  step is "same pixels, one owner" — and where a step cannot keep the pixels (C is
  the risk), that is called out and checked in a browser.
- **Not a token rewrite.** `tokens.css` is the scale and it is fine. The work is
  adopting it, not changing it.
- **Not module work.** § 1. If the inventory finds a page rule reaching at
  `molviewer-*`, the fix is to delete the reach, never to edit the module.

## 7. Open, for your decision

0. **The inspectors module's appearance lives in `results/style.css`** (§ 3.2) —
   20 rules for elements the module builds in JavaScript. Mount an inspector on
   `/molbuilder` or `/spectra`, where the module also loads, and it renders
   unstyled. The fix is a **module change**: repatriate the rules to a sheet under
   `lib/inspectors/` and link it wherever the module is mounted. It is listed here
   rather than done because § 1's rule cuts both ways — a module's CSS is changed
   inside the module, deliberately, not as a side effect of tidying a page.
   Recorded in `tests/test_css_module_boundary.py`'s `KNOWN` list until it lands.

1. **Does `modify` keep its own sheet?** This plan says yes and the measurements
   back it: 52 classes, no form schema, near-zero vocabulary overlap with the
   generator tabs. The alternative — folding it into a shared "workbench" tier —
   has one member and would be a tier invented for a single page.
2. **Do the three generator tabs get a shared sheet of their own?** They already
   share `form-schema` + `form-components`, and what is left is 3–12 classes each.
   This plan says **no** — there is not enough left to justify a fourth file, and an
   almost-empty shared sheet is a place for drift to hide. Worth re-asking after the
   inventory, when the number is exact rather than estimated.
3. **How far does `results` go?** It is 40 classes and a viewer, not a form. It may
   deserve the same treatment as `modify`, or it may be mostly module surface that
   belongs to the trajectory / inspector sheets. The inventory decides.
