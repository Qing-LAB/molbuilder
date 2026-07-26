# Code-audit playbook

Canonical reference for systematic code reviews of the molbuilder
codebase.  Memory entries point here — do **not** duplicate
playbook content in memory.

> **How to use:** before starting any systematic audit (multi-agent
> review, "find everything wrong with X" sweep, dimension-by-
> dimension code-shape pass), read § 1 + § 2 + the relevant entries
> in § 3 + § 4.  When a new bug class is found, add it to § 3 + § 4
> here.  Memory entries that point at "code audit" should reference
> this document, not duplicate it.

---

## § 1 Audit principles

Rules that govern every audit, regardless of dimension.

1. **Trust but verify.**  Agent findings are CANDIDATES.  Verify
   each one against the consumer side, the design intent (read git
   blame + the file's surrounding context), and the actual runtime
   behavior before proposing an action.  Same applies to dependent
   chain reactions: "no caller" claims should be verified across
   tests, docs, and sibling modules — not just one grep pass.

2. **Structural audits miss behavioral bugs.**  Slicing by code
   dimension (CSS / JS / API / tests as separate sweeps) catches
   shape problems but routinely misses "user clicks X → expected
   surfaces don't update" because no slice traces the chain.  For
   each user-facing feature, do a per-feature traversal (§ 3.2)
   in addition to the dimension slices.

3. **Honest counting.**  When an agent flags "N items," verify the
   count by hand on the first 3–5 before assuming the rest are
   real.  Past sessions have caught audits over-reporting by 4–5×
   (e.g., "40 demotion candidates" → 3 real on per-test review).

4. **Audits propagate.**  When a CSS / JS / API audit ships a fix
   for one file, propagate the same audit to every sibling file in
   the same commit.  Past gap: `style.css:357` audited `.issues-
   panel` for `[hidden]` precedence in 2026-05-28; `trajectory-
   inspector.css` was added later and inherited the bug because
   the audit didn't generalize (fixed 2026-06-14, commit `e7651cb`).

5. **Memory vs playbook split.**  Memory holds principles (rules
   I follow at all times).  This document holds playbooks (how to
   actually run an audit dimension).  When a new audit lesson is
   general, add an entry here — not a new memory file.

---

## § 2 Audit dimensions

What to audit.  Each has its own playbook in § 5.

| Dimension | What it covers |
|---|---|
| CSS | visual drift, dead selectors, attribute/class precedence |
| UI element contracts | visibility, state, event-wiring, dependent-surface updates |
| Python API | wire shapes, response envelopes, dead routes |
| UI ↔ API interface | wire-shape match, error UX, race conditions |
| Tests | coverage gaps, wrong-tier tests, fragile fixtures, vague intent |
| Scientific correctness | chemistry validation, target-platform correctness — see `scientific-validation.md` |

---

## § 3 Known traps

Bug patterns that have caught us before.  Each entry: name, symptom,
diagnostic (smoking gun), fix pattern, precedent commit/session,
audit invariant.

### 3.1 CSS `[hidden]`-attribute precedence

**Symptom.**  JS sets `element.hidden = true` but the element stays
visually present in the layout.  Users see a clickable "ghost"
control that does nothing on click — the JS handler fires correctly,
but its preconditions early-return because the element was supposed
to be hidden in the first place.

**Why.**  When CSS sets `display: <not none>` on a class/id selector,
it ties with the UA stylesheet's `[hidden] { display: none }` on
specificity and wins by source order.

**Diagnostic (DevTools console one-liner).**

```js
const el = document.getElementById("foo");
({ hidden_attr: el.hidden, computed_display: getComputedStyle(el).display })
```

Smoking gun: `hidden_attr: true, computed_display: <not "none">`.

**Fix pattern.**  Every CSS rule that sets `display:` on a class/id
pairs with a `[hidden]` guard in the same commit:

```css
.foo { display: flex; }
.foo[hidden] { display: none; }
```

**Precedents.**
- 2026-05-28 — `.issues-panel` audit, `style.css:357`
- 2026-06-14 — `.ctab-panel label.check`, commit `e7651cb`
  (trajectory-inspector hide-frozen "ghost toggle")

**Audit invariant.**  Grep every component CSS file for `display:`
on class/id selectors; cross-check that every one that sets a
non-`none` display has a matching `[hidden]` rule.  Worth a
source-text test under `tests/` once the codebase stops accruing
new violations.

### 3.2 Per-feature option-dependency traversal

**Symptom.**  A user-toggleable option exists in the UI.  Some
dependent computations honor it; others ignore it entirely.  Each
function in isolation looks correct because it reads its own state
correctly — the missing read is in a function that the user
expected to depend on the option.

**Why.**  Structural reviews slice by code dimension, not by
feature.  No slice asks "for option O, enumerate every
dependent computation; does each one read O?"

**Diagnostic.**  For each user-toggleable option, list every
function that should re-fire on change.  Grep each function body
for a reference to the option's id (e.g., `"hide-frozen"`).  Any
function in the should-fire list that doesn't grep is a candidate.

**Fix pattern.**  Add the option read + the same skip/filter idiom
the other dependent functions use.  Add a source-text invariant
test that asserts each function's body references the option (so
future regressions surface).

**Precedents.**
- 2026-06-14 — trajectory-inspector `refreshForcesStatus`
  ignored hide-frozen entirely.  Fixed in commit `65e8246`;
  invariants in commit `66dab9a`.
- 2026-06-13 — projection-toggle label inversion (different bug
  class but same audit-dimension gap; commit `92fc060`).

**Audit invariant.**  When auditing a UI option, list every
function in the should-fire-on-change set + check each one reads
the option.

### 3.3 Cross-source-of-truth gap

**Symptom.**  A parser detects a phenomenon (e.g., "constrained
atoms exist") via one source but exposes the relevant data (e.g.,
indices) only via another source.  Result: one consumer (a chart
that uses the value) works; another (a UI affordance that needs
the indices) silently no-ops or hides.

**Why.**  Two data paths converge on the same JS state field but
populate from different upstream sources.  When the user has only
one upstream available, the UI is half-functional.

**Diagnostic.**  For a non-working UI affordance: trace BOTH the
data it consumes AND what upstream sources could populate it.
Cross-check whether the user's environment provides each source.

**Fix pattern.**  Bridge: when source A is available but source B
(the canonical one) isn't, populate the JS state field from A too,
not just B.

**Precedents.**
- 2026-06-14 — SIESTA `runtime_info.frozen_atoms` populated only
  from `.molstruct.json` sidecar; chart's `max_force_constrained`
  came from `.out` parsing.  Trajectories without a sidecar but
  with `.fdf` constraints had the chart working and the hide-
  frozen toggle hidden.  Filed as task #392.

**Audit invariant.**  When a UI feature requires data X, trace
every code path that COULD provide X.  Verify each path is honored.

### 3.4 Listener-attached-but-element-missing

**Symptom.**  JS calls `_on(querySelector(...), "change", handler)`
but `querySelector` returns null, so `_on` silently no-ops.  User
clicks the element (when it appears later via DOM mutation), no
handler fires.

**Why.**  Listener wired at module-init time against a DOM that
isn't fully present, OR against an element that gets recreated
later, leaving the listener orphaned on a detached node.

**Diagnostic.**  DevTools console one-liners:

```js
// Are there extra copies of the element?
document.querySelectorAll("#foo").length

// After clicking, did .checked change?  (If not, click event isn't reaching the input.)
document.getElementById("foo")?.checked
```

**Fix pattern.**  Either (a) defer listener attachment to after the
DOM exists (e.g., from inside the partial-fetch resolve), or (b)
use event delegation on a stable parent that survives DOM
mutations.

**Audit invariant.**  For every `_on(querySelector(...), ...)`,
verify the queried element exists at the moment the line runs.

### 3.5 Two endpoints, same wire shape, different names

**Symptom.**  Two API endpoints return structurally similar
payloads but use different field names (`atoms` vs `atom_list`,
`errors` vs `errors_only`).  Client code branches on which
endpoint it called.

**Why.**  Endpoints added at different times; no canonical
envelope spec consulted.

**Fix pattern.**  Pick one canonical name; rename + add a self-
documenting comment block explaining what the field is for.  When
naming differences are deliberate (filtered subset vs full list),
name them so the relationship is obvious from the name
(`errors_only`, not `errors`).

**Precedents.**
- 2026-06-14 — transport `errors` → `errors_only` rename
  (commit `3c32ded`) so the relationship to `issues` is in the
  name.  Spec added to `web-api.md § 1.1` (commit `d36d3de`).

**Audit invariant.**  Before flagging a field as "duplicate" or
"redundant," read every consumer + the field's git history.  A
"deliberate filtered view" is not a duplicate.

---

## § 4 Smoking-gun diagnostics

DevTools console one-liners that surface specific bug classes.
Each entry: bug class, one-liner, expected normal output, smoking-gun
output.

### 4.1 Is the `[hidden]` attribute actually hiding?

```js
const el = document.getElementById("foo");
({ hidden_attr: el.hidden, computed_display: getComputedStyle(el).display })
```

- Normal (when `hidden` is set): `{ hidden_attr: true, computed_display: "none" }`
- Smoking gun: `{ hidden_attr: true, computed_display: "flex" }` (or any non-none)

→ § 3.1

### 4.2 Is the listener wired?

```js
// Existence + state after click
({
    count_in_dom: document.querySelectorAll("#foo").length,
    is_checked: document.getElementById("foo")?.checked,
    visible_checked_boxes: Array.from(
        document.querySelectorAll('input[type="checkbox"]:checked')
    ).map(el => ({ id: el.id,
                   label: el.closest('label')?.textContent?.trim().slice(0, 60) }))
})
```

- Normal: `count_in_dom: 1`, `is_checked` flips on click, the element
  appears in `visible_checked_boxes` with the expected id.
- Smoking gun A: `count_in_dom: 0` (wrong id, element absent)
- Smoking gun B: `count_in_dom > 1` (DOM duplication — listener may
  be attached to a different copy than the one the user clicks)
- Smoking gun C: `is_checked` doesn't change after click (click is
  reaching a different element)

→ § 3.4

### 4.3 What does the user actually see?

```js
({
    status: document.getElementById("status-readout")?.textContent,
    panel_visible: !document.getElementById("the-panel")?.hidden,
    panel_computed_display: getComputedStyle(document.getElementById("the-panel")).display,
})
```

For "feature does nothing" reports — distinguishes "the handler ran
but state didn't change" from "the handler didn't run at all."

---

## § 5 Per-dimension audit checklists

### 5.1 CSS audit checklist

For each component CSS file:

- [ ] Every class/id selector that sets `display: <not none>` has
      a matching `[hidden]` guard (§ 3.1).
- [ ] Every id used in HTML has a corresponding CSS rule OR a JS
      reference.  Dead selectors get deleted.
- [ ] Tokens are the only source for color/spacing values; raw
      hex literals outside `tokens.css` are flagged.
- [ ] Overlapping rules across CSS files either have identical
      values or carry an explicit "intentional override" comment.

### 5.2 UI element contract checklist

For each interactive element (checkbox, input, select, button-with-
state, menu-item):

- [ ] Documented contract: when is it visible / enabled / checked?
      When the user does action X, which surfaces update?
- [ ] L5 e2e test that drives the element through its states with
      a realistic data fixture (not just the happy path).
- [ ] Visibility honored at the computed-style level, not just the
      attribute (§ 3.1).
- [ ] Listener attached to the actually-existing DOM node, not an
      orphan (§ 3.4).

### 5.3 API audit checklist

For each endpoint:

- [ ] Response envelope follows `web-api.md § 1.1.1`.
- [ ] Every JS caller for this route is verified to exist (grep
      `static/`).
- [ ] If zero callers: trace the migration trail (what alternative
      replaced this endpoint?), verify chemistry / multi-backend
      contracts aren't entangled before deletion.
- [ ] Deletion includes routes + tests + documented helpers +
      cross-references in docs.

### 5.4 Test audit checklist

For each test file:

- [ ] Docstring cites a specific contract or regression (no "tests
      the foo function" without saying which contract).
- [ ] Test lives at the correct tier (L1–L5) per
      `docs/protocols/test-strategy.md`.
- [ ] No `time.sleep` or unconditional `wait_for_timeout` without
      a justifying physics reason.
- [ ] No mocking the thing the test is trying to test.

---

## § 6 How to add a new entry

When a new bug class is found:

1. Identify the trap.  Add it as a `§ 3.N` entry: name, symptom,
   why, diagnostic, fix pattern, precedents, audit invariant.
2. If there's a useful smoking-gun one-liner, add it to § 4.
3. Add to the relevant dimension's checklist in § 5.
4. Reference this entry in any related memory addition — don't
   duplicate the content there.
