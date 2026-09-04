# The code-audit playbook — how to run a systematic review

**Role:** reference
**Domain:** process
**Companions:** [`testing.md`](?doc=process/testing.md) — several audit invariants
below have graduated into enforced artifact lints (§ 5 rule 5); [`conventions.md`](?doc=process/conventions.md)
— the conventions those tests gate; [`web/web-api.md`](?doc=web/web-api.md) — the
response-envelope spec the API dimension checks against;
[`science/validation.md`](?doc=science/validation.md) — the scientific-correctness
dimension.

This is the canonical reference for **systematic code reviews** of the codebase —
the multi-agent "find everything wrong with X" sweeps and the dimension-by-dimension
code-shape passes. It exists so the traps that have bitten us before are written
down once, not re-learned each time.

> **How to use it.** Before starting an audit, read § 1 (the principles that govern
> every audit) + § 2 (pick your dimensions) + the § 3 traps relevant to what you're
> touching + the § 5 checklist for each dimension. When a new bug class turns up, add
> it here (§ 6) rather than to memory — memory holds *principles*, this doc holds
> *playbooks*. A memory note about auditing should point here, not restate it.

## 1. Audit principles

These hold regardless of which dimension you're auditing.

1. **Trust but verify — every agent finding is a *candidate*.** Before proposing any
   action, check the finding against the consumer side, the design intent (read `git
   blame` + the surrounding file), and the actual runtime behaviour. This applies
   doubly to "nothing uses this" claims: verify across tests, docs, and sibling
   modules — not one grep pass. (Past sessions have found roughly half of an agent's
   confident findings to be false on check.)
2. **Structural audits miss behavioural bugs.** Slicing by code dimension (CSS / JS /
   API / tests as separate sweeps) catches *shape* problems but routinely misses
   "user clicks X → the dependent surface doesn't update," because no single slice
   traces that chain. For each user-facing feature, add a per-feature traversal
   (§ 3.2) on top of the dimension slices.
3. **Count honestly.** When an agent reports "N problems," hand-verify the first 3–5
   before trusting the rest. Past audits have over-reported by 4–5× (a flagged "40
   demotion candidates" turned out to be 3 real ones on per-item review).
4. **Audits propagate — fix every sibling in the same commit.** When a CSS/JS/API fix
   lands for one file, apply the same audit to every sibling in that class right then.
   The classic miss: `style.css` was audited for `[hidden]`-attribute precedence in
   2026-05-28; `trajectory-inspector.css` was written later and silently inherited the
   same bug because the audit hadn't been generalised (fixed 2026-06-14, `e7651cb`).
5. **Graduate an invariant into a test once it stops finding new violations.** The end
   state of a mature audit dimension is an **artifact lint** that fails the build —
   see [`testing.md § 3a`](?doc=process/testing.md). Several of the traps below have
   already made that jump (noted inline). When an audit stops turning up fresh
   violations, that's the signal to write the test and retire the manual pass.

   **An artifact lint is not a source pin, and this step means only the first.**
   The distinction is what the assertion *quantifies over*:

   | | an artifact **lint** ✅ | a source **pin** ❌ |
   |---|---|---|
   | scope | **every** file in a class | **one** named file |
   | asserts | a property the shipped artifact must not violate | that one line is spelled a certain way |
   | a rename | is irrelevant to it | breaks it |
   | deleting the code | **fails** it | passes it, if the name survives in a comment |
   | examples | `test_no_inline_scripts.py`, `test_css_no_duplicate_selectors.py`, `test_page_ids_unique.py` | `assert "function _themeColors" in core_js` |

   A lint is a *generalised* audit — the same reading you did by hand, done over
   the whole class, forever. A pin is a transcript of one line you happened to be
   looking at, and it rots the moment the line moves.
   [`testing.md § 3a`](?doc=process/testing.md) forbids the second outright; this
   step licenses only the first.

## 1a. Rules for a loose-interface language *(user, 2026-08-24)*

Python and JavaScript will both accept a call that is missing the one fact
the function exists to use. **Nothing raises**, because the parameter had a
default and the default was a legal value. These rules exist because that
happened, three times in one afternoon, in code that had passed three
"reviews".

### D1 — A parameter that decides WHICH thing is never optional

If a default silently selects *which machine, which scope, which folder,
which queue*, it must be **required**. Pass `None` explicitly to mean "this
one" — an omitted argument says *I did not think about it*, an explicit
`None` says *I considered it and the answer is this*.

> **The incident.** `_bench_inputs(base, target=None)` enumerates a sweep
> grid from a machine's probed hardware. The browser's prep door called
> `_bench_inputs(dest)`. Python filled `target=None`, which is a *meaningful*
> state (one record, no ambiguity), so nothing raised and the tests passed —
> on any machine with a single record. On a machine holding two, `prep bench`
> failed at the WRITE while its own PREVIEW succeeded, because preview
> returns before that line. Signature is now `(base, target)`; the same call
> is now a `TypeError` at the first execution.

**Corollary — a default is a decision, and decisions get written down.** If
you cannot state in one sentence why the default is the *right answer* rather
than merely a *safe-looking* one, it should not be a default. `load_warm_files(engine)`
passes this test: its docstring says status reads a directory where the
description may not be in hand, so the shipped rules ARE the answer.

### D2 — Scope is stated, never achieved by construction

A call that works because an argument was *shaped* to make a fallback land
correctly is a trap for the next reader and the next refactor.

> `_render_sbatch_for(base / f"{name}.sh", ...)` derived its config scope from
> `script_path.parent`, and worked only because the path was chosen so the
> parent was the bundle root. It now passes `project_dir=base` outright.

### D3 — The UI answers its own preconditions; the server is not the validator

A control that fires into a backend refusal to discover the user skipped a
step is a control that has outsourced its own question.

> The Prep button called the API with no machine chosen and surfaced the
> server's paragraph about `--target` and probe commands — which reads as a
> fault, not as *"pick a machine first"*. It is now disabled with that
> sentence until the question above it is answered.

### D4 — Two implementations of one rule get a parity test, or one deletes

When a rule must exist twice because the surfaces cannot share code (a
browser cannot import Python), the duplication is permitted **only** with a
test that runs both over the same inputs and compares.

> `viewer.js` and `ask.py` both parse `"4h"` and `"128G"`. They had already
> drifted: the browser accepted SLURM's `7-00:00:00` — which is what the tab
> FILLS a field with — and Python refused it.

### D5 — Never overwrite an answer with a suggestion

A default fills what nobody answered. If a value was typed by a person or
loaded from their file, replacing it is data loss wearing a default's
clothes. Mark which is which **on the field itself**; a parallel record of
"what is in this box" is a second answer to one question.

---

## 1b. What "review" means *(user, 2026-08-24)*

**A grep is not a review.** Searching for a symbol, counting references, and
checking a doc table are *structural sweeps*. They find dead code, drift and
duplication. They cannot find a call that omits an optional argument, because
that call is syntactically perfect.

Three reviews were reported as complete having run only sweeps. The rules:

**R1 — Read the full text of what you changed.** Every function you wrote,
end to end, beside the signature of every function it calls. Compare the call
against the definition, argument by argument. The `_bench_inputs` defect is
visible in two seconds this way and invisible to every grep.

**R2 — Execute every path, not every function.** A door with modes (plan and
write; run and bench) has *four* paths. "I tested the door" means all four.
**A preview succeeding is not evidence that the write works** — it is
evidence that the code before the write works.

**R3 — Name what the check cannot see.** When reporting a review, say which
technique was used and what it is blind to. "Swept for unused symbols and
duplication; did not execute the bench write path" is honest. "Full review,
clean" — when it was greps — is a false statement about verification, and it
is worse than the bug because it stops anyone else looking.

**R4 — The last mile is the user's click.** For anything with a UI, drive the
real control in the real browser to completion before calling it done. Every
button defect this session survived unit tests, structural sweeps and a
passing full lane, and died the moment the button was actually pressed.

## 2. The audit dimensions

What there is to audit; each has a checklist in § 5.

| Dimension | What it covers |
|---|---|
| **CSS** | visual drift, dead selectors, attribute/class precedence |
| **UI element contracts** | visibility, state, event-wiring, dependent-surface updates |
| **Python API** | wire shapes, response envelopes, dead routes |
| **UI ↔ API interface** | wire-shape match, error UX, race conditions |
| **Tests** | coverage gaps, wrong-layer tests, fragile fixtures, vague intent |
| **Scientific correctness** | chemistry validation, engine-keyword correctness — see [`science/validation.md`](?doc=science/validation.md) |

## 3. Known traps

Bug patterns that have caught us before. Each entry: symptom, why it happens, the
smoking-gun diagnostic, the fix pattern, precedents, and the audit invariant to run.

### 3.1 CSS `[hidden]`-attribute precedence

**Symptom.** JS sets `element.hidden = true` but the element stays visually present.
The user sees a clickable "ghost" control that does nothing — the JS handler fires,
but its preconditions early-return because the element was supposed to be gone.

**Why.** When CSS sets `display: <not none>` on a class/id selector, it ties on
specificity with the UA stylesheet's `[hidden] { display: none }` and **wins by
source order** — so the attribute is silently overridden.

**Diagnostic (DevTools one-liner).**

```js
const el = document.getElementById("foo");
({ hidden_attr: el.hidden, computed_display: getComputedStyle(el).display })
```

Smoking gun: `hidden_attr: true, computed_display: <not "none">`.

**Fix pattern.** Every rule that sets `display:` on a class/id pairs with a `[hidden]`
guard in the same commit:

```css
.foo { display: flex; }
.foo[hidden] { display: none; }
```

**Precedents.**
- 2026-05-28 — `.issues-panel` (`style.css:357`).
- 2026-06-14 — `.ctab-panel label.check` (`trajectory-inspector.css`, commit `e7651cb`,
  the "ghost hide-frozen toggle").

**Audit invariant — now an enforced test.** This one has graduated:
`tests/test_css_hidden_attribute_audit.py` cross-references the IDs JS actually toggles
via `.hidden = …` against the CSS rules that target them, and fails if any such rule
sets a non-`none` `display` without a matching `[hidden]` guard. Run it; add new
violations to it rather than re-grepping by hand.

### 3.2 Per-feature option-dependency traversal

**Symptom.** A user-toggleable option exists in the UI. Some dependent computations
honour it; others ignore it entirely. Each function looks correct in isolation because
it reads *its own* state correctly — the missing read is in a function the user
*expected* to depend on the option.

**Why.** Structural reviews slice by code dimension, not by feature. No slice asks
"for option O, enumerate every dependent computation; does each one read O?"

**Diagnostic.** For each user-toggleable option, list every function that should
re-fire on change, then grep each body for the option's id (e.g. `"hide-frozen"`). Any
function in the should-fire list that doesn't reference it is a candidate.

**Fix pattern.** Add the option read + the same skip/filter idiom the other dependent
functions already use, then pin the regression **through the option itself**: toggle it
in a browser and assert the dependent surface changed.

> **Corrected 2026-09-03.** This step used to say "add a source-text invariant test
> asserting each function's body references the option", and
> `test_trajectory_hide_frozen_invariants_js.py` was written to it. That is a source
> pin, not a lint — it greps for `"hide-frozen"` inside a named function body, so it
> passes on a function that reads the option and then ignores it, and fails on a
> correct refactor that renames the local. The observable is a **visible checkbox with
> a visible effect** (frozen atoms' force arrows appear or don't), which is an e2e.
> See [`testing.md § 3a`](?doc=process/testing.md).

**Precedents.**
- 2026-06-14 — trajectory-inspector `refreshForcesStatus` ignored hide-frozen entirely
  (fixed `65e8246`; the source pins added in `66dab9a` are retired 2026-09-03 -- see § 5 rule 5).
- 2026-06-13 — a projection-toggle label inversion: a different bug, same
  audit-dimension gap (no per-feature traversal caught it).

**Audit invariant.** When auditing a UI option, enumerate the should-fire-on-change set
and confirm each member reads the option.

### 3.3 Cross-source-of-truth gap

**Symptom.** A parser detects a phenomenon (e.g. "constrained atoms exist") from one
source but exposes the data a UI affordance needs (e.g. the *indices*) only from
another. One consumer (a chart using the value) works; another (an affordance needing
the indices) silently no-ops or hides.

**Why.** Two data paths converge on the same JS state field but populate it from
different upstream sources. When the user's environment supplies only one, the UI is
half-functional.

**Diagnostic.** For a non-working affordance: trace **both** the data it consumes *and*
every upstream source that could populate it. Cross-check which of those the user's
environment actually provides.

**Fix pattern.** Bridge — when source A is present but the canonical source B isn't,
populate the JS state field from A too.

**Precedents.**
- 2026-06-14 — SIESTA `runtime_info.frozen_atoms` was populated only from the
  `.molstruct.json` sidecar, while the chart's `max_force_constrained` came from `.out`
  parsing. A trajectory with `.fdf` constraints but no sidecar had the chart working
  and the hide-frozen toggle hidden.

**Audit invariant.** When a UI feature needs data X, trace *every* code path that could
provide X and verify each is honoured.

### 3.4 Listener-attached-but-element-missing

**Symptom.** JS calls `_on(querySelector(...), "change", handler)` but `querySelector`
returns null, so `_on` silently no-ops. When the element later appears via a DOM
mutation, the user clicks it and no handler fires.

**Why.** The listener was wired at module-init against a DOM that wasn't fully present
yet, or against an element that gets recreated later — leaving the listener orphaned on
a detached node.

**Diagnostic (DevTools one-liners).**

```js
document.querySelectorAll("#foo").length          // extra copies of the element?
document.getElementById("foo")?.checked           // did .checked flip after the click?
```

**Fix pattern.** Either (a) defer listener attachment to after the DOM exists (e.g.
from inside the partial-fetch resolve), or (b) use event delegation on a stable parent
that survives the mutation.

**Audit invariant.** For every `_on(querySelector(...), ...)`, verify the queried
element exists at the moment the line runs.

### 3.5 Two endpoints, same wire shape, different names

**Symptom.** Two API endpoints return structurally similar payloads under different
field names (`atoms` vs `atom_list`, `errors` vs `errors_only`), and client code
branches on which one it called.

**Why.** The endpoints were added at different times without a canonical envelope spec.

**Fix pattern.** Pick one canonical name; rename + document what the field is for. Where
a naming difference is *deliberate* (a filtered subset vs the full list), name it so the
relationship is obvious (`errors_only`, not `errors`).

**Precedents.**
- 2026-06-14 — transport `errors` → `errors_only` (commit `3c32ded`) so the field's
  relationship to `issues` is in the name; the envelope spec was added to
  [`web/web-api.md § 1`](?doc=web/web-api.md) (commit `d36d3de`).

**Audit invariant.** Before flagging a field as "duplicate" or "redundant," read every
consumer *and* the field's git history. A deliberate filtered view is not a duplicate.

### 3.6 A reference table typed in by hand

**Symptom.** A literal table of physical or chemical constants appears in the
source — atomic masses, atomic numbers, covalent radii, isotope abundances,
element colours — often in the module that first needed it, and sometimes in
JavaScript because "the browser has no way to know."

**Why.** The need shows up far from where such data belongs, and typing twenty
entries is faster than finding out whether something already ships them. It
almost always *is* shipped: this program depends on ASE, and `ase.data` carries
masses, atomic numbers, covalent radii and more.

**Fix pattern.** Name the existing table instead of copying it — a one-line
lookup function in the L1 module that owns the concept (`chemistry.atomic_mass`
wrapping `ase.data.atomic_masses`), and every caller goes through that name. If
a browser panel is what needs the number, the server computes the answer and
sends it; shipping a periodic table into JavaScript is the same mistake with a
longer commute.

**Precedents.**
- 2026-08-05 — the vibrational-mode composition ("the motion is 91% C, 9% H")
  needed atomic masses in the Spectra panel. Added `chemistry.atomic_mass` over
  `ase.data` and computed the share server-side in `/api/spectra/load`; no table
  was written. See [`web/spectra.md § 4.2`](?doc=web/spectra.md).

**Audit invariant.** A hand-typed constant table is a second source of truth
with no test guarding it — a wrong value in a rarely used row is found by a
user, never by the suite. Before accepting one, check the dependencies already
installed; before writing one, say in a comment which authority it came from and
why the installed one would not do.

## 4. Smoking-gun diagnostics

DevTools one-liners that surface a specific bug class. Each: the one-liner, the normal
output, and the smoking-gun output.

**4.1 Is the `[hidden]` attribute actually hiding? (→ § 3.1)**

```js
const el = document.getElementById("foo");
({ hidden_attr: el.hidden, computed_display: getComputedStyle(el).display })
```

- Normal: `{ hidden_attr: true, computed_display: "none" }`.
- Smoking gun: `{ hidden_attr: true, computed_display: "flex" }` (or any non-none).

**4.2 Is the listener wired? (→ § 3.4)**

```js
({
  count_in_dom: document.querySelectorAll("#foo").length,
  is_checked: document.getElementById("foo")?.checked,
  visible_checked_boxes: Array.from(
    document.querySelectorAll('input[type="checkbox"]:checked')
  ).map(el => ({ id: el.id,
                 label: el.closest('label')?.textContent?.trim().slice(0, 60) })),
})
```

- Normal: `count_in_dom: 1`, `is_checked` flips on click, the element appears in
  `visible_checked_boxes` with the expected id.
- Smoking gun A: `count_in_dom: 0` — wrong id, element absent.
- Smoking gun B: `count_in_dom > 1` — DOM duplication; the listener may be on a
  different copy than the one clicked.
- Smoking gun C: `is_checked` doesn't change after click — the click is reaching a
  different element.

**4.3 What does the user actually see?**

```js
({
  status: document.getElementById("status-readout")?.textContent,
  panel_visible: !document.getElementById("the-panel")?.hidden,
  panel_computed_display: getComputedStyle(document.getElementById("the-panel")).display,
})
```

For "feature does nothing" reports — distinguishes "the handler ran but state didn't
change" from "the handler never ran."

## 5. Per-dimension checklists

### 5.1 CSS

For each component CSS file:

- [ ] Every class/id selector that sets `display: <not none>` has a matching `[hidden]`
      guard (§ 3.1) — enforced by `test_css_hidden_attribute_audit.py`.
- [ ] No full selector is defined in two CSS files (a cascade collision where the
      later-loaded copy silently wins) — enforced by `test_css_no_duplicate_selectors.py`.
- [ ] Every id used in HTML has a CSS rule *or* a JS reference; dead selectors get
      deleted.
- [ ] Colour/spacing values come from the token sheet; raw hex literals outside
      `tokens.css` are flagged.
- [ ] Overlapping rules across files either share identical values or carry an explicit
      "intentional override" comment.

### 5.2 UI element contracts

For each interactive element (checkbox, input, select, stateful button, menu item):

- [ ] A documented contract: when is it visible / enabled / checked, and which surfaces
      update when the user acts on it?
- [ ] An `e2e` test that drives it through its states with a realistic fixture — not
      just the happy path.
- [ ] Visibility honoured at the *computed-style* level, not just the attribute (§ 3.1).
- [ ] The listener attached to the actually-existing DOM node, not an orphan (§ 3.4).

### 5.3 API

For each endpoint:

- [ ] The response follows the envelope spec ([`web/web-api.md § 1`](?doc=web/web-api.md)).
- [ ] Every JS caller of the route is verified to exist (grep `static/`).
- [ ] If zero callers: trace the migration trail (what replaced it?) and verify no
      chemistry / multi-backend contract is entangled before deleting.
- [ ] Deletion covers routes + tests + documented helpers + doc cross-references.

### 5.4 Tests

For each test file:

- [ ] The docstring cites a specific contract or regression — not "tests the foo
      function" without saying which contract.
- [ ] The test lives at the right layer (`unit` / `module` / `interface` /
      `integration`, or `e2e` for browser-only) per [`testing.md § 1`](?doc=process/testing.md).
- [ ] No `time.sleep` / unconditional `wait_for_timeout` without a justifying physics
      reason — wait on state ([`testing.md § 5`](?doc=process/testing.md)).
- [ ] It doesn't mock the very thing it claims to test.

## 6. Adding a new entry

When a new bug class turns up:

1. Add it as a `§ 3.N` trap: symptom, why, diagnostic, fix pattern, precedents, audit
   invariant.
2. Add a smoking-gun one-liner to § 4 if there's a useful one.
3. Add the check to the relevant § 5 checklist.
4. Reference this entry from any related memory note — don't duplicate the content
   there.
5. Once the invariant stops finding fresh violations, graduate it into an **artifact
   lint** (§ 5 rule 5 — a property of every file in a class, never a pin on one
   line: [`testing.md § 3a`](?doc=process/testing.md)) and mark it enforced here.

> **Migration note.** Corrected against code during the docs migration: the `[hidden]`
> invariant § 3.1 only *wished for* a test in the legacy copy — it now exists
> (`test_css_hidden_attribute_audit.py`), as does the duplicate-selector guard
> (`test_css_no_duplicate_selectors.py`), so both are marked enforced. Test-layer
> terminology was aligned to the current marker scheme (`unit`/`module`/`interface`/
> `integration` + `e2e`, not the retired "L1–L5"). One precedent commit hash that no
> longer resolves in history was dropped while keeping its lesson.

## Diagram faithfulness — a standing pass *(user, 2026-08-21)*

**A diagram is a claim like any other, and it drifts like any other.**
When a review touches an area, every diagram and file-tree the area's
documents draw is checked AGAINST THE CODE it depicts — nodes that no
longer exist, files a tree omits, arrows whose direction changed.  A gap
found is reconciled in the same pass, on whichever side is wrong: fix the
document when the code is right, fix the code when the document records
the ruling.  (Found necessary the day it was named: validation.md's § 7
tree listed seven files while the package held eleven.)
