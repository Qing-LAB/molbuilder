# The editor module — CodeMirror behind one door


> **ARCHIVED 2026-09-01.**  Its open items moved to the one plan,
> [`plans/plan.md`](?doc=plans/plan.md); what stays here is the record of
> what was decided and built.  Nine plan documents were consolidated that
> day *(user: "We don't need ten plan files scattered")*, and a fact-check
> against the code found three of the nine headers stating the opposite of
> what had shipped.

**Role:** plan
**Domain:** web

**Companions — the contracts this is built against, and where the two disagree
those win:** [`web/molview.md`](?doc=web/molview.md) § 4, § 9.1 — the module
pattern this copies, and the only working example of it in the tree;
[`web/ui-contract.md`](?doc=web/ui-contract.md) § 1–2 — the stylesheet layers
and the rule that a shared widget has exactly one owner;
[`web/tabs.md`](?doc=web/tabs.md) — the surfaces that consume it.

**Not started, and deliberately sequenced after the parameter-tab → Task-setup
workflow** (user, 2026-08-16). This document exists so the design survives that
wait: it was worked out in one session against all three call sites, and a
decision that does not survive into a plan is one you pay for twice.

---

## 1. The problem, measured

CodeMirror 5.65.16 is vendored and used on three surfaces. Nothing owns it.

| | today |
|---|---|
| **loaders** | **three.** `lib/codemirror-load.js` (shared), and `lib/inspectors/markdown.js` carries its own inline `loadScript`/`loadCSS`, its own cached promise, and its own asset list |
| **stylesheets theming `.CodeMirror*`** | **three** — 25 rules in `projects-sidebar.css`, 4 in `task-setup/style.css`, 1 in `inspectors/markdown.css`; six separate `.CodeMirror` rules between them |
| **the hard-won limits** | **one surface only.** The 1500-line selection cap (Ctrl-A measured at **225 s** on a 2 MB document) and the 1 MB view-only bar live in `projects/preview.js`. Task setup has neither |

Three consequences that are already real, not hypothetical:

1. **`markdown.js`'s asset list is unguarded.** `test_codemirror_vendor_bundle.py`
   reads the *shared* loader's list, so a vendor change can break the markdown
   inspector silently — the exact drift that test exists to catch.
2. **It loads markdown without xml.** `markdown.min.js`'s module head is
   `require("../xml/xml")`. The dependency is declared in the shared loader's
   `CM_MODES` and does not reach this path.
3. **It can double-inject the core.** Two independent promise caches both
   guarding on `if (!window.CodeMirror)`; if two surfaces open together, both
   see `undefined` before either resolves.

And `ui-contract.md` § 1 is already being violated by the CSS: *"a widget used
on more than one page lives in a shared sheet… a shared element has exactly one
owner."* The editor is on three surfaces and has none.

---

## 2. The shape — MolView's, applied to CodeMirror

This is not a new pattern. `lib/molview/index.js` conceals vendored 3Dmol behind
one ES-module door, and states the rule:

> *"the single ES-module entry, and the whole of what is importable… **Every
> other file in the module is internal — a consumer that imports any of them
> directly has broken the module, not found a shortcut.**"*

```js
// lib/editor/index.js — the whole of what is importable
export { mount } from "./mount.js";
export { modeForPath } from "./_modes.js";   // wantable with no editor
```

```js
const ed = await mount(hostEl, {
    owner:    "task-setup",              // who these bytes belong to
    tag:      "task-setup.description",  // this editor SLOT — see § 4
    path,                                // the ONLY way a mode is chosen
    text,
    readOnly: false,
    restore:  true,
});
```

### The eleven verbs, and which call site earns each

| verb | earned by |
|---|---|
| `getValue()` · `setValue(text, {path})` | all three |
| `append(text)` | `preview.js` streaming a >16 MB file — `setValue(existing + chunk)` per chunk is O(n²) |
| `isDirty()` · `markClean()` | all three |
| `onChange(fn)` | dirty tracking; `markdown.js`'s debounced live preview |
| `onSave(fn)` | `markdown.js`'s `Ctrl-S` / `Cmd-S` binding |
| `setReadOnly(b)` | `preview.js`'s view ↔ edit toggle |
| `find()` | `preview.js`'s Find button |
| `refresh()` | Task setup — mounted inside a card that may be hidden or resized |
| `onScrollNearBottom(fn)` | `preview.js` asking for the next chunk |
| `destroy()` | all three (`markdown.js` calls it `dispose`) |

**The estimate went 4 → 9 → 11 as each call site was read.** That is the method
working, not failing — every one is a real need that would otherwise have been
found mid-port. But it means this is a **consolidation, not a simplification**:
the win is one loader, one theme, one set of limits and one thing to swap for
CM6. Not less code at the call sites.

### What is internal, and not negotiable

`window.CodeMirror`, the mode registry, the addon commands, `viewportMargin`,
the 1500-line selection cap, the 1 MB view-only bar.

**No `getCodeMirror()` escape hatch.** That is how this pattern dies: one caller
reaches through, and the door stops being a boundary.

`viewportMargin` is the worked example of why concealing is worth it.
`markdown.js` sets it to `Infinity` today — rendering every line, which is the
cliff the selection cap exists for. Inside the module it becomes the module's
decision from file size, and the markdown inspector inherits a fix it never had.

---

## 3. The boundary — what stays outside

**Byte-fetching stays in `projects/api.js`.** `preview.js` branches on the
server's 16 MB bulk ceiling and streams through `/api/files/read_range`. That is
HTTP, and the editor has no business in it.

**The buffer half comes inside** — `append` and `onScrollNearBottom` are about
the document and the viewport. The alternative is `preview.js` keeping its own
CodeMirror instance to stay fast, which defeats the exercise.

---

## 4. Persistence — tagged, so a slot has an identity

*User decision, 2026-08-16: the persistency module takes a **tag**, which can be
instance-specific.*

Keyed on the **slot**, not the file:

```
key:    molbuilder.editor.<tag>
record: { path, text, size, mtime, at }
```

**Why a tag rather than `(owner, path)`.** Task setup's editor follows the
projects sidebar, so its path changes under a live instance every time you pick
a different folder. Keyed on path it is a new identity each time and can never
restore itself. Keyed on a tag, the slot is stable and the path becomes part of
the *record* — which makes the restore check fall out: a stored path that does
not match what is being mounted means the slot held a different file last time,
so load fresh without prompting. It also lets the same file be open in two slots
(the preview modal and Task setup both on `task.json`) as two records rather
than one buffer they fight over.

**Three rules, because the naive version loses work or shows lies:**

1. **Persist only what is dirty.** A clean buffer is re-readable from disk, and
   disk is always more current. Storing it burns quota to duplicate a file.
2. **A size ceiling, below the 1 MB view-only bar.** `sessionStorage` is ~5 MB
   total and shared with the sidebar; a 2 MB unsaved buffer evicts everything
   else. Above it: keep editing, persist nothing, and **say so** — an editor
   that silently forgets is worse than one that admits it cannot remember.
3. **Store what it was loaded from, and check on restore.** If `size`/`mtime`
   moved, restoring silently would overwrite someone else's edit with a stale
   buffer. That is the stale-file handshake `tabs.md § 6` already requires: say
   *"you have unsaved edits from before; the file has changed since"* and let
   the user choose.

### The stale-file handshake belongs here too

*Folded in 2026-08-16 (user), from the Task-setup review's **F6**.*

Rule 3 above already requires the editor to store **what a buffer was loaded
from** — path, size, mtime — so a stale buffer cannot silently overwrite
someone else's edit on restore. **That is the same fact a save needs**, and the
same comparison:

| | asks | answers with |
|---|---|---|
| **restore** | the buffer I kept — is the file still what it was? | *"you have unsaved edits from before; the file has changed since"* |
| **save** | the file I loaded — has it moved under me? | *"this folder changed since you opened it — a `prep` ran, or somebody edited by hand"* |

`task-setup.md` § 8 and [`tabs.md § 6`](?doc=web/tabs.md) require the second;
Task setup ships without it today, and saving is last-write-wins. **Doing it in
the tab would put mtime tracking in a second place**, three months before the
editor puts it in the first — so the handshake is the module's, exposed on the
handle:

```js
ed.loadedFrom()        // { path, size, mtime } | null
ed.isStale()           // re-stat and compare; the caller decides what to do
```

Two callers, one comparison: `projects/preview.js` already has a save path with
the same exposure, and Task setup's save calls `isStale()` before writing.

**`lib/session-store.js`** — `get(tag)` / `set(tag, value)` / `drop(tag)`,
quota- and private-mode-safe in one place. The editor needs it, and four
existing ad-hoc `sessionStorage` sites (`transport_form`, the structure
inspector, the docs theme, the sidebar) could adopt it later **without being
dragged into this change**.

---

## 5. The order to do it in

1. `lib/session-store.js` + `lib/editor/` + `docs/web/editor.md` (the contract).
2. **Port `task-setup/viewer.js`** — smallest, newest, fully covered by tests.
3. **Port `projects/preview.js`** — the streaming one, and the real test of
   `append` / `onScrollNearBottom`.
4. **Port `inspectors/markdown.js` last** — no test coverage, its own loader,
   the missing xml dependency, and the split pane. If the door is too narrow,
   this is where it shows, and finding out last is deliberate: the first two
   ports prove the shape before the awkward one stresses it.
5. Fold the three stylesheets into `lib/editor/editor.css`, one owner, tokens
   only. The tab sheets keep genuine per-instance composition (heights, borders).

---

## 6. What this does to the CodeMirror 6 question

**It converts a migration into a swap, and that is the main argument for doing
it at all.**

CM6 is not a newer CM5 — it is a rewrite (`EditorState`/`EditorView`/extensions),
distributed as npm packages with **no official single-file browser build**. This
project has **no JS build step**: no `package.json`, an empty `package-lock.json`,
and browser assets vendored pre-built and shipped by `package-data` globs. So
adopting CM6 means either adding an npm + bundler pipeline — a deployment
change, against the offline/strict-CSP constraint `static/vendor/README.md`
states — or vendoring somebody's unofficial bundle, which discards the
modularity that is CM6's whole point.

That decision is **bigger than the editor** (it is really *"do we want a JS
build step?"*) and should be taken on its own terms. What this plan does is make
it cheap either way: three call sites that never name `CodeMirror` can be
repointed by replacing one module's internals. Three that construct it directly
cannot.

*(Before pursuing CM6, verify whether a usable prebuilt ESM bundle has appeared
since — that is the hinge, and it was not checked.)*
