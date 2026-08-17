# The Task setup tab — what is built, and what is left

**Role:** plan
**Domain:** web

**Companions — the contracts this work is built against, and where the two
disagree those win:** [`web/task-setup.md`](?doc=web/task-setup.md) — the
design, entire; [`engines/stages.md`](?doc=engines/stages.md) — `task.json`,
the one-stage rule, and § 6.5a's hand-over;
[`web/ui-contract.md`](?doc=web/ui-contract.md) — the five stylesheet layers and
the token rule; [`web/tabs.md`](?doc=web/tabs.md) — the roster, and the rule
that tabs do not hand each other data in memory.

This plan exists because the work arrived in pieces across one session and the
remaining pieces were only ever named in conversation. **It is a plan, so it
lives here and leaves in one of two directions** — its rules land in
`task-setup.md`, or it goes to `archive/` when the tab is finished.

---

## 1. What is built, and verified

| | |
|---|---|
| the tab, at `/task-setup` | seventh in the roster; follows the projects sidebar's selected folder |
| three-state open | a `task.json` → a description · no `task.json` but a `task.1st.json` → a hand-over · neither → nothing described yet |
| what it shows | the destination and folder state, the file list, the stage table, the machine rows, and `task.json` itself in the editor |
| the editor | vendored CodeMirror 5.65.16, **highlighting chosen by file suffix**, eight modes, loaded on first use |
| the hand-over | `POST /api/task-setup/handover` writes `<label>.template.toml` + `task.1st.json`; the button lives on the parameter tab |
| the CSS | layer 5, composition only; thirteen `--ts-*` tokens in `lib/tokens.css`; two tests fail on a raw colour or a magic number |

**It writes no description.** Save is rendered disabled and says
`molbuilder jobset describe` is what writes one today.

---

## 2. What is left, in dependency order

### T1 — Ask for the shape

**Blocks everything below.** `shape` is required with no default, because
inferring it "would hand somebody a directory tree they never asked for"
(`stages.md § 6.7`). The card is in the template already; it is not wired.

Two options, no default pre-selected, and the page cannot be saved until one is
chosen. `task-setup.md` § 4 has the two descriptions to show.

### T2 — Save

The write path, and the only thing that makes the tab more than a viewer.

* **Reuse `describe.build_description` + `describe.write_description`** — the
  shipped pair. `write_description` is already transactional: a staging
  directory beside the target, published with `os.replace` once every artifact
  exists, removed entirely if anything raises. Do not write the files by hand.
* **Consume the hand-over**: on success write `task.json` + the template, then
  **delete `task.1st.json`**, so the next visit finds one description and no
  ambiguity (`stages.md § 6.5a`).
* **Checkpoint first** — `task-setup.md` § 8: save the folder's current state
  before writing, offered and never taken silently (`checkpointing.md § 9`).
* **The edited buffer is the source.** The editor is where a person corrects the
  description before it is written; a save must take what is in the buffer, not
  re-serialise from the parsed model, or an edit made in the editor is silently
  discarded.
* **Refuse rather than repair** a buffer that does not parse, naming the line.

When this lands, `test_the_tab_writes_nothing_yet` changes **on purpose** — it
exists so enabling a write path cannot happen by accident.

### T3 — Edit the stage table

Today it renders. The ten operations and the rule that keeps each one safe are
`task-setup.md` § 9 — in particular: adding a column seeds every stage with the
template's value, removing one keeps the **last enabled** stage's, and removing
the last stage is refused.

### T4 — Edit the machine rows

One point is a choice, several is a measurement. Adding a point to a chosen
setting turns it into an axis and **keeps the value as the first point**, so
measuring never discards what you chose. `mpi_np` / `omp_threads` /
`max_memory_mb` can only ever be axes.

### T5 — What has already run

Read from the folder: how many attempts exist per stage, whether the last
converged. No target machine required, which is why it belongs here rather than
on Results.

---

## 2a. What comes after, and not before

**The editor module** ([`plans/editor-module-plan.md`](?doc=plans/editor-module-plan.md))
— CodeMirror behind one door — is sequenced **after** T1–T5 (user, 2026-08-16).
The workflow from the parameter tab through to a saved description is the thing
with a user waiting on it; the editor consolidation is a refactor of working
code whose payoff is mostly in what it prevents. Doing it first would mean
porting an editor whose call sites are still changing.

## 3. Two things deliberately not done

**The JSON mode is vendored; a mode for molbuilder's own formats is not.**
`.fdf`, `.xyz`, `.out` and the logs render as plain text because CodeMirror has
no upstream mode for them. Writing one is a real option and a separate piece of
work — it is not a gap in this plan.

**The tab does not poll a running job.** `task-setup.md` § 10. Reading a folder
needs no target machine; watching a cluster run is the Results tab's problem.

---

## 4. The two mistakes this work already made, so they are not repeated

Both were **names that were wrong rather than code that was wrong**, both
survived a green test suite, and both were found only by reading the API being
called:

1. `missing_ok` passed where `lib/projects/api.js` takes **`missingOk`** — so
   the option was silently dropped and every folder without a description
   logged a failed-resource error.
2. The hand-over button written against `structurePage.structureEnvelope()` and
   `formSchema.lastSchema()` — **neither exists anywhere in the tree**. It would
   have parsed, loaded, and done nothing.

**The rule this earns:** when wiring to an API, open it. A test that never calls
the thing it wraps cannot catch an argument name, and both of these are now
pinned by tests that read the source.
