# Code conventions

Small, enforced-by-review conventions that keep the codebase legible and safe to
change. Linked from `architecture.md` § 0. Add a convention here only when it has
bitten us and a rule prevents the recurrence.

---

## 1. Module-provenance header (MANDATORY for framework-level code)

**Every code file must open with a comment that states, explicitly:**

1. **MODULE** — which module/subsystem this file belongs to (and, if the module
   spans several files, the sibling files + the public surface each mounts).
2. **ROLE** — what the code is responsible for and, just as important, what it is
   **not** (the boundary it must not cross).
3. **USED BY** — the concrete callers / consumers of this file's public surface.

For **framework-level / system-wide code** (blueprints, shared `lib/*` modules,
persistence layers, anything multiple subsystems depend on) this header is **not
optional**. A file that bundles more than one concern must label **each** concern
group with its own MODULE / USED BY, so the groups can never be conflated.

### Why this exists

A single blueprint file, `blueprints/workingcopy.py`, silently hosted **three
unrelated concerns** behind one URL prefix:

- the **obsolete** structure "working-copy door" (`/open`, `/save`, `/update`, …),
  superseded by the projects-sidebar contract and no longer called by any client;
- the **live workspace/state-timeline persistence** (the `*-state` routes, used by
  `lib/workspace/dispatcher.js`) — since extracted to `blueprints/workspace.py` at
  `/api/workspace/state/*`, which is exactly the fix this rule drives;
- and functions imported by *other* live routes (`StructureCodec`, used by
  `/api/structure/resolve-cell`).

With no per-group provenance, a surface-level glance read the whole file as one
"working copy" thing, and a cleanup nearly deleted the **live** persistence layer
that molview's save/retract depends on. The header is what makes removability a
question you can answer by reading, not by guessing.

### The rules that go with it

- **Never judge code dead/removable from a surface grep.** Trace the actual
  callers end-to-end *and* read the provenance header.
- **Doc-vs-code disagreement = UNDETERMINED.** If a doc says a route/function is
  used but the code shows no caller (or vice-versa), do **not** delete it. Resolve
  the contradiction (usually a stale doc) and get an explicit decision first.
- **A file hosting multiple route groups / concerns must delineate them** with
  their own MODULE + USED BY blocks.

### Template

```
/* <one-line what-this-is>.
 *
 * MODULE: <subsystem> (<dir/>; contract: docs/protocols/<x>.md).
 *   <sibling files + the public surface each mounts, if the module spans files>
 *   <server backend routes, if any>
 *
 * ROLE: <responsibility>.  <the boundary it must NOT cross>.
 *
 * USED BY:
 *   - <caller file> — <how / which part of the surface>.
 *   - ...
 *
 * Public surface: <the exported/mounted API>.
 */
```

Reference implementations: `lib/workspace/dispatcher.js`,
`lib/workspace/snapshot-io.js`.
