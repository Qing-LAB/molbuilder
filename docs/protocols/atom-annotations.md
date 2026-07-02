# Unified atom annotations + fused viewer/selection — design

**Status: IN PROGRESS (2026-07-01). Phases 1-3 SHIPPED (Structure layer + sidecar v4 + fdf registry); Phases 4-6 remain — see §7.** Authoritative for
the work that (1) unifies per-atom metadata into one extensible model, (2) makes
selection always-available + filterable by any metadata, and (3) fuses the
MolViewer and atom-selection into one responsive component. Implement in the
phases in §7; each phase is test-gated + committed.

**Companions:** `types/structure.md` (the `Structure` dataclass this extends),
`atom-selection-guide.md` / `molviewer-guide.md` (the modules being fused),
`sidecar-contract.md` + `data-vocabulary.md` (the `.molstruct.json` persistence),
`engines/siesta.md` (the fdf emission this feeds).

---

## 1. Motivation

Per-atom metadata is fragmented and hard-coded:

- **`Structure`** (canonical) carries `regions: Dict[str,List[int]]` +
  `frozen_atoms: List[int]` — two *hard-coded* channels.
- The **JS selection store** mirrors them as `labels[]` + `isFrozen`.
- The **viewer** keeps only element + xyz.
- **fdf** consumes `frozen_atoms` → `%block Geometry.Constraints` and `regions`
  → transport/electrode blocks — again, hard-coded per field.

Adding *any* new per-atom concept (per-atom charge, spin, basis-set override,
constraint type, …) today means touching `Structure`, the sidecar, the JS store,
and the fdf emitter separately. We want **one extensible model** that all four
speak, so richer information flows to every application (fdf + future setup
scripts) without schema churn — and so the selector can filter on *any* of it.

---

## 2. The annotation model (the "open annotations layer")

`Structure` keeps its typed columns (`elements`, `positions`, `atom_names`,
`residue_ids`, `residue_names`, `chain_ids`). On top sits **one extensible
annotations layer**: a set of named **channels**, each a per-atom metadata
stream. Three channel **kinds** cover current + foreseeable needs:

| Kind | Shape | Meaning | Examples |
|---|---|---|---|
| `tag` | name → set of atom indices (multi-valued: an atom may be in many) | named region membership | `L-electrode`, `device`, `bridge` |
| `flag` | name → boolean per atom (a subset) | a yes/no property | `frozen` |
| `value` | name → scalar per atom (sparse map idx→value) | a per-atom number/enum | `charge`, `spin`, `basis_override`, `constraint` |

Each channel also carries light **presentation + emit hints**:
`{ kind, data, color?, fdf?: <emit-strategy id> }`. (`color?` is a *proposed*
centralization — region colors today live JS-side in
`region-label-definitions.js` / `selection-panel.js`, not in the model; the
design pulls them into the channel so viewer + panel share one source.)

**Built-ins (backward-compatible):**
- **each region label → one `tag` channel** (channel name = the label, e.g.
  `L-electrode`); the registry is `{L-electrode: tag, device: tag, …}`.
- `frozen_atoms` → a **`flag`** channel named `frozen`.

`Structure.regions` and `Structure.frozen_atoms` remain as **convenience
accessors backed by the annotations layer** — existing callers keep working;
they're just views over channels now.

### 2.1 Index stability (channels remap on structure edits)

Channels are **keyed by atom index**, so any structure mutation (add / delete
atom) MUST remap every channel or metadata silently corrupts (the same hazard
`selection_remap` guards for the selection). This already exists for the two
built-ins — `modify.py::remap_frozen_and_regions(old_to_new)` — and the
annotations layer **generalizes it to iterate all channels** (drop indices that
vanished, translate the rest through `old_to_new`; `value` channels remap their
key set). Every modify op that returns a `selection_remap` must also carry the
channel remap. This is a load-bearing correctness requirement, not an add-on.

**Structure API (as shipped, Phase 1):**
```python
struct.annotations                       # {name: AtomChannel}  -- extensible extras
struct.channels() -> {name: AtomChannel} # unified: regions(tag) + frozen(flag) + extras
struct.get_channel(name) -> AtomChannel | None
struct.atom_annotations(i) -> dict       # everything on atom i (for the UI/filter)
struct.set_channel(name, AtomChannel(...))  # set an EXTENSIBLE channel; built-in
                                            # names rejected -> edit .regions/.frozen_atoms
```
Built-ins keep their existing storage (`.regions` / `.frozen_atoms`) and are
*surfaced* as channels by `channels()`; extensible channels live in
`.annotations`. Module helpers `copy_annotations` / `remap_annotations` handle
copy + the all-channel index remap (§2.1).

---

## 3. Persistence (`.molstruct.json` schema v4)

Extend the sidecar: add **`annotations: {name: {kind, data, color?, fdf?}}`**
**alongside** the fields it already carries (`regions`, `frozen_atoms`,
`structure_hash`, `cell`, `pbc`) — purely additive. Keep writing `regions` +
`frozen_atoms` for one release (v4 reader maps a v3 file's `regions`/
`frozen_atoms` into channels; v4 writer emits both `annotations` AND the legacy
fields until consumers migrate). Bump `SCHEMA_VERSION` 3 → 4, register the v4
shape in `data-vocabulary.md`, and keep the atomic-write + label-propagation
rules (save-flow §4) per-channel.

---

## 4. fdf / setup-script flow (channel emit-strategy registry)

**Additive, not a rewrite.** The two built-ins keep their existing, Sol-
validated emission untouched — they ARE the built-in strategies:

| Built-in channel | Emits (unchanged) |
|---|---|
| `frozen` flag | `%block Geometry.Constraints` (1-based) — `siesta/input.py` |
| region tags (`L-electrode`/`device`/…) | transport/electrode blocks (`TS.Atoms`, electrode labels) — `transport/transiesta.py` |

On top, a **strategy registry** lets **extensible** channels emit: a channel in
`Structure.annotations` may carry an `fdf` = `<strategy-id>`; a registered
strategy `(channel, struct) -> fdf lines` is invoked during fdf assembly. A
channel with **no** registered strategy is **carried but not emitted** (a
generic user tag never becomes an electrode block; the validator may warn
"channel X present, no fdf consumer").

This is the extension point — new metadata (e.g. a `charge`/`initspin` value
channel) → register one strategy → it emits, **no emitter rewrite and no risk
to the proven frozen/region paths**. The same registry serves other setup
scripts (PySCF, transport) later. (A future cleanup MAY migrate the two
built-ins into the registry once it's battle-tested; not required now.)

---

## 5. JS unified model + always-on filterable selection

- The selection store's `Atom` generalises `labels[]`/`isFrozen` into the same
  **channels** (mirrors §2). `atom.annotations` exposes everything for the UI.
- **Selection is always mounted** (fused module, §6). The **filter** operates on
  **any channel** — element, residue, any `tag`, any `flag`, any `value` (range
  predicates for scalars) — so "select all `frozen`", "select `bridge` ∩ carbon",
  "select `charge > 0`" are uniform. This is the efficiency win: one filter UI
  over all metadata, available during build / modify / results.

---

## 6. Fused viewer + selection module + responsive UI

**Every** molview-embedding tab uses the **one** fused module — selection is
**always mounted** (not an opt-out). Selection is not just for editing: it
drives filtering + highlighting during build, modify, AND results inspection,
so it's valuable everywhere. Per-tab behavior is confined by **arguments**, not
by omitting selection.

- **Data:** the workspace selection store stays the single source of truth (no
  second copy — avoids the two-writer class we fixed); the fused module owns the
  viewer + panel + adapter + the unified API, over the § 2 annotation channels.
- **API:** `viewer.embed(host, opts) → handle`; the handle exposes view +
  selection/filter ops.
- **UI (responsive):** **wide** → viewer + selection as **two matching-height
  cards**; **narrow** → **one card with two tabs** (View / Selection). Uses the
  responsive-grid floor from `mobile-layout.md`.

**The two behavior-confining args (this is what differs per tab):**

| Arg | Values | Effect |
|---|---|---|
| `mode` | `"modify"` \| `"readonly"` | `modify`: full structure-edit ops + selection editing. `readonly`: view + selection for **filter/highlight/inspect only** — no edit ops, no writes (the Results inspectors). |
| `persistence` | `"workspace"` \| `"ephemeral"` | `workspace`: this view's structure+selection+annotations **are** the workspace — persisted to sessionStorage + restored on nav (the Molbuilder/Modify tab). `ephemeral`: a transient view **re-derived from its source each mount** — never persisted, never restored (the Results inspectors, driven by the selected result file). |

### 6.1 Data-persistence contract (when data is kept vs lost)

The user must be able to trust *when their work survives*. Unified rule:

> **Your working data = the workspace** (structure + selection + annotations).
> Under `persistence: workspace` it **survives tab-switches and page reloads
> within a session**, and the app **warns before discarding unsaved changes**
> (the dirty-gate). It becomes **durable only when you Save** (to `<name>.xyz`
> + its `.molstruct.json` sidecar). A `persistence: ephemeral` (read-only
> inspector) view **owns no data** — it is re-derived from its source file, so
> there is nothing to lose.

Per event:

| Event | `persistence: workspace` | `persistence: ephemeral` |
|---|---|---|
| Switch tab & return | **preserved** (restored from the snapshot) | re-derived from source (nothing owned) |
| Page reload | **preserved** (sessionStorage) | re-derived / empty |
| Load a different file with unsaved edits | **gated** — warning modal; lost only on explicit *Discard* | n/a |
| **Save** | written to disk (`.xyz` + `.molstruct.json`, incl. annotations) | n/a |
| Browser tab/session **closed** | **LOST unless Saved** (sessionStorage is per-session) — the UI must make this explicit | n/a |

Annotations (labels/frozen/new channels) follow the **same** contract as the
structure — persisted with the workspace + written to the sidecar on Save.
This section extends `workspace-contract.md` § 4 (persistence) + `save-flow.md`
(the dirty-gate); those remain authoritative for the mechanics.

---

## 7. Phased implementation plan (each: implement → test → commit)

0. **This design** — approve.
1. **Structure annotations layer** (Python) — **SHIPPED (2026-07-01).**
   `AtomChannel` + `Structure.annotations` + unified `channels()`/`get_channel()`/
   `atom_annotations()`/`set_channel()`; `copy_annotations`/`remap_annotations`;
   `modify.py` delete-remap + verbatim rebuilds carry annotations; `copy()`/
   `translated()` carry them. tests/test_atom_annotations.py (11).
2. **Sidecar v4** — **SHIPPED (2026-07-01).** `SCHEMA_VERSION 3->4`; `to_dict`
   writes `annotations` + dual-writes regions/frozen; `apply_to_structure`
   reads them; parse-side reads v3+v4 and validates channel indices at load;
   AtomChannel.to_json/from_json round-trip. tests/test_sidecar_annotations.py.
3. **fdf channel emit-strategy registry** — **SHIPPED (2026-07-01).** Additive:
   `annotations_fdf.py` (register_fdf_strategy/emit_channels/unregistered_
   channels); wired into `siesta/input.py` (no-op when no registered channels);
   built-in frozen/region emission untouched. tests/test_annotations_fdf.py;
   335 fdf tests unchanged.
4. **JS unified `Atom` + channel filter** — store carries channels; filter by any.
5. **Fused module + migrate ALL molview tabs** — embed mounts viewer+panel+
   adapter with the § 6 `mode`/`persistence` args; **every** molview-embedding
   tab moves to it (selection always mounted): Modify/Molbuilder =
   `mode:modify, persistence:workspace`; Results structure/trajectory/spectra
   inspectors = `mode:readonly, persistence:ephemeral`; others per the
   investigation below. Responsive 2-card ↔ tabbed UI. E2E-gate (nav-selection
   loop + full Modify E2E + the Results inspectors).
   - **Investigation first (§ 8):** before migrating, audit each embed site for
     data-structure / interface / logic changes; confirm the right
     `mode`/`persistence` per tab; only then migrate.
6. **Docs + data-vocabulary v4** — register the sidecar v4 shape; merge
   `molviewer-guide.md` + `atom-selection-guide.md` into one; update
   `embedded-viewer.md`, `atom-selection.md`, `types/structure.md`,
   `sidecar-contract.md`, `engines/siesta.md`; full E2E.

**Guardrails:** `Structure` is the lingua franca — keep back-compat accessors so
consumers don't break mid-migration; **channels remap on every add/delete**
(generalize `modify.py::remap_frozen_and_regions` to all channels — §2.1, a
correctness must); the selection store stays single-source (no duplicate state);
dual-write the sidecar until readers migrate; a channel with no fdf strategy is
carried-not-emitted; each phase gated by the relevant test suite before the next.

---

## 8. UI-integration investigation (per embed site)

Audit of the **5** current `viewer.embed` sites and what full integration needs
(the Phase 5 prerequisite).

| Tab / embed | today | target args | changes needed |
|---|---|---|---|
| **Modify/Molbuilder** (`modify/viewer.js`) | full edit; selection wired *externally* via `selection-bootstrap.js` (mounts panel + attaches adapter) | `mode:modify, persistence:workspace` | **interface**: the embed mounts the panel+adapter itself; delete the manual bootstrap wiring. **logic**: none new (behavior preserved). |
| **Results · structure** (`inspectors/structure.js`) | bare + already uses `selection.measurements` (pick→distance/angle chip) | `mode:readonly, persistence:ephemeral` | **gains** the selection panel + filter (module-provided). Pick/measure already present. |
| **Results · trajectory** (`trajectory/core.js`) | bare + `pick:triple` + measurements + animation | `mode:readonly, persistence:ephemeral` (keep `pick:triple`, animation) | **gains** panel+filter; keep animation + triple-pick. |
| **Results · spectra** (`spectra/core.js`) | bare + **`pick:{mode:"none"}`** (vibrational-mode viewer) | `mode:readonly, persistence:ephemeral, pick:none` | selection panel present but **pick stays off**; filter/highlight only. |
| *(future build/other)* | — | per case | assess when added. |

**Key findings driving the design:**
- **The fused module must OWN the panel host** — render the selection panel
  inside its own responsive card. Today only `modify.html` +
  `_trajectory_inspector.html` have panel/host partials; the inspectors don't.
  Making the module render the panel means *any* embed gets selection without
  each host shipping a `#selection-host`.
- **Three orthogonal args compose** — `mode` (edit vs readonly), `persistence`
  (workspace vs ephemeral), and the **existing `pick` opt** (spectra needs
  `pick:none` even in readonly). So "selection always mounted" ≠ "pick always
  on": the panel/filter is always available; click-pick is governed by `pick`.
- **No data-structure change for inspectors** — their structures come from
  parsed results and may carry no annotations; the filter still works on the
  always-present element/residue channels, plus any annotations that are there.
- **Persistence is already what each tab does** — Modify persists (workspace),
  inspectors re-derive from the selected result file (ephemeral). The args make
  the existing behavior *explicit + contractual* (§ 6.1), not new behavior.
