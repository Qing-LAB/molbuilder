# Unified atom annotations + fused viewer/selection — design

**Status: PROPOSED (2026-07-01). Design-first; no code yet.** Authoritative for
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
`{ kind, data, color?, fdf?: <emit-strategy id> }`.

**Built-ins (backward-compatible):**
- `regions` → **`tag`** channels (one namespace; the existing labels).
- `frozen_atoms` → a **`flag`** channel named `frozen`.

`Structure.regions` and `Structure.frozen_atoms` remain as **convenience
accessors backed by the annotations layer** — existing callers keep working;
they're just views over channels now.

**Structure API (proposed):**
```python
struct.annotations                       # {name: Channel}
struct.get_channel(name) -> Channel | None
struct.set_tag(name, indices)            # + add_to_tag / remove
struct.set_flag(name, indices)           # frozen = set_flag("frozen", …)
struct.set_value(name, {idx: val})       # future scalars
struct.atom_annotations(i) -> dict       # everything on atom i (for the UI/filter)
```

---

## 3. Persistence (`.molstruct.json` schema v4)

Extend the sidecar: add **`annotations: {name: {kind, data, color?, fdf?}}`**.
Keep writing `regions` + `frozen_atoms` for one release (v4 reader maps a v3
file's `regions`/`frozen_atoms` into channels; v4 writer emits both `annotations`
AND the legacy fields until consumers migrate). Register the v4 shape in
`data-vocabulary.md`. Atomic-write + label-propagation rules (save-flow §4)
apply per-channel.

---

## 4. fdf / setup-script flow (channel-driven emission)

The fdf emitter stops special-casing `frozen`/`regions` and instead **iterates
channels**, dispatching by kind + a per-channel emit strategy:

| Channel | Emits |
|---|---|
| `frozen` (flag) | `%block Geometry.Constraints` (1-based, as today) |
| region `tag`s | transport/electrode blocks (`TS.Atoms`, electrode labels) |
| future `value` (e.g. `charge`) | per-atom directives when a strategy is registered |

A channel with no registered strategy is **carried but not emitted** (and the
validator can warn "channel X present, no fdf consumer"). This is the extension
point: new metadata → register one emit strategy, no emitter rewrite. Same model
serves other setup scripts (PySCF, transport) later.

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

One component owns the 3D view **and** the selection (fully fused, selection
default-on; `selection: false` for the Results bare embeds — structure /
trajectory / spectra inspectors):

- **Data:** the workspace selection store stays the single source of truth
  (no second copy — avoids the two-writer class we fixed); the fused module owns
  the viewer + panel + adapter + the unified API.
- **API:** `viewer.embed(host, { selection, ... }) → handle`; the handle exposes
  both view ops and selection/filter ops.
- **UI (responsive):** **wide** → viewer + selection as **two matching-height
  cards** side by side; **narrow** → **one card with two tabs** (View /
  Selection). Uses the responsive-grid floor from `mobile-layout.md`.

---

## 7. Phased implementation plan (each: implement → test → commit)

0. **This design** — approve.
1. **Structure annotations layer** (Python) — channels + `regions`/`frozen_atoms`
   as backed accessors; full API compat; unit tests.
2. **Sidecar v4** — `annotations` + v3 back-read + dual-write; tests + data-vocabulary.
3. **fdf channel-driven emission** — frozen/region via channels + strategy
   registry; the existing siesta/transport fdf tests are the net.
4. **JS unified `Atom` + channel filter** — store carries channels; filter by any.
5. **Fused module + responsive UI** — embed mounts viewer+panel+adapter; migrate
   the Modify tab; set `selection:false` on the 3 Results embeds; E2E-gate
   (nav-selection loop + full Modify E2E).
6. **Docs** — merge `molviewer-guide.md` + `atom-selection-guide.md` into one;
   update `embedded-viewer.md`, `atom-selection.md`, `types/structure.md`,
   `sidecar-contract.md`, `engines/siesta.md`; full E2E.

**Guardrails:** `Structure` is the lingua franca — keep back-compat accessors so
consumers don't break mid-migration; the selection store stays single-source
(no duplicate state); dual-write the sidecar until readers migrate; each phase
gated by the relevant test suite before the next.
