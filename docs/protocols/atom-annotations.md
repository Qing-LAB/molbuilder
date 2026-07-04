> **Scope (2026-07-03).** This doc is the **per-atom annotation *channels* model**:
> the data model (§§ 2–4) that `structure.py`, `sidecars/molstruct.py`, `parse/`,
> `siesta/input.py`, and `script_emit.py` depend on, plus its JS mirror (§ 5).
>
> The fused viewer/selection design that had accreted here (former §§ 6–9) was
> **removed** 2026-07-03. The molview module — the viewer + selection, k-grid,
> measurement — is defined in [`molview-module.md`](molview-module.md).

# Unified atom annotations — the per-atom annotation channels model

**Status: IN PROGRESS (2026-07-02). Phases 1-3 + 4a-4c SHIPPED (annotations model, sidecar v4, fdf registry, JS channel model + filter); 4d DEFERRED (no value-channel producer yet); Phases 5-6 remain — see §7.** Authoritative for
the work that (1) unifies per-atom metadata into one extensible model, (2) makes
selection always-available + filterable by any metadata, and (3) fuses the
MolViewer and atom-selection into one responsive component. Implement in the
phases in §7; each phase is test-gated + committed.

**Companions:** `types/structure.md` (the `Structure` dataclass this extends),
`atom-selection-guide.md` / `molviewer-guide.md` (the modules being fused),
`sidecar-contract.md` + `data-vocabulary.md` (the `.molstruct.json` persistence),
`working-copy-persistence.md` (the module's load/edit/save, §6.1),
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

## 3. Data-model persistence (engine-agnostic, round-trippable)

The annotation channels are the molview/selector **DATA MODEL**. They are
persisted **identically wherever a structure is saved** — engine-agnostic, the
same shape everywhere, so a saved structure round-trips its full metadata:

- **`.molstruct.json` sidecar (schema v4)** — the primary sidecar record.
  Add `annotations: {name: {kind, data, color?, fdf?}}` **alongside** the
  existing `regions`/`frozen_atoms`/`structure_hash`/`cell`/`pbc` (additive).
  Dual-write regions/frozen for one release; v4 reader maps a v3 file's
  regions/frozen; bump `SCHEMA_VERSION` 3→4; register in `data-vocabulary.md`;
  keep atomic-write + label-propagation (save-flow §4) per-channel. **(Phase 2,
  shipped.)**
- **The `.fdf` / `.py` ATOM-METADATA reserved comment block** — the *same* data
  embedded in the generated script's comment area (`script-contract.md` §4;
  `script_emit.emit_atom_metadata` writes it, `apply_inbody_atom_metadata` reads
  it back). Carries regions + frozen + **the annotation channels** — the
  script's engine-agnostic copy of the data model (a PySCF script carries the
  identical block); block bumped to v4, round-trips via emit/apply, wired into
  the siesta/pyscf/transport emitters. **(SHIPPED 2026-07-01.)**

This is **data**, not engine setup. It says nothing about how a simulation runs;
it just records what the user labeled. See § 4 for the separate concern.

---

## 4. Engine parameter setup (extraction into the engine's required input)

**This is a DIFFERENT concern from § 3 — do not conflate them.** § 3 *persists*
the data model (engine-agnostic, round-trip). § 4 *reads* that data model and
**translates the relevant parts into the input blocks the engine REQUIRES** to
run a correct simulation. This is dictated by the engine's physics, is **one-way**
(data model → engine input), and is **not** how the data is stored.

The two built-in translations (keep their existing, Sol-validated code):

| Metadatum (from § 3) | Engine block it's translated INTO | Why the engine needs it |
|---|---|---|
| `frozen` | SIESTA `%block Geometry.Constraints` | tells the relaxer which atoms not to move — load-bearing for correctness (e.g. electrode atoms fixed so the self-energy / lead coupling is right) |
| region tags | transport/electrode blocks (`TS.Atoms`, …) | defines the device/lead partition the NEGF solver requires |

**The same `frozen` datum therefore plays two roles**: (a) it is *persisted* as
data in ATOM-METADATA + the sidecar (§ 3), AND (b) it is *translated* into
`Geometry.Constraints` as an engine parameter (§ 4). Same source, two distinct
outputs — persistence vs simulation setup.

**Extension point (additive).** A **strategy registry** lets *new* metadata
become engine parameters without touching the proven built-ins: a channel in
`Structure.annotations` may carry `fdf = "<strategy-id>"`; a registered strategy
`(channel, struct) → engine lines` is invoked during assembly (e.g. a future
`initspin` value channel → `%block DM.InitSpin`). A channel with **no** strategy
is **not translated** — it still *persists* via § 3; it simply isn't a
simulation parameter (the validator may warn "channel present, no engine
consumer"). No emitter rewrite; no risk to frozen/region. The same registry
serves PySCF / transport setup later.

---

## 5. JS unified model + always-on filterable selection (Phase 4)

The JS mirrors § 2: everything filterable is a **channel**. To keep this clean
(no bolting onto the store), a **pure channel-model layer** sits below the store,
panel, and viewer-adapter — a single place that answers "what channels does this
atom / this structure have?".

**Layers (each depends only on the one below):**

| Layer | Module | Owns |
|---|---|---|
| L1 low-level presentation API | **`lib/workspace/_atom-channels.js`** + **`_atom-index.js`** | `atomChannels`/`channelKinds` (channel taxonomy + order) and `toDisplay`/`fromDisplay`/`shiftExpression` (index base) — pure, **no DOM/store/HTTP** |
| L2 store | `_selection-store-impl.js` | holds atoms + filter drafts; `knownChannels()` (via L1); `_filterToRule` translates a draft → server rule |
| L3 UI | `selection-panel.js` (+ viewer-adapter) | renders the filter over `knownChannels()`; overlays colour by channel |
| server | `/api/selection/*` | supplies per-atom channels + evaluates rules |

### § 5.1 The L1 contract (low-level presentation API)

**L1 is the low-level presentation API.** It exists only to serve the UI (the
backend never converts index base or enumerates channels), so it *is*
presentation — the primitive tier. It owns two things and nothing else:

1. **Primitives + conventions** — the display value (`toDisplay(i) = i+1`), the
   channel taxonomy (`category`/`tag`/`flag`/`value`), and the stable channel
   enumeration order. It returns **values/model**, never finished presentation
   (no formatted `"#5"` string, no widget, no render).
2. **A contract higher layers build on.** L2 (store) and L3 (panel, viewer)
   **compose** L1's primitives into formatted, rendered UI and **must not
   re-derive** them (no rogue `atom.index + 1`, no re-implementing the
   regions-vs-frozen split).

**Conformance is bound by tests, not trust.** A layer that cannot import L1 at
runtime — the **standalone viewer embed** (`mol-viewer-embed.js`), which inlines
`+1` to stay self-contained — MUST still **provably conform**:

- `test_atom_index_js::test_viewer_index_labels_conform_to_l1` binds the viewer's
  inline `+1` to `toDisplay` (they can't drift).
- `test_atom_channels_js::test_selection_panel_frozen_name_matches_l1` binds the
  panel's `FROZEN_TAG_LABEL` to L1's `FROZEN_CHANNEL`.
- The L1 primitives themselves are pinned in `test_atom_index_js.py` /
  `test_atom_channels_js.py`; the panel's runtime 1-based display is pinned by
  the `test_atom_index_display_is_1_based` E2E.

This is the boundary: **low-level presentation API (values + conventions) below;
high-level presentation (format + render) above.**

**The channel model (L1):** an atom's channels unify its element, residue, each
tag/region, each flag (`frozen`), and each `value` channel:

```js
atomChannels(atom) -> { element:{kind:"category",value:"C"},
                        residue:{kind:"category",value:"ALA"},
                        "L-electrode":{kind:"tag"}, frozen:{kind:"flag"},
                        charge:{kind:"value",value:-1.0}, ... }
channelKinds(atoms) -> [{name,kind}]   // every filterable channel present
```

**Filter contract (L2):** filter drafts generalize from the current
`by_element`/`by_index`/`by_label` into a **uniform channel filter** —
`tag`/`flag` → membership, `category` → equals, `value` → a range predicate
(`>`, `<`, range). `knownChannels()` enumerates all filterable channels (so the
UI no longer special-cases regions-vs-frozen). Translation to server rules
stays in `_filterToRule`; tag/flag → `by_region`, value → a new `by_value` rule
(needs server support).

**Back-compat + boundaries:** the existing `by_element`/`by_index`/`by_label`
drafts keep working (they map onto the generalized model); the store stays the
single source of truth (§ 6, workspace-guide); value-channel filtering needs the
server to (a) include value channels in `/api/selection/atoms` and (b) resolve a
`by_value` rule — that server slice is called out, not hidden.

**Delivered incrementally (layered, node-tested):** (4a) the L1 pure module +
node tests; (4b) store `knownChannels` + generalized `_filterToRule`; (4c) panel
UI; (4d) server value-channel payload + `by_value`. Each is testable on its own.
