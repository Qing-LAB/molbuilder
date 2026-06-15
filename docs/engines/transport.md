# Transport engine (TranSIESTA)

> **Status: stub.**  The substantive contracts for the transport
> feature already live in sibling docs.  This file exists so the
> doc index has a stable landing for transport-engine-specific
> content as it grows, and so cross-references from the README
> + the GPU SIESTA env doc resolve.

## Where the live contracts already live

* **Tab UI + cross-tab workflow** — what the
  `/transport-calculation` tab does, its place in the file-driven
  task-tab model, the Phase D form skeleton:
  [`../tabs/architecture.md`](../tabs/architecture.md) § 8.
* **TranSIESTA generator** — `TransportConfig` dataclass + field
  metadata (workflow_group, validators, defaults), the
  `render_transiesta_fdf` emitter, the `TransiestaEngine` API:
  `molbuilder/transport/transiesta.py` (module is the canonical
  source; docstrings carry the per-function contract).
* **Web API** — `/api/transport/schema`, `/api/transport/render`,
  envelope shape, error semantics:
  [`../protocols/web-api.md`](../protocols/web-api.md).
* **Sidecar contract** — transport-job `.molstruct.json` shape
  (region labels: electrode / bridge / anchor; ordering
  invariants):
  [`../protocols/sidecar-contract.md`](../protocols/sidecar-contract.md).
* **Scientific validation** — Au-BDT-Au zero-bias fixture
  cross-check vs Reed 2006 / Stokbro 2003; the placeholder
  integration test that runs the emitted `.fdf` and asserts
  `T(E_F)` within factor-of-2:
  [`../protocols/scientific-validation.md`](../protocols/scientific-validation.md)
  + `tests/test_transport_au_bdt_au_validation.py`.
* **Roadmap** — bias scan, electrode `.TSHS` generation wizard,
  Transport results-tab framework, PySCF-NEGF backend:
  [`../roadmap.md`](../roadmap.md) (Transport section).
* **Decisions log** — every Transport-related decision (B.3
  zero-bias ship, atom-ordering preflight, engine_key metadata,
  mesh-cutoff default for Au, contour-bottom help text,
  high-bias warning, k_mesh_transverse tuple coercion):
  [`../design.md`](../design.md) — search for "Transport".

## What this file will own when it grows

* The end-to-end TranSIESTA emitter contract (input → output
  files, electrode lead orientation, k-mesh coupling, contour
  parameters).
* The PySCF-NEGF backend contract (when implemented).
* The transport results parser contract (`.transport.json`
  schema, parse_output API, T(E) + I-V data shapes).
* Per-engine integration tests + validation fixtures index.

Today these are split across the bullet list above; once the
transport feature consolidates (Transport results-tab framework
lands; bias-scan driver ships), the content lifts here.
