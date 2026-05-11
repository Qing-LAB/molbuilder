# molbuilder — documentation index

This directory holds the **test contracts** for what molbuilder must
do, organised by purpose.  Every feature has a focused spec; every
claim a spec makes must be testable; tests must reference the spec
rather than the implementation they test.

Project-level design (mission, architectural principles, decisions
made, active roadmap items) lives in [`design.md`](design.md) one
level up.  `design.md` sits *above* these per-feature specs.

## The rule

> **Tests must be derivable from the spec without reading the
> implementation.  Code reviews must verify code matches spec, not
> code matches reviewer's expectations.**

A bug shipped early in the project because a code review checked the
implementation against itself — tests asserted "the string `mol =
gto.M(...)` appears in the generated script" rather than "the
generated script must not truncate `<job>.log` between stages".  When
the implementation was wrong, the test was wrong in lock-step.  The
specs in this directory decouple the two.

## Directory layout — by purpose

```
docs/
├── design.md                   # cross-cutting design + decisions log
├── README.md                   # this file
├── img/                        # README screenshots
├── protocols/                  # cross-cutting interfaces
│   ├── web-api.md              #   build-side Flask endpoints
│   ├── watch-api.md            #   watch-side Flask endpoints
│   ├── cli.md                  #   click-based CLI surface
│   └── job-layout.md           #   on-disk run-directory convention (job-layout v1)
├── types/                      # L1 data-type contracts
│   ├── structure.md            #   Structure dataclass + XYZ/PDB I/O
│   ├── parsers.md              #   TrajectoryParser plug-in interface + per-engine specifics
│   └── chemistry.md            #   charge auto-detect, phosphate protonation, dipole estimate
├── engines/                    # per-engine emitter specs
│   ├── siesta.md               #   SIESTA .fdf emitter
│   ├── pyscf.md                #   PySCF runnable-script emitter
│   └── builders.md             #   build-backend contract (peptide/DNA/RNA/SMILES/name)
└── tabs/                       # per-tab UI + (when needed) supporting assets
    ├── modify.md               #   Modify tab: atoms, junctions, electrode placement
    └── watch.md                #   Watch tab: 3Dmol viewer + Plotly plots + control panels
```

**Categories.** What each folder is for:

- **`protocols/`** — how parts of the system talk to each other (HTTP
  API, CLI surface, on-disk file layout).  Specs here describe
  contracts between components.
- **`types/`** — L1 data-type and parser contracts.  Specs here
  describe the shape of values flowing between components.
- **`engines/`** — per-engine emitter specs (one per downstream code
  we generate input for, plus the build-backend contract).  Specs
  here describe what we *write*, not what we *do*.
- **`tabs/`** — per-tab UI specs.  Single-file specs stay as
  `<tab>.md`; tabs needing multiple assets (bibliography, sub-specs)
  become subfolders, e.g. `tabs/spectra/spec.md` +
  `tabs/spectra/references.bib`.

Bibliographies live alongside the spec they cite, in the same
subfolder.  Adding a future tab with citations means a new folder
with both files together — exactly one place to look when reading
either.

## Versioning

This is a 1.x project.  Spec changes that remove or rename promised
output files require a minor version bump (1.x → 1.x+1) AND a
deprecation note in the design.md decisions log.  Adding new
optional fields or files is a patch-level change.
