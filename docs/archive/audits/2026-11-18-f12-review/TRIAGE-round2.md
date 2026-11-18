# F12 review round 2 — holistic (one source per fact, one data shape, nothing duplicated or handcrafted)

14 readers, whole files of a coverage table, code and tests, read-only.
About **395 findings** (`r2-*.md`, this folder). Not yet re-read finding by
finding: each is re-read in the release that owns its module, when that
module is read whole for the fix (*a review list is input, not the work
order*).

## The duplicates that CAUSE round-1 defects (one redesign closes both)

| round 2 | its round-1 defect | the one home |
|---|---|---|
| R2-1 #1 save-then-write copied 4× (prep ×3, launch) | A-D1 traceback on a failed write | one "save, carry out, record how far" door, as `migrate` does |
| R2-3 F1 bundle lists hand-typed | B-P1 modules missing beside a job | `TRAVELS_WITH`, as the monitor bundle |
| R2-3 F2 the job's GPUs worked out 3× in bash | B-D1 GPU 0 shown/bound | one per-rank device answer |
| R2-4 P1 solvent name normalised 7× | D-D9 `eps=?`; **measured: `' water'` passes the check, prep traceback** | one normaliser |
| R2-4 P5 identity items typed in shared template | D-D3 PySCF `job_name` per stage | the engine record's `identity_field` |
| R2-4 P8 XC family from authors beside `xc_functional` | D-D1 vdW-DF never preparable | the template's own field |
| R2-5 #3 compose refusals caught by 3 private lists | E-D1 transport 500 / traceback | `rung.compose_refusals` |
| R2-8 C-5 O3′–P bridge computed twice | H-D2 false phosphopeptide warning | the builders' `_common` rule |
| R2-8 C-9 DNA input filtered before the grammar | H-D5 silent drops | the one sequence grammar |
| R2-10 F1 page re-implements the builders' grammars | (new) peptide panel refuses `AR[SEP]C` the server builds | the server's grammar |
| R2-11 F10 read-only MolView mount copied 3× | J2-D2 trajectory TypeError | one mount helper |
| R2-13 F10 lane-0 membership decided twice | L-D1 wrong stick highlighted | `_seal.js` |

## Safety

- **R2-14 #2** — the suite's refusing scheduler stubs `sbatch`/`srun`/`salloc`
  but not `squeue`/`scancel` (`scheduler/kinds.toml` names them): on a
  cluster login node a basic test could query or cancel a real job. (This
  workstation has no scheduler.)
- **R2-9 #5** — two file-name rules: rename/move/copy take any name, upload
  and write the strict one → a file renamed `my notes.txt` can no longer be
  saved from the preview; `.draft` vanishes from the sidebar.
- **R2-2 #5** — every new project's `pseudopotential/README.md` says
  `psml_lib = pseudopotential` reads the project's folder; it reads
  `projects/pseudopotential/` (seen on F12's own water run).

## Biggest one-home items by module (the rest in each report)

- **jobset**: describe written twice (CLI and hand-over), already differing;
  bench grid names typed 4×; SIESTA restart keywords named in shared code.
- **run files/records**: three readers disagree which catalogue row a name
  is (a person's `plot.py` described as an engine input); task/job-request
  refusal helpers copied; each ladder rule checked twice.
- **run script/monitor**: config row order re-types `template.SAID`;
  init-config detects its own sentence; source-build layout in 3 places.
- **template/engines**: catalogue ranges and anchors hand-copied
  (displacement range → double warning; `BENCH_FIELDS`; PySCF `mf.*` names).
- **transport**: lead order decided again by mean z with the axis typed;
  citation kind decided in 5 places, spelled two ways; window read by token
  position ignoring its unit.
- **spectra**: three JSON readers with three NaN rules; engine name spelled
  "siesta" in one data file and "SIESTA" in the others of the same export.
- **parse**: one run reading as two classes, the copy dropping untaught
  fields; three Fortran number readers (one refuses `1.0D-03`).
- **structure/builders**: helix forms and terminals have no home (unknown
  terminal → KeyError or quietly "OH"); a missing tool raised 4 ways;
  `PERIODIC_TABLE` hand-copied, stops at Z=103.
- **web server**: checkpoint routes classify errors themselves; modify
  routes re-check axis/centre words; atom-selection reads a second shape.
- **pages**: server facts recomputed (axis kinds, builder rules, memory
  unit "GB" vs "GiB" on one card); five script loaders; seven copies of the
  load-ended signal; bench node census built in the browser.
- **MolView**: contact-distance readout from a hand copy of a file nothing
  reads; dead filter helpers beside two hand-kept lists; three server rules
  copied in.
- **viewers**: server sentences drawn raw (` -- `); IR/Raman words with two
  homes; modes-table columns described four times.
- **tests**: a fixture/app/server copied per file (84 app builds; 18 copies
  of `app_at`); machine records written by hand 13 ways; a findings check
  expecting the wrong dash; a test-only TBtrans window reader looser than
  the product's.

## Deletions proposed by readers (shown before any removal)

`/api/files/write`'s generate-and-save branch and its tests; the unused
`watch` re-export in `web/blueprints/__init__.py`;
`test_blueprint_label_adoption.py`; the `capture_on_fail` marker and the
`window.__molbuilder_modify_test` hook it keeps alive; dead `FdfParams`
fields; MolView's unused `KIND`/`atomChannels`/`channelKinds`; the five
unread lifecycle aliases in `lib/spectra/core.js`.

## Design questions raised

- MolView's atom filter posts its own atom shape (documented exception,
  `web-api.md`); moving it onto the one envelope is a design change (R2-12 A8).
