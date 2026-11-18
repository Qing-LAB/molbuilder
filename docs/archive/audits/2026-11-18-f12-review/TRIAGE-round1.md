# F12 review round 1 — verified triage (2026-11-18, at f578482db / 2.77.2)

13 readers (contract chapters against code, read-only) → 187 findings → 8
independent verifiers re-read every one: **178 confirmed, 4 plausible, 5
refuted**. Standing: 27 wrong results (W), 7 crashes (C), 41 misleading
messages (M), 21 cosmetic (N), 85 chapter sentences (T).
Reports `r1-*.md`; verdicts `v-*.md` (this folder).

Road: **own** = a person meets it with what molbuilder writes and offers;
**foreign** = only with hand-made input (a symlink, a hand-edited file, a deck
molbuilder did not write, an unusual git or machine setting).

## A. The builders build what was asked — `builders/`, `peptide.py`, `pubchem.py`, `validation/geometry.py`

| id | class | road | what |
|---|---|---|---|
| H-D1 | W | own | modified residues ([SEP] [TPO] [PTR] [MLY] [ALY]) placed in lab directions: P 0.235 Å from CB |
| H-D3 | W | own | Z-DNA asked `CGCGCG` is built 5′-GCGCGC-3′, no word (the panel's own example) |
| H-D4 | W | own | one-nucleotide AmberTools build lacks its 3′-OH H — a radical, no word |
| H-D7 | W | own | build by name drops stereochemistry: "D-alanine" comes out L |
| H-D2 | M | own | every phosphopeptide warned "disconnected backbone" (P read as a nucleic-acid phosphate) |
| H-D5 | M | own | DNA input silently drops dashes/digits the RNA and peptide grammars refuse |
| H-D6 | M | own | RDKit backend says it has no 3′-phosphate; the installed RDKit builds one |
| H-D8 | M+C | own | network errors all worded as a 30 s timeout; and every builder refusal on the CLI is a traceback |

## B. The deck writes the value the person set — `siesta/`, `pyscf/`, `template.py`, `pseudos.py`, `validation/stages.py`, `periodicity_gate.py`

| id | class | road | what |
|---|---|---|---|
| D-D2 | W | own | SIESTA deck rounds: bias 0.00625 V → 0.0063; eta 5e-7 eV → 0 (no broadening); 5e-5 and 6e-5 V both run 1e-4 |
| D-D3 | W | own | PySCF `job_name` accepted per stage → that stage reads `<new>.chk`, starts cold, next stage finds nothing |
| G-D1 | W | own | Cell page Apply with nothing changed marks the structure modified → a blank net charge then drops the recorded charge/spin |
| P-D2 | W | own | open-shell PySCF relaxation: each trajectory frame shows another evaluation's SCF history |
| D-D6 | W small | own | charged SIESTA vibration: Makov–Payne script prints a displaced geometry's energy as the total |
| D-D1 | M (blocks) | own | vdW-DF (DRSLL/LMKLL) offered, but every PSML is refused "XC-family mismatch" — can never be prepared |
| D-D7 | M | own | ladder check misses a tenfold loosening 1e-10 → 1e-9 of `scf_conv_tol` |
| D-D5, D8, D9, D10, D12, D13 | M | own | wrong default in a help (energy shift 0.02 vs 0.01 Ry); tips to raise a history already 8; `eps=?` header; unconverged geometry titled "Optimized"; stability warning for an untested state; a refusal naming a stage the kind lacks |
| D-D4, D11 | N | own | "Verbose inline comments" off is never passed to the deck writer |
| F-D1 | W | own (by hand) | `mb_vibration.pyz finish` (§ 5.5 recovery) states relaxation/Raman "not requested", wrong remedy |
| F-D2 | W | own, rare | PySCF vibration with `already_relaxed` ticked: the mode frame set never reads `info.calculation` |

## C. The person's files are safe — `web/blueprints/files.py`, `checkpoint.py`, `web/blueprints/build.py`, `workspace_storage.py`

| id | class | road | what |
|---|---|---|---|
| I-D1 = J1 P-D1 | W | foreign (a symlink) | delete/rename/move/copy act on a link's target: deleting a link `rmtree`s the folder it points to |
| C-D1 | W | foreign (git settings) | checkpoints inherit the person's git ignore → small files not saved, a restore deletes them unwarned (your machine: no such setting) |
| I-D2 | W | own | Send a hand-over twice into one folder under a new name → Save refused "holds 2 templates" though the page said it replaced it |
| I-D3 | C | own (disk full) | workspace save on a full disk answers an HTML 500, the reason never shown |
| C-U3 | T→code | own | a restore lets a run number be used again |

## D. A failure is said in words, never a traceback or HTML 500 — `cli.py`, `jobset/`, `transport/record.py`, `parse/instruments/reports.py`

| id | class | road | what |
|---|---|---|---|
| A-D1 | C | own (disk full) | a write failing in prep/launch → traceback, no ledger line saying where it stopped |
| E-D1, E-D2 | C | foreign | a damaged junction pair or SHARED block → Results 500/400, `summarize` traceback |
| P-D3 | C | foreign | a report with a non-number `sent_at` → `/api/notify/reports` 500 |
| H-D8 (CLI) | C | own | `molbuilder peptide A-C` etc. — every builder refusal a traceback |

## E. The pages show the folder and the run they say — `task-setup/`, `results/`, `lib/trajectory/`, `structure-optimization/`, `modify/`, `this-machine/`, `lib/projects/`

| id | class | road | what |
|---|---|---|---|
| (F12 walk) | M | own | **fixed in the working tree**: Task setup's "What gets written" kept the last folder's structure names |
| J2-D1 | W | own | Export bar keeps the first-picked run's path → exports another run than the one on screen |
| SO-D1 | W | own | Structure optimization: a chosen frame of a frame set comes back as frame 1 after a reload; Send writes frame 1 |
| J2-D2 | C | own | trajectory view throws when the 3-D viewer could not start (no WebGL, narrow panel) — nothing drawn |
| MB-D2 | C | own | Molbuilder Load-add failing (server refusal, dropped connection) — silent, unhandled |
| TS-D1 | M | own, every time | a PySCF hand-over opened after a SIESTA folder is offered SIESTA's settings |
| TS-D2..D5 | M | own (timing / a failed fetch) | late folder answer overwrites template values; stale file rows; empty picker said "all columns used"; dropped error |
| J2-D4 | M | own | a SIESTA run with no frames yet: no "Stopped" badge, cover stays 15 s |
| SO-D2, SO-D3, MB-D3, P-D2, P-D3, TM-D1..D3 | M | own/rare | readouts naming the wrong file or state ("load failed: b" while a.xyz shown; "Not saved yet" for a loaded file; channel note dropped; filter count stale) |
| J2-D5..D7, DOC-D1..D3, P-D4..D6, MB-D1, MB-D4 | N | own | status-line wording, late Fit answer, Documents race, tree drops a sibling |

## F. MolView (sealed) — `lib/molview/`

| id | class | road | what |
|---|---|---|---|
| K-D1 | W | own | after a Delete, the saved point keeps the old selection → after Retract/reload the next Delete/Assign acts on atoms nobody picked |
| K-D8 | W | own (timing) | a label or cell change made while a slab/append request is out is silently overwritten |
| K-D10 | W low | own | whole-store JSON edit becomes several undo steps; one Retract leaves a never-shown state |
| K-D2 | W low | own | `labels_modified` lost on reload (no reader today) |
| K-D6, K-D7 | M | own | axis triads take the force arrows' gold; a measurement arrow drawn to a hidden pick under isolate |
| K-D3..D5, D9, D11 | N | own | rare frame race; Cell dot; play icon; shift-click; axis re-shown |

## G. VibrationView and the spectra viewers (sealed) — `lib/vibrationview/`, `lib/spectra/`, `lib/spectrumchart/`

| id | class | road | what |
|---|---|---|---|
| L-D2 | W (+C) | own | animation export mixes modes/amplitudes if the person clicks during it; a poll mid-export kills it with a raw error |
| L-D1 | M | own | spectrum chart's highlight colours land on the wrong sticks once a Raman lane has pending modes |
| L-D5 | M | own | remembered Size/Speed applied but the readouts still show 0.15 Å / 1.0× |
| L-D3, L-D4 | N | own | returning to the browser tab, or a live poll, rebuilds the 3-D box (camera, zoom lost) |

## H. The run script and the machine — `runwrap.py`, `scheduler/`, `monitor.py`, `envs/`, `cli.py serve`

| id | class | road | what |
|---|---|---|---|
| B-D2 | W | own | the run script's USER-CUSTOM zone runs after `rc=0` is written — nothing there reaches the engine |
| B-D3 | W (record only) | own (a Slurm refusing a column) | probe writes `max_submit_jobs: null` ("asked, none") for a question not asked |
| O-D6 | W | own | init-config replaces the person's own preamble with conda's hook, unsaid |
| B-D1, B-D4, C-D3 | M | own | GPU index shown for the device; the shown command lacks `MaxWalltime`; notify-token says failures are silent though logged |
| O-D1..O-D5, O-D7..O-D9 | M | own | `--activation`/`--projects` dropped on re-run; `serve status` probes 127.0.0.1 only; `::1` crash text; install help says 3.11; `<recipe>` printed literally |
| C-D2 | N | own | start card never states the wall |

## I. Chapter sentences (T, 85) — docs-only, no version

Each listed with the sentence to change in its `v-*.md`. Five need the code
changed rather than the chapter: A-U6 (`status --help` "Read-only" but it can
write `run.json`), B-U3 (a misspelled `qos` drops a queue silently), C-U3 (run
number reuse), H-U4 (= H-D1), H-U5 (= H-D5).

## Questions that are yours

1. **F-U1** — a SIESTA vibration's result drops the structure's own `info`
   notes; PySCF's keeps them. Make SIESTA keep them, or say it does not?
2. **C-OBS** — a `report-cards.json` copy that does not read stops the
   monitor of *every* run on that machine, runs without a notify block
   included (no closing record). Keep, or scope it to runs that notify?
