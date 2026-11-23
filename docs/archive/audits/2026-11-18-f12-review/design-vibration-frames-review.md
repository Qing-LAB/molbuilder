# The vibration framework's motion toolkit — the design review (2026-11-23)

**Verdict: the design closed and committed, documents only; built in R3d2b1 and R3d2b2**
([`plans/plan.md`](?doc=plans/plan.md) § 2).

*(The user, 2026-11-23: "this is entirely user's research and i created molbuilder to support this
which does not mean molbuilder does a fully packaged rigid result generator"; "transport calculation
should not overreach to design what the multiframe is. we need to cleanly separate the logic and
design"; "vibration calculation is the foundation that exposes the fundamental data of all modes";
"i want three through review of document/design/api/framework/example after this work by agents".)*

## What was designed

- **Three parts, one direction**: the vibration calculation and its record (the foundation,
  [`engines/vibration.md`](?doc=engines/vibration.md) § 6.11); the motion toolkit
  ([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)) — one mode's states and their
  draws, the molecule's motion and its snapshots (Monte Carlo at equal weights, antithetic pairs the
  person's choice, seeds that nest), a mode's own Gauss frames, a person's own frames through one
  writer; a calculation over frames (transport), which reads frames, weights and the structure's own
  facts alone ([`engines/transport.md`](?doc=engines/transport.md) § 2a.9, § 2a.12).
- **The science** ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md)): a
  driven mode as one coherent state, a warm mode's displaced thermal state; the whole molecule's
  independent modes and drive groups; the swing's own Gauss rule worked step by step; the snapshot
  average, its error and antithetic pairs; what each kind of set answers, comparing states at one
  temperature.
- **The record** (R3d2b1): per-atom motion shares, the largest zero-point move, the reduced mass and
  force constant by Gaussian's convention, the spread and mean quanta at 0–40 K beside the
  thermochemistry grid, in `/modes/thermal`.

## The user's choices (2026-11-23)

Per-atom shares stored; temperatures from 0 K; the reduced mass and force constant by Gaussian's
convention (an imaginary mode's positive, as Gaussian prints it); the swing's frames by its own Gauss
rule, after a step-by-step explanation; a driven mode the pure coherent state by default ("quantum
mechanically, the 'driven mode' should only have one possible state"); a scientific and an API reader
in every science release; three review rounds of the design before code.

## The review

| round | readers | findings |
|---|---|---|
| a science and an API review of 2.78.0 and the record's first design | science, api | 19 |
| 1 | science, api, documents, boundary | 56 |
| 2 (new readers) | science, api, documents, boundary | 55 |
| 3 (new readers) | science, api, documents, boundary | 49 + 1 note |
| the last fixes | science and api; documents and boundary | 24 |

Every finding fixed or recorded with its reason; every worked number recomputed independently by the
science readers (numpy/scipy, CODATA, never molbuilder's code); every new citation checked against
Crossref or its author's page. The decisions of each round were applied to the documents and read by
the next round's new readers. What the round-3 readers changed most: driven modes independent by
default and drive groups the person's, with stated lags (a mode's eigenvector sign is a convention);
random streams that nest, so more frames later continue the first; the transport data file carrying no
copy of a set's rows; each set stating its own assumptions as a row; one indexing rule, item i is
frame i.
