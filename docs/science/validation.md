# Scientific validation — the runtime machinery

**Role:** contract
**Domain:** science
**Companions:** `overview.md` (the **what** — the correctness principles + the
advisory-while-editing / enforcing-at-generation rule); `chemistry-correctness.md`
(the chemistry facts the analyzer encodes); [`model/chemistry.md`](?doc=model/chemistry.md)
(the L1 chemistry primitives this composes); the engine emitters (the consumers).

This is **how** molbuilder realises scientific correctness at runtime: the
structure's chemistry **facts** (`analyze_structure`), the one **electronic
state** every calculation carries (`electronic_state`,
[`chemistry-correctness.md`](?doc=science/chemistry-correctness.md) § 2a), and the
surfaces that read them — the checks, the deck writers, the chemistry card. Every
boundary passes a **frozen dataclass** (an immutable typed record — never an
untyped dict); JSON appears only at the HTTP wire (the network boundary), via the
record's own `as_dict()` or `dataclasses.asdict()`.

The core idea is **open-shell vs closed-shell**: *closed-shell* = every electron
paired (non-magnetic, most organics); *open-shell* = some electrons unpaired
(magnetic, most transition metals). Picking the wrong one gives a physically
wrong answer that often still "converges" silently — which is exactly what this
machinery prevents. (Cross-cutting terms — *SCF*, *DFT*, *2S*, *parity*, *KB
projector* — are in the [`overview.md` glossary](?doc=science/overview.md);
narrower ones are glossed inline below.)

---

## 1. Layered overview

```mermaid
flowchart TB
    subgraph L1["L1 — chemistry primitives (engine-agnostic, pure)"]
        chem["chemistry.py — resolve_element · total_electrons ·<br/>check_spin_charge_parity · explain_metal_spin · formal_charge_from_phosphates"]
    end
    subgraph L2["L2 — the facts (engine-agnostic)"]
        an["analyze_structure(struct) → ChemistryAnalysis"]
    end
    subgraph L3["L3 — the electronic state (one class, every engine)"]
        es["electronic_state(struct, cfg, kind=) → ElectronicState"]
    end
    subgraph L4["L4 — surfaces (consumers)"]
        val["validation/ — check_electronic_state (the one family)"]
        deck["the deck writers — SIESTA, PySCF, the transport rungs"]
        api["/api/structure/analyze → the chemistry card + each form's chip"]
    end
    chem --> an --> es
    es --> val
    es --> deck
    es --> api
```

Three typed boundaries:

| Boundary | Owner | Input → output |
|---|---|---|
| `analyze_structure(struct)` | `chemistry.py` | `Structure` → `ChemistryAnalysis` — facts, no decision |
| `electronic_state(struct, cfg, *, kind)` | `electronic_state.py` | `Structure` + a config's four state items → `ElectronicState` |
| `check_electronic_state(struct, cfg, *, calculation, engine, state)` | `validation/chemistry.py` | the state `validate()` worked out → `List[Issue]`, one finding per fact |

### 1.1 Three kinds of question, one door *(framework, 2026-09-03)*

The layers above are **one** of the three things validation does, and the three
differ in what they take in and — the part that decides where a finding
belongs — **what their verdict is about**:

```mermaid
flowchart TB
    subgraph Q["the three questions"]
      direction LR
      A["<b>ANALYSIS</b> — what IS this?<br/><i>in:</i> the structure<br/><i>verdict about:</i> facts<br/>L1-L4 above"]
      I["<b>INTEGRITY</b> — is this artifact sound?<br/><i>in:</i> ONE artifact<br/><i>verdict about:</i> that artifact<br/>a .psml missing · dead channel · wrong XC"]
      F["<b>FITNESS</b> — do the pieces fit?<br/><i>in:</i> the artifacts <b>+ the config</b><br/><i>verdict about:</i> <b>the configuration</b><br/>mesh_cutoff vs what these pseudos ask"]
    end
    D["<b>the ONE door</b><br/>validate(struct, cfg) → List[Issue]<br/>report(issues) — raises on ERROR"]
    A --> D
    I --> D
    F --> D
    D --> S1["<b>script generation</b><br/>render_deck"]
    D --> S2["<b>the web preflight</b><br/>the issues panel"]
    D --> S3["<b>jobset prep</b><br/>before a folder is written"]
```

| | analysis | integrity | fitness |
|---|---|---|---|
| **input** | the structure | ONE artifact | every artifact **+ the config** |
| **verdict about** | facts — *what this molecule is* | that artifact | **the configuration** |
| **keyed to** | nothing to fix | the element / file | the **config field** a person edits |
| **example** | *this has an open-shell metal* | *`S.psml` is absent* | *`mesh_cutoff` is below what these pseudos state* |

**Why the distinction earns its keep: it decides where a finding goes, and
being wrong about that hides it.** A fitness verdict keyed to an element reads
as *"something is wrong with sulfur"* when what is wrong is a number in the
form — so the person goes and re-downloads a perfectly good file. Key it to
`config.mesh_cutoff` and it lands on the field they change.

**An artifact may DECLARE a requirement, and then the configuration must
satisfy it — the strictest one in the set wins.** That is the fitness rule in
one sentence, and it is general: a PseudoDojo v0.5 file states its own
recommended cutoff, and any future declared fact (a required relativity, a
minimum basis) enters through the same reader and the same comparison. Where a
file states nothing, a literature default answers instead — **a declared number
outranks a guess**, the same rule the rank count follows
(`running-a-job.md` § 3.1: *read from a record, never guessed*).

**All three reach every surface through the one door, and that is the test of
placement.** `validate(...)` is called by script generation, the web
preflight and `jobset prep`, and `report()` gates generation and prep (the
preflight shows the findings while the form is open); a finding that needed new plumbing to reach a
surface would be a finding put in the wrong layer.

---

## 2. The facts (L2)

`chemistry.analyze_structure(struct) → ChemistryAnalysis` (`chemistry.py`) says
what a structure IS, chemically — the metals that bear on its spin, and their
usual spins — and decides nothing:

```python
@dataclass(frozen=True)
class ChemistryAnalysis:           # chemistry.py
    n_atoms:        int
    elements:       List[str]          # unique, sorted
    open_d_metals:  List[str]          # the open-d transition metals present
    noble_metals:   List[str]          # Cu, Ag, Au present
    metals:         List[str]          # both, open-d first; [] for organics
    metal_hints:    List[MetalHint]    # each metal's usual spins, low → high
```

**Design rules.** Pure function (no I/O, no engine imports); engine-agnostic
vocabulary; **facts, not a decision** — the charge and the spin are decided by the
electronic state (§ 3), at the calculation's own charge and periodicity. Every
label must name an element: `resolve_element` raises `KeyError` on one that does
not, because every reader goes on to count electrons, and a count with an atom
left out is a wrong one. The metal lists are split once here, so no reader
filters `metals` again.

> *(The analyzer suggests nothing: judged at charge 0 on the neutral structure
> whatever its cell, a suggestion tells a formate ion at −1 and a bulk gold lead
> to go open-shell. Deciding is the electronic state's, § 3.)*

### 2.1 The noble-metal distinction — three categories, not one flat set

The metals are split into three physically-grounded sets (`chemistry.py`), so a
gold junction is never read as open-shell:

| Set | Elements | Physics | What a blank spin becomes |
|---|---|---|---|
| `OPEN_D_TRANSITION_METALS` | Sc–Ni (3d), Y–Rh (4d, not Pd), Hf–Ir (5d, not Pt/Au), lanthanides, common actinides | incomplete d-shell; Stoner criterion / itinerant moments | **unrestricted** — the metal's usual count in a finite system, a floating moment in a repeating cell; warned until stated (ES8) |
| `NOBLE_METALS_S1` | Cu, Ag, Au | atomic nd¹⁰(n+1)s¹, but in any extended metallic context (cluster ≥ 4, surface, junction, bulk) the s-band delocalises | **restricted** for a cluster of four or more with an even count, and in any repeating cell; a single atom keeps its doublet |
| `CLOSED_D10_METALS` | Zn, Cd, Hg, **Pd** (4d¹⁰5s⁰), **Pt** (5d⁹6s¹ atom; metallic Pt closed-shell in surface DFT) | filled/effectively-filled d | no row of its own — the electron count's parity decides |

*(Plain-language keys: a **d-shell** is the set of d orbitals; **d¹⁰** = full (10
electrons, non-magnetic), an **incomplete** d-shell is the magnetic case. The
**Stoner criterion** is the textbook condition for a metal's delocalised
electrons to turn magnetic — it fails for Cu/Ag/Au, so they stay non-magnetic.
**NEGF** / **TranSIESTA**, in the references below, is the electron-transport
method these gold-junction papers used.)*

**The decision is the detection table** —
[`chemistry-correctness.md`](?doc=science/chemistry-correctness.md) § 2a.1b,
`electronic_state.recommend`. The first row that matches wins, so an open-d metal
decides even beside gold; a repeating cell never reads its count per cell as a
spin (ES3). The 4-atom cutoff (`NOBLE_CLUSTER_THRESHOLD = 4`,
`electronic_state.py`) is the conservative choice — overwhelmingly what published
Au transport / surface DFT does. When the noble closed-shell answer is wrong
(sub-4-atom Au cluster, single adatom on an insulator, a magnetic 3d co-adsorbate,
explicit Kondo / spin-orbit physics), state the spin.

**Worked example — both directions, end to end.** An **Au-BDT-Au** junction
(gold / benzene-1,4-dithiol / gold) and an Fe centre:

```python
>>> from molbuilder.electronic_state import electronic_state
>>> from molbuilder.validation.chemistry import check_electronic_state

# --- CLOSED: the Au junction, a repeating cell ---
>>> st = electronic_state(au_bdt_au, SiestaConfig(), kind="transport")
>>> st.spin_treatment.value, st.unpaired_electrons.value
('restricted', 0)
>>> st.spin_treatment.said
'detected: metallic Au in a repeating cell: the s-band delocalises and no moment forms'

# --- OPEN: an Fe centre (open-d 3d metal), the spin fields blank ---
>>> st = electronic_state(fe_porphyrin, PySCFConfig(), kind="optimization")
>>> st.spin_treatment.value, st.unpaired_electrons.value
('unrestricted', 2)          # Fe's usual count — a guess about coordination (ES8)

# REVERSE — a closed shell stated on the open-shell Fe system:
>>> cfg = PySCFConfig(spin_treatment="restricted")
>>> check_electronic_state(fe_porphyrin, cfg, calculation="optimization", engine="pyscf",
...                        state=electronic_state(fe_porphyrin, cfg, kind="optimization"))
[Issue(severity='warn', message='The spin treatment is restricted (stated), and the
       structure implies unrestricted, 2S = 2: … converges to a fictitious state …',
       where='config.spin_treatment')]
```

**References** (the noble-metal-is-closed-shell basis): Taylor, Brandbyge,
Stokbro, *PRB* **63**, 245407 (2001) — the original TranSIESTA Au-BDT-Au paper;
Ke, Baranger, Yang, *JCP* **122**, 074704 (2005) — Au-BDT-Au NEGF; Verzijl &
Thijssen, *JPCC* **116**, 24811 (2012) — DFT+Σ Au-alkanedithiol benchmark;
Marder, *Condensed Matter Physics* Ch. 17 — the Stoner-criterion derivation
(Cu/Ag/Au explicitly non-magnetic in bulk). The table's rows are pinned through
prep in `tests/test_electronic_state.py` (Au₄, Au₁, the gold lead, Fe, bcc Fe on
SIESTA; Au₂, Au₃, Au₅, Pd₂ and more on PySCF); `tests/test_chemistry_analyzer.py`
asserts the three sets are pairwise disjoint and Pd/Pt are excluded from
`OPEN_D_TRANSITION_METALS`.

---

## 3. The electronic state (L3)

One class decides the charge and the spin of a calculation — for the form, the
checks and every deck: `electronic_state(struct, cfg, *, kind)`. Its contract —
the four items, the one order a blank is answered in (stated → implied → recorded
→ detected), the detection table, what each engine can run, ES1–ES10 — is
[`chemistry-correctness.md`](?doc=science/chemistry-correctness.md) § 2a; this is
where it sits in the machinery. Each value is a `Resolved(value, source, why)`
with its phrasing `said`, so the deck comment, the prep report and the card say
the same words.

*(No per-engine adapter suggests values for a form to copy: a copy overwrites
the person's own values, and a suggestion judged at charge 0 on the neutral
structure is wrong for an ion or a repeating cell.)*

---

## 4. The consumers (L4)

**The settings gate** — `check_electronic_state` (`validation/chemistry.py`),
asked **once** by `validate()` for every engine and every kind: what the engine
cannot run (ES4–ES6), parity at the resolved charge for a finite system (ES3), a
stated closed shell on an open-shell structure and a constrained singlet (ES9), a
count a metal decided until it is stated (ES8), a charge on a transport
calculation (ES7). At most one finding per fact. The engine validators READ the
state — SIESTA's charged-cell notice and vacuum threshold, its spin-orbit
pseudopotential check — but raise none of its findings.
[`overview.md`](?doc=science/overview.md) § 4 lists them with their severities.

**The deck writers** — SIESTA optimization and vibration, the five transport
rungs, PySCF optimization and vibration — each spells the state its own way
(`Spin` / `Spin.Fix` + `Spin.Total` / `NetCharge`; PySCF's composed class and
`gto.M(charge=, spin=)`) and writes each value beside its source. A transport
ladder's rungs are handed the junction's state, decided once at prep.

**The forms** — `/api/structure/analyze` (`build.py`) takes the structure the page
would hand over (the envelope its viewer holds — the one the preflight and the
hand-over send) and each form's four items, and answers per engine
`ElectronicState.as_dict()` beside the facts. `lib/chemistry.js` shows it on the
chemistry card, each value with its source; `lib/detection-chip.js` in one line
on each form's chip. It is asked on every load and restore and on every edit to
one of the four items, and it **fills nothing in** — a blank is already the
instruction "work it out", and the card is its answer. With no structure or no
answer, the card is hidden.

**The invariant** ([`overview.md`](?doc=science/overview.md) § 5): the card, the chip, the checks
and the deck read one class, so they cannot disagree — the remedy for two-surface
drift is to delete the parallel path, not patch it.

---

## 4.1 The delivery contract — facts in, findings out (decided 2026-07-29)

A check that never runs, or runs on the wrong structure, or produces a finding
nobody sees, is worse than no check: it reads as a clean bill of health. The
clauses below close the three ways that happens — a check skipped because a
caller omitted an argument (F4), a finding sent to the server's stderr as a
Python warning (R5), and a request mixing live labels with coordinates read
from a page-local copy (F1).

**The layer between a tab and validation owns both directions.** That layer is
MolView's concealed data model (`molview.data`): the facts leave from it, the
findings come back to it for routing. A tab wires the two and contributes
nothing of its own. Everything below follows from that.

### Facts (inbound)

| # | Clause | Enforced by |
|---|---|---|
| **F1** | **One fact holder, read once.** Coordinates, elements, labels (regions / frozen), periodicity and annotations are read from `molview.data` **at request time** — never from a page-local mirror, a second fetch, or a re-read of disk. **One call assembles the whole body**, so a tab can neither send a partial set nor send the same facts twice from two reads: `exportFile(range)` returns them together, in the server's words, for the frame on screen ([`web/molview.md`](?doc=web/molview.md) § 9.3a). A body that carries `frozen_atoms` / `regions` / `periodicity` *beside* the structure has read them again, at another moment; the server reads the envelope alone (`struct_from_body`), so they reach nothing. | `molview.data.exportFile()`; `tests/test_build_e2e.py`'s Sends — the frame on screen after the frame bar moved, and the held atoms of the file loaded after a click elsewhere |
| **F2** | **No server-side second source.** The server builds its `Structure` from the request body alone — one seam applies the in-body labels and periodicity and runs the frame-contract gate. **The sidecar is never read for an emitted structure**, so a validated structure is never a body/disk mixture, and a body with no label keys declares *no labels*. No request names the structure's file: a hand-over's folder rides as `dest`, which anchors the pseudopotential check ([`web/handover-procedure.md`](?doc=web/handover-procedure.md) § 2.2) and is never a label source. | `struct_from_body` — the labels ride inside the envelope; `test_validation_delivery_contract.py` proves a sidecar on disk cannot reach an emitted deck |
| **F3** | **The model is complete by construction.** The model always carries periodicity — defaults when the pair has none, full values otherwise — so a tab is never in a position where it must invent a fact. | [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 7; `TestTabEmitContract` |
| **F4** | **Derived facts are derived from those facts, server-side.** The cell a check needs is `struct.resolve_cell()`, resolved inside `validate()` — never an argument a caller can forget. It is derived only when the structure actually **declares a box** (an explicit `cell`, a vacuum the structure states — an all-zero one included — or a non-isolated axis): a gas-phase molecule never asked for a lattice, and a *planar* one's bounding box has zero thickness, so checking it would report a degenerate cell for a calculation that has no cell. A check that cannot run says so as `info`; silence is never the answer. | `_structure_declares_a_box`; `validate()`'s cell default; contract tests |

### Findings (outbound)

| # | Clause | Enforced by |
|---|---|---|
| **R1** | **One result type, web-shaped by construction.** Every finding is an `Issue` → `{severity, message, where, card?, stage?}` from the one serializer (`_shared.issues_to_json`). `where` is the **stable machine-readable identifier** (`geometry.min_distance`, `cell.image_distance`, `config.mesh_cutoff`) — the UI binds behaviour to it, carrying it on the row as `data-where` and never showing it, and never parses `message`, which is prose for humans and may be reworded freely. | `issues_to_json`; contract tests |
| **R1a** | **A stage label rides beside `where`, never inside it** ([`engines/stages.md`](?doc=engines/stages.md) § 4 R2). A stage is validated as a *resolved whole*, so the same check fires for a stage as for a single run and produces the **same** id — folding the stage into the id would give one check as many ids as the ladder has stages, and R1 says the UI binds to the id. `stage` is **absent** for a single run *and* for a finding about the **sequence** (`engines/stages.md` § 4 R3 — a ladder that loosens is a fact about the description, not about a member of it). | `issues_to_json` omits the key when unset; `test_validation_across_stages.py` |
| **R2** | **One channel into the UI.** The layer that holds the facts also takes the result: a single client module (`lib/validation-findings.js`) receives `issues[]` and routes them — beside the control when the page hands it the ids its form rendered (`fieldIds` — each form's map from the renderer, `formSchema.fieldIds`, which the page assembles — keyed by stage first when one form holds several stages' controls, the finding's `stage` picking its map; a `where` outside `config.` such as `task.bias` looked up whole), on the card its field is drawn in where the finding names a config field, residual structure panel otherwise — and every page mounts it. No page implements its own renderer. | `validation-findings.js`; each page that shows findings draws every one its check returned, a row each and nothing besides — the optimization tab (`test_build_e2e.py`), the Spectrum tab (`test_spectrum_form_e2e.py`), the Transport tab (`test_transport_on_a_real_junction_e2e.py`, what a citation measured); one renderer held by review ([`code-audit.md`](?doc=process/code-audit.md) § 1c) |
| **R2a** | **A NOTICE IS A FINDING.** `{severity, message, where, about}` beside `ok` ([`web/web-api.md`](?doc=web/web-api.md)) is the same row the same module draws — a different word for the same channel, never a reason for a renderer of its own. An outcome line that carries its notices — a save's, a load's, an upload's — takes them from the module too: `outcome(sentence, notices, tone)` gives the line's words and the tone of the most severe notice (R4), the one composer those lines ask. | held by review: a severity reaching markup outside the module is a finding ([`code-audit.md`](?doc=process/code-audit.md) § 1c) |
| **R2b** | **Reuse is by IMPORT, downward, and it is not a breach of a module's boundary.** A module with its own design system may compose around the shared row — MolView sets its own type scale on `.issues-panel .issue-item` — but may not re-implement it: [`ui-contract.md § 5`](?doc=web/ui-contract.md) already rules that a surface owns *where* a message sits and never what a severity looks like. What a self-contained module may NOT do is take the renderer off a global: [`web/molview.md`](?doc=web/molview.md) § 4 is *"nothing it needs comes from a global"*, and mounting publishes nothing. So `lib/validation-findings.js` has **two delivery forms** — `export`s for the modules that import it, and `lib/validation-findings-global.js`, a separate entry point that publishes `molbuilder.validationFindings` for the classic scripts that cannot import. One implementation; the registration is not part of it. | `tests/test_molview_mount.py::test_mounting_needs_only_what_it_is_handed` |
| **R3** | **Nothing is dropped.** Rendered count equals received count. An unknown or missing `card` falls to the residual panel; it is never skipped, and the list is never truncated. | contract tests |
| **R4** | **Severity means the same everywhere.** `error` blocks generation and says why; `warn` renders without blocking; `info` is advisory. No surface downgrades a severity to keep a screen quiet, and the CLI prints the same three. **One exception, named:** a door that loads, builds or modifies a structure answers an `error` as a `warn` notice beside the structure (`periodicity_gate.notices_for_report`) — a load that refused would leave the structure unopenable, and so unfixable ([`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 8.2), and findings while editing never block ([`overview.md`](?doc=science/overview.md) § 3, the user's ruling); the same finding is an `error` at generation, where it refuses. | contract tests |
| **R5** | **One channel means one channel.** A finding never travels as a Python `warnings.warn` — it cannot reach a web user. Code that wants to warn returns an `Issue` from a validator. **A builder's findings are one check after the build** *(decided with the user, 2026-10-31, (b))*: a builder returns the structure, and `validation.builders.check_built`, run by the one build door every surface builds through (`builders.build`), says what it found — the structure's own, read off what was built (a peptide's C-terminus with no `OXT`, an aldehyde; a built nucleic acid's steric clash, two atoms of different residues nearly on each other — on every DNA or RNA build, said and never repaired unasked, *decided with the user, 2026-10-31, (b)*; a SMILES's or a name's molecules overlapping, two atoms of different molecules nearer than half the sum of their van der Waals radii, `engines/builders.md` § 6), and what the backend that built it does not build of what was asked (a helical form, a terminal state), in each backend's own answer (`builders.backends.built`, beside the table its build reads). The build route answers them as notices beside the structure, every builder panel drawing them through R2's module, and the command line prints them as `report` prints every finding. | `tests/test_build_findings.py` — each finding through the route, none raised as a Python warning |
| **R6** | **Visible before the irreversible step.** Findings accompany the artifact at render time *and* the preflight — before engine input is written or a job is submitted, never after. | endpoint tests |
| **R7** | **A message cites no document** *(the user, 2026-11-05: "Remove from both")*. A sentence a person reads, on a page or in the terminal, says what is wrong and the way on in its own words, and names no chapter or section. References stay in the documents and in code comments. Text written into a deck, a run script or a data file's definitions is that file's own; a link on a page is a link. | review, over each release's messages |

**Which severity a spatial check gets** (decided 2026-07-29). Two different
questions get two different answers, and conflating them is how a tool becomes
either nagging or dangerous:

* **Is there *enough* space?** — `cell.vacuum_thin`, `cell.image_distance`,
  `cell.volume`. These are **warnings, never blocking.** The cell is well-formed;
  what is in question is the *physics quality* of the result, and that is the
  user's call to make. They may be probing convergence, reproducing a published
  tight-box run, or deliberately accepting image interaction. molbuilder states
  the number and the recommendation and gets out of the way — it never resizes
  the box and never refuses the run.
* **Will this engine even USE the cell?** — `cell.periodic_in_gas_phase`. A
  **warning**, by the same rule: the PySCF renderer builds
  a molecular `gto.M()` with no lattice and no k-points, so a structure with a
  repeating axis produces an **isolated cluster** and the cell is dropped. That
  is not a rough version of what was asked for — it is a different calculation,
  and it is said. An isolated-cluster run of a periodic input
  is legal and occasionally deliberate, so the user is told, not stopped: the
  finding names the repeating axes, the lattice being ignored, and what comes
  out instead.

  It keys on `axis_kind`, not `pbc`. `axis_kind` is authoritative and never
  `None`; `pbc` is its derived view and collapses `transport` into the same
  `True` as `periodic` — both are wrong for a gas-phase script, but a check
  written on `pbc` alone could not tell a lead from a crystal axis, which
  `config.kgrid` depends on.

* **Does the k-point mesh contradict the axes?** — `kmesh.check`, on every
  mesh a deck writes (`config.kgrid`, a transmission's `config.tbt_k_grid`),
  **warnings**, save one refusal — a transport calculation's grid sampling an
  isolated axis more than once while the other axis of the electrode plane is
  periodic, which TranSIESTA stops on *(user, 2026-09-30)* — under one rule
  *(user, 2026-08-20)*: **`k > 1` is the user's
  explicit statement — "sample a supercell along this axis" — so that is the
  only place a consistency question exists. `k = 1` states nothing** (correct
  for an isolated axis, a legitimate Γ-only choice for a periodic one) **and is
  validated not at all**, save on a transport calculation, where one k-point
  along a direction the electrodes repeat in is warned
  ([`engines/transport.md`](?doc=engines/transport.md) § 0.3b; *user,
  2026-10-09*). The mesh and its roles are
  [`engines/siesta.md`](?doc=engines/siesta.md) § 6.1's.

  Where `k > 1`, two of the user's own statements can contradict it:

  | the sampled axis says | finding |
  |---|---|
  | `isolated` | **warn** — sampling a direction declared not to repeat: wasted cost |
  | `periodic` — or `transport` in a calculation that is not a transport one, where the deck is periodic along it *(user, 2026-09-30)* — but the **geometric gap** (cell extent − atom span) on that axis is ≥ 5 Å | **hint** — the gap is the real vacuum whether or not the `vacuum` field was set; images that far apart are usually meant to interact weakly or not at all, so the finding names the gap and says *"if deliberate, carry on"* — a minor-image-interaction setup is a legitimate choice only the user can judge. *(The arithmetic is exact for orthogonal cells with unwrapped coordinates — the common case here; a skewed cell's axis norm overstates the perpendicular image distance, and wrapped coordinates can suppress the hint. Both err on the quiet side of a hint-only check.)* |
  | `periodic`, tightly packed | silent — everything checks out |

  A transport calculation's own transport axis is not in this table: it is 1
  on the open rungs and a lead's own count on a lead, fixed by the mesh and
  refused otherwise on every door. One fact, one severity: the mesh fixes the
  transport axis, and every door refuses another value.

  *(Nothing judges an axis the user stated nothing about. A junction whose
  axis kinds were lost between tabs and kept its `2 2 1` grid is still
  caught: each wrongly-isolated axis sampled twice earns a warning.)*

* **Can this cell exist at all?** — `cell.no_volume`, `cell.left_handed` and
  `cell.unfittable`, all **errors**, from the one checker (`molbuilder/cell.py`), asked with
  the geometry's own checks by `validation.structure_findings` — the
  structural half of `validate()`, which `molbuilder validate` asks with no
  engine named, every finding of the box heard — a mirrored cell is an error
  with or without `--engine`. Upstream of
  them the gate refuses the edit outright (the frame contract,
  [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md)
  § 6.1). Not a judgement about
  quality: a zero-volume lattice makes SIESTA fail when it builds reciprocal
  vectors, so emitting it with a warning would hand the user a
  guaranteed-failed run dressed as a choice.

  They are separate ids because they are separate faults with separate
  repairs: a flat box needs volume, a mirrored one two vectors swapped.
  [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md)
  § 6.1a lists every id and which surface each reaches.

So: **adequacy is advisory, representability is blocking.** A check that reports
"your box is small" must not stop the run; a check that reports "this box is not
a box" must.

```mermaid
flowchart LR
    M["molview.data<br/>(the fact holder)"] -->|"F1 exportFile() — one read"| REQ["request body"]
    REQ -->|"F2 body only, no disk"| S["Structure"]
    S -->|"F4 cell = resolve_cell()"| V["validate(struct, cfg)<br/>the ONE gate"]
    V -->|"List[Issue]"| J["R1 issues_to_json"]
    J -->|"issues[]"| P["R2 validation-findings.js"]
    P -->|"named a config field"| C["the card its field is drawn in"]
    P -->|"otherwise (R3)"| RES["residual structure panel"]
```

**Worked example — a thin vacuum.** `hemeC-dithiol` with 2.5 Å of vacuum per
side surfaces `cell.image_distance` (`warn`, "min atom-to-nearest-image
distance is 5.15 Å …") in the structure panel *before* the deck is written,
with the actionable number in the message — the overlap SIESTA otherwise
reports only as *"Gamma-point calculation with multiply-connected orbital
pairs"*.

---

## 5. Labels are the user's

molbuilder reads the labels it **owns** — the reserved frozen label (SIESTA's
`Geometry.Constraints`, PySCF's geomeTRIC `$freeze`, the vibration's Hessian
mask) and a transport calculation's partition (`L-electrode`, `R-electrode`,
`bridge`, `buffer`, [`engines/transport.md`](?doc=engines/transport.md) § 4)
— and no other. Any other label in `struct.regions` is the user's: it rides
along untouched, and no gate refuses it, warns about it or gives it a meaning
by its name *(user, 2026-10-02)*.

---

## 6. Adding a new engine

1. Declare what it can run: its record in the engine table — with
   `computes_in = "free space"` if it builds the atoms as one molecule in
   free space, and then every question about the calculation's axes, the
   box's advice among them, is answered for a cluster
   ([`model/structure-periodicity.md`](?doc=model/structure-periodicity.md)
   § 2.1) — and a part on each kind it runs in the kind table
   ([`execution/architecture.md`](?doc=execution/architecture.md) § 3.3); and
   on the state's two items the choices each of its kinds offers (`offered`
   with the engine's key, [`engines/template.md`](?doc=engines/template.md)
   § 6.3a — no `free` among its counts if it cannot float a moment), with a
   `not_offered` row saying why for what it cannot. The form then offers exactly
   that, and `resolve` and the gate refuse the rest by name (ES4).
2. Give its config the electronic state's three items — `net_charge`,
   `spin_treatment`, `unpaired_electrons`, each blank by default and declared
   once in the catalogue ([`engines/template.md`](?doc=engines/template.md)
   § 6.3) — `method` if it has one, and the three rules
   the gate asks of every engine's config: why, under a spin treatment and
   kind, it cannot hold a stated number of unpaired electrons
   (`count_must_float`), what it does with an odd electron count under
   restricted spin (`odd_count_restricted`), and which setting of its own
   the engine stops on with the spin state (`spin_hint` — said as
   information, never refused; PySCF's answers `None`).
3. Spell the state in its deck writer — read `electronic_state(...)`, never the
   raw fields — and write each value beside its source.
4. Name its validator in its engine table entry (`validator`,
   [`execution/architecture.md`](?doc=execution/architecture.md) § 3.3) — a
   table without one is refused as it loads; the state's findings come to it
   through `validate()` without a line of its own, and the state itself is
   handed to it. Pin its decks through prep.

No endpoint change: `/api/structure/analyze` answers every engine the kind
has a part on in the kind table.

---

## 7. Where the validators live

`molbuilder/validation/` is a package split **by concern** so any caller imports
directly from the relevant submodule:

```
validation/
├── __init__.py     # public API: validate, report; an engine's checks are
│                   # the part its engine table entry names (`validator`), a
│                   # kind's science the part its kind table entry names; a
│                   # junction's checks where the kind's deck relaxes
├── geometry.py     # validate_geometry + geometry checks
├── metadata.py     # the catalogue's ranges and the fields' validate= callables;
│                   # what a kind fixes, does not offer, or leaves inactive
├── chemistry.py    # species labels; THE ELECTRONIC STATE'S ONE FAMILY
│                   # (check_electronic_state); peptide protonation
├── builders.py     # what a build says of what it built (`check_built`, R5)
├── sidecar.py      # frozen-atoms-consumed check; the junction's boundary
│                   # (engines/transport.md § 6.1c);
│                   # the relaxation-record check (engines/vibration.md § 2.2)
├── stages.py       # stage-ladder checks (shared by describe/dispatch)
├── task.py         # the TASK preflight — a description that is not one refuses
├── siesta.py       # SIESTA's validator (validate_siesta) + pseudo/mesh/Makov-Payne/
│                   # vacuum checks
├── pyscf.py        # PySCF's validator (validate_pyscf) + metal-basis adequacy
│                   # and the ECP hint
├── spectra.py      # the vibration kind's science (grid/amplitude/frozen atoms/
│                   # the relaxation record): validate_vibration_kind, its validator
└── transport.py    # the transport kind's science (the pole energy and the
                    # temperature, the window against the bias, I12):
                    # validate_transport_kind, its validator
```

Two rules make this safe to extend: **the call order inside `validate_siesta` /
`validate_pyscf` is the per-engine public contract** — load-bearing, since
tests count issues by position — and a helper **loses its `_` prefix when it
gains a cross-module caller** (e.g. `check_electronic_state` is public; the
others stay private until a PR forces the promotion, no back-compat shim). An
engine's checks are the part its entry in the engine table names
(`validator`), a kind's own science the part its entry in the kind table names
(`validator`) — `validate(struct, cfg, calculation=…)` is the one gate, and
the caller always states the kind. **The gate works out once what every check
reads**: the config's engine (`engine_table.engine_of_config`), the box, and
the electronic state the deck will be written from (`None` where a species
label names no element — the label check owns that finding); each is handed
to the engine's and the kind's checks, which compose none of their own. The
engine and shared validators' own tests are under `tests/validation/`
(geometry, chemistry, SIESTA, PySCF, the transport kind, the aggregator).

**One fact, one finding.** The kind's validator owns the families the kind's
science answers — on a vibration: the grid, the held atoms and what survives the
freeze — and the engine
validator, which
receives `calculation`, **defers** those families on that kind rather than
firing its own copy. Two findings for one fact is the failure this rule
closes: the engine's copy is reasoned from the wrong calculation (a
*"held fixed during relaxation"* line on a force-constant run that relaxes
nothing). Both engine validators
branch on the kind the same way (`validation/pyscf.py`, `validation/siesta.py`).
The charge and spin are neither's: they are the electronic state's one family,
asked once by `validate()` for every engine and kind (§ 4). **Nor is a
junction's**: that a lead comes through a relaxation unmoved, and the room at
its transport boundary, are asked once by `validate()` where the kind's deck
relaxes the structure (the kind table's `relaxes` — an optimization's), never
by an engine's validator. A vibration's held atoms are its own question, and
the free atoms are the system it calculates; a transport rung relaxes nothing,
and its kind refuses the boundary itself.
The same wrong calculation reached the transport kind: its rungs write no MD
block — the junction was relaxed upstream and every rung computes at that
geometry — so SIESTA's frozen-atom family is not asked there at all, rather
than deferred to anyone.

> *A kind's science is its kind-table record's `validator` and an engine's
> checks its engine-table record's, never a registry keyed on a config class:
> such a registry is only as live as the callers that construct the class, and
> every transport rung resolves a `SiestaConfig`.*

---

## 8. What the analyzer does NOT cover

Its scope is the chemistry that decides the **electronic state** — the charge, the
spin treatment and the count — **+ the open-shell-metal hints**: the place silent
chemistry errors hide. Out of scope
(and why): basis set + XC functional (user preference / budget), k-points / mesh
cutoff (geometry, not chemistry), pseudopotential family (covered by the
`pseudopotentials.md` validator pass), convergence thresholds and optimisation /
spectral-workflow choices. Growing it to cover everything would re-fragment the
cross-engine consistency claim.

---

## 9. Test invariants

- **One state, every deck** — each deck kind (SIESTA optimization, vibration, the
  transport rungs; PySCF optimization and vibration) carries the class's answer,
  each value beside its source, pinned through `jobset prep`:
  `tests/test_electronic_state.py` (the detection table on both engines, the
  charge step on every deck, the refusals),
  `tests/test_an_edit_on_a_recorded_structure_e2e.py` (the recorded step, and
  an edit voiding it) and
  `tests/test_transport_compose.py` (a blank spin decided once, on the junction).
- **The card and the chip** — they answer for exactly what each form says, about
  the structure the page holds, fill nothing in, and hide with nothing to answer:
  `tests/test_chemistry_card_e2e.py` (the three tabs, in a browser) and
  `tests/test_chemistry_module_js.py` (the supersede protocol, what the card
  says).
- **The route's refusals** — no structure, an unreadable one, a label naming no
  element, a form for an engine that does not run the kind:
  `tests/test_structure_analyze_endpoint.py`.
- **Every field reaches the deck** — a state item changes what the ENGINE reads,
  not only the comment beside it: `tests/test_every_form_field_reaches_the_deck.py`.

---

> **Note — the GPU eigensolver is not here.** The ELPA-CUDA / NVIDIA-MPS
> eigensolver machinery (the `molbuilder-siesta-gpu` env, the numerical-
> equivalence claim, the MPS rank policy, the `envs validate` probes) is a
> SIESTA-GPU engine/ops concern, documented in
> [`engines/siesta.md`](?doc=engines/siesta.md) § 7.1 and
> [`ops/installation.md`](?doc=ops/installation.md): it is not chemistry
> validation.
