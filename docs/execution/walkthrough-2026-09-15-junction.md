# A junction, walked end to end — 2026-09-15

**Role:** review record
**Domain:** execution
**Companions:** [`execution/worked-example.md`](?doc=execution/worked-example.md)
— the same idea for a plain relaxation, and the model for this document's gap
list; [`engines/transport.md`](?doc=engines/transport.md) § 3.4 — the workflow
this walks; [`science/junction-cell.md`](?doc=science/junction-cell.md) — the
crystallography § 1's findings are measured against.

*User instruction: "build a test case … build a simple molecule junction with
Au slab, and simulate the whole calculation by allowing high tolerance in SCF
calculation so that the calculation can conclude quickly … starting from
molbuilder design, to structure optimization then to the task setup/execution,
and last the pickup of transport calculation … all test should be done through
the existing framework, UI and cli/api. do not hack and be honest to detect the
missing gaps … we can run on local machine, using hierarchical dir design, but
should also test in flat dir configuration too."*

**The system.** Au(100) 2×2, **4 layers per side** (two whole ABAB stacking
periods, § 3.1's condition), an S–CH₂–S bridge, 2.40 Å S–Au contact, 37 atoms,
626 electrons. Small on purpose and structurally complete on purpose: periodic
leads, a stacking-period-multiple layer count, region labels, frozen electrode
layers, and a seam that continues the crystal. Deliberately cheap electronics —
SZ basis, 100 Ry mesh, `DM.Tolerance 1e-3`, `MD.NumCGSteps 2`, k 2×2×1 — which
is why the relaxation **concluded in 41 seconds**. None of this is a
convergence claim; it is a workflow claim.

**What a test can hold is held**: `tests/test_workflow_junction_walkthrough.py`
pins F4, F9, F13 and F17 as `xfail(strict=True)`, so each turns XPASS the day it
is fixed. The browser-side findings and the two prose findings are not in it,
and the file says why.

## 0. The short version

| | |
|---|---|
| **Reached, working** | build → label → relax → conclude → cite → describe, in both directory shapes |
| **Dead end** | `described` → `running`, for transport, in **every** door |
| **Worst single defect** | F9 — every junction the CLI builds has a defective crystal seam, silently |
| **Worst single blocker** | F17 + F18 — the transport ladder cannot be prepped or launched at all |

**Stages `electrode_L`, `electrode_R`, `device`, `transmission` and
`summarize run` were never reached.** Everything up to and including the
description is verified working; everything after it is blocked. So this
document proves the *front* of the transport workflow and disproves the *back*.

## 1. The findings

## Step 1 — build the junction (CLI `smiles` + `modify --electrode`)

F1. `modify --help`'s own worked example cannot run as printed.
    The 3-step pipe example mixes `--delete` and `--orient-axis` in ONE
    call ("molbuilder modify bdt.xyz - --orient-axis 0,3 --center
    midpoint" is fine, but the example's first line in the docstring
    pairs two op TYPES).  The refusal is correct and well worded; the
    EXAMPLE is what is wrong.  Severity: small (docs).

F2. Empty stdin reports a PDB error for xyz input.
    `... | molbuilder modify - out.xyz` with an empty upstream says
    "no ATOM/HETATM records found in PDB input", which sent me looking
    for a format-sniffing bug that does not exist.  Should say the input
    was empty.  Severity: small (diagnostics).

F3. `--orthogonal`'s help text is backwards, and nothing pre-validates.
    Help: "use ASE's orthogonal supercell (only meaningful for
    fcc(111))".  `FCC_ORTHOGONAL_CHOICES` in modify.py says the
    opposite: (100) and (110) accept ONLY orthogonal=True; (111) is the
    one surface where it is a free choice.  So the CLI default (False)
    is the single value a (100) slab cannot be built with -- exactly the
    failure the code comment records for the old UI ("the box starts
    unchecked, and unchecked is the one setting a (100) slab cannot be
    built with").  The web path avoids it by sending the table to the UI
    (blueprints/modify.py:169); the CLI has no equivalent.
    Severity: medium -- blocks (100)/(110) electrodes from the CLI.

F4. `_validate_orthogonal_compat` does not exist.
    modify.py:742 promises "Per-(plane, orthogonal) compatibility is
    enforced by :func:`_validate_orthogonal_compat` before the builder
    is called."  Grep: 1 reference (that docstring), 0 definitions.  The
    only guard is ASE's own exception.  Severity: small (a docstring
    naming a guard that was never written) -- but it is why F3 bites.

F5. The slab default lattice constant is a_experimental, not a_pbe.
    `_load_fcc_lattice()` ("back-compat shim") returns a_experimental
    for every metal, and that is what `build_electrode_slab` uses ->
    Au a = 4.078 A.  `engines/transport.md` 7.1 says "the lead must use
    the same one the device was built with ... For a PBE run that means
    a_pbe, not the room-temperature experimental value", and a_pbe =
    4.158 A -- a 1.9% difference, which is the magnitude invariant I10
    exists to prevent.  The default functional is GGA/PBE.  Nothing
    warns; `--lattice-constant` can override.  The shim's own docstring
    says callers needing the choice "should hit load_fcc_lattice_full
    directly (the web meta endpoint does)" -- so the web offers the
    choice and the CLI silently takes the wrong one for a PBE run.
    Severity: medium (science), by default rather than by mistake.

F6. THE SEAM VERDICT HAS ONE DOOR, and the CLI is not it.
    `science/junction-cell.md` 6 deliberately leaves c = z_extent
    ("A freshly built slab therefore gets c = z_extent ... That is not a
    usable cell ... and classify_seam says so -- collision.  That is the
    point.") and relies on the verdict being delivered: 6.1 "What the
    program owes you is a verdict on what you set, and classify_seam
    gives one on every build".
    MEASURED on the junction I built: image gap across z = 0.0000 A,
    and classify_seam(pos, cell) -> verdict='collision', message "the
    boundary leaves 0.00 A of room -- the box is not padded, and no
    engine can use it".
    But classify_seam has exactly ONE consumer in the tree:
    web/blueprints/modify.py:701 (the Modify tab's electrode endpoint).
    So:
      - CLI `modify --electrode` prints "Wrote ... 37 atoms" and no verdict.
      - `molbuilder validate` reports "cell volume / atom-bounding-volume
        = 1.78 -- cell is suspiciously tight", NOT the collision.  The one
        verdict that says "no engine can use it" is downgraded to
        "suspiciously tight" by the validator a person would run.
      - the Molbuilder tab's Cell page shows the lattice with no notice.
    Severity: HIGH -- a junction built through the documented CLI pipe is
    unusable and nothing says so until SIESTA stops.

F7. Molbuilder tab: "Load picked file" ADDS to the restored structure.
    A fresh visit restored a previous 306-atom structure; loading the
    37-atom junction gave "Added junction.xyz -- 37 atoms, 343 in total"
    and a notice that the added structure's cell was DISCARDED.  Adding
    is clearly deliberate (that is how slabs get built), but the first
    action on a restored session silently mixes two systems, and for a
    transport junction the discarded cell is the load-bearing part.
    Clear structure -> Load gives the right result, with a good
    confirmation dialog.  Severity: medium (workflow trap).

F8. Region labelling worked, and the rule is not kept.
    Filter (By atom index, 1-based, "6-21") -> Apply filter -> 16 of 37
    -> region dropdown -> Assign.  Correct, and the 1-based display
    convention is honoured (stores.js by_index shifts once).  The saved
    sidecar carries regions {R-electrode:16, L-electrode:16,
    frozen_atoms:32} and selection_rules: {} -- the RULE that defined
    the set is discarded, only the materialised indices survive.  The
    sidecar has a selection_rules field for exactly this.  Severity:
    small -- to confirm whether any writer populates it.
    (My first reading here was wrong: I concluded the filter was unwired
    because nothing happened on input.  There is an explicit "Apply
    filter" button; the two-step is by design.)

F9. **CLI `--electrode` builds a crystallographically DEFECTIVE seam,
    on every plane and every layer count.**  MEASURED, and it is the
    headline finding of this walkthrough.

    `add_slab` has two placement controls: `sequence` (the A->B->C walk)
    and `start_registry` (which site the first layer lands on).
    `cli.py:1103` passes `sequence=_CONTINUES_THE_CRYSTAL[side]` and
    leaves `start_registry` at its default 0 for BOTH slabs, and the
    `--electrode` spec grammar (ELEM:PLANE:MxNxL@contact=D:SIDE=IDX) has
    no field for it.  Measured with c = z_span + d per 6.1:

      plane  layers/side   junction-cell.md 3.1 says   CLI produces
      (111)  3, 6          continues                   TWIN
      (111)  4             eclipsed                    eclipsed
      (100)  4, 6          continues                   ECLIPSED
      (100)  3             eclipsed                    eclipsed
      (110)  4, 6          continues                   ECLIPSED
      (110)  3             eclipsed                    eclipsed

    NOT ONE "continues" case reproduces.  On (111) the seam step is the
    REVERSE of the in-slab step -- the twin signature 3.1 warns carries
    the right bond length, "which is exactly why a distance check misses
    it".

    `start_registry` is the control, and it works: sweeping it on
    (100)/4 gives +z=1, -z=0 -> continues, seam_step (1.442, 1.442),
    gap 2.8837 = a/sqrt2; on (111)/3, +z=1, -z=0 -> continues.  So
    `add_slab` is correct and the CLI does not drive it.

    The WEB slab card exposes both ("Start registry" + "Sequence"
    selects, modify.html:721-729; the route takes start_registry,
    blueprints/modify.py:607) AND returns the seam verdict at build
    time.  That door is sound.  The CLI is the broken one.
    Severity: HIGH (science) -- silent, and the default.

F10. `junction-cell.md` 3.2 states the wrong mechanism for `--electrode`.
    "`sequence="ACB"` is the alternative, and it is what `--electrode`
    now uses on that side" -- presented as what makes the crystal carry
    on.  Measured: `sequence` alone never changes the seam registry
    (every row of F9's table used the documented ACB mapping).  Only
    `start_registry` does, which 3.2 calls "translation, no flip" but
    does not name.  So the contract's own claim about its CLI is false,
    which is why F9 went unnoticed.
    Severity: HIGH (contract) -- fix the RULE, then the CLI.

F11. Setting `c` is browser-only, and `c` is a mandatory step.
    6 deliberately leaves c = z_extent ("not a usable cell") and makes
    setting it the user's decision.  No CLI verb and no API route sets a
    cell: `modify`'s ops are delete/orient/rotate/electrode, and the
    modify blueprint's routes are delete/add_atom/append/orient/rotate/
    translate/calibrate/slab/lattice-from-run -- `c` is edited in the
    client and persisted on save.  So a scripted junction cannot be
    completed, and the seam verdict (F6) is also absent from the
    cell-apply path -- the verdict is missing exactly where 6.1 says
    "what the program owes you is a verdict on what you set".
    Severity: medium (workflow) -- blocks CLI-only automation.

## Step 2-3 — describe, hand over, prep, launch  (all through the real doors)

WHAT WORKED, and is worth recording as working:
  * Structure-optimization tab loaded the junction WITH its regions.
  * Live preflight said the right things: mesh_cutoff 100 Ry below the
    ~150 Ry floor (my deliberate choice), "region label(s)
    ['L-electrode','R-electrode'] which the SIESTA run does NOT
    consume", and "32 atom(s) held fixed ... from struct.frozen_atoms".
  * Send to Task setup wrote the pair + 50 KB template + task.1st.json.
  * Task setup: shape asked with NO default (Flat / Hierarchical, both
    explained), hand-over shown read-only, one stage, Save wrote
    task.json + environment.json and removed task.1st.json.
  * `jobset prep run coarse --bundle ...` copied the 4 pseudos from the
    projects tree (the bare name `pseudopotential` resolved), rendered
    01_coarse/run-0, printed everything it resolved.
  * The DECK IS CORRECT: MeshCutoff 100, PAO.BasisSize SZ,
    DM.Tolerance 1e-03, MaxSCFIterations 30, MD.TypeOfRun CG,
    MD.MaxForceTol 0.2, kgrid 2x2x1, LatticeVectors 22.038 along z,
    coords shifted into [0,c), Geometry.Constraints for the 32 frozen.
  * `jobset launch run coarse --mode direct` -> mpirun -np 4 siesta,
    monitor attached.

F12. prep warns `cell.atoms_outside` on a cell that CONTAINS its atoms.
    prep printed: "Some atoms are outside the box ... a 2.02/-0.58,
    b 2.02/-0.58, c 10.00/-7.96".  Measured on the same sidecar:
    `cell.resolve(struct)` -> contains_atoms=True, clearances
    [(0.0,1.442),(0.0,1.442),(0.0,2.039)], box 5.767/5.767/22.038 at
    corner (-2.019,-2.019,-10.0).  `molbuilder validate` on the same
    file does NOT raise it.  And the rendered deck is right.
    So prep's preflight and prep's renderer disagree about the cell, and
    the printed clearances match neither.  ROOT CAUSE NOT ESTABLISHED --
    reported with the measurements rather than a guessed mechanism.
    It matters because `project-layout.md` makes prep's printout the
    thing that "makes submit a plain yes", and the remedy it offers
    ("move the corner") would break a correct cell.
    Severity: medium (false alarm at the authoritative moment).

F13. The run log's provenance block reports EVERY BLOCK-VALUED
     parameter as "not in the deck", including a k-grid that is set.
    The wrapper writes "-- not in the deck; the engine default applies
    --" and lists `kgrid (catalogue default (1, 1, 1))` while the deck
    it is describing carries `%block kgrid_Monkhorst_Pack / 2 2 1`.
    Mechanism, measured directly:
      script_emit.parameter('kgrid','siesta',deck_text=deck)
        -> writes=('%block kgrid_Monkhorst_Pack',) value=None
      same call for mesh_cutoff -> '100.0', basis_size -> 'SZ',
      dm_tolerance -> '1e-03'   (scalars read back fine)
    runwrap.py:1771 lists a parameter as absent when
    `param.writes and param.value is None`, so the read-back's
    inability to parse a `%block` becomes "the engine default applies".
    Both block-valued SIESTA parameters (kgrid, kgrid_displacement) are
    affected, and they are the only two in that list that are actually
    emitted.
    This inverts the one statement the block exists to make -- its own
    docstring: "a reader chasing a surprising number needs to know that
    it was never set rather than assume the deck is the whole story."
    LOAD-BEARING FOR TRANSPORT: the composite reads the transverse k
    from the cited relaxation's own deck (transport.md 7.1), so someone
    checking whether their 2x2x1 travelled reads that it did not.
    Severity: HIGH (provenance says the opposite of the truth).

F14. Send to Task setup is a silent no-op when a FILE is selected.
    The destination is the sidebar's selected FOLDER.  With
    `junction.xyz` still selected the button (enabled) did nothing --
    no dialog, no status line, no console error.  Selecting the folder
    made it work and navigate.  A disabled button with a reason, or a
    "pick a folder" status, is the difference between a 10-second and a
    10-minute detour.
    Severity: small (diagnostics).

## Step 4 — the transport pickup

WORKED: the citation dialog walked the tree, accepted the run directory,
and the meta line read
  "CONCLUDED (rc 0 at Tue Sep 15 05:43:49 PM MST 2026) - SZ - 100 Ry -
   GGA/PBE - k 2x2x1 - 37 atoms"
-- molbuilder's own marker answering first (it carries the rc), the
electronic contract read straight off the cited deck, and the un-nested
parenthetical from today's earlier fix.  The relaxation itself concluded
in 41 s (17:43:06 -> 17:43:47), SCF by DM+H, cell preserved at
5.767/5.767/22.038, and left .fdf + .XV + 0_NORMAL_EXIT + .concluded.
Re-prep made run-1 and left run-0 byte-intact.

F15. `bridge` is documented as IMPLICIT and required as EXPLICIT.
    `engines/transport.md` 4: "**`bridge`** -- the scattering region ...
    **Not** a TranSIESTA block -- it's implicit ('the atoms in no
    electrode region')."
    The composer refuses: "5 atom(s) carry no partition label: atom 0
    (S), atom 1 (C), atom 2 (S), atom 3 (H), atom 4 (H).  Every atom
    must be exactly one of L-electrode, R-electrode, bridge, buffer ...
    an unlabeled atom has no place in TranSIESTA's atom order and would
    be misassigned silently."
    Both positions are defensible; they contradict.  A person who reads
    4 labels only the two electrodes -- which is exactly what I did --
    and is refused at the citation step, after the relaxation has
    already run.  The refusal itself is excellent (names every atom and
    the reason); the DOC is what is wrong, and it is the doc a person
    reads first.
    Severity: medium (contract) -- one sentence to fix, and it costs a
    full relaxation cycle to discover.

F16. Fixing a label means re-running the relaxation.
    For a form-A citation the labels are read from the DECK's own
    atom-metadata block first (compose.labeled_citation_structure:
    "Form A's precedence is the deck's own block FIRST").  The deck is
    rendered at prep from the calculation folder's source sidecar, which
    was copied there by the hand-over.  So a label added after
    describing reaches the citation only after re-prep AND re-launch --
    the geometry is unchanged, but there is no way to say "same run, new
    labels".  Cheap here (41 s); on a real junction it is a day.
    Not obviously wrong -- the deck IS the record of what ran -- but
    worth stating in the contract, because F15 guarantees people hit it.
    Severity: small (workflow), conditional on F15.

F17. *** THE CLI CANNOT PREP A TRANSPORT DESCRIPTION AT ALL, and the
     WEB DOOR PREPS THE SAME FILE FINE. ***  The two doors disagree
     about whether the Transport tab's own output is valid.

    `molbuilder jobset prep run seed --bundle wf-AuSCS/transport/tr-h`:
      Error: the description fails its own preflight (stages.md 6.6):
        - 'varies' names 'tbt_k_grid', which is not a field of SiestaConfig
        - 'varies' names 'transmission_emax_ev', ... SiestaConfig
        - 'varies' names 'transmission_emin_ev', ... SiestaConfig
        - 'varies' names 'transmission_n_points', ... SiestaConfig
        - stage 'device' overrides 'transmission_emin_ev', ... SiestaConfig
        (+3 more)

    POST /api/task-setup/prep {"dest":"projects/wf-AuSCS/transport/tr-h",
    "kind":"run","stage":"seed"}  ->  ok=true, dirs=["01_seed"], and it
    wrote the whole bundle: 01_seed/tr-h_01_seed.fdf + run.sh + the four
    pseudos + monitor, job-set.json, STAGE-PLAN.md, environment.json,
    junction.cited.fdf, junction.xyz + .molstruct.json (the sorted
    copy), atom-permutation.json, slot-provenance.json.

    CAUSE, read not guessed: `validation/task.py::config_class_for`
    returns `known.get(task.engine)` with
    `known = {"siesta": SiestaConfig, "pyscf": PySCFConfig}`.  A
    transport task correctly carries engine.name = "siesta" (TranSIESTA
    IS siesta -- there is no separate binary), so the validator picks
    SiestaConfig and rejects the transport vocabulary field by field.
    `task.calculation == "transport"` is right there in the file and is
    never consulted.  `transport/stages.py` has the CORRECT check
    ("TransportConfig field names are the vocabulary") -- so there are
    two validators and the wrong one runs first.

    WHAT THIS MEANS FOR plan.md W26.  W26 records "**The Task setup seam
    works**: prep-plan answers for a transport description with the five
    stages ...".  That is TRUE and INCOMPLETE: I re-verified prep-plan
    answers ok=true with the five stages and the hierarchical shape.
    But prep-plan only PLANS.  The seam plans a ladder the CLI cannot
    prep, and the documented transport workflow in `transport.md` 3 is
    written entirely in CLI verbs:
        molbuilder jobset prep run seed   # then launch, stage by stage
        molbuilder jobset summarize run
    Every one of those is behind this refusal.
    Severity: CRITICAL -- the documented transport workflow is
    unreachable from the command line it is documented in.

F18. The web prep door writes a transport bundle with NO ATTEMPT
     DIRECTORY, so the stage cannot be launched either.
    After the successful web prep the tree holds `01_seed/` with the
    deck, run.sh, pseudos and monitor -- but no `01_seed/run-0/`.
      molbuilder jobset launch run seed --mode direct
        Error: job 'seed': no attempt is open under 01_seed/ -- a
        hierarchical stage runs in run-<n>, never in its own container
        (project-layout.md 1.5, 1.6).  Open one:
            molbuilder jobset prep run seed
    The remedy it prints is exactly the command F17 refuses.
    NOT a general web-prep gap -- measured: the same endpoint on the
    OPTIMIZATION bundle (`kind=run, stage=coarse`) returned ok=true and
    DID create `01_coarse/run-2`.  So opening the attempt works for a
    SIESTA ladder and not for the transport composite.

    *** NET RESULT: the transport five-stage ladder cannot be launched
    by ANY door.  CLI prep refuses the description; web prep accepts it
    but leaves no attempt to run in; CLI launch then refuses and sends
    you back to the CLI prep that refused.  The workflow dead-ends
    between "described" and "running". ***
    Severity: CRITICAL.

    (So stages electrode_L, electrode_R, device, transmission and
    `summarize run` were NOT reachable in this walkthrough.  Everything
    up to and including the description is verified working.)

## Both directory shapes — what actually differs

WORKED, both shapes, optimization half:
  * hierarchical: prep -> 01_coarse/run-0, relaunch -> run-1, run-0 left
    byte-intact.  Concluded 41 s.
  * flat: prep said "prepped 1 job dir(s) ... (flat: no attempt to open;
    runs are told apart by the wrapper's output index)" and put the
    deck, pseudos, run.sh and monitor in the calculation root.
    Concluded; left 0_NORMAL_EXIT, AuSCS_relax.XV, -run1.concluded.
  * BOTH are citable for transport (measured via
    /api/transport/describe_attempt):
      relax-f                         -> form='relaxation' concluded=True
      relax-h/01_coarse/run-1         -> form='relaxation' concluded=True

F19. A FLAT calculation with more than one stage is UNCITABLE for
     transport, and the refusal's remedy contradicts the shape.
    Measured: a flat directory holding two stage decks answers
      "holds 2 .fdf files (AuSCS_relax_01_coarse.fdf,
       AuSCS_relax_02_tight.fdf) -- the citation names a directory, so
       the directory must answer unambiguously.  Keep one deck, or cite
       a directory holding one."
    Flat's whole design is every stage's deck in the calculation root
    (`worked-example.md` 4), so "keep one deck" cannot be followed
    without abandoning the shape, and there is no per-stage directory to
    cite instead.  So: flat + 1 stage is citable, flat + >=2 stages is a
    dead end reached only AFTER the ladder has run.
    Task setup asks the shape with no default and says nothing about
    this; a transport pickup is the commonest reason to relax a junction.
    Severity: medium (design interaction) -- either the citation learns
    to take `directory + stage`, or the shape card warns.

## 2. What this says about the design, not just the code

**Three of the nineteen are a contract disagreeing with its own code**, and
each cost real time to discover rather than to fix:

* F10 — `junction-cell.md` § 3.2 says `--electrode` continues the crystal via
  `sequence="ACB"`. Measured: `sequence` never changes the seam registry.
  Because the contract says the CLI is right, nobody checked, and F9 has been
  shipping.
* F15 — `transport.md` § 4 calls `bridge` *implicit*; the composer requires it
  *explicitly*, and refuses the citation after the relaxation has run.
* F4 — a docstring names a validator that was never written, which is why the
  CLI's `--orthogonal` default reaches ASE instead of a molbuilder refusal.

**The pattern worth naming: the planning door and the doing door do not share
a validator.** F17 and F18 are the same shape — `prep-plan` answers, `prep`
refuses, the web `prep` writes an incomplete bundle, and `launch`'s remedy
points at the refusing CLI. `plan.md` **W26** recorded "the Task setup seam
works" on the strength of `prep-plan` alone; that was true and it was not
enough. **A seam that plans is not a seam that runs**, and only running it
found that out — which is the argument for this kind of walkthrough over any
amount of reading.

**And the safety nets that exist are not wired to every door.** `classify_seam`
gives exactly the right verdict — measured `collision` at 0.00 Å, `eclipsed`,
`twin`, `continues` — and has ONE consumer: the web slab endpoint. The CLI
build path, the cell-apply path and `molbuilder validate` all miss it, and
`validate` downgrades a 0.00 Å image collision to "cell is suspiciously
tight". The measurement is right; the delivery is not.

## 3. What went right, and is worth not breaking

Written down because a gap list reads as if nothing works, and most of this
worked first time:

* **The refusals are excellent where they fire.** Every one named the offending
  atoms/files and the remedy: the unlabeled-atom refusal listed all five atoms
  and the four legal labels; the two-deck citation named both files; the
  `no attempt is open` error cited `project-layout.md` § 1.5–1.6; the
  one-op-per-call refusal explained the pipe.
* **The preflight said the right things** at describe time — the 100 Ry mesh
  below the production floor, "region label(s) … which the SIESTA run does NOT
  consume", and the 32 frozen atoms it had picked up from the sidecar.
* **`fdf`-is-truth held.** The transport citation read `SZ · 100 Ry · GGA/PBE ·
  k 2x2x1 · 37 atoms` straight off the cited deck, and molbuilder's own
  concluded-marker answered ahead of SIESTA's, carrying the rc.
* **The deck was correct in every particular** — cell 22.038 along z, atoms
  shifted into `[0, c)`, `Geometry.Constraints` for the frozen 32, and every
  loose value I chose.
* **Attempts are immutable.** Re-prep made `run-1` and left `run-0`
  byte-intact; the flat shape said in one line why it opens no attempt.
* **Task setup asked the shape with no default**, explained both, and refused
  to be written until the shape and the machine were answered.

## 4. Order of work

| # | fix | why first |
|---|---|---|
| 1 | **F17** — `config_class_for` branches on `task.calculation` | one function; unblocks the entire documented transport CLI |
| 2 | **F18** — the web prep opens the attempt for a composite too | without it F17's fix still cannot launch |
| 3 | **F9 + F10** — drive `start_registry` from the side; correct § 3.2 first | silent science defect, and the contract is what hid it |
| 4 | **F13** — read `%block` values back, or report them as unknown rather than absent | the provenance block currently says the opposite of the truth |
| 5 | **F6** — give the CLI build and `validate` the seam verdict | the check exists and is right; only the delivery is missing |
| 6 | **F15** — one sentence in `transport.md` § 4 | costs a whole relaxation cycle to discover |
| 7 | F3/F4/F5, F11, F12, F14, F16, F19 | the rest, in severity order above |
