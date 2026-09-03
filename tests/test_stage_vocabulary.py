"""The four questions that decide whether the staged-run design landed.

They are things a reader should be able to *check*, not believe, and the
rules they check belong to the live contracts:

  1. Is there one way to say "stage"?  -- `engines/stages.md` § 1 + § 1.4:
     a stage is OURS, not the engine's, and one mechanism serves every
     engine.
  2. Does a stage's name survive -- is no file named by a stage's
     *position*?  -- `engines/stages.md` § 7.3 R5, with the one place a
     number does belong settled by `project-layout.md` § 4.1
     (`<seq>_<name>`, assigned once).
  3. Does everything run through the wrapper?  --
     `execution/running-a-job.md` §§ 2 and 5: activation and the engine
     launch live in the wrapper, nowhere else (`stages.md` § 8 hands them
     over by name).
  4. Does each stage start because someone said so?  -- `stages.md` § 8:
     a producer "asks for nothing this file does not carry, because there
     are no edges left to thread"; the job set carries none
     (`job-system.md`), and the one sequenced thing that exists -- a
     transport bias scan -- chains INSIDE one submission's walker
     (`engines/transport.md` § 6).

**All four are answered YES today** (Q2/Q3/Q4 on 2026-08-10, each noted at
its own section below; Q1 by the agreed mechanism set).  Three of them were
``xfail(strict=True)`` while the staged-run work was in flight -- strict, so
the day the behaviour started working the suite failed until the record was
updated.  What the module is FOR now is the other direction: each answer is
mechanical, so a regression is caught by a test rather than by the next
fresh-eyes review.

Run it as a module, from the repository root, for the baseline table::

    python -m tests.test_stage_vocabulary

(as a module, not a path: the reporter imports ``molbuilder`` and the package
is not installed into the env -- see docs/process/testing.md.)

WHAT THE LEDGERS BELOW ARE FOR.  Question 1 is answered by attributing every
declaration in the package whose name carries "stage" to one of the
mechanisms the design enumerates, or to a role that is not a mechanism at
all.  Retiring a mechanism is then a one-line edit -- which is what made
"these nine names are gone" provable rather than intended -- and growing a
new one fails until somebody writes down what it is.  Same shape as
``test_css_no_duplicate_selectors.py``: enumerate mechanically, attribute by
hand, assert the two agree.

THE BLIND SPOT, STATED SO NOBODY TRUSTS THIS FURTHER THAN IT GOES.  Detecting
by name cannot find a mechanism that does not carry the word.
``molwatch_log_basename(label, stage=N)`` is exactly that: it takes a stage and
writes its number into a filename, and no detector here sees it.  Question 2
catches it instead -- which is the general answer, and the reason the design
review says to search *by behaviour, not by name*.

The dated measurements of these counts, and the phasing that moved them, are
history: ``docs/archive/2026-08-11-staged-runs-architecture.md`` § 8c (where
the four questions were first written) and
``docs/archive/2026-08-19-staged-runs-implementation-plan.md``.  Read them for
provenance, never to decide what the rule is.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "molbuilder"
STATIC = PKG / "web" / "static"

# Third-party or generated -- not ours to inventory.
JS_EXCLUDE = ("vendor", "codemirror")


# ===================================================================== #
#  Question 1 -- how many ways are there to say "stage"?                #
# ===================================================================== #

# Declaration sites, not uses: an option a user types, a type a value has,
# a function that produces or transforms a ladder, a table of presets.
_PY_DETECTORS = (
    ("click option", re.compile(
        r'@click\.option\(\s*"(--[a-z0-9-]*stage[a-z0-9-]*)"')),
    ("class", re.compile(r'^class\s+(\w*[Ss]tage\w*)\b', re.M)),
    ("function", re.compile(r'^def\s+(\w*stage\w*)\s*\(', re.M)),
    ("constant", re.compile(r'^([A-Z][A-Z0-9_]*)\s*[:=]', re.M)),
)

#: Applied to the constant detector only: the other three already carry the
#: word in their pattern.  Written as a filter rather than baked into the
#: regex because ``^[A-Z][A-Z0-9_]*STAGE`` demanded a character BEFORE
#: "STAGE" and so missed ``STAGE_FIELDS`` outright -- caught 2026-08-07 when
#: P1 added two of them.
_CONSTANT_IS_STAGEY = lambda name: "STAGE" in name  # noqa: E731

# symbol -> (mechanism number per architecture § 8b, or None; where; role)
#
# A number means "this symbol implements that mechanism".  ``None`` means it
# reads, names or carries stages without being a way of *expressing* one --
# the observing side, or a different sense of the word entirely.
PY_LEDGER: dict[str, tuple[int | None, str, str]] = {
    # -- the ten -------------------------------------------------------
    "apply_siesta_stage": (
        1, "molbuilder/config/siesta.py", "applies that overlay"),
    "SIESTA_STAGE_PRESETS": (
        1, "molbuilder/config/siesta.py",
        "the tier values -- real science, kept as the defaults a new stage "
        "is created with"),
    "PYSCF_STAGE_PRESETS": (
        1, "molbuilder/config/pyscf.py",
        "the same table for PySCF (2026-08-18).  Not a new mechanism: it is "
        "mechanism 1 finally being the same shape on both engines, and the "
        "numbers in it are `tuning.md` § 2.4's, guarded by "
        "test_doc_claims.py rather than restated anywhere"),
    "vibration_stages": (
        2, "molbuilder/pyscf/stages.py",
        "the vibration kind's ONE-STAGE ladder (`stages.md` § 6.8, "
        "spectra-migration P1): a `freq` stage with no overrides, so a "
        "vibration description is an ordinary § 6.5 ladder rather than a "
        "stage-less special shape.  `jobset init` selects it by "
        "`--calculation vibration` and refuses --stage-strategy beside it "
        "-- the tiers are an optimization vocabulary"),
    "--stage-strategy": (
        2, "molbuilder/jobset/_cli.py",
        "named presets over the *enable* flags, on `jobset init` -- "
        "where the ladder is DESCRIBED rather than rendered, and for both "
        "engines through one option.  It was also on `molbuilder pyscf` "
        "until 2026-08-18; a command that writes ONE deck had no business "
        "carrying a ladder (`stages.md` § 1.1a)"),
    "SIESTA_STAGE_STRATEGY_PRESETS": (
        2, "molbuilder/config/siesta.py",
        "those presets -- pure enable-mask data now; the applier that turned "
        "them into SiestaStageSpec objects went with the class (P2)"),
    "STAGE_STRATEGY_PRESETS": (
        2, "molbuilder/config/pyscf.py",
        "PySCF's copy of the enable-mask table.  Two homes for one fact, "
        "and a drift test is what keeps them equal -- the honest reading is "
        "that this is mechanism 2's remaining debt, not a second mechanism. "
        "`default_pyscf_stages` reads it exactly as SIESTA's twin reads "
        "SIESTA_STAGE_STRATEGY_PRESETS"),
    "_refuse_duplicate_stage_names": (
        None, "molbuilder/task.py",
        "the ONE case-insensitive duplicate check (D10, 2026-08-13) -- "
        "names key filenames, and case-insensitive filesystems make "
        "'Tight' and 'tight' one deck; both parsers and the constructor "
        "call this instead of keeping their own loops"),
    # Mechanism 5 -- SiestaStageSpec, _default_siesta_stages,
    # validate_siesta_stages -- is RETIRED (P2 unit 2, 2026-08-07).  These
    # three rows are deleted rather than commented out: their absence is
    # what proves the subtraction, and the guard below fails if any of the
    # names comes back.  What replaced it is mechanism 11, which was
    # already here.
    # -- decision 27's one naming rule (P4 units 2-4, 2026-08-10) ------
    #  These are not an eleventh way to EXPRESS a stage -- they are the one
    #  way to NAME one, which question 2 is about.  Mechanism 6 renders the
    #  decks; these decide what those decks are called, in a single place
    #  that the emitter, the log, the browser and the decoder all read.
    "stage_token": (
        None, "molbuilder/identity.py",
        "builds `<NN>_<name>` -- the one token every per-stage artifact is "
        "named from, and the same one a stage directory uses"),
    "parse_stage_token": (
        None, "molbuilder/identity.py",
        "reads it back.  The decoder used to keep its own `-stage(N)` regex, "
        "a second spelling of the emitter's convention that could drift"),
    "_detect_stage_name": (
        None, "molbuilder/parse/dirs/job.py",
        "the observing side -- the half of the token a person reads"),
    # -- decision 28's one RESOLVER (§ 8f, 2026-08-10) -----------------
    #  Not an eleventh way to express a stage either: these are the one way
    #  to REFER to one.  The token above says what a stage's files are
    #  called; these say what a person types and what a screen prints, and
    #  they exist because six places were answering that separately -- two
    #  of them with the stage's ROW, which question 2 is about.
    "StageRef": (
        None, "molbuilder/identity.py",
        "the pair every surface refers to a stage by -- the assigned "
        "ordinal and the name.  `seq` is None where none is assigned (a "
        "sweep point), never filled in from a position"),
    "resolve_stage_ref": (
        None, "molbuilder/identity.py",
        "`tight` | `#3` -> one StageRef (user-settled 2026-08-21: a bare "
        "number and the token are legal NAMES, so both retired as "
        "ordinal spellings).  #N MATCHES seq AND NEVER A ROW, which is "
        "R5's whole point"),
    "render_stage_refs": (
        None, "molbuilder/identity.py",
        "the on-disk listing format (tokens) for tables, so a status "
        "table and a directory cannot show two vocabularies"),
    "render_stage_choices": (
        None, "molbuilder/identity.py",
        "the TYPEABLE listing format for refusals -- name (#N) -- split "
        "from render_stage_refs on purpose: a refusal must not offer the "
        "on-disk spelling the resolver no longer accepts"),
    #  `identity.seq_text` -- what a `seq` column prints -- belongs to this
    #  group but is NOT a row here: it carries no "stage" in its name, so the
    #  detector cannot see it and the reverse check would call the row stale.
    #  That is this module's stated blind spot, not an omission; the guard on
    #  it is question 2, which asks about behaviour rather than names.
    #  `Shape.stage_dir` / `Shape.stage_glob` (molbuilder/jobset/shape.py, the
    #  § 9 object, 2026-08-10) belong to this group and are NOT rows here.
    #  They are METHODS, and the function detector is `^def` -- anchored at
    #  column 0, because what it inventories is a declaration, and a method is
    #  part of its class's.  `Shape` itself carries no "stage" in its name, so
    #  nothing here sees it either.  Adding rows makes the REVERSE check fail
    #  ("the ledger names a symbol the package no longer declares"), which is
    #  the guard telling the truth: it cannot see them.  Same blind spot as
    #  `identity.seq_text` above, and question 2 is what covers the pair --
    #  they exist so a stage is found by NAME in flat and by PATH in the
    #  hierarchy, which is a behaviour, not a vocabulary.
    "stage_refs": (
        None, "molbuilder/jobset/materialize.py",
        "the after-produce half: every job's ref, with seq READ BACK off "
        "its own deck.  Total over the JobSet, and the only place the two "
        "kinds are told apart"),
    "_resolve_stage": (
        None, "molbuilder/jobset/_cli.py",
        "the CLI's single door onto the resolver -- which jobs a verb acts "
        "on, and the ladder's refusal to act on all of them without "
        "--chain (project-layout 1.6)"),
    "_resolve_stage_name": (
        None, "molbuilder/jobset/_cli.py",
        "the half of that door every verb needs -- WHICH job did the user "
        "name.  Split out because `status <stage>` must resolve a name "
        "without inheriting the whole-set refusal: a whole-ladder status is "
        "a legitimate thing to ask for, unlike a whole-ladder submit"),
    "_stage_bench_dir": (
        None, "molbuilder/jobset/_cli.py",
        "where a stage's benchmark lives -- the `bench/` container "
        "(job-contracts § 6.3) resolved through the description, one "
        "answer for the trials, the job-set and the verdict, with the "
        "same unknown-stage refusal as every other lookup (U1, "
        "2026-08-12)"),
    "render_stage_status": (
        None, "molbuilder/jobset/runstatus.py",
        "the per-stage view (job-system.md 5.3) -- what happened to ONE "
        "stage: its attempts, its launch record and what it continued from.  "
        "Answerable only because a try is a directory and a launch is a "
        "record (project-layout 1.5, 1.6)"),
    "_stage_science": (
        None, "molbuilder/siesta/input.py",
        "the deck's one-line stage comment, DERIVED from the config being "
        "rendered so it cannot drift from the keywords below it (decision "
        "27's second half)"),
    "SIESTA_STAGE_NAMES": (
        1, "molbuilder/config/siesta.py",
        "what each tier is CALLED -- read by both doors (the --stage overlay "
        "and default_siesta_stages) so one tier cannot have two names"),
    #  ``render_siesta_stages_runner`` stood here.  DELETED 2026-08-10 with the
    #  function, its 142-line bash template and its fourteen tests (P5 unit 3,
    #  decision 29).  The row is removed rather than commented: its ABSENCE is
    #  what proves the subtraction, and the guard below fails if the name comes
    #  back.  Mechanism 6 is now the deck renderer alone.
    #  ``STAGES`` -- the bash array in the runner's template -- went with the
    #  template on 2026-08-10.  It was the only stage vocabulary that lived in
    #  GENERATED TEXT, which is where questions 3 and 4 used to look.
    "default_pyscf_stages": (
        11, "molbuilder/pyscf/stages.py",
        "the same builder for PySCF, body-for-body: mechanism 1's preset "
        "table for the science, mechanism 2's for the enable mask, `restart` "
        "positional, ValueError on an unknown strategy.  It read neither "
        "table and invented its own strategy names until 2026-08-18, so "
        "`--stage-strategy vib-quality` silently ran the publishable ladder"),
    "default_siesta_stages": (
        11, "molbuilder/siesta/stages.py",
        "builds the shipped ladder as task.Stage objects, reading the tier "
        "values from mechanism 1's preset table and the enable mask from "
        "mechanism 2's.  It produces the design's own stage rather than "
        "being a way of expressing one, which is why it is 11 and not a "
        "twelfth"),
    "TRANSPORT_STAGES": (
        11, "molbuilder/transport/stages.py",
        "the transport composite's FIXED five-rung ladder "
        "(archive/2026-09-01-transport-design.md 4.2): seed, electrode_L, electrode_R, "
        "device, transmission -- a DAG of different programs, not a "
        "parameter ladder, so the tuple is a design constant rather than "
        "a strategy output.  Skipping is the per-stage enabled flag "
        "(ruling Q4), never a different ladder"),
    "render_stage_deck": (
        None, "molbuilder/transport/stages.py",
        "renders ONE transport rung's deck from the composed junction + "
        "the one TransportConfig (P4b).  Not a mechanism for expressing "
        "stages -- it consumes the ladder above the way prep's SIESTA arm "
        "consumes spec_for"),
    "StageError": (
        None, "molbuilder/transport/stages.py",
        "the refusal class for a transport rung that cannot render -- "
        "carries the stage's name in its message, same role PrepError "
        "plays one floor up (prep translates it)"),
    "stage_inputs": (
        None, "molbuilder/transport/stages.py",
        "the transport composite's § 4.2 DAG as data -- which files a "
        "rung CONSUMES from the concluded attempts of the rungs before "
        "it (P5's gather reads it).  Consumes the ladder like "
        "render_stage_deck; not a way of expressing stages"),
    # Mechanism 3 -- ``--stages-json``, ``stages_from_dicts`` and
    # ``stages_from_configs`` -- and mechanism 8 -- ``StageSpec``,
    # ``_default_stages``, ``validate_stages`` -- are RETIRED (2026-08-18).
    # `stages.md` § 1.1a made a PySCF ladder N decks and N jobs, which left
    # mechanism 8 with no reader and mechanism 3 with nothing to parse INTO.
    # Their rows are deleted rather than commented out: their absence is what
    # proves the subtraction, and the guard below fails if any name comes back.
    # What replaced them is mechanisms 1, 2 and 11 -- all three already here,
    # and all three now the same on both engines.
    #
    # Mechanism 10's Python half went with mechanism 8:
    # ``_stagespec_to_field_schemas`` walked a ``List[<dataclass>]`` config
    # field into a stage-table schema, and no config anywhere has such a field
    # now.  The JS renderer outlived it as reached-by-nothing until the
    # user's cleanup ask; it retired at the U6 close (2026-08-22), CSS
    # included -- the live stage table is Task setup's own.
    "Stage": (
        11, "molbuilder/task.py",
        "THE DESIGN'S OWN: a stage in task.json -- name, enabled, overrides, "
        "and no others.  This is the one P5 keeps; the other ten are what it "
        "measures itself against"),
    "_stage_from_obj": (
        11, "molbuilder/task.py", "parses one, refusing by name"),
    "STAGE_FIELDS": (
        11, "molbuilder/task.py", "the three fields § 2 allows"),
    "STAGE_NAME_RE": (
        11, "molbuilder/task.py",
        "the filename-safe set § 6.6 requires of a stage name"),
    # -- not a way of expressing a stage -------------------------------
    #  `stage_completion_tag` and `parse_stage_completion_tag` used to sit
    #  here.  Both are GONE, retired with the auto-tagger by
    #  `checkpointing.md` L4 -- *nothing tags a state on your behalf* -- and
    #  their rows are deleted rather than annotated, because this ledger's
    #  own rule is that the deletion IS the proof the subtraction happened.
    "StageStatus": (
        None, "molbuilder/jobset/runstatus.py", "the observing side"),
    "_stage_state": (
        None, "molbuilder/jobset/runstatus.py", "the observing side"),
    "_detect_stage": (
        None, "molbuilder/parse/dirs/job.py",
        "the decoder reading a stage back OUT of an emitted name -- "
        "question 2's other end"),
    "check_identical_stages": (
        None, "molbuilder/validation/stages.py",
        "§ 6.6a's check that a stage does not recompute the one before it -- "
        "a question ABOUT a ladder, not a way of expressing one"),
    # DIAG_1STAGE / DIAG_2STAGE (bench/__init__.py's diagonalisation
    # constants -- same word, unrelated role) left the ledger 2026-08-13
    # when the legacy siesta-gpu stack was deleted: the rows' removal is
    # the proof the subtraction happened, per this test's own rule.
    "_stage_of": (
        None, "molbuilder/resolve.py",
        "step 2's rung lookup -- which Stage this prep resolves, refusing "
        "rather than picking one (landed with the resolver, plan step 3)"),
    "_trial_stage_token": (
        None, "molbuilder/jobset/materialize.py",
        "the <NN>_<stage> a TRIAL's deck carries -- what nests "
        "bench-<point>/ inside the stage it measures (step 6 u3); a "
        "label-anchored parse of the deck's own name, never a count"),
}

# The JS side is attributed by FILE rather than by token: a hundred CSS class
# names inside one widget are one mechanism, and what actually matters is a
# *new file* growing stage machinery.
JS_LEDGER: dict[str, tuple[int | None, str]] = {
    "lib/form-schema.js": (
        None, "carries NO stage vocabulary since the U6 close -- the "
              "stage-table renderer (mechanism 10's JS half) retired "
              "2026-08-22 with its producer long gone; the live stage "
              "table is task-setup/viewer.js's own"),
    # structure-optimization/viewer.js was attributed here until 2026-08-16,
    # for a `p-stage-preset` that turned a stage NUMBER into a filename.  The
    # tab's cleanup deleted that path (the panel no longer names decks), so
    # the file carries no stage vocabulary at all and an entry for it would
    # be exactly the obsolete truth this map exists to prevent.
    "lib/molview/mount.js": (
        None, "molviewer-window-stage is a CSS layer -- unrelated role"),
    # Attributed 2026-08-16 with the Task-setup tab's editable stage table.
    # NOT a mechanism, for the reason mechanism 9's removal states three lines
    # up: what made that one a mechanism was turning a stage NUMBER into a deck
    # filename.  This file spells no ordinal and names no deck.  Its two uses:
    #   * `"stage" + n` -- the default NAME offered for a row the user just
    #     added (task-setup.md § 9), deduped against the names already taken.
    #     A name, which is the user's to change; the `<NN>` that orders it is
    #     assigned by `identity` at prep and never appears here (rule A1).
    #   * `c.group === "stage"` -- READING the catalogue's closed `group`
    #     vocabulary (template.md § 5) to decide which columns start ticked.
    #     Consuming a value the catalogue declares is the opposite of a second
    #     way to express a ladder.
    # The rest are `ts-*` DOM ids -- a CSS layer, like the row above.
    "task-setup/viewer.js": (
        None, "the stage TABLE -- offers a new row's default name and reads "
              "the catalogue's `stage` group; spells no ordinal, names no deck"),
    # Attributed 2026-08-26 with the sweep-summary charts.  Its single hit is
    # the word ELPA-1STAGE, in a comment, and it is not this vocabulary at
    # all: that is the name of an ELPA solver variant -- a VALUE a benchmark
    # sweeps `diag_algorithm` over, the way it sweeps a block size.  Same
    # letters, different layer.  The file spells no ordinal, names no deck,
    # and reads no `stage` group; a chart axis is the closest it comes to a
    # ladder and that ladder is the sweep's, not the pipeline's.
    "lib/inspectors/bench-summary.js": (
        None, "ELPA-1STAGE is an ELPA solver name -- a benchmark axis VALUE, "
              "not a pipeline stage"),
}

#: Eleven on 2026-08-07 morning, and it went UP on purpose: P1 added
#: task.json's own stage without retiring anything, which the plan calls "the
#: one phase that deliberately raises the mechanism count".
#:
#: TEN since P2 unit 2 the same day: mechanism 5 -- SiestaStageSpec and its
#: two helpers -- is deleted.  That is the first subtraction of the program,
#: and the number falling is what makes it a fact rather than a claim.
#:
#: NINE since 2026-08-11/12: `molbuilder fdf` went (plan step 5, C2) and took
#: mechanism 1's flag ``--stage`` and mechanism 4's ``--stage-resources``
#: with it -- the single-shot tier overlay is no longer expressible from any
#: surface (the presets remain as the ladder's defaults, applied by
#: ``apply_siesta_stage`` at resolve time).  ``--stage-strategy`` moved to
#: `jobset init`, where the ladder is DESCRIBED rather than rendered.
#: SEVEN since step 6 u5 (2026-08-12): mechanisms 6 and 7 -- the
#: produce-time renderer ``render_siesta_stage_fdfs`` and the producer
#: ``stages_to_jobset`` -- are DELETED with the second lifecycle.  The deck
#: is rendered by `prep` per resolved element, which carries no stage-named
#: symbol of its own: the ladder is data, not a mechanism.
#:
#: SIX since 2026-08-16: mechanism 9 -- the structure-optimization tab's
#: ``p-stage-preset``, which turned a stage NUMBER into a deck filename --
#: went with that tab's cleanup.  The panel names no decks now, so the last
#: place a stage was spelled in the browser is gone.
#:
#: FOUR since 2026-08-18: mechanisms 3 and 8 -- PySCF's own ``StageSpec``
#: ladder and the ``--stages-json`` route that authored one -- are DELETED
#: with the in-script loop that read them (`stages.md` § 1.1a).  **This is
#: the subtraction the whole PySCF exception was costing**: the four that
#: remained were 1 (tier values), 2 (enable masks), 10 (a dead JS renderer)
#: and 11 (task.json's own stage), and the first, second and fourth became
#: the SAME code on both engines rather than a pair of parallel ones.
#:
#: THREE since 2026-08-22 (the U6 close): mechanism 10's JS half -- the
#: form-schema stage-table renderer -- retired.  Its Python producer died
#: with mechanism 8, it had been recorded as reached-by-nothing awaiting a
#: UI ask, and the user's cleanup ask supplied one.  The three that remain
#: are 1, 2 and 11.
MECHANISM_COUNT = 3


def _py_sources() -> list[Path]:
    return sorted(PKG.rglob("*.py"))


def _js_sources() -> list[Path]:
    return sorted(p for p in STATIC.rglob("*.js")
                  if not any(x in str(p) for x in JS_EXCLUDE))


def detect_py_symbols() -> dict[str, set[str]]:
    """symbol -> the repo-relative files declaring it."""
    found: dict[str, set[str]] = {}
    for path in _py_sources():
        src = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(REPO).as_posix()
        for kind, rx in _PY_DETECTORS:
            for m in rx.finditer(src):
                name = m.group(1)
                if kind == "constant" and not _CONSTANT_IS_STAGEY(name):
                    continue
                found.setdefault(name, set()).add(rel)
    return found


def detect_js_files() -> set[str]:
    """Static JS files mentioning a stage token, relative to static/."""
    rx = re.compile(r'["\'][a-z0-9-]*stage[a-z0-9-]*["\']', re.I)
    return {p.relative_to(STATIC).as_posix()
            for p in _js_sources()
            if rx.search(p.read_text(encoding="utf-8", errors="replace"))}


def _expected_files(where) -> set[str]:
    """A ledger row's ``where`` as a set: one path, or several."""
    return {where} if isinstance(where, str) else set(where)


def measure_mechanisms() -> list[str]:
    """The distinct mechanisms the ledgers attribute, as '<n>: <where>'."""
    seen: dict[int, str] = {}
    for sym, (n, where, _role) in PY_LEDGER.items():
        if n is not None:
            seen.setdefault(n, f"{n}: {sym} ({where})")
    for f, (n, _role) in JS_LEDGER.items():
        if n is not None:
            seen.setdefault(n, f"{n}: {f}")
    return [seen[k] for k in sorted(seen)]


def test_every_stage_declaration_is_attributed():
    """Question 1.  Nothing in the package says "stage" unaccounted for."""
    found = detect_py_symbols()
    undeclared = sorted(set(found) - set(PY_LEDGER))
    assert not undeclared, (
        "new way(s) to say 'stage', not in the ledger:\n  "
        + "\n  ".join(f"{s} -- {', '.join(sorted(found[s]))}"
                      for s in undeclared)
        + "\n\nAdd each to PY_LEDGER with the mechanism it implements "
          "(architecture § 8b) or None plus the role it plays.")

    stale = sorted(set(PY_LEDGER) - set(found))
    assert not stale, (
        "ledger names symbol(s) the package no longer declares:\n  "
        + "\n  ".join(stale)
        + "\n\nIf a phase retired them, delete the rows -- that deletion IS "
          "the proof the subtraction happened.")

    # Set EQUALITY, not membership.  Membership let a symbol declared in
    # two files pass while the ledger named one -- which is precisely the
    # blind spot this file exists to close, and ``stages_from_dicts`` walked
    # into it on 2026-08-07 when task.py grew one beside PySCF's.  A row may
    # name a tuple of files when a name genuinely lives in more than one.
    moved = sorted(
        f"{s}: ledger says {PY_LEDGER[s][1]}, found in {', '.join(sorted(w))}"
        for s, w in found.items()
        if _expected_files(PY_LEDGER[s][1]) != w)
    assert not moved, "declared somewhere the ledger does not expect:\n  " \
                      + "\n  ".join(moved)


def test_every_stage_bearing_js_file_is_attributed():
    """Question 1, the browser half."""
    found = detect_js_files()
    assert found == set(JS_LEDGER), (
        f"unattributed: {sorted(found - set(JS_LEDGER))}\n"
        f"stale:        {sorted(set(JS_LEDGER) - found)}")


def test_mechanism_count_is_the_agreed_number():
    """The headline number.  It changes when a phase decides it changes."""
    mechs = measure_mechanisms()
    assert len(mechs) == MECHANISM_COUNT, (
        f"{len(mechs)} mechanisms, expected {MECHANISM_COUNT}:\n  "
        + "\n  ".join(mechs)
        + "\n\nMECHANISM_COUNT above is the authority, and the comment over "
          "it is the ledger: a phase that retires a mechanism moves the "
          "number and says which one, in the same commit.  The plan's § 6 "
          "table and archive § 8b are DATED measurements of this same count "
          "-- history, not restatements; do not edit them to match.")


# ===================================================================== #
#  Question 2 -- does a stage's name survive?                           #
# ===================================================================== #
#
# The rule is engines/stages.md § 7.3 R5: "the stage's position in the list
# must never appear in a filename ... Names are stable; positions are not."
#
# It is NOT "no number anywhere".  project-layout.md § 4.1 settles the one
# place a number belongs: a stage *directory* is ``<seq>_<name>`` -- number
# AND name, assigned once at produce and never reassigned.  What R5 forbids
# is a name carrying the position INSTEAD of the name, because inserting a
# stage then silently reassigns outputs that already exist.
#
# Two producers do that today; the flat ladder's ``<id>_<name>`` is correct.
# That is architecture § 8b's "three filename conventions" seen from the
# other side: 2 offenders + 1 correct = 3, target 1.

_POSITIONAL = re.compile(r"-stage\d")


def measure_positional_names() -> list[str]:
    offences = []

    from molbuilder.trajectory_log.format import molwatch_log_basename
    name = molwatch_log_basename("JOB", 1)
    if _POSITIONAL.search(name):
        offences.append(
            f"molwatch_log_basename('JOB', 1) -> {name!r} "
            "(project-layout.md § 4.1: '<id>_<name>.molwatch.log' flat, "
            "'<id>.molwatch.log' inside the stage directory)")

    # § 8c asks for more than "no offender I already know about": it asks that
    # deck, output and log AGREE.  So the decks and the runner get read too --
    # they are correct today, which is the point of pinning them before P4
    # moves anything.  (test_siesta_stages_emit.py pins the deck's exact name;
    # this asks the different question of whether ANY producer has drifted
    # onto a position, and three lines is a cheap way to not find out late.)
    # Repointed 2026-08-12 (u5): the render-time producer died; the LIVE
    # namer is `identity.stage_token` from each stage's place in the FULL
    # ladder (prep._token_for), so the scan runs over exactly the names
    # `prep` writes.
    from molbuilder.identity import stage_token
    from molbuilder.siesta.stages import default_siesta_stages
    decks = [f"JOB_{stage_token(i, st.name)}.fdf"
             for i, st in enumerate(default_siesta_stages(), start=1)
             if st.enabled]
    positional_decks = sorted(n for n in decks if _POSITIONAL.search(n))
    if positional_decks:
        offences.append(
            f"the live namer emits {positional_decks} -- the deck "
            "and its log would then key on different things")

    # The flat runner had a probe here until 2026-08-10; P5 unit 3 deleted the
    # function, so the probe could only ever return None.  What it guarded --
    # a positional name reaching an output redirect -- is now covered by the
    # deck scan above, because flat and hierarchical emit the SAME decks
    # (decision 29) and there is no second namer left to disagree with them.

    viewer = STATIC / "structure-optimization" / "viewer.js"
    src = viewer.read_text(encoding="utf-8", errors="replace")
    # The deck-name suffix, as a template literal appended to a basename.
    if re.search(r"`-stage\$\{", src):
        offences.append(
            "structure-optimization/viewer.js builds `-stage${N}` as a deck "
            "filename suffix from a preset dropdown, discarding names it "
            "already has (its presets are coarse/medium/tight)")

    return offences


# Was ``xfail(strict=True)`` until 2026-08-10 with the reason: "P4 -- one naming
# rule.  Two producers key a filename on a stage's position:
# molwatch_log_basename and the structure-optimization tab."  P4 units 2-4
# removed both, so the marker came off and this guards for real.
def test_no_emitted_name_is_keyed_on_a_stage_position():
    """Question 2."""
    offences = measure_positional_names()
    assert not offences, "\n  ".join(["positional filenames:"] + offences)


# ===================================================================== #
#  Question 3 -- does everything run through the wrapper?               #
# ===================================================================== #
#
# job-system.md decision #2: "reuse the single-job wrapper unchanged ... so
# everything true of a single run is automatically true of every job in a
# batch", and running-a-job.md § 2.2a: "bash is a bootstrap, not a program."
#
# The flat ladder's runner is a program.  Architecture § 8b measured it by
# grepping the emitted text for any sign of the wrapper and finding none;
# this renders the script for real and asks the same question.
#
# SCOPE, so the number is not read as more than it is: this measures the one
# generated script that LAUNCHES work outside the wrapper.  runwrap's own
# script invokes the engine because it *is* the wrapper, and PySCF's emitted
# .py runs inside it.  Proving there is no second offender belongs to P5,
# which is the phase that owns what a producer may emit.

# "run.sh" is NOT a marker (B-3, 2026-08-13): merely writing the
# wrapper's FILENAME somewhere in a module said nothing about the
# invocation arriving with activation and the monitor around it, and it
# exempted the whole module from the question.
WRAPPER_MARKERS = ("conda activate", "source activate", "module load",
                   "mb_monitor", "molwatch")

# The invocation may arrive behind a launcher (B-3: `mpirun -np 4
# siesta < deck` was invisible to the bare form, and every parallel
# invocation is spelled that way).
ENGINE_INVOCATION = re.compile(
    r"^\s*(?:if\s+!\s+)?"
    r"(?:\S*(?:mpirun|srun|exec)\s+(?:\S+\s+)*?)?"
    r"(\S*siesta)\s*<", re.M)


def test_question_3_sees_a_launcher_prefixed_invocation():
    """The regex's own mutation guard: the spellings a module could emit."""
    assert ENGINE_INVOCATION.search("siesta < in.fdf > out.out")
    assert ENGINE_INVOCATION.search("if ! siesta < in.fdf; then")
    assert ENGINE_INVOCATION.search("mpirun -np 4 siesta < in.fdf")
    assert ENGINE_INVOCATION.search("mpirun -np 4 --bind-to core siesta < a")
    assert ENGINE_INVOCATION.search("srun siesta < in.fdf")
    assert not ENGINE_INVOCATION.search("# siesta < deck is the shape")


#: The ONE module allowed to emit an engine invocation.  It is the wrapper
#: generator, so the invocation it writes arrives with activation, the rank
#: clamp, the monitor and the run index around it -- which is the entire
#: content of question 3.
ENGINE_EMITTER = "runwrap.py"


def measure_direct_engine_invocations() -> list[str]:
    """Any module that writes ``siesta <`` into a script, other than the
    wrapper generator.

    **Widened 2026-08-10, when the old probe went vacuous.** It rendered
    `render_siesta_stages_runner` and inspected the text; P5 deleted that
    function, so the probe found nothing and the question passed *because
    there was nothing left to ask*. A guard that can no longer fail is worse
    than no guard, because it reads as coverage.

    So it now scans the package: a module emitting the engine directly is one
    whose output has no activation, no rank clamp and no monitor -- and
    therefore no `.molwatch.log` for the Results tab or the trajectory
    viewer -- whether or not that module is the one this file was written
    against.
    """
    offences = []
    for path in sorted(PKG.rglob("*.py")):
        if path.name == ENGINE_EMITTER:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not ENGINE_INVOCATION.search(text):
            continue
        if any(m in text for m in WRAPPER_MARKERS):
            continue
        offences.append(
            f"{path.relative_to(REPO)}: emits an engine invocation and shows "
            "no sign of the wrapper -- no environment activation (so `siesta` "
            "must already be on PATH, and it lives in a conda env), no rank "
            "clamp, no monitor, so no .molwatch.log for the Results tab or "
            "the trajectory viewer")
    return offences


# QUESTION 3 IS ANSWERED YES, 2026-08-10.  This was xfail(strict) from the day
# the module was written: the flat ladder runner called `siesta < $fdf` with no
# activation, no rank clamp, no monitor and no .molwatch.log.  P5 unit 3 deleted
# it (decision 29 -- flat runs through `jobset prep` / `submit run --chain` like
# the hierarchy), the probe below found nothing left to complain about, and the
# strict marker did exactly what it is for: it FAILED, loudly, the moment the
# behaviour started working, so the fix could not land without this being said.
def test_no_generated_script_invokes_an_engine_directly():
    """Question 3."""
    offences = measure_direct_engine_invocations()
    assert not offences, "\n  ".join(["direct engine invocation:"] + offences)


# ===================================================================== #
#  Question 4 -- does each stage start because someone said so?         #
# ===================================================================== #
#
# engines/stages.md § 7.1: stages do not chain.  An attempt is immutable and
# somebody decides to run the next one.  Two producers chain today, and they
# are owned by different phases: P5 deletes the flat runner (taking its loop
# with it), and P7 is where `submit` stops wiring the ladder together.  The
# guard names P7 because that is where the QUESTION is answered -- half of it
# going away in P5 does not turn it green.


def measure_chaining_producers() -> list[str]:
    """Question 4, asked of what a produced job actually says.

    A produced job may name **itself** (`name`) and **its own deck**
    (`script`), and may carry opaque `traits`.  Anything else whose value is
    another job's name is that job waiting on another one, whatever the field
    is called -- so the check is on the VALUE, not on a list of field names.
    That is what makes it survive a rename, and what keeps a retired
    vocabulary out of this file.
    """
    import dataclasses
    offences = []

    # Repointed 2026-08-12 (u5) to the LIVE construction: each enabled
    # stage through the one seam, jobs built the way `prep._job_for`
    # builds them.
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.identity import stage_token
    from molbuilder.jobset.model import Job, JobSet, Resources
    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import (_traits, _warm_declaration,
                                          default_siesta_stages)
    jobs = []
    for i, st in enumerate(default_siesta_stages(), start=1):
        if not st.enabled:
            continue
        eff = effective_config(SiestaConfig(system_label="JOB"),
                               getattr(st, "overrides", None),
                               where=f"stage {st.name!r}")
        jobs.append(Job(name=st.name,
                        script=f"JOB_{stage_token(i, st.name)}.fdf",
                        resources=Resources(),
                        warm=_warm_declaration("JOB", eff),
                        traits=_traits(eff)))
    js = JobSet(name="JOB", engine="siesta", kind="ladder", jobs=jobs)
    names = {j.name for j in js.jobs}
    for j in js.jobs:
        for f in dataclasses.fields(j):
            if f.name in ("name", "script", "traits"):
                continue
            val = getattr(j, f.name)
            found = [o for o in names if o != j.name and o == val]
            if found:
                offences.append(
                    f"live construction: job {j.name!r} field {f.name!r} names "
                    f"another job ({found}) -- that job is waiting on one")

    # A stage says WHAT it would warm-start from; the run is named by a person
    # at prep.  A declaration that also named the source would be an edge.
    for j in js.jobs:
        for w in j.warm:
            extra = ({f.name for f in dataclasses.fields(w)}
                     - {"name", "requires_same"})
            if extra:
                offences.append(
                    f"live construction: {j.name!r} declares a warm file "
                    f"carrying {sorted(extra)} -- a warm declaration is a "
                    "filename and its condition, nothing else")

    return offences


# QUESTION 4 IS ANSWERED YES, 2026-08-10.  This was xfail(strict) from the day
# the module was written.  P5 unit 3 removed one of the two chaining producers
# with the flat runner; P7 unit 2 removed the other -- `stages_to_jobset` now
# emits neither `depends_on` nor `Carry`, so nothing starts a stage but a
# person.  The strict marker did its job again: it FAILED the moment the
# behaviour started working, so the fix could not land without this being said.
def test_stages_do_not_chain():
    """Question 4."""
    offences = measure_chaining_producers()
    assert not offences, "\n  ".join(["stages chained:"] + offences)


# ===================================================================== #
#  The baseline, as one command                                         #
# ===================================================================== #

def baseline() -> list[tuple[str, str, int, str, str]]:
    """(question, measure, count, target, owning phase)."""
    return [
        # Q1's owning phase moved P5 -> P2 when the plan's *Subtracts* was
        # corrected: deleting the model belongs to the model phase, and P5
        # owns the emitted SHAPE.  P2 has taken mechanism 5; the rest of the
        # set is a grammar decision and stays P9's.
        ("Q1", 'ways to say "stage"',
         len(measure_mechanisms()), "the agreed set", "P2/P9"),
        ("Q2", "emitted names keyed on a position",
         len(measure_positional_names()), "0", "P4"),
        ("Q3", "generated scripts invoking an engine",
         len(measure_direct_engine_invocations()), "0", "P5"),
        ("Q4", "producers that chain stages",
         len(measure_chaining_producers()), "0", "P7"),
    ]


if __name__ == "__main__":  # pragma: no cover -- the reporting surface
    rows = baseline()
    print("staged-run baseline -- engines/stages.md SS 1, 7.3 R5, 8")
    print()
    for q, measure, count, target, phase in rows:
        flag = "!!" if target == "0" and count else "  "
        print(f" {flag} {q}  {measure:<38s} {count:>3d}   "
              f"target {target:<16s} {phase}")
    print()
    for line in measure_mechanisms():
        print(f"      mechanism {line}")
