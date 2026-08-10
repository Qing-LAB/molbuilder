"""The four questions that decide whether the staged-run design landed.

``docs/web/staged-runs-architecture.md`` § 8c states them as things a reader
should be able to *check*, not believe:

  1. Is there one way to say "stage"?
  2. Does a stage's name survive -- is no file named by a stage's *position*?
  3. Does everything run through the wrapper?
  4. Does each stage start because someone said so?

Three of the four answers are wrong today.  This module makes all four
mechanical, so a regression is caught by a test rather than by the next
fresh-eyes review, and so the subtraction reviews of
``docs/execution/staged-runs-implementation-plan.md`` have a number to point
at instead of an opinion.  The three that are wrong are ``xfail(strict=True)``
naming the phase that fixes each: strict matters, because it fails loudly the
day the behaviour *starts* working, so a fix cannot land without the plan
being updated.

Run it as a module, from the repository root, for the baseline table::

    python -m tests.test_stage_vocabulary

(as a module, not a path: the reporter imports ``molbuilder`` and the package
is not installed into the env -- see docs/process/testing.md.)

WHAT THE LEDGERS BELOW ARE FOR.  Question 1 is answered by attributing every
declaration in the package whose name carries "stage" to one of the ten
mechanisms ``staged-runs-architecture.md`` § 8b enumerates, or to a role that
is not a mechanism at all.  Removing a mechanism is then a one-line edit --
which is what makes phase P5's "these nine names are gone" provable rather
than intended -- and growing an eleventh fails until somebody writes down what
it is.  Same shape as ``test_css_no_duplicate_selectors.py``: enumerate
mechanically, attribute by hand, assert the two agree.

THE BLIND SPOT, STATED SO NOBODY TRUSTS THIS FURTHER THAN IT GOES.  Detecting
by name cannot find a mechanism that does not carry the word.
``molwatch_log_basename(label, stage=N)`` is exactly that: it takes a stage and
writes its number into a filename, and no detector here sees it.  Question 2
catches it instead -- which is the general answer, and the reason the plan's
Review 2 says to search *by behaviour, not by name*.
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
    "--stage": (
        1, "molbuilder/cli.py",
        "single-shot tier overlay; P5 retires the flag, the presets stay"),
    "apply_siesta_stage": (
        1, "molbuilder/config/siesta.py", "applies that overlay"),
    "SIESTA_STAGE_PRESETS": (
        1, "molbuilder/config/siesta.py",
        "the tier values -- real science, kept as the defaults a new stage "
        "is created with"),
    "--stage-strategy": (
        2, "molbuilder/cli.py", "named presets over the *enable* flags"),
    "SIESTA_STAGE_STRATEGY_PRESETS": (
        2, "molbuilder/config/siesta.py",
        "those presets -- pure enable-mask data now; the applier that turned "
        "them into SiestaStageSpec objects went with the class (P2)"),
    "apply_stage_strategy": (
        2, "molbuilder/config/pyscf.py", "the same road for PySCF"),
    "STAGE_STRATEGY_PRESETS": (
        2, "molbuilder/config/pyscf.py",
        "PySCF's own copy of the strategy presets"),
    "--stages-json": (
        3, "molbuilder/cli.py",
        "the whole ladder from a file; its help says 'Unknown keys ignored', "
        "which P1 reverses"),
    "stages_from_dicts": (
        3, ("molbuilder/config/pyscf.py", "molbuilder/task.py"),
        "parses it.  TWO declarations of one name, and that is the point: "
        "PySCF's builds its own StageSpec; task.py's builds the design's "
        "own Stage, and it is what --stages-json reaches now that the "
        "SIESTA parser is deleted (P2)"),
    "--stage-resources": (
        4, "molbuilder/cli.py", "per-stage scheduler asks, a *second* file"),
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
    "_stage_tokens": (
        None, "molbuilder/siesta/input.py",
        "pairs each ENABLED stage with its token, numbering from its place "
        "in the FULL ladder so disabling one leaves a gap rather than "
        "renumbering what follows (project-layout 4.2)"),
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
        "`tight` | `3` | `03` | `03_tight` -> one StageRef.  A NUMBER "
        "MATCHES seq AND NEVER A ROW, which is R5's whole point"),
    "render_stage_refs": (
        None, "molbuilder/identity.py",
        "the one listing format, so a refusal, a status table and a help "
        "string cannot show three vocabularies for one set of jobs"),
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
    "render_siesta_stage_fdfs": (
        6, "molbuilder/siesta/input.py", "flat -- one deck per enabled stage"),
    #  ``render_siesta_stages_runner`` stood here.  DELETED 2026-08-10 with the
    #  function, its 142-line bash template and its fourteen tests (P5 unit 3,
    #  decision 29).  The row is removed rather than commented: its ABSENCE is
    #  what proves the subtraction, and the guard below fails if the name comes
    #  back.  Mechanism 6 is now the deck renderer alone.
    "_enabled_stages": (
        6, "molbuilder/siesta/input.py", "what both of those iterate"),
    #  ``STAGES`` -- the bash array in the runner's template -- went with the
    #  template on 2026-08-10.  It was the only stage vocabulary that lived in
    #  GENERATED TEXT, which is where questions 3 and 4 used to look.
    "default_siesta_stages": (
        11, "molbuilder/siesta/stages.py",
        "builds the shipped ladder as task.Stage objects, reading the tier "
        "values from mechanism 1's preset table and the enable mask from "
        "mechanism 2's.  It produces the design's own stage rather than "
        "being a way of expressing one, which is why it is 11 and not a "
        "twelfth"),
    "stages_to_jobset": (
        7, "molbuilder/siesta/stages.py",
        "hierarchical -- a ladder JobSet, wired stage-to-stage"),
    "StageSpec": (
        8, "molbuilder/config/pyscf.py",
        "PySCF's ladder: one process, one file, an in-script loop -- "
        "genuinely a different shape, and it stays"),
    "_default_stages": (
        8, "molbuilder/config/pyscf.py", "its default ladder"),
    "validate_stages": (
        8, "molbuilder/config/pyscf.py", "its gate"),
    "_emit_stages_loop": (
        8, "molbuilder/pyscf/input.py", "emits that in-script loop"),
    # Mechanism 9 is a DOM id, so it lives in the JS ledger below.
    "_stagespec_to_field_schemas": (
        10, "molbuilder/web/blueprints/_shared.py",
        "the Python end of the generic stage-table: any List[<dataclass>] "
        "becomes a per-stage grid"),

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
    "_emit_siesta_multi_stage": (
        None, "molbuilder/cli.py", "the CLI's glue over 3, 4, 6 and 7"),
    "StageStatus": (
        None, "molbuilder/jobset/runstatus.py", "the observing side"),
    "_stage_state": (
        None, "molbuilder/jobset/runstatus.py", "the observing side"),
    "_detect_stage": (
        None, "molbuilder/parse/dirs/job.py",
        "the decoder reading a stage back OUT of an emitted name -- "
        "question 2's other end"),
    "StageBundle": (
        None, "molbuilder/siesta/stages.py", "the carrier the producer returns"),
    "build_siesta_stage_bundle": (
        None, "molbuilder/siesta/stages.py",
        "composition of 6 and 7, and since 2026-08-10 (P5 unit 1) it READS "
        "the shape and emits ONE layout: flat -> decks + the runner, "
        "hierarchical -> decks + the JobSet.  It used to call both and let "
        "whatever command you typed next settle it -- the code that decided "
        "the shape by not deciding it"),
    "check_identical_stages": (
        None, "molbuilder/validation/stages.py",
        "§ 6.6a's check that a stage does not recompute the one before it -- "
        "a question ABOUT a ladder, not a way of expressing one"),
    "DIAG_1STAGE": (
        None, "molbuilder/bench/__init__.py",
        "SIESTA *diagonalisation*, unrelated role -- same word"),
    "DIAG_2STAGE": (
        None, "molbuilder/bench/__init__.py", "likewise"),
}

# The JS side is attributed by FILE rather than by token: a hundred CSS class
# names inside one widget are one mechanism, and what actually matters is a
# *new file* growing stage machinery.
JS_LEDGER: dict[str, tuple[int | None, str]] = {
    "lib/form-schema.js": (
        10, "the generic stage-table field kind -- rows are the per-stage "
            "parameters, columns are the stages, which is the panel "
            "task-setup-plan.md § 6 describes"),
    "structure-optimization/viewer.js": (
        9, "p-stage-preset: a stage NUMBER, into a filename.  (Its "
           "ELPA-1STAGE is the diagonaliser, a different sense)"),
    "lib/molview/mount.js": (
        None, "molviewer-window-stage is a CSS layer -- unrelated role"),
}

#: Eleven on 2026-08-07 morning, and it went UP on purpose: P1 added
#: task.json's own stage without retiring anything, which the plan calls "the
#: one phase that deliberately raises the mechanism count".
#:
#: TEN since P2 unit 2 the same day: mechanism 5 -- SiestaStageSpec and its
#: two helpers -- is deleted.  That is the first subtraction of the program,
#: and the number falling is what makes it a fact rather than a claim.
MECHANISM_COUNT = 10


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
        + "\n\nP5 is where this number comes down.  Moving it means updating "
          "architecture § 8b and the plan's § 6 baseline in the same commit.")


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
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import render_siesta_stage_fdfs
    from molbuilder.structure import Structure
    import numpy as np
    h2 = Structure(elements=["H", "H"],
                   positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                   vacuum=(12.0, 12.0, 12.0))
    from molbuilder.siesta.stages import default_siesta_stages
    decks = render_siesta_stage_fdfs(h2, SiestaConfig(system_label="JOB"),
                                     default_siesta_stages())
    positional_decks = sorted(n for n in decks if _POSITIONAL.search(n))
    if positional_decks:
        offences.append(
            f"render_siesta_stage_fdfs emits {positional_decks} -- the deck "
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

WRAPPER_MARKERS = ("conda activate", "source activate", "module load",
                   "run.sh", "mb_monitor", "molwatch")

ENGINE_INVOCATION = re.compile(r"^\s*(?:if\s+!\s+)?(\S*siesta)\s*<", re.M)


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
    offences = []

    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import (default_siesta_stages,
                                          stages_to_jobset)
    js = stages_to_jobset(SiestaConfig(system_label="JOB"),
                          default_siesta_stages())
    chained = [j.name for j in js.jobs if j.depends_on is not None]
    if chained:
        offences.append(
            f"stages_to_jobset: {len(chained)} of {len(js.jobs)} jobs carry "
            f"depends_on ({', '.join(chained)}) -- the ladder is wired "
            "stage-to-stage, so the scheduler starts the next one, not a person")

    # A `Carry` is an edge too, and the more dangerous one: it points at a file
    # in another stage's directory before that stage has run, so a produced
    # tree holds dangling symlinks and whatever runs there writes THROUGH one.
    carried = [j.name for j in js.jobs if j.carry]
    if carried:
        offences.append(
            f"stages_to_jobset: {len(carried)} job(s) carry a Carry edge "
            f"({', '.join(carried)}) -- what a stage continues from is a real "
            "file copied in at prep, from the run you name")

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
    print("staged-run baseline -- docs/web/staged-runs-architecture.md § 8c")
    print()
    for q, measure, count, target, phase in rows:
        flag = "!!" if target == "0" and count else "  "
        print(f" {flag} {q}  {measure:<38s} {count:>3d}   "
              f"target {target:<16s} {phase}")
    print()
    for line in measure_mechanisms():
        print(f"      mechanism {line}")
