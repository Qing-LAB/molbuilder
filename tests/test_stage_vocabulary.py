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
    ("constant", re.compile(r'^([A-Z][A-Z0-9_]*STAGE[A-Z0-9_]*)\s*[:=]', re.M)),
)

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
        2, "molbuilder/config/siesta.py", "those presets"),
    "apply_siesta_stage_strategy": (
        2, "molbuilder/config/siesta.py", "applies them"),
    "apply_stage_strategy": (
        2, "molbuilder/config/pyscf.py", "the same road for PySCF"),
    "--stages-json": (
        3, "molbuilder/cli.py",
        "the whole ladder from a file; its help says 'Unknown keys ignored', "
        "which P1 reverses"),
    "siesta_stages_from_dicts": (
        3, "molbuilder/config/siesta.py", "parses it"),
    "stages_from_dicts": (
        3, "molbuilder/config/pyscf.py", "the same for PySCF"),
    "--stage-resources": (
        4, "molbuilder/cli.py", "per-stage scheduler asks, a *second* file"),
    "SiestaStageSpec": (
        5, "molbuilder/config/siesta.py",
        "the in-memory model 1-4 all feed; eight fields where the contract "
        "says three"),
    "_default_siesta_stages": (
        5, "molbuilder/config/siesta.py", "the ladder a fresh config carries"),
    "validate_siesta_stages": (
        5, "molbuilder/config/siesta.py", "its gate"),
    "render_siesta_stage_fdfs": (
        6, "molbuilder/siesta/input.py", "flat -- one deck per enabled stage"),
    "render_siesta_stages_runner": (
        6, "molbuilder/siesta/input.py",
        "flat -- the bash loop; questions 3 and 4 both point here"),
    "_enabled_stages": (
        6, "molbuilder/siesta/input.py", "what both of those iterate"),
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

    # -- not a way of expressing a stage -------------------------------
    "stage_completion_tag": (
        None, "molbuilder/checkpoint.py", "names a checkpoint tag"),
    "parse_stage_completion_tag": (
        None, "molbuilder/checkpoint.py", "reads one back"),
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
        "composition of 6 and 7 that calls BOTH -- flat decks and a flat "
        "runner, plus a hierarchical JobSet whenever emit_jobset, which "
        "defaults true.  Not an eleventh mechanism: the code that decides "
        "the shape by not deciding it.  P5"),
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
        for _kind, rx in _PY_DETECTORS:
            for m in rx.finditer(src):
                found.setdefault(m.group(1), set()).add(rel)
    return found


def detect_js_files() -> set[str]:
    """Static JS files mentioning a stage token, relative to static/."""
    rx = re.compile(r'["\'][a-z0-9-]*stage[a-z0-9-]*["\']', re.I)
    return {p.relative_to(STATIC).as_posix()
            for p in _js_sources()
            if rx.search(p.read_text(encoding="utf-8", errors="replace"))}


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

    moved = sorted(
        f"{s}: ledger says {PY_LEDGER[s][1]}, found in {', '.join(sorted(w))}"
        for s, w in found.items() if PY_LEDGER[s][1] not in w)
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
    decks = render_siesta_stage_fdfs(h2, SiestaConfig(system_label="JOB"))
    positional_decks = sorted(n for n in decks if _POSITIONAL.search(n))
    if positional_decks:
        offences.append(
            f"render_siesta_stage_fdfs emits {positional_decks} -- the deck "
            "and its log would then key on different things")

    runner = _flat_runner_text()
    if runner is not None and _POSITIONAL.search(runner):
        offences.append(
            "render_siesta_stages_runner writes a positional name into the "
            "outputs it redirects")

    viewer = STATIC / "structure-optimization" / "viewer.js"
    src = viewer.read_text(encoding="utf-8", errors="replace")
    # The deck-name suffix, as a template literal appended to a basename.
    if re.search(r"`-stage\$\{", src):
        offences.append(
            "structure-optimization/viewer.js builds `-stage${N}` as a deck "
            "filename suffix from a preset dropdown, discarding names it "
            "already has (its presets are coarse/medium/tight)")

    return offences


@pytest.mark.xfail(strict=True, reason=(
    "P4 -- one naming rule.  Two producers key a filename on a stage's "
    "position: molwatch_log_basename and the structure-optimization tab."))
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


def _flat_runner_text() -> str | None:
    """The rendered flat ladder runner, or None once P5 has deleted it."""
    try:
        from molbuilder.siesta.input import render_siesta_stages_runner
    except ImportError:
        return None
    from molbuilder.config.siesta import SiestaConfig
    return render_siesta_stages_runner(SiestaConfig(system_label="JOB"))


def measure_direct_engine_invocations() -> list[str]:
    script = _flat_runner_text()
    if script is None:
        return []
    if not ENGINE_INVOCATION.search(script):
        return []
    if any(m in script for m in WRAPPER_MARKERS):
        return []
    return ["render_siesta_stages_runner: invokes the engine directly and "
            "shows no sign of the wrapper -- no environment activation (so "
            "`siesta` must already be on PATH, and it lives in a conda env), "
            "no rank clamp, no monitor, so no .molwatch.log for the Results "
            "tab or the trajectory viewer"]


@pytest.mark.xfail(strict=True, reason=(
    "P5 -- one shape out.  The flat ladder runner calls `siesta < $fdf` with "
    "no activation, no wrapper and no monitor."))
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
    from molbuilder.siesta.stages import stages_to_jobset
    js = stages_to_jobset(SiestaConfig(system_label="JOB"))
    chained = [j.name for j in js.jobs if j.depends_on is not None]
    if chained:
        offences.append(
            f"stages_to_jobset: {len(chained)} of {len(js.jobs)} jobs carry "
            f"depends_on ({', '.join(chained)}) -- the ladder is wired "
            "stage-to-stage, so the scheduler starts the next one, not a person")

    script = _flat_runner_text()
    if script is not None and re.search(r'for\s+\w+\s+in\s+"\$\{!STAGES\[@\]\}"',
                                        script):
        offences.append(
            "render_siesta_stages_runner: the emitted bash loops over every "
            "stage in one invocation")

    return offences


@pytest.mark.xfail(strict=True, reason=(
    "P7 -- one attempt, no chain.  stages_to_jobset wires each stage to its "
    "predecessor and the flat runner loops over all of them."))
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
        ("Q1", 'ways to say "stage"',
         len(measure_mechanisms()), "the agreed set", "P5"),
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
