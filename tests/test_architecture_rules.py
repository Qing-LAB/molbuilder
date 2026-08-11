"""The § 9.6 rules that had no test — A1, A4 and A7.

``docs/execution/staged-runs-implementation-plan.md`` § 9.6 opens by saying that
**a rule nobody checks is a wish**, and then listed three of its seven rules with
*(no test)* or *measured by hand* in the "checked by" column.  This file is the
answer to that.  Each test below names its rule, states what it can see, and --
just as important -- states what it cannot.

Why these three and not the other four
======================================

A2, A3, A5 and A6 are about *values flowing through the code at run time*, so
``test_jobset`` and friends can check them by running the code and looking at
what came out.  A1, A4 and A7 are about **the shape of the source itself** --
who is allowed to spell a name, who is allowed to build an object, who is
allowed to import whom.  No amount of running the program shows you those; you
have to read the source.  That is why they went unchecked for so long, and why
they all live here, in one file that parses ``molbuilder/`` rather than calling
it.

The failure they share
======================

All three are the § 9 diagnosis in different clothes: *somebody worked out an
answer that another part of the system already had*.  A1 is that habit applied
to a **name**, A4 to an **object**, A7 to a **decision**.  Catching them needs
the same tool, so they are tested together.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent / "molbuilder"
#: The contract this map must agree with.  It moved out of the implementation
#: plan on 2026-08-10: a plan may describe what does not exist yet, and this
#: design describes what IS, so it is a contract.
_CONTRACT = (Path(__file__).resolve().parent.parent
             / "docs" / "execution" / "architecture.md")


def _python_files() -> list[Path]:
    """Every ``.py`` under ``molbuilder/``, relative to it."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(_PKG):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        out += [Path(dirpath, fn).relative_to(_PKG)
                for fn in filenames if fn.endswith(".py")]
    return sorted(out)


# ===================================================================== #
#  A1 -- one namer                                                      #
# ===================================================================== #

#: How a stage token is spelled, and the only spelling that counts as one.
#: ``identity.stage_token`` is ``f"{int(seq):02d}_{name}"``; these patterns are
#: the ways that same string gets re-assembled somewhere else.  The guard is
#: deliberately about the ZERO-PADDED ordinal joined to a name, not about any
#: ``f"{a}_{b}"`` -- the padding is what makes it a token rather than a coincidence,
#: so this catches the real thing without shouting at every underscore.
_HAND_BUILT_TOKEN = re.compile(
    r""":0\d+d\}_        # f-string:   f"{seq:02d}_{name}"
      | %0\d+d_          # %-format:   "%02d_%s" % (seq, name)
      | \.zfill\(\s*2\s*\)   # str build: str(seq).zfill(2) + "_" + name
    """,
    re.VERBOSE,
)

#: The one module allowed to spell it.  Not a list that grows: the whole point
#: of A1 is that there is exactly one namer, so a second entry here would be
#: the rule being repealed rather than an exception being granted.
_THE_NAMER = Path("identity.py")


def test_a1_a_stage_token_is_spelled_in_exactly_one_place():
    """**A1 -- one namer.** Only ``identity.py`` builds ``<NN>_<name>``.

    ``run-identity.md`` § 3 rule 1 is that a name is normalised **once**, and
    the reason is not tidiness: the CLI, the web tab and the codec must all
    reach the same normaliser, or the id a surface shows is not the id the
    engine wrote.  A second place that assembles the same string is a second
    normaliser whether or not anyone calls it that.

    **What this cannot see.**  Someone who builds the token by a route this
    regex does not know -- string concatenation in a loop, say -- passes.  The
    guard covers the three spellings a person actually reaches for; it is a
    fence, not a proof.
    """
    offenders = []
    for rel in _python_files():
        if rel == _THE_NAMER:
            continue
        src = (_PKG / rel).read_text(encoding="utf-8")
        for m in _HAND_BUILT_TOKEN.finditer(src):
            line = src.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{line}  {m.group(0).strip()}")

    assert not offenders, (
        "these build a stage token by hand instead of asking "
        "`identity.stage_token` for one (§ 9.6 rule A1):\n  "
        + "\n  ".join(offenders))


# ===================================================================== #
#  A4 -- ask, do not work it out again                                  #
# ===================================================================== #

#: Where a ``StageRef`` may be created.  ``identity.py`` owns the class, and
#: ``jobset/materialize.py`` holds ``stage_refs``, the resolver that reads each
#: stage's ordinal back off its deck.  Anywhere else is a caller re-deriving
#: what one of those two already worked out.
_STAGEREF_OWNERS = {Path("identity.py"), Path("jobset/materialize.py")}


def test_a4_a_stage_ref_is_built_only_by_its_resolver():
    """**A4 -- one owning function per object**, for ``StageRef``.

    § 9.6 measured this rule by hand on 2026-08-10 and found exactly one
    violation: ``runstatus.py`` held a stage's name and number as two loose
    fields, so ``render_stage_status`` built a **second** ``StageRef`` out of
    them just to print the heading ``01_coarse``.  A caller working out an
    answer a floor below already held -- inside the very object created to stop
    that.  ``StageStatus`` now carries the ref whole, and this test is what
    keeps it that way.

    **What this cannot see.**  It checks one of the four objects in § 9.5.
    ``Attempt``, ``Shape`` and ``LaunchAgreement`` are each returned by one
    function today, but they are dataclasses a caller could construct too; a
    general version of this test would need a rule about which of those are
    meant to be constructible.  Extending it is a judgement, not a copy-paste,
    so it is left undone deliberately rather than half-done.
    """
    offenders = []
    for rel in _python_files():
        if rel in _STAGEREF_OWNERS:
            continue
        tree = ast.parse((_PKG / rel).read_text(encoding="utf-8"),
                         filename=str(rel))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "StageRef"):
                offenders.append(f"{rel}:{node.lineno}")

    assert not offenders, (
        "these build a StageRef instead of asking `stage_refs` for the one it "
        "already made (§ 9.6 rule A4):\n  " + "\n  ".join(offenders))


# ===================================================================== #
#  A7 -- nothing depends upwards                                        #
# ===================================================================== #
#
#  This is NOT `tests/test_layering.py` again.  That test checks *import
#  depth* -- L1 / L2 / L3, a coarser grouping in which the whole of `jobset`
#  is one tier.  It therefore cannot see `submit.py` (floor 5) importing
#  `runstatus.py` (floor 6): both are `L2`, so it has nothing to compare.
#  `jobset` alone spans four architectural floors inside that single tier,
#  which is exactly where an upward import would hide.

#: `execution/architecture.md` § 2.1's table, as code.  Keys are paths under
#: ``molbuilder/``; a key ending
#: in ``/`` covers a directory.  A file with no entry has no architectural
#: floor and is not judged -- most of the package is domain code that § 9's
#: stack says nothing about.
_FLOOR = {
    "identity.py":            1,
    "environment.py":         1,
    "persist.py":             1,
    "task.py":                2,
    "siesta/stages.py":       3,
    "bench/to_jobset.py":     3,
    "jobset/model.py":        3,
    "jobset/materialize.py":  4,
    "jobset/shape.py":        4,
    "jobset/submit.py":       5,
    "jobset/prep.py":         5,
    "runwrap.py":             5,
    "jobset/runstatus.py":    6,
    "parse/dirs/":            6,
    "cli.py":                 7,
    "jobset/_cli.py":         7,
    "web/":                   7,
}


def _floor_of(rel: str | Path) -> int | None:
    """The floor of a path under ``molbuilder/``, or ``None`` if unmapped."""
    s = str(rel).replace(os.sep, "/")
    if s in _FLOOR:
        return _FLOOR[s]
    for key, floor in _FLOOR.items():
        if key.endswith("/") and s.startswith(key):
            return floor
    return None


def _imported_modules(rel: Path, tree: ast.AST) -> set[str]:
    """Every intra-package import in ``rel``, as a path under ``molbuilder/``.

    Relative imports are resolved against the importing file's own package, so
    ``from ..identity import StageRef`` inside ``jobset/submit.py`` comes back
    as ``identity.py``.  Both the module and each imported *name* are offered
    as candidates, because ``from . import prep`` names its module in the
    alias, not in the module field -- and that is the form an upward import
    would most naturally take.
    """
    pkg = rel.parts[:-1]                      # ("jobset",) for jobset/submit.py
    out: set[str] = set()

    def add(parts: tuple[str, ...]) -> None:
        if parts:
            out.add("/".join(parts) + ".py")
            out.add("/".join(parts) + "/")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("molbuilder."):
                    add(tuple(alias.name.split(".")[1:]))
        elif isinstance(node, ast.ImportFrom):
            if node.level:                    # relative: from ..x import y
                base = pkg[:len(pkg) - (node.level - 1)]
            elif (node.module or "").startswith("molbuilder"):
                base = ()                     # absolute: from molbuilder.x ...
            else:
                continue                      # stdlib / third party
            mod = (node.module or "")
            if not node.level:
                mod = mod.split(".", 1)[1] if "." in mod else ""
            parts = base + tuple(p for p in mod.split(".") if p)
            add(parts)
            for alias in node.names:          # from . import prep
                add(parts + (alias.name,))
    return out


@pytest.mark.parametrize(
    "rel", [p for p in _python_files() if _floor_of(p) is not None],
    ids=lambda p: str(p))
def test_a7_nothing_imports_from_a_higher_floor(rel: Path):
    """**A7 -- nothing depends upwards.**

    § 9.3's rule is *a layer may call down and return up; it may never reach
    across*.  In import terms that is: a file on floor N may import files on
    floor N or below, never above.  Same-floor imports are allowed and normal
    -- ``submit`` (5) asking ``prep`` (5) for the launch agreement is two files
    on one floor sharing one answer, which is the opposite of the failure this
    rule is about.

    Why an upward import is the serious kind of mistake: floor 6 *observes*.
    The moment floor 5 imports it, launching depends on reading, and the
    question *what happened* has to be answerable before the thing has
    happened.  The cycle usually shows up first as an import error at startup,
    a long way from the line that caused it.

    **What this cannot see.**  A file with no entry in ``_FLOOR`` is not
    judged, and reaching a higher floor *through* an unmapped one would pass.
    That is the price of a map maintained by hand; the companion test below at
    least keeps the map honest about the doc.
    """
    mine = _floor_of(rel)
    tree = ast.parse((_PKG / rel).read_text(encoding="utf-8"), filename=str(rel))

    bad = []
    for target in _imported_modules(rel, tree):
        theirs = _floor_of(target)
        if theirs is not None and theirs > mine:
            bad.append(f"{target.rstrip('/')} (floor {theirs})")

    assert not bad, (
        f"{rel} is on floor {mine} and imports upward: {sorted(set(bad))}.  "
        "Either move the import down, or change § 9.3's table and this map "
        "together (§ 9.6 rule A7)."
    )


def test_a7_the_floor_map_still_matches_the_document():
    """The map above and the contract's floor table must name the same files.

    Without this the test passes forever while the design says something else
    -- precisely the drift the contract exists to prevent.  The table is
    located by its heading TEXT rather than a section number, because numbers
    move: this guard already had to follow the design from the implementation
    plan into `execution/architecture.md`.
    """
    text = _CONTRACT.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"^#+ .*seven floors\s*$", l, re.I)), None)
    assert start is not None, (
        "cannot find the floor table's heading (a line matching "
        f"'... seven floors') in {_CONTRACT.name} -- it was renamed, so this "
        "guard is now checking nothing.  Repoint it.")

    documented: set[str] = set()
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        cells = [c.strip().strip("*") for c in line.split("|")]
        # A floor row: | N | name | decision | files | entry points | ...
        # The leading empty cell from the opening pipe makes the number
        # cells[1], and the files column cells[4].
        if len(cells) > 5 and cells[1].isdigit():
            documented |= set(re.findall(r"`([^`]+)`", cells[4]))

    mapped = {k.rstrip("/").removesuffix(".py") for k in _FLOOR}
    assert documented == mapped, (
        "§ 9.3's table and the floor map in this file disagree.\n"
        f"  in the doc, not in the map: {sorted(documented - mapped)}\n"
        f"  in the map, not in the doc: {sorted(mapped - documented)}")


def test_the_floor_map_names_only_real_paths():
    """A stale entry exempts nothing and outlives the rename that broke it."""
    missing = sorted(k for k in _FLOOR if not (_PKG / k).exists())
    assert not missing, (
        f"the floor map names paths that are not in the tree: {missing}")
