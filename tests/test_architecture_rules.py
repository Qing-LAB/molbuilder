"""The architecture rules that had no test — A1, A4 and A7.

``docs/execution/architecture.md`` § 7 opens by saying that
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

All three are one diagnosis in different clothes: *somebody worked out an
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

#: How a stage token is spelled: ``identity.stage_token`` is
#: ``f"{int(seq):02d}_{name}"``.  The patterns are the ways that same string
#: gets assembled, so "does this file spell a token?" is answerable mechanically.
_SPELLS_A_TOKEN = re.compile(
    r""":0\d+d\}_        # f-string:   f"{seq:02d}_{name}"
      | %0\d+d_          # %-format:   "%02d_%s" % (seq, name)
      | \.zfill\(\s*2\s*\)   # str build: str(seq).zfill(2) + "_" + name
    """,
    re.VERBOSE,
)

#: A1's rule, as the set it permits: **one namer**.
_THE_NAMERS = {Path("identity.py")}


def test_a1_exactly_one_module_spells_a_stage_token():
    """**A1 — one namer**, stated as the set of files allowed to be one.

    `run-identity.md` § 3 rule 1: a name is tidied **once**.  The reason is not
    tidiness — the CLI, the web tab and the codec must all reach the same
    normaliser, or the id a surface shows is not the id the engine wrote.  A
    second place that assembles the same string is a second normaliser whether
    or not anyone calls it that.

    **An equality, not a hunt for offenders.** The set of modules that spell a
    token must BE ``{identity.py}`` — so a second speller fails, and so does
    `identity.py` quietly losing the ability to spell one (which would mean the
    namer moved and this guard had stopped watching anything).

    **What it cannot see:** a token assembled by a route these three patterns
    do not describe.  It covers the spellings a person actually reaches for; it
    is a fence, not a proof.
    """
    spellers = {rel for rel in _python_files()
                if _SPELLS_A_TOKEN.search((_PKG / rel).read_text(encoding="utf-8"))}
    assert spellers == _THE_NAMERS, (
        "A1 says exactly one module spells `<NN>_<name>`.\n"
        f"  also spelling one: {sorted(str(p) for p in spellers - _THE_NAMERS)}\n"
        f"  expected to, but does not: "
        f"{sorted(str(p) for p in _THE_NAMERS - spellers)}")


# ===================================================================== #
#  A4 -- ask, do not work it out again                                  #
# ===================================================================== #

#: Every § 3 object and the ONE function allowed to return it.  Measured, not
#: assumed: ``Shape`` is never constructed directly (callers use ``Shape.named``),
#: ``Attempt`` is built once inside ``prepare_attempt``, and
#: ``LaunchAgreement``'s three constructions are three ``return`` lines in one
#: function -- which is still one owner.  ``None`` means "only inside its own
#: class definition", the classmethod case.
_ONE_BUILDER = {
    "StageRef":        ("jobset/materialize.py", "stage_refs"),
    "Attempt":         ("jobset/materialize.py", "prepare_attempt"),
    "LaunchAgreement": ("jobset/agreement.py",   "launch_agreement"),
    "Shape":           ("jobset/shape.py",       None),
}


def test_a4_each_object_is_built_in_exactly_one_function():
    """**A4 — ask, do not work it out again.**

    Each object in `execution/architecture.md` § 3 answers one question once.
    A second construction site is a second answer, and stopping a caller from
    working out something a floor below already knows is the entire reason
    these objects were named.

    **The unit of the rule is the FUNCTION, not the module.** The owning module
    legitimately builds its own object; what must not happen is a *second*
    function building one — including inside the same file, which a
    module-level rule would wave through.

    The measurement behind this found the violation it now prevents:
    ``runstatus`` built a second ``StageRef`` to format a heading, because
    ``StageStatus`` carried the number and the name as loose fields instead of
    the ref the resolver had already made.

    **The class's own METHODS may build it** — that is how `Shape.named` works,
    and a classmethod returning an instance of its own class is the builder,
    not a second one. **Living in the same FILE is not enough**: `Attempt` is
    defined in `materialize.py`, so a file-level exemption would let every
    function in that module build one, which is exactly the rule this test
    exists to enforce. (It did, briefly, on 2026-08-11 — the two A4 tests were
    merged and the file-level clause came with them. The mutation that had been
    killed before the merge survived after it, which is why the merge is
    re-mutated rather than assumed.)
    """
    import ast
    offences = []
    for cls, (owner_rel, owner_fn) in _ONE_BUILDER.items():
        for rel in _python_files():
            text = (_PKG / rel).read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(rel))
            # the class's OWN methods -- not merely "this file defines it"
            own_methods = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef) and n.name == cls:
                    own_methods |= {
                        m for m in ast.walk(n)
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for fn in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                builds = [c.lineno for c in ast.walk(fn)
                          if isinstance(c, ast.Call)
                          and getattr(c.func, "id", "") == cls]
                if not builds:
                    continue
                here = str(rel).replace("\\", "/")
                allowed = ((here == owner_rel and fn.name == owner_fn)
                           or fn in own_methods)
                if not allowed:
                    offences.append(
                        f"{rel}:{min(builds)} {fn.name}() builds a {cls} — only "
                        f"{owner_rel}::{owner_fn or 'its own classmethod'} may")

    assert not offences, (
        "§ 3 gives each object exactly one builder (rule A4):\n  "
        + "\n  ".join(offences))


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
    "resolve.py":             3,
    "jobset/model.py":        3,
    "jobset/materialize.py":  4,
    "jobset/shape.py":        4,
    "jobset/submit.py":       5,
    "jobset/prep.py":         5,
    "jobset/agreement.py":    5,
    "runwrap.py":             5,
    "jobset/runstatus.py":    6,
    "parse/dirs/":            6,
    "cli.py":                 7,
    "jobset/_cli.py":         7,
    "jobset/ledger.py":       7,
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

    `execution/architecture.md` § 2.1's rule is *a floor may call down and
    return up; it may never reach
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
        "Either move the import down, or change § 2.1's floor table and this "
        "map together (§ 7 rule A7)."
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
        "§ 2.1's floor table and the floor map in this file disagree.\n"
        f"  in the doc, not in the map: {sorted(documented - mapped)}\n"
        f"  in the map, not in the doc: {sorted(mapped - documented)}")


def test_the_floor_map_names_only_real_paths():
    """A stale entry exempts nothing and outlives the rename that broke it."""
    missing = sorted(k for k in _FLOOR if not (_PKG / k).exists())
    assert not missing, (
        f"the floor map names paths that are not in the tree: {missing}")


# ===================================================================== #
#  The config map cannot go stale — P12 unit 1                          #
# ===================================================================== #
#
#  `execution/architecture.md` § 8.2 tabulates every config section, who reads
#  it, and where it lands.  Two documents used to hold PARTIAL tables --
#  deployment.md six, running-a-job.md four -- and neither said so, which is
#  how a reader concluded molbuilder.json is smaller than it is.
#
#  Writing this guard found three errors in that table on its first run:
#  `providers` is `auth.providers` and not a section, `admin_emails` is the
#  GETTER while the section is `admin`, and the count was 11 rather than 10.

_CONFIG_DOC_SECTION = "## 8. Configuration"


def _sections_the_code_reads() -> set[str]:
    """Every top-level key of ``molbuilder.json`` the reader looks at.

    Two populations, and the difference is worth keeping: what ``_normalise``
    validates when the file is loaded, and what a getter reads on demand later.
    Both are config sections; only the first is refused when malformed.
    """
    import inspect
    from molbuilder import runtime_config as rc
    validated = set(re.findall(r'out\["([a-z_]+)"\]',
                               inspect.getsource(rc._normalise)))
    lazy = set()
    for name in dir(rc):
        if not name.startswith("get_"):
            continue
        try:
            src = inspect.getsource(getattr(rc, name))
        except (TypeError, OSError):
            continue
        lazy |= set(re.findall(r'cfg\.get\(\s*"([a-z_]+)"', src))
        lazy |= set(re.findall(r'cfg\[\s*"([a-z_]+)"\s*\]', src))
    return validated | lazy


def _sections_the_contract_documents() -> set[str]:
    """The first column of § 8.2's table."""
    text = _CONTRACT.read_text(encoding="utf-8")
    start = text.index(_CONFIG_DOC_SECTION)
    end = text.index("\n## ", start + 1)
    out = set()
    for line in text[start:end].splitlines():
        cells = [c.strip() for c in line.split("|")]
        # a section row: | `name` | read by | reaches | what it decides |
        if len(cells) > 4 and cells[1].startswith("`") and cells[1].endswith("`"):
            out.add(cells[1].strip("`"))
    return out


def test_every_config_section_is_documented_and_every_documented_one_exists():
    """§ 8.2's table and the reader name the same sections.

    An equality in both directions: a section added to `runtime_config` without
    a row fails, and a row whose section was deleted fails too.  Either way the
    map and the code cannot drift apart silently, which is what let two guides
    each publish a different subset of one file.
    """
    code = _sections_the_code_reads()
    doc = _sections_the_contract_documents()
    assert code == doc, (
        "execution/architecture.md § 8.2 and molbuilder/runtime_config.py "
        "disagree about which config sections exist.\n"
        f"  read by the code, not in the table: {sorted(code - doc)}\n"
        f"  in the table, not read by the code: {sorted(doc - code)}")
