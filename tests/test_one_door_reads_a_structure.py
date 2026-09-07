"""One door reads a structure from disk, and it is `StructureCodec`.

WHAT THIS IS FOR, in one sentence: a `.xyz` and the `.molstruct.json` beside
it are one file, and a second piece of code that reads them is a second answer
that will drift.

**It did.** On 2026-09-07 four readers of that pair existed. The one a person
was most likely to reach for -- `molbuilder.load()`, the obvious name, in
`__all__`, docstring accurate for what it did -- read the geometry and not the
sidecar. It predated the sidecar by two months and nothing swept it. Its one
production caller was `jobset init`, the verb that turns a structure into a
calculation (`job-system.md` § 5.1), so a description was born with the
author's regions, frozen atoms and cell missing, and everything downstream
was then faithfully correct about the wrong thing. It is the same failure
`siesta/input.py` records having fixed at its own door -- *"the script relaxed
every atom of a structure whose author had frozen two."*

**A hand-written list of doors would not have caught it**, which is why this
guard reads the code. `load()` would have been ON such a list, looking
entirely reasonable.

THE RULE, and the two things it is not:

  * Reading a PATH and getting a `Structure` goes through `StructureCodec`.
  * Parsing TEXT is not this. The browser posts bytes, OpenBabel returns a
    PDB string; there is no file, so there is no sidecar to miss.
  * A `FileParser` in the parse registry reading an ENGINE's output is not
    this either. `siesta.XV`, `*_optimized.xyz` and friends are somebody
    else's format, not a molbuilder pair.

Shaped after `test_one_home_for_a_constant.py`, including its best idea: an
allowance carries the reason it was granted, and the guard fails when an
allowed site stops doing the thing it was allowed for -- so an exemption
cannot outlive its argument.
"""
from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "molbuilder"

#: The door. Only this may turn a path into a Structure.
OWNER = "molbuilder/workingcopy_structure.py"

#: The low-level readers. Calling one of these directly is what the door
#: exists to be instead of.
READERS = ("from_xyz", "from_pdb")

#: file -> (how many calls, why they are allowed).  The COUNT is part of the
#: allowance: a file that grows a second call has not inherited the first
#: one's reason, and must come here and say its own.
ALLOWED: dict[str, tuple[int, str]] = {
    OWNER: (2, "the door itself -- .xyz and .pdb, one each"),

    "molbuilder/chemistry.py": (
        3, "TEXT from OpenBabel/RDKit, produced in this process. No file "
           "exists, so there is no sidecar to miss"),
    "molbuilder/web/blueprints/build.py": (
        5, "four are TEXT the browser posted (the load door's xyz/pdb "
           "branches and `_xyz_to_structure`). The fifth is "
           "`/api/structure/analyze`, which DOES read a path and skip the "
           "sidecar -- harmless today because the analyzer reads only "
           "element names, and listed here so it is a known exception "
           "rather than an undiscovered one"),
    "molbuilder/parse/coords/pyscf_geom.py": (
        1, "a FileParser reading an ENGINE's output (`*_optimized.xyz`). "
           "Not a molbuilder pair; the parse registry is its own contract"),
    "molbuilder/transport/_cli.py": (
        1, "`_load_device` reads a path and DOES apply the whole sidecar, "
           "through `apply_to_structure` -- correct, but its own copy of "
           "the walk. A candidate for the door; not a defect"),
}


def _calls_in(path: Path) -> list[int]:
    """Line numbers of every `X.from_xyz(...)` / `X.from_pdb(...)`."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in READERS):
            out.append(node.lineno)
    return out


def _survey() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for py in sorted(PKG.rglob("*.py")):
        lines = _calls_in(py)
        if lines:
            found[str(py.relative_to(PKG.parent))] = lines
    return found


def test_no_second_reader_of_a_structure_appears_without_saying_why():
    """A new caller of the low-level readers is a finding, not a detail."""
    found = _survey()
    strangers = sorted(set(found) - set(ALLOWED))
    assert not strangers, (
        "these call `Structure.from_xyz` / `from_pdb` directly, which is a "
        "second reader of a structure:\n  "
        + "\n  ".join(f"{f} (lines {found[f]})" for f in strangers)
        + "\n\nIf it reads a PATH, use `StructureCodec` -- it reads the "
          "`.molstruct.json` beside the geometry, which is where the regions, "
          "the frozen atoms and the cell live.  If it parses TEXT, or reads "
          "an engine's own output format, add it to ALLOWED with the reason."
    )


def test_an_allowed_file_has_not_quietly_grown_another():
    """The count is the allowance. A second call has its own reason to give.

    Without this, one exemption shelters every later call in the same file --
    which is how `/api/structure/analyze` would have slipped in beside the
    load door's legitimate text parsing.
    """
    found = _survey()
    drift = []
    for rel, (expected, why) in ALLOWED.items():
        actual = len(found.get(rel, []))
        if actual != expected:
            drift.append(f"{rel}: allowed {expected} ({why[:48]}...), "
                         f"found {actual} at {found.get(rel, [])}")
    assert not drift, (
        "an allowance no longer describes the file:\n  " + "\n  ".join(drift)
        + "\n\nA new call needs its own reason here; a removed one means the "
          "allowance can go."
    )


def test_the_guard_can_actually_see_a_violation():
    """A lint whose pattern never matches stays green over a regression."""
    src = "s = Structure.from_xyz(p)\nt = Struct.from_pdb(q)\n"
    tmp = PKG / "_guard_selftest.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        assert len(_calls_in(tmp)) == 2, (
            "the walker no longer finds a direct reader call -- it would be "
            "green over a file that reintroduced one")
        assert "molbuilder/_guard_selftest.py" in _survey(), (
            "the survey missed a file the walker can read")
    finally:
        tmp.unlink()
