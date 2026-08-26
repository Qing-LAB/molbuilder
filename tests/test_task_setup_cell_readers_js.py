"""Every declared type, read the way the stage table reads it.

``docs/web/task-setup.md`` § 5.2: *"the declared type decides the cell — both
the widget and the value"*.  This drives the real readers under Node, one
case per row of behaviour.

**Why this file exists at all.**  The readers used to sit inside
``task-setup/viewer.js``, a 2500-line page controller that cannot be imported
without a DOM — so the only thing a test could check was that the KEYS
existed.  ``int3: (t) => t`` would have passed that, and the behaviour was
verified once, by hand, in a browser.  This codebase has been burned by
exactly that assurance before (`test_task_setup_tab.py`'s own note: *"a claim
in a contract that rests on somebody having driven it once"*), and the bug
this closes — ``"kgrid": "4,4,1"`` reaching a ``Tuple[int, int, int]`` field —
was one keystroke deep.

**Three types have no shipped catalogue item**: ``pow2``, ``text`` and
``intlist``.  A column's type comes from the shipped catalogue, so no cell
can carry one today.  They have readers anyway, because the failure mode
without one is SILENT — the lookup misses, the raw text is stored, and the
description carries a string.  That is the bug, and it is worth one line
each to make the day somebody adds such an item uneventful.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from _node_esm import run_node

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/task-setup/cell-readers.js"

#: A reader answers ``undefined`` for text that is not its type; ``setCell``
#: then keeps the text AS TYPED, and the save door refuses it by name
#: (`stages.md` § 6.6's declared-type row).  JSON has no ``undefined``, so
#: the harness maps it to this.
KEPT = "<<kept as typed>>"

CASES = [
    # (declared type, what was typed, what the description must carry)
    ("bool",    "true",             True),
    ("bool",    "false",            False),
    ("enum",    "Broyden",          "Broyden"),
    ("str",     "probe_run",        "probe_run"),
    ("text",    "%block X\n 1",     "%block X\n 1"),

    ("int",     "100",              100),
    # A whole-valued float IS an integer, by the same rule the preflight
    # applies to the value it ends up checking.  A stricter test here would
    # refuse a value the save door accepts.
    ("int",     "100.0",            100),
    ("int",     "-4",               -4),
    ("int",     "100.7",            KEPT),
    ("int",     "abc",              KEPT),
    ("pow2",    "64",               64),

    ("float",   "300",              300),
    ("float",   "0.02",             0.02),
    ("float",   "1e-5",             1e-05),
    ("float",   "abc",              KEPT),

    # The three spellings `--kgrid` itself takes (`cli.KGridParam`).
    ("int3",    "4,4,1",            [4, 4, 1]),
    ("int3",    "4x4x1",            [4, 4, 1]),
    ("int3",    "4 4 1",            [4, 4, 1]),
    ("int3",    "4, 4, 1",          [4, 4, 1]),
    ("int3",    "4,4",              KEPT),
    ("int3",    "4,4,1,1",          KEPT),
    ("int3",    "4,a,1",            KEPT),
    ("int3",    "4,4,1.5",          KEPT),

    ("float3",  "0.5,0.5,0.0",      [0.5, 0.5, 0.0]),
    ("float3",  "0 0 0",            [0, 0, 0]),
    ("float3",  "0.5,0.5",          KEPT),

    ("strlist", "Au,C,H,S",         ["Au", "C", "H", "S"]),
    ("strlist", "Au, C, H, S",      ["Au", "C", "H", "S"]),
    # `x` separates a GRID, never a list -- splitting on it here would cut
    # `Xe` in half, and an element symbol is not a thing to guess at.
    ("strlist", "Xe,Au",            ["Xe", "Au"]),
    ("strlist", "Au",               ["Au"]),

    ("intlist", "0,1,2",            [0, 1, 2]),
    ("intlist", "0 1 2",            [0, 1, 2]),
    ("intlist", "0,x,2",            KEPT),
]


@pytest.fixture(scope="module")
def read():
    """Every case through the real module, in ONE Node call."""
    snippet = (
        f"const m = await import({json.dumps(MODULE.resolve().as_uri())});\n"
        f"const cases = {json.dumps([[t, txt] for t, txt, _ in CASES])};\n"
        "const out = cases.map(([t, text]) => {\n"
        "  const r = m.CELL_READERS[t];\n"
        "  if (!r) return '<<no reader>>';\n"
        "  const v = r(text);\n"
        f"  return v === undefined ? {json.dumps(KEPT)} : v;\n"
        "});\n"
        "console.log(JSON.stringify(out));")
    got = run_node([], snippet)
    assert len(got) == len(CASES)
    return got


@pytest.mark.parametrize(
    "i", range(len(CASES)),
    ids=[f"{t}:{txt!r}" for t, txt, _ in CASES])
def test_a_cell_reads_as_its_declared_type(read, i):
    declared, typed, want = CASES[i]
    got = read[i]
    assert got == want, (
        f"a {declared} cell holding {typed!r} put {got!r} into the "
        f"description; it must carry {want!r}")
    # `1 == True` in Python, and a description carrying 1 where the config
    # declares bool is the `use_gpu` bug in its other direction.
    if isinstance(want, bool) or isinstance(got, bool):
        assert isinstance(got, bool) == isinstance(want, bool), (
            f"a {declared} cell produced {type(got).__name__}, not "
            f"{type(want).__name__}")


def test_every_declared_type_has_a_reader():
    """`template.TYPES` is the closed vocabulary a catalogue item may
    declare.  A type present there and missing here does not fail loudly:
    the lookup misses, ``setCell`` stores the raw text, and the description
    carries a string -- which is exactly how ``int3``, ``float3``,
    ``strlist`` and ``intlist`` behaved until 2026-08-25.  Pinning the two
    vocabularies against each other is the only thing that makes the next
    addition noisy."""
    from molbuilder.template import TYPES
    have = set(run_node(
        [], f"const m = await import({json.dumps(MODULE.resolve().as_uri())});\n"
            "console.log(JSON.stringify(Object.keys(m.CELL_READERS)));"))
    assert have == set(TYPES), (
        f"no reader for {sorted(set(TYPES) - have)}; "
        f"reader for a type the catalogue cannot declare: "
        f"{sorted(have - set(TYPES))}")


def test_every_declared_type_is_exercised_above():
    """The coverage test's own coverage.  A reader can exist and be wrong,
    so the table above must reach all of them -- otherwise this file pins
    eleven keys and one behaviour, which is what it replaced."""
    from molbuilder.template import TYPES
    covered = {t for t, _, _ in CASES}
    assert covered == set(TYPES), (
        f"no behaviour case for {sorted(set(TYPES) - covered)}")
