"""No test may take its facts from the repository's ``projects/`` tree.

USER RULE, 2026-08-03: *"Don't use a data file without confirming if it is
relevant. Garbage in garbage out."* and *"Code is designed to look forward."*

``projects/`` is the user's scientific record.  A test that reads it is broken
three ways, and the third hides the other two:

  1. **Unconfirmed relevance.** The file exists; that is not a claim it is
     current, correct, or wanted.  Several under ``projects/`` are residue from
     abandoned attempts, so a test asserting facts about them asserts nothing.
  2. **It changes meaning with no diff.** Regenerate the run and the test now
     measures something else -- no review, and no way to tell a real regression
     from a moved fixture.  That cost a bisect once (task #41): the failure
     "your fixture predates the label store" looks exactly like "your parser is
     broken".
  3. **It skips silently elsewhere.** Every one sat behind a
     ``pytest.skip("fixture absent")``.  On a machine without that directory the
     test SKIPS and the suite still reads green -- reporting coverage it does
     not have, which is worse than a red test.

THE RULE: build what you need.  ``tests/support/junction.py`` is the pattern --
an Au-BDT-Au junction defined in source, written through the application's own
writers (``render_fdf``, ``dump_spectra_json``) and read back through its own
readers.  A constructed fixture cannot go stale, its relevance is not a guess,
and it proves the emit and parse halves agree TODAY rather than agreeing with a
number captured once.

Engine OUTPUT is the one thing that cannot be constructed honestly -- a
hand-written ``.out`` tests a guess at SIESTA's format rather than SIESTA's.
Those live checked in at ``tests/watch/fixtures/siesta_frozen/``: real output,
versioned with the tests, reviewed when it changes.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest


TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

#: A path that resolves INTO the repository's own ``projects/`` tree.
#:
#: Deliberately narrow.  ``tmp_path / "projects"`` is a temp tree the test built
#: itself and is exactly right; so is a string literal that merely MENTIONS such
#: a path (the CSV-redaction tests feed one in as sample text and never open
#: it).  What is forbidden is rooting at the repo and reaching for real data.
#: A name that means "the repository checkout".
_ROOT_NAMES = {"ROOT", "REPO", "REPO_ROOT"}

#: ``projects/pseudopotential`` is a shared INPUT LIBRARY (the PSML files an
#: engine needs), not a record of anybody's results.  Listed so the exception
#: is a decision on the page rather than a hole in a pattern.
_ALLOWED_FIRST_SEGMENTS = {"pseudopotential"}

#: A line may opt out by saying so -- for a path that is sample TEXT under
#: test rather than a file to open.
_OPT_OUT = "# not-a-fixture"

_EXEMPT = {Path(__file__).name}


def _is_repo_root(node) -> bool:
    """Does *node* evaluate to the repository checkout?

    Two spellings: a module constant named ``ROOT``/``REPO``/``REPO_ROOT``,
    and a ``Path(__file__)`` walk -- ``.resolve().parent.parent``,
    ``.parents[1]``, any depth.
    """
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id in _ROOT_NAMES
    while isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id == "Path":
            return any(isinstance(a, ast.Name) and a.id == "__file__"
                       for a in node.args)
        node = f.value if isinstance(f, ast.Attribute) else None
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
    return False


def _div_chain(node):
    """``a / b / c`` as ``[a, b, c]`` -- the shape a path is built in."""
    parts = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parts.append(node.right)
        node = node.left
    parts.append(node)
    return list(reversed(parts))


def _offending_lines(path: Path):
    """Every expression in *path* that builds a path into the repo's tree.

    **AST, not text** (2026-09-06).  This was a regex over the file, and it
    was wrong three separate ways at once -- it read one LINE at a time so a
    path split across two was invisible; it required the literal
    ``"projects"`` so the ``"projects/_t_..."`` spelling that thirteen sites
    actually used never matched; and once those were fixed it flagged
    DOCSTRINGS that quote the rule, which made the rule unwriteable in its own
    words.  Each fix was a patch on a symptom.  A path is a syntax tree, so
    the check reads one: a division chain whose base is the checkout and whose
    first string segment is ``projects``.  All three problems stop existing --
    line breaks are not a concept here, a segment's value is its value, and a
    docstring is an ``Expr``, not a ``BinOp``.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    opted_out = {n for n, line in enumerate(lines, 1) if _OPT_OUT in line}
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return []

    # A DOCSTRING IS PROSE.  It is the one place a forbidden path may appear
    # as TEXT -- showing what a redaction turns into, or telling you not to
    # write one -- and flagging it makes the rule unwriteable in its own
    # words.  Only the whole-path branch below needs this: a division chain
    # cannot be a docstring.
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))

    out = []
    for node in ast.walk(tree):
        segs = None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            parts = _div_chain(node)
            if _is_repo_root(parts[0]):
                segs = [p.value for p in parts[1:]
                        if isinstance(p, ast.Constant) and isinstance(p.value, str)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            # A whole path written out: ".../molbuilder/projects/<something>"
            if "/molbuilder/projects/" in node.value:
                segs = node.value.split("/molbuilder/", 1)[1].split("/")
        if not segs:
            continue
        flat = [s for seg in segs for s in seg.split("/") if s]
        if not flat or flat[0] != "projects" or len(flat) < 2:
            continue
        if flat[1] in _ALLOWED_FIRST_SEGMENTS:
            continue
        if node.lineno in opted_out:
            continue
        out.append((node.lineno, lines[node.lineno - 1].strip()[:100]))
    return sorted(set(out))


def test_no_test_file_reads_the_projects_tree():
    hits = []
    for f in sorted(TESTS.rglob("*.py")):
        if f.name in _EXEMPT:
            continue
        for n, line in _offending_lines(f):
            hits.append(f"{f.relative_to(REPO)}:{n}  {line}")
    assert not hits, (
        "these tests read the user's projects/ tree, whose relevance nobody "
        "confirmed and which skips silently on any other machine:\n  "
        + "\n  ".join(hits)
        + "\n\nBuild what you need -- see tests/support/junction.py."
    )


def test_no_test_skips_itself_when_a_projects_fixture_is_missing():
    """The silent half.  A skip keyed on a missing projects/ path is how a
    suite reports coverage it does not have."""
    bad = []
    for f in sorted(TESTS.rglob("*.py")):
        if f.name in _EXEMPT:
            continue
        for n, raw in enumerate(f.read_text(encoding="utf-8",
                                            errors="replace").splitlines(), 1):
            code = raw.split("#", 1)[0]    # a comment may QUOTE the pattern
            m = re.search(r'pytest\.skip\(([^)]*)\)', code)
            if not m:
                continue
            arg = m.group(1)
            if "fixture" in arg and ("absent" in arg or "missing" in arg):
                bad.append(f"{f.relative_to(REPO)}:{n}  {raw.strip()[:80]}")
    assert not bad, (
        "these skip themselves when a fixture file is absent, so the suite "
        "reads green while proving nothing:\n  " + "\n  ".join(bad)
    )


# --------------------------------------------------------------------- #
#  The alternative has to work, or the rule above is just a prohibition #
# --------------------------------------------------------------------- #

def test_the_shared_junction_fixture_builds():
    import sys
    sys.path.insert(0, str(TESTS))
    from support.junction import build_junction, frozen, regions

    s = build_junction()
    assert s.n_atoms > 0
    assert set(s.regions) == set(regions())
    assert s.regions["frozen_atoms"] == sorted(frozen())


def test_a_built_run_directory_round_trips_its_labels(tmp_path):
    """End to end on constructed data: the application's own writer produces a
    deck whose labels the LIVE reader recovers intact.

    It said "the LIVE reader" while driving `_extract_script_source` until
    2026-09-05 -- a door with no production caller, deleted that day. So the
    one test claiming to cover how a finished run's labels are read was the
    only thing keeping that path alive, and covered nothing that ships. It
    now walks the real one: the run directory is scanned by
    `parse/dirs/atom_metadata.py` and applied by `apply_atom_metadata`, which
    is what `/api/build/load` and the transport composite both use.
    """
    import sys
    sys.path.insert(0, str(TESTS))
    from support.junction import frozen, run_dir
    from molbuilder.parse.dirs.atom_metadata import atom_metadata_json_for_run_dir
    from molbuilder.script_emit import apply_atom_metadata
    from molbuilder.structure import FROZEN_LABEL, Structure

    d = run_dir(tmp_path)
    recovered = atom_metadata_json_for_run_dir(d)
    assert recovered is not None, "the run dir's own deck yielded no label block"

    payload = json.loads(recovered)
    struct = Structure(elements=["C"] * payload["n_atoms_total"],
                       positions=np.zeros((payload["n_atoms_total"], 3)))
    assert apply_atom_metadata(struct, payload) is True
    assert sorted(struct.regions[FROZEN_LABEL]) == frozen()


def test_the_built_xv_round_trips_through_the_real_parser(tmp_path):
    """The .XV writer is hand-rolled -- the application only ever READS one --
    so it is verified the only honest way: parse it back and compare."""
    import sys
    import numpy as np
    sys.path.insert(0, str(TESTS))
    from support.junction import build_junction, xv_file
    from molbuilder.parse import parse

    s = build_junction()
    result = parse(xv_file(tmp_path / "j.XV"))
    assert list(result.structure.elements) == list(s.elements)
    assert np.allclose(result.structure.positions, s.positions, atol=1e-6)
    assert np.allclose(result.cell, s.resolve_cell(), atol=1e-6)


def test_the_built_spectra_sidecar_round_trips(tmp_path):
    import sys
    sys.path.insert(0, str(TESTS))
    from support.junction import spectra_sidecar
    from molbuilder.parse import parse

    result = parse(spectra_sidecar(tmp_path / "built.spectra.json"))
    assert result.schema.startswith("spectra/v")
    assert "schema_version" in result.payload


# ---------------------------------------------------------------------------
#  The detector's own truth table
# ---------------------------------------------------------------------------
#
#  Not "a test for a test": this file IS a static check, and a checker with no
#  known-good and known-bad inputs is how this one came to be wrong three ways
#  at once while reading green -- blind to a line break, blind to the
#  `"projects/_t_..."` spelling that thirteen sites actually used, and (once
#  those were fixed) flagging the docstrings that explain the rule.  Every row
#  below is a shape that really occurred in this repository.

_MUST_FIRE = [
    ("split across two lines",
     'from pathlib import Path\nP = (\n    Path(__file__).resolve().parent.parent\n'
     '    / "projects" / "BDT" / "H.psml"\n)\n'),
    ("the spelling thirteen sites used",
     'ROOT = 1\nd = ROOT / "projects/_t_handover/optimization/probe"\n'),
    ("separate segments",  'ROOT = 1\nd = ROOT / "projects" / "_t_x"\n'),
    ("a parents[N] walk",
     'from pathlib import Path\nd = Path(__file__).resolve().parents[1] / "projects/_t_y"\n'),
    ("a whole path as one literal", 'p = "/home/u/molbuilder/projects/BDT/run.out"\n'),
]

_MUST_NOT_FIRE = [
    ("the pseudopotential library", 'ROOT = 1\nd = ROOT / "projects" / "pseudopotential"\n'),
    ("the same, joined",  'ROOT = 1\nd = ROOT / "projects/pseudopotential/H.psml"\n'),
    ("a docstring quoting the rule",
     '"""Do not write /home/u/molbuilder/projects/BDT/x here."""\n'),
    ("an opted-out sample path",
     'p = "/home/u/molbuilder/projects/BDT/run.out"   # not-a-fixture\n'),
    ("an isolated root",
     'def f(isolated_projects_root):\n    return isolated_projects_root / "p/t/c"\n'),
]


@pytest.mark.parametrize("label,src", _MUST_FIRE, ids=[c[0] for c in _MUST_FIRE])
def test_the_detector_catches(tmp_path, label, src):
    f = tmp_path / "probe.py"
    f.write_text(src, encoding="utf-8")
    assert _offending_lines(f), f"the detector is blind to: {label}"


@pytest.mark.parametrize("label,src", _MUST_NOT_FIRE, ids=[c[0] for c in _MUST_NOT_FIRE])
def test_the_detector_leaves_alone(tmp_path, label, src):
    f = tmp_path / "probe.py"
    f.write_text(src, encoding="utf-8")
    assert not _offending_lines(f), f"the detector wrongly flags: {label}"
