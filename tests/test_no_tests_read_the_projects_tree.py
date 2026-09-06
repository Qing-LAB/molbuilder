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

import json
import re
from pathlib import Path

import numpy as np


TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

#: A path that resolves INTO the repository's own ``projects/`` tree.
#:
#: Deliberately narrow.  ``tmp_path / "projects"`` is a temp tree the test built
#: itself and is exactly right; so is a string literal that merely MENTIONS such
#: a path (the CSV-redaction tests feed one in as sample text and never open
#: it).  What is forbidden is rooting at the repo and reaching for real data.
#: WHITESPACE INCLUDES NEWLINES, and that is the point.  This was applied one
#: LINE at a time until 2026-09-06, so an expression broken across two lines
#: was invisible to it -- and one was::
#:
#:     _H_PSML_SOURCE = (
#:         Path(__file__).resolve().parent.parent
#:         / "projects" / "BDT" / "optimization" / "TJ-BDT-Au111" / "H.psml"
#:     )
#:
#: which is a real project's real pseudopotential, exactly the thing this file
#: forbids, sitting in the suite the whole time the guard read green.  The
#: scan is over the file's TEXT now, and `\s*` between the parts is what makes
#: a line break stop hiding anything.
_REPO_ROOTED = re.compile(
    r"""(?:REPO|REPO_ROOT|ROOT)\s*/\s*["']projects["']
      | Path\(__file__\)(?:\s*\.\s*\w+\s*\(\s*\))*[^\n]{0,80}?\s*/\s*["']projects["']
      | ["'][^"'\n]*/molbuilder/projects/[^"'\n]+["']
    """,
    re.VERBOSE,
)

#: ``projects/pseudopotential`` is a shared INPUT LIBRARY (the PSML files an
#: engine needs), not a record of anybody's results, and the tests naming it
#: assert on a message string rather than reading it.  Listed so the exception
#: is a decision on the page rather than a hole in a regex.
_ALLOWED = ('projects/pseudopotential', '"projects" / "pseudopotential"')

#: A line may opt out by saying so.  For a path that is sample TEXT under test
#: rather than a file to open.
_OPT_OUT = "# not-a-fixture"

_EXEMPT = {Path(__file__).name}


def _offending_lines(path: Path):
    """Every real-tree path in ``path``, as ``(line number, source)``.

    Scans the file's TEXT, not its lines: a path expression may be split
    across a line break, and reading one line at a time is how one hid here
    for months (see :data:`_REPO_ROOTED`).  Comments are blanked rather than
    removed so every offset still maps to its own line.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    # Blank out comments and opted-out lines, PRESERVING LENGTH, so a match
    # offset still names the line it came from.
    kept = []
    for line in lines:
        if _OPT_OUT in line:
            kept.append(" " * len(line))
            continue
        code = line.split("#", 1)[0]
        kept.append(code + " " * (len(line) - len(code)))
    code_text = "\n".join(kept)

    out = []
    for m in _REPO_ROOTED.finditer(code_text):
        start, end = m.start(), m.end()
        n = code_text.count("\n", 0, start) + 1
        # The allow-list is checked against the MATCH and the lines it spans,
        # so a multi-line exception is recognised the same way a single-line
        # one is.
        span = "\n".join(lines[n - 1:code_text.count("\n", 0, end) + 1])
        if any(a in span or a in m.group(0) for a in _ALLOWED):
            continue
        out.append((n, lines[n - 1].strip()[:100]))
    return out


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
