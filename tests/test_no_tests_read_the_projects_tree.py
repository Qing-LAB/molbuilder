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

import re
from pathlib import Path


TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

#: A path that resolves INTO the repository's own ``projects/`` tree.
#:
#: Deliberately narrow.  ``tmp_path / "projects"`` is a temp tree the test built
#: itself and is exactly right; so is a string literal that merely MENTIONS such
#: a path (the CSV-redaction tests feed one in as sample text and never open
#: it).  What is forbidden is rooting at the repo and reaching for real data.
_REPO_ROOTED = re.compile(
    r"""(?:REPO|REPO_ROOT|ROOT)\s*/\s*["']projects["']
      | Path\(__file__\)[^\n]*?/\s*["']projects["']
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
    out = []
    for n, raw in enumerate(path.read_text(encoding="utf-8",
                                           errors="replace").splitlines(), 1):
        if _OPT_OUT in raw:
            continue
        code = raw.split("#", 1)[0]           # a comment may DISCUSS the path
        if any(a in code for a in _ALLOWED):
            continue
        if _REPO_ROOTED.search(code):
            out.append((n, raw.strip()[:100]))
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
    deck whose labels the LIVE reader recovers intact.  (This went through
    BundleDirParser until 2026-08-29; the bundle parser retired with
    calculation-to-calculation passing, and the reader that survives it --
    the run decoder's own extractor -- is what a finished run's labels are
    actually read by.)"""
    import sys
    sys.path.insert(0, str(TESTS))
    from support.junction import frozen, run_dir
    from molbuilder.script_emit import extract_script_source

    d = run_dir(tmp_path)
    deck = sorted(d.glob("*.fdf"))[0]
    out = extract_script_source(deck.read_text())
    assert sorted(out["frozen_atoms"]) == frozen()


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
