"""What the doors ANSWER.

`tests/test_path_framework.py` proves nobody hand-spells a name.  That is a
claim about the source.  This file is the other half: each door must give the
right answer.

Every test here builds files on disk and asks the function.  None of them reads
source text.
"""
from __future__ import annotations

import pytest

from molbuilder import runfiles
from molbuilder.runfiles import canonical_role, find_by_role


def _touch(d, *names):
    for n in names:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# 1.  The patterned role -- the one row in the catalogue that is a family.     #
# --------------------------------------------------------------------------- #

def test_no_role_in_the_catalogue_is_a_GLOB():
    """A wildcard in a role is a coordinate that escaped into the vocabulary.

    § 5l.3: the stamp is a **field**, so the role is a template and every
    comparison is an ordinary equality.
    """
    starred = sorted(a.role for a in runfiles.WRITTEN if "*" in a.role)
    assert not starred, (
        f"these rows declare a glob as a role: {starred}. If the name varies, "
        f"declare the varying part as a FIELD (`fields=(...)` on the row, its "
        f"shape in `runfiles.FIELDS`) — see § 5l.1.")


def test_every_declared_field_has_a_declared_shape():
    """A field whose shape is not written down cannot be read back out.

    The shapes live in one place for the reason `QUALIFIERS` does: `compose`
    validates against it and `parse` reads with it, so the two cannot disagree
    about what a stamp looks like.
    """
    for a in runfiles.WRITTEN:
        for f in a.fields:
            assert f in runfiles.FIELDS, (
                f"role {a.role!r} names the field {f!r} with no shape in "
                f"`FIELDS`")
        # and the template must actually mention what it declares
        assert set(a.fields) == set(runfiles._fields_in(a.role)), (
            f"role {a.role!r} declares fields {a.fields} but its template "
            f"names {runfiles._fields_in(a.role)}")


def test_canonical_role_maps_a_name_back_to_its_TEMPLATE():
    """A plain role must come back unchanged and must NOT become fuzzy — ``.out``
    matching ``.pyscf.out``, or ``.log`` matching ``.pyscf.log``, is the failure
    mode that would make every comparison in the module quietly approximate.
    """
    assert canonical_role(".runwrap-20260908-120000.log") == (
        ".runwrap-{stamp}.log", {"stamp": "20260908-120000"})
    for plain in (".out", ".pyscf.log", ".log", "_geom.log"):
        assert canonical_role(plain) == (plain, {}), plain
    # a stamp-shaped thing that is not in the right place stays foreign
    assert canonical_role(".runwrap-nope.log") == (".runwrap-nope.log", {})


def test_a_field_round_trips_through_compose_and_parse():
    """`parse` then `.name` must give back the byte-identical filename.

    That is what makes the template a real declaration rather than a label: the
    grammar can rebuild the exact name it read.
    """
    written = "bdt_01_tight.runwrap-20260908-120000.log"
    rf = runfiles.parse(written, "bdt")
    assert rf is not None
    assert rf.role == ".runwrap-{stamp}.log"
    assert dict(rf.fields) == {"stamp": "20260908-120000"}
    assert rf.name == written


def test_a_template_cannot_be_composed_without_its_field():
    """An omission is refused, and so is a value of the wrong shape.

    Both would produce a name `parse` could not read back — which is exactly
    what the field's declared shape exists to prevent.
    """
    with pytest.raises(runfiles.RunFileError) as e:
        runfiles.compose("bdt", ".runwrap-{stamp}.log")
    assert "needs the field" in str(e.value)
    with pytest.raises(runfiles.RunFileError) as e:
        runfiles.compose("bdt", ".runwrap-{stamp}.log", stamp="not-a-stamp")
    assert "declared shape" in str(e.value)


def test_the_glob_family_is_unchanged_by_the_field():
    """`patterns()` must still emit `{label}.runwrap-*.log`.

    `identity.OUR_FILE_PATTERNS` and `runwrap`'s `--cold` sweep read that view,
    and a `--cold` run that stopped recognising the wrapper log would leave it to
    be appended to by the next launch. The star belongs in the GLOB view, which
    is the only place it is now produced.
    """
    from molbuilder.identity import OUR_FILE_PATTERNS
    fam = [p for p in runfiles.patterns() if "runwrap" in p]
    assert fam == ["{label}.runwrap-*.log", "{label}_*.runwrap-*.log"]
    assert [p for p in OUR_FILE_PATTERNS if "runwrap" in p] == fam


def test_find_by_role_answers_for_the_family_without_a_label(tmp_path):
    """Label-less, because the catalogue's role is recognisable on its own."""
    d = _touch(tmp_path, "a.runwrap-20260101-000000.log",
               "b.runwrap-20260102-000000.log", "a.out")
    assert [p.name for p in find_by_role(d, ".runwrap-{stamp}.log")] == [
        "a.runwrap-20260101-000000.log", "b.runwrap-20260102-000000.log"]


def test_find_by_role_still_refuses_a_role_the_catalogue_does_not_declare():
    """A typo is a refusal, not an empty list — the reason the check is there."""
    with pytest.raises(runfiles.RunFileError) as e:
        find_by_role(".", ".runwrap.log")           # no star: not a row
    assert "not a role molbuilder writes" in str(e.value)


# --------------------------------------------------------------------------- #
# 3.  The sidecar's finder and its composer.                                   #
# --------------------------------------------------------------------------- #

def test_sidecars_in_finds_what_sidecar_path_for_composes(tmp_path):
    """The § 4.5 pairing, asserted as a round trip rather than as two lists."""
    from molbuilder.sidecars.molstruct import (is_sidecar, sidecar_path_for,
                                               sidecars_in)
    for stem in ("relaxed", "bridge.spectra"):
        sidecar_path_for(tmp_path / f"{stem}.xyz").write_text("{}",
                                                             encoding="utf-8")
    found = sidecars_in(tmp_path)
    assert [p.name for p in found] == ["bridge.spectra.molstruct.json",
                                       "relaxed.molstruct.json"]
    assert all(is_sidecar(p) for p in found)
    assert not is_sidecar(tmp_path / "relaxed.xyz")


# --------------------------------------------------------------------------- #
# 4.  The other callers, asked for their answers.                              #
# --------------------------------------------------------------------------- #


def test_the_trial_search_reads_the_prefix_through_its_own_reader(tmp_path):
    """`materialize.trials_in` returns the trial directories and skips a
    directory that merely sits beside them.

    `paths.trial_name` composes `bench-<point>` and `paths.trial_point` reads it
    back. The pair is § 4.5's rule, and a door that re-spells one half is how
    the two drift.
    """
    from molbuilder.jobset.materialize import trials_in
    c = tmp_path / "01_coarse" / "bench"
    for d in ("bench-G1", "bench-G2K4", "notatrial", "bench-"):
        (c / d).mkdir(parents=True, exist_ok=True)
    assert [p.name for p in trials_in(c)] == ["bench-G1", "bench-G2K4"]
