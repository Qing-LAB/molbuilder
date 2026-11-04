"""What the doors ANSWER.

That nobody hand-spells a name is a claim about the source, kept by review
(`tools/classify_path_finders.py --check`; `process/code-audit.md` § 1c (f)).
This file is what a run can show: each door must give the right answer.

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
# 1.  The patterned role -- a row in the catalogue that is a family.          #
# --------------------------------------------------------------------------- #

def test_no_role_in_the_catalogue_is_a_GLOB():
    """A wildcard in a role is a coordinate that escaped into the vocabulary.

    § 5l.3: a channel is a **field**, so the role is a template and every
    comparison is an ordinary equality.
    """
    starred = sorted(a.role for a in runfiles.WRITTEN if "*" in a.role)
    assert not starred, (
        f"these rows declare a glob as a role: {starred}. If the name varies, "
        f"declare the varying part as a FIELD (`fields=(...)` on the row, its "
        f"shape in `runfiles.FIELDS`) — see § 5l.1.")


def test_every_declared_field_has_a_declared_shape():
    """A field whose shape is not written down cannot be read back out.

    The shapes live in one place: `compose` validates against it and `parse`
    reads with it, so the two cannot disagree about what a field looks
    like.
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
    assert canonical_role(".TBT_UP.digest.nc") == (
        runfiles.ROLE_TBT_DIGEST, {"tbt": "TBT_UP"})
    # ...and an engine's file the catalogue names (`runfiles.ENGINE_FILES`):
    # a lead's Green's functions at a voltage, a channel's data file.
    assert canonical_role(".v0.2.TSGFL") == (
        runfiles.SIESTA_TSGF, {"bias": "v0.2", "lead": "L"})
    assert canonical_role(".TBT_DN.nc") == (
        runfiles.TBTRANS_NC, {"tbt": "TBT_DN"})
    for plain in (".out", ".pyscf.log", ".log", "_geom_optim.xyz", ".XV"):
        assert canonical_role(plain) == (plain, {}), plain
    # a value outside the field's shape stays foreign
    assert canonical_role(".TBT_XX.digest.nc") == (".TBT_XX.digest.nc", {})
    assert canonical_role(".x0.2.TSGFL") == (".x0.2.TSGFL", {})


@pytest.mark.parametrize("name,role,fields", [
    ("tsgf/AuBDT.v0.2.TSGFL", runfiles.SIESTA_TSGF,
     {"bias": "v0.2", "lead": "L"}),
    ("AuBDT.TSHS", runfiles.SIESTA_TSGF, None),
], ids=["a-lead-of-ours", "not-of-the-role"])
def test_a_templated_engine_name_reads_back_its_fields(name, role, fields):
    """`runfiles.fields_of`: the fields an engine's file carries, read off
    the name it prints (``Using GF-file '<path>'``) by its row's pattern.
    THE SILENT FAILURE: a lead cut from the name by hand."""
    assert runfiles.fields_of(name, role) == fields


def test_a_field_round_trips_through_compose_and_parse():
    """`parse` then `.name` must give back the byte-identical filename.

    That is what makes the template a real declaration rather than a label: the
    grammar can rebuild the exact name it read.
    """
    written = "bdt.TBT_UP.digest.nc"
    rf = runfiles.parse(written, "bdt")
    assert rf is not None
    assert rf.role == runfiles.ROLE_TBT_DIGEST
    assert dict(rf.fields) == {"tbt": "TBT_UP"}
    assert rf.name == written


def test_a_template_cannot_be_composed_without_its_field():
    """An omission is refused, and so is a value of the wrong shape.

    Both would produce a name `parse` could not read back — which is exactly
    what the field's declared shape exists to prevent.
    """
    with pytest.raises(runfiles.RunFileError) as e:
        runfiles.compose("bdt", runfiles.ROLE_TBT_DIGEST)
    assert "needs the field" in str(e.value)
    with pytest.raises(runfiles.RunFileError) as e:
        runfiles.compose("bdt", runfiles.ROLE_TBT_DIGEST, tbt="TBT_XX")
    assert "declared shape" in str(e.value)


def test_find_by_role_answers_for_the_family_without_a_label(tmp_path):
    """Label-less, because the catalogue's role is recognisable on its own."""
    d = _touch(tmp_path, "a.TBT.digest.nc", "b.TBT_UP.digest.nc", "a.out")
    assert [p.name for p in find_by_role(d, runfiles.ROLE_TBT_DIGEST)] == [
        "a.TBT.digest.nc", "b.TBT_UP.digest.nc"]


@pytest.mark.parametrize("engine,calculation,theirs", [
    ("pyscf", "optimization", (runfiles.PYSCF_BUNDLE,)),
    ("siesta", "optimization", (runfiles.MAKOV_PAYNE_SCRIPT,)),
    ("siesta", "vibration", (runfiles.MAKOV_PAYNE_SCRIPT,
                             runfiles.COORDINATES_FILE)),
], ids=["a-PySCF-deck", "a-SIESTA-relaxation", "a-SIESTA-vibration"])
def test_what_travels_beside_a_deck_is_its_engines_and_its_kinds(
        engine, calculation, theirs):
    """`job-contracts.md` § 2.2: the files a launch brings beside a deck are
    the catalogue's travelling rows, each narrowed by its own engine and
    kind -- the run script, its header and the monitor beside every deck,
    then what this engine's and this kind's runs open.  THE SILENT FAILURE:
    one hand list for every deck -- SIESTA's charge-correction script and a
    vibration call's coordinates file brought beside a PySCF script, and
    claimed as prep's own beside it."""
    names = runfiles.RunNames("h2", "01_coarse")
    assert sorted(runfiles.beside_the_deck(engine, calculation, None,
                                           names)) == sorted(
        (names.run_script, names.header, runfiles.MONITOR_BUNDLE, *theirs))


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


