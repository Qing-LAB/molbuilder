"""**A pseudopotential declares requirements, and the calculation must satisfy
them** — layer 2 of `science/pseudopotentials.md` § 2a, and the layer-1 status
that goes with it.

The two facts this pins were both found by comparing the two PseudoDojo tables
after a real failure (user, 2026-09-03): their v0.5 sulfur broke a SIESTA run
and v0.4 fixed it.  v0.5 is v0.4 with **eleven elements re-generated** — Ba,
Bi, I, Pb, Po, Rb, Rn, S, Te, Tl, Xe — and those eleven, and only those, both
carry cutoff hints and have zeroed projectors.

**Layer 1 — `semilocal_only`.**  For sulfur the zeroed channel is **p**, and
S's valence is 3s² 3p⁴, so the channel with nothing in it is one of the two
that make sulfur bond.  The file writes `<proj l='p' ekb='0'>` with 462
explicit zeros — *not* an omission (PSML says "nothing here" by leaving the
`<proj>` out entirely) but a claim that the projector IS zero.  It is not.

**Layer 2 — the declared cutoff.**  The same eleven state
`cutoff_hint_normal`; sulfur says 147 Ry.  Nothing read it, while a generic
150 Ry literature floor was checked instead — a guess outranking a
measurement.

**The severities differ, and § 2a.2 says why.**  Layer 1 is an **ERROR**
(user ruling, 2026-09-03): the file states something false about a channel
carrying valence electrons, nothing in it says to read the semilocal block
instead, and a reader doing the normal thing therefore gets wrong physics with
no crash to notice — the `xc_family_mismatch` case.  Layer 2 is a **WARN**: a
cutoff below what a file recommends is a trade-off a person may make
knowingly, and the number is theirs to set.
"""
from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np

from _road import PSML_DIR
from molbuilder.pseudos import ERROR_STATUSES, check_coverage, parse_psml_header
from molbuilder.structure import Structure
from molbuilder.validation.siesta import _declared_cutoff_ry

#: The files themselves (`tests/fixtures/psml/README.md`).
S_V04 = PSML_DIR / "pseudodojo-v0.4" / "S.psml"
S_V05 = PSML_DIR / "pseudodojo-v0.5" / "S.psml"
BI_V05 = PSML_DIR / "pseudodojo-v0.5" / "Bi.psml"


def _lib(tmp_path, **files):
    """A library holding these real files under their species' names."""
    d = tmp_path / "psml"
    d.mkdir(exist_ok=True)
    for name, source in files.items():
        shutil.copy(source, d / f"{name}.psml")
    return d


def _found(lib, element):
    return [e for e in check_coverage([element], lib,
                                      expected_xc_family="GGA",
                                      expected_xc_authors="PBE")
            if e.status != "ok"]


# ===================================================================== #
#  Layer 1 -- the file: present-but-zero is a claim, and it is false    #
# ===================================================================== #

def test_the_file_that_broke_a_run_names_its_zeroed_valence_channel():
    """v0.5's sulfur: both p projectors zero, with an `<slps l="p">` carrying
    the channel, and p is in its valence (3s2 3p4).  One of its two `d`
    projectors is zero too and is not named: nothing occupies d, so nothing
    rides on it, and naming it would bury the one that matters.  C5 stays
    quiet: it asks whether a channel is MISSING, and this one is present and
    lying."""
    info = parse_psml_header(S_V05)
    assert info.semilocal_only_channels == ["p"]
    assert info.partial_projector_channels == []
    assert info.null_channels == []


def test_the_good_pseudo_says_nothing():
    """v0.4's sulfur, every channel a real projector."""
    info = parse_psml_header(S_V04)
    assert info.semilocal_only_channels == []
    assert info.partial_projector_channels == []
    assert info.null_channels == []


def test_it_BLOCKS(tmp_path):
    """ERROR, by user ruling 2026-09-03.

    The file states something false about a channel carrying four of
    sulfur's six valence electrons, and nothing in it says to read the
    semilocal block instead -- so a reader doing the normal thing gets wrong
    physics with no crash to notice.  That is the `xc_family_mismatch` case,
    and it blocks for the same reason: *silently* wrong is what a preflight
    is for.
    """
    entries = _found(_lib(tmp_path, S=S_V05), "S")
    assert [e.status for e in entries] == ["semilocal_only"], entries
    assert "semilocal_only" in ERROR_STATUSES
    msg = entries[0].message
    assert "ZERO" in msg and "not omitted" in msg, (
        "the message must say the projector is present and CLAIMS zero -- "
        "'missing' would send the reader looking for the wrong thing")


def test_a_PARTLY_zeroed_valence_channel_warns_rather_than_blocks(tmp_path):
    """v0.5's bismuth: p asks for two projectors and ships one real, one empty.

    A second projector is what makes a channel transferable across bonding
    environments, so this costs accuracy -- not the channel, which is why it
    is not the ERROR above.  Blocking would refuse six elements over a
    degradation their owner may knowingly accept; silence would hide a defect
    the file's own generator input contradicts.
    """
    entries = _found(_lib(tmp_path, Bi=BI_V05), "Bi")
    assert [e.status for e in entries] == ["partial_projectors"], entries
    assert "partial_projectors" not in ERROR_STATUSES
    info = parse_psml_header(BI_V05)
    assert "p" in info.partial_projector_channels
    assert info.semilocal_only_channels == []


# ===================================================================== #
#  Layer 2 -- the set + the config: what the files ask of the run       #
# ===================================================================== #

def _cutoff(lib, elements):
    struct = Structure(
        elements=elements,
        positions=np.array([[2.0 * i, 0.0, 0.0] for i in range(len(elements))]))
    return _declared_cutoff_ry(struct, SimpleNamespace(psml_lib=None),
                               dest_dir=lib)


def test_the_highest_requirement_wins_because_there_is_one_grid(tmp_path):
    """The mesh is a single global grid, so the most demanding element sets
    it.  A minimum or an average would answer LOWER the more species a
    system has, which is backwards: adding a species can only make the
    grid's job harder.  Bismuth states 36 Ry, sulfur 147."""
    lib = _lib(tmp_path, S=S_V05, Bi=BI_V05)
    assert _cutoff(lib, ["Bi", "S"]) == (147.0, "S")


def test_an_element_that_states_nothing_does_not_lower_the_bar(tmp_path):
    """Only the eleven re-generated elements carry hints, so a real system
    states fewer numbers than it has species.  Silence must abstain, not vote
    for a smaller number: asked with a silent species on each side of the one
    that states, in the order the species are read."""
    lib = _lib(tmp_path, S=S_V05, H=PSML_DIR / "H.psml", C=PSML_DIR / "C.psml")
    assert _cutoff(lib, ["H", "S", "C"]) == (147.0, "S")


def test_the_hints_are_read_whole_so_high_can_be_named():
    """`normal` is the threshold; `high` is what tight and vibrational work
    wants, so the message can offer it."""
    info = parse_psml_header(S_V05)
    assert info.cutoff_hints_ry == {"low": 72.0, "normal": 147.0, "high": 162.0}
    assert info.suggested_mesh_ry == 147.0, "the THRESHOLD is `normal`"


def test_a_file_that_states_nothing_leaves_the_field_None():
    """v0.4 files carry no hints at all, and that must read as *unknown*
    rather than as zero -- a zero requirement is satisfied by anything."""
    info = parse_psml_header(S_V04)
    assert info.cutoff_hints_ry == {}
    assert info.suggested_mesh_ry is None
