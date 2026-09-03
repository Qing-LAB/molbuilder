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

from molbuilder.pseudos import ERROR_STATUSES, check_coverage, parse_psml_header


#: The real ekb values read out of PseudoDojo v0.5's S.psml (2026-09-03).
_S_V05 = [("s", 6.77423395188), ("s", 0.542269569144),
          ("p", 0.0), ("p", 0.0),
          ("d", 0.0), ("d", 3.02242629395)]
#: v0.4's sulfur: every channel carries a real projector.
_S_V04 = [("s", 6.764), ("s", 0.574),
          ("p", 3.227), ("p", 0.887),
          ("d", -3.548), ("d", -0.987)]


def _psml(element, projectors, *, z, semilocal=(), shells=(), hints=None,
          creator="ONCVPSP-4.0.1+psml-4.0.1-76 (scalar-relativistic)"):
    """A PSML carrying the parts this layer reads.

    ``semilocal`` are the l-letters with an `<slps>` block, ``shells`` the
    OCCUPIED (n, l, occupation) rows, and ``hints`` the PseudoDojo cutoff
    annotation.  Everything is optional, because the point of several tests
    below is what happens when a file states nothing.
    """
    proj = "\n".join(
        f'<proj l="{l}" seq="{i + 1}" ekb="{ekb}" eref="0" type="oncv"/>'
        for i, (l, ekb) in enumerate(projectors))
    slps = "\n".join(f'<slps n="3" l="{l}" set="scalar_relativistic"/>'
                     for l in semilocal)
    sh = "\n".join(f'<shell n="{n}" l="{l}" occupation="{occ}"/>'
                   for n, l, occ in shells)
    hint_attrs = ""
    if hints:
        hint_attrs = " " + " ".join(f'cutoff_hint_{k}="{v}"'
                                    for k, v in hints.items())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<psml version="1.1" xmlns="http://esl.cecam.org/PSML/ns/1.1">
<provenance creator="{creator}"/>
<pseudo-atom-spec atomic-label="{element}" atomic-number="{z}"
 z-pseudo="{z}" relativity="scalar">
<annotation pseudo-energy="-10.1"{hint_attrs}/>
<valence-configuration total-valence-charge="6">
{sh}
</valence-configuration>
</pseudo-atom-spec>
<exchange-correlation><libxc-info>
<functional type="exchange" id="101"/>
<functional type="correlation" id="130"/>
</libxc-info></exchange-correlation>
<semilocal-potentials set="scalar_relativistic">
{slps}
</semilocal-potentials>
<nonlocal-projectors set="scalar_relativistic">
{proj}
</nonlocal-projectors>
</psml>"""


_S_SHELLS = [(3, "s", 2), (3, "p", 4)]          # sulfur's valence: 3s2 3p4
_S_HINTS = {"low": 72, "normal": 147, "high": 162}


# ===================================================================== #
#  Layer 1 — the file: present-but-zero is a claim, and it is false     #
# ===================================================================== #

def test_a_zeroed_valence_channel_the_semilocal_block_carries_is_named(tmp_path):
    """The v0.5 sulfur shape, exactly."""
    p = tmp_path / "S.psml"
    p.write_text(_psml("S", _S_V05, z=16, semilocal="spd",
                       shells=_S_SHELLS, hints=_S_HINTS))
    info = parse_psml_header(p)
    assert info.semilocal_only_channels == ["p"], (
        "sulfur's p projectors are both zero and an <slps l='p'> carries the "
        "channel -- that is the case that broke a real run")
    assert info.null_channels == [], (
        "C5 must stay quiet: it asks whether a channel is MISSING, and this "
        "one is present and lying.  A different question, a different check")


def test_an_UNOCCUPIED_zeroed_channel_is_not_named(tmp_path):
    """Sulfur's `d` is zeroed too, and d is not in 3s² 3p⁴.

    A dead channel nothing occupies is a representation choice with nothing
    riding on it.  Naming it would bury the one that matters in noise —
    which is the whole reason the valence configuration is read.
    """
    p = tmp_path / "S.psml"
    p.write_text(_psml("S", _S_V05, z=16, semilocal="spd",
                       shells=_S_SHELLS, hints=_S_HINTS))
    assert "d" not in parse_psml_header(p).semilocal_only_channels


def test_the_good_pseudo_says_nothing(tmp_path):
    """v0.4's sulfur, same element, same everything else."""
    p = tmp_path / "S.psml"
    p.write_text(_psml("S", _S_V04, z=16, semilocal="spd", shells=_S_SHELLS))
    info = parse_psml_header(p)
    assert info.semilocal_only_channels == []
    assert info.null_channels == []


def test_it_BLOCKS(tmp_path):
    """ERROR, by user ruling 2026-09-03.

    The file states something false about a channel carrying four of
    sulfur's six valence electrons, and nothing in it says to read the
    semilocal block instead — so a reader doing the normal thing gets wrong
    physics with no crash to notice.  That is the `xc_family_mismatch` case,
    and it blocks for the same reason: *silently* wrong is what a preflight
    is for.
    """
    (tmp_path / "S.psml").write_text(
        _psml("S", _S_V05, z=16, semilocal="spd", shells=_S_SHELLS,
              hints=_S_HINTS))
    entries = [e for e in check_coverage(["S"], tmp_path,
                                         expected_xc_family="GGA",
                                         expected_xc_authors="PBE")
               if e.status != "ok"]
    assert [e.status for e in entries] == ["semilocal_only"], entries
    assert "semilocal_only" in ERROR_STATUSES, (
        "a claim of zero on a valence channel must block -- the run would be "
        "wrong and would not say so")
    msg = entries[0].message
    assert "ZERO" in msg and "not omitted" in msg, (
        "the message must say the projector is present and CLAIMS zero -- "
        "'missing' would send the reader looking for the wrong thing")


# ===================================================================== #
#  Layer 2 — the set + the config: what the files ask of the run        #
# ===================================================================== #

def _lib(tmp_path, **files):
    d = tmp_path / "psml"
    d.mkdir(exist_ok=True)
    for name, text in files.items():
        (d / f"{name}.psml").write_text(text)
    return d


def test_the_highest_requirement_wins_because_there_is_one_grid(tmp_path):
    """The mesh is a single global grid, so the most demanding element sets
    it.  A minimum or an average would answer LOWER the more species a
    system has, which is backwards: adding a species can only make the
    grid's job harder."""
    from molbuilder.pseudos import parse_psml_header as ph
    lib = _lib(
        tmp_path,
        S=_psml("S", _S_V04, z=16, shells=_S_SHELLS,
                hints={"low": 72, "normal": 147, "high": 162}),
        C=_psml("C", _S_V04, z=6, shells=[(2, "s", 2), (2, "p", 2)],
                hints={"low": 30, "normal": 41, "high": 48}),
    )
    got = {el: ph(lib / f"{el}.psml").suggested_mesh_ry for el in ("S", "C")}
    assert got == {"S": 147.0, "C": 41.0}
    assert max(got.values()) == 147.0, "the strictest element sets the grid"


def test_an_element_that_states_nothing_does_not_lower_the_bar(tmp_path):
    """Only the eleven re-generated elements carry hints, so a real system
    states fewer numbers than it has species — BDT on gold states exactly
    one.  Silence must abstain, not vote for a smaller number."""
    from molbuilder.pseudos import parse_psml_header as ph
    lib = _lib(
        tmp_path,
        S=_psml("S", _S_V04, z=16, shells=_S_SHELLS, hints=_S_HINTS),
        Au=_psml("Au", _S_V04, z=79, shells=[(6, "s", 1)]),   # no hints
        H=_psml("H", _S_V04, z=1, shells=[(1, "s", 1)]),      # no hints
    )
    stated = [ph(lib / f"{el}.psml").suggested_mesh_ry
              for el in ("S", "Au", "H")]
    assert stated == [147.0, None, None]
    assert max(v for v in stated if v is not None) == 147.0


def test_the_hints_are_read_whole_so_high_can_be_named(tmp_path):
    """`normal` is the threshold; `high` is what tight and vibrational work
    wants, so the message can offer it."""
    p = tmp_path / "S.psml"
    p.write_text(_psml("S", _S_V04, z=16, shells=_S_SHELLS, hints=_S_HINTS))
    info = parse_psml_header(p)
    assert info.cutoff_hints_ry == {"low": 72.0, "normal": 147.0, "high": 162.0}
    assert info.suggested_mesh_ry == 147.0, "the THRESHOLD is `normal`"


def test_a_file_that_states_nothing_leaves_the_field_None(tmp_path):
    """v0.4 files carry no hints at all, and that must read as *unknown*
    rather than as zero — a zero requirement is satisfied by anything."""
    p = tmp_path / "S.psml"
    p.write_text(_psml("S", _S_V04, z=16, shells=_S_SHELLS))
    info = parse_psml_header(p)
    assert info.cutoff_hints_ry == {}
    assert info.suggested_mesh_ry is None
