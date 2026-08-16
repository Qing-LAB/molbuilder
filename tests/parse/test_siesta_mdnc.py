"""The .MD.nc reader — and the trap it exists to avoid.

`docs/model/parse.md` § 6 gives the shape; what needs pinning here is the
BOOKKEEPING, because every mistake in it is silent:

  * a .MD.nc row mixes two steps — ``xa`` is the geometry about to be tried,
    every scalar on the same row describes the geometry just evaluated.  Pair
    them naively and every energy in the trajectory is off by one move, with
    a plausible-looking plot and nothing raised;
  * the file has no row for the input geometry, and it ACCUMULATES across
    restarts, so aligning it to the .out by index arithmetic is right on a
    fresh run and wrong on a warm one;
  * ``xa`` is Bohr while ``volume`` is Ang**3 IN THE SAME FILE, so a blanket
    unit assumption is wrong even when it looks consistent.

The fixture pair is one real H2 relaxation (SIESTA 5.4.2, molbuilder-siesta
env): ``offsetH2.MD.nc`` and the ``offsetH2.out`` written by the same run, so
the two can be checked against each other rather than against my arithmetic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder.frame import Frame
from molbuilder.parse import parse
from molbuilder.parse.engines.siesta_mdnc import (SiestaMdNcFileParser,
                                                  align_to_reference)
from molbuilder.structure import Structure

FIX = Path(__file__).resolve().parent / "fixtures" / "siesta_mdnc"
MDNC = FIX / "offsetH2.MD.nc"
OUT = FIX / "offsetH2.out"


def _need(p: Path) -> Path:
    assert p.exists(), (
        f"committed fixture missing: {p}.  It is versioned with this test; "
        f"a checkout without it is broken, not a reason to skip.")
    return p


def _frames(*bonds: float):
    """Synthetic 2-atom frames, one per bond length."""
    out = []
    for i, b in enumerate(bonds):
        out.append(Frame(
            structure=Structure(elements=["H", "H"],
                                positions=np.array([[0.0, 0.0, 0.0],
                                                    [b, 0.0, 0.0]])),
            step_index=i))
    return out


# ---- detection --------------------------------------------------- #


def test_it_claims_a_real_md_nc():
    assert SiestaMdNcFileParser.can_parse(_need(MDNC))


def test_it_declines_the_out_that_sits_beside_it():
    """The .out is the other half of the same run; claiming it would steal
    the file from the parser that can actually read run state and SCF."""
    assert not SiestaMdNcFileParser.can_parse(_need(OUT))


def test_it_declines_a_file_that_only_has_the_name(tmp_path):
    """Name alone is not evidence.  Without the magic-number check a text
    file called ``x.MD.nc`` would be claimed and then blow up in parse()."""
    fake = tmp_path / "notreally.MD.nc"
    fake.write_text("this is not netCDF\n")
    assert not SiestaMdNcFileParser.can_parse(fake)


def test_the_registry_routes_a_md_nc_here():
    """Registration, not just the class, is what makes it reachable."""
    res = parse(str(_need(MDNC)))
    assert res.parser_name == "siesta-mdnc"
    assert res.source_format == "siesta-mdnc"


# ---- what the file does and does not know ------------------------ #


def test_run_state_is_unknown_not_finished():
    """The file says nothing about how the run ended, and a restart appends
    to it -- so a complete-looking file may belong to a running job.
    Reporting 'finished' here would put a green badge on a live run."""
    assert parse(str(_need(MDNC))).run_state == "unknown"


def test_forces_and_scf_history_are_absent_rather_than_invented():
    res = parse(str(_need(MDNC)))
    assert res.frames, "fixture parsed to no frames"
    assert all(f.forces is None for f in res.frames)
    assert all(f.scf_history is None for f in res.frames)


def test_units_are_converted_from_the_attribute():
    """xa is Bohr in the file; a bond length read as Bohr would be 1.89x
    too long.  H2 relaxes near 0.75-0.95 Ang, i.e. ~1.4-1.8 Bohr."""
    res = parse(str(_need(MDNC)))
    for f in res.frames:
        b = float(np.linalg.norm(f.structure.positions[1]
                                 - f.structure.positions[0]))
        assert 0.5 < b < 1.3, f"bond {b} A -- unit conversion looks wrong"


# ---- THE trap: a row is not a step ------------------------------- #


def test_geometry_and_energy_in_one_frame_belong_to_the_SAME_step():
    """The regression this module exists for.

    ``xa[k]`` is the geometry about to be tried; ``etot[k]`` is the energy of
    the geometry BEFORE that move.  Pairing them by row index attaches every
    energy to the previous geometry -- silently, and with a plot that still
    looks like a converging relaxation.

    Checked against the .out from the SAME run: for every frame the reader
    matches to a .out frame, the energies must agree to the .out's own
    printing precision (4 decimals in eV).
    """
    nc = parse(str(_need(MDNC)))
    out = parse(str(_need(OUT)))
    mapping = align_to_reference(out.frames, nc.frames)
    compared = 0
    for i, j in enumerate(mapping):
        if j is None:
            continue
        e_out, e_nc = out.frames[i].energy, nc.frames[j].energy
        if e_out is None or e_nc is None:
            continue
        compared += 1
        assert abs(e_out - e_nc) < 5e-4, (
            f".out frame {i} and .MD.nc frame {j} are the SAME geometry but "
            f"report different energies ({e_out} vs {e_nc}).  The row/step "
            f"pairing regressed: xa[k] goes with etot[k+1].")
    assert compared >= 2, (
        f"only {compared} frame(s) had both energies -- the fixture should "
        f"give at least two, so a passing run is not vacuous")


def test_the_final_geometry_has_no_energy_yet():
    """The last row's geometry has not been evaluated when the file was
    written, so its energy is None.  Recycling the previous value would be
    a fabricated number in the one place a reader is most likely to look."""
    nc = parse(str(_need(MDNC)))
    assert nc.frames[-1].energy is None


def test_the_per_step_series_are_shifted_with_the_energy():
    """temperature / pressure / volume / eks describe the evaluated
    geometry too, so they move with etot rather than with xa."""
    nc = parse(str(_need(MDNC)))
    eks = nc.runtime_info.get("mdnc_eks_eV")
    assert eks is not None and len(eks) == len(nc.frames)
    # H2 at 0 K has no smearing entropy, so eks == etot on this fixture.
    for f, e in zip(nc.frames, eks):
        if f.energy is not None and e is not None:
            assert abs(f.energy - e) < 1e-9


# ---- alignment --------------------------------------------------- #


def test_alignment_finds_the_measured_lag_on_the_real_pair():
    nc = parse(str(_need(MDNC)))
    out = parse(str(_need(OUT)))
    mapping = align_to_reference(out.frames, nc.frames)
    # The .out opens with the INPUT geometry, which .MD.nc never stores.
    assert mapping[0] is None
    assert [j for j in mapping if j is not None] == [0, 1, 2, 3]


def test_the_input_geometry_is_never_lost(user_requirement="2026-08-15"):
    """THE STARTING FRAME MUST SURVIVE (user, 2026-08-15).

    .MD.nc holds only post-move geometries -- it has no row for the structure
    the user submitted.  So a design that took its frame list from the netCDF
    would drop the initial geometry from the results display, which is the
    one frame a reader compares everything else against.

    The rule that prevents it: the .out defines the frame list, and .MD.nc
    only UPGRADES frames it matches.  Expressed here as two properties of
    ``align_to_reference`` -- it returns exactly one entry per reference
    frame, and a reference frame with no counterpart yields None (keep the
    .out's own values) rather than being dropped.
    """
    out = parse(str(_need(OUT)))
    nc = parse(str(_need(MDNC)))
    mapping = align_to_reference(out.frames, nc.frames)
    assert len(mapping) == len(out.frames), (
        "alignment changed the frame count; the .out's frame list is the "
        "trajectory and must survive intact")
    assert mapping[0] is None, (
        "the input geometry matched a .MD.nc row -- it should not, and if "
        "this ever passes the fixture no longer exercises the case")
    # And the .out's own first frame still carries a real geometry + energy,
    # which is what the display shows for step 0.
    assert out.frames[0].structure.positions.shape == (2, 3)
    assert out.frames[0].energy is not None


def test_alignment_matches_by_geometry_not_by_index():
    """A warm restart appends to the existing .MD.nc, so the candidate list
    can carry a previous run's frames in front.  Index arithmetic gets this
    wrong; matching does not."""
    ref = _frames(1.00, 1.10, 1.20)
    stale = _frames(9.00, 9.10)                     # a previous run's tail
    candidate = stale + ref
    assert align_to_reference(ref, candidate) == [2, 3, 4]


def test_alignment_reports_a_miss_rather_than_guessing():
    ref = _frames(1.00, 5.55, 1.20)
    candidate = _frames(1.00, 1.20)
    assert align_to_reference(ref, candidate) == [0, None, 1]


def test_a_repeated_final_geometry_does_not_claim_one_row_twice():
    """A relaxation prints its final geometry again at the end.  Both .out
    frames match the same coordinates, and a naive matcher would map both
    onto the single .MD.nc row -- double-counting the last step."""
    ref = _frames(1.00, 1.20, 1.20)
    candidate = _frames(1.00, 1.20)
    assert align_to_reference(ref, candidate) == [0, 1, None]


def test_alignment_is_empty_when_there_is_nothing_to_align():
    assert align_to_reference([], _frames(1.0)) == []
    assert align_to_reference(_frames(1.0), []) == [None]


# ---- refusing rather than assuming ------------------------------- #


# ---- the merge: .out stays the trajectory ------------------------ #


def _merged(tmp_path, *, with_nc=True, extra_nc=False, corrupt=False):
    """A private copy of the fixture pair, parsed through the .out."""
    import shutil
    shutil.copy(_need(OUT), tmp_path / "offsetH2.out")
    if with_nc:
        dest = tmp_path / "offsetH2.MD.nc"
        shutil.copy(_need(MDNC), dest)
        if corrupt:
            dest.write_bytes(b"CDF\x01" + b"\x00" * 64)
    if extra_nc:
        shutil.copy(_need(MDNC), tmp_path / "otherrun.MD.nc")
    return parse(str(tmp_path / "offsetH2.out"))


def test_the_merge_upgrades_precision_without_reshaping_the_trajectory(tmp_path):
    plain = parse(str(_need(OUT)))          # fixture dir has the sibling too
    merged = _merged(tmp_path)
    assert len(merged.frames) == len(plain.frames)
    assert merged.runtime_info["mdnc_coords_upgraded"] >= 3
    assert merged.runtime_info["mdnc_energies_upgraded"] >= 3
    # An upgraded energy carries more significant digits than the .out's
    # four decimals -- that IS the upgrade.
    upgraded = [f.energy for f in merged.frames
                if f.energy is not None and abs(f.energy * 1e4
                                                - round(f.energy * 1e4)) > 1e-6]
    assert upgraded, ("no frame gained precision; the merge did not run or "
                      "took the text values")


def test_the_merge_keeps_the_input_geometry_as_step_0(tmp_path):
    """The user's requirement, at the level that actually ships it.

    .MD.nc has no row for the submitted structure, so the merged result must
    still open with the .out's own frame 0 -- geometry and energy intact."""
    merged = _merged(tmp_path)
    first = merged.frames[0]
    bond = float(np.linalg.norm(first.structure.positions[1]
                                - first.structure.positions[0]))
    assert abs(bond - 0.95) < 1e-6, (
        f"step 0 is {bond} A, not the 0.95 A input geometry -- the starting "
        f"frame was replaced or dropped")
    assert first.energy is not None


def test_the_merge_never_touches_what_only_the_out_knows(tmp_path):
    merged = _merged(tmp_path)
    plain = _merged(tmp_path, with_nc=False)
    assert merged.run_state == plain.run_state == "finished"
    for a, b in zip(merged.frames, plain.frames):
        assert (a.forces is None) == (b.forces is None)
        if a.forces is not None:
            assert np.allclose(a.forces, b.forces)
        assert (a.scf_history or []) == (b.scf_history or [])


def test_no_sibling_parses_exactly_as_before(tmp_path):
    """A SIESTA built without -DCDF, or a run with WriteMDhistory off,
    writes no .MD.nc at all.  That must cost nothing."""
    res = _merged(tmp_path, with_nc=False)
    assert len(res.frames) == 6
    assert res.run_state == "finished"
    assert not [k for k in res.runtime_info if k.startswith("mdnc_")]


def test_an_unreadable_sibling_is_recorded_not_raised(tmp_path):
    """A truncated netCDF must not cost the user their results -- the .out
    still describes the whole run."""
    res = _merged(tmp_path, corrupt=True)
    assert len(res.frames) == 6
    assert res.run_state == "finished"
    assert "mdnc_error" in res.runtime_info
    assert "mdnc_coords_upgraded" not in res.runtime_info


def test_two_candidates_are_left_alone_rather_than_guessed(tmp_path):
    """Two .MD.nc in one directory means two runs.  Picking one would risk
    attaching another run's geometry to this run's energies."""
    import shutil
    shutil.copy(_need(OUT), tmp_path / "siesta.out")      # name != label
    shutil.copy(_need(MDNC), tmp_path / "runA.MD.nc")
    shutil.copy(_need(MDNC), tmp_path / "runB.MD.nc")
    res = parse(str(tmp_path / "siesta.out"))
    assert not [k for k in res.runtime_info if k.startswith("mdnc_")]


def test_a_differently_named_out_still_finds_the_only_sibling(tmp_path):
    """`siesta.out` with `SystemLabel hemeC` writes `hemeC.MD.nc`; the stem
    will not match, and the single-candidate rule is what saves it."""
    import shutil
    shutil.copy(_need(OUT), tmp_path / "siesta.out")
    shutil.copy(_need(MDNC), tmp_path / "hemeC.MD.nc")
    res = parse(str(tmp_path / "siesta.out"))
    assert res.runtime_info.get("mdnc_source") == "hemeC.MD.nc"
    assert res.runtime_info["mdnc_coords_upgraded"] >= 3


def test_it_reads_the_same_data_without_netCDF4(monkeypatch):
    """The reader claims two interchangeable backends: netCDF4 when present,
    scipy.io otherwise.  scipy is a hard dependency of this project and
    netCDF4 is not, so the fallback is the path on any machine that installed
    molbuilder without the SIESTA extras -- and an untested fallback is a
    claim, not a feature."""
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "netCDF4":
            raise ImportError("simulated: netCDF4 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(__import__("sys").modules, "netCDF4", raising=False)

    viascipy = SiestaMdNcFileParser.parse(_need(MDNC))
    monkeypatch.undo()
    native = SiestaMdNcFileParser.parse(_need(MDNC))

    assert len(viascipy.frames) == len(native.frames)
    for a, b in zip(viascipy.frames, native.frames):
        assert a.structure.elements == b.structure.elements
        assert np.allclose(a.structure.positions, b.structure.positions)
        assert (a.energy is None) == (b.energy is None)
        if a.energy is not None:
            assert abs(a.energy - b.energy) < 1e-12


def test_an_unknown_unit_is_refused_not_assumed(tmp_path):
    """A future SIESTA writing Angstrom would otherwise be scaled by 1.89
    with nothing to show for it."""
    netCDF4 = pytest.importorskip("netCDF4")
    p = tmp_path / "weird.MD.nc"
    ds = netCDF4.Dataset(str(p), "w", format="NETCDF3_CLASSIC")
    ds.createDimension("xyz", 3)
    ds.createDimension("atom", 1)
    ds.createDimension("step", None)
    v = ds.createVariable("xa", "f8", ("step", "atom", "xyz"))
    v.unit = "furlongs"
    v[0, 0, :] = [0.0, 0.0, 0.0]
    e = ds.createVariable("etot", "f8", ("step",))
    e.unit = "Ry"
    e[0] = -1.0
    z = ds.createVariable("iza", "i4", ("atom",))
    z[0] = 1
    ds.close()
    with pytest.raises(ValueError, match="furlongs"):
        SiestaMdNcFileParser.parse(p)
