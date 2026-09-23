"""Cell/boundary preservation in the transport emitter + sidecar
round-trip (the hex-Au(111) fix)."""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.transport import TransportConfig
from molbuilder.sidecars import molstruct as msj
from molbuilder.structure import Structure
from molbuilder.parse.fdf import parse_fdf_params
from molbuilder.transport.transiesta import axis_vacuum, _emit_geometry

# A hexagonal Au(111)-like cell: a,b at 60 deg ~17.3 A; c (transport) 40 A.
HEX = np.array([[17.30, 0.0, 0.0],
                [8.65, 14.98, 0.0],
                [0.0, 0.0, 40.0]])


def _hex_device():
    d = 2.40
    elems, pos, L, B, R = [], [], [], [], []
    i = 0
    for k in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, k * d]); L.append(i); i += 1
    z0 = 4 * d
    for j in range(2):
        elems.append("S"); pos.append([1.0, 1.0, z0 + 1.5 + j * 1.8]); B.append(i); i += 1
    z1 = z0 + 1.5 + 2 * 1.8 + 1.5
    for k in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, z1 + k * d]); R.append(i); i += 1
    return Structure(elements=elems, positions=np.asarray(pos, float), cell=HEX,
                     regions={"L-electrode": L, "bridge": B, "R-electrode": R})


# ------------------------------------------------------------------ #
#  Structure + sidecar carry the cell                                #
# ------------------------------------------------------------------ #


def test_structure_cell_and_periodicity_defaults():
    """No cell means a vacuum box; a stated cell means a lattice.  The
    boolean view follows from the kind rather than being stored beside it
    (`pbc` was retired as a field 2026-09-22)."""
    s = Structure(elements=["C"], positions=np.zeros((1, 3)))
    assert s.cell is None
    assert s.axis_kind == ("isolated",) * 3 and s.pbc() == (False,) * 3
    s2 = Structure(elements=["C"], positions=np.zeros((1, 3)), cell=HEX)
    assert s2.cell.shape == (3, 3)
    assert s2.axis_kind == ("periodic",) * 3 and s2.pbc() == (True,) * 3


def test_structure_cell_validation():
    with pytest.raises(ValueError):
        Structure(elements=["C"], positions=np.zeros((1, 3)),
                  cell=np.zeros((2, 3)))


def test_a_degenerate_cell_is_reported_rather_than_unopenable():
    """Zero volume (two parallel vectors) must be said out loud -- but by the
    checker, not by refusing to build the object.

    This asserted that `Structure(...)` raised, until 2026-09-21.  § 8.2 says
    reading does not judge: a model that cannot HOLD a bad box cannot show one
    on the Cell page either, so a pair whose sidecar carried one could not be
    opened and therefore could not be fixed.  The refusal lives at the two
    doors that are about to act on the box -- the edit gate and the emitter --
    and both still refuse.  What must never happen is the box travelling
    QUIETLY, which is what this now pins.
    """
    from molbuilder.cell import resolve_and_check
    bad = np.array([[10.0, 0, 0], [10.0, 0, 0], [0, 0, 10.0]])
    s = Structure(elements=["C"], positions=np.zeros((1, 3)), cell=bad)
    assert s.cell is not None, "the box the user has to fix was dropped"
    _rc, issues = resolve_and_check(s)
    assert [i.where for i in issues] == ["cell.no_volume"]


def test_copy_translated_preserve_cell_and_axis_kind():
    s = Structure(elements=["C", "N"], positions=np.array([[0, 0, 0], [1.0, 0, 0]]),
                  cell=HEX, axis_kind=("periodic", "periodic", "isolated"))
    for clone in (s.copy(), s.translated([1.0, 2.0, 3.0]), s.centered()):
        assert clone.cell is not None and np.allclose(clone.cell, HEX)
        assert clone.axis_kind == ("periodic", "periodic", "isolated")
        assert clone.pbc() == (True, True, False)
    # copy is independent (no shared array)
    s.copy().cell[0, 0] = 999.0
    assert s.cell[0, 0] == pytest.approx(17.30)


def test_sidecar_cell_round_trip(tmp_path):
    d = msj.to_dict({"cell": HEX,
                     "axis_kind": ("periodic", "periodic", "isolated")},
                    n_atoms_total=1, structure_hash="0" * 32)
    assert d["axis_kind"] == ["periodic", "periodic", "isolated"]
    assert "pbc" not in d, "the retired duplicate must not be written again"
    p = tmp_path / "x.molstruct.json"
    msj.save(p, d)
    loaded = msj.load(p)
    s = Structure(elements=["C"], positions=np.zeros((1, 3)))
    msj.apply_to_structure(s, loaded)
    assert np.allclose(s.cell, HEX)
    assert s.axis_kind == ("periodic", "periodic", "isolated")
    assert s.pbc() == (True, True, False)


def test_sidecar_without_cell_is_nonperiodic():
    d = msj.to_dict(n_atoms_total=1, structure_hash="0" * 32)
    del d["cell"]                                     # simulate an old v3 file
    s = Structure(elements=["C"], positions=np.zeros((1, 3)))
    msj.apply_to_structure(s, d)
    assert s.cell is None


# ------------------------------------------------------------------ #
#  Emitter preserves the explicit cell verbatim                      #
# ------------------------------------------------------------------ #


def test_emitter_preserves_hex_cell_verbatim():
    fdf = "\n".join(_emit_geometry(_hex_device()))
    p = parse_fdf_params(fdf)
    assert np.allclose(p.cell_ang[0], [17.30, 0.0, 0.0])
    assert np.allclose(p.cell_ang[1], [8.65, 14.98, 0.0])   # NOT squared off


def test_a_PERIODIC_axis_with_no_cell_is_refused_not_fabricated():
    """The Au(111) case, and § 7 is unambiguous about it: "the box is NOT
    recoverable from atom extents (padding fabricates an orthorhombic box
    that severs the periodic gold)".  A rectangle cannot tile a 60 degree
    lattice, so the lead would be a different crystal from the device.

    This asserted the opposite until 2026-09-23 -- that a fabricated box is
    produced and flagged.  Flagging was not enough: `as_structure` wrapped
    such a box in an explicit `cell=` and the deck then printed "Explicit
    lattice preserved from the structure (NOT recomputed from atom
    extents)" over one that had been.  `Structure.resolve_cell` has always
    refused this; the emitter now agrees with it.

    What is NOT refused is an ISOLATED axis -- see the nanowire tests
    below.  That is a real electrode and its box IS derived.
    """
    dev = _hex_device()
    dev.cell = None
    with pytest.raises(ValueError, match="periodic.*states no cell"):
        _emit_geometry(dev)


def test_axis_vacuum_flags_transport_axis_gap():
    # atoms span z 0..~21 in a 40 A c-axis -> ~19 A z-vacuum
    dev = _hex_device()
    vac = axis_vacuum(dev.cell, dev.positions)
    assert vac[2] > 5.0
    fdf = "\n".join(_emit_geometry(dev))
    assert "transport axis (c) has vacuum" in fdf


# ------------------------------------------------------------------ #
#  Wizard preserves the hex lateral vectors                          #
# ------------------------------------------------------------------ #



# The two wizard tests below this line went with `electrode_wizard`
# (2026-09-17).  `_emit_geometry` above is the LIVE emitter -- `deck.py`
# reuses it for every rung -- so the hexagonal-cell checks still guard the
# deck a person actually gets.


# ------------------------------------------------------------------ #
#  An ISOLATED electrode — a nanowire/chain lead in vacuum           #
# ------------------------------------------------------------------ #

def _wire():
    """A two-atom lead with no transverse lattice: vacuum across, periodic
    along transport.  A real shape -- a nanowire or chain electrode."""
    return dict(elements=["Au", "Au"],
                positions=np.array([[0., 0, 0], [0., 0, 2.4]]))


def test_an_isolated_electrode_gets_the_TRANSPORT_vacuum_by_default():
    """15 Å per side, not the framework's 3 Å.

    A lead is what the self-energy is built FROM, so its periodic images
    must be electrostatically isolated or Sigma describes a wire coupled to
    its own copies.  `Structure.effective_vacuum` answers 3 Å where nobody
    chose -- right for a molecule in a box, five times too thin here -- so
    transport answers with its own default, the way the electrode's dense
    transport-axis k is a default rather than something to discover
    (`engines/transport.md` § 2a.7).
    """
    from molbuilder.transport.transiesta import _compute_cell_from_extents
    a, b, _c = _compute_cell_from_extents(Structure(**_wire()))
    assert (a, b) == pytest.approx((30.0, 30.0)), "2 x 15 A per side"


@pytest.mark.parametrize("vac,expect", [
    ((8.0, 8.0, 0.0), (16.0, 16.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0)),
])
def test_a_stated_vacuum_is_obeyed_verbatim(vac, expect):
    """The three states (`structure-periodicity.md` § 2): a number is used,
    `[0,0,0]` means no gap DELIBERATELY and is also used, and only UNSET
    reaches the default above.

    This emitter ignored the field entirely until 2026-09-23: someone who
    typed 8 Å on the Cell page got 15 Å in the deck, silently -- the
    'control that appears to do something and does not' shape § 3.2 keeps
    finding.
    """
    from molbuilder.transport.transiesta import _compute_cell_from_extents
    a, b, _c = _compute_cell_from_extents(Structure(**_wire(), vacuum=vac))
    assert (a, b) == pytest.approx(expect)


def test_the_transport_axis_is_never_padded_with_vacuum():
    """`resolve_cell`'s rule, and transport follows it: vacuum is meaningless
    on a transport axis because the device length is MATCHED, not padded
    (§ 6.2).  The box there is the atom span and nothing else.

    This answered `int(bbox_z + 2) + 1` until 2026-09-23 -- a rounding that
    honoured neither the vacuum nor the kind and had no source.
    """
    from molbuilder.transport.transiesta import _compute_cell_from_extents
    s = Structure(**_wire(), vacuum=(8.0, 8.0, 8.0),
                  axis_kind=("isolated", "isolated", "transport"))
    a, b, c = _compute_cell_from_extents(s)
    assert (a, b) == pytest.approx((16.0 + 0.0, 16.0)), "2 x 8 A across"
    assert c == pytest.approx(2.4), (
        "the atom span, with no padding: an 8 A vacuum on the transport "
        "axis must not lengthen the device")

