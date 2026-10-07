"""Atom-identity binding (model/overview.md § 2): an atom a user labels is the
same physical atom (element + position) a rule selects after the label has been
through disk -- the place a wrong atom gets computed."""
import numpy as np
from molbuilder.structure import Structure


def _repeated_element_struct():
    """Five atoms with REPEATED elements, so identity cannot ride on the element.

    A test whose fixture cannot express the confusion is not testing against
    it. Three carbons here mean an index shift between the label and the
    selection lands on a DIFFERENT carbon and stays chemically plausible --
    exactly the failure that is silent.
    """
    els = ["C", "C", "N", "C", "O"]
    pos = np.array([[float(i), 0.0, 0.0] for i in range(5)])
    return Structure(elements=els, positions=pos,
                     cell=np.diag([50.0, 50.0, 50.0]),
                     regions={"lead": [0, 3], "device": [1, 2]})


def test_a_region_label_survives_the_sidecar_and_selects_the_same_atoms(tmp_path):
    """A `ByRegion` rule, after the labels have been written to a sidecar and
    read back, selects the SAME PHYSICAL ATOMS it selected before.

    THE FAILURE THIS CATCHES.  `regions` decides which atoms are computed --
    which are electrode and which are device, which are frozen. The write side
    (`Structure.metadata_to_dict` -> `molstruct.save`) and the read side
    (`molstruct.load` -> `apply_to_structure` -> `evaluate(ByRegion(...))`) are
    each covered on their own; an index shift introduced BETWEEN them is
    invisible there, and `model/overview.md` § 2 names index translation as exactly where an
    off-by-one happens.

    Contract: `model/overview.md` § 2 (the atom-identity binding) and
    `model/structure.md` § 2.2 (`apply_metadata_dict` is the single dict->struct
    authority).

    The assertion is on ELEMENT AND POSITION, never on the index: an index that
    round-trips while pointing at a different atom is the whole failure.
    """
    from molbuilder.sidecars import molstruct as msj
    from molbuilder.selection import ByRegion, evaluate

    src = _repeated_element_struct()
    want = {(src.elements[i], tuple(src.positions[i]))
            for i in sorted(evaluate(ByRegion("lead"), src))}
    assert len(want) == 2, "the fixture's own region is not two atoms"

    # --- through the real doors, not a hand-built dict ---------------------
    payload = msj.to_dict(src.metadata_to_dict(),
                          n_atoms_total=len(src.elements),
                          structure_hash="sha256:" + "0" * 64)
    path = msj.save(tmp_path / "probe.molstruct.json", payload)

    # A FRESH structure carrying the same atoms and no labels -- the reload
    # path, where the geometry comes from the .xyz and the metadata from here.
    reloaded = Structure(elements=list(src.elements),
                         positions=np.array(src.positions),
                         cell=np.diag([50.0, 50.0, 50.0]))
    assert not (reloaded.regions or {}), "the fresh structure is already labelled"
    msj.apply_to_structure(reloaded, msj.load(path))

    got = {(reloaded.elements[i], tuple(reloaded.positions[i]))
           for i in sorted(evaluate(ByRegion("lead"), reloaded))}
    assert got == want, (
        f"the region selects different physical atoms after the round trip:\n"
        f"  before {sorted(want)}\n  after  {sorted(got)}")

    # And the labels did not bleed: 'device' must not have absorbed 'lead'.
    other = {(reloaded.elements[i], tuple(reloaded.positions[i]))
             for i in sorted(evaluate(ByRegion("device"), reloaded))}
    assert other.isdisjoint(got), f"regions overlap after reload: {other & got}"
