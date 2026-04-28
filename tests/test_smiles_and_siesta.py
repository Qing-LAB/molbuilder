"""SMILES builder + siesta module smoke tests.

Skips RDKit-dependent parts cleanly if RDKit isn't installed.
"""

from __future__ import annotations

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    # ---- SMILES (needs RDKit) ---------------------------------------
    try:
        from molbuilder import build_from_smiles
        s = build_from_smiles("c1ccccc1", title="benzene")
    except ImportError as e:
        print(f"  (skip SMILES: RDKit not installed -- {e})")
        s = None

    if s is not None:
        # benzene = 12 atoms (6 C + 6 H)
        assert s.n_atoms == 12, s.n_atoms
        assert s.elements.count("C") == 6
        assert s.elements.count("H") == 6
        # All atoms in roughly the same plane (planar benzene)
        z_spread = s.positions[:, 2].max() - s.positions[:, 2].min()
        assert z_spread < 0.1, z_spread          # near-planar
        # Round-trip XYZ
        xyz = s.to_xyz()
        assert int(xyz.splitlines()[0]) == 12
        print(f"  SMILES OK: benzene = {s.n_atoms} atoms, "
              f"planar to {z_spread:.3f} A")

        # Bigger: BDT (1,4-benzenedithiol) = 6 C + 4 H + 2 S = 12 heavies + 4 H
        s2 = build_from_smiles("Sc1ccc(S)cc1", title="bdt")
        assert "S" in s2.elements
        assert s2.elements.count("S") == 2
        print(f"  SMILES OK: BDT = {s2.n_atoms} atoms")

    # ---- siesta.render_fdf (needs ASE; we already require it) -------
    from molbuilder.siesta import Config, render_fdf
    from molbuilder import build_dna

    dna = build_dna("ATGC")
    cfg = Config(
        system_name="test_dna", system_label="dna",
        kgrid=(4, 4, 1), mesh_cutoff=350.0, relax_type="none",
    )
    fdf = render_fdf(dna, cfg)
    assert "SystemName        test_dna" in fdf
    assert "MeshCutoff 350.0 Ry" in fdf
    assert "%block kgrid_Monkhorst_Pack" in fdf
    assert "MD.TypeOfRun" not in fdf, "relax_type='none' must drop MD block"
    # ChemicalSpeciesLabel for the 4 elements DNA contains: C, H, N, O, P
    for el in ("C", "H", "N", "O", "P"):
        assert f" {el}\n" in fdf, el

    # ---- siesta.convert end-to-end (XYZ -> FDF) ---------------------
    from molbuilder.siesta import convert
    with tempfile.TemporaryDirectory() as d:
        xyz_path = os.path.join(d, "dna.xyz")
        fdf_path = os.path.join(d, "out", "dna.fdf")
        dna.to_xyz(xyz_path)
        summary = convert(xyz_path, fdf_path, Config(
            system_label="dna4", kgrid=(2, 2, 1), relax_type="CG",
        ))
        assert summary["n_atoms"] == dna.n_atoms
        assert os.path.isfile(fdf_path)
        text = open(fdf_path).read()
        assert "SystemLabel       dna4" in text

    print(f"OK -- siesta module: rendered FDF for {dna.n_atoms}-atom DNA, "
          f"convert() round-trip works.")


if __name__ == "__main__":
    main()
