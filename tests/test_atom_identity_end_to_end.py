"""End-to-end atom-identity binding (model/overview.md § 2): the atom a user
freezes (internal 0-based) MUST be the same physical atom (element + position)
constrained in the generated engine input -- across BOTH engines, whose index
conventions differ (SIESTA .fdf 1-based; geomeTRIC $freeze 1-based; PySCF
mol.atom 0-based).  This is the catastrophic-wrong-atom guard."""
import re
import numpy as np
from molbuilder.structure import Structure
from molbuilder import engine_atom_index as eai


def _distinct_struct():
    # Distinct elements AND positions so (element, position) uniquely ids an atom.
    # Explicit large cell so render_fdf emits coords as-is (a cell-less struct is
    # auto-boxed + uniformly shifted -- identity-preserving but it would move the
    # absolute coords we assert on).
    els = ["H", "C", "N", "O", "F"]
    pos = np.array([[float(i), float(2 * i), float(3 * i)] for i in range(5)])
    return Structure(elements=els, positions=pos,
                     cell=np.diag([50.0, 50.0, 50.0]), pbc=[True, True, True])


# ---------------------------- SIESTA .fdf ---------------------------- #
def _fdf_coords(fdf):
    out, inblk = [], False
    for ln in fdf.splitlines():
        if "%block AtomicCoordinatesAndAtomicSpecies" in ln:
            inblk = True; continue
        if inblk and "%endblock" in ln:
            break
        if inblk:
            p = ln.split()
            if len(p) >= 3:
                out.append(np.array([float(p[0]), float(p[1]), float(p[2])]))
    return out


def _fdf_constrained_1based(fdf):
    idxs, inblk = set(), False
    for ln in fdf.splitlines():
        if "%block Geometry.Constraints" in ln:
            inblk = True; continue
        if inblk and "%endblock" in ln:
            break
        if inblk and ln.strip().startswith("position"):
            idxs.update(int(x) for x in ln.split()[1:])
    return idxs


def test_siesta_frozen_maps_to_correct_physical_atom():
    from molbuilder.siesta.input import render_fdf
    from molbuilder.config.siesta import SiestaConfig
    s = _distinct_struct()
    s.frozen_atoms = [1, 3]                       # freeze C (idx1) and O (idx3)
    fdf = render_fdf(s, SiestaConfig())
    coords = _fdf_coords(fdf)
    assert len(coords) == 5
    constrained = _fdf_constrained_1based(fdf)
    # render_fdf uniformly translates atoms (box-centering); identity is the
    # ORDER, invariant to that shift.  Derive the shift from atom 0, then verify
    # EVERY fdf line is the internal atom at that index (catches any reorder).
    shift = coords[0] - s.positions[0]
    for j in range(5):
        assert np.allclose(coords[j] - shift, s.positions[j]), (
            f"atom order not preserved at fdf line {j+1}")
    # Now the constraint must target the right physical atom.
    for i in s.frozen_atoms:
        eng = eai.siesta_atom_index(i)            # 0-based -> SIESTA 1-based
        assert eng in constrained, f"internal atom {i} not constrained"
        assert np.allclose(coords[eng - 1] - shift, s.positions[i]), (
            f"SIESTA constrains atom {eng} at {coords[eng-1]-shift} but internal "
            f"atom {i} is at {s.positions[i]} -- WRONG PHYSICAL ATOM")
    assert eai.siesta_atom_index(2) not in constrained   # N (idx2) not frozen


# --------------------------- PySCF / geomeTRIC ----------------------- #
_ATOM = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d{6,})\s+(-?\d+\.\d{6,})\s+(-?\d+\.\d{6,})\s*$")


def _pyscf_atoms(script):
    out = []
    for ln in script.splitlines():
        m = _ATOM.match(ln)
        if m:
            out.append((m.group(1),
                        np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])))
    return out


def test_pyscf_frozen_maps_to_correct_physical_atom():
    from molbuilder.pyscf.input import render_script
    from molbuilder.config.pyscf import PySCFConfig
    s = _distinct_struct()
    s.frozen_atoms = [1, 3]
    script = render_script(s, PySCFConfig(optimize=True, optimizer="geometric"))
    atoms = _pyscf_atoms(script)
    assert len(atoms) == 5, f"expected 5 atom lines, got {len(atoms)}"
    m = re.search(r"xyz ([\d,]+)", script)
    assert m, "geomeTRIC $freeze xyz line not found"
    frozen_1based = {int(x) for x in m.group(1).split(",")}
    for i in s.frozen_atoms:
        eng = eai.geometric_atom_index(i)         # 0-based -> geomeTRIC 1-based
        assert eng in frozen_1based, f"internal atom {i} not frozen"
        el, xyz = atoms[eng - 1]
        assert el == s.elements[i] and np.allclose(xyz, s.positions[i]), (
            f"geomeTRIC freezes atom {eng} ({el} at {xyz}) but internal atom "
            f"{i} is {s.elements[i]} at {s.positions[i]} -- WRONG PHYSICAL ATOM")
