"""`POST /api/modify/lattice-from-run` — the user's own bulk relax, measured.

`plans/modify-redesign-plan.md` § 3.3.  The radio it replaces ("Your bulk run")
had been **unreachable since it shipped**: its only home was a packaged column
that is `null` for all six metals and that nothing in the codebase writes, so
the control greyed itself out — correctly — forever.

The user's design put the value where it belongs: *"a `.xyz` or `.XV` result
where one single periodic lattice is correctly optimized with the same
pseudopotential/basis etc., but that is at user's hand … and the backend just
extracts the lattice from that result."*

So the division of labour is asserted here as much as the arithmetic: **two**
refusals, because guessing would be worse than stopping, and everything else a
note the user reads and overrules.
"""
from __future__ import annotations

import numpy as np
import pytest

A = 4.078
D_NN = A / np.sqrt(2.0)


def _au_supercell(nx=3, ny=3, nz=2, a=A, element="Au"):
    from molbuilder.structure import Structure
    base = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]) * a
    pos = np.vstack([base + np.array([i, j, k]) * a
                     for i in range(nx) for j in range(ny) for k in range(nz)])
    return Structure(elements=[element] * len(pos), positions=pos,
                     cell=np.diag([nx * a, ny * a, nz * a]))


@pytest.fixture()
def picker_root(tmp_path, monkeypatch):
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)
    return tmp_path


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


def _write(tmp_path, struct, name="Au-bulk.xyz"):
    from molbuilder.workingcopy_structure import StructureCodec
    path = tmp_path / name
    StructureCodec().write(struct, path)
    return path


def _ask(client, path, **extra):
    return client.post("/api/modify/lattice-from-run",
                       json={"path": str(path), **extra}).get_json()


class TestItAnswersFromTheAtoms:

    def test_a_relaxed_supercell_gives_the_lattice_constant(
            self, client, picker_root):
        """The whole point: a 72-atom 3×3×2 supercell — the shape a real lead
        relax has — answers with the CONVENTIONAL cubic edge, which is not any
        of its three cell edges."""
        path = _write(picker_root, _au_supercell())
        d = _ask(client, path)
        assert d["ok"] is True, d
        assert abs(d["a"] - A) < 1e-6, d["a"]
        assert abs(d["d_nn"] - D_NN) < 1e-6, d["d_nn"]
        assert d["coordination"] == 12
        assert abs(d["second_shell_ratio"] - np.sqrt(2.0)) < 1e-6
        assert d["n_atoms"] == 72
        assert d["element"] == "Au"

    def test_the_cell_edges_are_not_the_answer(self, client, picker_root):
        """Stated as its own test because it is the design decision: the box is
        3a × 3a × 2a, so every naive reading of the cell — a row, the smallest
        row, the cube root of the volume — gives a different wrong number."""
        struct = _au_supercell(3, 3, 2)
        d = _ask(client, _write(picker_root, struct))
        edges = [float(np.linalg.norm(row)) for row in struct.cell]
        assert all(abs(d["a"] - e) > 1.0 for e in edges), (
            f"a={d['a']} coincides with a cell edge {edges}")
        assert abs(d["a"] - float(np.linalg.det(struct.cell)) ** (1 / 3)) > 1.0

    def test_it_compares_against_the_literature_and_says_by_how_much(
            self, client, picker_root):
        """The derived line is the cross-check (§ 3.3).  Au's PBE reference is
        4.158 Å, so a 4.078 Å measurement must come back as about −1.9%."""
        d = _ask(client, _write(picker_root, _au_supercell()))
        said = " ".join(n["message"] for n in d["notes"])
        assert "PBE" in said and "experimental" in said, said
        pbe = [n for n in d["notes"] if "PBE" in n["message"]][0]
        assert "-1.9%" in pbe["message"], pbe["message"]
        assert pbe["level"] == "info", "2% from a reference is not a warning"


    def test_a_siesta_XV_reads_the_same_as_the_xyz(self, client, picker_root):
        """§ 3.3 names both readers, so both are driven.  A `.XV` is what a
        SIESTA relax actually leaves behind, and it must give the same answer
        as the same crystal written as extended XYZ — otherwise the route has
        two behaviours and the user has to know which file they picked."""
        # THE READER'S OWN CONSTANT, imported rather than retyped: a second
        # copy here would make the test measure the difference between two
        # Bohr radii instead of between two readers.  (It did, at 1.6e-6 Å,
        # which is how the five-spellings finding surfaced.)
        from molbuilder.parse.coords.siesta_xv import _ANGSTROM_PER_BOHR as BOHR
        struct = _au_supercell(2, 2, 2)
        pos_bohr = struct.positions / BOHR
        cell_bohr = np.asarray(struct.cell) / BOHR
        rows = ["  ".join(f"{v:.9f}" for v in row) + "   0.0 0.0 0.0"
                for row in cell_bohr]
        rows.append(f"  {len(pos_bohr)}")
        for p_ in pos_bohr:
            rows.append("  1  79  " + "  ".join(f"{v:.9f}" for v in p_)
                        + "  0.0 0.0 0.0")
        xv = picker_root / "Au.XV"
        xv.write_text("\n".join(rows) + "\n")

        from_xv = _ask(client, xv)
        from_xyz = _ask(client, _write(picker_root, struct, "same.xyz"))
        assert from_xv["ok"] is True, from_xv
        assert abs(from_xv["a"] - from_xyz["a"]) < 1e-6, (from_xv, from_xyz)
        assert abs(from_xv["a"] - A) < 1e-6
        assert from_xv["coordination"] == 12


class TestTheNotesTheUserReads:

    def test_a_slab_is_flagged_by_its_coordination(self, client, picker_root):
        """§ 3.3's first check.  A thin slab still yields the right `a`, so the
        answer is given — with a note saying the file is probably not the bulk
        crystal that was meant."""
        struct = _au_supercell(3, 3, 1)
        struct = type(struct)(elements=struct.elements, positions=struct.positions,
                              cell=np.diag([3 * A, 3 * A, 3 * A + 20.0]))
        d = _ask(client, _write(picker_root, struct, "Au-slab.xyz"))
        assert d["ok"] is True, d
        assert d["coordination"] != 12
        warned = [n for n in d["notes"] if "neighbours" in n["message"]]
        assert warned and warned[0]["level"] == "warn", d["notes"]
        assert "slab" in warned[0]["message"]

    def test_the_second_shell_mistake_reads_as_a_big_offset(
            self, client, picker_root):
        """§ 3.3: the one mistake anyone makes is a SECOND-shell pair, a factor
        1.414 out — and the comparison says so at once rather than the number
        passing silently."""
        d = _ask(client, _write(picker_root,
                                _au_supercell(a=A * np.sqrt(2.0)),
                                "Au-wrong.xyz"))
        pbe = [n for n in d["notes"] if "PBE" in n["message"]][0]
        assert pbe["level"] == "warn", pbe
        assert "+38" in pbe["message"] or "+39" in pbe["message"] \
            or "+41" in pbe["message"], pbe["message"]


    def test_a_file_too_big_to_measure_exactly_is_refused_not_sampled(
            self, client, picker_root):
        """The measurement is exact and O(n²), so the route caps it — and it
        REFUSES rather than sampling, because a silently truncated answer to
        "what is this crystal's lattice constant" is worse than being asked
        for a smaller cell."""
        big = _au_supercell(7, 7, 7)          # 1372 atoms
        r = client.post("/api/modify/lattice-from-run",
                        json={"path": str(_write(picker_root, big, "huge.xyz"))})
        assert r.status_code == 400
        said = r.get_json()["error"]
        assert "1372" in said and "smaller" in said, said


class TestTheTwoRefusals:

    def test_a_file_with_no_cell(self, client, picker_root):
        """No cell means no periodic images, and on a small cell the measured
        minimum is then simply wrong rather than absent — so this stops."""
        from molbuilder.structure import Structure
        struct = _au_supercell()
        bare = Structure(elements=struct.elements, positions=struct.positions)
        r = client.post("/api/modify/lattice-from-run",
                        json={"path": str(_write(picker_root, bare, "nocell.xyz"))})
        assert r.status_code == 400
        assert "no unit cell" in r.get_json()["error"]

    def test_more_than_one_element_with_none_named(self, client, picker_root):
        from molbuilder.structure import Structure
        struct = _au_supercell()
        mixed = Structure(elements=["Ag"] + list(struct.elements[1:]),
                          positions=struct.positions, cell=struct.cell)
        path = _write(picker_root, mixed, "alloy.xyz")
        r = client.post("/api/modify/lattice-from-run", json={"path": str(path)})
        assert r.status_code == 400
        assert "name which one" in r.get_json()["error"]
        # ...and naming one is the way through, on the same file.
        d = _ask(client, path, element="Au")
        assert d["ok"] is True and abs(d["a"] - A) < 1e-6

    def test_a_path_outside_the_picker_roots_is_refused(self, client, picker_root):
        """web-api.md § 2.1: the fence is at the route, before anything opens
        the file."""
        r = client.post("/api/modify/lattice-from-run", json={"path": "/etc/passwd"})
        assert r.status_code >= 400

    def test_an_element_the_file_does_not_hold(self, client, picker_root):
        d = _ask(client, _write(picker_root, _au_supercell()), element="Cu")
        assert d["ok"] is False and "holds no Cu" in d["error"]
