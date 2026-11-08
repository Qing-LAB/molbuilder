"""What a PySCF vibration's deck does with the settings that reach its several
mean fields -- read as Python, in the deck prep wrote -- and the kind's science
gate failing closed.

What a row can say of this deck, the science gate's findings at prep among it,
is a row of ``tests/data/the_deck.toml``.  Here is what only the program's
shape shows: which object a setting was applied to, whatever the deck calls it.
"""
from __future__ import annotations

import ast

import pytest


def _deck_tree(tmp_path, monkeypatch, **template):
    """The deck `jobset prep` wrote for a PySCF vibration of the road's H2,
    with ``template``'s values set in its template, parsed."""
    from support.road import (described, group_words, jobset, road_tree,
                              set_by_the_person, stage_file)
    bundle = described(road_tree(tmp_path, monkeypatch, engine="pyscf"),
                       calculation="vibration", engine="pyscf")
    set_by_the_person(bundle, template)
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    return ast.parse(stage_file(bundle, "freq", ".py").read_text())


@pytest.mark.parametrize("soscf", [True, False], ids=["on", "off"])
def test_the_mean_field_handed_to_the_relaxation_is_the_second_order_one(
        soscf, tmp_path, monkeypatch):
    """`engines/pyscf.md` § 7a's role table promises the `newton()` wrap at the
    vibration RELAXATION site; without it a `scf_soscf=true` run relaxes under
    DIIS while its equilibrium SCF runs Newton.

    Whichever name the deck gives it: the object the relaxation door takes
    (`_mb_relax`, `engines/pyscf.md` § 3) is rebound to its `newton()` first."""
    tree = _deck_tree(tmp_path, monkeypatch, scf_soscf=soscf)
    relax = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == "_mb_relax")
    given = ast.unparse(relax.args[0])
    wrapped = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and n.lineno < relax.lineno
               and ast.unparse(n) == f"{given} = {given}.newton()"]
    assert bool(wrapped) is soscf, (given, soscf, wrapped)


@pytest.mark.parametrize("method, spelled", [
    ("DFT", {"mf.xc": "'b3lyp'", "mf.grids.level": "4", "mf.disp": "'d3bj'"}),
    ("HF", {"mf.disp": "'d3bj'"}),
], ids=["a Kohn-Sham deck", "a Hartree-Fock deck"])
def test_the_level_of_theory_is_set_in_one_place_every_mean_field_goes_through(
        method, spelled, tmp_path, monkeypatch):
    """`engines/pyscf.md` § 7a, `engines/vibration.md` I9: the functional, grid
    and dispersion were spelled twice until 2026-08-21, the layout's for the
    optimization deck and hand constants in the vibration deck's constructions,
    free to disagree.  Whatever object it is on, the deck assigns each once, on
    either method -- HF has no functional and no grid, and takes the dispersion
    correction like any method -- in one function, and the function is called
    for the displaced and relaxing mean fields and for the equilibrium one.

    MUTATION THIS MUST FAIL AGAINST: the dresser emitted for DFT alone; a site
    that spells the functional itself, or skips the dresser."""
    tree = _deck_tree(tmp_path, monkeypatch, method=method, functional="b3lyp",
                      dispersion="d3bj", grid_level=4)

    def spellings(node):
        return {ast.unparse(n.targets[0]): ast.unparse(n.value)
                for n in ast.walk(node) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]).endswith(
                    (".xc", ".grids.level", ".disp"))}

    assigned = [ast.unparse(n.targets[0]) for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]).endswith(
                    (".xc", ".grids.level", ".disp"))]
    assert sorted(assigned) == sorted(spelled), assigned
    (dresser,) = [fn.name for fn in ast.walk(tree)
                  if isinstance(fn, ast.FunctionDef)
                  and spellings(fn) == spelled]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == dresser]
    assert len(calls) >= 2, f"{dresser} is called {len(calls)} time(s)"


@pytest.mark.parametrize("dispersion", ["d4", "none"])
def test_a_hartree_fock_deck_says_hartree_fock_where_it_says_its_method(
        dispersion, tmp_path, monkeypatch):
    """`engines/vibration.md` § 4.10 (V1.8): the level of theory is one
    answer, `is_dft`.  Described as Hartree-Fock with the functional changed
    to the hybrid PBE0, the deck's header and Methods paragraph name
    Hartree-Fock and no functional.  Dispersion is not a DFT question
    (`engines/pyscf.md` § 7a; user, 2026-09-28: "scientifically correct
    decision applied"): d4 reaches the deck in the dresser every construction
    calls, is named in the header and in the Methods paragraph with its own
    paper; "none" is plain Hartree-Fock.

    MUTATIONS THIS MUST FAIL AGAINST: the paragraph reading `cfg.functional`
    whatever the method; the dispersion dropped under Hartree-Fock."""
    tree = _deck_tree(tmp_path, monkeypatch, method="HF", functional="PBE0",
                      dispersion=dispersion, grid_level=3)
    head = ast.get_docstring(tree)
    methods = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "METHODS_TEXT")
    dresser = ast.unparse(next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        and n.name == "_mb_configure_theory"))
    assert "RHF (Hartree-Fock)" in head, head
    assert "Hartree-Fock (RHF)" in methods, methods[:600]
    for dft_word in ("PBE", "B3LYP"):
        assert dft_word not in methods, (dft_word, methods[:600])
    if dispersion == "d4":
        assert "Dispersion: d4" in head, head
        assert ("with the D4 dispersion correction [Caldeweyher2019]"
                in methods), methods[:600]
        assert "mf.disp = 'd4'" in dresser, dresser
    else:
        assert "Dispersion:" not in head, head
        assert "dispersion correction" not in methods, methods[:600]
        assert "mf.disp" not in dresser, dresser


def test_the_kind_gate_refuses_an_engine_it_has_no_science_for():
    """A check is never skipped in silence (science/validation.md F4).
    A config no engine's record names is refused at the gate on every
    kind -- it would otherwise come back with an EMPTY verdict, which every
    surface reads as 'checked, nothing found' -- and the vibration kind's
    own science refuses an engine it has no checks for by name.

    API-level, a refusal the road cannot reach: `molbuilder describe` takes no
    engine the table does not name, so a config of no engine is handed here."""
    import dataclasses

    import numpy as np

    from molbuilder.engine_table import EngineTableError
    from molbuilder.structure import Structure
    from molbuilder.validation import validate
    from molbuilder.validation.spectra import validate_vibration_kind

    @dataclasses.dataclass
    class OtherEngineConfig:
        basis: str = "x"

    water = Structure(elements=["O", "H", "H"],
                      positions=np.array([[0.0, 0.0, 0.119],
                                          [0.0, 0.757, -0.477],
                                          [0.0, -0.757, -0.477]]))
    for kind in ("vibration", "optimization"):
        with pytest.raises(EngineTableError, match="no engine's config"):
            validate(water, OtherEngineConfig(), calculation=kind)
    with pytest.raises(TypeError, match="vibration kind has no science"):
        validate_vibration_kind(water, OtherEngineConfig(), None,
                                engine="orca", state=None)
