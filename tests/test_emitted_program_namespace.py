"""A PySCF deck is a PROGRAM, and its namespace has an owner per name.

**Why this is one test and not one per setting.** The generator is a state
machine: it walks a declared layout and substitutes engine hooks at fixed
points. Data that goes through the walk cannot break this rule — the walk
writes `name = value` lines and nothing else. What CAN break it is the other
kind of variation: the code the generator *emits*, which introduces scopes of
its own. So the rule is stated once, over everything the machine can emit, and
the hook that emits code is what it is really about.

The namespace, from the deck itself rather than from a rule someone invented:

* **The engine's, unprefixed** — ``gto``, ``scf``, ``dft``, ``geometric_solver``,
  ``mol``, ``mf``. This is the vocabulary a person reads the deck in and edits.
* **molbuilder's, prefixed** — ``_mb_socket``, ``_os``, ``_cp``,
  ``_pyscf_lib``, and every import from ``mb_pyscf.pyz`` as ``_mb_<its name>``
  (`engines/pyscf.md` § 3). Prefixed *so that* molbuilder's machinery can
  never take a name the engine owns.

The convention is already in the code. What was missing is anything that made
it hold: ``molwatch``'s optimizer callback bound a bare ``scf``, rebinding
``from pyscf import gto, scf, dft`` to a list of dicts for the rest of that
function — in half of every configuration that optimises geometry. Nothing in
the function read the module, so nothing failed. That is the state in which the
next edit inside it reaches for ``scf.RHF``, gets a list, and the calculation
keeps running.
"""
from __future__ import annotations

import ast
import builtins
import itertools

import pytest

from support.road import (WATER, described, group_words, jobset, read_request,
                          road_tree, set_by_the_person, stage_file,
                          write_request)
from molbuilder.calculation import open as open_calculation

#: The axes that change WHICH CODE is emitted — not which values it carries.
#: A value axis (``mesh_cutoff``) cannot affect a namespace; an axis that turns
#: a callback, a solvent wrapper or a GPU shim on and off can.
_AXES = {
    "optimize":            [True, False],
    "density_fit":         [True, False],
    "solvent":             [None, "water"],
    # The theory and the spin treatment both change which class, module and
    # stability block are emitted (`pyscf/layout.scf_class`).
    "method":              ["DFT", "HF"],
    "spin_treatment":      ["restricted", "unrestricted"],
    "use_gpu":             [False, True],
}

#: The same for the vibration deck, whose code turns on its own settings: the
#: two spectra, the per-mode probe, the theory, the GPU shim.
_VIB_AXES = {
    "compute_ir":          [True, False],
    "compute_raman":       [True, False],
    "es_mode_selection":   ["skip", "all"],
    "method":              ["DFT", "HF"],
    "use_gpu":             [False, True],
}


def _decks(tmp_path, monkeypatch, calculation, stage, axes):
    """Every deck `jobset prep` writes over the code-shaping axes, one
    calculation prepared again for each -- the template's values as a person
    sets them, the GPU's on the run card, their home -- each what prep takes:
    a combination skipped here is a deck the rule no longer asks about, which
    is how half this sweep once went unasked, every GPU deck refused for a
    count it did not state."""
    # THE ROAD'S WATER, closed shell and free: an even count an open-shell
    # axis can hold two unpaired electrons on.
    bundle = described(road_tree(tmp_path, monkeypatch, WATER, engine="pyscf"),
                       calculation=calculation, engine="pyscf",
                       stage_strategy="")
    template = open_calculation(bundle).template.path
    as_described, request = template.read_text(), read_request(bundle)
    for combo in itertools.product(*axes.values()):
        over = dict(zip(axes, combo))
        card = {"use_gpu": over["use_gpu"],
                **({"gpu_count": 1} if over["use_gpu"] else {})}
        values = {k: v for k, v in over.items()
                  if k != "use_gpu" and v is not None}
        if over.get("spin_treatment") == "unrestricted":
            values["unpaired_electrons"] = 2   # water's even count
        template.write_text(as_described)
        write_request(bundle, {**request,
                               "execution": {**request["execution"], **card}})
        set_by_the_person(bundle, values)
        r = jobset("prep", "task", *group_words(bundle, stage),
                   "--bundle", bundle, "--target", "this")
        assert r.exit_code == 0, (over, r.output)
        yield over, stage_file(bundle, stage, ".py").read_text()


def _imported(tree):
    return {(a.asname or a.name).split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}


def _bound_in_scope(fn):
    """Names this function binds: assignments, loop targets, and parameters.

    Not descending into nested functions -- those are their own scope and are
    visited separately by the ``ast.walk`` in the test.
    """
    out = set()
    for x in ast.walk(fn):
        if isinstance(x, ast.Assign):
            for t in x.targets:
                out |= {nd.id for nd in ast.walk(t)
                        if isinstance(nd, ast.Name)
                        and isinstance(nd.ctx, ast.Store)}
        elif isinstance(x, ast.For):
            out |= {nd.id for nd in ast.walk(x.target)
                    if isinstance(nd, ast.Name)}
    out |= {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    return out


def _read_and_never_bound(tree):
    """The names a deck reads that nothing in it binds, scope aside: each a
    NameError waiting for the line that reads it to run."""
    bound = set(dir(builtins)) | {"__file__", "__name__"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
             and isinstance(n.ctx, ast.Load)} - bound - _imported(tree))


@pytest.mark.parametrize("calculation, stage, axes", [
    ("optimization", "coarse", _AXES),
    ("vibration", "freq", _VIB_AXES),
], ids=["optimization", "vibration"])
def test_every_deck_keeps_to_its_own_namespace(
        calculation, stage, axes, tmp_path, monkeypatch):
    """**The rule.**  molbuilder's emitted machinery may not take a name the
    engine owns, and a deck reads no name it never bound.

    Stated over every deck the generator can emit, because a rule that holds
    for the one deck someone happened to open is not a rule.  ``mol`` and
    ``mf`` are deliberately NOT covered by the first half: they are
    module-level assignments, not imports, and a callback receiving ``mol``
    from the optimizer's environment is *supposed* to shadow it.

    The second half is E-M4.7's shape: an HF deck imports no ``dft``, its
    method a render-time fact, and the original bug read that name with
    ``force_cpu=True`` AFTER the full Hessian was paid for."""
    offenders, unbound = [], []
    n = 0
    for over, text in _decks(tmp_path, monkeypatch, calculation, stage, axes):
        n += 1
        tree = ast.parse(text)
        imported = _imported(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for name in sorted(_bound_in_scope(fn) & imported):
                offenders.append(f"{fn.name}() rebinds imported {name!r} "
                                 f"[{over}]")
        unbound += [f"{name!r} [{over}]" for name in _read_and_never_bound(tree)]
    assert not offenders, (
        f"{len(offenders)} of {n} decks rebind a name the deck imports; "
        f"molbuilder's emitted code is prefixed (`_mb_`, `_`) so that it "
        f"cannot:\n  " + "\n  ".join(sorted(set(offenders))[:8]))
    assert not unbound, (
        "decks read names nothing binds, a NameError on the line that does:"
        "\n  " + "\n  ".join(sorted(set(unbound))[:8]))
