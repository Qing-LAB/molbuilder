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
import dataclasses
import itertools

import numpy as np

from molbuilder import script_emit as _sc
from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import spec_for
from molbuilder.structure import Structure

_STRUCT = Structure(
    elements=["O", "H", "H"],
    positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    vacuum=(8.0, 8.0, 8.0))

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


def _decks():
    """Every deck the generator emits over the code-shaping axes, the
    optimization deck's and the vibration deck's.

    A generator that refused EVERYTHING would make every consumer pass
    over nothing, so this raises if no combination rendered at all --
    the swallow is for individual refused combinations only."""
    yielded = 0
    for calculation, axes in (("optimization", _AXES),
                              ("vibration", _VIB_AXES)):
        for combo in itertools.product(*axes.values()):
            over = dict(zip(axes, combo))
            if over.get("spin_treatment") == "unrestricted":
                over["unpaired_electrons"] = 2   # water's even count
            try:
                cfg = dataclasses.replace(PySCFConfig(job_name="w"), **over)
                deck = str(_sc.render_deck(
                    spec_for(_STRUCT, cfg, calculation=calculation),
                    _STRUCT, cfg, verbose=cfg.verbose_comments))
            except Exception:        # a refused combination is not this
                continue             # test's subject; the gate owns that
            yielded += 1
            yield {"calculation": calculation, **over}, deck
    assert yielded, "every combination refused; the sweep rendered nothing"


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


def test_every_generated_deck_is_a_valid_program():
    """A deck that does not parse is a deck that cannot run, and the check
    gate reads lines rather than syntax -- so nothing else asks this."""
    broken = []
    for over, text in _decks():
        try:
            ast.parse(text)
        except SyntaxError as exc:
            broken.append(f"{over}: {exc.msg} at line {exc.lineno}")
    assert not broken, "generated decks that are not valid Python:\n  " \
                       + "\n  ".join(broken)


def test_no_emitted_function_rebinds_a_name_the_deck_imported():
    """**The rule.**  molbuilder's emitted machinery may not take a name the
    engine owns.

    Stated over every deck the generator can emit, because a rule that holds
    for the one deck someone happened to open is not a rule.  ``mol`` and
    ``mf`` are deliberately NOT covered: they are module-level assignments,
    not imports, and a callback receiving ``mol`` from the optimizer's
    environment is *supposed* to shadow it.
    """
    offenders = []
    n = 0
    for over, text in _decks():
        n += 1
        tree = ast.parse(text)
        imported = _imported(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for name in sorted(_bound_in_scope(fn) & imported):
                offenders.append(f"{fn.name}() rebinds imported {name!r} "
                                 f"[{over}]")
    assert n, "no decks were rendered -- the matrix is broken, not the rule"
    assert not offenders, (
        f"{len(offenders)} of {n} decks rebind a name the deck imports; "
        f"molbuilder's emitted code is prefixed (`_mb_`, `_`) so that it "
        f"cannot:\n  " + "\n  ".join(sorted(set(offenders))[:8]))


def test_each_import_from_the_bundle_is_bound_under_molbuilders_prefix():
    """The convention that makes the rule above hold by construction:
    **each import from the bundle is bound as ``_mb_<its name>``**
    (`engines/pyscf.md` § 3), so an import of ours can never take a name the
    engine owns.  Checked over every deck, so the prefix stays a rule rather
    than a habit that decays one import at a time; the bundle's members are
    read off `runwrap.PYSCF_COMPANIONS`, the list prep ships.

    MUTATION THIS MUST FAIL AGAINST: an emitter writing its own
    ``from relax_policy import relax`` beside the generated import lines.
    """
    from molbuilder.runwrap import PYSCF_COMPANIONS
    members = {name[:-len(".py")] for name in PYSCF_COMPANIONS}
    stray = set()
    for _, text in _decks():
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.ImportFrom) and n.module in members:
                if any(a.asname != f"_mb_{a.name}" for a in n.names):
                    stray.add(ast.unparse(n))
            elif isinstance(n, ast.Import):
                if any(a.name in members and a.asname != f"_mb_{a.name}"
                       for a in n.names):
                    stray.add(ast.unparse(n))
    assert not stray, (
        f"imports from the bundle not bound as _mb_<its name>, so they can "
        f"take a name the engine owns: {sorted(stray)}")
