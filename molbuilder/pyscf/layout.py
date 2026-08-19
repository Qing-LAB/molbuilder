"""PySCF's answers to the emission doors — `execution/script-preparation.md` § 4.2.

**Layout is engine knowledge, and it lives here as a table rather than as
control flow.**  The framework walks it, turns each catalogue name into a
:class:`~molbuilder.script_emit.Parameter`, and hands that to :func:`line`.  The
engine is never given a bare value to write, which is what makes *"a value is
written together with the reason it holds"* structural: the catalogue's note is
placed above whatever :func:`line` returns, by the framework, every time.

The catalogue's own ``group`` cannot drive this.  Its vocabulary is the FORM's
-- ``setup`` / ``stage`` / ``profile`` / ``budget`` / ``staging`` / ``output`` --
and one group cuts across several places in a script: ``stage`` alone holds
``scf_conv_tol``, ``grid_level`` and the six geometry-convergence knobs, which
belong in three different parts of the file.

**What is here and what is not.**  A PySCF script splits three ways, and only
the first is a per-parameter door:

* ``mf.*`` and ``mol.*`` **assignments** -- one catalogue item, one line.  These
  are here.
* ``gto.M(...)`` **constructor arguments** -- ``charge``, ``spin``, ``basis``,
  ``symmetry``, ``ecp`` are arguments inside one call, so they belong to the
  molecule block (door 3a) and not to a line of their own.
* **behaviour flags with no keyword at all** -- ``optimize``,
  ``save_optimized_xyz``, ``write_trajectory`` -- which generate control flow
  rather than a setting, and belong to the free-form body (door 3b).
"""
from __future__ import annotations

from typing import Optional

from ..script_emit import Parameter, Section

#: The SCF settings, in the order a reader meets them.  Names are CATALOGUE
#: names; the engine keyword each one writes is the catalogue's ``anchor`` and
#: is never spelled twice.
SCF_SECTION = Section(
    "SCF settings",
    ("scf_conv_tol", "scf_conv_tol_grad", "scf_max_cycle", "scf_init_guess",
     "level_shift", "diis_space", "damp"),
)


#: The DFT setup, in the order PySCF needs it applied to ``mf``.
DFT_SECTION = Section(
    "Functional, density fitting, dispersion",
    ("functional", "grid_level", "density_fit", "dispersion"),
)


#: The geometry-convergence ladder for **this rung**.  A PySCF ladder is N
#: decks and N jobs (`stages.md` § 1.1a), so one deck carries one rung's
#: targets and there is no list here to loop over.
GEOMETRY_SECTION = Section(
    "Geometry convergence (this stage)",
    ("geom_etol", "geom_grms", "geom_gmax", "geom_drms", "geom_dmax",
     "geom_max_steps"),
)

#: ``catalogue name -> the geomeTRIC keyword it becomes``.  The mapping is the
#: catalogue's own ``anchor``; this table exists only to spell the call, and is
#: read from the declaration rather than typed, so the two cannot disagree.
_GEOM_KWARG = {
    "geom_etol": "convergence_energy",
    "geom_grms": "convergence_grms",
    "geom_gmax": "convergence_gmax",
    "geom_drms": "convergence_drms",
    "geom_dmax": "convergence_dmax",
    "geom_max_steps": "maxsteps",
}


def geom_kwargs() -> "tuple":
    """The ``optimize(...)`` keyword lines, in the section's own order."""
    return tuple(
        f"        {_GEOM_KWARG[n]:<21} = _GEOM_{n[5:].upper()},"
        for n in GEOMETRY_SECTION.items)


def line(cfg, *, is_dft: bool):
    """**Door 2 — the engine's syntax, and there is one of it.**

    Returns ``(Parameter) -> str | None`` for EVERY item this engine lays out:
    the SCF settings, the DFT ones, and geomeTRIC's convergence targets.

    **It was three functions until 2026-08-18** -- ``line``, ``dft_line`` and
    ``geom_line`` -- and that is why the writer built a separate ``DeckSpec``
    per section: a spec carries ONE ``line``, so sections needing different
    syntax could not share one.  The framework was being worked around rather
    than used, and the whole-deck runner was unreachable as a result.  Three of
    the items depend on more than their own value, which is what the closure is
    for: the signature stays one-parameter-in, one-line-out, and context the
    engine needs is captured where the engine knows it.

    ``auxbasis`` is deliberately absent from every section: it is an *argument*
    of the density-fitting call rather than a line, the same category as the
    ``gto.M(...)`` arguments, so it is read here and emitted as part of that
    call.

    ``None`` means *not emitted for this configuration*, and it is the whole
    conditionality mechanism: no predicate table, and no ``if`` in the
    framework.  Three of the seven decline on their own terms --

    * ``scf_conv_tol_grad`` when it is not set, because PySCF derives it from
      the energy tolerance (``scf.hf.kernel``: ``if conv_tol_grad is None:
      conv_tol_grad = sqrt(conv_tol)``) and a line restating the derivation
      would be the script claiming a setting it does not make;
    * ``diis_space`` at PySCF's own 8, and ``damp`` at zero, because these are
      hard-SCF knobs and a tutorial script stays clean on the easy path.

    The exact spellings are load-bearing: ``mf.conv_tol`` is written with two
    spaces before its ``=`` because it is the first of a column that reads as a
    block, and the emitted text is what the tests and a human both read.
    """
    def _line(param: Parameter) -> Optional[str]:
        if not param.known or param.value is None:
            return None
        name, value = param.name, param.value
        return (_scf(name, value) or _dft(name, value) or _geom(name, value))

    def _dft(name, value):
        if name == "functional":
            return f'mf.xc = "{value}"' if is_dft else None
        if name == "grid_level":
            return f"mf.grids.level = {value}" if is_dft else None
        if name == "density_fit":
            if not value:
                return None
            aux = getattr(cfg, "auxbasis", None)
            return (f'mf = mf.density_fit(auxbasis="{aux}")' if aux
                    else "mf = mf.density_fit()")
        if name == "dispersion":
            if not value or str(value).lower() == "none" or not is_dft:
                return None
            return f'mf.disp = "{value}"'
        return None

    def _geom(name, value):
        """One target, as a named constant the single ``optimize(...)`` call
        reads -- so the value a reader wants to change sits beside the
        catalogue's explanation of what it does.  Nothing when this run does
        not relax: a deck that named convergence targets and never optimised
        would be claiming a setting it does not make."""
        if name not in GEOMETRY_SECTION.items:
            return None
        if not getattr(cfg, "optimize", False):
            return None
        return f"_GEOM_{name[5:].upper()} = {value!r}"

    def _scf(name, value):
        if name == "scf_conv_tol":
            return f"mf.conv_tol  = {value:.0e}"
        if name == "scf_conv_tol_grad":
            if not value or value <= 0:
                return None
            return f"mf.conv_tol_grad = {value:.0e}    # set explicitly"
        if name == "scf_max_cycle":
            return f"mf.max_cycle = {value}"
        if name == "scf_init_guess":
            return f'mf.init_guess = "{value}"'
        if name == "level_shift":
            return f"mf.level_shift = {value}" if value else None
        if name == "diis_space":
            # Only when bumped off PySCF's own default -- see the docstring.
            return f"mf.diis_space = {value}" if value != 8 else None
        if name == "damp":
            return f"mf.damp = {value}" if value else None
        return None

    return _line


def check_rules(text: str, struct=None, cfg=None):
    """PySCF's answer to *what must a finished deck of mine satisfy?*

    **It parses.**  A generated Python program that does not compile is a run
    that dies on the queue after the wait, and the failure is always the same
    shape: an escape mangled while building the emitter's own string.  That bug
    shipped four times in one day across two generators before anything read a
    produced file back.  ``ast.parse`` costs microseconds and is decisive.

    The remaining rules are the two facts a PySCF deck cannot be wrong about
    and still mean what it says: it must build a molecule, and its identity must
    be the one that was stamped -- the name every warm file is keyed by.
    """
    import ast

    from ..issues import Issue

    out = []
    try:
        ast.parse(text)
    except SyntaxError as exc:
        out.append(Issue(
            "error",
            f"the generated script does not parse: {exc.msg} "
            f"(line {exc.lineno})",
            where="deck.syntax"))
        return out          # nothing below is meaningful in a broken file
    if "gto.M(" not in text:
        out.append(Issue("error", "the script builds no molecule (no gto.M)",
                         where="deck.molecule"))
    label = getattr(cfg, "job_name", None)
    if label and f'JOB = "{label}"' not in text:
        out.append(Issue(
            "error",
            f"the script's JOB literal is not the identity it was written "
            f"for ({label!r}); the warm files are keyed by that name",
            where="deck.identity"))
    return out


#: Items whose effective value is not read from the attribute the catalogue
#: names.  ``charge`` / ``spin`` / ``basis`` / ``symmetry`` are ``gto.M(...)``
#: arguments, so their anchor is the constructor rather than a path; the
#: molecule keeps them under its own names.
_READBACK = {
    "net_charge":    "mol.charge",
    "spin":          "mol.spin",
    "basis":         "mol.basis",
    "symmetry":      "mol.symmetry",
    "max_memory_mb": "mol.max_memory",
}

def recorded_items():
    """**Every** catalogue item this engine declares, in catalogue order.

    Not a curated list.  The record is meant to answer *what was this run set
    to* without qualification, so it covers the whole vocabulary: the values a
    person changed, the ones left at the project default, and the ones that
    never reach the engine at all (a molbuilder-level flag like
    ``save_optimized_xyz`` still decides what the run produces).

    A hand-kept list would answer *what somebody remembered to add*, which is a
    different and much less useful question -- and would go stale on the first
    new item.
    """
    from .. import script_emit as _sc
    return [it.name for it in _sc.declarations(engine="pyscf")]


#: Anchors that name a CALL rather than a setting.  ``mf.newton()`` returns a
#: new solver object; reading ``mf.newton`` back would record a bound method,
#: which is not a value anybody asked about.  What it did is already visible in
#: ``scf_solver_class``.
_NOT_A_SETTING = {"mf.newton"}


def readback(param: Parameter):
    """The Python expression that reads this parameter's **effective** value.

    For most items it is the catalogue's own ``anchor`` -- ``mf.conv_tol``,
    ``mf.max_cycle``, ``mf.grids.level`` are already attribute paths, so the
    place to read the value back from is a fact the declaration already
    carries and this does not restate.  Only the constructor arguments need
    a table, and that table is five lines long.

    ``None`` when there is nothing to read -- a flag that generates control
    flow rather than setting anything.
    """
    if param.name in _READBACK:
        return _READBACK[param.name]
    for key in param.writes:
        if key in _NOT_A_SETTING:
            return None
        if key.startswith(("mf.", "mol.")):
            return key
    return None
