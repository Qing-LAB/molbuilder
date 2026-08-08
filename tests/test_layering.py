"""Enforce the L1 / L2 / L3 import-direction invariant from
``docs/design.md`` § "Layout".

Why this test exists
====================

The design doc commits to a strict layering:

  * L1 (core types) -- no imports from L2 or L3
  * L2 (domain verbs) -- may import L1; no imports from L3
  * L3 (surfaces: cli, web) -- may import everything

Nothing else in the suite checks this.  Without an enforcement test, a
helper added to ``structure.py`` (L1) that reaches into
``builders.backends`` (L2) for an "is_available" probe would land
silently, the layer boundary would erode, and circular-import
problems would start surfacing as elusive runtime errors at startup.

How it works
============

For every ``.py`` file under ``molbuilder/``, parse the AST, collect
its top-level ``import X`` / ``from molbuilder.X import ...``
statements, classify the importing module by its layer, and assert
that every import target is in the same layer or lower.  Test
modules + the ``__init__.py`` re-export sites are exempted (the
public API legitimately re-exports L2 verbs from L1's package
namespace).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


# --------------------------------------------------------------------- #
#  Layer assignment                                                     #
#                                                                       #
#  Each entry is a top-level module / subpackage name under             #
#  ``molbuilder/``.  Keep in declaration order to match design.md so a  #
#  future reader can cross-reference cleanly.                           #
# --------------------------------------------------------------------- #


_L1_MODULES = {
    "structure", "frame", "issues",
    "cell",              # the ONE cell resolver + checker (cell-plan.md § 6a).
                         # L1 because it imports ``structure`` and ``issues``
                         # and nothing else -- which is what lets the L1 gate,
                         # the L2 validators and the L2 emitter all ask the
                         # same question of it instead of each answering it.
    "periodicity_gate",  # the frame-contract gate (structure-periodicity.md
                         # 6.1/6.2): defaults + validation for cell / origin /
                         # vacuum / axis kinds.  L1 because it imports
                         # ``structure`` and nothing else, and both the L2
                         # codec seam and the L3 web door call it.
    "chemistry", "residues",
    "config",            # siesta / pyscf / spectra / transport config dataclasses
    "trajectory_log",    # format + emitter (data-shape, no domain logic)
    "selection",         # atom-selection rule dataclasses + evaluator (no domain deps)
    "runtime_info",      # cross-cutting threading / GPU / runtime-info emitters
                         # (string emitters + physical_core_count -- L1 because no
                         # domain deps; siesta + pyscf + spectra + runwrap all use it)
    "pseudos",           # PSML pseudopotential header parser + coverage check
                         # (pure XML parsing + dataclass; no domain deps)
    "checkpoint",        # run-dir checkpoint/restart helpers; L1 because the
                         # L1 ``config`` dataclasses (siesta/pyscf) import it,
                         # so it must sit at or below L1 (no domain deps).
    "annotations_fdf",   # fdf emit-strategy registry for extensible atom-
                         # annotation channels; pure (typing only, struct is
                         # duck-typed) -- siesta/input consumes it.
    "engine_atom_index", # single-point 0-based -> per-engine atom-index
                         # translation (pure int math); the engine translators
                         # consume it.


    "persist",           # versioned-document helpers (@major schema check +
                         # JSON IO); pure stdlib, no domain deps -- bench +
                         # jobset persisted artifacts share it.
    "identity",          # the run id: normalise once, build from inputs
                         # (execution/run-identity.md 2-3).  L1 on stdlib
                         # alone, and that is load-bearing: the CLI, the web
                         # tab and the codec must all reach ONE normaliser or
                         # the id a surface shows differs from the one the
                         # engine writes -- 3's rule 1.
    "task",              # the task.json codec (engines/stages.md 6) -- one
                         # calculation's description: engine name, shape,
                         # base, varies, stages.  L1 because it imports only
                         # ``persist``, and that is load-bearing rather than
                         # incidental: BOTH surfaces must read a description
                         # through this one module, so it has to sit below
                         # the engines rather than beside one.  The half of
                         # the 6.6 preflight that needs an engine's field
                         # schema belongs to resolution, not here.
    "reload_protocol",   # the two constants the supervisor and its child agree
                         # on (exit code + env flag).  L1 for a reason the
                         # design depends on: the SUPERVISOR reads them, and it
                         # must never import application code -- a child that
                         # fails to import has to leave the parent alive so the
                         # next reload can fix the mistake.  It first lived at
                         # ``web/reload_protocol.py``, where reading it ran
                         # ``web/__init__.py`` and pulled in the whole app plus
                         # Flask (docs/ops/server-reload-plan.md 3.3).
}

_L2_MODULES = {
    "peptide", "nucleic", "smiles", "pubchem",
    "modify", "validation",
    "workingcopy_structure",   # structure+sidecar codec for the workingcopy
                               # core (reuses structure L1 + sidecars L2)

    "builders",          # backends/* package
    "backends",          # back-compat shim re-exporting builders.backends
    "template",          # an engine config as a readable .fdf.template
                         # (job-contracts.md § 3.7).  L2 because it imports the
                         # L1 config dataclasses and ``script_emit``, and is
                         # imported by the siesta producers and (later) prep --
                         # a config-shaped verb, not a core type.
    "siesta", "pyscf",   # script generators
    "spectra",           # spectra engines + script renderers
    "transport",         # transport engines + results (Phase B.2)
    "parse",             # unified parse module (Phase A+B+C+D shipped 2026-06-19;
                         # composes engine/sidecar FileParsers + JobDirParser into
                         # typed ParseResult outputs.  L2 because it composes
                         # per-engine parsers + ScriptSourceTextParser.)
    "projects",          # filesystem layout / naming rules
    "runtime_config",    # molbuilder.json reader
    "diagnostics",       # capabilities snapshot
    "envs",              # subprocess dispatch
    "runwrap",           # bash-wrapper emitter
    "monitor",           # background job-monitor + notifier hooks (PoC,
                         # § 11.0b); stdlib-only, parses run artifacts
    "auth_setup",        # interactive auth/secret bootstrap wizard; L2 --
                         # imported only by runtime_config (L2) + cli (L3),
                         # imports no molbuilder domain modules
    "data",              # bundled JSON tables
    "bundle_writer",     # write-side bundle materializer: composes BundleResult
                         # + Structure + dirs/parse.  L2 because it knows
                         # engine-specific final-coords sources (.XV, .opt.xyz).
    "script_emit",       # write-side emit_* helpers rehomed from script_contract
                         # in H2.emit.  L2 because emitter helpers consume
                         # parse.types (TrajectoryResult / ParseResult).
    "sidecars",          # write-side .molstruct/.spectra/.transport JSON helpers
                         # rehomed from parsers in H2.sidecars.  L2 because the
                         # canonical dict builders consume parse.types.
    "bench",             # bench subcommand machinery -- consumes siesta config +
                         # runwrap; generates BENCH-MARKS-aware sweeps.
    "jobset",            # engine-agnostic staged-execution framework
                         # (model / materialize / prep / plan / submit /
                         # runstatus + the jobset CLI group).  L2: prep ->
                         # runwrap, submit -> runtime_config, runstatus ->
                         # parse -- all L2; the model is pure stdlib.  See
                         # docs/execution/job-system.md.
}

_L3_MODULES = {
    "cli",
    "web",
}


# Modules whose name we don't classify (top-level files inside the
# package that exist for other reasons).  __init__.py is treated as
# special further down because it legitimately re-exports.
_UNCLASSIFIED_FILES = {
    "__init__.py",
    "__main__.py",
}


def _module_layer(rel_path: Path) -> str | None:
    """Classify a file relative to ``molbuilder/`` into ``L1`` / ``L2`` /
    ``L3``.  ``__init__.py`` / ``__main__.py`` and unrecognised
    locations return None and are skipped by the test."""
    if rel_path.name in _UNCLASSIFIED_FILES:
        return None
    # Top-level name (either the parent dir for a subpackage, or the
    # filename stem for a top-level module).
    parts = rel_path.parts
    head = parts[0] if len(parts) > 1 else rel_path.stem
    if head in _L1_MODULES:
        return "L1"
    if head in _L2_MODULES:
        return "L2"
    if head in _L3_MODULES:
        return "L3"
    return None


def _import_targets(tree: ast.AST) -> set[str]:
    """Collect every ``molbuilder.<head>...`` import target as the
    top-level ``<head>``.  Imports of stdlib / third-party packages
    are filtered out -- only intra-package imports matter."""
    heads = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("molbuilder."):
                    heads.add(name.split(".")[1])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "molbuilder":
                # `from molbuilder import X` -- X can be any layer's
                # re-exported public symbol; treat as not-an-import-of-
                # a-specific-submodule (the __init__.py is the gate).
                continue
            if mod.startswith("molbuilder."):
                heads.add(mod.split(".")[1])
            elif mod == "" and node.level > 0:
                # Relative import: `from .X import Y` or `from ..X import Y`.
                # We don't try to resolve these to absolute layers; the
                # surrounding package boundary already enforces locality.
                continue
    return heads


def _all_python_files() -> list[Path]:
    pkg_root = Path(__file__).resolve().parent.parent / "molbuilder"
    found = []
    for dirpath, dirnames, filenames in os.walk(pkg_root):
        # Skip __pycache__ directories.
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py"):
                found.append(Path(dirpath, fn).relative_to(pkg_root))
    return sorted(found)


# Permitted exceptions: re-export sites that legitimately reach
# upward.  Each entry is the relative path within molbuilder/ and a
# short rationale -- adding to this set is the explicit "I know this
# crosses the layer line and that's intentional" signal.
_EXEMPT = {
    # The package __init__ is the public API: it re-exports L2 verbs
    # under the molbuilder.* namespace so callers can `import molbuilder`.
    Path("__init__.py"),
    # cli/web entry shims: re-export from web.app / cli for back-compat.
    Path("__main__.py"),
}


# --------------------------------------------------------------------- #
#  The test                                                             #
# --------------------------------------------------------------------- #


def _judged_files() -> list[Path]:
    """The files the layering rule actually applies to.

    A package ``__init__.py`` is excluded BY DESIGN: it is the public API
    surface and re-exports across layers on purpose (``import molbuilder``
    must hand you L2 verbs from the L1 package namespace).  ``_EXEMPT`` names
    the non-``__init__`` re-export shims.

    These used to be parametrized in and then ``pytest.skip``-ed one by one --
    25 skips in a suite whose other skips are real ("no node", "no pyscf in
    this env"), so the noise hid them.  Worse, the skip was reached through
    ``_module_layer(...) is None``, which means a NON-init file that stopped
    being classifiable would have gone quiet too.  Excluding them here and
    asserting the exclusion below turns 25 silent skips into one real check.
    """
    return [p for p in _all_python_files()
            if p.name != "__init__.py" and p not in _EXEMPT]


def test_only_package_inits_sit_outside_the_layer_rule():
    """Nothing escapes the rule by accident.

    Two ways it could: a named exemption that no longer names a real file
    (so the exemption outlived its subject), or a judged module the layer map
    cannot classify (so the rule silently did not apply).  Both are drift
    between the map and the tree, and both should FAIL rather than skip.

    A first cut also asserted that every unjudged file was an ``__init__`` or
    an exemption -- which is how ``_judged_files`` defines "unjudged", so it
    could never fail.  Deleted rather than kept as decoration.
    """
    # Every named exemption must still name a real file.  A stale entry
    # exempts nothing and quietly survives the rename that outdated it, so
    # the next reader trusts a guarantee that no longer has a subject.
    present = set(_all_python_files())
    stale = sorted(str(p) for p in _EXEMPT if p not in present)
    assert not stale, (
        "_EXEMPT names files that are not in the tree any more; drop them:\n  "
        + "\n  ".join(stale))

    unclassifiable = [p for p in _judged_files() if _module_layer(p) is None]
    assert not unclassifiable, (
        "these modules could not be assigned a layer, so the rule silently "
        "did not apply to them -- the layer map has drifted from the tree:\n  "
        + "\n  ".join(str(p) for p in unclassifiable))


@pytest.mark.parametrize("rel_path", _judged_files(),
                         ids=lambda p: str(p))
def test_module_does_not_import_from_higher_layer(rel_path: Path):
    """Every L1 file imports only from L1; every L2 file from L1 or L2;
    L3 may import anything (no constraint)."""
    layer = _module_layer(rel_path)
    if layer == "L3":
        # L3 is the top; it can import anything.
        return

    src = (Path(__file__).resolve().parent.parent
           / "molbuilder" / rel_path).read_text()
    tree = ast.parse(src, filename=str(rel_path))
    targets = _import_targets(tree)

    bad = []
    for head in targets:
        target_layer = (
            "L1" if head in _L1_MODULES
            else "L2" if head in _L2_MODULES
            else "L3" if head in _L3_MODULES
            else None
        )
        if target_layer is None:
            # Imported a name we don't classify (typically a
            # top-level file like ``trajectory_log`` whose head IS in
            # the table -- but this branch catches unknown additions).
            continue
        # Layer order: L1 < L2 < L3.  An L1 file must not import L2/L3;
        # an L2 file must not import L3.
        rank = {"L1": 1, "L2": 2, "L3": 3}
        if rank[target_layer] > rank[layer]:
            bad.append((head, target_layer))

    assert not bad, (
        f"{rel_path} is classified as {layer} but imports from a "
        f"higher layer: {bad}.  Either move the import out, or "
        f"update the layer assignment in tests/test_layering.py "
        f"(and the corresponding section of docs/design.md)."
    )


def test_layer_tables_cover_all_top_level_names():
    """Sanity: every top-level name under ``molbuilder/`` is classified
    in exactly one of L1 / L2 / L3.  Catches a new module landing
    without a layering decision."""
    pkg_root = Path(__file__).resolve().parent.parent / "molbuilder"
    top_names = set()
    for entry in pkg_root.iterdir():
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix != ".py":
            continue
        if entry.is_dir():
            top_names.add(entry.name)
        else:
            top_names.add(entry.stem)
    classified = _L1_MODULES | _L2_MODULES | _L3_MODULES
    unclassified = top_names - classified
    assert not unclassified, (
        f"New top-level names found under molbuilder/ that aren't "
        f"in any layer table: {sorted(unclassified)}.  Add them to "
        f"_L1_MODULES / _L2_MODULES / _L3_MODULES in this file."
    )
