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
    "chemistry", "residues",
    "config",            # siesta / pyscf / spectra / transport config dataclasses
    "trajectory_log",    # format + emitter (data-shape, no domain logic)
    "selection",         # atom-selection rule dataclasses + evaluator (no domain deps)
    "runtime_info",      # cross-cutting threading / GPU / runtime-info emitters
                         # (string emitters + physical_core_count -- L1 because no
                         # domain deps; siesta + pyscf + spectra + runwrap all use it)
    "pseudos",           # PSML pseudopotential header parser + coverage check
                         # (pure XML parsing + dataclass; no domain deps)
}

_L2_MODULES = {
    "peptide", "nucleic", "smiles", "pubchem",
    "modify", "validation",
    "builders",          # backends/* package
    "backends",          # back-compat shim re-exporting builders.backends
    "siesta", "pyscf",   # script generators
    "spectra",           # spectra engines + script renderers
    "transport",         # transport engines + results (Phase B.2)
    "parsers",           # trajectory parsers
    "projects",          # filesystem layout / naming rules
    "runtime_config",    # molbuilder.json reader
    "diagnostics",       # capabilities snapshot
    "envs",              # subprocess dispatch
    "runwrap",           # bash-wrapper emitter
    "data",              # bundled JSON tables
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


@pytest.mark.parametrize("rel_path", _all_python_files(),
                         ids=lambda p: str(p))
def test_module_does_not_import_from_higher_layer(rel_path: Path):
    """Every L1 file imports only from L1; every L2 file from L1 or L2;
    L3 may import anything (no constraint)."""
    if rel_path in _EXEMPT:
        pytest.skip(f"{rel_path} is an explicit re-export site")
    layer = _module_layer(rel_path)
    if layer is None:
        pytest.skip(f"{rel_path} is unclassified (probably __init__ shim)")
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
