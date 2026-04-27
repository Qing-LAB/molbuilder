"""Pluggable nucleic-acid builders.

Each backend implements ``build(kind, sequence, form, terminal, title)``
and returns a :class:`~molbuilder.structure.Structure`.

    Backend   | DNA  | RNA  | helix-shape   | install
    ----------+------+------+---------------+-----------------------------
    rdkit     |  yes |  yes | folded        | already a dep
    amber     |  yes |  yes | extended      | conda ambertools  (tleap)

  * rdkit produces correct chemistry / connectivity but the embedded
    conformer is whatever ETKDG decides -- not a helix.  Fine for
    short oligos that DFT will fully optimise; bad for 10+ mers.
  * amber drives AmberTools' ``tleap`` with a ``sequence { ... }``
    macro; output is chemically clean (Amber OL15 / OL3 force-field
    topology) in extended conformation.  AmberTools 23+ removed the
    original ``nab`` fiber builder; this is the closest in-AmberTools
    replacement.

For canonical B-form / A-form 3-D coordinates -- the only thing
neither backend provides -- install 3DNA externally and use its
``fiber`` command.  A 3DNA backend isn't wired up yet but would slot
in here as ``_threedna.py``.

(There used to be an in-house "fiber" chain-grow backend; it has been
removed because it produced incorrect 5'-end chemistry and could not
enforce the tetrahedral phosphate-bridge geometry.  Use ``rdkit`` or
``amber`` instead.)

``BackendUnavailable`` is raised when the user picks a backend whose
external dependency isn't installed (e.g. ``amber`` without ``tleap``).
"""

from __future__ import annotations

from typing import Callable, Dict


class BackendUnavailable(RuntimeError):
    """The requested backend's external dependency isn't installed."""


# ---------------------------------------------------------------- #
#  Registry                                                         #
# ---------------------------------------------------------------- #


def _load_backends() -> Dict[str, Callable]:
    """Lazy-load each backend module so a missing dep doesn't break
    the whole package."""
    from . import _amber, _rdkit
    return {
        "rdkit": _rdkit.build,
        "amber": _amber.build,
    }


def available_backends() -> Dict[str, bool]:
    """Map of backend-name -> whether it's runnable on this machine.

    'rdkit' is True if rdkit imports.
    'amber' is True if ``tleap`` is on PATH.
    """
    from . import _amber, _rdkit
    return {
        "rdkit": _rdkit.is_available(),
        "amber": _amber.is_available(),
    }


def dispatch(kind: str, sequence: str, *,
             backend: str = "auto",
             form: str = "B",
             terminal: str = "OH",
             title=None):
    """Build a nucleic acid using the named backend (or auto-detect)."""
    backends = _load_backends()

    if backend == "auto":
        # Prefer the AmberTools backend (correct chemistry + extended
        # geometry); fall back to RDKit (folded conformer, still valid
        # chemistry) if AmberTools isn't installed.
        order = ["amber", "rdkit"]
        avail = available_backends()
        for name in order:
            if avail.get(name):
                return backends[name](kind, sequence, form, terminal, title)
        raise BackendUnavailable(
            "No nucleic-acid backend available.  Either:\n"
            "  - `pip install rdkit`  (chemistry-only, folded conformer),\n"
            "  - `conda install -c conda-forge ambertools`  (extended chain)."
        )

    if backend not in backends:
        raise ValueError(
            f"Unknown backend {backend!r}; valid: "
            f"{sorted(backends) + ['auto']}"
        )

    return backends[backend](kind, sequence, form, terminal, title)


__all__ = ["BackendUnavailable", "available_backends", "dispatch"]
