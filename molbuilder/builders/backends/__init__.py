"""Pluggable nucleic-acid builders.

Each backend implements ``build(kind, sequence, form, terminal, title)``
and returns a :class:`~molbuilder.structure.Structure`.

    Backend   | DNA  | RNA  | helix-shape   | install
    ----------+------+------+---------------+-----------------------------
    threedna  |  yes |  yes | canonical     | x3dna.org (license-gated;
              |      |      | (B/A/Z/A-RNA) |  unpack tarball at repo root
              |      |      |               |  OR set $X3DNA)
    amber     |  yes |  yes | extended      | conda ambertools  (tleap)
    rdkit     |  yes |  yes | folded        | already a dep

  * threedna shells out to 3DNA's ``fiber`` for canonical helical
    geometry -- the only backend that produces a true B/A/Z helix.
    See ``_threedna.py`` for the detection chain (in-tree directory
    -> $X3DNA env -> fiber on PATH).  3DNA is non-commercial-use
    licensed and not pip/conda-installable; molbuilder does not
    auto-fetch it.
  * amber drives AmberTools' ``tleap`` with a ``sequence { ... }``
    macro; output is chemically clean (Amber OL15 / OL3 force-field
    topology) in extended conformation.  AmberTools 23+ removed the
    original ``nab`` fiber builder; this is the closest in-AmberTools
    replacement.
  * rdkit produces correct chemistry / connectivity but the embedded
    conformer is whatever ETKDG decides -- not a helix.  Fine for
    short oligos that DFT will fully optimise; bad for 10+ mers.

(There used to be an in-house "fiber" chain-grow backend; it has been
removed because it produced incorrect 5'-end chemistry and could not
enforce the tetrahedral phosphate-bridge geometry.  Use ``threedna``,
``amber``, or ``rdkit`` instead.)

``BackendUnavailable`` is raised when the user picks a backend whose
external dependency isn't installed (e.g. ``amber`` without ``tleap``,
or ``threedna`` without an x3dna install reachable via the detection
chain).
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
    from . import _amber, _rdkit, _threedna
    return {
        "threedna": _threedna.build,
        "amber":    _amber.build,
        "rdkit":    _rdkit.build,
    }


def available_backends() -> Dict[str, bool]:
    """Map of backend-name -> whether it's runnable on this machine.

    'threedna' is True if a 3DNA install is reachable via the
        detection chain (in-tree directory -> $X3DNA env -> fiber
        on PATH).
    'amber' is True if ``tleap`` is on PATH.
    'rdkit' is True if rdkit imports.
    """
    from . import _amber, _rdkit, _threedna
    return {
        "threedna": _threedna.is_available(),
        "amber":    _amber.is_available(),
        "rdkit":    _rdkit.is_available(),
    }


# Auto-detect order: best geometry first.  3DNA produces canonical
# helices, AmberTools an extended chain (correct chemistry, not a
# helix), RDKit a folded conformer (correct chemistry, no helix
# discipline at all).  When --backend auto is requested, dispatch
# falls through this list cleanly: each missing backend is skipped,
# the first available one runs, no error if at least one is present.
_AUTO_ORDER = ["threedna", "amber", "rdkit"]


def dispatch(kind: str, sequence: str, *,
             backend: str = "auto",
             form: str = "B",
             terminal: str = "OH",
             title=None):
    """Build a nucleic acid using the named backend (or auto-detect)."""
    backends = _load_backends()

    if backend == "auto":
        avail = available_backends()
        for name in _AUTO_ORDER:
            if avail.get(name):
                return backends[name](kind, sequence, form, terminal, title)
        raise BackendUnavailable(
            "No nucleic-acid backend available.  Either:\n"
            "  - install 3DNA (best geometry; download from http://x3dna.org/\n"
            "    after accepting the non-commercial license, then unpack\n"
            "    at the repo root or set $X3DNA),\n"
            "  - `conda install -c conda-forge ambertools`  (extended chain),\n"
            "  - `pip install rdkit`  (folded conformer, chemistry-only)."
        )

    if backend not in backends:
        raise ValueError(
            f"Unknown backend {backend!r}; valid: "
            f"{sorted(backends) + ['auto']}"
        )

    return backends[backend](kind, sequence, form, terminal, title)


__all__ = ["BackendUnavailable", "available_backends", "dispatch"]
