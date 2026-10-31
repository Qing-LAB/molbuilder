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

Every backend delegates to an external builder, and none grows the
chain residue-by-residue in-house: doing that gets the 5'-end chemistry
wrong and cannot enforce the tetrahedral phosphate-bridge geometry.

``BackendUnavailable`` is raised when the user picks a backend whose
external dependency isn't installed (e.g. ``amber`` without ``tleap``,
or ``threedna`` without an x3dna install reachable via the detection
chain).  It is `molbuilder.chemistry`'s type -- the lowest layer that
raises it, since ``add_hydrogens`` refuses with it too -- and each
backend names itself in its ``missing``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ...chemistry import BackendUnavailable


#: THE HELICAL FORM EACH KIND IS BUILT IN when none is asked -- the canonical
#: one, B for DNA and A for RNA -- and THE TERMINAL STATE, both ends -OH.
#: Every door that builds reads them here: the builders' own defaults, the
#: backends', the build route, the command line and the check after a build.
DEFAULT_FORM = {"dna": "B", "rna": "A"}
DEFAULT_TERMINAL = "OH"


@dataclass(frozen=True)
class Built:
    """What a backend builds of an ask *(decided with the user, 2026-10-31,
    (b))*: the helical ``form`` -- ``None`` where it shapes no helix -- and
    the ``terminal`` state, each with the words that say why when it is not
    what was asked.  Each backend answers it (its module's ``built``) beside
    the table its own build reads, and the check after the build
    (`validation.builders.check_built`) says what differs."""
    form: Optional[str]
    terminal: str
    form_words: str = ""
    terminal_words: str = ""


def built(backend: str, kind: str, form: str, terminal: str) -> Built:
    """What ``backend`` builds of a ``kind`` asked in ``form`` with
    ``terminal`` -- its own module's answer."""
    return _modules()[backend].built(kind, form, terminal)


def resolved_backend(backend: str) -> Optional[str]:
    """The backend a build with ``backend`` runs: ``auto`` resolved by
    :func:`auto_backend_name` -- ``None`` when none is installed -- any
    other name as given.  Asked once, before the build, by every door that
    builds: the build runs it and the check after it names it."""
    return auto_backend_name() if backend == "auto" else backend


# ---------------------------------------------------------------- #
#  Registry                                                         #
# ---------------------------------------------------------------- #


def _modules():
    """Each backend's module, by its name -- lazy, so a missing dep doesn't
    break the whole package."""
    from . import _amber, _rdkit, _threedna
    return {"threedna": _threedna, "amber": _amber, "rdkit": _rdkit}


def _load_backends() -> Dict[str, Callable]:
    """Each backend's ``build``, by its name."""
    return {name: m.build for name, m in _modules().items()}


def available_backends() -> Dict[str, bool]:
    """Map of backend-name -> whether it's runnable on this machine.

    'threedna' is True if a 3DNA install is reachable via the
        detection chain (in-tree directory -> $X3DNA env -> fiber
        on PATH).
    'amber' is True if ``tleap`` is on PATH.
    'rdkit' is True if rdkit imports.
    """
    return {name: m.is_available() for name, m in _modules().items()}


# Auto-detect order: best geometry first.  3DNA produces canonical
# helices, AmberTools an extended chain (correct chemistry, not a
# helix), RDKit a folded conformer (correct chemistry, no helix
# discipline at all).  When --backend auto is requested, dispatch
# falls through this list cleanly: each missing backend is skipped,
# the first available one runs, no error if at least one is present.
_AUTO_ORDER = ["threedna", "amber", "rdkit"]


def auto_backend_name() -> str | None:
    """Which backend ``dispatch(backend='auto')`` would pick on this
    machine, or ``None`` if no backend is available.

    Pure read-only -- doesn't run any backend.  Useful for surfacing
    the resolved choice in CLIs and the web UI before/after a build.
    """
    avail = available_backends()
    for name in _AUTO_ORDER:
        if avail.get(name):
            return name
    return None


def dispatch(kind: str, sequence: str, *,
             backend: str = "auto",
             form: Optional[str] = None,
             terminal: str = DEFAULT_TERMINAL,
             title=None,
             double_strand: bool = False,
             strand2: "str | None" = None):
    """Build a nucleic acid using the named backend (or auto-detect).

    ``double_strand`` builds a canonical Watson-Crick DUPLEX (only the threedna
    backend supports it; build_dna gates it to X3DNA + B-form).  ``strand2`` (an
    explicit second strand, 5'->3') builds an ARBITRARY / mismatched duplex via
    X3DNA ``rebuild`` -- threedna-only, B-form; other backends reject it.
    """
    backends = _load_backends()
    form = form or DEFAULT_FORM.get(kind)

    if backend == "auto":
        chosen = resolved_backend(backend)
        if chosen is None:
            raise BackendUnavailable(
                "No nucleic-acid backend available.  Either:\n"
                "  - install 3DNA (best geometry; download from http://x3dna.org/\n"
                "    after accepting the non-commercial license, then unpack\n"
                "    at the repo root or set $X3DNA),\n"
                "  - `conda install -c conda-forge ambertools`  (extended chain),\n"
                "  - `pip install rdkit`  (folded conformer, chemistry-only)."
            )
        return backends[chosen](kind, sequence, form, terminal, title,
                                double_strand=double_strand, strand2=strand2)

    if backend not in backends:
        raise ValueError(
            f"Unknown backend {backend!r}; valid: "
            f"{sorted(backends) + ['auto']}"
        )

    return backends[backend](kind, sequence, form, terminal, title,
                             double_strand=double_strand, strand2=strand2)


__all__ = [
    "DEFAULT_FORM", "DEFAULT_TERMINAL", "Built", "auto_backend_name",
    "available_backends", "built", "dispatch", "resolved_backend",
]
