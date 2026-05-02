"""Trajectory-parser interface.

Every output format (SIESTA .out, PySCF / geomeTRIC .xyz, future
NWChem / ORCA / Gaussian / OpenMM / ...) implements ``TrajectoryParser``.
The Flask app in ``molbuilder/web/blueprints/watch.py`` discovers
parsers via the registry in ``molbuilder/parsers/__init__.py`` and
never knows about specific file formats.

Return type from ``parse()``: a :class:`molbuilder.frame.Trajectory`
holding the per-step :class:`molbuilder.frame.Frame` objects plus
format-level metadata (``source_format``, optional shared
``lattice``).  Per-step physics (energy, forces, max_force,
scf_history) lives on each Frame; the Frame's ``structure`` field
carries the geometry as a :class:`molbuilder.structure.Structure`.

Use ``None`` for unknown values (energy / forces / scf_history / etc.);
they round-trip to JSON ``null`` in the legacy adapter and Plotly
draws those as gaps in the trace.

The watch web layer adapts a Trajectory back to the original molwatch
v1 JSON shape via :func:`molbuilder.parsers.trajectory_to_legacy_dict`
so the existing 3Dmol.js client keeps working unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..frame import Trajectory


class TrajectoryParser(ABC):
    """Subclass per output format."""

    #: Short identifier echoed back to the front-end ("siesta", "pyscf").
    name: str = "abstract"

    #: Human-readable name shown in the UI ("SIESTA .out", "PySCF / geomeTRIC").
    label: str = "abstract"

    #: One-line description of WHAT FILE the user should hand us.  Surfaced
    #: in the "no registered parser" error so users who upload the wrong
    #: file (e.g. PySCF .log instead of geomeTRIC _optim.xyz) get told
    #: which file to look for instead.
    hint: str = ""

    @classmethod
    @abstractmethod
    def can_parse(cls, path: str) -> bool:
        """Cheap check: does this parser handle this file?

        Implementations should peek at the first ~50 lines for format
        markers and return False fast on a mismatch.  Avoid raising;
        an unsupported file should yield False, not crash the registry.
        """

    @classmethod
    @abstractmethod
    def parse(cls, path: str) -> Trajectory:
        """Parse the entire file into a Trajectory.

        Re-callable; the watch app calls this on every mtime change.
        Implementations must be tolerant of in-progress files: torn
        frames at EOF are dropped, and a partial step that has
        coordinates but no energy yet stores ``energy=None`` on the
        Frame rather than raising.
        """
