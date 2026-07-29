"""Engine Protocol + registry for the Transport tab.

L2 plug-in surface, mirrored from :mod:`molbuilder.spectra.engine_base`.
Each engine (transiesta, pyscf-negf in B.3) provides four methods:

* :meth:`TransportEngine.render_script` — emit a self-contained
  script the user runs externally.
* :meth:`TransportEngine.parse_output` — read a
  ``<job>.transport.json`` produced by the script back into a
  typed :class:`TransportResults`.
* :meth:`TransportEngine.preflight` — pre-submission scientific +
  consistency checks against the current ``TransportConfig`` and
  any prior results on disk.  Returns ``List[Issue]`` so warns and
  errors share the rest of molbuilder's issues pipeline.
* :meth:`TransportEngine.methods_fragment` — engine-specific
  paragraph for the Methods section (NEGF code version, self-energy
  scheme, basis, citation keys).  Composed with the generic
  Methods template that lands when B.3 wires in backends.

Engines self-register via the ``@register_engine`` decorator at
import time; :func:`get_engine` dispatches by ``name``.

Adding a new engine (transiesta, pyscf-negf, inelastica, ...) is
mechanical: drop in ``<engine>_engine.py`` next to this file with
a registered class, add the engine name to the
``TransportConfig.engine`` ``choices`` metadata tuple, and add its
citations to ``docs/science/references.bib``.  Nothing else
changes — the web blueprint, the JSON parser, the Methods
generator, and the form-schema endpoint all dispatch through this
registry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Type, runtime_checkable

from ..config.transport import TransportConfig
from ..issues import Issue
from ..structure import Structure
from .results import TransportResults


class UnknownEngineError(LookupError):
    """Raised by :func:`get_engine` when the requested engine name
    isn't registered.  Carries the requested name + the list of
    available names so a typo surfaces with an actionable error.
    """

    def __init__(self, name: str, available):
        super().__init__(
            f"unknown Transport engine {name!r}; "
            f"registered engines: {sorted(available)}"
        )
        self.name = name
        self.available = sorted(available)


@runtime_checkable
class TransportEngine(Protocol):
    """Plug-in surface for a Transport-tab engine.

    Concrete engines are classes with classmethod implementations of
    the four operations below.  ``name`` and ``label`` are class
    attributes (not methods) so the registry can introspect them
    without instantiating.

    The Protocol is :func:`runtime_checkable` for the (rare) case
    where a caller wants to ``isinstance(cls, TransportEngine)``;
    normal dispatch is via the registry.
    """

    name: str     # registry key, e.g. "transiesta"
    label: str    # human-facing, e.g. "TranSIESTA (NEGF, pseudopotentials)"

    @classmethod
    def render_script(cls, struct: Structure,
                      cfg: TransportConfig) -> str:
        """Build the self-contained script the user runs externally.

        The script writes ``<cfg.job_name>.transport.json`` as it
        progresses, atomically replacing the file at each phase
        boundary so molwatch / the Results tab can live-render
        partial output.
        """
        ...

    @classmethod
    def parse_output(cls, path: str) -> TransportResults:
        """Read a ``<job>.transport.json`` produced by this engine
        back into a typed :class:`TransportResults`.

        The wire format is engine-independent (the JSON schema is
        shared across engines), but the engine knows about its own
        ``engine_metadata`` fields and can post-process them when
        populating the typed shape.  Implementations typically
        delegate to :meth:`TransportResults.from_dict` plus a small
        per-engine adaptation layer.
        """
        ...

    @classmethod
    def preflight(cls, struct: Structure,
                  cfg: TransportConfig,
                  prior: Optional[TransportResults] = None
                  ) -> List[Issue]:
        """Scientific + consistency checks before the script runs.

        Returns a list of :class:`Issue` — the web layer's
        validation panel renders these inline.  Severity matches
        the rest of molbuilder's pipeline:

          * ``error`` — the script would fail / produce bad data;
            blocks generation.  Examples: device region empty;
            electrode atoms not labelled in the .molstruct.json
            sidecar; basis set the engine doesn't recognise.
          * ``warn`` — scientifically dubious but not broken.
            Examples: bias window larger than ±2 V (out of the
            linear-response regime; NEGF still runs); device region
            < 10 atoms (likely under-converged).

        ``prior`` is the on-disk :class:`TransportResults` if this
        run is a resume / re-run after an earlier run on the same
        ``job_name``; ``None`` for a fresh run.  The engine uses
        ``prior`` to decide which phases to skip (electrode
        self-energies are reusable across V points; the
        equilibrium G is reusable for a finite-bias re-run).
        """
        ...

    @classmethod
    def methods_fragment(cls, cfg: TransportConfig,
                         results: TransportResults) -> str:
        """Engine-specific paragraph for the Methods section.

        Composed with the generic Methods template (in
        :mod:`molbuilder.transport.methods` when B.3 ships).
        An engine's fragment typically names:

          * the program + version ("TranSIESTA 4.1.5");
          * the self-energy scheme + decimation depth;
          * the basis / pseudopotential sources;
          * the citation keys via ``[CitationKey]`` markers that
            resolve against
            ``docs/science/references.bib``.

        Plain English, manuscript-ready prose.  The Methods
        generator interpolates citation keys + actual parameter
        values (k-grid, energy grid density, bias points, etc.)
        when composing the full Methods paragraph.
        """
        ...


# --------------------------------------------------------------------- #
#  Registry                                                              #
# --------------------------------------------------------------------- #


_ENGINES: Dict[str, Type[TransportEngine]] = {}


def register_engine(
        cls: Type[TransportEngine]) -> Type[TransportEngine]:
    """Class decorator: register an engine under its ``name``
    attribute so :func:`get_engine` can find it.

    Re-registering an existing ``name`` raises ``ValueError``
    rather than silently overwriting — if two engines claim the
    same name, that's a programmer error worth surfacing at
    import time.
    """
    name = getattr(cls, "name", None)
    if not name or not isinstance(name, str):
        raise TypeError(
            f"register_engine: {cls!r} must declare a non-empty "
            f"string `name` class attribute"
        )
    if name in _ENGINES and _ENGINES[name] is not cls:
        raise ValueError(
            f"register_engine: engine name {name!r} already "
            f"registered to {_ENGINES[name].__name__}; cannot "
            f"re-register to {cls.__name__}.  If you intended to "
            f"replace the engine, call unregister_engine({name!r}) "
            f"first."
        )
    _ENGINES[name] = cls
    return cls


def get_engine(name: str) -> Type[TransportEngine]:
    """Look up a registered engine by name; raise
    :class:`UnknownEngineError` if not registered.

    Typical caller (the web blueprint, once B.3 ships):

      .. code:: python

          engine = get_engine(cfg.engine)
          issues = engine.preflight(struct, cfg)
          script = engine.render_script(struct, cfg)
    """
    if name not in _ENGINES:
        raise UnknownEngineError(name, _ENGINES.keys())
    return _ENGINES[name]


def registered_engines() -> List[str]:
    """Return the names of all currently-registered engines,
    sorted alphabetically.  Useful for the schema endpoint's
    ``choices`` validation."""
    return sorted(_ENGINES.keys())


def unregister_engine(name: str) -> None:
    """Remove an engine from the registry.

    Test-only knob — not part of the production lifecycle.
    Provided so pytest fixtures can register a mock engine for the
    duration of a test without leaking it into the next.
    """
    _ENGINES.pop(name, None)


__all__ = [
    "TransportEngine",
    "UnknownEngineError",
    "register_engine",
    "get_engine",
    "registered_engines",
    "unregister_engine",
]
