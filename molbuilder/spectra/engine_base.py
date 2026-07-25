"""Engine Protocol + registry for the Spectra tab.

L2 plug-in surface.  Each engine (PySCF in v1; SIESTA reserved)
provides these methods:

* :meth:`SpectraEngine.render_script` -- emit a self-contained
  script the user runs externally.
* :meth:`SpectraEngine.parse_output` -- read a
  ``<job>.spectra.json`` produced by the script back into a
  typed :class:`SpectraResults`.
* :meth:`SpectraEngine.render_checks` -- THE render gate: PySCF
  scientific advisories that DO block generation.  This is what
  ``validate(struct, cfg)`` runs via the registered SpectraConfig
  validator (validation/__init__.py).
* :meth:`SpectraEngine.selector_checks` -- preflight-only
  selector / frequency-window availability UX (workflow-ordering
  soft-deps that must NOT block render).  The /spectra preflight
  endpoint adds these on top of ``render_checks``.
* :meth:`SpectraEngine.methods_fragment` -- engine-specific
  paragraph for the Methods section (functional implementation,
  code version, citation keys for method papers).  Composed with
  the generic Methods template in
  ``molbuilder.spectra.methods``.

The render-gate vs selector split (V1/V2, 2026-07): ``render_checks``
gates generation; ``selector_checks`` is preflight-only.  They replaced
a single combined ``preflight`` so the render gate runs only the science
checks -- a ``top_n`` script is valid to EMIT even before Raman data
exists, so its selector soft-dep must not block render.

Engines self-register via the ``@register_engine`` decorator at
import time; :func:`get_engine` dispatches by ``name``.

Adding a new engine (SIESTA, OpenMX with vib analysis, ...) is
mechanical: drop in ``<engine>_engine.py`` next to this file
with a registered class, add the engine name to
``SpectraConfig.engine``'s ``choices`` metadata tuple, and add
its citations to ``docs/tabs/spectra/references.bib``.  Nothing
else changes -- the web blueprint, the JSON parser, the
Methods generator, and the form-schema endpoint all dispatch
through this registry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Type, runtime_checkable

from ..config.spectra import SpectraConfig
from ..issues import Issue
from ..structure import Structure
from .results import ModeData, SpectraResults


class UnknownEngineError(LookupError):
    """Raised by :func:`get_engine` when the requested engine name
    isn't registered.  Carries the requested name + the list of
    available names so a typo surfaces with an actionable error.
    """

    def __init__(self, name: str, available):
        super().__init__(
            f"unknown Spectra engine {name!r}; "
            f"registered engines: {sorted(available)}"
        )
        self.name      = name
        self.available = sorted(available)


@runtime_checkable
class SpectraEngine(Protocol):
    """Plug-in surface for a Spectra-tab engine.

    Concrete engines are classes with classmethod implementations
    of the four operations below.  ``name`` and ``label`` are
    class attributes (not methods) so the registry can introspect
    them without instantiating.

    The Protocol is :func:`runtime_checkable` for the (rare) case
    where a caller wants to ``isinstance(cls, SpectraEngine)``
    check; normal dispatch is via the registry.
    """

    name:  str       # registry key, e.g. "pyscf"
    label: str       # human-facing, e.g. "PySCF (analytic Hessian + dα/dR)"

    @classmethod
    def render_script(cls, struct: Structure, cfg: SpectraConfig) -> str:
        """Build the self-contained script the user runs externally.

        The script writes ``<cfg.job_name>.spectra.json`` (spec § 6)
        as it progresses, atomically replacing the file at each
        phase boundary.  See spec § 6.1 for the live-watch contract.
        """
        ...

    @classmethod
    def parse_output(cls, path: str) -> SpectraResults:
        """Read a ``<job>.spectra.json`` produced by this engine
        back into a typed :class:`SpectraResults`.

        The parser is engine-independent at the wire level (the
        JSON schema is shared across engines), but the engine
        knows about its own ``engine_metadata`` fields and can
        post-process them when populating the typed shape.
        Implementations typically delegate to
        :meth:`SpectraResults.from_dict` plus a small per-engine
        adaptation layer.
        """
        ...

    @classmethod
    def render_checks(cls, struct: Structure,
                      cfg: SpectraConfig) -> List[Issue]:
        """THE render gate -- scientific advisories that block or warn on
        generation.  This is what ``validate(struct, cfg)`` runs via the
        registered SpectraConfig validator, so EVERY engine MUST implement
        it (a missing method would make ``validate()`` raise ``AttributeError``).

        Returns :class:`Issue` list; ``Issue.severity`` is ``"error"``
        (would fail / produce bad data -- blocks generation) or ``"warn"``
        (scientifically dubious but valid, e.g. hybrid functional at grid
        level < 4, or displacement amplitude outside [0.02, 0.20] Å).
        """
        ...

    @classmethod
    def selector_checks(cls, struct: Structure,
                        cfg: SpectraConfig,
                        prior: Optional[SpectraResults] = None) -> List[Issue]:
        """Preflight-ONLY selector / frequency-window availability checks
        (workflow-ordering soft-deps).  NOT part of the render gate: a
        ``top_n`` / ``threshold`` script is valid to emit before any Raman
        data exists (the selector resolves at run time), so these must not
        block render.  The /spectra preflight endpoint adds them on top of
        ``render_checks``.

        ``prior`` is the on-disk :class:`SpectraResults` for a resume /
        re-run on the same ``job_name`` (``None`` for a fresh run); the
        engine uses it to range-check explicit indices and decide whether
        rank-based selectors have the Raman data they need.
        """
        ...

    @classmethod
    def methods_fragment(cls, cfg: SpectraConfig,
                         modes: List[ModeData]) -> str:
        """Engine-specific paragraph for the Methods section.

        Composed with the generic Methods template
        (``molbuilder.spectra.methods``) which produces the full
        Methods text.  An engine's fragment typically names:

          * the program + version ("PySCF 2.x.y");
          * the specific implementation used ("analytic Hessian
            via ``pyscf.hessian.rks``");
          * the method papers cited via ``[CitationKey]`` markers
            that resolve against ``docs/tabs/spectra/references.bib``.

        Plain English, manuscript-ready prose.  The Methods
        generator interpolates the citation keys + the actual
        parameter values (functional, basis, etc.) when composing
        the full Methods paragraph.
        """
        ...


# --------------------------------------------------------------------- #
#  Registry                                                              #
# --------------------------------------------------------------------- #


_ENGINES: Dict[str, Type[SpectraEngine]] = {}


def register_engine(cls: Type[SpectraEngine]) -> Type[SpectraEngine]:
    """Class decorator: register an engine under its ``name``
    attribute so :func:`get_engine` can find it.

    Re-registering an existing ``name`` raises ``ValueError``
    rather than silently overwriting -- if two engines claim the
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
            f"register_engine: engine name {name!r} already registered "
            f"to {_ENGINES[name].__name__}; cannot re-register to "
            f"{cls.__name__}.  If you intended to replace the engine, "
            f"call unregister_engine({name!r}) first."
        )
    _ENGINES[name] = cls
    return cls


def get_engine(name: str) -> Type[SpectraEngine]:
    """Look up a registered engine by name; raise
    :class:`UnknownEngineError` if not registered.

    Typical caller (the web blueprint):

      .. code:: python

          engine = get_engine(cfg.engine)
          issues = engine.render_checks(struct, cfg)   # the render gate
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

    Test-only knob -- not part of the production lifecycle.
    Provided so pytest fixtures can register a mock engine for
    the duration of a test without leaking it into the next.
    """
    _ENGINES.pop(name, None)


__all__ = [
    "SpectraEngine",
    "UnknownEngineError",
    "register_engine",
    "get_engine",
    "registered_engines",
    "unregister_engine",
]
