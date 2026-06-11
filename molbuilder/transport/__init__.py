"""Electronic-transport tab — L2 engine layer.

Phase B.1 landed the dataclass
(:class:`molbuilder.config.transport.TransportConfig`).  Phase B.2
(this package) lands the engine abstraction parallel to
:mod:`molbuilder.spectra`:

  * :mod:`molbuilder.transport.engine_base` — the
    :class:`TransportEngine` Protocol + registry.
  * :mod:`molbuilder.transport.results` — the engine-agnostic
    :class:`TransportResults` dataclass.

Backend engines (transiesta, pyscf-negf) plug in by registering
their class against the Protocol; the web blueprint, the wire
parser, and the Methods generator all dispatch through the
registry.  No backend implementations land in B.2 — those follow
in B.3 once the abstraction is exercised by tests.
"""

from .engine_base import (
    TransportEngine,
    UnknownEngineError,
    register_engine,
    get_engine,
    registered_engines,
    unregister_engine,
)
from .results import TransportResults

# Concrete engines self-register on import via the
# ``@register_engine`` decorator.  Importing the module here at
# package-load time guarantees the registry is populated whenever
# anything reaches into :mod:`molbuilder.transport` (web blueprint,
# CLI, tests).  Mirrors the per-engine ``auto_defaults`` pattern
# from the chemistry middle layer.
from . import transiesta as _transiesta  # noqa: F401

__all__ = [
    "TransportEngine",
    "UnknownEngineError",
    "register_engine",
    "get_engine",
    "registered_engines",
    "unregister_engine",
    "TransportResults",
]
