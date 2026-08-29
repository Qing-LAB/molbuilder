"""Electronic transport — the COMPOSITE calculation's engine layer.

The front door of `plans/transport-design.md` § 4.1: one calculation
cites a finished junction attempt and derives five stages.  What lives
here:

  * :mod:`.compose`   — resolve the citation, extract + gate the
    electrodes, the travelling compose record.
  * :mod:`.stages`    — TRANSPORT_STAGES, the per-stage input DAG,
    ``config_for`` (the electronic contract read from the citation's
    own deck) and ``render_stage_deck``.
  * :mod:`.record`    — ``summarize run``'s ``<label>.transport.json``
    (``molbuilder/transport-result@1``).
  * :mod:`.transiesta` — the TranSIESTA deck emitter + preflight
    (registered engine); :mod:`.wizard` — the bulk-electrode derivation;
    :mod:`.preflight` — the cross-deck consistency checks; :mod:`.sort`
    — the categorical atom sort.
  * :mod:`.engine_base` — the :class:`TransportEngine` Protocol +
    registry; :mod:`.results` — the engine-agnostic
    :class:`TransportResults` dataclass (the pre-composite wire shape,
    still the sidecar's type).

A backend that registers itself (``@register_engine``) also adds its
choice to ``TransportConfig.engine`` in the same commit — the form
offers only registered engines.
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
