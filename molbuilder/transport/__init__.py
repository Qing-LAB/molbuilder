"""Electronic transport — the COMPOSITE calculation's engine layer.

The front door of `archive/2026-09-01-transport-design.md` § 4.1: one calculation
cites a finished junction attempt and derives five stages.  What lives
here:

  * :mod:`.compose`   — resolve the citation, extract + gate the
    electrodes, the travelling compose record.
  * :mod:`.stages`    — TRANSPORT_STAGES, the per-stage input DAG, and
    ``route_overrides`` (which rung owns each override).
  * :mod:`.deck`      — ``transport_spec``: the ``DeckSpec`` every one of
    the five rungs renders through, and ``SHAPE_OF_RUNG``, the table
    saying which of the three deck shapes each rung gets.
  * :mod:`.citation_defaults` — what the cited run contributes to the
    template, once, at ``jobset init``.
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


# Concrete engines self-register on import via the
# ``@register_engine`` decorator.  Importing the module here at
# package-load time guarantees the registry is populated whenever
# anything reaches into :mod:`molbuilder.transport` (web blueprint,
# CLI, tests).  Mirrors the per-engine ``auto_defaults`` pattern
# from the chemistry middle layer.
from . import transiesta as _transiesta  # noqa: F401

