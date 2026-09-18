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
  * :mod:`.transiesta` — the TranSIESTA **emission library**: the
    geometry table every rung writes and the NEGF electrode block the
    device and transmission rungs write, reused unchanged by
    :mod:`.deck`; :mod:`.wizard` — the bulk-electrode derivation;
    :mod:`.sort` — the categorical atom sort.

**THERE IS NO ENGINE REGISTRY, and this paragraph used to say there was.**
Until 2026-09-18 the list above also named ``.preflight``, ``.engine_base``
and ``.results`` — *three modules that no longer exist on disk* — described
``.transiesta`` as a "registered engine", and told a reader that a backend
registers itself with ``@register_engine``, a decorator that exists
nowhere.  Transport's registry went on 2026-09-17 with ``TransiestaEngine``;
the sweep that day corrected the same claim in ``transiesta.py`` and
``spectra/methods.py`` and missed this file, whose whole docstring was the
claim.  A new engine is **described and rendered through ``spec_for``**
(`engines/overview.md` § 5); it does not register.
"""

# NOT a side-effect import any more.  ``from . import transiesta`` stood
# here to "guarantee the registry is populated" -- there is no registry, and
# every user of the emission library imports it directly (`deck.py`,
# `wizard.py`, `stages.py`, `jobset/prep.py`).

