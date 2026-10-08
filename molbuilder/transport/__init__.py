"""Electronic transport — the COMPOSITE calculation's engine layer.

One calculation
cites a finished junction attempt and derives five stages.  What lives
here:

  * :mod:`.compose`   — resolve the citation, extract + gate the
    electrodes, the travelling compose record.
  * :mod:`.stages`    — TRANSPORT_STAGES and the per-stage input DAG.
    Which rung reads an override is the catalogue's (``template.reads``).
  * :mod:`.deck`      — ``transport_spec``: the ``DeckSpec`` every one of
    the five rungs renders through, and ``SHAPE_OF_RUNG``, the table
    saying which of the four deck shapes each rung gets.
  * :mod:`.citation_defaults` — what the cited run contributes to the
    template, once, at ``jobset init``.
  * :mod:`.record`    — ``summarize task``'s ``<label>.transport.json``
    (``molbuilder/transport-result@2``).
  * :mod:`.transiesta` — the TranSIESTA **emission library**: the
    geometry table every rung writes and the electrode and reservoir
    declarations the device and transmission rungs write, reused by
    :mod:`.deck` (their VALUES are catalogue items,
    `engines/transport.md` § 6.1b); :mod:`.wizard` — the bulk-electrode
    derivation;
    :mod:`.sort` — the categorical atom sort.

**THERE IS NO ENGINE REGISTRY.**  A new engine is **described and rendered through ``spec_for``**
(`engines/overview.md` § 5); it does not register.
"""

