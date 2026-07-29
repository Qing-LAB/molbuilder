"""Sidecar JSON write-side helpers.

Per ``docs/model/parse.md`` H2: this package hosts
the write-side of the ``.molstruct.json`` / ``.spectra.json`` /
``.transport.json`` sidecars.  The read-side lives in
``molbuilder.parse.sidecars``.

The two halves are intentionally split:

* ``parse/sidecars/`` — read-only parsers built on the
  ``FileParser`` ABC.  Pure I/O over local files; no subprocess,
  no network, no global mutable state.  Returns typed
  :class:`molbuilder.parse.types.SidecarResult` envelopes for
  uniform consumption by the registry + monitor.
* ``molbuilder/sidecars/`` (this package) — write-side: atomic
  save / canonical-dict builders / sidecar-path conventions /
  POSIX advisory locking / consumer-helper ``apply_to_structure``
  that maps a loaded sidecar payload back onto a Structure object.

Exception classes live HERE as their canonical home; the read-side
modules re-import them so callers can ``except`` on either side.

Per-format modules:

* :mod:`molbuilder.sidecars.molstruct` — per-atom regions +
  frozen_atoms metadata that rides next to a structure file.
* :mod:`molbuilder.sidecars.spectra` — spectrum-results envelope
  (peaks, dipole strengths, …).
* :mod:`molbuilder.sidecars.transport` — transport-results
  envelope (T(E), I-V, …).
"""
