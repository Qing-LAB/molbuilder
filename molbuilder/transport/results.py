"""L1 result type for the Transport tab.

Pinned shape — an engine-agnostic dataclass that any TransportEngine
backend populates from its own native output.  The Web blueprint,
the Methods generator, and the wire serialiser all consume
this shape uniformly.

Fields:

  * :attr:`metadata` — dict of engine name, code version, run
    timestamps, job name, structure source, etc.
  * :attr:`energy_grid_eV` — 1-D float array of energies (eV)
    at which transmission is evaluated.
  * :attr:`transmission` — 1-D float array, same length as
    ``energy_grid_eV``.  Total transmission T(E) in units of e²/h.
  * :attr:`fermi_energy_eV` — float, the Fermi level used as
    the zero of the energy grid.
  * :attr:`conductance_G0` — float, G(E_F) in units of the
    conductance quantum :math:`G_0 = 2e^2/h`.  Convenience
    scalar; equals ``transmission`` interpolated at
    ``fermi_energy_eV``.
  * :attr:`pdos` — optional ``Dict[str, np.ndarray]`` of
    projected DOS per orbital group (e.g. ``{"C-2pz": [...]}``);
    arrays share the energy grid.  May be empty for engines that
    don't decompose the DOS.
  * :attr:`bias_grid_V` — optional 1-D float array of
    source-drain voltages where the current is sampled.  ``None``
    for equilibrium-only runs.
  * :attr:`current_uA` — optional I(V) array (μA), same length
    as ``bias_grid_V``.  ``None`` iff ``bias_grid_V`` is None.
  * :attr:`methods_text` — engine-generated Methods paragraph
    (manuscript-ready prose with citation markers).
  * :attr:`bibliography_keys` — list of citation keys referenced
    in ``methods_text`` for resolution against
    ``docs/tabs/transport/references.bib``.
  * :attr:`complete` — False during a live-watched in-progress
    run, True once the engine has written the final phase.

The dataclass refuses equality: ``a == b`` raises ``TypeError``.
Comparison of two transport runs is a multi-dimensional question
("ΔG at E_F?", "Δintegrated current 0..0.5 V?", "biggest peak
position shift?") that the boolean answer wouldn't capture.  A
future ``transport.compare`` helper is the right shape for this.

JSON round-trip via :meth:`to_dict` / :meth:`from_dict`; numpy
arrays serialise as nested lists (re-coerced to dtype=float on
read).  Schema version pinned at ``"1"`` so a forward-compat
read sees the version field and can branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


SCHEMA_VERSION = "2"
# Schema history:
#   v1 (initial): metadata, energy_grid_eV, transmission, fermi_energy_eV,
#                 conductance_G0, pdos, bias_grid_V, current_uA,
#                 methods_text, bibliography_keys, complete.
#   v2 (2026-06-25): added regions + frozen_atoms so the sidecar carries
#                    the boundary conditions the calculation used --
#                    downstream comparators / manuscript generators can
#                    correlate results to input geometry without needing
#                    the sibling .molstruct.json.  Reader stays back-
#                    compat for v1 files (regions/frozen_atoms default
#                    to empty).


@dataclass
class TransportResults:
    """Engine-agnostic result of a transport calculation.  Phase B.2
    defines the shape; B.3 backends populate it."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    energy_grid_eV: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    transmission: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    fermi_energy_eV: float = 0.0
    conductance_G0: float = 0.0
    pdos: Dict[str, np.ndarray] = field(default_factory=dict)
    bias_grid_V: Optional[np.ndarray] = None
    current_uA: Optional[np.ndarray] = None
    methods_text: str = ""
    bibliography_keys: List[str] = field(default_factory=list)
    complete: bool = False
    # Boundary conditions the calculation used.  Schema v2 (2026-06-25).
    # Populated from struct.regions / struct.frozen_atoms at write time
    # so a parsed .transport.json can answer "which atoms were the
    # bridge / electrodes / frozen?" without needing the .molstruct.json
    # sibling.  Empty dict / empty list = no boundary conditions
    # declared on the input (legal for a free-electron sanity check
    # but unusual for a real device).
    regions: Dict[str, List[int]] = field(default_factory=dict)
    frozen_atoms: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Coerce array fields so callers don't have to fight dtype
        # surprises 100 lines downstream.  Loud failures here are
        # better than silent silent broadcasting later.
        self.energy_grid_eV = np.asarray(self.energy_grid_eV,
                                          dtype=float)
        self.transmission = np.asarray(self.transmission, dtype=float)
        if self.energy_grid_eV.shape != self.transmission.shape:
            raise ValueError(
                f"TransportResults: energy_grid_eV "
                f"{self.energy_grid_eV.shape} and transmission "
                f"{self.transmission.shape} must share shape"
            )
        for k, arr in list(self.pdos.items()):
            arr = np.asarray(arr, dtype=float)
            if arr.shape != self.energy_grid_eV.shape:
                raise ValueError(
                    f"TransportResults: pdos[{k!r}] shape {arr.shape} "
                    f"does not match energy_grid_eV "
                    f"{self.energy_grid_eV.shape}"
                )
            self.pdos[k] = arr
        if self.bias_grid_V is not None:
            self.bias_grid_V = np.asarray(self.bias_grid_V, dtype=float)
        if self.current_uA is not None:
            self.current_uA = np.asarray(self.current_uA, dtype=float)
        # I(V) and V grids must be paired
        if (self.bias_grid_V is None) != (self.current_uA is None):
            raise ValueError(
                "TransportResults: bias_grid_V and current_uA "
                "must be both provided or both None"
            )
        if (self.bias_grid_V is not None
                and self.bias_grid_V.shape != self.current_uA.shape):
            raise ValueError(
                f"TransportResults: bias_grid_V "
                f"{self.bias_grid_V.shape} and current_uA "
                f"{self.current_uA.shape} must share shape"
            )

    def __eq__(self, other: Any) -> bool:
        raise TypeError(
            "TransportResults: equality is not defined.  Two transport "
            "runs are compared dimensionally (ΔG at E_F, peak shift, "
            "integrated current, etc.), not yes/no.  Use a future "
            "transport.compare(a, b) helper for typed deltas."
        )

    def __hash__(self) -> int:  # noqa: D401
        # Mirror the __eq__ refusal — a dataclass that refuses
        # equality should also refuse hashing.
        raise TypeError(
            "TransportResults is unhashable (equality is undefined)"
        )

    # ------------------------------------------------------------------ #
    #  Wire format                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Wire encoding: nested-list arrays + scalar primitives.
        Round-trips losslessly through :meth:`from_dict`."""
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": dict(self.metadata),
            "energy_grid_eV": self.energy_grid_eV.tolist(),
            "transmission": self.transmission.tolist(),
            "fermi_energy_eV": float(self.fermi_energy_eV),
            "conductance_G0": float(self.conductance_G0),
            "pdos": {k: v.tolist() for k, v in self.pdos.items()},
            "bias_grid_V": (None if self.bias_grid_V is None
                            else self.bias_grid_V.tolist()),
            "current_uA": (None if self.current_uA is None
                           else self.current_uA.tolist()),
            "methods_text": self.methods_text,
            "bibliography_keys": list(self.bibliography_keys),
            "complete": bool(self.complete),
            "regions": {k: sorted(int(i) for i in v)
                        for k, v in self.regions.items()},
            "frozen_atoms": sorted(int(i) for i in self.frozen_atoms),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransportResults":
        """Decode a dict produced by :meth:`to_dict`.

        Back-compat: schema_version ``"1"`` is read transparently --
        the v1 -> v2 delta is purely additive (regions + frozen_atoms),
        so a v1 file decodes as a v2 record with empty boundary-
        condition fields.  This matters because pre-2026-06-25 runs
        wrote v1 sidecars and shouldn't break on re-read.

        Forward-compat: any other ``schema_version`` raises
        :class:`ValueError`.  A future v3+ writer is expected to add
        explicit migration here.
        """
        version = d.get("schema_version", SCHEMA_VERSION)
        if version not in (SCHEMA_VERSION, "1"):
            raise ValueError(
                f"TransportResults: unknown schema_version "
                f"{version!r}; this molbuilder build understands "
                f"{SCHEMA_VERSION!r} and back-compat reads {'1'!r}"
            )
        return cls(
            metadata=dict(d.get("metadata", {})),
            energy_grid_eV=np.asarray(d.get("energy_grid_eV", []),
                                       dtype=float),
            transmission=np.asarray(d.get("transmission", []),
                                     dtype=float),
            fermi_energy_eV=float(d.get("fermi_energy_eV", 0.0)),
            conductance_G0=float(d.get("conductance_G0", 0.0)),
            pdos={k: np.asarray(v, dtype=float)
                  for k, v in d.get("pdos", {}).items()},
            bias_grid_V=(None if d.get("bias_grid_V") is None
                         else np.asarray(d.get("bias_grid_V"),
                                          dtype=float)),
            current_uA=(None if d.get("current_uA") is None
                        else np.asarray(d.get("current_uA"),
                                         dtype=float)),
            methods_text=str(d.get("methods_text", "")),
            bibliography_keys=list(d.get("bibliography_keys", [])),
            complete=bool(d.get("complete", False)),
            # v2 fields -- back-compat: missing => empty (v1 sidecars).
            regions={k: [int(i) for i in v]
                     for k, v in (d.get("regions") or {}).items()},
            frozen_atoms=[int(i) for i in (d.get("frozen_atoms") or [])],
        )


__all__ = ["TransportResults", "SCHEMA_VERSION"]
