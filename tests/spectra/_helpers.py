"""Shared helpers for the spectra test suite: the structure and the config a
vibration deck is prepared from, no result of a run among them.

Plain module -- import directly:

    from tests.spectra._helpers import _spectra_cfg

(Not a pytest conftest fixture because the call sites use these as
factory functions with optional arguments -- fixture indirection would
add noise without value.)
"""

from __future__ import annotations

import numpy as np

from molbuilder.structure import Structure


def _struct_water() -> Structure:
    """Cheap real Structure for water -- used by engine preflight +
    script-template tests that need a Structure (not just a config)."""
    return Structure(
        elements  = ["O", "H", "H"],
        positions = np.array([[0., 0., 0.],
                              [0.96, 0., 0.],
                              [-0.24, 0.93, 0.]]),
    )


def _spectra_cfg(struct: "Structure" = None, **overrides):
    """The config shape the vibration deck and the Methods fragment
    actually read at run time.

    This builds the real thing: `PySCFConfig` seen through
    `pyscf.vibration_deck.VibrationConfigView`, which is the object the emitters
    and the kind's validation both receive.  A test written against this
    cannot pass on a shape production never sees.
    """
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.pyscf.vibration_deck import VibrationConfigView
    from molbuilder.structure import FROZEN_LABEL

    if struct is None:
        struct = _struct_water()
    # FROZEN ATOMS LIVE ON THE STRUCTURE, not on the config -- the region
    # store is where the browser puts them and where the deck reads them.
    # Accepting them here as a keyword and routing them to the store keeps
    # the call sites short without letting a test pretend the config
    # carries something it does not.
    frozen = overrides.pop("frozen_indices", None)
    if frozen is not None:
        regions = dict(getattr(struct, "regions", None) or {})
        regions[FROZEN_LABEL] = list(frozen)
        # Rebuilt, not `dataclasses.replace`d: replace() comes back with
        # regions EMPTY on this class, so a frozen set handed to it would
        # silently vanish and the test would pass against no frozen atoms.
        struct = type(struct)(elements=list(struct.elements),
                              positions=struct.positions.copy(),
                              regions=regions)
    return VibrationConfigView(PySCFConfig(**overrides), struct)
