"""The junctions real relaxations start from (`support.real_runs`), built in
source, each the smallest that shows what its tests read, and the layers'
positions of a chain sandwich (``tests/test_transport_compose.py``).
"""
from __future__ import annotations


def bridged_chain() -> dict:
    """The smallest junction a transport calculation's decks read fully: a
    hydrogen chain for leads, nine atoms each and held, as the real-junction
    test's are (`test_transport_on_a_real_junction_e2e.py`), with a bridge of
    two other elements, so a species order has something to order; a wire,
    isolated across in an 8 A box, one spacing of room along transport."""
    elements = ["H"] * 9 + ["C", "O"] + ["H"] * 9
    n = len(elements)
    regions = {"L-electrode": list(range(9)), "bridge": [9, 10],
               "R-electrode": list(range(11, n))}
    regions["frozen_atoms"] = regions["L-electrode"] + regions["R-electrode"]
    return {"elements": elements,
            "positions": [[4.0, 4.0, 1.1 * i] for i in range(n)],
            "regions": regions, "cell": [8.0, 8.0, 1.1 * n],
            "axis_kind": ["isolated", "isolated", "transport"]}


def uneven_chain() -> dict:
    """Hydrogen on a square lattice, stacked along transport at the spacing
    it has across, so its leads continue through the side boundaries as they
    must (`cell.transverse_vacuum`), a buffer atom past its left lead, its
    leads of eight and nine atoms and held: what a symmetric junction cannot
    tell, in the seconds hydrogen takes -- a count hardcoded for both leads, a
    buffer's anchors, the k-points across."""
    n = 22
    regions = {"buffer": [0], "L-electrode": list(range(1, 9)),
               "bridge": list(range(9, 13)), "R-electrode": list(range(13, n))}
    regions["frozen_atoms"] = regions["L-electrode"] + regions["R-electrode"]
    return {"elements": ["H"] * n,
            "positions": [[1.0, 1.0, 2.0 * i] for i in range(n)],
            "regions": regions, "cell": [2.0, 2.0, 2.0 * n],
            "axis_kind": ["periodic", "periodic", "transport"]}


# ---- a chain sandwich's layers --------------------------------------- #

#: Six 2.5 Å gold layers a side (12.5 Å span -- over the wizard's 12 Å lead
#: floor), a four-atom bridge between.
_LAYERS_L = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
_BRIDGE = [("S", 15.0), ("C", 16.4), ("C", 17.8), ("S", 19.2)]
_LAYERS_R = [22.0, 24.5, 27.0, 29.5, 32.0, 34.5]
