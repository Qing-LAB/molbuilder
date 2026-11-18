"""What 3DNA's and AmberTools' backends share: their programs write
coordinates, not bonds, so each checks its backbone after a build and keeps
the one chain it built."""

from __future__ import annotations

from typing import Optional

from ...structure import Structure


def verify_backbone_connectivity(struct: Structure) -> Optional[str]:
    """Return None if every nucleotide's link to the one before it in its
    chain is bridged (`chemistry.backbone_links`, the one rule), else a
    string describing the worst violation.  A
    duplex's strands number their residues on, so a residue's neighbour by
    number may be the other strand's, which no backbone joins -- the rule
    reads each chain on its own.

    The self-check of the backends whose programs write coordinates, not
    bonds -- 3DNA's and AmberTools' (RDKit builds the bonds itself): if it
    fails the backend raises rather than ship a broken structure.
    """
    from ...chemistry import O3P_BRIDGE_A, backbone_links
    broken = {where: link for where, link in backbone_links(struct).items()
              if not link.bridged}
    if not broken:
        return None
    (chain, b), link = max(broken.items(), key=lambda kv: kv[1].distance)
    return (f"chain {chain} residue {b - 1} O3' -> residue {b} P distance = "
            f"{link.distance:.2f} A (should be <= {O3P_BRIDGE_A}); backbone "
            f"is broken")


def first_chain(struct: Structure) -> Structure:
    """The atoms of the structure's first chain, by name: a backend returns
    one chain, whatever its program emitted."""
    first = min(struct.chain_ids, default=None)
    return struct.without(i for i, chain in enumerate(struct.chain_ids)
                          if chain != first)
