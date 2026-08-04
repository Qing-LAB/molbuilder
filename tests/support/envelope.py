"""THE way a test builds a request body carrying a structure.

`Structure.to_dict()` IS the envelope -- the same serialiser the sidecar, the
persistence layer and the CLI round-trip through, and the exact shape
`molview.exportFile()` produces and `struct_from_body` accepts.  So a test that
wants to post a structure builds a Structure and calls it.

WHY THIS FILE EXISTS.  When the request shape moved from `xyz` text to the
envelope, six test files each grew their own `_envelope()` that split an XYZ
string into elements and positions by hand.  Six hand-rolled XYZ parsers, in
the test suite, for a project whose contract says a coordinate document is the
server's format and the browser does not write one (molview.md § 11.7) -- and
each one free to drift from the real envelope in its own direction, which is
the exact duplication the code under test had just been cleaned of.

A structure a test can post should be CONSTRUCTED, not parsed out of a string:
`Structure(elements=[...], positions=[[...]])` states what it is, and
`to_dict()` puts it in the shape the door takes.  If the envelope grows a
field, it appears here for free -- there is nowhere for it to be forgotten.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from molbuilder.structure import Structure


def structure(elements: Sequence[str],
              positions: Sequence[Sequence[float]],
              *,
              regions: Optional[dict] = None,
              frozen: Optional[Sequence[int]] = None,
              cell=None,
              **fields) -> Structure:
    """A Structure built from what it IS, with its labels on it.

    ``frozen=`` is a convenience that lands in ``regions`` under the reserved
    label, because the frozen set is an ordinary label in the ONE store
    (``structure.py``: ``FROZEN_LABEL``) and there is no second key for it.
    """
    from molbuilder.structure import FROZEN_LABEL
    labels = dict(regions or {})
    if frozen is not None:
        labels[FROZEN_LABEL] = list(frozen)
    if labels:
        fields["regions"] = labels
    if cell is not None:
        fields["cell"] = cell
    return Structure(
        elements=list(elements),
        positions=np.asarray(positions, dtype=float),
        **fields,
    )


def envelope(elements, positions, **kw) -> dict:
    """The same, as the dict a request body carries under ``structure``."""
    return structure(elements, positions, **kw).to_dict()


def from_xyz(text: str, **kw) -> dict:
    """An envelope for a structure a fixture already has as XYZ TEXT.

    Parsing is done by ``Structure.from_xyz`` -- the application's own reader --
    rather than by splitting lines in the test.  Use this only where the
    fixture is genuinely a file's contents; prefer :func:`envelope` and say
    what the atoms are.
    """
    struct = Structure.from_xyz(text)
    return envelope(struct.elements, struct.positions,
                    title=struct.title or "", **kw)


def from_file(path, **kw) -> dict:
    """An envelope for a structure on disk (``.xyz`` or ``.pdb``), read through
    the application's own parsers."""
    text = path.read_text() if hasattr(path, "read_text") else open(path).read()
    if str(path).lower().endswith(".pdb"):
        struct = Structure.from_pdb(text)
    else:
        struct = Structure.from_xyz(text)
    out = envelope(struct.elements, struct.positions,
                   title=struct.title or "", **kw)
    # Identity columns a PDB carries and an XYZ does not.
    for col in ("atom_names", "residue_ids", "residue_names", "chain_ids"):
        val = getattr(struct, col, None)
        if val:
            out[col] = list(val)
    return out
