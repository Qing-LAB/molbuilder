"""THE SAVED STRUCTURE PAIR'S REFUSALS -- a structure file and the
``.molstruct.json`` sidecar beside it (`model/structure-molstruct.md` § 3):
a sidecar that does not read, metadata that is not for this structure, a
geometry changed since the pair was written.

MODULE  structure_pair (L1; stdlib)
ROLE    the three refusals, named once: the codec (`sidecars.molstruct`)
        raises them, and the structure (`structure.Structure`) raises the
        pairing refusal for a deck's atom-metadata block -- one name for one
        condition
TRAVELS beside a PySCF script and a vibration finish, with the codec and
        the structure (`runwrap.PYSCF_COMPANIONS`, `VIBRATION_COMPANIONS`)
"""
from __future__ import annotations


class MolstructJsonError(ValueError):
    """Sidecar JSON is malformed, refers to a different structure, or
    fails an invariant check.  Distinct exception type so callers can
    differentiate "user-error sidecar" from "I/O failure"."""


class MolstructPairingError(MolstructJsonError):
    """THIS METADATA IS NOT FOR THIS STRUCTURE -- the one condition above
    that no surface may answer leniently.

    `structure-molstruct.md` § 3: the atom and frame count, checked here on
    apply, answers *does this metadata describe these atoms?* -- and the
    answer is binary, because the metadata is
    indexed by atom position and a near-miss mis-assigns every label after
    the first inserted atom.  So § 2 says a mismatch "is refused, never
    mis-applied", and `apply_to_structure` is documented as never partially
    applying one.

    Separate TYPE, not a separate message, because the surfaces above differ
    on everything else.  A block whose *form* this build cannot read (a
    pre-v7 run's top-level `frozen_atoms`) is a version fact: the in-script
    block's surface answers it by loading without the labels and saying why,
    since refusing there would leave a finished run unopenable and so
    unfixable.  A block for a *different structure* is not a version fact and
    gets no such leniency anywhere.
    """


class GeometryChangedError(MolstructJsonError):
    """THESE ATOMS, AT OTHER COORDINATES: the sidecar's geometry hash is not
    the file's -- its atom lines or the cell changed after the pair was
    written, so its labels and cell may sit on the wrong atoms
    (`structure-molstruct.md` § 3).  Not a :class:`MolstructPairingError`:
    the labels still apply, atom for atom, so a door that only shows the
    structure opens it, saying so, and the person saves it anew; every door
    that builds something from it refuses."""


__all__ = ["GeometryChangedError", "MolstructJsonError",
           "MolstructPairingError"]
