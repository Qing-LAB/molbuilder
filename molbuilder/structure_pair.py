"""THE SAVED STRUCTURE PAIR'S REFUSALS -- a structure file and the
``.molstruct.json`` sidecar beside it (`model/structure-molstruct.md` § 3):
a sidecar that does not read, metadata that is not for this structure, a
geometry changed since the pair was written.

MODULE  structure_pair (L1; the standard library and `persist`'s mark)
ROLE    the three refusals, named once: the codec (`sidecars.molstruct`)
        raises them, and the structure (`structure.Structure`) raises the
        pairing refusal for a deck's atom-metadata block -- one name for one
        condition
TRAVELS beside a PySCF script and a vibration finish, with the codec and
        the structure (`runwrap.members`), and `persist` with it
        (:data:`TRAVELS_WITH`)
"""
from __future__ import annotations

try:                                        # inside molbuilder
    from .persist import FileRefused
except ImportError:                         # beside a job, in a bundle
    from persist import FileRefused  # type: ignore[no-redef]


class MolstructJsonError(ValueError):
    """Sidecar JSON is malformed, refers to a different structure, or
    fails an invariant check -- a payload built in code its builder
    refuses, or a structure pair that does not read
    (:class:`MolstructUnreadable`).  Distinct exception type so callers can
    differentiate "user-error sidecar" from "I/O failure"."""


class MolstructUnreadable(MolstructJsonError, FileRefused):
    """A STRUCTURE PAIR THAT DOES NOT READ -- its sidecar's bytes, its keys,
    its fields, its version, its atoms or geometry not the file's -- with
    the mark every reader's refusal of a file carries (`persist.FileRefused`,
    `execution/job-contracts.md` § 6.1).  A payload built in code that its
    builder refuses carries none."""


class MolstructVersionError(MolstructUnreadable):
    """A SIDECAR OF A VERSION THIS BUILD DOES NOT READ -- a pair saved before
    the one version it reads, or one a later molbuilder wrote
    (`structure-molstruct.md` § 2): its own kind, carrying ``fact`` -- the
    version found, when or by whom such a pair was written and the one
    version read, without the pair's way on -- and ``later``, whether a
    later molbuilder wrote it, so a door that holds such a pair for a
    calculation says that calculation's way on instead, the later one's
    way when it is later (`transport.compose.recorded_junction`,
    `jobset.rung.structure_for`, `vibration_frames`)."""

    def __init__(self, message: str, *, fact: str, later: bool = False):
        super().__init__(message)
        self.fact = fact
        self.later = later


class MolstructPairingError(MolstructUnreadable):
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


class GeometryChangedError(MolstructUnreadable):
    """THESE ATOMS, AT OTHER COORDINATES: the sidecar's geometry hash is not
    the file's -- its atom lines or the cell changed after the pair was
    written, so its labels and cell may sit on the wrong atoms
    (`structure-molstruct.md` § 3).  Not a :class:`MolstructPairingError`:
    the labels still apply, atom for atom, so a door that only shows the
    structure opens it, saying so, and the person saves it anew; every door
    that builds something from it refuses."""


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.persist",)


__all__ = ["GeometryChangedError", "MolstructJsonError",
           "MolstructPairingError", "MolstructUnreadable",
           "MolstructVersionError"]
