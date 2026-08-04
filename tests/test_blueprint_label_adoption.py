"""No blueprint applies labels itself -- they arrive WITH the structure.

This file used to assert the opposite, and was right to at the time: every route
that rebuilt a Structure from a request body had to remember to call
``apply_labels_to_struct(struct, body)`` afterwards, because the labels travelled
in a SEPARATE top-level key.  A route that forgot silently stripped the user's
boundary conditions -- ``/api/modify/*`` did exactly that -- so the guard
enumerated the callers and checked each one remembered.

A guard that checks everyone remembers is a guard against a design where
forgetting is possible.  Since 2026-08-03 it is not: the labels ride inside the
structure envelope, ``Structure.from_dict`` applies them as part of building the
Structure, and ``apply_labels_to_struct`` no longer exists.  There is one way in,
so there is nothing to forget -- and the test that made forgetting *visible* is
replaced by one that keeps the second door from being reopened.

The failure this now catches: a route that reads labels off the body itself,
which is how a second source comes back.  Two places labels can arrive from is
two places they can disagree, and no precedence rule fixes it -- "the envelope
had none" and "the envelope disagreed" look identical to any rule you can write.
"""
from pathlib import Path

_BLUEPRINTS_DIR = (Path(__file__).resolve().parent.parent
                   / "molbuilder" / "web" / "blueprints")

#: Reading one of these off the request body means a route is taking labels from
#: somewhere other than the structure they belong to.
_SECOND_SOURCE_READS = (
    'body.get("regions")',
    "body.get('regions')",
    'body["regions"]',
    "body['regions']",
    'body.get("frozen_atoms")',
    "body.get('frozen_atoms')",
    'body["frozen_atoms"]',
    "body['frozen_atoms']",
)


def _code_only(text: str) -> str:
    """Comments may DISCUSS the retired keys -- explaining why they went is how
    the next reader learns not to bring them back."""
    return "\n".join(
        line.split("#", 1)[0] for line in text.splitlines()
    )


def test_no_blueprint_reads_labels_off_the_request_body():
    offenders = []
    for path in sorted(_BLUEPRINTS_DIR.glob("*.py")):
        code = _code_only(path.read_text())
        for read in _SECOND_SOURCE_READS:
            if read in code:
                offenders.append(f"{path.name}: {read}")
    assert not offenders, (
        "these routes take labels from the request body instead of from the "
        "structure that carries them:\n  " + "\n  ".join(offenders)
        + "\n\nLabels live in `structure.metadata.regions` and are applied by "
          "`Structure.from_dict` -- the one deserialiser.  A second place they "
          "can arrive from is a second place they can be dropped from, which "
          "is what task #41 was."
    )


def test_the_second_applier_is_gone():
    """Named directly, so its return would be caught even if a caller spelled
    the body read in a way the list above does not cover."""
    from molbuilder.web.blueprints import _shared
    assert not hasattr(_shared, "apply_labels_to_struct"), (
        "`apply_labels_to_struct` is back.  It was the server-side half of the "
        "retired flat body shape: labels beside the structure rather than "
        "inside it.  Its last caller (the transport tab) was migrated to "
        "`molview.exportFile()` on 2026-08-03 and the function deleted."
    )


def test_labels_reach_the_structure_through_the_one_door():
    """The property the old guard protected, asserted directly rather than by
    checking that every route remembered a follow-up call."""
    from molbuilder.web.blueprints._shared import struct_from_body

    struct = struct_from_body({"structure": {
        "elements":  ["C", "H", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "metadata":  {"regions": {"L-electrode": [0], "frozen_atoms": [1, 2]}},
    }})
    assert struct.regions["L-electrode"] == [0]
    # The reserved label is an ordinary member of the one store, reachable
    # through its one designated accessor.
    assert list(struct.frozen_atoms) == [1, 2]
