"""No blueprint applies labels itself -- they arrive WITH the structure.

The labels ride inside the structure envelope and ``Structure.from_dict``
applies them as part of building the ``Structure``.  One way in, so there is
nothing for a route to remember and nothing for it to forget.

The failure this catches: a route that reads labels off the request body
itself, which is how a second source comes back.  Two places labels can arrive
from is two places they can disagree, and no precedence rule fixes it -- "the
envelope had none" and "the envelope disagreed" look identical to any rule you
can write.
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
