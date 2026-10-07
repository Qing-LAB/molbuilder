"""No blueprint applies labels itself -- they arrive WITH the structure.

The labels ride inside the structure envelope and ``Structure.from_dict``
applies them as part of building the ``Structure``.  One way in, so there is
nothing for a route to remember and nothing for it to forget.

The failure this catches: labels posted with a structure that never reach
it.  Two places labels can arrive from is two places they can disagree, and no
precedence rule fixes it -- "the envelope had none" and "the envelope
disagreed" look identical to any rule you can write -- so a route that reads a
second source off the request body is review's to refuse
(`process/code-audit.md` § 1c); what a test can see is the one door delivering.
"""


def test_labels_reach_the_structure_through_the_one_door():
    """Asserted directly rather than by checking that every route remembered
    a follow-up call."""
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
