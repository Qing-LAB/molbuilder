"""The browser's one path door (`lib/path-utils.js`, `web/runtime.md` § 3):
what a page works out of a path -- its name, its folder, a name joined to a
folder, where it lies in a tree -- by one rule each.

THE SILENT FAILURE.  Until the door, about twenty page sites worked these out
by hand, and "inside the projects folder" was a bare prefix: a folder named
`projects2` beside the tree counted as inside it, so the sidebar restored
into it and the notebook opened at `2/...`.  Each row is a rule a caller
leans on; the `projects2` rows are the ones the hand-written tests broke.

Category: API -- the module runs in node (`_node_esm.run_node`); no server,
no page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node

DOOR = (Path(__file__).resolve().parents[1]
        / "molbuilder/web/static/lib/path-utils.js")

pytestmark = pytest.mark.module

ROOT = "/t/projects"

#: (the call, its arguments, the answer).
ROWS = [
    # Where a path lies under a folder: the folder itself, below it, or not
    # at all -- a sibling whose name merely starts the same is outside.
    ("relative", [ROOT, ROOT], ""),
    ("relative", [ROOT + "/", ROOT], ""),
    ("relative", [ROOT + "/P/structure", ROOT], "P/structure"),
    ("relative", ["/t/projects2/P", ROOT], None),
    ("relative", ["/elsewhere/P", ROOT], None),
    ("isInside", [ROOT + "/P", ROOT], True),
    ("isInside", ["/t/projects2", ROOT], False),
    # A file's name and its folder.
    ("basename", [ROOT + "/P/structure/w.xyz"], "w.xyz"),
    ("dirname", [ROOT + "/P/structure/w.xyz"], ROOT + "/P/structure"),
    ("dirname", ["w.xyz"], ""),
    # A name joined to a folder: one separator, however the folder ends.
    ("join", [ROOT + "/P/", "w.xyz"], ROOT + "/P/w.xyz"),
    ("join", [ROOT + "/P", "w.xyz"], ROOT + "/P/w.xyz"),
]


@pytest.mark.parametrize("call, args, answer", ROWS,
                         ids=[f"{c}{tuple(a)}" for c, a, _ in ROWS])
def test_the_path_door_answers_by_its_rule(call, args, answer):
    got = run_node(
        [], f"const door = await import({json.dumps(DOOR.as_uri())});\n"
            f"console.log(JSON.stringify("
            f"door.{call}(...{json.dumps(args)})));")
    assert got == answer, (call, args, got)

