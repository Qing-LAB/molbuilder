"""The shared sender sends a tab's request through the door the tab names,
with the fields the tab hands it, and asks of none what the door does not
need (`web/handover-procedure.md`: the shared library compares no kind).

THE SILENT FAILURE.  The sender (`lib/task-handover.js`) choosing the door
or copying the fields by the kind's name: until 2.65.0 it did, so a field
it did not list -- until 2026-10-11 the electrode rename, `swap_electrodes`
-- never reached the describe door.  What this file guards is the library:
it sends through the door it is handed, every field as given, and asks for
what that door needs before sending anything.  Which fields a tab hands it
is that tab's (`transport/core.js` hands Describe's body whole); a browser
reads that composition (the F9 e2e), not this file.

Category: API -- the sender runs in node (`_node_esm.run_node`), its request
answered by the app's own route (`_node_esm.served`) and read back.  No
engine is involved, so each door refuses, by its own rule: the describe door
a junction no relaxation wrote, the hand-over door a SIESTA calculation with
no pseudopotential to start from.  An accepted request is the browser's to
follow: Describe on a junction a real relaxation wrote
(`test_transport_on_a_real_junction_e2e.py`), and a hand-over's files
written by its door (`test_build_e2e.py`, Send to Task setup).
What is read here of an answer is the sender's own rule for a refusal: the
door's sentence, word for word, and nothing written.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.envelope import envelope
from tests._node_esm import run_node, served

_ROOT = Path(__file__).resolve().parents[1]
_SENDER = _ROOT / "molbuilder/web/static/lib/task-handover.js"
#: The one reader of a server's answer, which the page's header loads before
#: any click can send (`web/ui-contract.md` § 12).
_READER = _ROOT / "molbuilder/web/static/lib/server.js"
#: ...and the one path door, which it loads beside it (`web/runtime.md` § 3).
_PATH_DOOR = _ROOT / "molbuilder/web/static/lib/path-utils.js"

pytestmark = pytest.mark.module

_JUNCTION = "P/optimization/R/01_x/run-0"

# (the tab's door, its kind, the fields its description carries): the
# Transport tab's door as `transport/core.js` names it, with the rename among
# its fields; the hand-over door every tab of a structure sends through.
ROWS = [
    pytest.param({"url": "/api/transport/describe",
                  "doing": "Describing\u2026",
                  "checked": "The description was checked:",
                  "next": "the road"},
                 "transport",
                 {"junction": _JUNCTION, "bias": [0.0], "stages": {},
                  "shared": {}, "swap_electrodes": True},
                 id="a-writing-door"),
    pytest.param("handover", "vibration",
                 {"structure": envelope(["H", "H"], [[0, 0, 0],
                                                     [0, 0, 0.74]]),
                  "params": {"temperature_k": 1}},
                 id="the-hand-over-door"),
]


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the sender (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.fixture
def dest(isolated_projects_root):
    """The folder the tab names -- the sidebar's, so one that is there --
    under this test's projects tree (`isolated_projects_root`)."""
    folder = isolated_projects_root / "P" / "transport" / "T"
    folder.mkdir(parents=True)
    return str(folder)


def _send(app_at, dest, door, calculation, fields):
    """What the sender sent through ``door`` with ``fields``, the sidebar's
    folder ``dest``; what the door answered; and what the sender said.  The
    sidebar's file layer answers its folder and nothing else: a write
    through it fails the run."""
    choice = json.dumps({"door": door, "calculation": calculation,
                         "fields": fields, "dest": dest})
    return run_node([_READER, _PATH_DOOR, _SENDER], f"""
        const choice = {choice};
        const projects = {{
            getCurrentDir: () => choice.dest,
            relativeToProjects: () => "P/transport/T",
            refresh: async () => ({{ ok: true }}),
        }};
        const handover = window.molbuilder.taskHandover;
        const door = choice.door === "handover" ? handover.HANDOVER
                                                : choice.door;
        const said = [];
        await handover.send({{
            projects, say: (tone, words) => said.push([tone, words]),
            engine: "siesta",
            calculation: choice.calculation, door, fields: choice.fields,
        }});
        const c = __calls[0];
        console.log(JSON.stringify({{
            sent: c ? {{ url: c.url, body: c.body }} : null, said,
            answer: __answered[0] || null }}));
    """, globals_js=served(app_at))


@pytest.mark.parametrize("door,calculation,fields", ROWS)
def test_the_request_goes_through_the_tabs_door_with_its_fields(
        app_at, dest, door, calculation, fields):
    got = _send(app_at, dest, door, calculation, fields)
    sent = got["sent"]
    assert sent["url"] == (door["url"] if isinstance(door, dict)
                           else "/api/task-setup/handover")
    # EVERY FIELD AS THE TAB GAVE IT -- the rename included.
    for key, value in fields.items():
        assert sent["body"][key] == value, key
    assert sent["body"]["calculation"] == calculation
    # THE FOLDER THE DOOR WRITES INTO (§ 3.8.2a).
    assert sent["body"]["dest"] == dest
    # THE DOOR'S REFUSAL, in its words, is the sender's last.
    assert got["answer"]["ok"] is False, got["answer"]
    assert got["said"][-1] == ["error", got["answer"]["error"]]


def test_a_field_the_door_needs_is_asked_for_before_anything_is_sent(
        app_at, dest):
    """The hand-over door needs a structure -- a description is OF one --
    and says so in its words, sending nothing (`handover-procedure.md`)."""
    got = _send(app_at, dest, "handover", "vibration", {"params": {}})
    assert got["sent"] is None
    assert got["said"] == [["warn", "Load a structure first — a description "
                                    "is of something."]]
