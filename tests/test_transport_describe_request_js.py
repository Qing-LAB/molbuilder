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

Category: API -- the sender runs in Node against a stand-in `fetch`, and the
request it builds is read back.  No engine is involved.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SENDER = _ROOT / "molbuilder/web/static/lib/task-handover.js"

pytestmark = [
    pytest.mark.module,
    pytest.mark.skipif(shutil.which("node") is None,
                       reason="node not installed; L2 Node-driven test"),
]

_DRIVER = r"""
const fs = require("fs"), vm = require("vm");
globalThis.window = globalThis;
let sent = null;
globalThis.fetch = async (url, init) => {
    sent = { url: url, body: JSON.parse(init.body) };
    return { ok: true, status: 200,
             json: async () => ({ ok: true, label: "T",
                                  written: ["T.template.toml", "task.json"],
                                  notices: [] }) };
};
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8"));
const choice = JSON.parse(process.argv[3]);
const projects = {
    getCurrentDir: () => "/tree/P/transport/T",
    relativeToProjects: () => "P/transport/T",
    refresh: async () => ({ ok: true }),
};
const handover = window.molbuilder.taskHandover;
const door = choice.door === "handover" ? handover.HANDOVER : choice.door;
const said = [];
handover.send({
    projects: projects, say: (tone, words) => said.push([tone, words]),
    engine: "siesta",
    calculation: choice.calculation, door: door, fields: choice.fields,
}).then(() => process.stdout.write(JSON.stringify({sent, said})));
"""

_JUNCTION = "P/optimization/R/01_x/run-0"

# (the tab's door, its kind, the fields its description carries): a writing
# door the tab names, with the rename among its fields; the hand-over door
# every tab of a structure sends through.
ROWS = [
    pytest.param({"url": "/api/transport/describe", "writes": True,
                  "next": "the road"},
                 "transport",
                 {"junction": _JUNCTION, "bias": [0.0], "stages": {},
                  "shared": {}, "swap_electrodes": True},
                 id="a-writing-door"),
    pytest.param("handover", "vibration",
                 {"structure": {"atoms": []}, "params": {"temperature_k": 1}},
                 id="the-hand-over-door"),
]


def _send(tmp_path, door, calculation, fields):
    driver = tmp_path / "drive.js"
    driver.write_text(_DRIVER)
    out = subprocess.run(
        ["node", str(driver), str(_SENDER),
         json.dumps({"door": door, "calculation": calculation,
                     "fields": fields})],
        capture_output=True, text=True, timeout=60, check=True).stdout
    return json.loads(out)


@pytest.mark.parametrize("door,calculation,fields", ROWS)
def test_the_request_goes_through_the_tabs_door_with_its_fields(
        tmp_path, door, calculation, fields):
    sent = _send(tmp_path, door, calculation, fields)["sent"]
    assert sent["url"] == (door["url"] if isinstance(door, dict)
                           else "/api/task-setup/handover")
    # EVERY FIELD AS THE TAB GAVE IT -- the rename included.
    for key, value in fields.items():
        assert sent["body"][key] == value, key
    assert sent["body"]["calculation"] == calculation
    # THE FOLDER THE DOOR WRITES INTO (§ 3.8.2a).
    assert sent["body"]["dest"] == "/tree/P/transport/T"


def test_a_field_the_door_needs_is_asked_for_before_anything_is_sent(
        tmp_path):
    """The hand-over door needs a structure -- a description is OF one --
    and says so in its words, sending nothing (`handover-procedure.md`)."""
    got = _send(tmp_path, "handover", "vibration", {"params": {}})
    assert got["sent"] is None
    assert got["said"] == [["warn", "Load a structure first — a description "
                                    "is of something."]]
