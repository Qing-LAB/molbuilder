"""The Transport tab's Describe request carries the choice the tab holds
(`engines/transport.md` § 4, § 3.8.2a).

THE SILENT FAILURE.  The tab holds a choice about its own copy of the cited
junction -- the electrode rename -- and shows the junction composed with it;
the sender (`lib/task-handover.js`) builds the request the describe door
writes the description from.  A choice the sender does not copy is a
description that silently differs from what the tab showed: until 2026-10-11 the sender never sent `swap_electrodes`, so a junction
shown with its leads traded was described with them as cited, and biased at
the other end.  The mapping crosses two files (`transport/core.js` hands the
options, the sender builds the body), which is why reading either alone did
not catch it.

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
window.molbuilder.taskHandover.send(Object.assign({
    projects: projects, say: () => {}, engine: "siesta",
    calculation: "transport", junction: "P/optimization/R/01_x/run-0",
    bias: [0.0], stages: {}, shared: {},
}, choice)).then(() => process.stdout.write(JSON.stringify(sent)));
"""


@pytest.mark.parametrize("swap", [True, False])
def test_the_describe_request_carries_the_tabs_rename_and_its_folder(
        tmp_path, swap):
    driver = tmp_path / "drive.js"
    driver.write_text(_DRIVER)
    out = subprocess.run(
        ["node", str(driver), str(_SENDER),
         json.dumps({"swap_electrodes": swap})],
        capture_output=True, text=True, timeout=60, check=True).stdout
    sent = json.loads(out)
    assert sent["url"] == "/api/transport/describe"
    assert sent["body"]["swap_electrodes"] is swap
    # THE FOLDER THE DOOR WRITES INTO (§ 3.8.2a).
    assert sent["body"]["dest"] == "/tree/P/transport/T"
