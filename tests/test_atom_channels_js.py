"""Phase 4a (atom-annotations.md § 5): the PURE L1 channel model
(lib/workspace/_atom-channels.js).  Runs under Node (no browser) since the
module is pure logic — deterministic, no E2E flakiness."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_NODE = shutil.which("node")
_MODULE = (Path(__file__).resolve().parent.parent
           / "molbuilder/web/static/lib/workspace/_atom-channels.js")

pytestmark = pytest.mark.skipif(_NODE is None, reason="node not available")


def _run(script: str) -> str:
    """Run a node script that requires the module; return stdout; raise on
    a nonzero exit (a failed assertion)."""
    full = f"const M = require({json.dumps(str(_MODULE))});\n" + script
    r = subprocess.run([_NODE, "-e", full], capture_output=True, text=True)
    assert r.returncode == 0, f"node failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout.strip()


def test_atom_channels_unifies_element_residue_tags_frozen_values():
    out = _run("""
      const a = { index:0, element:"C", residueName:"ALA",
                  labels:["L-electrode","bridge"], isFrozen:true,
                  values:{charge:-1.0} };
      const ch = M.atomChannels(a);
      const A = require("assert");
      A.strictEqual(ch.element.kind, "category"); A.strictEqual(ch.element.value, "C");
      A.strictEqual(ch.residue.value, "ALA");
      A.strictEqual(ch["L-electrode"].kind, "tag");
      A.strictEqual(ch.frozen.kind, "flag");
      A.strictEqual(ch.charge.kind, "value"); A.strictEqual(ch.charge.value, -1.0);
      console.log("ok");
    """)
    assert out == "ok"


def test_atom_channels_accepts_raw_wire_shape():
    _run("""
      const A = require("assert");
      const ch = M.atomChannels({element:"Au", regions:["dev"], is_frozen:true,
                                 residue_name:"AU"});
      A.strictEqual(ch.dev.kind, "tag");
      A.strictEqual(ch.frozen.kind, "flag");
      A.strictEqual(ch.residue.value, "AU");
      console.log("ok");
    """)


def test_channel_kinds_stable_order_and_dedup():
    out = _run("""
      const atoms = [
        {element:"C", labels:["bridge"], isFrozen:false, residueName:"MOL"},
        {element:"Au", labels:["L-electrode"], isFrozen:true, values:{charge:1}},
      ];
      console.log(JSON.stringify(M.channelKinds(atoms)));
    """)
    import json as _j
    got = _j.loads(out)
    # element, residue first; tags sorted; frozen; values sorted
    assert got == [
        {"name": "element", "kind": "category"},
        {"name": "residue", "kind": "category"},
        {"name": "L-electrode", "kind": "tag"},
        {"name": "bridge", "kind": "tag"},
        {"name": "frozen", "kind": "flag"},
        {"name": "charge", "kind": "value"},
    ]


def test_empty_and_null_safe():
    _run("""
      const A = require("assert");
      A.deepStrictEqual(M.atomChannels(null), {});
      A.deepStrictEqual(M.channelKinds([]), []);
      A.deepStrictEqual(M.channelKinds(null), []);
      console.log("ok");
    """)


def test_selection_panel_frozen_name_matches_l1():
    """The panel's FROZEN_TAG_LABEL and the L1 model's FROZEN_CHANNEL are
    intentionally independent constants (different layers) so the panel needs no
    L1 load-time dependency -- this test guards them against drift."""
    import re
    l1_frozen = _run(
        "console.log(JSON.stringify(M.FROZEN_CHANNEL));").strip().strip('"')
    panel = (Path(__file__).resolve().parent.parent
             / "molbuilder/web/static/lib/selection-panel.js").read_text()
    m = re.search(r'FROZEN_TAG_LABEL\s*=\s*"([^"]+)"', panel)
    assert m, "FROZEN_TAG_LABEL literal not found in selection-panel.js"
    assert m.group(1) == l1_frozen, (
        f"frozen name drift: panel={m.group(1)!r} vs L1={l1_frozen!r}")
