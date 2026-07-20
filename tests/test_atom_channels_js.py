"""Phase 4a (atom-annotations.md § 5): the PURE L1 channel model
(lib/molview/_atom-channels.js).  Runs under Node (no browser) since the
module is pure logic — deterministic, no E2E flakiness.

Loaded via the shared ES-module-capable harness (tests/_node_esm.py) and read through the
GLOBAL ``molbuilder.atomChannelModel`` -- which the file publishes whether it's a classic IIFE
or a native ES module -- so this test is stable across the MolView ESM migration.
"""
import re
from pathlib import Path

from _node_esm import run_node

_MODULE = (Path(__file__).resolve().parent.parent
           / "molbuilder/web/static/lib/molview/_atom-channels.js")

_M = "globalThis.molbuilder.atomChannelModel"   # published by classic OR ESM form


def test_atom_channels_unifies_element_residue_tags_frozen_values():
    ch = run_node([_MODULE], f"""
      const a = {{ index:0, element:"C", residueName:"ALA",
                   labels:["L-electrode","bridge"], isFrozen:true,
                   values:{{charge:-1.0}} }};
      console.log(JSON.stringify({_M}.atomChannels(a)));
    """)
    assert ch["element"] == {"kind": "category", "value": "C"}
    assert ch["residue"]["value"] == "ALA"
    assert ch["L-electrode"]["kind"] == "tag"
    assert ch["frozen"]["kind"] == "flag"
    assert ch["charge"] == {"kind": "value", "value": -1.0}


def test_atom_channels_accepts_raw_wire_shape():
    ch = run_node([_MODULE],
                  f'console.log(JSON.stringify({_M}.atomChannels('
                  f'{{element:"Au", regions:["dev"], is_frozen:true, residue_name:"AU"}})));')
    assert ch["dev"]["kind"] == "tag"
    assert ch["frozen"]["kind"] == "flag"
    assert ch["residue"]["value"] == "AU"


def test_channel_kinds_stable_order_and_dedup():
    got = run_node([_MODULE], f"""
      const atoms = [
        {{element:"C", labels:["bridge"], isFrozen:false, residueName:"MOL"}},
        {{element:"Au", labels:["L-electrode"], isFrozen:true, values:{{charge:1}}}},
      ];
      console.log(JSON.stringify({_M}.channelKinds(atoms)));
    """)
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
    out = run_node([_MODULE], f"""
      console.log(JSON.stringify({{ a: {_M}.atomChannels(null),
                                    b: {_M}.channelKinds([]),
                                    c: {_M}.channelKinds(null) }}));
    """)
    assert out == {"a": {}, "b": [], "c": []}


def test_selection_panel_frozen_name_matches_l1():
    """The panel's FROZEN_TAG_LABEL and the L1 model's FROZEN_CHANNEL are
    intentionally independent constants (different layers) so the panel needs no
    L1 load-time dependency -- this test guards them against drift."""
    l1_frozen = run_node([_MODULE], f'console.log(JSON.stringify({_M}.FROZEN_CHANNEL));')
    panel = (Path(__file__).resolve().parent.parent
             / "molbuilder/web/static/lib/molview/selection/panel.js").read_text()
    m = re.search(r'FROZEN_TAG_LABEL\s*=\s*"([^"]+)"', panel)
    assert m, "FROZEN_TAG_LABEL literal not found in selection-panel.js"
    assert m.group(1) == l1_frozen, (
        f"frozen name drift: panel={m.group(1)!r} vs L1={l1_frozen!r}")
