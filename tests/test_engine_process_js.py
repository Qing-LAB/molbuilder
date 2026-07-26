"""process -- the PURE per-frame processor (molview-render-streamline.md §2, §7.3).

Node unit test: processFrame is a pure function. We assert §2 end to end -- the selection
filter, the drawn->original sourceIndex map, original-index labels under isolation, the
selection-highlight index list (§8.1: WHICH atoms to glow, isolate off only), and force→arrow
derivation. The output carries semantic content, never rendering style (§7.3).
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molview/engine/process.js"
# process.js reuses the L1 index helper for 1-based (SIESTA) label text -- load it first.
INDEX_HELPER = ROOT / "molbuilder/web/static/lib/molview/_atom-index.js"


_BOOT = """
    const processFrame = globalThis.molbuilder.molview.engine.process.processFrame;
    // 4 atoms on the x-axis. identity is elements only -- process no longer reads annotations
    // (region/frozen halos removed, §8.1); the panel reads annotations straight from molview.data.
    const COORDS = [[0,0,0],[1,0,0],[2,0,0],[3,0,0]];
    const IDENTITY = { elements: ["C","N","O","H"] };
"""


def _run_node(snippet: str) -> object:
    # ES-module harness (tests/_node_esm): imports the L1 index module (ESM) + the process
    # module (classic IIFE), each publishing its global; the snippet reads through the global.
    return run_node([INDEX_HELPER, MODULE], _BOOT + snippet)


def test_no_isolate_draws_all_atoms_identity_source():
    out = _run_node("""
        const pf = processFrame({ coords: COORDS, forces: null }, IDENTITY,
                                { selection: [1, 3], isolate: false });
        console.log(JSON.stringify({ positions: pf.positions, sourceIndex: pf.sourceIndex,
                                     elements: pf.elements }));
    """)
    # selection WITHOUT isolate does not filter -- every atom drawn, identity source map.
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    assert out["sourceIndex"] == [0, 1, 2, 3]
    assert out["elements"] == ["C", "N", "O", "H"]


def test_isolate_keeps_only_selected_and_maps_source_index():
    out = _run_node("""
        const pf = processFrame({ coords: COORDS, forces: null }, IDENTITY,
                                { selection: [3, 1], isolate: true });   // out-of-order selection
        console.log(JSON.stringify({ positions: pf.positions, sourceIndex: pf.sourceIndex,
                                     elements: pf.elements }));
    """)
    # drawn in ORIGINAL order (ascending): atoms 1 and 3 kept.
    assert out["positions"] == [[1, 0, 0], [3, 0, 0]]
    assert out["sourceIndex"] == [1, 3]            # drawn m -> original atom
    assert out["elements"] == ["N", "H"]


def test_index_labels_show_original_index_under_isolation():
    out = _run_node("""
        const pf = processFrame({ coords: COORDS, forces: null }, IDENTITY,
                                { selection: [1, 3], isolate: true, showIndex: true });
        console.log(JSON.stringify({ labels: pf.labels }));
    """)
    # two drawn atoms; the label TEXT is the ORIGINAL index, 1-based (SIESTA convention via
    # atomIndexModel.toDisplay): original atoms 1 and 3 -> display "2" and "4". NOT the drawn
    # 0/1. Placed at the drawn atom's position.
    assert out["labels"] == [
        {"position": [1, 0, 0], "text": "2"},
        {"position": [3, 0, 0], "text": "4"},
    ]


def test_labels_null_when_showindex_off():
    out = _run_node("""
        const pf = processFrame({ coords: COORDS, forces: null }, IDENTITY, { showIndex: false });
        console.log(JSON.stringify({ labels: pf.labels }));
    """)
    assert out["labels"] is None


def test_selection_highlight_is_drawn_indices_when_isolate_off():
    out = _run_node("""
        const pf = processFrame({ coords: COORDS, forces: null }, IDENTITY,
                                { selection: [1, 3], isolate: false });
        console.log(JSON.stringify({ selection: pf.selection }));
    """)
    # §8.1: isolate off draws every atom in original order, so the highlight list is just the
    # selected atoms' DRAWN indices (== original index here). The embed glows these -- no style
    # is baked into the data, and there is no region/frozen halo.
    assert out["selection"] == [1, 3]


def test_selection_highlight_null_under_isolate():
    out = _run_node("""
        const pf = processFrame({ coords: COORDS, forces: null }, IDENTITY,
                                { selection: [1, 3], isolate: true });
        console.log(JSON.stringify({ selection: pf.selection }));
    """)
    # Under isolate the selection IS the entire drawn set -- an in-view highlight would add
    # nothing, so the engine sends null.
    assert out["selection"] is None


def test_selection_highlight_null_when_nothing_selected():
    out = _run_node("""
        const empty = processFrame({ coords: [[0,0,0]] }, { elements: ["C"] },
                                   { selection: [], isolate: false });
        const none  = processFrame({ coords: [[0,0,0]] }, { elements: ["C"] }, {});
        console.log(JSON.stringify({ empty: empty.selection, none: none.selection }));
    """)
    assert out["empty"] is None
    assert out["none"] is None


def test_forces_build_styled_arrows_for_drawn_atoms_with_scale():
    out = _run_node("""
        const forces = [[0,0,0],[0,1,0],[0,0,0],[0,0,2]];
        const pf = processFrame({ coords: COORDS, forces: forces }, IDENTITY,
                                { selection: [1, 3], isolate: true, showForces: true, forceScale: 2 });
        console.log(JSON.stringify({ arrows: pf.arrows }));
    """)
    arrows = out["arrows"]
    # drawn atoms 1 and 3, both non-zero -> two arrows; end = pos + force*scale(2).
    assert len(arrows) == 2
    assert arrows[0]["start"] == [1, 0, 0] and arrows[0]["end"] == [1, 2, 0]  # (0,1,0)*2 -> +2y
    assert arrows[1]["start"] == [3, 0, 0] and arrows[1]["end"] == [3, 0, 4]  # (0,0,2)*2 -> +4z
    # Styling (§2.4): the LARGEST drawn force (atom3, |f|=2) is gold-highlighted; the rest ramp
    # dim-red -> orange-red; radius grows with magnitude so the max force is the thickest arrow.
    assert arrows[1]["color"] == "#ffc400"
    assert arrows[0]["color"].startswith("rgb(")
    assert arrows[1]["radius"] > arrows[0]["radius"]


def test_forces_zero_vectors_are_suppressed_no_arrow():
    # A consumer hides a force (a frozen atom, or a sub-threshold magnitude) by handing a
    # ZERO vector; the engine draws NO arrow for it (process.js §2.4) -- so the consumer owns
    # WHICH forces show, the engine owns HOW they look.
    out = _run_node("""
        const forces = [[0,0,3],[0,0,0],[0,0,1],[0,0,0]];
        const pf = processFrame({ coords: COORDS, forces: forces }, IDENTITY,
                                { showForces: true, forceScale: 1 });
        console.log(JSON.stringify({ n: pf.arrows ? pf.arrows.length : null }));
    """)
    # atoms 0 and 2 carry a force; 1 and 3 are zeroed -> only two arrows drawn.
    assert out["n"] == 2


def test_arrows_null_without_forces_or_when_overlay_off():
    out = _run_node("""
        const noForce = processFrame({ coords: COORDS, forces: null }, IDENTITY, { showForces: true });
        const off     = processFrame({ coords: COORDS, forces: [[0,0,1],[0,0,1],[0,0,1],[0,0,1]] },
                                     IDENTITY, { showForces: false });
        console.log(JSON.stringify({ noForce: noForce.arrows, off: off.arrows }));
    """)
    assert out["noForce"] is None
    assert out["off"] is None


def test_single_frame_is_just_processed_once():
    # §2.1 single frame = frame[0]; the processor runs the same on a one-frame set.
    out = _run_node("""
        const pf = processFrame({ coords: [[5,5,5]], forces: null },
                                { elements: ["Fe"] }, {});
        console.log(JSON.stringify({ positions: pf.positions, sourceIndex: pf.sourceIndex }));
    """)
    assert out["positions"] == [[5, 5, 5]]
    assert out["sourceIndex"] == [0]
