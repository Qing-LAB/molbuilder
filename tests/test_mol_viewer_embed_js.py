"""Unit tests for ``lib/mol-viewer-embed.js`` pure helpers.

Implements the testing strategy from
``docs/protocols/embedded-viewer.md`` § 6 "Test coverage": the
pure-logic helpers (option normalisation, lattice-mode detection,
idempotence diffs) run under Node without 3Dmol or a browser.

The full live-mount Playwright tests come in a later commit when
the migration of /modify lands; this file covers the
no-3Dmol-dependency surface.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT   = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/mol-viewer-embed.js"


def _run_node(snippet: str) -> object:
    """Load the embed module under Node with the minimal globals it
    needs, run the snippet, return the parsed JSON output."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    bootstrap = """
        global.window = global;
        // The embed module references window.molbuilder.viewer.create
        // at embed time; the pure helpers we test don't need it but
        // we stub the namespace so module load is silent.
        global.window.molbuilder = global.window.molbuilder || {};
        global.window.molbuilder.viewer = global.window.molbuilder.viewer || {};
    """
    full = bootstrap + "\n" + MODULE.read_text() + "\n" + snippet
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", full],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- #
#  Option normalisation                                                 #
# --------------------------------------------------------------------- #


class TestNormaliseOpts:

    def test_empty_opts_yields_full_internal_shape(self):
        """``_normaliseOpts({})`` returns the full internal-state
        shape with every field defaulted -- this is the "ground
        truth" against which idempotence diffs are computed."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOpts({});
            console.log(JSON.stringify(Object.keys(r).sort()));
        ''')
        assert out == sorted([
            "xyz", "pdb", "style", "axes", "cell",
            "labels", "arrows", "pick", "lattice",
        ])

    def test_xyz_string_passes_through(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOpts({
                xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n"
            });
            console.log(JSON.stringify({
                xyz: typeof r.xyz === "string" && r.xyz.length > 0,
                pdb: r.pdb,
            }));
        ''')
        assert out == {"xyz": True, "pdb": None}

    def test_non_string_xyz_becomes_null(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOpts({xyz: 42});
            console.log(JSON.stringify({xyz: r.xyz}));
        ''')
        assert out == {"xyz": None}


class TestNormaliseAxes:

    def test_true_becomes_auto_mode(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAxes(true);
            console.log(JSON.stringify(r));
        ''')
        # length / origin / labels / colors / radius are intentionally
        # left undefined so the underlying mol-axes module's defaults
        # apply; serialisation drops them.
        assert out == {"mode": "auto"}

    def test_false_undefined_null_all_become_null(self):
        for falsy in ("false", "undefined", "null"):
            out = _run_node(
                f'''
                    const r = window.molbuilder.viewer._normaliseAxes({falsy});
                    console.log(JSON.stringify(r));
                '''
            )
            assert out is None, f"axes({falsy}) should normalise to null"

    def test_object_with_mode_preserved(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAxes({
                mode: "cartesian", length: 2.5
            });
            console.log(JSON.stringify(r));
        ''')
        assert out["mode"] == "cartesian"
        assert out["length"] == 2.5


class TestNormaliseStyle:

    def test_defaults_when_no_opts(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseStyle();
            console.log(JSON.stringify(r));
        ''')
        assert out == {
            "rep": "stick",
            "radiusScale": 1.0,
            "colorScheme": None,
            "background": "#ffffff",
            "showLabels": False,
        }

    def test_overrides_apply(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseStyle({
                rep: "sphere",
                radiusScale: 0.7,
                background: "#000000",
            });
            console.log(JSON.stringify(r));
        ''')
        assert out["rep"] == "sphere"
        assert out["radiusScale"] == 0.7
        assert out["background"] == "#000000"


# --------------------------------------------------------------------- #
#  Lattice validation                                                   #
# --------------------------------------------------------------------- #


class TestLatticeDetection:

    def test_valid_3x3_lattice_returns_copy(self):
        out = _run_node('''
            const L = [[10,0,0],[0,10,0],[0,0,15]];
            const r = window.molbuilder.viewer._normaliseLattice(L);
            console.log(JSON.stringify(r));
        ''')
        assert out == [[10, 0, 0], [0, 10, 0], [0, 0, 15]]

    def test_undefined_lattice_returns_null(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLattice();
            console.log(JSON.stringify({lattice: r}));
        ''')
        assert out == {"lattice": None}

    def test_2x3_rejected(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLattice(
                [[1,0,0],[0,1,0]]);
            console.log(JSON.stringify({lattice: r}));
        ''')
        assert out == {"lattice": None}

    def test_3x2_rejected(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLattice(
                [[1,0],[0,1],[0,0]]);
            console.log(JSON.stringify({lattice: r}));
        ''')
        assert out == {"lattice": None}

    def test_nan_or_infinity_rejected(self):
        out = _run_node('''
            const L = [[1,0,0],[0,NaN,0],[0,0,1]];
            const r = window.molbuilder.viewer._normaliseLattice(L);
            console.log(JSON.stringify({lattice: r}));
        ''')
        assert out == {"lattice": None}


# --------------------------------------------------------------------- #
#  Idempotence diffs                                                    #
# --------------------------------------------------------------------- #


class TestIdempotenceDiff:
    """``_equalNormalised`` is what each setX method uses to decide
    "did the option actually change?" before re-rendering.  Mis-
    detecting equality silently breaks the no-churn invariant from
    embedded-viewer.md § 3."""

    def test_two_default_axes_are_equal(self):
        out = _run_node('''
            const A = window.molbuilder.viewer._normaliseAxes(true);
            const B = window.molbuilder.viewer._normaliseAxes(true);
            const eq = window.molbuilder.viewer._equalNormalised(A, B);
            console.log(JSON.stringify({eq: eq}));
        ''')
        assert out == {"eq": True}

    def test_cartesian_vs_auto_axes_differ(self):
        out = _run_node('''
            const A = window.molbuilder.viewer._normaliseAxes(true);
            const B = window.molbuilder.viewer._normaliseAxes({
                mode: "cartesian"
            });
            const eq = window.molbuilder.viewer._equalNormalised(A, B);
            console.log(JSON.stringify({eq: eq}));
        ''')
        assert out == {"eq": False}

    def test_null_vs_null_equal(self):
        out = _run_node('''
            const eq = window.molbuilder.viewer._equalNormalised(null, null);
            console.log(JSON.stringify({eq: eq}));
        ''')
        assert out == {"eq": True}

    def test_null_vs_object_differ(self):
        out = _run_node('''
            const eq = window.molbuilder.viewer._equalNormalised(
                null, {mode: "auto"});
            console.log(JSON.stringify({eq: eq}));
        ''')
        assert out == {"eq": False}

    def test_pick_normalisation_drops_none_mode(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({mode: "none"});
            console.log(JSON.stringify({pick: r}));
        ''')
        assert out == {"pick": None}

    def test_pick_single_preserves_callback_type(self):
        """The onPick callback is preserved; identity isn't but the
        callable property is (so the wired handler still invokes
        the user's function)."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "single",
                onPick: function (idx) { /* test */ },
            });
            console.log(JSON.stringify({
                mode: r.mode,
                hasOnPick: typeof r.onPick === "function",
            }));
        ''')
        assert out == {"mode": "single", "hasOnPick": True}


# --------------------------------------------------------------------- #
#  Cell / labels normalisation                                          #
# --------------------------------------------------------------------- #


class TestCellAndLabels:

    def test_cell_true_yields_default_object(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseCell(true);
            console.log(JSON.stringify(r));
        ''')
        assert out == {"color": None, "radius": 0.04}

    def test_cell_false_yields_null(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseCell(false);
            console.log(JSON.stringify({cell: r}));
        ''')
        assert out == {"cell": None}

    def test_labels_true_yields_indices_mode(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels(true);
            console.log(JSON.stringify(r));
        ''')
        assert out == {"atoms": "indices", "fontSize": 12}

    def test_labels_index_list_preserved(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels({
                atoms: [0, 5, 12]
            });
            console.log(JSON.stringify(r.atoms));
        ''')
        assert out == [0, 5, 12]


# --------------------------------------------------------------------- #
#  Animation normalisation (stage 3)                                    #
# --------------------------------------------------------------------- #


class TestAnimationNormalisation:

    def test_null_or_undefined_yields_null(self):
        for falsy in ("null", "undefined"):
            out = _run_node(
                f'''
                    const r = window.molbuilder.viewer._normaliseAnimation({falsy});
                    console.log(JSON.stringify({{anim: r}}));
                '''
            )
            assert out == {"anim": None}, f"animation({falsy}) should be null"

    def test_unknown_kind_yields_null(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "spin", frames: 42
            });
            console.log(JSON.stringify({anim: r}));
        ''')
        assert out == {"anim": None}

    def test_vibration_requires_displacements_array(self):
        out = _run_node('''
            // No displacements -> invalid
            const a = window.molbuilder.viewer._normaliseAnimation({
                kind: "vibration"
            });
            // Displacements present -> valid
            const b = window.molbuilder.viewer._normaliseAnimation({
                kind: "vibration",
                displacements: [[1,0,0],[0,1,0],[0,0,1]],
            });
            console.log(JSON.stringify({
                aIsNull: a === null,
                bKind:   b ? b.kind : null,
                bAmp:    b ? b.amplitude : null,
                bHz:     b ? b.speedHz : null,
            }));
        ''')
        assert out == {
            "aIsNull": True,
            "bKind":   "vibration",
            "bAmp":    0.15,    # default
            "bHz":     1.0,     # default
        }

    def test_vibration_overrides_apply(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "vibration",
                displacements: [[1,0,0]],
                amplitude: 0.05,
                speedHz: 2.5,
                paused: true,
            });
            console.log(JSON.stringify(r));
        ''')
        assert out["kind"] == "vibration"
        assert out["amplitude"] == 0.05
        assert out["speedHz"] == 2.5
        assert out["paused"] is True

    def test_trajectory_requires_frames_array(self):
        out = _run_node('''
            // No frames -> invalid
            const a = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory"
            });
            // Empty frames array -> invalid
            const b = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [],
            });
            // Valid 2-frame trajectory
            const c = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [[[0,0,0],[1,0,0]], [[0,1,0],[1,1,0]]],
            });
            console.log(JSON.stringify({
                aIsNull: a === null,
                bIsNull: b === null,
                cKind:   c ? c.kind : null,
                cFrameCount: c ? c.frames.length : 0,
                cCurrent: c ? c.currentFrame : null,
                cFps:    c ? c.fps : null,
                cPaused: c ? c.paused : null,
                cLoop:   c ? c.loop : null,
            }));
        ''')
        assert out["aIsNull"] is True
        assert out["bIsNull"] is True
        assert out["cKind"] == "trajectory"
        assert out["cFrameCount"] == 2
        assert out["cCurrent"] == 0     # startFrame default
        assert out["cFps"] == 10        # default
        assert out["cPaused"] is True   # trajectory defaults paused
        assert out["cLoop"] is True     # default

    def test_trajectory_startframe_clamped(self):
        """A startFrame outside [0, n_frames) falls back to 0."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [[[0,0,0]],[[1,0,0]],[[2,0,0]]],
                startFrame: 99,
            });
            console.log(JSON.stringify({
                start:   r.startFrame,
                current: r.currentFrame,
            }));
        ''')
        assert out == {"start": 0, "current": 0}

    def test_trajectory_paused_defaults_true(self):
        """Trajectories don't auto-play by default -- the user
        expects to scrub via the slider OR click play."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [[[0,0,0]],[[1,0,0]]],
            });
            console.log(JSON.stringify({paused: r.paused}));
        ''')
        assert out == {"paused": True}

    def test_vibration_paused_defaults_false(self):
        """Vibration auto-plays by default -- the spectra UX
        expects "click a mode, see it move" without a separate
        play action."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "vibration",
                displacements: [[1,0,0]],
            });
            console.log(JSON.stringify({paused: r.paused}));
        ''')
        assert out == {"paused": False}
