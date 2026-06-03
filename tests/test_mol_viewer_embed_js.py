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
            "overlays", "knobs",
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
        # Per § 3.3 (review fix D3): ``showLabels`` was removed
        # because two paths to the same state caused precedence
        # ambiguity.  setLabels / opts.labels is the sole path.
        assert out == {
            "rep": "stick",
            "radiusScale": 1.0,
            "colorScheme": None,
            "background": "#ffffff",
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

    def test_labels_true_yields_all_atoms_index_format(self):
        """Per § 3.6 (review fix D2): ``labels: true`` defaults to
        labelling all atoms with index format.  The legacy
        ``atoms: "indices"`` sentinel is normalised to the split
        shape so downstream code only handles one form."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels(true);
            console.log(JSON.stringify({
                atoms:  r.atoms,
                format: r.format,
            }));
        ''')
        assert out == {"atoms": "all", "format": "index"}

    def test_labels_legacy_indices_normalises_to_split_shape(self):
        """Legacy callers may pass {atoms: "indices"}; normaliser
        converts to the modern split shape per § 3.6."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels({
                atoms: "indices",
            });
            console.log(JSON.stringify({
                atoms:  r.atoms,
                format: r.format,
            }));
        ''')
        assert out == {"atoms": "all", "format": "index"}

    def test_labels_legacy_names_normalises_to_name_format(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels({
                atoms: "names",
            });
            console.log(JSON.stringify({
                atoms:  r.atoms,
                format: r.format,
            }));
        ''')
        assert out == {"atoms": "all", "format": "name"}

    def test_labels_format_element_kept(self):
        """The new ``element`` format option from the Labels knob
        popover; must round-trip."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels({
                atoms:  "all",
                format: "element",
            });
            console.log(JSON.stringify({
                atoms:  r.atoms,
                format: r.format,
            }));
        ''')
        assert out == {"atoms": "all", "format": "element"}

    def test_labels_per_atom_list_with_format(self):
        """Per § 3.6: atoms (which) and format (what) vary
        independently."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels({
                atoms:  [1, 5, 9],
                format: "element",
            });
            console.log(JSON.stringify({
                atoms:  r.atoms,
                format: r.format,
            }));
        ''')
        assert out == {"atoms": [1, 5, 9], "format": "element"}

    def test_pick_default_yields_halo_plus_index_label(self):
        """Per § 3.8 (review fix D5): the default selection
        rendering is halo + index label (matches /modify's pre-
        embed behaviour).  No bespoke onPick wiring required."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "multi",
            });
            console.log(JSON.stringify({
                halo:  r.halo,
                label: r.label,
                style: r.style,
            }));
        ''')
        assert out["halo"]  == {"color": "#ffd54a",
                                "radius": 0.6, "opacity": 0.5}
        assert out["label"] == "index"
        assert out["style"] is None

    def test_pick_halo_false_explicitly_disables(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "multi", halo: false,
            });
            console.log(JSON.stringify({halo: r.halo}));
        ''')
        assert out == {"halo": None}

    def test_pick_halo_object_overrides_defaults(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "single",
                halo: { color: "#ff0000", radius: 0.9 },
            });
            console.log(JSON.stringify(r.halo));
        ''')
        assert out == {"color": "#ff0000", "radius": 0.9,
                       "opacity": 0.5}

    def test_pick_label_false_disables_auto_label(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "single", label: false,
            });
            console.log(JSON.stringify({label: r.label}));
        ''')
        assert out == {"label": False}

    def test_pick_label_format_options_kept(self):
        out = _run_node('''
            const r1 = window.molbuilder.viewer._normalisePick({
                mode: "single", label: "element",
            });
            const r2 = window.molbuilder.viewer._normalisePick({
                mode: "single", label: "name",
            });
            console.log(JSON.stringify({
                r1: r1.label, r2: r2.label,
            }));
        ''')
        assert out == {"r1": "element", "r2": "name"}

    def test_pick_legacy_haloColor_haloRadius_promoted(self):
        """Per § 3.8: when halo is absent and legacy haloColor /
        haloRadius are present, the legacy pair is synthesised
        into the new halo object."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "multi",
                haloColor: "#0000ff",
                haloRadius: 0.42,
            });
            console.log(JSON.stringify(r.halo));
        ''')
        assert out == {"color": "#0000ff", "radius": 0.42,
                       "opacity": 0.5}

    def test_pick_halo_supplied_ignores_legacy_fields(self):
        """Per § 3.8 deprecated-field precedence: if halo is
        supplied at all, deprecated haloColor / haloRadius are
        IGNORED entirely (no field-level merge)."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "multi",
                halo: { color: "#00ff00" },
                haloColor: "#ff0000",   // ignored
                haloRadius: 0.99,       // ignored
            });
            console.log(JSON.stringify(r.halo));
        ''')
        # halo.radius defaults from the halo object (0.6), NOT
        # from the legacy haloRadius (0.99).
        assert out == {"color": "#00ff00", "radius": 0.6,
                       "opacity": 0.5}

    def test_pick_style_override_kept(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normalisePick({
                mode: "single",
                style: { color: "#abc", opacity: 0.4 },
            });
            console.log(JSON.stringify(r.style));
        ''')
        assert out == {"color": "#abc", "opacity": 0.4}

    def test_labels_invalid_format_falls_back_to_index(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseLabels({
                atoms:  "all",
                format: "octopus",
            });
            console.log(JSON.stringify({format: r.format}));
        ''')
        assert out == {"format": "index"}

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


# --------------------------------------------------------------------- #
#  Error model (§ 3.14 + § 5)                                          #
# --------------------------------------------------------------------- #


class TestViewerError:
    """The closed error model that callers switch on by ``code``.

    These tests pin the shape (plain object, not Error subclass),
    the closed code list, and the rate-limit dispatcher invariants
    from § 5.4."""

    def test_error_codes_is_a_frozen_array_of_12(self):
        """The user-visible code union per § 5.2.  Number is
        load-bearing — adding a new code is a contract change."""
        out = _run_node('''
            const codes = window.molbuilder.viewer.ViewerErrorCodes;
            console.log(JSON.stringify({
                len:      codes.length,
                frozen:   Object.isFrozen(codes),
                has_io:   codes.indexOf("io_error") >= 0,
                has_abort: codes.indexOf("aborted") >= 0,
            }));
        ''')
        assert out == {
            "len": 12, "frozen": True,
            "has_io": True, "has_abort": True,
        }

    def test_makeError_shape_is_plain_object(self):
        """ViewerError is {code, message, cause} — NOT an Error
        subclass — so it round-trips through Promise.reject and
        structuredClone cleanly per § 5.1."""
        out = _run_node('''
            const e = window.molbuilder.viewer._makeError(
                "no_project", "no active dir", {detail: 1});
            console.log(JSON.stringify({
                code:    e.code,
                message: e.message,
                cause:   e.cause,
                isError: e instanceof Error,
                proto:   Object.getPrototypeOf(e) === Object.prototype,
            }));
        ''')
        assert out == {
            "code":    "no_project",
            "message": "no active dir",
            "cause":   {"detail": 1},
            "isError": False,
            "proto":   True,
        }

    def test_throwable_is_real_Error_subclass(self):
        """Review fix D4: synchronous throw paths (embed() hard-
        deps) must throw something ``instanceof Error`` so consumer
        try/catch + monitoring libraries handle it normally.  The
        {code, message, cause} fields are still readable on the
        Error instance."""
        out = _run_node('''
            const t = window.molbuilder.viewer._throwable;
            let caught;
            try { throw t("missing_dependency", "test msg", null); }
            catch (e) { caught = e; }
            console.log(JSON.stringify({
                isError:   caught instanceof Error,
                code:      caught.code,
                message:   caught.message,
                name:      caught.name,
                hasStack:  typeof caught.stack === "string"
                           && caught.stack.length > 0,
            }));
        ''')
        assert out == {
            "isError":  True,
            "code":     "missing_dependency",
            "message":  "test msg",
            "name":     "ViewerError",
            "hasStack": True,
        }

    def test_makeError_unknown_code_becomes_unknown(self):
        """A misspelled code in the embed itself is a programming
        error.  Fall back to "unknown" + a marker so tests catch
        it instead of silently shipping a bad code."""
        out = _run_node('''
            const e = window.molbuilder.viewer._makeError(
                "no_such_code", "oops");
            console.log(JSON.stringify({
                code:    e.code,
                hasMarker: e.message.indexOf("bad error code") >= 0,
            }));
        ''')
        assert out == {"code": "unknown", "hasMarker": True}

    def test_dispatchError_fires_onError_once(self):
        """Single error → single onError fire.  No rate-limit
        engagement on the first occurrence."""
        out = _run_node('''
            let count = 0;
            const state = { userOpts: {
                onError: (e) => { count++; }
            }};
            const err = window.molbuilder.viewer._makeError(
                "io_error", "disk full");
            window.molbuilder.viewer._dispatchError(state, err);
            console.log(JSON.stringify({count}));
        ''')
        assert out == {"count": 1}

    def test_dispatchError_rate_limits_same_code(self):
        """Per § 5.4: one fire per code per 500 ms per embed.
        Two errors with the same code in tight succession should
        fire onError exactly once."""
        out = _run_node('''
            let count = 0;
            const state = { userOpts: {
                onError: (e) => { count++; }
            }};
            const err = window.molbuilder.viewer._makeError(
                "io_error", "disk full");
            window.molbuilder.viewer._dispatchError(state, err);
            window.molbuilder.viewer._dispatchError(state, err);
            window.molbuilder.viewer._dispatchError(state, err);
            console.log(JSON.stringify({count}));
        ''')
        assert out == {"count": 1}

    def test_dispatchError_different_codes_pass_independently(self):
        """Rate-limit is per-code, not global — two DIFFERENT
        errors fire onError twice in tight succession."""
        out = _run_node('''
            let count = 0;
            const state = { userOpts: {
                onError: (e) => { count++; }
            }};
            window.molbuilder.viewer._dispatchError(state,
                window.molbuilder.viewer._makeError("io_error", "a"));
            window.molbuilder.viewer._dispatchError(state,
                window.molbuilder.viewer._makeError("no_project", "b"));
            console.log(JSON.stringify({count}));
        ''')
        assert out == {"count": 2}

    def test_dispatchError_swallows_onError_throws(self):
        """Per § 5.4: if onError throws, the embed catches and
        the original error path is uninterrupted (no propagation
        out of _dispatchError)."""
        out = _run_node('''
            let downstream = "not-reached";
            const state = { userOpts: {
                onError: () => { throw new Error("bad handler"); }
            }};
            try {
                window.molbuilder.viewer._dispatchError(state,
                    window.molbuilder.viewer._makeError("unknown", "x"));
                downstream = "reached";
            } catch (e) {
                downstream = "leaked-" + e.message;
            }
            console.log(JSON.stringify({downstream}));
        ''')
        assert out == {"downstream": "reached"}

    def test_dispatchError_noop_when_onError_absent(self):
        """No onError supplied → no fire, no throw, no console
        spam (well, we DO console.warn but that's expected
        infrastructure logging)."""
        out = _run_node('''
            const state = { userOpts: {} };  // no onError
            // Capture console.warn to confirm it fired:
            let warns = 0;
            const realWarn = console.warn;
            console.warn = () => { warns++; };
            try {
                window.molbuilder.viewer._dispatchError(state,
                    window.molbuilder.viewer._makeError("aborted", "x"));
            } finally {
                console.warn = realWarn;
            }
            console.log(JSON.stringify({warns}));
        ''')
        # Exactly one console.warn fire (the embed's "[viewer.embed]" log)
        # plus zero onError callbacks (none supplied).
        assert out == {"warns": 1}


# --------------------------------------------------------------------- #
#  OverlaySpec normalisation (§ 3.12)                                  #
# --------------------------------------------------------------------- #


class TestNormaliseOverlays:
    """Per-atom overlays drive /modify's frozen-atom greying,
    region tints, and selection halos; /spectra's spectator-atom
    greying; future custom-style use cases.  The normaliser
    must reject malformed inputs cleanly so the renderer can
    assume a uniform internal shape."""

    def test_undefined_or_null_returns_null(self):
        out = _run_node('''
            const a = window.molbuilder.viewer._normaliseOverlays(undefined);
            const b = window.molbuilder.viewer._normaliseOverlays(null);
            console.log(JSON.stringify({a, b}));
        ''')
        assert out == {"a": None, "b": None}

    def test_empty_atoms_array_returns_null(self):
        """No entries → null (renderer skips entirely)."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({atoms: []});
            console.log(JSON.stringify({r}));
        ''')
        assert out == {"r": None}

    def test_indices_selector_with_style_kept(self):
        """The most common use case (/modify frozen atoms)."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{
                    indices: [0, 2, 5],
                    style: { color: "#888", opacity: 0.55 },
                }],
            });
            console.log(JSON.stringify(r));
        ''')
        assert out == {
            "atoms": [{
                "selectorKind":  "indices",
                "selectorValue": [0, 2, 5],
                "style": {"color": "#888", "opacity": 0.55},
                "halo":   None,
                "marker": None,
            }],
        }

    def test_elements_selector_kept(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{
                    elements: ["C", "N"],
                    halo: { color: "red", radius: 0.7 },
                }],
            });
            console.log(JSON.stringify(r));
        ''')
        assert out["atoms"][0]["selectorKind"]  == "elements"
        assert out["atoms"][0]["selectorValue"] == ["C", "N"]
        assert out["atoms"][0]["halo"] == {
            "color": "red", "radius": 0.7, "opacity": 0.5,
        }

    def test_residues_selector_kept(self):
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{
                    residues: [1, 3],
                    marker: { kind: "lock", color: "#222" },
                }],
            });
            console.log(JSON.stringify(r));
        ''')
        assert out["atoms"][0]["selectorKind"]  == "residues"
        assert out["atoms"][0]["selectorValue"] == [1, 3]
        assert out["atoms"][0]["marker"] == {"kind": "lock", "color": "#222"}

    def test_no_selector_drops_entry(self):
        """Per § 3.12: exactly ONE of indices/elements/residues
        is required.  Zero selectors → dropped."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{ style: { color: "red" } }],   // no selector
            });
            console.log(JSON.stringify({r}));
        ''')
        assert out == {"r": None}

    def test_multiple_selectors_drops_entry(self):
        """Multiple selectors → also dropped (ambiguous intent)."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{
                    indices:  [1, 2],
                    elements: ["C"],
                    halo: { color: "red" },
                }],
            });
            console.log(JSON.stringify({r}));
        ''')
        assert out == {"r": None}

    def test_no_treatment_drops_entry(self):
        """Selector with no style/halo/marker is a no-op → drop."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{ indices: [0, 1] }],
            });
            console.log(JSON.stringify({r}));
        ''')
        assert out == {"r": None}

    def test_invalid_marker_kind_drops_marker_only(self):
        """Bad marker kind drops the marker field; if style/halo
        is also present, the entry survives without the marker."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{
                    indices: [0],
                    style:  { color: "red" },
                    marker: { kind: "octopus" },
                }],
            });
            console.log(JSON.stringify(r));
        ''')
        assert out["atoms"][0]["style"]  == {"color": "red"}
        assert out["atoms"][0]["marker"] is None

    def test_opacity_clamped_to_0_1(self):
        """Style opacity outside [0, 1] is clamped, not rejected."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [
                    { indices: [0], style: { opacity: 2.5 } },
                    { indices: [1], style: { opacity: -0.3 } },
                ],
            });
            console.log(JSON.stringify({
                hi: r.atoms[0].style.opacity,
                lo: r.atoms[1].style.opacity,
            }));
        ''')
        assert out == {"hi": 1.0, "lo": 0.0}

    def test_three_marker_kinds_accepted(self):
        """The closed marker enum per § 3.12: lock / star / dot."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOverlays({
                atoms: [
                    { indices: [0], marker: { kind: "lock" } },
                    { indices: [1], marker: { kind: "star" } },
                    { indices: [2], marker: { kind: "dot"  } },
                ],
            });
            console.log(JSON.stringify(r.atoms.map((e) => e.marker.kind)));
        ''')
        assert out == ["lock", "star", "dot"]

    def test_overlays_idempotence_equal(self):
        """Two equivalent overlay specs normalise to deeply-equal
        structures so _equalNormalised → no re-render."""
        out = _run_node('''
            const a = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{ indices: [1, 2], style: { color: "red" } }],
            });
            const b = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{ indices: [1, 2], style: { color: "red" } }],
            });
            console.log(JSON.stringify({
                eq: window.molbuilder.viewer._equalNormalised(a, b),
            }));
        ''')
        assert out == {"eq": True}

    def test_overlays_idempotence_differs_on_index_change(self):
        out = _run_node('''
            const a = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{ indices: [1, 2], style: { color: "red" } }],
            });
            const b = window.molbuilder.viewer._normaliseOverlays({
                atoms: [{ indices: [1, 3], style: { color: "red" } }],
            });
            console.log(JSON.stringify({
                eq: window.molbuilder.viewer._equalNormalised(a, b),
            }));
        ''')
        assert out == {"eq": False}

    def test_normaliseAnimation_trajectory_arrowsPerFrame_kept(self):
        """Per § 3.9: optional parallel array of per-frame arrow
        overlays.  Used by the trajectory inspector to flip force
        vectors per frame without 30Hz setArrows churn."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [[[0,0,0]], [[1,0,0]]],
                arrowsPerFrame: [
                    [{ start:[0,0,0], end:[1,0,0], color:"red" }],
                    [{ start:[1,0,0], end:[0,0,0], color:"blue" }],
                ],
            });
            console.log(JSON.stringify({
                hasApf: Array.isArray(r.arrowsPerFrame),
                len:    r.arrowsPerFrame.length,
                color0: r.arrowsPerFrame[0][0].color,
                color1: r.arrowsPerFrame[1][0].color,
            }));
        ''')
        assert out == {
            "hasApf": True,  "len": 2,
            "color0": "red", "color1": "blue",
        }

    def test_normaliseAnimation_trajectory_onFrame_kept(self):
        """onFrame is a function reference; non-function values
        normalise to null (so the renderer's call site can do a
        cheap typeof check)."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [[[0,0,0]]],
                onFrame: () => {},
            });
            const r2 = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [[[0,0,0]]],
                onFrame: "not a function",
            });
            console.log(JSON.stringify({
                kept: typeof r.onFrame === "function",
                drop: r2.onFrame === null,
            }));
        ''')
        assert out == {"kept": True, "drop": True}

    def test_normaliseAnimation_trajectory_apf_missing_is_null(self):
        """No arrowsPerFrame supplied → field is null (renderer
        skips the per-frame arrow branch)."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseAnimation({
                kind: "trajectory",
                frames: [[[0,0,0]]],
            });
            console.log(JSON.stringify({apf: r.arrowsPerFrame}));
        ''')
        assert out == {"apf": None}

    def test_normaliseOpts_includes_overlays(self):
        """Spot check that the overlay slot is plumbed through
        the top-level normaliser used at embed time."""
        out = _run_node('''
            const r = window.molbuilder.viewer._normaliseOpts({
                xyz: "dummy",
                overlays: {
                    atoms: [{ indices: [0], style: { color: "red" } }],
                },
            });
            console.log(JSON.stringify({
                kind: r.overlays.atoms[0].selectorKind,
                idx:  r.overlays.atoms[0].selectorValue,
            }));
        ''')
        assert out == {"kind": "indices", "idx": [0]}
