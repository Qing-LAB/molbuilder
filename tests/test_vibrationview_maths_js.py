"""VibrationView's maths — derived from the contract, not from the source.

Every test below names the rule in ``docs/web/vibrationview.md`` it exists to
guard, and asserts what that rule obliges (§ 14's table).  None of them reads the
implementation to decide what to expect.

Two things about the shape of this file are the point rather than an accident:

  * **No stubs.**  Not a fake ``requestAnimationFrame``, not a fake clock, not a
    stand-in viewer.  Level 2 is pure (§ 7), so a scatter is a function call.  The
    module this replaces could not be tested this way — its maths and its clock
    were the same object, so the old suite hand-rolled a rAF queue and pumped it
    by hand just to check an eigenvector.

  * **Reached by import, through the path the browser serves.**  Nothing is
    published to a global for a test to find, because a seam a test can reach is a
    seam anything can reach (§ 4).  A node-level test importing an internal file
    is not that seam: it is a build-time path, not a runtime discovery.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"

IMPORT = (
    'const M = await import("/static/lib/vibrationview/_maths.js");\n'
)


def _run(snippet: str) -> object:
    return run_node([], IMPORT + snippet, static_root=STATIC)


# ---------------------------------------------------------------------------
# § 6.3 — the free/held-still partition is one fact, given once
# ---------------------------------------------------------------------------

def test_held_still_is_exactly_the_complement_of_the_basis():
    """§ 6.3: held-still atoms are exactly the complement of the basis.

    The set is DERIVED, never given, so it cannot contradict the basis.
    """
    out = _run("""
        console.log(JSON.stringify({
            some:   M.scatter([[1,0,0],[0,1,0]], [1, 3], 5).heldStill,
            all:    M.scatter([], [], 4).heldStill,
            none:   M.scatter([[1,0,0],[0,1,0],[0,0,1]], [0,1,2], 3).heldStill,
            noBasis: M.scatter([[1,0,0],[0,1,0],[0,0,1]], null, 3).heldStill,
        }));
    """)
    assert out["some"] == [0, 2, 4]
    assert out["all"] == [0, 1, 2, 3]      # a basis naming nobody holds all still
    assert out["none"] == []
    assert out["noBasis"] == []            # no basis -> every atom moves (§ 6.3 table)


def test_scatter_places_free_rows_on_the_atoms_the_basis_names():
    """§ 6.3: the scatter turns free-atom rows into a full per-atom array with
    zeros where nothing moves."""
    out = _run("""
        console.log(JSON.stringify(
            M.scatter([[1, 0, 0], [0, 2, 0]], [2, 0], 4).displacements));
    """)
    assert out == [[0, 2, 0],    # basis row 1 -> atom 0
                   [0, 0, 0],    # held still
                   [1, 0, 0],    # basis row 0 -> atom 2
                   [0, 0, 0]]    # held still


def test_scatter_is_correct_when_the_basis_is_not_a_sorted_run():
    """§ 6.3: "authoritative even when the free set is not a sorted run".

    A length check would silently mis-order this: the row count happens to equal
    the atom count, yet the rows belong to atoms in a different order.
    """
    out = _run("""
        console.log(JSON.stringify(
            M.scatter([[1, 0, 0], [2, 0, 0], [3, 0, 0]], [2, 0, 1], 3).displacements));
    """)
    assert out == [[2, 0, 0], [3, 0, 0], [1, 0, 0]]


def test_scatter_with_no_basis_uses_the_rows_as_given():
    """§ 6.3 table: basis absent -> the rows are already one per atom, in order."""
    out = _run("""
        console.log(JSON.stringify(
            M.scatter([[1, 0, 0], [0, 1, 0]], null, 2).displacements));
    """)
    assert out == [[1, 0, 0], [0, 1, 0]]


# ---------------------------------------------------------------------------
# § 6.3 — a mode that does not fit its structure is refused, never padded
# ---------------------------------------------------------------------------

def test_a_mode_that_does_not_fit_the_structure_is_refused():
    """§ 6.3: "the door says no, and nothing is drawn".

    Each of these is a mode computed against a different molecule.  Padding with
    zeros would animate the structure partially, plausibly, and wrongly — which is
    exactly the failure that cannot be seen by looking at it.
    """
    out = _run("""
        function refused(fn) {
            try { fn(); return false; } catch (e) { return true; }
        }
        console.log(JSON.stringify({
            // the basis names an atom the structure does not have
            outOfRange:  refused(() => M.scatter([[1,0,0]], [7], 3)),
            negative:    refused(() => M.scatter([[1,0,0]], [-1], 3)),
            // more moving atoms than there are atoms
            tooManyRows: refused(() => M.scatter([[1,0,0],[0,1,0],[0,0,1],[1,1,1]],
                                                 [0,1,2,3], 3)),
            // rows and basis disagree about how many atoms move
            rowsVsBasis: refused(() => M.scatter([[1,0,0],[0,1,0]], [0], 3)),
            // no basis, and the rows do not cover the structure
            noBasisShort: refused(() => M.scatter([[1,0,0]], null, 3)),
            // one atom named twice: two rows claiming the same atom
            duplicate:   refused(() => M.scatter([[1,0,0],[0,1,0]], [1,1], 3)),
            // a row that is not a finite vector
            badRow:      refused(() => M.scatter([[1,0,"x"]], [0], 2)),
            nanRow:      refused(() => M.scatter([[1,0,NaN]], [0], 2)),
            // ...and a mode that DOES fit is not refused
            good:        refused(() => M.scatter([[1,0,0]], [0], 2)) === false,
        }));
    """)
    assert out["outOfRange"] is True
    assert out["negative"] is True
    assert out["tooManyRows"] is True
    assert out["rowsVsBasis"] is True
    assert out["noBasisShort"] is True
    assert out["duplicate"] is True
    assert out["badRow"] is True
    assert out["nanRow"] is True
    assert out["good"] is True


def test_a_refused_mode_produces_nothing_at_all():
    """§ 6.3: "nothing is drawn" — a refusal is not a partial answer.

    Guards the failure mode this rule exists for: a caller that swallowed the
    error must not be left holding a half-scattered array.
    """
    out = _run("""
        let value = "no value returned";
        try { value = M.scatter([[1,0,0],[0,1,0]], [0, 9], 3); }
        catch (e) { value = "threw"; }
        console.log(JSON.stringify({ value }));
    """)
    assert out["value"] == "threw"


# ---------------------------------------------------------------------------
# § 10 — how a frame gets drawn
# ---------------------------------------------------------------------------

def test_positions_are_the_equilibrium_plus_amplitude_times_cos_phase():
    """§ 10: position_i(φ) = equilibrium_i + amplitude · cos(φ) · displacement_i."""
    out = _run("""
        const eq   = [[0,0,0], [1,0,0]];
        const disp = [[0,0,0], [1,0,0]];        // atom 0 held still
        console.log(JSON.stringify({
            peak:  M.positionsAtPhase(eq, disp, 0.2, 0),              // cos = 1
            zero:  M.positionsAtPhase(eq, disp, 0.2, Math.PI / 2),    // cos = 0
            trough: M.positionsAtPhase(eq, disp, 0.2, Math.PI),       // cos = -1
        }));
    """)
    assert out["peak"] == [[0, 0, 0], [1.2, 0, 0]]
    assert out["trough"] == [[0, 0, 0], [0.8, 0, 0]]
    # at the zero crossing every atom is back at its equilibrium
    assert out["zero"][0] == [0, 0, 0]
    assert abs(out["zero"][1][0] - 1.0) < 1e-12


def test_held_still_atoms_never_move_at_any_phase():
    """§ 10: "held-still atoms have a zero displacement row, so they are not a
    special case in the loop — they simply do not move"."""
    out = _run("""
        const eq   = [[0,0,0], [1,1,1], [2,0,0]];
        const disp = M.scatter([[1,1,1]], [1], 3).displacements;      // only atom 1 moves
        const N = M.rate(30, 1.0).framesPerCycle;
        const moved = [];
        for (let n = 0; n < N; n++) {
            const p = M.positionsAtPhase(eq, disp, 3.0, M.phaseOfFrame(n, N));
            for (const i of [0, 2]) {
                if (p[i][0] !== eq[i][0] || p[i][1] !== eq[i][1]
                    || p[i][2] !== eq[i][2]) moved.push([n, i]);
            }
        }
        console.log(JSON.stringify({ moved }));
    """)
    assert out["moved"] == []


# ---------------------------------------------------------------------------
# § 10.1 — frames, not a wall clock
# ---------------------------------------------------------------------------

def test_a_cycle_is_a_whole_number_of_frames_and_closes_exactly():
    """§ 10.1: "a cycle is a whole number of frames at every rate, and frame 0 of
    the next cycle holds exactly the positions of frame 0 of this one" — which is
    what lets a one-cycle export loop without a seam."""
    out = _run("""
        const eq   = [[0,0,0], [1,0,0]];
        const disp = [[0,0,0], [1,0,0]];
        const rates = [[30, 1.0], [25, 0.3], [60, 1.0], [24, 1.7], [5, 4.0]];
        const rows = rates.map(([fps, cyc]) => {
            const N = M.rate(fps, cyc).framesPerCycle;
            const first = M.positionsAtPhase(eq, disp, 0.2, M.phaseOfFrame(0, N));
            const next  = M.positionsAtPhase(eq, disp, 0.2, M.phaseOfFrame(N, N));
            const mid   = M.positionsAtPhase(eq, disp, 0.2, M.phaseOfFrame(7, N));
            const midNext = M.positionsAtPhase(eq, disp, 0.2,
                                               M.phaseOfFrame(7 + 3 * N, N));
            return { fps, cyc, N,
                     whole:  Number.isInteger(N),
                     closes: JSON.stringify(first) === JSON.stringify(next),
                     stable: JSON.stringify(mid) === JSON.stringify(midNext) };
        });
        console.log(JSON.stringify(rows));
    """)
    for row in out:
        assert row["whole"] is True, row
        assert row["closes"] is True, row      # frame N == frame 0
        assert row["stable"] is True, row      # and it does not drift over cycles


def test_the_rounding_lands_on_the_duration_not_on_the_frame_count():
    """§ 10.1: a fractional frames-per-cycle is accepted, and what shifts is the
    cycle LENGTH, not the frame count.

    25 fps over 0.3 s is 7.5 frames; the count rounds to 8 and the cycle becomes
    0.32 s.  Refusing the rate instead would make a smoothness control throw at a
    user for dragging a slider.
    """
    out = _run("""
        console.log(JSON.stringify({
            even:      M.rate(30, 1.0).framesPerCycle,
            fractional: M.rate(25, 0.7).framesPerCycle,   // 17.5 -> 18
            roundsDown: M.rate(24, 1.7).framesPerCycle,   // 40.8 -> 41
        }));
    """)
    assert out["even"] == 30
    assert out["fractional"] == 18            # round(17.5)
    assert out["roundsDown"] == 41            # round(40.8)
    # the effective cycle length is what moved: 18 frames at 25 fps = 0.72 s
    assert abs(out["fractional"] / 25 - 0.72) < 1e-9


def test_a_rate_out_of_range_is_brought_into_it():
    """§ 10.1: "a frame rate below the floor or above the ceiling is brought into
    range rather than honoured or refused".

    This is the arithmetic half.  Whether the DOOR clamps — which is what stops
    the clock dividing by zero — is asserted where the door is,
    ``test_vibrationview_mount_js.py``.  Junk values are the door's business too:
    this function is handed real numbers, and giving it a second opinion about
    defaults is what put four of them in two files.
    """
    out = _run("""
        console.log(JSON.stringify({
            MIN: M.FRAMES_PER_CYCLE_MIN,
            MAX: M.FRAMES_PER_CYCLE_MAX,
            tooFew:   M.rate(2, 1.0).framesPerCycle,        // 2 -> floor
            tooMany:  M.rate(1000, 100).framesPerCycle,     // -> ceiling
            // a low rate is fine when the cycle is long: 5 fps x 4 s = 20 frames
            slowButSmooth: M.rate(5, 4.0).framesPerCycle,
            // a cycle of zero or less is a duration nobody can draw; the frame
            // floor contains it without a second bound of its own
            zeroCycle: M.rate(30, 0).framesPerCycle,
            negative:  M.rate(30, -2).framesPerCycle,
        }));
    """)
    assert out["tooFew"] == out["MIN"]
    assert out["tooMany"] == out["MAX"]
    assert out["slowButSmooth"] == 20
    assert out["zeroCycle"] == out["MIN"]
    assert out["negative"] == out["MIN"]


def test_the_cycle_length_reported_is_the_one_that_happened():
    """§ 10.1: "the returned cycleSec is what the rounding produced, not what was
    asked for".

    It is the number an export stamps into its metadata, so it has to describe the
    frames that exist rather than the request that produced them — a duration that
    survived into a caption while the frames said otherwise would be a caption
    that lies.  It is also why the seconds need no band of their own: the frame
    count is bounded, so a wild request is already contained.
    """
    out = _run("""
        const eff = (f, c) => { const r = M.rate(f, c);
            return { fps: r.fps, n: r.framesPerCycle, sec: r.cycleSec }; };
        console.log(JSON.stringify({
            plain:     eff(30, 1.0),      // asks for a second, gets one
            rounded:   eff(25, 0.3),      // 7.5 frames -> the floor, so it stretches
            enormous:  eff(30, 1000),     // 30000 frames -> the ceiling
            slow:      eff(5, 1.0),       // 5 frames -> the floor
        }));
    """)
    assert out["plain"] == {"fps": 30, "n": 30, "sec": 1.0}
    # the frame count is what is honoured; the duration is what moves
    assert out["rounded"]["n"] == 15 and out["rounded"]["sec"] == 15 / 25
    assert out["enormous"]["n"] == 1200 and out["enormous"]["sec"] == 1200 / 30
    assert out["slow"]["n"] == 15 and out["slow"]["sec"] == 3.0
    # every reported duration is exactly frames / rate -- no third opinion
    for row in out.values():
        assert abs(row["sec"] - row["n"] / row["fps"]) < 1e-12


def test_the_phase_comes_from_the_frame_number():
    """§ 10.1: "the phase comes from the frame number, not from the clock" — so
    frame n and frame n + N are the same phase EXACTLY, with no drift from
    accumulated addition, and the sequence an export encodes is the one the screen
    shows."""
    out = _run("""
        const N = M.rate(60, 1.0).framesPerCycle;      // 60 frames: quarters land on frames
        console.log(JSON.stringify({
            N,
            zero:    M.phaseOfFrame(0, N),
            quarter: M.phaseOfFrame(15, N),
            half:    M.phaseOfFrame(30, N),
            // exact equality after a thousand cycles
            drifts:  M.phaseOfFrame(7, N) !== M.phaseOfFrame(7 + 1000 * N, N),
            // a paused loop resuming from a kept frame number is the same phase
            resumes: M.phaseOfFrame(13, N) === M.phaseOfFrame(13, N),
            // a frame is a whole position in the cycle; half of one is the one before
            fractional: M.phaseOfFrame(7.5, N) === M.phaseOfFrame(7, N),
        }));
    """)
    assert out["N"] == 60
    assert out["zero"] == 0.0
    assert abs(out["quarter"] - 1.5707963267948966) < 1e-15
    assert abs(out["half"] - 3.141592653589793) < 1e-15
    assert out["drifts"] is False
    assert out["resumes"] is True
    assert out["fractional"] is True
