"""MolView § 6.3 / § 6.4 — the master copy, and the one number that says which frame.

Derived from ``docs/web/molview.md``, not from the source.  § 6.4:

    They are one fact, kept in one place.  A frame number without the range it is valid in
    cannot be used for anything ... Nothing anywhere keeps its own copy of either.  Not the
    renderEngine, not the sealed layer, not a tab, not the frame bar.

and the order it insists on:

    1. The master copy is updated first, and completely.  2. The range is recomputed from it —
    not from the drawing, and not from what the caller said it was adding.  3. The frame number
    is checked against that range and moved if it no longer fits.  4. Only then is anyone told,
    and what they see is a matching pair.

§ 13.3 rows guarded here: nothing keeps its own copy · master copy, then range, then frame,
then notify · an out-of-range write is resolved, not accepted · same atoms every frame · only
the master copy's count is offered · the renderEngine answers nothing.

The stand-in renderEngine below obeys the document rather than the code (§ 13.1): it accepts
commands and answers NOTHING, because § 9.7 says every entry on that surface is an
instruction.  A stand-in that offered ``frameCount()`` would describe a design this contract
forbids, and a suite built on it would confirm behaviour that cannot happen.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "molbuilder/web/static/lib/molview/data-model.js"

# Three atoms, so a frame is three coordinates and a wrong count is obvious.
_BOOT = f"""
    const mod = await import("file://{MODEL}");
    const data = mod.data;

    // A renderEngine that obeys § 9.7: COMMANDS ONLY.  It records what it was told and answers
    // nothing.  If the model tries to ask it a question, the call throws and the test fails.
    const told = [];
    const engine = new Proxy({{
        setDataSource(src) {{ engine._src = src; }},
        setFrameNotifier() {{}},
        dataChanged()   {{ told.push("dataChanged"); }},
        cellChanged()   {{ told.push("cellChanged"); }},
        appendFrames(f) {{ told.push("appendFrames:" + f.length); }},
        forcesChanged() {{ told.push("forcesChanged"); }},
        showFrame(i)    {{ told.push("showFrame:" + i); }},
        render()        {{}},
        dispose()       {{}},
    }}, {{
        get(t, k) {{
            if (k in t || typeof k === "symbol" || k === "_src") return t[k];
            throw new Error("the model asked the renderEngine a question: " + String(k));
        }},
    }});

    const ATOMS = [{{ element: "C", x: 0, y: 0, z: 0 }},
                   {{ element: "N", x: 1, y: 0, z: 0 }},
                   {{ element: "O", x: 2, y: 0, z: 0 }}];
    await data.selection.adoptSession({{ sourceFile: null, selection: [], atoms: ATOMS }});
    data.attachRenderEngine(engine);
    // Attaching backfills the already-loaded structure (a session restore attaches AFTER the
    // load), which is one legitimate `dataChanged`.  Start each test from a clean record.
    told.length = 0;
    const frame = (x) => [[x,0,0],[x+1,0,0],[x+2,0,0]];
"""


def _run(snippet: str) -> object:
    return run_node([], _BOOT + snippet)


def test_the_model_answers_which_frame_and_how_many_without_asking_anyone():
    """§ 6.4 / § 9.7 — the reads are the model's; the renderEngine is never consulted.

    The stand-in throws on any read, so this passes only if the model answers from the master
    copy it holds.  Before the truth moved, these three reads were forwarded downward into the
    layer that is supposed to hold no truth at all.
    """
    out = _run("""
        data.reloadFrames([frame(0), frame(10), frame(20)]);
        console.log(JSON.stringify({
            count:   data.frameCount(),
            current: data.currentFrame(),
            atFrame1: data.getFrameAllAtoms(1),
            told:    told,
        }));
    """)
    assert out["count"] == 3
    assert out["current"] == 0, "a full load resets to frame 0 (§ 10.8)"
    assert out["atFrame1"] == [[10, 0, 0], [11, 0, 0], [12, 0, 0]]
    assert out["told"] == ["dataChanged"], "the renderEngine is told, never asked"


def test_a_read_of_the_coordinates_cannot_be_used_to_write():
    """§ 9.3 — every read returns a copy, so changing it cannot change the viewer."""
    out = _run("""
        data.reloadFrames([frame(0), frame(10)]);
        const got = data.getFrameAllAtoms(0);
        got[0][0] = 999;                       // scribble on what we were handed
        console.log(JSON.stringify({ again: data.getFrameAllAtoms(0)[0] }));
    """)
    assert out["again"] == [0, 0, 0], "a read handed out the viewer's own array"


def test_an_out_of_range_write_is_resolved_against_the_range():
    """§ 6.4 — a number outside the range is resolved, never taken on trust.

    A tab following the end of a growing run should land on the last frame, not raise.
    """
    out = _run("""
        data.reloadFrames([frame(0), frame(10), frame(20)]);
        const past   = data.setCurrentFrame(99);
        const before = data.setCurrentFrame(-5);
        console.log(JSON.stringify({ past, before, ended: data.currentFrame() }));
    """)
    assert out["past"] == 2, "a seek past the end lands on the last frame"
    assert out["before"] == 0
    assert out["ended"] == 0


def test_everyone_is_told_once_whatever_moved_the_frame():
    """§ 6.4 — one write reaches EVERY subscriber, and a subscriber never asks who moved it."""
    out = _run("""
        const heard = [];
        data.onFrameChange(() => heard.push(["a", data.currentFrame(), data.frameCount()]));
        data.onFrameChange(() => heard.push(["b", data.currentFrame(), data.frameCount()]));
        data.reloadFrames([frame(0), frame(10), frame(20)]);
        data.setCurrentFrame(2);
        data.setCurrentFrame(2);        // already there: nothing moved, so nothing is announced
        console.log(JSON.stringify({ heard }));
    """)
    # The load tells both; the seek tells both; the no-op seek tells nobody.
    assert out["heard"] == [["a", 0, 3], ["b", 0, 3], ["a", 2, 3], ["b", 2, 3]]


def test_no_subscriber_ever_sees_a_new_range_beside_an_old_frame_number():
    """§ 6.4 — the ordering rule, which is the whole reason the two live together.

    Load a long trajectory, scrub to the end, then load a SHORT one over it.  A subscriber that
    woke up between "the range is new" and "the frame number is new" would read frame 9 of a
    2-frame structure — a pair that cannot be drawn, saved or exported.
    """
    out = _run("""
        data.reloadFrames(Array.from({ length: 10 }, (_, i) => frame(i)));
        data.setCurrentFrame(9);
        const seen = [];
        data.onFrameChange(() => seen.push([data.currentFrame(), data.frameCount()]));
        data.reloadFrames([frame(0), frame(1)]);      // the structure got shorter underneath
        console.log(JSON.stringify({
            seen,
            impossible: seen.filter(([i, n]) => i >= n),
        }));
    """)
    assert out["impossible"] == [], (
        "a subscriber saw a frame number the range cannot contain — the master copy, the range "
        "and the index were not settled before anyone was told")
    assert out["seen"] == [[0, 2]], "a full load resets to frame 0 and announces once"


def test_the_count_offered_is_the_master_copys_own_length():
    """§ 10.10 — the drawing's count is not reachable, and never answers this."""
    out = _run("""
        data.reloadFrames([frame(0), frame(10)]);
        const two = data.frameCount();
        data.addFrames([frame(20), frame(30)]);
        console.log(JSON.stringify({ two, four: data.frameCount(), told }));
    """)
    assert out["two"] == 2
    assert out["four"] == 4, "the count comes from the master copy, which grew"
    assert out["told"] == ["dataChanged", "appendFrames:2"]


def test_the_displayed_frame_does_not_move_while_a_run_grows_past_it():
    """§ 10.8 rule 5 — a user watching frame 1 keeps watching frame 1."""
    out = _run("""
        data.reloadFrames([frame(0), frame(10), frame(20)]);
        data.setCurrentFrame(1);
        const heard = [];
        data.onFrameChange(() => heard.push([data.currentFrame(), data.frameCount()]));
        data.addFrames([frame(30), frame(40)]);
        console.log(JSON.stringify({ at: data.currentFrame(), n: data.frameCount(), heard }));
    """)
    assert out["at"] == 1, "an append moved the user's frame"
    assert out["n"] == 5
    # The RANGE changed, so the bar's "i / N" has to hear about it even though i did not move.
    assert out["heard"] == [[1, 5]]


def test_a_frame_with_the_wrong_atom_count_is_a_hard_error():
    """§ 10.8 — never padded, never truncated, never guessed into fitting."""
    out = _run("""
        data.reloadFrames([frame(0)]);
        const tryIt = (fn) => { try { fn(); return null; } catch (e) { return String(e.message); } };
        console.log(JSON.stringify({
            short:  tryIt(() => data.addFrames([[[0,0,0],[1,0,0]]])),
            long:   tryIt(() => data.addFrames([[[0,0,0],[1,0,0],[2,0,0],[3,0,0]]])),
            reload: tryIt(() => data.reloadFrames([frame(0), [[0,0,0]]])),
            unchanged: data.frameCount(),
        }));
    """)
    assert out["short"] and "does not match" in out["short"]
    assert out["long"] and "does not match" in out["long"]
    assert out["reload"], "a frame set of mixed atom counts is refused whole"
    assert out["unchanged"] == 1, "a refused append left the master copy exactly as it was"
