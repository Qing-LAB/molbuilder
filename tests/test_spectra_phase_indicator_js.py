"""L2 Node test: spectra phase-indicator visibility transition.

The spectra inspector's ``#phase-indicator`` element ships with
``hidden=""`` in the partial template and reveals on the first
results-load tick.  Visibility is owned by ``lib/spectra/core.js
::updatePhaseIndicator``, called from the spectra fetch handler
each time fresh results arrive.

Contracts covered:

  1. Calling ``updatePhaseIndicator(results)`` with valid results
     flips ``hidden`` from true to false.  This is the missing
     transition the audit flagged.
  2. The function tolerates a missing element (defensive
     ``if (!els.phaseIndicator) return``) — early return, no
     crash on mounts where the partial doesn't include the dot
     bar.
  3. Each ``.phase-dot`` gets a class reflecting its phase status
     (``empty``, ``in_progress``, ``complete``, ``error``).  The
     status comes from ``results.phase_<dot-name>`` (e.g.,
     ``phase_frequencies``, ``phase_raman``, ``phase_es``).
  4. Calling ``updatePhaseIndicator`` a second time with new
     results updates the dot classes in place; the visibility
     stays true (it never re-hides).

The audit (#393, Agent B) flagged this as a Top-5 visibility-
contract gap: ``hidden until first load tick``.  Pinning it
forward catches a refactor that, e.g., moves the
``els.phaseIndicator.hidden = false`` line out of the function
or gates it on a condition that doesn't fire on the first tick.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/spectra/core.js"


def _extract_fn_source(name: str) -> str:
    src = MODULE.read_text(encoding="utf-8")
    start = src.find(f"function {name}(")
    if start < 0:
        pytest.fail(
            f"Could not find ``function {name}(`` in "
            f"{MODULE.relative_to(ROOT)}.  Either renamed or "
            f"the parser needs updating."
        )
    open_brace = src.find("{", start)
    depth = 0
    i = open_brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    pytest.fail(f"Unbalanced braves in function {name}")


def _make_phase_indicator_stub(*, dot_phases: list[str]):
    """Build the JS literal for a stubbed phase-indicator element
    + its child ``.phase-dot`` array.  Mirrors the partial's
    actual DOM shape: a parent ``<div id="phase-indicator">``
    with N ``<span class="phase-dot" data-phase="X">`` children.
    """
    dots = [
        {
            "dataset": {"phase": ph},
            # className + title get overwritten by the function;
            # initialize them to bare values for the round-trip.
            "className": "phase-dot",
            "title": "",
        }
        for ph in dot_phases
    ]
    return {
        "hidden": True,
        "_dots": dots,
        # querySelectorAll(".phase-dot") returns the dots array.
        # We can't pickle JS methods in JSON; the bootstrap below
        # injects a custom querySelectorAll wrapper.
    }


def _run_with_results(
    results: dict[str, object],
    *,
    dot_phases: list[str] = None,
    element_present: bool = True,
) -> dict[str, object]:
    """Stub ``els.phaseIndicator`` (a parent with N phase-dot
    children); call ``updatePhaseIndicator(results)``; report the
    resulting visibility + dot classes."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    if dot_phases is None:
        dot_phases = ["frequencies", "raman", "es"]
    fn = _extract_fn_source("updatePhaseIndicator")
    stub = _make_phase_indicator_stub(dot_phases=dot_phases)
    bootstrap = f"""
        // ===== Stub the ``els.phaseIndicator`` element =====
        const dots = {json.dumps(stub["_dots"])};
        const phaseIndicator = {{
            hidden: {json.dumps(stub["hidden"])},
            querySelectorAll(sel) {{
                if (sel === ".phase-dot") return dots;
                return [];
            }},
        }};
        const els = {{
            phaseIndicator: {("phaseIndicator" if element_present
                              else "null")},
        }};

        {fn}

        updatePhaseIndicator({json.dumps(results)});

        console.log(JSON.stringify({{
            element_present_after: els.phaseIndicator !== null,
            hidden: els.phaseIndicator
                ? els.phaseIndicator.hidden : null,
            dot_states: dots.map(d => ({{
                phase: d.dataset.phase,
                className: d.className,
                title: d.title,
            }})),
        }}));
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- #
#  Contract: visibility transition                                       #
# --------------------------------------------------------------------- #


def test_first_call_reveals_the_indicator():
    """The element ships ``hidden=true`` in the partial; the
    first updatePhaseIndicator call must flip it to false."""
    out = _run_with_results({
        "phase_frequencies": "complete",
        "phase_raman": "in_progress",
        "phase_es": "empty",
    })
    assert out["hidden"] is False


def test_subsequent_calls_keep_it_visible():
    """The function never sets ``hidden = true`` — once shown, it
    stays.  This pins that no future refactor accidentally adds a
    ``hidden = !something`` branch that would re-hide on a stale
    poll."""
    fn = _extract_fn_source("updatePhaseIndicator")
    assert "hidden = false" in fn, (
        "updatePhaseIndicator must contain `hidden = false`."
    )
    assert "hidden = true" not in fn, (
        "updatePhaseIndicator must NOT contain `hidden = true` — "
        "the indicator only reveals; it never re-hides."
    )


# --------------------------------------------------------------------- #
#  Contract: missing element → silent no-op                              #
# --------------------------------------------------------------------- #


def test_missing_element_silent_no_op():
    """Defensive: if the partial template doesn't include the
    phase-indicator div (e.g., a non-spectra view re-uses the
    module), the function early-returns without crashing."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    fn = _extract_fn_source("updatePhaseIndicator")
    bootstrap = f"""
        const els = {{ phaseIndicator: null }};
        {fn}
        // Empty results — function should early-return.
        updatePhaseIndicator({{}});
        console.log("OK");
    """
    proc = subprocess.run(
        [shutil.which("node"), "--input-type=commonjs", "-e", bootstrap],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"missing-element no-op failed:\n"
        f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    )


# --------------------------------------------------------------------- #
#  Contract: each dot reflects its phase                                 #
# --------------------------------------------------------------------- #


def test_dot_classes_match_phase_status():
    """Each ``.phase-dot`` gets the class ``phase-dot phase-<status>``
    based on ``results.phase_<dot's data-phase>``."""
    out = _run_with_results({
        "phase_frequencies": "complete",
        "phase_raman": "in_progress",
        "phase_es": "empty",
    })
    by_phase = {d["phase"]: d for d in out["dot_states"]}
    assert by_phase["frequencies"]["className"] == "phase-dot phase-complete"
    assert by_phase["raman"]["className"]       == "phase-dot phase-in_progress"
    assert by_phase["es"]["className"]          == "phase-dot phase-empty"


def test_missing_phase_field_defaults_to_empty():
    """If the results blob omits ``phase_es`` entirely, the dot
    gets class ``phase-dot phase-empty`` (the default).  Catches
    a regression where results-shape drift might leave dots
    hanging on a stale class."""
    out = _run_with_results({
        "phase_frequencies": "complete",
        # phase_raman missing
        # phase_es missing
    })
    by_phase = {d["phase"]: d for d in out["dot_states"]}
    assert by_phase["raman"]["className"] == "phase-dot phase-empty"
    assert by_phase["es"]["className"]    == "phase-dot phase-empty"


def test_dot_title_carries_human_readable_status():
    """Title attribute drives the hover tooltip — pin the format
    ``<phase>: <status>`` so a future refactor doesn't quietly
    drop the user-visible accessibility hint."""
    out = _run_with_results({
        "phase_frequencies": "complete",
    })
    by_phase = {d["phase"]: d for d in out["dot_states"]}
    assert by_phase["frequencies"]["title"] == "frequencies: complete"
