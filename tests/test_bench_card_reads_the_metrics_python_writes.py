"""The Bench card and the instrument parser must agree on five key names.

`util_csv_metrics` writes a metrics dict; it crosses to the browser as JSON
and `bench-summary.js::_usageLine` reads it back.  Nothing held the two ends
together: renaming `wall_s` to `monitored_elapsed_s` on 2026-09-05 required
editing one line of JavaScript, and had I missed it the card would simply
have stopped showing the duration — no error, no failing test, just a
shorter line that still looks right.

That is the write-then-read shape this project keeps getting caught by, so
it gets the same treatment as the rest: the REAL function runs, on the REAL
keys the REAL parser produces, and the assertion is on what a person sees.

Contract: `docs/web/bench-summary.md`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "molbuilder/web/static/lib/inspectors/bench-summary.js"


def _fn(src: str, name: str) -> str:
    """One `function <name>(` through to the next top-level `function `."""
    start = src.index(f"    function {name}(")
    nxt = src.find("\n    function ", start + 1)
    return src[start:nxt if nxt != -1 else len(src)].rstrip()


def _usage_line(metrics: dict) -> str:
    node = shutil.which("node")
    if node is None:                                    # pragma: no cover
        pytest.skip("node not available")
    src = CARD.read_text(encoding="utf-8")
    bootstrap = (
        f"{_fn(src, '_num')}\n"
        f"{_fn(src, '_usageLine')}\n"
        f"console.log(JSON.stringify("
        f"_usageLine({{metrics: {json.dumps(metrics)}}})));"
    )
    proc = subprocess.run([node, "--input-type=commonjs", "-e", bootstrap],
                          capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:                            # pragma: no cover
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip())


def _real_metrics() -> dict:
    """What the parser emits from what the MONITOR writes.

    Both halves come from the shipped code: `monitor._util_csv_header` and
    `monitor._util_csv_row` compose the file, `util_csv_metrics` reads it.
    Typing the columns by hand here produced `gpu_sm_pct` instead of
    `gpu0_sm`, so the GPU keys were silently absent and this test covered
    three of five names while appearing to cover all of them.

    Epochs ASCENDING, as the monitor stamps them: the means are
    time-weighted, so a row out of order contributes a negative interval.
    """
    from molbuilder.monitor import UtilSample, _util_csv_header, _util_csv_row
    from molbuilder.parse.instruments.util_csv import util_csv_metrics

    samples = [
        UtilSample(epoch=1000.0, cpu_pct=50.0, mem_gb=3.4, gpus=[(0, 60.0, 40.0, 7.5)]),
        UtilSample(epoch=1200.0, cpu_pct=90.0, mem_gb=4.2, gpus=[(0, 80.0, 55.0, 8.0)]),
        UtilSample(epoch=1400.0, cpu_pct=90.0, mem_gb=4.2, gpus=[(0, 80.0, 55.0, 8.0)]),
    ]
    csv = "\n".join([_util_csv_header(1)]
                    + [_util_csv_row(s, 1) for s in samples]) + "\n"
    return util_csv_metrics(csv)


def test_the_card_renders_every_metric_the_parser_produces():
    """Each key the parser emits reaches the card, or the card drops it silently."""
    metrics = _real_metrics()
    assert metrics, "the parser produced nothing — the fixture is wrong, not the card"

    line = _usage_line(metrics)

    # The card's own vocabulary for each key it reads.
    expected_labels = {
        "peak_rss_gb":         "peak",
        "cpu_mean_pct":        "cpu",
        "gpu_sm_mean_pct":     "gpu",
        "gpu_vram_peak_gb":    "vram",
        "monitored_elapsed_s": "s",
    }
    missing = [k for k in metrics if k in expected_labels
               and expected_labels[k] not in line]
    assert not missing, (
        f"the parser emitted {sorted(metrics)} and the card rendered {line!r}; "
        f"it reads none of {missing} — the two ends disagree on a key name")


def test_the_duration_reaches_the_card_under_its_current_name():
    """The rename's other end.

    `monitored_elapsed_s` was `wall_s` until 2026-09-05 (P-T1: a time field's
    suffix is its contract, and a duration may not wear a date's name).  The
    parser and the card are edited in different languages in different files;
    this is the only thing that fails if one moves without the other.
    """
    line = _usage_line({"monitored_elapsed_s": 41.0})
    assert line == "41 s", f"the duration did not render: {line!r}"

    # The OLD name must now render nothing, or the card is reading both and
    # the rename left a second door open.
    assert _usage_line({"wall_s": 41.0}) == "", (
        "the card still reads `wall_s` — the old spelling survives")
