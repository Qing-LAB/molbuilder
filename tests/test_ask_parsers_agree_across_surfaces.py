"""The browser and the terminal must read a time and a memory THE SAME WAY.

`jobset/ask.py` parses what a person types at `--time` / `--mem`;
`task-setup/viewer.js` parses the same strings as they are typed into the
tab's two fields, because it validates them against the chosen queue's
ceiling while the person is still typing.  Two implementations of one rule,
and the browser cannot import the Python one.

**They had already drifted** (found 2026-08-24 by running both over the same
inputs):

* `"7-00:00:00"` -- SLURM's own spelling, and what a queue's `max_time` IS
  in the machine record, so it is what the tab FILLS a time field with.  The
  browser accepted it; Python refused.  A person reading that value out of
  their own `task.json` could not type it back at `--time`.
* `"80GB"` -- which `prep --mem`'s own help text advertises.  The browser
  accepted it; `launch --mem` refused.  Two flags of one name disagreeing
  about a spelling one of them documents.

So the duplication stays (it must), and this pins the agreement.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from molbuilder.scheduler.quantities import (canonical_mem, canonical_time,
                                             parse_duration, parse_memory)

_VIEWER = (Path(__file__).resolve().parents[1] / "molbuilder" / "web"
           / "static" / "task-setup" / "viewer.js")

#: Every spelling either surface can produce or documents, plus what must
#: stay refused.  `None` = "says nothing"; "ERR" = refused.
_TIMES = ["4h", "90m", "45", "1.5h", "4H", "0.5",
          "7-00:00:00", "4:00:00", "00:15:00", "1-00:00:00",
          "", "   ", "0", "-3", "banana"]
_MEMS = ["128G", "0.5T", "128", "512M", "1024K", "4t", "80GB", "128GB",
         "", "   ", "0", "-5", "banana"]


def _py(fn, raw):
    try:
        v = fn(raw)
    except ValueError:
        return "ERR"
    return None if v is None else float(v)


def _same(a, b) -> bool:
    """Agreement is about the VALUE, not its JSON type: JS hands back `128`
    where Python hands back `128.0`, and a float round-trip differs in the
    ninth decimal.  Comparing the printed forms called those a disagreement
    -- a defect in the first version of this test, not in either parser."""
    if a is None or b is None or a == "ERR" or b == "ERR":
        return a == b
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


def _js(raws_t, raws_m):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = _VIEWER.read_text(encoding="utf-8")

    def _grab(name):
        i = src.index(f"function {name}(")
        # to the closing brace of the function, which sits at column 0
        j = src.index("\n}", i) + 2
        return src[i:j]

    prog = (_grab("_parseTime") + "\n" + _grab("_parseMem") + "\n"
            + "const T=" + json.dumps(raws_t) + ",M=" + json.dumps(raws_m)
            + ";\nconst norm=v=>Number.isNaN(v)?'ERR':"
              "(v===null?null:Math.round(v*1e9)/1e9);\n"
              "console.log(JSON.stringify({t:T.map(x=>norm(_parseTime(x))),"
              "m:M.map(x=>norm(_parseMem(x)))}));")
    out = subprocess.run([node, "--input-type=commonjs", "-e", prog],
                         capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        pytest.fail(f"node failed: {out.stderr}\n{out.stdout}")
    return json.loads(out.stdout)


def _js_canon(raws_t, raws_m):
    """The same extraction, for the two CANONICALISERS.

    They are the other half of the duplication: `_canonTime`/`_canonMem`
    decide what the browser WRITES into `task.json`, and
    `canonical_time`/`canonical_mem` decide what the CLI writes.  One
    record, one spelling -- so if these two drift, the file gets whichever
    surface last touched it, which is the state this whole rule replaced.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = _VIEWER.read_text(encoding="utf-8")

    def _grab(name):
        i = src.index(f"function {name}(")
        j = src.index("\n}", i) + 2
        return src[i:j]

    prog = ("\n".join(_grab(n) for n in ("_parseTime", "_parseMem",
                                         "_slurmTime", "_slurmMem",
                                         "_canonTime", "_canonMem"))
            + "\nconst T=" + json.dumps(raws_t) + ",M=" + json.dumps(raws_m)
            + ";\nconsole.log(JSON.stringify({"
              "t:T.map(x=>_canonTime(x)),m:M.map(x=>_canonMem(x))}));")
    out = subprocess.run([node, "--input-type=commonjs", "-e", prog],
                         capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        pytest.fail(f"node failed: {out.stderr}\n{out.stdout}")
    return json.loads(out.stdout)


#: Only the spellings that MEAN something -- canonicalising is what happens
#: to a value that parsed, and a refusal is the reader's job, tested above.
_GOOD_TIMES = ["4h", "90m", "45", "1.5h", "4H", "7-00:00:00", "4:00:00",
               "00:15:00", "1-00:00:00", "168h", ""]
#: Includes a SUB-MEGABYTE value on purpose.  Both writers round to whole
#: megabytes, and rounding a positive ask down to 0 hands SLURM its
#: "all the node's memory" spelling -- the smallest request becoming the
#: largest.  Python grew the guard and the browser did not, and this list
#: was too narrow to notice (2026-08-24).
_GOOD_MEMS = ["128G", "0.5T", "128", "512M", "1024K", "4t", "80GB",
              "128GB", "503.5G", "0.0001", "0.001", "0", ""]


@pytest.fixture(scope="module")
def both():
    return _js(_TIMES, _MEMS)


@pytest.fixture(scope="module")
def both_canon():
    return _js_canon(_GOOD_TIMES, _GOOD_MEMS)


def _py_canon(fn, raw):
    try:
        v = fn(raw)
    except ValueError:
        return raw.strip()          # the JS passes an unparseable through
    return v or ""


def test_the_two_time_writers_agree_on_every_spelling(both_canon):
    """What the browser writes into `task.json` and what the CLI writes
    must be the SAME STRING -- not merely the same duration."""
    bad = [(raw, js, _py_canon(canonical_time, raw))
           for raw, js in zip(_GOOD_TIMES, both_canon["t"])
           if js != _py_canon(canonical_time, raw)]
    assert not bad, (
        "the browser and the terminal WRITE a duration differently: "
        + ", ".join(f"{r!r}: browser={j!r} terminal={p!r}"
                    for r, j, p in bad))


def test_the_two_memory_writers_agree_on_every_spelling(both_canon):
    bad = [(raw, js, _py_canon(canonical_mem, raw))
           for raw, js in zip(_GOOD_MEMS, both_canon["m"])
           if js != _py_canon(canonical_mem, raw)]
    assert not bad, (
        "the browser and the terminal WRITE a memory differently: "
        + ", ".join(f"{r!r}: browser={j!r} terminal={p!r}"
                    for r, j, p in bad))


def test_canonical_is_what_sbatch_takes():
    """The point of the whole rule: `-t 4h` is not a thing SLURM accepts,
    and it is what the browser used to write.  2026-08-24."""
    import re
    assert canonical_time("4h") == "0-04:00:00"
    assert re.fullmatch(r"(?:\d+-)?\d+(?::\d{2}){0,2}",
                        canonical_time("4h"))
    assert canonical_mem("80GB") == "80G"
    assert re.fullmatch(r"\d+(?:\.\d+)?[KMGT]?", canonical_mem("80GB"))


def test_canonicalising_is_idempotent():
    """A record read and rewritten must not change -- otherwise every save
    rewrites the file and no two copies of it ever match."""
    for v in _GOOD_TIMES:
        once = canonical_time(v)
        assert canonical_time(once) == once
    for v in _GOOD_MEMS:
        once = canonical_mem(v)
        assert canonical_mem(once) == once


def test_zero_memory_survives_because_slurm_means_something_by_it():
    """`--mem`'s own help says *"'0' asks for all of the node's"*, and
    `parse_memory` refuses 0 because zero gigabytes is not an amount to fit
    a queue against.  Those are two different questions and the writer must
    not inherit the reader's refusal."""
    assert canonical_mem("0") == "0"
    with pytest.raises(ValueError):
        parse_memory("0")


def test_the_two_time_parsers_agree_on_every_spelling(both):
    mismatched = [(raw, js, _py(parse_duration, raw))
                  for raw, js in zip(_TIMES, both["t"])
                  if not _same(js, _py(parse_duration, raw))]
    assert not mismatched, (
        "the browser and the terminal read a duration differently: "
        + ", ".join(f"{r!r}: browser={j} terminal={p}"
                    for r, j, p in mismatched))


def test_the_two_memory_parsers_agree_on_every_spelling(both):
    mismatched = [(raw, js, _py(parse_memory, raw))
                  for raw, js in zip(_MEMS, both["m"])
                  if not _same(js, _py(parse_memory, raw))]
    assert not mismatched, (
        "the browser and the terminal read an amount of memory "
        "differently: "
        + ", ".join(f"{r!r}: browser={j} terminal={p}"
                    for r, j, p in mismatched))


def test_slurms_own_spelling_is_read_by_BOTH():
    """The concrete case: it is what a queue's ceiling looks like in the
    machine record, so the tab fills a field with it -- and a value the tool
    writes must be one the tool can read back."""
    assert parse_duration("7-00:00:00") == 7 * 86400


def test_the_spelling_prep_advertises_is_accepted_by_launch():
    """`prep --mem`'s help says *"e.g. 80GB"*.  It passed the string through
    unparsed while `launch --mem` refused it."""
    assert parse_memory("80GB") == 80.0


def test_nothing_that_should_be_refused_became_acceptable():
    for bad in ("banana", "0", "-3"):
        with pytest.raises(ValueError):
            parse_duration(bad)
        with pytest.raises(ValueError):
            parse_memory(bad)


# --------------------------------------------------------------------- #
#  One object, one module                                               #
# --------------------------------------------------------------------- #

def test_the_quantity_vocabulary_is_defined_in_exactly_one_module():
    """A duration and an amount of memory are written in one place.

    They were not.  On 2026-08-24 the same two objects had FIVE readers and
    writers across three modules: `slurm_time` here and a byte-identical
    `_slurm_walltime` in `runwrap.py` (whose docstring named the fold as a
    candidate "when the scheduler subsystem exists" -- it existed);
    `parse_memory` here, `parse_mem_gb` in `scheduler/admit.py`, and a dead
    `_mem_to_mb` in `runwrap.py` left behind by the estimation purge.  Two
    of those read the SAME dialect into different units, and `parse_memory`
    and `parse_mem_gb` still disagree by 1024x on a bare number -- correct
    per dialect, catastrophic if you hold the wrong one.

    Scattering is what let `jobset/submit.py` call the record reader on a
    human-written value and hand `sbatch` a `-t 4h` it refused.
    """
    import re
    root = Path(__file__).resolve().parents[1] / "molbuilder"
    names = ["parse_walltime", "parse_duration", "parse_memory",
             "slurm_time", "slurm_mem", "canonical_time", "canonical_mem",
             "human_wall", "parse_mem_gb"]
    where = {n: [] for n in names}
    for f in sorted(root.rglob("*.py")):
        if "static" in f.parts:
            continue
        src = f.read_text(encoding="utf-8")
        for n in names:
            if re.search(rf"^def {re.escape(n)}\(", src, re.M):
                where[n].append(f.relative_to(root).as_posix())
    bad = {n: v for n, v in where.items() if v != ["scheduler/quantities.py"]}
    assert not bad, (
        "the quantity vocabulary must live in scheduler/quantities.py and "
        f"nowhere else; found: {bad}")


def test_no_second_writer_of_the_slurm_walltime_spelling():
    """A content check, not a name check: the duplicate that existed was
    called something else.  `D-HH:MM:SS` is assembled in one place."""
    import re
    root = Path(__file__).resolve().parents[1] / "molbuilder"
    pattern = re.compile(r'\{d\}-\{h:02d\}:\{m:02d\}')
    hits = [f.relative_to(root).as_posix()
            for f in sorted(root.rglob("*.py"))
            if "static" not in f.parts
            and pattern.search(f.read_text(encoding="utf-8"))]
    assert hits == ["scheduler/quantities.py"], (
        f"more than one module assembles a SLURM walltime: {hits}")
