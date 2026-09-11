"""Does this node-driven test FAIL when the JS it loads is broken?

    SUBSUMPTION_TMP=<dir> python tools/vacuity_js.py tests/test_x_js.py [budget]

WHY THIS EXISTS.  `tools/vacuity.py` mutates Python, so for every test that
drives JS through node it reports JS-SUBJECT -- "outside what this instrument
can judge".  That was 43 files of ~190 swept, a sixth of the suite, and
"unjudgeable" is not an answer to *why should this test exist*: these are
exactly the tests the 2026-09-10 review doubted hardest, having been skipped
for months while the UI changed underneath them.

So mutate the JS.  A node-driven test names the module it loads -- all 33 of
them do, as `MODULE = ROOT / "molbuilder/web/static/..."` or a path literal --
so the subject is read out of the test itself rather than guessed.

Operators are textual, one occurrence at a time: comparison flips, boolean
operator swaps, numeric literal bumps.  **Every mutant is `node --check`ed
before it counts.**  A mutation that lands inside a comment or a string can
break the parse, and a test failing on a SyntaxError would score as a guard it
has not earned -- the instrument's own version of the fault
`verify_subsumption.py` documents.

Bias: textual noise pushes toward GUARDS, never toward VACUOUS.  That is the
safe direction, since a VACUOUS verdict is what deletes a test.
"""
from __future__ import annotations

import json, os, pathlib, re, shutil, subprocess, sys, tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
PYEXE = sys.executable

#: one edit each, applied at a chosen occurrence
_OPS = [
    (re.compile(r"==="), "!=="), (re.compile(r"!=="), "==="),
    (re.compile(r"&&"), "||"),  (re.compile(r"\|\|"), "&&"),
    # `>` only where it cannot be an arrow or a JSX-ish close
    (re.compile(r"(?<![=!<>\-])>=(?!=)"), "<"),
    (re.compile(r"(?<![=!<>])<=(?!=)"), ">"),
]
_NUM = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def _modules(testfile: str):
    """The JS files this test loads, read out of the test's own source."""
    src = pathlib.Path(testfile).read_text(errors="ignore")
    out, seen = [], set()
    static = REPO / "molbuilder/web/static"
    for m in re.finditer(r"""["']([^"']*?\.js)["']""", src):
        rel = m.group(1).lstrip("/")
        cands = [REPO / rel, static / rel]
        # A path assembled from components -- `/ "static" / "lib" / name` with
        # the basename alone in a parametrize list -- leaves only "foo.js" as a
        # literal, which resolves nowhere.  Fall back to basename search so
        # those files are measured instead of reported NO-JS-SUBJECT.
        if not any(c.is_file() for c in cands) and "/" not in rel:
            # ONLY WHEN IT IS UNAMBIGUOUS.  A bare "page.js" matches several
            # files under static/, and taking them all is worse than taking
            # none: test_molview_measurement.py resolved to 82 modules and a
            # 16-mutant budget spread over them measured nothing.
            hits = sorted(static.rglob(rel))
            if len(hits) == 1:
                cands += hits
        for cand in cands:
            if cand.is_file() and cand not in seen and "vendor/" not in str(cand):
                seen.add(cand); out.append(cand)
    # A budget cannot be spread over dozens of modules.  Keep the ones this
    # test talks about MOST -- a module named once in a layering list is not
    # the subject; the one named twenty times is.
    out.sort(key=lambda p: -src.count(p.name))
    return out[:6]


def _named_spans(testfile, src):
    """Byte ranges of the JS functions THIS TEST NAMES.

    Without JS coverage the budget otherwise strides across the whole module:
    measured on tests/test_trajectory_clocks_js.py, 8 mutants over
    lib/trajectory/core.js never touched `cumulativeElapsed`, the one helper
    the test evaluates, and the file read as blind.  Same fault as the Python
    side's first-N-lines sampling, same fix -- aim at the subject.

    A name counts when the test mentions it and the module DEFINES it.  Body
    found by brace matching from the definition.
    """
    tsrc = pathlib.Path(testfile).read_text(errors="ignore")
    names = set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]{2,}", tsrc))
    spans = []
    for name in names:
        for m in re.finditer(
                r"(?:function\s+%s\s*\(|%s\s*[:=]\s*(?:async\s+)?function\s*\(|"
                r"%s\s*[:=]\s*\([^)]*\)\s*=>)" % (re.escape(name), re.escape(name),
                                                    re.escape(name)), src):
            i = src.find("{", m.end() - 1)
            if i < 0:
                continue
            depth, j = 0, i
            while j < len(src):
                if src[j] == "{":
                    depth += 1
                elif src[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j > i:
                spans.append((i, j))
    return sorted(spans)


def _mutants(path: pathlib.Path, spans=None):
    """(description, mutated source) pairs, spread over the file."""
    src = path.read_text()

    def inside(pos):
        return (not spans) or any(a <= pos <= b for a, b in spans)

    for rx, repl in _OPS:
        hits = [h for h in rx.finditer(src) if inside(h.start())]
        # stride through the occurrences instead of taking the first few
        for i in range(0, len(hits), max(1, len(hits) // 4 or 1)):
            h = hits[i]
            yield (f"{rx.pattern}->{repl}@{h.start()}",
                   src[:h.start()] + repl + src[h.end():])
    nums = [h for h in _NUM.finditer(src) if inside(h.start())]
    for i in range(0, len(nums), max(1, len(nums) // 4 or 1)):
        h = nums[i]
        yield (f"{h.group(1)}->{int(h.group(1)) + 1}@{h.start()}",
               src[:h.start()] + str(int(h.group(1)) + 1) + src[h.end():])


def _parses(path: pathlib.Path) -> bool:
    node = shutil.which("node")
    if not node:
        return True
    return subprocess.run([node, "--check", str(path)],
                          capture_output=True).returncode == 0


def _green(testfile, root, timeout=420):
    try:
        r = subprocess.run([PYEXE, "-m", "pytest", "-q", "-p", "no:randomly",
                            "-x", testfile], cwd=root,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return r.returncode == 0, (r.stdout or "")


#: "module.js@offset" pairs to skip, comma-separated, from VACUITY_JS_SKIP.
#:
#: WHY.  A mutant at a module-wide switch kills every test that touches the
#: module, so GUARDS on it says only "this file notices the module being
#: broken" -- a bar nearly anything clears.  Measured: flipping
#: `const readOnly = opts.mode === "readonly"` at lib/molview/model.js@2489
#: makes EVERY model read-only (mode is normally undefined), the master copy
#: freezes, and six separate test files died on their first mutant at that one
#: offset.  Excluding it asks the question that matters: does this file notice
#: something its own subject got wrong?
def _skips():
    raw = os.environ.get("VACUITY_JS_SKIP", "")
    out = set()
    for item in raw.split(","):
        item = item.strip()
        if "@" in item:
            mod, _, off = item.rpartition("@")
            try: out.add((mod, int(off)))
            except ValueError: pass
    return out


def vacuity_js(testfile, budget=12):
    mods = _modules(testfile)
    if not mods:
        return {"verdict": "NO-JS-SUBJECT",
                "detail": "the test names no .js file that exists"}
    work = tempfile.mkdtemp(prefix="vacjs_")
    root = pathlib.Path(work) / "repo"
    subprocess.run(["git", "worktree", "add", "--detach", str(root), "HEAD"],
                   capture_output=True, cwd=REPO)
    try:
        base = _green(testfile, root)
        if base is None:
            return {"verdict": "NOT-MEASURED-TIMEOUT"}
        ok, out = base
        if not ok:
            return {"verdict": "NOT-GREEN-AT-HEAD", "detail": out[-200:]}
        m = re.search(r"(\d+) passed", out)
        if not m or int(m.group(1)) == 0:
            return {"verdict": "ALL-SKIPPED-HERE", "detail": out.strip()[-120:]}

        skip = _skips()
        tried = skipped = 0
        # Aim at what the test names; fall back to the whole module only if
        # nothing it mentions is defined there.
        streams = []
        for p in mods:
            spans = _named_spans(testfile, p.read_text())
            streams.append((p, _mutants(p, spans)))
        while streams and tried < budget:
            for p, it in list(streams):
                rel = p.relative_to(REPO)
                target = root / rel
                if not target.exists():
                    streams.remove((p, it)); continue
                try:
                    desc, mutated = next(it)
                except StopIteration:
                    streams.remove((p, it)); continue
                off = int(desc.rsplit("@", 1)[1]) if "@" in desc else -1
                if (str(rel), off) in skip or (p.name, off) in skip:
                    skipped += 1
                    continue
                original = target.read_text()
                target.write_text(mutated)
                try:
                    if not _parses(target):
                        continue        # a SyntaxError is not a mutant
                    g = _green(testfile, root)
                finally:
                    target.write_text(original)
                if g is None:
                    continue
                tried += 1
                if not g[0]:
                    return {"verdict": "GUARDS", "tried": tried,
                            "skipped": skipped, "first": f"{rel}: {desc}"}
                if tried >= budget:
                    break
        if not tried:
            return {"verdict": "NO-MUTANTS", "modules": [str(p.relative_to(REPO)) for p in mods]}
        return {"verdict": "VACUOUS" if not streams else "VACUOUS-AT-BUDGET",
                "tried": tried, "budget": budget,
                "modules": [str(p.relative_to(REPO)) for p in mods]}
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(root)],
                       capture_output=True, cwd=REPO)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    f = sys.argv[1]
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    print(json.dumps({"file": f, **vacuity_js(f, b)}), flush=True)
