"""Does this test file FAIL when the code it runs is broken?

    SUBSUMPTION_TMP=<dir> python tools/vacuity.py tests/test_x.py [budget]

WHY THIS EXISTS.  Every test-retirement audit before this one asked "can this
test be kept" -- and a test with a plausible docstring always can.  That
question defends the suite by default and settles nothing; the 2026-09-10
sweeps spent a day on it and produced arguments, not verdicts.

The question a test has to answer is "why should this exist", and exactly one
answer counts: it goes red when the code is wrong.  So ask the code.  Mutate
the molbuilder lines this file actually executes -- one edit at a time,
comparison flips, literal changes, boolean clauses dropped -- and run the file
against each mutant.

    VACUOUS   the file stayed GREEN through every mutant of the code it ran.
              It executes production code and notices nothing about it.
    NO-CODE   the file executes no molbuilder line at all.
    GUARDS n  n mutants turned it red.  That is the reason to exist, measured.

A VACUOUS verdict is bounded by the operator set (it cannot mutate a dict key
or a string's contents), so it is evidence, not proof -- but it inverts the
burden correctly: the file had its chance to object and did not.
"""
from __future__ import annotations

import json, os, pathlib, shutil, subprocess, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_subsumption import covered_lines, mutants_for, PY as PYEXE  # noqa: E402


def _named_modules(testfile):
    """The molbuilder modules this test file imports directly."""
    import ast as _ast
    out = set()
    try:
        tree = _ast.parse(pathlib.Path(testfile).read_text())
    except Exception:
        return out
    for n in _ast.walk(tree):
        if isinstance(n, _ast.ImportFrom) and (n.module or "").startswith("molbuilder"):
            out.add(n.module[len("molbuilder."):] if n.module != "molbuilder" else "")
            for a in n.names:
                out.add(((n.module + "." + a.name)[len("molbuilder."):]))
        elif isinstance(n, _ast.Import):
            for a in n.names:
                if a.name.startswith("molbuilder"):
                    out.add(a.name[len("molbuilder."):])
    return {m for m in out if m}


def run_file(nodeid, root, timeout):
    try:
        r = subprocess.run([PYEXE, "-m", "pytest", "-q", "-p", "no:randomly",
                            "-x", nodeid], cwd=root, capture_output=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None                      # unusable signal, not a verdict
    return r.returncode == 0


def vacuity(testfile, budget=10, timeout=420):
    try:
        lines = covered_lines(testfile)
    except SystemExit as e:
        return {"verdict": "NO-COVERAGE-RUN", "detail": str(e)[:200]}
    if not lines:
        return {"verdict": "NO-CODE", "killed": 0}
    work = tempfile.mkdtemp(prefix="vacuity_")
    root = pathlib.Path(work) / "repo"
    subprocess.run(["git", "worktree", "add", "--detach", str(root), "HEAD"],
                   capture_output=True, cwd=".")
    try:
        # A file whose tests all SKIP here is green at HEAD and green under
        # every mutant -- "VACUOUS" would be a verdict on this machine's
        # environment, not on the test.  A skip counts as a pass; that single
        # fact is what hid 717 tests from this suite for months.
        base = subprocess.run([PYEXE, "-m", "pytest", "-q", "-p", "no:randomly",
                               testfile], cwd=root, capture_output=True,
                              text=True, timeout=timeout)
        tail = (base.stdout or "")[-400:]
        if base.returncode != 0:
            return {"verdict": "NOT-GREEN-AT-HEAD", "detail": tail[-200:]}
        import re as _re
        passed = _re.search(r"(\d+) passed", tail)
        if not passed or int(passed.group(1)) == 0:
            return {"verdict": "ALL-SKIPPED-HERE", "detail": tail.strip()[-120:]}
        by_file = {}
        for f, l in lines:
            by_file.setdefault(f, set()).add(l)
        # MUTATE WHAT THE FILE NAMES, round-robin, SUBJECT FIRST.
        # Two instrument faults, both measured on tests/test_persist.py:
        # spending the budget on the widest-covered module measures the
        # framework the test travelled through, and spending it on ONE named
        # module starves the rest.  `test_persist.py` covers 618 lines across
        # 17 files; 8 mutants all landed in `jobset/model.py` and the file was
        # reported blind.  So: the module whose name matches the test file
        # goes first, then the other direct imports, one mutant each in turn.
        named = _named_modules(testfile)
        subject = pathlib.Path(testfile).stem[len("test_"):]

        def _rank(rel):
            mod = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel
            if mod.split(".")[-1] == subject or mod == subject:
                return 0
            return 1 if any(mod.endswith(n) or n.endswith(mod) for n in named) else 2

        streams = []
        for rel in sorted(by_file, key=lambda r: (_rank(r), r)):
            if _rank(rel) == 2:
                continue                 # framework the test merely passed through
            if not (root / "molbuilder" / rel).exists():
                continue
            streams.append((rel, iter(mutants_for(pathlib.Path("molbuilder") / rel,
                                                  by_file[rel], root))))
        if not streams:                  # nothing it names is reachable
            streams = [(rel, iter(mutants_for(pathlib.Path("molbuilder") / rel,
                                              by_file[rel], root)))
                       for rel in sorted(by_file, key=lambda r: -len(by_file[r]))[:3]
                       if (root / "molbuilder" / rel).exists()]
        tried = killed = 0
        while streams and tried < budget:
            for rel, it in list(streams):
                try:
                    ln, mutated = next(it)
                except StopIteration:
                    streams.remove((rel, it)); continue
                target = root / "molbuilder" / rel
                original = target.read_text()
                target.write_text(mutated)
                try:
                    green = run_file(testfile, root, timeout)
                finally:
                    target.write_text(original)
                if green is None:
                    continue
                tried += 1
                if not green:
                    return {"verdict": "GUARDS", "killed": 1, "tried": tried,
                            "first": f"{rel}:{ln}"}
                if tried >= budget:
                    break
        return {"verdict": "VACUOUS" if tried else "NO-MUTANTS",
                "killed": 0, "tried": tried,
                "lines": len(lines), "files": len(by_file)}
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(root)],
                       capture_output=True, cwd=".")
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    f = sys.argv[1]
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(json.dumps({"file": f, **vacuity(f, b)}), flush=True)
