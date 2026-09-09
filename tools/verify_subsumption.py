"""Does the COVERER die whenever the CANDIDATE dies?  Verify one "subsumed" claim.

    python tools/verify_subsumption.py pairs.json
    # pairs.json: [["tests/f.py::test_candidate", "tests/f.py::test_coverer"], ...]

WHY THIS EXISTS.  A test-retirement audit (2026-09-08) classified ~110 tests
CUT-SUBSUMED -- "another test fails for the same cause".  That is a claim about
which INPUTS reach which code, and it was measured wrong 1 time in 5:

    ok    CONFIRMED     test_preserves_unrelated_record       both died, 2/2
    ok    CONFIRMED     test_can_parse                        both died, 3/3
    ok    CONFIRMED     test_block_size_cap_derives_from_...  both died, 2/2
    KEEP  NOT-SUBSUMED  test_parse_is_deterministic           coverer survived 1/12
    ??    INCONCLUSIVE  test_root_landing_path_follows_TABS   no mutant killed it
    ??    INCONCLUSIVE  test_carbon_is_twelve_times_hydrogen  no mutant killed it

The false one is instructive.  `test_parse_is_deterministic` compares the whole
legacy dict for ONE file parsed twice, so it reaches the wall-clock-to-elapsed
derivation at `parse/engines/_helpers.py:303`.  Its supposed coverer compares two
DIFFERENT files and never reaches that.  **Two tests can cover one line and still
assert different things about it** -- which is why coverage alone cannot settle
this, and why an argument cannot either.

COVERAGE WAS TRIED FIRST AND IS NOT ENOUGH.  On the control pair both tests
execute the same lines; the gap was that no frozen fixture carries both
`SCF_NOT_CONV` and ">> End of run".  Coverage sees lines; subsumption is about
inputs reaching them.  So coverage is used only to CHOOSE where to mutate.

HOW IT WORKS.  Trace the candidate (stdlib `sys.settrace`, no new dependency),
mutate one line it covers, run both tests.  Candidate red + coverer green = the
claim is false.  A mutant the candidate survives is uninformative and skipped.

WHAT IT CANNOT DO, and each of these was hit in the first run:

  * **INCONCLUSIVE** means NO OPINION, never "verified".  The operators are
    mechanical -- flip a comparison, negate a bool, perturb a literal -- and
    cannot reach every assertion.
  * A **class-nested test needs its full nodeid** (`file.py::Class::test_x`).  A
    bare name matches nothing and reports NO-COVERAGE, which looks like a finding
    and is not.
  * It is **slow**: a tree copy plus two pytest runs per mutant.
  * A CONFIRMED verdict is evidence, not proof: the operator set is finite.
"""

import ast, json, os, subprocess, sys, pathlib, shutil, tempfile

TMP = os.environ["CLAUDE_JOB_DIR"] + "/tmp"
PY  = sys.executable


def covered_lines(nodeid):
    out = f"{TMP}/cov_one.json"
    if os.path.exists(out):
        os.unlink(out)
    env = dict(os.environ, PYTHONPATH=TMP, MB_COV_OUT=out)
    subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:randomly", "-p", "covplug",
                    nodeid], env=env, capture_output=True, timeout=900)
    if not os.path.exists(out):
        return set()
    d = json.load(open(out))
    hits = set()
    for k, v in d.items():          # a parametrized id expands to several
        for s in v:
            f, _, l = s.rpartition(":")
            hits.add((f, int(l)))
    return hits


class _Mut(ast.NodeTransformer):
    """Apply exactly ONE edit, at `target` line, of kind `kind`."""
    FLIP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE,
            ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt,
            ast.In: ast.NotIn, ast.NotIn: ast.In,
            ast.Is: ast.IsNot, ast.IsNot: ast.Is}

    def __init__(self, target):
        self.target, self.done = target, False

    def visit_Compare(self, node):
        self.generic_visit(node)
        if (not self.done and getattr(node, "lineno", None) == self.target
                and len(node.ops) == 1 and type(node.ops[0]) in self.FLIP):
            node.ops = [self.FLIP[type(node.ops[0])]()]
            self.done = True
        return node

    def visit_Constant(self, node):
        """Perturb a literal.

        BOOLS AND COMPARISONS ARE NOT ENOUGH, measured 2026-09-09: two of five
        pairs came back INCONCLUSIVE ("no mutant killed the candidate") because
        the code they exercise has no comparison to flip -- a constant table
        (`atomic_mass`) and a list of route paths.  A verdict of "no opinion" on
        40% of the bucket makes the harness useless for it, so a literal is
        perturbed too: a number by +1, a non-empty string by a suffix.
        """
        if self.done or getattr(node, "lineno", None) != self.target:
            return node
        v = node.value
        if isinstance(v, bool):
            node.value = not v
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            node.value = v + 1
        elif isinstance(v, str) and v:
            node.value = v + "\u0000mut"
        else:
            return node
        self.done = True
        return node


def mutants_for(rel, lines, root):
    src = (root / rel).read_text()
    try:
        base = ast.parse(src)
    except SyntaxError:
        return
    for ln in sorted(lines):
        tree = ast.parse(src)
        m = _Mut(ln)
        tree = m.visit(tree)
        if not m.done:
            continue
        ast.fix_missing_locations(tree)
        try:
            yield ln, ast.unparse(tree)
        except Exception:
            continue


def run(nodeid, root):
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:randomly", "-x",
                        nodeid], cwd=root, capture_output=True, timeout=900)
    return r.returncode == 0        # True = green


def check(candidate, coverer, budget=12):
    lines = covered_lines(candidate)
    if not lines:
        return {"verdict": "NO-COVERAGE", "detail": "candidate covers no molbuilder line"}
    work = tempfile.mkdtemp(prefix="mutsub_")
    root = pathlib.Path(work) / "repo"
    subprocess.run(["git", "worktree", "add", "--detach", str(root), "HEAD"],
                   capture_output=True, cwd=".")
    try:
        by_file = {}
        for f, l in lines:
            by_file.setdefault(f, set()).add(l)
        tried = killed_x = survived_c = 0
        escapes = []
        for rel, lns in sorted(by_file.items()):
            target = root / "molbuilder" / rel
            if not target.exists():
                continue
            original = target.read_text()
            for ln, mutated in mutants_for(pathlib.Path("molbuilder") / rel,
                                           lns, root):
                if tried >= budget:
                    break
                target.write_text(mutated)
                try:
                    x_green = run(candidate, root)
                    if x_green:
                        continue            # mutant not seen by X: uninformative
                    tried += 1; killed_x += 1
                    c_green = run(coverer, root)
                    if c_green:
                        survived_c += 1
                        escapes.append(f"{rel}:{ln}")
                finally:
                    target.write_text(original)
            if tried >= budget:
                break
        if killed_x == 0:
            return {"verdict": "INCONCLUSIVE",
                    "detail": "no mutant killed the candidate"}
        return {"verdict": "NOT-SUBSUMED" if escapes else "CONFIRMED",
                "killed_x": killed_x, "survived_c": survived_c,
                "escapes": escapes[:5]}
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(root)],
                       capture_output=True, cwd=".")
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    pairs = json.load(open(sys.argv[1]))
    for cand, cov in pairs:
        r = check(cand, cov)
        v = r["verdict"]
        mark = {"CONFIRMED": "ok  ", "NOT-SUBSUMED": "KEEP", }.get(v, "??  ")
        print(f"{mark} {v:14} {cand.split('::')[-1][:56]}")
        if v == "NOT-SUBSUMED":
            print(f"       coverer survived {r['survived_c']}/{r['killed_x']} "
                  f"mutants that killed the candidate: {r['escapes']}")
        elif v == "CONFIRMED":
            print(f"       both died on all {r['killed_x']} informative mutants")
        else:
            print(f"       {r.get('detail','')}")
        sys.stdout.flush()
