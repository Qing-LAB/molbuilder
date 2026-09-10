"""Does the COVERER die whenever the CANDIDATE dies?  Verify one "subsumed" claim.

    python tools/verify_subsumption.py pairs.json
    # pairs.json: [["tests/f.py::test_candidate", "tests/f.py::test_coverer"], ...]

WHY THIS EXISTS.  A test-retirement audit (2026-09-08) classified ~110 tests
CUT-SUBSUMED -- "another test fails for the same cause".  That is a claim about
which INPUTS reach which code, and with the FIXED operator set below it was
measured wrong on 9 of 19 candidates -- NEARLY HALF.  The 1-in-5 figure this
file used to publish was measured before `visit_BoolOp` existed:

    ok    CONFIRMED     test_preserves_unrelated_record       both died, 2/2
    ok    CONFIRMED     test_can_parse                        both died, 3/3
    ok    CONFIRMED     test_block_size_cap_derives_from_...  both died, 2/2
    KEEP  NOT-SUBSUMED  test_parse_is_deterministic           coverer survived 1/12
    ??    INCONCLUSIVE  test_root_landing_path_follows_TABS   no mutant killed it
    ??    INCONCLUSIVE  test_carbon_is_twelve_times_hydrogen  no mutant killed it

A second batch on 2026-09-09 made the number WORSE, not better:

    KEEP  NOT-SUBSUMED  test_a_traversal_is_refused_by_...    coverer survived 2/5
    KEEP  NOT-SUBSUMED  test_a_missing_path_argument_is_...   coverer survived 1/4
    ok    CONFIRMED     test_the_listing_and_the_submissi...  both died, 1/1
    ok    CONFIRMED     test_nothing_that_should_be_refus...  both died, 2/2
    ??    INCONCLUSIVE  test_the_request_is_shown_even_...    no mutant killed it

**That was still the blind set.**  With clause-dropping added, a re-run of 22
pairs over 19 candidates gives **9 KEEP and 10 cuttable -- wrong on nearly
half**, and three of the nine had been CLEARED FOR DELETION.  Both
new false verdicts are the same shape: the candidate drives a route that reaches
an arm of the fence (`files.py:173`, the missing-path refusal; `:192`, the `..`
refusal) which the named coverer's input never reaches.

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
    cannot reach every assertion.  Two of seven landed here on the first run;
    adding literal perturbation decided one of them
    (`test_root_landing_path_follows_TABS`, a list of route paths -> CONFIRMED).

    **The other is undecidable BY DESIGN, and the reason generalises: the value
    asserted is not ours.**  `test_carbon_is_twelve_times_hydrogen` exercises
    `chemistry.atomic_mass`, whose whole body is a lookup into ASE's table --
    the module docstring says so: *"ASE ships the IUPAC standard atomic weights,
    so this is a name for a table rather than a copy of one."*  There is no
    molbuilder literal to perturb, and this tool deliberately mutates only
    `molbuilder/`.  For that class, read instead of mutate: both tests reach the
    same one-line function, and the sibling pins C to 12.011+-1e-3 and H to
    1.008+-1e-3, confining the ratio to [11.903, 11.928] -- strictly inside the
    candidate's 11.9+-0.2.  Arithmetic, not a mutation question.
  * A **class-nested test needs its full nodeid** (`file.py::Class::test_x`).  A
    bare name matches nothing and reports NO-COVERAGE, which looks like a finding
    and is not.
  * It is **slow**: a tree copy plus two pytest runs per mutant.
  * **A verdict is against the NAMED coverer, never against the whole suite.**
    "Is C subsumed by X" is the question; "is C subsumed by anything" is not.
    Measured 2026-09-09: `test_a_traversal_is_refused_by_its_own_name` came back
    NOT-SUBSUMED against the door test, was re-run against the SECOND coverer the
    auditor named, and came back NOT-SUBSUMED again -- but the two coverers
    between them do reach both arms of the fence. Name every coverer the claim
    rests on, or the verdict is about your pair list.

  * **NO-COVERAGE can be the instrument's reach, not the test's.** `sys.settrace`
    does not follow a subprocess, so a test that probes forked behaviour
    (`test_admin_reload.py`'s `_serve_probe` family) reports zero molbuilder
    lines however much code it drives. Four of eight pairs landed here on
    2026-09-09. Treat NO-COVERAGE as "ask a different way", never as a finding.

  * A CONFIRMED verdict is evidence, not proof: the operator set is finite.
"""

import ast, json, os, subprocess, sys, pathlib, shutil, tempfile

# Scratch for the coverage plug-in and the mutated copies.  It honoured only
# `CLAUDE_JOB_DIR` until 2026-09-09 and raised `KeyError` without it -- so the
# tool ran in exactly the session that wrote it and nowhere else, which is the
# opposite of what a verification tool is for.  `--tmp` first, then the
# environment, then a temp directory it makes itself.
def _scratch() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--tmp" and i + 1 < len(sys.argv):
            d = sys.argv[i + 1]
            sys.argv[i:i + 2] = []
            return d
    for var in ("SUBSUMPTION_TMP", "CLAUDE_JOB_DIR"):
        if os.environ.get(var):
            return os.environ[var].rstrip("/") + "/tmp"
    return tempfile.mkdtemp(prefix="verify_subsumption.")


TMP = _scratch()
os.makedirs(TMP, exist_ok=True)
PY  = sys.executable


#: The per-test coverage plug-in, IN THIS REPOSITORY.  It was loaded as
#: `-p covplug` until 2026-09-09 -- a module written into the scratch directory
#: of the session that first ran this tool and never committed -- so the tool
#: shipped unable to reproduce its own published result, and did so SILENTLY:
#: `covered_lines` returned an empty set and every pair came back NO-COVERAGE,
#: which reads like a finding rather than a broken instrument.  That is the
#: exact fault this tool exists to catch, in the tool itself.
_COV_PLUGIN = "_pertest_coverage"
_TOOLS_DIR = str(pathlib.Path(__file__).resolve().parent)


def covered_lines(nodeid):
    out = f"{TMP}/cov_one.json"
    if os.path.exists(out):
        os.unlink(out)
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join([TMP, _TOOLS_DIR,
                                           os.environ.get("PYTHONPATH", "")]),
               MB_COV_OUT=out)
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:randomly",
                        "-p", _COV_PLUGIN, nodeid],
                       env=env, capture_output=True, timeout=900, text=True)
    if not os.path.exists(out):
        # LOUD.  An empty coverage set is indistinguishable from "this test
        # touches no molbuilder code", and reporting the second when the first
        # happened is how a broken instrument reads as a verdict.
        raise SystemExit(
            f"the coverage plug-in produced no output for {nodeid}.\n"
            f"  plug-in: {_COV_PLUGIN} (from {_TOOLS_DIR})\n"
            f"  pytest exit {r.returncode}\n"
            f"  --- stdout ---\n{(r.stdout or '')[-1500:]}\n"
            f"  --- stderr ---\n{(r.stderr or '')[-800:]}")
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


    def visit_BoolOp(self, node):
        """DROP ONE CLAUSE of an `and`/`or` -- the operator that was missing, and
        the one that matters most for a GUARD.

        MEASURED 2026-09-09, and it cost three real tests nearly being cut.  A
        validator is usually one condition with two clauses:

            if not isinstance(providers, list) or not providers:   # runtime_config:446
            if not isinstance(val, str)        or not val:         # :193
            if not isinstance(tenant, str)     or not tenant:      # :269

        and the two tests on it reach ONE CLAUSE EACH -- missing/non-list hits
        the first, empty hits the second.  Flipping a comparison or perturbing a
        literal kills both together, so the harness said CONFIRMED (subsumed) for
        all three.  Drop `or not providers` and the empty-list test fails while
        the missing-key test passes: they were never subsumed, and cutting them
        would have let `providers: []` and `id: ""` through -- a site whose login
        page has no buttons, and a callback route of `/oauth-callback/`.

        So this operator is not an extra: without it the tool is blind to exactly
        the shape it is most often pointed at.  Each clause is dropped in turn,
        which is why `mutants_for` asks for one mutant per (line, index).
        """
        self.generic_visit(node)
        if (self.done or getattr(node, "lineno", None) != self.target
                or len(node.values) < 2):
            return node
        i = getattr(self, "boolop_drop", None)
        if i is None or i >= len(node.values):
            return node
        kept = [v for j, v in enumerate(node.values) if j != i]
        self.done = True
        return kept[0] if len(kept) == 1 else ast.BoolOp(op=node.op, values=kept)


def mutants_for(rel, lines, root):
    src = (root / rel).read_text()
    try:
        ast.parse(src)          # a syntax guard only; each mutant re-parses
    except SyntaxError:
        return
    for ln in sorted(lines):
        # index None = the comparison/literal operators; 0..2 = drop that clause
        # of a boolean operator on this line (see `_Mut.visit_BoolOp`).
        for drop in (None, 0, 1, 2):
            tree = ast.parse(src)
            m = _Mut(ln)
            m.boolop_drop = drop
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
