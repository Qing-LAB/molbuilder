"""Per-test line coverage over molbuilder/, using only the stdlib.

WHY THIS EXISTS.  A CUT-SUBSUMED verdict claims "test C fails for the same cause
as test X".  That is a claim about which CODE each one reaches.  If X executes a
line C never executes, C cannot fail for a defect on that line, and the claim is
false.  Coverage subsumption is therefore a NECESSARY condition -- cheap,
mechanical, and needing no guess about what to mutate.

It is not SUFFICIENT: two tests can cover one line and assert different things
about it.  So this filters; real mutation confirms what survives the filter.
"""
import json, os, sys, pytest

ROOT = os.path.abspath("molbuilder") + os.sep
_hits = {}
_cur = None

def _trace(frame, event, arg):
    if event == "line":
        f = frame.f_code.co_filename
        if f.startswith(ROOT):
            _hits.setdefault(_cur, set()).add((f[len(ROOT):], frame.f_lineno))
    return _trace

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    global _cur
    _cur = item.nodeid
    _hits.setdefault(_cur, set())
    old = sys.gettrace()
    sys.settrace(_trace)
    try:
        yield
    finally:
        sys.settrace(old)
        _cur = None

def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("MB_COV_OUT")
    if out:
        with open(out, "w") as fh:
            json.dump({k: sorted(f"{a}:{b}" for a, b in v)
                       for k, v in _hits.items()}, fh)
