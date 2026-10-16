"""What a page drew of the findings its check returned -- for a test that asks
a page shows every one (`science/validation.md` § 4.1 R2, R3: a row each,
the drawn count the count received)."""
from __future__ import annotations

from collections import Counter

#: Each row the one renderer drew inside the page's ``scopes`` (CSS selectors
#: of its panel and its form), as `lib/validation-findings.js` writes a row:
#: its severity, its message, its ``where``.  A row inside two scopes is one.
_DRAWN = """(scopes) => {
    const rows = new Set(scopes.flatMap(
        (s) => [...document.querySelectorAll(s + " li.issue-item")]));
    return [...rows].map((li) => [
        li.dataset.severity,
        li.querySelector(".issue-msg").textContent,
        (li.querySelector(".issue-where") || {}).textContent || null]);
}"""


def _said(issues) -> list:
    return [[i["severity"], i["message"], i.get("where") or None]
            for i in issues]


def assert_every_finding_drawn(page, scopes, issues, *, timeout=10_000):
    """``issues`` -- one answer of the page's check -- drawn inside
    ``scopes``, a row each and nothing besides: waited for until the rows
    there are those findings, then compared row for row."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    want = _said(issues)
    try:
        page.wait_for_function(
            f"""([scopes, want]) => {{
                const key = (r) => JSON.stringify(r);
                const got = ({_DRAWN})(scopes).map(key).sort();
                const exp = want.map(key).sort();
                return got.length === exp.length
                    && got.every((k, i) => k === exp[i]); }}""",
            arg=[scopes, want], timeout=timeout)
    except PlaywrightTimeout:
        pass                    # the comparison below says what differs
    drawn = page.evaluate(_DRAWN, scopes)
    said, got = (Counter(map(tuple, want)), Counter(map(tuple, drawn)))
    assert got == said, {"said and not drawn": list((said - got).elements()),
                         "drawn and not said": list((got - said).elements())}
