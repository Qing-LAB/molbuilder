"""Every served page must declare each element id at most once.

The bug this test is here to catch
==================================

On 2026-08-22 the Task-setup tab lost its bench panel entirely.  The
cause was not the panel: a card added days earlier -- "Which machine is
this for" -- was given ``id="ts-machine-card"``, which the older bench
card ("The machine, and what to try on it") already carried.  Both
cards ship ``hidden``; the renderer unhides one by id; and
``getElementById`` answers with the FIRST match in document order.  So
the renderer faithfully built every bench row into a card that stayed
hidden, and the panel simply was not there.

Two words, one layer apart -- "the machine we prepare FOR" and "the
machine settings we measure ON" -- collapsed onto one id.

No existing test caught it, and one test actively hid it: the card's
own test asserted ``'id="ts-machine-card"' in body``, which a duplicate
satisfies twice over.  A substring pin cannot tell one card from two.

The rule is the HTML sibling of ``test_css_no_duplicate_selectors.py``:
one home per id, per page.  It is checked over EVERY served page rather
than per tab, because nothing about this failure was Task-setup's --
any page that hides a card and reaches for it by id can lose it the
same way.
"""
from __future__ import annotations

import re

import pytest


#: ``id="..."`` in served markup.  Attribute order varies by hand, so
#: this reads the attribute itself rather than assuming a tag shape.
_ID = re.compile(r'\bid="([^"]+)"')


def _html_get_routes(app):
    """Every parameter-free GET route that answers with a page.

    Enumerated from the url_map rather than listed by hand: a page
    added tomorrow is covered without anyone remembering to add it,
    which is the whole point of a structural guard.
    """
    out = []
    for rule in app.url_map.iter_rules():
        if rule.arguments:                      # needs a parameter
            continue
        if "GET" not in (rule.methods or set()):
            continue
        if str(rule).startswith("/api/"):       # JSON, not markup
            continue
        out.append(str(rule))
    return sorted(set(out))


@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    return create_app(config={})


def test_no_page_declares_an_id_twice(_app):
    offenders = {}
    client = _app.test_client()
    for path in _html_get_routes(_app):
        resp = client.get(path)
        if resp.status_code != 200:
            continue                            # redirect / partial / 404
        if "html" not in resp.headers.get("Content-Type", ""):
            continue
        body = resp.data.decode()
        seen, dupes = set(), set()
        for name in _ID.findall(body):
            (dupes if name in seen else seen).add(name)
        if dupes:
            offenders[path] = sorted(dupes)
    assert not offenders, (
        "an element id is declared more than once on a page; "
        "getElementById answers with the first, so whichever element "
        "loses is unreachable and any card it holds silently "
        "disappears:\n"
        + "\n".join(f"  {p}: {', '.join(ids)}" for p, ids in offenders.items())
    )


def test_the_two_task_setup_machine_cards_are_separate_things(_app):
    """The specific collision, pinned by ROLE rather than by string.

    ``ts-target-card`` asks which machine the deck is for;
    ``ts-machine-card`` holds the settings measured on it.  A future
    edit that merges the names would pass the duplicate sweep above
    only by deleting a card, so this names what each one is for.
    """
    body = _app.test_client().get("/task-setup").data.decode()
    assert body.count('id="ts-target-card"') == 1
    assert body.count('id="ts-machine-card"') == 1
    # ...and each still owns the children its renderer reaches for.
    assert 'id="ts-target-choice"' in body     # loadMachines() fills this
    assert 'id="ts-machine-rows"' in body      # renderMachine() fills this
