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

The rule is one home per id, per page.  It is checked over EVERY served page rather
than per tab, because nothing about this failure was Task-setup's --
any page that hides a card and reaches for it by id can lose it the
same way.
"""
from __future__ import annotations

from html.parser import HTMLParser

import pytest


class _Ids(HTMLParser):
    """Each element's ``id``, as the browser parses the page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []

    def handle_starttag(self, tag, attrs):
        self.ids += [value for name, value in attrs if name == "id"]

    handle_startendtag = handle_starttag


@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    return create_app(config={})


def test_no_page_declares_an_id_twice(isolated_projects_root):
    """Every page the app serves (`support/pages.every_served_page`: each
    one its route map names, signed out and signed in), parsed.

    MUTATION THIS MUST FAIL AGAINST: an id a page already carries given to
    an element of a partial it includes -- the account strip a signed-in
    page alone carries among them."""
    pytest.importorskip("flask")
    from support.pages import every_served_page
    offenders = {}
    for path, html in sorted(every_served_page().items()):
        parsed = _Ids()
        parsed.feed(html)
        parsed.close()
        dupes = sorted({i for i in parsed.ids if parsed.ids.count(i) > 1})
        if dupes:
            offenders[path] = dupes
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
