"""THE way a browser test opens Task setup on a folder, its cards, its
machine card and its editor -- as a person does (`process/testing.md` § 5,
*Drive the page as a person does*).

A folder is opened as a person opens it: the tab, then the folder's rows in
the sidebar (`support.sidebar.walk_to`).  Every card of the page folds, and a
folder opened anew shows them folded to their heads (`task-setup.md`
§ 9a.1) -- so a test that works inside a card opens it first, as a person
does: a click on its head.  Once the folder has been read: the read folds
the main column's cards when it starts, so a card opened before then would
be folded back under the test.  The column says it is being rewritten
(`aria-busy`, set in the template and by every read) until the read -- and
the machines' list it waits on -- has ended.

The two records are read where a person reads them: the page's one editor,
a tab each (§ 9a).
"""
from __future__ import annotations

from support.sidebar import WAIT_MS

#: The page's one editor (`task-setup.md` § 9a): a document per record, the
#: tab chosen showing its own.
EDITOR = "#ts-editor-card .CodeMirror"

#: The machine card's name for the machine the server runs on
#: (`task-setup.md` § 6.4).
THIS_MACHINE = "(this machine)"


def task_setup_on(page, base: str, folder) -> None:
    """Task setup opened on ``folder`` as a person opens it: the tab, the
    folder's rows walked in the sidebar (`support.sidebar.walk_to`), and --
    its read ended -- every card of the main column opened (:func:`unfold`).
    """
    from support.sidebar import walk_to
    page.goto(f"{base}/task-setup")
    walk_to(page, folder)
    unfold(page)


def unfold(page, *cards: str) -> None:
    """Open each card ``cards`` names (by id) -- with none named, every card
    the main column shows -- once the folder is read."""
    page.wait_for_selector("#ts-main:not([aria-busy])", state="attached",
                           timeout=WAIT_MS)
    if not cards:
        cards = tuple(page.eval_on_selector_all(
            "#ts-main > details.fold:not([hidden])", "ds => ds.map(d => d.id)"))
    for card in cards:
        head = page.locator(f"#{card} > summary")
        head.wait_for(state="visible", timeout=WAIT_MS)
        if not page.locator(f"#{card}[open]").count():
            head.click()
        page.locator(f"#{card}[open]").wait_for(state="attached",
                                                timeout=WAIT_MS)


def every_card_text(page) -> str:
    """What the page shows with every card open: the main column's at once,
    the rail's one at a time (one is open at a time), each with every group
    it folds opened -- the text a person reads opening them all."""
    unfold(page)
    texts = [page.inner_text("#ts-main")]
    for card in page.eval_on_selector_all(
            ".page-aside > details.fold:not([hidden])",
            "ds => ds.map(d => d.id)"):
        unfold(page, card)
        closed = page.locator(f"#{card} details.fold:not([open]) > summary")
        for _ in range(64):                 # each click opens one
            if not closed.count():
                break
            closed.first.click()
        assert not closed.count(), f"a group of #{card} would not open"
        texts.append(page.inner_text(f"#{card}"))
    return "\n".join(texts)


def machine(page, name: str = THIS_MACHINE):
    """The machine card's choice ``name`` -- :data:`THIS_MACHINE` unless
    named: the button a person presses to prepare for that machine, pressed
    (``aria-pressed``) while it is the one chosen -- by the person, or by
    the page where one machine is listed and no choice is required
    (`task-setup.md` § 6.4)."""
    return page.locator(f'#ts-target-choice .opt[data-machine="{name}"]')


def picked_by_the_page(page) -> None:
    """The machine card showing :data:`THIS_MACHINE` chosen by the page
    itself: the one machine listed, so no choice is required (`task-setup.md`
    § 6.4) -- asserted, never clicked in the page's place."""
    from playwright.sync_api import expect
    expect(machine(page)).to_have_attribute("aria-pressed", "true",
                                            timeout=WAIT_MS)


def page_free(page) -> None:
    """Wait as a person waits: until the busy cover is gone -- every card
    edit, Save and Prep writes under it (`ui-contract.md` § 10)."""
    page.wait_for_selector(".page-busy-cover", state="hidden",
                           timeout=WAIT_MS)


def _shown(page, which: str) -> None:
    """Record ``which``'s tab -- ``task`` or ``request`` -- chosen in the
    files card, the card opened as a person opens it."""
    unfold(page, "ts-editor-card")
    page.click(f"#ts-editor-tab-{which}")


def file_text(page, which: str) -> str:
    """The text the editor holds for record ``which`` -- ``task`` or
    ``request`` -- read as a person reads it: once the page is free, so a
    card's write has landed (:func:`page_free`), its tab chosen in the files
    card (`task-setup.md` § 9a)."""
    page_free(page)
    _shown(page, which)
    return page.locator(EDITOR).evaluate("n => n.CodeMirror.getValue()")


def edited_by_hand(page, which: str, text: str) -> None:
    """Record ``which``'s text in the editor -- ``task`` or ``request`` --
    replaced by ``text`` as a person edits it there by hand: its tab chosen
    in the files card, the editor's text replaced whole
    (`support.sidebar.replace_text`) -- which the page reads back into its
    cards (`task-setup.md` § 9a)."""
    from support.sidebar import replace_text
    _shown(page, which)
    replace_text(page, EDITOR, text)


def wait_for_file(page, which: str, pattern: str) -> None:
    """Wait until record ``which``'s text, its tab chosen, matches the
    regular expression ``pattern`` -- what a card wrote, or the folder's
    read, reaching the buffer Save sends."""
    _shown(page, which)
    page.wait_for_function(
        "([sel, re]) => { const n = document.querySelector(sel);"
        " return !!n && new RegExp(re).test(n.CodeMirror.getValue()); }",
        arg=[EDITOR, pattern], timeout=WAIT_MS)
