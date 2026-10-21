"""THE way a browser test opens a Task setup card.

Every card of the page folds, and a folder opened anew shows them folded to
their heads (`task-setup.md` § 9a.1) -- so a test that works inside a card
opens it first, as a person does: a click on its head.  Once the folder has
been read: the read folds the main column's cards when it starts, so a card
opened before then would be folded back under the test.  The column says it
is being rewritten (`aria-busy`, set in the template and by every read) until
the read -- and the machines' list it waits on -- has ended.
"""
from __future__ import annotations


def unfold(page, *cards: str) -> None:
    """Open each card ``cards`` names (by id) -- with none named, every card
    the main column shows -- once the folder is read."""
    page.wait_for_selector("#ts-main:not([aria-busy])", state="attached",
                           timeout=20000)
    if not cards:
        cards = tuple(page.eval_on_selector_all(
            "#ts-main > details.fold:not([hidden])", "ds => ds.map(d => d.id)"))
    for card in cards:
        head = page.locator(f"#{card} > summary")
        head.wait_for(state="visible", timeout=20000)
        if not page.locator(f"#{card}[open]").count():
            head.click()
        page.locator(f"#{card}[open]").wait_for(state="attached",
                                                timeout=5000)


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


def file_text(page, which: str) -> str:
    """The text the editor holds for record ``which`` -- ``task`` or
    ``request`` -- read as a person reads it: its tab chosen in the files
    card (`task-setup.md` § 9a)."""
    unfold(page, "ts-editor-card")
    page.click(f"#ts-editor-tab-{which}")
    return page.evaluate(
        "() => document.querySelector('.CodeMirror').CodeMirror.getValue()")


def wait_for_file(page, which: str, pattern: str,
                  timeout: int = 10000) -> None:
    """Wait until record ``which``'s text, its tab chosen, matches the
    regular expression ``pattern`` -- what a card wrote reaching the
    buffer Save sends."""
    unfold(page, "ts-editor-card")
    page.click(f"#ts-editor-tab-{which}")
    page.wait_for_function(
        "(re) => { const cm = document.querySelector('.CodeMirror');"
        " return !!cm && new RegExp(re).test(cm.CodeMirror.getValue()); }",
        arg=pattern, timeout=timeout)
