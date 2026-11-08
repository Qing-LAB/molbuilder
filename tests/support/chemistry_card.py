"""THE CHEMISTRY CARD, READ AS A PERSON READS IT -- the charge and spin a
calculation will carry, and why, drawn for each engine's form
(`science/chemistry-correctness.md` § 2.5, § 2a.5): once it has answered
(:func:`card_answered`), its blocks by their headings (:func:`card_blocks`).

Used by the card's own browser tests (``tests/test_chemistry_card_e2e.py``)
and the Transport tab's, which asks it of a cited junction
(``tests/test_transport_on_a_real_junction_e2e.py``).
"""
from __future__ import annotations


def card_answered(page, needle: str = "") -> None:
    """Until the card is shown with an answer -- holding ``needle``, when
    one is named."""
    page.wait_for_function(
        "(n) => { const p = document.getElementById('chemistry-panel');"
        "  const s = document.getElementById('chemistry-state');"
        "  return p && !p.hidden && s && s.textContent.length > 0"
        "    && s.textContent.includes(n); }",
        arg=needle, timeout=15000)


def card_blocks(page) -> dict:
    """Each engine's block of the card, by its heading."""
    return page.evaluate(
        "() => Object.fromEntries(Array.from(document.querySelectorAll("
        "  '#chemistry-state .chemistry-engine')).map(b => ["
        "    b.querySelector('h3').textContent, b.textContent]))")
