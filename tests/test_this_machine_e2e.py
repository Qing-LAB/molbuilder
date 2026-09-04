"""**What a person actually does on /this-machine, and what they see after.**

Written 2026-09-03 to replace a source pin, and the replacement is not
like-for-like: the pin read `this_machine.html` off disk and checked three
substrings were in it, then said in its own docstring *"the behaviour itself
is pinned by `test_config_dir_has_one_home.py`"*.

**That file has never existed.**  So a page nothing tested carried a docstring
telling the next reader it was tested elsewhere, which is worse than no test —
it closes the question (`process/testing.md` § 3a.1).

Nothing else visits this page either: it was the only route in the app with no
browser test at all, while being the one page that handles credentials.

**Everything here goes through the page's own controls** — typing in the
fields, choosing the kind, pressing Save, opening the disclosure.  Driving
`/api/notify/channels` with `fetch` from inside the browser would be an API
test wearing a browser costume, and the API already has one
(`tests/test_notify_setup.py`); what only a browser can answer is whether the
form a person fills in reaches that API and whether what comes back is safe to
put on screen.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


#: A Slack webhook: the URL *is* the credential, so it must never render in
#: full.  The tail is deliberately distinctive, so a partial leak is caught
#: as surely as a whole one.
_SECRET_URL = ("https://hooks.slack.com/services/"
               "T00000000/B11111111/zzTOPSECRETzz9876")


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """An isolated config dir, so no test reads or writes the real one.

    Function-scoped, hence the function-scoped server below: these tests
    write a credentials file, and `$MOLBUILDER_CONFIG_DIR` is where the
    monitor's own `default_notify_path()` resolves it to
    (`configuration.md` § 2.1c).
    """
    root = tmp_path / "config-root"
    root.mkdir()
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    (tmp_path / "user-home").mkdir()
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return root


@pytest.fixture
def flask_server(config_home):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, base_url):
    """Load the page and wait for its channel card to have painted.

    Not ``#tm-list``: that div is EMPTY, and therefore hidden, until a
    channel exists — which is the state every one of these tests starts in.
    ``#tm-save`` is the card's own control and is there either way.
    """
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{base_url}/this-machine")
    page.wait_for_selector("#tm-save", timeout=5000)
    return errors


def _add_channel(page, name, url, *, key=None):
    """Fill the form and press Save, the way a person adds a channel."""
    page.fill("#tm-name", name)
    page.check("#tm-kind-slack")
    page.fill("#tm-url", url)
    if key is not None:
        page.fill("#tm-key", key)
    page.click("#tm-save")
    page.wait_for_function(
        """(n) => {
            const l = document.querySelector("#tm-list");
            return l && l.textContent.includes(n);
        }""", arg=name, timeout=5000)


def test_the_page_states_every_branch_of_the_config_dir_rule(
        page, flask_server):
    """All three, because naming only two sends a person to a directory the
    monitor does not read — and silently, since an absent notify file simply
    means "no notifier" (`configuration.md` § 2.1c).

    The rule lives inside a ``<details>``, so this OPENS it first, as a
    person must.  That is the half a source pin could not see: a rule
    present in the file but sealed inside a disclosure that never opens is
    not stated to anybody.
    """
    _open(page, flask_server)
    details = page.locator("details.tm-expects")
    assert details.count() == 1, "the 'what the target machine needs' block is gone"
    page.locator("details.tm-expects > summary").click()
    page.wait_for_function(
        "() => document.querySelector('details.tm-expects').open",
        timeout=3000)

    shown = details.inner_text()
    for branch in ("$MOLBUILDER_CONFIG_DIR",
                   "$XDG_CONFIG_HOME/molbuilder",
                   "~/.config/molbuilder"):
        assert branch in shown, (
            f"the page does not tell the reader about {branch}.  It is "
            f"resolved on the far machine, so stating it is the only thing "
            f"this page can honestly do about it.")
    assert "notify" in shown, (
        "the page names no file, so a reader learns the directory and not "
        "what to put in it")


def test_a_channels_secret_never_reaches_the_page(page, flask_server):
    """The property the whole `_row` door exists to hold.

    A Slack webhook URL is the entire credential — anyone holding it can post
    as you.  Add one through the form, then read the page back: it must not
    be there, in text or in any attribute.

    Checked against the serialised DOM rather than the visible text, because
    a value can sit in a ``title``, a ``value``, or a ``data-`` attribute and
    still be one clipboard away from a person.
    """
    _open(page, flask_server)
    _add_channel(page, "prod-alerts", _SECRET_URL)

    dom = page.content()
    assert "prod-alerts" in dom, "the saved channel is not on the page at all"
    assert _SECRET_URL not in dom, (
        "the webhook URL is rendered in full.  For Slack and Discord the URL "
        "IS the credential, so this is the same defect as printing a key.")
    assert "zzTOPSECRETzz9876" not in dom, (
        "the secret tail of the webhook URL reached the page -- a partial "
        "leak of a credential is a leak")
    # ...and the masked form IS shown, or the assertions above would also
    # pass on a page that renders no channel at all.
    assert "hooks.slack.com" in dom, (
        "nothing about the address is shown, so the reader cannot tell which "
        "service the channel points at -- masking is not hiding")


# NOT asserted here: that a successful save clears `#tm-url`.  It does not,
# and `web/this-machine.md` § 3.1 enumerates what a save DOES clear -- the
# previous format's top-level `url`/`key`/`headers` in the FILE -- a closed
# list the form is not on.  The observation is real (a saved webhook stays in
# the box until the tab closes) but it is a UI decision nobody has made, and a
# test may not invent a contract for one (`process/testing.md`: tests serve
# the contract, never write it).
def test_saving_writes_the_file_the_page_tells_you_to_write(
        page, flask_server, config_home):
    """The page instructs a person to put a 0600 file at
    ``<config dir>/notify``.  Saving through the form must produce exactly
    that file, or the instruction on screen is about something the app does
    not do.

    This is the end of the chain the page describes: form → API → the file a
    monitor on the compute node will open.
    """
    _open(page, flask_server)
    _add_channel(page, "prod-alerts", _SECRET_URL)

    dest = config_home / "notify"
    assert dest.exists(), (
        f"the page says the monitor reads <config dir>/notify, but saving "
        f"through the form wrote no such file under {config_home}")
    assert (dest.stat().st_mode & 0o077) == 0, (
        "the notify file holds credentials and the page states mode 0600; "
        f"it was written {oct(dest.stat().st_mode & 0o777)}")
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert "prod-alerts" in json.dumps(doc), (
        "the channel is not in the file the monitor will read")
