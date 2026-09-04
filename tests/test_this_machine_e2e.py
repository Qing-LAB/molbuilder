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
(`tests/test_notify_setup_api.py`, 40-odd tests); what only a browser can
answer is whether the
form a person fills in reaches that API and whether what comes back is safe to
put on screen.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


#: A Slack webhook: the URL *is* the credential — whoever holds it can post
#: into the channel as you, with no second factor — so it must never render
#: in full.
#:
#: The last four characters ARE shown, on purpose: `_mask` keeps
#: ``MASK_TAIL = 4`` so two webhooks on one page can be told apart.  So the
#: secret part of this fixture is the middle, and the assertions below name
#: the boundary rather than saying "no part of it may appear", which would
#: be a different rule from the one the code implements.
_SECRET_URL = ("https://hooks.slack.com/services/"
               "T00000000/B11111111/zzTOPSECRETzz9876")

#: The path segment that must never be readable.  Deliberately not the last
#: four characters, which the mask is entitled to show.
_SECRET_MIDDLE = "zzTOPSECRETzz"

#: A molbuilder listener's address.  Not itself a secret -- and masked all
#: the same, because the alternative is asking "does it have a key?", which
#: a mislabelled channel answers wrongly.
_LISTENER_URL = "https://lab.example.org/molbuilder/report/aa11bb22cc33"

#: A signing key: the SECOND credential a channel can carry, and unlike the
#: address it has no visible form at all — `_row` reports only ``has_key``.
_SECRET_KEY = "sk-live-DO-NOT-PRINT-4242"

#: **Nothing here talks to Slack, and nothing is a real credential.**  The
#: kinds are routing labels the page itself offers (`#tm-kind-slack`,
#: `-discord`, `-listener`); what is under test is the MECHANISM behind them
#: — `_mask`, the `_row` door, the file that gets written.  The one control
#: that would send anything outward is Test, and no test here presses it.
#: The values above are self-labelling fakes on a documentation domain, and
#: the server is a throwaway on 127.0.0.1.

#: Names the box's own hint forbids: "Letters, digits, `-` and `_`".
_FORBIDDEN = {"has space", "semi;colon", "sla/sh", "back\\slash"}

#: ...and two it must keep accepting, so a fix cannot overshoot.
_ALLOWED = {"ok-name_1", "plain"}


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

    Returns the JS-error list, which every caller **asserts is empty**.
    That assertion is not ceremony: it is what found the defect below.

    ``pattern="[A-Za-z0-9_-]{1,64}"`` on ``#tm-name`` looks fine and was
    dead.  Chrome compiles a `pattern` attribute under the `v` flag, where
    an unescaped ``-`` before ``]`` is a syntax error -- so the browser
    DISCARDED the constraint and `checkValidity()` accepted "has space" and
    "sla/sh", while the hint beside the box promised "Letters, digits, -
    and _".  The page stated a rule it did not enforce, in silence, for as
    long as the attribute had been there.

    The failure was visible the whole time, as a console error on every
    load, and nothing was listening.  Fixed 2026-09-03 by escaping the
    dash; pinned by ``test_a_name_the_page_forbids_is_refused`` below.
    """
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))
    page.goto(f"{base_url}/this-machine")
    page.wait_for_selector("#tm-save", timeout=5000)
    return errors


def _add_channel(page, name, url, *, kind="slack", key=None):
    """Fill the form and press Save, the way a person adds a channel.

    ``kind`` is not decoration: **the Key box only exists for a listener.**
    `#tm-key-field` ships `hidden` and the kind radio reveals it, so asking
    a Slack channel for a key is asking for a control the page does not
    offer -- which is how the first version of this file timed out on a
    `page.fill` and taught me the two credential shapes are genuinely
    different surfaces, not one surface with an optional extra.
    """
    page.fill("#tm-name", name)
    page.check(f"#tm-kind-{kind}")
    page.fill("#tm-url", url)
    if key is not None:
        page.wait_for_selector("#tm-key", state="visible", timeout=5000)
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
    errors = _open(page, flask_server)
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
    assert errors == [], f"the page reported JS errors: {errors}"


def test_a_channels_secret_never_reaches_the_page(page, flask_server):
    """The property the whole `_row` door exists to hold.

    A Slack webhook URL is the entire credential — anyone holding it can post
    as you.  Add one through the form, then read the page back: it must not
    be there, in text or in any attribute.

    Checked against the serialised DOM rather than the visible text, because
    a value can sit in a ``title``, a ``value``, or a ``data-`` attribute and
    still be one clipboard away from a person.
    """
    errors = _open(page, flask_server)
    _add_channel(page, "prod-alerts", _SECRET_URL)

    dom = page.content()
    assert "prod-alerts" in dom, "the saved channel is not on the page at all"
    assert _SECRET_URL not in dom, (
        "the webhook URL is rendered in full.  For Slack and Discord the URL "
        "IS the credential, so this is the same defect as printing a key.")
    assert _SECRET_MIDDLE not in dom, (
        f"the readable part of the webhook path reached the page.  The mask "
        f"is entitled to the last {4} characters -- that is how two channels "
        f"are told apart -- but everything before them identifies the hook "
        f"well enough to replay it.")
    # ...and the masked form IS shown, or every assertion above would also
    # pass on a page that renders no channel at all.
    assert "hooks.slack.com" in dom, (
        "nothing about the address is shown, so the reader cannot tell which "
        "service the channel points at -- masking is not hiding")
    assert errors == [], f"the page reported JS errors: {errors}"


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
    # READ IT THE WAY THE MONITOR WILL.  `monitor.load_channels()` is the
    # function that opens this file on the compute node -- so asserting
    # through it answers "will the monitor find this channel", which is the
    # only question the page's instruction is making a promise about.
    # `json.loads` would answer the weaker "is there a channel-shaped thing
    # in the file", and the two differ: the monitor refuses a file whose
    # top level is the retired single-destination shape, and says so.
    from molbuilder.monitor import load_channels

    chans = load_channels(str(dest))
    assert "prod-alerts" in chans, (
        f"the monitor's own reader does not find the channel in {dest}; it "
        f"sees {sorted(chans)}.  The page told the user to put a file here "
        f"for the monitor to read, so this is the promise it made.")
    blob = json.dumps(chans)
    # The other half of the secret story: what must NOT reach the screen
    # must still reach the FILE.  Without this, a save that quietly dropped
    # the key would satisfy "the key never renders" perfectly, and the
    # channel would fail to authenticate on a compute node with nothing on
    # any surface to say why.
    assert _SECRET_URL in blob, (
        "the notify file is missing the address that was typed.  The monitor "
        "reads this file and nothing else, so a credential that is masked on "
        "screen AND absent here is simply lost -- the channel would fail to "
        "authenticate on a compute node with nothing on any surface saying "
        "why.")


def test_a_listeners_key_never_reaches_the_page(page, flask_server,
                                                config_home):
    """The OTHER credential shape, and the one `_mask` warns about.

    A molbuilder listener carries a plain address **plus a key that signs
    each report and never travels with it** — so it is the only kind whose
    Key box the page reveals at all.

    `_mask`'s own docstring records the trap: the kind used to be derived
    here as ``"listener" if key else "webhook"``, so *whether a key is
    stored* decided whether the address was printed in full — and "a rule
    that can be defeated by mislabelling is not a rule".  Hence **every**
    address is masked now, a listener's included, and it loses nothing:
    the tail still names the segment and the listener section below shows
    the route in full.

    Two things must therefore hold at once, and only one of them is
    obvious: the key must not render, AND the address must be masked even
    though it is not itself a secret.
    """
    errors = _open(page, flask_server)
    _add_channel(page, "lab-listener", _LISTENER_URL,
                 kind="listener", key=_SECRET_KEY)

    dom = page.content()
    assert "lab-listener" in dom, "the saved listener is not on the page"
    assert _SECRET_KEY not in dom, (
        "the signing key is on the page.  It has no masked form and no "
        "reason to be rendered at all: `_row` reports only `has_key`, "
        "because *is a signature attached* is the whole question a reader "
        "has about it -- and the key is the one thing that must never "
        "leave the file it is stored in.")
    assert _LISTENER_URL not in dom, (
        "the listener's address is printed in full.  It is masked like "
        "every other kind on purpose: deriving 'is this a secret?' from "
        "'does it have a key?' is the rule mislabelling defeats "
        "(`notify_setup._mask`).")
    assert "lab.example.org" in dom, (
        "nothing about the listener's address is shown, so a reader cannot "
        "tell where reports go -- masking is not hiding")
    assert errors == [], f"the page reported JS errors: {errors}"

    # The key must still reach the FILE.  A page that satisfies "the key
    # never renders" by never storing it would leave every report from that
    # machine unsigned, and nothing on screen would say so.
    from molbuilder.monitor import load_channels

    stored = load_channels(str(config_home / "notify"))
    assert _SECRET_KEY in json.dumps(stored), (
        "the key was never written to <config dir>/notify, so the monitor "
        "has nothing to sign reports with")


def test_a_name_the_page_forbids_is_refused(page, flask_server):
    """The box says "Letters, digits, ``-`` and ``_``", so it must mean it.

    **This is not a test about regex syntax.**  A channel name is what you
    tick on a calculation and what the monitor looks up in the file on the
    compute node, so a name with a space or a slash in it is a channel that
    is named one thing here and looked for as another there — and nothing
    reports that, because "no channel by that name" is indistinguishable
    from "no notifier" (`configuration.md` § 2.1c).

    The constraint was dead from the day it was written: ``pattern`` is
    compiled under the `v` flag, where an unescaped ``-`` before ``]`` is a
    syntax error, and a `pattern` that does not compile is discarded rather
    than enforced.  So the form accepted every name and the hint beside it
    was decoration.

    Driven through ``checkValidity()`` rather than by pressing Save,
    because refusing on submit is the browser's job here and what is under
    test is whether the rule reaches it at all.
    """
    _open(page, flask_server)
    verdicts = page.evaluate("""(names) => {
        const n = document.getElementById("tm-name");
        const out = {};
        for (const v of names) { n.value = v; out[v] = n.checkValidity(); }
        n.value = "";
        return out;
    }""", sorted(_FORBIDDEN | _ALLOWED))
    accepted = {k for k, ok in verdicts.items() if ok}

    # Report only what actually went wrong, on each side.  The first draft
    # printed EVERY accepted name under "the form accepts names its own hint
    # forbids", so a real failure listed `ok-name_1` and `plain` among the
    # offenders and sent the reader hunting for a defect in names that were
    # behaving perfectly.
    leaked = sorted(_FORBIDDEN & accepted)
    assert not leaked, (
        f"the form accepts {leaked}, which its own hint forbids.  A "
        f"`pattern` that fails to compile is DISCARDED by the browser, not "
        f"enforced -- so the field reads as validated and is not.  Check the "
        f"attribute compiles: an unescaped `-` before `]` is a syntax error "
        f"under the `v` flag that `pattern` uses, and the console says so on "
        f"every page load.")
    over_tight = sorted(_ALLOWED - accepted)
    assert not over_tight, (
        f"the form refuses {over_tight}, which the hint beside the box says "
        f"are legal.  Over-tightening is the same defect pointing the other "
        f"way: a person cannot name a channel the thing they already called "
        f"it on the compute node.")
