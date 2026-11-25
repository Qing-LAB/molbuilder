"""The pages the app serves, as the browser receives them -- for a test that
reads what was served rather than the templates it was rendered from --
what a page's request is served of a result (:func:`spectra_served`), a
server sentence as a page draws it (:func:`drawn`), what a chart's own
Export hands over (:func:`chart_exported`), where it stands beside the
chart's toolbar (:func:`export_clear_of_the_chart`), and the one save-where
question answered (:func:`where_and_name_answered`)."""
from __future__ import annotations

from pathlib import Path


def chart_exported(page, box, form: str, to) -> Path:
    """WHAT A CHART'S OWN EXPORT HANDS OVER (`web/results.md` § 1.2): the
    Export button in ``box`` -- a locator of the chart's box, or of the wrap
    around a sealed chart -- opened and
    ``form`` (``PNG``, ``SVG``, ``NetCDF``, ``CSV``) pressed, as a person
    presses them; the download saved in ``to`` under the name it was handed
    over with, its path answered."""
    from playwright.sync_api import TimeoutError as NoDownload
    box.locator(".chart-export-open").click()
    try:
        with page.expect_download(timeout=120000) as got:
            box.locator(".chart-export-items").get_by_role(
                "button", name=form, exact=True).click()
    except NoDownload:
        said = box.locator(".chart-export-said").inner_text()
        raise AssertionError(f"{form}: no download; the chart's Export "
                             f"said {said!r}") from None
    saved = Path(to) / got.value.suggested_filename
    got.value.save_as(saved)
    return saved


def export_clear_of_the_chart(box) -> None:
    """WHERE A CHART'S OWN EXPORT STANDS (`web/results.md` § 1.2): the chart
    in ``box`` -- a locator of its box, or of the wrap around a sealed chart
    -- keeps Plotly's toolbar, its buttons the view's alone -- zoom, pan and
    reset among them, no picture, no share, nothing that leaves the page;
    and the Export button sits above the drawing, so it covers neither the
    toolbar nor anything the chart draws."""
    box.locator(".chart-export-open").wait_for(timeout=60000)
    box.locator(".modebar-btn").first.wait_for(state="attached", timeout=60000)
    titles = box.locator(".modebar-btn").evaluate_all(
        "bs => bs.map(b => b.getAttribute('data-title'))")
    assert {"Zoom", "Pan", "Reset axes"} <= set(titles) <= {
        "Zoom", "Pan", "Zoom in", "Zoom out", "Autoscale", "Reset axes"}, titles
    # The toolbar is inside the drawing's container, so above the one is
    # above the other.
    export = box.locator(".chart-export-open").bounding_box()
    drawing = box.locator(".svg-container").first.bounding_box()
    assert export["y"] + export["height"] <= drawing["y"] + 0.5, (
        export, drawing)


def drawn(said: str) -> str:
    """``said`` -- a sentence the server writes for the terminal and the page
    alike -- as a page draws it (`web/ui-contract.md` § 12.1): a command in
    backticks shown as code, its backticks gone; the terminal's dash, ` -- `,
    a dash."""
    return said.replace("``", "`").replace("`", "").replace(" -- ", " \u2014 ")


def served_pages(app, *, user=None):
    """Every page ``app`` serves: each GET route of its map with no URL
    argument, outside ``/api/`` and ``/static/``, that answers 200 with
    HTML -- ``(path, html)`` pairs -- signed in as ``user`` when one is
    given.  A page that answers an error fails here, naming it: a page that
    cannot render is not one to skip.

    Asked over HTTPS, the only scheme the session cookie rides; and signed
    in afresh before each page, because ``/login`` and ``/logout`` act on
    the session."""
    web = app.test_client()
    base = "https://localhost"
    for rule in app.url_map.iter_rules():
        path = str(rule)
        if (path.startswith(("/api/", "/static/")) or rule.arguments
                or "GET" not in rule.methods):
            continue
        if user is not None:
            with web.session_transaction(base_url=base) as session:
                session.clear()
                session["user"] = dict(user)
        r = web.get(path, base_url=base)
        assert r.status_code < 400, f"{path} answered {r.status_code}"
        if r.status_code == 200 and r.mimetype == "text/html":
            yield path, r.get_data(as_text=True)


#: One sign-in of every kind molbuilder offers, none needing a secret to be
#: built: an app built with them serves the sign-in page the plain test app
#: does not have, each kind's button drawn.
_ALLOWED = {"allowed_users": ["user@example.com"]}
SIGN_INS = [
    {"id": "cas", "label": "Sign in", "kind": "cas",
     "login_url": "https://cas.example.com/cas/login",
     "email_domain": "example.com", **_ALLOWED},
    *({"id": kind, "label": f"Sign in with {kind}", "kind": kind,
       "client_id": "a-client", **_ALLOWED}
      for kind in ("google", "github", "microsoft", "orcid")),
]

#: The person signed in through the CAS one, as its login stores them.
SIGNED_IN = {"email": "user@example.com", "provider_id": "cas",
             "name": "A person", "picture": None, "sub": None}


def every_served_page() -> dict:
    """Every page the app serves -- the plain app's; one that has sign-in
    of every kind, signed out (its sign-in page) and signed in (every page,
    with the account strip a signed-in page carries, keyed
    ``"<path> (signed in)"``) -- ``{path: html}``."""
    from molbuilder.runtime_config import _PROVIDER_KINDS
    from molbuilder.web.app import create_app
    assert {s["kind"] for s in SIGN_INS} == set(_PROVIDER_KINDS), \
        sorted(_PROVIDER_KINDS)
    quiet = {"rate_limit": {"enabled": False}}
    pages = dict(served_pages(create_app(config=quiet)))
    signed = create_app(config={**quiet, "auth": {"providers": SIGN_INS}})
    pages.update(served_pages(signed))
    pages.update({f"{path} (signed in)": html
                  for path, html in served_pages(signed, user=SIGNED_IN)})
    return pages


def ids_of(html: str) -> list:
    """Each element ``id`` the page ``html`` declares, as the browser parses
    it, in the order declared -- one declared twice is listed twice."""
    from html.parser import HTMLParser

    class _Ids(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.ids = []

        def handle_starttag(self, tag, attrs):
            self.ids += [value for name, value in attrs if name == "id"]

        handle_startendtag = handle_starttag

    parser = _Ids()
    parser.feed(html)
    parser.close()
    return parser.ids


def spectra_served(web_client, monkeypatch, path, **ask) -> dict:
    """What the load door serves of the result at ``path``, asked as the page
    asks (`POST /api/spectra/load`, ``ask`` beside the path), inside the tree
    it lies in, the one door pointed at it (the fence, `web-api.md` § 2.1;
    `conftest.point_the_one_door_at`): the answer's ``results``."""
    from conftest import point_the_one_door_at
    from molbuilder.projects import find_projects_root
    point_the_one_door_at(find_projects_root(path).parent, monkeypatch.setenv)
    r = web_client.post("/api/spectra/load", json={"path": str(path), **ask})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["ok"] is True
    return r.get_json()["results"]


def where_and_name_answered(page, project, where, stem):
    """The one save-where question (`projects.chooseSavePath`), answered as
    a person answers it: WHERE -- the picker's Choose stays disabled until a
    folder is selected; opening the project and clicking the folder's row
    is the selection -- then WHAT TO CALL IT.  The projects root is the
    tree, never an answer: its row cannot be chosen (`projects.md` § 4.1).
    The picker opens at the root each time (a closed dialog is gone), and
    is anchored at its top: the project opened, its top has not moved
    (`ui-contract.md` § 11 -- a centred picker moved its rows under the
    pointer between the two clicks of a double-click)."""
    from support.sidebar import WAIT_MS
    dialog = page.locator("dialog.molbuilder-tree-picker-dialog")
    root_row = dialog.locator(
        f'li.tp-node[data-path="{project.parent}"] > .tp-row')
    root_row.wait_for(state="visible", timeout=WAIT_MS)
    assert "tp-row--inert" in (root_row.get_attribute("class") or "")
    top = dialog.bounding_box()["y"]
    dialog.locator(f'li.tp-node[data-path="{project}"]'
                   " > .tp-row > .tp-twisty").click()
    row = dialog.locator(f'li.tp-node[data-path="{where}"] > .tp-row')
    row.wait_for(state="visible", timeout=WAIT_MS)
    assert dialog.bounding_box()["y"] == top
    row.click()
    page.locator("dialog [data-action='confirm']").first.click()
    name = page.locator("dialog input[data-role='name']")
    name.wait_for(state="visible", timeout=WAIT_MS)
    name.fill(stem)
    page.locator("dialog [data-action='confirm']").last.click()
