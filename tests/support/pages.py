"""The pages the app serves, as the browser receives them -- for a test that
reads what was served rather than the templates it was rendered from."""
from __future__ import annotations


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


#: A CAS sign-in, which needs no secret: an app built with it serves the
#: sign-in page the plain test app does not have.
SIGN_IN = {"id": "cas", "label": "Sign in", "kind": "cas",
           "allowed_users": ["user@example.com"],
           "login_url": "https://cas.example.com/cas/login",
           "email_domain": "example.com"}

#: The person signed in through it, as its login stores them.
SIGNED_IN = {"email": "user@example.com", "provider_id": "cas",
             "name": "A person", "picture": None, "sub": None}


def every_served_page() -> dict:
    """Every page the app serves -- the plain app's; one that has sign-in,
    signed out (its sign-in page) and signed in (every page, with the
    account strip a signed-in page carries, keyed ``"<path> (signed in)"``)
    -- ``{path: html}``."""
    from molbuilder.web.app import create_app
    quiet = {"rate_limit": {"enabled": False}}
    pages = dict(served_pages(create_app(config=quiet)))
    signed = create_app(config={**quiet, "auth": {"providers": [SIGN_IN]}})
    pages.update(served_pages(signed))
    pages.update({f"{path} (signed in)": html
                  for path, html in served_pages(signed, user=SIGNED_IN)})
    return pages
