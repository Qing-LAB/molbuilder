"""The pages the app serves, as the browser receives them -- for a test that
reads what was served rather than the templates it was rendered from."""
from __future__ import annotations


def served_pages(app):
    """Every page ``app`` serves: each GET route of its map with no URL
    argument, outside ``/api/`` and ``/static/``, that answers 200 with
    HTML -- ``(path, html)`` pairs."""
    web = app.test_client()
    for rule in app.url_map.iter_rules():
        path = str(rule)
        if (path.startswith(("/api/", "/static/")) or rule.arguments
                or "GET" not in rule.methods):
            continue
        r = web.get(path)
        if r.status_code == 200 and r.mimetype == "text/html":
            yield path, r.get_data(as_text=True)


#: A CAS sign-in, which needs no secret: an app built with it serves the
#: sign-in page the plain test app does not have.
SIGN_IN = {"id": "cas", "label": "Sign in", "kind": "cas",
           "allowed_users": ["user@example.com"],
           "login_url": "https://cas.example.com/cas/login",
           "email_domain": "example.com"}


def every_served_page() -> dict:
    """Every page the app serves -- the plain app's, and the sign-in page of
    one that has sign-in -- ``{path: html}``."""
    from molbuilder.web.app import create_app
    quiet = {"rate_limit": {"enabled": False}}
    pages = dict(served_pages(create_app(config=quiet)))
    pages.update(served_pages(create_app(
        config={**quiet, "auth": {"providers": [SIGN_IN]}})))
    return pages
