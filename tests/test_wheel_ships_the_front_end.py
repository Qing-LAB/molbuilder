"""**An installed copy has its whole front end**: a built wheel carries every
file of the package, and every asset a served page names is there.

molbuilder is not pip-installed to be run today, and PyPI is planned (the
user, 2026-10-15), so its package-data list is kept correct
(`pyproject.toml`): it has been wrong twice in ways only an installed copy
showed -- no `.toml` pattern until 2026-08-14, so the
parameter catalogue and the warm-file rules never shipped; and on 2026-09-03
a wheel carrying 90 of the 141 static files, every directory added after
the hand-kept glob list was last edited missing.  The repo worked both
times, because Flask serves `web/static/` off disk.

Both halves are asked of the artifact: the wheel, built from the tree as it
stands, and the pages, as the server answers them.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_a_built_wheel_carries_every_file_of_the_package(tmp_path):
    """Every file git tracks under `molbuilder/` is in the wheel built from
    it -- the scripts, styles, templates and data files an installed copy
    reads as well as its modules.

    Built from a copy of those files as they stand on disk, with the
    environment's own setuptools (`pip wheel --no-build-isolation`):
    nothing is installed, and the checkout is left as it was.

    MUTATION THIS MUST FAIL AGAINST: the static tree's pattern one level
    deep again, as the hand-kept list was."""
    import shutil
    import subprocess
    import sys
    import zipfile
    files = subprocess.run(["git", "ls-files", "molbuilder"], cwd=REPO,
                           capture_output=True, text=True,
                           check=True).stdout.split()
    src = tmp_path / "src"
    for rel in files + ["pyproject.toml", "README.md", "LICENSE"]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, src / rel)
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(src), "--no-deps",
         "--no-build-isolation", "-q", "-w", str(tmp_path / "dist")],
        capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr[-2000:]
    wheel = next((tmp_path / "dist").glob("molbuilder-*.whl"))
    shipped = set(zipfile.ZipFile(wheel).namelist())
    missing = [f for f in files if f not in shipped]
    assert not missing, (
        f"{len(missing)} of the package's {len(files)} files are not in the "
        "wheel -- an installed copy lacks them:\n  "
        + "\n  ".join(missing[:40]))


class _Assets(HTMLParser):
    """The assets a page names as the browser reads them: every script's
    ``src``, every stylesheet or icon ``href``, every image's ``src``."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        url = (a.get("src") if tag in ("script", "img")
               else a.get("href") if tag == "link" else None)
        if url and url.startswith("/") and not url.startswith("//"):
            self.urls.append(url)

    handle_startendtag = handle_starttag


def test_every_asset_a_served_page_names_answers(isolated_projects_root):
    """THE SILENT FAILURE: a page that names a file nobody wrote, or one moved
    -- a 404 in the browser's console and a feature that does nothing.

    Every page the app serves, parsed as the browser parses it, and each
    asset it names asked of the server.

    MUTATION THIS MUST FAIL AGAINST: a stylesheet a page links, gone."""
    from molbuilder.web.app import create_app
    from support.pages import every_served_page
    web = create_app(config={"rate_limit": {"enabled": False}}).test_client()
    named = {}
    for path, html in every_served_page().items():
        parsed = _Assets()
        parsed.feed(html)
        parsed.close()
        for url in parsed.urls:
            named.setdefault(url.split("?")[0], path)
    assert any(u.startswith("/static/") for u in named), sorted(named)
    dangling = {}
    for url, page in sorted(named.items()):
        status = web.get(url).status_code
        if status != 200:
            dangling[url] = f"{status}, named by {page}"
    assert not dangling, dangling
