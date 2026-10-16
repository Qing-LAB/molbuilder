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

Both halves are asked of the artifacts: the wheel, built from what git
tracks as it stands on disk, and the pages, as the server answers them --
each asset they name looked for in that wheel.  And the third-party files
it carries are asked of it too: each ships with its notice.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The origin the pages are asked from (`support/pages.served_pages`).
ORIGIN = "https://localhost"


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    """The wheel built from every file git tracks under `molbuilder/`, as
    they stand on disk, with the environment's own setuptools (`pip wheel
    --no-build-isolation`): nothing is installed, and the checkout is left
    as it was -- ``(the tracked files, the wheel's names, the wheel)``."""
    import shutil
    import subprocess
    import sys
    import zipfile
    tmp = tmp_path_factory.mktemp("wheel")
    files = subprocess.run(["git", "ls-files", "molbuilder"], cwd=REPO,
                           capture_output=True, text=True,
                           check=True).stdout.split()
    src = tmp / "src"
    for rel in files + ["pyproject.toml", "README.md", "LICENSE"]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, src / rel)
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(src), "--no-deps",
         "--no-build-isolation", "-q", "-w", str(tmp / "dist")],
        capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr[-2000:]
    built_wheel = next((tmp / "dist").glob("molbuilder-*.whl"))
    return files, set(zipfile.ZipFile(built_wheel).namelist()), built_wheel


def test_a_built_wheel_carries_every_file_of_the_package(wheel):
    """Every file git tracks under `molbuilder/` is in the wheel built from
    it -- the scripts, styles, templates and data files an installed copy
    reads as well as its modules.

    MUTATION THIS MUST FAIL AGAINST: the static tree's pattern one level
    deep again, as the hand-kept list was."""
    files, shipped, _ = wheel
    missing = [f for f in files if f not in shipped]
    assert not missing, (
        f"{len(missing)} of the package's {len(files)} files are not in the "
        "wheel -- an installed copy lacks them:\n  "
        + "\n  ".join(missing[:40]))


class _Assets(HTMLParser):
    """The assets a page names as the browser reads them: every script's
    ``src``, every stylesheet or icon ``href``, every image's ``src`` --
    ``(tag, url)``."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.named = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        url = (a.get("src") if tag in ("script", "img")
               else a.get("href") if tag == "link" else None)
        if url:
            self.named.append((tag, url))

    handle_startendtag = handle_starttag


def test_every_asset_a_served_page_names_answers(isolated_projects_root,
                                                  wheel):
    """THE SILENT FAILURE: a page that names a file nobody wrote, or one moved
    -- a 404 in the browser's console and a feature that does nothing; one
    the checkout has and git does not, which the repo serves off disk and an
    installed copy lacks; or one named on another origin, which the
    browser's `default-src 'self'` refuses (`web-api.md` § 2 -- an image
    may also be a ``data:`` URL).

    Every page the app serves, parsed as the browser parses it; each asset
    it names, resolved against the page, asked of the server, and a
    ``/static/`` one looked for in the wheel.

    MUTATIONS THIS MUST FAIL AGAINST: a stylesheet a page links, gone; a
    script a page names that git does not track; a script from a CDN."""
    from molbuilder.web.app import create_app
    from support.pages import every_served_page
    _, shipped, _ = wheel
    web = create_app(config={"rate_limit": {"enabled": False}}).test_client()
    named, elsewhere = {}, {}
    for key, html in every_served_page().items():
        page = key.split()[0]                     # "<path> (signed in)"
        parsed = _Assets()
        parsed.feed(html)
        parsed.close()
        for tag, url in parsed.named:
            if tag == "img" and url.startswith("data:"):
                continue
            where = urlsplit(urljoin(ORIGIN + page, url))
            if f"{where.scheme}://{where.netloc}" != ORIGIN:
                elsewhere.setdefault(url, key)
            else:
                named.setdefault(where.path, key)
    assert not elsewhere, (
        f"assets named off molbuilder's own origin, which the CSP refuses: "
        f"{elsewhere}")
    assert any(u.startswith("/static/") for u in named), sorted(named)
    dangling = {}
    for url, page in sorted(named.items()):
        status = web.get(url, base_url=ORIGIN).status_code
        if status != 200:
            dangling[url] = f"{status}, named by {page}"
        elif (url.startswith("/static/")
              and f"molbuilder/web{url}" not in shipped):
            dangling[url] = f"not in the wheel, named by {page}"
    assert not dangling, dangling


#: The vendored browser assets, as the wheel holds them.
VENDOR = "molbuilder/web/static/vendor/"


def _inventory(readme: str):
    """The inventory table of the vendor directory's README, as the wheel
    carries it: ``(component, files cell, license cell)`` per row."""
    start = readme.index("## Inventory")
    rest = readme[start + 3:]
    section = rest[:rest.find("\n## ")] if "\n## " in rest else rest
    rows = []
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| Component") \
                or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[1], cells[-1]))
    return rows


def test_every_vendored_file_ships_with_its_notice(wheel):
    """THE SILENT FAILURE: a third-party file redistributed without its
    licence -- added to `static/vendor/` with no inventory row, or a row
    whose licence file is not shipped beside it (`static/vendor/README.md`,
    the inventory; the user, 2026-10-09: *"explicit citation and license for
    all embedded packages ... should have the right files and information
    included in the package/git file structure"*).

    Asked of the wheel, what ships: every vendored file in it is a file an
    inventory row names or a row's notice, and every row's licence file is
    in it.  Each row's release and citation are the inventory's, kept by
    review.

    MUTATIONS THIS MUST FAIL AGAINST: a vendored file with no row; a row's
    licence file not shipped."""
    import fnmatch
    import re
    import zipfile
    _, shipped, built = wheel
    readme = zipfile.ZipFile(built).read(VENDOR + "README.md").decode()
    rows = _inventory(readme)
    assert rows, "the shipped inventory has no rows"
    covered = []
    for _name, files, notice in rows:
        covered += [g for g in re.findall(r"`([^`]+)`", files)
                    if not g.startswith("/")]
        covered += re.findall(r"`([^`]+)`", notice)
        covered += re.findall(r"\]\(([^)]+)\)", notice)
    vendored = sorted(n[len(VENDOR):] for n in shipped
                      if n.startswith(VENDOR) and not n.endswith("/")
                      and n != VENDOR + "README.md")
    orphans = [f for f in vendored
               if not any(fnmatch.fnmatch(f, g) for g in covered)]
    assert not orphans, (
        f"shipped with no inventory row naming them: {orphans} -- add the "
        f"component's row, its license file and its citation to "
        f"static/vendor/README.md")
    unshipped = [(name, link) for name, _files, notice in rows
                 for link in re.findall(r"\]\(([^)]+)\)", notice)
                 if VENDOR + link not in shipped]
    assert not unshipped, f"licence files the wheel does not carry: {unshipped}"
