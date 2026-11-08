"""Task Setup tab — the page and its doors.

`docs/web/task-setup.md` is the contract. This holds the page's parts and
its stylesheet layers as served, and the doors the page asks: the hand-over
route (`handover-procedure.md`), the save door (§ 8), the pickers and
presets (§ 5), the folder door (§ 2.1), the machine choice
(`preparing-for-another-machine.md` § 4) and the
chain from a hand-over to a rendered deck.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT   = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"


# --------------------------------------------------------------------- #
#  The page                                                             #
# --------------------------------------------------------------------- #


#: A DOCUMENT THAT ANSWERS AS THE SERVED PAGE DOES, for the page's script
#: run under node: an element asked for by its id is found when the page as
#: served holds that id (``__held``) or the script made one with it, and is
#: ``null`` otherwise, as a browser answers.  Each element holds what the
#: script puts in it -- its children, attributes, text -- and below it, one
#: element for any one selector, none for a list of them; a script or a
#: stylesheet put in the head loads at once, running nothing, and the editor
#: the page mounts (CodeMirror, vendored and loaded that way) holds each
#: record's text.  Nothing else of a browser is here: what this asks is
#: whether the script finds the elements it needs.
_THE_SERVED_DOCUMENT = r"""
globalThis.__made = new Map();
const __found = new Map();
class __El {
  constructor(tag) {
    this.tagName = String(tag || "").toUpperCase(); this.nodeType = 1;
    this.children = []; this.parentNode = null; this.attrs = {};
    this.style = { setProperty() {}, removeProperty() {} }; this.dataset = {};
    this._text = ""; this.hidden = false; this.disabled = false;
    this.value = ""; this.checked = false; this.open = false;
    this.className = ""; this.title = ""; this.type = "";
    const cls = new Set();
    this.classList = { add: (...c) => c.forEach((x) => cls.add(x)),
                       remove: (...c) => c.forEach((x) => cls.delete(x)),
                       toggle: (c, on) => ((on ?? !cls.has(c))
                                           ? (cls.add(c), true)
                                           : (cls.delete(c), false)),
                       contains: (c) => cls.has(c) };
  }
  get id() { return this.attrs.id || ""; }
  set id(v) { this.setAttribute("id", v); }
  setAttribute(k, v) { this.attrs[k] = String(v);
                       if (k === "id") __made.set(String(v), this); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; }
  removeAttribute(k) { delete this.attrs[k]; }
  toggleAttribute(k, on) { const set = on ?? !(k in this.attrs);
                           if (set) this.attrs[k] = ""; else delete this.attrs[k];
                           return set; }
  appendChild(c) {
    if (c && c.__fragment) { for (const k of [...c.children]) this.appendChild(k);
                             return c; }
    if (c.parentNode) c.parentNode.removeChild(c);
    this.children.push(c); c.parentNode = this;
    if (c.tagName === "SCRIPT" || c.tagName === "LINK") {
      setTimeout(() => c.onload && c.onload(), 0);
    }
    return c;
  }
  append(...cs) { for (const c of cs) this.appendChild(
      typeof c === "string" ? document.createTextNode(c) : c); }
  prepend(...cs) { const had = this.children; this.children = [];
                   this.append(...cs); this.children.push(...had); }
  insertBefore(c, ref) { this.appendChild(c); if (!ref) return c;
    this.children.pop(); const i = this.children.indexOf(ref);
    this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; }
  replaceChildren(...cs) { this.children = []; this._text = "";
                           this.append(...cs); }
  replaceWith(...cs) { const p = this.parentNode; if (!p) return;
    const i = p.children.indexOf(this); p.children.splice(i, 1);
    this.parentNode = null; for (const c of cs.reverse()) {
      p.children.splice(i, 0, c); c.parentNode = p; } }
  removeChild(c) { const i = this.children.indexOf(c);
                   if (i >= 0) this.children.splice(i, 1);
                   c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  cloneNode() { return new __El(this.tagName); }
  get firstChild() { return this.children[0] || null; }
  get lastChild() { return this.children[this.children.length - 1] || null; }
  get firstElementChild() { return this.firstChild; }
  get lastElementChild() { return this.lastChild; }
  get childNodes() { return this.children; }
  get childElementCount() { return this.children.length; }
  get parentElement() { return this.parentNode; }
  get nextSibling() { return null; }
  get previousSibling() { return null; }
  get nextElementSibling() { return null; }
  get previousElementSibling() { return null; }
  get isConnected() { return true; }
  get textContent() { return this._text + this.children.map(
      (c) => c.textContent || "").join(""); }
  set textContent(t) { this.children = []; this._text = String(t ?? ""); }
  get innerText() { return this.textContent; }
  set innerText(t) { this.textContent = t; }
  get innerHTML() { return ""; }
  set innerHTML(_v) { this.children = []; this._text = ""; }
  addEventListener() {} removeEventListener() {}
  dispatchEvent() { return true; }
  querySelector(sel) { this._q = this._q || new Map();
    if (!this._q.has(sel)) this._q.set(sel, new __El("div"));
    return this._q.get(sel); }
  querySelectorAll() { return []; }
  getElementsByTagName() { return []; }
  closest() { return null; } matches() { return false; }
  contains(o) { return o === this; }
  focus() {} blur() {} click() {} select() {} scrollIntoView() {}
  getBoundingClientRect() { return { top: 0, left: 0, right: 0, bottom: 0,
                                     width: 0, height: 0 }; }
}
class __Text { constructor(t) { this.textContent = String(t ?? "");
                                this.nodeType = 3; this.parentNode = null; } }
globalThis.document = {
  readyState: "complete",
  head: new __El("head"), body: new __El("body"),
  documentElement: new __El("html"), activeElement: null,
  getElementById(id) {
    if (__held.has(id)) {
      if (!__found.has(id)) { const e = new __El("div");
                              e.attrs.id = id; __found.set(id, e); }
      return __found.get(id);
    }
    return __made.get(id) || null;
  },
  createElement: (tag) => new __El(tag),
  createElementNS: (_ns, tag) => new __El(tag),
  createTextNode: (t) => new __Text(t),
  createDocumentFragment: () => Object.assign(new __El("#fragment"),
                                              { __fragment: true }),
  querySelector: () => null, querySelectorAll: () => [],
  getElementsByTagName: () => [],
  addEventListener() {}, removeEventListener() {},
  dispatchEvent() { return true; },
};
for (const name of ["addEventListener", "removeEventListener"]) {
  globalThis[name] = () => {};
}
globalThis.requestAnimationFrame = (f) => setTimeout(f, 0);
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
const __store = () => { const m = new Map(); return {
  getItem: (k) => (m.has(k) ? m.get(k) : null),
  setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) }; };
globalThis.sessionStorage = __store();
globalThis.localStorage = __store();
globalThis.location = new URL("http://127.0.0.1/task-setup");
globalThis.history = { replaceState() {}, pushState() {} };
for (const name of ["MutationObserver", "ResizeObserver",
                    "IntersectionObserver"]) {
  globalThis[name] = class { observe() {} disconnect() {} unobserve() {} };
}
const __doc = (text) => { let t = text; return {
  getValue: () => t, setValue: (v) => { t = String(v); },
  on() {}, off() {}, clearHistory() {}, getCursor: () => null,
  setCursor() {}, markClean() {}, isClean: () => true }; };
globalThis.CodeMirror = Object.assign(
  (_host, opts) => ({ ...__doc(opts && opts.value || ""), refresh() {},
                      swapDoc() {}, setOption() {}, focus() {},
                      getWrapperElement: () => new __El("div") }),
  { Doc: (text) => __doc(text), modes: {}, commands: {},
    defineOption() {}, defineExtension() {} });
"""


def test_the_page_script_reads_a_calculation_on_the_page_as_served(
        tmp_path, monkeypatch):
    """THE PAGE'S SCRIPT FINDS THE ELEMENTS IT NEEDS ON THE PAGE AS SERVED.
    `task-setup/viewer.js` is loaded under node as the page loads it --
    after the classic scripts the served page names, the sidebar's
    selection a calculation `molbuilder describe` wrote, each request
    answered by the app serving the tree (`_node_esm.served`) -- each
    element it asks for by id found when the served page holds it: the read
    ends with the calculation loaded and its machine card drawn, and no
    error.

    THE SILENT FAILURE: an id renamed in the template alone -- a card's
    renderer meets ``null`` when a calculation loads (the machine card's
    rows, measured 2.67.11: it throws, while the page boots clean).  The
    ids are the script's own, asked as it runs; this test holds no list of
    them."""
    import json
    import urllib.request

    from support.live_server import serve
    from support.pages import ids_of
    from support.road import described, road_tree
    from tests._node_esm import run_node, served
    calc = described(road_tree(tmp_path, monkeypatch))
    with serve() as base:
        with urllib.request.urlopen(f"{base}/task-setup") as r:
            page = r.read().decode()
        classic = re.findall(r'<script src="([^"]+)"(?![^>]*\bdefer\b)', page)
        got = run_node(
            [STATIC / src.split("/static/", 1)[1].split("?")[0]
             for src in classic],
            """
            const folder = %s;
            window.molbuilder.runtime.register("projects", {
              getCurrentDir: () => folder,
              onChange: () => {}, onCommit: () => {},
              onFolderChanged: () => {},
            });
            await import(%s);
            await __quiet();
            const $ = (id) => document.getElementById(id);
            console.log(JSON.stringify({
              state: $("ts-state").getAttribute("data-state"),
              said: $("ts-state-title").textContent,
              machine_card_shown: !$("ts-machine-card").hidden }));
            """ % (json.dumps(str(calc)),
                   json.dumps((STATIC / "task-setup" / "viewer.js").as_uri())),
            globals_js=("globalThis.__held = new Set(%s);\n"
                        % json.dumps(sorted(ids_of(page)))
                        + _THE_SERVED_DOCUMENT + served(base)),
            static_root=STATIC)
    assert got == {"state": "loaded", "said": "Loaded — H2",
                   "machine_card_shown": True}, got


def test_the_page_loads_the_shared_stylesheet_layers(web_client):
    """Layers 1-4 before layer 5, so tokens land first (`ui-contract.md` § 1)."""
    body = web_client.get("/task-setup").data.decode()
    order = [body.index(x) for x in (
        "lib/tokens.css", "lib/page-shell.css",
        "lib/form-components.css", "task-setup/style.css")]
    assert order == sorted(order), "stylesheet layers load out of order"


# --------------------------------------------------------------------- #
#  The hand-over (`stages.md` § 6.5a)                                    #
# --------------------------------------------------------------------- #

import json as _json

from support.road import fresh_calc_dir, save_both
from support.machines import SLURM, WORKSTATION


def _folder(client, d):
    """The folder answer -- the one door Task setup reads a folder through
    (`web/web-api.md`, `/api/task-setup/folder`)."""
    return client.get("/api/task-setup/folder?dir=" + str(d)).get_json()


def _envelope():
    """Through the ONE builder (`tests/support/envelope.py`), so a field the
    envelope grows lands here rather than being forgotten in a hand-listed
    copy."""
    from support.envelope import envelope
    return {"structure": envelope(["H", "H"],
                                  [[0, 0, 0], [0, 0, 0.74]])}


def _as_the_page_saves(over):
    """The description and the job request the page builds from the
    hand-over ``over`` and sends to Save (`task-setup/viewer.js`,
    `proposedFromHandover`): the hand-over's own fields, the ladder it
    proposes, what that ladder varies, each stage's proposed group, and the
    one layout it carries (`task-setup.md` § 4) -- every value the server's,
    none typed here: ``(task, request)``."""
    task = {"schema": over["task_schema"], "engine": over["engine"],
            "calculation": over["calculation"], "shape": over["task_shape"],
            "run": over["run"], "structure": over["structure"],
            "varies": over["proposed_varies"],
            "stages": over["proposed_stages"]}
    request = {"schema": over["job_request_schema"],
               "stages": {n: {"group": g}
                          for n, g in over["proposed_groups"].items()}}
    return task, request


def test_handover_renders_and_writes_nothing(web_client, isolated_projects_root):
    """The endpoint returns TEXTS; the browser writes them.

    `web/projects.md` § 1 puts raw bytes in the content-blind file layer that
    "every tab can use" — a tab that writes files itself bypasses the roots
    guard, the lock, the uniform envelope and the sidebar re-list.  What is
    genuinely server-side is the RENDER: only Python can turn a config into
    `<label>.template.toml`.
    """
    d = fresh_calc_dir(isolated_projects_root)
    r = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", calculation="optimization",
        name="probe calc", params={"system_label": "probe"}))
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out["ok"] is True
    assert out["template_text"].strip(), "no template rendered"
    assert out["handover_name"] == "task.1st.json"
    assert list(d.iterdir()) == [], (
        "the render endpoint wrote into the folder; the browser writes, "
        "through the file layer (projects.writeFile)")

    h = _json.loads(out["handover_text"])
    assert h["schema"] == "molbuilder/task-handover@1"
    assert "shape" not in h and "stages" not in h
    # THE ONE LAYOUT the description states, for the page to copy
    # (`task.LAYOUT`, `web/web-api.md`); the hand-over waits on its
    # stages alone (`engines/stages.md` § 6.5a).
    assert h["task_shape"] == "hierarchical"
    assert h["awaiting"] == ["stages"]
    assert h["_what"], "the file does not say what it is"



@pytest.mark.parametrize("length, taken", [(1, True), (100, False)],
                         ids=["inside-the-ladders-cap", "past-it"])
def test_the_handover_names_its_calculation_as_its_save_reads_it(
        web_client, isolated_projects_root, length, taken):
    """`run-identity.md` § 3: the label is the one rule's, its cap counting
    the ladder's own stage token once the ladder is known -- and the
    hand-over knows it (`task.label_of`): a name the description takes is
    taken, its label the one the save reads; one past the cap is refused by
    name, 400.  THE SILENT FAILURE: the hand-over capping a name by a
    guessed stage budget its save does not use, a name the description
    takes refused at the hand-over (until 2.65.3, a 500)."""
    from molbuilder.identity import MAX_LABEL_BYTES
    from molbuilder.jobset.ladder import fresh_stages
    from molbuilder.task import label_of
    fresh_calc_dir(isolated_projects_root)
    name = "a" * (MAX_LABEL_BYTES + length)
    stages = tuple(s.name for s in fresh_stages("optimization", "siesta"))
    r = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", calculation="optimization",
        name=name, params={"system_label": name}))
    if taken:
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["label"] == label_of(
            name, calculation="optimization", stage_names=stages)
    else:
        assert r.status_code == 400, r.get_json()
        assert r.get_json()["error"].startswith("name:"), r.get_json()

# THE PROPOSED LADDER RIDES THE HAND-OVER, numbered, made by the server's one
# rule (`ladder.fresh_stages`, `jobset.group.numbered_fresh`): a vibration's
# one stage, whichever way the box is ticked -- on SIESTA the relaxation is
# a phase of its run, not a stage (`engines/vibration.md` § 5.11 § 1) --
# the ordinary one-stage start otherwise (`engines/stages.md` § 6.5).  Until
# 2.15.3 the page derived it in JavaScript, a second copy of the rule.
@pytest.mark.parametrize("engine,calculation,params,ladder", [
    ("siesta", "vibration", {"system_label": "probe"}, [("freq", 1)]),
    ("siesta", "vibration", {"system_label": "probe", "already_relaxed": True},
     [("freq", 1)]),
    ("pyscf", "vibration", {"job_name": "probe"}, [("freq", 1)]),
    ("siesta", "optimization", {"system_label": "probe"}, [("coarse", 1)]),
], ids=["siesta vibration, the box unticked", "siesta vibration, ticked",
        "pyscf vibration", "an optimization"])
def test_the_handover_carries_the_proposed_ladder(web_client,
                                                  isolated_projects_root,
                                                  engine, calculation,
                                                  params, ladder):
    rendered = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine=engine, name="probe", calculation=calculation,
        params=params)).get_json()
    assert rendered["ok"], rendered
    over = _json.loads(rendered["handover_text"])
    assert [(s["name"], over["proposed_groups"][s["name"]])
            for s in over["proposed_stages"]] == ladder
    assert all(s["overrides"] == {} for s in over["proposed_stages"])


# --------------------------------------------------------------------- #
#  T2 save                                                               #
# --------------------------------------------------------------------- #


def test_save_writes_the_description_and_reports_the_handover(web_client, isolated_projects_root):
    """The save door owns `task.json` — the same reason `/api/structure/save`
    owns the sidecar: a browser-authored, schema-stamped file that the loader
    would reject is the save-then-reload trap `projects.md` § 3 describes.

    It does NOT delete the hand-over; it reports that one is there, and the
    browser removes it through `projects.deleteEntry`.
    """
    d = fresh_calc_dir(isolated_projects_root)
    rendered = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", calculation="optimization",
        name="probe", params={"system_label": "probe"})).get_json()
    # the browser's half, through the file layer
    (d / rendered["handover_name"]).write_text(rendered["handover_text"])
    proposed, request = _as_the_page_saves(
        _json.loads(rendered["handover_text"]))
    r = save_both(d, proposed, request, web_client)
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert (d / "task.json").is_file()
    assert (d / "job-request.json").is_file()
    assert out["stages"] == ["coarse"]
    assert out["handover_here"] is True, (
        "the save door should report the hand-over for the browser to remove")
    assert (d / "task.1st.json").is_file(), (
        "the save door deleted it; moving bytes is the file layer's job")


def test_save_never_writes_over_a_file_changed_since_it_was_read(
        web_client, isolated_projects_root):
    """`web/task-setup.md` § 8: Save compares what each record was when the
    page read it (the folder answer's `read_as`) with the file on disk --
    one changed since, by another tab's Save, a migrate or a hand edit, is
    refused, 409, nothing saved or written; a body with no `read_as` is
    refused, 400.  The 2.27.0 review: the later Save silently overwrote
    the other writer's change.

    MUTATIONS THIS MUST FAIL AGAINST: the comparison dropped (the second
    Save writes over the hand edit); `read_as` not required (a body naming
    none saves)."""
    import json
    from support.road import read_as
    d = fresh_calc_dir(isolated_projects_root)
    rendered = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", calculation="optimization",
        name="probe", params={"system_label": "probe"})).get_json()
    (d / rendered["handover_name"]).write_text(rendered["handover_text"])
    task, request = _as_the_page_saves(_json.loads(rendered["handover_text"]))
    assert save_both(d, task, request, web_client).status_code == 200
    seen = read_as(d, web_client)
    # ANOTHER WRITER, after the page read the folder:
    edited = dict(request, execution={"mpi_np": 4})
    (d / "job-request.json").write_text(json.dumps(edited))
    body = {"dest": str(d), "text": json.dumps(task),
            "request_text": json.dumps(request)}
    r = web_client.post("/api/task-setup/save", json=dict(body, read_as=seen))
    assert r.status_code == 409, r.get_json()
    assert "job-request.json changed on disk" in r.get_json()["error"]
    assert json.loads((d / "job-request.json").read_text()) == edited
    r = web_client.post("/api/task-setup/save", json=body)
    assert r.status_code == 400 and "read_as" in r.get_json()["error"]


def test_a_stage_stating_a_default_may_share_the_group_above(
        web_client, isolated_projects_root):
    """`job-request.md` § 3.3, § 4: a group's members share one job's
    launch, compared as the run takes it -- a setting no card states runs
    at the catalogue's default, so a stage stating that default (`use_gpu:
    false`) states what the stage above does, and its row may repeat the
    group number.  The 2.28.0 review: the cell said "states other values"
    and the numbering split two stages prep takes as one job.

    MUTATION THIS MUST FAIL AGAINST: the cards compared as stated, an
    unstated setting not its default."""
    d = fresh_calc_dir(isolated_projects_root)
    rendered = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", calculation="optimization",
        name="probe", params={"system_label": "probe"})).get_json()
    # WHAT THE HAND-OVER SENDS, written as the page's road writes it.
    for f in rendered["structure_files"]:
        (d / f["name"]).write_text(f["text"])
    (d / rendered["template_name"]).write_text(rendered["template_text"])
    (d / rendered["handover_name"]).write_text(rendered["handover_text"])
    task, request = _as_the_page_saves(_json.loads(rendered["handover_text"]))
    # A STAGE THE PERSON ADDS below the proposed one, in a group of its own,
    # starting clean and its card stating the default (`use_gpu: false`) --
    # as the page's ladder and run card write them, in one Save.
    task["stages"] = [*task["stages"],
                      {"name": "medium", "overrides": {"restart": "clean"}}]
    task["varies"] = sorted(set(task["varies"]) | {"restart"})
    request["stages"]["medium"] = {"group": 2,
                                   "execution": {"use_gpu": False}}
    assert save_both(d, task, request, web_client).status_code == 200
    rows = web_client.get("/api/task-setup/folder?dir=" + str(d)
                          ).get_json()["groups"]["stages"]
    assert [r["why_same"] for r in rows] == [None, None], rows


def test_save_refuses_rather_than_repairs(web_client, isolated_projects_root,
                                         tmp_path, monkeypatch):
    """The text goes through `task.from_text` — the same door `prep` uses — so
    a browser cannot become a second, drifting writer of descriptions."""
    d = fresh_calc_dir(isolated_projects_root)
    from support.road import described, read_as, road_tree
    r = web_client.post("/api/task-setup/save",
                        json={"dest": str(d), "text": "{ not json",
                              "request_text": "{}",
                              "read_as": read_as(d, web_client)})
    assert r.status_code == 400
    assert not (d / "task.json").exists(), "a bad description was written"

    # complete but for `stages`, so the STAGES refusal is what fires: the
    # description `molbuilder describe` wrote, its ladder taken out in
    # the page's editor, as a person may.
    calc = described(road_tree(tmp_path, monkeypatch))
    was = (calc / "task.json").read_bytes()
    no_stages = _json.loads(was)
    del no_stages["stages"]
    r2 = web_client.post("/api/task-setup/save", json={
        "dest": str(calc), "text": _json.dumps(no_stages),
        "request_text": (calc / "job-request.json").read_text(),
        "read_as": read_as(calc, web_client)})
    assert r2.status_code == 400
    assert "no 'stages'. A job has at least one stage" in \
        r2.get_json()["error"], "the refusal should be the reader's own words"
    assert (calc / "task.json").read_bytes() == was, (
        "a bad description was written")


# --------------------------------------------------------------------- #
#  The checkpoint API (F3) and the two guards (F1, F2)                   #
# --------------------------------------------------------------------- #


def test_the_column_picker_offers_no_run_setting(web_client):
    """A run setting is the rung's run card's, never a column (`stages.md`
    § 6.2, § 6.8d; plan § 5w K5): the machine's answers, and a person's --
    `use_gpu`, the solver.  `use_gpu` was a column until 2026-09-30, and a
    rung's `use_gpu` set as one reached its deck and not the scheduler's
    device ask (SO-C1).  `restart` says what a stage builds on -- the
    description's, a column (`job-request.md` § 2)."""
    from molbuilder.template import run_settings
    # The set is the catalogue's own answer; that it holds the settings this
    # rule is about is asserted, so an empty answer cannot pass vacuously.
    assert {"use_gpu", "diag_algorithm", "mpi_np"} <= run_settings("siesta")
    j = web_client.get("/api/task-setup/columns?engine=siesta"
                       "&calculation=optimization").get_json()
    names = [i["name"] for i in j["items"]]
    assert names, "the picker offered nothing at all"
    assert not (run_settings("siesta") & set(names)), sorted(
        run_settings("siesta") & set(names))
    # ...and a value that binds EVERY rung of the folder's kind is not a
    # column either (template.md § 6.4 `shared`: "no stage overrides it").
    # Measured 2026-09-24: a transport folder's picker offered thirteen of
    # them, `mesh_cutoff` and `basis_size` among them -- each a per-rung
    # override `prep` refuses by name.  A rung-owned item stays a column,
    # and the payload names its owners, so the table can disable the
    # other rungs' cells.
    from molbuilder.template import catalogue, select
    cat = catalogue()
    shared = {it.name for it in select(cat, engine="siesta", shared=True)
              if "transport" in it.shared}
    assert shared, "the fixture catalogue marks nothing shared"
    t = web_client.get("/api/task-setup/columns?engine=siesta"
                       "&calculation=transport").get_json()
    by = {i["name"]: i for i in t["items"]}
    assert not (set(by) & shared), sorted(set(by) & shared)
    assert by["transmission_n_points"]["stages"] == ["transmission"]
    # ...while an optimization's picker, whose kind shares nothing, keeps
    # the same rows as controls: `mesh_cutoff` is a legitimate per-stage
    # column of a relaxation ladder.
    assert "mesh_cutoff" in names


def test_only_execution_category_parameters_may_be_swept(web_client):
    """`stages.md § 6.8` — sweeping anything else means each point silently
    measures a DIFFERENT calculation."""
    j = web_client.get("/api/task-setup/sweepable?engine=siesta"
                       "&calculation=optimization").get_json()
    assert j["ok"] and j["items"]
    from molbuilder.template import load_catalogue, read_template, one
    t = read_template(load_catalogue())
    for item in j["items"]:
        it = one(t, item["name"])
        assert "execution" in it.category, (
            f"{item['name']} is offered for sweeping but is not `execution`")


def test_the_sweepable_list_says_which_the_machine_answers(web_client):
    """An allocation resolver means a description may never carry a value
    (`template.md § 6.4`), so those can only ever be measured."""
    j = web_client.get("/api/task-setup/sweepable?engine=siesta"
                       "&calculation=optimization").get_json()
    by = {i["name"]: i["machine_answers"] for i in j["items"]}
    for machine in ("mpi_np", "omp_threads", "max_memory_mb", "gpu_count"):
        assert by.get(machine) is True, f"{machine} not flagged machine-answered"
    assert by.get("use_gpu") is False, (
        "the GPU is a user decision, not something the machine answers "
        "(engines/overview.md § 3a: the user decides the GPU)")


@pytest.mark.parametrize("engine,table", [
    ("siesta", "molbuilder.config.siesta:SIESTA_STAGE_PRESETS"),
    # PySCF's menu served only the tiers the default strategy runs (two of
    # three) until 2.53.1.
    ("pyscf", "molbuilder.config.pyscf:PYSCF_STAGE_PRESETS"),
])
def test_the_presets_come_from_the_shipped_table(web_client, engine, table):
    """The same tiers the engine's shipped ladder is built from, so a stage
    filled here and stage N of that ladder cannot drift -- every tier,
    whatever a strategy runs.  `tuning.md § 4` is the authority for the
    numbers; this serves them, never restates them."""
    import importlib
    module, name = table.split(":")
    presets = getattr(importlib.import_module(module), name)
    j = web_client.get(f"/api/task-setup/presets?engine={engine}"
                       "&calculation=optimization").get_json()
    from molbuilder.config.stages import STAGE_NAMES
    assert j["ok"] and [ps["tier"] for ps in j["presets"]] == sorted(presets)
    for ps in j["presets"]:
        assert ps["name"] == STAGE_NAMES[ps["tier"]]
        assert ps["values"] == presets[ps["tier"]], (
            "the endpoint restates the tier values instead of serving them")


def test_the_presets_are_offered_per_kind(web_client):
    """A tier is offered only where every field it carries may be a column
    of the kind's table (`task-setup.md` § 9)."""
    # A transport rung owns none of the relaxation fields, so its rows get
    # no menu; a vibration's one stage owns them all.
    # Until 2026-09-24 every transport rung offered coarse/medium/tight
    # (plan W31, archived 2026-09-29), a menu whose every entry `prep` would refuse.
    t = web_client.get("/api/task-setup/presets?engine=siesta"
                       "&calculation=transport").get_json()
    assert t["ok"] and t["presets"] == [], t
    v = web_client.get("/api/task-setup/presets?engine=siesta"
                       "&calculation=vibration").get_json()
    assert v["ok"] and len(v["presets"]) == 3


def test_the_folder_template_is_what_an_empty_cell_names(web_client, isolated_projects_root):
    """`stages.md` § 6.2: a stage that sets nothing uses THE TEMPLATE'S value.

    So the whole point of the hand-over — the k-grid a person chose in the
    parameter tab — has to survive into what Task setup shows.  Before this
    endpoint the tab read the catalogue's default and named a number the job
    would not run whenever the sender had changed that parameter.
    """
    d = fresh_calc_dir(isolated_projects_root)
    rendered = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", calculation="optimization",
        name="probe",
        params={"system_label": "probe", "kgrid": [4, 4, 1],
                "mesh_cutoff": 450.0})).get_json()
    # WHAT THE BROWSER WRITES: the hand-over's files, as the route
    # returned them -- the template beside the hand-over that names it.
    (d / rendered["template_name"]).write_text(rendered["template_text"])
    (d / rendered["handover_name"]).write_text(rendered["handover_text"])

    j = _folder(web_client, d)["template"]
    assert j["ok"], j
    assert j["name"] == rendered["template_name"]
    assert j["values"]["mesh_cutoff"] == 450.0, (
        "the value the parameter tab collected did not reach Task setup; "
        "the catalogue default (300.0) would be shown instead")
    assert j["values"]["kgrid"] == [4, 4, 1], j["values"].get("kgrid")


def test_each_template_value_says_whose_it_is(web_client, isolated_projects_root):
    """`engines/template.md` § 6.6 obligation 2 (plan § 5w K7): the template
    a form hands over records whose each value is, and the folder door says
    it in the template's own words (`template.SOURCE_WORDS`).

    The form sends what it holds, a field nobody chose blank
    (`web/form-schema.md` § 1.1), and the one door reads a blank as not
    chosen: the field is written at THIS KIND's recommendation and recorded
    as nobody's.  Until K7 the form drew every default as a value and sent
    it, so no template could tell the person's 450 Ry from nobody's 300."""
    from molbuilder.template import one, read_template
    d = fresh_calc_dir(isolated_projects_root)
    rendered = web_client.post("/api/task-setup/handover", json=dict(
        _envelope(), engine="siesta", name="probe", calculation="vibration",
        params={"system_label": "probe", "mesh_cutoff": 450.0,
                "relax_force_tol": None, "basis_size": None})).get_json()
    assert rendered["ok"], rendered
    tmpl = read_template(rendered["template_text"])
    assert one(tmpl, "mesh_cutoff").source == "person"
    assert one(tmpl, "system_label").source == "person"
    # Not chosen: the vibration's own recommendation, held tighter than an
    # optimization's 0.02 (`template.md` § 6.3a), and nobody's.
    rft = one(tmpl, "relax_force_tol")
    assert (rft.value, rft.source) == (0.01, "default"), rft
    assert one(tmpl, "basis_size").source == "default"

    (d / rendered["template_name"]).write_text(rendered["template_text"])
    (d / rendered["handover_name"]).write_text(rendered["handover_text"])
    said = _folder(web_client, d)["template"]["said"]
    assert said["mesh_cutoff"] == "you set this", said["mesh_cutoff"]
    assert said["relax_force_tol"] == "not chosen", said["relax_force_tol"]

    # ...AND, ONCE DESCRIBED, EVERY VALUE SHARED BY EVERY STAGE IS ECHOED with
    # whose it is (`task-setup.md` § 3, obligation 3): no stage may change
    # one, so no column of the stage table shows it.  The calculation's
    # identity is one; a value a stage may set is none of them.  Saved as
    # the page saves the hand-over: its proposed ladder and groups, and the
    # one layout it carries (§ 4).
    s = save_both(d, *_as_the_page_saves(
        _json.loads(rendered["handover_text"])), web_client)
    assert s.status_code == 200, s.get_json()
    shared = {r["name"]: r for r in _folder(web_client, d)["template"]["shared"]}
    assert (shared["system_label"]["value"],
            shared["system_label"]["said"]) == ("probe", "you set this"), shared
    assert "mesh_cutoff" not in shared, sorted(shared)


def test_a_folder_with_no_template_is_not_an_error(web_client, isolated_projects_root):
    """An empty folder is an ordinary state, not a failure — the cells fall
    back to the catalogue, which is exactly right when nothing was sent."""
    d = fresh_calc_dir(isolated_projects_root)
    j = _folder(web_client, d)["template"]
    assert j["ok"] and j["name"] is None and j["values"] == {}


from conftest import write_pseudos as _pseudos_for


def _child_env_with_a_config(tmp_path):
    """Environment for a spawned `molbuilder`, with a machine config of its own.

    The tests below run the real CLI in a SUBPROCESS, so the child is given
    a root of its own, holding the one thing the render requires: a probed
    record, carrying the activation the generator reads
    (`configuration.md` § 4).  ``monkeypatch`` cannot reach across a
    process boundary, which is why the environment is built here rather than
    set on the parent.
    """
    import os
    root = tmp_path / "child-config-root"
    root.mkdir(parents=True, exist_ok=True)
    # THE CHILD'S MACHINE IS PROBED.  The child resolves its scope from the
    # env var below, not from this process's -- and prep refuses without a
    # record there (`running-a-job.md` § 3.1).
    from conftest import write_machine_record
    write_machine_record(at=root)
    env = dict(os.environ)
    env["MOLBUILDER_CONFIG_DIR"] = str(root)
    return env


def test_the_whole_chain_from_structure_to_rendered_deck(web_client, tmp_path, isolated_projects_root):
    """§ 7's bar, automated: structure -> hand-over -> description -> deck.

    **Every link can hold while the thing being carried is lost between them.**
    That is what the four shape-checks could not see, and why this compares the
    DECK's own lattice against the structure that started the chain rather than
    checking that each step returned `ok`.
    """
    import json as _json, subprocess, sys
    d = fresh_calc_dir(isolated_projects_root)
    # a periodic slab: a cell, a region label, a frozen atom
    cell = [[5.77, 0.0, 0.0], [2.885, 4.997, 0.0], [0.0, 0.0, 20.0]]
    # BUILT, not a two-atom envelope overwritten field by field: the
    # canonical dict carries the per-atom columns too, and replacing
    # `elements` alone left `atom_names` describing the old atoms.
    from support.envelope import envelope
    env = {"structure": envelope(
        ["Au", "Au", "S"],
        [[0, 0, 0], [2.885, 0, 0], [0, 0, 4.755]],
        regions={"frozen_atoms": [0], "slab": [0, 1]},
        cell=cell)}
    r = web_client.post("/api/task-setup/handover", json=dict(
        env, engine="siesta", calculation="optimization", name="chain",
        params={"system_label": "chain", "kgrid": [8, 8, 1],
                "kgrid_displacement": [0.5, 0.5, 0.0], "mesh_cutoff": 350.0}))
    assert r.status_code == 200, r.get_json()
    out = r.get_json()

    for f in out["structure_files"]:
        (d / f["name"]).write_text(f["text"])
    (d / out["template_name"]).write_text(out["template_text"])
    over = _json.loads(out["handover_text"])

    # The run card states its launch shape, as a run's must
    # (`architecture.md` § 5.2) -- the person's, on the page.
    described, request = _as_the_page_saves(over)
    request["execution"] = {"mpi_np": 1, "omp_threads": 1}
    s = save_both(d, described, request, web_client)
    assert s.status_code == 200, s.get_json()

    # the data files the engine will open -- prep refuses without them
    _pseudos_for(d, ["Au", "S"])

    # …and now the part no shape-check reaches: RENDER IT.
    p = subprocess.run(
        [sys.executable, "-m", "molbuilder.cli", "jobset", "prep", "task",
         "--group", "1", "--bundle", str(d)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
        env=_child_env_with_a_config(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "already under way" not in p.stdout, (
        "the hand-over's own files are being reported as engine leftovers:\n"
        + p.stdout)

    decks = sorted((d / "01_coarse").glob("*_01_coarse.fdf"))
    assert decks, sorted(x.name for x in d.iterdir())
    deck = decks[0].read_text()

    # THE LATTICE the k-grid indexes — the thing the hand-over used to drop
    lat = re.search(r"%block LatticeVectors(.*?)%endblock", deck, re.S)
    assert lat, "the deck has no cell — a periodic run became a molecule"
    rows = [[float(v) for v in ln.split()]
            for ln in lat.group(1).strip().splitlines()]
    assert rows == cell, f"deck lattice {rows} != source {cell}"

    kg = re.search(r"%block kgrid_Monkhorst_Pack(.*?)%endblock", deck, re.S)
    assert kg and "8 0 0 0.5" in " ".join(kg.group(1).split()), (
        kg and kg.group(1))
    assert "MeshCutoff 350.0 Ry" in deck

    # the ATOMS, row by row, against the pair the hand-over wrote: the same
    # atoms in the same order, placed by the one rule
    # (`model/structure-periodicity.md` § 6.0) -- the design coordinates
    # plus the engine offset -- and the deck's own record states that
    # offset.  "Nothing lost" means one translation for every atom, and
    # the one the deck says.
    import numpy as _np
    from molbuilder import cell as _cell
    from molbuilder.deck_record import extract_engine_offset
    from molbuilder.workingcopy_structure import StructureCodec
    blk = re.search(r"%block AtomicCoordinatesAndAtomicSpecies(.*?)%endblock",
                    deck, re.S)
    fdf = _np.array([[float(v) for v in ln.split()[:3]]
                     for ln in blk.group(1).strip().splitlines()])
    design = StructureCodec().load(d / over["structure"]["source"])
    assert fdf.shape == (3, 3)
    # Nobody assigned an origin, so the pair the hand-over wrote leaves the
    # placement to the rule (Automatic) rather than freezing a corner.
    assert design.engine_offset is None, design.engine_offset
    offset = _cell.engine_offset(design)
    assert _np.allclose(fdf, design.positions + offset, atol=1e-6), (
        fdf, design.positions, offset)
    record = extract_engine_offset(deck)
    assert record is not None, "the deck does not say where it put the atoms"
    assert _np.allclose(record["applied_offset"], offset, atol=1e-7), record
    assert record["stated"] is False, record


def test_a_dispersion_turned_off_on_the_form_is_off_in_the_deck(
        web_client, tmp_path, isolated_projects_root):
    """``dispersion = "none"`` from a form reaches the deck as no correction.

    "none" is the item's VALUE for no correction (`config/pyscf.py`'s note on
    the field).  Until 2026-09-28 the form's server side turned it into None,
    None is what an UNSET item reads as, the template wrote the item valueless
    and `prep` filled the class default: a person who chose "none" ran D3BJ.
    Every link returned ok -- so this follows the value through all of them,
    hand-over to template to `prep` to the deck `prep` renders.
    """
    import json as _json, subprocess, sys
    d = fresh_calc_dir(isolated_projects_root)
    from support.envelope import envelope
    env = {"structure": envelope(
        ["O", "H", "H"],
        [[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]])}
    r = web_client.post("/api/task-setup/handover", json=dict(
        env, engine="pyscf", calculation="optimization", name="nodisp",
        params={"method": "DFT", "dispersion": "none"}))
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    for f in out["structure_files"]:
        (d / f["name"]).write_text(f["text"])
    (d / out["template_name"]).write_text(out["template_text"])
    blk = out["template_text"].split("[item.dispersion]", 1)[1].split("help", 1)[0]
    assert 'value = "none"' in blk, blk

    described, request = _as_the_page_saves(_json.loads(out["handover_text"]))
    request["execution"] = {"threads": 1}
    s = save_both(d, described, request, web_client)
    assert s.status_code == 200, s.get_json()

    p = subprocess.run(
        [sys.executable, "-m", "molbuilder.cli", "jobset", "prep", "task",
         "--group", "1", "--bundle", str(d)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
        env=_child_env_with_a_config(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    decks = sorted((d / "01_coarse").glob("*_01_coarse.py"))
    assert decks, sorted(x.name for x in d.iterdir())
    deck = decks[0].read_text()
    assert 'mf.xc = "' in deck, "not a DFT deck -- the check below proves nothing"
    assert "mf.disp = " not in deck, [
        ln for ln in deck.splitlines() if "mf.disp" in ln]


def test_a_cpu_description_gets_a_cpu_benchmark(web_client, tmp_path, isolated_projects_root):
    """The machine half of § 7's bar, which the chain test above does not reach.

    **Where the grid comes from is settled and it is not the description.**
    `generator.md` § 4.3 — *a sweep and an allocation are both inputs to `prep`,
    never fields of the description* — and § 10's class 3 puts `mpi_np`,
    `omp_threads` and `max_memory_mb` at prep, *"never floor 2"*.  So this test
    does NOT write points into `task.json`; it states them in the job request,
    as the Measure card does (§ 4.3a), and asks `prep bench`.

    **What the description DOES answer is the GPU**, and that is equally
    settled: `web/task-setup.md` § 6.2, *"use GPU or not is set up only at the
    Job Prep UI"*.  `use_gpu` is a `staging` item carrying a real value, and
    it rides the template like any other.

    Until 2026-08-17 `_bench_inputs` pinned `use_gpu=True` and
    `diag_algorithm='ELPA-1STAGE'` flat, so **every trial measured a GPU
    whatever was asked for** — and on a machine whose probe finds no GPU the
    verb refused outright, which made a CPU benchmark impossible to run at all.
    That is the case here: an ordinary CPU description, benchmarked.

    Three claims:

    * it RUNS, and produces more than one trial;
    * no trial carries the GPU keyword the description declined;
    * every trial is separately labelled (`project-layout.md` § 7 invariant 5)
      — two trials sharing a SystemLabel warm-start off each other's `.DM` and
      the timings stop being comparable, which is the one thing a benchmark
      exists to produce.
    """
    import json as _json, subprocess, sys
    d = fresh_calc_dir(isolated_projects_root)
    env = _envelope()
    r = web_client.post("/api/task-setup/handover", json=dict(
        env, engine="siesta", calculation="optimization", name="grid",
        params={"system_label": "grid", "mesh_cutoff": 200.0,
                "use_gpu": False}))
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    for f in out["structure_files"]:
        (d / f["name"]).write_text(f["text"])
    (d / out["template_name"]).write_text(out["template_text"])
    described, request = _as_the_page_saves(
        _json.loads(out["handover_text"]))
    request["bench"] = {"mpi_np": [1, 2], "omp_threads": [1]}
    s = save_both(d, described, request, web_client)
    assert s.status_code == 200, s.get_json()
    _pseudos_for(d, ["H"])

    p = subprocess.run(
        [sys.executable, "-m", "molbuilder.cli", "jobset", "prep", "bench",
         "coarse", "--bundle", str(d)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
        env=_child_env_with_a_config(tmp_path))
    assert p.returncode == 0, (
        "a CPU description cannot be benchmarked:\n" + p.stdout + p.stderr)

    # EACH TRIAL'S RENDERED DECK, one per point of the sweep.
    decks = sorted(d.rglob("*.fdf"))
    assert len(decks) > 1, (
        "a benchmark is a set of points; got "
        + repr([str(x.relative_to(d)) for x in decks]))

    labels = []
    for deck in decks:
        text = deck.read_text()
        m = re.search(r"^SystemLabel\s+(\S+)", text, re.M)
        assert m, f"{deck.name} has no SystemLabel"
        labels.append(m.group(1))
        # The description said CPU.  A trial that turns the GPU on is
        # measuring a calculation nobody asked to run.
        g = re.search(r"^Diag\.ELPA\.GPU\s+(\S+)", text, re.M)
        assert not (g and g.group(1).lower().strip(".") == "true"), (
            f"{deck.name} enables the GPU against the description's "
            f"use_gpu = false -- the Job Prep UI's answer was "
            f"overridden by a pin (web/task-setup.md § 6.2)")
    assert len(set(labels)) == len(labels), (
        f"trials share a SystemLabel {labels} -- they will warm-start off "
        f"each other's .DM and the timings stop being comparable")

    js = sorted(d.rglob("job-set.json"))
    assert js, sorted(str(x.relative_to(d)) for x in d.rglob("*"))
    plan = _json.loads(js[0].read_text())
    assert plan["kind"] == "sweep", plan["kind"]
    assert len(plan["jobs"]) == len(decks), (
        [j.get("script") for j in plan["jobs"]])
    # A CPU sweep asks for no GPU.  `gres` set here would queue every
    # trial behind a GPU node it never uses.
    for j in plan["jobs"]:
        res = j.get("resources") or {}
        assert not (j.get("gres") or res.get("gres")), (
            f"a CPU trial asks for {j.get('gres') or res.get('gres')!r}")


def test_a_refused_cell_is_the_door_s_400_not_a_500(web_client):
    """`checked_periodicity` RAISES on a box it will not accept, and the app
    turns that into a 400 carrying the gate's own sentence.  The hand-over runs
    the same gate, so a refusal has to leave as the same answer — a 500 would
    show a stack trace where the reason belongs."""
    env = _envelope()
    env["structure"]["metadata"] = {
        "regions": {}, "cell": [[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]],
        "engine_offset": None, "axis_kind": None, "vacuum": None,
    }
    r = web_client.post("/api/task-setup/handover", json=dict(
        env, engine="siesta", calculation="optimization", name="bad",
        params={"system_label": "bad"}))
    # Two equal rows: a flat box, an ERROR finding (`cell.no_volume`), so the
    # gate refuses and the door answers 400 with that sentence.
    assert r.status_code == 400, (r.status_code, r.get_json())
    assert "flat" in ((r.get_json() or {}).get("error") or ""), (
        "the 400 does not carry the gate's reason", r.get_json())


# --------------------------------------------------------------------- #
#  The value SHAPE reaches the tab (user, 2026-08-20)                    #
# --------------------------------------------------------------------- #

def test_the_run_card_offers_only_what_the_kind_carries(web_client):
    """A kind's run card offers the run settings that kind carries
    (`template.md` § 6.3's sibling rule, as the columns are narrowed):
    TranSIESTA's solver is a transport calculation's, and an optimization's
    card does not carry it (the item's `calculations`; a value its decks
    would ignore)."""
    def names(kind):
        j = web_client.get("/api/task-setup/sweepable?engine=siesta"
                           f"&calculation={kind}").get_json()
        return {i["name"] for i in j["items"]}
    assert "ts_solution_method" in names("transport")
    assert "ts_solution_method" not in names("optimization")
    assert "ts_solution_method" not in names("vibration")
    assert "use_gpu" in names("vibration")


def test_both_pickers_payloads_carry_the_value_shape(web_client):
    """A bool or enum parameter edits through a dropdown of its legal
    values, and the tab can only build one by asking the catalogue -- so
    BOTH payloads carry `type`/`choices` (+ the sweepable's `default`,
    which births a row at its value in force).  Until 2026-08-20 the
    sweepable payload had no type at all, and every added setting was born
    as the number 1 -- `use_gpu` included."""
    sw = web_client.get("/api/task-setup/sweepable?engine=siesta"
                        "&calculation=optimization").get_json()
    items = {i["name"]: i for i in sw["items"]}
    assert items["use_gpu"]["type"] == "bool"
    assert items["diag_algorithm"]["type"] == "enum"
    assert items["diag_algorithm"]["choices"] == [
        "ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE"]
    assert items["diag_algorithm"]["default"] == "ELPA-1STAGE"

    from molbuilder.template import catalogue, one
    cols = web_client.get("/api/task-setup/columns?engine=siesta"
                          "&calculation=optimization").get_json()
    citems = {i["name"]: i for i in cols["items"]}
    assert citems["relax_type"]["type"] == "enum"
    assert citems["relax_type"]["choices"] == list(
        one(catalogue(), "relax_type", engine="siesta").choices)

    # THE KIND'S OWN DEFAULT, never the general one -- what the hover calls
    # *Recommended* (the M11 review's PS-C12: a vibration folder was told
    # an optimization's 0.02).
    vib = web_client.get("/api/task-setup/columns?engine=siesta"
                         "&calculation=vibration").get_json()
    vitems = {i["name"]: i for i in vib["items"]}
    assert vitems["relax_force_tol"]["default"] == 0.01
    assert citems["relax_force_tol"]["default"] == 0.02


def test_another_kinds_items_stay_out_of_the_optimization_surfaces(
        web_client):
    """`template.md` § 6.3's sibling rule at the two web doors
    (spectra-migration P0, 2026-08-20): the vibration items leaked into
    the Build form and the column picker the day their catalogue rows
    landed -- an item another calculation kind owns stays out by its own
    `calculations` declaration."""
    import json as _json
    schema = _json.dumps(
        web_client.get("/api/build/schema/pyscf"
                       "?calculation=optimization").get_json())
    cols = _json.dumps(
        web_client.get("/api/task-setup/columns?engine=pyscf"
                       "&calculation=optimization").get_json())
    for name in ("already_relaxed", "compute_raman", "es_mode_selection",
                 "displacement_amplitude_ang"):
        assert f'"{name}"' not in schema, f"{name} leaked into the form"
        assert f'"{name}"' not in cols, f"{name} leaked into the columns"
    assert '"basis"' in schema, "shared items must stay"


def test_save_runs_gate_three_and_refuses_a_failing_preflight(
        web_client, tmp_path, monkeypatch):
    """Gate ③ fires at save (G-1b, 2026-08-21): `workflow.md` § 9 names this
    door beside `describe` and dispatch, and until then only the codec ran
    here -- a stage override outside its field's bounds saved cleanly and
    failed at prep, on the cluster, hours later.  The refusal is the SAME
    function the CLI runs (`validation.task.preflight`), so the two
    surfaces answer alike.  Asked of a calculation `molbuilder describe`
    wrote, the override set on its first stage as the page sets one."""
    from support.road import (described, description_with_overrides,
                              read_request, road_tree)
    d = described(road_tree(tmp_path, monkeypatch))
    was = (d / "task.json").read_bytes()
    # A k-point count of 0 is past kgrid's hard limit
    # (`engines/template.md` § 5.3); the codec has no opinion about
    # values, so only the preflight can catch this.  (A value merely
    # outside a recommended range is warned, and saves.)
    bad_bounds = description_with_overrides(
        d, {"coarse": {"kgrid": [0, 4, 4]}})
    r = save_both(d, bad_bounds, read_request(d), web_client)
    assert r.status_code == 400, r.get_json()
    body = r.get_json()
    assert "preflight" in body["error"], body
    assert "kgrid" in body["error"], (
        "the refusal must name the failing field")
    assert (d / "task.json").read_bytes() == was, (
        "a description that fails its own preflight was written anyway")
    # The findings ride as data too, for the tab to render.
    assert any("kgrid" in f.get("message", "")
               for f in body.get("findings", []))


# --------------------------------------------------------------------- #
#  Which machine this is prepared FOR                                    #
# --------------------------------------------------------------------- #


class TestTheMachineChoiceIsAskedNotGuessed:
    """`preparing-for-another-machine.md` § 4: the choice is the user's
    whenever more than one machine could be meant.

    The tab does not carry its own rule for this -- the CLI refuses without
    a choice, and the tab asks the same question from the same records, so
    the two cannot come to different conclusions about what is ambiguous.
    """

    def test_the_route_lists_the_records_and_says_when_a_choice_is_required(
            self, web_client):
        """`choice_required` mirrors the CLI's C1 rule: any NAMED record
        makes 'this machine' ambiguous.  This box's record is the one the
        suite probes for every test (`conftest.write_machine_record`, a
        workstation's); the named one is written by the record's writer."""
        from molbuilder.config_dir import ensure_private_dir
        from molbuilder.scheduler import (Environment, environments_dir,
                                          named_environment_path,
                                          write_environment)
        r = web_client.get("/api/task-setup/machines").get_json()
        assert r["ok"] and r["choice_required"] is False, r
        assert [m["name"] for m in r["machines"]] == ["(this machine)"]

        ensure_private_dir(environments_dir())
        write_environment(Environment(scheduler=SLURM),
                          named_environment_path("sol"))
        r = web_client.get("/api/task-setup/machines").get_json()
        assert r["choice_required"] is True, r
        assert {m["name"] for m in r["machines"]} == {"sol", "(this machine)"}
        assert all(m["readable"] for m in r["machines"])

    def test_an_unreadable_record_is_listed_and_marked(self, web_client):
        """Shown rather than hidden: a record the user wrote and molbuilder
        cannot read is a thing to fix, and hiding it leaves them waiting.
        THE SILENT FAILURE: the record listed as readable, a machine offered
        that no prep can use.  Its text is typed: no writer of ours makes a
        record its reader refuses, and the refusal is what this asks."""
        from molbuilder.config_dir import ensure_private_dir
        from molbuilder.scheduler import environments_dir, named_environment_path
        ensure_private_dir(environments_dir())
        named_environment_path("sol").write_text('{"schema": "nope@9"}')
        r = web_client.get("/api/task-setup/machines").get_json()
        sol = next(m for m in r["machines"] if m["name"] == "sol")
        assert sol["readable"] is False
        assert "probe --write --name sol" in sol["summary"]


class TestTheTabShowsWhatAPrepWouldResolve:
    """`preparing-for-another-machine.md` § 5: the tab shows what `prep`
    resolved using the provenance `prep` already computes -- not a
    hand-written notice, which would be a second account of the same facts,
    free to drift from the one the terminal prints."""

    def _folder(self, tree, tmp_path, monkeypatch):
        import json as _json
        b = tree / "P" / "optimization" / "w"
        b.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        # THE SANDBOX IS THE CONFIG ROOT: without naming the directory the
        # write lands in a file nothing opens, and the test passes having
        # configured nothing.
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
        # ...and the box is probed: this moves the machine scope, and
        # prep refuses without a record (`running-a-job.md` § 3.1).
        from conftest import write_machine_record
        write_machine_record()
        # ...and a NAMED record beside it, so which machine a fresh folder is
        # for is not decided yet: the card shows no queues, and is not hidden
        # (W54 C10).
        from molbuilder.scheduler import (Environment,
                                          named_environment_path,
                                          write_environment)
        named = named_environment_path("sol")
        named.parent.mkdir(parents=True, exist_ok=True)
        write_environment(Environment(scheduler=WORKSTATION), named)
        (tmp_path / "molbuilder.json").write_text(_json.dumps(
            {"launch": {"mode": "direct"}}))
        return b

    def test_it_serves_the_same_facts_prep_prints(
            self, web_client, isolated_projects_root, tmp_path, monkeypatch):
        b = self._folder(isolated_projects_root, tmp_path, monkeypatch)
        r = _folder(web_client, b)["provenance"]
        assert r["ok"], r
        # the shape config_provenance produces, not a re-description of it --
        # this machine's molbuilder.json.  WHICH MACHINE RECORD answers is the
        # prep answer's, a preview's too: it depends on the machine a prep
        # names (`test_prep_from_the_browser`).
        assert {s["scope"] for s in r["sources"]} == {"machine"}
        assert "launch.mode" in r["effective"]
        assert "from" in r["effective"]["launch.mode"]


# --------------------------------------------------------------------- #
#  The TRANSPORT describe seam (engines/transport.md § 3.1, § 3.8)            #
#  the slot picker's describe seam                                      #
# --------------------------------------------------------------------- #

_T_CITE = "transport/optimization/Relax/01_only/run-0"


def test_the_handover_door_refuses_transport_by_name(web_client):
    r = web_client.post("/api/task-setup/handover", json=dict(
        engine="siesta", calculation="transport", name="T",
        junction=_T_CITE, bias=[0.0]))
    assert r.status_code == 400
    assert "describes on its own tab" in r.get_json()["error"]


def test_describe_attempt_stays_inside_the_tree(web_client):
    r = web_client.get("/api/transport/describe_attempt?path=../../etc")
    assert r.status_code == 400


# --------------------------------------------------------------------- #
#  The folder door -- Task setup's one per-directory answer              #
# --------------------------------------------------------------------- #


class TestTheFolderDoor:
    """`web/task-setup.md` § 2.1: *"the page holds no state of its own…
    the folder is the only link."*

    The page did not keep that.  It assembled from twelve endpoints and
    cleared the per-folder ones with `_resetPerFolderState()`, a
    hand-written list of eight clears in a file with twenty-five module
    variables -- so a six-trial bench plan rode into a calculation that
    declared none (2026-09-19).  This door answers the folder once.
    """

    @pytest.fixture
    def described(self, tmp_path, monkeypatch):
        """A described SIESTA calculation, as `molbuilder describe` leaves
        one (`support.road`), and the page's client."""
        from molbuilder.web.app import create_app
        from support.road import described, road_tree
        return (described(road_tree(tmp_path, monkeypatch)),
                create_app(config={}).test_client())


    def test_a_description_its_reader_refuses_is_said_on_the_page(self,
                                                                  described):
        """A `task.json` its reader refuses -- the road's own record, stating
        a major this build does not read: the reader's words reach the page,
        where one part of the answer forgot the description might not read and
        answered a bare 500 with nothing said."""
        from molbuilder.task import FILENAME as TASK_FILENAME
        dest, client = described
        path = dest / TASK_FILENAME
        record = _json.loads(path.read_text(encoding="utf-8"))
        record["schema"] = "molbuilder/task@3"
        path.write_text(_json.dumps(record), encoding="utf-8")
        r = client.get("/api/task-setup/folder",
                       query_string={"dir": str(dest)})
        assert r.status_code == 200, r.get_data(as_text=True)[:300]
        said = r.get_json()
        assert said["mode"] == "description"
        assert "molbuilder/task@3" in said["refused"]

    def test_the_answer_names_the_folder_it_is_about(self, described):
        """The half a reset list cannot cover.

        Two things make a page show the wrong folder: state that LINGERS,
        and an answer that LANDS LATE.  Clearing fixes only the first, and
        Task setup has no `AbortController` across any of its twelve calls
        -- so a response for the folder you just left can still arrive.
        `dir` is what lets a consumer that has moved on discard it, which
        is `calcdir.json`'s rule (§ 1.4a) applied to the wire.
        """
        dest, client = described
        one = client.get("/api/task-setup/folder?dir=" + str(dest)).get_json()
        assert one["dir"] == str(dest.resolve()), (
            "the answer must name its own subject; got " + repr(one["dir"]))

    def test_it_reports_the_mode_rather_than_the_page_keeping_one(
            self, described):
        """A save writes `task.json` and deletes the hand-over, so which is
        on disk IS the mode -- the page need not remember one."""
        dest, client = described
        from molbuilder.task import FILENAME as TASK_FILENAME
        assert client.get("/api/task-setup/folder?dir=" + str(dest)
                          ).get_json()["mode"] == "description"

        # ...THE HAND-OVER in its place, as its door renders one and the
        # page writes it.
        (dest / TASK_FILENAME).unlink()
        rendered = client.post("/api/task-setup/handover", json=dict(
            _envelope(), engine="siesta", calculation="optimization",
            name="probe", params={"system_label": "probe"})).get_json()
        assert rendered["ok"], rendered
        (dest / rendered["handover_name"]).write_text(
            rendered["handover_text"])
        body = client.get(
            "/api/task-setup/folder?dir=" + str(dest)).get_json()
        assert body["mode"] == "handover"
        assert body["description"] is None and body["handover"] is not None

        (dest / rendered["handover_name"]).unlink()
        assert client.get("/api/task-setup/folder?dir=" + str(dest)
                          ).get_json()["mode"] == "empty"

    def test_one_bad_part_does_not_cost_the_others(self, described):
        """A malformed template must not take the description with it."""
        dest, client = described
        tmpl = next(dest.glob("*.template.toml"))
        tmpl.write_text("this is not toml = = =\n")
        body = client.get(
            "/api/task-setup/folder?dir=" + str(dest)).get_json()
        assert body["ok"] is True, "the answer survives one bad part"
        assert body["template"]["ok"] is False, "and says which part failed"
        assert body["description"] is not None, (
            "the description is still there to read")



#: (calculation, the kind's entry the folder answer serves) -- what "What came
#: over" draws its rows by (`web/task-setup.md` § 3): a kind of a structure
#: cites nothing and carries none of a transport calculation's fields; a
#: transport calculation cites its junction, carries the rename, the bias
#: list and the calculation it reuses, and has two leads.
KIND_ENTRY_ROWS = [
    ("optimization", {"cites": None, "carries": [], "leads": []}),
    ("vibration", {"cites": None, "carries": [], "leads": []}),
    ("transport", {"cites": "junction",
                   "carries": ["swap_electrodes", "bias", "reuses"],
                   "leads": ["L-electrode", "R-electrode"]}),
]


@pytest.mark.parametrize("calculation, entry", KIND_ENTRY_ROWS,
                         ids=[r[0] for r in KIND_ENTRY_ROWS])
def test_the_folder_answer_states_its_kind_s_entry(tmp_path, monkeypatch,
                                                   calculation, entry):
    """Which rows "What came over" draws is its kind's entry, served in the
    folder answer -- the slot it cites, the fields it carries and its leads
    (`web/task-setup.md` § 3) -- so the page
    names no kind.  A kind of a structure is read through the road (its
    folder answer); a transport description needs a resolvable citation,
    which only an engine run makes, so its entry is asked of the answer's
    composer (`build._folder_kind`).

    THE SILENT FAILURE (until 2.65.0): the page compared the kind's name to
    "transport", so a second kind that cites would have been drawn as one
    that holds a structure.  The page's reading of the entry is the
    browser's (the F9 e2e)."""
    if entry["cites"] is not None:
        from molbuilder.web.blueprints.build import _folder_kind
        assert _folder_kind(calculation) == entry
        return
    from molbuilder.web.app import create_app
    from support.road import described, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch),
                       calculation=calculation)
    client = create_app(config={}).test_client()
    assert _folder(client, bundle)["kind"] == entry
