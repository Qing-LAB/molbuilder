# Workspace — saving your work so a reload or undo can bring it back

**Role:** contract
**Domain:** web
**Companions:** [`molview.md`](?doc=web/molview.md) — owns the molecule the
workspace saves; the workspace holds no data of its own. `web-api.md` — the
`/api/workspace-storage/*` server routes (web wave). The Modify tab's Save panel
and the projects file doors — a *file* save is their job, not the workspace's
(§ 8).

When you edit a molecule in a tab and then reload the page — or click Undo far
enough back — your work is still there. That is the **workspace** module: it
quietly saves the in-progress state of a tab so it can be brought back, without
you ever pressing "save". It has one job, and it never looks at *what* it is
saving — it just moves the bytes it is handed.

Two things up front:

- The workspace does **not** hold the molecule. The structure you see and edit
  lives in MolView ([`molview.data`](?doc=web/molview.md)); the workspace only
  saves and restores whatever bytes MolView hands it.
- It is a plain ES module — nothing here is waiting on a conversion.

## 1. The module

One file: **`dispatcher.js`** — the front door, `window.molbuilder.workspace`.
Every tab talks to this. It decides *where* saved bytes go and hands back what
was saved.

There was a second file that kept a copy in the browser's own short-term storage.
Nothing ever restored from it — every restore went to the server anyway — so it
cost a write on every edit and bought nothing. It is gone, and with it the only
other place your work could be.

The server side is three small routes — `/api/workspace-storage/{write,read,prune}`
in `workspace_storage.py` — described in § 9.

> **They were called `/api/state-timeline/*`, in `state_timeline.py`, and the name
> was wrong.** It named MolView's **timeline** — the sequence of saved points, the
> position on it, the badge, and the policy of what a save records and what to
> prune. None of that is on the server. That is `lib/molview/history.js`, and
> nothing in these routes knows a sequence exists: no order, no position, no
> notion that index 3 follows index 2. They store one opaque blob per
> `{workspace_id, state_index}` for **any** tag — a page's settings as well as
> a viewer's history. Renamed 2026-08-02 (§ 2a).

## 2. Where your work is kept

**One place: files on the server, under your project directory.**

```
<projects_root>/.molbuilder_workspace/states/<workspace_id>.<step>.wc.json
```

```mermaid
flowchart LR
    MV["MolView<br/>holds the molecule, decides when to save"]
    WS["workspace.persist"]
    F["one file per step, on the server<br/>under the project directory"]
    MV -->|"here are the bytes, and my tag"| WS
    WS -->|"sent, not waited for"| F
    F -.->|"reload the tab, or Retract"| MV
```

Two kinds of file, both here:

- **A numbered point for each save.** That is what Undo walks back through, and
  it survives a reload. The server keeps the most recent **30** per browser tab
  and tag, and drops older ones on its own.
- **A draft**, one file, rewritten every time you change something. That is what
  makes a reload of the tab find the work you never got round to saving.

They are separate files on purpose: an edit rewriting the point you last saved
would mean Retract took you back to something that had changed since you chose
it.

When MolView decides something is worth keeping, it calls:

```js
window.molbuilder.workspace.persist(tag, bytes, step);   // a numbered point
window.molbuilder.workspace.persistDraft(tag, bytes);    // the draft
```

**The caller names its tag and, for a point, its step — never a file.** The
workspace names the file from them (§ 9), so no saver can write into another's.

**It does not wait.** The write is sent and the call returns, so a slow disk
never freezes the editor — which means "it worked" cannot come back from here. A
failure turns up afterwards through `onPersistError`. The workspace never looks
inside the bytes; that is MolView's business.

## 2a. What the workspace is not

Jobs this module could plausibly grow into, and does not. Each boundary is here
because something has already drifted across it, or has been read as though it
had.

| Not | Whose job it is | Why the boundary is there |
|---|---|---|
| the **timeline** — the sequence of saved points, where you are on it, what a save records, what to prune, how far a step goes | **MolView**, in `lib/molview/history.js` (molview.md § 11.2) | The workspace stores numbered *states*; the **timeline is the sequence MolView makes of them**. The workspace has no idea one exists — it never compares two indices, and index 3 means nothing more to it than "a different file from index 2". A server that knew the order would be a second opinion about a history it cannot see the contents of |
| a **reader** of what it stores | nobody — the bytes are opaque | It moves bytes and never parses them. That is what lets a trajectory become restorable with no change here (molview.md § 11.2), and it is why the same routes serve a molecule, a history point and a page's settings without knowing the difference |
| a **file** save | the Modify tab's Save panel + the projects doors (§ 8) | Saving to a project is a decision a user makes about a name and a place. The workspace saves what you did not ask it to save |
| the **owner** of where `projects/` is | the app, through `Capabilities` | The root is resolved once for the whole app. Answering it separately here is exactly how the storage came to write somewhere the file picker was not looking |

**The first row is the one that has actually cost time.** The server half was
named `state_timeline.py`, serving `/api/state-timeline/*`, after a concept that
lives entirely in the browser and entirely inside MolView. Reading the name
instead of the callers led, on 2026-08-02, to two opposite wrong conclusions
about the same file within an hour — that it was a shared domain module to be
promoted out of the web layer, and that it was MolView-private plumbing to be
hidden. It is neither: it is this module's storage, and this module is public.

---

## 3. A worked example — reload the tab, undo

Say you open a molecule in the Modify tab, delete an atom, then move another:

1. Each edit rewrites **the draft** — one file, replaced each time — and lays
   down a numbered point of its own (`molview.md` § 1.1), dropping any points
   above it: two edits, two points.
2. You pick out a few atoms and press **Save state**: a point for a state no
   edit produced.
3. You **reload the tab**, or go to another page and come back in the same
   tab. The tab reads its draft back and your edited molecule is there, with
   the atoms you picked out. **Closing the tab ends its set**: a new tab starts
   with none, and the closed tab's files stay in the folder, unread (§ 4, § 9).
4. You click **Retract**. Each press steps back one point, read back from its
   own numbered file — the selection, then the move, then the delete. Work no
   point holds yet (the badge says when) is what the first press throws away.
5. If a write ever fails — disk full, network drop — the editor does not freeze.
   The failure is reported through `onPersistError` and a page event, never
   swallowed. A read that fails is reported the same way: the tab opens without
   what it could not read, and says so.

## 4. The tag — how several savers share one page

A page can have more than one thing worth keeping. The Results tab shows a
structure inspector and a trajectory inspector side by side. The Modify tab has a
viewer holding a molecule *and* its own panel state. All of them want to survive a
reload, and none of them should be able to tread on another.

**The tag is what separates them.** Every save and every load names who it belongs
to — `"results:structure"`, `"results:trajectory"`, `"modify:panel"`, a
viewer's `"<owner>:ui"` view context (molview.md § 11.2b) — and the
workspace names the files from that name and the browser tab (below). Two tags
are two slots. Nothing
is merged, so there is no moment where one writer's copy sits half over
another's.

**Each browser tab has its own files** *(the user, 2026-11-11: "same module,
one set per tab")*: the workspace id is the browser tab's own id and the tag —
`ws-<tab>-<tag>` — so two tabs on the same page never share a file (but a
tab the browser **duplicates** starts with the original's id, and shares its
files), and
within a tab the tag is all that separates two savers: isolation between pages
rests on the owner literals staying distinct. Picking a new owner string means
checking it against the ones in use. **A tab's id** is made once, twelve
lowercase letters and digits, and kept in that tab (`pagePrefs`,
[`runtime.md`](?doc=web/runtime.md) § 3), so a reload of the tab finds its
files and no other tab ever does; a browser that refuses to keep it makes a
new one per load, and the work is kept for the page but not across its reload.
**Two different tags are never one id**: the tag is spelled for a file name,
one spelling per character and none shared — a letter or a digit as it is, a
`:` as `-`, a `-` as `_d`, a `_` as `_u`, any other character as `_x` and its
six hex digits. **A tag never ends in `:draft`** — refused, as a missing tag
is: a tag's draft is its id with `-draft`, which `:draft` would spell the
same.

**Two things the workspace promises:**

1. **There is one way to save and one way to load — the calls in § 5.** If you
   find another way to write these bytes, that is a bug to report, not a
   shortcut to use. Bytes written any other way carry no tag, and the tag is the
   only thing keeping two savers off each other.
2. **What you save under your tag stays yours.** Another tag's save cannot
   change it, hide it or delete it — whichever of you saved first. And when you
   read, you get back what *you* wrote and never what somebody else did. That
   holds because two tags are two different places, not because callers are
   careful.

**The tag goes in the call. Nothing is switched on beforehand.** That is the whole
of the difference, and it is worth spelling out what the other way does:

> You call `useNamespace("modify")` once. That sets one variable inside the
> workspace. From then on every save — from anyone on the page — goes into
> whatever slot that variable last named, because the save itself says nothing
> about who it belongs to. So the viewer sets it to its own name, the panel later
> sets it to *its* own name, and the viewer's next save lands in the panel's slot.
> Nobody passed a wrong value; the variable changed under them. And the two only
> find out later, when one of them reads back something it never wrote.

Passing the tag with each call removes the window entirely: there is nothing to
set first, so there is nothing to change underneath anyone.

> **This is how the code works now** (changed 2026-08-02). Until then the tag was
> that one variable, `persist` took no tag at all, and the browser-storage half
> was published as a second door anyone could write through. All three are gone,
> and `tests/test_workspace_tag_isolation.py` fails if a call stops taking its
> tag.

## 5. The public surface

Everything a consumer uses is on `window.molbuilder.workspace`:

**Every call names its tag** (§ 4), so nothing has to be set up beforehand and no
two callers can disagree about whose slot they are in.

| Member | What it does |
|---|---|
| `persist(tag, bytes, step)` | Send these bytes as the tag's numbered point `step`. Does not wait; `true` means sent, not saved. |
| `readState(tag, step)` | Read the tag's point `step` back. `null` for a file that is not there **and** for any failure, deliberately: the only safe reading of it is "nothing to put back". A failure is still said — through `onPersistError`, as a read — never passed off as nothing saved. |
| `pruneStatesAbove(tag, step)` | Drop the tag's points above `step` (used when Undo-then-a-new-edit throws away the redo tail); `-1` drops every point. The draft stays. |
| `persistDraft(tag, bytes)` | Send these bytes as the tag's draft — its one file, rewritten each time. Does not wait, as `persist`. |
| `readDraft(tag)` | Read the tag's draft back; `null` as `readState`. |
| `onPersistError(fn)` | Be told when the store fails — a background save, or a read back — never silent. Returns an unsubscribe. |

**No call takes a file name or an id.** The tag, and for a point its step, are
all a caller says; where the bytes go is worked out here, from them and the
browser tab's own id.

**This is the whole of it.** There is no second way to save or load (§ 4's
first guarantee).

## 6. What the workspace does, and what you have to do

The workspace stores bytes and gives them back. That is all of it. It cannot
read them, so everything that needs to *understand* what was saved is yours.

**What it does**

- Puts your bytes in a file on the server — the tag's draft, or one of its
  numbered points — sent without waiting.
- Gives them back when you ask, by tag, and by step number for a point.
- Keeps one tag's bytes away from another's.
- Tells you when a save to the server failed.

**What it never does**

- Open your bytes. It cannot tell a molecule from a shopping list.
- Decide when you should save.
- Decide what a "step" means.

There is no `workspace.getAtoms()` and there never will be. Two functions once
broke that rule — they opened your saved bytes to answer "is there a molecule in
here?" and "which file did it come from?" — and one of them got it wrong and wiped
a molecule someone had just built (§ 7). Both are gone.

### What you have to do

**Say your tag on every call.** No tag is remembered between calls. Use the same
word every time for the same thing: the viewer on the Modify tab always says
`"modify"`, the structure inspector on Results always says `"results:structure"`.
Miss it and the call fails; use a different word by accident and you are reading
an empty slot while your work sits in the other one.

**Decide what goes in the bytes — and be able to read them back.** The workspace
hands you exactly what you gave it. If you later want to know "was there a
molecule in there?", open it and look. Nothing else can answer that for you.

**Decide when to save.** Nothing is written on a timer and nothing is written
because something changed. You call `persist` when something happened that you
do not want to lose.

**Number your own steps.** You say which number a saved point is. If you go back
three steps and then start working again, the points above are now abandoned and
they are yours to delete — call `pruneStatesAbove`.

**Know that the save is not finished when the call returns.** `persist` and
`persistDraft` say only that the bytes were sent. Whether they reached the
server is not known yet — that is the point, so a slow disk never freezes your
editor. If it fails you find out afterwards, through `onPersistError`. If your
user should be told, subscribe to it.

**Do not go round the front door.** If you find some other way to write these
bytes, report it — bytes written any other way carry no tag, and then two savers
can land on each other.

## 7. Bringing work back when a page opens

You open the Molbuilder tab. Two things can be true at the same time:

- a file is highlighted in the sidebar, because you clicked it before you left;
- you were working on a molecule in this tab before a reload and never saved it
  to a file.

The tab has to show one of them. Load the highlighted file and that work is
gone; bring back that work and the highlighted file is ignored.

**The tab decides, and it decides by reading its own saved bytes.** It reads
its draft back (`readDraft(tag)` — or, for a
MolView-held structure, `load(0)`), looks at what comes back, and asks itself whether
there is work in there worth keeping. If there is, that wins and the highlighted
file is left alone — opening a file is something you do on purpose, by clicking
Load, not something that happens because a name was still highlighted.

**Do not decide by comparing file names.** A molecule you built from SMILES, or
from a peptide sequence, was never in a file — so "which file did this come
from?" answers *nothing* for it, and a tab that reads that as "there is no saved
work" wipes the molecule you just built. That happened. Ask whether there is a
structure in your saved bytes, not where it came from.

> The workspace used to answer both of those questions itself, by opening the
> saved bytes and hunting for a molecule inside them. That is how the wipe
> happened: it was reading data it does not understand. Both questions are yours
> now — you wrote the bytes, so you know what is in them.

## 8. A file save is not the workspace's job

Saving your work-in-progress (everything above) is a different thing from
**saving a file into a project.** When you click Save-to-project, that runs
through the Modify tab's Save panel (`modify/structure/save.js`) to the projects
file door (`projects.molviewFiles.save` → `POST /api/structure/save`), which writes
the `.xyz` + `.molstruct.json` pair. The workspace is **not** in that path at
all. (The details live in the Modify-tab and projects docs.)

## 9. The state files on the server, and two ordering rules

For the record: the server keeps each tab's history as small JSON files at
`<projects_root>/.molbuilder_workspace/states/<workspace_id>.<step>.wc.json`. It
never parses them as molecules — it just moves JSON bytes. A closed browser
tab, or a tag no longer used, leaves its files behind, since nothing asks for
its id again; these are **not** auto-cleaned — crash recovery is out of scope by design — but once the pile
passes 300 files the server logs a one-time warning so an operator can clear the
folder by hand.

### 9.1 A session's history is bounded — the last 30 saves

**A saved sequence has a floor, and it is 30 points deep.** On every write the
server drops any step older than the newest 30, so the history is a rolling
window rather than an unbounded pile.

This is a **contract term, not a capacity accident**. Saved state is a
convenience for stepping back through recent work, and a convenience that grows
without limit stops being one: an afternoon of editing would otherwise leave
hundreds of full structure snapshots per tab on disk, forever, for a depth
nobody reaches. Thirty is the boundary that keeps "you can step back" honest and
finite.

What it means where you can feel it:

- **Retract reaches back 30 saves, and no further.** At the floor there is
  nothing older to return to. That is the designed answer, and the viewer says
  so rather than pretending it moved (§ 9.2).
- **Your unsaved work is never at risk.** The draft is a *different* workspace —
  `<id>-draft`, a single file pinned at step 0 (`persistDraft`) — so a
  one-entry history has nothing to trim. The window bounds *how far back you can
  step*, never *whether your latest work survives*.
- **The bound is per browser tab and tag.** Each has its own id and therefore
  its own thirty; two savers on a page, or two tabs, do not share the
  allowance.

### 9.2 Who prunes which end

Both ends of a sequence get trimmed, by different owners, for different reasons.
Stating it once here because the two are easy to read as one rule:

| End | Who | When | Why |
| --- | --- | --- | --- |
| the **future** — everything above the current point | **MolView** | on `save(step)`, via `pruneStatesAbove` | a save on top of a retracted point makes the abandoned branch meaningless (`molview.md` § 11.2) |
| the **past** — everything below the newest 30 | **the workspace** | on every write | the bound above |

So MolView owns *what a save means and how far a step goes*; the workspace owns
*where the bytes live and how many of them there are*. MolView decides where you
are in the sequence; it does not decide how long the sequence may be.

Because the floor moves without telling the client, **MolView's idea of the
sequence can outrun what is on disk** — its position and highest-point live in
memory. `load` already answers this honestly: it returns `null` when the point
asked for is not there, which the caller must report as *there is nothing
further back* rather than as a move. "Retracted to state #45" when nothing moved
is worse than silence.

> *(Task #47 landed: `retractState` reads `load`'s answer and reports "there
> is nothing further back" at the floor instead of a move that did not
> happen.)*

**An id belongs to a browser tab and a tag, worked out from them.** The id the
state files are named after (`<workspace_id>.<step>.wc.json`) is `"ws-"`, the
tab's own id and the tag, spelled for a file name (§ 4) — the same every time
the tag comes back in that tab, so a reload finds its work, and never another
tab's (a duplicated tab excepted, § 4) or another tag's, since no caller names
a file. **Isolation holds for the
identity as well as for the content**, by construction.

Two small ordering rules keep the history consistent:

- **Same-step saves stay in order.** Rapid save → undo → save can send two
  writes to the *same* numbered file; the front door chains them so a stale one
  can't land last.
- **Clear the tail before re-anchoring.** When Undo-then-edit throws away the
  redo tail, that delete (`pruneStatesAbove`) finishes before the new baseline
  snapshot is written, so the history can't briefly contain both.

Both rules are about one workspace id's files, so **each id has its own
queue**: a tag's draft, an id of its own, never waits behind its points'
writes or their delete — an edit's draft goes out at once, so a reload right
after the edit finds it.

## 10. Test map

- `test_workspace_tag_isolation.py` — against the real storage routes
  (`served`), read back through the calls of § 5: a missing tag and a `:draft`
  one refused (one example of each clause of the one rule) + two tags stay two
  slots, their points and their drafts; each browser tab its own (the same tab
  reloaded finds its work, another finds none and changes none); the two
  ordering rules of § 9.2, each with a request held — the later of two writes
  to one file, a point's or the draft's, is what stays; the tail delete
  finishes before the next point is written — and an edit's draft never waits
  behind the delete; MolView's
  save and edit, its molecule loaded by the real load route.
- `test_workspace_storage_api.py` — the server `/api/workspace-storage/*` routes.
- `test_molview_model.py` / `test_molview_stores_history.py` — restore-at-mount
  through the real model (`load(0)` adopts; a restored session still receives
  frames).

*(Four rows here used to name files that do not exist —
`test_workspace_dispatcher_js.py` and kin.  A test map that names phantom
files fails the reader exactly when they need it: corrected 2026-08-19.)*
