/* Projects tab — column-view server-side file explorer.
 *
 * Layout: horizontally-arranged columns (macOS Finder vibe).
 *   col[0]   roots
 *   col[1+]  entries of the directory chosen in the previous column
 *   col[end] when the chosen entry is a file: a metadata panel
 *
 * Selection model:
 *   - One *active* column = the one being navigated.
 *   - One *selected* entry within each non-final column = its "open
 *     subdirectory" (highlighted, shown in a later column).
 *   - At most one final-leaf selection at a time = the user's
 *     "current selection."
 *
 * Shared state (sessionStorage; cross-tab via the 'storage' event):
 *   molbuilder.current_dir   = absolute path of the deepest opened dir
 *   molbuilder.current_file  = absolute path of the selected file (or "")
 *
 * Other tabs subscribe to 'storage' and read these keys to update
 * their "Use current selection" UI.
 *
 * v1 scope: navigation + selection only.  No upload/rename/delete.
 */

(function () {
  "use strict";

  // ----- DOM refs ------------------------------------------------- //
  const elExplorer  = document.getElementById("explorer");
  const elStatus    = document.getElementById("status-path");
  const elActions   = document.getElementById("actions-bar");

  // ----- shared-state keys --------------------------------------- //
  const SS_DIR  = "molbuilder.current_dir";
  const SS_FILE = "molbuilder.current_file";

  // Which tabs can a file with the given extension be loaded into?
  // The "Open in <Tab>" buttons get rendered from this map.  Each
  // entry: {tab: <route>, label: <button-text>}.
  const OPEN_TARGETS = {
    ".xyz":           [{tab: "/modify",  label: "Open in Modify"},
                       {tab: "/",        label: "Open in Build"}],
    ".pdb":           [{tab: "/modify",  label: "Open in Modify"}],
    ".molwatch.log":  [{tab: "/watch",   label: "Open in Watch"}],
    ".spectra.json":  [{tab: "/spectra", label: "Open in Spectra"}],
    ".log":           [{tab: "/watch",   label: "Open in Watch"}],
  };

  // Pick the longest matching extension from the OPEN_TARGETS keys.
  // Order matters because .spectra.json is a more specific match
  // than .json.  Sort once at module load.
  const EXT_KEYS = Object.keys(OPEN_TARGETS).sort(
    (a, b) => b.length - a.length
  );

  function pickTargets(name) {
    const lower = name.toLowerCase();
    for (const ext of EXT_KEYS) {
      if (lower.endsWith(ext)) return OPEN_TARGETS[ext];
    }
    return [];
  }

  // ----- API helpers --------------------------------------------- //
  async function apiRoots() {
    const r = await fetch("/api/files/roots");
    return (await r.json()).roots || [];
  }

  async function apiList(path) {
    const r = await fetch(
      "/api/files/list?path=" + encodeURIComponent(path)
    );
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || "list failed");
    return j;
  }

  // ----- Shared-state writers ----------------------------------- //
  // sessionStorage.setItem doesn't fire the 'storage' event inside
  // the writing tab (only on OTHER tabs of the same origin), so we
  // also dispatch a CustomEvent locally for same-page subscribers.
  function setShared(dir, file) {
    sessionStorage.setItem(SS_DIR,  dir  || "");
    sessionStorage.setItem(SS_FILE, file || "");
    window.dispatchEvent(new CustomEvent("molbuilder.selection", {
      detail: {dir: dir || "", file: file || ""},
    }));
  }

  // ----- Column rendering --------------------------------------- //

  function clearColumnsFrom(idx) {
    while (elExplorer.children.length > idx) {
      elExplorer.removeChild(elExplorer.lastChild);
    }
  }

  function makeColumn(headerLabel) {
    const col = document.createElement("div");
    col.className = "col";
    const hdr = document.createElement("div");
    hdr.className = "col-header";
    hdr.textContent = headerLabel;
    hdr.title = headerLabel;
    const list = document.createElement("ul");
    list.className = "col-list";
    col.appendChild(hdr);
    col.appendChild(list);
    elExplorer.appendChild(col);
    return {col, list};
  }

  function renderEntry(list, entry, onClick) {
    const li = document.createElement("li");
    li.className = "entry " + (entry.kind === "directory" ? "is-dir" : "is-file");
    li.dataset.name = entry.name;
    li.dataset.kind = entry.kind;

    const icon = document.createElement("span");
    icon.className = "entry-icon";
    icon.textContent = entry.kind === "directory" ? "▸" : "·";
    li.appendChild(icon);

    const name = document.createElement("span");
    name.className = "entry-name";
    name.textContent = entry.name;
    li.appendChild(name);

    if (entry.kind === "directory") {
      const arr = document.createElement("span");
      arr.className = "entry-arrow";
      li.appendChild(arr);
    }

    li.addEventListener("click", () => onClick(li, entry));
    list.appendChild(li);
    return li;
  }

  function markSelected(list, li) {
    list.querySelectorAll(".entry.is-selected")
        .forEach((n) => n.classList.remove("is-selected"));
    if (li) li.classList.add("is-selected");
  }

  // ----- Roots column (always first) ----------------------------- //

  async function renderRoots() {
    elExplorer.innerHTML = "";
    const roots = await apiRoots();
    const {list} = makeColumn("Roots");
    if (roots.length === 0) {
      list.classList.add("is-empty");
      return;
    }
    for (const r of roots) {
      const entry = {
        name: r.label,
        kind: "directory",
        _path: r.path,
        _exists: r.exists,
      };
      const li = renderEntry(list, entry, async (li, e) => {
        markSelected(list, li);
        if (!e._exists) {
          // Show empty column for non-existent roots (e.g., projects/
          // before any project is created).
          clearColumnsFrom(1);
          const {list: emptyList} = makeColumn(e._path + " (missing)");
          emptyList.classList.add("is-empty");
          setShared(e._path, "");
          updateStatus();
          return;
        }
        await openDir(e._path, 1);
      });
      if (!r.exists) {
        li.style.opacity = "0.55";
        li.title = r.path + " (does not exist yet)";
      } else {
        li.title = r.path;
      }
    }
  }

  // ----- Open a directory into column[columnIdx] ----------------- //

  async function openDir(absPath, columnIdx) {
    clearColumnsFrom(columnIdx);
    let resp;
    try {
      resp = await apiList(absPath);
    } catch (e) {
      const {list} = makeColumn(absPath + " (error)");
      list.classList.add("is-empty");
      const li = document.createElement("li");
      li.style.padding = "0.5rem";
      li.style.color = "#a33";
      li.textContent = e.message;
      list.appendChild(li);
      setShared(absPath, "");
      updateStatus();
      return;
    }

    const {list} = makeColumn(resp.path);
    if (resp.entries.length === 0) {
      list.classList.add("is-empty");
      setShared(resp.path, "");
      updateStatus();
      return;
    }

    for (const e of resp.entries) {
      const fullPath = resp.path.replace(/\/$/, "") + "/" + e.name;
      renderEntry(list, e, async (li, _e) => {
        markSelected(list, li);
        if (e.kind === "directory") {
          await openDir(fullPath, columnIdx + 1);
        } else if (e.kind === "file") {
          clearColumnsFrom(columnIdx + 1);
          renderMetaCol(e, fullPath);
          setShared(resp.path, fullPath);
          updateStatus();
        } else {
          // symlink / other — show metadata only, don't try to descend.
          clearColumnsFrom(columnIdx + 1);
          renderMetaCol(e, fullPath);
          setShared(resp.path, fullPath);
          updateStatus();
        }
      });
    }
    // Default to the directory itself as the current selection
    // (file=""), so other tabs can see "user is browsing here."
    setShared(resp.path, "");
    updateStatus();
  }

  // ----- File metadata panel ------------------------------------- //

  function renderMetaCol(entry, fullPath) {
    const col = document.createElement("div");
    col.className = "col meta-col";
    const hdr = document.createElement("div");
    hdr.className = "col-header";
    hdr.textContent = entry.name;
    hdr.title = fullPath;
    col.appendChild(hdr);

    const body = document.createElement("div");
    body.className = "meta-body";

    const rows = [
      ["Path",  fullPath],
      ["Kind",  entry.kind],
      ["Size",  entry.size === null ? "—" : humanSize(entry.size)],
      ["Mtime", entry.mtime ? new Date(entry.mtime * 1000).toLocaleString() : "—"],
    ];
    for (const [k, v] of rows) {
      const row = document.createElement("div");
      row.className = "meta-row";
      const key = document.createElement("span");
      key.className = "meta-key";
      key.textContent = k;
      const val = document.createElement("span");
      val.className = "meta-val";
      val.textContent = v;
      row.appendChild(key);
      row.appendChild(val);
      body.appendChild(row);
    }
    col.appendChild(body);
    elExplorer.appendChild(col);
  }

  function humanSize(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MB";
    return (n / 1024 / 1024 / 1024).toFixed(1) + " GB";
  }

  // ----- Status line + open-in actions --------------------------- //

  function updateStatus() {
    const dir  = sessionStorage.getItem(SS_DIR)  || "";
    const file = sessionStorage.getItem(SS_FILE) || "";
    elStatus.textContent = file || dir || "(none)";
    // Build the "Open in <Tab>" buttons for the selected file (if any).
    elActions.innerHTML = "";
    if (!file) return;
    const targets = pickTargets(file.split("/").pop());
    if (targets.length === 0) {
      const span = document.createElement("span");
      span.className = "hint";
      span.style.cssText = "font-size: 0.8rem; color: #888;";
      span.textContent = "(no quick-open target for this file type)";
      elActions.appendChild(span);
      return;
    }
    for (const t of targets) {
      const btn = document.createElement("button");
      btn.textContent = t.label;
      btn.addEventListener("click", () => {
        window.location.href = t.tab;
      });
      elActions.appendChild(btn);
    }
  }

  // ----- Init ---------------------------------------------------- //

  renderRoots().then(updateStatus).catch((e) => {
    elExplorer.innerHTML = "";
    const div = document.createElement("div");
    div.style.cssText = "padding: 1rem; color: #a33;";
    div.textContent = "Failed to load roots: " + e.message;
    elExplorer.appendChild(div);
  });

  // If the user came here with a prior selection in sessionStorage,
  // the status line shows it immediately.  (Re-navigating to the
  // exact dir/file would require remembering the column path, which
  // we skip for v1.)
  updateStatus();
})();
