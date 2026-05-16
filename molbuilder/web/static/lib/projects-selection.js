/* projects-selection.js -- shared shim for "use current Projects-tab
 * selection" UI in any tab.
 *
 * Each tab calls:
 *
 *     molbuilderProjectsSelection.init({
 *         bannerEl: document.getElementById("projects-banner"),
 *         extensions: [".spectra.json"],     // optional: which file types this tab handles
 *         onLoad: async function (path, text) { ... },
 *     });
 *
 * The shim:
 *   - reads sessionStorage.molbuilder.current_file at startup
 *   - subscribes to the cross-tab 'storage' event AND the same-tab
 *     'molbuilder.selection' CustomEvent the Projects tab dispatches
 *   - shows / hides the banner based on whether the current
 *     selection matches the extensions filter
 *   - on click of the banner's "Use" button: fetches the file via
 *     /api/files/read and invokes onLoad(path, text)
 *
 * The bannerEl must contain elements with these classes:
 *   .ps-path     -- updated with the file path
 *   .ps-use-btn  -- the "Use" button
 *   .ps-clear    -- (optional) a button to clear the selection
 *
 * Keeping all the DOM wiring on the caller (rather than injecting a
 * fixed widget) lets each tab fit the banner into its own layout
 * without fighting CSS.
 */

(function (root) {
  "use strict";

  const SS_DIR  = "molbuilder.current_dir";
  const SS_FILE = "molbuilder.current_file";

  function nameOf(path) {
    if (!path) return "";
    const parts = path.split("/");
    return parts[parts.length - 1];
  }

  function endsWithAny(name, exts) {
    if (!exts || exts.length === 0) return true;
    const lower = name.toLowerCase();
    return exts.some((e) => lower.endsWith(e.toLowerCase()));
  }

  function init(opts) {
    const banner = opts.bannerEl;
    if (!banner) {
      console.warn("projects-selection: bannerEl required");
      return;
    }
    const extensions = opts.extensions || [];
    const onLoad     = opts.onLoad     || function () {};

    const pathSpan = banner.querySelector(".ps-path");
    const useBtn   = banner.querySelector(".ps-use-btn");
    const clearBtn = banner.querySelector(".ps-clear");

    function refresh() {
      const file = sessionStorage.getItem(SS_FILE) || "";
      if (!file || !endsWithAny(nameOf(file), extensions)) {
        banner.hidden = true;
        return;
      }
      banner.hidden = false;
      if (pathSpan) {
        pathSpan.textContent = file;
        pathSpan.title = file;
      }
      if (useBtn) useBtn.disabled = false;
    }

    if (useBtn) {
      useBtn.addEventListener("click", async function () {
        const path = sessionStorage.getItem(SS_FILE) || "";
        if (!path) return;
        useBtn.disabled = true;
        try {
          const r = await fetch(
            "/api/files/read?path=" + encodeURIComponent(path)
          );
          const j = await r.json();
          if (!j.ok) {
            alert("Failed to read " + path + ": " + j.error);
            return;
          }
          await onLoad(path, j.text);
        } catch (e) {
          alert("Failed to load: " + e.message);
        } finally {
          useBtn.disabled = false;
        }
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        sessionStorage.removeItem(SS_FILE);
        window.dispatchEvent(new CustomEvent("molbuilder.selection", {
          detail: {dir: sessionStorage.getItem(SS_DIR) || "", file: ""},
        }));
        refresh();
      });
    }

    // Cross-tab updates (sessionStorage write in OTHER tab).
    window.addEventListener("storage", function (e) {
      if (e.key === SS_FILE) refresh();
    });
    // Same-tab updates dispatched by explorer.js.
    window.addEventListener("molbuilder.selection", refresh);

    refresh();
  }

  root.molbuilderProjectsSelection = {init: init};
})(window);
