/* projects/api.js -- HTTP wrappers for /api/files/* + /api/projects/*.
 *
 * Pure functions: no DOM, no module-level state.  Each function
 * issues one HTTP request and returns the parsed JSON body (already
 * shaped as ``{ok, ...}``).  Network errors return a synthetic
 * ``{ok: false, error: "..."}`` so callers don't need a separate
 * try/catch around every call.
 *
 * The backend contract is in ``docs/protocols/selection.md`` § and
 * ``docs/protocols/web-api.md`` § ``/api/files/*``.
 */

export async function apiRoots() {
  const r = await fetch("/api/files/roots");
  return (await r.json()).roots || [];
}

export async function apiList(path) {
  const r = await fetch(
    "/api/files/list?path=" + encodeURIComponent(path)
  );
  return await r.json();
}

export async function apiStat(path) {
  const r = await fetch(
    "/api/files/stat?path=" + encodeURIComponent(path)
  );
  return await r.json();
}

export async function apiRead(path) {
  const r = await fetch(
    "/api/files/read?path=" + encodeURIComponent(path)
  );
  return await r.json();
}

export async function apiMkdir(parent, name) {
  const r = await fetch("/api/files/mkdir", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({parent: parent, name: name}),
  });
  return await r.json();
}

export async function apiCreateProject(name) {
  const r = await fetch("/api/projects/create", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name: name}),
  });
  return await r.json();
}

export async function apiUpload(targetDir, file) {
  const fd = new FormData();
  fd.append("target_dir", targetDir);
  fd.append("file", file);
  const r = await fetch("/api/files/upload", {method: "POST", body: fd});
  // 501 still returns valid JSON; the inline error UX renders it.
  try { return await r.json(); }
  catch (_) {
    return {ok: false, error: "upload server returned non-JSON (status "
                               + r.status + ")"};
  }
}

export async function apiDelete(path, recursive) {
  const r = await fetch("/api/files/delete", {
    method: "DELETE",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({path: path, recursive: !!recursive}),
  });
  try { return await r.json(); }
  catch (_) {
    return {ok: false, error: "delete server returned non-JSON (status "
                               + r.status + ")"};
  }
}

export async function apiWrite(path, text, opts) {
  opts = opts || {};
  const body = {path: path, text: text};
  if (opts.overwrite) body.overwrite = true;
  if (opts.expected_mtime != null) body.expected_mtime = opts.expected_mtime;
  let r;
  try {
    r = await fetch("/api/files/write", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify(body),
    });
    return await r.json();
  } catch (e) {
    return {ok: false, error: "network error: " + e.message};
  }
}
