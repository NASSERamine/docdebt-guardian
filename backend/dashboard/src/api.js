// Every call the dashboard makes to the service, in one place.

const TOKEN_KEY = "docdebt.dashboardToken";

export async function getJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

// The token only lives in this browser; storage can be unavailable (private
// window, blocked site data), in which case the user is simply asked again.
export function readToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function writeToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* not persisted: asked again next time */
  }
}

function askToken() {
  return readToken() || window.prompt("Dashboard token (DOCDEBT_DASHBOARD_TOKEN):") || null;
}

// A change (decision, settings) with the dashboard token. Resolves to
// { status, body }; a 401 forgets the stored token so the next try asks again,
// a 2xx remembers it.
export async function sendWithToken(method, path, payload) {
  const token = askToken();
  if (!token) return { status: 0, body: { detail: "No token given." } };
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const body = await res.json().catch(() => ({}));
  if (res.status === 401) writeToken(null);
  else if (res.ok) writeToken(token);
  return { status: res.status, body };
}

// FastAPI's 422 detail is a list of { loc, msg }; anything else is a string.
export function errorText(body, status) {
  if (Array.isArray(body?.detail)) return body.detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join("; ");
  if (status === 401) return "Wrong dashboard token — try again.";
  return body?.detail || `Request failed (${status}).`;
}
