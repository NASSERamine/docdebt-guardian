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

// In sign-in mode the session cookie authorizes every change: no token.
let authMode = false;
export function setAuthMode(enabled) {
  authMode = Boolean(enabled);
}

function askToken() {
  return readToken() || window.prompt("Dashboard token (DOCDEBT_DASHBOARD_TOKEN):") || null;
}

// A change (decision, settings) with the dashboard token. Resolves to
// { status, body }; a 401 forgets the stored token so the next try asks again,
// a 2xx remembers it.
export async function sendWithToken(method, path, payload) {
  const headers = { "Content-Type": "application/json" };
  let token = null;
  if (!authMode) {
    token = askToken();
    if (!token) return { status: 0, body: { detail: "No token given." } };
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch(path, {
    method,
    headers,
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const body = await res.json().catch(() => ({}));
  if (token && res.status === 401) writeToken(null);
  else if (token && res.ok) writeToken(token);
  return { status: res.status, body };
}

// What to do about an error, from its text: every error the user can fix
// says where (U4). Returns { to, label, external } or null.
export function actionFor(text) {
  const t = (text || "").toLowerCase();
  if (/api key|model page|provider refused|could not call the model|no model is configured|choose a model/.test(t))
    return { to: "/model", label: "Open the Model page" };
  if (/not connected|install the app|no github installation/.test(t)) return { to: "/repositories", label: "Add or connect the repository" };
  if (/cost cap/.test(t)) return { to: "/settings", label: "Change the cost cap" };
  if (/sign in|session ended|sign-in expired/.test(t)) return { to: "/", label: "Sign in again" };
  if (/regenerate first|no verified regeneration/.test(t)) return { to: null, label: "Use “Regenerate” first" };
  if (/checks/.test(t))
    return {
      to: "https://docs.github.com/en/apps/maintaining-github-apps/modifying-a-github-app-registration",
      label: "Give the App the Checks permission",
      external: true,
    };
  return null;
}

// FastAPI's 422 detail is a list of { loc, msg }; anything else is a string.
export function errorText(body, status) {
  if (Array.isArray(body?.detail)) return body.detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join("; ");
  if (status === 401) return authMode ? "Your session ended: sign in again." : "Wrong dashboard token — try again.";
  return body?.detail || `Request failed (${status}).`;
}
