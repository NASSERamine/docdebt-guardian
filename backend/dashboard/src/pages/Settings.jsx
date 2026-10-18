import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { errorText, sendWithToken } from "../api.js";
import { ConnectButton, ErrorNote, StateNote, SyncButton, formatDate, repoPath } from "../components.jsx";
import { useApi } from "../hooks.js";

function DefaultHint({ name, settings, format = (v) => String(v) }) {
  const overridden = settings.overridden.includes(name) && settings.effective[name] !== settings.defaults[name];
  if (!overridden) return <span className="field-hint">default</span>;
  return <span className="field-hint changed">changed here · default: {format(settings.defaults[name])}</span>;
}

// What is fixed, for determinism, and what remains a setting (the cost cap).
function FixedChoices({ settings }) {
  return (
    <div className="field">
      <span className="field-label">How every pull request is analysed (fixed)</span>
      <ul className="plain-list">
        <li>
          The model chosen on the <Link to="/model">Model page</Link>, at temperature 0, with its fixed prompt.
        </li>
        <li>One detection on the whole Javadoc: debt when an exact rule fails or the model says so. No threshold.</li>
        <li>
          Project context always retrieved: documented methods of the same file, then hybrid search in the knowledge
          base ({settings.rag_embeddings === "bm25" ? "BM25 only" : "BM25 + all-MiniLM-L6-v2 vectors"}, fused by rank).
        </li>
        <li>The field-by-field fix is proposed; a full regeneration is computed only when you ask for it.</li>
      </ul>
    </div>
  );
}

function SettingsForm({ settings, onSaved }) {
  const [form, setForm] = useState(settings.effective);
  const [message, setMessage] = useState(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => setForm(settings.effective), [settings]);

  const set = (name, value) => setForm((prev) => ({ ...prev, [name]: value }));
  const dirty = JSON.stringify(form) !== JSON.stringify(settings.effective);

  async function send(method, payload, success) {
    setSaving(true);
    setMessage(null);
    try {
      const { status, body } = await sendWithToken(method, "/api/settings", payload);
      if (status === 200) {
        setMessage({ ok: true, text: success });
        onSaved();
      } else if (status !== 0) {
        setMessage({ ok: false, text: errorText(body, status) });
      }
    } catch (e) {
      setMessage({ ok: false, text: `Could not reach the service: ${e}` });
    } finally {
      setSaving(false);
    }
  }

  const capOn = form.max_total_cost_usd !== null;
  return (
    <form
      className="settings-form"
      onSubmit={(e) => {
        e.preventDefault();
        send("PUT", form, "Saved. The next pull request uses these settings.");
      }}
    >
      {!settings.editable && (
        <p className="notice">
          Read-only: sign in with GitHub, or set <span className="mono">DOCDEBT_DASHBOARD_TOKEN</span> in{" "}
          <span className="mono">.env</span>, to change settings from here.
        </p>
      )}
      <FixedChoices settings={settings} />
      <fieldset disabled={!settings.editable || saving}>
        <div className="field">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={capOn}
              onChange={(e) => set("max_total_cost_usd", e.target.checked ? 5 : null)}
            />
            Cost cap
          </label>
          {capOn && (
            <span className="cap-input">
              $
              <input
                type="number"
                className="filter-input"
                min="0"
                step="0.01"
                value={form.max_total_cost_usd}
                onChange={(e) => set("max_total_cost_usd", e.target.value === "" ? 0 : Number(e.target.value))}
                aria-label="Cost cap in US dollars"
              />
            </span>
          )}
          <DefaultHint name="max_total_cost_usd" settings={settings} format={(v) => (v === null ? "no cap" : `$${v}`)} />
          <p className="field-help">
            ${settings.spent_usd.toFixed(4)} spent so far, all repositories (with the prices given on the Model page).
            Once the cap is reached, new pushes are not reviewed and the PR gets a note saying why. A budget, not a
            decision threshold: off by default.
          </p>
        </div>

        <div className="form-actions">
          <button type="submit" className="primary-btn" disabled={!dirty}>
            Save
          </button>
          {settings.overridden.length > 0 && (
            <button
              type="button"
              className="page-btn"
              onClick={() => window.confirm("Forget the changes made here and use the defaults?") && send("DELETE", undefined, "Back to the defaults.")}
            >
              Reset to defaults
            </button>
          )}
          {message && (message.ok ? <span className="feed-result ok">{message.text}</span> : <ErrorNote text={message.text} />)}
        </div>
      </fieldset>
    </form>
  );
}

function Repositories({ repos, connectUrl, onSynced }) {
  const connected = repos.filter((r) => r.connected);
  const removed = repos.filter((r) => r.installed_before);
  return (
    <>
      <p className="field-help">
        Connecting and disconnecting happen on GitHub, in the App&apos;s installation settings; this list follows
        automatically.
      </p>
      <ul className="repo-list">
        {connected.map((r) => (
          <li key={`${r.owner}/${r.repo}`}>
            <Link className="repo-row" to={repoPath(r.owner, r.repo)}>
              <span className="dot connected" />
              <span className="repo-name">
                {r.owner}/{r.repo}
              </span>
              <span className="repo-stats">{r.connected_at ? `since ${formatDate(r.connected_at)}` : ""}</span>
            </Link>
          </li>
        ))}
        {removed.map((r) => (
          <li key={`${r.owner}/${r.repo}`}>
            <Link className="repo-row" to={repoPath(r.owner, r.repo)}>
              <span className="dot disconnected" />
              <span className="repo-name">
                {r.owner}/{r.repo}
              </span>
              <span className="repo-stats">disconnected</span>
            </Link>
          </li>
        ))}
      </ul>
      {connected.length === 0 && removed.length === 0 && <p className="state-note">No installation recorded yet.</p>}
      <div className="form-actions">
        <ConnectButton connectUrl={connectUrl}>Connect or remove repositories on GitHub ↗</ConnectButton>
      </div>
      <div className="form-actions">
        <SyncButton onSynced={onSynced} />
      </div>
      <p className="field-help">
        A repository missing here, or shown as not connected although the App is installed on it? Sync asks GitHub
        directly (also done each time the service starts).
      </p>
    </>
  );
}

export default function Settings() {
  const { data, error, loading, refresh } = useApi(["/api/settings", "/api/repos", "/api/connect-url"]);
  return (
    <>
      <h1 className="page-title">Settings</h1>
      <StateNote error={!data && error} loading={!data && loading}>
        {data && (
          <div className="columns">
            <div className="panel">
              <h2>Review</h2>
              <SettingsForm settings={data[0]} onSaved={refresh} />
            </div>
            <div className="panel">
              <h2>Repositories</h2>
              <Repositories repos={data[1]} connectUrl={data[2].url} onSynced={refresh} />
            </div>
          </div>
        )}
      </StateNote>
    </>
  );
}
