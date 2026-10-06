import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { errorText, sendWithToken } from "../api.js";
import { useAuth } from "../auth.jsx";
import { StateNote, SyncButton, formatDate, repoPath } from "../components.jsx";
import { useApi } from "../hooks.js";

function AddRepository({ onAdded }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null); // { kind: "error" | "install" | "ok", ... }
  const navigate = useNavigate();

  async function add(value = text) {
    if (!value.trim()) return;
    setBusy(true);
    setResult(null);
    try {
      const { status, body } = await sendWithToken("POST", "/api/repos", { repo: value });
      if (status === 200 && body.status === "connected") {
        onAdded();
        navigate(repoPath(body.owner, body.repo));
      } else if (status === 200 && body.status === "not_installed") {
        setResult({ kind: "install", ...body });
      } else if (status !== 0) {
        setResult({ kind: "error", text: errorText(body, status) });
      }
    } catch (e) {
      setResult({ kind: "error", text: `Could not reach the service: ${e}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="add-repo">
      <form
        className="filters"
        onSubmit={(e) => {
          e.preventDefault();
          add();
        }}
      >
        <input
          className="filter-input grow"
          placeholder="owner/name or https://github.com/owner/name"
          value={text}
          onChange={(e) => setText(e.target.value)}
          aria-label="Repository"
          autoFocus
        />
        <button type="submit" className="primary-btn" disabled={busy || !text.trim()}>
          {busy ? "Checking…" : "Add repository"}
        </button>
      </form>
      {result?.kind === "error" && <p className="feed-result error">{result.text}</p>}
      {result?.kind === "install" && (
        <div className="install-card">
          <p>
            <strong className="mono">
              {result.owner}/{result.repo}
            </strong>{" "}
            exists and you can push to it, but the DocDebt App is not installed on it yet.
          </p>
          <ol>
            <li>
              <a className="primary-btn" href={result.install_url} target="_blank" rel="noreferrer">
                Install DocDebt on GitHub ↗
              </a>{" "}
              — choose <em>Only select repositories</em>, pick <span className="mono">{result.repo}</span>, then Install.
            </li>
            <li>
              Come back here and{" "}
              <button type="button" className="page-btn" disabled={busy} onClick={() => add(`${result.owner}/${result.repo}`)}>
                check again
              </button>
            </li>
          </ol>
        </div>
      )}
    </div>
  );
}

export default function Repositories() {
  const { me } = useAuth();
  const { data, error, loading, refresh } = useApi(["/api/repos"]);
  const [removing, setRemoving] = useState(null);

  async function remove(r) {
    if (!window.confirm(`Remove ${r.owner}/${r.repo} from your list? The App stays installed on GitHub.`)) return;
    setRemoving(`${r.owner}/${r.repo}`);
    await sendWithToken("DELETE", `/api/repos/${encodeURIComponent(r.owner)}/${encodeURIComponent(r.repo)}`);
    setRemoving(null);
    refresh();
  }

  return (
    <>
      <h1 className="page-title">Repositories</h1>
      <section className="panel add-panel">
        <h2>Add a repository</h2>
        <p className="field-help">
          Type the exact name or paste the GitHub URL.
          {me?.auth_enabled && " You need write access to it: accepting a fix commits to it."}
        </p>
        <AddRepository onAdded={refresh} />
      </section>

      <section>
        <div className="panel-heading">
          <h2 className="section-title">{me?.auth_enabled ? "Your repositories" : "Connected repositories"}</h2>
          <SyncButton onSynced={refresh} />
        </div>
        <StateNote error={!data && error} loading={!data && loading}>
          {data &&
            (data[0].length === 0 ? (
              <p className="state-note">No repository yet: add one above.</p>
            ) : (
              <ul className="repo-list">
                {data[0].map((r) => (
                  <li key={`${r.owner}/${r.repo}`} className="repo-item">
                    <Link className="repo-row" to={repoPath(r.owner, r.repo)}>
                      <span className={`dot ${r.connected ? "connected" : "disconnected"}`} />
                      <span className="repo-name">
                        {r.owner}/{r.repo}
                      </span>
                      <span className="repo-stats">
                        {r.pending > 0 && <span className="pending">{r.pending} pending</span>}
                        {r.accepted > 0 && <span className="accepted">{r.accepted} accepted</span>}
                        {!r.connected && <span>not connected</span>}
                        {r.connected && r.connected_at && <span>since {formatDate(r.connected_at)}</span>}
                      </span>
                    </Link>
                    {me?.auth_enabled && (
                      <button
                        type="button"
                        className="page-btn"
                        disabled={removing === `${r.owner}/${r.repo}`}
                        onClick={() => remove(r)}
                        title="Remove from your list"
                      >
                        Remove
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            ))}
        </StateNote>
      </section>
    </>
  );
}
