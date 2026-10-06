import { useEffect, useState } from "react";
import UsageChart from "./UsageChart.jsx";

const REFRESH_MS = 15000;

async function getJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function timeAgo(unixSeconds) {
  const seconds = Math.max(0, Date.now() / 1000 - unixSeconds);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function StatusLine({ summary }) {
  return (
    <div className="statusline">
      <h1>DocDebt Guardian</h1>
      <span className="metric">
        <strong>{summary.repositories}</strong> repos
      </span>
      <span className="metric">
        <strong className="pending">{summary.pending}</strong> pending
      </span>
      <span className="metric">
        <strong className="accepted">{summary.accepted}</strong> accepted
      </span>
      <span className="metric">
        <strong>${summary.cost_usd.toFixed(2)}</strong> total cost
      </span>
    </div>
  );
}

function ConnectButton({ connectUrl }) {
  if (!connectUrl) return null;
  return (
    <a className="connect-btn" href={connectUrl} target="_blank" rel="noreferrer">
      + Connect a repository
    </a>
  );
}

function RepoList({ repos, connectUrl }) {
  if (repos.length === 0) {
    return (
      <div className="empty-state">
        <p>No repository connected yet. Install the App on a repository to start reviewing its documentation.</p>
        <ConnectButton connectUrl={connectUrl} />
      </div>
    );
  }
  return (
    <>
      <div className="panel-actions">
        <span />
        <ConnectButton connectUrl={connectUrl} />
      </div>
      <ul className="repo-list">
        {repos.map((r) => (
          <li className="repo-row" key={`${r.owner}/${r.repo}`}>
            <span className={`dot ${r.connected ? "connected" : "disconnected"}`} title={r.connected ? "connected" : "not connected"} />
            <span className="repo-name">
              {r.owner}/{r.repo}
            </span>
            {r.proposals !== undefined && (
              <span className="repo-stats">
                {r.pending > 0 && <span className="pending">{r.pending} pending</span>}
                {r.accepted > 0 && <span className="accepted">{r.accepted} accepted</span>}
                {r.rejected > 0 && <span className="rejected">{r.rejected} rejected</span>}
                {r.unknown > 0 && <span title="Resolved before this dashboard could record the decision">{r.unknown} unknown</span>}
              </span>
            )}
          </li>
        ))}
      </ul>
    </>
  );
}

function marker(proposal) {
  if (proposal.decision === "accept") return { icon: "●", cls: "accepted", label: "accepted" };
  if (proposal.decision === "reject") return { icon: "✕", cls: "rejected", label: "rejected" };
  if (proposal.resolved) {
    // Resolved before the "decision" column existed (a database migrated from
    // an earlier version of this service): the outcome is genuinely unknown,
    // so say that rather than call it "pending" -- it is not waiting on anyone.
    return { icon: "◐", cls: "", label: "resolved (decision unknown)" };
  }
  return { icon: "○", cls: "pending", label: "pending" };
}

function ActivityFeed({ proposals }) {
  if (proposals.length === 0) {
    return <p className="state-note">No proposal has been posted yet.</p>;
  }
  return (
    <ul className="feed">
      {proposals.map((p) => {
        const m = marker(p);
        return (
          <li className="feed-item" key={p.short_id}>
            <span className={`feed-marker ${m.cls}`}>{m.icon}</span>
            <div className="feed-body">
              <div className="feed-title">
                <span className="ref">
                  {p.owner}/{p.repo}#{p.pr_number}
                </span>{" "}
                <span className="method">{p.method_label}</span>
              </div>
              <div className="feed-meta">
                {m.label} {timeAgo(p.created_at)}
                {m.label === "pending" && (
                  <>
                    {" · reply "}
                    <span className="command">/docdebt accept {p.short_id}</span>
                  </>
                )}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export default function App() {
  const [summary, setSummary] = useState(null);
  const [repos, setRepos] = useState([]);
  const [proposals, setProposals] = useState([]);
  const [usage, setUsage] = useState([]);
  const [connectUrl, setConnectUrl] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      try {
        const [s, r, p, u, c] = await Promise.all([
          getJson("/api/summary"),
          getJson("/api/repos"),
          getJson("/api/proposals?limit=20"),
          getJson("/api/usage?days=14"),
          getJson("/api/connect-url"),
        ]);
        if (cancelled) return;
        setSummary(s);
        setRepos(r);
        setProposals(p);
        setUsage(u);
        setConnectUrl(c.url);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    }

    refresh();
    const id = setInterval(refresh, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (error) {
    return (
      <div className="app">
        <p className="state-note">Could not reach the service: {error}</p>
      </div>
    );
  }
  if (!summary) {
    return (
      <div className="app">
        <p className="state-note">Loading…</p>
      </div>
    );
  }

  return (
    <div className="app">
      <StatusLine summary={summary} />
      <div className="columns">
        <div className="panel">
          <h2>Repositories</h2>
          <RepoList repos={repos} connectUrl={connectUrl} />
        </div>
        <div className="panel">
          <h2>Recent activity</h2>
          <ActivityFeed proposals={proposals} />
        </div>
      </div>
      <UsageChart series={usage} />
    </div>
  );
}
