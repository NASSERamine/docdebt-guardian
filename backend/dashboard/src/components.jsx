import { useState } from "react";
import { Link } from "react-router-dom";
import { errorText, sendWithToken } from "./api.js";

export function timeAgo(unixSeconds) {
  const seconds = Math.max(0, Date.now() / 1000 - unixSeconds);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function formatDate(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function repoPath(owner, repo) {
  return `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`;
}

export function marker(proposal) {
  if (proposal.decision === "accept") return { icon: "●", cls: "accepted", label: "accepted" };
  if (proposal.decision === "reject") return { icon: "✕", cls: "rejected", label: "rejected" };
  if (proposal.decision === "expired") return { icon: "◌", cls: "", label: "expired (push again for a fresh review)" };
  if (proposal.resolved) {
    // Resolved before the "decision" column existed (a database migrated from
    // an earlier version of this service): the outcome is genuinely unknown,
    // so say that rather than call it "pending" -- it is not waiting on anyone.
    return { icon: "◐", cls: "", label: "resolved (decision unknown)" };
  }
  return { icon: "○", cls: "pending", label: "pending" };
}

export function ConnectButton({ connectUrl, children = "+ Connect a repository" }) {
  if (!connectUrl) return null;
  return (
    <a className="connect-btn" href={connectUrl} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

export function StateNote({ error, loading, children }) {
  if (error) return <p className="state-note">Could not reach the service: {error}</p>;
  if (loading) return <p className="state-note">Loading…</p>;
  return children ?? null;
}

// Asks GitHub which repositories the App is installed on and updates the
// list: for repositories whose installation webhook was missed.
export function SyncButton({ onSynced }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  async function sync() {
    setBusy(true);
    setMessage(null);
    try {
      const { status, body } = await sendWithToken("POST", "/api/installations/sync");
      if (status === 200) {
        const parts = [`${body.connected.length} connected`];
        if (body.disconnected.length) parts.push(`${body.disconnected.length} disconnected`);
        setMessage({ ok: true, text: `Synced with GitHub: ${parts.join(" · ")}.` });
        onSynced?.();
      } else if (status !== 0) {
        setMessage({ ok: false, text: errorText(body, status) });
      }
    } catch (e) {
      setMessage({ ok: false, text: `Could not reach the service: ${e}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <span className="sync-box">
      <button type="button" className="page-btn" disabled={busy} onClick={sync}>
        {busy ? "Syncing…" : "Sync with GitHub"}
      </button>
      {message && <span className={`feed-result ${message.ok ? "ok" : "error"}`}>{message.text}</span>}
    </span>
  );
}

export function StatTiles({ tiles }) {
  return (
    <div className="tiles">
      {tiles.map((t) => (
        <div className="tile" key={t.label}>
          <div className={`tile-value ${t.cls || ""}`}>{t.value}</div>
          <div className="tile-label">{t.label}</div>
        </div>
      ))}
    </div>
  );
}

export const REJECT_REASONS = [
  { key: "false_positive", label: "False positive", help: "The documentation was right: there was no debt." },
  { key: "incorrect_fix", label: "Incorrect fix", help: "The debt is real but the proposed Javadoc is wrong." },
  { key: "manual", label: "I'll fix it myself", help: "The debt is real; the author prefers to write the fix." },
  { key: "other", label: "Other", help: "Any other reason." },
];

// Reject in two clicks: "Reject", then "Why?" — the reason is what the
// precision and the threshold suggestion learn from. "No reason" still works.
export function RejectButton({ proposal, busy, onDecide, big = false }) {
  const [open, setOpen] = useState(false);
  const choose = (reason) => {
    setOpen(false);
    onDecide(proposal, "reject", reason);
  };
  return (
    <span className="reject-menu">
      <button
        type="button"
        className={`decide-btn reject${big ? " big" : ""}`}
        disabled={busy}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        title="Dismiss this proposal; nothing is committed"
      >
        Reject ▾
      </button>
      {open && (
        <span className="reject-popover" role="menu">
          <span className="reject-why">Why?</span>
          {REJECT_REASONS.map((r) => (
            <button key={r.key} type="button" role="menuitem" title={r.help} onClick={() => choose(r.key)}>
              {r.label}
            </button>
          ))}
          <button type="button" role="menuitem" className="muted" onClick={() => choose(null)}>
            No reason
          </button>
        </span>
      )}
    </span>
  );
}

function DecisionButtons({ proposal, busy, onDecide }) {
  return (
    <span className="decision-buttons">
      <button
        type="button"
        className="decide-btn accept"
        disabled={busy}
        onClick={() => onDecide(proposal, "accept")}
        title="Commit this fix to the pull request branch"
      >
        Accept
      </button>
      <RejectButton proposal={proposal} busy={busy} onDecide={onDecide} />
    </span>
  );
}

export function percent(value) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

const MECHANISM_LABELS = { hybrid: "hybrid", regeneration: "regeneration" };
const REASON_LABELS = Object.fromEntries(REJECT_REASONS.map((r) => [r.key, r.label]));

// Observed precision: accepted / (accepted + rejected) over a period, with
// the breakdowns that explain it. Shown once there are enough decisions.
export function PrecisionCard({ precision, title = "Observed precision" }) {
  const p = precision;
  return (
    <div className="panel precision-card">
      <div className="panel-heading">
        <h2>{title}</h2>
        <span className="field-hint">last {p.days} days</span>
      </div>
      {p.decisions === 0 ? (
        <p className="state-note">No accepted or rejected proposal yet: each decision is a label DocDebt learns from.</p>
      ) : (
        <>
          <div className="precision-head">
            <span className={`precision-value ${p.enough ? "" : "muted"}`}>{percent(p.precision)}</span>
            <span className="field-help">
              {p.accepted} accepted / {p.decisions} decided
              {!p.enough && ` · indicative until ${p.min_decisions} decisions`}
            </span>
          </div>
          <div className="precision-grid">
            {[
              ["By debt kind", p.by_kind, (k) => k],
              ["By model", p.by_model, (k) => k],
              ["By repair mechanism", p.by_mechanism, (k) => MECHANISM_LABELS[k] || k],
            ].map(([label, rows, fmt]) =>
              rows.length ? (
                <div key={label}>
                  <div className="quote-label">{label}</div>
                  <ul className="mini-table">
                    {rows.map((r) => (
                      <li key={r.key}>
                        <span className="mono">{fmt(r.key)}</span>
                        <span>
                          {percent(r.precision)} <span className="muted">({r.accepted}/{r.accepted + r.rejected})</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null,
            )}
            {Object.keys(p.reject_reasons).length > 0 && (
              <div>
                <div className="quote-label">Why proposals were rejected</div>
                <ul className="mini-table">
                  {Object.entries(p.reject_reasons).map(([k, n]) => (
                    <li key={k}>
                      <span>{REASON_LABELS[k] || k}</span>
                      <span>{n}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// The list of proposals used by every page. `decisions` is the object
// returned by useDecision; `showRepo` is false on a repository's own page.
export function ProposalList({ proposals, decisionsEnabled, decisions, showRepo = true, absoluteDates = false }) {
  if (proposals.length === 0) {
    return <p className="state-note">No proposal matches.</p>;
  }
  return (
    <ul className="feed">
      {proposals.map((p) => {
        const m = marker(p);
        const result = decisions.results[p.short_id];
        const pending = m.label === "pending";
        return (
          <li className="feed-item" key={p.short_id}>
            <span className={`feed-marker ${m.cls}`}>{m.icon}</span>
            <div className="feed-body">
              <div className="feed-title">
                {showRepo && (
                  <Link className="ref" to={repoPath(p.owner, p.repo)}>
                    {p.owner}/{p.repo}
                  </Link>
                )}
                <a className="ref" href={p.pr_url} target="_blank" rel="noreferrer" title="Open the pull request">
                  #{p.pr_number}
                </a>{" "}
                <Link className="method" to={`/proposals/${p.short_id}`} title="See the debt and the proposed fix">
                  {p.method_label}
                </Link>
              </div>
              <div className="feed-meta">
                {m.label}
                {p.reject_reason && ` (${REASON_LABELS[p.reject_reason] || p.reject_reason})`} · {absoluteDates ? formatDate(p.created_at) : timeAgo(p.created_at)} ·{" "}
                <span className="mono">{p.file_path}</span>
                {pending && !decisionsEnabled && (
                  <>
                    {" · reply "}
                    <span className="command">/docdebt accept {p.short_id}</span>
                  </>
                )}
                {decisions.busyId === p.short_id && " · working…"}
              </div>
              {result && <div className={`feed-result ${result.ok ? "ok" : "error"}`}>{result.message}</div>}
            </div>
            <span className="row-actions">
              <Link className="page-btn" to={`/proposals/${p.short_id}`}>
                {pending ? "Review" : "Details"}
              </Link>
              {pending && decisionsEnabled && (
                <DecisionButtons proposal={p} busy={decisions.busyId !== null} onDecide={decisions.decide} />
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
