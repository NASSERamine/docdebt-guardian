import { useCallback, useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { errorText, getJson, sendWithToken } from "../api.js";
import { ErrorNote, PrecisionCard, StateNote, formatDate, percent, timeAgo } from "../components.jsx";
import { useApi } from "../hooks.js";

// The repository page's "insights": documentation health of the default
// branch, the knowledge base (RAG) indexed by the same scan and kept up to
// date by every push, the merge block, and the observed precision.

const POLL_MS = 2000;

const SOURCE_LABELS = {
  method: "documented methods",
  test: "test methods",
  readme: "README sections",
  validated: "accepted fixes",
};

function apiBase(owner, repo) {
  return `/api/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`;
}

function scoreClass(score) {
  if (score == null) return "";
  if (score >= 0.8) return "accepted";
  if (score >= 0.5) return "pending";
  return "rejected";
}

function ChartTip({ active, payload, render }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tip">{render(payload[0].payload)}</div>;
}

function HealthTrend({ trend }) {
  if (trend.length < 2) return null;
  const data = trend.map((t) => ({ ...t, pct: Math.round(t.score * 1000) / 10 }));
  return (
    <ResponsiveContainer width="100%" height={110}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={(d) => new Date(d * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
          tick={{ fontSize: 11, fill: "var(--ink-faint)" }}
          axisLine={{ stroke: "var(--border)" }}
          tickLine={false}
        />
        <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "var(--ink-faint)" }} axisLine={false} tickLine={false} />
        <Tooltip
          content={
            <ChartTip render={(r) => `${formatDate(r.date)} · ${r.pct}%${r.commit_sha ? ` · ${r.commit_sha.slice(0, 7)}` : ""}`} />
          }
        />
        <Line isAnimationActive={false} type="monotone" dataKey="pct" stroke="var(--accepted)" strokeWidth={2} dot={{ r: 2 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// Measures the default branch (deterministic checks only, no LLM, no cost)
// and rebuilds the knowledge base in the same pass. Polls until it ends.
function useHealthScan(owner, repo, running, onFinished) {
  const [scan, setScan] = useState(running);
  const [error, setError] = useState(null);
  useEffect(() => setScan(running), [running?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const active = scan?.status === "running";
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(async () => {
      try {
        const next = await getJson(`/api/health-scans/${scan.id}`);
        setScan(next);
        if (next.status !== "running") {
          if (next.status === "failed") setError(`Scan failed: ${next.error}`);
          onFinished();
        }
      } catch (e) {
        setError(`Lost track of the scan: ${e}`);
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [active, scan?.id, onFinished]);

  async function start() {
    setError(null);
    const { status, body } = await sendWithToken("POST", `${apiBase(owner, repo)}/health-scan`);
    if (status === 202) setScan(body);
    else if (status !== 0) setError(errorText(body, status));
  }

  return { active, error, start, progress: active ? { done: scan.progress_done || 0, total: scan.progress_total || 0 } : null };
}

// Live progress of a running scan (U5).
export function Progress({ progress, label }) {
  if (!progress) return null;
  const pct = progress.total ? Math.round((100 * progress.done) / progress.total) : 0;
  return (
    <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-bar" style={{ width: `${pct}%` }} />
      <span className="progress-label">
        {label} {progress.total ? `${progress.done}/${progress.total}` : "starting…"}
      </span>
    </div>
  );
}

function HealthCard({ health, scan, enabled, connected }) {
  const h = health.latest;
  const delta = h && health.previous_score != null ? h.score - health.previous_score : null;
  return (
    <div className="panel">
      <div className="panel-heading">
        <h2>Documentation health</h2>
        {h && <span className="field-hint">{h.branch} · {h.commit_sha?.slice(0, 7)} · {timeAgo(h.finished_at)}</span>}
      </div>
      {!h ? (
        <p className="state-note">
          Not measured yet. The scan reads the default branch and counts, among public and protected methods, those
          documented without structural debt. No LLM call, no cost.
        </p>
      ) : (
        <>
          <div className="precision-head">
            <span className={`precision-value ${scoreClass(h.score)}`}>{percent(h.score)}</span>
            {delta != null && Math.abs(delta) >= 0.0005 && (
              <span className={`delta ${delta > 0 ? "accepted" : "rejected"}`}>
                {delta > 0 ? "▲" : "▼"} {Math.abs(delta * 100).toFixed(1)} pts
              </span>
            )}
            <span className="field-help">
              {h.documented - h.debt_methods} of {h.methods} methods documented without debt · {h.debt_methods} with
              structural debt · {h.methods - h.documented} undocumented · {h.files} files
              {h.truncated && " (first files only: raise DOCDEBT_HEALTH_MAX_FILES)"}
            </span>
          </div>
          <HealthTrend trend={health.trend} />
          <div className="precision-grid">
            {h.packages.length > 0 && (
              <div>
                <div className="quote-label">Packages, least healthy first</div>
                <ul className="mini-table">
                  {h.packages.slice(0, 8).map((p) => (
                    <li key={p.package}>
                      <span className="mono">{p.package || "(default)"}</span>
                      <span className={scoreClass(p.score)}>
                        {percent(p.score)} <span className="muted">· {p.methods} m.</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {h.worst_files.length > 0 && (
              <div>
                <div className="quote-label">Files with the most to fix</div>
                <ul className="mini-table">
                  {h.worst_files.slice(0, 8).map((f) => (
                    <li key={f.path}>
                      <span className="mono ellipsis" title={f.path}>
                        {f.path.split("/").pop()}
                      </span>
                      <span className="muted">
                        {f.debt_methods} debt · {f.undocumented} undoc.
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </>
      )}
      {connected && (
        <div className="form-actions">
          <button
            type="button"
            className="primary-btn"
            disabled={!enabled || scan.active}
            onClick={scan.start}
            title={enabled ? "Measure the default branch now" : "Sign in or set DOCDEBT_DASHBOARD_TOKEN"}
          >
            {scan.active ? "Analysing…" : "Analyse the default branch"}
          </button>
          <Progress progress={scan.progress} label="Reading files" />
          {(scan.error || health.last_error) && <ErrorNote text={scan.error || `Last scan failed: ${health.last_error}`} />}
        </div>
      )}
    </div>
  );
}

function KnowledgeCard({ kb, scan, enabled, connected }) {
  return (
    <div className="panel">
      <div className="panel-heading">
        <h2>Knowledge base (RAG)</h2>
        <span className="badge ok">always used in reviews</span>
      </div>
      <p className="field-help">
        What DocDebt knows about this project. For each method it gives the model up to three documented methods of
        the same file, then the four best excerpts of this base: hybrid search, BM25 keywords and{" "}
        <span className="mono">all-MiniLM-L6-v2</span> vectors fused by rank (no similarity threshold). Every accepted
        fix — an edited one too — is added as an example; every push to the default branch re-indexes the files it
        changed.
      </p>
      <div className="kb-counts">
        {Object.entries(kb.counts).map(([kind, n]) => (
          <div className="tile" key={kind}>
            <div className="tile-value">{n}</div>
            <div className="tile-label">{SOURCE_LABELS[kind] || kind}</div>
          </div>
        ))}
      </div>
      <p className="field-help">
        {kb.indexed_at ? (
          <>
            Indexed {timeAgo(kb.indexed_at)}
            {kb.commit_sha && ` at ${kb.commit_sha.slice(0, 7)}`} · vectors:{" "}
            <span className="mono">{kb.embedder === "bm25" ? "none (BM25 only)" : kb.embedder}</span>
          </>
        ) : (
          <span className="pending">
            Not indexed yet: reviews use the methods of the file under review only. Index it with the button below.
          </span>
        )}
      </p>
      {connected && (
        <div className="form-actions">
          <button type="button" className="page-btn" disabled={!enabled || scan.active} onClick={scan.start}>
            {scan.active ? "Indexing…" : kb.indexed_at ? "Re-index from the default branch" : "Index the knowledge base"}
          </button>
          <Progress progress={scan.progress} label="Indexing" />
        </div>
      )}
    </div>
  );
}

function PolicyCard({ owner, repo, policy, enabled, onSaved }) {
  const [message, setMessage] = useState(null);
  const [saving, setSaving] = useState(false);

  async function save(values) {
    setSaving(true);
    setMessage(null);
    try {
      const { status, body } = await sendWithToken("PUT", `${apiBase(owner, repo)}/settings`, values);
      if (status === 200) {
        setMessage({ ok: true, text: "Saved: applies from the next push." });
        onSaved();
      } else if (status !== 0) setMessage({ ok: false, text: errorText(body, status) });
    } catch (e) {
      setMessage({ ok: false, text: `Could not reach the service: ${e}` });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="panel">
      <h2>Merge check</h2>
      <fieldset className="plain-fieldset" disabled={!enabled || saving}>
        <div className="field">
          <label className="checkbox">
            <input type="checkbox" checked={Boolean(policy.block_on_debt)} onChange={(e) => save({ block_on_debt: e.target.checked })} />
            Block the merge while documentation debt is pending
          </label>
          <p className="field-help">
            The &quot;DocDebt Guardian&quot; check run fails while a proposal of the pull request waits for a decision
            (otherwise it is neutral). Make it required in the branch protection rules to block merging.
          </p>
        </div>
      </fieldset>
      {!enabled && <p className="field-help">Sign in or set DOCDEBT_DASHBOARD_TOKEN to change it.</p>}
      {message && (message.ok ? <span className="feed-result ok">{message.text}</span> : <ErrorNote text={message.text} />)}
    </div>
  );
}

export default function RepoInsights({ owner, repo, connected, enabled }) {
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((k) => k + 1), []);
  const base = apiBase(owner, repo);
  const { data, error, loading } = useApi(
    [
      `${base}/health`,
      `${base}/knowledge`,
      `${base}/settings`,
      `/api/precision?repo=${encodeURIComponent(`${owner}/${repo}`)}&days=90`,
    ],
    { reloadKey },
  );
  const scan = useHealthScan(owner, repo, data?.[0]?.running ?? null, reload);

  return (
    <StateNote error={!data && error} loading={!data && loading}>
      {data &&
        (() => {
          const [health, kb, policy, precision] = data;
          return (
            <>
              <h2 className="section-title">Documentation health and knowledge</h2>
              <div className="columns">
                <HealthCard health={health} scan={scan} enabled={enabled} connected={connected} />
                <KnowledgeCard kb={kb} scan={scan} enabled={enabled} connected={connected} />
              </div>
              <h2 className="section-title">Merge check and decisions</h2>
              <div className="columns">
                <PolicyCard owner={owner} repo={repo} policy={policy} enabled={enabled} onSaved={reload} />
                <PrecisionCard precision={precision} title="Observed precision on this repository" />
              </div>
            </>
          );
        })()}
    </StateNote>
  );
}
