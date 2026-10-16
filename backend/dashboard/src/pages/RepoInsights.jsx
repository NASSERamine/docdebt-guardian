import { useCallback, useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { errorText, getJson, sendWithToken } from "../api.js";
import { PrecisionCard, StateNote, formatDate, percent, timeAgo } from "../components.jsx";
import { useApi } from "../hooks.js";

// The repository page's "insights": documentation health of the default
// branch, the knowledge base (RAG) built by the same scan, the repository's
// policy, and what DocDebt learnt from the reviewers' decisions.

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

  return { active, error, start };
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
          {(scan.error || health.last_error) && (
            <span className="feed-result error">{scan.error || `Last scan failed: ${health.last_error}`}</span>
          )}
        </div>
      )}
    </div>
  );
}

function KnowledgeCard({ kb, scan, enabled, connected, ragEnabled }) {
  return (
    <div className="panel">
      <div className="panel-heading">
        <h2>Knowledge base (RAG)</h2>
        <span className={`badge ${ragEnabled ? "ok" : ""}`}>{ragEnabled ? "used in reviews" : "off for this repository"}</span>
      </div>
      <p className="field-help">
        What DocDebt knows about this project: it retrieves the relevant excerpts for each method and gives them to the
        model as reference, so fixes follow the project&apos;s vocabulary and conventions. Each accepted fix is added
        as an example.
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
            {kb.commit_sha && ` at ${kb.commit_sha.slice(0, 7)}`} · ranking:{" "}
            <span className="mono">{kb.embedder === "bm25" ? "BM25 (keywords)" : `embeddings, ${kb.embedder}`}</span>
          </>
        ) : (
          "Not indexed yet: reviews still use the file under review (same-file methods, called methods)."
        )}
      </p>
      {connected && (
        <div className="form-actions">
          <button type="button" className="page-btn" disabled={!enabled || scan.active} onClick={scan.start}>
            {scan.active ? "Indexing…" : "Re-index from the default branch"}
          </button>
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

  const toggles = [
    {
      key: "allow_cloud",
      label: policy.cloud_model ? `Allow the cloud model (${policy.cloud_model})` : "Allow the cloud model",
      help: policy.cloud_model
        ? "Cascade: the local model first; the cloud model only when the local one fails or is not confident enough. Off: the code never leaves your machine."
        : "No known cloud model is configured (DOCDEBT_CLOUD_MODEL, OPENAI_API_KEY): reviews stay local.",
    },
    {
      key: "rag_enabled",
      label: "Use the knowledge base (RAG)",
      help: "Give the model excerpts of this project as reference.",
    },
    {
      key: "block_on_structural",
      label: "Block the merge on structural debt",
      help: 'The "DocDebt Guardian" check run fails while structural debt is pending. Make it required in the branch protection rules to block merging.',
    },
  ];
  return (
    <div className="panel">
      <h2>Repository policy</h2>
      <fieldset className="plain-fieldset" disabled={!enabled || saving}>
        {toggles.map((t) => (
          <div className="field" key={t.key}>
            <label className="checkbox">
              <input type="checkbox" checked={Boolean(policy[t.key])} onChange={(e) => save({ [t.key]: e.target.checked })} />
              {t.label}
            </label>
            <p className="field-help">{t.help}</p>
          </div>
        ))}
        <div className="field">
          <span className="field-label">Confidence threshold</span>
          <p className="field-help">
            {policy.confidence_threshold != null ? (
              <>
                <strong>{policy.confidence_threshold.toFixed(2)}</strong> for this repository (global:{" "}
                {policy.global_confidence_threshold.toFixed(2)}).{" "}
                <button type="button" className="link-btn" onClick={() => save({ clear_threshold: true })}>
                  Use the global value
                </button>
              </>
            ) : (
              <>Global value: {policy.global_confidence_threshold.toFixed(2)} (Settings).</>
            )}
          </p>
        </div>
      </fieldset>
      {!enabled && <p className="field-help">Sign in or set DOCDEBT_DASHBOARD_TOKEN to change the policy.</p>}
      {message && <span className={`feed-result ${message.ok ? "ok" : "error"}`}>{message.text}</span>}
    </div>
  );
}

function ThresholdCard({ owner, repo, suggestion, enabled, onApplied }) {
  const s = suggestion;
  const [message, setMessage] = useState(null);

  async function apply() {
    const { status, body } = await sendWithToken("PUT", `${apiBase(owner, repo)}/settings`, {
      confidence_threshold: s.suggested,
    });
    if (status === 200) {
      setMessage({ ok: true, text: `Threshold ${s.suggested.toFixed(2)} applied to this repository.` });
      onApplied();
    } else if (status !== 0) setMessage({ ok: false, text: errorText(body, status) });
  }

  const data = s.curve.map((r) => ({ ...r, pct: r.precision == null ? null : Math.round(r.precision * 100) }));
  return (
    <div className="panel">
      <div className="panel-heading">
        <h2>Confidence threshold, learnt</h2>
        <span className="field-hint">target precision {percent(s.target)}</span>
      </div>
      <p className="field-help">
        For semantic debt, the precision the accepted/rejected decisions show at each threshold of the Judge. A higher
        threshold reports less debt, with fewer false alarms. Nothing changes until you apply it.
      </p>
      {s.curve.length === 0 ? (
        <p className="state-note">No decided semantic proposal yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={130}>
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="threshold" tick={{ fontSize: 11, fill: "var(--ink-faint)" }} axisLine={{ stroke: "var(--border)" }} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "var(--ink-faint)" }} axisLine={false} tickLine={false} />
            <Tooltip content={<ChartTip render={(r) => `≥ ${r.threshold}: ${r.pct}% precision · ${r.kept} proposals kept`} />} />
            <ReferenceLine y={s.target * 100} stroke="var(--pending)" strokeDasharray="4 4" />
            {s.current != null && <ReferenceLine x={s.current} stroke="var(--ink-faint)" label={{ value: "now", fill: "var(--ink-faint)", fontSize: 10 }} />}
            <Line isAnimationActive={false} type="stepAfter" dataKey="pct" stroke="var(--action)" strokeWidth={2} dot={{ r: 2 }} />
          </LineChart>
        </ResponsiveContainer>
      )}
      <div className="form-actions">
        {s.suggested != null ? (
          <>
            <span>
              Suggested: <strong className="mono">{s.suggested.toFixed(2)}</strong>
              {s.current != null && <span className="muted"> (now {s.current.toFixed(2)})</span>}
            </span>
            <button
              type="button"
              className="primary-btn"
              disabled={!enabled || (s.current != null && Math.abs(s.current - s.suggested) < 1e-9)}
              onClick={apply}
            >
              Apply to this repository
            </button>
          </>
        ) : (
          <span className="field-help">
            {s.enough
              ? "No threshold reaches the target precision with enough proposals kept."
              : `A suggestion appears after ${s.min_decisions} decided semantic proposals (${s.decisions} so far).`}
          </span>
        )}
        {message && <span className={`feed-result ${message.ok ? "ok" : "error"}`}>{message.text}</span>}
      </div>
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
      `${base}/threshold-suggestion`,
    ],
    { reloadKey },
  );
  const scan = useHealthScan(owner, repo, data?.[0]?.running ?? null, reload);

  return (
    <StateNote error={!data && error} loading={!data && loading}>
      {data &&
        (() => {
          const [health, kb, policy, precision, suggestion] = data;
          return (
            <>
              <h2 className="section-title">Documentation health and knowledge</h2>
              <div className="columns">
                <HealthCard health={health} scan={scan} enabled={enabled} connected={connected} />
                <KnowledgeCard kb={kb} scan={scan} enabled={enabled} connected={connected} ragEnabled={policy.rag_enabled} />
              </div>
              <h2 className="section-title">Policy and learning</h2>
              <div className="columns">
                <PolicyCard owner={owner} repo={repo} policy={policy} enabled={enabled} onSaved={reload} />
                <ThresholdCard owner={owner} repo={repo} suggestion={suggestion} enabled={enabled} onApplied={reload} />
              </div>
              <PrecisionCard precision={precision} title="Observed precision on this repository" />
            </>
          );
        })()}
    </StateNote>
  );
}
