import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { errorText, sendWithToken } from "../api.js";
import { ConnectButton, StateNote, SyncButton, formatDate, repoPath } from "../components.jsx";
import { useApi } from "../hooks.js";

const PATTERN_LABELS = { zeroshot: "zero-shot", cot: "chain of thought" };

const MECHANISM_HELP = {
  hybrid:
    "Hybrid replacement (recommended): a deterministic plan fixes the structure (renames, deletions), the model only writes or judges the fields it is asked about, and a post-validation reverts invented identifiers and cosmetic edits. Untouched lines stay byte-identical.",
  regeneration:
    "Regeneration: the model rewrites the whole Javadoc with the screened repair prompt, then the Verifier checks it. Only for models with a screened prompt; others always use the hybrid mechanism.",
};

const RAG_SOURCES = {
  siblings: ["Same file", "documented methods of the file under review"],
  callees: ["Called methods", "the documentation of the methods it calls"],
  similar: ["Similar methods", "the closest documented methods of the whole project"],
  validated: ["Accepted fixes", "Javadocs reviewers accepted from DocDebt"],
  tests: ["Tests", "test methods that call it"],
  readme: ["README", "the most relevant README section"],
};

function CloudInfo({ cloud }) {
  if (!cloud.model) {
    return <p className="field-help">No cloud model configured (DOCDEBT_CLOUD_MODEL): every review stays local.</p>;
  }
  const problems = [];
  if (!cloud.known) problems.push("unknown model id");
  if (!cloud.api_key_set) problems.push("OPENAI_API_KEY is not set");
  return (
    <p className="field-help">
      Cloud model: <span className="badge model">{cloud.model}</span> —{" "}
      {cloud.price_in_per_m != null || cloud.price_out_per_m != null
        ? `$${cloud.price_in_per_m ?? "?"}/M input tokens, $${cloud.price_out_per_m ?? "?"}/M output tokens.`
        : "catalog prices (set DOCDEBT_CLOUD_PRICE_IN_PER_M / _OUT_PER_M to override)."}{" "}
      Used only on repositories whose policy allows it, after the local model,
      and within the cost cap.
      {problems.length > 0 && <span className="feed-result error"> Not usable: {problems.join(", ")}.</span>}
    </p>
  );
}

function DefaultHint({ name, settings, format = (v) => String(v) }) {
  const overridden = settings.overridden.includes(name) && settings.effective[name] !== settings.defaults[name];
  if (!overridden) return <span className="field-hint">default from the environment</span>;
  return <span className="field-hint changed">changed here · environment default: {format(settings.defaults[name])}</span>;
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
          Read-only: set <span className="mono">DOCDEBT_DASHBOARD_TOKEN</span> in <span className="mono">.env</span> and
          restart the service to change settings from here.
        </p>
      )}
      <fieldset disabled={!settings.editable || saving}>
        <div className="field">
          <label htmlFor="model">Model</label>
          <select id="model" className="filter-input" value={form.repair_model} onChange={(e) => set("repair_model", e.target.value)}>
            {[...new Set([...settings.choices.models, form.repair_model])].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <DefaultHint name="repair_model" settings={settings} />
          <p className="field-help">
            Local model: writes the repairs and, when semantic detection is on, judges the documentation. Must be
            available in Ollama.
          </p>
          <CloudInfo cloud={settings.cloud} />
        </div>

        <div className="field">
          <label htmlFor="mechanism">Repair mechanism</label>
          <select
            id="mechanism"
            className="filter-input"
            value={form.repair_mechanism}
            onChange={(e) => set("repair_mechanism", e.target.value)}
          >
            {settings.choices.mechanisms.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <DefaultHint name="repair_mechanism" settings={settings} />
          <p className="field-help">{MECHANISM_HELP[form.repair_mechanism]}</p>
        </div>

        <div className="field">
          <label htmlFor="pattern">Repair prompt</label>
          <select id="pattern" className="filter-input" value={form.repair_pattern} onChange={(e) => set("repair_pattern", e.target.value)}>
            {[...new Set([...settings.choices.patterns, form.repair_pattern])].map((p) => (
              <option key={p} value={p}>
                {PATTERN_LABELS[p] || p}
              </option>
            ))}
          </select>
          <DefaultHint name="repair_pattern" settings={settings} format={(v) => PATTERN_LABELS[v] || v} />
        </div>

        <div className="field">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={!form.structural_only}
              onChange={(e) => set("structural_only", !e.target.checked)}
            />
            Semantic detection (LLM Judge)
          </label>
          <DefaultHint name="structural_only" settings={settings} format={(v) => (v ? "off" : "on")} />
          <p className="field-help">
            Off: only exact structural checks (@param, @return, @throws). On: the model also looks for descriptions that
            contradict the code — slower, and it costs LLM calls.
          </p>
        </div>

        <div className="field">
          <label htmlFor="threshold">Confidence threshold: {Number(form.confidence_threshold).toFixed(2)}</label>
          <input
            id="threshold"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={form.confidence_threshold}
            onChange={(e) => set("confidence_threshold", Number(e.target.value))}
            disabled={form.structural_only}
          />
          <DefaultHint name="confidence_threshold" settings={settings} />
          <p className="field-help">
            Below this share of agreeing votes the Judge abstains instead of reporting semantic debt.
            {form.structural_only && " Only used when semantic detection is on."}
          </p>
        </div>

        <div className="field">
          <span className="field-label">Project context (RAG)</span>
          <label className="checkbox">
            <input type="checkbox" checked={form.rag_repair} onChange={(e) => set("rag_repair", e.target.checked)} />
            Give the context to the repair
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={form.rag_detection}
              onChange={(e) => set("rag_detection", e.target.checked)}
              disabled={form.structural_only}
            />
            Give the context to the semantic detection
          </label>
          <div className="source-grid">
            {settings.choices.rag_sources.map((key) => {
              const [label, help] = RAG_SOURCES[key] || [key, ""];
              const on = form.rag_sources.includes(key);
              return (
                <label className="checkbox" key={key} title={help}>
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={(e) =>
                      set(
                        "rag_sources",
                        settings.choices.rag_sources.filter((k) => (k === key ? e.target.checked : form.rag_sources.includes(k))),
                      )
                    }
                  />
                  {label}
                </label>
              );
            })}
          </div>
          <label htmlFor="topk">Similar methods per prompt: {form.rag_top_k}</label>
          <input
            id="topk"
            type="range"
            min="1"
            max="5"
            step="1"
            value={form.rag_top_k}
            onChange={(e) => set("rag_top_k", Number(e.target.value))}
          />
          <p className="field-help">
            Excerpts of the repository retrieved for each method (ranking: {settings.rag_embeddings === "bm25" ? "BM25" : `${settings.rag_embeddings} embeddings, BM25 as fallback`})
            and given to the model as reference only. Each repository can turn it off in its policy. Detection prompts
            were screened without context: adding it to detection is experimental.
          </p>
        </div>

        <div className="field">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={capOn}
              onChange={(e) => set("max_total_cost_usd", e.target.checked ? settings.defaults.max_total_cost_usd ?? 5 : null)}
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
            ${settings.spent_usd.toFixed(4)} spent so far, all repositories. Once the cap is reached, new pushes are not
            reviewed and the PR gets a note saying why. Local Ollama models cost nothing.
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
              onClick={() => window.confirm("Forget every change made here and use the environment defaults?") && send("DELETE", undefined, "Back to the environment defaults.")}
            >
              Reset to defaults
            </button>
          )}
          {message && <span className={`feed-result ${message.ok ? "ok" : "error"}`}>{message.text}</span>}
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
