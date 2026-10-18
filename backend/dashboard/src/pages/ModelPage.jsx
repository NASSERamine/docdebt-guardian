import { useEffect, useState } from "react";
import { errorText, sendWithToken } from "../api.js";
import { ErrorNote, StateNote, formatDate } from "../components.jsx";
import { useApi } from "../hooks.js";

// The Model page: choose one of the offered models, paste its API key once,
// test it, save it. The key is stored encrypted by the service and never comes
// back to the browser (only its last four characters).

const KEY_HELP = {
  openai: "An OpenAI API key (platform.openai.com → API keys), starting with sk-.",
  anthropic: "An Anthropic API key (console.anthropic.com → API keys), starting with sk-ant-.",
  google: "A Google AI Studio API key (aistudio.google.com → Get API key).",
};

function ModelForm({ state, onSaved }) {
  const saved = state.config;
  const defaultModel = state.choices.find((c) => c.default)?.id;
  const [model, setModel] = useState(saved?.model || defaultModel);
  const [apiKey, setApiKey] = useState("");
  const [priceIn, setPriceIn] = useState(saved?.price_in_per_m ?? "");
  const [priceOut, setPriceOut] = useState(saved?.price_out_per_m ?? "");
  const [busy, setBusy] = useState(null);
  const [message, setMessage] = useState(null);
  useEffect(() => {
    setModel(saved?.model || defaultModel);
    setPriceIn(saved?.price_in_per_m ?? "");
    setPriceOut(saved?.price_out_per_m ?? "");
  }, [saved?.model, saved?.price_in_per_m, saved?.price_out_per_m, defaultModel]);

  const choice = state.choices.find((c) => c.id === model);
  const providers = [...new Set(state.choices.map((c) => c.provider_label))];
  const payload = () => ({
    model,
    ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
    price_in_per_m: priceIn === "" ? null : Number(priceIn),
    price_out_per_m: priceOut === "" ? null : Number(priceOut),
  });
  const keyNeeded = !saved && !apiKey.trim();

  async function run(kind) {
    setBusy(kind);
    setMessage(null);
    try {
      const { status, body } =
        kind === "test"
          ? await sendWithToken("POST", "/api/llm-config/test", payload())
          : kind === "delete"
            ? await sendWithToken("DELETE", "/api/llm-config")
            : await sendWithToken("PUT", "/api/llm-config", payload());
      if (status === 200 && kind === "test") {
        setMessage(body.ok ? { ok: true, text: "Connection successful: the provider answered." } : { ok: false, text: body.error });
      } else if (status === 200) {
        setApiKey("");
        setMessage({ ok: true, text: kind === "delete" ? "Configuration removed." : "Connection successful — saved. The next pull request uses this model." });
        onSaved();
      } else if (status !== 0) setMessage({ ok: false, text: errorText(body, status) });
    } catch (e) {
      setMessage({ ok: false, text: `Could not reach the service: ${e}` });
    } finally {
      setBusy(null);
    }
  }

  return (
    <form
      className="settings-form"
      onSubmit={(e) => {
        e.preventDefault();
        run("save");
      }}
    >
      {!state.editable && (
        <p className="notice">
          Read-only: sign in with GitHub, or set <span className="mono">DOCDEBT_DASHBOARD_TOKEN</span>, to choose a model.
        </p>
      )}
      <fieldset disabled={!state.editable || busy !== null}>
        <div className="field">
          <label htmlFor="model">Model</label>
          <select id="model" className="filter-input" value={model} onChange={(e) => setModel(e.target.value)}>
            {providers.map((provider) => (
              <optgroup key={provider} label={provider}>
                {state.choices
                  .filter((c) => c.provider_label === provider)
                  .map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label}
                      {c.default ? " (default)" : ""}
                    </option>
                  ))}
              </optgroup>
            ))}
          </select>
          <p className="field-help">
            One model for everything: detection on the whole Javadoc, the field-by-field fix and, on demand, the
            regeneration. Temperature 0 and a fixed prompt per model, so the same method gives the same answer.
          </p>
        </div>

        <div className="field">
          <label htmlFor="api-key">API key {saved && <span className="field-hint">saved: {saved.key_hint}</span>}</label>
          <input
            id="api-key"
            type="password"
            className="filter-input key-input"
            autoComplete="off"
            placeholder={saved ? "Leave empty to keep the saved key" : "Paste the API key"}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <p className="field-help">
            {KEY_HELP[choice?.provider]} Stored encrypted on the service; never shown again, never sent to the browser.
          </p>
        </div>

        <div className="field">
          <span className="field-label">Prices (optional, US$ per million tokens)</span>
          <span className="price-inputs">
            <label>
              input
              <input type="number" min="0" step="0.01" className="filter-input" value={priceIn} onChange={(e) => setPriceIn(e.target.value)} />
            </label>
            <label>
              output
              <input type="number" min="0" step="0.01" className="filter-input" value={priceOut} onChange={(e) => setPriceOut(e.target.value)} />
            </label>
          </span>
          <p className="field-help">From the provider&apos;s price list: they give the cost of each proposal, pull request and month.</p>
        </div>

        <div className="form-actions">
          <button type="button" className="page-btn" disabled={keyNeeded} onClick={() => run("test")}>
            {busy === "test" ? "Testing…" : "Test the connection"}
          </button>
          <button type="submit" className="primary-btn" disabled={keyNeeded}>
            {busy === "save" ? "Testing and saving…" : "Save"}
          </button>
          {saved && (
            <button
              type="button"
              className="page-btn"
              onClick={() => window.confirm("Remove the saved model and key? Reviews then run the exact rules only.") && run("delete")}
            >
              Remove
            </button>
          )}
        </div>
        {message && (message.ok ? <span className="feed-result ok">{message.text}</span> : <ErrorNote text={message.text} />)}
      </fieldset>
    </form>
  );
}

function CurrentModel({ config, choices }) {
  if (!config) {
    return (
      <p className="state-note">
        No model saved yet: pull requests are checked with the exact rules only, and no fix can be proposed. Choose a
        model and paste its key.
      </p>
    );
  }
  const label = choices.find((c) => c.id === config.model)?.label || config.model;
  return (
    <div className="current-model">
      <span className={`dot ${config.last_test_ok === false ? "disconnected" : "connected"}`} />
      <strong>{label}</strong> · key {config.key_hint}
      {config.last_test_at && <span className="muted"> · last checked {formatDate(config.last_test_at)}</span>}
      {config.last_test_ok === false && (
        <div>
          <ErrorNote text={`The last call failed: ${config.last_test_error}. Check the API key on the Model page.`} />
        </div>
      )}
    </div>
  );
}

export default function ModelPage() {
  const { data, error, loading, refresh } = useApi(["/api/llm-config"]);
  return (
    <>
      <h1 className="page-title">Model</h1>
      <StateNote error={!data && error} loading={!data && loading}>
        {data && (
          <div className="columns">
            <div className="panel">
              <h2>Choose the model</h2>
              <ModelForm state={data[0]} onSaved={refresh} />
            </div>
            <div className="panel">
              <h2>In use</h2>
              <CurrentModel config={data[0].config} choices={data[0].choices} />
              <p className="field-help">
                GPT-5.6 Luna is the default. GPT-5.6, Claude Sonnet 5 and Gemini 3.1 Pro are the API models of the
                research behind DocDebt; Luna uses the prompts screened for GPT-5.6 (same family). The key is tested
                before it is saved: a refused key is never stored.
              </p>
            </div>
          </div>
        )}
      </StateNote>
    </>
  );
}
