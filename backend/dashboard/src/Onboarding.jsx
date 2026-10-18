import { Link } from "react-router-dom";

// The getting-started guide (U1): five steps, shown on the dashboard until
// every one is done. Each step says what to do and links to where to do it.

const STEPS = {
  sign_in: { title: "Sign in with GitHub", help: "Done — you are in.", to: null },
  install_app: {
    title: "Install the App on a repository",
    help: "Add the repository by its name on the Repositories page; DocDebt gives you the GitHub installation link if needed.",
    to: "/repositories",
    action: "Add a repository",
  },
  configure_model: {
    title: "Choose the model and paste its API key",
    help: "GPT-5.6 Luna by default. The key is tested before it is saved.",
    to: "/model",
    action: "Open the Model page",
  },
  index_knowledge: {
    title: "Index the knowledge base",
    help: "On the repository page, “Analyse the default branch”: measures the documentation health and indexes the project for retrieval.",
    to: "/repositories",
    action: "Choose a repository",
  },
  first_pull_request: {
    title: "Open a pull request",
    help: "Change a Java method whose Javadoc is out of date (for example rename a parameter): the proposal appears on the PR and here.",
    to: null,
  },
};

export default function Onboarding({ steps }) {
  if (!steps || steps.every((s) => s.done)) return null;
  const doneCount = steps.filter((s) => s.done).length;
  const next = steps.find((s) => !s.done)?.key;
  return (
    <section className="panel onboarding">
      <div className="panel-heading">
        <h2>Getting started</h2>
        <span className="field-hint">
          {doneCount} of {steps.length} done
        </span>
      </div>
      <ol className="onboarding-steps">
        {steps.map((s, i) => {
          const info = STEPS[s.key];
          return (
            <li key={s.key} className={s.done ? "done" : s.key === next ? "next" : ""}>
              <span className="step-mark">{s.done ? "✓" : i + 1}</span>
              <div>
                <strong>{info.title}</strong>
                {!s.done && <p className="field-help">{info.help}</p>}
              </div>
              {!s.done && info.to && s.key === next && (
                <Link className="primary-btn" to={info.to}>
                  {info.action}
                </Link>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
