import { Link } from "react-router-dom";
import { useApi } from "../hooks.js";

// One running example, followed through every step: what the jury sees on a
// real proposal page is exactly this.
const EXAMPLE_BEFORE = `/**
 * Converts celsius to kelvin.
 * @param tempInFahrenheit temperature in celsius
 * @return temperature in kelvin
 */
public double toKelvin(double tempInCelsius) {
    return tempInCelsius + 273.15;
}`;

function Step({ n, kind, title, children, example, branch }) {
  return (
    <li className="step">
      <div className="step-rail">
        <span className={`step-node ${kind.cls}`}>{n}</span>
      </div>
      <div className="step-card">
        <div className="step-head">
          <h3>{title}</h3>
          <span className={`step-kind ${kind.cls}`}>{kind.label}</span>
        </div>
        <div className="step-text">{children}</div>
        {example && (
          <div className="step-example">
            <span className="quote-label">On the example</span>
            {example}
          </div>
        )}
        {branch && <div className="step-branch">↳ {branch}</div>}
      </div>
    </li>
  );
}

const KIND = {
  trigger: { cls: "trigger", label: "trigger" },
  exact: { cls: "exact", label: "deterministic · no AI" },
  llm: { cls: "llm", label: "AI model" },
  check: { cls: "exact", label: "deterministic check" },
  human: { cls: "human", label: "a person decides" },
};

export default function HowItWorks() {
  const { data } = useApi(["/api/settings"]);
  const runtime = data?.[0]?.effective;
  const judgeOn = runtime ? !runtime.structural_only : null;
  const model = runtime?.repair_model ?? "the configured model";

  return (
    <>
      <h1 className="page-title">How it works</h1>
      <p className="lead">
        Every method touched by a pull request goes through the same pipeline. Exact rules run first, the AI model only
        writes and judges where rules cannot, and every AI answer is checked before anyone sees it. Nothing reaches the
        code until a person accepts it.
      </p>

      <section className="problem">
        <div>
          <h2 className="section-title">The problem: documentation debt</h2>
          <p className="step-text">
            A method changes, its Javadoc does not. Here the parameter was renamed to <code>tempInCelsius</code> but the
            comment still documents <code>tempInFahrenheit</code>: anyone reading the documentation is misled, and no
            compiler notices.
          </p>
        </div>
        <pre className="code-block small">{EXAMPLE_BEFORE}</pre>
      </section>

      <ol className="pipeline">
        <Step
          n="0"
          kind={KIND.trigger}
          title="A pull request changes Java code"
          example={<span>The PR adds <code>Celsius.toKelvin</code>.</span>}
        >
          GitHub notifies DocDebt on every push (or you press <strong>Scan now</strong> on a repository). Only the
          methods the pull request actually touches are analysed, not the whole repository.
        </Step>

        <Step
          n="1"
          kind={KIND.exact}
          title="Auditor — exact structural rules"
          example={
            <ul className="mini-findings">
              <li>
                <span className="rule-chip">unknown-param</span> <code>tempInFahrenheit</code> is not a parameter
              </li>
              <li>
                <span className="rule-chip">missing-param</span> <code>tempInCelsius</code> is not documented
              </li>
            </ul>
          }
          branch="These errors are certain: no AI answer can cancel them."
        >
          Compares the Javadoc with the method signature: every parameter documented, no phantom parameter,{" "}
          <code>@return</code> when something is returned, every declared exception documented. Instant, free, and
          always right.
        </Step>

        <Step
          n="2"
          kind={KIND.llm}
          title="Judge — does the description still match the code?"
          example={<span>“Converts celsius to kelvin” matches <code>tempInCelsius + 273.15</code>: no semantic debt.</span>}
          branch={
            judgeOn === null
              ? "Can be switched on or off in Settings."
              : judgeOn
                ? `Currently ON (${model}). A local model answers first; a stronger one is asked only when it is unsure, if allowed.`
                : "Currently OFF (structural checks only): switch it on in Settings."
          }
        >
          Rules cannot read sentences. The AI model votes “debt” or “clean” and must quote the exact sentence that is
          wrong and the code that contradicts it. Both quotes are checked word for word against the real file: an
          invented quote is thrown away, so the model can miss an alert but never invent one.
        </Step>

        <Step
          n="3"
          kind={KIND.llm}
          title="Repairer — write the fix"
          example={
            <pre className="code-block small diff">
              <div className="diff-line del">- * @param tempInFahrenheit temperature in celsius</div>
              <div className="diff-line add">+ * @param tempInCelsius temperature in celsius</div>
            </pre>
          }
          branch="No debt at steps 1–2? The method is clean: nothing is posted."
        >
          {model} rewrites the Javadoc. DocDebt then keeps <em>only</em> the fields that had to change and copies every
          other line from the original, byte for byte, so the proposed diff stays as small as possible.
        </Step>

        <Step
          n="4"
          kind={KIND.check}
          title="Verifier — check the fix before anyone sees it"
          example={<span>No structural error left, the other four lines unchanged: verified in 1 attempt.</span>}
          branch="Rejected? The model gets the reasons and tries again. Still wrong after the last attempt: nothing is proposed — a bad fix is never shown."
        >
          Runs the Auditor again on the fixed Javadoc, checks every requested field is present and not empty, that a
          contradicted sentence is really gone, and that no other line was touched.
        </Step>

        <Step
          n="5"
          kind={KIND.human}
          title="Human review — accept or reject"
          example={
            <span>
              Accept → commit <code>docdebt: fix documentation for Celsius.toKelvin</code> on the PR branch.
            </span>
          }
        >
          The proposal appears as a comment on the pull request and in this dashboard, with the debt and the diff. A
          person accepts it (DocDebt commits the fix to the PR branch) or rejects it (nothing changes). A decision is
          remembered: the same proposal is not posted twice.
        </Step>
      </ol>

      <section>
        <h2 className="section-title">What DocDebt guarantees</h2>
        <div className="guarantees">
          <div className="tile">
            <div className="guarantee-title">No commit without a person</div>
            <p className="field-help">Every fix waits for Accept. Comments written by bots are never read as commands.</p>
          </div>
          <div className="tile">
            <div className="guarantee-title">Rules before AI</div>
            <p className="field-help">Exact checks decide first; the model only adds what rules cannot see.</p>
          </div>
          <div className="tile">
            <div className="guarantee-title">No invented alerts</div>
            <p className="field-help">A semantic alert needs two quotes found word for word in the real code.</p>
          </div>
          <div className="tile">
            <div className="guarantee-title">Minimal, verified diffs</div>
            <p className="field-help">Only the faulty fields change, and only after the Verifier accepts them.</p>
          </div>
        </div>
      </section>

      <p className="field-help how-next">
        See it on real data: <Link to="/proposals">History</Link> → <strong>Review</strong> on any proposal.
      </p>
    </>
  );
}
