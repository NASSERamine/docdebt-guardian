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
  const { data } = useApi(["/api/status"]);
  const status = data?.[0];
  const model = status?.signed_in && status.model.configured ? status.model.model : "the model chosen on the Model page";

  return (
    <>
      <h1 className="page-title">How it works</h1>
      <p className="lead">
        Every method touched by a pull request goes through the same fixed pipeline: same input, same answer. Exact
        rules and the AI model both look at the whole Javadoc, the project context is always retrieved, every AI answer
        is checked before anyone sees it, and nothing reaches the code until a person accepts it.
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
          kind={KIND.exact}
          title="Context — what does the project already say? (RAG, always on)"
          example={
            <ul className="mini-findings">
              <li>
                <span className="badge source siblings">same file</span> <code>Celsius.toFahrenheit</code>, documented
                next to it
              </li>
              <li>
                <span className="badge source similar">similar method</span> <code>Fahrenheit.toKelvin</code>, found by
                the hybrid search
              </li>
            </ul>
          }
          branch="Each excerpt given to the model is shown on the proposal page, with its rank."
        >
          First, up to three documented methods of the same file (the context that helped most in the research). Then
          the four best excerpts of the repository&apos;s knowledge base — debt-free documented methods, tests, README
          sections and accepted fixes — found by a hybrid search: BM25 keywords and all-MiniLM-L6-v2 vectors (run
          locally, stored in Chroma) fused by rank. No similarity threshold, fixed sizes, and never a method with the
          same name: the answer cannot leak in.
        </Step>

        <Step
          n="3"
          kind={KIND.llm}
          title="Detection — one verdict on the whole Javadoc"
          example={
            <span>
              Rules: two errors. Model: “debt — the @param names a parameter the method does not have”. Verdict: debt.
            </span>
          }
          branch={`Debt when an exact rule fails OR ${model} says so — one call, temperature 0, no threshold.`}
        >
          The exact rules of step 1 and the AI model judge the same thing: is this Javadoc still true for this code? The
          model answers “debt” or “clean” with the research&apos;s fixed prompt, then explains what is wrong; the
          sentences and code it quotes are highlighted when they are found word for word in the method. It is a
          proposal, never an automatic change: the person decides.
        </Step>

        <Step
          n="4"
          kind={KIND.llm}
          title="Fix — field-by-field replacement"
          example={
            <pre className="code-block small diff">
              <div className="diff-line del">- * @param tempInFahrenheit temperature in celsius</div>
              <div className="diff-line add">+ * @param tempInCelsius temperature in celsius</div>
            </pre>
          }
          branch="No debt at steps 1 and 3? The method is clean: nothing is posted. Want another version? “Regenerate” asks the model for a full rewrite to compare."
        >
          A deterministic plan handles the structure first (a renamed parameter keeps its description, a phantom tag is
          deleted). The model then only writes the missing fields and judges the disputed ones. A post-validation
          reverts invented identifiers and cosmetic edits, and every untouched line is copied byte for byte, so the
          diff stays as small as possible. A full regeneration is computed only when the reviewer asks for it.
        </Step>

        <Step
          n="5"
          kind={KIND.check}
          title="Verifier — check the fix before anyone sees it"
          example={<span>No structural error left, the other four lines unchanged: verified in 1 attempt.</span>}
          branch="Rejected? The model gets the reasons and tries again. Still wrong after the last attempt: nothing is proposed — a bad fix is never shown."
        >
          Runs the Auditor again on the fixed Javadoc, checks every requested field is present and not empty, that a
          contradicted sentence is really gone, and that no other line was touched.
        </Step>

        <Step
          n="6"
          kind={KIND.human}
          title="Human review — accept, edit, regenerate or reject"
          example={
            <span>
              Accept → commit <code>docdebt: fix documentation for Celsius.toKelvin</code> on the PR branch.
            </span>
          }
        >
          The proposal appears as a comment on the pull request and in this application, with the explanation and the
          three versions side by side (current, field-by-field fix, regeneration). A person accepts one of them, edits
          the fix before accepting (the edit is checked by the exact rules), or rejects with a reason (nothing
          changes). A <strong>DocDebt Guardian</strong> check run sums it up on the commit and can block the merge while
          a proposal is pending. Every decision feeds the observed precision, and accepted fixes become examples in the
          knowledge base.
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
            <div className="guarantee-title">Deterministic</div>
            <p className="field-help">One model, fixed prompts, temperature 0, fixed context sizes, no threshold.</p>
          </div>
          <div className="tile">
            <div className="guarantee-title">Minimal, verified diffs</div>
            <p className="field-help">Only the faulty fields change, and only after the Verifier accepts them.</p>
          </div>
        </div>
      </section>

      <p className="field-help how-next">
        See it on real data: <Link to="/review">To review</Link>, or <Link to="/proposals">History</Link> →{" "}
        <strong>Review</strong> on any proposal.
      </p>
    </>
  );
}
