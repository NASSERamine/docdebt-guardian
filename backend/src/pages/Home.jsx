import { Link, useSearchParams } from "react-router-dom";
import { canUseApp, useAuth } from "../auth.jsx";

const LOGIN_ERRORS = {
  state: "The sign-in took too long or was opened in another tab. Try again.",
  github: "GitHub refused the sign-in. Try again; if it keeps failing, check the App's client ID and secret.",
};

function GithubMark() {
  return (
    <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

export function SignInButton({ big = false }) {
  // A full page load: the server redirects to GitHub and back.
  return (
    <a className={`github-btn ${big ? "big" : ""}`} href="/auth/login">
      <GithubMark /> Sign in with GitHub
    </a>
  );
}

export default function Home() {
  const { me } = useAuth();
  const [params] = useSearchParams();
  const loginError = LOGIN_ERRORS[params.get("login_error")];
  const inside = canUseApp(me);

  return (
    <div className="home">
      {loginError && <p className="notice">{loginError}</p>}

      <section className="hero">
        <p className="eyebrow">Documentation debt, caught in review</p>
        <h1>Keep your Javadoc true to your code.</h1>
        <p className="hero-sub">
          DocDebt Guardian reads every pull request, finds the Java documentation that no longer matches the code, and
          proposes a fix it has verified. You accept or reject it in one click; nothing changes without you.
        </p>
        <div className="hero-actions">
          {inside ? (
            <Link className="primary-btn big" to="/overview">
              Open the dashboard →
            </Link>
          ) : me?.auth_enabled ? (
            <SignInButton big />
          ) : null}
          <Link className="page-btn big" to="/how-it-works">
            How it works
          </Link>
        </div>
        {me?.user && (
          <p className="field-help">
            Signed in as <strong>{me.user.login}</strong>.
          </p>
        )}
      </section>

      <section className="features">
        <div className="tile">
          <div className="guarantee-title">Finds what drifted</div>
          <p className="field-help">
            A renamed parameter still documented under its old name, a missing <code>@return</code>, a description the
            code now contradicts.
          </p>
        </div>
        <div className="tile">
          <div className="guarantee-title">Fixes it, checked</div>
          <p className="field-help">
            An AI model writes the correction; exact checks verify it before you ever see it. Only the wrong lines
            change.
          </p>
        </div>
        <div className="tile">
          <div className="guarantee-title">You decide</div>
          <p className="field-help">
            Every fix waits for your Accept, on the pull request or here. Accepted fixes are committed to the PR branch
            for you.
          </p>
        </div>
      </section>

      <section>
        <h2 className="section-title">Start in three steps</h2>
        <ol className="start-steps">
          <li>
            <span className="step-node exact">1</span>
            <div>
              <strong>Sign in with GitHub.</strong>
              <p className="field-help">Your repositories where DocDebt is installed and you can push are added for you.</p>
            </div>
          </li>
          <li>
            <span className="step-node llm">2</span>
            <div>
              <strong>Add a repository.</strong>
              <p className="field-help">
                Type <code>owner/name</code> or paste its GitHub URL; install the DocDebt App on it if asked.
              </p>
            </div>
          </li>
          <li>
            <span className="step-node human">3</span>
            <div>
              <strong>Open a pull request, or press Scan now.</strong>
              <p className="field-help">Review each proposal — the debt, the fix — and accept or reject it.</p>
            </div>
          </li>
        </ol>
      </section>
    </div>
  );
}
