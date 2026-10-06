import { Link, NavLink, Route, Routes } from "react-router-dom";
import { RequireUser, canUseApp, useAuth } from "./auth.jsx";
import History from "./pages/History.jsx";
import Home, { SignInButton } from "./pages/Home.jsx";
import HowItWorks from "./pages/HowItWorks.jsx";
import Overview from "./pages/Overview.jsx";
import ProposalDetail from "./pages/ProposalDetail.jsx";
import RepoDetail from "./pages/RepoDetail.jsx";
import Repositories from "./pages/Repositories.jsx";
import Settings from "./pages/Settings.jsx";

function NotFound() {
  return (
    <p className="state-note">
      This page does not exist. <Link to="/">Back to the home page</Link>
    </p>
  );
}

function UserMenu() {
  const { me, signOut } = useAuth();
  if (!me?.auth_enabled) return null;
  if (!me.user) return <SignInButton />;
  return (
    <span className="user-menu">
      {me.user.avatar_url && <img className="avatar" src={me.user.avatar_url} alt="" />}
      <span className="user-login">{me.user.login}</span>
      <button type="button" className="page-btn" onClick={signOut}>
        Sign out
      </button>
    </span>
  );
}

// Every page of the application itself needs a signed-in user (or local mode).
const app = (element) => <RequireUser>{element}</RequireUser>;

export default function App() {
  const { me } = useAuth();
  const inside = canUseApp(me);
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          DocDebt Guardian
        </Link>
        <nav className="nav">
          {inside && (
            <>
              <NavLink to="/overview">Dashboard</NavLink>
              <NavLink to="/repositories">Repositories</NavLink>
              <NavLink to="/proposals">History</NavLink>
              <NavLink to="/settings">Settings</NavLink>
            </>
          )}
          <NavLink to="/how-it-works">How it works</NavLink>
        </nav>
        <UserMenu />
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/overview" element={app(<Overview />)} />
          <Route path="/repositories" element={app(<Repositories />)} />
          <Route path="/proposals" element={app(<History />)} />
          <Route path="/proposals/:shortId" element={app(<ProposalDetail />)} />
          <Route path="/repos/:owner/:repo" element={app(<RepoDetail />)} />
          <Route path="/settings" element={app(<Settings />)} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}
