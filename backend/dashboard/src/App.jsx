import { Link, NavLink, Route, Routes } from "react-router-dom";
import History from "./pages/History.jsx";
import Overview from "./pages/Overview.jsx";
import ProposalDetail from "./pages/ProposalDetail.jsx";
import RepoDetail from "./pages/RepoDetail.jsx";
import Settings from "./pages/Settings.jsx";

function NotFound() {
  return (
    <p className="state-note">
      This page does not exist. <Link to="/">Back to the overview</Link>
    </p>
  );
}

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          DocDebt Guardian
        </Link>
        <nav className="nav">
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/proposals">History</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/proposals" element={<History />} />
          <Route path="/proposals/:shortId" element={<ProposalDetail />} />
          <Route path="/repos/:owner/:repo" element={<RepoDetail />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}
