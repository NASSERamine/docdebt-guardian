import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { getJson, setAuthMode } from "./api.js";

// Who is signed in. In local mode (no GitHub OAuth configured on the service)
// there is no sign-in: everything is open to the single operator.
const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [me, setMe] = useState(null); // { auth_enabled, user } once loaded
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const next = await getJson("/api/me");
      setAuthMode(next.auth_enabled);
      setMe(next);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const signOut = useCallback(async () => {
    await fetch("/auth/logout", { method: "POST" });
    await refresh();
  }, [refresh]);

  return <AuthContext.Provider value={{ me, error, refresh, signOut }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}

// The dashboard is open in local mode, and for a signed-in user otherwise.
export function canUseApp(me) {
  return Boolean(me && (!me.auth_enabled || me.user));
}

export function RequireUser({ children }) {
  const { me, error } = useAuth();
  const location = useLocation();
  if (error) return <p className="state-note">Could not reach the service: {error}</p>;
  if (!me) return <p className="state-note">Loading…</p>;
  if (!canUseApp(me)) return <Navigate to="/" replace state={{ from: location.pathname }} />;
  return children;
}
