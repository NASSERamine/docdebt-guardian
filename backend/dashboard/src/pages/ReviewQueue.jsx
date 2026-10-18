import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { StateNote } from "../components.jsx";
import { useApi } from "../hooks.js";
import { ProposalView } from "./ProposalDetail.jsx";

// "To review" (U8): every pending proposal of every repository, one after the
// other. A accepts the field-by-field fix, G regenerates, E edits, R opens the
// reject menu, ← / → move between proposals. The queue is reloaded after each
// decision, so a decided proposal leaves it.

function isTyping(event) {
  const tag = event.target?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target?.isContentEditable;
}

export default function ReviewQueue() {
  const { data, error, loading, refresh } = useApi(["/api/review-queue"]);
  const queue = data?.[0] || [];
  const [index, setIndex] = useState(0);
  const actionsRef = useRef(null);
  const safeIndex = Math.min(index, Math.max(0, queue.length - 1));
  const current = queue[safeIndex];

  const move = useCallback((step) => setIndex((i) => Math.max(0, Math.min(queue.length - 1, i + step))), [queue.length]);

  useEffect(() => {
    function onKey(event) {
      if (isTyping(event) || event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key.toLowerCase();
      if (key === "arrowright" || key === "j") move(1);
      else if (key === "arrowleft" || key === "k") move(-1);
      else if (key === "a") actionsRef.current?.accept();
      else if (key === "g") actionsRef.current?.regenerate();
      else if (key === "e") {
        event.preventDefault();
        actionsRef.current?.edit();
      } else if (key === "r") document.querySelector(".review-current .reject-menu button")?.click();
      else return;
      event.preventDefault();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [move]);

  return (
    <>
      <div className="page-heading">
        <h1 className="page-title">To review</h1>
        {queue.length > 0 && (
          <span className="queue-nav">
            <button type="button" className="page-btn" disabled={safeIndex === 0} onClick={() => move(-1)} title="Shortcut: ←">
              ← Previous
            </button>
            <span className="mono">
              {safeIndex + 1} / {queue.length}
            </span>
            <button
              type="button"
              className="page-btn"
              disabled={safeIndex >= queue.length - 1}
              onClick={() => move(1)}
              title="Shortcut: →"
            >
              Next →
            </button>
          </span>
        )}
      </div>
      <p className="field-help shortcuts">
        Shortcuts: <kbd>A</kbd> accept the fix · <kbd>R</kbd> reject (with a reason) · <kbd>G</kbd> regenerate ·{" "}
        <kbd>E</kbd> edit, then accept · <kbd>←</kbd> <kbd>→</kbd> previous / next
      </p>
      <StateNote error={!data && error} loading={!data && loading}>
        {queue.length === 0 ? (
          <div className="empty-state">
            <p>Nothing waiting for a decision. New proposals appear here as soon as a pull request is reviewed.</p>
            <Link className="connect-btn" to="/proposals">
              See the history
            </Link>
          </div>
        ) : (
          current && (
            <div className="review-current" key={current.short_id}>
              <ProposalView shortId={current.short_id} onDecided={refresh} actionsRef={actionsRef} />
            </div>
          )
        )}
      </StateNote>
    </>
  );
}
