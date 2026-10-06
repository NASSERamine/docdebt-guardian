import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";

function formatDay(iso) {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div
      style={{
        background: "var(--surface-raised)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        padding: "6px 10px",
        fontSize: 12,
        color: "var(--ink)",
      }}
    >
      <div className="mono">{formatDay(label)}</div>
      <div style={{ color: "var(--ink-dim)" }}>
        {row.calls} call{row.calls === 1 ? "" : "s"} · ${row.cost_usd.toFixed(4)}
      </div>
    </div>
  );
}

export default function UsageChart({ series }) {
  const totalCalls = series.reduce((sum, row) => sum + row.calls, 0);
  const totalCost = series.reduce((sum, row) => sum + row.cost_usd, 0);

  return (
    <div className="usage-panel">
      <div className="usage-header">
        <h2>LLM usage, last {series.length} days</h2>
        <span className="total">
          {totalCalls} calls · ${totalCost.toFixed(4)}
        </span>
      </div>
      {totalCalls === 0 ? (
        <p className="state-note">No LLM call recorded yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={72}>
          <BarChart data={series} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" tickFormatter={formatDay} tick={{ fontSize: 11, fill: "var(--ink-faint)" }} axisLine={{ stroke: "var(--border)" }} tickLine={false} interval={Math.ceil(series.length / 7) - 1} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--surface)" }} />
            <Bar dataKey="calls" fill="var(--action)" radius={[2, 2, 0, 0]} maxBarSize={18} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
