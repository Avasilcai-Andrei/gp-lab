import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from "recharts";

const API = "http://localhost:8000";

const TEAM_COLORS = {
  "Red Bull Racing": "#3671C6",
  "Mercedes": "#27F4D2",
  "Ferrari": "#E8002D",
  "McLaren": "#FF8000",
  "Aston Martin": "#358C75",
  "Alpine": "#FF87BC",
  "Williams": "#64C4FF",
  "AlphaTauri": "#5E8FAA",
  "Alfa Romeo": "#C92D4B",
  "Haas F1 Team": "#B6BABD",
};

const DRIVER_PRESET = [
  { driver: "VER", team: "Red Bull Racing", grid: 7 },
  { driver: "HAD", team: "Red Bull Racing", grid: 8 },
  { driver: "ANT", team: "Mercedes", grid: 2 },
  { driver: "RUS", team: "Mercedes", grid: 1 },
  { driver: "LEC", team: "Ferrari", grid: 3 },
  { driver: "HAM", team: "Ferrari", grid: 4 },
  { driver: "NOR", team: "McLaren", grid: 5 },
  { driver: "PIA", team: "McLaren", grid: 6 },
  { driver: "ALO", team: "Aston Martin", grid: 9 },
  { driver: "STR", team: "Aston Martin", grid: 10 },
];

function PosBadge({ pos }) {
  const cls = pos === 1 ? "pos-1" : pos === 2 ? "pos-2" : pos === 3 ? "pos-3" : "pos-other";
  return <span className={`pos-badge ${cls}`}>{pos}</span>;
}

function ProbBar({ value, color = "#00d2be" }) {
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar-bg">
        <div className="prob-bar-fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "0.75rem", color: "#999", minWidth: "36px" }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export default function PredictorView() {
  const [year, setYear] = useState("2026");
  const [gp, setGp] = useState("Monza");
  const [grid, setGrid] = useState(DRIVER_PRESET);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelStatus, setModelStatus] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainYears, setTrainYears] = useState("2022,2023,2024,2025,2026");

  const updateDriver = (entryToUpdate, field, val) => {
    setGrid(g => g.map(d => d === entryToUpdate ? { ...d, [field]: field === "grid" ? parseInt(val) || d.grid : val } : d));
  };

  const predict = async () => {
    setLoading(true); setError(""); setResults(null);
    try {
      const res = await fetch(`${API}/predict/race`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          year: parseInt(year),
          grand_prix: gp,
          qualifying_results: grid,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      const json = await res.json();
      setResults(json.predictions);
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const checkStatus = async () => {
    const res = await fetch(`${API}/predict/status`);
    setModelStatus(await res.json());
  };

  const triggerTrain = async () => {
    setTraining(true);
    await fetch(`${API}/predict/train?years=${trainYears}`, { method: "POST" });
    setTimeout(checkStatus, 2000);
    setTraining(false);
  };

  const podiumData = results?.slice(0, 5).map(r => ({
    name: r.driver,
    podium: Math.round(r.podium_probability * 100),
    points: Math.round(r.points_probability * 100),
    color: TEAM_COLORS[r.team] || "#666",
  }));

  return (
    <div>
      <div className="section-title">Race Predictor</div>
      <div className="section-subtitle">ml model · grid-based prediction · driver form · team performance</div>

      <div style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <div className="control-group">
          <span className="control-label">Season</span>
          <input className="control-input" style={{ width: "80px" }} value={year} onChange={e => setYear(e.target.value)} />
        </div>
        <div className="control-group">
          <span className="control-label">Grand Prix</span>
          <input className="control-input" style={{ width: "160px" }} value={gp} onChange={e => setGp(e.target.value)} />
        </div>
        <button className="btn btn-primary" onClick={predict} disabled={loading}>
          {loading ? "Predicting..." : "▶ Run Prediction"}
        </button>
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-end" }}>
          <div className="control-group">
            <span className="control-label">Train on years</span>
            <input className="control-input" style={{ width: "120px" }} value={trainYears} onChange={e => setTrainYears(e.target.value)} placeholder="2022,2023" />
          </div>
          <button className="btn btn-outline" onClick={triggerTrain} disabled={training}>{training ? "Starting..." : "Train Model"}</button>
          <button className="btn btn-outline" onClick={checkStatus}>Model Status</button>
        </div>
      </div>

      {modelStatus && (
        <div style={{ marginBottom: "1rem", padding: "0.75rem 1rem", background: "#111", border: "1px solid #2a2a2a", borderRadius: 4, fontFamily: "IBM Plex Mono, monospace", fontSize: "0.75rem" }}>
          {modelStatus.trained ? (
            <span style={{ color: "#00d2be" }}>
              ✓ Model trained · {modelStatus.training_races} races · {modelStatus.years_covered?.join(", ")}
            </span>
          ) : (
            <span style={{ color: "#e8002d" }}>⚠ {modelStatus.message}</span>
          )}
        </div>
      )}

      {error && <div className="error-msg">⚠ {error}</div>}

      <div className="grid-2" style={{ gap: "1.5rem" }}>
        <div className="card">
          <div className="card-title">Qualifying Grid</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
            {grid
              .slice()
              .sort((a, b) => a.grid - b.grid)
              .map((entry, i) => (
              <div key={i} style={{
                display: "flex", gap: "0.5rem", alignItems: "center",
                padding: "0.4rem 0.5rem", borderRadius: 2,
                background: i % 2 === 0 ? "#141414" : "transparent",
              }}>
                <span style={{
                  fontFamily: "Bebas Neue, sans-serif", fontSize: "1.1rem",
                  color: "#444", width: "20px", textAlign: "center"
                }}>{i + 1}</span>
                <input
                  className="control-input"
                  style={{ width: "55px", padding: "0.3rem 0.5rem", fontSize: "0.8rem" }}
                  value={entry.driver}
                  onChange={e => updateDriver(entry, "driver", e.target.value)}
                />
                <input
                  className="control-input"
                  style={{ flex: 1, padding: "0.3rem 0.5rem", fontSize: "0.75rem" }}
                  value={entry.team}
                  onChange={e => updateDriver(entry, "team", e.target.value)}
                />
                <div style={{
                  width: 10, height: 10, borderRadius: "50%",
                  background: TEAM_COLORS[entry.team] || "#444",
                  flexShrink: 0
                }} />
              </div>
            ))}
          </div>
          <button
            className="btn btn-outline"
            style={{ marginTop: "0.75rem", width: "100%", fontSize: "0.7rem" }}
            onClick={() => setGrid(g => [...g, { driver: "NEW", team: "Unknown", grid: g.length + 1 }])}
          >
            + Add Driver
          </button>
        </div>

        <div>
          {results ? (
            <>
              <div className="card" style={{ marginBottom: "1rem" }}>
                <div className="card-title">Predicted Finishing Order</div>
                <table className="pred-table">
                  <thead>
                    <tr>
                      <th>Pos</th>
                      <th>Driver</th>
                      <th>Team</th>
                      <th>Grid</th>
                      <th>Podium %</th>
                      <th>Points %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr key={r.driver}>
                        <td><PosBadge pos={i + 1} /></td>
                        <td>
                          <span style={{ fontFamily: "Bebas Neue, sans-serif", fontSize: "1.1rem", letterSpacing: "0.05em" }}>
                            {r.driver}
                          </span>
                        </td>
                        <td>
                          <span style={{ color: TEAM_COLORS[r.team] || "#666", fontSize: "0.78rem" }}>
                            {r.team}
                          </span>
                        </td>
                        <td style={{ color: "#666" }}>{r.grid_position}</td>
                        <td style={{ minWidth: "120px" }}><ProbBar value={r.podium_probability} color="#e8002d" /></td>
                        <td style={{ minWidth: "120px" }}><ProbBar value={r.points_probability} color="#00d2be" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="card">
                <div className="card-title">Podium Probability — Top 5</div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={podiumData} margin={{ top: 4, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontFamily: "Bebas Neue", fontSize: 14 }} stroke="#333" />
                    <YAxis tick={{ fontSize: 10 }} stroke="#333" tickFormatter={v => `${v}%`} />
                    <Tooltip
                      cursor={{ fill: "rgba(255,255,255,0.04)" }}
                      contentStyle={{ background: "#111", border: "1px solid #2a2a2a", borderRadius: 2, fontFamily: "IBM Plex Mono, monospace", fontSize: "0.75rem" }}
                      formatter={(v, n) => [`${v}%`, n]}
                    />
                    <Bar dataKey="podium" name="Podium %" radius={[2, 2, 0, 0]}>
                      {podiumData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div className="card" style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "300px" }}>
              <div style={{ textAlign: "center", color: "#444" }}>
                <div style={{ fontFamily: "Bebas Neue", fontSize: "3rem", letterSpacing: "0.1em" }}>READY</div>
                <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "0.75rem", marginTop: "0.5rem" }}>
                  Set the grid and run a prediction
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}