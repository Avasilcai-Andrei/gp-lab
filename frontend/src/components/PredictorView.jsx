import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from "recharts";
import {
  DndContext, closestCenter, PointerSensor, useSensor, useSensors
} from "@dnd-kit/core";
import {
  SortableContext, arrayMove, verticalListSortingStrategy, useSortable
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

const API = "http://localhost:8000";

const TEAM_COLORS = {
  "Red Bull Racing": "#3671C6",
  "Mercedes": "#27F4D2",
  "Ferrari": "#E8002D",
  "McLaren": "#FF8000",
  "Aston Martin": "#358C75",
  "Alpine": "#FF87BC",
  "Williams": "#64C4FF",
  "Racing Bulls": "#6692FF",
  "RB": "#6692FF",
  "Kick Sauber": "#52E252",
  "Audi": "#BB0A30",
  "Cadillac": "#B6862C",
  "Haas F1 Team": "#B6BABD",
  // legacy names kept for older seasons
  "AlphaTauri": "#5E8FAA",
  "Alfa Romeo": "#C92D4B",
};

// Offline fallback only — the grid is normally loaded from /predict/grid.
const DRIVER_PRESET = [
  { driver: "RUS", team: "Mercedes" },
  { driver: "ANT", team: "Mercedes" },
  { driver: "LEC", team: "Ferrari" },
  { driver: "HAM", team: "Ferrari" },
  { driver: "NOR", team: "McLaren" },
  { driver: "PIA", team: "McLaren" },
  { driver: "VER", team: "Red Bull Racing" },
  { driver: "HAD", team: "Red Bull Racing" },
  { driver: "ALO", team: "Aston Martin" },
  { driver: "STR", team: "Aston Martin" },
];

let _uid = 0;
const makeId = () => `g${_uid++}`;
// Array order is the single source of truth; numeric grid is derived from index.
const withIds = (list) => list.map((e) => ({ id: makeId(), driver: e.driver, team: e.team }));

function PosBadge({ pos }) {
  const cls = pos === 1 ? "pos-1" : pos === 2 ? "pos-2" : pos === 3 ? "pos-3" : "pos-other";
  return <span className={`pos-badge ${cls}`}>{pos}</span>;
}

function SortableRow({ entry, pos, onRemove }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: entry.id });
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    display: "flex", gap: "0.5rem", alignItems: "center",
    padding: "0.45rem 0.5rem", borderRadius: 2,
    background: isDragging ? "#1c1c1c" : (pos % 2 === 1 ? "#141414" : "transparent"),
    boxShadow: isDragging ? "0 6px 18px rgba(0,0,0,0.55)" : "none",
    border: `1px solid ${isDragging ? "#e8002d" : "transparent"}`,
    cursor: "grab", userSelect: "none",
    position: "relative", zIndex: isDragging ? 20 : "auto",
  };
  return (
    <div ref={setNodeRef} style={style} {...attributes} {...listeners}>
      <span style={{ color: "#444", fontSize: "0.85rem", width: "14px", textAlign: "center" }}>⠿</span>
      <span style={{ fontFamily: "Bebas Neue, sans-serif", fontSize: "1.1rem", color: "#444", width: "22px", textAlign: "center" }}>{pos}</span>
      <span style={{ fontFamily: "Bebas Neue, sans-serif", fontSize: "1.05rem", letterSpacing: "0.05em", width: "50px" }}>{entry.driver}</span>
      <span style={{ flex: 1, fontFamily: "IBM Plex Mono, monospace", fontSize: "0.75rem", color: "#bbb" }}>{entry.team}</span>
      <div style={{ width: 10, height: 10, borderRadius: "50%", background: TEAM_COLORS[entry.team] || "#444", flexShrink: 0 }} />
      <button
        onClick={(e) => { e.stopPropagation(); onRemove(entry.id); }}
        onPointerDown={(e) => e.stopPropagation()}
        title="Remove"
        style={{ background: "none", border: "none", color: "#555", cursor: "pointer", fontSize: "1rem", lineHeight: 1, padding: "0 0.25rem" }}
      >×</button>
    </div>
  );
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
  const [grid, setGrid] = useState(() => withIds(DRIVER_PRESET));
  const [gridLoading, setGridLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelStatus, setModelStatus] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainYears, setTrainYears] = useState("2022,2023,2024,2025,2026");

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } })
  );

  // Load the real grid on mount and whenever Season / Grand Prix change (debounced).
  useEffect(() => {
    if (!year || !gp) return;
    const t = setTimeout(async () => {
      setGridLoading(true);
      try {
        const res = await fetch(`${API}/predict/grid/${parseInt(year)}/${encodeURIComponent(gp)}`);
        if (!res.ok) throw new Error("grid fetch failed");
        const data = await res.json();
        if (Array.isArray(data) && data.length) setGrid(withIds(data));
        else throw new Error("empty grid");
      } catch {
        setGrid(withIds(DRIVER_PRESET)); // offline fallback
      } finally {
        setGridLoading(false);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [year, gp]);

  const onDragEnd = ({ active, over }) => {
    if (over && active.id !== over.id) {
      setGrid((items) => {
        const oldI = items.findIndex((i) => i.id === active.id);
        const newI = items.findIndex((i) => i.id === over.id);
        return arrayMove(items, oldI, newI);
      });
    }
  };

  const removeRow = (id) => setGrid((items) => items.filter((i) => i.id !== id));
  const addRow = () => setGrid((items) => [...items, { id: makeId(), driver: "NEW", team: "Unknown" }]);

  const randomizeGrid = () => setGrid((items) => {
    const a = items.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  });

  const predict = async () => {
    setLoading(true); setError(""); setResults(null);
    try {
      const res = await fetch(`${API}/predict/race`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          year: parseInt(year),
          grand_prix: gp,
          // Grid = current array order; payload shape stays [{driver, team, grid}].
          qualifying_results: grid.map((e, i) => ({ driver: e.driver, team: e.team, grid: i + 1 })),
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
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.6rem" }}>
            <div className="card-title" style={{ marginBottom: 0 }}>
              Qualifying Grid
              {gridLoading && (
                <span style={{ marginLeft: "0.5rem", fontFamily: "IBM Plex Mono, monospace", fontSize: "0.65rem", color: "#666" }}>
                  loading…
                </span>
              )}
            </div>
            <button
              className="btn btn-outline"
              style={{ fontSize: "0.65rem", padding: "0.3rem 0.6rem" }}
              onClick={randomizeGrid}
            >
              ⤮ Randomize
            </button>
          </div>
          <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "0.65rem", color: "#555", marginBottom: "0.5rem" }}>
            drag to reorder · position sets the starting grid
          </div>
          <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
            <SortableContext items={grid.map((e) => e.id)} strategy={verticalListSortingStrategy}>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                {grid.map((entry, i) => (
                  <SortableRow key={entry.id} entry={entry} pos={i + 1} onRemove={removeRow} />
                ))}
              </div>
            </SortableContext>
          </DndContext>
          <button
            className="btn btn-outline"
            style={{ marginTop: "0.75rem", width: "100%", fontSize: "0.7rem" }}
            onClick={addRow}
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