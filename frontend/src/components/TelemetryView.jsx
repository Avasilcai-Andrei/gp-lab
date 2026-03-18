import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from "recharts";

const API = "http://localhost:8000";

const CHANNELS = [
  { key: "speed",    label: "Speed (km/h)",   color: "#e8002d", domain: [0, 380] },
  { key: "throttle", label: "Throttle (%)",    color: "#00d2be", domain: [0, 100] },
  { key: "brake",    label: "Brake",           color: "#ff6b35", domain: [0, 1] },
  { key: "gear",     label: "Gear",            color: "#ffd700", domain: [0, 9] },
  { key: "rpm",      label: "RPM",             color: "#9b59b6", domain: [4000, 16000] },
];

function formatLapTime(raw) {
  // raw looks like "0 days 00:01:20.307000"
  if (!raw) return "--";
  const match = raw.match(/(\d+):(\d+)\.(\d+)/);
  if (!match) return raw;
  const mins = parseInt(match[1]);
  const secs = match[2];
  const ms = match[3].slice(0, 3);
  return `${mins}:${secs}.${ms}`;
}


function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#111", border: "1px solid #2a2a2a",
      padding: "0.6rem 0.9rem", borderRadius: 2,
      fontFamily: "IBM Plex Mono, monospace", fontSize: "0.75rem"
    }}>
      <div style={{ color: "#666", marginBottom: 4 }}>{Math.round(label)}m</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
        </div>
      ))}
    </div>
  );
}

// Downsample data for performance
function downsample(arr, maxPoints = 600) {
  if (!arr || arr.length <= maxPoints) return arr;
  const step = Math.ceil(arr.length / maxPoints);
  return arr.filter((_, i) => i % step === 0);
}

function buildChartData(tel, downsampleTo = 600) {
  if (!tel?.distance) return [];
  const dist = tel.distance;
  const step = Math.max(1, Math.floor(dist.length / downsampleTo));
  return dist
    .filter((_, i) => i % step === 0)
    .map((d, i) => {
      const idx = i * step;
      return {
        distance: Math.round(d),
        speed: tel.speed?.[idx],
        throttle: tel.throttle?.[idx],
        brake: tel.brake?.[idx],
        gear: tel.gear?.[idx],
        rpm: tel.rpm?.[idx],
      };
    });
}

// Single driver telemetry panel
function SingleTelemetry() {
  const [form, setForm] = useState({ year: "2023", gp: "Monza", driver: "VER", session: "Q" });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedChannels, setSelectedChannels] = useState(["speed", "throttle", "brake"]);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const fetchData = async () => {
    setLoading(true); setError("");
    try {
      const res = await fetch(
        `${API}/telemetry/fastest-lap?year=${form.year}&gp=${encodeURIComponent(form.gp)}&driver=${form.driver}&session=${form.session}`
      );
      if (!res.ok) throw new Error((await res.json()).detail);
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = data ? buildChartData(data.telemetry) : [];

  return (
    <div>
      <div className="control-row">
        {[
          { k: "year", label: "Season", w: "90px" },
          { k: "gp", label: "Grand Prix", w: "160px" },
          { k: "driver", label: "Driver", w: "100px" },
          { k: "session", label: "Session", w: "100px" },
        ].map(({ k, label, w }) => (
          <div className="control-group" key={k}>
            <span className="control-label">{label}</span>
            <input
              className="control-input"
              style={{ width: w }}
              value={form[k]}
              onChange={e => set(k, e.target.value)}
              placeholder={label}
            />
          </div>
        ))}
        <button className="btn btn-primary" onClick={fetchData} disabled={loading}>
          {loading ? "Loading..." : "Load Lap"}
        </button>
      </div>

      {loading && <div className="status-bar"><span className="status-dot" /> Fetching from FastF1...</div>}
      {error && <div className="error-msg">⚠ {error}</div>}

      {data && (
        <>
          <div style={{ display: "flex", gap: "2rem", marginBottom: "1.5rem", flexWrap: "wrap" }}>
            <Stat label="Driver" value={data.driver} />
            <Stat label="Lap Time" value={formatLapTime(data.lap_time)} />
            <Stat label="Lap" value={`#${data.lap_number}`} />
            <Stat label="Tyre" value={data.compound} />
          </div>

          {/* Channel toggles */}
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            {CHANNELS.map(ch => (
              <button
                key={ch.key}
                onClick={() => setSelectedChannels(s =>
                  s.includes(ch.key) ? s.filter(x => x !== ch.key) : [...s, ch.key]
                )}
                style={{
                  fontFamily: "IBM Plex Mono, monospace", fontSize: "0.7rem",
                  letterSpacing: "0.08em", textTransform: "uppercase",
                  padding: "0.3rem 0.75rem", borderRadius: 2, cursor: "pointer",
                  border: `1px solid ${selectedChannels.includes(ch.key) ? ch.color : "#2a2a2a"}`,
                  background: selectedChannels.includes(ch.key) ? `${ch.color}22` : "transparent",
                  color: selectedChannels.includes(ch.key) ? ch.color : "#666",
                  transition: "all 0.15s",
                }}
              >
                {ch.label}
              </button>
            ))}
          </div>

          {/* Charts */}
          {CHANNELS.filter(ch => selectedChannels.includes(ch.key)).map(ch => (
            <div className="chart-container" key={ch.key}>
              <div className="chart-label">{ch.label}</div>
              <ResponsiveContainer width="100%" height={140}>
                <LineChart data={chartData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                  <XAxis dataKey="distance" tick={{ fontSize: 10 }} tickFormatter={v => `${v}m`} stroke="#333" />
                  <YAxis domain={ch.domain} tick={{ fontSize: 10 }} stroke="#333" width={40} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone" dataKey={ch.key} dot={false}
                    stroke={ch.color} strokeWidth={1.5} name={ch.label}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}

          {/* Lap Time Chart */}
          <LapTimePanel year={form.year} gp={form.gp} driver={form.driver} session={form.session} />
        </>
      )}
    </div>
  );
}

// Lap time evolution chart
function LapTimePanel({ year, gp, driver, session }) {
  const [lapData, setLapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetch_ = async () => {
    setLoading(true);
    setError(""); 
    try {
      const res = await fetch(`${API}/telemetry/lap-times?year=${year}&gp=${encodeURIComponent(gp)}&driver=${driver}&session=${session}`);
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to load stint data.");
      }
      
      const json = await res.json();
      setLapData(json);
    } catch (e) {
      setError(e.message);
    } finally { 
      setLoading(false); 
    }
  };

  const COMPOUND_COLORS = {
    SOFT: "#e8002d", MEDIUM: "#ffd700", HARD: "#f0f0f0",
    INTERMEDIATE: "#00d2be", WET: "#0067ff", UNKNOWN: "#666",
  };

  return (
    <div className="chart-container" style={{ marginTop: "2rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
        <div className="chart-label">Lap Time Evolution</div>
        <button className="btn btn-outline" style={{ fontSize: "0.65rem", padding: "0.3rem 0.75rem" }} onClick={fetch_} disabled={loading}>
          {loading ? "..." : "Load Stint"}
        </button>
      </div>

      {error && <div className="error-msg" style={{ marginBottom: "1rem" }}>⚠ {error}</div>}

      {lapData && !error && (
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={lapData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
            <XAxis dataKey="lap_number" tick={{ fontSize: 10 }} stroke="#333" />
            <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10 }} stroke="#333" width={45}
              tickFormatter={v => `${v.toFixed(0)}s`} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload;
              return (
                <div style={{ background: "#111", border: "1px solid #2a2a2a", padding: "0.6rem", borderRadius: 2, fontFamily: "IBM Plex Mono, monospace", fontSize: "0.75rem" }}>
                  <div style={{ color: "#666" }}>Lap {d.lap_number}</div>
                  <div style={{ color: "#f0f0f0" }}>{formatLapTime(d.lap_time_str)}</div>
                  <div style={{ color: COMPOUND_COLORS[d.compound] || "#666" }}>{d.compound}</div>
                </div>
              );
            }} />
            <Line type="monotone" dataKey="lap_time_seconds" dot={(props) => {
              const { cx, cy, payload } = props;
              const color = COMPOUND_COLORS[payload.compound] || "#666";
              return <circle key={payload.lap_number} cx={cx} cy={cy} r={3} fill={color} stroke="none" />;
            }}
              stroke="#333" strokeWidth={1} isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// Dual driver comparison panel
function CompareTelemetry() {
  const [form, setForm] = useState({
    year: "2023", gp: "Monza", session: "Q",
    driver1: "VER", lap1: "1",
    driver2: "HAM", lap2: "1",
  });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [channel, setChannel] = useState("speed");

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const fetchData = async () => {
    setLoading(true); setError("");
    try {
      const res = await fetch(
        `${API}/telemetry/compare?year=${form.year}&gp=${encodeURIComponent(form.gp)}&session=${form.session}` +
        `&driver1=${form.driver1}&lap1=${form.lap1}&driver2=${form.driver2}&lap2=${form.lap2}`
      );
      if (!res.ok) throw new Error((await res.json()).detail);
      setData(await res.json());
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  // Merge two telemetry datasets by interpolating on common distance axis
  const buildCompareData = () => {
    if (!data) return [];
    const d1 = data.driver1.telemetry;
    const d2 = data.driver2.telemetry;
    const maxDist = Math.max(...d1.distance, ...d2.distance);
    const points = 600;
    const result = [];
    for (let i = 0; i <= points; i++) {
      const dist = (maxDist / points) * i;
      const interp = (arr_d, arr_v) => {
        const idx = arr_d.findIndex(d => d >= dist);
        if (idx <= 0) return arr_v[0];
        if (idx >= arr_d.length) return arr_v[arr_v.length - 1];
        const t = (dist - arr_d[idx - 1]) / (arr_d[idx] - arr_d[idx - 1]);
        return arr_v[idx - 1] + t * (arr_v[idx] - arr_v[idx - 1]);
      };
      result.push({
        distance: Math.round(dist),
        [data.driver1.driver]: interp(d1.distance, d1[channel]),
        [data.driver2.driver]: interp(d2.distance, d2[channel]),
        delta: interp(d1.distance, d1[channel]) - interp(d2.distance, d2[channel]),
      });
    }
    return result;
  };

  const chartData = buildCompareData();
  const ch = CHANNELS.find(c => c.key === channel);

  return (
    <div>
      <div className="control-row">
        {[
          { k: "year", label: "Season", w: "80px" },
          { k: "gp", label: "Grand Prix", w: "160px" },
          { k: "session", label: "Session", w: "90px" },
        ].map(({ k, label, w }) => (
          <div className="control-group" key={k}>
            <span className="control-label">{label}</span>
            <input className="control-input" style={{ width: w }} value={form[k]} onChange={e => set(k, e.target.value)} />
          </div>
        ))}
      </div>
      <div className="control-row">
        <div style={{ display: "flex", gap: "1rem", alignItems: "flex-end", padding: "0.75rem 1rem", background: "#1a1a1a", borderRadius: 4, border: "1px solid #e8002d22", flex: 1 }}>
          <div className="control-group">
            <span className="control-label" style={{ color: "#e8002d" }}>Driver 1</span>
            <input className="control-input" style={{ width: "90px" }} value={form.driver1} onChange={e => set("driver1", e.target.value)} />
          </div>
          <div className="control-group">
            <span className="control-label">Lap #</span>
            <input className="control-input" style={{ width: "60px" }} value={form.lap1} onChange={e => set("lap1", e.target.value)} />
          </div>
        </div>
        <div style={{ display: "flex", gap: "1rem", alignItems: "flex-end", padding: "0.75rem 1rem", background: "#1a1a1a", borderRadius: 4, border: "1px solid #00d2be22", flex: 1 }}>
          <div className="control-group">
            <span className="control-label" style={{ color: "#00d2be" }}>Driver 2</span>
            <input className="control-input" style={{ width: "90px" }} value={form.driver2} onChange={e => set("driver2", e.target.value)} />
          </div>
          <div className="control-group">
            <span className="control-label">Lap #</span>
            <input className="control-input" style={{ width: "60px" }} value={form.lap2} onChange={e => set("lap2", e.target.value)} />
          </div>
        </div>
        <button className="btn btn-primary" onClick={fetchData} disabled={loading} style={{ alignSelf: "flex-end" }}>
          {loading ? "Loading..." : "Compare"}
        </button>
      </div>

      {error && <div className="error-msg">⚠ {error}</div>}

      {data && (
        <>
          <div style={{ display: "flex", gap: "2rem", marginBottom: "1.5rem", flexWrap: "wrap" }}>
            <div style={{ display: "flex", gap: "1.5rem" }}>
              <Stat label={data.driver1.driver} value={formatLapTime(data.driver1.lap_time)} color="#e8002d" />
              <Stat label={data.driver2.driver} value={formatLapTime(data.driver2.lap_time)} color="#00d2be" />
            </div>
          </div>

          {/* Channel selector */}
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
            {CHANNELS.map(c => (
              <button key={c.key} onClick={() => setChannel(c.key)}
                className={`inner-tab ${channel === c.key ? "active" : ""}`}
                style={channel === c.key ? { background: c.color, borderColor: c.color } : {}}>
                {c.key}
              </button>
            ))}
          </div>

          {/* Overlay chart */}
          <div className="chart-label">{ch.label} — Comparison</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
              <XAxis dataKey="distance" tick={{ fontSize: 10 }} tickFormatter={v => `${v}m`} stroke="#333" />
              <YAxis domain={ch.domain} tick={{ fontSize: 10 }} stroke="#333" width={45} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "0.7rem" }} />
              <Line type="monotone" dataKey={data.driver1.driver} dot={false} stroke="#e8002d" strokeWidth={2} isAnimationActive={false} />
              <Line type="monotone" dataKey={data.driver2.driver} dot={false} stroke="#00d2be" strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>

          {/* Delta chart */}
          <div className="chart-label" style={{ marginTop: "1.5rem" }}>Delta ({data.driver1.driver} – {data.driver2.driver})</div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={chartData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
              <XAxis dataKey="distance" tick={{ fontSize: 10 }} tickFormatter={v => `${v}m`} stroke="#333" />
              <YAxis tick={{ fontSize: 10 }} stroke="#333" width={45} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke="#444" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="delta" dot={false} stroke="#ffd700" strokeWidth={1.5} isAnimationActive={false} name="Delta" />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div>
      <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "0.62rem", letterSpacing: "0.12em", textTransform: "uppercase", color: color || "#666", marginBottom: "0.2rem" }}>{label}</div>
      <div style={{ fontFamily: "Bebas Neue, sans-serif", fontSize: "1.4rem", letterSpacing: "0.05em", color: "#f0f0f0" }}>{value}</div>
    </div>
  );
}

export default function TelemetryView() {
  const [tab, setTab] = useState("single");
  return (
    <div>
      <div className="section-title">Telemetry</div>
      <div className="section-subtitle">real lap data via fastf1 · speed · throttle · brake · gear</div>

      <div className="inner-tabs">
        <button className={`inner-tab ${tab === "single" ? "active" : ""}`} onClick={() => setTab("single")}>Single Lap</button>
        <button className={`inner-tab ${tab === "compare" ? "active" : ""}`} onClick={() => setTab("compare")}>Compare Drivers</button>
      </div>

      <div className="card">
        {tab === "single" ? <SingleTelemetry /> : <CompareTelemetry />}
      </div>
    </div>
  );
}
