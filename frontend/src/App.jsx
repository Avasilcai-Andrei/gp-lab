import { useState } from "react";
import TelemetryView from "./components/TelemetryView";
import PredictorView from "./components/PredictorView";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("telemetry");

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-flag">🏁</span>
            <span className="logo-text">F1<span className="logo-accent">LAB</span></span>
          </div>
          <nav className="nav">
            <button
              className={`nav-btn ${activeTab === "telemetry" ? "active" : ""}`}
              onClick={() => setActiveTab("telemetry")}
            >
              Telemetry
            </button>
            <button
              className={`nav-btn ${activeTab === "predictor" ? "active" : ""}`}
              onClick={() => setActiveTab("predictor")}
            >
              Race Predictor
            </button>
          </nav>
        </div>
      </header>

      <main className="main">
        {activeTab === "telemetry" ? <TelemetryView /> : <PredictorView />}
      </main>
    </div>
  );
}
