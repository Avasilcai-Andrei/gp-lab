# 🏎️ F1 Lab — Telemetry & Race Predictor

A full-stack F1 data app built with **FastF1**, **FastAPI**, **React**, and **XGBoost**. Load real telemetry data from any F1 session, compare drivers lap-by-lap, and predict race finishing orders using a machine learning model trained on historical results.

---

## Features

### Telemetry Viewer
- Load any driver's fastest lap from any session (Race, Qualifying, FP1/2/3)
- Charts: Speed, Throttle, Brake, RPM, Gear vs. Distance
- Toggle channels on/off with colored buttons
- Lap time evolution chart with tyre compound coloring (soft/medium/hard/inter/wet)
- **Driver comparison mode** — overlay two drivers with interpolated delta trace
- Data sourced live from FastF1 and cached locally

### Race Predictor
- Edit a full qualifying grid in the UI
- Predict finishing order using XGBoost trained on real historical F1 data
- Features used: grid position, driver form (recency-weighted rolling avg), team form, DNF rate, regulation era flag
- Soft probability distribution — podium % and points % across the full field
- Train the model directly from the UI on any range of seasons
- Per-driver form nudge (-5 to +5) to apply domain knowledge on top of the model

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | FastF1 (real F1 telemetry + session results) |
| Backend | Python, FastAPI, Uvicorn |
| ML | XGBoost, scikit-learn, pandas, numpy |
| Frontend | React, Vite, Recharts |
| Communication | REST API (JSON) |

---

## Project Structure

```
f1-lab/
├── backend/
│   ├── main.py           # FastAPI server + all endpoints
│   ├── telemetry.py      # FastF1 data fetching & processing
│   ├── predictor.py      # ML model training & inference
│   ├── models/           # Saved .pkl model (auto-created, gitignored)
│   ├── f1_cache/         # FastF1 data cache (auto-created, gitignored)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── App.css
    │   └── components/
    │       ├── TelemetryView.jsx
    │       └── PredictorView.jsx
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Backend

```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

Backend runs at **http://localhost:8000**
Interactive API docs at **http://localhost:8000/docs**

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:5173**

---

## API Endpoints

### Telemetry

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/telemetry/fastest-lap` | Fastest lap telemetry for a driver |
| GET | `/telemetry/lap` | Telemetry for a specific lap number |
| GET | `/telemetry/compare` | Compare two drivers on the same chart |
| GET | `/telemetry/lap-times` | Full stint lap time data |
| GET | `/telemetry/drivers` | List all drivers in a session |

**Example:**
```
GET /telemetry/fastest-lap?year=2023&gp=Monza&driver=VER&session=Q
```

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/race` | Predict race from qualifying grid |
| POST | `/predict/train?years=2024,2025` | Train model in background |
| GET | `/predict/status` | Check model status |

**Example request body:**
```json
{
  "year": 2026,
  "grand_prix": "Silverstone",
  "qualifying_results": [
    { "driver": "RUS", "team": "Mercedes", "grid": 1, "form_nudge": 0 },
    { "driver": "ANT", "team": "Mercedes", "grid": 2, "form_nudge": 0 },
    { "driver": "HAM", "team": "Ferrari",  "grid": 3, "form_nudge": 0 }
  ]
}
```

---

## Usage Guide

### Telemetry
1. Enter **Year**, **Grand Prix** (e.g. `Monza`, `Monaco`, `Silverstone`), **Driver** (e.g. `VER`, `HAM`, `LEC`), **Session** (`Q`, `R`, `FP1`)
2. Click **Load Lap** — first load takes 30-60s, subsequent loads instant from cache
3. Toggle channels with the colored buttons
4. Click **Load Stint** for lap time evolution

### Driver Comparison
1. Switch to **Compare Drivers** tab
2. Enter two drivers and lap numbers
3. Switch between Speed / Throttle / Brake / Gear / RPM channels

### Race Predictor
1. Set train years (e.g. `2024,2025,2026`) and click **Train Model**
2. Click **Model Status** to check progress
3. Edit the qualifying grid with current drivers and teams
4. Use the **form nudge** (− / +) buttons to override the model's form estimate per driver
5. Click **▶ Run Prediction**

---

## ML Model Details

- **Algorithm:** XGBoost Regressor
- **Features:** grid position, recency-weighted form (exponential decay over last 10 races), team form, DNF rate, season progress, regulation era flag
- **Sample weights:** 2026+ races weighted 5x to reflect current regulation order
- **Known limitation:** Accuracy improves as more races from the current season are added. Retrain required periodically.

---

## Credits

- Telemetry and session data via **[FastF1](https://github.com/theOehrly/Fast-F1)** (MIT License)
- ML via **XGBoost** and **scikit-learn**
- Charts via **Recharts**
- API via **FastAPI**

---

## Known Limitations

- FastF1 rate limits to ~500 API calls/hour — training on multiple years may require waiting between runs
- DNFs caused by reliability temporarily skew a driver's form score downward
- New drivers with no historical data fall back to grid position as their form estimate

---

## Potential Improvements

- [ ] Track map — color-coded speed visualization using X/Y telemetry coordinates
- [ ] Sector times — purple/green/yellow mini-sector comparison
- [ ] Tyre strategy predictor

