# 🏁 GP Lab — Grand Prix Telemetry & ML Predictor

GP Lab is a full-stack performance analysis tool for Formula racing. It combines real-time telemetry visualization with a Gradient Boosting model to predict race outcomes based on historical trends, driver form, and circuit characteristics.

> **Disclaimer:** This project is an independent educational tool and is not affiliated with, sponsored by, or endorsed by the Formula 1 companies. F1, FORMULA ONE, and related marks are trademarks of Formula One Licensing B.V.

---

## 🚀 Key Features

### 1. Advanced Telemetry Analytics
- **Multi-Channel Visualization:** Real-time speed, throttle, brake, RPM, and gear data synchronized across distance.
- **Delta Comparison Mode:** Overlay two drivers to visualize time lost/gained through corners using interpolated trace data.
- **Stint Evolution:** Track lap time degradation with automated tyre compound detection.

### 2. ML Race Predictor
- **Dynamic Grid Engine:** Interactive UI to set the starting grid and run simulations.
- **Feature Engineering:** Uses recency-weighted form, team-based performance baselines, and circuit-specific "affinity" scores.
- **Probabilistic Outcomes:** Uses a modified Softmax decay to calculate podium and points probabilities across the field.

---

## 🛠️ Technical Implementation (Deep Dive)

### Machine Learning Architecture
The prediction engine uses an **XGBoost Regressor** with custom sample weighting. 

- **Time-Decay Weighting:** To account for the rapid evolution of car development, training data is weighted using an exponential decay function: 
  $$W_i = e^{0.015 \cdot (Index_i - Index_{max})}$$
  This ensures that the model prioritizes recent performance while still learning from historical data.
- **Cold-Start Mitigation:** For rookies or drivers new to a team (e.g., 2026 scenarios), the model implements a hierarchical fallback: 
  `Driver Form` → `Team Baseline` → `Grid Position`.
- **Circuit Characteristics:** Instead of just treating every track the same, the model ingests a "Circuit Fingerprint" (Street vs. Permanent, Overtaking Difficulty, Power Dependency, and Tyre Degradation).

### Data Pipeline
- **FastF1 Integration:** Automated fetching and local caching of telemetry and session results.
- **Grid Correction:** Unlike basic predictors that use Qualifying results, this tool pulls the **actual starting grid** from the race session to account for post-qualifying penalties and pit-lane starts.

---

## 💻 Tech Stack

- **Frontend:** React 18, Vite, Recharts (Customized for dark-mode performance data)
- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Data/ML:** XGBoost, Pandas, FastF1
- **Styling:** CSS3 (Bebas Neue & IBM Plex Mono for a "pit wall" aesthetic)

---

## 🔧 Setup & Installation

### Backend
1. `cd backend`
2. `python -m venv venv`
3. `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. `pip install -r requirements.txt`
5. `python main.py`

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`

---

## Credits & License
- Data provided by **FastF1** (MIT License).
- Built for educational and portfolio purposes.