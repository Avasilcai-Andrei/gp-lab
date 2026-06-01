import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor



CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

MODEL_PATH = Path("./models/race_predictor.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)


# ─── Circuit Characteristics ─────────────────────────────────────────────────
# street        : 1 = street circuit, 0 = permanent
# overtaking    : 0 = very hard (Monaco) → 1 = very easy (Monza)
# power_dep     : how much raw engine power matters
# downforce_dep : how much aero downforce matters
# tire_deg      : how aggressive tyre degradation is

CIRCUIT_FEATURES = {
    "Monaco":       {"street": 1, "overtaking": 0.05, "power_dep": 0.2, "downforce_dep": 0.9,  "tire_deg": 0.3},
    "Baku":         {"street": 1, "overtaking": 0.55, "power_dep": 0.8, "downforce_dep": 0.5,  "tire_deg": 0.4},
    "Singapore":    {"street": 1, "overtaking": 0.2,  "power_dep": 0.3, "downforce_dep": 0.95, "tire_deg": 0.5},
    "Jeddah":       {"street": 1, "overtaking": 0.45, "power_dep": 0.85,"downforce_dep": 0.6,  "tire_deg": 0.3},
    "LasVegas":     {"street": 1, "overtaking": 0.6,  "power_dep": 0.9, "downforce_dep": 0.45, "tire_deg": 0.35},
    "Monza":        {"street": 0, "overtaking": 0.85, "power_dep": 1.0, "downforce_dep": 0.1,  "tire_deg": 0.4},
    "Silverstone":  {"street": 0, "overtaking": 0.55, "power_dep": 0.65,"downforce_dep": 0.75, "tire_deg": 0.75},
    "Spa":          {"street": 0, "overtaking": 0.7,  "power_dep": 0.85,"downforce_dep": 0.6,  "tire_deg": 0.55},
    "Bahrain":      {"street": 0, "overtaking": 0.75, "power_dep": 0.7, "downforce_dep": 0.65, "tire_deg": 0.85},
    "Suzuka":       {"street": 0, "overtaking": 0.35, "power_dep": 0.6, "downforce_dep": 0.9,  "tire_deg": 0.6},
    "Barcelona":    {"street": 0, "overtaking": 0.3,  "power_dep": 0.6, "downforce_dep": 0.85, "tire_deg": 0.9},
    "Zandvoort":    {"street": 0, "overtaking": 0.2,  "power_dep": 0.55,"downforce_dep": 0.85, "tire_deg": 0.7},
    "Interlagos":   {"street": 0, "overtaking": 0.65, "power_dep": 0.6, "downforce_dep": 0.7,  "tire_deg": 0.65},
    "Melbourne":    {"street": 0, "overtaking": 0.4,  "power_dep": 0.65,"downforce_dep": 0.7,  "tire_deg": 0.5},
    "Imola":        {"street": 0, "overtaking": 0.25, "power_dep": 0.6, "downforce_dep": 0.8,  "tire_deg": 0.6},
    "Miami":        {"street": 0, "overtaking": 0.5,  "power_dep": 0.7, "downforce_dep": 0.7,  "tire_deg": 0.6},
    "Montreal":     {"street": 0, "overtaking": 0.65, "power_dep": 0.75,"downforce_dep": 0.55, "tire_deg": 0.5},
    "RedBullRing":  {"street": 0, "overtaking": 0.6,  "power_dep": 0.7, "downforce_dep": 0.65, "tire_deg": 0.7},
    "Hungaroring":  {"street": 0, "overtaking": 0.2,  "power_dep": 0.5, "downforce_dep": 0.9,  "tire_deg": 0.75},
    "Losail":       {"street": 0, "overtaking": 0.5,  "power_dep": 0.7, "downforce_dep": 0.75, "tire_deg": 0.8},
    "COTA":         {"street": 0, "overtaking": 0.6,  "power_dep": 0.65,"downforce_dep": 0.8,  "tire_deg": 0.7},
    "AbuDhabi":     {"street": 0, "overtaking": 0.45, "power_dep": 0.7, "downforce_dep": 0.75, "tire_deg": 0.5},
    "Shanghai":     {"street": 0, "overtaking": 0.55, "power_dep": 0.65,"downforce_dep": 0.75, "tire_deg": 0.75},
    "Mexico":       {"street": 0, "overtaking": 0.4,  "power_dep": 0.5, "downforce_dep": 0.95, "tire_deg": 0.5},
}

CIRCUIT_ALIASES = {
    "bahrain grand prix": "Bahrain",
    "australian grand prix": "Melbourne",
    "saudi arabian grand prix": "Jeddah",
    "japanese grand prix": "Suzuka",
    "chinese grand prix": "Shanghai",
    "miami grand prix": "Miami",
    "emilia romagna grand prix": "Imola",
    "monaco grand prix": "Monaco",
    "canadian grand prix": "Montreal",
    "spanish grand prix": "Barcelona",
    "austrian grand prix": "RedBullRing",
    "british grand prix": "Silverstone",
    "hungarian grand prix": "Hungaroring",
    "belgian grand prix": "Spa",
    "dutch grand prix": "Zandvoort",
    "italian grand prix": "Monza",
    "azerbaijan grand prix": "Baku",
    "singapore grand prix": "Singapore",
    "united states grand prix": "COTA",
    "mexico city grand prix": "Mexico",
    "são paulo grand prix": "Interlagos",
    "sao paulo grand prix": "Interlagos",
    "las vegas grand prix": "LasVegas",
    "qatar grand prix": "Losail",
    "abu dhabi grand prix": "AbuDhabi",
    "bahrain": "Bahrain",
    "australia": "Melbourne", "melbourne": "Melbourne",
    "saudi": "Jeddah", "jeddah": "Jeddah",
    "japan": "Suzuka", "suzuka": "Suzuka",
    "china": "Shanghai", "shanghai": "Shanghai",
    "miami": "Miami",
    "imola": "Imola", "emilia": "Imola",
    "monaco": "Monaco",
    "canada": "Montreal", "montreal": "Montreal",
    "spain": "Barcelona", "barcelona": "Barcelona",
    "austria": "RedBullRing", "red bull ring": "RedBullRing",
    "silverstone": "Silverstone", "britain": "Silverstone", "british": "Silverstone",
    "hungary": "Hungaroring", "hungaroring": "Hungaroring",
    "belgium": "Spa", "spa": "Spa",
    "netherlands": "Zandvoort", "zandvoort": "Zandvoort", "dutch": "Zandvoort",
    "monza": "Monza", "italy": "Monza", "italian": "Monza",
    "baku": "Baku", "azerbaijan": "Baku",
    "singapore": "Singapore",
    "cota": "COTA", "texas": "COTA", "austin": "COTA", "americas": "COTA",
    "mexico": "Mexico", "mexico city": "Mexico",
    "brazil": "Interlagos", "interlagos": "Interlagos",
    "vegas": "LasVegas", "las vegas": "LasVegas",
    "qatar": "Losail", "losail": "Losail",
    "abu dhabi": "AbuDhabi", "abudhabi": "AbuDhabi",
}

def match_circuit_key(grand_prix: str):
    """Resolve a GP name to its canonical circuit key via the alias map, or None.

    Shares the same matching logic as get_circuit_features; exposed so other
    layers (e.g. the grid endpoint) can reuse the alias resolution."""
    gp = grand_prix.strip()
    if gp in CIRCUIT_FEATURES:
        return gp
    gp_lower = gp.lower()
    for alias, key in CIRCUIT_ALIASES.items():
        if alias in gp_lower:
            return key
    return None


def get_circuit_features(grand_prix: str) -> dict:
    """Match a GP name to circuit characteristics, with fuzzy fallback."""
    key = match_circuit_key(grand_prix)
    if key is not None:
        return CIRCUIT_FEATURES[key]
    # Generic average circuit if unknown
    return {"street": 0, "overtaking": 0.5, "power_dep": 0.6,
            "downforce_dep": 0.65, "tire_deg": 0.6}


# ─── Data Collection ─────────────────────────────────────────────────────────

def collect_training_data(years: list[int]) -> pd.DataFrame:
    records = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            for _, event in schedule.iterrows():
                try:
                    race = fastf1.get_session(year, event["EventName"], "R")
                    race.load(telemetry=False, weather=False, messages=False, laps=False)
                    
                    # Use actual Race Results for Ground Truth and Starting Grid
                    results = race.results
                    gp_name = event["EventName"]

                    for _, row in results.iterrows():
                        records.append({
                            "year": year,
                            "round": int(event.get("RoundNumber", 0)),
                            "grand_prix": gp_name,
                            "driver": row["Abbreviation"],
                            "team": row.get("TeamName", "Unknown"),
                            "grid_position": int(row["GridPosition"]), # Corrected: Real starting position
                            "finish_position": int(row["Position"]) if pd.notna(row["Position"]) else 20,
                            "points": float(row["Points"]) if pd.notna(row["Points"]) else 0,
                            "status": row.get("Status", "Finished"),
                        })
                except Exception as e:
                    print(f"Fetch error {year} {event['EventName']}: {str(e)[:50]}")
                    continue
        except Exception as e:
            print(f"Schedule fetch error for {year}: {str(e)[:50]}")
            continue
            
    return pd.DataFrame(records)


# ─── Feature Engineering ─────────────────────────────────────────────────────

REG_CHANGE_YEARS = {2022, 2026}

def engineer_features(df: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    # Sort for time-series operations
    df = df.sort_values(["driver", "year", "round"]).copy()

    df["new_era"] = df["year"].apply(lambda y: 1 if y in REG_CHANGE_YEARS else 0)
    df["dnf"] = (~df["status"].str.contains("Finished|Lapped", na=False)).astype(int)

    # Driver Form (Rolling 3-race window)
    df["form_3"] = (
        df.groupby("driver")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    
    # Points Form (The missing key!)
    df["points_form_3"] = (
        df.groupby("driver")["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    
    # Team Performance Baseline
    df["team_form_3"] = (
        df.groupby("team")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Reliability Metric
    df["dnf_rate"] = (
        df.groupby("driver")["dnf"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Normalized Season Progress
    df["season_progress"] = (
        df["round"] / df.groupby("year")["round"].transform("max")
    )

    # Historical Performance at specific Venue
    df["circuit_affinity"] = (
        df.groupby(["driver", "grand_prix"])["finish_position"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    
    # Fill missing values using hierarchical fallbacks
    df["points_form_3"] = df["points_form_3"].fillna(0.0)
    df["circuit_affinity"] = df["circuit_affinity"].fillna(df["form_3"]).fillna(df["team_form_3"])
    df["form_3"] = df["form_3"].fillna(df["team_form_3"])

    # Map Static Circuit Metadata
    for col in ["street", "overtaking", "power_dep", "downforce_dep", "tire_deg"]:
        df[col] = df["grand_prix"].apply(lambda gp: get_circuit_features(gp)[col])

    # Drop any rows where we still don't have essential data. Inference passes
    # drop_incomplete=False so candidate rows for rookies/new teams survive (their
    # NaN features are then filled with the training means, just like in fit).
    if drop_incomplete:
        return df.dropna(subset=["form_3", "grid_position"])
    return df



# ─── Model Training ───────────────────────────────────────────────────────────

# Single source of truth for the feature set, model config, and fit procedure so
# that the backtest harness trains exactly the same model as production.
FEATURES = [
    "grid_position", "form_3", "points_form_3", "team_form_3",
    "dnf_rate", "season_progress", "new_era", "circuit_affinity",
    "street", "overtaking", "power_dep", "downforce_dep", "tire_deg",
]

# Legacy post-adjustment that pulls street-circuit predictions back toward grid.
# Redundant now that the model predicts a grid-anchored delta; kept behind a flag
# so the backtest can decide whether it still helps. Off by default.
USE_STREET_PULL = False

# Exclude DNF rows from the regression target (a filled-in finish of 20 is not a
# pace signal). On by default; toggled off only for attribution backtests.
EXCLUDE_DNF = True

# Fallback noise scale (positions) for the Monte Carlo probability simulation,
# used when a model wasn't shipped with a calibrated residual_std. The authoritative
# value comes from the walk-forward out-of-sample residuals (see backtest_calibration).
RESIDUAL_STD = 3.0

# Number of Monte Carlo simulations per race for probability estimation.
N_SIMS = 4000

# Predicted-position bin edges (right-open) for position-dependent noise std.
# A few coarse bins keep the heteroscedastic fit robust without overfitting.
POS_BINS = [1, 4, 8, 13, 21]


def _make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.04,
        random_state=42,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        gamma=1.0,
    )


def fit_model(raw_df: pd.DataFrame) -> dict:
    """Engineer features and fit the model on a RAW (un-engineered) dataframe.

    Engineering happens here via engineer_features() so that training and
    inference share one feature function — inference re-runs the same
    engineer_features() over (history + the upcoming race). This kills the
    train/serve skew that previously existed (training used rolling-3 means while
    inference used an exp-weighted last-10 stat).

    The target is the GRID DELTA (finish_position - grid_position): positions
    gained/lost relative to the start. This anchors predictions to the grid so
    the model only has to learn the deviation, instead of compressing absolute
    positions toward the mean.

    DNF rows are excluded from the regression target — a DNF's filled-in
    finish_position of 20 is not a pace signal and poisons the delta. Reliability
    is still represented via the `dnf_rate` feature, computed from the full
    history (DNF rows are kept in the engineered frame, only dropped from the fit).

    Returns the dict pickled to MODEL_PATH, plus:
      - "raw_history": the raw rows, so inference can re-engineer identically
      - "feature_means": training fill values, applied to NaNs at inference too
      - "target": "delta" marker so predict re-anchors to grid
    """
    raw_df = raw_df.copy()
    df = engineer_features(raw_df)
    df["race_index"] = df.groupby(["year", "round"]).ngroup()
    max_idx = df["race_index"].max()

    # Train on classified finishers only (unless attribution disables it).
    finishers = df[df["dnf"] == 0] if EXCLUDE_DNF else df

    feature_means = finishers[FEATURES].mean()
    X = finishers[FEATURES].fillna(feature_means)
    y = finishers["finish_position"] - finishers["grid_position"]

    # Time-decay weights: recent races count exponentially more
    # Combined with a reg-change boost for 2022/2026 races
    time_weights = np.exp(0.015 * (finishers["race_index"] - max_idx))
    reg_boost = finishers["new_era"].apply(lambda x: 3.0 if x == 1 else 1.0)
    sample_weights = time_weights * reg_boost

    model = _make_model()
    model.fit(X, y, sample_weight=sample_weights)

    return {
        "model": model,
        "features": FEATURES,
        "training_data": df,
        "raw_history": raw_df,
        "feature_means": feature_means,
        "target": "delta",
    }


def train_model(years: list[int] = [2022, 2023, 2024, 2025, 2026]):
    print("Collecting training data...")
    df = collect_training_data(years)

    saved = fit_model(df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(saved, f)

    print(f"Model saved to {MODEL_PATH}")
    return saved["model"], saved["training_data"]


def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ─── Prediction ───────────────────────────────────────────────────────────────

def _soft_probabilities(positions: list, decay: float) -> list:
    scores = [np.exp(-decay * (p - 1)) for p in positions]
    total = sum(scores)
    return [round(s / total, 3) for s in scores]


def monte_carlo_probabilities(predicted_positions, dnf_rates, residual_std,
                              n_sims: int = N_SIMS, seed: int = 0,
                              dnf_floor: float = 0.0):
    """Monte Carlo finishing-order simulation → P(win), P(podium), P(points).

    Each simulation perturbs every car's predicted finish by Gaussian noise,
    independently draws each car as a DNF, sends DNF'd cars to the back (out of
    contention), re-ranks, and tallies. Probabilities are the frequency of
    win / top-3 / top-10 over sims.

    - `residual_std` may be a scalar OR a per-car array (position-dependent
      variance: tighter at the front, wider in the midfield).
    - `dnf_floor` is a minimum per-car incident probability (the "chaos floor"),
      so no car is ever treated as a near-certain finisher.

    Fully vectorized (n_sims x n_drivers) so it runs inline in the API.
    Returns three numpy arrays aligned to the input order.
    """
    pred = np.asarray(predicted_positions, dtype=float)
    dnf_p = np.clip(np.maximum(np.asarray(dnf_rates, dtype=float), dnf_floor), 0.0, 1.0)
    n = pred.shape[0]
    if n == 0:
        empty = np.zeros(0)
        return empty, empty, empty

    sigma = np.asarray(residual_std, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(n, float(sigma))

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_sims, n)) * sigma[None, :]
    sim_pos = pred[None, :] + noise

    # DNF draws → shove retired cars far to the back for that sim.
    dnf_draw = rng.random((n_sims, n)) < dnf_p[None, :]
    sim_pos = np.where(dnf_draw, sim_pos + 1000.0, sim_pos)

    # Rank within each sim (1 = best). Double argsort gives each car's rank.
    ranks = sim_pos.argsort(axis=1).argsort(axis=1) + 1

    p_win = (ranks == 1).mean(axis=0)
    p_podium = (ranks <= 3).mean(axis=0)
    p_points = (ranks <= 10).mean(axis=0)
    return p_win, p_podium, p_points


def _baseline_predict(qualifying_results: list[dict]) -> list[dict]:
    """Grid-based noisy fallback used when no trained model is available."""
    results = []
    for entry in qualifying_results:
        noise = np.random.normal(0, 1.2)
        pred_pos = max(1, min(20, entry["grid"] + noise))
        results.append({
            "driver": entry["driver"],
            "team": entry["team"],
            "grid_position": entry["grid"],
            "predicted_position": round(pred_pos, 1),
            "podium_probability": 0,
            "points_probability": 0,
        })
    results = sorted(results, key=lambda x: x["predicted_position"])
    positions = [r["predicted_position"] for r in results]
    for i, r in enumerate(results):
        r["podium_probability"] = _soft_probabilities(positions, 0.45)[i]
        r["points_probability"] = _soft_probabilities(positions, 0.22)[i]
    return results


def predict_race(year: int, grand_prix: str, qualifying_results: list[dict]) -> list[dict]:
    saved = load_model()
    if saved is None:
        return _baseline_predict(qualifying_results)
    return predict_with_model(saved, year, grand_prix, qualifying_results)


def predict_with_model(saved: dict, year: int, grand_prix: str,
                       qualifying_results: list[dict],
                       round_number: int = None) -> list[dict]:
    """Run the trained-model prediction path against an in-memory `saved` dict.

    Identical to the model branch of predict_race; factored out so the backtest
    harness can drive it with a per-race model/training_data without touching
    the on-disk MODEL_PATH. `round_number` is the round being predicted; if None
    it is inferred from how many of this season's races are already in history.
    """
    circuit = get_circuit_features(grand_prix)
    model = saved["model"]
    features = saved["features"]
    is_delta = saved.get("target") == "delta"

    # Legacy pickles (trained before the skew fix) lack raw_history; fall back to
    # the old inline feature path so production /predict keeps working on them.
    if "raw_history" not in saved:
        return _predict_with_model_legacy(saved, year, grand_prix, qualifying_results)

    raw = saved["raw_history"]
    feature_means = saved["feature_means"]

    # season_progress at serve time: infer the round from this season's races
    # already in history (or use the passed round), and normalize by the longest
    # season seen in history. Previously hardcoded 0.5 — the same train/serve
    # skew class we removed for the form features.
    if round_number is None:
        round_number = int(raw[raw["year"] == year]["round"].nunique()) + 1
    season_len = int(raw.groupby("year")["round"].nunique().max()) if len(raw) else round_number
    season_progress = min(1.0, round_number / max(season_len, round_number))

    # Build candidate rows for the upcoming race and engineer their features
    # through the SAME engineer_features() used in fit_model — single source of
    # truth, zero train/serve skew. A sentinel round places them chronologically
    # after all history, so only past data feeds their shifted/rolling features.
    next_round = int(raw["round"].max()) + 1 if len(raw) else 1
    candidates = pd.DataFrame([{
        "year": year,
        "round": next_round,
        "grand_prix": grand_prix,
        "driver": entry["driver"],
        "team": entry["team"],
        "grid_position": int(entry["grid"]),
        "finish_position": np.nan,
        "points": np.nan,
        "status": "Finished",
    } for entry in qualifying_results])

    combined = pd.concat([raw, candidates], ignore_index=True)
    engineered = engineer_features(combined, drop_incomplete=False)
    cand = engineered[(engineered["year"] == year) & (engineered["round"] == next_round)]
    cand_by_driver = {row["driver"]: row for _, row in cand.iterrows()}

    results = []
    for entry in qualifying_results:
        row = cand_by_driver[entry["driver"]]
        feature_row = pd.DataFrame([{
            "grid_position": entry["grid"],
            "form_3": row["form_3"],
            "points_form_3": row["points_form_3"],
            "team_form_3": row["team_form_3"],
            "dnf_rate": row["dnf_rate"],
            "season_progress": season_progress,
            "new_era": 1 if year >= 2026 else 0,
            "circuit_affinity": row["circuit_affinity"],
            "street": circuit["street"],
            "overtaking": circuit["overtaking"],
            "power_dep": circuit["power_dep"],
            "downforce_dep": circuit["downforce_dep"],
            "tire_deg": circuit["tire_deg"],
        }])
        # Fill residual NaNs (rookies / new teams) with the same training means
        # used in fit_model, so the fallback is identical on both sides.
        feature_row = feature_row.fillna(feature_means)

        available = [f for f in features if f in feature_row.columns]
        raw_pred = float(model.predict(feature_row[available])[0])

        # Delta models predict positions gained/lost; re-anchor to the grid.
        pred_pos = entry["grid"] + raw_pred if is_delta else raw_pred

        # Manual adjustments for specific track types (legacy, off by default)
        if USE_STREET_PULL:
            street_pull = circuit["street"] * 0.3
            pred_pos = (pred_pos * (1 - street_pull)) + (entry["grid"] * street_pull)

        results.append({
            "driver": entry["driver"],
            "team": entry["team"],
            "grid_position": entry["grid"],
            "predicted_position": round(max(1.0, min(20.0, pred_pos)), 2),
            "dnf_rate": round(float(np.clip(feature_row["dnf_rate"].iloc[0], 0.0, 1.0)), 3),
        })

    results = sorted(results, key=lambda x: x["predicted_position"])

    # Probabilities via Monte Carlo over the point predictions + DNF draws.
    # Position-dependent std and chaos floor are used if the model was shipped
    # with a fitted noise model; otherwise fall back to a single scalar std.
    posmodel = saved.get("position_noise")
    preds_list = [r["predicted_position"] for r in results]
    sigma = _position_std_array(preds_list, posmodel) if posmodel else \
        float(saved.get("residual_std", RESIDUAL_STD))
    p_win, p_podium, p_points = monte_carlo_probabilities(
        preds_list,
        [r["dnf_rate"] for r in results],
        sigma,
        dnf_floor=float(saved.get("dnf_floor", 0.0)),
    )
    for i, r in enumerate(results):
        r["win_probability"] = round(float(p_win[i]), 3)
        r["podium_probability"] = round(float(p_podium[i]), 3)
        r["points_probability"] = round(float(p_points[i]), 3)

    return results


def _predict_with_model_legacy(saved: dict, year: int, grand_prix: str,
                               qualifying_results: list[dict]) -> list[dict]:
    """Old inline-feature inference, kept only for pre-skew-fix on-disk pickles
    that have no `raw_history`. New models go through predict_with_model's unified
    engineer_features() path; retrain to migrate off this."""
    circuit = get_circuit_features(grand_prix)
    model = saved["model"]
    features = saved["features"]
    historical = saved["training_data"].sort_values(["year", "round"])

    def get_weighted_stat(df: pd.DataFrame, column: str, n: int = 10):
        if len(df) == 0: return None
        data = df.tail(n).copy()
        weights = np.exp(0.4 * np.arange(len(data)))
        return float(np.average(data[column].values, weights=weights))

    results = []
    for entry in qualifying_results:
        driver_hist = historical[historical["driver"] == entry["driver"]]
        team_hist = historical[historical["team"] == entry["team"]]

        form = get_weighted_stat(driver_hist, "finish_position")
        points_form = get_weighted_stat(driver_hist, "points") or 0.0
        team_form = get_weighted_stat(team_hist, "finish_position") or 10.0
        form = form if form is not None else team_form
        dnf_rate = float(driver_hist["dnf"].mean()) if len(driver_hist) > 0 else 0.1

        at_circuit = driver_hist[driver_hist["grand_prix"] == grand_prix]
        if len(at_circuit) >= 3:
            affinity = (at_circuit["finish_position"].mean() * 0.5) + (form * 0.5)
        else:
            affinity = form

        feature_row = pd.DataFrame([{
            "grid_position": entry["grid"],
            "form_3": form,
            "points_form_3": points_form,
            "team_form_3": team_form,
            "dnf_rate": dnf_rate,
            "season_progress": 0.5,
            "new_era": 1 if year >= 2026 else 0,
            "circuit_affinity": affinity,
            "street": circuit["street"],
            "overtaking": circuit["overtaking"],
            "power_dep": circuit["power_dep"],
            "downforce_dep": circuit["downforce_dep"],
            "tire_deg": circuit["tire_deg"],
        }])

        available = [f for f in features if f in feature_row.columns]
        pred_pos = float(model.predict(feature_row[available])[0])
        if saved.get("target") == "delta":
            pred_pos = entry["grid"] + pred_pos

        results.append({
            "driver": entry["driver"],
            "team": entry["team"],
            "grid_position": entry["grid"],
            "predicted_position": round(max(1.0, min(20.0, pred_pos)), 2),
        })

    results = sorted(results, key=lambda x: x["predicted_position"])
    positions = [r["predicted_position"] for r in results]
    ot = circuit["overtaking"]
    podium_probs = _soft_probabilities(positions, 0.45 + (1 - ot) * 0.3)
    points_probs = _soft_probabilities(positions, 0.22 + (1 - ot) * 0.1)
    for i, r in enumerate(results):
        r["podium_probability"] = podium_probs[i]
        r["points_probability"] = points_probs[i]
    return results


# ─── Model Status ─────────────────────────────────────────────────────────────

def get_model_status() -> dict:
    saved = load_model()
    if saved is None:
        return {"trained": False, "message": "No trained model found. Use /train endpoint."}
    df = saved["training_data"]
    return {
        "trained": True,
        "training_races": len(df.drop_duplicates(subset=["year", "round"])),
        "training_drivers": len(df["driver"].unique()),
        "years_covered": sorted(df["year"].unique().tolist()),
        "features": saved["features"],
    }


# ─── Backtesting ──────────────────────────────────────────────────────────────
# Measures the *current* model's accuracy. It reuses fit_model() and
# predict_with_model() verbatim — the same code the production path runs — so the
# numbers reflect the real model, not a re-implementation.
#
# Leakage prevention: for each held-out race (year Y, round K) we slice the raw
# results to rows strictly BEFORE it — (year < Y) OR (year == Y AND round < K) —
# and only then run engineer_features() + fit_model() on that slice. Because the
# feature engineering and the per-driver form lookups at inference time both
# operate on this before-only frame, no row from the test race or any later race
# can ever enter the model or its features.


def _build_qualifying_from_results(race_rows: pd.DataFrame) -> list[dict]:
    """Reconstruct the prediction input (driver, team, actual starting grid)."""
    entries = []
    for _, row in race_rows.iterrows():
        entries.append({
            "driver": row["driver"],
            "team": row["team"],
            "grid": int(row["grid_position"]),
        })
    return entries


def _grid_sort_key(grid: int) -> int:
    """Grid 0 means pit-lane / no grid slot → sort to the back, not the front."""
    return grid if grid > 0 else 99


def _grid_baseline_preds(qualifying_results: list[dict]) -> list[dict]:
    """Naive prediction: finishing order == starting grid order, no model.

    Returns the same `preds` shape (sorted best-first) that _race_metrics expects,
    so the grid baseline is scored through the exact same code as the model.
    """
    ordered = sorted(qualifying_results, key=lambda e: _grid_sort_key(e["grid"]))
    return [{"driver": e["driver"]} for e in ordered]


def _pole_winner_hit(race_rows: pd.DataFrame) -> float:
    """1.0 if the front-of-grid (pole) driver actually won, else 0.0."""
    valid = race_rows[race_rows["grid_position"] > 0]
    if valid.empty:
        return float("nan")
    pole_row = valid.loc[valid["grid_position"].idxmin()]
    return 1.0 if int(pole_row["finish_position"]) == 1 else 0.0


def _race_metrics(preds: list[dict], race_rows: pd.DataFrame) -> dict:
    """Score one race's predicted order against its actual finishing order."""
    actual = {row["driver"]: int(row["finish_position"]) for _, row in race_rows.iterrows()}

    # preds arrive sorted by predicted_position → index gives the predicted rank.
    pred_rank = {r["driver"]: i + 1 for i, r in enumerate(preds)}

    drivers = [d for d in pred_rank if d in actual]
    pr = [pred_rank[d] for d in drivers]
    ac = [actual[d] for d in drivers]

    mae = float(np.mean([abs(a - b) for a, b in zip(pr, ac)]))

    pred_winner = preds[0]["driver"]
    winner_hit = 1.0 if actual.get(pred_winner) == 1 else 0.0

    pred_top3 = {r["driver"] for r in preds[:3]}
    actual_top3 = {d for d, p in actual.items() if p <= 3}
    podium_hit = len(pred_top3 & actual_top3) / 3.0

    # Spearman via pandas (handles ties, no scipy dependency).
    spearman = (
        float(pd.Series(pr).corr(pd.Series(ac), method="spearman"))
        if len(pr) > 2 else float("nan")
    )

    return {"mae": mae, "winner_hit": winner_hit,
            "podium_hit": podium_hit, "spearman": spearman}


def _print_race_line(m: dict) -> None:
    print(
        f"  {m['year']} R{m['round']:>2} {m['grand_prix'][:30]:<30} "
        f"MAE={m['mae']:5.2f}  Win={'Y' if m['winner_hit'] else '-'}  "
        f"Podium={m['podium_hit'] * 3:.0f}/3  rho={m['spearman']:+.2f}"
    )


def _aggregate(per_race: list[dict]) -> dict:
    """Average the per-race metrics over all backtested races."""
    return {
        "races": len(per_race),
        "mae": float(np.mean([r["mae"] for r in per_race])),
        "winner_hit_rate": float(np.mean([r["winner_hit"] for r in per_race])),
        "podium_hit_rate": float(np.mean([r["podium_hit"] for r in per_race])),
        "spearman": float(np.nanmean([r["spearman"] for r in per_race])),
    }


def _print_comparison(model_agg: dict, naive_agg: dict, pole_won_rate: float) -> None:
    """Print model vs grid-order baseline side by side, with the delta."""
    print("\n" + "=" * 74)
    print(f"BACKTEST SUMMARY  ({model_agg['races']} races)")
    print("=" * 74)
    print(f"  {'Metric':<32}{'Model':>11}{'Grid (naive)':>14}{'d(model-naive)':>16}")
    print("  " + "-" * 70)

    def row(label: str, m: float, n: float, *,
            pct: bool, lower_better: bool, signed: bool = False) -> None:
        delta = m - n
        if pct:
            mstr, nstr, dstr = f"{m*100:.1f}%", f"{n*100:.1f}%", f"{delta*100:+.1f}pts"
        else:
            vfmt = "{:+.3f}" if signed else "{:.3f}"
            mstr, nstr, dstr = vfmt.format(m), vfmt.format(n), f"{delta:+.3f}"
        # Mark which side is better for this metric.
        better = (delta < 0) if lower_better else (delta > 0)
        flag = "  <- model" if better and abs(delta) > 1e-9 else ""
        print(f"  {label:<32}{mstr:>11}{nstr:>14}{dstr:>16}{flag}")

    row("Mean Absolute Error (pos)", model_agg["mae"], naive_agg["mae"],
        pct=False, lower_better=True)
    row("Winner hit rate", model_agg["winner_hit_rate"], naive_agg["winner_hit_rate"],
        pct=True, lower_better=False)
    row("Podium hit rate (of top 3)", model_agg["podium_hit_rate"], naive_agg["podium_hit_rate"],
        pct=True, lower_better=False)
    row("Spearman rank correlation", model_agg["spearman"], naive_agg["spearman"],
        pct=False, lower_better=False, signed=True)
    print("  " + "-" * 70)
    print(f"  Pole sitter actually won:       {pole_won_rate * 100:5.1f}%  "
          f"(= the grid baseline's winner prediction)")
    print("=" * 74)


def backtest(test_years: list[int], test_rounds: list[int] = None,
             history_start: int = 2022) -> dict:
    """Walk-forward backtest of the current model.

    For every race in `test_years` (optionally narrowed to `test_rounds`), train
    a fresh model on all races strictly before it and score the prediction
    against the real result. History is pulled from `history_start` onward.
    """
    test_years = sorted(test_years)
    all_years = list(range(min(history_start, min(test_years)), max(test_years) + 1))

    print(f"Collecting data for {all_years} (first run downloads via FastF1)...")
    raw = collect_training_data(all_years)
    if raw.empty:
        raise RuntimeError("No data collected — check FastF1 availability.")

    per_race = []
    naive_per_race = []
    pole_won = []
    for ty in test_years:
        year_races = raw[raw["year"] == ty]
        rounds = sorted(int(r) for r in year_races["round"].unique())
        if test_rounds:
            rounds = [r for r in rounds if r in test_rounds]

        for rnd in rounds:
            race_rows = year_races[year_races["round"] == rnd]
            if race_rows.empty:
                continue
            gp_name = race_rows.iloc[0]["grand_prix"]

            # Strictly-before slice — this is the leakage barrier.
            before = raw[(raw["year"] < ty) | ((raw["year"] == ty) & (raw["round"] < rnd))]
            if before.empty:
                print(f"  skip {ty} R{rnd} {gp_name}: no prior data")
                continue
            if len(before) < 20:
                print(f"  skip {ty} R{rnd} {gp_name}: insufficient training rows")
                continue

            # fit_model engineers features internally (same fn inference uses).
            saved = fit_model(before.copy())
            quali = _build_qualifying_from_results(race_rows)
            preds = predict_with_model(saved, ty, gp_name, quali, round_number=rnd)

            metrics = _race_metrics(preds, race_rows)
            metrics.update({"year": ty, "round": rnd, "grand_prix": gp_name, "n": len(quali)})
            per_race.append(metrics)

            # Naive baseline: grid order, scored through the exact same metrics.
            naive_metrics = _race_metrics(_grid_baseline_preds(quali), race_rows)
            naive_metrics.update({"year": ty, "round": rnd, "grand_prix": gp_name})
            naive_per_race.append(naive_metrics)
            pole_won.append(_pole_winner_hit(race_rows))

            _print_race_line(metrics)

    if not per_race:
        print("No races were backtested.")
        return None

    agg = _aggregate(per_race)
    naive_agg = _aggregate(naive_per_race)
    pole_won_rate = float(np.nanmean(pole_won))
    _print_comparison(agg, naive_agg, pole_won_rate)
    return {
        "per_race": per_race,
        "aggregate": agg,
        "naive_per_race": naive_per_race,
        "naive_aggregate": naive_agg,
        "pole_won_rate": pole_won_rate,
    }


# ─── Probability calibration validation ───────────────────────────────────────

def _walk_forward_races(raw: pd.DataFrame, test_years: list[int],
                        test_rounds: list[int] = None,
                        history_start: int = 2022) -> list[dict]:
    """Walk-forward over the test set, returning per-race OOS prediction records.

    Each record holds the model's point prediction plus the actual finish and the
    car's dnf_rate, so callers can both calibrate residuals and score probabilities
    — all strictly out-of-sample (every model is trained on before-only data)."""
    test_years = sorted(test_years)
    races = []
    for ty in test_years:
        year_races = raw[raw["year"] == ty]
        rounds = sorted(int(r) for r in year_races["round"].unique())
        if test_rounds:
            rounds = [r for r in rounds if r in test_rounds]

        for rnd in rounds:
            race_rows = year_races[year_races["round"] == rnd]
            if race_rows.empty:
                continue
            gp_name = race_rows.iloc[0]["grand_prix"]
            before = raw[(raw["year"] < ty) | ((raw["year"] == ty) & (raw["round"] < rnd))]
            if len(before) < 20:
                continue

            saved = fit_model(before.copy())
            quali = _build_qualifying_from_results(race_rows)
            preds = predict_with_model(saved, ty, gp_name, quali, round_number=rnd)

            actual = {r["driver"]: int(r["finish_position"]) for _, r in race_rows.iterrows()}
            # Same DNF definition engineer_features uses, for residual exclusion.
            dnf_flag = (~race_rows["status"].str.contains("Finished|Lapped", na=False))
            dnf_actual = {r["driver"]: bool(d) for (_, r), d in zip(race_rows.iterrows(), dnf_flag)}

            recs = [{
                "driver": p["driver"],
                "pred": p["predicted_position"],
                "dnf_rate": p["dnf_rate"],
                "actual": actual[p["driver"]],
                "actual_dnf": dnf_actual[p["driver"]],
            } for p in preds if p["driver"] in actual]

            races.append({"year": ty, "round": rnd, "grand_prix": gp_name, "records": recs})
            print(f"  walk-forward {ty} R{rnd:>2} {gp_name[:28]:<28} ({len(recs)} cars)")
    return races


def _reliability_table(market: str, pairs: list[tuple]) -> None:
    """Print a decile reliability table: predicted prob vs empirical frequency."""
    probs = np.array([p for p, _ in pairs])
    outs = np.array([o for _, o in pairs], dtype=float)
    print(f"\nReliability - {market}  (n={len(pairs)})")
    print(f"  {'bin':<12}{'count':>7}{'pred':>10}{'empirical':>12}")
    print("  " + "-" * 41)
    edges = np.round(np.arange(0.0, 1.0001, 0.1), 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi) if hi >= 1.0 else (probs >= lo) & (probs < hi)
        c = int(mask.sum())
        if c == 0:
            continue
        print(f"  [{lo:.1f},{hi:.1f}){c:>7}{probs[mask].mean():>10.3f}{outs[mask].mean():>12.3f}")


def _proba_scores(pairs: list[tuple]) -> dict:
    probs = np.array([p for p, _ in pairs])
    outs = np.array([o for _, o in pairs], dtype=float)
    brier = float(np.mean((probs - outs) ** 2))
    pc = np.clip(probs, 1e-15, 1 - 1e-15)
    logloss = float(-np.mean(outs * np.log(pc) + (1 - outs) * np.log(1 - pc)))
    return {"brier": brier, "logloss": logloss, "base_rate": float(outs.mean())}


# ─── Noise model: position-dependent variance + chaos floor ───────────────────

def _finisher_residuals(races: list[dict]):
    """(predicted_position, residual) pairs for classified finishers only."""
    preds, res = [], []
    for race in races:
        for r in race["records"]:
            if not r["actual_dnf"]:
                preds.append(r["pred"])
                res.append(r["actual"] - r["pred"])
    return np.array(preds), np.array(res)


def _fit_position_std(races: list[dict]) -> dict:
    """Fit a per-position-bin noise std from OOS residuals (heteroscedastic).

    A few coarse predicted-position bins; bins with too few samples fall back to
    the global std. Returns {edges, stds, global}."""
    preds, res = _finisher_residuals(races)
    global_std = float(np.std(res)) if len(res) else RESIDUAL_STD
    stds = []
    for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]):
        mask = (preds >= lo) & (preds < hi)
        stds.append(float(np.std(res[mask])) if int(mask.sum()) > 20 else global_std)
    return {"edges": list(POS_BINS), "stds": stds, "global": global_std}


def _position_std_array(predicted_positions, posmodel: dict):
    """Map each predicted position to its bin's std."""
    edges, stds = posmodel["edges"], posmodel["stds"]
    pp = np.asarray(predicted_positions, dtype=float)
    idx = np.clip(np.searchsorted(edges, pp, side="right") - 1, 0, len(stds) - 1)
    return np.asarray(stds, dtype=float)[idx]


def _calibration_pairs(races, single_std, posmodel, dnf_floor, n_sims, seed):
    """Run the sim over every race and collect (prob, outcome) pairs + prob sums.

    `single_std` (scalar) is used when `posmodel` is None, else position-dependent
    stds are derived from each car's predicted position."""
    markets = ["win", "podium", "points"]
    pairs = {m: [] for m in markets}
    sums = {m: [] for m in markets}
    for race in races:
        recs = race["records"]
        preds = [r["pred"] for r in recs]
        sigma = _position_std_array(preds, posmodel) if posmodel else float(single_std)
        p = dict(zip(markets, monte_carlo_probabilities(
            preds, [r["dnf_rate"] for r in recs], sigma,
            n_sims=n_sims, seed=seed, dnf_floor=dnf_floor)))
        for m in markets:
            sums[m].append(float(p[m].sum()))
        for i, r in enumerate(recs):
            a = r["actual"]
            outcome = {"win": int(a == 1), "podium": int(a <= 3), "points": int(a <= 10)}
            for m in markets:
                pairs[m].append((float(p[m][i]), outcome[m]))
    return pairs, sums


def _fit_chaos_floor(fit_races, posmodel, n_sims, seed) -> float:
    """Grid-search a minimum per-car DNF probability so the top points decile on
    the FIT set lands near its empirical rate (fixes near-lock overconfidence)."""
    best_floor, best_err = 0.0, float("inf")
    for floor in np.round(np.arange(0.0, 0.201, 0.01), 2):
        pairs, _ = _calibration_pairs(fit_races, None, posmodel, float(floor), n_sims, seed)
        probs = np.array([p for p, _ in pairs["points"]])
        outs = np.array([o for _, o in pairs["points"]], dtype=float)
        mask = probs >= 0.9
        if int(mask.sum()) < 10:
            continue
        err = abs(probs[mask].mean() - outs[mask].mean())
        if err < best_err:
            best_err, best_floor = err, float(floor)
    return best_floor


def _evaluate_calibration(races, label, single_std, posmodel, dnf_floor,
                          n_sims, seed) -> dict:
    """Print sanity sums, reliability tables and scores for one noise config."""
    print("\n" + "#" * 64)
    print(f"# {label}")
    print("#" * 64)
    markets = ["win", "podium", "points"]
    pairs, sums = _calibration_pairs(races, single_std, posmodel, dnf_floor, n_sims, seed)
    print("Sanity (mean per race):  "
          f"sum P(win)={np.mean(sums['win']):.3f}  "
          f"P(podium)={np.mean(sums['podium']):.3f}  "
          f"P(points)={np.mean(sums['points']):.3f}")
    scores = {}
    for m in markets:
        _reliability_table(m, pairs[m])
        scores[m] = _proba_scores(pairs[m])
    print("\n  " + f"{'Market':<10}{'Brier':>12}{'LogLoss':>12}{'BaseRate':>12}")
    print("  " + "-" * 46)
    for m in markets:
        s = scores[m]
        print(f"  {m:<10}{s['brier']:>12.4f}{s['logloss']:>12.4f}{s['base_rate']:>12.3f}")
    return scores


def backtest_calibration(test_years: list[int], test_rounds: list[int] = None,
                         history_start: int = 2022, n_sims: int = N_SIMS,
                         seed: int = 0) -> dict:
    """Validate the Monte Carlo probabilities out-of-sample over the walk-forward.

    Calibrates a single noise std from OOS residuals, generates win/podium/points
    probabilities for every race, then reports reliability tables and Brier /
    log-loss scores per market."""
    test_years = sorted(test_years)
    all_years = list(range(min(history_start, min(test_years)), max(test_years) + 1))
    print(f"Collecting data for {all_years} (first run downloads via FastF1)...")
    raw = collect_training_data(all_years)
    if raw.empty:
        raise RuntimeError("No data collected — check FastF1 availability.")

    print("Walk-forward predictions...")
    races = _walk_forward_races(raw, test_years, test_rounds, history_start)
    if not races:
        print("No races were backtested.")
        return None

    # Calibrate noise from OUT-OF-SAMPLE residuals, finishers only (a DNF's
    # filled-in 20 is not a point-prediction error). Single global std — each
    # residual is OOS; only the scalar aggregates across the whole set.
    residuals = [r["actual"] - r["pred"]
                 for race in races for r in race["records"] if not r["actual_dnf"]]
    residual_std = float(np.std(residuals))
    print(f"\nCalibrated residual std (OOS, {len(residuals)} finisher rows): "
          f"{residual_std:.3f} positions")

    scores = _evaluate_calibration(
        races, f"single std = {residual_std:.3f}  (no chaos floor)",
        residual_std, None, 0.0, n_sims, seed)
    return {"residual_std": residual_std, "scores": scores}


def backtest_calibration_holdout(fit_years: list[int], holdout_years: list[int],
                                 history_start: int = 2022, n_sims: int = N_SIMS,
                                 seed: int = 0) -> dict:
    """Honest calibration: fit the NOISE model on `fit_years` OOS residuals, then
    validate the reliability tables + Brier on the held-out `holdout_years`.

    The point model stays walk-forward everywhere. Only the noise layer
    (position-dependent std + chaos floor) is fit on the fit window, so the
    calibration check on the holdout is genuinely out-of-sample. Reports
    before (single std) vs after (position std + chaos floor) on the holdout."""
    fit_years, holdout_years = sorted(fit_years), sorted(holdout_years)
    lo = min(history_start, min(fit_years), min(holdout_years))
    all_years = list(range(lo, max(holdout_years) + 1))
    print(f"Collecting data for {all_years} (first run downloads via FastF1)...")
    raw = collect_training_data(all_years)
    if raw.empty:
        raise RuntimeError("No data collected — check FastF1 availability.")

    print(f"\nWalk-forward predictions on FIT years {fit_years}...")
    fit_races = _walk_forward_races(raw, fit_years, None, history_start)
    print(f"\nWalk-forward predictions on HOLDOUT years {holdout_years}...")
    hold_races = _walk_forward_races(raw, holdout_years, None, history_start)
    if not fit_races or not hold_races:
        print("Not enough races to fit/validate.")
        return None

    # Fit the noise model on the FIT window only.
    _, fit_res = _finisher_residuals(fit_races)
    before_std = float(np.std(fit_res))
    posmodel = _fit_position_std(fit_races)
    dnf_floor = _fit_chaos_floor(fit_races, posmodel, n_sims, seed)

    print("\n" + "=" * 64)
    print(f"NOISE MODEL fit on {fit_years} ({len(fit_res)} finisher residuals)")
    print("=" * 64)
    print(f"  Before: single std = {before_std:.3f}")
    print("  After:  position-dependent std by predicted-position bin:")
    for (lo_b, hi_b), s in zip(zip(posmodel["edges"][:-1], posmodel["edges"][1:]),
                               posmodel["stds"]):
        print(f"            P{lo_b:>2}-{hi_b - 1:<2}: std = {s:.3f}")
    print(f"          chaos floor (min DNF prob) = {dnf_floor:.2f}")

    # Validate both configs on the HOLDOUT.
    before = _evaluate_calibration(
        hold_races, f"BEFORE  -  single std={before_std:.3f}, no chaos floor  "
                    f"[holdout {holdout_years}]", before_std, None, 0.0, n_sims, seed)
    after = _evaluate_calibration(
        hold_races, f"AFTER  -  position std + chaos floor={dnf_floor:.2f}  "
                    f"[holdout {holdout_years}]", None, posmodel, dnf_floor, n_sims, seed)

    print("\n" + "=" * 64)
    print(f"BRIER BEFORE vs AFTER  (holdout {holdout_years}, lower = better)")
    print("=" * 64)
    print(f"  {'Market':<10}{'Before':>12}{'After':>12}{'Delta':>12}")
    print("  " + "-" * 46)
    for m in ["win", "podium", "points"]:
        b, a = before[m]["brier"], after[m]["brier"]
        print(f"  {m:<10}{b:>12.4f}{a:>12.4f}{a - b:>+12.4f}")
    print("=" * 64)

    return {"before_std": before_std, "position_noise": posmodel,
            "dnf_floor": dnf_floor, "before": before, "after": after}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest the F1 race predictor.")
    parser.add_argument("--years", default="2025",
                        help="Comma-separated test years (default: 2025)")
    parser.add_argument("--rounds", default=None,
                        help="Comma-separated rounds to limit to (default: all)")
    parser.add_argument("--history-start", type=int, default=2022,
                        help="Earliest season to train history from (default: 2022)")
    parser.add_argument("--street-pull", action="store_true",
                        help="Re-enable the legacy street-circuit grid pull (off by default)")
    parser.add_argument("--include-dnf", action="store_true",
                        help="Attribution: train the delta target on DNF rows too (off by default)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibration holdout: fit noise on all but the last --years, "
                             "validate on the last (before/after reliability + Brier)")
    parser.add_argument("--sims", type=int, default=N_SIMS,
                        help=f"Monte Carlo simulations per race (default: {N_SIMS})")
    args = parser.parse_args()

    USE_STREET_PULL = args.street_pull
    EXCLUDE_DNF = not args.include_dnf

    years = [int(y) for y in args.years.split(",")]
    rounds = [int(r) for r in args.rounds.split(",")] if args.rounds else None
    if args.calibrate:
        if len(years) < 2:
            parser.error("--calibrate needs >=2 years: fit on all but the last, hold out the last")
        backtest_calibration_holdout(years[:-1], [years[-1]],
                                     history_start=args.history_start, n_sims=args.sims)
    else:
        backtest(years, rounds, history_start=args.history_start)

