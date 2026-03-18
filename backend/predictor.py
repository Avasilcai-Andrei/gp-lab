import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

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

def get_circuit_features(grand_prix: str) -> dict:
    """Match a GP name to circuit characteristics, with fuzzy fallback."""
    gp = grand_prix.strip()
    if gp in CIRCUIT_FEATURES:
        return CIRCUIT_FEATURES[gp]
    gp_lower = gp.lower()
    for alias, key in CIRCUIT_ALIASES.items():
        if alias in gp_lower:
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # Drop any rows where we still don't have essential data
    return df.dropna(subset=["form_3", "grid_position"])



# ─── Model Training ───────────────────────────────────────────────────────────

def train_model(years: list[int] = [2022, 2023, 2024, 2025, 2026]):

    from xgboost import XGBRegressor
    
    print("Collecting training data...")
    df = collect_training_data(years)
    df = engineer_features(df)

    features = [
    "grid_position", "form_3", "points_form_3", "team_form_3",
    "dnf_rate", "season_progress", "new_era", "circuit_affinity",
    "street", "overtaking", "power_dep", "downforce_dep", "tire_deg",
]

    X = df[features].fillna(df[features].mean())
    y = df["finish_position"]

    # Time-decay weights: recent races count exponentially more
    # Combined with a reg-change boost for 2022/2026 races
    df = df.copy()
    df["race_index"] = df.groupby(["year", "round"]).ngroup()
    max_idx = df["race_index"].max()
    time_weights = np.exp(0.015 * (df["race_index"] - max_idx))
    reg_boost = df["new_era"].apply(lambda x: 3.0 if x == 1 else 1.0)
    sample_weights = time_weights * reg_boost

    model = XGBRegressor(
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
    model.fit(X, y, sample_weight=sample_weights)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": features, "training_data": df}, f)

    print(f"Model saved to {MODEL_PATH}")
    return model, df


def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_race(year: int, grand_prix: str, qualifying_results: list[dict]) -> list[dict]:
    saved = load_model()
    circuit = get_circuit_features(grand_prix)

    def soft_probabilities(positions: list, decay: float) -> list:
        scores = [np.exp(-decay * (p - 1)) for p in positions]
        total = sum(scores)
        return [round(s / total, 3) for s in scores]

    # --- Baseline Prediction (No Model) ---
    if saved is None:
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
            r["podium_probability"] = soft_probabilities(positions, 0.45)[i]
            r["points_probability"] = soft_probabilities(positions, 0.22)[i]
        return results

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

        # Feature Extraction with Fallbacks
        form = get_weighted_stat(driver_hist, "finish_position")
        points_form = get_weighted_stat(driver_hist, "points") or 0.0
        team_form = get_weighted_stat(team_hist, "finish_position") or 10.0
        
        # Rookie logic: use team form if driver form is missing
        form = form if form is not None else team_form
        dnf_rate = float(driver_hist["dnf"].mean()) if len(driver_hist) > 0 else 0.1
        
        # Circuit Affinity
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

        # Inference
        available = [f for f in features if f in feature_row.columns]
        pred_pos = float(model.predict(feature_row[available])[0])

        # Manual adjustments for specific track types
        street_pull = circuit["street"] * 0.3
        pred_pos = (pred_pos * (1 - street_pull)) + (entry["grid"] * street_pull)
        
        results.append({
            "driver": entry["driver"],
            "team": entry["team"],
            "grid_position": entry["grid"],
            "predicted_position": round(max(1.0, min(20.0, pred_pos)), 2),
        })

    # Sort and Calculate Probabilities
    results = sorted(results, key=lambda x: x["predicted_position"])
    positions = [r["predicted_position"] for r in results]
    
    ot = circuit["overtaking"]
    podium_probs = soft_probabilities(positions, 0.45 + (1 - ot) * 0.3)
    points_probs = soft_probabilities(positions, 0.22 + (1 - ot) * 0.1)

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
    
