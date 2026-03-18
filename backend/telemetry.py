import fastf1
import pandas as pd
import numpy as np
from pathlib import Path

# Enable cache to avoid re-downloading data
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_session(year: int, grand_prix: str, session_type: str = "R"):
    """Load an F1 session. session_type: R=Race, Q=Qualifying, FP1/FP2/FP3"""
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()
    return session


def get_driver_lap_telemetry(year: int, grand_prix: str, session_type: str, driver: str, lap_number: int):
    """
    Returns telemetry for a specific driver lap.
    Data includes: Distance, Speed, Throttle, Brake, RPM, Gear, DRS
    """
    session = get_session(year, grand_prix, session_type)
    laps = session.laps.pick_driver(driver)
    if laps.empty:
        raise ValueError(f"No lap data found for driver {driver} in this session.")

    target_lap = laps[laps["LapNumber"] == lap_number]
    if target_lap.empty:
        raise ValueError(f"Lap {lap_number} not found for driver {driver}.")
    
    lap = target_lap.iloc[0]
    tel = lap.get_telemetry().add_distance()

    result = {
        "driver": driver,
        "lap_number": lap_number,
        "lap_time": str(lap["LapTime"]),
        "compound": lap["Compound"] if "Compound" in lap else "Unknown",
        "telemetry": {
            "distance": tel["Distance"].tolist(),
            "speed": tel["Speed"].tolist(),
            "throttle": tel["Throttle"].tolist(),
            "brake": tel["Brake"].astype(int).tolist(),
            "rpm": tel["RPM"].tolist(),
            "gear": tel["nGear"].tolist(),
            "drs": tel["DRS"].tolist(),
            "x": tel["X"].tolist(),
            "y": tel["Y"].tolist(),
        }
    }
    return result


def compare_laps(year: int, grand_prix: str, session_type: str,
                 driver1: str, lap1: int, driver2: str, lap2: int):
    """
    Compare telemetry between two drivers/laps.
    Returns both datasets normalized to distance for overlay charts.
    """
    session = get_session(year, grand_prix, session_type)

    def extract(driver, lap_number):
        laps = session.laps.pick_driver(driver)
        target_lap = laps[laps["LapNumber"] == lap_number]
        if target_lap.empty:
            raise ValueError(f"Lap {lap_number} not found for driver {driver}.")
        lap = target_lap.iloc[0]
        tel = lap.get_telemetry().add_distance()
        return {
            "driver": driver,
            "lap_number": lap_number,
            "lap_time": str(lap["LapTime"]),
            "compound": lap.get("Compound", "Unknown"),
            "telemetry": {
                "distance": tel["Distance"].tolist(),
                "speed": tel["Speed"].tolist(),
                "throttle": tel["Throttle"].tolist(),
                "brake": tel["Brake"].astype(int).tolist(),
                "rpm": tel["RPM"].tolist(),
                "gear": tel["nGear"].tolist(),
            }
        }

    return {
        "session": f"{year} {grand_prix} {session_type}",
        "driver1": extract(driver1, lap1),
        "driver2": extract(driver2, lap2),
    }


def get_fastest_lap_telemetry(year: int, grand_prix: str, session_type: str, driver: str):
    """Get telemetry for a driver's fastest lap in the session."""
    session = get_session(year, grand_prix, session_type)
    
    driver_laps = session.laps.pick_driver(driver)
    
    if driver_laps.empty:
        raise ValueError(f"No lap data found for driver {driver} in this session.")
        
    fastest_lap = driver_laps.pick_fastest()
    
    if pd.isna(fastest_lap['LapTime']):
        raise ValueError(f"Driver {driver} did not set a valid lap time in this session.")
        
    tel = fastest_lap.get_telemetry().add_distance()

    return {
        "driver": driver,
        "lap_number": int(fastest_lap["LapNumber"]),
        "lap_time": str(fastest_lap["LapTime"]),
        "compound": fastest_lap.get("Compound", "Unknown"),
        "telemetry": {
            "distance": tel["Distance"].tolist(),
            "speed": tel["Speed"].tolist(),
            "throttle": tel["Throttle"].tolist(),
            "brake": tel["Brake"].astype(int).tolist(),
            "rpm": tel["RPM"].tolist(),
            "gear": tel["nGear"].tolist(),
            "drs": tel["DRS"].tolist(),
            "x": tel["X"].tolist(),
            "y": tel["Y"].tolist(),
        }
    }

def get_session_drivers(year: int, grand_prix: str, session_type: str):
    """List all drivers in a session."""
    session = get_session(year, grand_prix, session_type)
    drivers = session.laps["Driver"].unique().tolist()
    results = []
    for d in drivers:
        try:
            info = session.get_driver(d)
            results.append({
                "abbreviation": d,
                "full_name": info.get("FullName", d),
                "team": info.get("TeamName", "Unknown"),
                "team_color": "#" + info.get("TeamColor", "FFFFFF"),
            })
        except Exception:
            results.append({"abbreviation": d, "full_name": d, "team": "Unknown", "team_color": "#FFFFFF"})
    return results


def get_lap_times(year: int, grand_prix: str, session_type: str, driver: str):
    """Return all lap times for a driver in a session."""
    session = get_session(year, grand_prix, session_type)
    laps = session.laps.pick_driver(driver)

    result = []
    for _, lap in laps.iterrows():
        if pd.notna(lap["LapTime"]):
            result.append({
                "lap_number": int(lap["LapNumber"]),
                "lap_time_seconds": lap["LapTime"].total_seconds(),
                "lap_time_str": str(lap["LapTime"]),
                "compound": lap.get("Compound", "Unknown"),
                "is_personal_best": bool(lap.get("IsPersonalBest", False)),
            })
    return result
