from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from telemetry import (
    get_driver_lap_telemetry,
    get_fastest_lap_telemetry,
    compare_laps,
    get_session_drivers,
    get_lap_times,
)
from predictor import predict_race, train_model, get_model_status

app = FastAPI(
    title="F1 Telemetry & Predictor API",
    description="Real F1 data via FastF1 + ML race predictions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Telemetry Endpoints ─────────────────────────────────────────────────────

@app.get("/telemetry/drivers")
def list_drivers(year: int, gp: str, session: str = "R"):
    """List all drivers in a session."""
    try:
        return get_session_drivers(year, gp, session)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/telemetry/fastest-lap")
def fastest_lap(year: int, gp: str, driver: str, session: str = "Q"):
    """Get telemetry for a driver's fastest lap."""
    try:
        return get_fastest_lap_telemetry(year, gp, session, driver)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/telemetry/lap")
def lap_telemetry(year: int, gp: str, driver: str, lap: int, session: str = "R"):
    """Get telemetry for a specific lap number."""
    try:
        return get_driver_lap_telemetry(year, gp, session, driver, lap)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/telemetry/compare")
def compare(
    year: int, gp: str,
    driver1: str, lap1: int,
    driver2: str, lap2: int,
    session: str = "Q"
):
    """Compare two drivers' telemetry on the same chart."""
    try:
        return compare_laps(year, gp, session, driver1, lap1, driver2, lap2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/telemetry/lap-times")
def lap_times(year: int, gp: str, driver: str, session: str = "R"):
    """Get all lap times for a driver in a session."""
    try:
        return get_lap_times(year, gp, session, driver)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── ML Prediction Endpoints ─────────────────────────────────────────────────

class QualifyingEntry(BaseModel):
    driver: str
    team: str
    grid: int


class PredictionRequest(BaseModel):
    year: int
    grand_prix: str
    qualifying_results: list[QualifyingEntry]


@app.post("/predict/race")
def predict(request: PredictionRequest):
    """Predict race finishing order from qualifying results."""
    try:
        entries = [e.dict() for e in request.qualifying_results]
        results = predict_race(request.year, request.grand_prix, entries)
        return {"grand_prix": request.grand_prix, "year": request.year, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/train")
def trigger_training(background_tasks: BackgroundTasks, years: str = "2022,2023"):
    """Trigger model training in the background. Pass comma-separated years."""
    year_list = [int(y.strip()) for y in years.split(",")]
    background_tasks.add_task(train_model, year_list)
    return {"message": f"Training started for years: {year_list}. Check /predict/status."}


@app.get("/predict/status")
def model_status():
    """Get current model training status and metadata."""
    return get_model_status()


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "F1 API running 🏎️"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
