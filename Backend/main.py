# main.py - Crisis Connect API
from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional, Any
from pydantic import BaseModel
from services.predict import collect_all_data, generate_risk_scores, calculate_household_resources, DISTRICT_COORDS
from services.alert_generate import generate_alerts_from_db
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import logging
from models.model import AlertModel, SimulateRequest
from datetime import datetime
from utils.db import init_mongo, close_mongo, get_db, ensure_indexes
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- App Setup ---
app = FastAPI(
    title="Crisis Connect API",
    description="Real-time flood risk prediction, alerting, and resource planning",
    version="1.0.0"
)

# --- CORS Middleware ---
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000, https://crisisconnect.streamlit.app. http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
model = None  # Will be loaded on startup
executor = ThreadPoolExecutor(max_workers=2)

# --- Geocoder ---
_geolocator = Nominatim(user_agent="crisis-connect")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1)

# --- File Paths ---
HISTORICAL_XLSX = "data_disaster.xlsx"
WEATHER_CSV = "latest_data.csv"
ALERTS_CSV = "alerts.csv"

# --- Helper Functions ---
def serialize_doc(doc):
    """Convert MongoDB doc for JSON serialization."""
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

def _json_safe(value):
    if isinstance(value, (np.floating, float)):
        try:
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def _df_to_json_records(df: pd.DataFrame) -> List[dict]:
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    return [{k: _json_safe(v) for k, v in r.items()} for r in records]

def _sanitize_records(records: List[dict]) -> List[dict]:
    return [{k: _json_safe(v) for k, v in r.items()} for r in records]

def _strip_mongo_ids(records: List[dict]) -> List[dict]:
    for r in records:
        r.pop("_id", None)
    return records

# --- Startup & Shutdown ---
@app.on_event("startup")
async def startup_event():
    global model
    await init_mongo(app)
    try:
        db = get_db(app)
        await ensure_indexes(db)
        logger.info("✅ MongoDB indexes ensured")
    except Exception as e:
        logger.warning(f"Failed to ensure MongoDB indexes: {e}")

    # Load ML model
    try:
        model = joblib.load("rf_model.pkl")
        logger.info("✅ ML Model loaded successfully")
    except Exception as e:
        logger.critical(f"❌ Failed to load model: {e}")
        raise RuntimeError("Model not found. Run `python predict.py` first.")

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo(app)
    executor.shutdown(wait=True)

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})

# --- Health Check ---
@app.get("/health", summary="Service health check")
async def health():
    try:
        db = get_db(app)
        await db.command("ping")
        if model is None:
            return JSONResponse(status_code=503, content={"status": "error", "detail": "Model not loaded"})
        return {"status": "ok", "model_loaded": True, "db_connected": True}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})

# --- API Endpoints ---

@app.get("/")
def root():
    return {"message": "Welcome to Crisis Connect API", "docs": "/docs"}

# --- Weather Data Collection ---
@app.get("/collect", summary="Collect weather data for predefined districts")
async def collect_data():
    try:
        df = collect_all_data()
        count = len(df)
        if count > 0:
            db = get_db(app)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            from pymongo import UpdateOne
            ops = [
                UpdateOne(
                    {"location": r["location"], "timestamp": r.get("timestamp") or now},
                    {"$set": r},
                    upsert=True
                )
                for r in _df_to_json_records(df)
            ]
            await db["weather_data"].bulk_write(ops)
        logger.info(f"Collected {count} weather records")
        return {"message": "Data collected", "count": count}
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

class CollectRequest(BaseModel):
    locations: Optional[Any] = None

@app.post("/collect", summary="Collect weather data for custom locations")
async def collect_data_custom(payload: CollectRequest):
    try:
        df = collect_all_data(payload.locations)
        count = len(df)
        if count > 0:
            db = get_db(app)
            records = _df_to_json_records(df)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            from pymongo import UpdateOne
            ops = [
                UpdateOne(
                    {"location": r["location"], "timestamp": r.get("timestamp") or now},
                    {"$set": r},
                    upsert=True
                )
                for r in records
            ]
            await db["weather_data"].bulk_write(ops)
            records = records if count <= 50 else []
        return {"message": "Data collected", "count": count, "records": records}
    except Exception as e:
        logger.error(f"Error in custom collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Risk Assessment ---
@app.get("/risk-assessment", summary="Get latest risk scores (read-only)")
async def assess_risk():
    db = get_db(app)
    
    # Just read from predictions collection
    try:
        predictions = await db["predictions"].find(
            {},
            {"_id": 0}  # Exclude MongoDB ID
        ).sort("timestamp", -1).to_list(length=10000)

        if not predictions:
            logger.warning("No predictions found in DB")
            return []

        logger.info(f"✅ Returned {len(predictions)} predictions")
        return predictions

    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve risk assessments")
    
# --- Predict Endpoint ---
# main.py

@app.post("/predict", summary="Score weather data using ML models")
async def predict_risk(generate_alerts: bool = False):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    db = get_db(app)

    # --- Step 1: Load existing predictions (e.g., from simulation) ---
    existing_predictions = await db["predictions"].find(
        {"composite_risk_score": {"$exists": True}},
        {"location": 1, "composite_risk_score": 1, "scenario": 1}
    ).to_list(length=10000)

    # Identify protected locations: simulated or high-risk (>= 70)
    protected_locations = {
        p["location"]: p
        for p in existing_predictions
        if p.get("scenario") and p["scenario"] != "real-time"
        or p.get("composite_risk_score", 0) >= 70
    }

    logger.info(f"🛡️ Protected {len(protected_locations)} locations from overwrite: {list(protected_locations.keys())}")

    # --- Step 2: Collect data only for unprotected locations ---
    locations_to_collect = {
        loc: coords for loc, coords in DISTRICT_COORDS.items()
        if loc not in protected_locations
    }

    logger.info(f"📡 Collecting real-time data for {len(locations_to_collect)} locations")

    try:
        df_new = collect_all_data(locations_to_collect) if locations_to_collect else pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ collect_all_data failed: {e}")
        df_new = pd.DataFrame()

    # Add 'scenario' to distinguish real-time data
    if not df_new.empty:
        df_new["scenario"] = "real-time"

    # --- Step 3: Load simulated/high-risk data ---
    df_protected = pd.DataFrame([
        {k: v for k, v in doc.items() if k != "_id"}
        for doc in protected_locations.values()
    ])

    # --- Step 4: Combine datasets ---
    if df_protected.empty:
        df = df_new
    elif df_new.empty:
        df = df_protected
    else:
        df = pd.concat([df_new, df_protected], ignore_index=True)

    if df.empty:
        logger.warning("No data to score")
        raise HTTPException(status_code=404, detail="No weather data to score")

    # --- Step 5: Run risk scoring only on new data (protected already scored) ---
    # But re-score real-time data
    df_new_risk = generate_risk_scores(model, df_new) if not df_new.empty else pd.DataFrame()
    df_final = df_new_risk.copy()

    # Append protected (already scored) data
    if not df_protected.empty:
        df_final = pd.concat([df_final, df_protected], ignore_index=True)

    # Ensure household_resources is dict
    df_final['household_resources'] = df_final['household_resources'].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    predictions = _df_to_json_records(df_final)

    # --- Step 6: Save predictions to MongoDB ---
    if predictions:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from pymongo import UpdateOne
        ops = [
            UpdateOne(
                {"location": p["location"], "timestamp": p.get("timestamp") or now},
                {"$set": p},
                upsert=True
            )
            for p in predictions
        ]
        await db["predictions"].bulk_write(ops)
        logger.info(f"✅ Saved {len(predictions)} predictions to MongoDB")
    else:
        logger.warning("No predictions to save")
        return {"message": "No predictions to save", "predictions": [], "alerts_generated": 0}

    # --- Step 7: Enrich with historical data (optional) ---
    try:
        hist_summary = await db["historical_summary"].find({}, {"_id": 0}).to_list(100000)
        if hist_summary:
            loc_to_summary = {
                h["location"].lower(): {k: v for k, v in h.items() if k != "location"}
                for h in hist_summary
            }
            for p in predictions:
                loc = str(p.get("location", "")).lower()
                if loc in loc_to_summary:
                    p["historical_profile"] = loc_to_summary[loc]
    except Exception as e:
        logger.warning(f"Failed to enrich with historical profile: {e}")

    # --- Step 8: Generate alerts if requested ---
    alerts_generated = []
    if generate_alerts:
        try:
            alerts_generated = await generate_alerts_from_db(db)
        except Exception as e:
            logger.warning(f"Failed to generate alerts: {e}")

    logger.info(f"📊 Scored {len(predictions)} records. Alerts generated: {len(alerts_generated)}")
    return {
        "message": f"{len(predictions)} records scored.",
        "predictions": predictions,
        "alerts_generated": len(alerts_generated),
    }

# --- Alerts ---
@app.post("/alerts", summary="Create a new alert")
async def create_alert(alert: AlertModel):
    try:
        db = get_db(app)
        doc = alert.dict()
        result = await db["alerts"].insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        logger.info(f"Created alert for {alert.location}")
        return {"message": "Alert created", "alert": doc}
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/alerts/history", summary="Get alert history")
async def get_alerts(
    location: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    db = get_db(app)
    query = {}
    if location:
        query["location"] = {"$regex": f"^{location}$", "$options": "i"}
    if level:
        query["risk_level"] = level.upper()

    cursor = db["alerts"].find(query).sort("timestamp", -1).limit(limit)
    alerts = await cursor.to_list(length=limit)
    alerts = [_strip_mongo_ids([a])[0] for a in alerts]
    return {"count": len(alerts), "alerts": alerts}

# --- Historical Data ---
@app.get("/api/historical", response_model=List[dict])
async def get_historical_data():
    db = get_db(app)
    docs = await db["historical_events"].find().to_list(length=100000)
    if not docs:
        try:
            df = pd.read_excel(HISTORICAL_XLSX)
            df.columns = df.columns.str.strip().str.lower()
            df.rename(columns={"location": "location"}, inplace=True)
            records = df.to_dict(orient="records")
            if records:
                await db["historical_events"].insert_many(records)
                summary = df.groupby("location")["severity"].value_counts().unstack(fill_value=0).reset_index().to_dict(orient="records")
                await db["historical_summary"].delete_many({})
                await db["historical_summary"].insert_many(summary)
            return records
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise HTTPException(status_code=500, detail="Historical data not available")
    return [_strip_mongo_ids([d])[0] for d in docs]

@app.get("/api/locations")
async def get_all_locations():
    db = get_db(app)
    docs = await db["historical_events"].distinct("location")
    if not docs:
        try:
            df = pd.read_excel(HISTORICAL_XLSX)
            locations = df["location"].dropna().unique().tolist()
            return locations
        except:
            return list(DISTRICT_COORDS.keys())
    return [loc for loc in docs if loc]

@app.get("/api/risk/{location}", description="Assess historical risk profile for a location")
async def assess_risk_by_location(location: str):
    db = get_db(app)
    docs = await db["historical_events"].find({
        "location": {"$regex": f"^{location}$", "$options": "i"}
    }).to_list(length=10000)

    if not docs:
        try:
            df = pd.read_excel(HISTORICAL_XLSX)
            df.columns = df.columns.str.strip().str.lower()  # Normalize
            required_cols = {'location', 'total_deaths'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

            # Handle missing values
            df['total_deaths'] = pd.to_numeric(df['total_deaths'], errors='coerce').fillna(0)
            df['location'] = df['location'].astype(str).str.strip()

            filtered = df[df['location'].str.contains(location, case=False, na=False)]
            if filtered.empty:
                raise HTTPException(status_code=404, detail="No historical data for this location")

            severity_count = filtered["severity"].value_counts().to_dict() if "severity" in filtered.columns else {"Unknown": len(filtered)}
            total = int(filtered['total_deaths'].sum())

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise HTTPException(status_code=500, detail="Data load failed")
    else:
        df = pd.DataFrame(_strip_mongo_ids(docs))
        severity_count = df["severity"].value_counts().to_dict() if "severity" in df.columns else {"Unknown": len(df)}
        total = len(df)

    return {
        "location": location,
        "total_events": total,
        "risk_profile": severity_count
    }
# --- Resource Calculator ---
class ResourceRequest(BaseModel):
    place_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    household_size: int = 4

@app.post("/resources", summary="Calculate household resource needs")
async def calculate_resources(request: ResourceRequest):
    db = get_db(app)
    location = request.place_name
    lat = request.lat
    lon = request.lon
    household_size = request.household_size

    if household_size < 1:
        raise HTTPException(status_code=400, detail="Household size must be >= 1")

    # Resolve location to coordinates
    if location and not (lat and lon):
        try:
            loop = asyncio.get_event_loop()
            geocode_result = await loop.run_in_executor(executor, _geocode, location)
            if geocode_result:
                lat, lon = geocode_result.latitude, geocode_result.longitude
            else:
                raise HTTPException(status_code=404, detail="Location not found")
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            raise HTTPException(status_code=400, detail="Geocoding failed")

    if not (lat and lon):
        raise HTTPException(status_code=400, detail="Valid coordinates required")

    # Find latest prediction
    query = {
        "lat": {"$gte": lat - 0.1, "$lte": lat + 0.1},
        "lon": {"$gte": lon - 0.1, "$lte": lon + 0.1}
    }
    prediction_doc = await db["predictions"].find(query).sort("timestamp", -1).limit(1).to_list(1)
    
    if not prediction_doc:
        # Fallback: collect fresh data
        try:
            df = collect_all_data({location or "custom": (lat, lon)})
            if df.empty:
                resources = calculate_household_resources("Low", household_size)
                return {
                    "location": location or "Custom",
                    "risk_category": "Low",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "household_size": household_size,
                    "resources": resources,
                    "message": "No data; assuming Low risk"
                }
            df = generate_risk_scores(model, df)
            prediction = df.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Fresh collect failed: {e}")
            raise HTTPException(status_code=502, detail="Data collection failed")
    else:
        prediction = _strip_mongo_ids(prediction_doc)[0]

    risk_category = prediction.get("risk_category", "Low")
    try:
        resources = calculate_household_resources(risk_category, household_size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response = {
        "location": prediction.get("location", location or "Custom"),
        "risk_category": risk_category,
        "timestamp": prediction.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "household_size": household_size,
        "resources": resources,
        **{k: v for k, v in prediction.items() if k in ['anomaly_score', 'precip_mm', 'wind_kph', 'wave_height', 'model_risk_score']}
    }
    return response

# --- Generate Alerts ---
@app.post("/alerts/generate")
async def trigger_alert_generation(limit: int = Query(500)):
    db = get_db(app)
    try:
        alerts = await generate_alerts_from_db(db, limit=limit)
        return {"generated": len(alerts), "alerts": [_strip_mongo_ids([a])[0] for a in alerts]}
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate alerts")


# Disaster scenario templates
SCENARIOS = {
    "flood": {
        "precip_mm": 60.0,
        "temp_c": 18.0,
        "humidity": 92.0,
        "wind_kph": 30.0,
        "pressure_mb": 1000.0,
        "cloud": 95.0,
        "wave_height": 0.5,
        "description": "Severe inland flooding due to prolonged rainfall"
    },
    "storm": {
        "precip_mm": 40.0,
        "temp_c": 22.0,
        "humidity": 85.0,
        "wind_kph": 80.0,
        "pressure_mb": 995.0,
        "cloud": 100.0,
        "wave_height": 1.2,
        "description": "Intense thunderstorm with strong winds"
    },
    "coastal_cyclone": {
        "precip_mm": 50.0,
        "temp_c": 25.0,
        "humidity": 90.0,
        "wind_kph": 110.0,
        "pressure_mb": 980.0,
        "cloud": 100.0,
        "wave_height": 4.5,
        "description": "Coastal cyclone with storm surge"
    },
    "flash_flood": {
        "precip_mm": 80.0,
        "temp_c": 20.0,
        "humidity": 95.0,
        "wind_kph": 40.0,
        "pressure_mb": 990.0,
        "cloud": 98.0,
        "wave_height": 0.3,
        "description": "Extreme flash flooding in urban/rural areas"
    }
}

@app.post("/simulate", summary="Simulate a disaster scenario for testing")
async def simulate_disaster(request: SimulateRequest):
    db = get_db(app)
    location = request.location.strip()
    scenario = request.scenario
    household_size = request.household_size

    # Get coordinates: from input or fallback to known districts
    lat, lon = None, None
    if request.lat and request.lon:
        lat, lon = float(request.lat), float(request.lon)
    else:
        # Try to match to known district
        matched = None
        for k, (la, lo) in DISTRICT_COORDS.items():
            if location.lower() in k.lower():
                matched = (la, lo)
                break
        if matched:
            lat, lon = matched
        else:
            # Try geocoding
            try:
                loop = asyncio.get_event_loop()
                geo = await loop.run_in_executor(executor, _geocode, location)
                if geo:
                    lat, lon = geo.latitude, geo.longitude
                else:
                    raise HTTPException(status_code=404, detail="Location not found and no coordinates provided")
            except Exception as e:
                logger.warning(f"Geocoding failed: {e}")
                raise HTTPException(status_code=404, detail="Could not resolve location")

    # Build simulated weather
    template = SCENARIOS[scenario]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    simulated_features = {
        "location": location,
        "lat": lat,
        "lon": lon,
        "temp_c": template["temp_c"],
        "humidity": template["humidity"],
        "wind_kph": template["wind_kph"],
        "pressure_mb": template["pressure_mb"],
        "precip_mm": template["precip_mm"],
        "cloud": template["cloud"],
        "wave_height": template["wave_height"],
        "timestamp": now,
        "is_severe": 1,
        "anomaly_score": 95.0,
        "model_risk_score": 88.0,
        "composite_risk_score": 92.0,
        "risk_category": "High",
        "scenario": scenario,
        "description": template["description"]
    }

    # Add household resources
    resources = calculate_household_resources("High", household_size=household_size)
    simulated_features["household_resources"] = resources

    # Save to weather_data and predictions
    from pymongo import UpdateOne
    ops = [
        UpdateOne(
            {"location": location, "scenario": scenario, "timestamp": now},
            {"$set": simulated_features},
            upsert=True
        )
    ]
    await db["simulated_events"].bulk_write(ops)
    await db["weather_data"].bulk_write(ops)
    await db["predictions"].bulk_write(ops)

    logger.info(f"🔥 Simulated {scenario} in {location} (lat={lat}, lon={lon})")

    return {
        "message": f"✅ Simulated {scenario.upper()} in {location}",
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "scenario": scenario,
        "risk_score": 92.0,
        "composite_risk_score": 92.0,
        "household_resources": resources,
        "description": template["description"],
        "timestamp": now
    }