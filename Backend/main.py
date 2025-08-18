from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional, Any, Dict, Tuple
from pydantic import BaseModel
from services.predict import collect_all_data, generate_risk_scores, fetch_weather_and_marine_data, extract_features
from services.alert_generate import generate_alerts_from_db
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import logging
from models.model import AlertModel, LocationRequest
from fuzzywuzzy import process
from pymongo.collection import Collection
from collections import defaultdict
from datetime import datetime
from utils.db import init_mongo, close_mongo, get_db, ensure_indexes
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-deployed-app-url"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORICAL_XLSX = "data_disaster.xlsx"
WEATHER_CSV = "latest_data.csv"  # Assumed CSV path for weather data
ALERTS_CSV = "alerts.csv"  # If alerts history is stored in CSV

# --- Helper functions to load data from files ---
from bson import ObjectId

def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc
def load_historical_data() -> pd.DataFrame:
    if not os.path.exists(HISTORICAL_XLSX):
        logger.error(f"Historical data file {HISTORICAL_XLSX} not found")
        raise HTTPException(status_code=500, detail="Historical data file not found")
    try:
        df = pd.read_excel(HISTORICAL_XLSX)
        df.columns = df.columns.str.strip().str.lower()
        # Standardize columns if needed
        df.rename(columns={
            'location': 'location',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'total deaths': 'total_deaths'
        }, inplace=True)

        def classify_severity(row):
            deaths = row.get('total_deaths') or 0
            try:
                deaths = int(deaths)
            except:
                deaths = 0
            if deaths > 100:
                return 'High'
            elif deaths > 10:
                return 'Medium'
            elif deaths > 0:
                return 'Low'
            return 'Unknown'

        df['severity'] = df.apply(classify_severity, axis=1)
        df.fillna({'location': 'Unknown', 'severity': 'Unknown', 'latitude': np.nan, 'longitude': np.nan}, inplace=True)
        logger.info(f"Loaded {len(df)} historical records from {HISTORICAL_XLSX}")
        return df
    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load historical data: {e}")

def load_weather_data() -> pd.DataFrame:
    if not os.path.exists(WEATHER_CSV):
        logger.error(f"Weather data file {WEATHER_CSV} not found")
        raise HTTPException(status_code=500, detail="Weather data file not found")
    try:
        df = pd.read_csv(WEATHER_CSV)
        logger.info(f"Loaded {len(df)} weather records from {WEATHER_CSV}")
        return df
    except Exception as e:
        logger.error(f"Failed to load weather data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load weather data: {e}")

def load_alerts_data() -> pd.DataFrame:
    if not os.path.exists(ALERTS_CSV):
        logger.warning(f"Alerts data file {ALERTS_CSV} not found, returning empty list")
        return pd.DataFrame()  # Return empty if no alerts file
    try:
        df = pd.read_csv(ALERTS_CSV)
        logger.info(f"Loaded {len(df)} alert records from {ALERTS_CSV}")
        return df
    except Exception as e:
        logger.error(f"Failed to load alerts data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load alerts data: {e}")

# --- API Endpoints ---

def _json_safe(value):
    # Normalize numpy scalars and guard against NaN/Inf
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
    sanitized: List[dict] = []
    for r in records:
        sanitized.append({k: _json_safe(v) for k, v in r.items()})
    return sanitized

def _strip_mongo_ids(records: List[dict]) -> List[dict]:
    for r in records:
        r.pop("_id", None)
    return records


@app.on_event("startup")
async def startup_event():
    await init_mongo(app)
    # Ensure indexes for efficient queries and deduplication
    try:
        db = get_db(app)
        await ensure_indexes(db)
        logger.info("MongoDB indexes ensured")
    except Exception as e:
        logger.warning(f"Failed to ensure MongoDB indexes: {e}")


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", summary="Service health check")
async def health():
    try:
        db = get_db(app)
        await db.command("ping")
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo(app)


@app.get("/weather", summary="Retrieve all weather data")
async def get_weather():
    # Prefer DB, fallback to CSV
    db = get_db(app)
    records = await db["weather_data"].find().sort("timestamp", -1).to_list(length=10000)
    if not records:
        # Collect fresh data instead of reading CSV
        try:
            df = collect_all_data()
        except Exception as e:
            logger.error(f"Collect failed: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to collect live weather data: {e}")
        if not df.empty:
            from pymongo import UpdateOne
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ops = []
            for r in df.to_dict(orient="records"):
                key = {"location": r.get("location"), "timestamp": r.get("timestamp") or now}
                r["timestamp"] = key["timestamp"]
                ops.append(UpdateOne(key, {"$set": r}, upsert=True))
            if ops:
                await db["weather_data"].bulk_write(ops)
            records = _df_to_json_records(df)
        else:
            records = []
    else:
        records = _strip_mongo_ids(records)
        records = _sanitize_records(records)
    return {"count": len(records), "records": records}

@app.post("/predict", summary="Score weather data using ML models")
async def predict_risk(generate_alerts: bool = False):
    db = get_db(app)
    records = await db["weather_data"].find().to_list(length=10000)
    if not records:
        # Collect fresh data from services instead of CSV
        try:
            df = collect_all_data()
        except Exception as e:
            logger.error(f"Collect failed: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to collect live weather data: {e}")
        if not df.empty:
            from pymongo import UpdateOne
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ops = []
            for r in df.to_dict(orient="records"):
                key = {"location": r.get("location"), "timestamp": r.get("timestamp") or now}
                r["timestamp"] = key["timestamp"]
                ops.append(UpdateOne(key, {"$set": r}, upsert=True))
            if ops:
                await db["weather_data"].bulk_write(ops)
    else:
        df = pd.DataFrame(_strip_mongo_ids(records))
    if df.empty:
        logger.warning("No weather data to score")
        raise HTTPException(status_code=404, detail="No weather data to score")
    df = generate_risk_scores(model, df)
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.where(pd.notnull(df), None)
    predictions = df.to_dict(orient="records")
    # Enrich with historical summary for quick reference
    try:
        hist_summary = await db["historical_summary"].find({}, {"_id": 0}).to_list(length=100000)
        if hist_summary:
            loc_to_summary = {h.get("location", "").lower(): {k: v for k, v in h.items() if k != "location"} for h in hist_summary}
            for p in predictions:
                loc = str(p.get("location", "")).lower()
                if loc in loc_to_summary:
                    p["historical_profile"] = loc_to_summary[loc]
    except Exception as e:
        logger.warning(f"Failed to enrich predictions with historical summary: {e}")
    # Save predictions with upsert-like behavior using bulk_write
    if predictions:
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from pymongo import UpdateOne
        ops = []
        for p in predictions:
            key = {
                "location": p.get("location"),
                "timestamp": p.get("timestamp") or now_ts,
            }
            p["timestamp"] = key["timestamp"]
            ops.append(UpdateOne(key, {"$set": p}, upsert=True))
        if ops:
            await db["predictions"].bulk_write(ops)
    # Optionally generate alerts right after scoring
    alerts_generated = []
    if generate_alerts:
        try:
            alerts_generated = await generate_alerts_from_db(db)
        except Exception as e:
            logger.warning(f"Failed to generate alerts: {e}")

    logger.info(f"Scored {len(predictions)} weather records")
    return {
        "message": f"{len(predictions)} records scored.",
        "predictions": predictions,
        "alerts_generated": len(alerts_generated),
    }

from fastapi import Request

@app.post("/alerts", summary="Create a new flood alert")
async def create_alert(request: Request):
    try:
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body}")
        alert_data = await request.json()
        logger.info(f"Parsed JSON: {alert_data}")
        alert = AlertModel(**alert_data)
        logger.info(f"Validated alert: {alert}")
        db = get_db(app)
        doc = alert.dict()
        await db["alerts"].insert_one(doc)
        doc.pop("_id", None)
        return {"message": "Alert stored", "alert": doc}
    except ValueError as e:
        logger.error(f"Invalid JSON or missing fields: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid or missing request body: {e}")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.get("/alerts/history", summary="Retrieve alert history with optional filters")
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
    if not alerts:
        # fallback to CSV
        df = load_alerts_data()
        if df.empty:
            return {"count": 0, "alerts": []}
        if location:
            df = df[df["location"].str.lower() == location.lower()]
        if level:
            df = df[df["risk_level"].str.upper() == level.upper()]
        df = df.sort_values(by="timestamp", ascending=False).head(limit)
        alerts = df.to_dict(orient="records")
    else:
        alerts = _strip_mongo_ids(alerts)
    logger.info(f"Loaded {len(alerts)} alerts with filters")
    return {"count": len(alerts), "alerts": alerts}

@app.get("/api/historical", response_model=List[dict], description="Retrieve historical disaster data")
async def get_historical_data():
    db = get_db(app)
    docs = await db["historical_events"].find().to_list(length=100000)
    if not docs:
        df = load_historical_data()
        records = df.to_dict(orient="records")
        if records:
            await db["historical_events"].insert_many(records)
            # Also compute and store a summary per location for quick joins
            summary = (
                df.groupby("location")["severity"].value_counts().unstack(fill_value=0).reset_index()
            )
            summary_records = summary.to_dict(orient="records")
            await db["historical_summary"].delete_many({})
            if summary_records:
                await db["historical_summary"].insert_many(summary_records)
        return records
    return _strip_mongo_ids(docs)

@app.get("/api/locations", response_model=List[str], description="Get all unique locations from historical data")
async def get_all_locations():
    db = get_db(app)
    docs = await db["historical_events"].distinct("location")
    if not docs:
        df = load_historical_data()
        locations = df["location"].dropna().unique().tolist()
        logger.info(f"Loaded {len(locations)} unique locations (from CSV)")
        return locations
    logger.info(f"Loaded {len(docs)} unique locations (from DB)")
    return [loc for loc in docs if loc]

@app.get("/api/risk/{location}", description="Assess historical risk profile for a location")
async def assess_risk_by_location(location: str):
    db = get_db(app)
    docs = await db["historical_events"].find({"location": {"$regex": f"^{location}$", "$options": "i"}}).to_list(length=10000)
    if not docs:
        df = load_historical_data()
        filtered = df[df["location"].str.lower() == location.lower()]
        if filtered.empty:
            logger.warning(f"No historical data for location: {location}")
            raise HTTPException(status_code=404, detail="No historical data for this location")
        severity_count = filtered["severity"].value_counts().to_dict()
        total = sum(severity_count.values())
    else:
        df = pd.DataFrame(_strip_mongo_ids(docs))
        severity_count = df["severity"].value_counts().to_dict()
        total = int(sum(severity_count.values()))
    logger.info(f"Assessed risk for {location}: {total} events")
    return {"location": location, "total_events": total, "risk_profile": severity_count}

@app.get("/", description="API root endpoint")
def root():
    return {"message": "Welcome to the Crisis Connect API"}

@app.get("/collect", description="Collect weather and marine data for predefined districts")
async def collect_data():
    try:
        df = collect_all_data()
        count = len(df)
        # Save to DB
        if count > 0:
            db = get_db(app)
            records = df.to_dict(orient="records")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            from pymongo import UpdateOne
            ops = []
            for r in records:
                key = {
                    "location": r.get("location"),
                    "timestamp": r.get("timestamp") or now,
                }
                r["timestamp"] = key["timestamp"]
                ops.append(UpdateOne(key, {"$set": r}, upsert=True))
            if ops:
                await db["weather_data"].bulk_write(ops)
        logger.info(f"Collected {count} weather records")
        return {"message": "Data collected", "count": count}
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error collecting data: {e}")


class CollectRequest(BaseModel):
    locations: Optional[Any] = None


@app.post("/collect", description="Collect weather/marine data for provided locations (names, coords, or mapping)")
async def collect_data_custom(payload: CollectRequest):
    try:
        records = []
        df = collect_all_data(payload.locations)
        count = len(df)
        if count > 0:
            db = get_db(app)
            records = _df_to_json_records(df)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            from pymongo import UpdateOne
            ops = []
            for r in records:
                key = {"location": r.get("location"), "timestamp": r.get("timestamp") or now}
                r["timestamp"] = key["timestamp"]
                ops.append(UpdateOne(key, {"$set": r}, upsert=True))
            if ops:
                await db["weather_data"].bulk_write(ops)
        logger.info(f"Custom-collected {count} weather records")
        return {"message": "Data collected", "count": count, "records": records if count <= 50 else []}
    except Exception as e:
        logger.error(f"Error in custom collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error in custom collection: {e}")

@app.get("/risk-assessment", description="Generate risk scores for collected data")
async def assess_risk():
    db = get_db(app)
    records = await db["weather_data"].find().to_list(length=10000)
    if not records:
        # Collect fresh data from services instead of CSV
        try:
            df = collect_all_data()
        except Exception as e:
            logger.error(f"Collect failed: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to collect live weather data: {e}")
        if not df.empty:
            from pymongo import UpdateOne
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ops = []
            for r in df.to_dict(orient="records"):
                key = {"location": r.get("location"), "timestamp": r.get("timestamp") or now}
                r["timestamp"] = key["timestamp"]
                ops.append(UpdateOne(key, {"$set": r}, upsert=True))
            if ops:
                await db["weather_data"].bulk_write(ops)
    else:
        df = pd.DataFrame(_strip_mongo_ids(records))
    if df.empty:
        logger.warning("No weather data available")
        raise HTTPException(status_code=404, detail="No weather data available")
    df = generate_risk_scores(model, df)
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.where(pd.notnull(df), None)
    predictions = df.to_dict(orient="records")
    # Enrich with historical summary
    try:
        hist_summary = await db["historical_summary"].find({}, {"_id": 0}).to_list(length=100000)
        if hist_summary:
            loc_to_summary = {h.get("location", "").lower(): {k: v for k, v in h.items() if k != "location"} for h in hist_summary}
            for p in predictions:
                loc = str(p.get("location", "")).lower()
                if loc in loc_to_summary:
                    p["historical_profile"] = loc_to_summary[loc]
    except Exception as e:
        logger.warning(f"Failed to enrich predictions with historical summary: {e}")
    # Save with upsert to avoid duplicates
    if predictions:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from pymongo import UpdateOne
        ops = []
        for p in predictions:
            key = {
                "location": p.get("location"),
                "timestamp": p.get("timestamp") or now,
            }
            p["timestamp"] = key["timestamp"]
            ops.append(UpdateOne(key, {"$set": p}, upsert=True))
        if ops:
            await db["predictions"].bulk_write(ops)
    logger.info(f"Generated risk scores for {len(predictions)} records")
    return predictions


@app.post("/alerts/generate", summary="Generate alerts from recent predictions and store them")
async def trigger_alert_generation(limit: int = Query(500, ge=1, le=5000)):
    db = get_db(app)
    try:
        alerts = await generate_alerts_from_db(db, limit=limit)
        return {"generated": len(alerts), "alerts": _strip_mongo_ids(alerts)}
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert generation failed: {e}")

@app.post("/risk/location", description="Assess risk for a specific location")
async def location_risk(request: Request):
    try:
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body}")
        data = await request.json()
        logger.info(f"Parsed JSON: {data}")
        location_data = LocationRequest(**data)
        logger.info(f"Validated location request: {location_data}")

        db = get_db(app)
        query = {}
        search_term = location_data.place_name or location_data.district
        if not search_term:
            if location_data.lat and location_data.lon:
                # Optional: Add geocoding reverse lookup if lat/lon provided
                raise HTTPException(status_code=501, detail="Lat/lon lookup not implemented")
            raise HTTPException(status_code=400, detail="Either place_name or district must be provided")

        # Query MongoDB with case-insensitive regex
        docs = await db["historical_events"].find({
            "location": {"$regex": f"^{search_term}$", "$options": "i"}
        }).to_list(length=10000)

        if not docs:
            # Fallback to CSV and use fuzzy matching
            df = load_historical_data()
            all_locations = df["location"].dropna().unique()
            best_match, score = process.extractOne(search_term.lower(), [loc.lower() for loc in all_locations])
            if score < 80:  # Adjust threshold as needed
                logger.warning(f"No historical data for location: {search_term}")
                raise HTTPException(status_code=404, detail=f"No historical data for location: {search_term}")
            filtered = df[df["location"].str.lower() == best_match]
            if filtered.empty:
                logger.warning(f"No historical data for matched location: {best_match}")
                raise HTTPException(status_code=404, detail=f"No historical data for matched location: {best_match}")
            severity_count = filtered["severity"].value_counts().to_dict()
            total = len(filtered)
        else:
            df = pd.DataFrame(_strip_mongo_ids(docs))
            severity_count = df["severity"].value_counts().to_dict()
            total = len(df)

        logger.info(f"Assessed risk for {search_term}: {total} events")
        return {
            "location": search_term,
            "total_events": total,
            "risk_profile": severity_count,
            "is_coastal": location_data.is_coastal
        }
    except ValueError as e:
        logger.error(f"Invalid JSON or missing fields: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid or missing request body: {e}")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
# Load ML model on startup
try:
    model = joblib.load("rf_model.pkl")
    logger.info("Loaded Random Forest model")
except Exception:
    logger.error("Model not found")
    raise HTTPException(status_code=500, detail="Model not found, please train the models first")
