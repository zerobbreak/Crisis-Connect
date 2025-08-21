from typing import Any, List, Tuple
import os
from datetime import datetime
import pandas as pd
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # Fallback if google-genai is not installed

logger = logging.getLogger("crisisconnect.alert_generation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Thresholds for generating alerts from predictions
HIGH_RISK_THRESHOLD = 70
MODERATE_RISK_THRESHOLD = 40

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TEMPLATES = {
    "high": (
        "⚠️ [HIGH RISK] Severe weather in {location}.\n"
        "Risk Score: {risk_score}%.\n"
        "Wave Height: {wave_height}m.\n"
        "For a family of 4: {food_packs} food packs, {water_gallons} gallons of water needed.\n"
        "Shelter: {'Required' if shelter_needed else 'Not required'}.\n"
        "Boats: {boats_needed} (shared).\n"
        "Move to higher ground and follow local updates."
    ),
    "moderate": (
        "⚠️ [MODERATE RISK] Unstable weather in {location}.\n"
        "Risk Score: {risk_score}%.\n"
        "Wave Height: {wave_height}m.\n"
        "For a family of 4: {food_packs} food packs, {water_gallons} gallons of water needed.\n"
        "Shelter: {'Required' if shelter_needed else 'Not required'}.\n"
        "Boats: {boats_needed} (shared).\n"
        "Monitor updates and stay cautious."
    ),
}

def translate_with_gemini(text: str, target_language: str) -> str:
    if not GEMINI_API_KEY or genai is None:
        # Fallback: return original text when translator is unavailable
        return text

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"Translate the following English text to {target_language}:\n{text}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    content_obj = response.candidates[0].content
    if hasattr(content_obj, "parts") and len(content_obj.parts) > 0:
        return content_obj.parts[0].text.strip()
    if hasattr(content_obj, "text"):
        return content_obj.text.strip()
    return str(content_obj)

# services/alert_generate.py
def generate_alerts(data: pd.DataFrame) -> List[dict]:
    alerts = []
    translation_cache = {}

    if data.empty:
        logger.warning("No data to generate alerts from")
        return alerts

    for _, row in data.iterrows():
        # ✅ Safely extract composite_risk_score
        try:
            risk_score = float(row.get("composite_risk_score", 0))
        except (TypeError, ValueError):
            risk_score = 0.0

        # ✅ Only alert if above threshold
        if risk_score < 70.0:
            logger.debug(f"Skipped {row.get('location')}: risk_score={risk_score}")
            continue

        location = str(row.get("location", "Unknown"))
        wave_height = float(row.get("wave_height", 0) or 0)

        # ✅ Safely get household_resources
        hr = row.get("household_resources")
        if not isinstance(hr, dict):
            hr = {}
        food_packs = hr.get("food_packs", 0)
        water_gallons = hr.get("water_gallons", 0)
        shelter_needed = hr.get("shelter_needed", False)
        boats_needed = hr.get("boats_needed", 0.0)

        level = "SEVERE" if risk_score >= 85 else "HIGH"

        message_en = TEMPLATES["high"].format(
            location=location,
            risk_score=int(risk_score),
            wave_height=round(wave_height, 1),
            food_packs=food_packs,
            water_gallons=water_gallons,
            shelter_needed=shelter_needed,
            boats_needed=round(boats_needed, 2),
        )

        if message_en not in translation_cache:
            message_zu = translate_with_gemini(message_en, "isiZulu")
            message_xh = translate_with_gemini(message_en, "isiXhosa")
            translation_cache[message_en] = (message_zu, message_xh)
        else:
            message_zu, message_xh = translation_cache[message_en]

        alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": location,
            "risk_score": risk_score,
            "risk_level": level,
            "message_en": message_en,
            "message_zu": message_zu,
            "message_xh": message_xh,
            "household_resources": hr,
            "scenario": row.get("scenario", "real-time")
        }
        alerts.append(alert)
        logger.info(f"✅ Alert generated for {location} (score: {risk_score:.1f})")

    return alerts

async def generate_alerts_from_db(db: AsyncIOMotorDatabase, limit: int = 1000):
    cursor = (
        db["predictions"]
        .find(
            {"composite_risk_score": {"$exists": True, "$ne": None}},  # ← Only fetch with score
            {
                "location": 1,
                "composite_risk_score": 1,
                "risk_category": 1,
                "wave_height": 1,
                "household_resources": 1,
                "scenario": 1
            }
        )
        .sort("timestamp", -1)
        .limit(limit)
    )
    predictions = await cursor.to_list(length=limit)

    logger.info(f"🔍 Found {len(predictions)} predictions in DB")
    for p in predictions:
        logger.info(f"📊 DB Prediction: {p['location']} → score={p.get('composite_risk_score')}")

    if not predictions:
        return []

    df = pd.DataFrame(predictions)
    alerts = generate_alerts(df)

    if alerts:
        await db["alerts"].insert_many(alerts)
        logger.info(f"✅ Generated and saved {len(alerts)} alerts")

    return alerts