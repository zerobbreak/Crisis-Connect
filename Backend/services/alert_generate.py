from typing import Any, List, Tuple
import os
from datetime import datetime
import pandas as pd

try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # Fallback if google-genai is not installed
from motor.motor_asyncio import AsyncIOMotorDatabase

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

def generate_alerts(data: pd.DataFrame) -> List[dict[str, Any]]:
    alerts: List[dict[str, Any]] = []
    translation_cache: dict[str, Tuple[str, str]] = {}

    if data.empty:
        return alerts

    for _, row in data.iterrows():
        risk_score = float(row.get("composite_risk_score", 0))  # Use composite_risk_score
        risk_category = str(row.get("risk_category", "Low")).lower()  # Use risk_category
        location = str(row.get("location", "Unknown"))
        wave_height = float(row.get("wave_height", 0) or 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        household_resources = row.get("household_resources", {}) or {}

        # Extract household resources with defaults
        food_packs = household_resources.get("food_packs", 0)
        water_gallons = household_resources.get("water_gallons", 0)
        shelter_needed = household_resources.get("shelter_needed", False)
        boats_needed = household_resources.get("boats_needed", 0.0)

        # Map risk_category to alert level
        if risk_category == "high":
            level = "high"
        elif risk_category == "medium":  # Handle both 'Medium' and 'Moderate'
            level = "moderate"
        else:
            continue  # Skip low-risk alerts

        # Format the alert message
        message_en = TEMPLATES[level].format(
            location=location,
            risk_score=int(risk_score),
            wave_height=round(wave_height, 1),
            food_packs=food_packs,
            water_gallons=water_gallons,
            shelter_needed=shelter_needed,
            boats_needed=boats_needed,
        )

        # Cache and translate the message
        if message_en not in translation_cache:
            message_zu = translate_with_gemini(message_en, "isiZulu")
            message_xh = translate_with_gemini(message_en, "isiXhosa")
            translation_cache[message_en] = (message_zu, message_xh)
        else:
            message_zu, message_xh = translation_cache[message_en]

        alerts.append(
            {
                "timestamp": timestamp,
                "location": location,
                "risk_score": risk_score,
                "risk_level": level.upper(),
                "message_en": message_en,
                "message_zu": message_zu,
                "message_xh": message_xh,
                "household_resources": household_resources,  # Include raw resource data
            }
        )

    return alerts

async def generate_alerts_from_db(db: AsyncIOMotorDatabase, limit: int = 1000) -> List[dict[str, Any]]:
    """Fetch recent predictions from MongoDB, generate alerts, and store them.

    Returns the list of generated alert documents.
    """
    # Pull latest predictions, including household_resources
    cursor = (
        db["predictions"]
        .find(
            {},
            {
                "location": 1,
                "composite_risk_score": 1,
                "risk_category": 1,
                "wave_height": 1,
                "timestamp": 1,
                "household_resources": 1,
            }
        )
        .sort("timestamp", -1)
        .limit(limit)
    )
    predictions = await cursor.to_list(length=limit)

    if not predictions:
        return []

    df = pd.DataFrame(predictions)
    alerts = generate_alerts(df)

    if not alerts:
        return []

    await db["alerts"].insert_many(alerts)
    return alerts