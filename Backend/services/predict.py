from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import numpy as np
import openmeteo_requests
from pydantic import ValidationError
import requests_cache
from retry_requests import retry
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import branca.colormap as cm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging
from datetime import timedelta, datetime
from pydantic import BaseModel, model_validator, field_validator, constr, confloat,Field, ConfigDict
from typing import List, Optional, Literal

from models.model import WeatherBatch, WeatherEntry

# Configuration
logger = logging.getLogger("crisisconnect.predict")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

DISTRICT_COORDS: Dict[str, Tuple[float, float]] = {
    "eThekwini (Durban)": (-29.8587, 31.0218),
    "King Cetshwayo (Richards Bay)": (-28.7807, 32.0383),
    "Ugu (Port Shepstone)": (-30.7414, 30.4540),
    "iLembe (Ballito)": (-29.5389, 31.2140),
    "uThukela (Ladysmith)": (-28.5539, 29.7784),
    "uMgungundlovu (Pietermaritzburg)": (-29.6006, 30.3794),
    "Amajuba (Newcastle)": (-27.7577, 29.9318),
    "Uthungulu (Empangeni)": (-28.7489, 31.8933),
    "UMkhanyakude (Mtubatuba)": (-28.4176, 32.1822),
    "Zululand (Vryheid)": (-27.7695, 30.7916),
    "Cape Town": (-33.9249, 18.4241),
    "George": (-33.9630, 22.4617),
    "Mossel Bay": (-34.1830, 22.1460),
    "Hermanus": (-34.4187, 19.2345),
    "Saldanha": (-33.0117, 17.9440),
    "Knysna": (-34.0363, 23.0479),
    "Gqeberha (Port Elizabeth)": (-33.9608, 25.6022),
    "East London": (-33.0153, 27.9116),
    "Mthatha": (-31.5889, 28.7844),
    "Johannesburg": (-26.2041, 28.0473),
    "Pretoria": (-25.7479, 28.2293),
    "Bloemfontein": (-29.0852, 26.1596),
    "Polokwane": (-23.9045, 29.4689),
    "Mbombela (Nelspruit)": (-25.4658, 30.9853),
    "Rustenburg": (-25.6676, 27.2421),
    "Kimberley": (-28.7282, 24.7499),
}

FEATURE_COLS = ['lat', 'lon', 'temp_c', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud', 'wave_height']

# Updated baseline with mean and std (placeholder; replace with real historical data)
BASELINE_WEATHER = {
    district: {
        'temp_c': {'mean': 20.0, 'std': 5.0},
        'humidity': {'mean': 50.0, 'std': 10.0},
        'wind_kph': {'mean': 15.0, 'std': 10.0},
        'pressure_mb': {'mean': 1013.0, 'std': 5.0},
        'precip_mm': {'mean': 2.0, 'std': 2.0},
        'cloud': {'mean': 20.0, 'std': 10.0},
        'wave_height': {'mean': 1.0 if any(k.lower() in district.lower() for k in [
            "Durban", "Richards Bay", "Port Shepstone", "Ballito", "Cape Town",
            "George", "Mossel Bay", "Hermanus", "Saldanha", "Knysna",
            "Gqeberha", "Port Elizabeth", "East London"
        ]) else 0.0, 'std': 0.5}
    } for district in DISTRICT_COORDS.keys()
}

# Setup Open-Meteo client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Geocoder
_geolocator = Nominatim(user_agent="crisis-connect")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1)

def fetch_historical_baseline(lat: float, lon: float, days: int = 365) -> Dict[str, Dict[str, float]]:
    """
    Fetch 1-year historical data to compute realistic baselines.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        "end_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean,pressure_msl_mean,cloud_cover_mean",
        "timezone": "auto"
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        if not responses:
            raise ValueError("No response from historical API")
        daily = responses[0].Daily()
        df = pd.DataFrame({
            "temp_c": daily.Variables(0).ValuesAsNumpy(),
            "precip_mm": daily.Variables(1).ValuesAsNumpy(),
            "wind_kph": daily.Variables(2).ValuesAsNumpy() * 3.6,
            "humidity": daily.Variables(3).ValuesAsNumpy(),
            "pressure_mb": daily.Variables(4).ValuesAsNumpy(),
            "cloud": daily.Variables(5).ValuesAsNumpy(),
        })
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("No valid historical data")

        return {
            col: {"mean": df[col].mean(), "std": df[col].std()}
            for col in df.columns
        }
    except Exception as e:
        logger.warning(f"Failed to fetch historical baseline: {e}. Using fallback.")
        return {
            "temp_c": {"mean": 20.0, "std": 5.0},
            "humidity": {"mean": 50.0, "std": 10.0},
            "wind_kph": {"mean": 15.0, "std": 10.0},
            "pressure_mb": {"mean": 1013.0, "std": 5.0},
            "precip_mm": {"mean": 2.0, "std": 2.0},
            "cloud": {"mean": 20.0, "std": 10.0},
            "wave_height": {"mean": 1.0, "std": 0.5}
        }

def load_all_baselines():
    """Precompute baselines for all districts."""
    global BASELINE_WEATHER
    for district, (lat, lon) in DISTRICT_COORDS.items():
        logger.info(f"Fetching baseline for {district}")
        BASELINE_WEATHER[district] = fetch_historical_baseline(lat, lon)
    logger.info("✅ All baselines loaded")

def fetch_weather_and_marine_data(lat, lon, is_coastal=False):
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m,pressure_msl,cloud_cover",
        "timezone": "auto"
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height",
        "timezone": "auto"
    } if is_coastal else None

    try:
        weather_response = openmeteo.weather_api(WEATHER_URL, params=weather_params)[0]
        weather_hourly = weather_response.Hourly()

        marine_hourly = None
        if is_coastal:
            marine_response = openmeteo.weather_api(MARINE_URL, params=marine_params)[0]
            marine_hourly = marine_response.Hourly()

        return weather_hourly, marine_hourly
    except Exception as e:
        logger.error(f"Failed to fetch data for ({lat}, {lon}): {e}")
        return None, None

def compute_anomaly_score(features: Dict[str, float], district: str) -> float:
    baseline = BASELINE_WEATHER.get(district, BASELINE_WEATHER.get("fallback", fetch_historical_baseline(0,0)))
    z_scores = []
    for feature in ['precip_mm', 'wind_kph', 'wave_height']:
        current = features.get(feature, 0)
        mean = baseline.get(feature, {}).get('mean', 0)
        std = baseline.get(feature, {}).get('std', 1)
        z_score = abs(current - mean) / max(std, 1e-6)
        z_scores.append(z_score)
    return min(np.mean(z_scores) * 20, 100)  # Scale to 0–100

def extract_features(district: str, lat: float, lon: float, weather_hourly, marine_hourly=None):
    if not weather_hourly:
        return None

    try:
        # Last 24 hours
        temp_c = weather_hourly.Variables(0).ValuesAsNumpy()[-24:]
        precip = weather_hourly.Variables(1).ValuesAsNumpy()[-24:]
        wind = weather_hourly.Variables(2).ValuesAsNumpy()[-24:]
        humidity = weather_hourly.Variables(3).ValuesAsNumpy()[-24:]
        pressure = weather_hourly.Variables(4).ValuesAsNumpy()[-24:]
        cloud = weather_hourly.Variables(5).ValuesAsNumpy()[-24:]
        wave_height = marine_hourly.Variables(0).ValuesAsNumpy()[-24:].mean() if marine_hourly else 0.0

        # Aggregates
        features = {
            'location': district,
            'lat': lat,
            'lon': lon,
            'temp_c': temp_c.mean(),
            'humidity': humidity.mean(),
            'wind_kph': wind.max() * 3.6,
            'pressure_mb': pressure.mean(),
            'precip_mm': precip.sum(),
            'cloud': cloud.mean(),
            'wave_height': wave_height,

            # New trend features
            'precip_3h_trend': (precip[-3:].sum() - precip[-6:-3].sum()) / (precip[-6:-3].sum() + 1e-6),
            'pressure_trend': (pressure[-1] - pressure[-6]) / 6,
            'wind_gust_factor': (wind.max() / (wind.mean() + 1e-6))
        }

        # is_severe: now only for labeling, not used in final risk
        features['is_severe'] = int(
            (features['precip_mm'] > 50 and features['wind_kph'] > 60) or
            (features['wave_height'] > 3 and features['precip_mm'] > 20) or
            (features['humidity'] > 95 and features['temp_c'] < 5 and features['precip_mm'] > 15)
        )

        features['anomaly_score'] = compute_anomaly_score(features, district)
        return features

    except Exception as e:
        logger.error(f"Error extracting features for {district}: {e}")
        return None

def collect_all_data(locations: Optional[Union[Dict, List]] = None) -> pd.DataFrame:
    data = []
    coastal_keywords = ["Durban", "Richards Bay", "Cape Town", "Gqeberha", "East London", "Port Shepstone"]

    if locations is None:
        target_locations = DISTRICT_COORDS
    elif isinstance(locations, dict):
        target_locations = {str(k): (float(v[0]), float(v[1])) for k, v in locations.items() if v}
    else:
        target_locations = {}
        for idx, item in enumerate(locations):
            if isinstance(item, str):
                try:
                    geo = _geocode(item)
                    if geo:
                        target_locations[item] = (geo.latitude, geo.longitude)
                    else:
                        logger.warning(f"Geocoding failed: {item}")
                except Exception as e:
                    logger.error(f"Geocode error: {e}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                target_locations[f"custom_{idx}"] = (float(item[0]), float(item[1]))
            elif isinstance(item, dict) and 'lat' in item and 'lon' in item:
                target_locations[item.get('name', f'custom_{idx}')] = (float(item['lat']), float(item['lon']))
            else:
                logger.warning(f"Invalid location format: {item}")

    for district, (lat, lon) in target_locations.items():
        is_coastal = any(k.lower() in district.lower() for k in coastal_keywords)
        weather_hourly, marine_hourly = fetch_weather_and_marine_data(lat, lon, is_coastal)
        if weather_hourly:
            features = extract_features(district, lat, lon, weather_hourly, marine_hourly)
            if features:
                # Validate with Pydantic before adding
                try:
                    entry = WeatherEntry(
                        temperature=features['temp_c'],
                        humidity=features['humidity'],
                        rainfall=features['precip_mm'],
                        wind_speed=features['wind_kph'],
                        wave_height=features['wave_height'],
                        location=district,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        latitude=lat,
                        longitude=lon
                    )
                    data.append(features)
                    logger.info(f"✅ Validated data for {district}")
                except ValidationError as ve:
                    logger.warning(f"Validation failed for {district}: {ve}")
        else:
            logger.warning(f"❌ Failed to fetch data for {district}")

    df = pd.DataFrame(data)
    return df

def train_model(df: pd.DataFrame):
    if df.empty or 'is_severe' not in df or df['is_severe'].nunique() < 2:
        logger.warning("Not enough classes to train. Adding synthetic samples.")
        synthetic = pd.DataFrame([{col: 0.0 for col in FEATURE_COLS + ['is_severe', 'anomaly_score', 'lat', 'lon']}, 
                                  {col: 0.0 for col in FEATURE_COLS + ['is_severe', 'anomaly_score', 'lat', 'lon']}])
        synthetic['is_severe'] = [0, 0]
        df = pd.concat([df, synthetic], ignore_index=True)

    X = df[FEATURE_COLS]
    y = df['is_severe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")

    # Feature importance
    plt.figure(figsize=(8, 6))
    sns.barplot(x=model.feature_importances_, y=FEATURE_COLS)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    joblib.dump(model, 'rf_model.pkl')
    logger.info("💾 Model saved as 'rf_model.pkl'")
    return model

def calculate_household_resources(severity: str, household_size: int = 4) -> Dict[str, float]:
    if severity not in ['Low', 'Medium', 'High', 'Moderate']:
        raise ValueError("Invalid severity level")

    severity = 'Medium' if severity == 'Moderate' else severity

    days = {'Low': 3, 'Medium': 7, 'High': 14}[severity]
    water = household_size * 1.0 * days
    food = household_size * 1.0 * days
    shelter = 1 if severity in ['Medium', 'High'] else 0
    boats = 0.1 if severity == 'High' else 0.0

    return {
        'food_packs': round(food, 1),
        'water_gallons': round(water, 1),
        'shelter_needed': bool(shelter),
        'boats_needed': round(boats, 2)
    }


def generate_risk_scores(model, df: pd.DataFrame):
    X = df[FEATURE_COLS]
    df['model_risk_score'] = model.predict_proba(X)[:, 1] * 100
    df['composite_risk_score'] = 0.7 * df['model_risk_score'] + 0.3 * df['anomaly_score']
    
    # Categorize
    df['risk_category'] = pd.cut(
        df['composite_risk_score'],
        bins=[0, 40, 70, 100],
        labels=['Low', 'Medium', 'High']
    ).astype(str)

    # Ensure household_resources is safe
    df['household_resources'] = df['risk_category'].apply(
        lambda s: calculate_household_resources(s)
    ).apply(lambda x: x if isinstance(x, dict) else {})

    return df  # ← Must return updated df

def visualize_data(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_severe', data=df, palette='coolwarm')
    plt.title("Class Distribution: Severe vs Normal")
    plt.xlabel("Severe (1 = Yes, 0 = No)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(df['composite_risk_score'], bins=20, kde=True, color='skyblue')
    plt.title("Distribution of Composite Risk Scores")
    plt.xlabel("Composite Risk Score (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df[FEATURE_COLS].hist(bins=20, figsize=(14, 8), color='skyblue', edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURE_COLS + ['is_severe', 'composite_risk_score']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def visualize_map_with_scores(df):
    map = folium.Map(location=[-30.5595, 22.9375], zoom_start=5)
    colormap = cm.LinearColormap(['green', 'orange', 'red'], vmin=0, vmax=100, caption="Flood Risk (%)")

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            popup=(f"{row['location']}:\n{row['risk_category']} Risk ({row['composite_risk_score']:.1f}%)\n"
                   f"Wave Height: {row['wave_height']:.1f}m\nAnomaly Score: {row['anomaly_score']:.1f}%"),
            color=colormap(row['composite_risk_score']),
            fill=True,
            fill_color=colormap(row['composite_risk_score']),
            fill_opacity=0.7
        ).add_to(map)

    colormap.add_to(map)
    map.save("weather_risk_map.html")
    print("🗺️ Map with composite risk scores saved to 'weather_risk_map.html'")

def main():
    global model

    # Load baselines
    load_all_baselines()

    # Collect data
    df = collect_all_data()
    if df.empty:
        logger.error("No data collected")
        return

    # Merge historical disaster data (if available)
    try:
        df_hist = pd.read_excel("data_disaster.xlsx")
        # ... merge logic (as before)
    except Exception as e:
        logger.warning(f"Failed to load historical disaster data: {e}")

    # Train or load model
    try:
        model = joblib.load("rf_model.pkl")
        logger.info("✅ Loaded existing model")
    except:
        logger.info("🔁 Training new model")
        model = train_model(df)

    # Generate predictions
    df = generate_risk_scores(model, df)
    df['household_resources'] = df['risk_category'].apply(
        lambda s: calculate_household_resources(s)
    )

    # Save
    df.to_csv("weather_data_scored.csv", index=False)
    logger.info("✅ Predictions saved")

    return df

if __name__ == "__main__":
    main()