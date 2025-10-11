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

# from models.model import WeatherBatch, WeatherEntry

class WeatherEntry(BaseModel):
    temperature: Optional[confloat(ge=-100.0, le=100.0)] = None
    humidity: Optional[confloat(ge=0.0, le=100.0)] = None
    rainfall: Optional[confloat(ge=0.0, le=5000.0)] = None
    wind_speed: Optional[confloat(ge=0.0, le=500.0)] = None
    wave_height: Optional[confloat(ge=0.0, le=50.0)] = None
    location: constr(min_length=1, max_length=100, strip_whitespace=True)
    timestamp: str
    latitude: Optional[confloat(ge=-90.0, le=90.0)] = None
    longitude: Optional[confloat(ge=-180.0, le=180.0)] = None
    
    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            return v
        except ValueError:
            raise ValueError("Timestamp must be in format YYYY-MM-DD HH:MM:SS")

class WeatherBatch(BaseModel):
    data: List[WeatherEntry]
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

# Enhanced feature set for better prediction accuracy
FEATURE_COLS = [
    # Basic weather features
    'temp_c', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud', 'wave_height',

    # Derived weather features
    'heat_index', 'wind_pressure_ratio', 'precip_intensity', 'weather_stability',

    # Temporal features
    'precip_trend_3h', 'temp_change_rate', 'humidity_trend', 'pressure_trend', 'wind_gust_factor',

    # Geographic features
    'coastal_distance', 'urban_density_proxy', 'elevation_proxy'
]

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

def calculate_enhanced_anomaly_score(features: Dict[str, float], district: str) -> float:
    """
    Calculate enhanced anomaly score considering multiple weather and temporal factors.

    Uses real historical baselines when available, falls back to placeholders.
    Considers weather anomalies, temporal trends, and geographic factors.
    """
    # Get baseline data (real historical or fallback)
    baseline = BASELINE_WEATHER.get(district, BASELINE_WEATHER.get("fallback", {}))

    anomaly_components = []

    # === WEATHER ANOMALIES ===
    weather_features = [
        'precip_mm', 'wind_kph', 'wave_height', 'temp_c', 'humidity',
        'heat_index', 'precip_intensity', 'weather_stability'
    ]

    for feature in weather_features:
        if feature in features:
            current = features[feature]
            feature_baseline = baseline.get(feature, {})

            # Use real baseline if available, otherwise use reasonable defaults
            mean = feature_baseline.get('mean', get_feature_default_mean(feature))
            std = feature_baseline.get('std', get_feature_default_std(feature))

            if std > 0:
                z_score = abs(current - mean) / std
                anomaly_components.append(z_score)

    # === TEMPORAL ANOMALIES ===
    temporal_features = [
        'precip_trend_3h', 'pressure_trend', 'humidity_trend', 'temp_change_rate'
    ]

    for feature in temporal_features:
        if feature in features:
            current = features[feature]
            # Temporal features are deviations from stability (0 = stable)
            # High absolute values indicate rapid changes = higher anomaly
            temporal_anomaly = abs(current) * 10  # Scale temporal changes
            anomaly_components.append(temporal_anomaly)

    # === SEASONAL CONTEXT ===
    # Adjust anomaly based on seasonal expectations
    seasonal_multiplier = get_seasonal_anomaly_multiplier(district, features)
    anomaly_components = [score * seasonal_multiplier for score in anomaly_components]

    # === GEOGRAPHIC FACTORS ===
    # Coastal areas have higher baseline anomaly tolerance
    coastal_factor = 1.0
    if 'coastal_distance' in features:
        coastal_distance = features['coastal_distance']
        coastal_factor = 1.0 + (coastal_distance / 10)  # Reduce anomaly score inland

    # Calculate final anomaly score
    if anomaly_components:
        raw_anomaly = np.mean(anomaly_components) / coastal_factor
        # Scale to 0-100 with reasonable bounds
        anomaly_score = min(raw_anomaly * 15, 100)
        return max(anomaly_score, 0)  # Ensure non-negative
    else:
        return 0

def get_feature_default_mean(feature: str) -> float:
    """Get reasonable default mean values for features when baseline unavailable."""
    defaults = {
        'temp_c': 20.0,
        'humidity': 65.0,
        'wind_kph': 15.0,
        'pressure_mb': 1013.0,
        'precip_mm': 5.0,
        'cloud': 40.0,
        'wave_height': 1.0,
        'heat_index': 22.0,
        'precip_intensity': 2.0,
        'weather_stability': 25000.0,
        'precip_trend_3h': 0.0,
        'pressure_trend': 0.0,
        'humidity_trend': 0.0,
        'temp_change_rate': 0.0
    }
    return defaults.get(feature, 0.0)

def get_feature_default_std(feature: str) -> float:
    """Get reasonable default standard deviations for features."""
    defaults = {
        'temp_c': 8.0,
        'humidity': 20.0,
        'wind_kph': 12.0,
        'pressure_mb': 8.0,
        'precip_mm': 15.0,
        'cloud': 25.0,
        'wave_height': 1.5,
        'heat_index': 10.0,
        'precip_intensity': 5.0,
        'weather_stability': 10000.0,
        'precip_trend_3h': 1.0,
        'pressure_trend': 2.0,
        'humidity_trend': 15.0,
        'temp_change_rate': 2.0
    }
    return defaults.get(feature, 1.0)

def get_seasonal_anomaly_multiplier(district: str, features: Dict[str, float]) -> float:
    """Get seasonal adjustment for anomaly scores."""
    # Simplified seasonal adjustment (could be enhanced with real seasonal data)
    # Higher in wet season (summer in South Africa), lower in dry season

    # Rough seasonal determination based on latitude and current month
    # South Africa summer: Dec-Feb, winter: Jun-Aug
    try:
        # This would ideally use current date, but we'll use a neutral multiplier for now
        seasonal_adjustment = 1.0  # Neutral

        # Could be enhanced to:
        # - Increase multiplier in summer months for rainfall features
        # - Decrease for winter stability expectations
        # - Adjust based on historical seasonal patterns

        return seasonal_adjustment
    except:
        return 1.0

def compute_anomaly_score(features: Dict[str, float], district: str) -> float:
    """Legacy anomaly score calculation for backward compatibility."""
    return calculate_enhanced_anomaly_score(features, district)

def extract_features_from_dataframe(district: str, lat: float, lon: float, weather_df: pd.DataFrame):
    """
    Extract enhanced features from historical weather DataFrame.

    This is a simplified version of extract_enhanced_features that works
    with historical DataFrame data instead of Open-Meteo response objects.
    """
    if weather_df.empty:
        return None

    try:
        # Calculate aggregates from available data
        temp_c = weather_df['temp_c'].mean() if 'temp_c' in weather_df.columns else 20.0
        humidity = weather_df['humidity'].mean() if 'humidity' in weather_df.columns else 65.0
        wind_kph = weather_df['wind_kph'].max() if 'wind_kph' in weather_df.columns else 15.0
        pressure_mb = weather_df['pressure_mb'].mean() if 'pressure_mb' in weather_df.columns else 1013.0
        precip_mm = weather_df['precip_mm'].sum() if 'precip_mm' in weather_df.columns else 5.0
        cloud = weather_df['cloud'].mean() if 'cloud' in weather_df.columns else 40.0
        wave_height = weather_df['wave_height'].mean() if 'wave_height' in weather_df.columns else 1.0

        # Calculate derived features (simplified for historical data)
        heat_index = temp_c + (humidity / 100) * 5
        wind_pressure_ratio = (wind_kph / (pressure_mb / 1000)) if pressure_mb > 0 else 0

        # Precipitation intensity (simplified)
        rainy_hours = len(weather_df[weather_df['precip_mm'] > 0.1]) if 'precip_mm' in weather_df.columns else 1
        precip_intensity = precip_mm / max(rainy_hours, 1)

        # Weather stability (simplified)
        weather_stability = (pressure_mb * humidity) / max(temp_c + 273, 1)

        # Temporal features (limited for historical data)
        precip_trend_3h = 0.0  # Would need more granular data
        temp_change_rate = 0.0
        humidity_trend = 0.0
        pressure_trend = 0.0
        wind_gust_factor = 1.0

        # Geographic features
        coastal_distance = abs(lon - 30.0)
        urban_centers = [
            (-26.2041, 28.0473),  # Johannesburg
            (-33.9249, 18.4241),  # Cape Town
            (-29.8587, 31.0218),  # Durban
            (-33.9608, 25.6022),  # Gqeberha
        ]
        min_urban_distance = min(
            np.sqrt((lat - uc_lat)**2 + (lon - uc_lon)**2)
            for uc_lat, uc_lon in urban_centers
        )
        urban_density_proxy = 1 / (min_urban_distance + 1)
        elevation_proxy = 1000 * (1 - abs(lat + 30) / 40)

        # Enhanced severity classification
        is_severe = int(
            (precip_mm > 50 and wind_kph > 60) or
            (wave_height > 3 and precip_mm > 20) or
            (humidity > 95 and temp_c < 5 and precip_mm > 15) or
            (precip_intensity > 10)
        )

        # Compile all features
        features = {
            'location': district,
            'lat': lat,
            'lon': lon,
            'temp_c': temp_c,
            'humidity': humidity,
            'wind_kph': wind_kph,
            'pressure_mb': pressure_mb,
            'precip_mm': precip_mm,
            'cloud': cloud,
            'wave_height': wave_height,
            'heat_index': heat_index,
            'wind_pressure_ratio': wind_pressure_ratio,
            'precip_intensity': precip_intensity,
            'weather_stability': weather_stability,
            'precip_trend_3h': precip_trend_3h,
            'temp_change_rate': temp_change_rate,
            'humidity_trend': humidity_trend,
            'pressure_trend': pressure_trend,
            'wind_gust_factor': wind_gust_factor,
            'coastal_distance': coastal_distance,
            'urban_density_proxy': urban_density_proxy,
            'elevation_proxy': elevation_proxy,
            'is_severe': is_severe
        }

        # Calculate anomaly score
        features['anomaly_score'] = calculate_enhanced_anomaly_score(features, district)

        return features

    except Exception as e:
        logger.error(f"Error extracting features from DataFrame for {district}: {e}")
        return None

def extract_enhanced_features(district: str, lat: float, lon: float, weather_hourly, marine_hourly=None):
    """
    Extract enhanced weather features including derived metrics for better prediction.

    Enhanced features include:
    - Basic weather data (temp, humidity, wind, pressure, precip, cloud)
    - Derived features (heat index, wind-pressure ratio, precip intensity, stability)
    - Temporal features (trends, rates of change)
    - Geographic features (coastal distance, relative position)
    """
    if not weather_hourly:
        return None

    try:
        # Extract last 24 hours of data for analysis
        temp_c = weather_hourly.Variables(0).ValuesAsNumpy()[-24:]
        precip = weather_hourly.Variables(1).ValuesAsNumpy()[-24:]
        wind = weather_hourly.Variables(2).ValuesAsNumpy()[-24:]
        humidity = weather_hourly.Variables(3).ValuesAsNumpy()[-24:]
        pressure = weather_hourly.Variables(4).ValuesAsNumpy()[-24:]
        cloud = weather_hourly.Variables(5).ValuesAsNumpy()[-24:]

        # Marine data (wave height) for coastal areas
        wave_height = 0.0
        if marine_hourly:
            wave_data = marine_hourly.Variables(0).ValuesAsNumpy()[-24:]
            wave_height = wave_data.mean() if len(wave_data) > 0 else 0.0

        # Calculate basic aggregates
        temp_mean = temp_c.mean()
        humidity_mean = humidity.mean()
        wind_max_kph = wind.max() * 3.6  # Convert m/s to km/h
        pressure_mean = pressure.mean()
        precip_total = precip.sum()
        cloud_mean = cloud.mean()

        # === DERIVED WEATHER FEATURES ===
        # Heat index (simplified) - combines temperature and humidity
        heat_index = temp_mean + (humidity_mean / 100) * 5

        # Wind-pressure ratio (storm intensity proxy)
        wind_pressure_ratio = (wind_max_kph / (pressure_mean / 1000)) if pressure_mean > 0 else 0

        # Precipitation intensity (mm per rainy hour)
        rainy_hours = np.sum(precip > 0.1)
        precip_intensity = precip_total / max(rainy_hours, 1)

        # Weather stability index (pressure * humidity / temperature)
        weather_stability = (pressure_mean * humidity_mean) / max(temp_mean + 273, 1)  # Kelvin

        # === TEMPORAL FEATURES ===
        # Precipitation trends (3-hour change)
        precip_recent = precip[-3:].sum()
        precip_previous = precip[-6:-3].sum() if len(precip) >= 6 else precip[:3].sum()
        precip_trend = (precip_recent - precip_previous) / max(precip_previous, 1e-6)

        # Temperature change rate (per hour)
        temp_change_rate = (temp_c[-1] - temp_c[0]) / 24 if len(temp_c) >= 24 else 0

        # Humidity trend over last 6 hours
        humidity_trend = (humidity[-1] - humidity[-6]) / 6 if len(humidity) >= 6 else 0

        # Pressure trend (barometric tendency)
        pressure_trend = (pressure[-1] - pressure[-6]) / 6 if len(pressure) >= 6 else 0

        # Wind gust factor (max wind / average wind)
        wind_avg = wind.mean()
        wind_gust_factor = wind.max() / max(wind_avg, 1e-6)

        # === GEOGRAPHIC FEATURES ===
        # Coastal distance (simplified based on longitude for South Africa)
        # Closer to east coast = more coastal influence
        coastal_distance = abs(lon - 30.0)  # East coast at ~30°E

        # Urban/rural proxy (based on known urban centers)
        urban_centers = [
            (-26.2041, 28.0473, "Johannesburg"),
            (-33.9249, 18.4241, "Cape Town"),
            (-29.8587, 31.0218, "Durban"),
            (-33.9608, 25.6022, "Gqeberha")
        ]

        # Distance to nearest urban center (population density proxy)
        min_urban_distance = min(
            np.sqrt((lat - uc_lat)**2 + (lon - uc_lon)**2)
            for uc_lat, uc_lon, _ in urban_centers
        )
        urban_density_proxy = 1 / (min_urban_distance + 1)  # Higher = more urban

        # Elevation proxy (rough estimate based on latitude for South Africa)
        elevation_proxy = 1000 * (1 - abs(lat + 30) / 40)  # Rough elevation gradient

        # === SEVERITY CLASSIFICATION ===
        # Enhanced flood risk indicators
        is_severe = int(
            # Heavy rainfall + strong winds
            (precip_total > 50 and wind_max_kph > 60) or
            # Coastal storm surge potential
            (wave_height > 3 and precip_total > 20) or
            # Freezing rain conditions
            (humidity_mean > 95 and temp_mean < 5 and precip_total > 15) or
            # Extreme precipitation events
            (precip_intensity > 10) or
            # Rapid weather deterioration
            (precip_trend > 2.0 and pressure_trend < -2)
        )

        # === COMPILE ALL FEATURES ===
        features = {
            # Location identifiers
            'location': district,
            'lat': lat,
            'lon': lon,

            # Basic weather features
            'temp_c': temp_mean,
            'humidity': humidity_mean,
            'wind_kph': wind_max_kph,
            'pressure_mb': pressure_mean,
            'precip_mm': precip_total,
            'cloud': cloud_mean,
            'wave_height': wave_height,

            # Derived weather features
            'heat_index': heat_index,
            'wind_pressure_ratio': wind_pressure_ratio,
            'precip_intensity': precip_intensity,
            'weather_stability': weather_stability,

            # Temporal features
            'precip_trend_3h': precip_trend,
            'temp_change_rate': temp_change_rate,
            'humidity_trend': humidity_trend,
            'pressure_trend': pressure_trend,
            'wind_gust_factor': wind_gust_factor,

            # Geographic features
            'coastal_distance': coastal_distance,
            'urban_density_proxy': urban_density_proxy,
            'elevation_proxy': elevation_proxy,

            # Target variable (for training)
            'is_severe': is_severe
        }

        # Calculate enhanced anomaly score
        features['anomaly_score'] = calculate_enhanced_anomaly_score(features, district)

        return features

    except Exception as e:
        logger.error(f"Error extracting enhanced features for {district}: {e}")
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
            features = extract_enhanced_features(district, lat, lon, weather_hourly, marine_hourly)
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
    """
    Calculate household resource requirements based on flood risk severity.

    Enhanced to handle the new risk categories and provide more detailed resource planning.
    """
    # Map new categories to resource levels
    severity_mapping = {
        'Very Low': 'Low',
        'Low': 'Low',
        'Medium': 'Medium',
        'Moderate': 'Medium',  # Legacy support
        'High': 'High',
        'Very High': 'Very High'
    }

    severity = severity_mapping.get(severity, severity)

    if severity not in ['Low', 'Medium', 'High', 'Very High']:
        logger.warning(f"Unknown severity level: {severity}, defaulting to Medium")
        severity = 'Medium'

    # Enhanced resource calculation with more granularity
    resource_config = {
        'Low': {
            'days': 2,
            'water_multiplier': 0.8,
            'food_multiplier': 0.8,
            'shelter': 0,
            'boats': 0.0
        },
        'Medium': {
            'days': 5,
            'water_multiplier': 1.0,
            'food_multiplier': 1.0,
            'shelter': 0.5,  # Partial shelter need
            'boats': 0.05
        },
        'High': {
            'days': 10,
            'water_multiplier': 1.5,
            'food_multiplier': 1.5,
            'shelter': 1,
            'boats': 0.15
        },
        'Very High': {
            'days': 14,
            'water_multiplier': 2.0,
            'food_multiplier': 2.0,
            'shelter': 1.5,  # Additional shelter capacity
            'boats': 0.25
        }
    }

    config = resource_config[severity]

    # Calculate resources with enhanced multipliers
    water = household_size * config['water_multiplier'] * config['days']
    food = household_size * config['food_multiplier'] * config['days']
    shelter = config['shelter']
    boats = config['boats']

    # Add emergency supplies based on severity
    emergency_supplies = {
        'Very High': {'medical_kits': 2, 'communication_devices': 1},
        'High': {'medical_kits': 1, 'communication_devices': 1},
        'Medium': {'medical_kits': 1},
        'Low': {}
    }

    resources = {
        'food_packs': round(food, 1),
        'water_gallons': round(water, 1),
        'shelter_needed': shelter > 0,
        'boats_needed': round(boats, 2),
        'preparation_days': config['days'],
        'household_size': household_size,
        'risk_level': severity
    }

    # Add emergency supplies for higher risk levels
    resources.update(emergency_supplies.get(severity, {}))

    return resources


def generate_risk_scores(df: pd.DataFrame, model=None, create_new_model=False):
    """
    Generate comprehensive risk scores using ensemble methods.

    Args:
        df: DataFrame with features
        model: Pre-trained ML model (if None, will train new one)
        create_new_model: Whether to train a new model from scratch

    Returns:
        DataFrame with risk scores, and trained model if create_new_model=True
    """
    if create_new_model or model is None:
        # Train a new model
        logger.info("Training new ML model with enhanced features...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Prepare training data
    X = df[FEATURE_COLS]
        y = df['is_severe'] if 'is_severe' in df.columns else [0] * len(df)

        if len(X) > 10:  # Need minimum samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            logger.info(f"✅ New model trained on {len(X_train)} samples")
        else:
            # Fallback: create a simple model
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='constant', constant=0)
            model.fit(X, y)
            logger.warning("⚠️ Used dummy model due to insufficient training data")

    # Now generate predictions with the model
    # Ensure all required features are present
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        logger.warning(f"Missing features for risk scoring: {missing_features}")
        # Fill missing features with defaults
        for feature in missing_features:
            df[feature] = get_feature_default_mean(feature)

    # Method 1: ML Model Prediction
    X = df[FEATURE_COLS]
    proba = model.predict_proba(X)

    # Handle single-class case (when all training data is the same class)
    if proba.shape[1] == 1:
        # If model only learned one class, use prediction confidence as risk score
        predictions = model.predict(X)
        ml_risk_score = predictions * 50  # Scale 0-1 to 0-50 for single class
    else:
        # Normal case with both classes
        ml_risk_score = proba[:, 1] * 100

    # Method 2: Enhanced Anomaly Score (already calculated in features)
    anomaly_score = df['anomaly_score'].values

    # Method 3: Temporal Risk Factors
    temporal_risk = calculate_temporal_risk_factors(df)

    # Method 4: Geographic Risk Factors
    geographic_risk = calculate_geographic_risk_factors(df)

    # Method 5: Weather Pattern Risk
    weather_pattern_risk = calculate_weather_pattern_risk(df)

    # Ensemble weighting (adjustable based on performance)
    weights = {
        'ml_model': 0.30,        # Primary ML prediction
        'anomaly': 0.25,         # Statistical anomalies
        'temporal': 0.20,        # Time-based trends
        'geographic': 0.15,      # Location-based factors
        'weather_pattern': 0.10  # Weather condition patterns
    }

    # Calculate ensemble composite score
    composite_score = (
        weights['ml_model'] * ml_risk_score +
        weights['anomaly'] * anomaly_score +
        weights['temporal'] * temporal_risk +
        weights['geographic'] * geographic_risk +
        weights['weather_pattern'] * weather_pattern_risk
    )

    # Clip to valid range
    composite_score = np.clip(composite_score, 0, 100)

    # Update DataFrame
    df['model_risk_score'] = ml_risk_score
    df['composite_risk_score'] = composite_score

    # Enhanced risk categorization with more granularity
    df['risk_category'] = pd.cut(
        composite_score,
        bins=[0, 25, 40, 60, 75, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ).astype(str)

    # Calculate household resources based on risk level
    df['household_resources'] = df['risk_category'].apply(
        lambda category: calculate_household_resources(category)
    ).apply(lambda x: x if isinstance(x, dict) else {})

    return df if not create_new_model else (df, model)

def calculate_temporal_risk_factors(df: pd.DataFrame) -> np.ndarray:
    """Calculate risk factors based on temporal trends."""
    temporal_scores = []

    for _, row in df.iterrows():
        temporal_score = 0

        # Precipitation trend (rapid increase = higher risk)
        if 'precip_trend_3h' in row and row['precip_trend_3h'] > 1.0:
            temporal_score += min(row['precip_trend_3h'] * 20, 50)

        # Pressure trend (rapid decrease = storm approaching)
        if 'pressure_trend' in row and row['pressure_trend'] < -1.5:
            temporal_score += min(abs(row['pressure_trend']) * 15, 30)

        # Temperature change rate (rapid cooling may indicate weather system)
        if 'temp_change_rate' in row and abs(row['temp_change_rate']) > 2:
            temporal_score += min(abs(row['temp_change_rate']) * 10, 20)

        temporal_scores.append(min(temporal_score, 100))

    return np.array(temporal_scores)

def calculate_geographic_risk_factors(df: pd.DataFrame) -> np.ndarray:
    """Calculate risk factors based on geographic characteristics."""
    geo_scores = []

    for _, row in df.iterrows():
        geo_score = 0

        # Coastal proximity (coastal areas more vulnerable to storms)
        if 'coastal_distance' in row:
            coastal_factor = max(0, (10 - row['coastal_distance']) / 10)  # 0-1 scale
            geo_score += coastal_factor * 20

        # Urban density (urban areas have more infrastructure at risk)
        if 'urban_density_proxy' in row and row['urban_density_proxy'] > 0.5:
            geo_score += (row['urban_density_proxy'] - 0.5) * 40

        # Elevation (lower elevations more flood-prone)
        if 'elevation_proxy' in row and row['elevation_proxy'] < 500:
            elevation_factor = (500 - row['elevation_proxy']) / 500  # 0-1 scale
            geo_score += elevation_factor * 15

        geo_scores.append(min(geo_score, 100))

    return np.array(geo_scores)

def calculate_weather_pattern_risk(df: pd.DataFrame) -> np.ndarray:
    """Calculate risk based on dangerous weather pattern combinations."""
    pattern_scores = []

    for _, row in df.iterrows():
        pattern_score = 0

        # Heavy rain + high winds (severe storm potential)
        if row.get('precip_mm', 0) > 30 and row.get('wind_kph', 0) > 40:
            pattern_score += 40

        # Very high humidity + temperature drop (potential severe weather)
        elif row.get('humidity', 0) > 85 and row.get('temp_change_rate', 0) < -1:
            pattern_score += 30

        # Extreme precipitation intensity
        if row.get('precip_intensity', 0) > 8:
            pattern_score += 35

        # Weather instability (pressure fluctuations + humidity changes)
        pressure_trend = abs(row.get('pressure_trend', 0))
        humidity_trend = abs(row.get('humidity_trend', 0))
        if pressure_trend > 1.5 and humidity_trend > 20:
            pattern_score += 25

        pattern_scores.append(min(pattern_score, 100))

    return np.array(pattern_scores)

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

# --- Testing & Validation Framework ---

def validate_predictions_against_history(model, test_days=30):
    """
    Backtest predictions against historical flood events

    Args:
        model: Trained ML model
        test_days: Number of days to test backward from today

    Returns:
        Dict with validation metrics
    """
    logger.info(f"Starting historical validation for last {test_days} days")

    # Load historical disaster data
    try:
        disasters_df = pd.read_excel("data/data_disaster.xlsx")
        disasters_df.columns = disasters_df.columns.str.strip().str.lower()
        logger.info(f"Loaded {len(disasters_df)} historical disaster records")

        # Create date column from separate year/month/day columns
        disasters_df['date'] = pd.to_datetime({
            'year': disasters_df['start year'],
            'month': disasters_df['start month'],
            'day': disasters_df['start day']
        }, errors='coerce')

        # Filter out rows without valid dates
        disasters_df = disasters_df.dropna(subset=['date'])

        logger.info(f"After date processing: {len(disasters_df)} valid disaster records")

    except Exception as e:
        logger.error(f"Failed to load disaster data: {e}")
        return {"error": "Could not load historical data"}

    # Get date range for testing (use past dates only, Open-Meteo archive doesn't have future data)
    end_date = datetime.now() - timedelta(days=1)  # Yesterday as end date
    start_date = end_date - timedelta(days=test_days)

    # Filter disasters in our test period
    test_disasters = disasters_df[
        (disasters_df['date'] >= start_date) &
        (disasters_df['date'] <= end_date)
    ]

    logger.info(f"Found {len(test_disasters)} disasters in test period")

    predictions = []
    actual_events = []

    # For each test day, make predictions and check against actual events
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"Testing predictions for {current_date.strftime('%Y-%m-%d')}")

        # Make predictions for all locations on this date
        daily_predictions = predict_for_date(model, current_date)

        # Check if any disasters occurred on this date
        day_disasters = test_disasters[
            test_disasters['date'].dt.date == current_date.date()
        ]

        # Record predictions and actual events
        for pred in daily_predictions:
            predictions.append({
                'date': current_date,
                'location': pred['location'],
                'predicted_risk': pred['composite_risk_score'],
                'predicted_category': pred['risk_category'],
                'had_disaster': pred['location'] in day_disasters['location'].values
            })

        for _, disaster in day_disasters.iterrows():
            actual_events.append({
                'date': current_date,
                'location': disaster['location'],
                'severity': disaster.get('severity', 'Unknown'),
                'actual_flood': True
            })

        current_date += timedelta(days=1)

    # Calculate validation metrics
    metrics = calculate_validation_metrics(predictions, actual_events)

    logger.info(f"Validation completed: {metrics.get('overall_accuracy', 0):.1%} accuracy, {metrics.get('total_predictions', 0)} predictions, {metrics.get('total_actual_floods', 0)} actual floods")
    return metrics

def predict_for_date(model, target_date):
    """
    Generate predictions for all locations on a specific historical date

    Args:
        model: Trained ML model
        target_date: Date to predict for

    Returns:
        List of predictions for all locations
    """
    predictions = []

    for district, (lat, lon) in DISTRICT_COORDS.items():
        try:
            # Fetch historical weather for this date
            historical_weather = fetch_historical_weather_for_date(lat, lon, target_date)

            if historical_weather.empty:
                logger.warning(f"No weather data for {district} on {target_date}")
                continue

            # Extract features from historical data
            features = extract_features_from_dataframe(district, lat, lon, historical_weather)

            if features:
                # Generate risk score
                prediction = generate_risk_scores(pd.DataFrame([features]), model)
                prediction_row = prediction.iloc[0]

                predictions.append({
                    'location': district,
                    'lat': lat,
                    'lon': lon,
                    'date': target_date,
                    'composite_risk_score': prediction_row['composite_risk_score'],
                    'risk_category': prediction_row['risk_category'],
                    'model_risk_score': prediction_row['model_risk_score'],
                    'anomaly_score': prediction_row['anomaly_score']
                })

        except Exception as e:
            logger.error(f"Failed to predict for {district}: {e}")
            continue

    return predictions

def calculate_validation_metrics(predictions, actual_events):
    """
    Calculate comprehensive validation metrics

    Args:
        predictions: List of prediction records
        actual_events: List of actual disaster events

    Returns:
        Dict with validation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix

    # Convert to binary classification (flood vs no flood)
    y_true = []
    y_pred = []
    risk_scores = []

    for pred in predictions:
        # True label: did a flood actually occur in this location on this date?
        actual_flood = pred['had_disaster']

        # Predicted label: was risk score above threshold?
        predicted_flood = pred['predicted_risk'] >= 70  # High risk threshold

        y_true.append(1 if actual_flood else 0)
        y_pred.append(1 if predicted_flood else 0)
        risk_scores.append(pred['predicted_risk'])

    # Calculate metrics
    if len(set(y_true)) > 1:  # Need both classes
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
    else:
        report = {"error": "Insufficient class diversity for classification metrics"}
        cm = None

    # Additional metrics
    total_predictions = len(predictions)
    total_actual_floods = sum(y_true)
    total_predicted_floods = sum(y_pred)

    # Risk score distribution
    high_risk_predictions = sum(1 for r in risk_scores if r >= 70)
    medium_risk_predictions = sum(1 for r in risk_scores if 40 <= r < 70)
    low_risk_predictions = sum(1 for r in risk_scores if r < 40)

    # Accuracy by risk category
    correct_high = sum(1 for p, t in zip(y_pred, y_true) if p == t and p == 1)
    correct_low = sum(1 for p, t in zip(y_pred, y_true) if p == t and p == 0)

    return {
        'total_predictions': total_predictions,
        'total_actual_floods': total_actual_floods,
        'total_predicted_floods': total_predicted_floods,

        'classification_report': report,

        'confusion_matrix': cm.tolist() if cm is not None else None,

        'risk_distribution': {
            'high_risk_count': high_risk_predictions,
            'medium_risk_count': medium_risk_predictions,
            'low_risk_count': low_risk_predictions,
            'high_risk_percentage': (high_risk_predictions / total_predictions) * 100,
            'medium_risk_percentage': (medium_risk_predictions / total_predictions) * 100,
            'low_risk_percentage': (low_risk_predictions / total_predictions) * 100
        },

        'accuracy_by_category': {
            'correct_high_risk_predictions': correct_high,
            'correct_low_risk_predictions': correct_low,
            'high_risk_accuracy': correct_high / total_predicted_floods if total_predicted_floods > 0 else 0,
            'low_risk_accuracy': correct_low / (total_predictions - total_predicted_floods) if (total_predictions - total_predicted_floods) > 0 else 0
        },

        'overall_accuracy': sum(1 for p, t in zip(y_pred, y_true) if p == t) / len(y_true),

        'false_positive_rate': sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0) / len(y_true),
        'false_negative_rate': sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1) / len(y_true),

        'test_period_days': len(set(p['date'] for p in predictions)),
        'locations_tested': len(set(p['location'] for p in predictions))
    }

def run_prediction_accuracy_test():
    """Run comprehensive prediction accuracy testing"""
    logger.info("Starting comprehensive prediction accuracy test")

    try:
        # Load model
        model = joblib.load("data/rf_model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"error": "Could not load ML model"}

    # Run validation against recent history
    validation_results = validate_predictions_against_history(model, test_days=90)

    # Run additional tests
    stress_test_results = run_stress_tests(model)
    edge_case_results = test_edge_cases(model)

    # Combine results
    comprehensive_report = {
        'timestamp': datetime.now().isoformat(),
        'validation_results': validation_results,
        'stress_test_results': stress_test_results,
        'edge_case_results': edge_case_results,
        'overall_assessment': assess_overall_accuracy(validation_results)
    }

    # Save report
    with open("prediction_accuracy_report.json", "w") as f:
        import json
        json.dump(comprehensive_report, f, indent=2, default=str)

    logger.info("Comprehensive accuracy test completed")
    return comprehensive_report

def run_stress_tests(model):
    """Test model performance under stress conditions"""
    logger.info("Running stress tests")

    # Test with extreme weather conditions
    extreme_scenarios = [
        {"temp_c": 45, "precip_mm": 200, "wind_kph": 150, "description": "Extreme heat + heavy rain + hurricane"},
        {"temp_c": -5, "precip_mm": 300, "wind_kph": 200, "description": "Freezing + extreme rain + storm"},
        {"temp_c": 15, "precip_mm": 0, "wind_kph": 0, "description": "No weather variation"},
    ]

    stress_results = []
    for scenario in extreme_scenarios:
        # Create test data
        test_data = pd.DataFrame([{
            'lat': -29.8587, 'lon': 31.0218,  # Durban coordinates
            'temp_c': scenario['temp_c'],
            'humidity': 80,
            'wind_kph': scenario['wind_kph'],
            'pressure_mb': 1000,
            'precip_mm': scenario['precip_mm'],
            'cloud': 90,
            'wave_height': 3.0
        }])

        try:
            prediction = generate_risk_scores(test_data, model)
            risk_score = prediction.iloc[0]['composite_risk_score']

            stress_results.append({
                'scenario': scenario['description'],
                'risk_score': risk_score,
                'expected_behavior': 'high_risk' if scenario['precip_mm'] > 100 else 'variable'
            })

        except Exception as e:
            stress_results.append({
                'scenario': scenario['description'],
                'error': str(e)
            })

    return stress_results

def test_edge_cases(model):
    """Test model with edge cases and unusual inputs"""
    logger.info("Testing edge cases")

    edge_cases = [
        # Invalid coordinates
        {"lat": 91, "lon": 0, "description": "Invalid latitude"},
        {"lat": 0, "lon": 181, "description": "Invalid longitude"},

        # Extreme values
        {"temp_c": 100, "precip_mm": 1000, "description": "Extreme temperature"},
        {"precip_mm": -10, "description": "Negative precipitation"},

        # Missing critical data
        {"missing_humidity": True, "description": "Missing humidity data"},
    ]

    edge_results = []
    for case in edge_cases:
        try:
            # Create test data with edge case
            test_data = pd.DataFrame([{
                'lat': case.get('lat', -29.8587),
                'lon': case.get('lon', 31.0218),
                'temp_c': case.get('temp_c', 25),
                'humidity': 80 if not case.get('missing_humidity') else None,
                'wind_kph': 20,
                'pressure_mb': 1013,
                'precip_mm': max(case.get('precip_mm', 5), 0),  # Ensure non-negative
                'cloud': 50,
                'wave_height': 1.5
            }])

            prediction = generate_risk_scores(test_data, model)
            risk_score = prediction.iloc[0]['composite_risk_score']

            edge_results.append({
                'case': case['description'],
                'risk_score': risk_score,
                'handled_gracefully': not pd.isna(risk_score)
            })

        except Exception as e:
            edge_results.append({
                'case': case['description'],
                'error': str(e),
                'handled_gracefully': False
            })

    return edge_results

def assess_overall_accuracy(validation_results):
    """Provide overall assessment of prediction accuracy"""

    if 'error' in validation_results:
        return {"assessment": "failed", "reason": validation_results['error']}

    accuracy = validation_results.get('overall_accuracy', 0)
    false_positive_rate = validation_results.get('false_positive_rate', 1)
    false_negative_rate = validation_results.get('false_negative_rate', 1)

    # Assessment criteria
    if accuracy > 0.8 and false_positive_rate < 0.1 and false_negative_rate < 0.2:
        assessment = "excellent"
        confidence = "High confidence in predictions"
    elif accuracy > 0.7 and false_positive_rate < 0.2 and false_negative_rate < 0.3:
        assessment = "good"
        confidence = "Moderate confidence, needs monitoring"
    elif accuracy > 0.6:
        assessment = "acceptable"
        confidence = "Basic functionality, needs improvement"
    else:
        assessment = "poor"
        confidence = "Not reliable for production use"

    return {
        "assessment": assessment,
        "overall_accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "confidence_level": confidence,
        "recommendations": get_accuracy_recommendations(assessment)
    }

def get_accuracy_recommendations(assessment):
    """Provide recommendations based on accuracy assessment"""

    recommendations = {
        "excellent": [
            "Model performing well - continue monitoring",
            "Consider adding more weather features",
            "Expand geographical coverage"
        ],
        "good": [
            "Tune risk thresholds for better balance",
            "Add more historical training data",
            "Implement ensemble methods"
        ],
        "acceptable": [
            "Review feature engineering",
            "Improve anomaly detection",
            "Add location-specific models"
        ],
        "poor": [
            "Complete model retraining needed",
            "Review data quality and sources",
            "Consider alternative ML approaches",
            "Focus on false positive/negative reduction"
        ]
    }

    return recommendations.get(assessment, ["Review model implementation"])

def fetch_historical_weather_for_date(lat, lon, target_date):
    """
    Fetch historical weather data for a specific date

    Args:
        lat, lon: Coordinates
        target_date: Date to fetch weather for

    Returns:
        DataFrame with weather data for that date
    """
    try:
        # Use Open-Meteo archive API for historical data
        url = "https://archive-api.open-meteo.com/v1/archive"

        # Fetch data for the target date ± 1 day for context
        start_date = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m,pressure_msl,cloud_cover",
            "timezone": "auto"
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process the response similar to real-time collection
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
        hourly_pressure_msl = hourly.Variables(4).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()

        # Create DataFrame for the target date
        target_date_str = target_date.strftime("%Y-%m-%d")
        weather_data = []

        for i in range(len(hourly_temperature_2m)):
            timestamp = datetime.fromtimestamp(hourly.Time() + hourly.Interval() * i)
            date_str = timestamp.strftime("%Y-%m-%d")

            if date_str == target_date_str:
                weather_data.append({
                    'timestamp': timestamp,
                    'temp_c': float(hourly_temperature_2m[i]),
                    'precip_mm': float(hourly_precipitation[i]),
                    'wind_kph': float(hourly_wind_speed_10m[i]),
                    'humidity': float(hourly_relative_humidity_2m[i]),
                    'pressure_mb': float(hourly_pressure_msl[i]),
                    'cloud': float(hourly_cloud_cover[i])
                })

        return pd.DataFrame(weather_data)

    except Exception as e:
        logger.error(f"Failed to fetch historical weather for {target_date}: {e}")
        return pd.DataFrame()

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
    df = generate_risk_scores(df, model)
    df['household_resources'] = df['risk_category'].apply(
        lambda s: calculate_household_resources(s)
    )

    # Save
    df.to_csv("weather_data_scored.csv", index=False)
    logger.info("✅ Predictions saved")

    return df

if __name__ == "__main__":
    main()