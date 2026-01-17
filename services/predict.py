from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import numpy as np
import openmeteo_requests
from pydantic import ValidationError
import requests_cache
from pydantic import BaseModel, model_validator, field_validator, constr, confloat,Field, ConfigDict
from typing import List, Optional, Literal
import logging

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

def fetch_historical_baseline(lat: float, lon: float, days: int = 3650) -> Dict[str, Dict[str, float]]:
    """
    Fetch historical weather data to compute baseline mean and std.
    Defaults to 10 years (3650 days) of history.
    """
    # Setup Open-Meteo client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean,pressure_msl_mean,cloud_cover_mean",
        "timezone": "auto"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        if not responses:
            raise ValueError("No response from historical API")
        daily = responses[0].Daily()
        
        # Extract data
        df = pd.DataFrame({
            "temp_c": daily.Variables(0).ValuesAsNumpy(),
            "precip_mm": daily.Variables(1).ValuesAsNumpy(),
            "wind_kph": daily.Variables(2).ValuesAsNumpy() * 3.6, # m/s to km/h
            "humidity": daily.Variables(3).ValuesAsNumpy(),
            "pressure_mb": daily.Variables(4).ValuesAsNumpy(),
            "cloud": daily.Variables(5).ValuesAsNumpy(),
        })
        
        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if df.empty:
            raise ValueError("No valid historical data")

        return {
            col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
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

def compare_with_historical_disasters(current_weather: Dict[str, float], location_name: str) -> float:
    """
    Compare current weather against known historical disasters.
    Returns a risk boost score (0-100) if a match is found.
    """
    # Placeholder for known disaster signatures (in real app, fetch from DB)
    # Example: 2022 KZN Floods signature
    KNOWN_DISASTERS = [
        {
            "name": "2022 KZN Floods",
            "signature": {
                "precip_mm": 300.0, # Extreme rain
                "wind_kph": 40.0,
                "humidity": 90.0
            },
            "threshold": 0.8 # Similarity threshold
        }
    ]
    
    max_similarity = 0.0
    
    for disaster in KNOWN_DISASTERS:
        # Calculate similarity (simplified cosine similarity or distance)
        # Here we use a simple ratio check for key metrics
        
        precip_ratio = min(current_weather.get('precip_mm', 0) / disaster['signature']['precip_mm'], 1.0)
        
        # If precipitation is approaching disaster levels
        if precip_ratio > 0.7:
            logger.warning(f"Weather pattern resembles {disaster['name']}! Precip Ratio: {precip_ratio:.2f}")
            return 95.0 # Force high risk
            
    return 0.0

def fetch_weather_and_marine_data(lat, lon, is_coastal=False):
    # Setup Open-Meteo client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

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
    """
    Adjust anomaly score based on seasonal expectations.
    e.g., Rain in winter in Cape Town is normal (1.0), but in Durban it's rare (1.5).
    """
    # Simplified seasonal logic
    month = datetime.now().month
    is_winter = 5 <= month <= 8
    
    # Western Cape (Winter Rainfall)
    if any(place in district for place in ["Cape Town", "George", "Stellenbosch"]):
        if is_winter:
            return 1.0 # Normal
        else:
            return 1.2 # Summer rain is slightly anomalous
            
    # KZN / Gauteng (Summer Rainfall)
    else:
        if is_winter:
            return 1.5 # Winter rain is anomalous
        else:
            return 1.0 # Summer rain is normal
            
    return 1.0

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

def extract_enhanced_features(district: str, lat: float, lon: float, weather_hourly, marine_hourly=None):
    """
    Extract enhanced weather features including derived metrics for better prediction.
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
        heat_index = temp_mean + (humidity_mean / 100) * 5
        wind_pressure_ratio = (wind_max_kph / (pressure_mean / 1000)) if pressure_mean > 0 else 0
        rainy_hours = np.sum(precip > 0.1)
        precip_intensity = precip_total / max(rainy_hours, 1)
        weather_stability = (pressure_mean * humidity_mean) / max(temp_mean + 273, 1)

        # === TEMPORAL FEATURES ===
        precip_recent = precip[-3:].sum()
        precip_previous = precip[-6:-3].sum() if len(precip) >= 6 else precip[:3].sum()
        precip_trend = (precip_recent - precip_previous) / max(precip_previous, 1e-6)
        temp_change_rate = (temp_c[-1] - temp_c[0]) / 24 if len(temp_c) >= 24 else 0
        humidity_trend = (humidity[-1] - humidity[-6]) / 6 if len(humidity) >= 6 else 0
        pressure_trend = (pressure[-1] - pressure[-6]) / 6 if len(pressure) >= 6 else 0
        wind_avg = wind.mean()
        wind_gust_factor = wind.max() / max(wind_avg, 1e-6)

        # === GEOGRAPHIC FEATURES ===
        coastal_distance = abs(lon - 30.0)
        urban_centers = [
            (-26.2041, 28.0473, "Johannesburg"),
            (-33.9249, 18.4241, "Cape Town"),
            (-29.8587, 31.0218, "Durban"),
            (-33.9608, 25.6022, "Gqeberha")
        ]
        min_urban_distance = min(
            np.sqrt((lat - uc_lat)**2 + (lon - uc_lon)**2)
            for uc_lat, uc_lon, _ in urban_centers
        )
        urban_density_proxy = 1 / (min_urban_distance + 1)
        elevation_proxy = 1000 * (1 - abs(lat + 30) / 40)

        # === SEVERITY CLASSIFICATION ===
        is_severe = int(
            (precip_total > 50 and wind_max_kph > 60) or
            (wave_height > 3 and precip_total > 20) or
            (humidity_mean > 95 and temp_mean < 5 and precip_total > 15) or
            (precip_intensity > 10) or
            (precip_trend > 2.0 and pressure_trend < -2)
        )

        # === COMPILE ALL FEATURES ===
        features = {
            'location': district,
            'lat': lat,
            'lon': lon,
            'temp_c': temp_mean,
            'humidity': humidity_mean,
            'wind_kph': wind_max_kph,
            'pressure_mb': pressure_mean,
            'precip_mm': precip_total,
            'cloud': cloud_mean,
            'wave_height': wave_height,
            'heat_index': heat_index,
            'wind_pressure_ratio': wind_pressure_ratio,
            'precip_intensity': precip_intensity,
            'weather_stability': weather_stability,
            'precip_trend_3h': precip_trend,
            'temp_change_rate': temp_change_rate,
            'humidity_trend': humidity_trend,
            'pressure_trend': pressure_trend,
            'wind_gust_factor': wind_gust_factor,
            'coastal_distance': coastal_distance,
            'urban_density_proxy': urban_density_proxy,
            'elevation_proxy': elevation_proxy,
            'is_severe': is_severe
        }

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
        # Simplified handling for list input
        pass 

    for district, (lat, lon) in target_locations.items():
        is_coastal = any(k.lower() in district.lower() for k in coastal_keywords)
        weather_hourly, marine_hourly = fetch_weather_and_marine_data(lat, lon, is_coastal)
        if weather_hourly:
            features = extract_enhanced_features(district, lat, lon, weather_hourly, marine_hourly)
            if features:
                try:
                    # Simple validation check
                    if features['temp_c'] > -100: 
                        data.append(features)
                        logger.info(f"✅ Validated data for {district}")
                except Exception as ve:
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

    joblib.dump(model, 'rf_model.pkl')
    logger.info("💾 Model saved as 'rf_model.pkl'")
    return model

def calculate_household_resources(severity: str, household_size: int = 4) -> Dict[str, float]:
    # Map new categories to resource levels
    severity_mapping = {
        'Very Low': 'Low',
        'Low': 'Low',
        'Medium': 'Medium',
        'Moderate': 'Medium',
        'High': 'High',
        'Very High': 'Very High'
    }
    severity = severity_mapping.get(severity, severity)
    if severity not in ['Low', 'Medium', 'High', 'Very High']:
        severity = 'Medium'

    resource_config = {
        'Low': {'days': 2, 'water_multiplier': 0.8, 'food_multiplier': 0.8, 'shelter': 0, 'boats': 0.0},
        'Medium': {'days': 5, 'water_multiplier': 1.0, 'food_multiplier': 1.0, 'shelter': 0.5, 'boats': 0.05},
        'High': {'days': 10, 'water_multiplier': 1.5, 'food_multiplier': 1.5, 'shelter': 1, 'boats': 0.15},
        'Very High': {'days': 14, 'water_multiplier': 2.0, 'food_multiplier': 2.0, 'shelter': 1.5, 'boats': 0.25}
    }
    config = resource_config[severity]

    water = household_size * config['water_multiplier'] * config['days']
    food = household_size * config['food_multiplier'] * config['days']
    shelter = config['shelter']
    boats = config['boats']

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
    resources.update(emergency_supplies.get(severity, {}))
    return resources

def calculate_temporal_risk_factors(df: pd.DataFrame) -> np.ndarray:
    """Calculate risk factors based on temporal trends."""
    temporal_scores = []
    for _, row in df.iterrows():
        temporal_score = 0
        if 'precip_trend_3h' in row and row['precip_trend_3h'] > 1.0:
            temporal_score += min(row['precip_trend_3h'] * 20, 50)
        if 'pressure_trend' in row and row['pressure_trend'] < -1.0:
            temporal_score += min(abs(row['pressure_trend']) * 15, 50)
        temporal_scores.append(temporal_score)
    return np.array(temporal_scores)

def calculate_geographic_risk_factors(df: pd.DataFrame) -> np.ndarray:
    """Calculate risk factors based on geography."""
    geo_scores = []
    for _, row in df.iterrows():
        geo_score = 0
        if 'coastal_distance' in row and row['coastal_distance'] < 0.5:
            geo_score += 20
        if 'urban_density_proxy' in row and row['urban_density_proxy'] > 0.5:
            geo_score += 15
        geo_scores.append(geo_score)
    return np.array(geo_scores)

def calculate_weather_pattern_risk(df: pd.DataFrame) -> np.ndarray:
    """Calculate risk based on specific weather patterns."""
    pattern_scores = []
    for _, row in df.iterrows():
        score = 0
        # Check for disaster match
        if 'precip_mm' in row:
             # Simplified check, real logic is in compare_with_historical_disasters
             pass
        pattern_scores.append(score)
    return np.array(pattern_scores)

def generate_risk_scores(df: pd.DataFrame, model=None, create_new_model=False):
    """
    Generate comprehensive risk scores using ensemble methods.
    """
    if create_new_model or model is None:
        logger.info("Training new ML model with enhanced features...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X = df[FEATURE_COLS]
        y = df['is_severe'] if 'is_severe' in df.columns else [0] * len(df)

        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            logger.info(f"✅ New model trained on {len(X_train)} samples")
        else:
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='constant', constant=0)
            model.fit(X, y)
            logger.warning("⚠️ Used dummy model due to insufficient training data")

    # Ensure all required features are present
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        for feature in missing_features:
            df[feature] = get_feature_default_mean(feature)

    # Method 1: ML Model Prediction
    X = df[FEATURE_COLS]
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        predictions = model.predict(X)
        ml_risk_score = predictions * 50
    else:
        ml_risk_score = proba[:, 1] * 100

    # Method 2: Enhanced Anomaly Score
    anomaly_score = df['anomaly_score'].values

    # Method 3: Temporal Risk Factors
    temporal_risk = calculate_temporal_risk_factors(df)

    # Method 4: Geographic Risk Factors
    geographic_risk = calculate_geographic_risk_factors(df)

    # Method 5: Weather Pattern Risk
    weather_pattern_risk = calculate_weather_pattern_risk(df)
    
    # Check for historical disaster matches
    for idx, row in df.iterrows():
        match_score = compare_with_historical_disasters(row.to_dict(), row.get('location', 'Unknown'))
        if match_score > 0:
            weather_pattern_risk[idx] = max(weather_pattern_risk[idx], match_score)

    # Ensemble weighting
    weights = {
        'ml_model': 0.30,
        'anomaly': 0.25,
        'temporal': 0.20,
        'geographic': 0.15,
        'weather_pattern': 0.10
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

    # Enhanced risk categorization
    df['risk_category'] = pd.cut(
        composite_score,
        bins=[0, 25, 40, 60, 75, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ).astype(str)

    # Calculate household resources
    df['household_resources'] = df['risk_category'].apply(
        lambda category: calculate_household_resources(category)
    ).apply(lambda x: x if isinstance(x, dict) else {})

    return df if not create_new_model else (df, model)

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