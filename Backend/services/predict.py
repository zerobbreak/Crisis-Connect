from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import numpy as np
import openmeteo_requests
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
from sklearn.metrics import classification_report
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Configuration
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
    }
    
    try:
        weather_response = openmeteo.weather_api(WEATHER_URL, params=weather_params)[0]
        weather_hourly = weather_response.Hourly()
        marine_response = openmeteo.weather_api(MARINE_URL, params=marine_params)[0] if is_coastal else None
        marine_hourly = marine_response.Hourly() if marine_response else None
        return weather_hourly, marine_hourly
    except Exception as e:
        print(f"⚠️ Error fetching data for lat={lat}, lon={lon}: {e}")
        return None, None

def compute_anomaly_score(features: Dict[str, float], district: str) -> float:
    """Calculate anomaly score using z-scores."""
    baseline = BASELINE_WEATHER.get(district, {
        'temp_c': {'mean': 20.0, 'std': 5.0},
        'humidity': {'mean': 50.0, 'std': 10.0},
        'wind_kph': {'mean': 15.0, 'std': 10.0},
        'pressure_mb': {'mean': 1013.0, 'std': 5.0},
        'precip_mm': {'mean': 2.0, 'std': 2.0},
        'cloud': {'mean': 20.0, 'std': 10.0},
        'wave_height': {'mean': 0.0, 'std': 0.5}
    })
    
    z_scores = []
    for feature in ['precip_mm', 'wind_kph', 'wave_height']:
        current = features.get(feature, 0)
        mean = baseline[feature]['mean']
        std = baseline[feature]['std']
        z_score = abs(current - mean) / max(std, 1e-10)
        z_scores.append(z_score)
    
    # Average z-score, scaled to 0–100
    anomaly_score = min(np.mean(z_scores) * 20, 100)  # Scale factor 20 for reasonable range
    return anomaly_score

def extract_features(district, lat, lon, weather_hourly, marine_hourly=None):
    if not weather_hourly:
        return None
    
    try:
        temp_c = weather_hourly.Variables(0).ValuesAsNumpy()[-24:].mean()
        precip_mm = weather_hourly.Variables(1).ValuesAsNumpy()[-24:].sum()
        wind_kph = weather_hourly.Variables(2).ValuesAsNumpy()[-24:].max() * 3.6
        humidity = weather_hourly.Variables(3).ValuesAsNumpy()[-24:].mean()
        pressure_mb = weather_hourly.Variables(4).ValuesAsNumpy()[-24:].mean()
        cloud = weather_hourly.Variables(5).ValuesAsNumpy()[-24:].mean()
        wave_height = marine_hourly.Variables(0).ValuesAsNumpy()[-24:].mean() if marine_hourly else 0
    except Exception as e:
        print(f"⚠️ Error extracting features for {district}: {e}")
        return None

    is_severe = int(
        (precip_mm > 25 and wind_kph > 50) or
        (wave_height > 2 and precip_mm > 15) or
        (humidity > 90 and temp_c < 10 and precip_mm > 10)
    )

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
        'is_severe': is_severe
    }

    features['anomaly_score'] = compute_anomaly_score(features, district)
    return features

def collect_all_data(locations: Optional[Union[Dict[str, Tuple[float, float]], List[Union[str, Tuple[float, float], Dict[str, float]]]]] = None) -> pd.DataFrame:
    data: List[Dict[str, float]] = []
    
    # Handle different input types for locations
    if locations is None:
        target_locations = DISTRICT_COORDS
    elif isinstance(locations, dict):
        target_locations = {str(k): (float(v[0]), float(v[1])) for k, v in locations.items() if v is not None}
    else:
        target_locations = {}
        for idx, item in enumerate(locations):
            if isinstance(item, str):
                # Geocode string to coordinates
                try:
                    geocode_result = _geocode(item)
                    if geocode_result:
                        target_locations[item] = (geocode_result.latitude, geocode_result.longitude)
                    else:
                        print(f"⚠️ Geocoding failed for location: {item}")
                        continue
                except Exception as e:
                    print(f"⚠️ Geocoding error for {item}: {e}")
                    continue
            elif isinstance(item, tuple) and len(item) == 2:
                target_locations[f"place_{idx}"] = (float(item[0]), float(item[1]))
            elif isinstance(item, dict) and item.get('lat') is not None and item.get('lon') is not None:
                target_locations[item.get('name', f'place_{idx}')] = (float(item['lat']), float(item['lon']))
            else:
                print(f"⚠️ Invalid location format at index {idx}: {item}")
                continue

    if not target_locations:
        print("⚠️ No valid locations provided")
        return pd.DataFrame()

    coastal_keywords = [
        "Durban", "Richards Bay", "Port Shepstone", "Ballito", "Cape Town",
        "George", "Mossel Bay", "Hermanus", "Saldanha", "Knysna",
        "Gqeberha", "Port Elizabeth", "East London"
    ]

    for district, (lat, lon) in target_locations.items():
        is_coastal = any(k.lower() in district.lower() for k in coastal_keywords)
        weather_hourly, marine_hourly = fetch_weather_and_marine_data(lat, lon, is_coastal=is_coastal)
        if weather_hourly:
            features = extract_features(district, lat, lon, weather_hourly, marine_hourly)
            if features:
                data.append(features)
                print(f"✅ Collected data for {district}: {features}")
        else:
            print(f"❌ Failed to fetch data for {district}")

    df = pd.DataFrame(data)
    
    if df.empty:
        print("⚠️ No data collected for any location")
        return df

    if df['is_severe'].nunique() < 2:
        print("⚠️ Only one class detected, adding synthetic negative samples")
        synthetic_data = []
        for district, (lat, lon) in DISTRICT_COORDS.items():
            synthetic_data.append({
                'location': f"{district}_synthetic",
                'lat': lat + 0.01,
                'lon': lon + 0.01,
                'temp_c': 20.0,
                'humidity': 50.0,
                'wind_kph': 10.0,
                'pressure_mb': 1013.0,
                'precip_mm': 0.0,
                'cloud': 20.0,
                'wave_height': 0.0,
                'is_severe': 0,
                'anomaly_score': 0.0
            })
        df_synthetic = pd.DataFrame(synthetic_data)
        df = pd.concat([df, df_synthetic], ignore_index=True)
    
    return df

def train_model(df):
    print("Class distribution before training:\n", df['is_severe'].value_counts())
    if df['is_severe'].nunique() < 2:
        raise ValueError("Cannot train model: Data contains only one class")

    X = df[FEATURE_COLS]
    y = df['is_severe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("\n🌲 Random Forest Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))

    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"📊 Cross-validation scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    importances = rf_model.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=FEATURE_COLS)
    plt.title("Feature Importance in Flood Prediction")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    print("\n📉 Logistic Regression Report:\n", classification_report(y_test, y_pred_log, zero_division=0))

    joblib.dump(rf_model, 'rf_model.pkl')
    print("💾 Random Forest model saved as 'rf_model.pkl'")
    return rf_model

def calculate_household_resources(
    severity: str,
    household_size: int = 4,
    days_low: int = 3,
    days_medium: int = 7,
    days_high: int = 14
) -> Dict[str, float]:
    """
    Calculate estimated resources needed for a household based on flood severity.
    
    Args:
        severity (str): Disaster severity level ('Low', 'Medium', 'High', or 'Moderate').
        household_size (int): Number of people in the household.
        days_low/medium/high (int): Days of supply for each severity level.
    
    Returns:
        Dict[str, float]: Dictionary of resource estimates.
    
    Raises:
        ValueError: If invalid severity provided.
    """
    if severity not in ['Low', 'Medium', 'High', 'Moderate']:
        raise ValueError("Severity must be 'Low', 'Medium', 'High', or 'Moderate'.")
    
    # Map 'Moderate' to 'Medium' for consistency
    if severity == 'Moderate':
        severity = 'Medium'
    
    # Base multipliers per person
    water_per_person_day = 1.0  # Gallons
    food_packs_per_person_day = 1.0  # Assuming 1 pack = 3 meals/day
    
    if severity == 'Low':
        days_supply = days_low
        shelter_needed = 0
        boats_needed = 0.0
    elif severity == 'Medium':
        days_supply = days_medium
        shelter_needed = 1
        boats_needed = 0.0
    else:  # High
        days_supply = days_high
        shelter_needed = 1
        boats_needed = 0.1
    
    # Calculations
    water_gallons = household_size * water_per_person_day * days_supply
    food_packs = household_size * food_packs_per_person_day * days_supply
    
    return {
        'food_packs': round(food_packs, 1),
        'water_gallons': round(water_gallons, 1),
        'shelter_needed': bool(shelter_needed),
        'boats_needed': round(boats_needed, 2)
    }

def generate_risk_scores(model, df):
    X = df[FEATURE_COLS]
    df['model_risk_score'] = model.predict_proba(X)[:, 1] * 100
    df['composite_risk_score'] = 0.0
    df['risk_category'] = 'Low'

    for idx, row in df.iterrows():
        threshold_severe = row['is_severe'] == 1
        model_score = row['model_risk_score']
        anomaly_score = row['anomaly_score']
        
        if threshold_severe:
            composite_score = min(max(model_score, 80), 100)
            risk_category = "High"
        else:
            composite_score = 0.8 * model_score + 0.2 * anomaly_score
            if composite_score > 70 or (model_score > 40 and anomaly_score > 50):
                risk_category = "High"
            elif composite_score > 40 or (model_score > 20 and anomaly_score > 30):
                risk_category = "Medium"
            else:
                risk_category = "Low"
        
        df.at[idx, 'composite_risk_score'] = composite_score
        df.at[idx, 'risk_category'] = risk_category

    # Add household resources calculation
    df['household_resources'] = df['risk_category'].apply(lambda s: calculate_household_resources(s))
    
    return df

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
    df = collect_all_data()
    if df.empty:
        print("❌ No data collected. Exiting.")
        return

    try:
        df_disasters = pd.read_excel("data_disaster.xlsx", sheet_name=0)
        weather_disasters = df_disasters[
            df_disasters['Disaster Type'].str.lower().isin([
                'flood', 'storm', 'drought', 'cold wave', 'heat wave'
            ])
        ].dropna(subset=['Latitude', 'Longitude'])

        # Impute missing weather data for historical entries
        median_features = df[df['location'].str.contains('_synthetic') == False][FEATURE_COLS].median()
        historical_data = []
        for _, row in weather_disasters.iterrows():
            historical_data.append({
                'location': row.get('Location', 'Unknown'),
                'lat': row['Latitude'],
                'lon': row['Longitude'],
                'temp_c': median_features['temp_c'],
                'humidity': median_features['humidity'],
                'wind_kph': median_features['wind_kph'],
                'pressure_mb': median_features['pressure_mb'],
                'precip_mm': median_features['precip_mm'],
                'cloud': median_features['cloud'],
                'wave_height': median_features['wave_height'],
                'is_severe': 1,
                'anomaly_score': 0.0
            })

        df_hist = pd.DataFrame(historical_data)
        df = pd.concat([df, df_hist], ignore_index=True)
        print("✅ Merged and imputed historical disaster data")

    except Exception as e:
        print(f"⚠️ Error loading disaster data: {e}")

    print("Class distribution:\n", df['is_severe'].value_counts())
    if df['is_severe'].nunique() < 2:
        print("❌ Not enough classes to train a classifier. Adding more negative samples.")
        synthetic_data = [{
            'location': 'synthetic_normal',
            'lat': -29.0,
            'lon': 30.0,
            'temp_c': 20.0,
            'humidity': 50.0,
            'wind_kph': 10.0,
            'pressure_mb': 1013.0,
            'precip_mm': 0.0,
            'cloud': 20.0,
            'wave_height': 0.0,
            'is_severe': 0,
            'anomaly_score': 0.0
        }]
        df = pd.concat([df, pd.DataFrame(synthetic_data)], ignore_index=True)

    model = train_model(df)
    df = generate_risk_scores(model, df)
    df.to_csv("weather_data_scored.csv", index=False)
    print("💾 Data with composite risk scores saved to 'weather_data_scored.csv'")

    visualize_data(df)
    visualize_map_with_scores(df)

if __name__ == "__main__":
    main()