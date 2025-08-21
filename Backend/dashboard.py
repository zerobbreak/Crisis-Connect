# dashboard.py - Crisis Connect Dashboard
import streamlit as st
import requests
import pandas as pd
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
from datetime import datetime
import branca.colormap as cm
import plotly.express as px
import time

# --- Configuration ---
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; }
    .stButton button { background-color: #007BFF; color: white; border: none; }
    .stButton button:hover { background-color: #0056b3; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #007BFF;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.title("🌊 Crisis Connect Dashviard")
st.markdown("Early natural disaster warning system")

# --- Data Fetching with Caching ---
@st.cache_data(ttl=300)
def fetch_api_data(endpoint, params=None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"📡 Failed to reach API: `{str(e)}`")
        st.info("💡 Ensure `main.py` is running at `uvicorn main:app --reload`")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_api_post_data(endpoint, json_data):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.post(url, json=json_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"📡 Failed to submit to API: `{str(e)}`")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

# --- Health Check ---
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                st.success("✅ API & Model: Healthy")
                return True
            else:
                st.warning("⚠️ API OK, but model not loaded. Run `python predict.py`")
                return False
        else:
            st.error("🔴 API unreachable")
            return False
    except:
        st.error("🔴 API unreachable. Is the backend running?")
        return False

with st.expander("🔧 System Status", expanded=False):
    if "health_checked" not in st.session_state:
        st.session_state.health_checked = check_api_health()

# --- Sidebar: Simulate Disaster ---
st.sidebar.title("🧨 Simulate Disaster")
simulate_location = st.sidebar.text_input("📍 Location (e.g., eThekwini, Cape Town)")
scenario = st.sidebar.selectbox(
    "Choose Scenario",
    ["flood", "storm", "coastal_cyclone", "flash_flood"]
)
household_size_sim = st.sidebar.number_input("👥 Household Size", 1, 20, 4)

if st.sidebar.button("🧨 Run Simulation"):
    if not simulate_location.strip():
        st.sidebar.error("Please enter a location")
    else:
        with st.spinner("Simulating disaster..."):
            resp = fetch_api_post_data("/simulate", {
                "location": simulate_location,
                "scenario": scenario,
                "household_size": household_size_sim
            })
            if resp:
                st.success(f"🔥 {resp['message']}")
                with st.expander("📊 Simulation Details"):
                    st.json(resp)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📍 Heatmap", "🚨 Alerts", "📦 Resources", "📊 Predictions", 
    "📈 History", "📋 Summary", "📊 Alert Stats"
])

# --- TAB 1: Heatmap ---
with tab1:
    st.subheader("Flood Risk Heatmap")
    
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🔄", help="Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner("Loading risk data..."):
        risk_data = fetch_api_data("/risk-assessment")
    
    if risk_data and len(risk_data) > 0:
        m = folium.Map(location=[-30.5595, 22.9375], zoom_start=5, tiles="CartoDB positron")
        heat_data = []
        icon_colors = {"Low": "green", "Medium": "orange", "High": "red"}

        for record in risk_data:
            lat = record.get('lat')
            lon = record.get('lon')
            score = record.get('composite_risk_score', 0)
            if lat and lon and score is not None:
                heat_data.append([lat, lon, score / 100])
                popup = folium.Popup(
                    f"<b>{record['location']}</b><br>"
                    f"Risk: <b>{record.get('risk_category', 'N/A')}</b><br>"
                    f"Score: <b>{score:.1f}%</b><br>"
                    f"🌧️ Rain (24h): {record.get('precip_mm', 'N/A')} mm<br>"
                    f"🌊 Wave: {record.get('wave_height', 'N/A')} m<br>"
                    f"💨 Wind: {record.get('wind_kph', 'N/A')} km/h",
                    max_width=300
                )
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=7,
                    popup=popup,
                    color=icon_colors.get(record.get('risk_category'), "blue"),
                    fill=True,
                    fill_opacity=0.8
                ).add_to(m)

        HeatMap(heat_data, radius=20, blur=15, max_opacity=0.8).add_to(m)

        colormap = cm.LinearColormap(
            colors=['#00FF00', '#FFFF00', '#FFA500', '#FF4500', '#8B0000'],
            vmin=0, vmax=100, caption="Flood Risk Score (%)"
        )
        colormap.add_to(m)

        folium_static(m, width=1200, height=600)
        
        risk_df = pd.DataFrame(risk_data)
        high_risk = len(risk_df[risk_df['risk_category'] == 'High'])
        st.markdown(f"🔍 <div class='metric-card'><div class='metric-label'>High-Risk Areas</div><div class='metric-value'>{high_risk}</div></div>", unsafe_allow_html=True)
    else:
        st.warning("No risk data available. Try running a simulation.")

# --- TAB 2: Alerts ---
with tab2:
    st.subheader("Recent Flood Alerts")
    
    level_filter = st.selectbox("Filter by Risk Level", ["All", "HIGH", "MODERATE", "LOW"], index=0)
    params = {"limit": 50}
    if level_filter != "All":
        params["level"] = level_filter

    with st.spinner("Fetching alerts..."):
        alerts_data = fetch_api_data("/alerts/history", params=params)
    
    if alerts_data and alerts_data.get("alerts"):
        alerts_df = pd.DataFrame(alerts_data["alerts"])
        if 'timestamp' in alerts_df.columns:
            # Handle ISO 8601 timestamps like "2025-08-18T16:23:00"
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'], format='mixed').dt.strftime('%Y-%m-%d %H:%M')       
        def highlight_risk(row):
            color = "red" if row['risk_level'] == 'HIGH' else "orange" if row['risk_level'] == 'MODERATE' else "green"
            return [f'color: {color}; font-weight: bold'] * len(row)
        
        styled_df = alerts_df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No alerts to display.")

    # Generate Alerts Button
    if st.button("⚡ Generate Alerts from Predictions"):
        result = fetch_api_post_data("/alerts/generate", {})
        if result:
            st.success(f"✅ Generated {result['generated']} alerts")
            st.json(result["alerts"][:5])  # Show first 5

# --- TAB 3: Resource Calculator ---
with tab3:
    st.subheader("Household Resource Calculator")
    
    with st.form("resource_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            place_name = st.text_input("📍 Location (e.g., eThekwini, Cape Town)", "").strip()
        with col2:
            household_size = st.number_input("👥 Household Size", min_value=1, max_value=50, value=4)

        st.caption("Or enter coordinates manually:")
        lat = st.number_input("🗺️ Latitude", value=0.0, format="%.6f", step=0.000001)
        lon = st.number_input("🌍 Longitude", value=0.0, format="%.6f", step=0.000001)

        submitted = st.form_submit_button("🧮 Calculate Resources")

    if submitted:
        if not place_name and (abs(lat) < 0.0001 or abs(lon) < 0.0001):
            st.warning("Please enter a location name or valid coordinates.")
            st.stop()

        with st.spinner("Calculating required resources..."):
            payload = {
                "place_name": place_name or None,
                "lat": lat if abs(lat) > 0.0001 else None,
                "lon": lon if abs(lon) > 0.0001 else None,
                "household_size": household_size
            }
            result = fetch_api_post_data("/resources", payload)
        
        if result:
            st.success(f"✅ Resources for {result['location']} ({result['risk_category']} risk)")
            
            res = result["resources"]
            cols = st.columns(5)
            cols[0].metric("🍽️ Food Packs", res["food_packs"])
            cols[1].metric("💧 Water (Gallons)", res["water_gallons"])
            cols[2].metric("🏠 Shelter", "Yes" if res["shelter_needed"] else "No")
            cols[3].metric("🚤 Boats Needed", res["boats_needed"])
            cols[4].metric("⏱️ Prep Time", "2–6 hrs" if res["shelter_needed"] else "1–2 hrs")

            with st.expander("🔍 View Risk Details"):
                st.write(f"**Anomaly Score**: {result.get('anomaly_score', 'N/A')}%")
                st.write(f"**Precipitation (24h)**: {result.get('precip_mm', 'N/A')} mm")
                st.write(f"**Wind Speed**: {result.get('wind_kph', 'N/A')} km/h")
                st.write(f"**Wave Height**: {result.get('wave_height', 'N/A')} m")
                st.write(f"**Model Risk Score**: {result.get('model_risk_score', 'N/A'):.1f}%")

            st.caption(f"⏱️ Generated at: {result['timestamp']}")
        else:
            st.error("Could not calculate resources. Check inputs and backend status.")

# --- TAB 4: Predictions ---
with tab4:
    st.subheader("All Risk Predictions")
    with st.spinner("Loading predictions..."):
        risk_data = fetch_api_data("/risk-assessment")
    
    if risk_data:
        df = pd.DataFrame(risk_data)
        df = df.sort_values("composite_risk_score", ascending=False)

        filters = st.multiselect(
            "Filter by Risk Category",
            options=df['risk_category'].unique(),
            default=df['risk_category'].unique()
        )
        filtered_df = df[df['risk_category'].isin(filters)]

        search = st.text_input("🔍 Search Location", "")
        if search:
            filtered_df = filtered_df[filtered_df['location'].str.contains(search, case=False, na=False)]

        cols_to_show = ['location', 'risk_category', 'composite_risk_score', 'precip_mm', 'wind_kph', 'wave_height']
        cols_to_show = [c for c in cols_to_show if c in filtered_df.columns]

        st.dataframe(
            filtered_df[cols_to_show],
            use_container_width=True,
            hide_index=True,
            column_config={
                "composite_risk_score": st.column_config.NumberColumn(format="%.1f%%"),
                "precip_mm": st.column_config.NumberColumn(format="%.1f mm"),
                "wind_kph": st.column_config.NumberColumn(format="%.1f km/h"),
                "wave_height": st.column_config.NumberColumn(format="%.1f m"),
            }
        )
    else:
        st.info("No predictions available.")

# --- TAB 5: History ---
with tab5:
    st.subheader("Historical Disaster Data")
    with st.spinner("Loading historical data..."):
        hist_data = fetch_api_data("/api/historical")
    
    if hist_data:
        df = pd.DataFrame(hist_data)
        total = len(df)
        deaths = pd.to_numeric(df.get('total_deaths', pd.Series()), errors='coerce').sum()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-label">Total Events</div><div class="metric-value">{}</div></div>'.format(total), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-label">Fatalities</div><div class="metric-value">{}</div></div>'.format(int(deaths) if not pd.isna(deaths) else 0), unsafe_allow_html=True)

        if 'severity' in df.columns:
            fig = px.pie(df, names='severity', title="Event Severity Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            yearly = df.groupby(df['event_date'].dt.year).size()
            fig2 = px.line(yearly, x=yearly.index, y=yearly.values, title="Disaster Events Over Time")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No historical data available.")

# --- TAB 6: Summary ---
with tab6:
    st.subheader("Risk Summary Overview")
    risk_data = fetch_api_data("/risk-assessment")
    
    if risk_data:
        df = pd.DataFrame(risk_data)
        cat_counts = df['risk_category'].value_counts()
        
        fig = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title="Risk Category Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)

        top5 = df.nlargest(5, 'composite_risk_score')[['location', 'composite_risk_score', 'precip_mm', 'risk_category']]
        st.markdown("### 🔺 Top 5 High-Risk Locations")
        st.dataframe(top5, use_container_width=True)
    else:
        st.info("No data to summarize.")

# --- TAB 7: Alert Stats ---
with tab7:
    st.subheader("Alert Statistics")
    alerts_data = fetch_api_data("/alerts/history", params={"limit": 100})
    
    if alerts_data and alerts_data.get("alerts"):
        df = pd.DataFrame(alerts_data["alerts"])
        st.metric("Total Alerts Issued", len(df))

        if 'risk_level' in df.columns:
            fig = px.pie(df, names='risk_level', title="Alert Risk Levels", hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

        top_locs = df['location'].value_counts().head(10).reset_index()
        top_locs.columns = ['Location', 'Alert Count']
        fig2 = px.bar(top_locs, x='Alert Count', y='Location', orientation='h', title="Top 10 Alerted Locations")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No alert history available.")

# --- Footer ---
st.markdown("---")
st.markdown(
    f"<div class='footer'>🌊 Powered by Crisis Connect | 🕒 Last updated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</div>",
    unsafe_allow_html=True
)