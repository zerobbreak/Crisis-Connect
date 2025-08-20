import streamlit as st
import requests
import pandas as pd
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
from datetime import datetime
import branca.colormap as cm
import matplotlib.pyplot as plt

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# Streamlit page configuration
st.set_page_config(page_title="Crisis Connect Dashboard", layout="wide")

# Title and description
st.title("Crisis Connect Flood Risk Dashboard")
st.markdown("Visualize flood risk predictions, alerts, and household resource needs for South African locations.")

# Function to fetch data from API
def fetch_api_data(endpoint, params=None):
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching data from {endpoint}: {str(e)}")
        return None

def fetch_api_post_data(endpoint, json_data):
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=json_data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching data from {endpoint}: {str(e)}")
        return None

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Heatmap", "Recent Alerts", "Resource Calculator", 
    "Risk Predictions", "Historical Data", "Summary", "Alert Stats"
])

# --- TAB 1: Heatmap ---
with tab1:
    st.header("Flood Risk Heatmap")
    risk_data = fetch_api_data("/risk-assessment")
    if risk_data:
        m = folium.Map(location=[-30.5595, 22.9375], zoom_start=5, tiles="CartoDB positron")
        risk_locations = []
        icon_colors = {"Low": "green", "Medium": "orange", "High": "red"}

        for record in risk_data:
            if 'lat' in record and 'lon' in record and 'composite_risk_score' in record:
                risk_locations.append([record['lat'], record['lon'], record['composite_risk_score']/100])
                folium.Marker(
                    location=[record['lat'], record['lon']],
                    popup=f"<b>{record['location']}</b><br>Risk: {record['risk_category']}<br>Score: {record['composite_risk_score']:.1f}%",
                    icon=folium.Icon(color=icon_colors.get(record['risk_category'], "blue"), icon="exclamation-triangle", prefix="fa")
                ).add_to(m)

        HeatMap(risk_locations, radius=15, min_opacity=0.5, blur=10).add_to(m)
        colormap = cm.LinearColormap(['blue', 'lime', 'yellow', 'orange', 'red'], vmin=0, vmax=100, caption="Flood Risk (%)")
        colormap.add_to(m)
        folium_static(m)
    else:
        st.warning("No risk data available. Run the backend to generate risk assessments.")

# --- TAB 2: Recent Alerts ---
with tab2:
    st.header("Recent Flood Alerts")
    alerts_data = fetch_api_data("/alerts/history", params={"limit": 20})
    if alerts_data and alerts_data.get("alerts"):
        alerts_df = pd.DataFrame(alerts_data["alerts"])
        st.dataframe(alerts_df, hide_index=True)
    else:
        st.info("No recent alerts available.")

# --- TAB 3: Resource Calculator ---
with tab3:
    st.header("Household Resource Calculator")
    with st.form("resource_form"):
        col1, col2 = st.columns(2)
        with col1:
            place_name = st.text_input("Location (e.g., eThekwini (Durban))", "")
            household_size = st.number_input("Household Size", min_value=1, value=4, step=1)
        with col2:
            lat = st.number_input("Latitude (optional)", value=0.0, format="%.6f")
            lon = st.number_input("Longitude (optional)", value=0.0, format="%.6f")
        submit_button = st.form_submit_button("Calculate Resources")

        if submit_button:
            payload = {"place_name": place_name if place_name else None, "lat": lat, "lon": lon, "household_size": household_size}
            result = fetch_api_post_data("/resources", payload)
            if result:
                st.subheader(f"Resources for {result['location']} (Risk: {result['risk_category']})")
                resources = result["resources"]
                st.write(
                    f"- **Food Packs**: {resources['food_packs']} (for {household_size} people)\n"
                    f"- **Water Gallons**: {resources['water_gallons']}\n"
                    f"- **Shelter Needed**: {'Yes' if resources['shelter_needed'] else 'No'}\n"
                    f"- **Boats Needed**: {resources['boats_needed']}\n"
                    f"- **Timestamp**: {result['timestamp']}"
                )

# --- TAB 4: Risk Predictions ---
with tab4:
    st.header("Risk Predictions for All Locations")
    risk_data = fetch_api_data("/risk-assessment")
    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        # Filtering option
        risk_filter = st.multiselect("Filter by Risk Category", risk_df['risk_category'].unique())
        if risk_filter:
            risk_df = risk_df[risk_df['risk_category'].isin(risk_filter)]
        st.dataframe(risk_df, hide_index=True)
    else:
        st.info("No risk predictions available.")

# --- TAB 5: Historical Data ---
with tab5:
    st.header("Historical Disaster Statistics")
    historical_data = fetch_api_data("/api/historical")
    if historical_data:
        hist_df = pd.DataFrame(historical_data)
        total_events = len(hist_df)
        total_deaths = hist_df.get('total_deaths', pd.Series()).sum()

        col1, col2 = st.columns(2)
        col1.metric("Total Events", total_events)
        col2.metric("Total Deaths", int(total_deaths))

        if 'severity' in hist_df.columns:
            severity_counts = hist_df['severity'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(severity_counts.index, severity_counts.values)
            ax.set_title("Severity Distribution")
            st.pyplot(fig)
    else:
        st.info("No historical disaster data available.")

# --- TAB 6: Summary ---
with tab6:
    st.header("Risk Assessment Summary")
    risk_data = fetch_api_data("/risk-assessment")
    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        if 'risk_category' in risk_df.columns:
            risk_table = risk_df['risk_category'].value_counts().reset_index()
            risk_table.columns = ['Risk Category', 'Count']
            st.table(risk_table)

        if 'location' in risk_df.columns and 'composite_risk_score' in risk_df.columns:
            top_risks = risk_df.sort_values("composite_risk_score", ascending=False).head(5)
            st.subheader("Top 5 Highest Risk Locations")
            st.table(top_risks[['location', 'composite_risk_score', 'risk_category']])
    else:
        st.info("No risk assessment data available.")

# --- TAB 7: Alert Stats ---
with tab7:
    st.header("Alert Statistics")
    alerts_data = fetch_api_data("/alerts/history", params={"limit": 100})
    if alerts_data and alerts_data.get("alerts"):
        alerts_df = pd.DataFrame(alerts_data["alerts"])
        total_alerts = len(alerts_df)
        st.metric("Total Alerts Issued", total_alerts)

        if 'risk_level' in alerts_df.columns:
            risk_level_counts = alerts_df['risk_level'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(risk_level_counts, labels=risk_level_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title("Alert Risk Level Distribution")
            st.pyplot(fig)

            st.subheader("Alerts by Location")
            location_table = alerts_df['location'].value_counts().head(10).reset_index()
            location_table.columns = ['Location', 'Number of Alerts']
            st.table(location_table)
    else:
        st.info("No alert data available.")

# Footer
st.markdown("---")
st.markdown("Powered by Crisis Connect API | Last updated: August 19, 2025")
