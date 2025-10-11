"""
Crisis Connect - Modern Admin Dashboard
A beautiful, modern Streamlit interface for managing the Crisis Connect system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import base64
from typing import Dict, List, Optional
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Crisis Connect Admin",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/crisis-connect',
        'Report a bug': "https://github.com/your-repo/crisis-connect/issues",
        'About': "Crisis Connect Admin Dashboard v1.0"
    }
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1e40af;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-color: #1f2937;
        --light-color: #f8fafc;
        --border-color: #e5e7eb;
    }
    
    /* Global styles */
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .header p {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: var(--dark-color);
        margin: 0;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        color: #6b7280;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-healthy {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Alert styling */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background-color: #eff6ff;
        border-color: var(--primary-color);
        color: #1e40af;
    }
    
    .alert-warning {
        background-color: #fffbeb;
        border-color: var(--warning-color);
        color: #92400e;
    }
    
    .alert-danger {
        background-color: #fef2f2;
        border-color: var(--danger-color);
        color: #991b1b;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2rem;
        }
        
    .metric-value {
            font-size: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
CACHE_DURATION = 300  # 5 minutes

# Helper functions
@st.cache_data(ttl=CACHE_DURATION)
def fetch_api_data(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Fetch data from the Crisis Connect API with caching"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from {endpoint}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_DURATION)
def fetch_health_status() -> Dict:
    """Fetch system health status"""
    return fetch_api_data("/health") or {"status": "unknown", "services": {}}

@st.cache_data(ttl=CACHE_DURATION)
def fetch_metrics() -> Dict:
    """Fetch system metrics"""
    return fetch_api_data("/metrics") or {}

@st.cache_data(ttl=CACHE_DURATION)
def fetch_alerts(limit: int = 100) -> List[Dict]:
    """Fetch recent alerts"""
    data = fetch_api_data("/alerts/history", {"limit": limit})
    return data.get("alerts", []) if data else []

@st.cache_data(ttl=CACHE_DURATION)
def fetch_risk_assessments() -> List[Dict]:
    """Fetch latest risk assessments"""
    data = fetch_api_data("/risk-assessment")
    return data.get("predictions", []) if data else []

def get_status_color(status: str) -> str:
    """Get color for status indicators"""
    status_colors = {
        "healthy": "status-healthy",
        "degraded": "status-warning", 
        "unhealthy": "status-error",
        "unknown": "status-warning"
    }
    return status_colors.get(status.lower(), "status-warning")

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def create_metric_card(title: str, value: str, change: Optional[str] = None, change_type: str = "neutral"):
    """Create a styled metric card"""
    change_html = ""
    if change:
        change_color = "#10b981" if change_type == "positive" else "#ef4444" if change_type == "negative" else "#6b7280"
        change_html = f'<p style="color: {change_color}; font-size: 0.875rem; margin: 0.25rem 0 0 0;">{change}</p>'
    
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{title}</p>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🚨 Crisis Connect Admin</h1>
        <p>Real-time flood risk monitoring and alert management system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")
        
        page = st.selectbox(
            "Select Page",
            ["📊 Dashboard", "🚨 Alerts", "🌤️ Weather Data", "📈 Analytics", "⚙️ Settings"],
            key="page_selector"
        )
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🔍 System Status")
        health = fetch_health_status()
        
        if health:
            status_class = get_status_color(health.get("status", "unknown"))
            st.markdown(f"""
            <div class="status-indicator {status_class}">
                {health.get("status", "unknown").upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Service status
            services = health.get("services", {})
            for service, info in services.items():
                service_status = info.get("status", "unknown")
                service_class = get_status_color(service_status)
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <span style="font-weight: 500;">{service.title()}:</span>
                    <span class="status-indicator {service_class}" style="margin-left: 0.5rem;">
                        {service_status.upper()}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
        if st.button("🚨 Generate Alerts", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/alerts/generate", timeout=30)
                if response.status_code == 200:
                    st.success("Alerts generated successfully!")
                else:
                    st.error("Failed to generate alerts")
            except:
                st.error("Could not connect to API")
        
        if st.button("📊 Collect Weather Data", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE_URL}/collect", timeout=60)
                if response.status_code == 200:
                    st.success("Weather data collected!")
                else:
                    st.error("Failed to collect weather data")
            except:
                st.error("Could not connect to API")
    
    # Main content based on selected page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🚨 Alerts":
        show_alerts_page()
    elif page == "🌤️ Weather Data":
        show_weather_page()
    elif page == "📈 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_dashboard():
    """Show main dashboard overview"""
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metrics = fetch_metrics()
        total_alerts = metrics.get("total_alerts", 0)
        create_metric_card("Total Alerts", str(total_alerts))
    
    with col2:
        recent_alerts = metrics.get("recent_alerts_24h", 0)
        create_metric_card("Recent Alerts (24h)", str(recent_alerts))
    
    with col3:
        high_risk = metrics.get("high_risk_locations", 0)
        create_metric_card("High Risk Locations", str(high_risk))
    
    with col4:
        system_uptime = metrics.get("system_uptime", "Unknown")
        create_metric_card("System Uptime", system_uptime)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Risk Level Distribution")
        
        risk_data = fetch_risk_assessments()
        if risk_data:
            risk_counts = {}
            for prediction in risk_data:
                category = prediction.get("risk_category", "Unknown")
                risk_counts[category] = risk_counts.get(category, 0) + 1
            
            if risk_counts:
                fig = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk assessment data available")
        else:
            st.info("No risk assessment data available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📈 Alert Trends (Last 7 Days)")
        
        alerts = fetch_alerts(limit=1000)
        if alerts:
            # Convert to DataFrame and group by date
            df = pd.DataFrame(alerts)
            if not df.empty and 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily_counts = df.groupby('date').size().reset_index(name='count')
                
                fig = px.line(
                    daily_counts,
                    x='date',
                    y='count',
                    title="Daily Alert Count",
                    color_discrete_sequence=[px.colors.qualitative.Set1[0]]
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Number of Alerts",
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No alert trend data available")
        else:
            st.info("No alert data available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent alerts table
    st.markdown("### 🚨 Recent Alerts")
    
    alerts = fetch_alerts(limit=20)
    if alerts:
        df = pd.DataFrame(alerts)
        if not df.empty:
            # Select and rename columns for display
            display_columns = ['location', 'risk_level', 'message', 'language', 'timestamp']
            available_columns = [col for col in display_columns if col in df.columns]
            
            if available_columns:
                display_df = df[available_columns].copy()
                
                # Format timestamp
                if 'timestamp' in display_df.columns:
                    display_df['timestamp'] = display_df['timestamp'].apply(format_timestamp)
                
                # Rename columns for better display
                column_mapping = {
                    'location': 'Location',
                    'risk_level': 'Risk Level',
                    'message': 'Message',
                    'language': 'Language',
                    'timestamp': 'Timestamp'
                }
                display_df = display_df.rename(columns=column_mapping)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No alert data available")
        else:
            st.info("No alerts found")
    else:
        st.info("No alerts available")

def show_alerts_page():
    """Show alerts management page"""
    st.markdown("### 🚨 Alert Management")
    
    # Alert filters
    col1, col2, col3 = st.columns(3)
    
        with col1:
        location_filter = st.selectbox("Filter by Location", ["All"] + ["Durban", "Cape Town", "Johannesburg"])
    
        with col2:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "LOW", "MODERATE", "HIGH", "CRITICAL"])
    
    with col3:
        language_filter = st.selectbox("Filter by Language", ["All", "en", "zu", "xh"])
    
    # Fetch and filter alerts
    alerts = fetch_alerts(limit=500)
    
    if alerts:
        df = pd.DataFrame(alerts)
        
        # Apply filters
        if location_filter != "All":
            df = df[df['location'].str.contains(location_filter, case=False, na=False)]
        
        if risk_filter != "All":
            df = df[df['risk_level'] == risk_filter]
        
        if language_filter != "All":
            df = df[df['language'] == language_filter]
        
        # Display filtered alerts
        if not df.empty:
            st.markdown(f"**Found {len(df)} alerts matching your criteria**")
            
            # Alert statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Alerts", len(df))
            
            with col2:
                high_risk = len(df[df['risk_level'] == 'HIGH'])
                st.metric("High Risk", high_risk)
            
            with col3:
                recent = len(df[pd.to_datetime(df['timestamp']) > datetime.now() - timedelta(hours=24)])
                st.metric("Last 24h", recent)
            
            with col4:
                languages = df['language'].nunique()
                st.metric("Languages", languages)
            
            # Display alerts table
        st.dataframe(
                df[['location', 'risk_level', 'message', 'language', 'timestamp']],
            use_container_width=True,
                hide_index=True
        )
    else:
            st.info("No alerts match your filter criteria")
    else:
        st.info("No alerts available")

def show_weather_page():
    """Show weather data page"""
    st.markdown("### 🌤️ Weather Data")
    
    # Weather data collection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Weather Data Collection")
        
        if st.button("🌤️ Collect Latest Weather Data", use_container_width=True):
            with st.spinner("Collecting weather data..."):
                try:
                    response = requests.get(f"{API_BASE_URL}/collect", timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Collected data for {data.get('count', 0)} locations")
                    else:
                        st.error("❌ Failed to collect weather data")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("#### Custom Location")
        
        lat = st.number_input("Latitude", value=-29.8587, format="%.4f")
        lon = st.number_input("Longitude", value=31.0218, format="%.4f")
        
        if st.button("📍 Collect for Location", use_container_width=True):
            with st.spinner("Collecting data..."):
                try:
                    payload = {"locations": [{"name": "Custom", "lat": lat, "lon": lon}]}
                    response = requests.post(f"{API_BASE_URL}/collect", json=payload, timeout=30)
                    if response.status_code == 200:
                        st.success("✅ Data collected for custom location")
                    else:
                        st.error("❌ Failed to collect data")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Weather data visualization
    st.markdown("#### 📊 Weather Data Overview")
    
    # This would typically show weather data from the API
    st.info("Weather data visualization would be displayed here based on collected data")

def show_analytics_page():
    """Show analytics page"""
    st.markdown("### 📈 Analytics & Reports")
    
    # Analytics options
        col1, col2 = st.columns(2)
    
        with col1:
        st.markdown("#### Alert Analytics")
        
        alerts = fetch_alerts(limit=1000)
        if alerts:
            df = pd.DataFrame(alerts)
            
            # Risk level distribution
            if 'risk_level' in df.columns:
                risk_counts = df['risk_level'].value_counts()
                
                fig = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title="Alert Distribution by Risk Level",
                    color=risk_counts.values,
                    color_continuous_scale="RdYlBu_r"
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Risk Level",
                    yaxis_title="Number of Alerts",
                    font=dict(family="Inter", size=12)
                )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Location Analytics")
        
        if alerts:
            if 'location' in df.columns:
                location_counts = df['location'].value_counts().head(10)
        
        fig = px.pie(
                    values=location_counts.values,
                    names=location_counts.index,
                    title="Top 10 Locations by Alert Count"
                )
                fig.update_layout(
                    height=400,
                    font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

    # System performance metrics
    st.markdown("#### 🔧 System Performance")
    
    metrics = fetch_metrics()
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Response Time", f"{metrics.get('avg_response_time', 0):.2f}ms")
        
        with col2:
            st.metric("Database Connections", metrics.get('db_connections', 0))
        
        with col3:
            st.metric("Cache Hit Rate", f"{metrics.get('cache_hit_rate', 0):.1f}%")

def show_settings_page():
    """Show settings page"""
    st.markdown("### ⚙️ System Settings")
    
    # API Configuration
    st.markdown("#### 🔗 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        st.info("Update the API URL if your backend is running on a different port")
    
    with col2:
        cache_duration = st.slider("Cache Duration (seconds)", 60, 600, CACHE_DURATION)
        st.info("How long to cache API responses")
    
    # System Information
    st.markdown("#### ℹ️ System Information")
    
    health = fetch_health_status()
    if health:
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(health)
        
        with col2:
            st.markdown("""
            #### 📋 Quick Actions
            
            - **Clear Cache**: Refresh all cached data
            - **Test Connection**: Verify API connectivity
            - **Export Data**: Download system data
            - **Reset Settings**: Restore default configuration
            """)
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
            
            if st.button("🔍 Test Connection", use_container_width=True):
                try:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Connection successful!")
    else:
                        st.warning("⚠️ Connection issues detected")
                except:
                    st.error("❌ Connection failed")

if __name__ == "__main__":
    main()
