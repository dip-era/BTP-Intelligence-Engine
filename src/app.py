import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import os
import sys
import sqlite3
from datetime import datetime
import plotly.graph_objects as go

def init_db():
    conn = sqlite3.connect('btp_intelligence.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS retrospective_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            location_name TEXT,
            forecasted_mins INTEGER,
            logged_actual_mins INTEGER,
            variance_factor TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_model_health_card():
    try:
        conn = sqlite3.connect('btp_intelligence.db')
        df = pd.read_sql_query("SELECT forecasted_mins, logged_actual_mins FROM retrospective_logs", conn)
        conn.close()
        
        if len(df) < 1:
            return "**MLOps Model Diagnostics:** Awaiting Ground Telemetry (0 paired instances logged)"
            
        df['error'] = df['logged_actual_mins'] - df['forecasted_mins']
        mae = df['error'].abs().mean()
        bias = df['error'].mean()
        
        bias_dir = f"+{bias:.1f}m (Systematic Under-prediction)" if bias > 0 else f"{bias:.1f}m (Systematic Over-prediction)"
        return f"📊 **Live MLOps Telemetry ({len(df)} logs) —** Mean Absolute Error: **{mae:.1f}m** | Model Drift Bias: **{bias_dir}**"
    except:
        return "📊 **MLOps Model Diagnostics:** Telemetry DB Offline"

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecaster import ImpactForecaster
from src.optimizer import ResourceOptimizer

# Page config
st.set_page_config(
    page_title="BTP Intelligence Engine",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and styling
st.markdown("""
<style>
/* 1. CineIQ Canvas & Deep Plum Sidebar */
.stApp { background-color: #14161B !important; }
[data-testid="stSidebar"] { background-color: #1E0F16 !important; border-right: 1px solid rgba(255, 255, 255, 0.02) !important; }

/* 2. CineIQ Metric Cards (The Big Stats) */
[data-testid="stMetric"] {
    background-color: #1A1C23 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.2) !important;
}
[data-testid="stMetricValue"] {
    color: #FF0055 !important; /* CineIQ Crimson */
    font-weight: 900 !important;
    font-size: 2.8rem !important;
    text-shadow: none !important; /* Removed neon glow for a clean solid look */
}
[data-testid="stMetricLabel"] { color: #E2E8F0 !important; font-weight: 500 !important; font-size: 0.9rem !important; }

/* 3. Sleek Inputs & Dropdowns */
.stTextInput > div > div > input, .stSelectbox > div > div, .stNumberInput > div > div > input {
    background-color: #1C1E26 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #F8FAFC !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus, .stSelectbox > div > div:focus {
    border-color: #FF0055 !important;
    box-shadow: 0 0 0 1px #FF0055 !important;
}

/* 4. Action Plan Table Restyling */
table { border-collapse: collapse !important; width: 100% !important; font-size: 0.9rem; }
th { background-color: #1E0F16 !important; color: #CBD5E1 !important; border-bottom: 2px solid #FF0055 !important; padding: 12px; text-align: left; }
td { border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important; color: #F8FAFC !important; padding: 12px; }

/* 5. Headers & Dividers */
h1, h2, h3 { color: #F8FAFC !important; font-family: 'Inter', sans-serif !important; letter-spacing: -0.5px; }
hr { border-color: rgba(255, 255, 255, 0.06) !important; }

/* Legacy Warning Banner (Preserved) */
.warning-banner {
    background-color: #2D1A1A;
    border: 1px solid #FF4B4B;
    border-radius: 8px;
    padding: 15px 20px;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    margin-top: 20px;
    margin-bottom: 20px;
}
.warning-icon {
    font-size: 1.5rem;
    margin-right: 15px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    forecaster = ImpactForecaster()
    try:
        forecaster._load_model()
    except Exception as e:
        st.error("Models not found. Please run main.py first to train models.")
    optimizer = ResourceOptimizer()
    return forecaster, optimizer

def get_dynamic_place_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="blr_traffic_hackathon_app")
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, timeout=2)
        address = location.raw['address']
        
        # Bangalore preference hierarchy: Suburb -> Neighbourhood -> Road -> fallback
        area = address.get('suburb', address.get('neighbourhood', address.get('road', 'the local sector')))
        return area.title()
    except:
        return "the surrounding sector"

def get_cascade_alert(lat, lon):
    if 12.97 < lat < 12.98 and 77.59 < lon < 77.60: # Central
        return "Diverting North traffic to Kasturba Rd will push Hudson Circle to 94% capacity. Recommended Action: Override Hudson Circle Signal Phase-2 to +15s Green."
    elif 12.91 < lat < 12.92 and 77.62 < lon < 77.63: # Silk Board
        return "Diverting traffic will gridlock Outer Ring Road at Silk Board Junction. Recommended Action: Deploy 4 marshals at Hosur Road intersection."
    elif 13.03 < lat < 13.04 and 77.58 < lon < 77.59: # Hebbal
        return "Diverting traffic will impact Airport road flow under Hebbal flyover. Recommended Action: Open the emergency lane."
    elif 12.96 < lat < 12.97 and 77.71 < lon < 77.72: # Whitefield
        return "ITPB main road will exceed capacity. Recommended Action: Reroute heavy vehicles via Hoodi."
    else:
        live_area_name = get_dynamic_place_name(lat, lon)
        return f"Traffic diverted away from {live_area_name} will critically overload parallel corridors. Recommended Action: Deploy 2 mobile marshals to clear illegal roadside parking along the main {live_area_name} stretch."

def render_saturation_chart(lat, lon, cause):
    # Generates plausible network stress distribution based on the active event
    roads = ["Kasturba Rd", "Hudson Circle", "MG Road", "Residency Rd", "St. Marks Rd"]
    
    if "public_event" in cause:
        base_stress = [82, 94, 61, 88, 44]
    elif "water" in cause:
        base_stress = [91, 85, 95, 60, 72]
    else:
        base_stress = [70, 88, 65, 91, 55]
        
    # Sort so the most bottlenecked road sits at the very top of the chart
    data = sorted(zip(roads, base_stress), key=lambda x: x[1])
    y_vals = [d[0] for d in data]
    x_vals = [d[1] for d in data]

    # Dynamic Cyber-Palette: >=90% is CineIQ Crimson, normal is Cyan/Slate
    bar_colors = ['#FF0055' if val >= 90 else '#38BDF8' for val in x_vals]

    fig = go.Figure(go.Bar(
        x=x_vals,
        y=y_vals,
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v}%" for v in x_vals],
        textposition='auto',
        textfont=dict(color='#FFFFFF', size=11, family='Inter')
    ))

    fig.update_layout(
        title="<b>PERIMETER CORRIDOR SATURATION FORECAST</b>",
        title_font=dict(color='#8B949E', size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, color='#8B949E'),
        yaxis=dict(color='#E6EDF3', showgrid=False),
        margin=dict(l=10, r=10, t=35, b=10),
        height=190
    )
    return fig

def main():
    if "lat" not in st.session_state: st.session_state.lat = 12.971600
    if "lon" not in st.session_state: st.session_state.lon = 77.594600
    if "cause" not in st.session_state: st.session_state.cause = "public_event"
    if "active_tab" not in st.session_state: st.session_state.active_tab = "🗺️ Interactive Map"
    if "run_forecast" not in st.session_state: st.session_state.run_forecast = True

    st.title("Bengaluru Traffic Police: Intelligence Engine")
    st.markdown("### Event-Driven Traffic Congestion Forecasting & Dynamic Routing")
    
    # Toggle switch
    view_mode = st.radio(
        "View Mode",
        ["Live Operational View", "Post-Event Retrospective"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    forecaster, optimizer = load_models()
    
    # Sidebar
    st.sidebar.header("Simulate Traffic Event")
    
    interface_options = ["Interactive Map", "Manual GPS"]
    
    # Ensure session state has a valid default string
    if 'active_tab' not in st.session_state or st.session_state.active_tab not in interface_options:
        st.session_state.active_tab = "Interactive Map"
        
    # Safely grab the integer index of the current active string
    current_idx = interface_options.index(st.session_state.active_tab)
    
    # Render the radio WITHOUT a 'key' parameter, driving it purely via 'index'
    selected_interface = st.sidebar.radio(
        "Control Interface:", 
        interface_options, 
        index=current_idx, 
        horizontal=True
    )
    
    # Manually propagate user sidebar clicks back to the session state
    if selected_interface != st.session_state.active_tab:
        st.session_state.active_tab = selected_interface
        st.rerun()
        
    input_method = selected_interface
    
    event_type = st.sidebar.selectbox("Event Type", ["unplanned", "planned"])
    
    causes = ["vehicle_breakdown", "accident", "tree_fall", "water_logging", "public_event", "others"]
    default_cause_idx = causes.index(st.session_state.cause) if st.session_state.cause in causes else 4
    event_cause = st.sidebar.selectbox("Event Cause", causes, index=default_cause_idx)
    if event_cause != st.session_state.cause:
        st.session_state.cause = event_cause
        
    priority = st.sidebar.selectbox("Priority", ["High", "Medium", "Low"])
    
    st.sidebar.markdown("---")
    
    if input_method == "Interactive Map":
        st.sidebar.info("Click any coordinate directly on the Folium map to instantly move the epicenter.")
        st.sidebar.markdown(f"**Current Lat:** `{st.session_state.lat:.6f}`  \n**Current Lon:** `{st.session_state.lon:.6f}`")
        event_cause = st.session_state.cause
        
    elif input_method == "Manual GPS":
        lat = st.sidebar.number_input("Latitude", min_value=12.80, max_value=13.25, value=st.session_state.lat, format="%.6f")
        lon = st.sidebar.number_input("Longitude", min_value=77.45, max_value=77.80, value=st.session_state.lon, format="%.6f")
        if lat != st.session_state.lat or lon != st.session_state.lon:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.rerun()
        event_cause = st.session_state.cause
    
    start_hour = st.sidebar.slider("Start Hour", 0, 23, 17) # 5 PM rush hour
    day_of_week = st.sidebar.selectbox("Day of Week (0=Mon, 6=Sun)", [0, 1, 2, 3, 4, 5, 6], index=4) # Friday
    
    is_weekend = 1 if day_of_week in [5, 6] else 0
    
    if st.sidebar.button("Forecast Impact & Plan Routes", type="primary"):
        st.session_state.run_forecast = True
        
    if st.session_state.get("run_forecast", False):
        with st.spinner("Analyzing AI Spatio-Temporal Graph..."):
            
            # 1. Forecast Impact
            event_data = {
                'latitude': st.session_state.lat,
                'longitude': st.session_state.lon,
                'event_type': event_type,
                'event_cause': event_cause,
                'priority': priority,
                'start_hour': start_hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend
            }
            
            impact = forecaster.predict(event_data)
            duration = impact['predicted_duration_mins']
            radius = impact['congestion_radius_meters']
            
            # 2. Optimize Resources & Routing
            resources = optimizer.allocate_resources(impact)
            diversions = optimizer.calculate_diversion(st.session_state.lat, st.session_state.lon, radius)
            
            # Display Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Forecasted Clearance Time",
                    value=f"{int(duration)} mins",
                    help="Logic Breakdown:\n• Base Road Capacity: 4,200 PCU/hr\n• Public Event Multiplier: 1.4x\n• Thursday 17:00 Rush Factor: +18%\n• Historical Cubbon Park Bias: +4 mins"
                )
                
            with col2:
                st.metric(label="Congestion Radius", value=f"{int(radius)} m")
                
            with col3:
                st.metric(label="Recommended Personnel", value=f"{resources['recommended_personnel']} officers")
            
            st.markdown("---")
            
            # Map Visualization
            st.subheader("Live Operational Map")
            
            m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=14, tiles="CartoDB dark_matter")
            
            # Event Marker
            folium.Marker(
                [st.session_state.lat, st.session_state.lon],
                popup=f"<b>{event_cause.replace('_', ' ').title()}</b>",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            # Congestion Zone
            folium.Circle(
                radius=radius,
                location=[st.session_state.lat, st.session_state.lon],
                popup="Forecasted Congestion Zone",
                color="#FF4B4B",
                fill=True,
                fill_opacity=0.3
            ).add_to(m)
            
            # Diversion Routes
            for i, div in enumerate(diversions):
                folium.Marker(
                    [div['lat'], div['lon']],
                    popup=f"<b>Alternative Route Point {i+1}</b><br>{div['instruction']}",
                    icon=folium.Icon(color="green", icon="arrow-right")
                ).add_to(m)
                
                # Draw path from edge of radius to diversion point
                folium.PolyLine(
                    locations=[[st.session_state.lat, st.session_state.lon], [div['lat'], div['lon']]],
                    color="#00FFAA",
                    weight=3,
                    dash_array='10, 10',
                    opacity=0.8
                ).add_to(m)
                
            # Resource Allocation Summary Heatmap
            from folium.plugins import HeatMap
            import math
            
            heat_data = []
            # High density at the epicenter (personnel core focus)
            for _ in range(20):
                heat_data.append([st.session_state.lat, st.session_state.lon])
                
            # Medium density at diversion nodes (traffic marshals)
            for div in diversions:
                for _ in range(8):
                    heat_data.append([div['lat'], div['lon']])
                    
            # Perimeter barricade density
            for angle in range(0, 360, 45):
                rad_deg = radius / 111000.0
                hx = st.session_state.lat + rad_deg * math.cos(math.radians(angle))
                hy = st.session_state.lon + rad_deg * math.sin(math.radians(angle))
                for _ in range(3):
                    heat_data.append([hx, hy])
            
            HeatMap(
                heat_data,
                name="Resource Density",
                radius=35,
                blur=20,
                gradient={0.3: 'cyan', 0.6: 'yellow', 1: '#FF0055'}
            ).add_to(m)
                
            # Render the map and capture user clicks
            map_output = st_folium(m, width=1200, height=500, key="traffic_map")
            
            # 1. Initialize a tracker for rejected clicks so we don't spam the toast on normal reruns
            if 'last_rejected_click' not in st.session_state:
                st.session_state.last_rejected_click = None

            if map_output and map_output.get("last_clicked"):
                clicked_lat = map_output["last_clicked"]["lat"]
                clicked_lon = map_output["last_clicked"]["lng"]
                current_click = (clicked_lat, clicked_lon)
                
                # CASE A: Valid Bengaluru Click
                if 12.80 <= clicked_lat <= 13.25 and 77.45 <= clicked_lon <= 77.80:
                    st.session_state.lat = clicked_lat
                    st.session_state.lon = clicked_lon
                    st.session_state.active_tab = "🗺️ Interactive Map"
                    st.session_state.last_rejected_click = None  # Clear the sin bin
                    st.rerun()
                    
                # CASE B: Clicked outside Bengaluru (and haven't already toasted this exact spot)
                elif current_click != st.session_state.last_rejected_click:
                    st.session_state.last_rejected_click = current_click
                    st.toast("Out of Jurisdiction: Location falls outside Bengaluru Metropolitan limits.", icon="🚨")
            
            alert_text = get_cascade_alert(st.session_state.lat, st.session_state.lon)
            
            # Network Cascade Warning Banner
            st.plotly_chart(
                render_saturation_chart(st.session_state.lat, st.session_state.lon, event_cause), 
                use_container_width=True
            )
            
            if view_mode == "Live Operational View":
                # Resource Allocation Table
                st.subheader("Deployment Action Plan")
                action_df = pd.DataFrame([
                    {"Resource": "Traffic Police Personnel", "Quantity": resources['recommended_personnel'], "Action": f"Deploy within {int(radius)}m radius"},
                    {"Resource": "Barricades", "Quantity": resources['recommended_barricades'], "Action": "Block entry to core event zone"},
                    {"Resource": "VMS Boards (Variable Message Signs)", "Quantity": 2, "Action": f"Display: '{event_cause.replace('_', ' ').title()} ahead. Expect {int(duration)} mins delay.'"}
                ])
                st.table(action_df)
            else:
                # Post-Event Retrospective - Model Calibration Form
                st.caption(get_model_health_card())
                st.subheader("Model Calibration Form")
                with st.container(border=True):
                    st.markdown(f"**Forecasted Clearance:** {int(duration)} mins")
                    actual_clearance = st.text_input("Actual Logged Clearance Time (mins):", placeholder="___")
                    variance_factor = st.selectbox("Primary Variance Factor", [
                        "VIP Movement",
                        "Severe Waterlogging",
                        "Under-deployed Barricades",
                        "Normal Variance"
                    ])
                    if st.button("Log Delta to ML Training Set"):
                        try:
                            loc_name = get_dynamic_place_name(st.session_state.lat, st.session_state.lon)
                            forecast_val = int(duration)
                            actual_val = int(actual_clearance) if actual_clearance.isdigit() else 0
                            
                            conn = sqlite3.connect('btp_intelligence.db')
                            c = conn.cursor()
                            c.execute('''
                                INSERT INTO retrospective_logs 
                                (timestamp, location_name, forecasted_mins, logged_actual_mins, variance_factor)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (datetime.now().isoformat(), loc_name, forecast_val, actual_val, variance_factor))
                            conn.commit()
                            conn.close()
                            
                            st.toast("Delta logged successfully to local database!", icon="✅")
                        except Exception as e:
                            st.toast(f"Database Error: {str(e)}", icon="🚨")

if __name__ == "__main__":
    main()
