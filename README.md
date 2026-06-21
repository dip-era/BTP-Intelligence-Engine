# Bengaluru Traffic Police: Intelligence Engine

**Flipkart Gridlock 2.0 Hackathon Submission**
**Problem Statement 2:** Event-Driven Traffic Congestion Forecasting & Dynamic Routing

The **BTP Intelligence Engine** is a production-grade MLOps prototype designed to simulate a live command center tool. Moving beyond standard static machine learning, it provides a comprehensive end-to-end simulation capable of predicting, visualizing, and coordinating responses to localized traffic breakdown events (both planned and unplanned) across the Bengaluru Metropolitan Area.

## Key Features

### 1. Ultra-Premium "Cyber-Neon" Aesthetic
A beautiful, high-contrast dark-mode UI built natively on Streamlit. Features custom CSS for frosted glass metric cards, glowing crimson accents, and centralized state management for a seamless, professional experience.

### 2. Dual-Mode Input Controller
A robust state controller that perfectly synchronizes input commands without lag:
- **Interactive Map**: Click anywhere on the map to instantly set the event epicenter.
- **Manual GPS**: Precise latitude and longitude inputs with strict geo-fencing (Lat 12.80-13.25, Lon 77.45-77.80) to prevent coordinate drifting to the Arctic. Out-of-bounds clicks trigger non-intrusive toast notifications.

### 3. Live Spatio-Temporal Forecasting Pipeline
- **Impact Forecaster**: Utilizes advanced ML models (XGBoost) trained on temporal/spatial features to predict congestion radius and clearance times.
- **Operations Optimizer**: Employs NetworkX graph logic to generate dynamic routing diversions, along with deterministic models to calculate the required ground personnel and barricades.

### 4. Advanced Geospatial Visualizations
- **Live Operational Map**: Interactive `Folium` rendering with an overlay of the congestion zone (Red Circle), diversion points (Green Markers), and calculated polyline escape routes.
- **Resource Allocation Summary Heatmap**: A `folium.plugins.HeatMap` that mathematically distributes density coordinates (epicenter, diversion nodes, and perimeter barricades) to visually communicate the deployment footprint of BTP personnel.

### 5. Network Cascade Alerts & Plotly Telemetry
- **Dynamic OSM Lookups**: Automatically queries the OpenStreetMap API via `geopy` to reverse-geocode map clicks, generating hyper-local neighborhood-aware warnings.
- **Perimeter Saturation Chart**: A live, horizontal Plotly bar chart visually mapping the predicted stress load onto surrounding arterial roads, shifting to Crimson when capacity hits $\ge 90\%$.

### 6. Post-Event Retrospective & MLOps Database
- **Model Calibration Form**: Swap the dashboard to retrospective mode. Command center operators can log the actual clearance times and variance factors directly against the AI forecast.
- **SQLite Persistence**: Ground-truth deltas are seamlessly committed to a local `btp_intelligence.db` database using native Python SQLite3.
- **Live MLOps Health Card**: A live diagnostic banner computes Mean Absolute Error (MAE) and Systematic Bias in real-time directly from the database to track model drift.

## Tech Stack
- **Frontend & App State**: Streamlit, Streamlit-Folium
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, Numpy, Scipy
- **Geospatial & Routing**: Folium, NetworkX, Geopy
- **Data Visualization**: Plotly
- **Database Persistence**: SQLite3

## Installation & Execution

1. **Navigate to the project directory.**
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the local ML Model ensuring binaries exist. It will also launch the intelligence engine at the end:**
   ```bash
   python main.py
   ```
## Design Philosophy
The system implements a "Demo Mode" initialization, instantly loading a default event profile centered around Cubbon Park (Lat 12.9716, Lon 77.5946) to prevent a cold-start blank screen during hackathon presentations.
