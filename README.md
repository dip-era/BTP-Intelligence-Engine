# Bengaluru Traffic Police: Intelligence Engine

**Flipkart Gridlock 2.0 Hackathon Submission**
**Problem Statement 2:** Event-Driven Traffic Congestion Forecasting & Dynamic Routing

The **BTP Intelligence Engine** is a production-grade MLOps prototype designed to simulate a live command center tool. Moving beyond standard static machine learning, it provides a comprehensive end-to-end simulation capable of predicting, visualizing, and coordinating responses to localized traffic breakdown events (both planned and unplanned) across the Bengaluru Metropolitan Area.

[CLick here](https://youtu.be/XTd-b6_Qsls) for demo video 

[Click here](https://huggingface.co/spaces/ranbyDipz/BTP-Intelligence-Engine) for the application 

## Key Features

### 1. Ultra-Premium "Cyber-Neon" Aesthetic
A beautiful, high-contrast dark-mode UI built natively on Streamlit. Features custom CSS for frosted glass metric cards, glowing crimson accents, and centralized state management for a seamless, professional experience.

### 2. Dual-Mode Input Controller
A robust state controller that perfectly synchronizes input commands without lag:
- **Interactive Map**: Click anywhere on the map to instantly set the event epicenter.
- **Manual GPS**: Precise latitude and longitude inputs with strict geo-fencing (Lat 12.80-13.25, Lon 77.45-77.80) to prevent coordinate drifting to the Arctic. Out-of-bounds clicks trigger non-intrusive toast notifications.

### 3. Live Spatio-Temporal Forecasting Pipeline
- **Impact Forecaster**: Utilizes advanced ML models trained on temporal/spatial features to predict congestion radius and clearance times.
- **Operations Optimizer**: Employs deterministic models to calculate the required ground personnel and barricades based on the severity of the event.

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

---

## Mathematical & Algorithmic Core
Our engine doesn't just display static data; it generates live predictions using sophisticated algorithms under the hood:
1. **XGBoost Regression (Gradient Boosting):** We engineered a highly-tuned `XGBRegressor` to model complex, non-linear relationships between spatial coordinates, time of day, and event types to output precise clearance minute durations.
2. **Graph Theory & Dijkstra's Algorithm:** Using **NetworkX**, we built a directed graph representation of the road network. When an event occurs, the optimizer calculates the shortest, least-congested alternative escape corridors (Diversions) using algorithmic pathfinding.
3. **Haversine Distance & Trigonometric Modeling:** For the spatial mapping (like calculating the perimeter barricades on the Heatmap), we utilize Haversine approximations and Cosine/Sine trigonometric functions to mathematically generate circular deployment bounds in geographic space.

---

## Tech Stack
- **Frontend & App State**: Streamlit, Streamlit-Folium
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, Numpy, Scipy
- **Geospatial & Routing**: Folium, NetworkX, Geopy
- **Data Visualization**: Plotly
- **Database Persistence**: SQLite3
- **Cloud Deployment**: Hosted inside a Streamlit Docker Container on **Hugging Face Spaces**.

---

## Instructions to Run (Local Hosting)

Follow these exact steps to clone, configure, and locally host the BTP Intelligence Engine.

**1. Clone the Repository**
```bash
git clone https://github.com/dip-era/BTP-Intelligence-Engine.git
cd BTP-Intelligence-Engine
```

**2. Place the Datasets**
Move the raw dataset into the root/parent folder of the cloned repository. The pipeline expects these files to be present:
- `Astram event data_anonymized.csv`
  
**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Master Script**
Our architecture is designed for simplicity. You do not need to run separate `uvicorn` and `streamlit` servers. A single execution of the master python file will automatically ingest the datasets, train the ML models, save the binary states, and immediately boot up the Streamlit dashboard on your `localhost`.
```bash
python main.py
```
*(Wait a few moments for the model training to complete. The browser window will pop up automatically).*

---

## Design Philosophy
The system implements a "Demo Mode" initialization, instantly loading a default event profile centered around Cubbon Park (Lat 12.9716, Lon 77.5946) to prevent a cold-start blank screen during hackathon presentations.
