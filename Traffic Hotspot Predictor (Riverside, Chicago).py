# -*- coding: utf-8 -*-
"""
# Oleksii Prokopchenko All rights reserved, 2025-2026
# Riverside,  Illinois, USA
# Tested in Google Colab - As-Is
"""

# Install necessary libraries
!pip install -q xgboost ipywidgets pandas numpy scikit-learn plotly

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output

# 1. Simulate historical traffic and incident data for Riverside, Chicago
np.random.seed(42)
n_samples = 2500

locations = [
    {"name": "Longcommon Rd & E Burlington St", "lat": 41.8295, "lon": -87.8182},
    {"name": "Riverside Rd & Bloomingbank Rd", "lat": 41.8312, "lon": -87.8225},
    {"name": "First Ave & 31st St Intersection", "lat": 41.8385, "lon": -87.8341},
    {"name": "Woodside Rd & Harlem Ave", "lat": 41.8250, "lon": -87.8050},
    {"name": "Pasadena Dr & Ridgewood Rd", "lat": 41.8270, "lon": -87.8150},
    {"name": "Delaplaine Rd & Quincy St", "lat": 41.8340, "lon": -87.8190}
]

data_rows = []
for _ in range(n_samples):
    loc = np.random.choice(locations)
    hour = np.random.randint(0, 24)
    day_of_week = np.random.randint(0, 7)
    temperature = np.random.normal(15, 10)
    precipitation = np.random.choice([0.0, 0.5, 2.5, 10.0], p=[0.7, 0.2, 0.08, 0.02])
    traffic_flow = np.random.randint(100, 1500)
    is_roadwork = np.random.choice([0, 1], p=[0.85, 0.15])

    # Calculate synthetic risk score
    risk_score = (
        (1.5 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0.5) * 0.3 +
        (1.2 if day_of_week < 5 else 0.8) * 0.1 +
        (1.4 if precipitation > 2.0 else 1.0) * 0.2 +
        (traffic_flow / 1000.0) * 0.3 +
        (1.5 if is_roadwork == 1 else 1.0) * 0.1
    )
    risk_score += np.random.normal(0, 0.1)
    target = 1 if risk_score > 1.15 else 0

    data_rows.append({
        'location_name': loc['name'],
        'lat': loc['lat'],
        'lon': loc['lon'],
        'hour': hour,
        'day_of_week': day_of_week,
        'temperature': round(temperature, 1),
        'precipitation': precipitation,
        'traffic_flow': traffic_flow,
        'is_roadwork': is_roadwork,
        'target_incident': target
    })

df = pd.DataFrame(data_rows)

# 2. Train XGBoost Model
features = ['hour', 'day_of_week', 'temperature', 'precipitation', 'traffic_flow', 'is_roadwork']
X = df[features]
y = df['target_incident']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 3. Interactive Widgets Setup
location_widget = widgets.Dropdown(options=[(loc['name'], i) for i, loc in enumerate(locations)], value=0, description='Intersection:')
hour_widget = widgets.IntSlider(value=8, min=0, max=23, step=1, description='Hour of Day:')
day_widget = widgets.Dropdown(options=[('Monday', 0), ('Tuesday', 1), ('Wednesday', 2), ('Thursday', 3), ('Friday', 4), ('Saturday', 5), ('Sunday', 6)], value=0, description='Day of Week:')
temp_widget = widgets.FloatSlider(value=18.0, min=-15.0, max=35.0, step=0.5, description='Temp (°C):')
precip_widget = widgets.Dropdown(options=[('None', 0.0), ('Light Rain', 0.5), ('Heavy Rain', 2.5), ('Storm / Snow', 10.0)], value=0.0, description='Precipitation:')
flow_widget = widgets.IntSlider(value=800, min=100, max=1500, step=50, description='Traffic (veh/h):')
roadwork_widget = widgets.Checkbox(value=False, description='Roadwork Active')

out = widgets.Output()

def update_simulation_map(b=None):
    with out:
        clear_output(wait=True)
        selected_loc_idx = location_widget.value
        selected_loc = locations[selected_loc_idx]

        input_data = pd.DataFrame([{
            'hour': hour_widget.value,
            'day_of_week': day_widget.value,
            'temperature': temp_widget.value,
            'precipitation': precip_widget.value,
            'traffic_flow': flow_widget.value,
            'is_roadwork': int(roadwork_widget.value)
        }])

        probability = model.predict_proba(input_data)[0][1] * 100

        print(f"📍 Selected Location: {selected_loc['name']} (Riverside, Chicago)")
        print(f"🚨 Predicted Hotspot / Incident Probability: {probability:.1f}%")
        if probability > 60:
            print("⚠️ WARNING: High risk of traffic hotspot! Patrol unit deployment recommended.")
        elif probability > 30:
            print("⚡ Moderate risk. Monitoring advised.")
        else:
            print("✅ Traffic situation is stable.")

        # Build data for Plotly map across all locations
        map_data = []
        for i, loc in enumerate(locations):
            p = probability if i == selected_loc_idx else np.random.uniform(10, 40)
            map_data.append({
                'name': loc['name'],
                'lat': loc['lat'],
                'lon': loc['lon'],
                'Risk_Probability': round(p, 1),
                'Status': 'Selected Location' if i == selected_loc_idx else 'Other Segments'
            })

        map_df = pd.DataFrame(map_data)

        # Plotly Interactive Map Rendering
        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="Risk_Probability",
            size="Risk_Probability",
            color_continuous_scale="Reds",
            range_color=[0, 100],
            hover_name="name",
            zoom=13,
            height=400,
            title="Traffic Risk Hotspots Map - Riverside Suburb"
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        fig.show()

run_button = widgets.Button(description='Calculate & Update Map', button_style='success', icon='play')
run_button.on_click(update_simulation_map)

display(widgets.VBox([
    widgets.HTML("<h3>Traffic Hotspot Predictor (Riverside, Chicago)</h3>"),
    location_widget,
    widgets.HBox([hour_widget, day_widget]),
    widgets.HBox([temp_widget, precip_widget]),
    widgets.HBox([flow_widget, roadwork_widget]),
    run_button,
    widgets.HTML("<hr>")
], layout=widgets.Layout(width='100%')), out)

update_simulation_map()
