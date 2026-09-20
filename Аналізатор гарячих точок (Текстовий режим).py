# -*- coding: utf-8 -*-
"""
# Oleksii Prokopchenko All rights reserved, 2025-2026
# Riverside,  Illinois, USA
# Tested in Google Colab - As-Is
"""

# Встановлення необхідних бібліотек
!pip install -q xgboost ipywidgets pandas numpy scikit-learn matplotlib

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# 1. Симуляція історичних даних для Ріверсайду (Чикаго)
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
        'hour': hour,
        'day_of_week': day_of_week,
        'temperature': round(temperature, 1),
        'precipitation': precipitation,
        'traffic_flow': traffic_flow,
        'is_roadwork': is_roadwork,
        'target_incident': target
    })

df = pd.DataFrame(data_rows)

# 2. Навчання моделі XGBoost
features = ['hour', 'day_of_week', 'temperature', 'precipitation', 'traffic_flow', 'is_roadwork']
X = df[features]
y = df['target_incident']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 3. Інтефейс керування (ipywidgets)
location_widget = widgets.Dropdown(options=[(loc['name'], i) for i, loc in enumerate(locations)], value=0, description='Перехрестя:')
hour_widget = widgets.IntSlider(value=8, min=0, max=23, step=1, description='Година доби:')
day_widget = widgets.Dropdown(options=[('Понеділок', 0), ('Вівторок', 1), ('Середа', 2), ('Четвер', 3), ('П’ятниця', 4), ('Субота', 5), ('Неділя', 6)], value=0, description='День тижня:')
temp_widget = widgets.FloatSlider(value=18.0, min=-15.0, max=35.0, step=0.5, description='Температура:')
precip_widget = widgets.Dropdown(options=[('Немає', 0.0), ('Слабкий дощ', 0.5), ('Сильний дощ', 2.5), ('Злива', 10.0)], value=0.0, description='Опади:')
flow_widget = widgets.IntSlider(value=800, min=100, max=1500, step=50, description='Трафік авт/год:')
roadwork_widget = widgets.Checkbox(value=False, description='Дорожні роботи')

out = widgets.Output()

def update_simulation(b=None):
    with out:
        clear_output(wait=True)
        selected_loc = locations[location_widget.value]

        input_data = pd.DataFrame([{
            'hour': hour_widget.value,
            'day_of_week': day_widget.value,
            'temperature': temp_widget.value,
            'precipitation': precip_widget.value,
            'traffic_flow': flow_widget.value,
            'is_roadwork': int(roadwork_widget.value)
        }])

        probability = model.predict_proba(input_data)[0][1] * 100

        print(f"📍 Локація: {selected_loc['name']}")
        print(f"🚨 Прогнозована ймовірність затору/інциденту: {probability:.1f}%")
        if probability > 60:
            print("⚠️ УВАГА: Високий ризик гарячої точки! Потрібен патруль.")
        elif probability > 30:
            print("⚡ Помірний ризик.")
        else:
            print("✅ Дорожня ситуація стабільна.")

        loc_history = df[df['location_name'] == selected_loc['name']]
        hist_pct = loc_history['target_incident'].mean() * 100
        print(f"📊 Історична частота інцидентів на цій ділянці: {hist_pct:.1f}%\n")

        # Візуалізація важливості ознак моделі за допомогою matplotlib
        fig, ax = plt.subplots(figsize=(8, 3))
        xgb.plot_importance(model, ax=ax, max_num_features=5, importance_type='weight', color='teal')
        ax.set_title("Головні фактори впливу на модель XGBoost")
        plt.tight_layout()
        plt.show()

run_button = widgets.Button(description='Розрахувати', button_style='success')
run_button.on_click(update_simulation)

display(widgets.VBox([widgets.HTML("<h3>Аналізатор гарячих точок (Текстовий режим)</h3>"), location_widget, hour_widget, day_widget, temp_widget, precip_widget, flow_widget, roadwork_widget, run_button]), out)
update_simulation()
