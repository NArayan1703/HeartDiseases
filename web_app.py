from __future__ import annotations

import csv
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
DATA_PATH = BASE_DIR / "health_data.csv"
PREDICTION_THRESHOLD = 0.35

FEATURES = [
    {
        "name": "sex",
        "label": "Sex",
        "default": 1.0,
        "type": "select",
        "help": "Biological sex recorded in the dataset.",
        "options": [(1.0, "Male"), (0.0, "Female")],
    },
    {
        "name": "age",
        "label": "Age",
        "default": 45.0,
        "unit": "years",
        "type": "number",
        "help": "Age of the person.",
    },
    {
        "name": "smokes",
        "label": "Smokes tobacco or cigarettes",
        "default": 0.0,
        "type": "select",
        "help": "Current tobacco or cigarette use.",
        "options": [(0.0, "No"), (1.0, "Yes")],
    },
    {
        "name": "total_cholesterol",
        "label": "Total cholesterol",
        "default": 145.0,
        "type": "number",
        "help": "Total cholesterol measurement from the dataset.",
    },
    {
        "name": "fasting_blood_sugar",
        "label": "Fasting blood sugar",
        "default": 94.0,
        "unit": "mg/dl",
        "type": "number",
        "help": "Fasting blood sugar measurement.",
    },
    {
        "name": "vigorous_minutes",
        "label": "Vigorous activity",
        "default": 0.0,
        "unit": "minutes per week",
        "type": "number",
        "help": "Weekly vigorous activity minutes.",
    },
    {
        "name": "moderate_minutes",
        "label": "Moderate activity",
        "default": 0.0,
        "unit": "minutes per week",
        "type": "number",
        "help": "Weekly moderate activity minutes.",
    },
    {
        "name": "walking_cycling_minutes",
        "label": "Walking or cycling",
        "default": 0.0,
        "unit": "minutes per week",
        "type": "number",
        "help": "Weekly walking or cycling minutes.",
    },
    {
        "name": "sitting_minutes",
        "label": "Sitting time",
        "default": 240.0,
        "unit": "minutes per day",
        "type": "number",
        "help": "Usual minutes spent sitting per day.",
    },
    {
        "name": "fruit_servings_per_week",
        "label": "Fruit servings",
        "default": 7.0,
        "unit": "servings per week",
        "type": "number",
        "help": "Fruit servings per week.",
    },
    {
        "name": "vegetable_servings_per_week",
        "label": "Vegetable servings",
        "default": 7.0,
        "unit": "servings per week",
        "type": "number",
        "help": "Vegetable servings per week.",
    },
    {
        "name": "systolic_bp",
        "label": "Systolic blood pressure",
        "default": 120.0,
        "unit": "mmHg",
        "type": "number",
        "help": "Systolic blood pressure.",
    },
    {
        "name": "diastolic_bp",
        "label": "Diastolic blood pressure",
        "default": 80.0,
        "unit": "mmHg",
        "type": "number",
        "help": "Diastolic blood pressure.",
    },
]

FEATURE_NAMES = [feature["name"] for feature in FEATURES]

COLUMN_RENAMES = {
    "sex": "sex",
    "age": "age",
    "Smoke tobacoo or ciggrate": "smokes",
    "Total cholesterol": "total_cholesterol",
    "Fasting Blood Sugar": "fasting_blood_sugar",
    "vigorous_minutes": "vigorous_minutes",
    "moderate_minutes": "moderate_minutes",
    "cycling or walking minutes": "walking_cycling_minutes",
    "minutes sitting per day moderate-intensity sports": "sitting_minutes",
    "serving of foods per week": "fruit_servings_per_week",
    "Vegetable_serving per week": "vegetable_servings_per_week",
    "Blood presure Systolic ": "systolic_bp",
    "Blood Pressure Diastolic": "diastolic_bp",
}


def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def csv_float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def load_feature_stats():
    stats = {name: {"min": None, "max": None, "mean": 0.0, "count": 0} for name in FEATURE_NAMES}
    if not DATA_PATH.exists():
        return stats

    with DATA_PATH.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            normalized = {new: csv_float(row, old) for old, new in COLUMN_RENAMES.items()}
            for name in FEATURE_NAMES:
                value = normalized.get(name)
                if value is None:
                    continue
                item = stats[name]
                item["min"] = value if item["min"] is None else min(item["min"], value)
                item["max"] = value if item["max"] is None else max(item["max"], value)
                item["mean"] += value
                item["count"] += 1

    for name in stats:
        item = stats[name]
        if item["count"]:
            item["mean"] = item["mean"] / item["count"]
    return stats


app = Flask(__name__)

try:
    MODEL = load_model()
    STARTUP_ERROR = None
except Exception as exc:  # noqa: BLE001 - shown in the local setup page.
    MODEL = None
    STARTUP_ERROR = exc

FEATURE_STATS = load_feature_stats()


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heart Disease Predictor</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #18212f;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 5vw, 4.25rem);
      line-height: 0.95;
      letter-spacing: 0;
      color: #121824;
    }
    .subhead { margin: 0; color: #526074; max-width: 720px; line-height: 1.5; }
    .status {
      min-width: 180px;
      padding: 12px 14px;
      border: 1px solid #d6deea;
      border-radius: 8px;
      background: #fff;
      color: #2d3a4d;
      font-size: 0.92rem;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(300px, 0.85fr);
      gap: 20px;
      align-items: start;
    }
    form, .panel {
      background: #fff;
      border: 1px solid #dce3ee;
      border-radius: 8px;
      box-shadow: 0 18px 48px rgba(18, 24, 36, 0.07);
    }
    form { padding: 20px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    label { display: block; font-weight: 700; color: #253044; margin-bottom: 7px; }
    input, select {
      width: 100%;
      min-height: 42px;
      border: 1px solid #c9d3e2;
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      color: #172033;
      background: #fff;
    }
    small { display: block; min-height: 18px; margin-top: 5px; color: #65748a; }
    button {
      margin-top: 20px;
      width: 100%;
      min-height: 46px;
      border: 0;
      border-radius: 6px;
      background: #146c63;
      color: #fff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    button:hover { background: #10584f; }
    .panel { padding: 20px; }
    .result {
      display: grid;
      gap: 12px;
      margin-bottom: 18px;
    }
    .score {
      font-size: clamp(2.5rem, 8vw, 5.5rem);
      line-height: 1;
      font-weight: 900;
      color: #a93f2c;
    }
    .verdict {
      display: inline-flex;
      width: fit-content;
      padding: 7px 10px;
      border-radius: 6px;
      background: #f2f5f9;
      color: #253044;
      font-weight: 800;
    }
    .error {
      border-left: 4px solid #b23b3b;
      background: #fff4f4;
      padding: 14px;
      border-radius: 6px;
      color: #632222;
      overflow-wrap: anywhere;
    }
    .note { color: #5e6c80; line-height: 1.5; margin: 0; }
    @media (max-width: 800px) {
      header, .layout { grid-template-columns: 1fr; display: grid; }
      .grid { grid-template-columns: 1fr; }
      .status { min-width: 0; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Heart Disease Predictor</h1>
        <p class="subhead">Local Flask app using an XGBoost model trained from health, activity, diet, and blood pressure inputs.</p>
      </div>
      <div class="status">{{ rows }} records | {{ features|length }} model inputs</div>
    </header>

    {% if startup_error %}
      <div class="error">
        <strong>Model could not be loaded.</strong><br>
        {{ startup_error }}<br><br>
        Train the model or install the project dependencies, then restart the server.
      </div>
    {% endif %}

    <div class="layout">
      <form method="post">
        <div class="grid">
          {% for feature in features %}
            <div>
              <label for="{{ feature.name }}">{{ feature.label }}</label>
              {% if feature.type == "select" %}
                <select id="{{ feature.name }}" name="{{ feature.name }}">
                  {% for value, label in feature.options %}
                    <option value="{{ value }}" {% if values[feature.name] == value %}selected{% endif %}>{{ label }}</option>
                  {% endfor %}
                </select>
              {% else %}
                <input id="{{ feature.name }}" name="{{ feature.name }}" type="number" step="any" value="{{ values[feature.name] }}">
              {% endif %}
              <small>
                {% if feature.type == "number" and stats[feature.name].count %}
                  {{ feature.help }} Range {{ "%.2f"|format(stats[feature.name].min) }} to {{ "%.2f"|format(stats[feature.name].max) }} {{ feature.unit }}.
                {% else %}
                  {{ feature.help }}
                {% endif %}
              </small>
            </div>
          {% endfor %}
        </div>
        <button type="submit">Predict Risk</button>
      </form>

      <aside class="panel">
        {% if result %}
          <div class="result">
            <span class="verdict">{{ result.label }}</span>
            <div class="score">{{ "%.1f"|format(result.probability * 100) }}%</div>
          </div>
          <p class="note">Classification threshold: {{ threshold }}. This is a screening estimate from survey-style data, not a clinical diagnosis.</p>
        {% else %}
          <p class="note">Enter health measurements and lifestyle values, then run a prediction. Defaults come from the dataset means where available.</p>
        {% endif %}
      </aside>
    </div>
  </main>
</body>
</html>
"""


def default_values():
    values = {}
    for feature in FEATURES:
        name = feature["name"]
        fallback = feature["default"]
        if feature["type"] == "select":
            values[name] = fallback
            continue
        mean = FEATURE_STATS[name]["mean"]
        values[name] = round(mean if FEATURE_STATS[name]["count"] else fallback, 4)
    return values


def parse_values():
    values = {}
    defaults = default_values()
    for name in FEATURE_NAMES:
        raw_value = request.form.get(name, "")
        try:
            values[name] = float(raw_value)
        except ValueError:
            values[name] = defaults[name]
    return values


@app.route("/", methods=["GET", "POST"])
def index():
    values = parse_values() if request.method == "POST" else default_values()
    result = None

    if request.method == "POST" and STARTUP_ERROR is None:
        ordered_values = pd.DataFrame(
            [[values[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )
        probability = float(MODEL.predict_proba(ordered_values)[0][1])
        result = {
            "probability": probability,
            "label": "Higher predicted risk" if probability >= PREDICTION_THRESHOLD else "Lower predicted risk",
        }

    row_count = FEATURE_STATS["age"]["count"]
    return render_template_string(
        TEMPLATE,
        features=FEATURES,
        values=values,
        stats=FEATURE_STATS,
        result=result,
        rows=f"{row_count:,}",
        startup_error=STARTUP_ERROR,
        threshold=PREDICTION_THRESHOLD,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
