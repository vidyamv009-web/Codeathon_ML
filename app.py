"""
Simple Flask app for car price prediction.

1. Run the joblib cell in CarPricePrediction.ipynb first
2. python app.py
3. Open http://127.0.0.1:5000
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
ARTIFACTS_PATH = BASE / "model" / "car_price_artifacts.joblib"

app = Flask(__name__)

# Simple UI fields only (other features use saved defaults)
UI_NUM = ["enginesize", "curbweight", "horsepower", "highwaympg", "carwidth"]
UI_CAT = ["company", "fueltype", "carbody", "aspiration", "drivewheel"]


def load_artifacts():
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ARTIFACTS_PATH}. Run the joblib save cell in the notebook first."
        )
    return joblib.load(ARTIFACTS_PATH)


def predict_price(form_data, artifacts):
    row = dict(artifacts["defaults"])

    for col in UI_NUM:
        if form_data.get(col) not in (None, ""):
            row[col] = float(form_data[col])

    for col in UI_CAT:
        if form_data.get(col) not in (None, ""):
            row[col] = form_data[col]

    for col in artifacts["cat_cols"]:
        le = artifacts["encoders"][col]
        value = row[col]
        if value not in le.classes_:
            value = artifacts["defaults"][col]
        row[col] = int(le.transform([value])[0])

    X = pd.DataFrame(
        [[row[c] for c in artifacts["feature_names"]]],
        columns=artifacts["feature_names"],
    )
    return round(float(artifacts["model"].predict(X)[0]), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    artifacts = load_artifacts()
    options = artifacts["category_options"]
    defaults = artifacts["defaults"]
    prediction = None
    error = None
    values = {col: defaults[col] for col in UI_NUM + UI_CAT}

    if request.method == "POST":
        values = {col: request.form.get(col, defaults[col]) for col in UI_NUM + UI_CAT}
        try:
            prediction = predict_price(request.form, artifacts)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        options=options,
        values=values,
        ui_num=UI_NUM,
        ui_cat=UI_CAT,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
