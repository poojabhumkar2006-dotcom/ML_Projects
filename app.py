from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib

app = Flask(__name__)

# =====================================================
# Base Directory
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "students_placement.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.csv")

# =====================================================
# Load Model & Dataset
# =====================================================

model = joblib.load(MODEL_FILE)

data = pd.read_csv(DATA_FILE)

# =====================================================
# Create Prediction History File
# =====================================================

if not os.path.exists(HISTORY_FILE):

    pd.DataFrame(columns=[
        "CGPA",
        "IQ",
        "Profile Score",
        "Prediction",
        "Confidence",
        "Date & Time"
    ]).to_csv(HISTORY_FILE, index=False)
    # =====================================================
# Home Page
# =====================================================

@app.route("/")
def home():

    history = pd.read_csv(HISTORY_FILE)

    total_predictions = len(history)

    placed = len(
        history[history["Prediction"] == "Placed"]
    )

    not_placed = len(
        history[history["Prediction"] == "Not Placed"]
    )

    placement_rate = round(
        (placed / total_predictions) * 100,
        2
    ) if total_predictions else 0

    average_cgpa = round(
        history["CGPA"].mean(),
        2
    ) if total_predictions else 0

    confidence_numeric = pd.to_numeric(
        history["Confidence"],
        errors="coerce"
    )

    highest_confidence = round(
        confidence_numeric.max(),
        2
    ) if not confidence_numeric.dropna().empty else 0

    return render_template(
        "index.html",
        history=history.values.tolist()[::-1],
        total_predictions=total_predictions,
        placed=placed,
        not_placed=not_placed,
        placement_rate=placement_rate,
        average_cgpa=average_cgpa,
        highest_confidence=highest_confidence,
        result=None,
        confidence="N/A",
        recommendation="",
        error=""
    )
# =====================================================
# Prediction Route
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    history = pd.read_csv(HISTORY_FILE)

    try:

        # Get input values
        cgpa = float(request.form["cgpa"])
        iq = float(request.form["iq"])
        profile_score = float(request.form["profile_score"])

        features = np.array([[cgpa, iq, profile_score]])

        # Predict
        prediction = model.predict(features)[0]

        if prediction == 0:
            result = "Placed"
        else:
            result = "Not Placed"

        # Confidence
        try:
            confidence = round(
                max(model.predict_proba(features)[0]) * 100,
                2
            )
        except Exception:
            confidence = "N/A"

        # Recommendation
        if result == "Placed":
            recommendation = (
                "Excellent profile! Keep improving your communication, "
                "aptitude and interview skills."
            )
        else:
            recommendation = (
                "Improve your CGPA, strengthen your profile, "
                "practice aptitude and build more technical projects."
            )

        # Save Prediction
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        new_record = pd.DataFrame([{
            "CGPA": cgpa,
            "IQ": iq,
            "Profile Score": profile_score,
            "Prediction": result,
            "Confidence": confidence,
            "Date & Time": current_time
        }])

        history = pd.concat(
            [history, new_record],
            ignore_index=True
        )

        history.to_csv(HISTORY_FILE, index=False)

    except Exception as e:

        total_predictions = len(history)

        placed = len(
            history[history["Prediction"] == "Placed"]
        )

        not_placed = len(
            history[history["Prediction"] == "Not Placed"]
        )

        placement_rate = round(
            (placed / total_predictions) * 100,
            2
        ) if total_predictions else 0

        average_cgpa = round(
            history["CGPA"].mean(),
            2
        ) if total_predictions else 0

        confidence_numeric = pd.to_numeric(
            history["Confidence"],
            errors="coerce"
        )

        highest_confidence = round(
            confidence_numeric.max(),
            2
        ) if not confidence_numeric.dropna().empty else 0

        return render_template(
            "index.html",
            history=history.values.tolist()[::-1],
            total_predictions=total_predictions,
            placed=placed,
            not_placed=not_placed,
            placement_rate=placement_rate,
            average_cgpa=average_cgpa,
            highest_confidence=highest_confidence,
            result=None,
            confidence="N/A",
            recommendation="",
            error=str(e)
        )

    # Dashboard Statistics
    total_predictions = len(history)

    placed = len(
        history[history["Prediction"] == "Placed"]
    )

    not_placed = len(
        history[history["Prediction"] == "Not Placed"]
    )

    placement_rate = round(
        (placed / total_predictions) * 100,
        2
    ) if total_predictions else 0

    average_cgpa = round(
        history["CGPA"].mean(),
        2
    ) if total_predictions else 0

    confidence_numeric = pd.to_numeric(
        history["Confidence"],
        errors="coerce"
    )

    highest_confidence = round(
        confidence_numeric.max(),
        2
    ) if not confidence_numeric.dropna().empty else 0

    return render_template(
        "index.html",
        history=history.values.tolist()[::-1],
        total_predictions=total_predictions,
        placed=placed,
        not_placed=not_placed,
        placement_rate=placement_rate,
        average_cgpa=average_cgpa,
        highest_confidence=highest_confidence,
        result=result,
        confidence=confidence,
        recommendation=recommendation,
        error=""
    )
# =====================================================
# Download Prediction History
# =====================================================

@app.route("/download")
def download():

    if os.path.exists(HISTORY_FILE):

        return send_file(
            HISTORY_FILE,
            as_attachment=True
        )

    return redirect(url_for("home"))


# =====================================================
# Clear Prediction History
# =====================================================

@app.route("/clear")
def clear():

    df = pd.DataFrame(columns=[
        "CGPA",
        "IQ",
        "Profile Score",
        "Prediction",
        "Confidence",
        "Date & Time"
    ])

    df.to_csv(HISTORY_FILE, index=False)

    return redirect(url_for("home"))


# =====================================================
# Chart Data
# =====================================================

@app.route("/chart-data")
def chart_data():

    history = pd.read_csv(HISTORY_FILE)

    placed = len(
        history[history["Prediction"] == "Placed"]
    )

    not_placed = len(
        history[history["Prediction"] == "Not Placed"]
    )

    confidence_numeric = pd.to_numeric(
        history["Confidence"],
        errors="coerce"
    ).fillna(0)

    return {
        "placed": placed,
        "not_placed": not_placed,
        "cgpa": history["CGPA"].tolist(),
        "confidence": confidence_numeric.tolist()
    }


# =====================================================
# 404 Page
# =====================================================

@app.errorhandler(404)
def page_not_found(e):

    return render_template("404.html"), 404


# =====================================================
# Run Flask App
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)