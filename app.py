from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib

model = joblib.load("model.pkl")


app = Flask(__name__)

# ----------------------------
# Load ML Model
# ----------------------------
MODEL_FILE = "model.pkl"
HISTORY_FILE = "prediction_history.csv"

model = joblib.load(MODEL_FILE)

# ----------------------------
# Create History File
# ----------------------------
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "CGPA",
        "IQ",
        "Profile Score",
        "Prediction",
        "Confidence",
        "Date & Time"
    ]).to_csv(HISTORY_FILE, index=False)


# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def home():

    history = pd.read_csv(HISTORY_FILE)

    total_predictions = len(history)

    placed = len(history[history["Prediction"] == "Placed"])

    not_placed = len(history[history["Prediction"] == "Not Placed"])

    placement_rate = round(
        (placed / total_predictions) * 100,
        2
    ) if total_predictions else 0

    average_cgpa = round(
        history["CGPA"].mean(),
        2
    ) if total_predictions else 0

    highest_confidence = round(
        history["Confidence"].max(),
        2
    ) if total_predictions else 0

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
# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get input values
        cgpa = float(request.form["cgpa"])
        iq = float(request.form["iq"])
        profile_score = float(request.form["profile_score"])
        features = np.array([[cgpa, iq, profile_score]])

        prediction = model.predict(features)[0]

        print("Input:", features)
        print("Raw Prediction:", prediction)

        if prediction == 0:
         result = "Placed"
        else:
         result = "Not Placed"

        # Confidence Score
        try:
            confidence = round(
                max(model.predict_proba(features)[0]) * 100,
                2
            )
        except:
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
                "practice aptitude and work on technical projects."
            )

        # Timestamp
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # Save Prediction History
        new_record = pd.DataFrame([{
            "CGPA": cgpa,
            "IQ": iq,
            "Profile Score": profile_score,
            "Prediction": result,
            "Confidence": confidence,
            "Date & Time": current_time
        }])

        history = pd.read_csv(HISTORY_FILE)

        history = pd.concat(
            [history, new_record],
            ignore_index=True
        )

        history.to_csv(HISTORY_FILE, index=False)

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
        )

        # Handle Confidence column safely
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
            result=result,
            confidence=confidence,
            recommendation=recommendation,
            history=history.values.tolist()[::-1],
            total_predictions=total_predictions,
            placed=placed,
            not_placed=not_placed,
            placement_rate=placement_rate,
            average_cgpa=average_cgpa,
            highest_confidence=highest_confidence,
            error=""
        )

    except Exception as e:

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
        total_predictions=total_predictions,
        placed=placed,
        not_placed=not_placed,
        placement_rate=placement_rate,
        history=history,
        result=result,
        confidence=confidence,
        recommendation=recommendation,
        error=os.error
        )
    # ==========================================
# Download Prediction History
# ==========================================

@app.route("/download")
def download():

    if os.path.exists(HISTORY_FILE):

        return send_file(
            HISTORY_FILE,
            as_attachment=True
        )

    return redirect(url_for("home"))


# ==========================================
# Clear Prediction History
# ==========================================

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

    df.to_csv(
        HISTORY_FILE,
        index=False
    )

    return redirect(url_for("home"))
# ==========================================
# Chart Data
# ==========================================

@app.route("/chart-data")
def chart_data():

    history = pd.read_csv(HISTORY_FILE)

    placed = len(
        history[
            history["Prediction"] == "Placed"
        ]
    )

    not_placed = len(
        history[
            history["Prediction"] == "Not Placed"
        ]
    )

    return {
        "placed": placed,
        "not_placed": not_placed,
        "cgpa": history["CGPA"].tolist(),
        "confidence": pd.to_numeric(
            history["Confidence"],
            errors="coerce"
        ).fillna(0).tolist()
    }
# ==========================================
# 404 Page
# ==========================================

@app.errorhandler(404)
def page_not_found(e):

    return render_template(
        "404.html"
    ), 404
# ==========================================
# Run Flask App
# ==========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)