import os
import pandas as pd
from datetime import datetime


HISTORY_FILE = "history.csv"


# ============================================================
# VALIDATION
# ============================================================

def validate_input(
    city,
    owner,
    availability
):

    errors = []

    if not city.strip():
        errors.append(
            "Please enter the city name."
        )

    if not owner.strip():
        errors.append(
            "Please enter the owner type."
        )

    if not availability.strip():
        errors.append(
            "Please enter availability status."
        )

    return errors


# ============================================================
# HOUSE SCORE
# ============================================================

def calculate_house_score(
    bhk,
    size_sqft,
    school,
    hospital,
    parking,
    security,
    furnished
):

    score = 0

    score += min(bhk * 5, 20)

    score += min(size_sqft / 100, 20)

    score += min(school * 2, 15)

    score += min(hospital * 2, 15)

    if parking == "Yes":
        score += 10

    if security == "Yes":
        score += 10

    if furnished == "Furnished":
        score += 10

    elif furnished == "Semi-furnished":
        score += 5

    return min(round(score, 2), 100)


# ============================================================
# EMI
# ============================================================

def calculate_emi(
    price_lakhs,
    annual_rate=8.5,
    years=20
):

    principal = price_lakhs * 100000

    monthly_rate = (
        annual_rate / 12 / 100
    )

    months = years * 12

    if monthly_rate == 0:

        return principal / months

    emi = (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** months
    ) / (
        (1 + monthly_rate) ** months - 1
    )

    return round(emi, 2)


# ============================================================
# PRICE CATEGORY
# ============================================================

def price_category(price):

    if price < 50:

        return "Affordable 🟢"

    elif price < 100:

        return "Mid-Range 🟡"

    else:

        return "Premium / Luxury 🔴"


# ============================================================
# MARKET PRICE
# ============================================================

def calculate_market_price(
    size_sqft,
    price_sqft
):

    return (
        size_sqft * price_sqft
    ) / 100000


# ============================================================
# INVESTMENT ADVICE
# ============================================================

def investment_advice(
    age,
    school,
    hospital
):

    if (
        age <= 5
        and school >= 5
        and hospital >= 3
    ):

        return (
            "Excellent Investment",
            "The property is relatively new and has good nearby facilities."
        )

    elif age <= 10:

        return (
            "Good Investment",
            "The property has reasonable long-term investment potential."
        )

    else:

        return (
            "Consider Carefully",
            "The property is older and may require additional maintenance."
        )


# ============================================================
# SAVE HISTORY
# ============================================================

def save_prediction(
    input_data,
    prediction
):

    record = input_data.copy()

    record["Prediction"] = prediction

    record["Date"] = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    df = pd.DataFrame([record])

    if os.path.exists(HISTORY_FILE):

        df.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            HISTORY_FILE,
            index=False
        )


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if os.path.exists(HISTORY_FILE):

        return pd.read_csv(
            HISTORY_FILE
        )

    return pd.DataFrame()


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    input_data,
    prediction
):

    report = pd.DataFrame({

        "Feature":
            list(input_data.keys()),

        "Value":
            list(input_data.values())

    })

    report.loc[len(report)] = [
        "Predicted Price",
        prediction
    ]

    return report


# ============================================================
# PROPERTY SUMMARY
# ============================================================

def property_summary(
    state,
    city,
    property_type,
    bhk,
    size_sqft,
    year,
    prediction
):

    return pd.DataFrame({

        "Feature": [

            "State",
            "City",
            "Property Type",
            "BHK",
            "Area",
            "Year Built",
            "Estimated Price"

        ],

        "Value": [

            state,
            city,
            property_type,
            bhk,
            f"{size_sqft} Sq.Ft",
            year,
            f"₹ {prediction:,.2f} Lakhs"

        ]

    })