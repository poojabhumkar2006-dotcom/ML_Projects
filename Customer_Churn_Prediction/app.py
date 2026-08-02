import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="🏦 Customer Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================

model = pickle.load(open("random_forest_churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ======================================================
# LOAD DATASET
# ======================================================

df = pd.read_csv("Churn_Modelling.csv")

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.stApp{
    background-color:#F5F7FA;
}

h1,h2,h3{
    color:#003366;
}

[data-testid="stSidebar"]{
    background:#0B3C5D;
}

[data-testid="stSidebar"] *{
    color:white;
}

.stButton>button{
    width:100%;
    height:55px;
    background:#2563EB;
    color:white;
    border-radius:10px;
    border:none;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1E40AF;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=90
)

st.sidebar.title("🏦 Banking Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Prediction",
        "ℹ About"
    ]
)
# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.title("🏦 Customer Churn Prediction System")

    st.markdown(
        "<h4 style='color:#2563EB;'>AI Powered Banking Analytics Dashboard</h4>",
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=1600",
        use_container_width=True
    )

    st.write("")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Customers", f"{len(df):,}")

    with col2:
        st.metric("🤖 Model", "Random Forest")

    with col3:
        st.metric("🎯 Accuracy", "86%")

    with col4:
        st.metric("📊 Features", "11")

    st.write("---")

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📖 About the Project")

        st.write("""
Customer Churn Prediction helps banks identify customers who are likely to leave the bank.

This application uses a trained **Random Forest Machine Learning Model** to predict whether a customer will stay or exit based on banking details.

### Features

- ✅ Customer Churn Prediction
- ✅ Banking Analytics Dashboard
- ✅ Random Forest Algorithm
- ✅ Stay & Exit Probability
- ✅ Risk Analysis
- ✅ Interactive User Interface
- ✅ Download Prediction Report

This system helps banks improve customer retention by identifying high-risk customers early.
""")

    with right:

        st.info("""
### 📌 Project Information

🏦 **Domain:** Banking

🤖 **Model:** Random Forest

📊 **Dataset:** Churn Modelling

👥 **Records:** 10,000 Customers

🎯 **Target:** Customer Churn

💻 **Framework:** Streamlit
""")

    st.write("---")

    st.subheader("📊 Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

    st.write("")

    st.subheader("📈 Dataset Statistics")

    st.dataframe(df.describe(), use_container_width=True)
    # ======================================================
# PREDICTION PAGE
# ======================================================

elif page == "📊 Prediction":

    st.title("📊 Customer Churn Prediction")

    st.markdown(
        "<h4 style='color:#2563EB;'>Enter Customer Details</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns(2)

    # ===========================
    # LEFT COLUMN
    # ===========================

    with left:

        credit_score = st.number_input(
            "💳 Credit Score",
            min_value=300,
            max_value=900,
            value=600
        )

        age = st.number_input(
            "🎂 Age",
            min_value=18,
            max_value=100,
            value=35
        )

        tenure = st.number_input(
            "📅 Tenure",
            min_value=0,
            max_value=10,
            value=5
        )

        balance = st.number_input(
            "💰 Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        num_products = st.selectbox(
            "🏦 Number of Products",
            [1, 2, 3, 4]
        )

    # ===========================
    # RIGHT COLUMN
    # ===========================

    with right:

        has_card = st.selectbox(
            "💳 Has Credit Card",
            ["Yes", "No"]
        )

        active_member = st.selectbox(
            "🟢 Active Member",
            ["Yes", "No"]
        )

        salary = st.number_input(
            "💼 Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        country = st.selectbox(
            "🌍 Country",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "👤 Gender",
            ["Male", "Female"]
        )

    st.write("")

    st.markdown("---")

    st.subheader("📋 Customer Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Credit Score", credit_score)

    with c2:
        st.metric("Age", age)

    with c3:
        st.metric("Balance", f"₹{balance:,.2f}")

    st.write("")

    predict = st.button("🔍 Predict Customer Churn")

    # ===========================
    # Data Encoding
    # ===========================

    germany = 1 if country == "Germany" else 0
    spain = 1 if country == "Spain" else 0
    male = 1 if gender == "Male" else 0

    has_card = 1 if has_card == "Yes" else 0
    active_member = 1 if active_member == "Yes" else 0
        # ======================================================
    # PREDICTION LOGIC
    # ======================================================

    if predict:

        # Create input array
        input_data = np.array([[
            credit_score,
            age,
            tenure,
            balance,
            num_products,
            has_card,
            active_member,
            salary,
            germany,
            spain,
            male
        ]])

        # Scale input
        input_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        stay_prob = probability[0][0] * 100
        churn_prob = probability[0][1] * 100

        st.write("")
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🟢 Stay Probability",
                f"{stay_prob:.2f}%"
            )

        with col2:
            st.metric(
                "🔴 Churn Probability",
                f"{churn_prob:.2f}%"
            )

        st.progress(int(churn_prob))

        # ======================================================
        # Probability Chart
        # ======================================================

        chart_df = pd.DataFrame(
            {
                "Probability": [
                    stay_prob,
                    churn_prob
                ]
            },
            index=[
                "Stay",
                "Churn"
            ]
        )

        st.subheader("📈 Probability Comparison")

        st.bar_chart(chart_df)

        # ======================================================
        # Prediction Message
        # ======================================================

        if prediction[0] == 1:

            st.error("🚨 High Risk Customer")

            st.markdown("""
### ⚠ Customer is likely to EXIT

Recommended Actions

- Contact customer immediately
- Offer cashback or discount
- Provide loyalty rewards
- Assign Relationship Manager
- Offer premium banking services
""")

        else:

            st.success("🎉 Customer is likely to STAY")

            st.balloons()

            st.markdown("""
### ✅ Customer is likely to STAY

Recommended Actions

- Continue customer engagement
- Offer Premium Credit Card
- Increase Reward Points
- Exclusive Banking Offers
- Maintain Customer Satisfaction
""")

        # ======================================================
        # Risk Level
        # ======================================================

        st.write("---")
        st.subheader("📈 Risk Level")

        if churn_prob < 30:

            st.success("🟢 LOW RISK")

        elif churn_prob < 70:

            st.warning("🟡 MEDIUM RISK")

        else:

            st.error("🔴 HIGH RISK")
                    # ======================================================
        # CUSTOMER SUMMARY
        # ======================================================

        st.write("")
        st.subheader("📋 Customer Summary")

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Credit Score",
                    "Age",
                    "Tenure",
                    "Country",
                    "Gender",
                    "Balance",
                    "Estimated Salary",
                    "Number of Products",
                    "Has Credit Card",
                    "Active Member",
                    "Prediction"
                ],
                "Value": [
                    credit_score,
                    age,
                    tenure,
                    country,
                    gender,
                    f"₹{balance:,.2f}",
                    f"₹{salary:,.2f}",
                    num_products,
                    "Yes" if has_card else "No",
                    "Yes" if active_member else "No",
                    "Exit" if prediction[0] == 1 else "Stay"
                ]
            }
        )

        st.table(summary)

        # ======================================================
        # DOWNLOAD REPORT
        # ======================================================

        csv = summary.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="customer_prediction_report.csv",
            mime="text/csv"
        )

        # ======================================================
        # MODEL INFORMATION
        # ======================================================

        st.write("---")

        with st.expander("🤖 Model Information"):

            st.write("""
### Random Forest Classifier

**Algorithm:** Random Forest Classifier

**Dataset:** Churn Modelling Dataset

**Training Samples:** 10,000 Customers

**Input Features:** 11

**Output:**
- Stay
- Exit

**Framework:** Scikit-Learn
            """)

        # ======================================================
        # DATASET PREVIEW
        # ======================================================

        with st.expander("📊 Dataset Preview"):

            st.dataframe(df.head(), use_container_width=True)

            st.write("Shape :", df.shape)

        # ======================================================
        # PREDICTION TIPS
        # ======================================================

        with st.expander("💡 Prediction Tips"):

            st.info("""
✔ Enter valid customer details.

✔ Higher age may increase churn probability.

✔ Active members usually have lower churn.

✔ Higher balance alone does not guarantee churn.

✔ This prediction is based on the trained machine learning model.
""")

        st.success("✅ Prediction Completed Successfully")
        # ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ About":

    st.title("ℹ About Customer Churn Prediction System")

    st.write("")

    left, right = st.columns([2, 1])

    with left:

        st.markdown("""
## 🏦 Project Overview

The Customer Churn Prediction System is a Machine Learning application that predicts whether a customer is likely to leave the bank.

The application uses a trained Random Forest Classifier to analyze customer information and determine whether the customer will Stay or Exit.

### Features

✅ Customer Churn Prediction

✅ Banking Dashboard

✅ Probability Prediction

✅ Risk Analysis

✅ Customer Summary

✅ Download Report

✅ Interactive User Interface

### Input Features

• Credit Score

• Age

• Tenure

• Balance

• Number of Products

• Credit Card Status

• Active Membership

• Estimated Salary

• Country

• Gender

The system helps banks identify high-risk customers and improve customer retention.
""")

    with right:

        st.info("""
### 📌 Project Details

🏦 Domain
Banking

🤖 Algorithm
Random Forest Classifier

📊 Dataset
Churn Modelling Dataset

👥 Records
10,000 Customers

💻 Framework
Streamlit

🐍 Language
Python
""")

    st.write("---")

    st.subheader("🛠 Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.success("""
🐍 Python

✔ NumPy

✔ Pandas

✔ Pickle
""")

    with c2:

        st.success("""
🤖 Machine Learning

✔ Random Forest

✔ Scikit-Learn

✔ Prediction Model
""")

    with c3:

        st.success("""
🌐 Frontend

✔ Streamlit

✔ HTML

✔ CSS
""")

    st.write("---")

    st.subheader("👩‍💻 Developer")

    st.info("""
Name : Pooja Dnyaneshwar Bhumkar

Project : Customer Churn Prediction System

Technology :

• Python

• Streamlit

• Machine Learning

• Scikit-Learn
""")

    st.write("---")

    st.markdown(
        """
<div style="
text-align:center;
padding:25px;
background:#003366;
border-radius:12px;
color:white;">

<h2>🏦 Customer Churn Prediction System</h2>

<p>
Developed using Python | Streamlit | Machine Learning
</p>

<p>
© 2026 All Rights Reserved
</p>

</div>
""",
        unsafe_allow_html=True
    )

# ======================================================
# FOOTER
# ======================================================

st.write("---")

st.markdown(
"""
<div style="
text-align:center;
color:gray;
font-size:15px;
padding:15px;
">

🏦 <b>Customer Churn Prediction System</b>

<br><br>

Made with ❤️ using Python, Streamlit & Scikit-Learn

</div>
""",
unsafe_allow_html=True
)