import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sales Intelligence 2026",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BACKGROUND IMAGE
# ============================================================

BACKGROUND_IMAGE = (
    "https://images.unsplash.com/"
    "photo-1604719312566-8912e9227c6a"
    "?auto=format&fit=crop&w=2400&q=90"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{

    background-image:
        linear-gradient(
            rgba(3, 8, 18, 0.78),
            rgba(3, 8, 18, 0.88)
        ),
        url("{BACKGROUND_IMAGE}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.main .block-container {{
    max-width: 1400px;
    padding-top: 1rem;
    padding-bottom: 4rem;
}}


/* SIDEBAR */

section[data-testid="stSidebar"] {{

    background:
        linear-gradient(
            180deg,
            rgba(3, 8, 18, 0.98),
            rgba(7, 15, 30, 0.98)
        );

    border-right:
        1px solid rgba(56,189,248,0.20);
}}

section[data-testid="stSidebar"] * {{
    color: white;
}}


/* INPUTS */

div[data-baseweb="input"] > div {{
    background: rgba(5,10,20,0.85) !important;
    border: 1px solid rgba(148,163,184,0.30) !important;
    border-radius: 10px !important;
}}

div[data-baseweb="select"] > div {{
    background: rgba(5,10,20,0.85) !important;
    border: 1px solid rgba(148,163,184,0.30) !important;
    border-radius: 10px !important;
}}

input {{
    color: white !important;
}}

label {{
    color: white !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}}


/* BUTTON */

.stButton > button {{

    width: 100%;
    height: 58px;

    border-radius: 14px;

    border: 1px solid rgba(125,211,252,0.5);

    background:
        linear-gradient(
            135deg,
            #0ea5e9,
            #2563eb
        );

    color: white;

    font-size: 15px;
    font-weight: 800;

    box-shadow:
        0 10px 30px rgba(14,165,233,0.35);

    transition: 0.3s;
}}

.stButton > button:hover {{

    transform: translateY(-3px);

    background:
        linear-gradient(
            135deg,
            #38bdf8,
            #1d4ed8
        );

    box-shadow:
        0 15px 40px rgba(14,165,233,0.55);
}}


/* METRICS */

div[data-testid="stMetric"] {{

    background:
        rgba(5,10,20,0.72);

    border:
        1px solid rgba(255,255,255,0.14);

    border-radius: 15px;

    padding: 15px;
}}

div[data-testid="stMetricLabel"] {{
    color: #94a3b8 !important;
}}

div[data-testid="stMetricValue"] {{
    color: white !important;
}}


/* EXPANDER */

div[data-testid="stExpander"] {{

    background:
        rgba(5,10,20,0.75);

    border:
        1px solid rgba(255,255,255,0.12);

    border-radius: 15px;
}}


/* TABS */

button[data-baseweb="tab"] {{
    color: #cbd5e1 !important;
    font-weight: 700 !important;
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# IMPORTANT:
# st.html() is used instead of st.markdown()
# ============================================================

st.html(
    """
    <div style="
        text-align:center;
        padding:25px 10px 35px 10px;
    ">

        <div style="
            display:inline-block;
            padding:9px 20px;
            border-radius:30px;
            background:rgba(56,189,248,0.15);
            border:1px solid rgba(56,189,248,0.5);
            color:#7dd3fc;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
            margin-bottom:15px;
        ">
            🧠 AI RETAIL INTELLIGENCE PLATFORM
        </div>


        <div style="
            font-size:50px;
            font-weight:800;
            color:white;
            line-height:1.1;
            text-shadow:0 5px 25px rgba(0,0,0,0.8);
        ">
            Sales
            <span style="
                color:#38bdf8;
                text-shadow:0 0 25px rgba(56,189,248,0.6);
            ">
                Intelligence
            </span>
            2026
        </div>


        <div style="
            color:#cbd5e1;
            font-size:14px;
            margin-top:15px;
            line-height:1.8;
            letter-spacing:0.5px;
        ">
            Predict • Analyze • Optimize • Grow
            <br>
            K-Nearest Neighbors Regression for Retail Sales Forecasting
        </div>

    </div>
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("KNN_reg_outlet_sales.csv")

    # Item Weight
    if "Item_Weight" in df.columns:

        df["Item_Weight"] = (
            df["Item_Weight"]
            .fillna(
                df["Item_Weight"].median()
            )
        )

    # Fat Content
    if "Item_Fat_Content" in df.columns:

        df["Item_Fat_Content"] = (
            df["Item_Fat_Content"]
            .replace({
                "Low Fat": 0,
                "Regular": 1,
                "LF": 0,
                "reg": 1,
                "low fat": 0
            })
        )

    # Outlet Size
    if "Outlet_Size" in df.columns:

        df["Outlet_Size"] = (
            df["Outlet_Size"]
            .replace({
                "Small": 0,
                "Medium": 1,
                "High": 2
            })
        )

        df["Outlet_Size"] = (
            df["Outlet_Size"]
            .fillna(1)
        )

    # Location
    if "Outlet_Location_Type" in df.columns:

        df["Outlet_Location_Type"] = (
            df["Outlet_Location_Type"]
            .replace({
                "Tier 1": 0,
                "Tier 2": 1,
                "Tier 3": 2
            })
        )

    # One-hot encoding
    categorical_columns = [
        "Item_Type",
        "Outlet_Identifier",
        "Outlet_Type"
    ]

    existing = [
        col
        for col in categorical_columns
        if col in df.columns
    ]

    df = pd.get_dummies(
        df,
        columns=existing,
        drop_first=True
    )

    # Remove identifier
    if "Item_Identifier" in df.columns:

        df.drop(
            "Item_Identifier",
            axis=1,
            inplace=True
        )

    return df


# ============================================================
# DATASET
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"❌ Error loading dataset: {e}"
    )

    st.stop()


if "Item_Outlet_Sales" not in df.columns:

    st.error(
        "❌ Item_Outlet_Sales column not found."
    )

    st.stop()


# ============================================================
# FEATURES
# ============================================================

X = df.drop(
    "Item_Outlet_Sales",
    axis=1
)

y = df[
    "Item_Outlet_Sales"
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 AI Model Control"
    )

    st.write(
        "Tune the KNN regression engine "
        "and observe performance."
    )

    st.divider()

    k_neighbors = st.slider(
        "K Neighbors",
        1,
        25,
        7
    )

    p_metric = st.selectbox(
        "Distance Metric",
        [2, 1],

        format_func=lambda x:
        "Euclidean Distance"
        if x == 2
        else
        "Manhattan Distance"
    )

    test_size = st.slider(
        "Test Dataset Ratio",
        0.10,
        0.30,
        0.15,
        0.05
    )


# ============================================================
# MODEL
# ============================================================

scaler = MinMaxScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=test_size,
    random_state=42
)


model = KNeighborsRegressor(
    n_neighbors=k_neighbors,
    p=p_metric
)


model.fit(
    X_train,
    y_train
)


y_pred = model.predict(
    X_test
)


r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    np.mean(
        (y_test - y_pred) ** 2
    )
)


# ============================================================
# SIDEBAR METRICS
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown(
        "### 📊 Model Performance"
    )

    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    st.metric(
        "MAE",
        f"₹ {mae:,.0f}"
    )

    st.metric(
        "RMSE",
        f"₹ {rmse:,.0f}"
    )

    st.success(
        "🟢 MODEL ONLINE"
    )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.html(
    """
    <div style="
        display:flex;
        align-items:center;
        gap:15px;
        margin:15px 0 20px 0;
    ">

        <div style="
            color:white;
            font-size:20px;
            font-weight:800;
        ">
            📊 Executive Overview
        </div>

        <div style="
            height:2px;
            flex:1;
            background:linear-gradient(
                90deg,
                #38bdf8,
                transparent
            );
        "></div>

    </div>
    """
)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


def kpi_card(icon, title, value):

    st.html(
        f"""
        <div style="
            background:rgba(5,10,20,0.78);
            border:1px solid rgba(255,255,255,0.14);
            border-radius:18px;
            padding:20px;
            min-height:115px;
            box-shadow:0 12px 30px rgba(0,0,0,0.3);
        ">

            <div style="
                font-size:25px;
            ">
                {icon}
            </div>

            <div style="
                color:#94a3b8;
                font-size:10px;
                font-weight:800;
                letter-spacing:1.3px;
                margin-top:8px;
            ">
                {title}
            </div>

            <div style="
                color:#ffffff;
                font-size:26px;
                font-weight:800;
                margin-top:5px;
            ">
                {value}
            </div>

        </div>
        """
    )


with c1:

    kpi_card(
        "📦",
        "DATASET RECORDS",
        f"{len(df):,}"
    )


with c2:

    kpi_card(
        "🧬",
        "MODEL FEATURES",
        X.shape[1]
    )


with c3:

    kpi_card(
        "🎯",
        "ACTIVE K VALUE",
        k_neighbors
    )


with c4:

    kpi_card(
        "📈",
        "R² SCORE",
        f"{r2:.3f}"
    )


st.write("")


# ============================================================
# PRODUCT INPUTS
# ============================================================

st.html(
    """
    <div style="
        color:white;
        font-size:20px;
        font-weight:800;
        margin:25px 0 20px 0;
    ">
        📦 Product Intelligence
    </div>
    """
)


c1, c2, c3 = st.columns(3)


with c1:

    item_weight = st.slider(
        "Item Weight (kg)",
        0.0,
        50.0,
        12.5,
        0.1
    )

    item_fat = st.selectbox(
        "Fat Classification",
        [
            "Low Fat",
            "Regular"
        ]
    )


with c2:

    item_visibility = st.slider(
        "Visibility Index",
        0.0,
        0.35,
        0.06,
        0.005
    )

    item_mrp = st.slider(
        "Maximum Retail Price (₹)",
        30.0,
        300.0,
        141.8,
        1.0
    )


with c3:

    item_type = st.selectbox(
        "Product Category",
        [
            "Baking Goods",
            "Breads",
            "Breakfast",
            "Canned",
            "Dairy",
            "Frozen Foods",
            "Fruits and Vegetables",
            "Hard Drinks",
            "Health and Hygiene",
            "Household",
            "Meat",
            "Others",
            "Seafood",
            "Snack Foods",
            "Soft Drinks",
            "Starchy Foods"
        ]
    )


# ============================================================
# OUTLET INPUTS
# ============================================================

st.html(
    """
    <div style="
        color:white;
        font-size:20px;
        font-weight:800;
        margin:30px 0 20px 0;
    ">
        🏪 Store Intelligence
    </div>
    """
)


c1, c2, c3 = st.columns(3)


with c1:

    outlet_identifier = st.selectbox(
        "Outlet Identifier",
        [
            "OUT010",
            "OUT013",
            "OUT017",
            "OUT018",
            "OUT019",
            "OUT027",
            "OUT035",
            "OUT045",
            "OUT046",
            "OUT049"
        ]
    )

    establishment_year = st.slider(
        "Establishment Year",
        1985,
        2026,
        1999
    )


with c2:

    outlet_size = st.selectbox(
        "Outlet Size",
        [
            "Small",
            "Medium",
            "High"
        ]
    )

    outlet_location = st.selectbox(
        "Location Tier",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )


with c3:

    outlet_type = st.selectbox(
        "Outlet Format",
        [
            "Grocery Store",
            "Supermarket Type1",
            "Supermarket Type2",
            "Supermarket Type3"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict = st.button(
    "🚀 RUN AI SALES PREDICTION"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # Encode values

    fat_val = (
        0
        if item_fat == "Low Fat"
        else 1
    )

    size_val = {
        "Small": 0,
        "Medium": 1,
        "High": 2
    }[outlet_size]

    loc_val = {
        "Tier 1": 0,
        "Tier 2": 1,
        "Tier 3": 2
    }[outlet_location]


    # Input dataframe

    input_df = pd.DataFrame({

        "Item_Weight":
            [item_weight],

        "Item_Fat_Content":
            [fat_val],

        "Item_Visibility":
            [item_visibility],

        "Item_MRP":
            [item_mrp],

        "Outlet_Establishment_Year":
            [establishment_year],

        "Outlet_Size":
            [size_val],

        "Outlet_Location_Type":
            [loc_val]

    })


    # Add missing columns

    for col in X.columns:

        if col not in input_df.columns:

            input_df[col] = 0


    # Item Type

    item_col = (
        "Item_Type_"
        + item_type
    )

    if item_col in input_df.columns:

        input_df[item_col] = 1


    # Outlet Identifier

    outlet_col = (
        "Outlet_Identifier_"
        + outlet_identifier
    )

    if outlet_col in input_df.columns:

        input_df[outlet_col] = 1


    # Outlet Type

    outlet_type_col = (
        "Outlet_Type_"
        + outlet_type
    )

    if outlet_type_col in input_df.columns:

        input_df[outlet_type_col] = 1


    # Correct order

    input_df = input_df.reindex(
        columns=X.columns,
        fill_value=0
    )


    # Scale

    input_scaled = scaler.transform(
        input_df
    )


    # Predict

    prediction = model.predict(
        input_scaled
    )[0]


    lower = max(
        0,
        prediction - mae
    )

    upper = (
        prediction + mae
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.html(
        f"""
        <div style="
            margin-top:30px;
            padding:35px;
            text-align:center;

            background:
                linear-gradient(
                    135deg,
                    rgba(8,47,73,0.96),
                    rgba(15,23,42,0.96)
                );

            border:
                1px solid rgba(56,189,248,0.65);

            border-radius:22px;

            box-shadow:
                0 20px 55px rgba(0,0,0,0.55);
        ">

            <div style="
                color:#7dd3fc;
                font-size:12px;
                font-weight:800;
                letter-spacing:2px;
            ">
                ✨ AI FORECAST RESULT
            </div>


            <div style="
                color:white;
                font-size:58px;
                font-weight:800;
                margin:10px 0;
                text-shadow:
                    0 0 30px
                    rgba(56,189,248,0.6);
            ">
                ₹ {prediction:,.2f}
            </div>


            <div style="
                color:#94a3b8;
                font-size:13px;
            ">
                Estimated Item Outlet Sales
            </div>


            <div style="
                color:#cbd5e1;
                font-size:13px;
                margin-top:15px;
            ">
                Expected range:
                <b style="color:#38bdf8;">
                    ₹ {lower:,.0f}
                    —
                    ₹ {upper:,.0f}
                </b>
            </div>

        </div>
        """
    )


# ============================================================
# ANALYTICS
# ============================================================

st.html(
    """
    <div style="
        color:white;
        font-size:20px;
        font-weight:800;
        margin:35px 0 20px 0;
    ">
        📊 AI Analytics Center
    </div>
    """
)


tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Actual vs Predicted",
        "🔥 Correlation Matrix",
        "📈 Sales Distribution"
    ]
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

with tab1:

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            name="Predictions",

            marker=dict(
                size=7,
                color="#38bdf8",
                opacity=0.65
            )
        )
    )


    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )


    fig.add_trace(
        go.Scatter(
            x=[
                minimum,
                maximum
            ],

            y=[
                minimum,
                maximum
            ],

            mode="lines",

            name="Perfect Prediction",

            line=dict(
                color="#fbbf24",
                dash="dash",
                width=2
            )
        )
    )


    fig.update_layout(

        title="Actual vs Predicted Outlet Sales",

        xaxis_title="Actual Sales (₹)",

        yaxis_title="Predicted Sales (₹)",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white"
        ),

        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CORRELATION
# ============================================================

with tab2:

    columns = [
        "Item_Weight",
        "Item_Visibility",
        "Item_MRP",
        "Item_Outlet_Sales"
    ]

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    corr = (
        df[columns]
        .corr()
    )


    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Feature Correlation Matrix",
        color_continuous_scale="Blues"
    )


    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white"
        ),

        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SALES DISTRIBUTION
# ============================================================

with tab3:

    fig = px.histogram(

        df,

        x="Item_Outlet_Sales",

        nbins=40,

        title="Outlet Sales Distribution",

        color_discrete_sequence=[
            "#38bdf8"
        ]
    )


    fig.update_layout(

        xaxis_title="Outlet Sales (₹)",

        yaxis_title="Number of Products",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white"
        ),

        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

with st.expander(
    "🧠 Advanced Model Diagnostics"
):

    a, b, c = st.columns(3)


    with a:

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

        st.caption(
            "Variance explained by KNN."
        )


    with b:

        st.metric(
            "Mean Absolute Error",
            f"₹ {mae:,.2f}"
        )

        st.caption(
            "Average prediction error."
        )


    with c:

        st.metric(
            "RMSE",
            f"₹ {rmse:,.2f}"
        )

        st.caption(
            "Penalizes larger errors."
        )


# ============================================================
# DATASET
# ============================================================

with st.expander(
    "📋 Explore Dataset"
):

    st.dataframe(
        df.head(30),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        text-align:center;
        padding:35px 10px;
        color:#94a3b8;
        font-size:12px;
    ">

        <span style="
            color:#38bdf8;
            font-weight:800;
        ">
            🛒 SALES INTELLIGENCE 2026
        </span>

        &nbsp; • &nbsp;

        KNN Regression

        &nbsp; • &nbsp;

        Streamlit + Scikit-learn

    </div>
    """
)