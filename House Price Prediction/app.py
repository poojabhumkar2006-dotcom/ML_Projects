import streamlit as st
import pandas as pd
import pickle
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EstateAI | House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "house_price_model.pkl"
)

VECTORIZER_FILE = os.path.join(
    BASE_DIR,
    "vectorizer.pkl"
)

ENCODER_FILE = os.path.join(
    BASE_DIR,
    "encoder.pkl"
)

FEATURES_FILE = os.path.join(
    BASE_DIR,
    "features.pkl"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "prediction_history.csv"
)


# ============================================================
# ONLINE BACKGROUND IMAGE
# ============================================================

BACKGROUND_IMAGE = (
    "https://images.unsplash.com/"
    "photo-1600585154340-be6161a56a0c"
    "?auto=format&fit=crop&w=2000&q=85"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {{
        background-color: #0B1220;

        background-image:
            linear-gradient(
                rgba(8, 15, 28, 0.60),
                rgba(8, 15, 28, 0.88)
            ),
            url("{BACKGROUND_IMAGE}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                #101827 0%,
                #18243A 100%
            );
    }}

    section[data-testid="stSidebar"] h1 {{
        color: #FFD166 !important;
        font-weight: 800 !important;
    }}

    section[data-testid="stSidebar"] h2 {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] p {{
        color: #D7DEE8 !important;
    }}


    /* ========================================================
       SIDEBAR BUTTONS
       ======================================================== */

    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;

        background-color: #FFFFFF !important;

        color: #172033 !important;

        border: none !important;

        border-radius: 14px !important;

        min-height: 55px;

        font-size: 16px !important;

        font-weight: 700 !important;

        box-shadow:
            0 5px 15px rgba(0, 0, 0, 0.20);

        transition: all 0.25s ease;
    }}

    section[data-testid="stSidebar"]
    .stButton > button p {{
        color: #172033 !important;
        font-weight: 700 !important;
    }}

    section[data-testid="stSidebar"]
    .stButton > button:hover {{
        background-color: #FFD166 !important;

        color: #111827 !important;

        transform: translateX(4px);

        box-shadow:
            0 8px 20px rgba(255, 209, 102, 0.35);
    }}

    section[data-testid="stSidebar"]
    .stButton > button:hover p {{
        color: #111827 !important;
    }}


    /* ========================================================
       MAIN TEXT
       ======================================================== */

    h1 {{
        color: #FFFFFF !important;

        font-weight: 800 !important;

        text-shadow:
            0 3px 10px rgba(0, 0, 0, 0.8);
    }}

    h2 {{
        color: #FFD166 !important;

        font-weight: 800 !important;

        text-shadow:
            0 2px 8px rgba(0, 0, 0, 0.7);
    }}

    h3 {{
        color: #FFFFFF !important;

        font-weight: 700 !important;
    }}

    .stApp p {{
        color: #F1F5F9;
    }}


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    [data-testid="stMetric"] {{
        background:
            rgba(15, 23, 42, 0.90);

        border:
            1px solid rgba(255, 255, 255, 0.15);

        border-radius: 16px;

        padding: 18px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.30);
    }}

    [data-testid="stMetricLabel"] {{
        color: #CBD5E1 !important;

        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: #FFD166 !important;

        font-weight: 800 !important;
    }}


    /* ========================================================
       MAIN BUTTONS
       ======================================================== */

    .stButton > button {{
        background-color: #FFFFFF !important;

        color: #172033 !important;

        border: none !important;

        border-radius: 12px !important;

        min-height: 48px;

        font-size: 16px !important;

        font-weight: 700 !important;

        box-shadow:
            0 5px 15px rgba(0, 0, 0, 0.22);

        transition: all 0.25s ease;
    }}

    .stButton > button p {{
        color: #172033 !important;

        font-weight: 700 !important;
    }}

    .stButton > button:hover {{
        background-color: #FFD166 !important;

        color: #111827 !important;

        transform: translateY(-2px);
    }}

    .stButton > button[kind="primary"] {{
        background:
            linear-gradient(
                135deg,
                #FF4B4B,
                #FF7043
            ) !important;

        color: #FFFFFF !important;
    }}

    .stButton > button[kind="primary"] p {{
        color: #FFFFFF !important;
    }}


    /* ========================================================
       INPUTS
       ======================================================== */

    input,
    textarea {{
        color: #FFFFFF !important;

        background-color: #172033 !important;
    }}

    div[data-baseweb="select"] {{
        background-color: #172033 !important;
    }}


    /* ========================================================
       CARDS
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background:
            rgba(15, 23, 42, 0.82);

        border:
            1px solid rgba(255, 255, 255, 0.14);

        border-radius: 16px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.30);
    }}


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {{
        color: #FFFFFF !important;

        font-weight: 600 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #FFD166 !important;
    }}


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {{
        border-color:
            rgba(255, 255, 255, 0.18);
    }}


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    [data-testid="stProgressBar"] > div > div {{
        background-color: #FFD166 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_models():

    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_FILE, "rb") as file:
        vectorizer = pickle.load(file)

    with open(ENCODER_FILE, "rb") as file:
        encoder = pickle.load(file)

    with open(FEATURES_FILE, "rb") as file:
        features = pickle.load(file)

    return model, vectorizer, encoder, features


# ============================================================
# MODEL STATUS
# ============================================================

try:

    model, dv, encoder, features = load_models()

    model_loaded = True

    model_error = None

except Exception as e:

    model_loaded = False

    model_error = str(e)

    model = None
    dv = None
    encoder = None
    features = None


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Dashboard"


if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "prediction_data" not in st.session_state:

    st.session_state.prediction_data = None


if "property_score" not in st.session_state:

    st.session_state.property_score = None


# ============================================================
# HELPER FUNCTION - SAVE HISTORY
# ============================================================

def save_prediction_history(data, prediction):

    record = data.copy()

    record["Prediction_Lakhs"] = prediction

    new_data = pd.DataFrame([record])

    if os.path.exists(HISTORY_FILE):

        try:

            old_data = pd.read_csv(
                HISTORY_FILE
            )

            final_data = pd.concat(
                [
                    old_data,
                    new_data
                ],
                ignore_index=True
            )

        except Exception:

            final_data = new_data

    else:

        final_data = new_data

    final_data.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏠 EstateAI")

    st.caption(
        "Smart Real Estate Intelligence"
    )

    st.divider()

    st.subheader("Navigation")


    if st.button(
        "🏠  Dashboard",
        use_container_width=True
    ):

        st.session_state.page = "Dashboard"

        st.rerun()


    if st.button(
        "🔮  Predict Price",
        use_container_width=True
    ):

        st.session_state.page = "Predict"

        st.rerun()


    if st.button(
        "📊  Analytics",
        use_container_width=True
    ):

        st.session_state.page = "Analytics"

        st.rerun()


    if st.button(
        "💰  EMI Calculator",
        use_container_width=True
    ):

        st.session_state.page = "Finance"

        st.rerun()


    if st.button(
        "📋  Prediction History",
        use_container_width=True
    ):

        st.session_state.page = "History"

        st.rerun()


    st.divider()

    st.subheader("🤖 Model Status")


    if model_loaded:

        st.success(
            "🟢 Model Online"
        )

    else:

        st.error(
            "🔴 Model Offline"
        )

        with st.expander(
            "Show model error"
        ):

            st.code(
                model_error
            )


    st.info(
        "Decision Tree Regressor"
    )

    st.caption(
        "19 property features"
    )

    st.divider()

    st.caption(
        "EstateAI • Smart Property Intelligence"
    )


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard():

    st.title(
        "🏠 EstateAI"
    )

    st.subheader(
        "Smart House Price Prediction"
    )

    st.write(
        "Predict property prices, analyze real-estate "
        "features and make smarter property decisions "
        "using Machine Learning."
    )

    st.write("")


    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        if st.button(
            "🔮 Predict House Price",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = "Predict"

            st.rerun()


    with col2:

        if st.button(
            "📊 Explore Analytics",
            use_container_width=True
        ):

            st.session_state.page = "Analytics"

            st.rerun()


    with col3:

        if st.button(
            "💰 Calculate EMI",
            use_container_width=True
        ):

            st.session_state.page = "Finance"

            st.rerun()


    st.write("")

    st.divider()


    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.subheader(
        "📌 EstateAI Overview"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🤖 AI Model",
            "Decision Tree"
        )


    with col2:

        st.metric(
            "📊 Features",
            "19"
        )


    with col3:

        st.metric(
            "⚡ Prediction",
            "Real-Time"
        )


    with col4:

        if model_loaded:

            st.metric(
                "🟢 Status",
                "Online"
            )

        else:

            st.metric(
                "🔴 Status",
                "Offline"
            )


    st.divider()


    # --------------------------------------------------------
    # LATEST PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "💰 Latest Property Valuation"
    )


    if st.session_state.prediction is not None:

        data = st.session_state.prediction_data

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Estimated Price",
                f"₹ {st.session_state.prediction:,.2f} L"
            )


        with col2:

            st.metric(
                "📍 Location",
                data["City"].title()
            )


        with col3:

            st.metric(
                "📐 Property Area",
                f"{data['Size_in_SqFt']:,} Sq.Ft"
            )

    else:

        st.info(
            "🏠 No prediction available yet. "
            "Click 'Predict House Price' to start."
        )


    st.divider()


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader(
        "✨ Why Use EstateAI?"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        with st.container(border=True):

            st.markdown("## 🔮")

            st.markdown(
                "### AI Price Prediction"
            )

            st.write(
                "Estimate house prices using "
                "your trained Machine Learning model."
            )


    with col2:

        with st.container(border=True):

            st.markdown("## 📊")

            st.markdown(
                "### Property Analytics"
            )

            st.write(
                "Understand property characteristics "
                "and nearby facilities."
            )


    with col3:

        with st.container(border=True):

            st.markdown("## 💰")

            st.markdown(
                "### EMI Calculator"
            )

            st.write(
                "Calculate monthly EMI, total interest "
                "and total loan repayment."
            )


    st.divider()


    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    st.subheader(
        "⚡ How EstateAI Works"
    )


    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1️⃣ Enter",
            "2️⃣ Process",
            "3️⃣ Predict",
            "4️⃣ Analyze"
        ]
    )


    with tab1:

        st.write(
            "Enter location, BHK, property size, "
            "age and other property details."
        )


    with tab2:

        st.write(
            "EstateAI converts the entered information "
            "into the format required by the ML model."
        )


    with tab3:

        st.write(
            "The Decision Tree Regressor predicts "
            "the estimated property price."
        )


    with tab4:

        st.write(
            "View the estimated value, property score "
            "and smart property insights."
        )


# ============================================================
# PREDICTION PAGE
# ============================================================

def show_prediction():

    st.title(
        "🔮 AI House Price Predictor"
    )

    st.write(
        "Enter property details to calculate "
        "the estimated house price."
    )

    st.divider()


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader(
            "📍 Location"
        )

        col1, col2 = st.columns(2)


        with col1:

            state = st.selectbox(
                "State",
                [
                    "maharashtra",
                    "karnataka",
                    "gujarat",
                    "delhi",
                    "tamil nadu"
                ]
            )


        with col2:

            city = st.text_input(
                "City",
                placeholder="Example: Pune"
            )


    st.write("")


    # --------------------------------------------------------
    # PROPERTY DETAILS
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader(
            "🏠 Property Details"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            property_type = st.selectbox(
                "Property Type",
                [
                    "Apartment",
                    "Independent House",
                    "Villa"
                ]
            )


        with col2:

            bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=2
            )


        with col3:

            size_sqft = st.number_input(
                "Size (Sq.Ft)",
                min_value=300,
                max_value=10000,
                value=1200
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            price_sqft = st.number_input(
                "Price Per Sq.Ft (₹)",
                min_value=1000,
                value=5000
            )


        with col2:

            year = st.number_input(
                "Year Built",
                min_value=1980,
                max_value=2026,
                value=2018
            )


        with col3:

            age = st.number_input(
                "Age of Property",
                min_value=0,
                max_value=100,
                value=5
            )


    st.write("")


    # --------------------------------------------------------
    # BUILDING DETAILS
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader(
            "🏗️ Building Details"
        )

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            furnished = st.selectbox(
                "Furnished Status",
                [
                    "Unfurnished",
                    "Semi-furnished",
                    "Furnished"
                ]
            )


        with col2:

            floor = st.number_input(
                "Floor Number",
                min_value=0,
                value=2
            )


        with col3:

            total_floor = st.number_input(
                "Total Floors",
                min_value=1,
                value=10
            )


        with col4:

            facing = st.selectbox(
                "Facing",
                [
                    "South",
                    "East",
                    "West",
                    "North"
                ]
            )


    st.write("")


    # --------------------------------------------------------
    # FACILITIES
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader(
            "✨ Facilities"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            school = st.number_input(
                "Nearby Schools",
                min_value=0,
                value=5
            )


        with col2:

            hospital = st.number_input(
                "Nearby Hospitals",
                min_value=0,
                value=3
            )


        with col3:

            transport = st.selectbox(
                "Public Transport",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            parking = st.selectbox(
                "Parking Space",
                [
                    "Yes",
                    "No"
                ]
            )


        with col2:

            security = st.selectbox(
                "Security",
                [
                    "No",
                    "Yes"
                ]
            )


        with col3:

            amenities = st.text_input(
                "Amenities",
                placeholder="Gym, Lift, Garden..."
            )


    st.write("")


    # --------------------------------------------------------
    # ADDITIONAL DETAILS
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader(
            "👤 Additional Details"
        )

        col1, col2 = st.columns(2)


        with col1:

            owner = st.text_input(
                "Owner Type",
                placeholder="Example: First Owner"
            )


        with col2:

            availability = st.text_input(
                "Availability Status",
                placeholder="Example: Ready to Move"
            )


    st.divider()


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict House Price",
        type="primary",
        use_container_width=True
    ):

        if not model_loaded:

            st.error(
                "❌ Machine Learning model is not available."
            )

            return


        if city.strip() == "":

            st.warning(
                "⚠️ Please enter the city."
            )

            return


        try:

            with st.spinner(
                "🤖 Analyzing property..."
            ):

                # --------------------------------------------
                # ORDINAL ENCODING
                # --------------------------------------------

                ordinal_df = pd.DataFrame(
                    {
                        "Property_Type": [
                            property_type
                        ],

                        "Furnished_Status": [
                            furnished
                        ],

                        "Public_Transport_Accessibility": [
                            transport
                        ],

                        "Facing": [
                            facing
                        ],

                        "Security": [
                            security
                        ]
                    }
                )


                ordinal_encoded = encoder.transform(
                    ordinal_df
                )


                # --------------------------------------------
                # INPUT DATA
                # --------------------------------------------

                input_data = {

                    "State":
                        state.lower(),

                    "City":
                        city.lower(),

                    "Property_Type":
                        ordinal_encoded[0][0],

                    "BHK":
                        bhk,

                    "Size_in_SqFt":
                        size_sqft,

                    "Price_per_SqFt":
                        price_sqft,

                    "Year_Built":
                        year,

                    "Furnished_Status":
                        ordinal_encoded[0][1],

                    "Floor_No":
                        floor,

                    "Total_Floors":
                        total_floor,

                    "Age_of_Property":
                        age,

                    "Nearby_Schools":
                        school,

                    "Nearby_Hospitals":
                        hospital,

                    "Public_Transport_Accessibility":
                        ordinal_encoded[0][2],

                    "Parking_Space":
                        parking.lower(),

                    "Security":
                        ordinal_encoded[0][4],

                    "Amenities":
                        amenities.lower(),

                    "Facing":
                        ordinal_encoded[0][3],

                    "Owner_Type":
                        owner.lower(),

                    "Availability_Status":
                        availability.lower()
                }


                # --------------------------------------------
                # VECTORIZE
                # --------------------------------------------

                X = dv.transform(
                    [input_data]
                )


                # --------------------------------------------
                # PREDICT
                # --------------------------------------------

                prediction = float(
                    model.predict(X)[0]
                )


                # --------------------------------------------
                # SESSION STATE
                # --------------------------------------------

                st.session_state.prediction = prediction

                st.session_state.prediction_data = input_data


            st.success(
                "🎉 Prediction completed successfully!"
            )


            # --------------------------------------------
            # RESULT
            # --------------------------------------------

            st.subheader(
                "💰 Estimated House Price"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Estimated Price",
                    f"₹ {prediction:,.2f} Lakhs"
                )


            with col2:

                st.metric(
                    "Approximate Value",
                    f"₹ {prediction * 100000:,.0f}"
                )


            # --------------------------------------------
            # PROPERTY SCORE
            # --------------------------------------------

            score = 50


            if size_sqft >= 1500:

                score += 10


            if school >= 5:

                score += 10


            if hospital >= 3:

                score += 10


            if parking == "Yes":

                score += 5


            if security == "Yes":

                score += 5


            if age <= 5:

                score += 10


            score = min(
                score,
                100
            )


            st.session_state.property_score = score


            st.subheader(
                "🏆 Property Quality Score"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Property Score",
                    f"{score}/100"
                )


            with col2:

                st.progress(
                    score / 100
                )


            # --------------------------------------------
            # SMART INSIGHTS
            # --------------------------------------------

            st.subheader(
                "💡 Smart Insights"
            )


            col1, col2 = st.columns(2)


            with col1:

                if size_sqft >= 1500:

                    st.success(
                        "📐 Spacious property."
                    )

                else:

                    st.info(
                        "📐 Standard-sized property."
                    )


                if school >= 5:

                    st.success(
                        "🏫 Good school accessibility."
                    )

                else:

                    st.info(
                        "🏫 Moderate school accessibility."
                    )


            with col2:

                if hospital >= 3:

                    st.success(
                        "🏥 Good healthcare accessibility."
                    )

                else:

                    st.info(
                        "🏥 Moderate healthcare accessibility."
                    )


                if parking == "Yes":

                    st.success(
                        "🚗 Parking facility available."
                    )

                else:

                    st.warning(
                        "🚗 Parking facility not available."
                    )


            # --------------------------------------------
            # SAVE HISTORY
            # --------------------------------------------

            try:

                save_prediction_history(
                    input_data,
                    prediction
                )

                st.success(
                    "📋 Prediction saved to history."
                )

            except Exception as history_error:

                st.warning(
                    "Prediction worked, but history "
                    "could not be saved."
                )

                st.caption(
                    str(history_error)
                )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# ANALYTICS PAGE
# ============================================================

def show_analytics():

    st.title(
        "📊 Property Analytics"
    )

    st.write(
        "Analyze the most recent property prediction."
    )

    st.divider()


    if st.session_state.prediction_data is None:

        st.info(
            "🔮 Please make a prediction first."
        )

        return


    data = st.session_state.prediction_data

    prediction = st.session_state.prediction

    score = st.session_state.property_score


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "💰 Price",
            f"₹ {prediction:,.2f} L"
        )


    with col2:

        st.metric(
            "📐 Area",
            f"{data['Size_in_SqFt']:,} Sq.Ft"
        )


    with col3:

        st.metric(
            "🛏️ BHK",
            data["BHK"]
        )


    with col4:

        st.metric(
            "🏆 Score",
            f"{score}/100"
        )


    st.divider()


    # --------------------------------------------------------
    # PROPERTY DETAILS
    # --------------------------------------------------------

    st.subheader(
        "🏠 Property Details"
    )


    details = pd.DataFrame(
        {
            "Feature": [
                "State",
                "City",
                "Property Type",
                "BHK",
                "Size",
                "Price/Sq.Ft",
                "Year Built",
                "Furnished",
                "Floor",
                "Total Floors",
                "Age",
                "Facing",
                "Parking",
                "Security"
            ],

            "Value": [
                data["State"].title(),
                data["City"].title(),
                str(data["Property_Type"]),
                data["BHK"],
                f"{data['Size_in_SqFt']:,} Sq.Ft",
                f"₹ {data['Price_per_SqFt']:,}",
                data["Year_Built"],
                str(data["Furnished_Status"]),
                data["Floor_No"],
                data["Total_Floors"],
                data["Age_of_Property"],
                str(data["Facing"]),
                data["Parking_Space"].title(),
                str(data["Security"])
            ]
        }
    )


    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # --------------------------------------------------------
    # FACILITIES
    # --------------------------------------------------------

    st.subheader(
        "✨ Nearby Facilities"
    )


    facilities = pd.DataFrame(
        {
            "Facility": [
                "Schools",
                "Hospitals"
            ],

            "Count": [
                data["Nearby_Schools"],
                data["Nearby_Hospitals"]
            ]
        }
    )


    st.bar_chart(
        facilities.set_index(
            "Facility"
        )
    )


# ============================================================
# EMI CALCULATOR
# ============================================================

def show_finance():

    st.title(
        "💰 Home Loan EMI Calculator"
    )

    st.write(
        "Calculate monthly EMI, total interest "
        "and total repayment."
    )

    st.divider()


    if st.session_state.prediction is not None:

        default_price = float(
            st.session_state.prediction
        )

    else:

        default_price = 50.0


    col1, col2 = st.columns(2)


    with col1:

        property_price = st.number_input(
            "Property Price (₹ Lakhs)",
            min_value=1.0,
            value=default_price,
            step=1.0
        )


        down_payment = st.number_input(
            "Down Payment (₹ Lakhs)",
            min_value=0.0,
            value=10.0,
            step=1.0
        )


    with col2:

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.1,
            max_value=30.0,
            value=8.5,
            step=0.1
        )


        loan_years = st.number_input(
            "Loan Tenure (Years)",
            min_value=1,
            max_value=40,
            value=20
        )


    st.divider()


    if down_payment >= property_price:

        st.error(
            "❌ Down payment must be less than property price."
        )

        return


    loan_amount = (
        property_price -
        down_payment
    )


    principal = (
        loan_amount * 100000
    )


    monthly_rate = (
        interest_rate / 12 / 100
    )


    months = (
        loan_years * 12
    )


    if monthly_rate == 0:

        emi = principal / months

    else:

        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** months
            /
            (
                (1 + monthly_rate) ** months
                - 1
            )
        )


    total_payment = (
        emi * months
    )


    total_interest = (
        total_payment -
        principal
    )


    # --------------------------------------------------------
    # EMI SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📊 EMI Summary"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🏠 Loan Amount",
            f"₹ {loan_amount:.1f} L"
        )


    with col2:

        st.metric(
            "📅 Monthly EMI",
            f"₹ {emi:,.0f}"
        )


    with col3:

        st.metric(
            "💸 Interest",
            f"₹ {total_interest:,.0f}"
        )


    with col4:

        st.metric(
            "💰 Total Payment",
            f"₹ {total_payment:,.0f}"
        )


    st.divider()


    # --------------------------------------------------------
    # PAYMENT BREAKDOWN
    # --------------------------------------------------------

    st.subheader(
        "📈 Payment Breakdown"
    )


    payment_df = pd.DataFrame(
        {
            "Component": [
                "Principal",
                "Interest"
            ],

            "Amount": [
                principal,
                total_interest
            ]
        }
    )


    st.bar_chart(
        payment_df.set_index(
            "Component"
        )
    )


# ============================================================
# HISTORY PAGE
# ============================================================

def show_history():

    st.title(
        "📋 Prediction History"
    )

    st.write(
        "View all previously generated predictions."
    )

    st.divider()


    if not os.path.exists(
        HISTORY_FILE
    ):

        st.info(
            "📭 No prediction history available yet."
        )

        return


    try:

        history = pd.read_csv(
            HISTORY_FILE
        )

    except Exception as e:

        st.error(
            "❌ Unable to read prediction history."
        )

        st.exception(e)

        return


    if history.empty:

        st.info(
            "📭 Prediction history is empty."
        )

        return


    if "Prediction_Lakhs" not in history.columns:

        st.warning(
            "Prediction_Lakhs column is missing "
            "from the history file."
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        return


    prices = pd.to_numeric(
        history["Prediction_Lakhs"],
        errors="coerce"
    ).dropna()


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🔢 Predictions",
            len(history)
        )


    with col2:

        if not prices.empty:

            st.metric(
                "📊 Average",
                f"₹ {prices.mean():,.2f} L"
            )

        else:

            st.metric(
                "📊 Average",
                "N/A"
            )


    with col3:

        if not prices.empty:

            st.metric(
                "⬆️ Highest",
                f"₹ {prices.max():,.2f} L"
            )

        else:

            st.metric(
                "⬆️ Highest",
                "N/A"
            )


    with col4:

        if not prices.empty:

            st.metric(
                "⬇️ Lowest",
                f"₹ {prices.min():,.2f} L"
            )

        else:

            st.metric(
                "⬇️ Lowest",
                "N/A"
            )


    st.divider()


    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    if not prices.empty:

        st.subheader(
            "📈 Prediction Trend"
        )


        chart_df = pd.DataFrame(
            {
                "Prediction (Lakhs)":
                    prices.reset_index(
                        drop=True
                    )
            }
        )


        st.line_chart(
            chart_df
        )


    st.divider()


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 All Predictions"
    )


    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = history.to_csv(
        index=False
    )


    st.download_button(
        "📥 Download Prediction History",
        data=csv_data,
        file_name="estateai_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# PAGE ROUTING
# ============================================================

if st.session_state.page == "Dashboard":

    show_dashboard()


elif st.session_state.page == "Predict":

    show_prediction()


elif st.session_state.page == "Analytics":

    show_analytics()


elif st.session_state.page == "Finance":

    show_finance()


elif st.session_state.page == "History":

    show_history()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏠 EstateAI • Smart Real Estate Intelligence • "
    "Powered by Python, Pandas, Scikit-Learn and Streamlit"
)