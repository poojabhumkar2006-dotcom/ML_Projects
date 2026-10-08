import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoMetrics AI | Vehicle Analytics & Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# BACKGROUND IMAGE & VISIBILITY CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Full Page Background Image with High Visibility Overlay */
    .stApp {
        background: linear-gradient(
            rgba(15, 23, 42, 0.45), 
            rgba(15, 23, 42, 0.65)
        ), 
        url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=2070&auto=format&fit=crop") no-repeat center center fixed !important;
        background-size: cover !important;
        color: #F8FAFC !important;
    }
    
    /* Top Dashboard Title Styling */
    .app-title-top {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: #FFFFFF !important;
        text-shadow: 0px 2px 10px rgba(0, 0, 0, 0.7);
        margin-bottom: 12px;
    }

    /* General Typography */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #F8FAFC !important;
    }

    /* FIX FOR SELECTBOX & DROPDOWN TEXT VISIBILITY */
    div[data-baseweb="select"] {
        background-color: #1E293B !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }

    div[data-baseweb="select"] svg {
        fill: #38BDF8 !important;
    }

    /* Selectbox Dropdown Menu Popup Items */
    ul[data-baseweb="menu"] {
        background-color: #0F172A !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    ul[data-baseweb="menu"] li {
        color: #FFFFFF !important;
        background-color: #0F172A !important;
    }

    ul[data-baseweb="menu"] li:hover {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    /* Horizontal Radio Container Styling */
    div[data-testid="stRadio"] > label {
        display: none !important;
    }
    
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 6px 10px !important;
        border-radius: 12px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
        width: 100% !important;
    }
    
    /* Horizontal Navigation Tab Items */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1 1 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        background-color: transparent !important;
        color: #CBD5E1 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        text-align: center !important;
        padding: 8px 14px !important;
        border-radius: 8px !important;
        margin: 0 4px !important;
        border: none !important;
        transition: all 0.25s ease-in-out !important;
        cursor: pointer !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.15) !important;
    }

    /* Active Selected Navigation Tab */
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.5) !important;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    /* Reduced Height & Modern Dark Glassmorphism Header Box */
    .main-header {
        background: rgba(30, 41, 59, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        padding: 12px 20px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35) !important;
        margin-bottom: 20px !important;
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }
    .main-header p {
        color: #38BDF8 !important;
        font-size: 0.88rem !important;
        margin-top: 2px !important;
        margin-bottom: 0 !important;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-left: 5px solid #38BDF8;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: left;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .metric-card h3 {
        color: #94A3B8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-card p {
        color: #FFFFFF !important;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }

    /* Prediction Result Highlight Card */
    .prediction-box {
        background: rgba(15, 23, 42, 0.85);
        border: 2px solid #38BDF8;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.2);
        margin-top: 10px;
    }
    .prediction-title {
        color: #94A3B8 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .prediction-value {
        color: #38BDF8 !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 6px 0;
    }

    /* Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(file_path='autos_dataset.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        np.random.seed(42)
        n = 205
        df = pd.DataFrame({
            'wheel-base': np.random.uniform(86.6, 120.9, n),
            'length': np.random.uniform(141.1, 208.1, n),
            'width': np.random.uniform(60.3, 72.3, n),
            'height': np.random.uniform(47.8, 59.8, n),
            'curb-weight': np.random.uniform(1488, 4066, n),
            'num-of-cylinders': np.random.choice(['four', 'six', 'five', 'three', 'twelve', 'two', 'eight'], n),
            'engine-size': np.random.uniform(61, 326, n),
            'horsepower': np.random.choice([np.nan, 48, 68, 100, 150, 200, 288], n),
            'peak-rpm': np.random.choice([np.nan, 4150, 5000, 5500, 6000], n),
            'city-mpg': np.random.uniform(13, 49, n),
            'highway-mpg': np.random.uniform(16, 54, n),
            'price': np.random.choice([np.nan, 5118, 10000, 16500, 22000, 35000, 45000], n)
        })

    cyl_map = {'four': 4, 'six': 6, 'five': 5, 'three': 3, 'twelve': 12, 'two': 2, 'eight': 8}
    if df['num-of-cylinders'].dtype == object:
        df['num-of-cylinders'] = df['num-of-cylinders'].map(cyl_map)

    for col in ['price', 'peak-rpm', 'horsepower']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['price'].fillna(df['price'].mean(), inplace=True)
    df['peak-rpm'].fillna(df['peak-rpm'].mean(), inplace=True)
    df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)

    feature_cols = [
        'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'num-of-cylinders', 'engine-size', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg'
    ]
    return df[feature_cols + ['price']].copy()

df = load_and_clean_data()

# -----------------------------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(data, max_depth, min_samples_leaf, splitter):
    X = data.drop('price', axis=1)
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        splitter=splitter,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    metrics = {
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
    }
    return model, metrics, X.columns.tolist()

# -----------------------------------------------------------------------------
# SIDEBAR - MODEL CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Model Controls")
    st.markdown("---")
    st.markdown("Adjust hyper-parameters for the **Decision Tree Regressor**:")
    
    param_max_depth = st.slider("Max Depth", 1, 20, 7)
    param_min_samples = st.slider("Min Samples Leaf", 1, 10, 2)
    param_splitter = st.selectbox("Splitter Strategy", ["best", "random"])

model, metrics, feature_names = train_model(df, param_max_depth, param_min_samples, param_splitter)

# -----------------------------------------------------------------------------
# TOP TITLE ABOVE DASHBOARD
# -----------------------------------------------------------------------------
st.markdown('<div class="app-title-top">🚗 AutoMetrics Analytics Dashboard</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HORIZONTAL TOP NAVIGATION BAR
# -----------------------------------------------------------------------------
navigation = st.radio(
    label="Horizontal Navigation",
    options=[
        "📊 Dashboard Insights",
        "💰 Price Predictor",
        "⚙️ Model Performance",
        "📁 Raw Data View"
    ],
    horizontal=True
)

# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD INSIGHTS
# -----------------------------------------------------------------------------
if navigation == "📊 Dashboard Insights":
    st.markdown("""
    <div class="main-header">
        <h1>Vehicle Market Analytics</h1>
        <p>Exploratory performance analysis of auto features, engine metrics, and pricing distribution curves.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Total Records</h3><p>{len(df)}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>Avg Price</h3><p>${df["price"].mean():,.0f}</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>Avg Horsepower</h3><p>{df["horsepower"].mean():.0f} HP</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>Avg City MPG</h3><p>{df["city-mpg"].mean():.1f}</p></div>', unsafe_allow_html=True)

    st.write("")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig1 = px.scatter(
            df, x='horsepower', y='price', size='engine-size', color='num-of-cylinders',
            title='<b>Horsepower vs. Price</b> (Sized by Engine Size)',
            template='plotly_dark', color_continuous_scale=px.colors.sequential.Blues
        )
        fig1.update_layout(paper_bgcolor='rgba(15, 23, 42, 0.75)', plot_bgcolor='rgba(15, 23, 42, 0.75)')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = px.box(
            df, x='num-of-cylinders', y='price', points='all',
            title='<b>Price Spread by Cylinder Count</b>',
            template='plotly_dark', color_discrete_sequence=['#38BDF8']
        )
        fig3.update_layout(paper_bgcolor='rgba(15, 23, 42, 0.75)', plot_bgcolor='rgba(15, 23, 42, 0.75)')
        st.plotly_chart(fig3, use_container_width=True)

    with col_right:
        fig2 = px.scatter(
            df, x='city-mpg', y='highway-mpg', color='price',
            title='<b>City vs. Highway Fuel Efficiency</b>',
            template='plotly_dark', color_continuous_scale=px.colors.sequential.Viridis
        )
        fig2.update_layout(paper_bgcolor='rgba(15, 23, 42, 0.75)', plot_bgcolor='rgba(15, 23, 42, 0.75)')
        st.plotly_chart(fig2, use_container_width=True)

        corr = df.corr()
        fig4 = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="<b>Correlation Heatmap</b>",
            color_continuous_scale="Blues", template="plotly_dark"
        )
        fig4.update_layout(paper_bgcolor='rgba(15, 23, 42, 0.75)', plot_bgcolor='rgba(15, 23, 42, 0.75)')
        st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 2: PRICE PREDICTOR
# -----------------------------------------------------------------------------
elif navigation == "💰 Price Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>Car Price Prediction Engine</h1>
        <p>Set custom vehicle parameters below and click predict to estimate real-time market value.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_inputs, col_results = st.columns([2, 1])
    
    with col_inputs:
        st.subheader("🛠️ Vehicle Specifications")
        
        c_i, c_ii, c_iii = st.columns(3)
        with c_i:
            inp_hp = st.slider("Horsepower", int(df['horsepower'].min()), int(df['horsepower'].max()), int(df['horsepower'].median()))
            inp_engine = st.slider("Engine Size", int(df['engine-size'].min()), int(df['engine-size'].max()), int(df['engine-size'].median()))
            inp_cylinders = st.selectbox("Cylinders", sorted(df['num-of-cylinders'].unique()))
            inp_weight = st.slider("Curb Weight (lbs)", int(df['curb-weight'].min()), int(df['curb-weight'].max()), int(df['curb-weight'].median()))
            
        with c_ii:
            inp_length = st.slider("Length", float(df['length'].min()), float(df['length'].max()), float(df['length'].median()))
            inp_width = st.slider("Width", float(df['width'].min()), float(df['width'].max()), float(df['width'].median()))
            inp_height = st.slider("Height", float(df['height'].min()), float(df['height'].max()), float(df['height'].median()))
            inp_wheelbase = st.slider("Wheel Base", float(df['wheel-base'].min()), float(df['wheel-base'].max()), float(df['wheel-base'].median()))

        with c_iii:
            inp_city = st.slider("City MPG", int(df['city-mpg'].min()), int(df['city-mpg'].max()), int(df['city-mpg'].median()))
            inp_hwy = st.slider("Highway MPG", int(df['highway-mpg'].min()), int(df['highway-mpg'].max()), int(df['highway-mpg'].median()))
            inp_rpm = st.slider("Peak RPM", int(df['peak-rpm'].min()), int(df['peak-rpm'].max()), int(df['peak-rpm'].median()))

        st.write("")
        predict_clicked = st.button("🏷️ Predict Vehicle Price", type="primary")

    with col_results:
        st.subheader("🎯 Price Estimation")
        
        input_data = pd.DataFrame([[
            inp_wheelbase, inp_length, inp_width, inp_height, inp_weight,
            inp_cylinders, inp_engine, inp_hp, inp_rpm, inp_city, inp_hwy
        ]], columns=feature_names)
        
        predicted_price = model.predict(input_data)[0]

        if predict_clicked or 'has_predicted' in st.session_state:
            st.session_state['has_predicted'] = True
            
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-title">Estimated Market Price</div>
                <div class="prediction-value">${predicted_price:,.2f}</div>
                <p style="color: #94A3B8; font-size: 0.85rem; margin:0;">DecisionTreeRegressor Estimation</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_price,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Valuation Range Gauge", 'font': {'color': '#F8FAFC', 'size': 14}},
                gauge={
                    'axis': {'range': [df['price'].min(), df['price'].max()], 'tickcolor': "#F8FAFC"},
                    'bar': {'color': "#38BDF8"},
                    'bgcolor': "rgba(15, 23, 42, 0.6)",
                    'bordercolor': "rgba(255, 255, 255, 0.2)"
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#F8FAFC"}, height=230, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Adjust the parameters on the left and click **'Predict Vehicle Price'** to run model inference.")

# -----------------------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif navigation == "⚙️ Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>Decision Tree Model Performance</h1>
        <p>Review training vs. testing accuracy metrics and feature contribution scores.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("📈 Performance Metrics")
        m_df = pd.DataFrame({
            'Metric': ['R² Score', 'MAE ($)', 'RMSE ($)'],
            'Training Set': [f"{metrics['train_r2']:.4f}", f"{metrics['train_mae']:.2f}", f"{metrics['train_rmse']:.2f}"],
            'Testing Set': [f"{metrics['test_r2']:.4f}", f"{metrics['test_mae']:.2f}", f"{metrics['test_rmse']:.2f}"]
        })
        st.table(m_df)
        
    with col_m2:
        st.subheader("🔑 Feature Importance")
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(
            importances, x='Importance', y='Feature', orientation='h',
            title="<b>Feature Importance Analysis</b>",
            template="plotly_dark",
            color='Importance',
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(height=340, paper_bgcolor='rgba(15, 23, 42, 0.75)', plot_bgcolor='rgba(15, 23, 42, 0.75)')
        st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: RAW DATA VIEW
# -----------------------------------------------------------------------------
elif navigation == "📁 Raw Data View":
    st.markdown("""
    <div class="main-header">
        <h1>Autos Dataset Table</h1>
        <p>Inspect and download the preprocessed data source.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df, use_container_width=True)
    
    st.download_button(
        label="📥 Download Cleaned CSV Dataset",
        data=df.to_csv(index=False),
        file_name="cleaned_autos_dataset.csv",
        mime="text/csv"
    )