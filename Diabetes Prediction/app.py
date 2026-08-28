import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk AI Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #38bdf8;
        margin-top: 5px;
    }
    .risk-high {
        background: rgba(225, 29, 72, 0.15);
        border: 1px solid #f43f5e;
        padding: 20px;
        border-radius: 12px;
        color: #fda4af;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        padding: 20px;
        border-radius: 12px;
        color: #6ee7b7;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA & MODEL PIPELINE (CACHED)
# =========================================================
@st.cache_data
def load_data():
    # Replace with path or st.file_uploader for dynamic input
    return pd.read_csv("diabetes.csv")

@st.cache_resource
def train_model(df, n_estimators, max_depth):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Data Preprocessing: Handling structural zeros in clinical measurements
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean = df.copy()
    for col in zero_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    X_clean = df_clean.drop("Outcome", axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, X_clean, X_test_scaled, y_test, y_pred, acc

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🏠 Executive Dashboard", "🤖 Patient Risk Analytics", "📊 Exploratory Data Hub", "📈 Model Diagnostics"]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Model Configuration")
n_estimators = st.sidebar.slider("Forest Estimators", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("Tree Max Depth", 2, 20, 8)

model, scaler, X, X_test_scaled, y_test, y_pred, accuracy = train_model(df, n_estimators, max_depth)

# =========================================================
# 1. EXECUTIVE DASHBOARD
# =========================================================
if page == "🏠 Executive Dashboard":
    st.title("🩺 Diabetes Analytics Platform")
    st.caption("AI-driven diagnostic modeling and clinical risk visualization.")
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Records</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Clinical Features</div><div class="metric-value">{X.shape[1]}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Diabetic Ratio</div><div class="metric-value">{df["Outcome"].mean()*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Model Accuracy</div><div class="metric-value">{accuracy*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Target Distribution")
        fig_target = px.pie(
            df, names=df["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}),
            hole=0.5, color_discrete_sequence=["#10b981", "#f43f5e"]
        )
        fig_target.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_target, use_container_width=True)

    with c2:
        st.subheader("Glucose Level Dispersion")
        fig_gluc = px.box(
            df, x="Outcome", y="Glucose", color="Outcome",
            color_discrete_sequence=["#10b981", "#f43f5e"],
            labels={0: "Non-Diabetic", 1: "Diabetic"}
        )
        fig_gluc.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_gluc, use_container_width=True)

# =========================================================
# 2. PATIENT RISK ANALYTICS (PREDICTION)
# =========================================================
elif page == "🤖 Patient Risk Analytics":
    st.title("🤖 Patient Risk Inference")
    st.caption("Input clinical parameters to compute real-time probability vectors.")
    st.divider()

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
            bp = st.number_input("Blood Pressure (mmHg)", 0, 200, 70)
            skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01)
            age = st.slider("Age (Years)", 1, 120, 33)

        submit = st.form_submit_button("Run Risk Assessment", type="primary", use_container_width=True)

    if submit:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        
        prob = model.predict_proba(input_scaled)[0][1]
        
        st.subheader("Inference Summary")
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Predicted Diabetes Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#f43f5e" if prob > 0.5 else "#10b981"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(244, 63, 94, 0.2)"}
                ]
            }
        ))
        gauge_fig.update_layout(template="plotly_dark", height=300, margin=dict(t=50, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

        if prob > 0.5:
            st.markdown(f'<div class="risk-high"><b>⚠️ High Risk Detected</b><br>The statistical probability of diabetes is {prob*100:.1f}%. Recommend immediate clinical follow-up.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><b>✅ Low Risk Detected</b><br>The statistical probability of diabetes is {prob*100:.1f}%. Regular health monitoring recommended.</div>', unsafe_allow_html=True)

# =========================================================
# 3. EXPLORATORY DATA HUB
# =========================================================
elif page == "📊 Exploratory Data Hub":
    st.title("📊 Multi-Dimensional Data Exploration")
    st.divider()

    tab1, tab2 = st.tabs(["Correlation Matrix", "Feature Distribution"])

    with tab1:
        corr = df.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="Viridis"
        )
        fig_corr.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab2:
        selected_feature = st.selectbox("Select Feature to Inspect", X.columns)
        fig_dist = px.histogram(
            df, x=selected_feature, color="Outcome",
            barmode="overlay", marginal="rug",
            color_discrete_sequence=["#10b981", "#f43f5e"]
        )
        fig_dist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_dist, use_container_width=True)

# =========================================================
# 4. MODEL DIAGNOSTICS
# =========================================================
elif page == "📈 Model Diagnostics":
    st.title("📈 Model Evaluation & Feature Attribution")
    st.divider()

    col1, col2 = c1, c2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Non-Diabetic", "Diabetic"],
            y=["Non-Diabetic", "Diabetic"],
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(template="plotly_dark", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Global Feature Importance")
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(
            importances, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale="teal"
        )
        fig_imp.update_layout(template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)