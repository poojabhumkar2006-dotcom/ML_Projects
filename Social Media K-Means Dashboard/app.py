import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Social Media K-Means Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }

    .dashboard-subtitle {
        color: #667085;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e7eaf0;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.05);
        min-height: 115px;
    }

    .metric-label {
        color: #667085;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .info-box {
        background: white;
        padding: 15px 18px;
        border-radius: 12px;
        border: 1px solid #e7eaf0;
        margin-bottom: 10px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e7eaf0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    # Default file expected beside app.py
    return pd.read_csv("Live.csv")


def format_number(value):
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


@st.cache_data
def prepare_data(raw_df):
    df = raw_df.copy()

    # Same preprocessing as the supplied K-Means notebook
    redundant_cols = ["Column1", "Column2", "Column3", "Column4"]
    redundant_cols = [c for c in redundant_cols if c in df.columns]
    if redundant_cols:
        df = df.drop(columns=redundant_cols)

    id_cols = ["status_id", "status_published"]
    id_cols = [c for c in id_cols if c in df.columns]

    working = df.drop(columns=id_cols).copy()

    if "status_type" not in working.columns:
        raise ValueError("The dataset must contain a 'status_type' column.")

    # Keep original category labels for display
    original_status = working["status_type"].astype(str)

    # Fill missing values
    numeric_cols = working.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        working[col] = working[col].fillna(working[col].median())

    working["status_type"] = working["status_type"].fillna("Unknown").astype(str)

    # Label encode status_type
    label_encoder = LabelEncoder()
    working["status_type_encoded"] = label_encoder.fit_transform(
        working["status_type"]
    )

    # Features used by the notebook:
    # all columns after dropping ID/date columns, including encoded status_type
    feature_df = working.drop(columns=["status_type"]).copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feature_df)

    return (
        df,
        working,
        feature_df,
        X_scaled,
        label_encoder,
        original_status
    )


@st.cache_data
def calculate_elbow(X_scaled, max_k=10):
    upper = min(max_k, len(X_scaled) - 1)
    ks = list(range(2, upper + 1))
    inertias = []

    for k in ks:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=42
        )
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    return ks, inertias


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV dataset",
    type=["csv"],
    help="Upload Live.csv or another compatible social-media dataset."
)

try:
    raw_df = load_data(uploaded_file)
    (
        cleaned_df,
        working_df,
        feature_df,
        X_scaled,
        label_encoder,
        original_status
    ) = prepare_data(raw_df)
except FileNotFoundError:
    st.error(
        "Live.csv was not found. Keep Live.csv in the same folder as app.py "
        "or upload it from the sidebar."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not prepare the dataset: {e}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
max_clusters = min(10, len(X_scaled) - 1)

n_clusters = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=max_clusters,
    value=min(4, max_clusters),
    step=1
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Preprocessing follows the supplied K-Means notebook: "
    "remove unused columns → encode status type → Min-Max scaling."
)


# ---------------------------------------------------------
# K-MEANS MODEL
# ---------------------------------------------------------
model = KMeans(
    n_clusters=n_clusters,
    init="k-means++",
    n_init=10,
    random_state=random_state
)

cluster_labels = model.fit_predict(X_scaled)

result_df = cleaned_df.copy()
result_df["Cluster"] = cluster_labels

if "status_type" in result_df.columns:
    result_df["status_type"] = result_df["status_type"].fillna("Unknown").astype(str)

silhouette = silhouette_score(X_scaled, cluster_labels)

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=random_state)
pca_values = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": pca_values[:, 0],
    "PC2": pca_values[:, 1],
    "Cluster": cluster_labels.astype(str),
    "Status Type": working_df["status_type"].values
})


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="dashboard-title">📊 Social Media K-Means Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="dashboard-subtitle">'
    'Interactive clustering and engagement analysis of Facebook-style post data'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
total_posts = len(result_df)
total_clusters = result_df["Cluster"].nunique()

numeric_engagement = [
    c for c in [
        "num_reactions",
        "num_comments",
        "num_shares",
        "num_likes"
    ] if c in result_df.columns
]

total_reactions = result_df["num_reactions"].sum() if "num_reactions" in result_df else 0
total_comments = result_df["num_comments"].sum() if "num_comments" in result_df else 0
total_shares = result_df["num_shares"].sum() if "num_shares" in result_df else 0

c1, c2, c3, c4, c5 = st.columns(5)

cards = [
    (c1, "Total Posts", format_number(total_posts)),
    (c2, "Clusters", str(total_clusters)),
    (c3, "Total Reactions", format_number(total_reactions)),
    (c4, "Total Comments", format_number(total_comments)),
    (c5, "Silhouette Score", f"{silhouette:.3f}")
]

for col, label, value in cards:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("")


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "🎯 Cluster Analysis",
    "🔎 PCA Visualization",
    "📐 Elbow Method",
    "📋 Data Explorer"
])


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Engagement Overview</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        if "status_type" in result_df.columns:
            type_counts = (
                result_df["status_type"]
                .value_counts()
                .reset_index()
            )
            type_counts.columns = ["Status Type", "Posts"]

            fig_type = px.bar(
                type_counts,
                x="Status Type",
                y="Posts",
                text="Posts",
                title="Posts by Status Type"
            )
            fig_type.update_layout(
                height=420,
                xaxis_title="",
                yaxis_title="Number of Posts",
                showlegend=False
            )
            st.plotly_chart(fig_type, use_container_width=True)

    with right:
        if numeric_engagement:
            selected_metric = st.selectbox(
                "Select engagement metric",
                numeric_engagement,
                format_func=lambda x: x.replace("num_", "").replace("_", " ").title()
            )

            fig_hist = px.histogram(
                result_df,
                x=selected_metric,
                nbins=40,
                title=f"Distribution of {selected_metric.replace('num_', '').title()}"
            )
            fig_hist.update_layout(
                height=420,
                xaxis_title=selected_metric.replace("num_", "").title(),
                yaxis_title="Posts"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-title">Engagement Comparison</div>', unsafe_allow_html=True)

    engagement_cols = [
        c for c in [
            "num_reactions",
            "num_comments",
            "num_shares",
            "num_likes",
            "num_loves",
            "num_wows",
            "num_hahas",
            "num_sads",
            "num_angrys"
        ] if c in result_df.columns
    ]

    if engagement_cols:
        totals = result_df[engagement_cols].sum().sort_values(ascending=False)
        totals_df = totals.reset_index()
        totals_df.columns = ["Metric", "Total"]

        fig_eng = px.bar(
            totals_df,
            x="Metric",
            y="Total",
            text_auto=".3s",
            title="Overall Engagement Metrics"
        )
        fig_eng.update_layout(height=450)
        st.plotly_chart(fig_eng, use_container_width=True)


# =========================================================
# TAB 2 - CLUSTER ANALYSIS
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Cluster Distribution</div>', unsafe_allow_html=True)

    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_counts.columns = ["Cluster", "Posts"]

    left, right = st.columns(2)

    with left:
        fig_cluster = px.bar(
            cluster_counts,
            x="Cluster",
            y="Posts",
            text="Posts",
            title="Posts in Each Cluster"
        )
        fig_cluster.update_layout(height=400)
        st.plotly_chart(fig_cluster, use_container_width=True)

    with right:
        fig_pie = px.pie(
            cluster_counts,
            names="Cluster",
            values="Posts",
            hole=0.45,
            title="Cluster Share"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title">Cluster Profile</div>', unsafe_allow_html=True)

    profile_cols = [
        c for c in [
            "num_reactions",
            "num_comments",
            "num_shares",
            "num_likes",
            "num_loves",
            "num_wows",
            "num_hahas",
            "num_sads",
            "num_angrys"
        ] if c in result_df.columns
    ]

    if profile_cols:
        profile = result_df.groupby("Cluster")[profile_cols].mean().round(2)
        profile_display = profile.copy()

        st.dataframe(
            profile_display,
            use_container_width=True
        )

        # Normalize the cluster profile for a comparable chart
        profile_long = (
            profile.reset_index()
            .melt(id_vars="Cluster", var_name="Metric", value_name="Average")
        )

        fig_profile = px.bar(
            profile_long,
            x="Metric",
            y="Average",
            color="Cluster",
            barmode="group",
            title="Average Engagement by Cluster"
        )
        fig_profile.update_layout(
            height=500,
            xaxis_title="",
            yaxis_title="Average Value"
        )
        st.plotly_chart(fig_profile, use_container_width=True)

    if "status_type" in result_df.columns:
        st.markdown('<div class="section-title">Status Type vs Cluster</div>', unsafe_allow_html=True)

        cross_tab = pd.crosstab(
            result_df["status_type"],
            result_df["Cluster"]
        )

        fig_heat = px.imshow(
            cross_tab,
            text_auto=True,
            aspect="auto",
            title="Number of Posts by Status Type and Cluster"
        )
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, use_container_width=True)


# =========================================================
# TAB 3 - PCA
# =========================================================
with tab3:
    st.markdown('<div class="section-title">2D PCA Cluster Visualization</div>', unsafe_allow_html=True)

    st.info(
        f"PCA reduces the scaled feature space to two dimensions for visualization. "
        f"The first two components explain "
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}% of the variance."
    )

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Status Type"],
        title="K-Means Clusters in PCA Space",
        opacity=0.75
    )

    fig_pca.update_layout(
        height=650,
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2"
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown('<div class="section-title">Cluster Centroids</div>', unsafe_allow_html=True)

    centroid_df = pd.DataFrame(
        model.cluster_centers_,
        columns=feature_df.columns
    )
    centroid_df.index.name = "Cluster"

    st.dataframe(
        centroid_df.round(3),
        use_container_width=True
    )


# =========================================================
# TAB 4 - ELBOW METHOD
# =========================================================
with tab4:
    st.markdown('<div class="section-title">Elbow Method</div>', unsafe_allow_html=True)

    st.write(
        "The elbow method compares K-Means inertia for different values of K. "
        "A sharp reduction followed by a slower improvement can indicate a useful "
        "cluster count."
    )

    elbow_ks, elbow_inertias = calculate_elbow(X_scaled, max_k=10)

    elbow_df = pd.DataFrame({
        "K": elbow_ks,
        "Inertia": elbow_inertias
    })

    fig_elbow = px.line(
        elbow_df,
        x="K",
        y="Inertia",
        markers=True,
        title="K-Means Elbow Curve"
    )

    fig_elbow.update_layout(
        height=500,
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia"
    )

    st.plotly_chart(fig_elbow, use_container_width=True)

    selected_inertia = elbow_df.loc[
        elbow_df["K"] == n_clusters, "Inertia"
    ].iloc[0]

    st.success(
        f"Current setting: K = {n_clusters} | "
        f"Inertia = {selected_inertia:,.2f} | "
        f"Silhouette Score = {silhouette:.3f}"
    )


# =========================================================
# TAB 5 - DATA EXPLORER
# =========================================================
with tab5:
    st.markdown('<div class="section-title">Clustered Dataset</div>', unsafe_allow_html=True)

    filter_cluster = st.multiselect(
        "Filter by cluster",
        sorted(result_df["Cluster"].unique()),
        default=sorted(result_df["Cluster"].unique())
    )

    filter_type = None
    if "status_type" in result_df.columns:
        available_types = sorted(result_df["status_type"].dropna().unique())
        filter_type = st.multiselect(
            "Filter by status type",
            available_types,
            default=available_types
        )

    filtered = result_df[result_df["Cluster"].isin(filter_cluster)].copy()

    if filter_type is not None:
        filtered = filtered[filtered["status_type"].isin(filter_type)]

    st.write(f"Showing **{len(filtered):,}** of **{len(result_df):,}** posts.")

    st.dataframe(
        filtered,
        use_container_width=True,
        height=520
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="clustered_social_media_data.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit • K-Means Clustering • Min-Max Scaling • PCA • Plotly"
)
