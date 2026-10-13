import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="MachinePulse | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Polished dashboard styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #07111f 0%, #0b1728 55%, #101d32 100%); color: #eef4ff; }
    [data-testid="stSidebar"] { background: #0a1423; border-right: 1px solid #1d3048; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f4f7ff; }
    p, label, .stMarkdown, [data-testid="stMetricLabel"] { color: #c6d3e5; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #12243a, #0e1b2d);
        border: 1px solid #203954; border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,.15);
    }
    div[data-testid="stMetricValue"] { color: #ffffff; font-weight: 700; }
    .hero {
        padding: 28px 30px; border-radius: 22px; margin-bottom: 20px;
        background: radial-gradient(circle at 85% 10%, rgba(38, 219, 184, .18), transparent 32%),
                    linear-gradient(120deg, #122b45, #102035 70%, #142a3d);
        border: 1px solid #29465f;
    }
    .eyebrow { color: #39e6c2; text-transform: uppercase; letter-spacing: 2px; font-size: 11px; font-weight: 700; }
    .hero h1 { font-size: clamp(30px, 4vw, 44px); margin: 7px 0 8px 0; }
    .hero p { color: #c0d1e5; font-size: 15px; margin-bottom: 0; max-width: 760px; }
    .pill { display: inline-block; padding: 6px 10px; border-radius: 99px; font-size: 11px; font-weight: 700; letter-spacing: .5px; background: #123c3a; color: #62f4d3; border: 1px solid #216b63; }
    .section-label { color: #39e6c2; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px; }
    .panel { background: #0e1b2c; border: 1px solid #1e344d; border-radius: 18px; padding: 18px 20px; }
    .small-muted { color: #92a8c1; font-size: 12px; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 11px; border: 1px solid #35d8b5; font-weight: 700;
        background: linear-gradient(90deg, #16b99b, #29d9b5); color: #06131f;
        min-height: 42px; transition: all .2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: #8fffe8; transform: translateY(-1px); }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] { background: #102137; border-radius: 10px; padding: 10px 18px; }
    button[data-baseweb="tab"][aria-selected="true"] { background: #173a4b; border-bottom: 2px solid #39e6c2; }
    div[data-testid="stExpander"] { border: 1px solid #1e344d; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Model configuration — transition matrix from the first app
# ------------------------------------------------------------
STATES = ["Operational", "Minor Wear", "Major Fault", "Failure"]
STATE_CODES = ["S1", "S2", "S3", "S4"]
STATE_COLORS = {
    "Operational": "#35d6b0",
    "Minor Wear": "#f3c969",
    "Major Fault": "#ff9c5a",
    "Failure": "#ff647c",
}
P_DEFAULT = np.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.2, 0.5, 0.2, 0.1],
    [0.0, 0.3, 0.4, 0.3],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)


def simulate_chain(start_state, steps, transition_matrix, seed):
    """Return a Markov-chain state sequence including the starting state."""
    rng = np.random.default_rng(seed)
    states = [int(start_state)]
    current = int(start_state)
    for _ in range(int(steps)):
        current = int(rng.choice(len(STATES), p=transition_matrix[current]))
        states.append(current)
    return states


def forecast_distribution(start_state, steps, transition_matrix):
    """Exact state distribution after each step: pi(t+1) = pi(t) @ P."""
    distribution = np.zeros(len(STATES), dtype=float)
    distribution[int(start_state)] = 1.0
    rows = [distribution.copy()]
    for _ in range(int(steps)):
        distribution = distribution @ transition_matrix
        rows.append(distribution.copy())
    return np.asarray(rows)


def state_advice(state_name):
    advice = {
        "Operational": ("NORMAL", "Machine appears operational in this simulated state. Continue routine monitoring.", "success"),
        "Minor Wear": ("WATCH", "Minor wear is indicated. Plan an inspection and preventive maintenance.", "warning"),
        "Major Fault": ("ACTION NEEDED", "Major fault is indicated. Prioritize inspection and schedule maintenance promptly.", "warning"),
        "Failure": ("FAILURE", "The model has entered its absorbing failure state. Stop and inspect the machine according to site safety procedures.", "error"),
    }
    return advice[state_name]


def styled_plot(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c6d3e5", family="DM Sans"),
        margin=dict(l=12, r=12, t=45, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#13263b", font_color="#ffffff"),
    )
    fig.update_xaxes(gridcolor="#20334a", zerolinecolor="#20334a")
    fig.update_yaxes(gridcolor="#20334a", zerolinecolor="#20334a")
    return fig


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ MachinePulse")
    st.caption("PREDICTIVE MAINTENANCE LAB")
    st.divider()

    st.markdown("### Simulation controls")
    initial_state_name = st.selectbox("Starting condition", STATES, index=0)
    horizon = st.slider("Forecast horizon (steps)", 5, 100, 30, step=5)
    seed = st.number_input("Simulation seed", min_value=0, max_value=999999, value=42, step=1)

    st.markdown("### Model tuning")
    st.caption("Adjust transition probabilities row by row. Each row is normalized automatically.")
    with st.expander("Operational transitions", expanded=False):
        op_minor = st.slider("Operational → Minor Wear", 0, 100, 20, 5, key="op_minor")
        op_major = st.slider("Operational → Major Fault", 0, 100 - op_minor, 10, 5, key="op_major")
        op_fail = 0
        op_ok = 100 - op_minor - op_major
        st.caption(f"Remain operational: {op_ok}% · Minor wear: {op_minor}% · Major fault: {op_major}%")
    with st.expander("Minor wear transitions", expanded=False):
        mw_ok = st.slider("Minor Wear → Operational", 0, 100, 20, 5, key="mw_ok")
        mw_major = st.slider("Minor Wear → Major Fault", 0, 100 - mw_ok, 20, 5, key="mw_major")
        mw_fail = st.slider("Minor Wear → Failure", 0, 100 - mw_ok - mw_major, 10, 5, key="mw_fail")
        mw_wear = 100 - mw_ok - mw_major - mw_fail
        st.caption(f"Remain in minor wear: {mw_wear}%")
    with st.expander("Major fault transitions", expanded=False):
        mf_ok = st.slider("Major Fault → Operational", 0, 100, 0, 5, key="mf_ok")
        mf_wear = st.slider("Major Fault → Minor Wear", 0, 100 - mf_ok, 30, 5, key="mf_wear")
        mf_fail = st.slider("Major Fault → Failure", 0, 100 - mf_ok - mf_wear, 30, 5, key="mf_fail")
        mf_major = 100 - mf_ok - mf_wear - mf_fail
        st.caption(f"Remain in major fault: {mf_major}%")

    # Failure is absorbing by default; the other rows sum to 1.
    P = np.array([
        [op_ok, op_minor, op_major, op_fail],
        [mw_ok, mw_wear, mw_major, mw_fail],
        [mf_ok, mf_wear, mf_major, mf_fail],
        [0, 0, 0, 100],
    ], dtype=float) / 100.0

    run = st.button("▶ Run prediction", use_container_width=True)
    reset = st.button("↺ Reset model defaults", use_container_width=True)
    if reset:
        for key in ["op_minor", "op_major", "mw_ok", "mw_major", "mw_fail", "mf_ok", "mf_wear", "mf_fail"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.divider()
    st.markdown('<span class="small-muted">MODEL TYPE</span>', unsafe_allow_html=True)
    st.markdown("**Discrete-time Markov chain**")
    st.caption("Educational simulation · Not a certified industrial safety system")

# ------------------------------------------------------------
# Run / persist simulation
# ------------------------------------------------------------
if "mp_history" not in st.session_state:
    st.session_state.mp_history = simulate_chain(0, 30, P_DEFAULT, 42)
    st.session_state.mp_P = P_DEFAULT.copy()
    st.session_state.mp_start = 0
    st.session_state.mp_horizon = 30
    st.session_state.mp_seed = 42

if run:
    start_idx = STATES.index(initial_state_name)
    st.session_state.mp_history = simulate_chain(start_idx, horizon, P, int(seed))
    st.session_state.mp_P = P.copy()
    st.session_state.mp_start = start_idx
    st.session_state.mp_horizon = horizon
    st.session_state.mp_seed = int(seed)

history = st.session_state.mp_history
P_used = st.session_state.mp_P
start_idx = st.session_state.mp_start
horizon_used = st.session_state.mp_horizon
forecast = forecast_distribution(start_idx, horizon_used, P_used)

history_df = pd.DataFrame({
    "Time step": range(len(history)),
    "State code": [STATE_CODES[s] for s in history],
    "Machine condition": [STATES[s] for s in history],
})
final_state = STATES[history[-1]]
failure_steps = sum(s == 3 for s in history[1:])
failure_step = next((i for i, s in enumerate(history) if s == 3), None)
expected_failure_probability = float(forecast[-1, 3])
major_or_failure = float(forecast[-1, 2] + forecast[-1, 3])

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">● Intelligent asset monitoring</div>
  <h1>Predictive Machine Maintenance</h1>
  <p>Explore machine health transitions, estimate future state probabilities, and identify when preventive maintenance may be needed — powered by a Markov process.</p>
  <div style="margin-top:16px"><span class="pill">MARKOV MODEL ACTIVE</span></div>
</div>
""", unsafe_allow_html=True)

# Status banner
status_title, status_message, status_type = state_advice(final_state)
if status_type == "success":
    st.success(f"**{status_title}** — {status_message}")
elif status_type == "warning":
    st.warning(f"**{status_title}** — {status_message}")
else:
    st.error(f"**{status_title}** — {status_message}")

# Metric row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Final simulated state", final_state, delta=f"Step {len(history)-1}")
c2.metric("Failure probability at horizon", f"{expected_failure_probability:.1%}", help="Exact Markov-chain probability at the selected forecast horizon.")
c3.metric("Major fault or failure", f"{major_or_failure:.1%}", help="Probability of being in Major Fault or Failure at the selected horizon.")
c4.metric("Failure steps in simulation", f"{failure_steps}", help="Count of simulated time steps that were in the Failure state.")

tabs = st.tabs(["📊 Overview", "🔮 Probability forecast", "🧭 Transition matrix", "📋 Simulation log"])

with tabs[0]:
    st.markdown("### Machine health timeline")
    st.caption("Hover over points to inspect the machine state at each time step.")
    timeline = pd.DataFrame({
        "Time step": range(len(history)),
        "State index": history,
        "Machine condition": [STATES[s] for s in history],
    })
    fig = px.line(
        timeline, x="Time step", y="State index", markers=True,
        hover_data=["Machine condition"],
        category_orders={"Machine condition": STATES},
    )
    fig.update_traces(line=dict(color="#39e6c2", width=3), marker=dict(size=7, color="#f4f7ff"))
    fig.update_yaxes(
        tickmode="array", tickvals=[0, 1, 2, 3],
        ticktext=["Operational", "Minor Wear", "Major Fault", "Failure"],
        title="Machine condition",
    )
    fig.update_xaxes(title="Time step")
    st.plotly_chart(styled_plot(fig, 390), use_container_width=True)

    a, b = st.columns([1.15, 1])
    with a:
        st.markdown("### State distribution in this simulation")
        counts = pd.Series([STATES[s] for s in history]).value_counts().reindex(STATES, fill_value=0)
        count_df = pd.DataFrame({"Condition": counts.index, "Steps": counts.values})
        fig2 = px.bar(count_df, x="Condition", y="Steps", color="Condition",
                      color_discrete_map=STATE_COLORS, text="Steps")
        fig2.update_layout(showlegend=False)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(styled_plot(fig2, 300), use_container_width=True)
    with b:
        st.markdown("### Maintenance recommendation")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"**Predicted final condition:** {final_state}")
        if final_state == "Operational":
            st.write("Keep the regular inspection schedule and continue monitoring.")
        elif final_state == "Minor Wear":
            st.write("Plan preventive maintenance and inspect wear-related components.")
        elif final_state == "Major Fault":
            st.write("Prioritize a technical inspection and plan corrective maintenance.")
        else:
            st.write("The simulated machine reached failure. Follow site shutdown and safety procedures.")
        if failure_step is not None:
            st.write(f"First simulated failure occurs at step **{failure_step}**.")
        else:
            st.write("No failure occurred in this single simulated path.")
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("### Exact Markov-chain forecast")
    st.caption("Unlike the single simulated path, this chart shows the model's probability distribution across all possible states at each step.")
    forecast_df = pd.DataFrame(forecast, columns=STATES)
    forecast_df.insert(0, "Time step", range(len(forecast_df)))
    long_df = forecast_df.melt(id_vars="Time step", var_name="Machine condition", value_name="Probability")
    fig3 = px.area(
        long_df, x="Time step", y="Probability", color="Machine condition",
        color_discrete_map=STATE_COLORS,
        category_orders={"Machine condition": STATES},
    )
    fig3.update_yaxes(tickformat=".0%", range=[0, 1])
    fig3.update_xaxes(title="Time step")
    st.plotly_chart(styled_plot(fig3, 420), use_container_width=True)

    st.markdown("### Forecast at selected horizon")
    final_dist = pd.DataFrame({
        "Machine condition": STATES,
        "Probability": forecast[-1],
        "Probability (%)": [f"{v:.2%}" for v in forecast[-1]],
    })
    fc1, fc2 = st.columns([1, 1.3])
    with fc1:
        fig4 = px.pie(final_dist, names="Machine condition", values="Probability",
                      color="Machine condition", color_discrete_map=STATE_COLORS, hole=0.62)
        fig4.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(styled_plot(fig4, 350), use_container_width=True)
    with fc2:
        st.dataframe(final_dist[["Machine condition", "Probability (%)"]], hide_index=True, use_container_width=True)
        st.caption("These are model-based probabilities, not guarantees or calibrated predictions from real sensor data.")

with tabs[2]:
    st.markdown("### Transition probability matrix")
    st.caption("Each row represents the current state; each column represents the next state.")
    matrix_df = pd.DataFrame(P_used, index=STATES, columns=STATES)
    st.dataframe(matrix_df.style.format("{:.0%}"), use_container_width=True)
    heat = go.Figure(data=go.Heatmap(
        z=P_used, x=STATES, y=STATES, zmin=0, zmax=1,
        colorscale=[[0, "#0e1b2c"], [0.5, "#176b70"], [1, "#39e6c2"]],
        text=np.vectorize(lambda x: f"{x:.0%}")(P_used),
        texttemplate="%{text}", hovertemplate="Current: %{y}<br>Next: %{x}<br>Probability: %{z:.1%}<extra></extra>",
        colorbar=dict(tickformat=".0%"),
    ))
    heat.update_layout(title="Transition probabilities", xaxis_title="Next state", yaxis_title="Current state")
    st.plotly_chart(styled_plot(heat, 360), use_container_width=True)
    st.info("The Failure state is absorbing in this model: its next-state probability remains 100% Failure.")

with tabs[3]:
    st.markdown("### Step-by-step simulation log")
    search = st.text_input("Filter machine condition", placeholder="Type e.g. Major Fault")
    filtered = history_df[history_df["Machine condition"].str.contains(search, case=False, na=False)] if search else history_df
    st.dataframe(filtered, hide_index=True, use_container_width=True)
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download full simulation log (CSV)", csv, "machinepulse_simulation.csv", "text/csv")
    st.caption(f"Showing {len(filtered)} of {len(history_df)} simulation rows.")

with st.expander("ℹ️ Model assumptions and limitations"):
    st.markdown("""
    - The machine is represented by four discrete states: Operational, Minor Wear, Major Fault, and Failure.
    - The transition matrix specifies the probability of moving from one state to another in one time step.
    - The single-path simulation is stochastic; use the same seed to reproduce the same path.
    - The forecast chart computes exact state probabilities using repeated multiplication by the transition matrix.
    - This demo does not ingest sensor data or train on maintenance records. Results depend entirely on the selected probabilities.
    - Do not use this educational demo as the sole basis for real industrial maintenance or safety decisions.
    """)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#7790aa;font-size:12px;padding:8px">'
    'MACHINEPULSE · MARKOV PROCESS DEMO · BUILT WITH STREAMLIT'
    '</div>',
    unsafe_allow_html=True,
)
