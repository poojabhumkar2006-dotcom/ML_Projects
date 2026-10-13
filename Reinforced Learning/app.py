"""
Reinforcement Learning for Engineering Design Optimization
Interactive Streamlit dashboard based on the uploaded notebook:
1) Machine energy-control optimization using Q-learning
2) Robot navigation in a grid world using Q-learning

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import random
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ----------------------------- Page setup ------------------------------------
st.set_page_config(
    page_title="RL Engineering Optimizer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0b1220 0%, #111b2e 55%, #101827 100%); color: #edf4ff; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { background: #0c1423; border-right: 1px solid #26354d; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f3f7ff; }
p, label, li { color: #c4d0e3; }
.hero { padding: 1.55rem 1.8rem; border-radius: 22px; border: 1px solid #2a405c;
        background: linear-gradient(115deg, rgba(20,184,166,.16), rgba(59,130,246,.14) 55%, rgba(139,92,246,.12));
        margin-bottom: 1rem; }
.eyebrow { color: #5eead4; font-weight: 700; letter-spacing: .13em; font-size: .76rem; text-transform: uppercase; }
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.7rem, 3vw, 2.7rem); font-weight: 700; line-height: 1.12; margin: .45rem 0; }
.hero-copy { color: #c5d3e7; max-width: 780px; font-size: 1rem; }
.metric-card { background: linear-gradient(145deg, #16243a, #111d30); border: 1px solid #2b405b;
               padding: 1rem 1.15rem; border-radius: 16px; min-height: 112px; }
.metric-label { font-size: .78rem; color: #9fb2cb; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: #f5f8ff; margin-top: .25rem; }
.metric-foot { font-size: .78rem; color: #5eead4; margin-top: .15rem; }
.section-card { background: #111d30; border: 1px solid #2a3d56; border-radius: 18px; padding: 1.1rem 1.25rem; }
div.stButton > button { border-radius: 11px; border: 1px solid #2dd4bf; background: linear-gradient(90deg,#0f9f95,#2563eb);
                        color: white; font-weight: 700; min-height: 2.8rem; }
div.stButton > button:hover { border-color: #99f6e4; color: white; filter: brightness(1.1); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: #142137; border-radius: 10px 10px 0 0; padding: 12px 18px; color: #c4d0e3; }
.stTabs [aria-selected="true"] { background: #164e63 !important; color: #ecfeff !important; }
div[data-testid="stDataFrame"] { border: 1px solid #2a3d56; border-radius: 12px; overflow: hidden; }
.small-note { color: #93a7c2; font-size: .83rem; }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, foot=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce7f6", family="DM Sans"),
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#26364d", zerolinecolor="#26364d")
    fig.update_yaxes(gridcolor="#26364d", zerolinecolor="#26364d")
    return fig


def q_train_energy(episodes, alpha, gamma, epsilon, seed):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    states = ["Low", "Medium", "High"]
    actions = ["Reduce power", "Maintain power", "Increase power"]
    rewards = np.array([[5, 2, -1], [3, 4, 0], [-2, 2, 5]], dtype=float)
    q = np.zeros((3, 3), dtype=float)
    reward_history = []
    for _ in range(episodes):
        state = rng.randrange(3)
        total = 0
        for _step in range(12):
            if rng.random() < epsilon:
                action = rng.randrange(3)
            else:
                best = np.flatnonzero(q[state] == q[state].max())
                action = int(rng.choice(list(best)))
            reward = rewards[state, action]
            next_state = int(np_rng.integers(0, 3))
            q[state, action] += alpha * (
                reward + gamma * np.max(q[next_state]) - q[state, action]
            )
            total += reward
            state = next_state
        reward_history.append(total)
    policy = [actions[int(np.argmax(q[i]))] for i in range(3)]
    return q, reward_history, states, actions, policy


def q_train_robot(grid_size, start, goal, obstacles, episodes, alpha, gamma, epsilon, seed, max_steps):
    rng = random.Random(seed)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    move_names = ["Right →", "Left ←", "Down ↓", "Up ↑"]
    q = np.zeros((grid_size, grid_size, 4), dtype=float)

    def transition(state, action):
        dr, dc = moves[action]
        candidate = (max(0, min(grid_size - 1, state[0] + dr)),
                     max(0, min(grid_size - 1, state[1] + dc)))
        if candidate in obstacles:
            return state
        return candidate

    def reward(state):
        if state == goal:
            return 100.0
        if state == start:
            return -1.0
        return -1.0

    episode_returns = []
    episode_lengths = []
    for _ in range(episodes):
        state = start
        total = 0.0
        for step in range(max_steps):
            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                best = np.flatnonzero(q[state[0], state[1]] == q[state[0], state[1]].max())
                action = int(rng.choice(list(best)))
            new_state = transition(state, action)
            r = reward(new_state)
            q[state[0], state[1], action] += alpha * (
                r + gamma * np.max(q[new_state[0], new_state[1]]) - q[state[0], state[1], action]
            )
            total += r
            state = new_state
            if state == goal:
                break
        episode_returns.append(total)
        episode_lengths.append(step + 1)

    # Greedy path after training; detect loops and report honestly if not found.
    state = start
    path = [start]
    visited = {start}
    reached = False
    for _ in range(max_steps * 2):
        if state == goal:
            reached = True
            break
        values = q[state[0], state[1]]
        ranked = list(np.argsort(values)[::-1])
        next_state = None
        for action in ranked:
            candidate = transition(state, int(action))
            if candidate != state and (candidate not in visited or candidate == goal):
                next_state = candidate
                break
        if next_state is None:
            for action in ranked:
                candidate = transition(state, int(action))
                if candidate != state:
                    next_state = candidate
                    break
        if next_state is None:
            break
        state = next_state
        path.append(state)
        if state in visited and state != goal:
            break
        visited.add(state)
    reached = state == goal
    return q, episode_returns, episode_lengths, path, reached, move_names


def render_grid(grid_size, start, goal, obstacles, path=None):
    path = path or []
    path_set = set(path)
    rows = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            pos = (r, c)
            if pos == start:
                row.append("🟢 START")
            elif pos == goal:
                row.append("🏁 GOAL")
            elif pos in obstacles:
                row.append("⬛ WALL")
            elif pos in path_set:
                row.append("➡️ PATH")
            else:
                row.append("·")
        rows.append(row)
    return pd.DataFrame(rows, index=[f"Row {i}" for i in range(grid_size)],
                         columns=[f"Col {i}" for i in range(grid_size)])


# -------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.caption("Configure the learning experiment")
    page = st.radio("Choose workspace", ["🏭 Energy Optimizer", "🤖 Robot Navigation", "📘 How it works"])
    st.divider()
    st.markdown("### Learning parameters")
    episodes = st.slider("Training episodes", 100, 3000, 1000, 100)
    alpha = st.slider("Learning rate (α)", 0.01, 1.0, 0.10, 0.01)
    gamma = st.slider("Discount factor (γ)", 0.0, 0.99, 0.90, 0.01)
    epsilon = st.slider("Exploration (ε)", 0.0, 1.0, 0.20, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    st.markdown("---")
    st.markdown('<div class="small-note">Built with Q-learning · Interactive engineering sandbox</div>', unsafe_allow_html=True)


# --------------------------------- Header ------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Reinforcement learning · Experiment 08</div>
  <div class="hero-title">Engineering Design <span style="color:#5eead4">Optimizer</span></div>
  <div class="hero-copy">Train an agent to make smarter machine-power decisions and navigate a robot through a grid.
  Tune the learning strategy, inspect rewards, and explore the policy learned from trial and feedback.</div>
</div>
""", unsafe_allow_html=True)

if page == "🏭 Energy Optimizer":
    st.markdown("### 🏭 Energy consumption optimization")
    st.write("The agent learns which power action is most rewarding for each machine-load state.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("States", "03", "Low · Medium · High load")
    with c2:
        metric_card("Actions", "03", "Reduce · Maintain · Increase")
    with c3:
        metric_card("Algorithm", "Q-learning", "Reward-driven policy")

    if st.button("🚀 Train energy-control agent", use_container_width=True):
        q, rewards_history, states, actions, policy = q_train_energy(
            episodes, alpha, gamma, epsilon, int(seed)
        )
        st.session_state["energy_result"] = (q, rewards_history, states, actions, policy)

    if "energy_result" not in st.session_state:
        q, rewards_history, states, actions, policy = q_train_energy(
            episodes, alpha, gamma, epsilon, int(seed)
        )
        st.session_state["energy_result"] = (q, rewards_history, states, actions, policy)

    q, rewards_history, states, actions, policy = st.session_state["energy_result"]
    st.markdown("#### Learned Q-table")
    qdf = pd.DataFrame(np.round(q, 2), index=[f"{s} load" for s in states], columns=actions)
    st.dataframe(qdf.style.format("{:.2f}").background_gradient(cmap="YlGnBu", axis=None), use_container_width=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Action values by machine load")
        if PLOTLY_AVAILABLE:
            heat = go.Figure(data=go.Heatmap(
                z=q, x=actions, y=states, colorscale="Tealgrn",
                text=np.round(q, 1), texttemplate="%{text:.1f}", hoverongaps=False
            ))
            heat.update_layout(title="Higher Q-value = preferred action", xaxis_title="Action",
                               yaxis_title="Machine load")
            st.plotly_chart(chart_layout(heat, 360), use_container_width=True)
        else:
            st.bar_chart(qdf)
    with right:
        st.markdown("#### Recommended operating policy")
        for state, action in zip(states, policy):
            st.markdown(f"""
            <div class="section-card" style="margin-bottom:10px">
              <div style="color:#8da6c5;font-size:.82rem">{state.upper()} LOAD</div>
              <div style="font-size:1.16rem;font-weight:700;color:#5eead4;margin-top:4px">✦ {action}</div>
            </div>""", unsafe_allow_html=True)
        st.caption("Policy is selected by taking the highest learned Q-value in each state.")
    st.markdown("#### Learning progress")
    window = max(1, min(50, len(rewards_history) // 10))
    smoothed = pd.Series(rewards_history).rolling(window=window, min_periods=1).mean()
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rewards_history, mode="lines", name="Episode return",
                                 line=dict(color="#355575", width=1), opacity=.35))
        fig.add_trace(go.Scatter(y=smoothed, mode="lines", name=f"Rolling mean ({window})",
                                 line=dict(color="#2dd4bf", width=3)))
        fig.update_layout(title="Reward per training episode", xaxis_title="Episode", yaxis_title="Total reward")
        st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
    else:
        st.line_chart(pd.DataFrame({"Episode reward": rewards_history, "Rolling mean": smoothed}))
    st.info("Interpretation: the Q-table estimates long-term reward for each action. These rewards are a simplified teaching model, not measured industrial energy data.")

elif page == "🤖 Robot Navigation":
    st.markdown("### 🤖 Robot navigation · Grid world")
    st.write("Train a robot to reach the goal while avoiding walls. Each move costs a small penalty; reaching the goal earns a large reward.")
    a, b, c = st.columns(3)
    with a:
        grid_size = st.slider("Grid size", 4, 8, 5)
    with b:
        robot_episodes = episodes
        st.metric("Training episodes", f"{robot_episodes:,}")
    with c:
        max_steps = st.slider("Max steps per episode", 30, 250, 100, 10)

    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)
    candidates = [(0, 2), (1, 2), (2, 2), (3, 1), (3, 3)]
    default_obstacles = [p for p in candidates if p[0] < grid_size and p[1] < grid_size and p not in (start, goal)]
    obstacle_options = [(r, c_) for r in range(grid_size) for c_ in range(grid_size)
                        if (r, c_) not in (start, goal)]
    default_selection = [p for p in default_obstacles if p in obstacle_options]
    selected = st.multiselect(
        "Place obstacles (row, column)", options=obstacle_options,
        default=default_selection, format_func=lambda p: f"({p[0]}, {p[1]})",
        help="Keep a clear route to the goal for the easiest experiment."
    )
    obstacles = set(selected)
    if st.button("🧭 Train navigation agent", use_container_width=True):
        with st.spinner("Training Q-learning agent…"):
            result = q_train_robot(grid_size, start, goal, obstacles, robot_episodes,
                                   alpha, gamma, epsilon, int(seed), max_steps)
        st.session_state["robot_result"] = (result, grid_size, start, goal, obstacles, robot_episodes, max_steps)

    if "robot_result" not in st.session_state or st.session_state["robot_result"][1] != grid_size or st.session_state["robot_result"][4] != obstacles or st.session_state["robot_result"][5] != robot_episodes or st.session_state["robot_result"][6] != max_steps:
        result = q_train_robot(grid_size, start, goal, obstacles, robot_episodes,
                               alpha, gamma, epsilon, int(seed), max_steps)
        st.session_state["robot_result"] = (result, grid_size, start, goal, obstacles, robot_episodes, max_steps)

    result, _, _, _, _, _, _ = st.session_state["robot_result"]
    q, episode_returns, episode_lengths, path, reached, move_names = result
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Route status", "Goal reached" if reached else "Not reached", "Greedy route after training")
    with m2:
        metric_card("Path length", str(max(0, len(path) - 1)), "Moves in displayed route")
    with m3:
        metric_card("Obstacles", str(len(obstacles)), "Blocked cells")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### 🗺️ Learned route")
        st.dataframe(render_grid(grid_size, start, goal, obstacles, path),
                     use_container_width=True, height=min(440, 45 * grid_size + 45))
        st.caption("🟢 Start · 🏁 Goal · ⬛ Wall · ➡️ Learned path")
        if reached:
            st.success("The greedy policy reached the goal in the displayed route.")
        else:
            st.warning("The current policy did not find a complete route. Try more episodes, lower exploration, or fewer obstacles.")
    with right:
        st.markdown("#### Training performance")
        if PLOTLY_AVAILABLE:
            window = max(1, min(25, len(episode_returns) // 10))
            avg = pd.Series(episode_returns).rolling(window=window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=episode_returns, mode="lines", name="Episode return",
                                     line=dict(color="#42617f", width=1), opacity=.35))
            fig.add_trace(go.Scatter(y=avg, mode="lines", name="Rolling mean",
                                     line=dict(color="#a78bfa", width=3)))
            fig.update_layout(title="Reward during training", xaxis_title="Episode", yaxis_title="Return")
            st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
        else:
            st.line_chart(pd.DataFrame({"Episode return": episode_returns}))
        st.markdown("#### Route coordinates")
        route_df = pd.DataFrame([{"Step": i, "Row": pos[0], "Column": pos[1]} for i, pos in enumerate(path)])
        st.dataframe(route_df, use_container_width=True, height=210)
    st.info("The grid is a simplified simulation. Moves into walls or beyond the boundary leave the robot in place; each step has a penalty and the goal has a positive reward.")

else:
    st.markdown("### 📘 Quick guide")
    st.markdown("""
    <div class="section-card">
    <h3>What is reinforcement learning?</h3>
    An agent learns by interacting with an environment. It receives rewards or penalties and improves its action choices over repeated episodes.
    </div><br>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-card">
        <h3>🏭 Energy optimization</h3>
        <ul>
          <li><b>State:</b> machine load — Low, Medium, High</li>
          <li><b>Actions:</b> Reduce, Maintain, Increase power</li>
          <li><b>Reward:</b> the notebook's illustrative reward matrix</li>
          <li><b>Output:</b> a learned action for each load state</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="section-card">
        <h3>🤖 Robot navigation</h3>
        <ul>
          <li><b>State:</b> robot's row and column in the grid</li>
          <li><b>Actions:</b> move right, left, down, or up</li>
          <li><b>Reward:</b> +100 for reaching the goal, −1 per step</li>
          <li><b>Output:</b> a route based on learned Q-values</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("#### Q-learning update")
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \\alpha\\left[r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)\\right]")
    st.markdown("""
    - **Learning rate (α):** how strongly new experience updates old estimates.
    - **Discount factor (γ):** how much future rewards matter.
    - **Exploration (ε):** chance of trying a random action instead of the current best action.
    - **Episodes:** how many training attempts the agent performs.
    """)
    st.warning("This dashboard is an educational demonstration based on the uploaded notebook's simplified Q-learning examples. It is not a validated engineering controller.")

st.markdown("---")
st.markdown('<div class="small-note" style="text-align:center">RL Engineering Optimizer · Q-learning demo · Tune parameters in the sidebar to experiment</div>', unsafe_allow_html=True)
