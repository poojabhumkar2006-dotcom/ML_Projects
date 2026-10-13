# ⚙️ Reinforcement Learning for Engineering Design Optimization

An interactive **Streamlit dashboard** demonstrating how **Q-learning**, a reinforcement learning algorithm, can learn action policies for simplified engineering optimization problems.

The application includes two interactive simulations:
- 🏭 **Machine Energy-Control Optimization** — learns a recommended power action for low, medium, and high machine-load states.
- 🤖 **Robot Navigation** — trains an agent to navigate a grid world toward a goal while avoiding obstacles.

> **Note:** This project is an educational simulation. Its reward values and environments are simplified examples, not validated industrial measurements or production-ready engineering controllers.

## ✨ Features

- Interactive dashboard built with Streamlit.
- Adjustable Q-learning parameters:
  - **Training episodes**
  - **Learning rate (α)**
  - **Discount factor (γ)**
  - **Exploration rate (ε)**
  - **Random seed**
- Machine energy-control Q-table and recommended policy.
- Robot grid-world simulation with configurable grid size and obstacles.
- Visualized training rewards and rolling-average performance.
- Display of the robot's learned route, path length, and route coordinates.
- Educational guide explaining reinforcement learning and the Q-learning update.
- Plotly visualizations when Plotly is available, with basic chart fallbacks.

## 🧠 How It Works

Q-learning estimates the expected long-term reward for taking an action in a state. After interacting with the environment, the agent updates its Q-value using:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
\]

Where:
- **State (`s`)**: the current situation of the agent.
- **Action (`a`)**: a choice the agent can make.
- **Reward (`r`)**: feedback received after an action.
- **Learning rate (`α`)**: controls how strongly new experience updates the Q-value.
- **Discount factor (`γ`)**: controls the importance of future rewards.
- **Exploration rate (`ε`)**: controls how often the agent tries a random action.

### 1. Machine Energy-Control Optimization

The simulation models three machine-load states and three available actions.

| Component | Options |
|---|---|
| States | Low, Medium, High load |
| Actions | Reduce power, Maintain power, Increase power |
| Learning method | Q-learning |
| Output | Q-table and recommended action for each load state |

The agent learns from an illustrative reward matrix. These rewards are designed for demonstration and do not represent measured energy consumption.

### 2. Robot Navigation

The robot learns a route from a start cell to a goal cell in a grid environment.

| Component | Description |
|---|---|
| State | Robot's row and column in the grid |
| Actions | Move right, left, down, or up |
| Goal reward | +100 |
| Step penalty | −1 per step |
| Obstacles | Configurable blocked cells |
| Output | Learned route, route coordinates, and training rewards |

The displayed route is derived from the learned Q-values. Depending on the settings and obstacles, the agent may not always find a complete route.

## 🛠️ Tech Stack

- **Python** — application and reinforcement-learning logic
- **Streamlit** — interactive web dashboard
- **NumPy** — numerical calculations and Q-tables
- **Pandas** — data tables and reward-history processing
- **Plotly** — interactive charts and heatmaps (optional)

## 📁 Project Structure

```text
reinforcement-learning-engineering-optimization/
├── app.py
├── requirements.txt
├── E10_Implementation_Reinforced_Learning_for_optimizing_engineering_designs.ipynb
└── README.md
```

The notebook filename shown above is illustrative. Keep the actual notebook filename in your repository, or rename it to match this structure.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later recommended
- `pip`

### 1. Clone the repository

Replace `<your-username>` and `<your-repository>` with your GitHub details.

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. (Recommended) Create a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The current requirements include:

```text
streamlit>=1.32
numpy>=1.24
pandas>=2.0
plotly>=5.18
```

### 4. Run the application

If your Python file is named `app.py`:

```bash
streamlit run app.py
```

If your uploaded file still has a different name, either rename it to `app.py` or pass its actual filename to Streamlit.

Streamlit will print a local URL in the terminal (commonly `http://localhost:8501`). Open that URL in your browser.

## 🖥️ Using the Dashboard

1. Launch the application.
2. Use the sidebar to select **Energy Optimizer**, **Robot Navigation**, or **How it works**.
3. Adjust the training episodes and Q-learning parameters.
4. For energy optimization, select **Train energy-control agent** to inspect the learned Q-table, policy, and reward history.
5. For robot navigation, choose the grid size, configure obstacles, and select **Train navigation agent**.
6. Review the learned route and training performance. If the goal is not reached, try more episodes, a different exploration rate, or fewer obstacles.

## 📊 Evaluation and Limitations

The dashboard provides visual feedback through Q-tables, reward histories, rolling averages, and the learned robot route. These are demonstration metrics rather than a rigorous benchmark against real engineering systems.

Current limitations:
- The environments use simplified, predefined reward rules.
- Machine energy rewards are illustrative and are not measured from physical equipment.
- Learning outcomes can vary with hyperparameters, random seed, and obstacle placement.
- The application does not connect to real machines, robots, or industrial sensors.
- A learned policy should not be used for real-world control without domain validation, safety analysis, and appropriate testing.

## 🔮 Possible Future Improvements

- Add comparisons with other reinforcement-learning algorithms.
- Introduce realistic energy-consumption or engineering-design datasets.
- Add experiment export and downloadable results.
- Support configurable start and goal positions for robot navigation.
- Add repeatable benchmark experiments across multiple random seeds.
- Evaluate policies using defined baselines and statistical metrics.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request describing the improvement.

## 📄 License

No license is specified by the supplied project files. If you intend to make the repository public, choose and add a license file that matches how you want others to use your work.

## 👤 Author

**Your Name**  
GitHub: [@your-username](https://github.com/your-username)

Replace the author and GitHub placeholders with your details.
