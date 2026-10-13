⚙️ MachinePulse — Predictive Machine Maintenance Using a Markov Process


MachinePulse is an interactive Streamlit dashboard that demonstrates how a discrete-time Markov chain can model machine health, estimate future failure probability, and support preventive-maintenance planning.
The project includes a notebook-based experiment and a dashboard for exploring machine-state transitions and forecast probabilities.
📌 Project Overview
In manufacturing environments, machine condition can gradually change from healthy operation to wear, faults, and eventual failure. This project models those changes using a Markov process.
The application simulates one possible sequence of machine conditions and also calculates the exact probability distribution across all conditions over a selected forecast horizon.
Note: This is an educational simulation. It does not use live sensor readings, train on real maintenance records, or provide certified industrial safety predictions.

🎯 Objectives
- Model machine health using a Markov chain.
- Simulate possible changes in machine condition over time.
- Estimate the probability of machine failure at a selected forecast horizon.
- Visualize state transitions and probability distributions.
- Explore how changing transition probabilities affects the forecast.
- Provide simple maintenance guidance based on the simulated final state.
🧰 Technologies Used
- Python — application logic and simulation
- Streamlit — interactive web dashboard
- NumPy — transition probabilities and Markov-chain calculations
- Pandas — simulation logs and tabular data
- Plotly Express / Plotly Graph Objects — interactive charts and heatmaps
- Jupyter Notebook — experiment and demonstration
🏭 Machine Health States
Code	State	Meaning
S1	Operational	Machine is operating normally
S2	Minor Wear	Wear is present; inspection may be needed
S3	Major Fault	A significant fault is indicated
S4	Failure	Machine has entered the failure state


🔢 Default Transition Probability Matrix
Each row represents the current state, and each column represents the next state.
Current State	Operational (S1)	Minor Wear (S2)	Major Fault (S3)	Failure (S4)
Operational (S1)	0.7	0.2	0.1	0.0
Minor Wear (S2)	0.2	0.5	0.2	0.1
Major Fault (S3)	0.0	0.3	0.4	0.3
Failure (S4)	0.0	0.0	0.0	1.0


The Failure state is absorbing in this model: once entered, the machine remains in that state in subsequent steps.
✨ Dashboard Features
- Simulation controls: choose the starting machine condition, forecast horizon, and random seed.
- Model tuning: adjust transition probabilities for operational, minor-wear, and major-fault states.
- Machine health timeline: inspect the simulated condition at each time step.
- State distribution chart: view how often each condition occurs in the simulated path.
- Probability forecast: view the model-based probability of each condition over time.
- Transition matrix: inspect the selected probabilities in a table and heatmap.
- Simulation log: filter the log by condition and download it as a CSV file.
- Maintenance guidance: view general guidance associated with the simulated final state.
- Reset option: restore the default model settings.
📁 Repository Files
The repository should contain the following project files:
.
├── app.py
├── E09_Predictive Machine Maintenance using Markov Process.ipynb
├── README.md
└── requirements.txt
- app.py — interactive Streamlit dashboard.
- E09_Predictive Machine Maintenance using Markov Process.ipynb — notebook experiment explaining the states, transition matrix, simulation, and visualization.
- README.md — project documentation.
- requirements.txt — Python dependencies.
If your uploaded files have different names, either rename them to match this structure or update the names above.
⚙️ Installation and Setup
1. Clone the repository
Replace YOUR-USERNAME and YOUR-REPOSITORY with your GitHub details.
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
2. Create a virtual environment (recommended)
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
Create a requirements.txt file containing:
streamlit
numpy
pandas
plotly
Then install the packages:
pip install -r requirements.txt
Jupyter Notebook is only needed if you want to run the notebook separately:
pip install notebook
4. Run the dashboard
Run this command from the project directory:
streamlit run app.py
Streamlit will provide a local URL in the terminal, usually http://localhost:8501. Open that URL in your browser.
5. Run the notebook (optional)
Start Jupyter:
jupyter notebook
Open the machine-maintenance notebook and run its cells in order.
🧪 How to Use the Dashboard
1. Start the app with streamlit run app.py.
2. Select the initial machine condition in the sidebar.
3. Choose a forecast horizon and simulation seed.
4. Adjust transition probabilities if you want to test a different scenario.
5. Select Run prediction.
6. Explore the Overview, Probability Forecast, Transition Matrix, and Simulation Log tabs.
7. Download the simulation log as a CSV if needed.
8. Use Reset model defaults to restore the default settings.
🧠 How the Model Works
A Markov chain assumes that the next state depends on the current state and the transition probabilities.
If \(P\) is the transition matrix and \(\pi_t\) is the state-probability distribution at time \(t\), the next distribution is:
\[
\pi_{t+1} = \pi_t P
\]
The dashboard uses two complementary approaches:
1. Random simulation: samples one possible sequence of states. The same random seed can reproduce the same path when the inputs remain unchanged.
2. Probability forecast: repeatedly multiplies the state distribution by the transition matrix to calculate the probability of each state at each step.
These results are based on the selected model probabilities. They are not guarantees about a real machine.
📊 Maintenance Guidance
Simulated final state	General guidance shown by the app
Operational	Continue routine monitoring and regular inspections
Minor Wear	Plan an inspection and preventive maintenance
Major Fault	Prioritize technical inspection and corrective maintenance
Failure	Follow site shutdown and safety procedures


Always follow your organization's maintenance procedures and qualified personnel's advice for real equipment.
⚠️ Assumptions and Limitations
- The model contains four discrete machine-health states.
- Transition probabilities are illustrative and configurable.
- The Failure state is absorbing.
- The simulation is stochastic; results depend on the random seed and selected parameters.
- The application does not ingest real sensor data or train on historical maintenance records.
- Forecast probabilities depend entirely on the transition matrix.
- The dashboard is for learning and demonstration, not for safety-critical decisions.
🔮 Possible Future Enhancements
- Integrate real machine sensor data.
- Estimate transition probabilities from historical maintenance records.
- Compare forecasts with actual machine failures.
- Add machine-wise records and maintenance history.
- Add alerts when model-based risk exceeds a configurable threshold.
- Evaluate model performance against real-world outcomes.
👩‍🎓 Academic Context
This project demonstrates the use of a Markov process for modeling manufacturing processes and machine maintenance. It illustrates how state-transition probabilities can be used to explore machine degradation and support maintenance planning.
📄 License
Choose a license before publishing this repository. For example, you may use the MIT License if it suits your project and you have permission to license all included materials.
