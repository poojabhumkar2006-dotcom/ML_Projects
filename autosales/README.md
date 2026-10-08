# 🚗 Auto Sales Prediction

An interactive **Auto Sales / Vehicle Price Prediction** application built using **Python, Machine Learning, and Streamlit**. The project uses a **Decision Tree Regressor** to predict vehicle prices based on different automobile characteristics.

The project also provides an interactive dashboard for exploring vehicle data, adjusting model parameters, viewing evaluation metrics, and generating price predictions.

---

## 📌 Project Overview

The **Auto Sales Prediction** project applies Machine Learning to estimate the price of automobiles using important vehicle specifications such as:

- Wheel Base
- Length
- Width
- Height
- Curb Weight
- Number of Cylinders
- Engine Size
- Horsepower
- Peak RPM
- City MPG
- Highway MPG

A **Decision Tree Regression** algorithm is used to train the prediction model.

The project includes an interactive **Streamlit web application** that makes it easier to understand the dataset, model performance, and predicted vehicle prices.

---

## 🎯 Objectives

- Analyze automobile-related data.
- Perform data cleaning and preprocessing.
- Convert categorical values into numerical form where required.
- Handle missing values.
- Train a Machine Learning regression model.
- Predict automobile prices.
- Evaluate model performance using regression metrics.
- Provide an interactive dashboard using Streamlit.
- Allow users to modify Decision Tree hyperparameters.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Streamlit | Web application/dashboard |
| Plotly | Interactive visualizations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory Data Analysis |
| Jupyter Notebook | Model development and experimentation |

---

## 🤖 Machine Learning Algorithm

### Decision Tree Regressor

The project uses the **Decision Tree Regression** algorithm to predict automobile prices.

A Decision Tree Regressor works by dividing the dataset into smaller groups based on feature values and making predictions using the resulting decision rules.

### Model Parameters

The Streamlit application allows users to control:

- **Max Depth**
- **Minimum Samples Leaf**
- **Splitter Strategy**
  - `best`
  - `random`

This makes it possible to experiment with the model and observe how hyperparameters affect performance.

---

## 📊 Dataset

The project uses an automobile dataset containing **205 records and 26 columns**.

The dataset contains information about:

- Automobile manufacturer
- Fuel type
- Body style
- Drive wheels
- Engine specifications
- Horsepower
- Mileage
- Vehicle dimensions
- Vehicle price

### Target Variable

```text
price
```

The `price` column is the target variable that the model attempts to predict.

### Main Features Used

```text
wheel-base
length
width
height
curb-weight
num-of-cylinders
engine-size
horsepower
peak-rpm
city-mpg
highway-mpg
```

---

## 🧹 Data Preprocessing

The application performs several preprocessing steps before training the model.

### 1. Handling Cylinder Values

Categorical cylinder values are converted into numerical values.

For example:

```text
two     → 2
three   → 3
four    → 4
five    → 5
six     → 6
eight   → 8
twelve  → 12
```

### 2. Numeric Conversion

Important columns such as:

- Price
- Horsepower
- Peak RPM

are converted into numeric values.

### 3. Missing Values

Missing values are handled using the mean of the respective numerical column.

### 4. Feature Selection

Only relevant numerical features are selected for training the Decision Tree model.

---

## 🔄 Machine Learning Workflow

```text
        Automobile Dataset
                ↓
        Data Loading
                ↓
       Data Preprocessing
                ↓
       Missing Value Handling
                ↓
        Feature Selection
                ↓
        Train-Test Split
                ↓
      Decision Tree Regressor
                ↓
          Model Training
                ↓
         Price Prediction
                ↓
       Model Evaluation
                ↓
       Streamlit Dashboard
```

---

## 📈 Model Evaluation

The model performance is evaluated using:

### R² Score

Measures how well the model explains the variation in the target variable.

```text
R² = 1 - (SSres / SStot)
```

A value closer to **1** generally indicates better performance.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

```text
MAE = Average(|Actual - Predicted|)
```

### Root Mean Squared Error (RMSE)

Measures prediction error while giving higher weight to larger errors.

```text
RMSE = √MSE
```

The application displays both **training** and **testing** performance.

---

## 🖥️ Streamlit Dashboard

The project contains an interactive Streamlit application named:

```text
app (2).py
```

The dashboard provides:

- 🚗 Vehicle price prediction
- 📊 Model performance metrics
- 📈 Interactive charts
- ⚙️ Decision Tree model controls
- 🔧 Hyperparameter adjustment
- 📋 Vehicle data analysis
- 📌 Prediction results

The application also includes a modern automobile-themed interface.

---

## 📂 Project Structure

```text
autosales/
│
├── app (2).py
│
├── autos_dataset (1).csv
│
├── Day 60_Decision Tree Linear regression_Auto Data Set (1).ipynb
│
└── anaconda_projects/
    └── db/
```

### Important Files

**`app (2).py`**

Main Streamlit application containing the dashboard, preprocessing, model training, evaluation, and prediction functionality.

**`autos_dataset (1).csv`**

Automobile dataset used for training and analysis.

**`Day 60_Decision Tree Linear regression_Auto Data Set (1).ipynb`**

Jupyter Notebook containing the Machine Learning experimentation and Decision Tree Regression implementation.

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### Step 2: Navigate to the Project Folder

```bash
cd autosales
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy scikit-learn streamlit plotly matplotlib seaborn
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run "app.py"
```

After running the command, Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your web browser to use the application.

---

## 🔍 Example Prediction Workflow

1. Open the Streamlit application.
2. Adjust the Decision Tree parameters from the sidebar.
3. Enter/select the required automobile characteristics.
4. Run the prediction.
5. View the estimated vehicle price.
6. Analyze model performance using R², MAE, and RMSE.

---

## 💡 Key Features

- ✅ Machine Learning-based vehicle price prediction
- ✅ Decision Tree Regression
- ✅ Interactive Streamlit interface
- ✅ Data preprocessing
- ✅ Missing value handling
- ✅ Hyperparameter controls
- ✅ Training and testing metrics
- ✅ Interactive visualizations
- ✅ Automobile-focused dashboard
- ✅ Beginner-friendly Machine Learning project

---

## 🚀 Future Improvements

The project can be further improved by:

- Adding Linear Regression and Random Forest models.
- Comparing multiple Machine Learning algorithms.
- Adding categorical feature encoding.
- Implementing cross-validation.
- Adding automated hyperparameter tuning.
- Improving prediction accuracy.
- Adding downloadable prediction reports.
- Deploying the application using Streamlit Community Cloud.
- Adding a larger and more recent automobile dataset.

---

## 🎓 Learning Outcomes

Through this project, the following concepts can be understood:

- Data preprocessing
- Exploratory Data Analysis
- Feature selection
- Train-test splitting
- Decision Tree Regression
- Hyperparameter tuning
- Regression evaluation metrics
- Streamlit application development
- Interactive data visualization
- Machine Learning model deployment

---

## 👩‍💻 Author

**Pooja Bhumkar**

B.E. Artificial Intelligence & Data Science

---

## ⭐ Acknowledgement

This project was developed as part of Machine Learning practice and experimentation using Python and Scikit-learn.

---

