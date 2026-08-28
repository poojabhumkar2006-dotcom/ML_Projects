# 🩺 Diabetes Risk AI Platform

An interactive **AI-powered Diabetes Risk Prediction and Analytics Platform** built using **Python and Streamlit**. The application uses a **Random Forest Classifier** to estimate diabetes risk based on clinical parameters and provides interactive dashboards for data exploration and model evaluation.

---

## 📌 Project Overview

The **Diabetes Risk AI Platform** is designed to demonstrate how Machine Learning can be applied to healthcare-related data for diabetes risk analysis.

Users can enter clinical parameters such as:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* BMI
* Diabetes Pedigree Function
* Age

The application processes these inputs and generates a **predicted diabetes risk probability**.

> ⚠️ **Disclaimer:** This project is intended for educational and demonstration purposes only. It should not be used as a substitute for professional medical diagnosis or advice.

---

## ✨ Features

### 🏠 Executive Dashboard

Provides an overview of the dataset and model performance, including:

* Total number of records
* Number of clinical features
* Diabetic ratio
* Model accuracy
* Target distribution visualization
* Glucose level distribution

### 🤖 Patient Risk Analytics

Allows users to enter patient-related clinical parameters and perform a real-time risk assessment.

The application displays:

* Predicted diabetes risk percentage
* Risk gauge visualization
* High-risk / low-risk indication

The prediction functionality is implemented using the trained Random Forest model.

### 📊 Exploratory Data Hub

Interactive data analysis features include:

* Correlation Matrix
* Feature Distribution
* Interactive histograms
* Multi-dimensional data exploration

### 📈 Model Diagnostics

The platform provides model evaluation through:

* Confusion Matrix
* Global Feature Importance
* Model Accuracy

---

## 🧠 Machine Learning Workflow

The project follows a basic Machine Learning pipeline:

```text
Diabetes Dataset
       ↓
Data Preprocessing
       ↓
Handling Structural Zeros
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Random Forest Classifier
       ↓
Model Prediction
       ↓
Model Evaluation
       ↓
Risk Visualization
```

The application replaces structural zeros in selected clinical measurements with missing values and fills them using the median before training the model.

---

## 🛠️ Technologies Used

| Technology     | Purpose                    |
| -------------- | -------------------------- |
| Python         | Programming language       |
| Streamlit      | Web application            |
| Pandas         | Data manipulation          |
| NumPy          | Numerical operations       |
| Scikit-learn   | Machine Learning           |
| Plotly         | Interactive visualizations |
| Random Forest  | Classification model       |
| StandardScaler | Feature scaling            |

The project imports and uses `RandomForestClassifier`, `StandardScaler`, and classification evaluation metrics from Scikit-learn.

---

## 📂 Project Structure

```text
Diabetes-Risk-AI-Platform/
│
├── app.py
├── diabetes.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Diabetes-Risk-AI-Platform
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Dataset

The project uses a diabetes dataset containing clinical attributes such as:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age
* Outcome

The dataset is loaded from `diabetes.csv`.

---

## 🎛️ Model Configuration

The application allows users to configure the Random Forest model through the sidebar.

### Forest Estimators

Users can adjust the number of trees used by the Random Forest model.

### Maximum Tree Depth

Users can control the maximum depth of individual trees.

These parameters are directly exposed through the Streamlit sidebar.

---

## 📈 Model Evaluation

The model is evaluated using:

* **Accuracy Score**
* **Confusion Matrix**
* **Feature Importance**

The confusion matrix compares actual and predicted diabetes outcomes, while feature importance shows the contribution of individual features to the Random Forest model.

---

## 💡 Learning Outcomes

Through this project, I gained practical experience in:

* Machine Learning classification
* Random Forest algorithms
* Data preprocessing
* Feature scaling
* Model evaluation
* Exploratory Data Analysis
* Data visualization
* Streamlit application development
* Building interactive ML dashboards

---

## 🚀 Future Enhancements

Possible improvements include:

* Adding additional Machine Learning models
* Hyperparameter optimization
* Model comparison dashboard
* User authentication
* Prediction history
* Downloadable prediction reports
* Cloud deployment
* Integration with a secure database
* Improved clinical interpretability

---

## ⚠️ Disclaimer

This application is created for **educational and demonstration purposes**. The predictions generated by the model should **not be considered medical advice or a clinical diagnosis**. Always consult a qualified healthcare professional for medical decisions.

---

## 👩‍💻 Author

**Pooja Dnyaneshwar Bhumkar**

B.E. Artificial Intelligence & Data Science

---



