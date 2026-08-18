# 🛒 KNN Regression – Outlet Sales Prediction

## 📌 Project Overview

This project uses **Machine Learning and K-Nearest Neighbors (KNN) Regression** to predict **Item Outlet Sales** based on product and outlet-related features.

The project demonstrates the complete machine learning workflow, including:

* Data loading and exploration
* Data preprocessing
* Handling missing values
* Encoding categorical variables
* Feature scaling
* KNN Regression model building
* Model evaluation
* Sales prediction

---

## 🎯 Objective

The main objective of this project is to predict the sales of a product at a particular outlet using historical sales data.

**Target Variable:**

`Item_Outlet_Sales`

The model uses product characteristics and outlet information to estimate the expected sales.

---

## 📂 Dataset

The dataset contains **8,523 records and 12 columns**.

### Features

| Feature                     | Description                             |
| --------------------------- | --------------------------------------- |
| `Item_Identifier`           | Unique identifier of the product        |
| `Item_Weight`               | Weight of the product                   |
| `Item_Fat_Content`          | Fat content category                    |
| `Item_Visibility`           | Visibility of the product in the outlet |
| `Item_Type`                 | Type/category of the product            |
| `Item_MRP`                  | Maximum Retail Price of the product     |
| `Outlet_Identifier`         | Unique outlet identifier                |
| `Outlet_Establishment_Year` | Year in which outlet was established    |
| `Outlet_Size`               | Size of the outlet                      |
| `Outlet_Location_Type`      | Location category of the outlet         |
| `Outlet_Type`               | Type of outlet                          |
| `Item_Outlet_Sales`         | **Target variable – outlet sales**      |

---

## 🤖 Machine Learning Algorithm

### K-Nearest Neighbors (KNN) Regression

KNN Regression predicts the output value by considering the target values of the **K nearest data points**.

The basic process is:

1. Select the value of **K**.
2. Calculate the distance between the test sample and training samples.
3. Identify the K nearest neighbors.
4. Calculate the average target value of those neighbors.
5. Use the average as the predicted sales value.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Handle Missing Values
   ↓
Categorical Encoding
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
KNN Regression
   ↓
Prediction
   ↓
Model Evaluation
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook / VS Code**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/poojabhumkar2006/ML_Projects
/Market_Sales_Prediction.git
```

Navigate to the project folder:

```bash
cd ML_Projects
/Market_Sales_Prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run

1. Download or clone the repository.
2. Open the project in **Jupyter Notebook** or **VS Code**.
3. Make sure the dataset is present in the project directory.
4. Run the Python/Jupyter Notebook file.
5. Execute the cells step by step.
6. View the predicted outlet sales and model evaluation results.

---

## 📊 Model Evaluation

The performance of the KNN Regression model can be evaluated using:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

### Evaluation Metrics

```text
MAE  → Measures average absolute prediction error
MSE  → Measures average squared prediction error
RMSE → Measures the square root of MSE
R²   → Measures how well the model explains the target variable
```

---

## 📁 Project Structure

```text
KNN-Regression-Outlet-Sales/
│
├── KNN_reg_outlet_sales(1).csv
├── KNN_Regression.ipynb
├── README.md
└── requirements.txt
```

---

## 💡 Key Insights

* Product characteristics influence outlet sales.
* `Item_MRP` is an important factor for predicting sales.
* Outlet characteristics can significantly affect sales performance.
* Feature scaling is important for KNN because the algorithm is distance-based.
* Choosing an appropriate value of **K** can improve model performance.

---

## 🚀 Future Improvements

* Perform hyperparameter tuning to find the optimal K value.
* Compare KNN with Random Forest, Decision Tree, Linear Regression, and XGBoost.
* Create an interactive **Streamlit web application**.
* Add advanced feature engineering.
* Deploy the prediction model online.

---

## 👩‍💻 Author

**Pooja Dnyaneshwar Bhumkar**

B.E. Artificial Intelligence & Data Science
Jai Hind College of Engineering, Kuran

---

## ⭐ Acknowledgement

This project was developed as part of my learning journey in **Machine Learning and Data Science**, focusing on practical implementation of regression algorithms.

---

## 📜 License

This project is created for **educational and learning purposes**.
