# 🏠 EstateAI – House Price Prediction System

## 📌 Project Overview

**EstateAI** is a Machine Learning-based House Price Prediction System developed using **Python, Scikit-Learn, Pandas, and Streamlit**.

The application predicts the estimated price of a property based on details such as location, property type, BHK, size, age, facilities, furnishing status, floor information, and other property-related features.

The project provides an interactive dashboard where users can:

* 🔮 Predict house prices
* 📊 Analyze property details
* 💰 Calculate home-loan EMI
* 📋 View prediction history
* 🏆 Calculate a property quality score
* 💡 Get basic property insights

---

## ✨ Key Features

### 🔮 1. House Price Prediction

Users can enter:

* State
* City
* Property Type
* BHK
* Property Size
* Price per Sq.Ft
* Year Built
* Property Age
* Furnished Status
* Floor Number
* Total Floors
* Nearby Schools
* Nearby Hospitals
* Public Transport
* Parking
* Security
* Amenities
* Facing
* Owner Type
* Availability Status

The trained Machine Learning model then predicts the estimated house price.

---

### 📊 2. Property Analytics

The analytics section displays:

* Estimated property price
* Property size
* BHK
* Property quality score
* Property details
* Nearby facilities
* Basic charts

---

### 💰 3. EMI Calculator

The built-in EMI calculator calculates:

* Loan amount
* Monthly EMI
* Total interest
* Total repayment
* Principal vs. interest breakdown

---

### 📋 4. Prediction History

Every successful prediction can be saved in:

```text
prediction_history.csv
```

Users can:

* View previous predictions
* See average predicted price
* Find highest and lowest predictions
* View prediction trends
* Download prediction history

---

### 🏆 5. Property Quality Score

EstateAI generates a simple property score based on factors such as:

* Property size
* Nearby schools
* Nearby hospitals
* Parking availability
* Security
* Property age

The score is displayed out of **100**.

---

## 🤖 Machine Learning Model

The project uses a:

**Decision Tree Regressor**

The application uses the following ML components:

| Component               | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| Decision Tree Regressor | House price prediction                         |
| DictVectorizer          | Converts input features into numerical vectors |
| Ordinal Encoder         | Encodes categorical features                   |
| Pandas                  | Data processing                                |
| Pickle                  | Saves and loads trained ML objects             |
| Streamlit               | Web application                                |

---

## 🛠️ Technologies Used

* 🐍 Python
* 📊 Pandas
* 🤖 Scikit-Learn
* 🎨 Streamlit
* 🔢 NumPy
* 💾 Pickle
* 📄 CSV
* 🌐 HTML/CSS through Streamlit

---

## 📁 Project Structure

```text
House Price Prediction/
│
├── app.py
│
├── house_price_model.pkl
├── vectorizer.pkl
├── encoder.pkl
├── features.pkl
│
├── prediction_history.csv
│
├── requirements.txt
│
└── README.md
```

### File Description

| File                     | Description                 |
| ------------------------ | --------------------------- |
| `app.py`                 | Main Streamlit application  |
| `house_price_model.pkl`  | Trained Decision Tree model |
| `vectorizer.pkl`         | Saved DictVectorizer        |
| `encoder.pkl`            | Saved categorical encoder   |
| `features.pkl`           | Saved feature information   |
| `prediction_history.csv` | Stores previous predictions |
| `requirements.txt`       | Required Python libraries   |
| `README.md`              | Project documentation       |

---

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Open the project

```bash
cd House-Price-Prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually:

```text
http://localhost:8501
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
```

If your saved pickle files were created using a particular Scikit-Learn version, use the compatible version to avoid pickle compatibility warnings/errors.

---

## 🔄 Application Workflow

```text
User
  ↓
Enter Property Details
  ↓
Data Validation
  ↓
Ordinal Encoding
  ↓
DictVectorizer
  ↓
Machine Learning Model
  ↓
House Price Prediction
  ↓
Property Score
  ↓
Smart Insights
  ↓
Save Prediction History
```

---

## 📊 Input Features

The model works with property-related features such as:

```text
State
City
Property_Type
BHK
Size_in_SqFt
Price_per_SqFt
Year_Built
Furnished_Status
Floor_No
Total_Floors
Age_of_Property
Nearby_Schools
Nearby_Hospitals
Public_Transport_Accessibility
Parking_Space
Security
Amenities
Facing
Owner_Type
Availability_Status
```

---

## 🎨 User Interface

The application includes a modern real-estate themed interface with:

* 🏠 EstateAI branding
* 🌄 Real-estate background image
* 🌙 Dark dashboard design
* 🟡 Gold highlights
* 🔴 Prediction action button
* 📊 Analytics dashboard
* 💰 EMI calculator
* 📋 Prediction history
* 📱 Responsive Streamlit layout

---

## 🚀 Future Improvements

The project can be further improved by adding:

* 📍 Google Maps integration
* 📈 Advanced property price charts
* 🧠 Random Forest/XGBoost comparison
* 🎯 Model accuracy metrics
* 🗺️ Location-based price prediction
* 🔐 User authentication
* 🗄️ MySQL database
* 📊 Interactive Plotly dashboards
* 📥 PDF property reports
* ☁️ Cloud deployment
* 🤖 AI-based property recommendations

---

## 🎯 Project Objectives

The main objectives of EstateAI are:

1. To predict house prices using Machine Learning.
2. To provide an easy-to-use real-estate prediction interface.
3. To analyze important property characteristics.
4. To provide useful property insights.
5. To calculate home-loan EMI.
6. To maintain prediction history.

---

## 👩‍💻 Author

**Pooja Dnyaneshwar Bhumkar**

**B.E. Artificial Intelligence & Data Science**

---


