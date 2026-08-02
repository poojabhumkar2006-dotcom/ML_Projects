# 🎓 Placement Prediction System

A Machine Learning-based web application that predicts whether a student is likely to be placed based on academic and skill-related parameters. The application is built using **Python, Flask, HTML, CSS, and Scikit-learn**.

---

## 📌 Features

- 🔐 User-friendly web interface
- 📊 Predicts placement chances using a trained Machine Learning model
- 📈 Displays prediction confidence
- 📝 Stores prediction history
- 📂 Uses CSV dataset for model training
- 💻 Responsive and attractive UI

---

## 🛠️ Technologies Used

- Python
- Flask
- HTML5
- CSS3
- Bootstrap
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## 📁 Project Structure

```
Placement-Prediction/
│
├── static/
│   ├── css/
│   ├── images/
│   └── js/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
│
├── app.py
├── model.pkl
├── students_placement.csv
├── prediction_history.csv
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Dataset

The project uses a student placement dataset containing attributes such as:

- CGPA
- IQ
- Previous Semester Percentage
- Communication Skills
- Technical Skills
- Internships
- Projects
- Placement Status

The dataset is used to train the Machine Learning model.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/poojabhumkar2006/placement-prediction.git
```

### 2. Navigate to Project Folder

```bash
cd placement-prediction
```

### 3. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## 🧠 Machine Learning Model

The prediction model is trained using **Support Vector Machine (SVM)**.

### Workflow

- Data Collection
- Data Preprocessing
- Feature Selection
- Model Training
- Model Evaluation
- Prediction
- Deployment using Flask

---

## 📷 Application Screenshots

Add screenshots here after uploading them.

### Home Page

```
screenshots/home.png
```

### Prediction Result

```
screenshots/result.png
```

### Prediction History

```
screenshots/history.png
```

---

## 📈 Future Enhancements

- User Authentication
- Database Integration (MySQL)
- Graphical Analytics Dashboard
- Export Prediction Reports
- Email Notifications
- Cloud Deployment

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Pooja Dnyaneshwar Bhumkar**

- B.E. Artificial Intelligence & Data Science
- Savitribai Phule Pune University

GitHub: https://github.com/poojabhumkar2006-dotcom

---
