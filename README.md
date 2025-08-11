# 💤 Sleep Disorder Detection System (Streamlit App)

This project is a complete **Machine Learning pipeline** to predict sleep disorders — including **Insomnia**, **Sleep Apnea**, or **No Disorder** — based on user health and lifestyle attributes. It also includes a **Streamlit Web App** where users can enter their details and get real-time predictions.

## 🚀 Live Demo

👉 [Try the app](https://my-sleep-detector.streamlit.app/)

---

## 📌 Project Overview

Sleep disorders can significantly impact an individual’s health, productivity, and quality of life. This project:

- Builds a **realistic synthetic dataset** from an available synthetic data file in Kaggle.
- Applies **data preprocessing, EDA, feature engineering**, and **ML modeling**.
- Trains an **XGBoost Classifier** with **hyperparameter tuning**.
- Evaluates model performance using accuracy, classification report, AUC scores, and visualizations.
- Creates a **user-friendly Streamlit interface** for real-time predictions.

---

## 📊 Dataset Details

| Feature                   | Description                                                             |
|--------------------------|-------------------------------------------------------------------------|
| Gender                   | Male/Female                                                             |
| Age                      | 18 - 80 years                                                           |
| Sleep Duration           | Hours of sleep per day (2 to 10 hrs)                                    |
| Quality of Sleep         | Subjective score (1 to 10)                                              |
| Physical Activity Level  | Minutes per day (5 to 100 mins)                                         |
| Stress Level             | Subjective score (1 to 10)                                              |
| BMI Category             | Normal / Overweight                                                     |
| Heart Rate               | Beats per minute (55 to 110 bpm)                                        |
| Daily Steps              | Number of steps (1000 to 13000)                                         |
| Sleep Efficiency         | Derived feature (Quality of Sleep / Sleep Duration)    |
| **Sleep Disorder**       | **Target variable: No disorder / Insomnia / Sleep Apnea**               |


---

## 🔧 Tools & Technologies Used

- **Python 3.9+**
- **Pandas, NumPy, Seaborn, Matplotlib**
- **Scikit-learn, XGBoost**
- **Imbalanced-learn (SMOTE)**
- **Streamlit** for app deployment
- **Joblib** for model/encoder saving
- **Visualizations:** Correlation matrix, feature importance, ROC curves

---

## 🧠 Model Training Pipeline

### 1. Data Preprocessing

### 2. Feature Engineering

### 3. Model Selection & Evaluation
- Compared:
  - Logistic Regression
  - Random Forest
  - XGBoost (Best)
- Used:
  - 5-fold Cross-Validation
  - RandomizedSearchCV for tuning
- Metrics:
  - Accuracy
  - Classification Report
  - ROC-AUC (Macro, Weighted)
  - Confusion Matrix

### 4. Best Model: **XGBoost Classifier**
- Tuned with best parameters
- Saved as `xgb_sleep_model.pkl`

---

## 📦 Streamlit App Features

The Streamlit web app:
- Asks for user inputs (Age, Gender, Sleep Duration, etc.)
- Automatically calculates **Sleep Efficiency**
- Encodes and scales features
- Loads trained model and predicts the sleep disorder
- Returns result in human-readable form

### 💻 Screenshot:
> _Add a screenshot of the app UI here if possible._

---

🔮 Future Enhancements
Add explainability using SHAP values.
Add more lifestyle features like Caffeine Intake, Blue Light Exposure, etc.

🙋‍♀️ Author
👩‍💻 Developed by Anusha
📧 Contact: anushamahalinga243@gmail.com

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
