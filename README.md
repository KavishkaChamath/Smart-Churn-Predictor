# 📡 Smart Churn Predictor

A machine learning web application that predicts customer churn for a telecom company using the Telco Customer Churn dataset. Built with Streamlit and trained using Logistic Regression, Decision Tree, Random Forest, and SVM models.

---

## 🗂️ Project Structure

```
Smart-Churn-Predictor/
├── app/
│   └── app.py                  # Streamlit web application
├── src/
│   ├── preprocess.py           # Data cleaning & feature engineering
│   ├── train.py                # Model training & saving
│   ├── evaluate.py             # Model evaluation & metrics export
│   └── predict.py              # Standalone prediction script
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned & encoded dataset
├── models/                     # Saved .pkl model files & evaluation results
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/KavishkaChamath/Smart-Churn-Predictor.git
cd Smart-Churn-Predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 🌐 Live Demo

> Hosted on Streamlit Community Cloud — *https://smart-churn-predictor-telco-customer.streamlit.app/*

---
