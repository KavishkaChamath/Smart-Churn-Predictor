
import pandas as pd
import joblib
import numpy as np
import os

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "/kaggle/working/Smart-Churn-Predictor/models/best_model.pkl"
model = joblib.load(MODEL_PATH)

# -----------------------------
# LOAD SCALER (ONLY IF EXISTS)
# -----------------------------
scaler_path = "/kaggle/working/Smart-Churn-Predictor/models/svm_scaler.pkl"

scaler = None
if os.path.exists(scaler_path):
   scaler = joblib.load(scaler_path)

# -----------------------------
# FEATURE LIST (IMPORTANT)
# -----------------------------
MODEL_FEATURES = model.feature_names_in_

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess_input(input_dict):
   df = pd.DataFrame([input_dict])
   df = pd.get_dummies(df)
   df = df.reindex(columns=MODEL_FEATURES, fill_value=0)
   return df

# -----------------------------
# PREDICT
# -----------------------------
def predict_churn(input_data):

   df = preprocess_input(input_data)

   # -------------------------
   # IMPORTANT FIX HERE
   # -------------------------
   # ONLY SCALE IF MODEL IS NOT LOGISTIC REGRESSION
   if scaler is not None and "logistic" not in str(type(model)).lower():
       df_input = scaler.transform(df)
   else:
       df_input = df  # KEEP DATAFRAME FOR LOGISTIC REGRESSION

   # -------------------------
   # PREDICTION
   # -------------------------
   prediction = model.predict(df_input)[0]

   # -------------------------
   # PROBABILITY
   # -------------------------
   probability = None

   if hasattr(model, "predict_proba"):
       probability = model.predict_proba(df_input)[0][1]

   elif hasattr(model, "decision_function"):
       scores = model.decision_function(df_input)
       probability = 1 / (1 + np.exp(-scores))

   return {
       "prediction": int(prediction),
       "churn_probability": float(probability) if probability is not None else None
   }

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":

   sample_input = {
   "gender": "Male",
   "SeniorCitizen": 1,
   "Partner": "No",
   "Dependents": "No",
   "tenure": 2,
   "PhoneService": "Yes",
   "MultipleLines": "No",
   "InternetService": "Fiber optic",
   "OnlineSecurity": "No",
   "OnlineBackup": "No",
   "DeviceProtection": "No",
   "TechSupport": "No",
   "StreamingTV": "Yes",
   "StreamingMovies": "Yes",
   "Contract": "Month-to-month",
   "PaperlessBilling": "Yes",
   "PaymentMethod": "Electronic check",
   "MonthlyCharges": 105.0,
   "TotalCharges": "210"
}
   

   result = predict_churn(sample_input)

   print("\n=== CHURN PREDICTION ===")
   print("Prediction:", "Will Churn" if result["prediction"] == 1 else "Will Stay")
   print("Probability:", round(result["churn_probability"], 4))
