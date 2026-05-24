
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
   accuracy_score,
   precision_score,
   recall_score,
   f1_score,
   roc_auc_score,
   confusion_matrix,
   RocCurveDisplay
)

# -----------------------------
# LOAD DATA
# -----------------------------
DATA_PATH = "/kaggle/working/Smart-Churn-Predictor/data/processed/processed_churn.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.2,
   random_state=42,
   stratify=y
)

# -----------------------------
# MODEL PATH
# -----------------------------
model_path = "/kaggle/working/Smart-Churn-Predictor/models"

model_files = [
   f for f in os.listdir(model_path)
   if f.endswith(".pkl") and "scaler" not in f and "best_model" not in f
]

# -----------------------------
# STORE RESULTS
# -----------------------------
results = []

# -----------------------------
# ROC CURVE SETUP (IMPORTANT FIX)
# -----------------------------
plt.figure(figsize=(8, 6))

# -----------------------------
# EVALUATE EACH MODEL
# -----------------------------
for model_file in model_files:

   model_name = model_file.replace(".pkl", "")
   model = joblib.load(os.path.join(model_path, model_file))

   # -----------------------------
   # HANDLE SVM SCALING
   # -----------------------------
   if model_name == "svm":
       scaler = joblib.load(os.path.join(model_path, "svm_scaler.pkl"))
       X_test_input = scaler.transform(X_test)
   else:
       X_test_input = X_test

   # -----------------------------
   # PREDICTIONS
   # -----------------------------
   y_pred = model.predict(X_test_input)

   # -----------------------------
   # ROC-AUC + PROBABILITY FIX
   # -----------------------------
   roc_auc = None
   y_prob = None

   # Try probability first
   if hasattr(model, "predict_proba"):
       y_prob = model.predict_proba(X_test_input)[:, 1]

   # Fallback for SVM or others
   elif hasattr(model, "decision_function"):
       y_scores = model.decision_function(X_test_input)
       y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

   # Compute ROC-AUC if possible
   if y_prob is not None:
       roc_auc = roc_auc_score(y_test, y_prob)

       RocCurveDisplay.from_predictions(
           y_test,
           y_prob,
           name=model_name
       )

   # -----------------------------
   # METRICS
   # -----------------------------
   acc = accuracy_score(y_test, y_pred)
   prec = precision_score(y_test, y_pred)
   rec = recall_score(y_test, y_pred)
   f1 = f1_score(y_test, y_pred)

   # -----------------------------
   # CONFUSION MATRIX
   # -----------------------------
   cm = confusion_matrix(y_test, y_pred)

   plt.figure()
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
   plt.title(f"Confusion Matrix - {model_name}")
   plt.xlabel("Predicted")
   plt.ylabel("Actual")
   plt.show()

   # -----------------------------
   # STORE RESULTS
   # -----------------------------
   results.append({
       "Model": model_name,
       "Accuracy": acc,
       "Precision": prec,
       "Recall": rec,
       "F1-Score": f1,
       "ROC-AUC": roc_auc
   })

   # -----------------------------
   # PRINT RESULTS
   # -----------------------------
   print("\n==============================")
   print("Model:", model_name)
   print("Accuracy:", round(acc, 4))
   print("Precision:", round(prec, 4))
   print("Recall:", round(rec, 4))
   print("F1 Score:", round(f1, 4))
   print("ROC-AUC:", round(roc_auc, 4) if roc_auc else "N/A")

# -----------------------------
# FINAL TABLE
# -----------------------------
results_df = pd.DataFrame(results)

print("\n\n===== FINAL MODEL COMPARISON =====")
print(results_df.sort_values(by="ROC-AUC", ascending=False))

results_df.to_csv("models/model_results.csv", index=False)

# # -----------------------------
# # FINAL ROC CURVE
# # # -----------------------------
# plt.title("ROC Curve Comparison (All Models)")
# plt.legend()
# plt.show()
