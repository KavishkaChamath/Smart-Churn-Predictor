import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")   # no display needed when running from terminal
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# ─── PATHS (local, not Kaggle) ───────────────────────────────────────────────────
BASE_PATH  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
DATA_PATH  = os.path.join(BASE_PATH, "data", "processed", "processed_churn.csv")
MODEL_PATH = os.path.join(BASE_PATH, "models")

# ─── LOAD DATA ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df.drop("Churn", axis=1)
y  = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── MODEL FILES (skip scaler and best_model) ────────────────────────────────────
model_files = [
    f for f in os.listdir(MODEL_PATH)
    if f.endswith(".pkl") and "scaler" not in f and "best_model" not in f
]

results     = []
json_results = {}

# ─── ROC CURVE combined plot setup ───────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")

# ─── EVALUATE EACH MODEL ─────────────────────────────────────────────────────────
for model_file in sorted(model_files):
    model_name = model_file.replace(".pkl", "")
    model      = joblib.load(os.path.join(MODEL_PATH, model_file))

    # Scale only for SVM
    if model_name == "svm":
        scaler       = joblib.load(os.path.join(MODEL_PATH, "svm_scaler.pkl"))
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test

    y_pred = model.predict(X_test_input)

    # ROC-AUC + probabilities
    roc_auc = None
    y_prob  = None
    fpr_list = tpr_list = None

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_input)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test_input)
        y_prob   = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    if y_prob is not None:
        roc_auc          = roc_auc_score(y_test, y_prob)
        fpr, tpr, _      = roc_curve(y_test, y_prob)
        fpr_list         = fpr.tolist()
        tpr_list         = tpr.tolist()
        # Add to combined ROC plot
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*30}\nModel: {model_name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}" if roc_auc else "ROC-AUC: N/A")

    results.append({
        "Model": model_name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc
    })

    # Build JSON entry (with ROC curve data for Streamlit)
    display_name = model_name.replace("_", " ").title()
    json_results[display_name] = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "roc_auc":          round(roc_auc, 4) if roc_auc else None,
        "confusion_matrix": cm.tolist(),
        "fpr":              fpr_list,   # for ROC plot in Streamlit
        "tpr":              tpr_list,
    }

# ─── Save combined ROC plot image ────────────────────────────────────────────────
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — All Models")
plt.legend(loc="lower right")
plt.tight_layout()
roc_img_path = os.path.join(MODEL_PATH, "roc_curves.png")
plt.savefig(roc_img_path, dpi=150)
plt.close()
print(f"\n✅ ROC curve image saved: {roc_img_path}")

# ─── Save JSON ───────────────────────────────────────────────────────────────────
json_path = os.path.join(MODEL_PATH, "evaluation_results.json")
with open(json_path, "w") as f:
    json.dump(json_results, f, indent=2)
print(f"✅ Evaluation JSON saved:  {json_path}")

# ─── Save CSV ────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
csv_path   = os.path.join(MODEL_PATH, "model_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"✅ Results CSV saved:      {csv_path}")

print("\n===== FINAL MODEL COMPARISON =====")
print(results_df.sort_values(by="F1-Score", ascending=False).to_string(index=False))

best = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
print(f"\n🏆 Best Model (F1): {best}")