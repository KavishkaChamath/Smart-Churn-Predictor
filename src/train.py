
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
DATA_PATH = "/kaggle/working/Smart-Churn-Predictor/data/processed/processed_churn.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------
# SPLIT DATA
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# SCALING (FOR SVM ONLY)
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=200),
    "svm": SVC()
}

# -----------------------------
# SAVE FOLDER
# -----------------------------
model_path = "/kaggle/working/Smart-Churn-Predictor/models"
os.makedirs(model_path, exist_ok=True)

# -----------------------------
# STORE RESULTS
# -----------------------------
results = []

best_model = None
best_cv_f1 = 0
best_name = ""

# -----------------------------
# TRAIN + CV + EVALUATION
# -----------------------------
for name, model in models.items():

    # -------------------------
    # CROSS VALIDATION (TRAIN ONLY)
    # -------------------------
    if name == "svm":
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        joblib.dump(scaler, f"{model_path}/svm_scaler.pkl")

    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    cv_f1 = cv_scores.mean()

    # -------------------------
    # METRICS (TEST SET)
    # -------------------------
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    joblib.dump(model, f"{model_path}/{name}.pkl")

    # -------------------------
    # STORE RESULTS
    # -------------------------
    results.append({
    "Model": name,
    "CV_F1_Mean": cv_f1,
    "CV_F1_Std": cv_scores.std(),
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1
})

    # -------------------------
    # PRINT RESULTS
    # -------------------------
    print("\n==============================")
    print("Model:", name)
    print("CV F1 Score (Mean):", round(cv_f1, 4))
    print("CV F1 Scores (Each Fold):", cv_scores)
    print("CV Std:", round(cv_scores.std(), 4))
    print("Test F1 Score:", round(f1, 4))
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    # -------------------------
    # BEST MODEL (BASED ON CV F1)
    # -------------------------
    if cv_f1 > best_cv_f1:
        best_cv_f1 = cv_f1
        best_model = model
        best_name = name

# -----------------------------
# SAVE BEST MODEL
# -----------------------------
joblib.dump(best_model, f"{model_path}/best_model.pkl")

# -----------------------------
# RESULTS TABLE
# -----------------------------
results_df = pd.DataFrame(results)
print("\n\n===== MODEL COMPARISON TABLE =====")
print(results_df)

print("\nBest Model:", best_name)
print("Best CV F1 Score:", best_cv_f1)
print("\nAll models saved in /models folder")
