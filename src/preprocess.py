
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_PATH, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_PATH = os.path.join(BASE_PATH, "data", "processed", "processed_churn.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


# -----------------------------
# CLEAN + FEATURE ENGINEERING
# -----------------------------
def clean_data(df):

    # Drop ID column (not useful)
    df = df.drop("customerID", axis=1)

    # Fix TotalCharges (IMPORTANT ISSUE IN THIS DATASET)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# -----------------------------
# ENCODING
# -----------------------------
def encode_features(df):

    # One-hot encoding for all categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df


# -----------------------------
# SPLIT DATA
# -----------------------------
def split_data(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# -----------------------------
# SCALING (ONLY NUMERIC IMPACT)
# -----------------------------
def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# -----------------------------
# SAVE PROCESSED DATASET
# -----------------------------
def save_processed_data(df):

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Processed dataset saved at: {PROCESSED_PATH}")


# -----------------------------
# MAIN PIPELINE FUNCTION
# -----------------------------
def run_preprocessing():

    df = load_data()

    print("Data Loaded:", df.shape)

    df = clean_data(df)

    df = encode_features(df)

    save_processed_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, scaler = scale_data(X_train, X_test)

    print("Preprocessing Completed Successfully!")

    return X_train, X_test, y_train, y_test, scaler


# Optional test run
if __name__ == "__main__":
    run_preprocessing()
