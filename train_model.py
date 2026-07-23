"""
Trains the loan-approval model using the same preprocessing steps as the
original notebook (missing-value imputation, log transforms, label encoding),
and saves everything the Flask app needs to make predictions.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/Loan_Prediction.csv"
MODEL_PATH = "model.joblib"

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                     "Self_Employed", "Property_Area"]
FEATURE_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
                "Credit_History", "Property_Area", "ApplicantIncomeLog",
                "CoapplicantIncomeLog", "LoanAmountLog", "Loan_Amount_TermLog",
                "Total_Income_Log"]


def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)

    # Fill missing values: mean for numerical, mode for categorical
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mean())
    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Engineered features (single, correct log transforms)
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["ApplicantIncomeLog"] = np.log(df["ApplicantIncome"].clip(lower=1))
    df["CoapplicantIncomeLog"] = np.log(df["CoapplicantIncome"] + 1)
    df["LoanAmountLog"] = np.log(df["LoanAmount"].clip(lower=1))
    df["Loan_Amount_TermLog"] = np.log(df["Loan_Amount_Term"].clip(lower=1))
    df["Total_Income_Log"] = np.log(df["Total_Income"].clip(lower=1))

    return df


def train():
    df = load_and_prepare()

    encoders = {}
    for col in CATEGORICAL_COLS + ["Loan_Status"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURE_COLS]
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tuned Random Forest, mirroring the notebook's hyperparameter search
    model = RandomForestClassifier(
        n_estimators=200, min_samples_split=15, max_depth=7, random_state=42
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc * 100:.2f}%")

    joblib.dump(
        {"model": model, "encoders": encoders, "feature_cols": FEATURE_COLS},
        MODEL_PATH,
    )
    print(f"Saved model + encoders to {MODEL_PATH}")


if __name__ == "__main__":
    train()
