import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

bundle = joblib.load("model.joblib")
model = bundle["model"]
encoders = bundle["encoders"]
FEATURE_COLS = bundle["feature_cols"]


def encode(col, value):
    """Encode a categorical value using the saved LabelEncoder, with a safe fallback."""
    le = encoders[col]
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    gender = form.get("gender", "Male")
    married = form.get("married", "No")
    dependents = form.get("dependents", "0")
    education = form.get("education", "Graduate")
    self_employed = form.get("self_employed", "No")
    property_area = form.get("property_area", "Urban")
    credit_history = float(form.get("credit_history", 1))

    applicant_income = float(form.get("applicant_income", 0))
    coapplicant_income = float(form.get("coapplicant_income", 0))
    loan_amount = float(form.get("loan_amount", 0))
    loan_term = float(form.get("loan_term", 360))

    total_income = applicant_income + coapplicant_income

    row = {
        "Gender": encode("Gender", gender),
        "Married": encode("Married", married),
        "Dependents": encode("Dependents", dependents),
        "Education": encode("Education", education),
        "Self_Employed": encode("Self_Employed", self_employed),
        "Credit_History": credit_history,
        "Property_Area": encode("Property_Area", property_area),
        "ApplicantIncomeLog": np.log(max(applicant_income, 1)),
        "CoapplicantIncomeLog": np.log(coapplicant_income + 1),
        "LoanAmountLog": np.log(max(loan_amount, 1)),
        "Loan_Amount_TermLog": np.log(max(loan_term, 1)),
        "Total_Income_Log": np.log(max(total_income, 1)),
    }

    X = [[row[c] for c in FEATURE_COLS]]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    approved = bool(pred == encoders["Loan_Status"].transform(["Y"])[0])
    confidence = round(float(max(proba)) * 100, 1)

    return render_template(
        "index.html",
        result={
            "approved": approved,
            "confidence": confidence,
            "form": form,
        },
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
