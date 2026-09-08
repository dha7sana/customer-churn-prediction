print("THIS IS MY APP.PY")
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- Basic fields ---
        gender        = 1 if request.form['gender'] == 'Male' else 0
        senior        = int(request.form['SeniorCitizen'])
        partner       = 1 if request.form['Partner'] == 'Yes' else 0
        dependents    = 1 if request.form['Dependents'] == 'Yes' else 0
        tenure        = float(request.form['tenure'])
        phone         = 1 if request.form['PhoneService'] == 'Yes' else 0
        paperless     = 1 if request.form['PaperlessBilling'] == 'Yes' else 0
        monthly       = float(request.form['MonthlyCharges'])
        total         = float(request.form['TotalCharges'])

        # --- MultipleLines (one-hot) ---
        multiple_lines = request.form['MultipleLines']
        ml_no_phone = 1 if multiple_lines == 'No phone service' else 0
        ml_yes      = 1 if multiple_lines == 'Yes' else 0

        # --- InternetService (one-hot) ---
        internet = request.form['InternetService']
        is_fiber = 1 if internet == 'Fiber optic' else 0
        is_no    = 1 if internet == 'No' else 0

        # --- OnlineSecurity (one-hot) ---
        online_sec = request.form['OnlineSecurity']
        os_no_int = 1 if online_sec == 'No internet service' else 0
        os_yes    = 1 if online_sec == 'Yes' else 0

        # --- OnlineBackup (one-hot) ---
        online_bk = request.form['OnlineBackup']
        ob_no_int = 1 if online_bk == 'No internet service' else 0
        ob_yes    = 1 if online_bk == 'Yes' else 0

        # --- DeviceProtection (one-hot) ---
        device = request.form['DeviceProtection']
        dp_no_int = 1 if device == 'No internet service' else 0
        dp_yes    = 1 if device == 'Yes' else 0

        # --- TechSupport (one-hot) ---
        tech = request.form['TechSupport']
        ts_no_int = 1 if tech == 'No internet service' else 0
        ts_yes    = 1 if tech == 'Yes' else 0

        # --- StreamingTV (one-hot) ---
        stv = request.form['StreamingTV']
        stv_no_int = 1 if stv == 'No internet service' else 0
        stv_yes    = 1 if stv == 'Yes' else 0

        # --- StreamingMovies (one-hot) ---
        smv = request.form['StreamingMovies']
        smv_no_int = 1 if smv == 'No internet service' else 0
        smv_yes    = 1 if smv == 'Yes' else 0

        # --- Contract (one-hot) ---
        contract = request.form['Contract']
        c_one_year = 1 if contract == 'One year' else 0
        c_two_year = 1 if contract == 'Two year' else 0

        # --- PaymentMethod (one-hot) ---
        payment = request.form['PaymentMethod']
        pm_credit   = 1 if payment == 'Credit card (automatic)' else 0
        pm_echeck   = 1 if payment == 'Electronic check' else 0
        pm_mailed   = 1 if payment == 'Mailed check' else 0

        # --- TenureGroup (one-hot) ---
        tg = request.form['TenureGroup']
        tg_loyal  = 1 if tg == 'Loyal' else 0
        tg_medium = 1 if tg == 'Medium-term' else 0
        tg_new    = 1 if tg == 'New' else 0
        tg_short  = 1 if tg == 'Short-term' else 0

        # --- SpendCategory (one-hot) ---
        sc = request.form['SpendCategory']
        sc_medium = 1 if sc == 'Medium' else 0
        sc_high   = 1 if sc == 'High' else 0

        # --- Assemble feature array in exact model order ---
        features = pd.DataFrame([[
            gender, senior, partner, dependents, tenure, phone, paperless,
            monthly, total,
            ml_no_phone, ml_yes,
            is_fiber, is_no,
            os_no_int, os_yes,
            ob_no_int, ob_yes,
            dp_no_int, dp_yes,
            ts_no_int, ts_yes,
            stv_no_int, stv_yes,
            smv_no_int, smv_yes,
            c_one_year, c_two_year,
            pm_credit, pm_echeck, pm_mailed,
            tg_loyal, tg_medium, tg_new, tg_short,
            sc_medium, sc_high
        ]], columns=model.feature_names_in_)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        result = "⚠️ Customer is likely to CHURN" if prediction == 1 else "✅ Customer is NOT likely to churn"
        color  = "#e74c3c" if prediction == 1 else "#27ae60"

        return render_template("index.html",
                               result=result,
                               probability=f"{probability:.1f}%",
                               color=color)

    except Exception as e:
        return render_template("index.html",
                               result=f"❌ Error: {str(e)}",
                               color="#e74c3c")

if __name__ == "__main__":
    app.run(debug=True)
