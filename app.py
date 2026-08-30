from flask import Flask, render_template, request
import pandas as pd
import joblib


# ==========================================================
# CREATE FLASK APPLICATION
# ==========================================================

app = Flask(__name__)


# ==========================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# ==========================================================

model = joblib.load("model/loan_model.joblib")


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        form_data={}
    )


# ==========================================================
# LOAN PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # ------------------------------------------------------
    # GET DATA FROM FORM
    # ------------------------------------------------------

    form_data = {
        "Gender": request.form["Gender"],
        "Married": request.form["Married"],
        "Dependents": request.form["Dependents"],
        "Education": request.form["Education"],
        "Self_Employed": request.form["Self_Employed"],
        "ApplicantIncome": request.form["ApplicantIncome"],
        "CoapplicantIncome": request.form["CoapplicantIncome"],
        "LoanAmount": request.form["LoanAmount"],
        "Loan_Amount_Term": request.form["Loan_Amount_Term"],
        "Credit_History": request.form["Credit_History"],
        "Property_Area": request.form["Property_Area"]
    }


    # ------------------------------------------------------
    # CONVERT DATA TO CORRECT TYPES
    # ------------------------------------------------------

    applicant_data = pd.DataFrame([{

        "Gender": form_data["Gender"],

        "Married": form_data["Married"],

        "Dependents": form_data["Dependents"],

        "Education": form_data["Education"],

        "Self_Employed": form_data["Self_Employed"],

        "ApplicantIncome": float(
            form_data["ApplicantIncome"]
        ),

        "CoapplicantIncome": float(
            form_data["CoapplicantIncome"]
        ),

        "LoanAmount": float(
            form_data["LoanAmount"]
        ),

        "Loan_Amount_Term": float(
            form_data["Loan_Amount_Term"]
        ),

        "Credit_History": float(
            form_data["Credit_History"]
        ),

        "Property_Area": form_data["Property_Area"]

    }])


    # ------------------------------------------------------
    # MAKE PREDICTION
    # ------------------------------------------------------

    prediction = model.predict(applicant_data)[0]


    # ------------------------------------------------------
    # GET APPROVAL PROBABILITY
    # ------------------------------------------------------

    probability = model.predict_proba(
        applicant_data
    )[0][1]


    approval_percentage = round(
        probability * 100,
        2
    )


    # ------------------------------------------------------
    # DETERMINE RESULT
    # ------------------------------------------------------

    if prediction == 1:

        result = "Loan Approved"

        result_class = "approved"

        result_icon = "✓"

        result_message = (
            "Based on the information provided, "
            "the machine learning model predicts "
            "that this application is likely to be approved."
        )

    else:

        result = "Loan Not Approved"

        result_class = "rejected"

        result_icon = "!"

        result_message = (
            "Based on the information provided, "
            "the machine learning model predicts "
            "that this application is unlikely to be approved."
        )


    # ------------------------------------------------------
    # SEND RESULT TO RESULT PAGE
    # ------------------------------------------------------

    return render_template(

        "result.html",

        form_data=form_data,

        result=result,

        result_class=result_class,

        result_icon=result_icon,

        result_message=result_message,

        approval_percentage=approval_percentage

    )


# ==========================================================
# RUN FLASK APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(debug=True)