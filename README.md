

# Loan Approval Prediction System

A Machine Learning-based web application that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant information.

The system uses Machine Learning to analyze different factors such as income, education, credit history, loan amount, property area, and other applicant details.

## Features

- Predicts whether a loan application is likely to be Approved or Rejected
- Displays the loan approval probability
- Simple and user-friendly interface
- Uses Machine Learning for prediction
- Accepts different applicant and loan details
- Displays the prediction result on a separate result page
- Responsive web interface

## Technologies Used

- Python
- Flask
- Machine Learning
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- Git & GitHub

## How the System Works

1. The user enters the applicant's information.
2. The information is sent to the Flask application.
3. The trained Machine Learning model processes the information.
4. The system predicts the loan status.
5. The result is displayed as:
   - Approved, or
   - Rejected
6. The system also displays the approval probability.

## Machine Learning Model

The system was trained using a loan dataset containing applicant and loan information.

Two Machine Learning algorithms were tested:

- Logistic Regression
- Random Forest Classifier

After evaluation, **Logistic Regression** was selected as the final model because it achieved better performance for this dataset.

The trained model is saved as:

model/loan_model.joblib








