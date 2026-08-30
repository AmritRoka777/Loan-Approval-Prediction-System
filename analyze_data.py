import pandas as pd


# --------------------------------------------------
# Load the dataset
# --------------------------------------------------

file_path = "dataset/Dataset.csv"

data = pd.read_csv(file_path)


# --------------------------------------------------
# Loan approval distribution
# --------------------------------------------------

print("\n========== LOAN STATUS ==========")
print(data["Loan_Status"].value_counts())


# --------------------------------------------------
# Gender distribution
# --------------------------------------------------

print("\n========== GENDER ==========")
print(data["Gender"].value_counts())


# --------------------------------------------------
# Education distribution
# --------------------------------------------------

print("\n========== EDUCATION ==========")
print(data["Education"].value_counts())


# --------------------------------------------------
# Married distribution
# --------------------------------------------------

print("\n========== MARRIED ==========")
print(data["Married"].value_counts())


# --------------------------------------------------
# Property area distribution
# --------------------------------------------------

print("\n========== PROPERTY AREA ==========")
print(data["Property_Area"].value_counts())


# --------------------------------------------------
# Credit history distribution
# --------------------------------------------------

print("\n========== CREDIT HISTORY ==========")
print(data["Credit_History"].value_counts())


# --------------------------------------------------
# Self-employed distribution
# --------------------------------------------------

print("\n========== SELF EMPLOYED ==========")
print(data["Self_Employed"].value_counts())


# --------------------------------------------------
# Dependents distribution
# --------------------------------------------------

print("\n========== DEPENDENTS ==========")
print(data["Dependents"].value_counts())