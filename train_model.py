import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

file_path = "dataset/Dataset.csv"

data = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Total records:", len(data))


# ==========================================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================================

# Loan_ID is only an identification number.
# It does not provide useful information for prediction.

data = data.drop("Loan_ID", axis=1)


# ==========================================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================================

# X = input features
# y = value we want to predict

X = data.drop("Loan_Status", axis=1)

y = data["Loan_Status"]


# Convert:
# Y = 1 (Approved)
# N = 0 (Rejected)

y = y.map({
    "Y": 1,
    "N": 0
})


# ==========================================================
# 4. DEFINE COLUMN TYPES
# ==========================================================

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]


# ==========================================================
# 5. PREPROCESSING FOR NUMERICAL DATA
# ==========================================================

numerical_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ==========================================================
# 6. PREPROCESSING FOR CATEGORICAL DATA
# ==========================================================

categorical_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ==========================================================
# 7. COMBINE BOTH PREPROCESSING PIPELINES
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ==========================================================
# 8. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================================
# 9. CREATE LOGISTIC REGRESSION PIPELINE
# ==========================================================

logistic_model = Pipeline(steps=[
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(max_iter=1000)
    )
])


# ==========================================================
# 10. CREATE RANDOM FOREST PIPELINE
# ==========================================================

random_forest_model = Pipeline(steps=[
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )
])


# ==========================================================
# 11. TRAIN LOGISTIC REGRESSION
# ==========================================================

print("\n========== TRAINING LOGISTIC REGRESSION ==========")

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)


# ==========================================================
# 12. EVALUATE LOGISTIC REGRESSION
# ==========================================================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_confusion = confusion_matrix(
    y_test,
    logistic_predictions
)


# ==========================================================
# 13. TRAIN RANDOM FOREST
# ==========================================================

print("\n========== TRAINING RANDOM FOREST ==========")

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)


# ==========================================================
# 14. EVALUATE RANDOM FOREST
# ==========================================================

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

random_forest_precision = precision_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

random_forest_recall = recall_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

random_forest_f1 = f1_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

random_forest_confusion = confusion_matrix(
    y_test,
    random_forest_predictions
)


# ==========================================================
# 15. DISPLAY RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print("\nLOGISTIC REGRESSION")
print("-" * 30)
print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1-Score :", round(logistic_f1, 4))

print("\nConfusion Matrix:")
print(logistic_confusion)


print("\nRANDOM FOREST")
print("-" * 30)
print("Accuracy :", round(random_forest_accuracy, 4))
print("Precision:", round(random_forest_precision, 4))
print("Recall   :", round(random_forest_recall, 4))
print("F1-Score :", round(random_forest_f1, 4))

print("\nConfusion Matrix:")
print(random_forest_confusion)


# ==========================================================
# 16. SELECT THE BETTER MODEL
# ==========================================================

if random_forest_f1 > logistic_f1:
    best_model = random_forest_model
    best_model_name = "Random Forest"
    best_f1 = random_forest_f1

else:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_f1 = logistic_f1


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Selected model:", best_model_name)
print("F1-Score:", round(best_f1, 4))


# ==========================================================
# 17. SAVE THE BEST MODEL
# ==========================================================

model_path = "model/loan_model.joblib"

joblib.dump(best_model, model_path)

print("\nModel saved successfully!")
print("Saved to:", model_path)