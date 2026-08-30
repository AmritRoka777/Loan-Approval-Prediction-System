import pandas as pd


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

file_path = "dataset/Dataset.csv"

data = pd.read_csv(file_path)


# --------------------------------------------------
# 2. Display basic information
# --------------------------------------------------

print("\n========== DATASET SHAPE ==========")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])


# --------------------------------------------------
# 3. Display column names
# --------------------------------------------------

print("\n========== COLUMN NAMES ==========")

for column in data.columns:
    print(column)


# --------------------------------------------------
# 4. Display first 5 rows
# --------------------------------------------------

print("\n========== FIRST 5 ROWS ==========")
print(data.head())


# --------------------------------------------------
# 5. Display data types
# --------------------------------------------------

print("\n========== DATA TYPES ==========")
print(data.dtypes)


# --------------------------------------------------
# 6. Check missing values
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())


# --------------------------------------------------
# 7. Display statistical information
# --------------------------------------------------

print("\n========== STATISTICAL INFORMATION ==========")
print(data.describe())