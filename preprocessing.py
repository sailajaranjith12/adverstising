import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Dataset/Advertising Budget and Sales.csv"
df = pd.read_csv(file_path)

# Drop the unnamed index column if it exists
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Check for missing values and handle them
df = df.dropna()  # Drop rows with missing values

# Check for duplicates
df = df.drop_duplicates()

# Standardize the feature variables
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Save the preprocessed data to a single file
df.to_csv("Dataset/processed_data.csv", index=False)
