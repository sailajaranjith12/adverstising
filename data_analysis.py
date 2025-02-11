import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
file_path = "Dataset/processed_data.csv"
df = pd.read_csv(file_path)

# Display basic information and statistics
print("Dataset Info:\n", df.info())
print("\nSummary Statistics:\n", df.describe())

# Check for correlations
correlation_matrix = df.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Visualizations
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot for relationships between variables
sns.pairplot(df)
plt.show()

# Distribution of Sales
df['Sales ($)'].hist(bins=20, edgecolor='black')
plt.xlabel("Sales ($)")
plt.ylabel("Frequency")
plt.title("Sales Distribution")
plt.show()

# Scatter plots for feature vs sales
features = df.columns[:-1]  # Exclude target variable
for feature in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df['Sales ($)'])
    plt.xlabel(feature)
    plt.ylabel("Sales ($)")
    plt.title(f"{feature} vs Sales ($)")
    plt.show()
