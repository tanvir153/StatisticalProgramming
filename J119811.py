
# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore


# Load Dataset

file_path = r"D:\03 Statistical Programming Project\Dataset for House Price Analysis Tanvir\Dataset.csv"
df = pd.read_csv(file_path)

# Select relevant features
selected_features = [
    "Land_Area", "Floor_Area", "Num_bathrooms", "Num_rooms",
    "Crimerate in area", "distance to nearest MRT Station", "distance to nearest Hospital"
]
target = "Price"

# Create dataset with selected features
df_selected = df[selected_features + [target]].copy()


# Display Dataset Information

print("\n Dataset Information:")
print(df_selected.info())

# Display summary statistics
print("\n Summary Statistics:")
print(df_selected.describe())

# Check for missing values
print("\n Missing Values in Dataset:")
missing_values = df_selected.isnull().sum()
print(missing_values)

# Handle missing values (if any)
if missing_values.sum() > 0:
    df_selected.fillna(df_selected.mean(), inplace=True)
    print("\n Missing values filled with column means.")


# Exploratory Data Analysis (EDA)
# Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_selected.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#  Price Distribution Before Log Transformation
plt.figure(figsize=(8, 6))
sns.histplot(df_selected["Price"], bins=50, kde=True, color="blue")
plt.title("House Price Distribution (Before Log Transformation)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

#  Box Plot Before Outlier Removal
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_selected["Price"])
plt.title("Box Plot Before Outlier Removal")
plt.show()


# Apply Log Transformation & Remove Outliers

df_selected["Log_Price"] = np.log1p(df_selected["Price"])
df_selected.drop(columns=["Price"], inplace=True)

# Log-Transformed Price Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df_selected["Log_Price"], bins=50, kde=True, color="green")
plt.title("Log-Transformed House Price Distribution")
plt.xlabel("Log Price")
plt.ylabel("Frequency")
plt.show()

# Remove Outliers using Z-score
df_selected["Log_Price_Zscore"] = zscore(df_selected["Log_Price"])
df_selected = df_selected[(df_selected["Log_Price_Zscore"] > -3) & (df_selected["Log_Price_Zscore"] < 3)]
df_selected.drop(columns=["Log_Price_Zscore"], inplace=True)

print(f"\n Dataset Size After Removing Outliers: {df_selected.shape}")

#  Box Plot After Outlier Removal
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_selected["Log_Price"])
plt.title("Box Plot After Outlier Removal")
plt.show()


# Data Splitting & Feature Scaling

X = df_selected.drop(columns=["Log_Price"])
y = df_selected["Log_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Apply Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Hyperparameter Tuning for Ridge Regression
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100, 1000]}
ridge_grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring="r2")
ridge_grid.fit(X_train_poly, y_train)

# Get the best alpha value
best_alpha = ridge_grid.best_params_["alpha"]
print(f"\n Best Alpha for Ridge Regression: {best_alpha}")

# Train Ridge Regression with Best Alpha
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_poly, y_train)

# Feature Importance Analysis
feature_importance = pd.DataFrame({'Feature': poly.get_feature_names_out(), 'Coefficient': ridge_model.coef_})
feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

print("\n Feature Importance (Top 10 Features):")
print(feature_importance.head(10))

# **Feature Importance Plot**
plt.figure(figsize=(10, 6))
sns.barplot(
    x=feature_importance["Coefficient"][:10], 
    y=feature_importance["Feature"][:10], 
    palette="viridis"
)
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.title(" Feature Importance (Top 10 Features)")
plt.show()


# Make Predictions & Evaluate Model
y_pred = ridge_model.predict(X_test_poly)
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

mse = mean_squared_error(y_test_exp, y_pred_exp)
r2 = r2_score(y_test_exp, y_pred_exp)

print("\n Model Performance:")
print(f" Mean Squared Error (MSE): {mse:.2f}")
print(f" R-squared Score (R²): {r2:.4f}")


# Visualization - Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test_exp, y_pred_exp, alpha=0.5, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Ridge Regression)")
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], color='red', linestyle='dashed')
plt.show()

# Hyperparameter Tuning Visualization
alpha_values = param_grid["alpha"]
mean_test_scores = ridge_grid.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 6))
plt.plot(alpha_values, mean_test_scores, marker="o", linestyle="dashed", color="green")
plt.xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Mean R² Score")
plt.title("Hyperparameter Tuning: R² Score vs. Alpha")
plt.show()
