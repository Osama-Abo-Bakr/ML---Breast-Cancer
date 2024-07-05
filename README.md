# Breast Cancer Detection Project

## Project Overview

This project focuses on predicting breast cancer using machine learning algorithms. The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. The project involves data preprocessing, visualization, model building, and evaluation.

## Libraries and Tools

- **Pandas**: For data manipulation
- **NumPy**: For numerical computations
- **Matplotlib**: For data visualization
- **Seaborn**: For statistical data visualization
- **Scikit-learn**: For data preprocessing and model building
- **XGBoost**: For advanced gradient boosting

## Project Steps

1. **Reading and Exploring Data**:
    - Loaded the dataset and displayed the first few records.
    - Checked for null values and data types.

2. **Data Cleaning**:
    - Removed irrelevant columns.
    - Dropped rows with outlier values based on specific thresholds.

3. **Data Visualization**:
    - Plotted heatmap to visualize correlations between features.
    - Displayed histograms for the distribution of features.

4. **Data Preprocessing**:
    - Label encoded the target variable.
    - Split the dataset into training and testing sets.

5. **Model Building and Evaluation**:
    - **Logistic Regression**:
        - Trained and evaluated the model.
    - **Random Forest Classifier**:
        - Tuned hyperparameters and trained the model.
    - **XGBoost Classifier**:
        - Built and evaluated the model with specific hyperparameters.

## Results

- **Logistic Regression**:
    - Training Accuracy: 0.956
    - Testing Accuracy: 0.934

- **Random Forest Classifier**:
    - Training Accuracy: 0.976
    - Testing Accuracy: 0.95

- **XGBoost Classifier**:
    - Training Accuracy: 1.0
    - Testing Accuracy: 0.98

## Conclusion

The project demonstrates the effective use of machine learning algorithms for breast cancer prediction. The models were able to achieve high accuracy, showcasing the potential of machine learning in medical diagnostics.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/breast-cancer-detection.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script to see the results.

## Contact

Feel free to reach out if you have any questions or suggestions.

- **Email**: [My Mail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn Profile](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Reading the data
data = pd.read_csv('path_to_your_data.csv')
data.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Visualizing correlations
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, square=True, cmap="Blues", fmt="0.2f", cbar=True)
plt.show()

# Data preprocessing
data.drop(data[data["smoothness_worst"] >= 0.20].index, inplace=True)
data.drop(data[data["smoothness_mean"] >= 0.15].index, inplace=True)
data.drop(data[data["fractal_dimension_worst"] > 0.15].index, inplace=True)
data.drop(data[data["area_mean"] > 2000].index, inplace=True)
data.drop(data[data["smoothness_se"] > 0.02].index, inplace=True)

# Label encoding
label_encoder = LabelEncoder()
data["diagnosis"] = label_encoder.fit_transform(data["diagnosis"])

# Splitting data
X = data.drop(columns="diagnosis")
Y = data["diagnosis"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Logistic Regression model
model_Lo = LogisticRegression(max_iter=15000)
model_Lo.fit(x_train, y_train)
print("Logistic Regression - Training Accuracy:", model_Lo.score(x_train, y_train))
print("Logistic Regression - Testing Accuracy:", model_Lo.score(x_test, y_test))

# Random Forest model
model_RF = RandomForestClassifier(n_estimators=1000, max_depth=1000, min_samples_leaf=7, min_samples_split=7, max_leaf_nodes=5, max_features=5)
model_RF.fit(x_train, y_train)
print("Random Forest - Training Accuracy:", model_RF.score(x_train, y_train))
print("Random Forest - Testing Accuracy:", model_RF.score(x_test, y_test))

# XGBoost model
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.5)
model_xgb.fit(x_train, y_train)
print("XGBoost - Training Accuracy:", model_xgb.score(x_train, y_train))
print("XGBoost - Testing Accuracy:", model_xgb.score(x_test, y_test))
```
