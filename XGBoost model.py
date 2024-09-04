# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:20:11 2024

@author: Yulin
"""


#XGBoost model

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

  
# Define dependent variable and independent variables
y = 
X = 

# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store evaluation results for each fold
mse_list = []
rmse_list = []
r2_list = []

# Iterate through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Build the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000, max_depth=6, learning_rate=0.2,random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importance using SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance for this fold
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Append results to lists
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Calculate and print the average performance metrics across all folds
average_mse = np.mean(mse_list)
average_rmse = np.mean(rmse_list)
average_r2 = np.mean(r2_list)

print(f"XGBoost Average RMSE: {average_rmse}")
print(f"XGBoost Average R-squared: {average_r2}")