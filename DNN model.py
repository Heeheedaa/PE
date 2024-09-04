# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:25:08 2024

@author: Yulin
"""


# DNN model with GPU acceleration

# Import necessary libraries
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

 
# Check for GPU availability
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Define dependent variable and independent variables
y = 
X = 

# Standardize the data including dependent variable
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


# Standardize the data including dependent variable
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store evaluation results for each fold
mse_list = []
rmse_list = []
r2_list = []


# Iterate through each fold
for train_index, test_index in kf.split(X):
    X_train_scaled_DNN, X_test_scaled_DNN = X_scaled[train_index], X_scaled[test_index]
    y_train_DNN, y_test_DNN = y.iloc[train_index], y.iloc[test_index]    
    
    # Build the neural network model
    model_DNN = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_scaled_DNN.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model_DNN.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on the GPU if available
    with tf.device('/GPU:0'):
        model_DNN.fit(X_train_scaled_DNN, y_train_DNN, epochs=50, batch_size=64, verbose=0)


    # Make predictions on the test data
    y_pred_DNN = model_DNN.predict(X_test_scaled_DNN).flatten()
    
    # Evaluate the model's performance for this fold
    mse = mean_squared_error(y_test_DNN, y_pred_DNN)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_DNN, y_pred_DNN)

    # Append results to lists
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Calculate and print the average performance metrics across all folds
average_mse_DNN = np.mean(mse_list)
average_rmse_DNN = np.mean(rmse_list)
average_r2_DNN = np.mean(r2_list)

print(f"DNN Average RMSE: {average_rmse_DNN}")
print(f"DNN Average R-squared: {average_r2_DNN}")