import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Load the dataset
try:
    df = pd.read_csv("measures_v2.csv")
except FileNotFoundError:
    print("Error: measures_v2.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Separate test profiles (profile_id 65 and 72)
df_test = df[(df['profile_id'] == 65) | (df['profile_id'] == 72)]
df = df[(df['profile_id'] != 65) & (df['profile_id'] != 72)]

# Define features (X) and target (y)
# 'pm' is the Permanent Magnet Temperature, which is our target
features = ['u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth',
            'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient', 'torque']
target = 'pm'

X = df[features]
y = df[target]

X_test_final = df_test[features]
y_test_final = df_test[target]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_final_scaled = scaler.transform(X_test_final) # Scale the final test set as well

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val_scaled)

# Evaluate the model on the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)

print(f"Validation Mean Squared Error: {mse_val:.2f}")
print(f"Validation Root Mean Squared Error: {rmse_val:.2f}")
print(f"Validation R-squared: {r2_val:.2f}")

# Make predictions on the final test set
y_pred_test_final = model.predict(X_test_final_scaled)

# Evaluate the model on the final test set
mse_test_final = mean_squared_error(y_test_final, y_pred_test_final)
rmse_test_final = np.sqrt(mse_test_final)
r2_test_final = r2_score(y_test_final, y_pred_test_final)

print(f"Final Test Mean Squared Error: {mse_test_final:.2f}")
print(f"Final Test Root Mean Squared Error: {rmse_test_final:.2f}")
print(f"Final Test R-squared: {r2_test_final:.2f}")

# Save the trained model and the scaler to .pkl files
model_filename = 'linear_regression_model.pkl'
scaler_filename = 'scaler.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Trained model saved as {model_filename}")

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved as {scaler_filename}")

print("\nModel training and saving complete. You can now use 'linear_regression_model.pkl' and 'scaler.pkl' for predictions.")
