from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('linear_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please ensure 'linear_regression_model.pkl' and 'scaler.pkl' are in the same directory as 'app.py'.")
    exit()
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Define the features that the model expects
# This list must match the 'features' list used during training
MODEL_FEATURES = ['u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth',
                  'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient', 'torque']

@app.route('/')
def home():
    """Renders the HTML form for temperature prediction."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives input data from the HTML form, scales it,
    makes a prediction using the loaded model, and returns the result.
    """
    try:
        # Get data from the form
        data = request.form.to_dict()

        # Convert input values to float, handling potential errors
        input_values = []
        for feature in MODEL_FEATURES:
            try:
                input_values.append(float(data.get(feature, 0.0))) # Default to 0.0 if not provided
            except ValueError:
                return jsonify({'error': f"Invalid input for {feature}. Please enter a numeric value."}), 400

        # Convert the list of input values to a NumPy array and reshape for scaling
        input_array = np.array(input_values).reshape(1, -1)

        # Scale the input features using the loaded scaler
        scaled_input_array = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input_array)[0]

        return jsonify({'predicted_temperature': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, use a production-ready WSGI server like Gunicorn
    app.run(debug=True) # debug=True allows for automatic reloading on code changes
# For vercel deployment
from flask import Response

def handler(environ, start_response):
    return app.wsgi_app(environ, start_response)
