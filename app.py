import random
import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Load your trained model (assumed to be a scikit-learn model saved using joblib)
class RealRULModel:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None  # Fallback if model fails

    def predict(self, data):
        if self.model:
            try:
                # Return prediction result (just ensure the input is correct)
                prediction = self.model.predict(data)
                return prediction
            except Exception as e:
                print(f"Error during model prediction: {e}")
        # Fallback to default RUL if model fails
        return [100]  # Assume safe RUL as fallback


# Instantiate the model
MODEL_PATH = 'rul_predictor_model.pkl'  # Path to your trained model
rul_model = RealRULModel(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# JSON file for storing prediction history
JSON_FILE = 'rul_history.json'


# Initialize JSON storage
def init_json():
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
            json.dump([], f)


init_json()


# RUL prediction and recommendation logic with logical randomization
def predict_rul_and_analyze(sensor_data, rul_model):
    try:
        # Function to simulate natural fluctuations in sensor data within a logical range
        def add_sensor_fluctuation(value, fluctuation_range):
            # Introduce random fluctuation based on the value range
            return value + random.uniform(-fluctuation_range, fluctuation_range)

        # Map sensor data and apply logical fluctuations based on expected ranges
        mapped_data = {
            "sensor_measurement_1": add_sensor_fluctuation(sensor_data.get("vibration", 0.0), 0.1),
            "sensor_measurement_2": add_sensor_fluctuation(sensor_data.get("temperature", 0.0), 0.5),
            "sensor_measurement_3": add_sensor_fluctuation(sensor_data.get("pressure", 0.0), 0.3),
            "sensor_measurement_4": add_sensor_fluctuation(sensor_data.get("humidity", 0.0), 2.0),
            "sensor_measurement_5": add_sensor_fluctuation(sensor_data.get("rpm", 0.0), 20.0)
        }

        # Assuming the model expects the features in a certain order
        required_features = ['sensor_measurement_1', 'sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4', 'sensor_measurement_5']

        # Aligning the data with required features
        aligned_data = {feature: mapped_data.get(feature, 0.0) for feature in required_features}
        sensor_data_df = pd.DataFrame([aligned_data], columns=required_features)

        # Debugging: Log aligned data
        print("Aligned Data for Model Input:", aligned_data)

        # Predict RUL using the model
        rul_prediction = rul_model.predict(sensor_data_df)[0]  # Predict returns an array; get the first value

        # Add slight noise to the prediction to introduce variation
        fluctuation = random.uniform(-5, 5)  # Add noise in a range of -5 to 5 RUL
        rul_prediction_with_noise = max(rul_prediction + fluctuation, 0)  # Ensure RUL doesn't go below 0

        # Provide recommendations based on RUL
        recommendations = get_recommendations_based_on_rul(rul_prediction_with_noise)

        return {
            "Predicted RUL": round(rul_prediction_with_noise, 2),
            "Recommendations": recommendations,
            "RUL Explanation": generate_rul_explanation(rul_prediction_with_noise)
        }

    except Exception as e:
        raise RuntimeError(f"Error during RUL analysis: {e}")


# Function to generate explanation of the RUL value
def generate_rul_explanation(rul):
    if rul > 100:
        return "The system is operating well within its useful life, with no immediate need for maintenance."
    elif 50 <= rul <= 100:
        return "The system is showing signs of wear. It's advised to keep monitoring the system closely and plan maintenance soon."
    else:
        return "The system is approaching failure. Immediate action is required, such as inspection and repair."


# Function to provide recommendations based on the RUL value
def get_recommendations_based_on_rul(rul):
    if rul > 100:
        return ["No immediate action required."]
    elif 50 <= rul <= 100:
        return [
            "Monitor the system closely.",
            "Schedule maintenance in the near future.",
            "Inspect components for early signs of wear."
        ]
    else:
        return [
            "Immediate inspection required.",
            "Repair or replace faulty components.",
            "Perform a complete system diagnostic."
        ]


# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Flask route for analyzing RUL
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Collect sensor data from the form
        sensor_data = {
            "vibration": float(request.form.get('vibration', 0.0)),
            "temperature": float(request.form.get('temperature', 0.0)),
            "pressure": float(request.form.get('pressure', 0.0)),
            "humidity": float(request.form.get('humidity', 0.0)),
            "rpm": float(request.form.get('rpm', 0.0))
        }

        # Debugging: Print collected sensor data
        print("Collected Sensor Data:", sensor_data)

        # Perform RUL prediction and analysis
        result = predict_rul_and_analyze(sensor_data, rul_model)

        # Save results to JSON file
        with open(JSON_FILE, 'r+') as f:
            history = json.load(f)
            history.append({
                "sensor_data": sensor_data,
                "predicted_rul": result["Predicted RUL"],
                "recommendations": result["Recommendations"]
            })
            f.seek(0)
            json.dump(history, f, indent=4)

        # Render result on a web page
        return render_template('result.html', result=result, sensor_data=sensor_data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
