from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load or train a simple model (for demonstration)
try:
    # Try to load the model if it exists
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Train a dummy model if no model file exists
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=7, n_classes=3, n_informative=5, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    with open('crop_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Dummy crop mapping (replace with actual crop names from your dataset)
crop_mapping = {0: 'Rice', 1: 'Maize', 2: 'Wheat'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract input features
        features = [
            data['nitrogen'],
            data['phosphorus'],
            data['potassium'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        # Convert to numpy array
        features = np.array([features])
        # Make prediction
        prediction = model.predict(features)[0]
        crop = crop_mapping.get(prediction, 'Unknown')
        return jsonify({'crop': crop})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)