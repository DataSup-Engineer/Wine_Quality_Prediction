"""
Flask web application for Wine Quality Classifier.

This application provides a web interface for users to input wine characteristics
and receive quality predictions from the trained model.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import WineQualityModel

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Global variables for model and scaler
model = None
scaler = None

# Feature names in the correct order (11 physicochemical properties only)
FEATURE_NAMES = [
    'fixed_acidity',
    'volatile_acidity',
    'citric_acid',
    'residual_sugar',
    'chlorides',
    'free_sulfur_dioxide',
    'total_sulfur_dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# Feature display names for the form
FEATURE_LABELS = {
    'fixed_acidity': 'Fixed Acidity',
    'volatile_acidity': 'Volatile Acidity',
    'citric_acid': 'Citric Acid',
    'residual_sugar': 'Residual Sugar',
    'chlorides': 'Chlorides',
    'free_sulfur_dioxide': 'Free Sulfur Dioxide',
    'total_sulfur_dioxide': 'Total Sulfur Dioxide',
    'density': 'Density',
    'pH': 'pH',
    'sulphates': 'Sulphates',
    'alcohol': 'Alcohol'
}


def load_model_at_startup():
    """Load the trained model and scaler at application startup."""
    global model, scaler
    
    model_path = 'models/trained_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Model files not found. Please train the model first by running: python train.py"
        )
    
    model = WineQualityModel()
    scaler = model.load_model(model_path, scaler_path)
    print("✓ Model and scaler loaded successfully")


def validate_input(form_data: dict) -> tuple:
    """
    Validate that all required features are present and numeric.
    
    Args:
        form_data: Dictionary containing form input data
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: Boolean indicating if input is valid
        - error_message: String describing the error, or empty string if valid
    """
    # Check all required features are present
    missing_features = []
    for feature in FEATURE_NAMES:
        if feature not in form_data or form_data[feature] == '':
            missing_features.append(FEATURE_LABELS[feature])
    
    if missing_features:
        return False, f"Missing required fields: {', '.join(missing_features)}"
    
    # Check all values are numeric
    invalid_features = []
    for feature in FEATURE_NAMES:
        try:
            float(form_data[feature])
        except (ValueError, TypeError):
            invalid_features.append(FEATURE_LABELS[feature])
    
    if invalid_features:
        return False, f"Invalid numeric values for: {', '.join(invalid_features)}"
    
    return True, ""


def prepare_features(form_data: dict) -> np.ndarray:
    """
    Extract features from form data and convert to numpy array.
    
    Args:
        form_data: Dictionary containing form input data
    
    Returns:
        Numpy array of shape (1, 11) with features in correct order
    """
    features = []
    for feature in FEATURE_NAMES:
        features.append(float(form_data[feature]))
    
    # Convert to numpy array with shape (1, n_features)
    feature_array = np.array(features).reshape(1, -1)
    return feature_array


@app.route('/')
def index():
    """Render the home page with input form."""
    return render_template('index.html', 
                         feature_names=FEATURE_NAMES,
                         feature_labels=FEATURE_LABELS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    
    Accepts form data with wine characteristics and returns predicted quality score.
    """
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate input
        is_valid, error_message = validate_input(form_data)
        if not is_valid:
            return render_template('index.html',
                                 feature_names=FEATURE_NAMES,
                                 feature_labels=FEATURE_LABELS,
                                 error=error_message,
                                 form_data=form_data), 400
        
        # Prepare features
        features = prepare_features(form_data)
        
        # Scale features using the loaded scaler
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get probability for predicted class
        predicted_class_idx = np.where(model.classes_ == prediction)[0][0]
        confidence = probabilities[predicted_class_idx] * 100
        
        # Render result page
        return render_template('result.html',
                             prediction=int(prediction),
                             confidence=f"{confidence:.1f}",
                             features=form_data,
                             feature_labels=FEATURE_LABELS)
    
    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        return render_template('index.html',
                             feature_names=FEATURE_NAMES,
                             feature_labels=FEATURE_LABELS,
                             error=error_message,
                             form_data=request.form.to_dict()), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions (JSON format).
    
    Accepts JSON with wine characteristics and returns JSON with prediction.
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Prepare features
        features = prepare_features(data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probabilities': {
                int(cls): float(prob) 
                for cls, prob in zip(model.classes_, probabilities)
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200


if __name__ == '__main__':
    # Load model at startup
    try:
        load_model_at_startup()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Run the Flask app
    print("\n" + "=" * 60)
    print("Wine Quality Classifier - Web Application")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Access the application at: http://127.0.0.1:5001")
    print("Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
