"""
Flask Application for Stock Market Prediction API
Serves machine learning model predictions via REST API
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
MODEL = None
SCALER = None
FEATURE_NAMES = None
MODEL_METADATA = None

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


def load_model_and_scaler():
    """Load the trained model and scaler"""
    global MODEL, SCALER, FEATURE_NAMES, MODEL_METADATA
    
    try:
        # Find the latest model file
        model_files = list(MODELS_DIR.glob("*.pkl"))
        
        if not model_files:
            logger.warning("No model files found. Model endpoints will not work.")
            return False
        
        # Load the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        MODEL = joblib.load(latest_model)
        logger.info(f"Loaded model from {latest_model}")
        
        # Try to load scaler if exists
        scaler_files = list(MODELS_DIR.glob("*scaler*.pkl"))
        if scaler_files:
            latest_scaler = max(scaler_files, key=lambda x: x.stat().st_mtime)
            SCALER = joblib.load(latest_scaler)
            logger.info(f"Loaded scaler from {latest_scaler}")
        
        # Try to load metadata if exists
        metadata_files = list(MODELS_DIR.glob("*metadata*.pkl"))
        if metadata_files:
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            MODEL_METADATA = joblib.load(latest_metadata)
            FEATURE_NAMES = MODEL_METADATA.get('feature_names')
            logger.info(f"Loaded metadata from {latest_metadata}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


# HTML template for home page
HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock Market Prediction API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .endpoint {
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .method {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            margin-right: 10px;
        }
        .get { background-color: #61affe; color: white; }
        .post { background-color: #49cc90; color: white; }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status.ok { background-color: #d4edda; color: #155724; }
        .status.error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Stock Market Prediction API</h1>
        <p>RESTful API for stock market predictions using machine learning</p>
    </div>
    
    <div class="status {{ status_class }}">
        <strong>Model Status:</strong> {{ model_status }}
    </div>
    
    <h2>📚 Available Endpoints</h2>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /</h3>
        <p>Home page with API documentation</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /health</h3>
        <p>Health check endpoint</p>
        <p><strong>Response:</strong></p>
        <pre>{
    "status": "healthy",
    "timestamp": "2024-01-30T10:00:00",
    "model_loaded": true
}</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /predict</h3>
        <p>Make a prediction using the trained model</p>
        <p><strong>Request Body:</strong></p>
        <pre>{
    "features": {
        "feature1": value1,
        "feature2": value2,
        ...
    }
}</pre>
        <p><strong>Response:</strong></p>
        <pre>{
    "prediction": 123.45,
    "timestamp": "2024-01-30T10:00:00",
    "model": "xgboost"
}</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /predict_batch</h3>
        <p>Make predictions for multiple samples</p>
        <p><strong>Request Body:</strong></p>
        <pre>{
    "samples": [
        {"feature1": value1, "feature2": value2, ...},
        {"feature1": value3, "feature2": value4, ...}
    ]
}</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /model/info</h3>
        <p>Get information about the loaded model</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /metrics</h3>
        <p>Get model performance metrics (if available)</p>
    </div>
    
    <h2>📖 Usage Example</h2>
    <pre>
import requests

# Make a prediction
response = requests.post('http://localhost:5000/predict', json={
    'features': {
        'Close_MA_20': 150.5,
        'RSI_14': 45.2,
        'MACD': 2.3,
        ...
    }
})

result = response.json()
print(f"Prediction: {result['prediction']}")
    </pre>
    
    <footer style="margin-top: 50px; text-align: center; color: #666;">
        <p>Stock Market Prediction API v1.0 | Built with Flask</p>
    </footer>
</body>
</html>
"""


@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    model_loaded = MODEL is not None
    status_class = 'ok' if model_loaded else 'error'
    model_status = 'Model loaded and ready' if model_loaded else 'No model loaded'
    
    return render_template_string(
        HOME_PAGE,
        model_status=model_status,
        status_class=status_class
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': MODEL is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Make a single prediction"""
    try:
        if MODEL is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure a trained model is available'
            }), 503
        
        # Get request data
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = data['features']
        
        # Convert to DataFrame
        if isinstance(features, dict):
            X = pd.DataFrame([features])
        elif isinstance(features, list):
            X = pd.DataFrame([features], columns=FEATURE_NAMES if FEATURE_NAMES else None)
        else:
            return jsonify({'error': 'Invalid features format'}), 400
        
        # Scale features if scaler is available
        if SCALER is not None:
            X_scaled = SCALER.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make prediction
        prediction = MODEL.predict(X)[0]
        
        # Prepare response
        response = {
            'prediction': float(prediction),
            'timestamp': datetime.now().isoformat(),
            'model': MODEL_METADATA.get('model_name', 'unknown') if MODEL_METADATA else 'unknown'
        }
        
        logger.info(f"Prediction made: {prediction}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make predictions for multiple samples"""
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get request data
        data = request.get_json()
        
        if 'samples' not in data:
            return jsonify({'error': 'Missing samples in request'}), 400
        
        samples = data['samples']
        
        # Convert to DataFrame
        X = pd.DataFrame(samples)
        
        # Scale features if scaler is available
        if SCALER is not None:
            X_scaled = SCALER.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make predictions
        predictions = MODEL.predict(X)
        
        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction made: {len(predictions)} samples")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    info = {
        'model_loaded': True,
        'model_type': type(MODEL).__name__,
        'feature_count': len(FEATURE_NAMES) if FEATURE_NAMES else 'unknown',
        'feature_names': FEATURE_NAMES[:20] if FEATURE_NAMES else None  # First 20 features
    }
    
    if MODEL_METADATA:
        info.update(MODEL_METADATA)
    
    return jsonify(info)


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    if MODEL_METADATA is None:
        return jsonify({'error': 'No metadata available'}), 404
    
    metrics = {
        'train_metrics': {
            'rmse': MODEL_METADATA.get('train_rmse'),
            'mae': MODEL_METADATA.get('train_mae'),
            'r2': MODEL_METADATA.get('train_r2')
        },
        'validation_metrics': {
            'rmse': MODEL_METADATA.get('val_rmse'),
            'mae': MODEL_METADATA.get('val_mae'),
            'r2': MODEL_METADATA.get('val_r2')
        }
    }
    
    return jsonify(metrics)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Flask application...")
    load_model_and_scaler()
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
