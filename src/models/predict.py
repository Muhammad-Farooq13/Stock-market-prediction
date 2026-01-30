"""
Prediction Module
Makes predictions using trained models
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Class to handle model predictions"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize Predictor
        
        Args:
            model_path: Path to the trained model
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Load trained model and scaler
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the scaler (optional)
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load_model() first.")
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict_with_intervals(self, X: pd.DataFrame, 
                              confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals
        
        Args:
            X: Input features
            confidence: Confidence level
            
        Returns:
            Dictionary with predictions and intervals
        """
        predictions = self.predict(X)
        
        # Simple approach: use standard deviation of training residuals
        # In practice, you'd load this from saved metadata
        std_error = 0.1 * np.std(predictions)  # Placeholder
        
        from scipy import stats
        alpha = 1 - confidence
        critical_value = stats.norm.ppf(1 - alpha/2)
        margin = critical_value * std_error
        
        return {
            'predictions': predictions,
            'lower_bound': predictions - margin,
            'upper_bound': predictions + margin,
            'confidence': confidence
        }
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Make prediction for a single sample
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Single prediction
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.predict(X)[0]
        
        return prediction
    
    def batch_predict(self, X: pd.DataFrame, batch_size: int = 1000) -> np.ndarray:
        """
        Make predictions in batches
        
        Args:
            X: Input features
            batch_size: Size of each batch
            
        Returns:
            All predictions
        """
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i+batch_size]
            batch_pred = self.predict(batch)
            predictions.extend(batch_pred)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(X)-1)//batch_size + 1}")
        
        return np.array(predictions)


def main():
    """Main function to demonstrate predictions"""
    print("\n=== Prediction Demo ===")
    print("Load a trained model and make predictions")
    
    # Example usage:
    # predictor = Predictor(model_path="models/xgboost_model.pkl")
    # predictions = predictor.predict(X_test)


if __name__ == "__main__":
    main()
