"""
Unit tests for models module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.models.predict import Predictor


class TestModelTrainer:
    """Test cases for ModelTrainer class"""
    
    def test_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer()
        assert trainer.model_dir.exists()
        assert trainer.trained_models == {}
    
    def test_get_model(self):
        """Test model retrieval"""
        trainer = ModelTrainer()
        
        # Test valid models
        model = trainer.get_model('linear_regression')
        assert model is not None
        
        model = trainer.get_model('random_forest', n_estimators=50)
        assert model is not None
        
        # Test invalid model
        with pytest.raises(ValueError):
            trainer.get_model('invalid_model')
    
    def test_train_model(self):
        """Test model training"""
        trainer = ModelTrainer()
        
        # Create sample data
        X_train = pd.DataFrame(np.random.randn(100, 5))
        y_train = pd.Series(np.random.randn(100))
        
        # Train model
        model, metrics = trainer.train_model(
            X_train, y_train, model_name='linear_regression'
        )
        
        assert model is not None
        assert 'train_rmse' in metrics
        assert 'train_mae' in metrics
        assert 'train_r2' in metrics
    
    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        trainer = ModelTrainer()
        
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
        y_train = pd.Series(np.random.randn(100))
        
        model, _ = trainer.train_model(X_train, y_train, model_name='random_forest')
        
        importance = trainer.get_feature_importance(model, X_train.columns.tolist())
        
        assert importance is not None
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestModelEvaluator:
    """Test cases for ModelEvaluator class"""
    
    def test_initialization(self):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator()
        assert evaluator.results_dir.exists()
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'mape' in metrics
        assert metrics['rmse'] > 0
        assert metrics['r2_score'] <= 1
    
    def test_calculate_residuals(self):
        """Test residual calculation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
        
        residuals = evaluator.calculate_residuals(y_true, y_pred)
        
        assert 'residuals' in residuals
        assert 'abs_residuals' in residuals
        assert len(residuals['residuals']) == len(y_true)


class TestPredictor:
    """Test cases for Predictor class"""
    
    def test_initialization(self):
        """Test Predictor initialization"""
        predictor = Predictor()
        assert predictor.model is None
        assert predictor.scaler is None
    
    def test_predict_single(self):
        """Test single prediction"""
        from sklearn.linear_model import LinearRegression
        
        # Create and train a simple model
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        model = LinearRegression()
        model.fit(X, y)
        
        # Create predictor and load model
        predictor = Predictor()
        predictor.model = model
        
        # Make prediction
        features = {'0': 0.5, '1': 0.3, '2': 0.7}
        prediction = predictor.predict_single(features)
        
        assert isinstance(prediction, (int, float, np.number))


def test_model_pipeline_integration():
    """Integration test for model pipeline"""
    # Create sample data
    X_train = pd.DataFrame(np.random.randn(100, 5))
    y_train = pd.Series(np.random.randn(100))
    X_test = pd.DataFrame(np.random.randn(20, 5))
    y_test = pd.Series(np.random.randn(20))
    
    # Train model
    trainer = ModelTrainer()
    model, metrics = trainer.train_model(X_train, y_train, model_name='linear_regression')
    
    # Evaluate model
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(model, X_test, y_test)
    
    assert result is not None
    assert 'metrics' in result
    assert 'predictions' in result


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
