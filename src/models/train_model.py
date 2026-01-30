"""
Model Training Module
Trains various machine learning models for stock market prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training operations"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize ModelTrainer
        
        Args:
            model_dir: Directory to save trained models
        """
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models = {}
        self.training_history = []
    
    def get_model(self, model_name: str, **kwargs) -> Any:
        """
        Get a model instance by name
        
        Args:
            model_name: Name of the model
            **kwargs: Model hyperparameters
            
        Returns:
            Model instance
        """
        models = {
            'linear_regression': LinearRegression(**kwargs),
            'ridge': Ridge(**kwargs),
            'lasso': Lasso(**kwargs),
            'random_forest': RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42)
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return models[model_name]
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame = None, y_val: pd.Series = None,
                   model_name: str = 'xgboost', **kwargs) -> Tuple[Any, Dict]:
        """
        Train a single model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            model_name: Name of the model to train
            **kwargs: Model hyperparameters
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Training {model_name} model...")
        
        # Get model instance
        model = self.get_model(model_name, **kwargs)
        
        # Train model
        start_time = datetime.now()
        
        if model_name in ['xgboost', 'lightgbm'] and X_val is not None:
            # Use early stopping for gradient boosting models
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'training_time': training_time,
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred)
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            metrics.update({
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_r2': r2_score(y_val, y_val_pred)
            })
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Train RMSE: {metrics['train_rmse']:.4f}, Train R²: {metrics['train_r2']:.4f}")
        
        if 'val_rmse' in metrics:
            logger.info(f"Val RMSE: {metrics['val_rmse']:.4f}, Val R²: {metrics['val_r2']:.4f}")
        
        # Store model and history
        self.trained_models[model_name] = model
        self.training_history.append(metrics)
        
        return model, metrics
    
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame = None, y_val: pd.Series = None,
                             model_names: list = None) -> Dict[str, Tuple[Any, Dict]]:
        """
        Train multiple models and compare performance
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            model_names: List of model names to train
            
        Returns:
            Dictionary of {model_name: (model, metrics)}
        """
        if model_names is None:
            model_names = ['linear_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        logger.info(f"Training {len(model_names)} models...")
        
        results = {}
        
        for model_name in model_names:
            try:
                model, metrics = self.train_model(
                    X_train, y_train, X_val, y_val, model_name
                )
                results[model_name] = (model, metrics)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Log comparison
        logger.info("\n=== Model Comparison ===")
        comparison_df = pd.DataFrame([metrics for _, metrics in results.values()])
        logger.info(f"\n{comparison_df.to_string()}")
        
        return results
    
    def save_model(self, model: Any, model_name: str, metadata: Dict = None):
        """
        Save trained model to disk
        
        Args:
            model: Trained model
            model_name: Name for the saved model
            metadata: Additional metadata to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.model_dir / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        if metadata:
            metadata_path = self.model_dir / f"{model_name}_{timestamp}_metadata.pkl"
            joblib.dump(metadata, metadata_path)
            logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def get_feature_importance(self, model: Any, feature_names: list,
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            logger.warning("Model does not have feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             model_name: str, param_grid: Dict,
                             cv: int = 5) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        
        # Get base model
        base_model = self.get_model(model_name)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_


def main():
    """Main function to demonstrate model training"""
    from src.data.load_data import DataLoader
    from src.data.preprocessing import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_stock_data()
    
    # Note: Adjust column names based on your actual data
    # preprocessor = DataPreprocessor()
    # data = preprocessor.preprocess_pipeline(df, target_column='Close', date_column='Date')
    
    # Train models
    # trainer = ModelTrainer()
    # results = trainer.train_multiple_models(
    #     data['X_train'], data['y_train'],
    #     data['X_val'], data['y_val']
    # )
    
    print("\n=== Model Training Demo ===")
    print("See code for implementation details")


if __name__ == "__main__":
    main()
