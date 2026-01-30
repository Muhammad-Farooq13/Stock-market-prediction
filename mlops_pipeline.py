"""
MLOps Pipeline
Implements continuous integration and deployment for ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import joblib
from typing import Dict, Any
import mlflow
import mlflow.sklearn

from src.data.load_data import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.utils.config import get_config
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("mlops_pipeline", log_dir="logs", level="INFO")


class MLOpsPipeline:
    """Class to handle MLOps pipeline operations"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize MLOps Pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
    
    def run_data_pipeline(self) -> Dict[str, Any]:
        """
        Run the data preprocessing and feature engineering pipeline
        
        Returns:
            Dictionary containing processed datasets
        """
        logger.info("=" * 80)
        logger.info("Starting Data Pipeline")
        logger.info("=" * 80)
        
        # Load data
        logger.info("Step 1: Loading data...")
        df = self.data_loader.load_stock_data()
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Feature engineering
        logger.info("Step 2: Feature engineering...")
        # Note: Adjust column names based on your actual dataset
        # df_features = self.feature_engineer.create_all_features(
        #     df, price_column='Close', date_column='Date'
        # )
        df_features = df  # Placeholder
        
        # Preprocessing
        logger.info("Step 3: Preprocessing...")
        # Note: Adjust column names based on your actual dataset
        # processed_data = self.preprocessor.preprocess_pipeline(
        #     df_features, target_column='Close', date_column='Date'
        # )
        processed_data = {'X_train': None}  # Placeholder
        
        logger.info("Data pipeline completed successfully")
        return processed_data
    
    def run_training_pipeline(self, data: Dict[str, Any], 
                             model_names: list = None) -> Dict[str, Any]:
        """
        Run the model training pipeline
        
        Args:
            data: Processed data dictionary
            model_names: List of models to train
            
        Returns:
            Dictionary containing trained models and metrics
        """
        logger.info("=" * 80)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 80)
        
        if model_names is None:
            model_names = ['linear_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        
        results = {}
        
        for model_name in model_names:
            logger.info(f"\nTraining {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params({
                    'model_name': model_name,
                    'n_features': X_train.shape[1],
                    'n_samples': X_train.shape[0]
                })
                
                # Train model
                model, metrics = self.model_trainer.train_model(
                    X_train, y_train, X_val, y_val, model_name
                )
                
                # Log metrics
                mlflow.log_metrics({
                    'train_rmse': metrics['train_rmse'],
                    'train_mae': metrics['train_mae'],
                    'train_r2': metrics['train_r2'],
                    'val_rmse': metrics.get('val_rmse', 0),
                    'val_mae': metrics.get('val_mae', 0),
                    'val_r2': metrics.get('val_r2', 0)
                })
                
                # Log model
                mlflow.sklearn.log_model(model, model_name)
                
                results[model_name] = {'model': model, 'metrics': metrics}
        
        logger.info("Training pipeline completed successfully")
        return results
    
    def run_evaluation_pipeline(self, models: Dict[str, Any], 
                                data: Dict[str, Any]) -> pd.DataFrame:
        """
        Run the model evaluation pipeline
        
        Args:
            models: Dictionary of trained models
            data: Test data dictionary
            
        Returns:
            DataFrame with model comparison
        """
        logger.info("=" * 80)
        logger.info("Starting Evaluation Pipeline")
        logger.info("=" * 80)
        
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Extract just the models from results
        model_dict = {name: result['model'] for name, result in models.items()}
        
        # Evaluate models
        comparison = self.evaluator.evaluate_multiple_models(
            model_dict, X_test, y_test
        )
        
        logger.info("Evaluation pipeline completed successfully")
        return comparison
    
    def select_best_model(self, comparison: pd.DataFrame, 
                         metric: str = 'rmse') -> str:
        """
        Select the best model based on evaluation metric
        
        Args:
            comparison: Model comparison DataFrame
            metric: Metric to use for selection
            
        Returns:
            Name of the best model
        """
        if metric in ['rmse', 'mae', 'mse', 'mape']:
            best_model = comparison.loc[comparison[metric].idxmin(), 'model']
        else:  # r2_score, explained_variance
            best_model = comparison.loc[comparison[metric].idxmax(), 'model']
        
        logger.info(f"Best model selected: {best_model} (based on {metric})")
        return best_model
    
    def deploy_model(self, model: Any, model_name: str, 
                    scaler: Any = None, metadata: Dict = None):
        """
        Deploy model by saving it to the models directory
        
        Args:
            model: Trained model to deploy
            model_name: Name of the model
            scaler: Scaler used for preprocessing
            metadata: Model metadata
        """
        logger.info(f"Deploying model: {model_name}")
        
        # Save model
        self.model_trainer.save_model(model, model_name, metadata)
        
        # Save scaler if provided
        if scaler is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scaler_path = self.config.MODELS_DIR / f"scaler_{timestamp}.pkl"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        logger.info("Model deployment completed")
    
    def run_full_pipeline(self, auto_deploy: bool = True):
        """
        Run the complete MLOps pipeline
        
        Args:
            auto_deploy: Automatically deploy the best model
        """
        logger.info("=" * 80)
        logger.info("Starting Full MLOps Pipeline")
        logger.info("=" * 80)
        
        try:
            # 1. Data pipeline
            data = self.run_data_pipeline()
            
            # Check if data pipeline succeeded
            if data['X_train'] is None:
                logger.error("Data pipeline failed. Please check the data and configuration.")
                return
            
            # 2. Training pipeline
            results = self.run_training_pipeline(data)
            
            # 3. Evaluation pipeline
            comparison = self.run_evaluation_pipeline(results, data)
            
            # 4. Select and deploy best model
            if auto_deploy:
                best_model_name = self.select_best_model(comparison)
                best_model = results[best_model_name]['model']
                best_metrics = results[best_model_name]['metrics']
                
                self.deploy_model(
                    best_model,
                    best_model_name,
                    scaler=data.get('scaler'),
                    metadata=best_metrics
                )
            
            logger.info("=" * 80)
            logger.info("Full MLOps Pipeline Completed Successfully")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def monitor_model_performance(self, model_name: str, 
                                  new_data: pd.DataFrame) -> Dict[str, float]:
        """
        Monitor deployed model performance on new data
        
        Args:
            model_name: Name of the deployed model
            new_data: New data for monitoring
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Monitoring model: {model_name}")
        
        # Load deployed model
        model_path = list(self.config.MODELS_DIR.glob(f"{model_name}*.pkl"))[0]
        model = joblib.load(model_path)
        
        # Make predictions and calculate metrics
        # Implementation depends on your specific needs
        
        logger.info("Model monitoring completed")
        return {}
    
    def retrain_model(self, model_name: str, trigger: str = "manual"):
        """
        Retrain a model (triggered manually or automatically)
        
        Args:
            model_name: Name of the model to retrain
            trigger: Trigger type (manual, scheduled, performance_degradation)
        """
        logger.info(f"Retraining model: {model_name} (trigger: {trigger})")
        
        # Run the full pipeline
        self.run_full_pipeline(auto_deploy=True)
        
        logger.info("Model retraining completed")


def main():
    """Main function to run the MLOps pipeline"""
    logger.info("Initializing MLOps Pipeline")
    
    # Create pipeline instance
    pipeline = MLOpsPipeline()
    
    # Run full pipeline
    # Uncomment when you have proper data configuration
    # pipeline.run_full_pipeline(auto_deploy=True)
    
    logger.info("MLOps Pipeline execution completed")


if __name__ == "__main__":
    main()
