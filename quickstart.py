"""
Quick Start Script
Demonstrates basic usage of the project
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data.load_data import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.visualization.visualize import Visualizer
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("quickstart", log_dir="logs")


def main():
    """Quick start demonstration"""
    
    logger.info("=" * 80)
    logger.info("Stock Market Prediction - Quick Start")
    logger.info("=" * 80)
    
    # 1. Load Data
    logger.info("\n1. Loading data...")
    loader = DataLoader()
    df = loader.load_stock_data()
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # 2. Display basic information
    logger.info("\n2. Dataset Overview:")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"\nFirst few rows:")
    print(df.head())
    
    # 3. Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info("\n3. Missing values found:")
        logger.info(missing[missing > 0])
    else:
        logger.info("\n3. No missing values found!")
    
    # 4. Basic statistics
    logger.info("\n4. Basic Statistics:")
    print(df.describe())
    
    logger.info("\n" + "=" * 80)
    logger.info("Quick Start Complete!")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("1. Explore the data in notebooks/01_exploratory_analysis.ipynb")
    logger.info("2. Configure your target and date columns in src/utils/config.py")
    logger.info("3. Run the full MLOps pipeline: python mlops_pipeline.py")
    logger.info("4. Start the Flask API: python flask_app.py")


if __name__ == "__main__":
    main()
