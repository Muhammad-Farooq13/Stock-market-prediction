"""
Configuration Management Module
Handles application configuration
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
import json
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for the project"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    
    # Data settings
    DATASET_NAME: str = "Daily_Global_Stock_Market_Indicators.csv"
    TARGET_COLUMN: str = "Close"
    DATE_COLUMN: str = "Date"
    
    # Model settings
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.1
    RANDOM_STATE: int = 42
    
    # Training settings
    N_ESTIMATORS: int = 100
    MAX_DEPTH: int = 10
    LEARNING_RATE: float = 0.1
    
    # Feature engineering settings
    MA_WINDOWS: list = None
    EMA_SPANS: list = None
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    # Flask settings
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = False
    
    # MLOps settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "stock_market_prediction"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Initialize default values"""
        if self.MA_WINDOWS is None:
            self.MA_WINDOWS = [5, 10, 20, 50, 200]
        if self.EMA_SPANS is None:
            self.EMA_SPANS = [12, 26]
        
        # Create directories if they don't exist
        for dir_path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
                        self.MODELS_DIR, self.LOGS_DIR, self.NOTEBOOKS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'project_root': str(self.PROJECT_ROOT),
            'data_dir': str(self.DATA_DIR),
            'models_dir': str(self.MODELS_DIR),
            'dataset_name': self.DATASET_NAME,
            'target_column': self.TARGET_COLUMN,
            'date_column': self.DATE_COLUMN,
            'test_size': self.TEST_SIZE,
            'validation_size': self.VALIDATION_SIZE,
            'random_state': self.RANDOM_STATE,
            'flask_host': self.FLASK_HOST,
            'flask_port': self.FLASK_PORT
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """Load configuration from file"""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
        
        return cls(**config_dict)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get global config instance"""
    return config


def main():
    """Main function to demonstrate configuration"""
    config = get_config()
    
    print("\n=== Project Configuration ===")
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Models Directory: {config.MODELS_DIR}")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Target Column: {config.TARGET_COLUMN}")
    print(f"Flask Port: {config.FLASK_PORT}")


if __name__ == "__main__":
    main()
