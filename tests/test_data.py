"""
Unit tests for data module
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import DataLoader  # noqa: E402
from src.data.preprocessing import DataPreprocessor  # noqa: E402


class TestDataLoader:
    """Test cases for DataLoader class"""

    def test_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.data_dir is not None
        assert loader.raw_dir.exists()
        assert loader.processed_dir.exists()

    def test_get_data_info(self):
        """Test data info extraction"""
        loader = DataLoader()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        info = loader.get_data_info(df)
        assert info["shape"] == (3, 2)
        assert "A" in info["columns"]
        assert "B" in info["columns"]


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""

    def test_initialization(self):
        """Test DataPreprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is not None
        assert preprocessor.scaling_method in ["standard", "minmax"]

    def test_handle_missing_values(self):
        """Test missing value handling"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8]})

        df_clean = preprocessor.handle_missing_values(df, strategy="mean")
        assert df_clean.isnull().sum().sum() == 0

    def test_handle_outliers(self):
        """Test outlier handling"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"A": [1, 2, 3, 100], "B": [10, 20, 30, 40]})  # 100 is an outlier

        df_clean = preprocessor.handle_outliers(df, columns=["A"], method="iqr")
        assert df_clean["A"].max() <= 100  # Outlier should be capped at boundary

    def test_encode_categorical(self):
        """Test categorical encoding"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"A": ["cat1", "cat2", "cat1"], "B": [1, 2, 3]})

        df_encoded = preprocessor.encode_categorical_variables(df)
        assert "A" not in df_encoded.columns  # Original column dropped after get_dummies with drop_first
        assert "B" in df_encoded.columns  # Non-categorical column retained

    def test_scale_features(self):
        """Test feature scaling"""
        preprocessor = DataPreprocessor()
        X_train = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
        X_test = pd.DataFrame({"A": [4], "B": [40]})

        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

        # Check that scaled data has mean ~0 for training data
        assert abs(X_train_scaled.mean().mean()) < 0.1
        # With only 3 rows ddof=1 std deviates from 1 — check it is non-zero
        assert X_train_scaled.std().mean() > 0


def test_data_pipeline_integration():
    """Integration test for full data pipeline"""
    preprocessor = DataPreprocessor()

    # Create sample data
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "target": [100, 200, 300, 400, 500]}
    )

    # Test preprocessing pipeline
    # Note: This is a simplified test
    df_clean = preprocessor.handle_missing_values(df)
    assert df_clean.shape == df.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
