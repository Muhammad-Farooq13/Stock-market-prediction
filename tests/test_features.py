"""
Unit tests for features module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        assert engineer.created_features == []
    
    def test_create_moving_averages(self):
        """Test moving average creation"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        df_features = engineer.create_moving_averages(df, 'Close', windows=[3, 5])
        
        assert 'Close_MA_3' in df_features.columns
        assert 'Close_MA_5' in df_features.columns
        assert not df_features['Close_MA_3'].iloc[:2].isna().all()  # First values are NaN
    
    def test_create_rsi(self):
        """Test RSI creation"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                     110, 112, 111, 113, 115, 114, 116]
        })
        
        df_features = engineer.create_rsi(df, 'Close', period=14)
        
        assert 'Close_RSI_14' in df_features.columns
        # RSI should be between 0 and 100
        rsi_values = df_features['Close_RSI_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_create_macd(self):
        """Test MACD creation"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Close': np.random.randn(50).cumsum() + 100
        })
        
        df_features = engineer.create_macd(df, 'Close')
        
        assert 'Close_MACD' in df_features.columns
        assert 'Close_MACD_Signal' in df_features.columns
        assert 'Close_MACD_Histogram' in df_features.columns
    
    def test_create_bollinger_bands(self):
        """Test Bollinger Bands creation"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Close': np.random.randn(50).cumsum() + 100
        })
        
        df_features = engineer.create_bollinger_bands(df, 'Close', window=20)
        
        assert 'Close_BB_Upper' in df_features.columns
        assert 'Close_BB_Lower' in df_features.columns
        assert 'Close_BB_Width' in df_features.columns
        assert 'Close_BB_PercentB' in df_features.columns
    
    def test_create_price_changes(self):
        """Test price change features"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105]
        })
        
        df_features = engineer.create_price_changes(df, 'Close', periods=[1])
        
        assert 'Close_Change_1d' in df_features.columns
        assert 'Close_PctChange_1d' in df_features.columns
        
        # Check calculation
        assert df_features['Close_Change_1d'].iloc[1] == 2  # 102 - 100
    
    def test_create_volatility_features(self):
        """Test volatility features"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Close': np.random.randn(30).cumsum() + 100
        })
        
        df_features = engineer.create_volatility_features(df, 'Close', windows=[5])
        
        assert 'Close_Volatility_5d' in df_features.columns
    
    def test_create_lag_features(self):
        """Test lag features"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        })
        
        df_features = engineer.create_lag_features(df, ['Close'], lags=[1, 2])
        
        assert 'Close_Lag_1' in df_features.columns
        assert 'Close_Lag_2' in df_features.columns
        assert df_features['Close_Lag_1'].iloc[1] == 100
    
    def test_create_time_features(self):
        """Test time features"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10)
        })
        
        df_features = engineer.create_time_features(df, 'Date')
        
        assert 'Year' in df_features.columns
        assert 'Month' in df_features.columns
        assert 'Day' in df_features.columns
        assert 'DayOfWeek' in df_features.columns


def test_feature_engineering_integration():
    """Integration test for feature engineering"""
    engineer = FeatureEngineer()
    
    # Create sample data
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Close': np.random.randn(100).cumsum() + 100
    })
    
    # Create all features
    df_features = engineer.create_all_features(df, 'Close', 'Date')
    
    # Check that features were created
    assert df_features.shape[1] > df.shape[1]
    assert len(engineer.created_features) > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
