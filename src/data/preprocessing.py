"""
Data Preprocessing Module
Handles data cleaning, transformation, and preparation
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class to handle data preprocessing operations"""
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize DataPreprocessor
        
        Args:
            scaling_method: 'standard' or 'minmax'
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.feature_names = None
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', 'mode', 'forward_fill', 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()
        
        if missing_before == 0:
            logger.info("No missing values found")
            return df_clean
        
        logger.info(f"Found {missing_before} missing values")
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'mode':
            df_clean = df_clean.fillna(df_clean.mode().iloc[0])
        elif strategy == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'drop':
            df_clean = df_clean.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in specified columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = (z_scores > threshold).sum()
                
                # Cap outliers
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
            if outliers > 0:
                logger.info(f"Handled {outliers} outliers in column '{col}'")
        
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                     columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not columns:
            logger.info("No categorical columns to encode")
            return df_encoded
        
        logger.info(f"Encoding categorical columns: {columns}")
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        
        logger.info(f"Shape after encoding: {df_encoded.shape}")
        return df_encoded
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple:
        """
        Scale features using the specified scaling method
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of (scaled_train, scaled_test) or just scaled_train
        """
        self.feature_names = X_train.columns.tolist()
        
        # Fit on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        
        logger.info(f"Scaled training features using {self.scaling_method} scaling")
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
            logger.info(f"Scaled test features")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def create_time_series_splits(self, df: pd.DataFrame, date_column: str,
                                  train_size: float = 0.7, val_size: float = 0.15) -> Tuple:
        """
        Create time-based train/validation/test splits
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        
        n = len(df_sorted)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"Train size: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        logger.info(f"Validation size: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        logger.info(f"Test size: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str = None,
                           date_column: str = None) -> dict:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            date_column: Name of the date column (for time-series split)
            
        Returns:
            Dictionary containing processed datasets
        """
        logger.info("Starting preprocessing pipeline")
        
        # 1. Handle missing values
        df_clean = self.handle_missing_values(df, strategy='forward_fill')
        
        # 2. Handle outliers
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        df_clean = self.handle_outliers(df_clean, columns=numeric_cols)
        
        # 3. Encode categorical variables
        df_encoded = self.encode_categorical_variables(df_clean)
        
        # 4. Split features and target
        if target_column:
            X = df_encoded.drop(columns=[target_column])
            y = df_encoded[target_column]
        else:
            X = df_encoded
            y = None
        
        # Remove date column from features if present
        if date_column and date_column in X.columns:
            X = X.drop(columns=[date_column])
        
        # 5. Create train/test splits
        if date_column and date_column in df_encoded.columns:
            # Time-based split
            train_df, val_df, test_df = self.create_time_series_splits(
                df_encoded, date_column
            )
            
            X_train = train_df.drop(columns=[target_column, date_column] if target_column else [date_column])
            y_train = train_df[target_column] if target_column else None
            
            X_val = val_df.drop(columns=[target_column, date_column] if target_column else [date_column])
            y_val = val_df[target_column] if target_column else None
            
            X_test = test_df.drop(columns=[target_column, date_column] if target_column else [date_column])
            y_test = test_df[target_column] if target_column else None
            
        else:
            # Random split
            if target_column:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42
                )
            else:
                X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
                X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
                y_train = y_val = y_test = None
        
        # 6. Scale features
        X_train_scaled, X_val_scaled = self.scale_features(X_train, X_val)
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_names,
            index=X_test.index
        )
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }


def main():
    """Main function to demonstrate preprocessing"""
    from load_data import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_stock_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method='standard')
    
    # Run preprocessing (adjust column names based on your data)
    # result = preprocessor.preprocess_pipeline(df, target_column='Close', date_column='Date')
    
    print("\n=== Preprocessing Demo ===")
    print(f"Original shape: {df.shape}")
    
    # Handle missing values
    df_clean = preprocessor.handle_missing_values(df)
    print(f"Shape after handling missing values: {df_clean.shape}")


if __name__ == "__main__":
    main()
