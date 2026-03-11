"""
Feature Engineering Module
Creates technical indicators and features for stock market prediction
"""

import logging
from typing import List, Optional

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class to handle feature engineering operations"""

    def __init__(self):
        """Initialize FeatureEngineer"""
        self.created_features = []

    def create_moving_averages(
        self, df: pd.DataFrame, column: str, windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Create moving average features

        Args:
            df: Input DataFrame
            column: Column to calculate moving averages on
            windows: List of window sizes

        Returns:
            DataFrame with moving average features
        """
        df_features = df.copy()

        for window in windows:
            feature_name = f"{column}_MA_{window}"
            df_features[feature_name] = df_features[column].rolling(window=window).mean()
            self.created_features.append(feature_name)
            logger.info(f"Created feature: {feature_name}")

        return df_features

    def create_exponential_moving_averages(
        self, df: pd.DataFrame, column: str, spans: List[int] = [12, 26]
    ) -> pd.DataFrame:
        """
        Create exponential moving average features

        Args:
            df: Input DataFrame
            column: Column to calculate EMA on
            spans: List of span values

        Returns:
            DataFrame with EMA features
        """
        df_features = df.copy()

        for span in spans:
            feature_name = f"{column}_EMA_{span}"
            df_features[feature_name] = df_features[column].ewm(span=span, adjust=False).mean()
            self.created_features.append(feature_name)
            logger.info(f"Created feature: {feature_name}")

        return df_features

    def create_rsi(self, df: pd.DataFrame, column: str, period: int = 14) -> pd.DataFrame:
        """
        Create Relative Strength Index (RSI) feature

        Args:
            df: Input DataFrame
            column: Column to calculate RSI on
            period: RSI period

        Returns:
            DataFrame with RSI feature
        """
        df_features = df.copy()

        # Calculate price changes
        delta = df_features[column].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        feature_name = f"{column}_RSI_{period}"
        df_features[feature_name] = rsi
        self.created_features.append(feature_name)
        logger.info(f"Created feature: {feature_name}")

        return df_features

    def create_macd(
        self,
        df: pd.DataFrame,
        column: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        """
        Create MACD (Moving Average Convergence Divergence) features

        Args:
            df: Input DataFrame
            column: Column to calculate MACD on
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            DataFrame with MACD features
        """
        df_features = df.copy()

        # Calculate MACD line
        ema_fast = df_features[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df_features[column].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line

        # Add features
        macd_feature = f"{column}_MACD"
        signal_feature = f"{column}_MACD_Signal"
        histogram_feature = f"{column}_MACD_Histogram"

        df_features[macd_feature] = macd_line
        df_features[signal_feature] = signal_line
        df_features[histogram_feature] = macd_histogram

        self.created_features.extend([macd_feature, signal_feature, histogram_feature])
        logger.info(f"Created MACD features for {column}")

        return df_features

    def create_bollinger_bands(
        self, df: pd.DataFrame, column: str, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Create Bollinger Bands features

        Args:
            df: Input DataFrame
            column: Column to calculate Bollinger Bands on
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            DataFrame with Bollinger Bands features
        """
        df_features = df.copy()

        # Calculate middle band (SMA)
        middle_band = df_features[column].rolling(window=window).mean()

        # Calculate standard deviation
        std = df_features[column].rolling(window=window).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)

        # Calculate band width and %B
        band_width = (upper_band - lower_band) / middle_band
        percent_b = (df_features[column] - lower_band) / (upper_band - lower_band)

        # Add features
        upper_feature = f"{column}_BB_Upper"
        lower_feature = f"{column}_BB_Lower"
        width_feature = f"{column}_BB_Width"
        percentb_feature = f"{column}_BB_PercentB"

        df_features[upper_feature] = upper_band
        df_features[lower_feature] = lower_band
        df_features[width_feature] = band_width
        df_features[percentb_feature] = percent_b

        self.created_features.extend(
            [upper_feature, lower_feature, width_feature, percentb_feature]
        )
        logger.info(f"Created Bollinger Bands features for {column}")

        return df_features

    def create_price_changes(
        self, df: pd.DataFrame, column: str, periods: List[int] = [1, 5, 10, 30]
    ) -> pd.DataFrame:
        """
        Create price change and return features

        Args:
            df: Input DataFrame
            column: Column to calculate changes on
            periods: List of periods for changes

        Returns:
            DataFrame with price change features
        """
        df_features = df.copy()

        for period in periods:
            # Absolute change
            change_feature = f"{column}_Change_{period}d"
            df_features[change_feature] = df_features[column].diff(period)

            # Percentage change
            pct_change_feature = f"{column}_PctChange_{period}d"
            df_features[pct_change_feature] = df_features[column].pct_change(period)

            self.created_features.extend([change_feature, pct_change_feature])

        logger.info(f"Created price change features for {column}")
        return df_features

    def create_volatility_features(
        self, df: pd.DataFrame, column: str, windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create volatility features

        Args:
            df: Input DataFrame
            column: Column to calculate volatility on
            windows: List of rolling window sizes

        Returns:
            DataFrame with volatility features
        """
        df_features = df.copy()

        returns = df_features[column].pct_change()

        for window in windows:
            volatility_feature = f"{column}_Volatility_{window}d"
            df_features[volatility_feature] = returns.rolling(window=window).std()
            self.created_features.append(volatility_feature)

        logger.info(f"Created volatility features for {column}")
        return df_features

    def create_lag_features(
        self, df: pd.DataFrame, columns: List[str], lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lagged features

        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        df_features = df.copy()

        for col in columns:
            if col not in df_features.columns:
                continue

            for lag in lags:
                lag_feature = f"{col}_Lag_{lag}"
                df_features[lag_feature] = df_features[col].shift(lag)
                self.created_features.append(lag_feature)

        logger.info(f"Created lag features for {len(columns)} columns")
        return df_features

    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create time-based features from date column

        Args:
            df: Input DataFrame
            date_column: Name of the date column

        Returns:
            DataFrame with time features
        """
        df_features = df.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features[date_column]):
            df_features[date_column] = pd.to_datetime(df_features[date_column])

        # Extract time features
        df_features["Year"] = df_features[date_column].dt.year
        df_features["Month"] = df_features[date_column].dt.month
        df_features["Day"] = df_features[date_column].dt.day
        df_features["DayOfWeek"] = df_features[date_column].dt.dayofweek
        df_features["Quarter"] = df_features[date_column].dt.quarter
        df_features["WeekOfYear"] = df_features[date_column].dt.isocalendar().week
        df_features["IsMonthStart"] = df_features[date_column].dt.is_month_start.astype(int)
        df_features["IsMonthEnd"] = df_features[date_column].dt.is_month_end.astype(int)

        time_features = [
            "Year",
            "Month",
            "Day",
            "DayOfWeek",
            "Quarter",
            "WeekOfYear",
            "IsMonthStart",
            "IsMonthEnd",
        ]
        self.created_features.extend(time_features)

        logger.info("Created time-based features")
        return df_features

    def create_all_features(
        self, df: pd.DataFrame, price_column: str, date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create all technical indicators and features

        Args:
            df: Input DataFrame
            price_column: Column containing price data
            date_column: Column containing date data (optional)

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering")

        df_features = df.copy()

        # Technical indicators
        df_features = self.create_moving_averages(df_features, price_column)
        df_features = self.create_exponential_moving_averages(df_features, price_column)
        df_features = self.create_rsi(df_features, price_column)
        df_features = self.create_macd(df_features, price_column)
        df_features = self.create_bollinger_bands(df_features, price_column)

        # Price-based features
        df_features = self.create_price_changes(df_features, price_column)
        df_features = self.create_volatility_features(df_features, price_column)

        # Lag features
        df_features = self.create_lag_features(df_features, [price_column])

        # Time features
        if date_column:
            df_features = self.create_time_features(df_features, date_column)

        logger.info(f"Feature engineering complete. Created {len(self.created_features)} features")
        logger.info(f"Final shape: {df_features.shape}")

        return df_features


def main():
    """Main function to demonstrate feature engineering"""
    from src.data.load_data import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_stock_data()

    print("\n=== Feature Engineering Demo ===")
    print(f"Original shape: {df.shape}")

    # Note: Adjust column names based on your actual data.
    # Example: FeatureEngineer().create_all_features(df, price_column='Close', date_column='Date')


if __name__ == "__main__":
    main()
