"""
Data Loading Module
Handles loading of stock market data from various sources
"""

import logging
from pathlib import Path

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class to handle data loading operations"""

    def __init__(self, data_dir: str = None):
        """
        Initialize DataLoader

        Args:
            data_dir: Directory containing data files
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str, from_processed: bool = False) -> pd.DataFrame:
        """
        Load CSV file from raw or processed directory

        Args:
            filename: Name of the CSV file
            from_processed: If True, load from processed directory

        Returns:
            DataFrame containing the data
        """
        try:
            if from_processed:
                filepath = self.processed_dir / filename
            else:
                filepath = self.raw_dir / filename

            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {len(df)} rows")
            return df

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_stock_data(
        self, filename: str = "Daily_Global_Stock_Market_Indicators.csv"
    ) -> pd.DataFrame:
        """
        Load stock market data with specific processing

        Args:
            filename: Name of the stock data file

        Returns:
            DataFrame containing stock market data
        """
        try:
            # Try loading from root directory first
            root_path = self.data_dir.parent / filename
            if root_path.exists():
                logger.info(f"Loading stock data from {root_path}")
                df = pd.read_csv(root_path)
            else:
                df = self.load_csv(filename)

            # Convert date columns if present
            date_columns = ["Date", "date", "DATE", "Timestamp", "timestamp"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    logger.info(f"Converted column '{col}' to datetime")
                    break

            logger.info(f"Loaded stock data with shape {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            logger.error(f"Error loading stock data: {str(e)}")
            raise

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to processed directory

        Args:
            df: DataFrame to save
            filename: Name for the output file
        """
        try:
            filepath = self.processed_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Saved processed data to {filepath}")

        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing dataset information
        """
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "duplicate_rows": df.duplicated().sum(),
        }

        return info


def main():
    """Main function to demonstrate data loading"""
    loader = DataLoader()

    # Load stock market data
    df = loader.load_stock_data()

    # Get data info
    info = loader.get_data_info(df)

    print("\n=== Data Information ===")
    print(f"Shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    print("\nMissing Values:")
    for col, count in info["missing_values"].items():
        if count > 0:
            print(f"  {col}: {count}")
    print(f"\nMemory Usage: {info['memory_usage']:.2f} MB")
    print(f"Duplicate Rows: {info['duplicate_rows']}")

    # Display first few rows
    print("\n=== First 5 Rows ===")
    print(df.head())

    # Display basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())


if __name__ == "__main__":
    main()
