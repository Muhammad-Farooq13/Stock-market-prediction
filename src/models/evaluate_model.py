"""
Model Evaluation Module
Evaluates model performance with various metrics
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to handle model evaluation operations"""

    def __init__(self, results_dir: str = None):
        """
        Initialize ModelEvaluator

        Args:
            results_dir: Directory to save evaluation results
        """
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent.parent / "logs" / "evaluation"
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_results = []

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate various evaluation metrics

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "explained_variance": explained_variance_score(y_true, y_pred),
        }

        # Additional custom metrics
        metrics["mean_error"] = np.mean(y_true - y_pred)
        metrics["std_error"] = np.std(y_true - y_pred)
        metrics["max_error"] = np.max(np.abs(y_true - y_pred))
        metrics["median_error"] = np.median(np.abs(y_true - y_pred))

        return metrics

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate a single model

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, y_pred)

        # Create evaluation result
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "predictions": y_pred,
            "actuals": y_test.values,
        }

        # Log results
        logger.info(f"\n=== {model_name} Evaluation Results ===")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")

        self.evaluation_results.append(result)

        return result

    def evaluate_multiple_models(
        self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and compare performance

        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with comparison of all models
        """
        logger.info(f"Evaluating {len(models)} models...")

        results = []

        for model_name, model in models.items():
            result = self.evaluate_model(model, X_test, y_test, model_name)
            results.append({"model": model_name, **result["metrics"]})

        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values("rmse")

        logger.info("\n=== Model Comparison ===")
        logger.info(f"\n{comparison_df.to_string()}")

        # Save comparison
        comparison_path = self.results_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison saved to {comparison_path}")

        return comparison_df

    def calculate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate residuals and related statistics

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary containing residual statistics
        """
        residuals = y_true - y_pred

        residual_stats = {
            "residuals": residuals,
            "abs_residuals": np.abs(residuals),
            "squared_residuals": residuals**2,
            "percentage_errors": (residuals / y_true) * 100,
        }

        return residual_stats

    def calculate_prediction_intervals(
        self, y_pred: np.ndarray, residuals: np.ndarray, confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals

        Args:
            y_pred: Predicted values
            residuals: Residuals from predictions
            confidence: Confidence level

        Returns:
            Dictionary with lower and upper bounds
        """
        from scipy import stats

        # Calculate standard error
        std_error = np.std(residuals)

        # Calculate critical value
        alpha = 1 - confidence
        critical_value = stats.norm.ppf(1 - alpha / 2)

        # Calculate intervals
        margin = critical_value * std_error

        intervals = {
            "lower_bound": y_pred - margin,
            "upper_bound": y_pred + margin,
            "margin": margin,
        }

        return intervals

    def cross_validation_score(
        self, model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation

        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import cross_val_score

        logger.info(f"Performing {cv}-fold cross-validation...")

        # Calculate scores for different metrics
        rmse_scores = np.sqrt(
            -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
        )
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        results = {
            "rmse_mean": rmse_scores.mean(),
            "rmse_std": rmse_scores.std(),
            "mae_mean": mae_scores.mean(),
            "mae_std": mae_scores.std(),
            "r2_mean": r2_scores.mean(),
            "r2_std": r2_scores.std(),
        }

        logger.info(f"CV RMSE: {results['rmse_mean']:.4f} (+/- {results['rmse_std']:.4f})")
        logger.info(f"CV R²: {results['r2_mean']:.4f} (+/- {results['r2_std']:.4f})")

        return results

    def save_evaluation_results(self, results: Dict[str, Any], filename: str):
        """
        Save evaluation results to file

        Args:
            results: Evaluation results dictionary
            filename: Name for the output file
        """
        output_path = self.results_dir / filename

        # Remove non-serializable items
        serializable_results = {
            k: v for k, v in results.items() if k not in ["predictions", "actuals", "residuals"]
        }

        # Convert numpy values to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            else:
                return obj

        serializable_results = convert_to_serializable(serializable_results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=4)

        logger.info(f"Evaluation results saved to {output_path}")

    def create_predictions_dataframe(
        self, y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Series = None
    ) -> pd.DataFrame:
        """
        Create a dataframe with predictions and actuals

        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Date index (optional)

        Returns:
            DataFrame with predictions
        """
        df = pd.DataFrame(
            {
                "actual": y_true,
                "predicted": y_pred,
                "error": y_true - y_pred,
                "abs_error": np.abs(y_true - y_pred),
                "percentage_error": ((y_true - y_pred) / y_true) * 100,
            }
        )

        if dates is not None:
            df["date"] = dates.values

        return df


def main():
    """Main function to demonstrate model evaluation"""
    print("\n=== Model Evaluation Demo ===")
    print("Load data, train models, and evaluate using ModelEvaluator")

    # Example usage:
    # evaluator = ModelEvaluator()
    # result = evaluator.evaluate_model(model, X_test, y_test, "XGBoost")
    # comparison = evaluator.evaluate_multiple_models(models, X_test, y_test)


if __name__ == "__main__":
    main()
