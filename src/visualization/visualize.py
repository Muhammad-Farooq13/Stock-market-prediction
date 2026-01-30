"""
Data Visualization Module
Creates various plots and visualizations for data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """Class to handle data visualization operations"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize Visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / "logs" / "visualizations"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_time_series(self, df: pd.DataFrame, date_column: str, 
                        value_columns: List[str], title: str = "Time Series Plot",
                        save_name: str = None):
        """
        Plot time series data
        
        Args:
            df: DataFrame containing the data
            date_column: Name of the date column
            value_columns: List of columns to plot
            title: Plot title
            save_name: Filename to save the plot
        """
        fig = go.Figure()
        
        for col in value_columns:
            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df[col],
                mode='lines',
                name=col
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_name:
            fig.write_html(self.output_dir / f"{save_name}.html")
            logger.info(f"Saved plot to {save_name}.html")
        
        fig.show()
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, title: str = "Correlation Matrix",
                               save_name: str = None, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation matrix heatmap
        
        Args:
            df: DataFrame containing the data
            title: Plot title
            save_name: Filename to save the plot
            figsize: Figure size
        """
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_name}.png")
        
        plt.show()
    
    def plot_distribution(self, df: pd.DataFrame, columns: List[str],
                         title: str = "Distribution Plot", save_name: str = None):
        """
        Plot distribution of numerical columns
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to plot
            title: Plot title
            save_name: Filename to save the plot
        """
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if col in df.columns:
                sns.histplot(df[col], kde=True, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_name}.png")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               title: str = "Feature Importance",
                               top_n: int = 20, save_name: str = None):
        """
        Plot feature importance
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            title: Plot title
            top_n: Number of top features to show
            save_name: Filename to save the plot
        """
        # Get top features
        top_features = feature_importance.head(top_n)
        
        # Create bar plot
        fig = px.bar(top_features, x='importance', y='feature',
                    orientation='h', title=title,
                    labels={'importance': 'Importance', 'feature': 'Feature'})
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            height=max(400, top_n * 20)
        )
        
        if save_name:
            fig.write_html(self.output_dir / f"{save_name}.html")
            logger.info(f"Saved plot to {save_name}.html")
        
        fig.show()
        return fig
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = "Predictions vs Actual",
                                   save_name: str = None):
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_name: Filename to save the plot
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Scatter Plot', 'Time Series Comparison')
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers',
                      name='Predictions', marker=dict(size=5, opacity=0.6)),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Time series comparison
        indices = np.arange(len(y_true))
        fig.add_trace(
            go.Scatter(x=indices, y=y_true, mode='lines',
                      name='Actual', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=indices, y=y_pred, mode='lines',
                      name='Predicted', line=dict(color='red')),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Sample Index", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        
        fig.update_layout(
            title_text=title,
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        if save_name:
            fig.write_html(self.output_dir / f"{save_name}.html")
            logger.info(f"Saved plot to {save_name}.html")
        
        fig.show()
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Analysis", save_name: str = None):
        """
        Plot residual analysis
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_name: Filename to save the plot
        """
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Predicted', 'Residual Distribution',
                          'Q-Q Plot', 'Residuals Over Time')
        )
        
        # Residuals vs predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers',
                      marker=dict(size=5, opacity=0.6)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residual distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=50),
            row=1, col=2
        )
        
        # Q-Q plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers',
                      marker=dict(size=5, opacity=0.6)),
            row=2, col=1
        )
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Residuals over time
        indices = np.arange(len(residuals))
        fig.add_trace(
            go.Scatter(x=indices, y=residuals, mode='lines'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Index", row=2, col=2)
        fig.update_yaxes(title_text="Residuals", row=2, col=2)
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            template='plotly_white',
            height=800
        )
        
        if save_name:
            fig.write_html(self.output_dir / f"{save_name}.html")
            logger.info(f"Saved plot to {save_name}.html")
        
        fig.show()
        return fig
    
    def plot_candlestick(self, df: pd.DataFrame, date_column: str,
                        open_col: str, high_col: str, low_col: str, close_col: str,
                        volume_col: str = None, title: str = "Candlestick Chart",
                        save_name: str = None):
        """
        Plot candlestick chart for stock data
        
        Args:
            df: DataFrame containing stock data
            date_column: Name of date column
            open_col, high_col, low_col, close_col: OHLC column names
            volume_col: Volume column name (optional)
            title: Plot title
            save_name: Filename to save the plot
        """
        if volume_col:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df[date_column],
                    open=df[open_col],
                    high=df[high_col],
                    low=df[low_col],
                    close=df[close_col],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=df[date_column], y=df[volume_col], name='Volume'),
                row=2, col=1
            )
            
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        else:
            fig = go.Figure(data=[go.Candlestick(
                x=df[date_column],
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col]
            )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        if save_name:
            fig.write_html(self.output_dir / f"{save_name}.html")
            logger.info(f"Saved plot to {save_name}.html")
        
        fig.show()
        return fig


def main():
    """Main function to demonstrate visualization"""
    print("\n=== Visualization Demo ===")
    print("Create various plots for data analysis")
    
    # Example usage:
    # visualizer = Visualizer()
    # visualizer.plot_time_series(df, 'Date', ['Close', 'MA_20'])


if __name__ == "__main__":
    main()
