# Global Stock Market Indicators Analysis and Prediction

## 📊 Project Overview

This project provides a comprehensive analysis and prediction framework for global stock market indicators. It implements machine learning models to forecast market trends and provides insights into historical patterns across various global markets.

### Objectives
- Analyze historical global stock market indicators
- Build predictive models for market forecasting
- Deploy the model as a REST API using Flask
- Implement MLOps best practices for continuous integration and deployment
- Provide interactive visualizations and dashboards

## 📁 Project Structure

```
stock market/
├── data/
│   ├── raw/                          # Raw, immutable data
│   └── processed/                    # Cleaned and processed data
├── notebooks/
│   └── 01_exploratory_analysis.ipynb # Jupyter notebooks for exploration
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py             # Data loading utilities
│   │   └── preprocessing.py          # Data cleaning and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py    # Feature creation and selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py           # Model training scripts
│   │   ├── evaluate_model.py        # Model evaluation metrics
│   │   └── predict.py               # Prediction utilities
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py             # Visualization functions
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── logger.py                # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
├── models/                          # Saved model artifacts
├── logs/                            # Application logs
├── flask_app.py                     # Flask API application
├── mlops_pipeline.py                # MLOps CI/CD pipeline
├── Dockerfile                       # Docker configuration
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## 📈 Dataset Overview

### Source
**Dataset**: Daily Global Stock Market Indicators
- **File**: `Daily_Global_Stock_Market_Indicators.csv`
- **Description**: Contains daily stock market indicators from major global markets

### Features
The dataset includes various stock market metrics such as:
- Market indices (S&P 500, NASDAQ, Dow Jones, etc.)
- Trading volumes
- Opening, closing, high, and low prices
- Technical indicators
- Date/time information

### Preprocessing Steps
1. **Data Cleaning**: Handle missing values, outliers, and duplicates
2. **Feature Engineering**: Create technical indicators (MA, RSI, MACD, etc.)
3. **Normalization**: Scale features for model training
4. **Time-series Split**: Split data chronologically for train/validation/test sets

## 🤖 Model Development

### Model Selection
The project evaluates multiple models:
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble method for non-linear patterns
- **XGBoost**: Gradient boosting for high performance
- **LSTM**: Deep learning for time-series forecasting

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

### Hyperparameter Tuning
- Grid Search CV for traditional models
- Optuna for advanced optimization
- Cross-validation strategies for robust evaluation

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock market
```

2. **Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up the data**
```bash
# Move the dataset to the raw data folder
# The data is already in the root, scripts will handle it
```

## 💻 Usage

### 1. Data Preprocessing
```bash
python src/data/preprocessing.py
```

### 2. Feature Engineering
```bash
python src/features/feature_engineering.py
```

### 3. Model Training
```bash
python src/models/train_model.py
```

### 4. Model Evaluation
```bash
python src/models/evaluate_model.py
```

### 5. Run Flask Application
```bash
python flask_app.py
```
The API will be available at `http://localhost:5000`

### API Endpoints
- `GET /`: Health check
- `POST /predict`: Make predictions
  ```json
  {
    "features": [value1, value2, value3, ...]
  }
  ```
- `GET /metrics`: Get model performance metrics

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t stock-market-predictor .
```

### Run Docker Container
```bash
docker run -p 5000:5000 stock-market-predictor
```

### Docker Compose (Optional)
```bash
docker-compose up -d
```

## 🔄 MLOps Pipeline

### Continuous Integration
The project implements:
- **Version Control**: Git for code versioning
- **Automated Testing**: pytest for unit and integration tests
- **Code Quality**: pylint, flake8 for code linting
- **Model Versioning**: MLflow for experiment tracking

### Continuous Deployment
```bash
python mlops_pipeline.py
```

Features:
- Automated model retraining
- Model performance monitoring
- A/B testing capabilities
- Automated deployment to production

### Model Monitoring
- Track prediction accuracy over time
- Monitor data drift
- Alert system for model degradation
- Logging of all predictions and errors

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_models.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📊 Visualization

Generate visualizations:
```bash
python src/visualization/visualize.py
```

This creates:
- Time series plots
- Correlation heatmaps
- Feature importance charts
- Model performance comparisons
- Prediction vs actual plots

## 🛠️ Development

### Project Dependencies
All dependencies are listed in `requirements.txt`:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning
- xgboost: Gradient boosting
- tensorflow/keras: Deep learning
- flask: API framework
- mlflow: Experiment tracking
- pytest: Testing framework
- plotly, matplotlib: Visualization

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all functions with docstrings
- Keep functions focused and modular

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Muhammad Farooq - Initial work - [@Muhammad-Farooq-13](https://github.com/Muhammad-Farooq-13)

## 🙏 Acknowledgments

- Data source: Global Stock Market Indicators
- Open source ML community
- Contributors and maintainers

## 📞 Contact

For questions or feedback, please reach out:
- Email: mfarooqshafee333@gmail.com
- GitHub: [@Muhammad-Farooq-13](https://github.com/Muhammad-Farooq-13)
- Project: [Stock Market Prediction](https://github.com/Muhammad-Farooq-13/stock-market-prediction)

## 🔮 Future Enhancements

- [ ] Add real-time data streaming
- [ ] Implement ensemble models
- [ ] Create interactive dashboard with Streamlit
- [ ] Add support for cryptocurrency markets
- [ ] Implement reinforcement learning for trading strategies
- [ ] Deploy on cloud platforms (AWS, Azure, GCP)
