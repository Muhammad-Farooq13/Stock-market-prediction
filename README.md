# Global Stock Market Indicators — Analysis & Prediction

> **End-to-end ML pipeline that forecasts daily stock-market direction and price level across major global indices, giving portfolio managers and quant teams an explainable, reproducible baseline for systematic trading decisions.**

[![CI](https://github.com/Muhammad-Farooq13/Stock-market-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/Stock-market-prediction/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.5.0-blue.svg)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.2.0-blueviolet.svg)](https://optuna.org/)
[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Demo-%E2%96%B6%20Launch-FF4B4B?style=flat)](https://Muhammad-Farooq13.github.io)

---

## Problem Statement

- **Regression task** — predict the next-day closing price of a market index (continuous target).
- **Classification task** — predict the next-day direction (Up / Down / Neutral) for position sizing.
- Both tasks share the same feature-engineering pipeline and are tuned end-to-end with Optuna HPO.
- Every prediction is accompanied by SHAP force-plots so analysts understand **why** the model made a call.
- All experiments are tracked in MLflow for full reproducibility.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Name** | Daily Global Stock Market Indicators |
| **File** | `Daily_Global_Stock_Market_Indicators.csv` |
| **Coverage** | Major global indices (S&P 500, NASDAQ, FTSE, Nikkei, DAX, …) |
| **Features** | OHLCV + technical indicators (MA, RSI, MACD, Bollinger Bands) |
| **Target (regression)** | Next-day closing price |
| **Target (classification)** | Next-day price direction (3-class) |

---

## Model Results

| Model | Task | RMSE | MAE | R² | F1-weighted |
|-------|------|------|-----|----|-------------|
| XGBoost | Regression | ~12.6 | — | ~0.51 | — |
| XGBoost | Classification | — | — | — | ~0.45 |
| LightGBM | Regression | — | — | — | — |
| Baseline (mean) | Regression | — | — | 0.00 | — |

> Results are indicative; re-run `make pipeline` for up-to-date numbers on your data split.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **ML / Boosting** | XGBoost 1.7, LightGBM 4.0, scikit-learn 1.3 |
| **HPO** | Optuna 3.2, Hyperopt |
| **Tracking** | MLflow 2.5, DVC 3.15 |
| **API** | Flask 2.3, Gunicorn |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Testing** | pytest, pytest-cov, pytest-mock |
| **Code Quality** | black, flake8, isort |
| **CI/CD** | GitHub Actions |
| **Containers** | Docker, Docker Compose |

---

## Quick Start

```bash
# 1 — clone
git clone https://github.com/Muhammad-Farooq13/Stock-market-prediction.git
cd Stock-market-prediction

# 2 — create virtual environment & install
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# 3 — run the full ML pipeline
python -m src.data.preprocessing
python -m src.features.feature_engineering
python -m src.models.train_model
python -m src.models.evaluate_model

# 4 — start the REST API
python flask_app.py          # → http://localhost:5000
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Return price / direction prediction |
| `GET` | `/metrics` | Latest model performance metrics |

**Predict payload example:**
```json
{ "features": [value1, value2, "..."] }
```

---

## Docker Quick Start

```bash
# build
docker build -t stock-market-predictor .

# run (API on port 5000)
docker run -p 5000:5000 stock-market-predictor

# or with docker-compose
docker-compose up -d
```

---

## Project Structure

```
Stock-market-prediction/
├── data/
│   ├── raw/                              # Immutable source data
│   └── processed/                        # Cleaned & feature-engineered data
├── src/
│   ├── data/
│   │   ├── load_data.py                  # Data loading utilities
│   │   └── preprocessing.py             # Cleaning, splitting, scaling
│   ├── features/
│   │   └── feature_engineering.py       # Technical indicator construction
│   ├── models/
│   │   ├── train_model.py               # Training + Optuna HPO
│   │   ├── evaluate_model.py            # Metric computation
│   │   └── predict.py                   # Inference helpers
│   ├── visualization/
│   │   └── visualize.py                 # Plots & dashboards
│   └── utils/
│       ├── config.py                    # Configuration management
│       └── logger.py                    # Structured logging
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
├── models/                              # Saved model artefacts (.joblib)
├── logs/                                # Application logs
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── flask_app.py                         # Flask REST API
├── mlops_pipeline.py                    # Automated MLOps pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## ⚠️ Limitations

- The dataset is sourced from publicly available historical OHLCV data. It **does not** incorporate alternative data (news sentiment, order-book depth, macro releases).
- Model accuracy is sensitive to the chosen train/test split boundary — results may not generalise to live trading without further validation.
- No look-ahead bias mitigation beyond a simple chronological split; walk-forward cross-validation is a recommended next step.
- Classification F1 scores (~0.45) are only marginally above a naive majority-class baseline on this dataset.

---

## Screenshots

> _Screenshots coming soon — dashboard visualisations will be added after UI polish._

---

## Testing

```bash
# run all tests
pytest tests/ -v

# with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit with a descriptive message: `git commit -m "feat: describe your change"`
4. Push and open a Pull Request

---

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS* (SHAP).
- Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD '19*.
- Yahoo Finance / Kaggle — Global Stock Market Indicators dataset.

---

## License

MIT © [Muhammad Farooq](https://github.com/Muhammad-Farooq13)

---

<p align="center">
  <a href="https://Muhammad-Farooq13.github.io">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Demo-Try%20the%20App%20Now-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
</p>
- [ ] Implement reinforcement learning for trading strategies
- [ ] Deploy on cloud platforms (AWS, Azure, GCP)
