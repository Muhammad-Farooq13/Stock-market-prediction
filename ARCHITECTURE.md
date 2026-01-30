# Project Architecture

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Raw Data  →  Data Loader  →  Preprocessor  →  Processed Data  │
│  (.csv)         (clean)        (transform)        (scaled)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  Technical Indicators  |  Lag Features  |  Time Features        │
│  (MA, RSI, MACD, BB)   |  (Historical)  |  (Date/Time)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  Linear Reg  →  Random Forest  →  XGBoost  →  LightGBM         │
│  (baseline)     (ensemble)        (boosting)  (fast)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Metrics Calculation  →  Model Comparison  →  Best Model        │
│  (RMSE, MAE, R²)         (Performance)         (Selection)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│              Flask API  ←→  Docker Container                     │
│                   ↓              ↓                               │
│              REST Endpoints  Production Server                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MLOPS LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  MLflow  →  Monitoring  →  Retraining  →  Versioning           │
│  (track)    (metrics)      (automated)     (models)             │
└─────────────────────────────────────────────────────────────────┘
```

## 🗂️ Directory Structure

```
stock market/
│
├── 📁 data/                          # Data storage
│   ├── raw/                          # Original datasets
│   │   └── .gitkeep
│   └── processed/                    # Cleaned datasets
│       └── .gitkeep
│
├── 📁 notebooks/                     # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb
│
├── 📁 src/                           # Source code
│   ├── __init__.py
│   │
│   ├── 📁 data/                      # Data handling
│   │   ├── __init__.py
│   │   ├── load_data.py             # Data loading
│   │   └── preprocessing.py          # Data preprocessing
│   │
│   ├── 📁 features/                  # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py    # Technical indicators
│   │
│   ├── 📁 models/                    # ML models
│   │   ├── __init__.py
│   │   ├── train_model.py           # Model training
│   │   ├── evaluate_model.py        # Model evaluation
│   │   └── predict.py               # Predictions
│   │
│   ├── 📁 visualization/             # Plotting
│   │   ├── __init__.py
│   │   └── visualize.py             # Visualization tools
│   │
│   └── 📁 utils/                     # Utilities
│       ├── __init__.py
│       ├── config.py                # Configuration
│       └── logger.py                # Logging
│
├── 📁 tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
│
├── 📁 models/                        # Saved models
│   └── .gitkeep
│
├── 📁 logs/                          # Application logs
│   └── .gitkeep
│
├── 📁 .github/                       # GitHub Actions
│   └── workflows/
│       └── ci.yml                   # CI/CD pipeline
│
├── 📄 flask_app.py                   # Flask API
├── 📄 mlops_pipeline.py              # MLOps automation
├── 📄 run_pipeline.py                # Pipeline runner
├── 📄 quickstart.py                  # Quick demo
│
├── 📄 Dockerfile                     # Docker config
├── 📄 docker-compose.yml             # Multi-container
├── 📄 requirements.txt               # Dependencies
├── 📄 setup.py                       # Package setup
├── 📄 config.yaml                    # Configuration
├── 📄 pytest.ini                     # Test config
├── 📄 Makefile                       # Commands
│
├── 📄 .gitignore                     # Git ignore rules
├── 📄 README.md                      # Main documentation
├── 📄 PROJECT_SUMMARY.md             # Project overview
├── 📄 GETTING_STARTED.md             # Getting started
├── 📄 CONTRIBUTING.md                # Contribution guide
└── 📄 LICENSE                        # MIT License
```

## 🔄 Data Flow

```
1. RAW DATA
   └─→ Daily_Global_Stock_Market_Indicators.csv

2. DATA LOADING (src/data/load_data.py)
   └─→ DataLoader.load_stock_data()
       └─→ DataFrame

3. PREPROCESSING (src/data/preprocessing.py)
   ├─→ Handle missing values
   ├─→ Remove outliers
   ├─→ Encode categories
   └─→ Scale features
       └─→ Processed DataFrame

4. FEATURE ENGINEERING (src/features/feature_engineering.py)
   ├─→ Moving Averages (MA_5, MA_10, MA_20, ...)
   ├─→ RSI (14-period)
   ├─→ MACD (12, 26, 9)
   ├─→ Bollinger Bands
   ├─→ Lag features
   └─→ Time features
       └─→ Enriched DataFrame

5. MODEL TRAINING (src/models/train_model.py)
   ├─→ Train/Val/Test split
   ├─→ Train multiple models
   │   ├─→ Linear Regression
   │   ├─→ Random Forest
   │   ├─→ XGBoost
   │   └─→ LightGBM
   └─→ Save best model
       └─→ model.pkl

6. EVALUATION (src/models/evaluate_model.py)
   ├─→ Calculate metrics (RMSE, MAE, R²)
   ├─→ Compare models
   └─→ Generate reports
       └─→ model_comparison.csv

7. DEPLOYMENT (flask_app.py)
   ├─→ Load trained model
   ├─→ Create API endpoints
   │   ├─→ /health
   │   ├─→ /predict
   │   ├─→ /predict_batch
   │   └─→ /metrics
   └─→ Serve predictions
       └─→ HTTP 200 OK

8. MONITORING (mlops_pipeline.py)
   ├─→ Track with MLflow
   ├─→ Monitor performance
   └─→ Trigger retraining
       └─→ Continuous improvement
```

## 🔧 Component Interactions

```
┌──────────────┐
│   Config     │  ←──── config.yaml
│   Manager    │  ←──── src/utils/config.py
└──────┬───────┘
       │ provides settings to
       ↓
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  DataLoader  │  →→→  │ Preprocessor │  →→→  │   Feature    │
│              │       │              │       │  Engineer    │
└──────────────┘       └──────────────┘       └──────┬───────┘
                                                      │
                                                      ↓
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  Visualizer  │  ←←←  │  Evaluator   │  ←←←  │   Trainer    │
│              │       │              │       │              │
└──────────────┘       └──────────────┘       └──────┬───────┘
                                                      │
                                                      ↓
                                               ┌──────────────┐
                                               │     Model    │
                                               │   (.pkl)     │
                                               └──────┬───────┘
                                                      │
                                                      ↓
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    Docker    │  ←←←  │   Flask API  │  ←←←  │  Predictor   │
│  Container   │       │              │       │              │
└──────────────┘       └──────────────┘       └──────────────┘
       ↓
┌──────────────┐
│    Users     │
│  (Requests)  │
└──────────────┘
```

## 🚀 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT                              │
├─────────────────────────────────────────────────────────────┤
│  Local Machine                                               │
│  ├── Python Virtual Environment                             │
│  ├── Jupyter Notebooks                                      │
│  ├── Model Training                                         │
│  └── Testing                                                │
└─────────────────────────────────────────────────────────────┘
                         ↓  git push
┌─────────────────────────────────────────────────────────────┐
│                     VERSION CONTROL                          │
├─────────────────────────────────────────────────────────────┤
│  GitHub Repository                                           │
│  ├── Source Code                                            │
│  ├── Documentation                                          │
│  └── CI/CD Workflows                                        │
└─────────────────────────────────────────────────────────────┘
                         ↓  trigger
┌─────────────────────────────────────────────────────────────┐
│                     CI/CD PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│  GitHub Actions                                              │
│  ├── Run Tests                                              │
│  ├── Check Code Quality                                     │
│  ├── Build Docker Image                                     │
│  └── Deploy (if tests pass)                                │
└─────────────────────────────────────────────────────────────┘
                         ↓  deploy
┌─────────────────────────────────────────────────────────────┐
│                     PRODUCTION                               │
├─────────────────────────────────────────────────────────────┤
│  Docker Container                                            │
│  ├── Flask API (Port 5000)                                  │
│  ├── Gunicorn (WSGI Server)                                 │
│  ├── Health Checks                                          │
│  └── Monitoring                                             │
│                                                              │
│  MLflow Server (Port 5001)                                   │
│  ├── Experiment Tracking                                    │
│  ├── Model Registry                                         │
│  └── Artifact Storage                                       │
└─────────────────────────────────────────────────────────────┘
                         ↓  requests
┌─────────────────────────────────────────────────────────────┐
│                     END USERS                                │
├─────────────────────────────────────────────────────────────┤
│  Web Applications                                            │
│  Mobile Apps                                                 │
│  Other Services                                              │
│  Data Scientists                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📊 MLOps Workflow

```
┌─────────────┐
│   Train     │  ──→  New data arrives
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Evaluate   │  ──→  Check performance metrics
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Deploy    │  ──→  If metrics acceptable
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Monitor    │  ──→  Track predictions & accuracy
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Retrain?   │  ──→  If performance degrades
└──────┬──────┘
       │
       └──────→  Back to Train
```

---

## 📝 Notes

- **All components are modular** and can be used independently
- **Configuration is centralized** in config.yaml and src/utils/config.py
- **Logging is consistent** across all modules
- **Tests cover** all major functionality
- **Docker enables** easy deployment anywhere
- **MLflow provides** experiment tracking and model versioning

---

This architecture ensures:
✅ **Scalability** - Easy to add new models or features
✅ **Maintainability** - Clear separation of concerns
✅ **Testability** - Comprehensive test coverage
✅ **Deployability** - Ready for production
✅ **Reproducibility** - Version control and tracking
