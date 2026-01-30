# 🚀 Getting Started Guide

## Welcome!

This guide will help you get started with the Stock Market Prediction project. Follow these steps to set up, explore, and deploy your machine learning models.

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **pip** package manager
- **Git** for version control
- **Docker** (optional, for containerized deployment)
- **Jupyter** (optional, for notebooks)

Check your Python version:
```bash
python --version
```

---

## 🛠️ Installation

### Step 1: Clone or Download the Project

If you're reading this, you likely already have the project. If not:

```bash
git clone <your-repository-url>
cd "stock market"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- pandas, numpy (data processing)
- scikit-learn, xgboost, lightgbm (ML models)
- flask (API)
- mlflow (experiment tracking)
- pytest (testing)
- And more...

---

## 📊 Understanding Your Data

### Step 1: Inspect the Dataset

Your dataset is: `Daily_Global_Stock_Market_Indicators.csv`

Run the quickstart script to see basic info:

```bash
python quickstart.py
```

This will show you:
- Dataset shape
- Column names
- Missing values
- Basic statistics

### Step 2: Configure Column Names

**Important:** Update these files with your actual column names:

1. **config.yaml** - Set your target and date columns:
```yaml
data:
  target_column: "Close"  # Change to your price column
  date_column: "Date"     # Change to your date column
```

2. **src/utils/config.py** - Verify default settings match your data

---

## 🔍 Exploratory Data Analysis

### Open Jupyter Notebook

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

This notebook will guide you through:
1. Loading data
2. Data visualization
3. Feature engineering
4. Model training
5. Evaluation

**Follow the notebook cells** - they're designed to work step-by-step!

---

## 🤖 Training Models

### Option 1: Quick Training (Recommended for first time)

Train models using the automated pipeline:

```bash
python mlops_pipeline.py
```

This will:
- ✅ Load and preprocess data
- ✅ Engineer features
- ✅ Train multiple models
- ✅ Evaluate performance
- ✅ Save the best model

### Option 2: Custom Training

Create a Python script:

```python
from src.data.load_data import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.train_model import ModelTrainer

# Load data
loader = DataLoader()
df = loader.load_stock_data()

# Preprocess
preprocessor = DataPreprocessor()
data = preprocessor.preprocess_pipeline(
    df, 
    target_column='Close',  # Your target
    date_column='Date'      # Your date column
)

# Train models
trainer = ModelTrainer()
results = trainer.train_multiple_models(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val']
)

# Best model is automatically saved!
```

---

## 🌐 Deploying the API

### Option 1: Local Deployment

Start the Flask API:

```bash
python flask_app.py
```

Visit: **http://localhost:5000**

You'll see a beautiful documentation page!

### Option 2: Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t stock-market-predictor .

# Run container
docker run -p 5000:5000 stock-market-predictor
```

### Option 3: Docker Compose (Multiple Services)

```bash
# Start all services (API + MLflow)
docker-compose up -d

# Stop services
docker-compose down
```

---

## 📡 Using the API

### Health Check

```bash
curl http://localhost:5000/health
```

### Make a Prediction

**Using curl:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Close_MA_20": 150.5,
      "RSI_14": 45.2,
      "MACD": 2.3
    }
  }'
```

**Using Python:**

```python
import requests

response = requests.post(
    'http://localhost:5000/predict',
    json={
        'features': {
            'Close_MA_20': 150.5,
            'RSI_14': 45.2,
            'MACD': 2.3
        }
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
```

---

## 🧪 Running Tests

Ensure everything works correctly:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
# Windows: start htmlcov/index.html
# Linux/Mac: open htmlcov/index.html
```

---

## 📈 Monitoring & MLOps

### View MLflow Experiments

```bash
mlflow ui
```

Visit: **http://localhost:5001**

Here you can:
- Track experiments
- Compare model performance
- View parameters and metrics
- Download model artifacts

### Model Retraining

The pipeline supports automated retraining:

```python
from mlops_pipeline import MLOpsPipeline

pipeline = MLOpsPipeline()
pipeline.retrain_model(
    model_name='xgboost',
    trigger='scheduled'
)
```

---

## 🎯 Common Workflows

### Workflow 1: First-Time Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run quickstart: `python quickstart.py`
3. Update config.yaml with your column names
4. Explore in notebook: `jupyter notebook notebooks/01_exploratory_analysis.ipynb`
5. Train models: `python mlops_pipeline.py`
6. Start API: `python flask_app.py`

### Workflow 2: Development

1. Make changes to code
2. Run tests: `pytest tests/ -v`
3. Format code: `black src/ tests/`
4. Commit changes: `git commit -m "Your message"`

### Workflow 3: Production Deployment

1. Train and validate models
2. Build Docker image: `docker build -t stock-market-predictor .`
3. Run container: `docker run -p 5000:5000 stock-market-predictor`
4. Set up monitoring
5. Configure automated retraining

---

## 🐛 Troubleshooting

### Problem: Module Not Found

**Solution:** Make sure you're in the project directory and virtual environment is activated

```bash
pip install -r requirements.txt
```

### Problem: Model Not Loading in API

**Solution:** Ensure you've trained a model first

```bash
python mlops_pipeline.py
```

The model will be saved in the `models/` directory.

### Problem: Port Already in Use

**Solution:** Use a different port

```bash
# Edit flask_app.py and change port to 5001
# Or kill the process using port 5000
```

### Problem: Import Errors

**Solution:** Add project to PYTHONPATH

```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;.

# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:."
```

---

## 📚 Next Steps

1. **Customize Features**: Edit `src/features/feature_engineering.py` to add custom indicators
2. **Tune Hyperparameters**: Modify `config.yaml` for better performance
3. **Add New Models**: Extend `ModelTrainer` class with new algorithms
4. **Improve API**: Add authentication, rate limiting, etc.
5. **Deploy to Cloud**: AWS, Azure, GCP, Heroku, etc.

---

## 💡 Tips & Best Practices

1. **Always work in a virtual environment**
2. **Run tests before committing code**
3. **Keep your requirements.txt updated**
4. **Use meaningful commit messages**
5. **Document your changes**
6. **Monitor model performance regularly**
7. **Version your models with MLflow**
8. **Keep sensitive data out of version control**

---

## 🤝 Getting Help

- **Read the Documentation**: Check `README.md` and `PROJECT_SUMMARY.md`
- **Check Logs**: Look in `logs/` directory for detailed error messages
- **Run Tests**: `pytest tests/ -v` to identify issues
- **GitHub Issues**: Open an issue if you find bugs

---

## ✅ Checklist

Before deploying to production:

- [ ] All tests pass
- [ ] Model performance is acceptable
- [ ] API documentation is complete
- [ ] Error handling is implemented
- [ ] Logging is configured
- [ ] Security measures in place
- [ ] Monitoring set up
- [ ] Backup strategy defined
- [ ] Deployment procedure documented

---

## 🎉 You're Ready!

Congratulations! You now have a complete understanding of the project.

**Start with the quickstart**, explore the notebook, train your models, and deploy your API!

Happy predicting! 📈🚀

---

*For more detailed information, see:*
- `README.md` - Comprehensive project documentation
- `PROJECT_SUMMARY.md` - Project overview
- `CONTRIBUTING.md` - How to contribute
