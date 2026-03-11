# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-03-11

### Added
- **End-to-end ML pipeline** — data loading, preprocessing, feature engineering, model training, evaluation, and prediction.
- **Multi-model support** — Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM.
- **Hyperparameter tuning** — GridSearchCV and Optuna integration for automated HPO.
- **MLflow experiment tracking** — all training runs logged with metrics, parameters, and artefacts.
- **Feature engineering module** — 20+ technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, ROC, Williams %R, Stochastic Oscillator.
- **REST API (Flask)** — `/predict`, `/metrics`, `/health`, `/models`, `/feature-importance`, `/history` endpoints with CORS support.
- **Docker support** — single-stage `python:3.11-slim` image, Gunicorn WSGI server.
- **CI/CD pipeline** — GitHub Actions workflow running flake8 lint, pytest with coverage, and Docker build on every push to `main`.
- **Missing-value warning** — logs detected NaN counts before `fillna` operations in data preprocessing.
- **Comprehensive test suite** — unit tests for data loading, preprocessing, feature engineering, model training, and utilities (`pytest-cov`).
- **Structured logging** — `loguru`-based logger with configurable log levels and file rotation.
- **Professional README** — badges, problem statement, dataset table, model results, tech stack, quick-start, Docker instructions, and honest limitations.
- **pyproject.toml** — PEP 517/518 packaging with `setuptools.build_meta` build backend; replaces legacy `setup.py`.

### Changed
- Upgraded base Docker image from `python:3.9-slim` to `python:3.11-slim`.
- Simplified Dockerfile to a single-stage build.
- CI requirements split into `requirements-ci.txt` (omits tensorflow/keras) for faster CI runner installs.

### Fixed
- `pyproject.toml` build-backend corrected from invalid `setuptools.backends.legacy:build` to `setuptools.build_meta`.
- Removed duplicate `known_first_party` entries from isort configuration.
- All `flake8` warnings resolved: unused imports (F401), lines >100 chars (E501), trailing whitespace (W291/W293).
- All `black` + `isort` formatting applied across `src/`, `tests/`.

### Removed
- Legacy `setup.py` — packaging now handled entirely by `pyproject.toml`.

---

## [0.1.0] — 2026-03-03

### Added
- Initial project scaffold: directory structure, placeholder modules, basic requirements.
- Initial changelog entry.

