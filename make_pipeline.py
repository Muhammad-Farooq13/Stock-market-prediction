"""
Pipeline script — run this once locally to train and save models.
Streamlit Cloud will load the pre-trained .joblib files from the models/ folder.

Usage:
    python make_pipeline.py
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "Daily_Global_Stock_Market_Indicators.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    logger.info("Loading dataset …")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values(["Index_Name", "Date"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features …")
    out = []
    for idx, grp in df.groupby("Index_Name"):
        grp = grp.copy().sort_values("Date")
        c = "Close"
        grp["MA_5"] = grp[c].rolling(5).mean()
        grp["MA_20"] = grp[c].rolling(20).mean()
        grp["MA_50"] = grp[c].rolling(50).mean()
        grp["EMA_12"] = grp[c].ewm(span=12, adjust=False).mean()
        grp["EMA_26"] = grp[c].ewm(span=26, adjust=False).mean()
        grp["MACD"] = grp["EMA_12"] - grp["EMA_26"]
        grp["Signal"] = grp["MACD"].ewm(span=9, adjust=False).mean()
        grp["MACD_Hist"] = grp["MACD"] - grp["Signal"]

        # RSI
        delta = grp[c].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        grp["RSI"] = 100 - 100 / (1 + rs)

        # Bollinger Bands
        grp["BB_mid"] = grp[c].rolling(20).mean()
        bb_std = grp[c].rolling(20).std()
        grp["BB_upper"] = grp["BB_mid"] + 2 * bb_std
        grp["BB_lower"] = grp["BB_mid"] - 2 * bb_std
        grp["BB_width"] = (grp["BB_upper"] - grp["BB_lower"]) / (grp["BB_mid"] + 1e-9)

        # Price momentum
        grp["Return_1d"] = grp[c].pct_change(1)
        grp["Return_5d"] = grp[c].pct_change(5)
        grp["Return_20d"] = grp[c].pct_change(20)

        # Volatility
        grp["Volatility_10"] = grp["Return_1d"].rolling(10).std()

        # Volume features
        grp["Vol_MA_10"] = grp["Volume"].rolling(10).mean()
        grp["Vol_ratio"] = grp["Volume"] / (grp["Vol_MA_10"] + 1e-9)

        # High/Low range
        grp["HL_range"] = (grp["High"] - grp["Low"]) / (grp["Close"] + 1e-9)

        # Lag features
        for lag in [1, 2, 3, 5]:
            grp[f"Close_lag_{lag}"] = grp[c].shift(lag)
            grp[f"Return_lag_{lag}"] = grp["Return_1d"].shift(lag)

        # Targets
        grp["target_regression"] = grp[c].shift(-1)  # next-day close
        next_ret = grp[c].shift(-1) / (grp[c] + 1e-9) - 1
        grp["target_direction"] = np.where(next_ret > 0.005, 1, np.where(next_ret < -0.005, -1, 0))

        out.append(grp)

    df_feat = pd.concat(out).reset_index(drop=True)
    return df_feat


# ─────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume", "Daily_Change_Percent",
    "MA_5", "MA_20", "MA_50", "EMA_12", "EMA_26",
    "MACD", "Signal", "MACD_Hist",
    "RSI", "BB_mid", "BB_upper", "BB_lower", "BB_width",
    "Return_1d", "Return_5d", "Return_20d", "Volatility_10",
    "Vol_MA_10", "Vol_ratio", "HL_range",
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5",
    "Return_lag_1", "Return_lag_2", "Return_lag_3", "Return_lag_5",
]


def train_models(df_feat: pd.DataFrame):
    df_feat = df_feat.dropna(subset=FEATURE_COLS + ["target_regression", "target_direction"])

    X = df_feat[FEATURE_COLS]
    y_reg = df_feat["target_regression"]
    y_clf = df_feat["target_direction"]

    # Encode direction: -1 → 0, 0 → 1, 1 → 2
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf_enc, test_size=0.2, shuffle=False
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Regression ──
    logger.info("Training regression model (XGBoost) …")
    reg_model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    reg_model.fit(X_train_sc, y_reg_train)
    y_pred_reg = reg_model.predict(X_test_sc)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    mae = mean_absolute_error(y_reg_test, y_pred_reg)
    r2 = r2_score(y_reg_test, y_pred_reg)
    logger.info(f"  Regression — RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

    # ── Classification ──
    logger.info("Training classification model (XGBoost) …")
    clf_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
        eval_metric="mlogloss",
    )
    clf_model.fit(X_train_sc, y_clf_train)
    y_pred_clf = clf_model.predict(X_test_sc)
    f1 = f1_score(y_clf_test, y_pred_clf, average="weighted")
    acc = accuracy_score(y_clf_test, y_pred_clf)
    logger.info(f"  Classification — F1 weighted: {f1:.4f} | Accuracy: {acc:.4f}")

    # ── Baseline (Ridge) ──
    ridge = Ridge()
    ridge.fit(X_train_sc, y_reg_train)

    # ── Save artefacts ──
    logger.info("Saving models …")
    joblib.dump(reg_model, MODELS_DIR / "regression_xgb.joblib")
    joblib.dump(clf_model, MODELS_DIR / "classification_xgb.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_xgb.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(
        {"rmse": rmse, "mae": mae, "r2": r2, "f1": f1, "accuracy": acc},
        MODELS_DIR / "metrics.joblib",
    )
    logger.info(f"Models saved to {MODELS_DIR}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "f1": f1, "accuracy": acc}


# ─────────────────────────────────────────────
# 4. Save processed features for app
# ─────────────────────────────────────────────
def save_features(df_feat: pd.DataFrame):
    out_path = ROOT / "data" / "processed" / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_path, index=False)
    logger.info(f"Features saved to {out_path}")


if __name__ == "__main__":
    df = load_data()
    df_feat = build_features(df)
    save_features(df_feat)
    metrics = train_models(df_feat)
    logger.info("Pipeline complete ✅")
    logger.info(f"Final metrics: {metrics}")
