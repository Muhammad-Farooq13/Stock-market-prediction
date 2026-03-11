"""
Streamlit Dashboard — Global Stock Market Prediction
Run locally:  streamlit run streamlit_app.py
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "Daily_Global_Stock_Market_Indicators.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

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

DIRECTION_LABELS = {0: "⬇ Down", 1: "➡ Neutral", 2: "⬆ Up"}
DIRECTION_COLORS = {0: "#FF4B4B", 1: "#F0A500", 2: "#21C55D"}

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    reg = joblib.load(MODELS_DIR / "regression_xgb.joblib")
    clf = joblib.load(MODELS_DIR / "classification_xgb.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler_xgb.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    metrics = joblib.load(MODELS_DIR / "metrics.joblib")
    return reg, clf, scaler, le, metrics


@st.cache_data(show_spinner="Loading dataset …")
def load_raw_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values(["Index_Name", "Date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading features …")
def load_features():
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    return None


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def compute_features_for_row(row: dict) -> np.ndarray:
    """Build a single feature vector from user inputs + auto-derived fields."""
    c = row["Close"]
    o = row["Open"]
    h = row["High"]
    lo = row["Low"]
    vol = row["Volume"]
    dc = row["Daily_Change_Percent"]

    # Derived — use single-point proxies (no history)
    ma5 = ma20 = ma50 = ema12 = ema26 = c
    macd = signal = macd_hist = 0.0
    rsi = 50.0
    bb_mid = c
    bb_upper = c * 1.02
    bb_lower = c * 0.98
    bb_width = 0.04
    ret1 = dc / 100
    ret5 = ret20 = ret1
    vol10 = 0.02
    vol_ma10 = vol
    vol_ratio = 1.0
    hl_range = (h - lo) / (c + 1e-9)
    cl1 = cl2 = cl3 = cl5 = c
    rl1 = rl2 = rl3 = rl5 = ret1

    return np.array(
        [
            o, h, lo, c, vol, dc,
            ma5, ma20, ma50, ema12, ema26,
            macd, signal, macd_hist,
            rsi, bb_mid, bb_upper, bb_lower, bb_width,
            ret1, ret5, ret20, vol10,
            vol_ma10, vol_ratio, hl_range,
            cl1, cl2, cl3, cl5,
            rl1, rl2, rl3, rl5,
        ],
        dtype=float,
    ).reshape(1, -1)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image(
    "https://img.shields.io/badge/Stock_Market-Predictor-FF4B4B?style=for-the-badge",
    use_container_width=True,
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Overview", "📊 Explorer", "🤖 Predict", "📈 Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with XGBoost · MLflow · Streamlit  \n"
    "[GitHub](https://github.com/Muhammad-Farooq13/Stock-market-prediction) · "
    "[Live Demo](https://Muhammad-Farooq13.github.io)"
)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
models_exist = all(
    (MODELS_DIR / f).exists()
    for f in ["regression_xgb.joblib", "classification_xgb.joblib", "scaler_xgb.joblib", "label_encoder.joblib"]
)

if not models_exist:
    st.error(
        "⚠️ Model files not found in `models/`.  \n"
        "Run `python make_pipeline.py` locally, then commit the `.joblib` files."
    )
    st.stop()

reg_model, clf_model, scaler, le, metrics = load_models()
df_raw = load_raw_data()
df_feat = load_features()

indices = sorted(df_raw["Index_Name"].unique())
countries = sorted(df_raw["Country"].unique())

# ─────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("📈 Global Stock Market Prediction")
    st.markdown(
        "> End-to-end ML pipeline forecasting next-day closing price and direction across major global indices."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Indices covered", str(len(indices)))
    col2.metric("Countries", str(len(countries)))
    col3.metric("Total records", f"{len(df_raw):,}")
    col4.metric(
        "Date range",
        f"{df_raw['Date'].min().strftime('%Y-%m-%d')} → {df_raw['Date'].max().strftime('%Y-%m-%d')}",
    )

    st.markdown("---")
    st.subheader("Model Performance")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("RMSE", f"{metrics['rmse']:.2f}", help="Root Mean Squared Error (regression)")
    mc2.metric("MAE", f"{metrics['mae']:.2f}", help="Mean Absolute Error (regression)")
    mc3.metric("R²", f"{metrics['r2']:.4f}", help="R-squared (regression, test set)")
    mc4.metric("F1 weighted", f"{metrics['f1']:.4f}", help="F1 score (direction classification)")
    mc5.metric("Accuracy", f"{metrics['accuracy']:.4f}", help="Direction accuracy")

    st.markdown("---")
    st.subheader("Price History — All Indices")
    df_pivot = df_raw.groupby(["Date", "Index_Name"])["Close"].mean().reset_index()
    sel_indices = st.multiselect("Select indices", indices, default=indices[:4])
    df_plot = df_pivot[df_pivot["Index_Name"].isin(sel_indices)]
    fig = px.line(
        df_plot, x="Date", y="Close", color="Index_Name",
        template="plotly_dark", title="Closing Price Over Time",
    )
    fig.update_layout(legend_title_text="Index", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: Explorer
# ─────────────────────────────────────────────
elif page == "📊 Explorer":
    st.title("📊 Data Explorer")

    col1, col2 = st.columns(2)
    sel_index = col1.selectbox("Index", indices)
    chart_type = col2.selectbox("Chart type", ["Candlestick", "Line", "Volume"])

    df_idx = df_raw[df_raw["Index_Name"] == sel_index].sort_values("Date")

    if chart_type == "Candlestick":
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_idx["Date"],
                    open=df_idx["Open"],
                    high=df_idx["High"],
                    low=df_idx["Low"],
                    close=df_idx["Close"],
                    name=sel_index,
                )
            ]
        )
        fig.update_layout(
            title=f"{sel_index} — Candlestick",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=500,
        )
    elif chart_type == "Line":
        fig = px.line(
            df_idx, x="Date", y=["Open", "High", "Low", "Close"],
            template="plotly_dark", title=f"{sel_index} — OHLC",
        )
        fig.update_layout(height=500)
    else:
        fig = px.bar(
            df_idx, x="Date", y="Volume",
            template="plotly_dark", title=f"{sel_index} — Trading Volume",
        )
        fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    # Stats
    st.subheader("Statistics")
    st.dataframe(df_idx[["Open", "High", "Low", "Close", "Volume", "Daily_Change_Percent"]].describe().round(2), use_container_width=True)

    # Feature chart (if available)
    if df_feat is not None:
        df_f = df_feat[df_feat["Index_Name"] == sel_index].sort_values("Date").dropna(subset=["RSI", "MACD"])
        if len(df_f) > 0:
            st.subheader("Technical Indicators")
            tab1, tab2, tab3 = st.tabs(["Moving Averages", "RSI", "MACD"])
            with tab1:
                fig2 = go.Figure()
                for col, color in [("Close", "white"), ("MA_5", "#FF4B4B"), ("MA_20", "#21C55D"), ("MA_50", "#F0A500")]:
                    if col in df_f.columns:
                        fig2.add_trace(go.Scatter(x=df_f["Date"], y=df_f[col], name=col, line=dict(color=color)))
                fig2.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig2, use_container_width=True)
            with tab2:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_f["Date"], y=df_f["RSI"], name="RSI", line=dict(color="#FF4B4B")))
                fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought 70")
                fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold 30")
                fig3.update_layout(template="plotly_dark", height=350, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig3, use_container_width=True)
            with tab3:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=df_f["Date"], y=df_f["MACD"], name="MACD", line=dict(color="#FF4B4B")))
                fig4.add_trace(go.Scatter(x=df_f["Date"], y=df_f["Signal"], name="Signal", line=dict(color="#21C55D")))
                fig4.add_trace(go.Bar(x=df_f["Date"], y=df_f["MACD_Hist"], name="Histogram", marker_color="#F0A500"))
                fig4.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: Predict
# ─────────────────────────────────────────────
elif page == "🤖 Predict":
    st.title("🤖 Next-Day Prediction")
    st.markdown(
        "Enter today's market values to predict **tomorrow's closing price** and **price direction**."
    )

    # Pre-fill with latest row for selected index
    sel_index = st.selectbox("Select index to pre-fill", indices)
    latest = df_raw[df_raw["Index_Name"] == sel_index].sort_values("Date").iloc[-1]

    with st.form("predict_form"):
        st.subheader("Market Inputs")
        c1, c2, c3 = st.columns(3)
        open_val = c1.number_input("Open", value=float(latest["Open"]), format="%.2f")
        high_val = c2.number_input("High", value=float(latest["High"]), format="%.2f")
        low_val = c3.number_input("Low", value=float(latest["Low"]), format="%.2f")

        c4, c5, c6 = st.columns(3)
        close_val = c4.number_input("Close", value=float(latest["Close"]), format="%.2f")
        vol_val = c5.number_input("Volume", value=float(latest["Volume"]), format="%.0f")
        dc_val = c6.number_input("Daily Change %", value=float(latest["Daily_Change_Percent"]), format="%.4f")

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        row = {"Open": open_val, "High": high_val, "Low": low_val,
               "Close": close_val, "Volume": vol_val, "Daily_Change_Percent": dc_val}

        # Use feature history from parquet if available
        if df_feat is not None:
            df_hist = (
                df_feat[df_feat["Index_Name"] == sel_index]
                .sort_values("Date")
                .dropna(subset=FEATURE_COLS)
            )
            if len(df_hist) > 0:
                feat_vec = df_hist[FEATURE_COLS].iloc[[-1]].values
            else:
                feat_vec = compute_features_for_row(row)
        else:
            # Fallback: use scaler's feature_names_in_ to build vector
            feat_vec = compute_features_for_row(row)

        feat_scaled = scaler.transform(feat_vec)

        pred_price = reg_model.predict(feat_scaled)[0]
        pred_dir_enc = clf_model.predict(feat_scaled)[0]
        pred_proba = clf_model.predict_proba(feat_scaled)[0]
        pred_dir_label = DIRECTION_LABELS[pred_dir_enc]
        pred_dir_color = DIRECTION_COLORS[pred_dir_enc]

        st.markdown("---")
        st.subheader("Prediction Results")
        r1, r2 = st.columns(2)
        r1.metric(
            "Predicted next-day close",
            f"{pred_price:,.2f}",
            delta=f"{pred_price - close_val:+.2f} ({(pred_price/close_val - 1)*100:+.2f}%)",
        )
        r2.markdown(
            f"<p style='font-size:18px; color:{pred_dir_color}; font-weight:600;'>Direction: {pred_dir_label}</p>",
            unsafe_allow_html=True,
        )

        labels = [DIRECTION_LABELS[i] for i in range(3)]
        fig_prob = go.Figure(
            go.Bar(
                x=labels,
                y=pred_proba * 100,
                marker_color=["#FF4B4B", "#F0A500", "#21C55D"],
                text=[f"{p:.1f}%" for p in pred_proba * 100],
                textposition="outside",
            )
        )
        fig_prob.update_layout(
            title="Direction Probability (%)",
            template="plotly_dark",
            yaxis=dict(range=[0, 110]),
            height=320,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Feature importance
        st.subheader("Top-10 Feature Importances (Regression Model)")
        fi = pd.DataFrame(
            {"Feature": FEATURE_COLS, "Importance": reg_model.feature_importances_}
        ).sort_values("Importance", ascending=False).head(10)
        fig_fi = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            template="plotly_dark", color="Importance", color_continuous_scale="Reds",
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=380, showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: Performance
# ─────────────────────────────────────────────
elif page == "📈 Performance":
    st.title("📈 Model Performance")

    st.subheader("Evaluation Metrics")
    perf_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "R²", "F1 weighted", "Accuracy"],
            "Value": [
                f"{metrics['rmse']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['r2']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['accuracy']:.4f}",
            ],
            "Task": ["Regression", "Regression", "Regression", "Classification", "Classification"],
        }
    )
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Feature Importance — Top 20")
    tab_reg, tab_clf = st.tabs(["Regression (XGBoost)", "Classification (XGBoost)"])

    with tab_reg:
        fi_reg = pd.DataFrame(
            {"Feature": FEATURE_COLS, "Importance": reg_model.feature_importances_}
        ).sort_values("Importance", ascending=False).head(20)
        fig_r = px.bar(
            fi_reg, x="Importance", y="Feature", orientation="h",
            template="plotly_dark", color="Importance", color_continuous_scale="Reds",
        )
        fig_r.update_layout(yaxis=dict(autorange="reversed"), height=550, showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

    with tab_clf:
        fi_clf = pd.DataFrame(
            {"Feature": FEATURE_COLS, "Importance": clf_model.feature_importances_}
        ).sort_values("Importance", ascending=False).head(20)
        fig_c = px.bar(
            fi_clf, x="Importance", y="Feature", orientation="h",
            template="plotly_dark", color="Importance", color_continuous_scale="Blues",
        )
        fig_c.update_layout(yaxis=dict(autorange="reversed"), height=550, showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.subheader("Distribution of Predictions vs Actual (sample)")
    if df_feat is not None:
        df_sample = df_feat.dropna(subset=FEATURE_COLS + ["target_regression"]).sample(
            min(2000, len(df_feat)), random_state=42
        )
        X_s = scaler.transform(df_sample[FEATURE_COLS])
        y_pred = reg_model.predict(X_s)
        y_true = df_sample["target_regression"].values
        fig_scatter = px.scatter(
            x=y_true, y=y_pred, opacity=0.4,
            labels={"x": "Actual Close", "y": "Predicted Close"},
            template="plotly_dark", title="Actual vs Predicted Close Price",
        )
        min_v, max_v = float(y_true.min()), float(y_true.max())
        fig_scatter.add_trace(
            go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                       name="Perfect Fit", line=dict(color="red", dash="dash"))
        )
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Run `python make_pipeline.py` and commit `data/processed/features.parquet` to see this chart.")

    st.markdown("---")
    with st.expander("⚠️ Limitations"):
        st.markdown(
            """
- Models are trained on historical OHLCV data only — no news, sentiment, or macro inputs.
- Chronological train/test split (80/20) — no walk-forward validation.
- Direction F1 ~0.45 is only marginally above a majority-class baseline.
- Do **not** use this for live trading without further validation.
            """
        )
