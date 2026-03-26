"""
predict_model.py
================
Evaluation and diagnostics pipeline: generates forecasts on the holdout set,
computes error metrics (RMSE, MAE, MAPE), runs residual diagnostics
(Ljung-Box, normality), and exports visualizations.

Usage:
    python -m src.models.predict_model configs/config.yaml
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(config: dict):
    """Load a serialized model from disk.

    Args:
        config: Project configuration dictionary.

    Returns:
        Tuple of (model, scaler_or_None).
    """
    active = config["model"]["active"]
    model_dir = Path(config["output"]["model_dir"])

    if active == "lstm":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_dir / "lstm_model.keras")
        with open(model_dir / "lstm_scaler.pkl", "rb") as f:
            data = pickle.load(f)
        return model, data.get("scaler")
    else:
        model_path = model_dir / f"{active}_model.pkl"
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data.get("scaler")


# ==============================================================================
# Forecasting
# ==============================================================================

def generate_forecast(model, test_df: pd.DataFrame, config: dict, scaler=None) -> np.ndarray:
    """Generate forecasts for the test period.

    Args:
        model: Trained model object.
        test_df: Test DataFrame.
        config: Project configuration dictionary.
        scaler: Optional scaler (for LSTM).

    Returns:
        Numpy array of forecasted values.
    """
    active = config["model"]["active"]
    target_col = config["data"]["columns"]["target"]
    n_test = len(test_df)

    if active in ("arima", "sarima"):
        forecast = model.predict(n_periods=n_test) if hasattr(model, "predict") else model.forecast(steps=n_test)
        forecast = np.array(forecast)

    elif active == "prophet":
        dt_col = config["data"]["columns"]["datetime"]
        future = test_df[[dt_col]].rename(columns={dt_col: "ds"})
        pred = model.predict(future)
        forecast = pred["yhat"].values

    elif active == "lstm":
        # Load full series for windowed prediction
        interim_dir = Path(config["data"]["interim_dir"])
        train_df = pd.read_csv(interim_dir / "train.csv")
        full_series = pd.concat([train_df[target_col], test_df[target_col]]).values
        window = config["model"]["lstm"]["input_window"]

        scaled = scaler.transform(full_series.reshape(-1, 1))
        X_test = []
        start = len(train_df) - window
        for i in range(start, start + n_test):
            X_test.append(scaled[i : i + window, 0])
        X_test = np.array(X_test).reshape(-1, window, 1)

        forecast_scaled = model.predict(X_test)
        forecast = scaler.inverse_transform(forecast_scaled).flatten()

    else:
        raise ValueError(f"Unknown model: {active}")

    logger.info("Forecast generated: %d data points", len(forecast))
    return forecast


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, and MAPE.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        Dictionary of metric names to values.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    # Avoid division by zero in MAPE
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    metrics = {"rmse": rmse, "mae": mae, "mape": mape}
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name.upper(), value)
    return metrics


# ==============================================================================
# Residual Diagnostics
# ==============================================================================

def run_residual_diagnostics(actual: np.ndarray, predicted: np.ndarray, config: dict) -> dict:
    """Run residual diagnostic tests.

    Tests:
        - Ljung-Box test for autocorrelation in residuals
        - Shapiro-Wilk test for normality of residuals

    Args:
        actual: Actual values.
        predicted: Predicted values.
        config: Project configuration dictionary.

    Returns:
        Dictionary of diagnostic results.
    """
    residuals = actual - predicted
    diagnostics = {}

    # Ljung-Box test
    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_result["lb_pvalue"].values[0]
    diagnostics["ljung_box"] = {
        "statistic": lb_result["lb_stat"].values[0],
        "p_value": lb_pvalue,
        "autocorrelation_present": lb_pvalue < 0.05,
    }
    logger.info(
        "Ljung-Box test — p-value: %.4f => %s",
        lb_pvalue,
        "Autocorrelation detected" if lb_pvalue < 0.05 else "No significant autocorrelation",
    )

    # Shapiro-Wilk normality test (on a sample if too many points)
    sample = residuals[:5000] if len(residuals) > 5000 else residuals
    sw_stat, sw_p = stats.shapiro(sample)
    diagnostics["shapiro_wilk"] = {
        "statistic": sw_stat,
        "p_value": sw_p,
        "residuals_normal": sw_p > 0.05,
    }
    logger.info(
        "Shapiro-Wilk test — p-value: %.4f => %s",
        sw_p,
        "Residuals are normal" if sw_p > 0.05 else "Residuals are NOT normal",
    )

    return diagnostics


# ==============================================================================
# Visualization
# ==============================================================================

def plot_forecast(actual: np.ndarray, predicted: np.ndarray, config: dict) -> None:
    """Plot forecast vs actual values and save to reports/figures/.

    Args:
        actual: Actual values.
        predicted: Predicted values.
        config: Project configuration dictionary.
    """
    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Forecast vs Actual
    axes[0].plot(actual, label="Actual", color="#2196F3", linewidth=1.2)
    axes[0].plot(predicted, label="Forecast", color="#FF5722", linewidth=1.2, linestyle="--")
    axes[0].set_title("Forecast vs Actual — Concurrent Players", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Concurrent Players")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = actual - predicted
    axes[1].bar(range(len(residuals)), residuals, color="#9C27B0", alpha=0.6, width=1.0)
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals", fontsize=12)
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Time Step")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = figures_dir / "forecast_vs_actual.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Forecast plot saved to %s", save_path)

    # Residual distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = figures_dir / "residual_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Residual distribution plot saved to %s", save_path)


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main(config_path: str) -> None:
    """Run the evaluation and diagnostics pipeline.

    Steps:
        1. Load configuration and trained model
        2. Load test data
        3. Generate forecast
        4. Compute evaluation metrics (RMSE, MAE, MAPE)
        5. Run residual diagnostics
        6. Export visualizations

    Args:
        config_path: Path to YAML configuration file.
    """
    config = load_config(config_path)

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("STAGE 4: Evaluation & Diagnostics")
    logger.info("=" * 60)

    # Load model
    model, scaler = load_model(config)

    # Load test data
    dt_col = config["data"]["columns"]["datetime"]
    target_col = config["data"]["columns"]["target"]
    test_df = pd.read_csv(
        Path(config["data"]["interim_dir"]) / "test.csv",
        parse_dates=[dt_col],
    )
    actual = test_df[target_col].values
    logger.info("Test data loaded: %d rows", len(actual))

    # Generate forecast
    forecast = generate_forecast(model, test_df, config, scaler)

    # Ensure equal length
    min_len = min(len(actual), len(forecast))
    actual, forecast = actual[:min_len], forecast[:min_len]

    # Metrics
    logger.info("--- Evaluation Metrics ---")
    metrics = compute_metrics(actual, forecast)

    # Diagnostics
    if config["evaluation"].get("residual_diagnostics", True):
        logger.info("--- Residual Diagnostics ---")
        diagnostics = run_residual_diagnostics(actual, forecast, config)

    # Visualization
    if config["evaluation"].get("forecast_plot", True):
        plot_forecast(actual, forecast, config)

    logger.info("Evaluation & diagnostics pipeline completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.models.predict_model <config_path>")
        sys.exit(1)
    main(sys.argv[1])
