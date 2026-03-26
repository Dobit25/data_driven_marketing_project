"""
train_model.py
==============
Model training pipeline: supports ARIMA, SARIMA, Prophet, and LSTM.
Trains the model specified in the config and serializes it to disk.

Usage:
    python -m src.models.train_model configs/config.yaml
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================================
# ARIMA / SARIMA Training
# ==============================================================================

def train_arima(train_series: pd.Series, config: dict):
    """Train an ARIMA model.

    Uses pmdarima's auto_arima if `auto_order` is True, otherwise fits
    with the specified (p, d, q) order.

    Args:
        train_series: Training time series.
        config: Model configuration dictionary.

    Returns:
        Fitted ARIMA model.
    """
    arima_cfg = config["model"]["arima"]

    if arima_cfg.get("auto_order", False):
        import pmdarima as pm
        model = pm.auto_arima(
            train_series,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            trace=True,
        )
        logger.info("Auto ARIMA order: %s", model.order)
    else:
        from statsmodels.tsa.arima.model import ARIMA
        order = tuple(arima_cfg["order"])
        model = ARIMA(train_series, order=order).fit()
        logger.info("ARIMA fitted with order %s", order)

    return model


def train_sarima(train_series: pd.Series, config: dict):
    """Train a SARIMA model.

    Uses pmdarima's auto_arima with seasonal=True if `auto_order` is True.

    Args:
        train_series: Training time series.
        config: Model configuration dictionary.

    Returns:
        Fitted SARIMA model.
    """
    sarima_cfg = config["model"]["sarima"]

    if sarima_cfg.get("auto_order", False):
        import pmdarima as pm
        seasonal_period = sarima_cfg["seasonal_order"][-1]
        model = pm.auto_arima(
            train_series,
            seasonal=True,
            m=seasonal_period,
            stepwise=True,
            suppress_warnings=True,
            trace=True,
        )
        logger.info("Auto SARIMA order: %s, seasonal: %s", model.order, model.seasonal_order)
    else:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = tuple(sarima_cfg["order"])
        seasonal_order = tuple(sarima_cfg["seasonal_order"])
        model = SARIMAX(
            train_series, order=order, seasonal_order=seasonal_order
        ).fit(disp=False)
        logger.info("SARIMA fitted — order: %s, seasonal: %s", order, seasonal_order)

    return model


# ==============================================================================
# Prophet Training
# ==============================================================================

def train_prophet(train_df: pd.DataFrame, config: dict):
    """Train a Facebook Prophet model.

    Args:
        train_df: Training DataFrame with 'ds' (datetime) and 'y' (target) columns.
        config: Model configuration dictionary.

    Returns:
        Fitted Prophet model.
    """
    from prophet import Prophet

    prophet_cfg = config["model"]["prophet"]

    model = Prophet(
        changepoint_prior_scale=prophet_cfg["changepoint_prior_scale"],
        seasonality_prior_scale=prophet_cfg["seasonality_prior_scale"],
        yearly_seasonality=prophet_cfg["yearly_seasonality"],
        weekly_seasonality=prophet_cfg["weekly_seasonality"],
        daily_seasonality=prophet_cfg["daily_seasonality"],
    )

    # Prophet expects columns named 'ds' and 'y'
    dt_col = config["data"]["columns"]["datetime"]
    target_col = config["data"]["columns"]["target"]
    prophet_df = train_df[[dt_col, target_col]].rename(
        columns={dt_col: "ds", target_col: "y"}
    )

    model.fit(prophet_df)
    logger.info("Prophet model fitted.")
    return model


# ==============================================================================
# LSTM Training
# ==============================================================================

def train_lstm(train_series: np.ndarray, config: dict):
    """Train an LSTM model using TensorFlow/Keras.

    Args:
        train_series: Normalized training series as numpy array.
        config: Model configuration dictionary.

    Returns:
        Tuple of (fitted model, scaler).
    """
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    lstm_cfg = config["model"]["lstm"]
    window = lstm_cfg["input_window"]

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train_series.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model
    model = tf.keras.Sequential()
    for i in range(lstm_cfg["num_layers"]):
        return_seq = i < lstm_cfg["num_layers"] - 1
        model.add(tf.keras.layers.LSTM(
            lstm_cfg["hidden_units"],
            return_sequences=return_seq,
            input_shape=(window, 1) if i == 0 else None,
        ))
        model.add(tf.keras.layers.Dropout(lstm_cfg["dropout"]))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lstm_cfg["learning_rate"]),
        loss="mse",
    )

    # Train
    model.fit(
        X, y,
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=lstm_cfg["early_stopping_patience"],
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    logger.info("LSTM model trained — %d epochs max, window=%d", lstm_cfg["epochs"], window)
    return model, scaler


# ==============================================================================
# Model Persistence
# ==============================================================================

def save_model(model, config: dict, model_name: str, scaler=None) -> Path:
    """Serialize and save the trained model to disk.

    Args:
        model: Trained model object.
        config: Project configuration dictionary.
        model_name: Name for the saved model file.
        scaler: Optional scaler object (for LSTM).

    Returns:
        Path to the saved model file.
    """
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    logger.info("Model saved to %s", model_path)
    return model_path


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main(config_path: str) -> None:
    """Run the model training pipeline.

    Steps:
        1. Load configuration
        2. Load training data
        3. Train the active model (ARIMA, SARIMA, Prophet, or LSTM)
        4. Save trained model

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
    logger.info("STAGE 3: Model Training")
    logger.info("=" * 60)

    # Load training data
    interim_dir = Path(config["data"]["interim_dir"])
    dt_col = config["data"]["columns"]["datetime"]
    target_col = config["data"]["columns"]["target"]

    train_df = pd.read_csv(interim_dir / "train.csv", parse_dates=[dt_col])
    train_series = train_df[target_col]

    logger.info("Training data loaded: %d rows", len(train_series))

    # Train based on active model
    active_model = config["model"]["active"]
    logger.info("Active model: %s", active_model)

    if active_model == "arima":
        model = train_arima(train_series, config)
        save_model(model, config, "arima_model")

    elif active_model == "sarima":
        model = train_sarima(train_series, config)
        save_model(model, config, "sarima_model")

    elif active_model == "prophet":
        model = train_prophet(train_df, config)
        save_model(model, config, "prophet_model")

    elif active_model == "lstm":
        model, scaler = train_lstm(train_series.values, config)
        # Save Keras model separately
        model_dir = Path(config["output"]["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / "lstm_model.keras")
        save_model(None, config, "lstm_scaler", scaler=scaler)
        logger.info("LSTM model and scaler saved.")

    else:
        raise ValueError(f"Unknown model type: {active_model}")

    logger.info("Model training pipeline completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.models.train_model <config_path>")
        sys.exit(1)
    main(sys.argv[1])
