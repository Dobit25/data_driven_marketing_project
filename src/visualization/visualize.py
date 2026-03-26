"""
visualize.py
============
Reusable plotting utilities for EDA, ACF/PACF, forecast, and diagnostics.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

logger = logging.getLogger(__name__)


def plot_time_series(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    title: str = "Time Series",
    save_path: str | None = None,
) -> None:
    """Plot a time series line chart.

    Args:
        df: DataFrame with datetime and target columns.
        datetime_col: Name of the datetime column.
        target_col: Name of the target column.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[datetime_col], df[target_col], linewidth=0.8, color="#1976D2")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot: %s", save_path)
    plt.close()


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 48,
    save_path: str | None = None,
) -> None:
    """Plot ACF and PACF side by side.

    Args:
        series: Time series data.
        lags: Number of lags to display.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[0].set_title("ACF", fontsize=12, fontweight="bold")
    axes[1].set_title("PACF", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved ACF/PACF plot: %s", save_path)
    plt.close()


def plot_seasonal_decomposition(
    series: pd.Series,
    period: int = 24,
    save_path: str | None = None,
) -> None:
    """Plot seasonal decomposition (trend, seasonal, residual).

    Args:
        series: Time series data.
        period: Seasonal period.
        save_path: Optional path to save the figure.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series.dropna(), model="additive", period=period)
    fig = result.plot()
    fig.set_size_inches(14, 8)
    fig.suptitle("Seasonal Decomposition", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved decomposition plot: %s", save_path)
    plt.close()
