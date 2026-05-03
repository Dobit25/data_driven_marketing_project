"""
evaluator.py — CLV Model Evaluation Module
=============================================
Evaluates CLV model predictions against actual holdout-period spending.

Metrics:
    - MAE  (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)

Also generates a calibration plot (predicted vs actual scatter).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger


class CLVEvaluator:
    """Evaluate CLV predictions against holdout actuals.

    Parameters
    ----------
    config : dict
        Configuration dictionary from config.yaml.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.reports_dir = Path(config["output"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path(config["output"]["figures_dir"])
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, float] = {}

    def evaluate(
        self,
        calibration_df: pd.DataFrame,
        holdout_rfm: pd.DataFrame,
        prediction_col: str = "predicted_clv_6m",
        actual_col: str = "Net_Sales",
    ) -> Dict[str, float]:
        """Compute evaluation metrics by comparing predictions vs actuals.

        Parameters
        ----------
        calibration_df : pd.DataFrame
            Calibration-period data with 'household_key' and prediction column.
        holdout_rfm : pd.DataFrame
            Holdout-period RFM with 'household_key' and actual spending column.
        prediction_col : str
            Column name in calibration_df containing CLV predictions.
        actual_col : str
            Column name in holdout_rfm containing actual spending.

        Returns
        -------
        dict
            Dictionary with 'mae', 'rmse', 'mape' keys.
        """
        logger.info("Evaluating CLV predictions against holdout actuals ...")

        # Join predictions with actuals on household_key
        pred = calibration_df[["household_key", prediction_col]].copy()
        actual = holdout_rfm[["household_key", actual_col]].copy()
        actual = actual.rename(columns={actual_col: "actual_clv"})

        merged = pred.merge(actual, on="household_key", how="inner")
        merged = merged.dropna(subset=[prediction_col, "actual_clv"])

        y_true = merged["actual_clv"].values
        y_pred = merged[prediction_col].values

        logger.info(f"  Matched households: {len(merged):,}")

        # MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # RMSE
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # MAPE (avoid division by zero)
        nonzero_mask = y_true > 0
        if nonzero_mask.sum() > 0:
            mape = float(
                np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                               / y_true[nonzero_mask])) * 100
            )
        else:
            mape = float("inf")

        self.metrics = {"mae": mae, "rmse": rmse, "mape": mape}

        logger.info(f"  MAE  = ${mae:,.2f}")
        logger.info(f"  RMSE = ${rmse:,.2f}")
        logger.info(f"  MAPE = {mape:.2f}%")

        # Save report
        self._save_report(merged, prediction_col)

        return self.metrics

    def calibration_plot(
        self,
        calibration_df: pd.DataFrame,
        holdout_rfm: pd.DataFrame,
        prediction_col: str = "predicted_clv_6m",
        actual_col: str = "Net_Sales",
    ) -> str:
        """Generate calibration scatter plot: predicted vs actual CLV.

        Parameters
        ----------
        calibration_df : pd.DataFrame
            DataFrame with predictions.
        holdout_rfm : pd.DataFrame
            DataFrame with actuals.
        prediction_col : str
            Prediction column name.
        actual_col : str
            Actual spending column name.

        Returns
        -------
        str
            Path to saved plot.
        """
        import matplotlib.pyplot as plt

        logger.info("Generating calibration plot ...")

        pred = calibration_df[["household_key", prediction_col]].copy()
        actual = holdout_rfm[["household_key", actual_col]].rename(
            columns={actual_col: "actual_clv"}
        )
        merged = pred.merge(actual, on="household_key", how="inner")
        merged = merged.dropna()

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(
            merged["actual_clv"],
            merged[prediction_col],
            alpha=0.3,
            s=10,
            color="#4E79A7",
            label="Customers",
        )

        # 45-degree line (perfect prediction)
        max_val = max(merged["actual_clv"].max(), merged[prediction_col].max())
        ax.plot(
            [0, max_val], [0, max_val],
            "r--", lw=2, alpha=0.8, label="Perfect prediction"
        )

        ax.set_xlabel("Actual Holdout CLV ($)", fontsize=12)
        ax.set_ylabel("Predicted CLV ($)", fontsize=12)
        ax.set_title("CLV Calibration Plot: Predicted vs Actual", fontsize=14)
        ax.legend(fontsize=10)

        # Add metrics annotation
        if self.metrics:
            text = (
                f"MAE  = ${self.metrics['mae']:,.2f}\n"
                f"RMSE = ${self.metrics['rmse']:,.2f}\n"
                f"MAPE = {self.metrics['mape']:.1f}%"
            )
            ax.text(
                0.05, 0.95, text,
                transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        plt.tight_layout()
        plot_path = self.figures_dir / "calibration_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"  Saved: {plot_path}")
        return str(plot_path)

    def _save_report(
        self,
        merged: pd.DataFrame,
        prediction_col: str,
    ) -> None:
        """Save evaluation report to text file."""
        report_path = self.reports_dir / "evaluation_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  CLV Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"  Households evaluated: {len(merged):,}\n")
            f.write(f"  Prediction column:    {prediction_col}\n\n")
            f.write(f"  MAE  = ${self.metrics['mae']:,.2f}\n")
            f.write(f"  RMSE = ${self.metrics['rmse']:,.2f}\n")
            f.write(f"  MAPE = {self.metrics['mape']:.2f}%\n\n")
            f.write("-" * 60 + "\n")
            f.write("  Prediction Distribution\n")
            f.write("-" * 60 + "\n")
            f.write(merged[[prediction_col, "actual_clv"]].describe().to_string())
            f.write("\n")

        logger.info(f"  Saved report: {report_path}")
