"""
time_splitter.py — Temporal Data Splitter
==========================================
Splits transaction data into Calibration (observation) and Holdout
(validation) periods based on WEEK_NO.

Constraint 5 Resolution:
    We use a TIME-BASED split, NOT random train_test_split.
    This is critical for CLV modeling because:
    - BG/NBD models require a clear calibration period to fit parameters
      and a holdout period to validate predictions.
    - Random splitting would break temporal causality — you can't predict
      the future using information from the future.
    - Default cutoff: Week 75 (calibration = weeks 1–75, holdout = weeks 76–102).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from loguru import logger


class TimeSplitter:
    """Split transaction data temporally into calibration and holdout periods.

    Parameters
    ----------
    config : dict
        Configuration dictionary with ``splitting`` section containing
        ``calibration_end_week`` and ``total_weeks``.

    Example
    -------
    >>> splitter = TimeSplitter(config)
    >>> cal_df, holdout_df = splitter.split(transactions_df)
    >>> print(f"Calibration: {len(cal_df):,} txns, Holdout: {len(holdout_df):,} txns")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        split_cfg = config["splitting"]
        self.calibration_end_week: int = split_cfg["calibration_end_week"]
        self.total_weeks: int = split_cfg["total_weeks"]

    def split(
        self,
        transactions: pd.DataFrame,
        cutoff_week: int | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split transactions into calibration and holdout by WEEK_NO.

        Calibration period: WEEK_NO <= cutoff_week
        Holdout period:     WEEK_NO >  cutoff_week

        Parameters
        ----------
        transactions : pd.DataFrame
            Transaction-level data with WEEK_NO column.
        cutoff_week : int, optional
            Week number to split on. Defaults to config value (75).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (calibration_df, holdout_df) — non-overlapping subsets.

        Raises
        ------
        ValueError
            If WEEK_NO column is missing or cutoff is invalid.
        """
        if cutoff_week is None:
            cutoff_week = self.calibration_end_week

        # Validation
        if "WEEK_NO" not in transactions.columns:
            raise ValueError(
                "Column 'WEEK_NO' not found in transactions DataFrame. "
                "Cannot perform temporal split."
            )

        week_min = int(transactions["WEEK_NO"].min())
        week_max = int(transactions["WEEK_NO"].max())

        if cutoff_week < week_min or cutoff_week >= week_max:
            raise ValueError(
                f"cutoff_week={cutoff_week} is out of valid range "
                f"[{week_min}, {week_max - 1}]. Must be between "
                f"min WEEK_NO and max WEEK_NO - 1."
            )

        logger.info(
            f"Splitting transactions at WEEK_NO = {cutoff_week} ..."
        )
        logger.info(
            f"  → Data range: Week {week_min} to Week {week_max} "
            f"({week_max - week_min + 1} weeks total)"
        )

        # ------------------------------------------------------------------
        # Temporal Split — NO RANDOMIZATION (Constraint 5)
        # ------------------------------------------------------------------
        calibration = transactions[transactions["WEEK_NO"] <= cutoff_week].copy()
        holdout = transactions[transactions["WEEK_NO"] > cutoff_week].copy()

        # ------------------------------------------------------------------
        # Validation: No temporal overlap
        # ------------------------------------------------------------------
        self._validate_no_overlap(calibration, holdout)

        # Summary statistics
        cal_weeks = f"Week {week_min}–{cutoff_week}"
        hold_weeks = f"Week {cutoff_week + 1}–{week_max}"

        cal_hh = calibration["household_key"].nunique()
        hold_hh = holdout["household_key"].nunique()
        hold_only = holdout[
            ~holdout["household_key"].isin(calibration["household_key"])
        ]["household_key"].nunique()

        logger.info(
            f"  ✓ Calibration ({cal_weeks}): "
            f"{len(calibration):,} txns | {cal_hh:,} households"
        )
        logger.info(
            f"  ✓ Holdout     ({hold_weeks}): "
            f"{len(holdout):,} txns | {hold_hh:,} households"
        )
        logger.info(
            f"  → {hold_only} households appear ONLY in holdout "
            f"(new customers after calibration)"
        )

        cal_pct = len(calibration) / len(transactions) * 100
        logger.info(
            f"  → Split ratio: {cal_pct:.1f}% calibration / "
            f"{100 - cal_pct:.1f}% holdout"
        )

        return calibration, holdout

    def _validate_no_overlap(
        self,
        calibration: pd.DataFrame,
        holdout: pd.DataFrame,
    ) -> None:
        """Verify that calibration and holdout periods don't overlap.

        Raises
        ------
        ValueError
            If any WEEK_NO appears in both calibration and holdout.
        """
        cal_weeks = set(calibration["WEEK_NO"].unique())
        hold_weeks = set(holdout["WEEK_NO"].unique())
        overlap = cal_weeks & hold_weeks

        if overlap:
            raise ValueError(
                f"Temporal overlap detected! Weeks {sorted(overlap)} appear "
                f"in both calibration and holdout periods. This should never "
                f"happen with a clean week-based split."
            )

    def get_split_info(self) -> Dict[str, Any]:
        """Return a dict summarizing the split configuration.

        Returns
        -------
        dict
            Contains calibration_end_week, calibration_range,
            holdout_range, total_weeks.
        """
        return {
            "calibration_end_week": self.calibration_end_week,
            "calibration_range": f"Week 1 – {self.calibration_end_week}",
            "holdout_range": (
                f"Week {self.calibration_end_week + 1} – {self.total_weeks}"
            ),
            "total_weeks": self.total_weeks,
            "calibration_pct": round(
                self.calibration_end_week / self.total_weeks * 100, 1
            ),
        }
