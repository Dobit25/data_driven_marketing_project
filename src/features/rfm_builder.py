"""
rfm_builder.py — RFM Aggregation Engine
========================================
Aggregates transaction-level data into household-level RFM features
for Customer Lifetime Value modeling.

Constraints Enforced:
    1. Entity Resolution: ALL aggregations use `household_key` as the
       groupby key. Output is validated to have exactly one row per household.
    2. Monetary Calculation: Computes BOTH Gross_Sales and Net_Sales.
       - Gross_Sales = SUM(SALES_VALUE)
       - Net_Sales   = SUM(SALES_VALUE + RETAIL_DISC + COUPON_DISC + COUPON_MATCH_DISC)
       Note: Discount columns store NEGATIVE values (e.g., -0.60), so adding
       them effectively subtracts the discount amount.
    3. Memory Optimization: This module produces a compact RFM table (~2500 rows)
       from 2.5M transaction rows. The caller should merge with demographics
       ONLY AFTER aggregation — never join raw transactions with demographics.

Design Decision:
    RFM for probabilistic CLV models (BG/NBD) uses a specific definition:
    - Recency (T - last purchase): weeks between last purchase and analysis end
    - Frequency: number of REPEAT purchases (total purchases - 1)
    - Monetary: average net revenue PER transaction (excluding first purchase)
    - T: customer "age" in weeks (analysis_end - first purchase)

    We also compute extended features (avg basket size, store diversity, etc.)
    for supervised ML approaches (XGBoost/LightGBM).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class RFMBuilder:
    """Build RFM (Recency-Frequency-Monetary) features from transaction data.

    All computations are performed at the `household_key` level to ensure
    correct entity resolution (Constraint 1).

    Parameters
    ----------
    config : dict
        Configuration dictionary with ``rfm`` section and ``splitting`` section.

    Example
    -------
    >>> builder = RFMBuilder(config)
    >>> rfm = builder.compute_rfm(transactions_df)
    >>> print(rfm.shape)  # (n_households, feature_columns)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.entity_key: str = config["rfm"]["entity_key"]  # "household_key"

    def compute_rfm(
        self,
        transactions: pd.DataFrame,
        analysis_end_week: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute RFM features aggregated by household_key.

        This is the CORE aggregation step. It reduces ~2.5M transaction rows
        into ~2500 household-level rows, which can then be safely merged
        with demographics (Constraint 3: Memory Optimization).

        Parameters
        ----------
        transactions : pd.DataFrame
            Transaction-level data with columns: household_key, BASKET_ID,
            WEEK_NO, SALES_VALUE, RETAIL_DISC, COUPON_DISC, COUPON_MATCH_DISC,
            QUANTITY, STORE_ID, PRODUCT_ID.
        analysis_end_week : int, optional
            The last week in the analysis window. Defaults to max WEEK_NO
            in the data.

        Returns
        -------
        pd.DataFrame
            Household-level RFM DataFrame with one row per household_key.
            Columns: household_key, Recency, Frequency, T, Gross_Sales,
            Net_Sales, avg_basket_size, avg_transaction_value,
            distinct_stores, tenure_weeks, total_baskets, coupon_usage_rate.
        """
        key = self.entity_key
        logger.info(f"Computing RFM features grouped by '{key}' ...")

        if analysis_end_week is None:
            analysis_end_week = int(transactions["WEEK_NO"].max())
            logger.info(f"  → analysis_end_week auto-set to {analysis_end_week}")

        # ------------------------------------------------------------------
        # Step 1: Compute Net_Sales per transaction line
        # ------------------------------------------------------------------
        # Constraint 2: Monetary Calculation
        # Gross = SALES_VALUE (what customer paid at POS)
        # Net   = SALES_VALUE + RETAIL_DISC + COUPON_DISC + COUPON_MATCH_DISC
        #         Discounts are stored as NEGATIVE, so adding them = subtracting
        #         e.g., SALES_VALUE=1.39, RETAIL_DISC=-0.60 → Net=0.79
        transactions = transactions.copy()
        transactions["_net_line"] = (
            transactions["SALES_VALUE"]
            + transactions["RETAIL_DISC"]
            + transactions["COUPON_DISC"]
            + transactions["COUPON_MATCH_DISC"]
        )

        # Flag: did this transaction line use a coupon?
        transactions["_has_coupon"] = (
            (transactions["COUPON_DISC"] < 0)
            | (transactions["COUPON_MATCH_DISC"] < 0)
        ).astype(np.int8)

        # ------------------------------------------------------------------
        # Step 2: Basket-level aggregation (intermediate step)
        # ------------------------------------------------------------------
        # First aggregate to basket level to compute per-basket metrics
        basket_agg = (
            transactions
            .groupby([key, "BASKET_ID", "WEEK_NO", "STORE_ID"], observed=True)
            .agg(
                basket_gross=("SALES_VALUE", "sum"),
                basket_net=("_net_line", "sum"),
                basket_items=("QUANTITY", "sum"),
                basket_has_coupon=("_has_coupon", "max"),
            )
            .reset_index()
        )

        # ------------------------------------------------------------------
        # Step 3: Household-level aggregation (final RFM)
        # ------------------------------------------------------------------
        hh_agg = (
            basket_agg
            .groupby(key, observed=True)
            .agg(
                # Recency & Tenure
                first_purchase_week=("WEEK_NO", "min"),
                last_purchase_week=("WEEK_NO", "max"),

                # Frequency: total distinct baskets (shopping trips)
                total_baskets=("BASKET_ID", "nunique"),

                # Monetary
                Gross_Sales=("basket_gross", "sum"),
                Net_Sales=("basket_net", "sum"),

                # Extended features
                total_items=("basket_items", "sum"),
                avg_basket_size=("basket_items", "mean"),
                avg_transaction_value=("basket_net", "mean"),
                distinct_stores=("STORE_ID", "nunique"),

                # Coupon usage
                baskets_with_coupon=("basket_has_coupon", "sum"),
            )
            .reset_index()
        )

        # ------------------------------------------------------------------
        # Step 4: Compute derived RFM columns
        # ------------------------------------------------------------------
        # T = customer "age" in weeks (from first purchase to analysis end)
        hh_agg["T"] = analysis_end_week - hh_agg["first_purchase_week"]

        # Recency = weeks since last purchase (relative to analysis end)
        # Lower recency = more recent purchase = more active customer
        hh_agg["Recency"] = analysis_end_week - hh_agg["last_purchase_week"]

        # Frequency for BG/NBD: number of REPEAT purchases (total - 1)
        # BG/NBD requires frequency ≥ 0 (customers with only 1 purchase have freq=0)
        hh_agg["Frequency"] = hh_agg["total_baskets"] - 1

        # Tenure in weeks
        hh_agg["tenure_weeks"] = (
            hh_agg["last_purchase_week"] - hh_agg["first_purchase_week"]
        )

        # Coupon usage rate: fraction of baskets using coupons
        hh_agg["coupon_usage_rate"] = (
            hh_agg["baskets_with_coupon"] / hh_agg["total_baskets"]
        ).fillna(0).astype(np.float32)

        # Average monetary per transaction (for Gamma-Gamma model)
        # Gamma-Gamma requires avg monetary for REPEAT customers only
        # For customers with frequency=0, set to 0 (they'll be excluded from GG)
        hh_agg["avg_monetary"] = np.where(
            hh_agg["Frequency"] > 0,
            hh_agg["Net_Sales"] / hh_agg["total_baskets"],
            0,
        )

        # ------------------------------------------------------------------
        # Step 5: Select and order final columns
        # ------------------------------------------------------------------
        output_cols = [
            key,
            "Recency",
            "Frequency",
            "T",
            "Gross_Sales",
            "Net_Sales",
            "avg_monetary",
            "total_baskets",
            "avg_basket_size",
            "avg_transaction_value",
            "distinct_stores",
            "tenure_weeks",
            "coupon_usage_rate",
            "first_purchase_week",
            "last_purchase_week",
        ]
        rfm = hh_agg[output_cols].copy()

        # ------------------------------------------------------------------
        # Step 6: Validate — Constraint 1 (Entity Resolution)
        # ------------------------------------------------------------------
        self._validate_entity_resolution(rfm)

        # Round floats for cleaner output
        float_cols = rfm.select_dtypes(include=[np.floating]).columns
        rfm[float_cols] = rfm[float_cols].round(4)

        logger.info(
            f"  → RFM table: {len(rfm):,} households × {len(rfm.columns)} features"
        )
        logger.info(
            f"    Frequency: min={rfm['Frequency'].min()}, "
            f"median={rfm['Frequency'].median():.0f}, "
            f"max={rfm['Frequency'].max()}"
        )
        logger.info(
            f"    Gross_Sales: total=${rfm['Gross_Sales'].sum():,.2f}, "
            f"avg=${rfm['Gross_Sales'].mean():,.2f}"
        )

        return rfm

    def _validate_entity_resolution(self, rfm: pd.DataFrame) -> None:
        """Ensure exactly one row per household_key (Constraint 1).

        Raises
        ------
        ValueError
            If duplicate household_keys are found in the RFM output.
        """
        key = self.entity_key
        duplicates = rfm[key].duplicated().sum()
        if duplicates > 0:
            raise ValueError(
                f"Entity Resolution FAILED: {duplicates} duplicate "
                f"'{key}' entries found in RFM table. "
                f"This violates Constraint 1 — all aggregations must "
                f"produce exactly one row per household."
            )
        logger.info(f"  ✓ Entity resolution validated: {len(rfm)} unique households")

    @staticmethod
    def compute_rfm_summary(rfm: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive statistics for the RFM table.

        Useful for EDA reports and understanding customer distribution.

        Parameters
        ----------
        rfm : pd.DataFrame
            Output of ``compute_rfm()``.

        Returns
        -------
        pd.DataFrame
            Descriptive statistics (count, mean, std, min, quartiles, max).
        """
        cols_to_describe = [
            "Recency", "Frequency", "T", "Gross_Sales", "Net_Sales",
            "avg_monetary", "total_baskets", "avg_basket_size",
            "avg_transaction_value", "distinct_stores", "tenure_weeks",
            "coupon_usage_rate",
        ]
        existing = [c for c in cols_to_describe if c in rfm.columns]
        return rfm[existing].describe().T.round(2)
