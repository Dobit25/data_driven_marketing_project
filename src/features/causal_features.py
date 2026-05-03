"""
causal_features.py — Promotional Feature Builder
==================================================
Extracts household-level promotion sensitivity features from causal_data.csv.

Features computed per household:
    - pct_display:  % of items purchased while on in-store display
    - pct_mailer:   % of items purchased while featured in mailer/flyer
    - promo_sensitivity: combined promotion sensitivity score

The causal_data.csv is 695 MB, so this module uses chunked processing
to avoid out-of-memory errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class CausalFeatureBuilder:
    """Build household-level promotional features from causal_data.

    Parameters
    ----------
    config : dict
        Configuration dictionary from config.yaml.
    chunksize : int
        Number of rows per chunk when reading causal_data.csv (default: 500_000).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        chunksize: int = 500_000,
    ) -> None:
        self.config = config
        self.chunksize = chunksize
        self.causal_path = Path(config["data"]["files"]["causal_data"])

    def build_promo_features(
        self,
        transactions: pd.DataFrame,
        analysis_end_week: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute promotion sensitivity features per household.

        Joins causal_data with transactions on (PRODUCT_ID, STORE_ID, WEEK_NO),
        then aggregates per household_key to compute the fraction of purchases
        made while the product was on display or in a mailer.

        Parameters
        ----------
        transactions : pd.DataFrame
            Transaction data with columns: household_key, PRODUCT_ID,
            STORE_ID, WEEK_NO, QUANTITY.
        analysis_end_week : int, optional
            Filter transactions up to this week.

        Returns
        -------
        pd.DataFrame
            One row per household_key with columns:
            pct_display, pct_mailer, promo_sensitivity.
        """
        logger.info("Building promotional features from causal_data ...")

        if not self.causal_path.exists():
            logger.warning(f"  causal_data not found at {self.causal_path}. "
                           f"Returning empty features.")
            return self._empty_features(transactions)

        # Filter transactions to analysis window
        txn = transactions.copy()
        if analysis_end_week is not None:
            txn = txn[txn["WEEK_NO"] <= analysis_end_week]

        # Keep only join keys + household info
        txn_slim = txn[
            ["household_key", "PRODUCT_ID", "STORE_ID", "WEEK_NO", "QUANTITY"]
        ].copy()

        # Process causal_data in chunks to avoid OOM
        logger.info(
            f"  Reading causal_data in chunks of {self.chunksize:,} rows ..."
        )

        # We'll build a lookup: (PRODUCT_ID, STORE_ID, WEEK_NO) → (display, mailer)
        # Since causal_data is huge, we chunk-read and only keep rows that
        # match our transactions.
        join_keys = set(
            zip(
                txn_slim["PRODUCT_ID"].values,
                txn_slim["STORE_ID"].values,
                txn_slim["WEEK_NO"].values,
            )
        )

        causal_records = []
        chunk_reader = pd.read_csv(
            self.causal_path,
            dtype={
                "PRODUCT_ID": np.int32,
                "STORE_ID": np.int16,
                "WEEK_NO": np.int8,
                "display": "category",
                "mailer": "category",
            },
            chunksize=self.chunksize,
        )

        for i, chunk in enumerate(chunk_reader):
            # Filter chunk to only rows matching our transactions
            chunk_keys = set(
                zip(
                    chunk["PRODUCT_ID"].values,
                    chunk["STORE_ID"].values,
                    chunk["WEEK_NO"].values,
                )
            )
            matching = chunk_keys & join_keys
            if matching:
                # Keep matching rows
                mask = pd.Series(
                    [
                        (p, s, w) in matching
                        for p, s, w in zip(
                            chunk["PRODUCT_ID"], chunk["STORE_ID"], chunk["WEEK_NO"]
                        )
                    ],
                    index=chunk.index,
                )
                causal_records.append(chunk[mask])

            if (i + 1) % 10 == 0:
                logger.info(f"    Processed {(i+1) * self.chunksize:,} rows ...")

        if not causal_records:
            logger.warning("  No matching causal data found. Returning defaults.")
            return self._empty_features(transactions)

        causal_df = pd.concat(causal_records, ignore_index=True)
        logger.info(f"  Matched causal records: {len(causal_df):,}")

        # Convert display/mailer to binary (0 = no promotion, anything else = yes)
        causal_df["has_display"] = (
            causal_df["display"].astype(str) != "0"
        ).astype(np.int8)
        causal_df["has_mailer"] = (
            causal_df["mailer"].astype(str) != "0"
        ).astype(np.int8)

        # Merge with transactions
        merged = txn_slim.merge(
            causal_df[["PRODUCT_ID", "STORE_ID", "WEEK_NO",
                       "has_display", "has_mailer"]],
            on=["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
            how="left",
        )

        # Fill NaN (no causal data = no promotion)
        merged["has_display"] = merged["has_display"].fillna(0).astype(np.int8)
        merged["has_mailer"] = merged["has_mailer"].fillna(0).astype(np.int8)

        # Aggregate per household
        promo_features = (
            merged.groupby("household_key", observed=True)
            .agg(
                total_items=("QUANTITY", "sum"),
                items_on_display=("has_display", "sum"),
                items_in_mailer=("has_mailer", "sum"),
            )
            .reset_index()
        )

        promo_features["pct_display"] = np.where(
            promo_features["total_items"] > 0,
            promo_features["items_on_display"] / promo_features["total_items"],
            0,
        )
        promo_features["pct_mailer"] = np.where(
            promo_features["total_items"] > 0,
            promo_features["items_in_mailer"] / promo_features["total_items"],
            0,
        )
        promo_features["promo_sensitivity"] = (
            promo_features["pct_display"] + promo_features["pct_mailer"]
        ) / 2

        # Select final columns
        result = promo_features[
            ["household_key", "pct_display", "pct_mailer", "promo_sensitivity"]
        ].copy()

        # Round
        for col in ["pct_display", "pct_mailer", "promo_sensitivity"]:
            result[col] = result[col].round(4).astype(np.float32)

        logger.info(
            f"  ✓ Promo features for {len(result):,} households"
        )
        logger.info(
            f"    Avg pct_display: {result['pct_display'].mean():.4f}"
        )
        logger.info(
            f"    Avg pct_mailer:  {result['pct_mailer'].mean():.4f}"
        )

        return result

    @staticmethod
    def _empty_features(transactions: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with zero-filled promo features."""
        hh_keys = transactions["household_key"].unique()
        return pd.DataFrame({
            "household_key": hh_keys,
            "pct_display": np.float32(0),
            "pct_mailer": np.float32(0),
            "promo_sensitivity": np.float32(0),
        })
