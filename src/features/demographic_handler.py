"""
demographic_handler.py — Missing Demographics Handler
=====================================================
Handles the merge of RFM features with demographic data and addresses
the missing demographics problem (only 801/2500 households have profiles).

Constraint 4 Resolution:
    Three-pronged approach:
    (a) has_demographics: binary flag indicating data availability
    (b) KNN imputation: for ordinal demographics, using RFM features as
        the similarity metric (neighbors with similar buying behavior
        likely share demographics)
    (c) Mode imputation: for purely categorical demographics (e.g.,
        marital status, homeowner type)

Design Decision:
    The merge happens AFTER RFM aggregation (Constraint 3: Memory).
    The RFM table has ~2500 rows, so the merge is cheap. We NEVER join
    the raw 2.5M-row transaction table with demographics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


class DemographicHandler:
    """Merge and impute demographic features for CLV modeling.

    Parameters
    ----------
    config : dict
        Configuration dictionary with ``demographics`` section defining
        imputation strategy, column types (ordinal vs categorical),
        and KNN parameters.

    Example
    -------
    >>> handler = DemographicHandler(config)
    >>> merged = handler.merge_demographics(rfm_df, demographics_df)
    >>> result = handler.handle_missing(merged)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        demo_cfg = config["demographics"]
        self.strategy: str = demo_cfg["missing_strategy"]
        self.imputation_cfg: Dict = demo_cfg.get("imputation", {})
        self.columns_cfg: Dict = demo_cfg.get("columns", {})

    def merge_demographics(
        self,
        rfm_df: pd.DataFrame,
        demographics_df: pd.DataFrame,
        entity_key: str = "household_key",
    ) -> pd.DataFrame:
        """Left-join RFM table with demographic table and add has_demographics flag.

        This is a SAFE operation because:
        - rfm_df has ~2500 rows (already aggregated)
        - demographics_df has 801 rows
        - Result is ~2500 rows (Constraint 3: Memory Optimization)

        Parameters
        ----------
        rfm_df : pd.DataFrame
            Household-level RFM features (output of RFMBuilder.compute_rfm).
        demographics_df : pd.DataFrame
            Demographic data (hh_demographic.csv loaded via DataLoader).
        entity_key : str
            Join key, default "household_key".

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with all RFM columns + demographic columns +
            has_demographics flag.
        """
        logger.info("Merging RFM with demographics ...")
        logger.info(
            f"  → RFM: {len(rfm_df):,} households | "
            f"Demographics: {len(demographics_df):,} households"
        )

        # Step 1: Add has_demographics flag BEFORE merge
        hh_with_demo = set(demographics_df[entity_key].unique())
        rfm_df = rfm_df.copy()

        # Step 2: Left join — keep ALL households, fill NaN for missing demos
        merged = rfm_df.merge(
            demographics_df,
            on=entity_key,
            how="left",
            validate="1:1",  # Ensure no duplicates from either side
        )

        # Step 3: Create the flag (Constraint 4a)
        merged["has_demographics"] = (
            merged[entity_key].isin(hh_with_demo).astype(np.int8)
        )

        n_with = merged["has_demographics"].sum()
        n_without = len(merged) - n_with
        coverage = n_with / len(merged) * 100

        logger.info(
            f"  → Merged: {len(merged):,} rows | "
            f"With demographics: {n_with:,} ({coverage:.1f}%) | "
            f"Without: {n_without:,} ({100 - coverage:.1f}%)"
        )

        return merged

    def handle_missing(
        self,
        df: pd.DataFrame,
        entity_key: str = "household_key",
    ) -> pd.DataFrame:
        """Apply imputation strategy for missing demographic values.

        Strategy is configured in config.yaml under ``demographics.missing_strategy``:
        - "flag_and_impute": KNN for ordinal, mode for categorical (recommended)
        - "flag_only": just keep the has_demographics flag, leave NaN as-is
        - "drop_missing": remove households without demographics

        Parameters
        ----------
        df : pd.DataFrame
            Merged RFM + demographics DataFrame (output of merge_demographics).
        entity_key : str
            Household key column name.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing demographics handled according to strategy.
        """
        logger.info(f"Handling missing demographics (strategy: '{self.strategy}') ...")

        if self.strategy == "flag_only":
            logger.info("  → Strategy 'flag_only': keeping NaN as-is with flag.")
            return df

        elif self.strategy == "drop_missing":
            before = len(df)
            df = df[df["has_demographics"] == 1].copy()
            logger.info(
                f"  → Strategy 'drop_missing': {before} → {len(df)} rows "
                f"(dropped {before - len(df)} without demographics)"
            )
            return df

        elif self.strategy == "flag_and_impute":
            return self._impute_missing(df, entity_key)

        else:
            logger.warning(
                f"  Unknown strategy '{self.strategy}', falling back to 'flag_only'"
            )
            return df

    def _impute_missing(
        self,
        df: pd.DataFrame,
        entity_key: str,
    ) -> pd.DataFrame:
        """Impute missing demographics using KNN (ordinal) and Mode (categorical).

        Ordinal Imputation (Constraint 4b):
            Demographics like AGE_DESC, INCOME_DESC have a natural order.
            We encode them as integers, then use KNN imputation with RFM
            features as the distance metric. Intuition: households with
            similar buying behavior likely share demographic characteristics.

        Categorical Imputation (Constraint 4c):
            Demographics like MARITAL_STATUS_CODE, HOMEOWNER_DESC have no
            natural order. We fill with the most frequent value (mode).

        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame with NaN in demographic columns.
        entity_key : str
            Household key column name.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed demographics.
        """
        df = df.copy()

        # ------------------------------------------------------------------
        # Part A: Ordinal imputation via KNN
        # ------------------------------------------------------------------
        ordinal_cfg = self.columns_cfg.get("ordinal", {})
        if ordinal_cfg:
            logger.info("  → Imputing ordinal columns with KNN ...")
            df = self._impute_ordinal_knn(df, ordinal_cfg, entity_key)

        # ------------------------------------------------------------------
        # Part B: Categorical imputation via Mode
        # ------------------------------------------------------------------
        cat_cols = self.columns_cfg.get("categorical", [])
        if cat_cols:
            logger.info("  → Imputing categorical columns with Mode ...")
            df = self._impute_categorical_mode(df, cat_cols)

        return df

    def _impute_ordinal_knn(
        self,
        df: pd.DataFrame,
        ordinal_cfg: Dict[str, Dict],
        entity_key: str,
    ) -> pd.DataFrame:
        """KNN imputation for ordinal demographic columns.

        Uses RFM features (Recency, Frequency, avg_monetary, T) as the
        distance metric to find similar households.
        """
        n_neighbors = self.imputation_cfg.get("knn_neighbors", 5)

        # RFM features to use as KNN distance features
        rfm_features = ["Recency", "Frequency", "T", "avg_monetary"]
        rfm_features = [c for c in rfm_features if c in df.columns]

        # Encode ordinal columns as integers
        ordinal_cols = list(ordinal_cfg.keys())
        existing_ordinal = [c for c in ordinal_cols if c in df.columns]

        if not existing_ordinal:
            logger.warning("    No ordinal demographic columns found in data.")
            return df

        # Build ordinal encoders
        encoders = {}
        for col in existing_ordinal:
            order = ordinal_cfg[col].get("order", [])
            if order:
                # Include any values not in the predefined order
                actual_values = df[col].dropna().unique().tolist()
                extra = [v for v in actual_values if v not in order]
                full_order = order + extra

                enc = OrdinalEncoder(
                    categories=[full_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                )
                # Fit and transform non-null values
                mask = df[col].notna()
                if mask.any():
                    encoded = enc.fit_transform(df.loc[mask, [col]])
                    df.loc[mask, f"_{col}_enc"] = encoded.ravel()
                else:
                    df[f"_{col}_enc"] = np.nan
                encoders[col] = (enc, full_order)
            else:
                # No order specified, skip
                df[f"_{col}_enc"] = pd.Categorical(df[col]).codes
                df.loc[df[col].isna(), f"_{col}_enc"] = np.nan

        # Build feature matrix for KNN: [RFM features, encoded ordinal cols]
        encoded_cols = [f"_{c}_enc" for c in existing_ordinal]
        knn_features = rfm_features + encoded_cols

        # Normalize RFM features for KNN distance calculation
        knn_matrix = df[knn_features].copy()
        for col in rfm_features:
            col_std = knn_matrix[col].std()
            if col_std > 0:
                knn_matrix[col] = (
                    (knn_matrix[col] - knn_matrix[col].mean()) / col_std
                )

        # Apply KNN imputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        imputed = imputer.fit_transform(knn_matrix)
        imputed_df = pd.DataFrame(imputed, columns=knn_features, index=df.index)

        # Decode back to original categories
        for col in existing_ordinal:
            enc_col = f"_{col}_enc"
            imputed_values = imputed_df[enc_col].round().astype(int)

            if col in encoders:
                _, full_order = encoders[col]
                # Map encoded integers back to category strings
                imputed_values = imputed_values.clip(0, len(full_order) - 1)
                df[col] = imputed_values.map(
                    lambda x, order=full_order: order[int(x)]
                )
            else:
                df[col] = imputed_values

            # Clean up temp column
            if enc_col in df.columns:
                df.drop(columns=[enc_col], inplace=True)

            n_imputed = df["has_demographics"].eq(0).sum()
            logger.info(f"    {col}: imputed {n_imputed} missing values via KNN")

        return df

    def _impute_categorical_mode(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Fill missing categorical demographics with the mode (most frequent value).

        This is appropriate for columns without a natural ordering
        (e.g., MARITAL_STATUS_CODE, HOMEOWNER_DESC).
        """
        for col in categorical_cols:
            if col not in df.columns:
                logger.warning(f"    Column '{col}' not found, skipping.")
                continue

            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue

            mode_value = df[col].mode()
            if len(mode_value) > 0:
                mode_value = mode_value.iloc[0]
                df[col] = df[col].fillna(mode_value)
                logger.info(
                    f"    {col}: filled {n_missing} missing with mode='{mode_value}'"
                )
            else:
                logger.warning(f"    {col}: no mode found, leaving as NaN.")

        return df
