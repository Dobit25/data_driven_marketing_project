"""
data_loader.py — DunnhumbyDataLoader
=====================================
OOP data loading module for the Dunnhumby "The Complete Journey" dataset.

Responsibilities:
    - Load each CSV with optimized dtypes to minimize memory usage
    - Provide schema exploration (dtypes, missing values, shape)
    - Document column meanings and table relationships in docstrings

Design Decisions:
    - Uses dtype downcasting (int64 → int32, float64 → float32) to reduce
      memory footprint by ~50%, critical for the 2.5M-row transaction table.
    - causal_data.csv (695 MB) is loaded with aggressive dtype optimization
      and only the columns needed for CLV features.
    - Each load method is independent — caller loads only what they need.

Table Relationships (Dunnhumby ERD):
    ┌─────────────┐      ┌──────────────────┐      ┌─────────┐
    │  household   │──1:N─│  transaction_data │──N:1─│ product │
    │  (2,500)     │      │  (2.5M rows)      │      │         │
    └──────┬───────┘      └──────────────────┘      └────┬────┘
           │                                              │
           │ 1:0..1                                       │ N:1
           ▼                                              ▼
    ┌──────────────┐      ┌───────────────┐      ┌───────────┐
    │hh_demographic│      │  causal_data  │──N:1─│  product   │
    │  (801 rows)  │      │  (store×week) │      │           │
    └──────────────┘      └───────────────┘      └───────────┘
           │
           │ 1:N
           ▼
    ┌──────────────┐      ┌───────────────┐
    │campaign_table│──N:1─│ campaign_desc │
    └──────┬───────┘      └───────────────┘
           │ 1:N
           ▼
    ┌──────────────┐      ┌───────────────┐
    │coupon_redempt│──N:1─│    coupon      │
    └──────────────┘      └───────────────┘
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class DunnhumbyDataLoader:
    """Load and explore the Dunnhumby 'The Complete Journey' dataset.

    This class provides memory-efficient loading of all 8 Dunnhumby CSV files
    with automatic dtype optimization and schema validation.

    Parameters
    ----------
    config : dict
        Configuration dictionary (loaded from config.yaml).
        Must contain ``data.files`` mapping with paths to all CSV files.

    Example
    -------
    >>> import yaml
    >>> config = yaml.safe_load(open("configs/config.yaml"))
    >>> loader = DunnhumbyDataLoader(config)
    >>> transactions = loader.load_transactions()
    >>> demographics = loader.load_demographics()
    >>> loader.explore(transactions, "transaction_data")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.file_paths: Dict[str, str] = config["data"]["files"]
        self._validate_paths()

    # ------------------------------------------------------------------
    # Path Validation
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        """Check that all configured CSV files exist on disk."""
        missing = []
        for name, path in self.file_paths.items():
            if not os.path.isfile(path):
                missing.append(f"  - {name}: {path}")
        if missing:
            msg = "Missing data files:\n" + "\n".join(missing)
            logger.warning(msg)

    # ------------------------------------------------------------------
    # Core Load Methods
    # ------------------------------------------------------------------
    def load_transactions(self) -> pd.DataFrame:
        """Load transaction_data.csv with optimized dtypes.

        Columns
        -------
        household_key : int32
            Unique household identifier — PRIMARY ENTITY for CLV analysis.
        BASKET_ID : int64
            Unique shopping trip / basket identifier.
        DAY : int16
            Day number relative to study start (1–711).
        PRODUCT_ID : int32
            Unique product identifier (joins to product.csv).
        QUANTITY : int16
            Number of units purchased for this line item.
        SALES_VALUE : float32
            Dollar amount the customer actually paid (after POS discounts).
        STORE_ID : int16
            Store where the transaction occurred.
        RETAIL_DISC : float32
            Retailer loyalty/store discount (NEGATIVE value, e.g., -0.60).
        TRANS_TIME : int16
            Transaction time in HHMM format (e.g., 1631 = 4:31 PM).
        WEEK_NO : int8
            Week number of the transaction (1–102).
        COUPON_DISC : float32
            Manufacturer coupon discount (NEGATIVE value).
        COUPON_MATCH_DISC : float32
            Additional discount when coupon matches retailer promo (NEGATIVE).

        Returns
        -------
        pd.DataFrame
            Transaction-level DataFrame (~2.5M rows).
        """
        dtypes = {
            "household_key": np.int32,
            "BASKET_ID": np.int64,
            "DAY": np.int16,
            "PRODUCT_ID": np.int32,
            "QUANTITY": np.int16,
            "SALES_VALUE": np.float32,
            "STORE_ID": np.int16,
            "RETAIL_DISC": np.float32,
            "TRANS_TIME": np.int16,
            "WEEK_NO": np.int8,
            "COUPON_DISC": np.float32,
            "COUPON_MATCH_DISC": np.float32,
        }
        path = self.file_paths["transaction_data"]
        logger.info(f"Loading transaction_data from {path} ...")
        df = pd.read_csv(path, dtype=dtypes)
        logger.info(
            f"  → Loaded {len(df):,} rows × {len(df.columns)} cols "
            f"({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)"
        )
        return df

    def load_demographics(self) -> pd.DataFrame:
        """Load hh_demographic.csv — demographic profiles for 801/2500 households.

        Columns
        -------
        household_key : int32
            Household identifier (joins to transaction_data).
        AGE_DESC : str (category)
            Age range of primary shopper: "19-24", "25-34", "35-44",
            "45-54", "55-64", "65+".
        MARITAL_STATUS_CODE : str (category)
            Marital status: "A" (married), "B" (single).
        INCOME_DESC : str (category)
            Annual household income range:
            "Under 15K", "15-24K", "25-34K", "35-49K", "50-74K",
            "75-99K", "100-124K", "125-149K", "150-174K", "175-199K",
            "200-249K", "250K+".
        HOMEOWNER_DESC : str (category)
            Home ownership: "Homeowner", "Renter", "Probable Homeowner",
            "Probable Renter", "Unknown".
        HH_COMP_DESC : str (category)
            Household composition: "1 Adult Kids", "2 Adults No Kids",
            "2 Adults Kids", "Single Female", "Single Male", etc.
        HOUSEHOLD_SIZE_DESC : str (category)
            Number of people in household: "1", "2", "3", "4", "5+".
        KID_CATEGORY_DESC : str (category)
            Presence of children: "None/Unknown", "1", "2", "3+".

        Returns
        -------
        pd.DataFrame
            Demographic DataFrame (801 rows).
        """
        path = self.file_paths["hh_demographic"]
        logger.info(f"Loading hh_demographic from {path} ...")
        df = pd.read_csv(path, dtype={"household_key": np.int32})

        # Convert string columns to pandas Categorical for memory efficiency
        cat_cols = [
            "AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC",
            "HOMEOWNER_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
            "KID_CATEGORY_DESC",
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        logger.info(
            f"  → Loaded {len(df):,} rows × {len(df.columns)} cols "
            f"({df.memory_usage(deep=True).sum() / 1e3:.1f} KB)"
        )
        return df

    def load_campaigns(self) -> pd.DataFrame:
        """Load campaign_table.csv — household-campaign participation records.

        Columns
        -------
        DESCRIPTION : str (category)
            Campaign type classification: "TypeA", "TypeB", "TypeC".
        household_key : int32
            Household that received the campaign.
        CAMPAIGN : int16
            Campaign identifier (joins to campaign_desc.csv).

        Returns
        -------
        pd.DataFrame
        """
        path = self.file_paths["campaign_table"]
        logger.info(f"Loading campaign_table from {path} ...")
        df = pd.read_csv(
            path,
            dtype={"household_key": np.int32, "CAMPAIGN": np.int16},
        )
        df["DESCRIPTION"] = df["DESCRIPTION"].astype("category")
        logger.info(f"  → Loaded {len(df):,} rows")
        return df

    def load_campaign_desc(self) -> pd.DataFrame:
        """Load campaign_desc.csv — campaign metadata (type, duration).

        Columns
        -------
        DESCRIPTION : str (category)
            Campaign type classification.
        CAMPAIGN : int16
            Unique campaign identifier.
        START_DAY : int16
            Day number when campaign begins.
        END_DAY : int16
            Day number when campaign ends.

        Returns
        -------
        pd.DataFrame
        """
        path = self.file_paths["campaign_desc"]
        logger.info(f"Loading campaign_desc from {path} ...")
        df = pd.read_csv(
            path,
            dtype={
                "CAMPAIGN": np.int16,
                "START_DAY": np.int16,
                "END_DAY": np.int16,
            },
        )
        df["DESCRIPTION"] = df["DESCRIPTION"].astype("category")
        logger.info(f"  → Loaded {len(df):,} rows")
        return df

    def load_coupons(self) -> pd.DataFrame:
        """Load coupon.csv — coupon-to-product-to-campaign mapping.

        Columns
        -------
        COUPON_UPC : int64
            Universal Product Code for the coupon itself.
        PRODUCT_ID : int32
            Product this coupon applies to (joins to product.csv).
        CAMPAIGN : int16
            Campaign this coupon belongs to (joins to campaign_desc.csv).

        Returns
        -------
        pd.DataFrame
        """
        path = self.file_paths["coupon"]
        logger.info(f"Loading coupon from {path} ...")
        df = pd.read_csv(
            path,
            dtype={
                "COUPON_UPC": np.int64,
                "PRODUCT_ID": np.int32,
                "CAMPAIGN": np.int16,
            },
        )
        logger.info(f"  → Loaded {len(df):,} rows")
        return df

    def load_coupon_redemptions(self) -> pd.DataFrame:
        """Load coupon_redempt.csv — actual coupon redemption events.

        Columns
        -------
        household_key : int32
            Household that redeemed the coupon.
        DAY : int16
            Day number when redemption occurred.
        COUPON_UPC : int64
            Coupon that was redeemed (joins to coupon.csv).
        CAMPAIGN : int16
            Campaign the redeemed coupon belongs to.

        Returns
        -------
        pd.DataFrame
        """
        path = self.file_paths["coupon_redempt"]
        logger.info(f"Loading coupon_redempt from {path} ...")
        df = pd.read_csv(
            path,
            dtype={
                "household_key": np.int32,
                "DAY": np.int16,
                "COUPON_UPC": np.int64,
                "CAMPAIGN": np.int16,
            },
        )
        logger.info(f"  → Loaded {len(df):,} rows")
        return df

    def load_products(self) -> pd.DataFrame:
        """Load product.csv — product master data with category hierarchy.

        Columns
        -------
        PRODUCT_ID : int32
            Unique product identifier (joins to transaction_data, coupon).
        MANUFACTURER : int32
            Manufacturer code.
        DEPARTMENT : str (category)
            Top-level department: "GROCERY", "DRUG GM", "PRODUCE",
            "MEAT-PCKGD", "PASTRY", "DELI", etc.
        BRAND : str (category)
            Brand type: "National" (branded) or "Private" (store brand).
        COMMODITY_DESC : str (category)
            Product category (e.g., "SOFT DRINKS", "FLUID MILK PRODUCTS").
        SUB_COMMODITY_DESC : str (category)
            Sub-category (e.g., "REGULAR COLA", "SS WHOLE MILK").
        CURR_SIZE_OF_PRODUCT : str
            Package size (e.g., "2 LT", "12 OZ", "1 GAL").

        Returns
        -------
        pd.DataFrame
        """
        path = self.file_paths["product"]
        logger.info(f"Loading product from {path} ...")
        df = pd.read_csv(
            path,
            dtype={"PRODUCT_ID": np.int32, "MANUFACTURER": np.int32},
        )
        for col in ["DEPARTMENT", "BRAND", "COMMODITY_DESC", "SUB_COMMODITY_DESC"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        logger.info(f"  → Loaded {len(df):,} rows")
        return df

    def load_causal_data(
        self,
        usecols: Optional[list] = None,
        chunksize: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load causal_data.csv — store-level promotional/display information.

        This is the LARGEST file (~695 MB). Uses aggressive dtype optimization.
        Consider loading with specific columns or in chunks for memory safety.

        Columns
        -------
        PRODUCT_ID : int32
            Product identifier.
        STORE_ID : int16
            Store identifier.
        WEEK_NO : int8
            Week number (1–102).
        display : str (category)
            In-store display placement code.
            "0" = no display, other codes indicate location (e.g., endcap).
        mailer : str (category)
            Mailer/flyer inclusion code.
            "0" = not in mailer, other codes indicate placement.

        Parameters
        ----------
        usecols : list, optional
            Subset of columns to load (saves memory).
        chunksize : int, optional
            If set, returns a TextFileReader for chunked processing.

        Returns
        -------
        pd.DataFrame or TextFileReader
        """
        dtypes = {
            "PRODUCT_ID": np.int32,
            "STORE_ID": np.int16,
            "WEEK_NO": np.int8,
            "display": "category",
            "mailer": "category",
        }
        path = self.file_paths["causal_data"]
        logger.info(f"Loading causal_data from {path} (large file, ~695 MB) ...")

        if chunksize:
            logger.info(f"  → Using chunked reading (chunksize={chunksize:,})")
            return pd.read_csv(
                path, dtype=dtypes, usecols=usecols, chunksize=chunksize
            )

        df = pd.read_csv(path, dtype=dtypes, usecols=usecols)
        logger.info(
            f"  → Loaded {len(df):,} rows × {len(df.columns)} cols "
            f"({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)"
        )
        return df

    # ------------------------------------------------------------------
    # Exploration & Profiling
    # ------------------------------------------------------------------
    @staticmethod
    def explore(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Print and return a comprehensive schema report for a DataFrame.

        Outputs to console:
            - Shape, memory usage
            - Column dtypes
            - Missing values count and percentage
            - Basic descriptive statistics
            - Sample rows

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to explore.
        table_name : str
            Human-readable name for console output.

        Returns
        -------
        dict
            Summary dict with keys: shape, dtypes, missing, memory_mb.
        """
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"  [DATA EXPLORATION] {table_name}")
        print(f"{separator}")

        # Shape
        print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Memory
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"  Memory: {mem_mb:.2f} MB")

        # Dtypes
        print(f"\n  {'Column':<30} {'Dtype':<15} {'Non-Null':>10} {'Missing':>10} {'Miss %':>8}")
        print(f"  {'-'*28}   {'-'*13}   {'-'*8}   {'-'*8}   {'-'*6}")
        for col in df.columns:
            non_null = df[col].notna().sum()
            missing = df[col].isna().sum()
            miss_pct = (missing / len(df)) * 100 if len(df) > 0 else 0
            print(
                f"  {col:<30} {str(df[col].dtype):<15} {non_null:>10,} "
                f"{missing:>10,} {miss_pct:>7.2f}%"
            )

        # Basic stats for numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            print(f"\n  [Numeric Summary]")
            stats = df[num_cols].describe().T
            for col_name in stats.index:
                row = stats.loc[col_name]
                print(
                    f"    {col_name:<28} "
                    f"min={row['min']:>12,.2f}  "
                    f"mean={row['mean']:>12,.2f}  "
                    f"max={row['max']:>12,.2f}  "
                    f"std={row['std']:>12,.2f}"
                )

        # Categorical columns — unique counts
        cat_cols = df.select_dtypes(include=["category", "object"]).columns
        if len(cat_cols) > 0:
            print(f"\n  [Categorical Summary]")
            for col in cat_cols:
                n_unique = df[col].nunique()
                top = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                print(f"    {col:<28} {n_unique:>5} unique   top: {top}")

        # Sample
        print(f"\n  [First 3 rows]")
        print(df.head(3).to_string(index=False, max_colwidth=30))

        print(f"\n{separator}\n")

        return {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isna().sum().to_dict(),
            "memory_mb": mem_mb,
        }

    def explore_all(self) -> Dict[str, Dict[str, Any]]:
        """Load and explore all core tables. Returns summary dict.

        This method loads transaction_data, hh_demographic, campaign_table,
        campaign_desc, coupon, coupon_redempt, and product.
        causal_data is excluded by default due to its size (695 MB).

        Returns
        -------
        dict
            Mapping of table_name → exploration summary dict.
        """
        summaries = {}

        tables = [
            ("transaction_data", self.load_transactions),
            ("hh_demographic", self.load_demographics),
            ("campaign_table", self.load_campaigns),
            ("campaign_desc", self.load_campaign_desc),
            ("coupon", self.load_coupons),
            ("coupon_redempt", self.load_coupon_redemptions),
            ("product", self.load_products),
        ]

        for name, load_fn in tables:
            try:
                df = load_fn()
                summaries[name] = self.explore(df, name)
                # Free memory after exploration (except transaction_data
                # which the caller likely needs)
                if name != "transaction_data":
                    del df
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                summaries[name] = {"error": str(e)}

        return summaries
