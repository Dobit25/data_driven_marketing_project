from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "data_clean.csv"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 3 core tables with optimized dtypes."""
    print("=" * 70)
    print("  [STEP 0] Loading Raw Data")
    print("=" * 70)

    txn_dtypes = {
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
    print("  Loading transaction_data.csv ...")
    txn = pd.read_csv(RAW_DIR / "transaction_data.csv", dtype=txn_dtypes)
    print(f"    -> {len(txn):,} rows x {len(txn.columns)} cols "
          f"({txn.memory_usage(deep=True).sum() / 1e6:.1f} MB)")

    print("  Loading hh_demographic.csv ...")
    demo = pd.read_csv(RAW_DIR / "hh_demographic.csv",
                        dtype={"household_key": np.int32})
    print(f"    -> {len(demo):,} rows x {len(demo.columns)} cols")

    print("  Loading product.csv ...")
    prod = pd.read_csv(RAW_DIR / "product.csv",
                        dtype={"PRODUCT_ID": np.int32, "MANUFACTURER": np.int32})
    print(f"    -> {len(prod):,} rows x {len(prod.columns)} cols")

    return txn, demo, prod

def merge_tables(
    txn: pd.DataFrame,
    demo: pd.DataFrame,
    prod: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join transaction_data with product and hh_demographic.

    Join keys:
        txn ←-> prod  on PRODUCT_ID
        txn ←-> demo  on household_key

    We use LEFT JOIN so we keep ALL transactions, even if:
        - a product has no metadata (unlikely but safe)
        - a household has no demographic data (1,699/2,500 missing)
    """
    print("\n" + "=" * 70)
    print("  [STEP 1] Merging Tables")
    print("=" * 70)

    prod_cols = ["PRODUCT_ID", "DEPARTMENT", "BRAND", "COMMODITY_DESC"]
    merged = txn.merge(prod[prod_cols], on="PRODUCT_ID", how="left")
    print(f"  After merge with product: {len(merged):,} rows")
    merged = merged.merge(demo, on="household_key", how="left")
    print(f"  After merge with demographics: {len(merged):,} rows")

    assert len(merged) == len(txn), (
        f"Row count mismatch after merge! Expected {len(txn):,}, got {len(merged):,}. "
        f"Check for duplicate keys in product or demographic tables."
    )
    print(f"  [OK] Row count preserved: {len(merged):,}")

    return merged


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data quality issues:
        a) Remove rows with QUANTITY <= 0 (returns, coupon adjustments)
        b) Remove rows with SALES_VALUE <= 0 (returns, void transactions)
        c) Cap extreme SALES_VALUE outliers (> 99.9th percentile)
        d) Ensure discount columns are non-positive (they store NEGATIVE values)
    """
    print("\n" + "=" * 70)
    print("  [STEP 2] Cleaning Anomalies")
    print("=" * 70)

    before = len(df)

    zero_qty = (df["QUANTITY"] <= 0).sum()
    print(f"  Rows with QUANTITY <= 0: {zero_qty:,} ({zero_qty/before*100:.2f}%)")
    df = df[df["QUANTITY"] > 0].copy()

    neg_sales = (df["SALES_VALUE"] <= 0).sum()
    print(f"  Rows with SALES_VALUE <= 0: {neg_sales:,} ({neg_sales/before*100:.2f}%)")
    df = df[df["SALES_VALUE"] > 0].copy()

    p999 = df["SALES_VALUE"].quantile(0.999)
    outliers = (df["SALES_VALUE"] > p999).sum()
    print(f"  SALES_VALUE > 99.9th pctl (${p999:.2f}): {outliers:,} rows -> capped")
    df["SALES_VALUE"] = df["SALES_VALUE"].clip(upper=p999)

    df["Net_Sales"] = (
        df["SALES_VALUE"]
        + df["RETAIL_DISC"]
        + df["COUPON_DISC"]
        + df["COUPON_MATCH_DISC"]
    )

    neg_net = (df["Net_Sales"] < 0).sum()
    print(f"  Rows with Net_Sales < 0 (over-discounted): {neg_net:,} -> floored to 0")
    df["Net_Sales"] = df["Net_Sales"].clip(lower=0)

    after = len(df)
    removed = before - after
    print(f"\n  Summary: {before:,} -> {after:,} rows (removed {removed:,}, "
          f"{removed/before*100:.2f}%)")

    return df

def handle_missing_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only 801/2,500 households have demographic data.
    Strategy: fill NaN with 'Unknown' and add a binary flag.

    Columns handled:
        AGE_DESC, MARITAL_STATUS_CODE, INCOME_DESC,
        HOMEOWNER_DESC, HH_COMP_DESC, HOUSEHOLD_SIZE_DESC,
        KID_CATEGORY_DESC
    """
    print("\n" + "=" * 70)
    print("  [STEP 3] Handling Missing Demographics")
    print("=" * 70)

    demo_cols = [
        "AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC",
        "HOMEOWNER_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
        "KID_CATEGORY_DESC",
    ]

    existing_demo_cols = [c for c in demo_cols if c in df.columns]
    df["has_demographics"] = df[existing_demo_cols].notna().any(axis=1).astype(np.int8)

    n_with = df.groupby("household_key")["has_demographics"].first()
    pct = n_with.mean() * 100
    print(f"  Households with demographics: {n_with.sum():,}/{len(n_with):,} "
          f"({pct:.1f}%)")

    for col in existing_demo_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna("Unknown")
            print(f"    {col}: filled {n_missing:,} NaN -> 'Unknown'")

    remaining_na = df[existing_demo_cols].isna().sum().sum()
    print(f"\n  [OK] Remaining NaN in demographic columns: {remaining_na}")

    return df

def aggregate_weekly_household(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to (household_key, WEEK_NO) level.

    Output columns:
        - household_key, WEEK_NO          : group keys
        - total_baskets                   : distinct shopping trips
        - total_items                     : sum of QUANTITY
        - Gross_Sales                     : sum of SALES_VALUE
        - Net_Sales                       : sum of Net_Sales (after discounts)
        - total_discount                  : sum of all discount columns
        - avg_item_price                  : Gross_Sales / total_items
        - n_products                      : distinct PRODUCT_ID count
        - n_stores                        : distinct STORE_ID count
        - n_departments                   : distinct DEPARTMENT count
        - pct_private_brand               : % of items bought as Private label
        - has_coupon                      : 1 if any coupon used, 0 otherwise
        - Demographics (carried forward from household level)
        - has_demographics                : flag
    """
    print("\n" + "=" * 70)
    print("  [STEP 4] Aggregating to Weekly/Household Level")
    print("=" * 70)

    group_keys = ["household_key", "WEEK_NO"]

    demo_cols = [
        "AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC",
        "HOMEOWNER_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
        "KID_CATEGORY_DESC", "has_demographics",
    ]
    existing_demo = [c for c in demo_cols if c in df.columns]

    df["_is_private"] = (df["BRAND"] == "Private").astype(np.int8)
    df["_has_coupon"] = (
        (df["COUPON_DISC"] < 0) | (df["COUPON_MATCH_DISC"] < 0)
    ).astype(np.int8)
    df["_total_disc"] = (
        df["RETAIL_DISC"] + df["COUPON_DISC"] + df["COUPON_MATCH_DISC"]
    )

    print(f"  Grouping by {group_keys} ...")

    agg = (
        df
        .groupby(group_keys, observed=True)
        .agg(
            total_baskets=("BASKET_ID", "nunique"),
            total_items=("QUANTITY", "sum"),
            Gross_Sales=("SALES_VALUE", "sum"),
            Net_Sales=("Net_Sales", "sum"),
            total_discount=("_total_disc", "sum"),
            n_products=("PRODUCT_ID", "nunique"),
            n_stores=("STORE_ID", "nunique"),
            n_departments=("DEPARTMENT", "nunique"),
            pct_private_brand=("_is_private", "mean"),
            has_coupon=("_has_coupon", "max"),
        )
        .reset_index()
    )

    agg["avg_item_price"] = np.where(
        agg["total_items"] > 0,
        agg["Gross_Sales"] / agg["total_items"],
        0,
    )

    demo_lookup = (
        df
        .groupby("household_key", observed=True)[existing_demo]
        .first()
        .reset_index()
    )
    agg = agg.merge(demo_lookup, on="household_key", how="left")

    float_cols = agg.select_dtypes(include=[np.floating]).columns
    agg[float_cols] = agg[float_cols].round(4)

    agg = agg.sort_values(["household_key", "WEEK_NO"]).reset_index(drop=True)

    print(f"  -> Aggregated: {len(agg):,} rows "
          f"({agg['household_key'].nunique():,} households x "
          f"up to {agg['WEEK_NO'].nunique()} weeks)")
    print(f"  Columns: {list(agg.columns)}")

    return agg

def save_output(df: pd.DataFrame) -> str:
    """Save the cleaned, aggregated dataset to CSV."""
    print("\n" + "=" * 70)
    print("  [STEP 5] Saving Output")
    print("=" * 70)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"  [OK] Saved: {OUTPUT_FILE}")
    print(f"    Size: {size_mb:.2f} MB")
    print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    return str(OUTPUT_FILE)


def print_summary(df: pd.DataFrame, elapsed: float) -> None:
    """Print a final summary of the cleaned dataset."""
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — Summary Report")
    print("=" * 70)

    print(f"\n  Total time: {elapsed:.1f} seconds")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Shape:  {df.shape[0]:,} rows x {df.shape[1]} columns")

    print(f"\n  --- Key Statistics ---")
    print(f"  Households:       {df['household_key'].nunique():,}")
    print(f"  Weeks:            {df['WEEK_NO'].nunique()}")
    print(f"  Week range:       {df['WEEK_NO'].min()} – {df['WEEK_NO'].max()}")
    print(f"  Total Gross Sales: ${df['Gross_Sales'].sum():,.2f}")
    print(f"  Total Net Sales:   ${df['Net_Sales'].sum():,.2f}")
    print(f"  Total Discounts:   ${df['total_discount'].sum():,.2f}")
    print(f"  Avg baskets/week/HH: {df['total_baskets'].mean():.2f}")

    if "has_demographics" in df.columns:
        hh_demo = df.groupby("household_key")["has_demographics"].first()
        pct = hh_demo.mean() * 100
        print(f"  Demographics coverage: {hh_demo.sum()}/{len(hh_demo)} "
              f"({pct:.1f}%)")

    print(f"\n  --- Columns in output ---")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        na = df[col].isna().sum()
        print(f"    {i:2d}. {col:<25} {str(dtype):<12} NaN={na}")

    print("\n" + "=" * 70)

def run_data_prep() -> pd.DataFrame:
    """Execute the full ETL pipeline."""
    start = time.time()

    print("\n" + "=" * 70)
    print("  >>> DATA ENGINEERING & ETL PIPELINE")
    print("  >>> Author: Hieu (Core Coder)")
    print("=" * 70)

    txn, demo, prod = load_raw_data()

    merged = merge_tables(txn, demo, prod)
    del txn

    cleaned = clean_anomalies(merged)
    del merged

    cleaned = handle_missing_demographics(cleaned)

    agg = aggregate_weekly_household(cleaned)
    del cleaned

    save_output(agg)

    elapsed = time.time() - start
    print_summary(agg, elapsed)

    return agg

if __name__ == "__main__":
    run_data_prep()
