"""
run_preprocessing.py — CLV Preprocessing Pipeline Orchestrator
================================================================
Main entry point for the end-to-end preprocessing pipeline.

Usage:
    python -m src.pipeline.run_preprocessing configs/config.yaml

Pipeline Steps:
    1. Load all Dunnhumby datasets (with dtype optimization)
    2. Explore / profile each table (schema, missing values, stats)
    3. Split transactions into Calibration / Holdout periods (Constraint 5)
    4. Build RFM features for calibration period (Constraints 1, 2, 3)
    5. Build RFM features for holdout period (for evaluation)
    6. Merge with demographics and handle missing data (Constraint 4)
    7. Generate EDA visualizations → reports/figures/
    8. Save processed datasets → data/processed/ and data/interim/

Constraints Summary:
    1. Entity Resolution:    groupby household_key ONLY
    2. Monetary Calculation:  Gross_Sales and Net_Sales (transparent formula)
    3. Memory Optimization:   Aggregate BEFORE merge
    4. Missing Demographics:  flag + KNN/Mode imputation
    5. Time-based Splitting:  WEEK_NO cutoff, NO random split
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from loguru import logger

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DunnhumbyDataLoader
from src.features.rfm_builder import RFMBuilder
from src.features.demographic_handler import DemographicHandler
from src.features.time_splitter import TimeSplitter
from src.visualization.eda_plots import EDAPlotter


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure loguru logger based on config.yaml settings."""
    log_cfg = config.get("logging", {})
    log_level = log_cfg.get("level", "INFO")
    log_file = log_cfg.get("log_file", "logs/pipeline.log")
    log_to_console = log_cfg.get("log_to_console", True)

    # Remove default logger
    logger.remove()

    # Console output
    if log_to_console:
        logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level:<8}</level> | "
                "<cyan>{message}</cyan>"
            ),
        )

    # File output
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def run_pipeline(config_path: str) -> None:
    """Execute the full CLV preprocessing pipeline.

    Parameters
    ----------
    config_path : str
        Path to the config.yaml file.
    """
    # ==================================================================
    # Step 0: Load Configuration
    # ==================================================================
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(config)

    logger.info("=" * 70)
    logger.info("  >>> CLV PREPROCESSING PIPELINE")
    logger.info(f"  Project: {config['project']['name']}")
    logger.info(f"  Config:  {config_path}")
    logger.info("=" * 70)

    start_time = time.time()

    # Ensure output directories exist
    for dir_key in ["interim_dir", "processed_dir"]:
        Path(config["data"][dir_key]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Step 1: Load Datasets
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 1] Loading Dunnhumby Datasets")
    logger.info("=" * 70)

    loader = DunnhumbyDataLoader(config)
    transactions = loader.load_transactions()
    demographics = loader.load_demographics()

    # Load auxiliary tables for exploration (smaller tables)
    campaigns = loader.load_campaigns()
    campaign_desc = loader.load_campaign_desc()
    coupons = loader.load_coupons()
    coupon_redemptions = loader.load_coupon_redemptions()
    products = loader.load_products()

    # ==================================================================
    # Step 2: Explore / Profile All Tables
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 2] Data Exploration & Profiling")
    logger.info("=" * 70)

    loader.explore(transactions, "transaction_data")
    loader.explore(demographics, "hh_demographic")
    loader.explore(campaigns, "campaign_table")
    loader.explore(campaign_desc, "campaign_desc")
    loader.explore(coupons, "coupon")
    loader.explore(coupon_redemptions, "coupon_redempt")
    loader.explore(products, "product")

    # Print table relationship summary
    _print_relationships(transactions, demographics, campaigns, coupon_redemptions)

    # Free auxiliary tables after exploration (Constraint 3: Memory)
    del campaigns, campaign_desc, coupons, coupon_redemptions, products
    logger.info("  → Freed auxiliary table memory after exploration.")

    # ==================================================================
    # Step 3: Temporal Split (Constraint 5)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 3] Temporal Split -- Calibration / Holdout")
    logger.info("=" * 70)

    splitter = TimeSplitter(config)
    calibration_end_week = config["splitting"]["calibration_end_week"]
    cal_transactions, holdout_transactions = splitter.split(transactions)

    # Log split info
    split_info = splitter.get_split_info()
    for k, v in split_info.items():
        logger.info(f"  {k}: {v}")

    # ==================================================================
    # Step 4: Build RFM — Calibration Period (Constraints 1, 2, 3)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 4] RFM Aggregation -- Calibration Period")
    logger.info("=" * 70)

    builder = RFMBuilder(config)
    # Compute analysis_end_day from DAY column for daily-granularity RFM
    calibration_end_day = int(cal_transactions["DAY"].max())
    logger.info(f"  Calibration end day: {calibration_end_day}")
    rfm_calibration = builder.compute_rfm(
        cal_transactions,
        analysis_end_day=calibration_end_day,
    )

    # Print RFM summary statistics
    rfm_summary = builder.compute_rfm_summary(rfm_calibration)
    print("\n  [RFM Summary Statistics (Calibration)]")
    print(rfm_summary.to_string())
    print()

    # Save RFM calibration to interim
    rfm_cal_path = Path(config["data"]["interim_dir"]) / "rfm_calibration.csv"
    rfm_calibration.to_csv(rfm_cal_path, index=False)
    logger.info(f"  Saved: {rfm_cal_path}")

    # ==================================================================
    # Step 5: Build RFM — Holdout Period (for future evaluation)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 5] RFM Aggregation -- Holdout Period")
    logger.info("=" * 70)

    total_weeks = config["splitting"]["total_weeks"]
    total_end_day = int(holdout_transactions["DAY"].max())
    logger.info(f"  Holdout end day: {total_end_day}")
    rfm_holdout = builder.compute_rfm(
        holdout_transactions,
        analysis_end_day=total_end_day,
    )

    # Save RFM holdout to interim
    rfm_hold_path = Path(config["data"]["interim_dir"]) / "rfm_holdout.csv"
    rfm_holdout.to_csv(rfm_hold_path, index=False)
    logger.info(f"  Saved: {rfm_hold_path}")

    # ==================================================================
    # Step 6: Merge Demographics & Handle Missing (Constraint 4)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 6] Merge Demographics & Impute Missing")
    logger.info("=" * 70)

    handler = DemographicHandler(config)

    # Merge and impute for calibration RFM
    merged_cal = handler.merge_demographics(rfm_calibration, demographics)
    merged_cal = handler.handle_missing(merged_cal)

    # Merge and impute for holdout RFM
    merged_holdout = handler.merge_demographics(rfm_holdout, demographics)
    merged_holdout = handler.handle_missing(merged_holdout)

    # Save final processed datasets
    processed_dir = Path(config["data"]["processed_dir"])

    cal_out_path = processed_dir / "clv_features_calibration.csv"
    merged_cal.to_csv(cal_out_path, index=False)
    logger.info(f"  Saved: {cal_out_path} ({len(merged_cal):,} rows)")

    holdout_out_path = processed_dir / "clv_features_holdout.csv"
    merged_holdout.to_csv(holdout_out_path, index=False)
    logger.info(f"  Saved: {holdout_out_path} ({len(merged_holdout):,} rows)")

    # ==================================================================
    # Step 7: Generate EDA Visualizations
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 7] Generating EDA Visualizations")
    logger.info("=" * 70)

    plotter = EDAPlotter(config)
    plot_paths = plotter.run_all(
        transactions=transactions,
        rfm=rfm_calibration,
        demographics=demographics,
        merged_df=merged_cal,
        calibration_end_week=calibration_end_week,
    )

    # ==================================================================
    # Step 8: Causal / Promotional Features (Issue #15)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 8] Building Promotional Features (causal_data)")
    logger.info("=" * 70)

    from src.features.causal_features import CausalFeatureBuilder

    causal_builder = CausalFeatureBuilder(config)
    promo_features = causal_builder.build_promo_features(
        cal_transactions, analysis_end_week=calibration_end_week,
    )

    # Merge promo features into calibration features
    merged_cal = merged_cal.merge(promo_features, on="household_key", how="left")
    for col in ["pct_display", "pct_mailer", "promo_sensitivity"]:
        if col in merged_cal.columns:
            merged_cal[col] = merged_cal[col].fillna(0)

    # Re-save enriched calibration features
    cal_out_path = processed_dir / "clv_features_calibration.csv"
    merged_cal.to_csv(cal_out_path, index=False)
    logger.info(f"  Re-saved enriched features: {cal_out_path}")

    # ==================================================================
    # Step 9: CLV Modeling (K-Means + BG/NBD + Gamma-Gamma + XGBoost)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 9] CLV Modeling Pipeline")
    logger.info("=" * 70)

    from src.models.clv_models import CLVModeler

    modeler = CLVModeler(config)
    clv_result = modeler.run_all(
        rfm_calibration=rfm_calibration,
        rfm_holdout=rfm_holdout,
        full_features=merged_cal,
    )

    # ==================================================================
    # Step 10: Model Evaluation (Holdout Validation)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 10] Evaluating CLV Predictions Against Holdout")
    logger.info("=" * 70)

    from src.models.evaluator import CLVEvaluator

    evaluator = CLVEvaluator(config)
    metrics = evaluator.evaluate(clv_result, rfm_holdout)
    evaluator.calibration_plot(clv_result, rfm_holdout)

    # ==================================================================
    # Step 11: Market Basket Analysis (Apriori)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  [STEP 11] Market Basket Analysis")
    logger.info("=" * 70)

    from src.features.mba_builder import MBABuilder

    # MBA uses full transactions (not split) and product-level data
    # First merge product info for DEPARTMENT column
    products = loader.load_products()
    txn_with_dept = transactions.merge(
        products[["PRODUCT_ID", "DEPARTMENT"]],
        on="PRODUCT_ID", how="left",
    )
    del products

    mba_builder = MBABuilder(config)
    mba_rules = mba_builder.run_all(txn_with_dept, item_col="DEPARTMENT")
    del txn_with_dept

    # ==================================================================
    # Step 12: Final Summary
    # ==================================================================
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {elapsed:.1f} seconds")
    logger.info(f"  Calibration RFM: {len(rfm_calibration):,} households")
    logger.info(f"  Holdout RFM:     {len(rfm_holdout):,} households")
    logger.info(f"  CLV predictions: {len(clv_result):,} households")
    logger.info(f"  MBA rules:       {len(mba_rules):,} rules")
    logger.info(f"  Plots saved:     {len(plot_paths)} charts")
    if metrics:
        logger.info(f"  Evaluation: MAE=${metrics['mae']:,.2f}, "
                     f"RMSE=${metrics['rmse']:,.2f}, MAPE={metrics['mape']:.1f}%")
    logger.info("")
    logger.info("  Output files:")
    logger.info(f"    {rfm_cal_path}")
    logger.info(f"    {rfm_hold_path}")
    logger.info(f"    {cal_out_path}")
    logger.info(f"    {holdout_out_path}")
    for name, path in plot_paths.items():
        logger.info(f"    {path}")
    logger.info("=" * 70)


def _print_relationships(
    transactions: pd.DataFrame,
    demographics: pd.DataFrame,
    campaigns: pd.DataFrame,
    coupon_redemptions: pd.DataFrame,
) -> None:
    """Print a summary of how tables relate via shared keys."""
    separator = "-" * 60
    print(f"\n{separator}")
    print("  [TABLE RELATIONSHIPS]")
    print(separator)

    # Households across tables
    txn_hh = set(transactions["household_key"].unique())
    demo_hh = set(demographics["household_key"].unique())
    camp_hh = set(campaigns["household_key"].unique())
    coupon_hh = set(coupon_redemptions["household_key"].unique())

    print(f"\n  Households in transaction_data:  {len(txn_hh):,}")
    print(f"  Households in hh_demographic:    {len(demo_hh):,}  "
          f"({len(txn_hh & demo_hh):,} overlap with transactions)")
    print(f"  Households in campaign_table:    {len(camp_hh):,}  "
          f"({len(txn_hh & camp_hh):,} overlap with transactions)")
    print(f"  Households in coupon_redempt:    {len(coupon_hh):,}  "
          f"({len(txn_hh & coupon_hh):,} overlap with transactions)")

    # Demographics coverage
    demo_coverage = len(txn_hh & demo_hh) / len(txn_hh) * 100
    print(f"\n  [!] Demographic coverage: {demo_coverage:.1f}% "
          f"({len(txn_hh & demo_hh)}/{len(txn_hh)} households)")

    # Campaign participation
    camp_coverage = len(txn_hh & camp_hh) / len(txn_hh) * 100
    print(f"  Campaign participation: {camp_coverage:.1f}% "
          f"({len(txn_hh & camp_hh)}/{len(txn_hh)} households)")

    print(f"\n{separator}\n")


# ==================================================================
# Entry Point
# ==================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.run_preprocessing <config.yaml>")
        print("Example: python -m src.pipeline.run_preprocessing configs/config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]
    if not os.path.isfile(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    run_pipeline(config_file)
