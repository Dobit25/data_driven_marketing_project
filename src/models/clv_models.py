"""
clv_models.py — CLV Modeling Module
=====================================
OOP module for Customer Lifetime Value segmentation and forecasting.

Responsibilities:
    - K-Means clustering with Silhouette-based optimal K selection
    - Business-meaningful segment labeling (Champions, Loyal, At Risk, etc.)
    - BG/NBD + Gamma-Gamma probabilistic CLV forecasting
    - XGBoost/LightGBM supervised CLV prediction
    - Model persistence via joblib

Design Decisions:
    - All models use CONSISTENT monetary definition (avg_monetary) for
      alignment between clustering and probabilistic models.
    - XGBoost uses avg_monetary as a FEATURE, and holdout Net_Sales as TARGET.
    - Segment labels are assigned automatically by analyzing cluster centroids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class CLVModeler:
    """Customer Lifetime Value modeling pipeline.

    Handles segmentation (K-Means), probabilistic CLV (BG/NBD + Gamma-Gamma),
    and supervised ML (XGBoost/LightGBM).

    Parameters
    ----------
    config : dict
        Configuration dictionary from config.yaml.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_dir = Path(config["output"]["model_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.seed = config["project"].get("random_seed", 42)

        # Models (populated after fitting)
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.bgf = None  # BetaGeoFitter
        self.ggf = None  # GammaGammaFitter
        self.supervised_model = None  # XGBoost or LightGBM

    # ------------------------------------------------------------------
    # K-Means Segmentation
    # ------------------------------------------------------------------
    def segment_customers(
        self,
        rfm: pd.DataFrame,
        k_range: Tuple[int, int] = (2, 9),
    ) -> pd.DataFrame:
        """Run K-Means clustering on RFM features with Silhouette optimization.

        Uses ['Recency', 'Frequency', 'avg_monetary'] — consistent with
        the monetary definition used by BG/NBD and Gamma-Gamma.

        Parameters
        ----------
        rfm : pd.DataFrame
            RFM table from RFMBuilder (must contain Recency, Frequency,
            avg_monetary columns).
        k_range : tuple of int
            (min_k, max_k) range to search for optimal clusters.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added 'Cluster' and 'Segment' columns.
        """
        logger.info("Segmenting customers with K-Means ...")
        features = ["Recency", "Frequency", "avg_monetary"]

        X = rfm[features].copy()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal K via Silhouette Score
        best_k, best_score = k_range[0], -1
        logger.info(f"  Searching K in range [{k_range[0]}, {k_range[1]}) ...")

        for k in range(k_range[0], k_range[1]):
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            logger.info(f"    K={k} → Silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k

        logger.info(f"  ✓ Optimal K={best_k} (Silhouette={best_score:.4f})")

        # Fit final model
        self.kmeans = KMeans(n_clusters=best_k, random_state=self.seed, n_init=10)
        rfm = rfm.copy()
        rfm["Cluster"] = self.kmeans.fit_predict(X_scaled)

        # Assign business labels
        rfm = self._assign_segment_labels(rfm)

        # Save customer labels
        labels_path = self.processed_dir / "customer_labels.csv"
        rfm[["household_key", "Cluster", "Segment"]].to_csv(
            labels_path, index=False
        )
        logger.info(f"  Saved: {labels_path}")

        return rfm

    def _assign_segment_labels(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Map cluster IDs to business-readable segment labels.

        Logic: Analyze cluster centroids on (Recency, Frequency, avg_monetary)
        and assign labels based on RFM profile:
            - Low R + High F + High M → Champions
            - Low R + High F + Med M  → Loyal Customers
            - Low R + Low F           → New Customers
            - Med R + Med F           → Promising
            - High R + High F         → At Risk
            - High R + Low F          → Hibernating
        """
        centroids = rfm.groupby("Cluster")[
            ["Recency", "Frequency", "avg_monetary"]
        ].mean()

        # Rank each cluster on each dimension (lower rank = lower value)
        n_clusters = len(centroids)
        r_rank = centroids["Recency"].rank()          # low R is good
        f_rank = centroids["Frequency"].rank()         # high F is good
        m_rank = centroids["avg_monetary"].rank()       # high M is good

        # Composite score: good customers have low R rank + high F rank + high M rank
        # Score = (n - R_rank) + F_rank + M_rank  → higher is better
        score = (n_clusters + 1 - r_rank) + f_rank + m_rank
        sorted_clusters = score.sort_values(ascending=False).index.tolist()

        # Map labels based on rank order
        label_pool = [
            "Champions", "Loyal Customers", "Promising",
            "Needs Attention", "At Risk", "Hibernating",
            "About to Sleep", "Lost", "Cannot Lose",
        ]
        segment_map = {}
        for i, cluster_id in enumerate(sorted_clusters):
            label = label_pool[i] if i < len(label_pool) else f"Segment_{i}"
            segment_map[cluster_id] = label

        rfm["Segment"] = rfm["Cluster"].map(segment_map)

        for cid, label in segment_map.items():
            count = (rfm["Cluster"] == cid).sum()
            logger.info(f"    Cluster {cid} → '{label}' ({count:,} customers)")

        return rfm

    # ------------------------------------------------------------------
    # BG/NBD + Gamma-Gamma (Probabilistic CLV)
    # ------------------------------------------------------------------
    def fit_bgnbd(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Fit BG/NBD model and predict future purchases.

        Includes data sanitization to handle extreme outliers that cause
        numerical overflow in the BG/NBD log-likelihood:
            1. Winsorize Frequency at 99th percentile
            2. Progressive penalizer fallback if convergence fails

        Parameters
        ----------
        rfm : pd.DataFrame
            Must contain: Frequency, Recency, T columns.
            BG/NBD requires Frequency > 0 (repeat customers).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 'predicted_purchases_6m' column added.
        """
        from lifetimes import BetaGeoFitter

        logger.info("Fitting BG/NBD model ...")
        rfm = rfm.copy()

        # Filter repeat customers
        repeat_mask = rfm["Frequency"] > 0
        repeat_df = rfm.loc[repeat_mask].copy()
        logger.info(
            f"  Repeat customers: {len(repeat_df):,}/{len(rfm):,} "
            f"({len(repeat_df)/len(rfm)*100:.1f}%)"
        )

        # --- Data sanitization: Winsorize Frequency at 99th percentile ---
        freq_raw_max = repeat_df["Frequency"].max()
        freq_p99 = int(repeat_df["Frequency"].quantile(0.99))
        repeat_df["Frequency"] = repeat_df["Frequency"].clip(upper=freq_p99)

        # --- Enforce mathematical constraint: Frequency <= T ---
        # A customer cannot make more repeat purchases than the number of periods (days)
        repeat_df["Frequency"] = np.minimum(repeat_df["Frequency"], repeat_df["T"])

        logger.info(
            f"  Sanitized: Frequency capped at p99={freq_p99:.0f} "
            f"(raw max={freq_raw_max}), and enforced Freq ≤ T"
        )

        # --- Progressive penalizer fitting ---
        base_penalizer = self.config["model"]["bgnbd_gg"].get(
            "penalizer_coef", 0.05
        )
        penalizers_to_try = [
            base_penalizer,
            max(base_penalizer, 0.1),
            0.5,
            1.0,
            5.0,
            10.0,
        ]
        # Deduplicate while preserving order
        seen = set()
        penalizers_to_try = [
            p for p in penalizers_to_try
            if not (p in seen or seen.add(p))
        ]

        fitted = False
        for pen in penalizers_to_try:
            try:
                self.bgf = BetaGeoFitter(penalizer_coef=pen)
                self.bgf.fit(
                    repeat_df["Frequency"],
                    repeat_df["Recency"],
                    repeat_df["T"],
                )
                logger.info(
                    f"  ✓ BG/NBD fitted successfully "
                    f"(penalizer={pen})"
                )
                fitted = True
                break
            except Exception as e:
                logger.warning(
                    f"  ✗ BG/NBD failed with penalizer={pen}: {e}"
                )

        if not fitted:
            logger.error(
                "  ✗ BG/NBD could not converge with any penalizer. "
                "Setting predicted_purchases_6m = Frequency-based fallback."
            )
            # Fallback: simple rate-based prediction (in days)
            rate = rfm.loc[repeat_mask, "Frequency"] / rfm.loc[repeat_mask, "T"]
            rfm.loc[repeat_mask, "predicted_purchases_6m"] = rate * 182
            rfm["predicted_purchases_6m"] = rfm["predicted_purchases_6m"].fillna(0)
            return rfm

        # Predict purchases for next 182 days (6 months)
        # Clip Frequency at p99 and Freq <= T for prediction consistency
        t = 182
        pred_freq = np.minimum(
            rfm.loc[repeat_mask, "Frequency"].clip(upper=freq_p99),
            rfm.loc[repeat_mask, "T"]
        )
        pred_rec = rfm.loc[repeat_mask, "Recency"]
        pred_T = rfm.loc[repeat_mask, "T"]

        rfm.loc[repeat_mask, "predicted_purchases_6m"] = (
            self.bgf.conditional_expected_number_of_purchases_up_to_time(
                t, pred_freq, pred_rec, pred_T,
            )
        )
        rfm["predicted_purchases_6m"] = rfm["predicted_purchases_6m"].fillna(0)

        logger.info(
            f"  Predicted purchases (6m): "
            f"mean={rfm['predicted_purchases_6m'].mean():.2f}, "
            f"max={rfm['predicted_purchases_6m'].max():.2f}"
        )
        return rfm

    def fit_gamma_gamma(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Fit Gamma-Gamma model and compute CLV.

        Includes progressive penalizer and graceful fallback,
        mirroring the BG/NBD protection strategy.

        Parameters
        ----------
        rfm : pd.DataFrame
            Must contain: Frequency, Recency, T, avg_monetary, and
            predicted_purchases_6m columns.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 'predicted_clv_6m' column added.
        """
        from lifetimes import GammaGammaFitter

        logger.info("Fitting Gamma-Gamma model ...")
        rfm = rfm.copy()

        # Filter: Frequency > 0 AND avg_monetary > 0
        gg_mask = (rfm["Frequency"] > 0) & (rfm["avg_monetary"] > 0)
        gg_df = rfm.loc[gg_mask].copy()
        logger.info(f"  Customers for Gamma-Gamma: {len(gg_df):,}")

        # Winsorize to improve convergence
        freq_p99 = int(gg_df["Frequency"].quantile(0.99))
        mon_p99 = gg_df["avg_monetary"].quantile(0.99)
        gg_df["Frequency"] = gg_df["Frequency"].clip(upper=freq_p99)
        gg_df["avg_monetary"] = gg_df["avg_monetary"].clip(upper=mon_p99)

        # Enforce mathematical constraint: Frequency <= T
        gg_df["Frequency"] = np.minimum(gg_df["Frequency"], gg_df["T"])

        logger.info(
            f"  Sanitized: Frequency≤{freq_p99:.0f} (and Freq≤T), "
            f"avg_monetary≤{mon_p99:.2f}"
        )

        # Progressive penalizer
        penalizers = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        gg_fitted = False
        for pen in penalizers:
            try:
                self.ggf = GammaGammaFitter(penalizer_coef=pen)
                self.ggf.fit(gg_df["Frequency"], gg_df["avg_monetary"])
                logger.info(f"  ✓ Gamma-Gamma fitted (penalizer={pen})")
                gg_fitted = True
                break
            except Exception as e:
                logger.warning(f"  ✗ GG failed with penalizer={pen}: {e}")

        if not gg_fitted:
            logger.error(
                "  ✗ Gamma-Gamma could not converge. "
                "CLV fallback: avg_monetary × predicted_purchases_6m"
            )
            rfm.loc[gg_mask, "predicted_clv_6m"] = (
                rfm.loc[gg_mask, "avg_monetary"]
                * rfm.loc[gg_mask, "predicted_purchases_6m"]
            )
            rfm["predicted_clv_6m"] = rfm["predicted_clv_6m"].fillna(0)
            return rfm

        # Check if BG/NBD converged properly
        bgf_valid = (
            hasattr(self, "bgf")
            and self.bgf is not None
            and hasattr(self.bgf, "params_")
        )

        if bgf_valid:
            # Full CLV: BG/NBD × Gamma-Gamma with discounting
            pred_freq = np.minimum(
                rfm.loc[gg_mask, "Frequency"].clip(upper=freq_p99),
                rfm.loc[gg_mask, "T"]
            )
            rfm.loc[gg_mask, "predicted_clv_6m"] = (
                self.ggf.customer_lifetime_value(
                    self.bgf,
                    pred_freq,
                    rfm.loc[gg_mask, "Recency"],
                    rfm.loc[gg_mask, "T"],
                    rfm.loc[gg_mask, "avg_monetary"].clip(upper=mon_p99),
                    time=182,
                    freq="D",
                    discount_rate=0.01,
                )
            )
            logger.info("  CLV via BG/NBD × Gamma-Gamma (full model)")
        else:
            # BG/NBD failed → E[monetary] × predicted_purchases
            logger.warning(
                "  BG/NBD not fitted → "
                "CLV = E[monetary] × predicted_purchases_6m"
            )
            expected_m = self.ggf.conditional_expected_average_profit(
                gg_df["Frequency"], gg_df["avg_monetary"]
            )
            rfm.loc[gg_mask, "predicted_clv_6m"] = (
                expected_m * rfm.loc[gg_mask, "predicted_purchases_6m"]
            )

        rfm["predicted_clv_6m"] = rfm["predicted_clv_6m"].fillna(0)

        logger.info(
            f"  Predicted CLV (6m): "
            f"mean=${rfm['predicted_clv_6m'].mean():,.2f}, "
            f"median=${rfm['predicted_clv_6m'].median():,.2f}, "
            f"max=${rfm['predicted_clv_6m'].max():,.2f}"
        )
        return rfm

    # ------------------------------------------------------------------
    # Supervised ML (XGBoost / LightGBM)
    # ------------------------------------------------------------------
    def fit_supervised(
        self,
        train_features: pd.DataFrame,
        holdout_rfm: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit XGBoost or LightGBM to predict CLV using rich feature set.

        The target variable is holdout-period Net_Sales (actual spending).
        Features include RFM + demographics + causal/promotion features.

        Parameters
        ----------
        train_features : pd.DataFrame
            Calibration-period features (RFM + demographics + promo features)
            with 'household_key' column.
        holdout_rfm : pd.DataFrame
            Holdout-period RFM with 'household_key' and 'Net_Sales' columns.
            Net_Sales is used as the target variable.

        Returns
        -------
        pd.DataFrame
            train_features with 'predicted_clv_supervised' column added.
        """
        active_model = self.config["model"].get("active", "bgnbd_gg")
        if active_model not in ("xgboost", "lightgbm"):
            logger.info("  Supervised model not active in config. Skipping.")
            return train_features

        logger.info(f"Fitting supervised model: {active_model} ...")

        # Prepare target: holdout Net_Sales
        target = holdout_rfm[["household_key", "Net_Sales"]].rename(
            columns={"Net_Sales": "target_clv"}
        )

        # Merge target into training features
        merged = train_features.merge(target, on="household_key", how="inner")

        # Select feature columns (exclude keys and targets)
        exclude_cols = {
            "household_key", "target_clv", "Cluster", "Segment",
            "predicted_purchases_6m", "predicted_clv_6m",
            "first_purchase_week", "last_purchase_week",
        }
        feature_cols = [c for c in merged.columns if c not in exclude_cols]

        # Handle categorical columns
        cat_cols = merged[feature_cols].select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        for col in cat_cols:
            merged[col] = merged[col].astype("category")

        X = merged[feature_cols]
        y = merged["target_clv"]

        logger.info(f"  Features: {len(feature_cols)}, Samples: {len(X):,}")

        # Train/validation split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if active_model == "xgboost":
            import xgboost as xgb

            params = self.config["model"]["xgboost"]
            self.supervised_model = xgb.XGBRegressor(
                n_estimators=params.get("n_estimators", 500),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                random_state=self.seed,
                enable_categorical=True,
            )
            self.supervised_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        elif active_model == "lightgbm":
            import lightgbm as lgb

            params = self.config["model"]["lightgbm"]
            self.supervised_model = lgb.LGBMRegressor(
                n_estimators=params.get("n_estimators", 500),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                random_state=self.seed,
                verbose=-1,
            )
            self.supervised_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )

        logger.info(f"  ✓ {active_model} fitted successfully")

        # Predict on all data
        merged["predicted_clv_supervised"] = self.supervised_model.predict(X)

        # Merge predictions back
        train_features = train_features.merge(
            merged[["household_key", "predicted_clv_supervised"]],
            on="household_key", how="left",
        )

        return train_features

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------
    def save_models(self) -> None:
        """Save all fitted models to disk via joblib."""
        logger.info("Saving models ...")

        if self.kmeans is not None:
            path = self.model_dir / "kmeans_model.pkl"
            joblib.dump({"kmeans": self.kmeans, "scaler": self.scaler}, path)
            logger.info(f"  Saved: {path}")

        if self.bgf is not None:
            path = self.model_dir / "bgnbd_model.pkl"
            joblib.dump(self.bgf, path)
            logger.info(f"  Saved: {path}")

        if self.ggf is not None:
            path = self.model_dir / "gg_model.pkl"
            joblib.dump(self.ggf, path)
            logger.info(f"  Saved: {path}")

        if self.supervised_model is not None:
            active = self.config["model"].get("active", "xgboost")
            path = self.model_dir / f"{active}_model.pkl"
            joblib.dump(self.supervised_model, path)
            logger.info(f"  Saved: {path}")

        logger.info("  ✓ All models saved")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def run_all(
        self,
        rfm_calibration: pd.DataFrame,
        rfm_holdout: Optional[pd.DataFrame] = None,
        full_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Execute the full modeling pipeline.

        Parameters
        ----------
        rfm_calibration : pd.DataFrame
            Calibration-period RFM features.
        rfm_holdout : pd.DataFrame, optional
            Holdout-period RFM (for supervised model target).
        full_features : pd.DataFrame, optional
            Enriched features (RFM + demographics + causal) for supervised model.

        Returns
        -------
        pd.DataFrame
            Final DataFrame with clusters, segments, and CLV predictions.
        """
        # Step 1: K-Means segmentation
        result = self.segment_customers(rfm_calibration)

        # Step 2: BG/NBD
        result = self.fit_bgnbd(result)

        # Step 3: Gamma-Gamma
        result = self.fit_gamma_gamma(result)

        # Step 4: Supervised ML (if configured and data available)
        if full_features is not None and rfm_holdout is not None:
            full_features = self.fit_supervised(full_features, rfm_holdout)
            # Merge supervised predictions back
            if "predicted_clv_supervised" in full_features.columns:
                sup_preds = full_features[
                    ["household_key", "predicted_clv_supervised"]
                ]
                result = result.merge(sup_preds, on="household_key", how="left")

        # Step 5: Save models
        self.save_models()

        # Step 6: Save final output
        final_path = self.processed_dir / "rfm_clv_final.csv"
        result.to_csv(final_path, index=False)
        logger.info(f"  Saved final CLV table: {final_path}")

        return result
