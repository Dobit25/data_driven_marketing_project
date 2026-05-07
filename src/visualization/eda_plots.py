"""
eda_plots.py — EDA Visualization Module
=========================================
Generates and saves exploratory data analysis plots for CLV pipeline.

All plots are saved as high-resolution PNG files to `reports/figures/`.
Uses matplotlib + seaborn for static plots with a consistent dark theme.

Output Files:
    reports/figures/
    ├── 01_transaction_volume_by_week.png
    ├── 02_sales_distribution.png
    ├── 03_rfm_distributions.png
    ├── 04_rfm_correlations.png
    ├── 05_demographic_coverage.png
    ├── 06_demographic_distributions.png
    ├── 07_monetary_gross_vs_net.png
    ├── 08_customer_segments_rfm.png
    ├── 09_coupon_impact.png
    └── 10_calibration_holdout_split.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


class EDAPlotter:
    """Generate and save EDA visualizations for CLV analysis.

    Parameters
    ----------
    config : dict
        Configuration dictionary with ``output.figures_dir`` path.
    style : str
        Matplotlib style. Default "seaborn-v0_8-darkgrid" for modern look.

    Example
    -------
    >>> plotter = EDAPlotter(config)
    >>> plotter.plot_transaction_volume(transactions_df)
    >>> plotter.plot_rfm_distributions(rfm_df)
    """

    # Color palette — curated for CLV analysis
    COLORS = {
        "primary": "#4E79A7",
        "secondary": "#F28E2B",
        "accent": "#E15759",
        "success": "#59A14F",
        "info": "#76B7B2",
        "muted": "#BAB0AC",
        "highlight": "#EDC948",
        "purple": "#B07AA1",
        "gradient": ["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#76B7B2"],
    }

    def __init__(self, config: Dict[str, Any], style: str = "seaborn-v0_8-darkgrid") -> None:
        self.config = config
        self.figures_dir = Path(config["output"]["figures_dir"])
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Try to set the style; fall back to a safe default if unavailable
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("ggplot")

        # Global plot settings
        plt.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        })

        logger.info(f"EDAPlotter initialized. Saving figures to: {self.figures_dir}")

    def _save_fig(self, fig: plt.Figure, filename: str) -> str:
        """Save figure to the figures directory and close it.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure object.
        filename : str
            Filename (e.g., '01_transaction_volume_by_week.png').

        Returns
        -------
        str
            Full path to the saved file.
        """
        filepath = self.figures_dir / filename
        fig.savefig(filepath, facecolor="white", edgecolor="none")
        plt.close(fig)
        logger.info(f"  Saved: {filepath}")
        return str(filepath)

    # ==================================================================
    # Plot 1: Transaction Volume Over Time
    # ==================================================================
    def plot_transaction_volume(
        self,
        transactions: pd.DataFrame,
        calibration_end_week: Optional[int] = None,
    ) -> str:
        """Plot weekly transaction count and total sales over time.

        Shows the overall shopping volume trend and marks the
        calibration/holdout split point if provided.

        Parameters
        ----------
        transactions : pd.DataFrame
            Raw transaction data with WEEK_NO, SALES_VALUE columns.
        calibration_end_week : int, optional
            If provided, draws a vertical line at the split point.

        Returns
        -------
        str
            Path to saved figure.
        """
        weekly = (
            transactions
            .groupby("WEEK_NO")
            .agg(
                n_transactions=("BASKET_ID", "count"),
                total_sales=("SALES_VALUE", "sum"),
                n_households=("household_key", "nunique"),
            )
            .reset_index()
        )

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(
            "Transaction Volume Over 102 Weeks",
            fontsize=15, fontweight="bold", y=0.98,
        )

        # Panel 1: Transaction count
        axes[0].fill_between(
            weekly["WEEK_NO"], weekly["n_transactions"],
            alpha=0.3, color=self.COLORS["primary"],
        )
        axes[0].plot(
            weekly["WEEK_NO"], weekly["n_transactions"],
            color=self.COLORS["primary"], linewidth=1.5,
        )
        axes[0].set_ylabel("Transaction Lines")
        axes[0].set_title("Weekly Transaction Volume")
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # Panel 2: Total sales
        axes[1].fill_between(
            weekly["WEEK_NO"], weekly["total_sales"],
            alpha=0.3, color=self.COLORS["success"],
        )
        axes[1].plot(
            weekly["WEEK_NO"], weekly["total_sales"],
            color=self.COLORS["success"], linewidth=1.5,
        )
        axes[1].set_ylabel("Total Sales ($)")
        axes[1].set_xlabel("Week Number")
        axes[1].set_title("Weekly Total Sales (SALES_VALUE)")
        axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # Mark calibration/holdout split
        if calibration_end_week:
            for ax in axes:
                ax.axvline(
                    x=calibration_end_week, color=self.COLORS["accent"],
                    linestyle="--", linewidth=2, alpha=0.8,
                )
            axes[0].text(
                calibration_end_week + 1, axes[0].get_ylim()[1] * 0.9,
                f"<-- Calibration | Holdout -->\n(Week {calibration_end_week})",
                fontsize=9, color=self.COLORS["accent"],
            )

        plt.tight_layout()
        return self._save_fig(fig, "01_transaction_volume_by_week.png")

    # ==================================================================
    # Plot 2: Sales Value Distribution
    # ==================================================================
    def plot_sales_distribution(self, transactions: pd.DataFrame) -> str:
        """Plot distribution of SALES_VALUE and discount columns.

        Parameters
        ----------
        transactions : pd.DataFrame
            Raw transaction data.

        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Sales & Discount Distributions (Transaction Level)",
            fontsize=15, fontweight="bold",
        )

        # Sales Value
        sales = transactions["SALES_VALUE"].clip(upper=transactions["SALES_VALUE"].quantile(0.99))
        axes[0, 0].hist(sales, bins=80, color=self.COLORS["primary"], alpha=0.7, edgecolor="white")
        axes[0, 0].set_title("SALES_VALUE Distribution")
        axes[0, 0].set_xlabel("Sales Value ($)")
        axes[0, 0].axvline(sales.mean(), color=self.COLORS["accent"], linestyle="--", label=f"Mean: ${sales.mean():.2f}")
        axes[0, 0].legend()

        # Retail Discount
        disc = transactions["RETAIL_DISC"]
        disc_nonzero = disc[disc != 0]
        if len(disc_nonzero) > 0:
            axes[0, 1].hist(disc_nonzero, bins=60, color=self.COLORS["secondary"], alpha=0.7, edgecolor="white")
        axes[0, 1].set_title("RETAIL_DISC (non-zero)")
        axes[0, 1].set_xlabel("Discount ($)")

        # Coupon Discount
        coupon = transactions["COUPON_DISC"]
        coupon_nonzero = coupon[coupon != 0]
        if len(coupon_nonzero) > 0:
            axes[1, 0].hist(coupon_nonzero, bins=60, color=self.COLORS["info"], alpha=0.7, edgecolor="white")
        axes[1, 0].set_title("COUPON_DISC (non-zero)")
        axes[1, 0].set_xlabel("Discount ($)")

        # Discount usage breakdown (pie chart)
        has_retail = (transactions["RETAIL_DISC"] != 0).sum()
        has_coupon = (transactions["COUPON_DISC"] != 0).sum()
        has_match = (transactions["COUPON_MATCH_DISC"] != 0).sum()
        no_disc = len(transactions) - (has_retail | has_coupon | has_match)

        labels = ["Retail Disc", "Coupon Disc", "Match Disc", "No Discount"]
        sizes = [has_retail, has_coupon, has_match, max(no_disc, 0)]
        colors = [self.COLORS["secondary"], self.COLORS["info"], self.COLORS["purple"], self.COLORS["muted"]]
        axes[1, 1].pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.85,
        )
        axes[1, 1].set_title("Discount Usage Breakdown")

        plt.tight_layout()
        return self._save_fig(fig, "02_sales_distribution.png")

    # ==================================================================
    # Plot 3: RFM Distributions
    # ==================================================================
    def plot_rfm_distributions(self, rfm: pd.DataFrame) -> str:
        """Plot histograms of core RFM features.

        Parameters
        ----------
        rfm : pd.DataFrame
            Household-level RFM table (output of RFMBuilder).

        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            "RFM Feature Distributions (Household Level)",
            fontsize=15, fontweight="bold",
        )

        features = [
            ("Recency", "Weeks since last purchase", self.COLORS["primary"]),
            ("Frequency", "Repeat purchases (baskets - 1)", self.COLORS["secondary"]),
            ("Net_Sales", "Total Net Sales ($)", self.COLORS["success"]),
            ("T", "Customer age (weeks)", self.COLORS["info"]),
            ("avg_basket_size", "Avg items per basket", self.COLORS["purple"]),
            ("coupon_usage_rate", "Coupon usage rate", self.COLORS["highlight"]),
        ]

        for idx, (col, xlabel, color) in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            if col in rfm.columns:
                data = rfm[col].dropna()
                # Clip outliers for better visualization
                upper = data.quantile(0.99)
                data_clipped = data.clip(upper=upper)
                ax.hist(data_clipped, bins=50, color=color, alpha=0.7, edgecolor="white")
                ax.axvline(data.median(), color=self.COLORS["accent"], linestyle="--",
                           label=f"Median: {data.median():.1f}")
                ax.set_title(col)
                ax.set_xlabel(xlabel)
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f"'{col}' not found", transform=ax.transAxes,
                        ha="center", va="center")

        plt.tight_layout()
        return self._save_fig(fig, "03_rfm_distributions.png")

    # ==================================================================
    # Plot 4: RFM Correlations
    # ==================================================================
    def plot_rfm_correlations(self, rfm: pd.DataFrame) -> str:
        """Plot correlation heatmap of RFM features.

        Parameters
        ----------
        rfm : pd.DataFrame
            Household-level RFM table.

        Returns
        -------
        str
            Path to saved figure.
        """
        corr_cols = [
            "Recency", "Frequency", "T", "Gross_Sales", "Net_Sales",
            "avg_monetary", "avg_basket_size", "avg_transaction_value",
            "distinct_stores", "tenure_weeks", "coupon_usage_rate",
        ]
        existing = [c for c in corr_cols if c in rfm.columns]
        corr_matrix = rfm[existing].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(
            "RFM Feature Correlation Matrix",
            fontsize=15, fontweight="bold", pad=20,
        )

        plt.tight_layout()
        return self._save_fig(fig, "04_rfm_correlations.png")

    # ==================================================================
    # Plot 5: Demographic Coverage
    # ==================================================================
    def plot_demographic_coverage(self, merged_df: pd.DataFrame) -> str:
        """Visualize the demographic data coverage (801/2500 households).

        Parameters
        ----------
        merged_df : pd.DataFrame
            RFM + demographics merged DataFrame with has_demographics flag.

        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Demographic Data Coverage Analysis",
            fontsize=15, fontweight="bold",
        )

        # Panel 1: Coverage pie chart
        n_with = merged_df["has_demographics"].sum()
        n_without = len(merged_df) - n_with
        sizes = [n_with, n_without]
        labels = [
            f"With Demographics\n({n_with:,} HH, {n_with/len(merged_df)*100:.1f}%)",
            f"Without Demographics\n({n_without:,} HH, {n_without/len(merged_df)*100:.1f}%)",
        ]
        colors_pie = [self.COLORS["success"], self.COLORS["muted"]]
        wedges, texts, autotexts = axes[0].pie(
            sizes, labels=labels, colors=colors_pie,
            autopct="", startangle=90, textprops={"fontsize": 10},
        )
        axes[0].set_title("Demographic Coverage")

        # Panel 2: Compare RFM distributions (with vs without demographics)
        for has_demo, color, label in [
            (1, self.COLORS["success"], "With Demographics"),
            (0, self.COLORS["muted"], "Without Demographics"),
        ]:
            subset = merged_df[merged_df["has_demographics"] == has_demo]
            if "Net_Sales" in subset.columns and len(subset) > 0:
                data = subset["Net_Sales"].clip(upper=subset["Net_Sales"].quantile(0.99))
                axes[1].hist(
                    data, bins=40, alpha=0.5, label=f"{label} (n={len(subset)})",
                    color=color, edgecolor="white",
                )
        axes[1].set_title("Net Sales Distribution by Demographic Availability")
        axes[1].set_xlabel("Net Sales ($)")
        axes[1].legend()

        plt.tight_layout()
        return self._save_fig(fig, "05_demographic_coverage.png")

    # ==================================================================
    # Plot 6: Demographic Distributions
    # ==================================================================
    def plot_demographic_distributions(self, demographics: pd.DataFrame) -> str:
        """Plot distributions of demographic categories.

        Parameters
        ----------
        demographics : pd.DataFrame
            Demographic data (hh_demographic.csv, 801 rows).

        Returns
        -------
        str
            Path to saved figure.
        """
        demo_cols = [
            "AGE_DESC", "INCOME_DESC", "MARITAL_STATUS_CODE",
            "HOMEOWNER_DESC", "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC",
        ]
        existing = [c for c in demo_cols if c in demographics.columns]
        n_cols = len(existing)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
        fig.suptitle(
            "Demographic Feature Distributions (801 Households with Data)",
            fontsize=15, fontweight="bold",
        )
        axes_flat = axes.flatten() if n_cols > 2 else [axes] if n_cols == 1 else axes

        for idx, col in enumerate(existing):
            ax = axes_flat[idx]
            counts = demographics[col].value_counts()
            bars = ax.barh(
                range(len(counts)), counts.values,
                color=self.COLORS["gradient"][idx % len(self.COLORS["gradient"])],
                alpha=0.8, edgecolor="white",
            )
            ax.set_yticks(range(len(counts)))
            ax.set_yticklabels(counts.index, fontsize=8)
            ax.set_title(col)
            ax.set_xlabel("Count")

            # Add count labels
            for bar, count in zip(bars, counts.values):
                ax.text(
                    bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=8,
                )

        # Hide empty subplots
        for idx in range(len(existing), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        return self._save_fig(fig, "06_demographic_distributions.png")

    # ==================================================================
    # Plot 7: Gross vs Net Sales
    # ==================================================================
    def plot_monetary_comparison(self, rfm: pd.DataFrame) -> str:
        """Compare Gross_Sales vs Net_Sales to visualize discount impact.

        Parameters
        ----------
        rfm : pd.DataFrame
            Household-level RFM table with Gross_Sales and Net_Sales.

        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Monetary Analysis: Gross vs. Net Sales",
            fontsize=15, fontweight="bold",
        )

        # Panel 1: Scatter plot
        gross_clip = rfm["Gross_Sales"].clip(upper=rfm["Gross_Sales"].quantile(0.99))
        net_clip = rfm["Net_Sales"].clip(upper=rfm["Net_Sales"].quantile(0.99))

        axes[0].scatter(
            gross_clip, net_clip,
            alpha=0.4, s=15, color=self.COLORS["primary"],
        )
        # Add 45-degree reference line
        max_val = max(gross_clip.max(), net_clip.max())
        axes[0].plot(
            [0, max_val], [0, max_val],
            "--", color=self.COLORS["accent"], alpha=0.6, label="1:1 line",
        )
        axes[0].set_xlabel("Gross Sales ($)")
        axes[0].set_ylabel("Net Sales ($)")
        axes[0].set_title("Gross vs Net Sales per Household")
        axes[0].legend()

        # Panel 2: Discount percentage distribution
        rfm_plot = rfm.copy()
        rfm_plot["discount_pct"] = np.where(
            rfm_plot["Gross_Sales"] > 0,
            (1 - rfm_plot["Net_Sales"] / rfm_plot["Gross_Sales"]) * 100,
            0,
        )
        disc_pct = rfm_plot["discount_pct"].clip(lower=0, upper=100)
        axes[1].hist(
            disc_pct, bins=50, color=self.COLORS["secondary"],
            alpha=0.7, edgecolor="white",
        )
        axes[1].axvline(
            disc_pct.median(), color=self.COLORS["accent"], linestyle="--",
            label=f"Median: {disc_pct.median():.1f}%",
        )
        axes[1].set_title("Discount Percentage per Household")
        axes[1].set_xlabel("Discount % (1 - Net/Gross x 100)")
        axes[1].legend()

        plt.tight_layout()
        return self._save_fig(fig, "07_monetary_gross_vs_net.png")

    # ==================================================================
    # Plot 8: Customer Segmentation (RFM Scatter)
    # ==================================================================
    def plot_customer_segments(self, rfm: pd.DataFrame) -> str:
        """Scatter plot of Recency vs Frequency colored by monetary value.

        Parameters
        ----------
        rfm : pd.DataFrame
            Household-level RFM table.

        Returns
        -------
        str
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        monetary = rfm["Net_Sales"].clip(upper=rfm["Net_Sales"].quantile(0.99))
        scatter = ax.scatter(
            rfm["Frequency"],
            rfm["Recency"],
            c=monetary,
            cmap="YlOrRd",
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Net Sales ($)", fontsize=11)

        ax.set_xlabel("Frequency (repeat purchases)")
        ax.set_ylabel("Recency (weeks since last purchase)")
        ax.set_title(
            "Customer Segmentation: Recency x Frequency x Monetary",
            fontsize=14, fontweight="bold",
        )

        # Add quadrant annotations
        freq_median = rfm["Frequency"].median()
        rec_median = rfm["Recency"].median()
        ax.axvline(freq_median, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(rec_median, color="gray", linestyle=":", alpha=0.5)

        # Quadrant labels
        ax.text(rfm["Frequency"].max() * 0.75, rfm["Recency"].max() * 0.05,
                "Champions\n(High F, Low R)", fontsize=9, ha="center",
                color=self.COLORS["success"], fontweight="bold")
        ax.text(rfm["Frequency"].max() * 0.05, rfm["Recency"].max() * 0.95,
                "At Risk\n(Low F, High R)", fontsize=9, ha="center",
                color=self.COLORS["accent"], fontweight="bold")

        plt.tight_layout()
        return self._save_fig(fig, "08_customer_segments_rfm.png")

    # ==================================================================
    # Plot 9: Coupon Impact Analysis
    # ==================================================================
    def plot_coupon_impact(self, rfm: pd.DataFrame) -> str:
        """Analyze relationship between coupon usage and customer value.

        Parameters
        ----------
        rfm : pd.DataFrame
            Household-level RFM table with coupon_usage_rate.

        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Coupon Usage Impact on Customer Value",
            fontsize=15, fontweight="bold",
        )

        if "coupon_usage_rate" not in rfm.columns:
            logger.warning("coupon_usage_rate not in RFM table, skipping coupon plot.")
            plt.close(fig)
            return ""

        # Panel 1: Coupon usage rate distribution
        axes[0].hist(
            rfm["coupon_usage_rate"], bins=40,
            color=self.COLORS["info"], alpha=0.7, edgecolor="white",
        )
        axes[0].set_title("Coupon Usage Rate Distribution")
        axes[0].set_xlabel("Fraction of baskets with coupons")
        axes[0].set_ylabel("Number of households")

        # Panel 2: Coupon usage vs Net Sales
        axes[1].scatter(
            rfm["coupon_usage_rate"],
            rfm["Net_Sales"].clip(upper=rfm["Net_Sales"].quantile(0.99)),
            alpha=0.3, s=15, color=self.COLORS["purple"],
        )
        axes[1].set_title("Coupon Usage vs. Net Sales")
        axes[1].set_xlabel("Coupon usage rate")
        axes[1].set_ylabel("Net Sales ($)")

        plt.tight_layout()
        return self._save_fig(fig, "09_coupon_impact.png")

    # ==================================================================
    # Plot 10: Calibration/Holdout Split Visualization
    # ==================================================================
    def plot_calibration_holdout(
        self,
        transactions: pd.DataFrame,
        calibration_end_week: int,
    ) -> str:
        """Visualize the temporal split between calibration and holdout.

        Parameters
        ----------
        transactions : pd.DataFrame
            Raw transaction data with WEEK_NO.
        calibration_end_week : int
            Week number where calibration ends.

        Returns
        -------
        str
            Path to saved figure.
        """
        weekly_hh = (
            transactions
            .groupby("WEEK_NO")["household_key"]
            .nunique()
            .reset_index(name="n_households")
        )

        fig, ax = plt.subplots(figsize=(14, 6))

        # Color bars by period
        colors = [
            self.COLORS["primary"] if w <= calibration_end_week
            else self.COLORS["secondary"]
            for w in weekly_hh["WEEK_NO"]
        ]
        ax.bar(weekly_hh["WEEK_NO"], weekly_hh["n_households"], color=colors, alpha=0.8)

        # Split line
        ax.axvline(
            x=calibration_end_week + 0.5, color=self.COLORS["accent"],
            linestyle="--", linewidth=2,
        )
        ax.text(
            calibration_end_week - 15, ax.get_ylim()[1] * 0.95,
            f"Calibration\nWeeks 1-{calibration_end_week}",
            fontsize=11, fontweight="bold", color=self.COLORS["primary"],
            ha="center",
        )
        ax.text(
            calibration_end_week + 14, ax.get_ylim()[1] * 0.95,
            f"Holdout\nWeeks {calibration_end_week+1}-102",
            fontsize=11, fontweight="bold", color=self.COLORS["secondary"],
            ha="center",
        )

        ax.set_xlabel("Week Number")
        ax.set_ylabel("Active Households")
        ax.set_title(
            "Calibration / Holdout Split -- Active Households per Week",
            fontsize=14, fontweight="bold",
        )

        plt.tight_layout()
        return self._save_fig(fig, "10_calibration_holdout_split.png")

    # ==================================================================
    # Run All Plots
    # ==================================================================
    def run_all(
        self,
        transactions: pd.DataFrame,
        rfm: pd.DataFrame,
        demographics: pd.DataFrame,
        merged_df: pd.DataFrame,
        calibration_end_week: int,
    ) -> Dict[str, str]:
        """Generate all EDA plots and return dict of saved file paths.

        Parameters
        ----------
        transactions : pd.DataFrame
            Raw transaction data.
        rfm : pd.DataFrame
            Household-level RFM table.
        demographics : pd.DataFrame
            Raw demographic data (801 rows).
        merged_df : pd.DataFrame
            RFM + demographics merged DataFrame.
        calibration_end_week : int
            Week number for calibration/holdout split.

        Returns
        -------
        dict
            Mapping of plot name → file path.
        """
        logger.info("=" * 60)
        logger.info("  Generating EDA Visualizations ...")
        logger.info("=" * 60)

        paths = {}

        paths["transaction_volume"] = self.plot_transaction_volume(
            transactions, calibration_end_week
        )
        paths["sales_distribution"] = self.plot_sales_distribution(transactions)
        paths["rfm_distributions"] = self.plot_rfm_distributions(rfm)
        paths["rfm_correlations"] = self.plot_rfm_correlations(rfm)
        paths["demographic_coverage"] = self.plot_demographic_coverage(merged_df)
        paths["demographic_distributions"] = self.plot_demographic_distributions(
            demographics
        )
        paths["monetary_comparison"] = self.plot_monetary_comparison(rfm)
        paths["customer_segments"] = self.plot_customer_segments(rfm)
        paths["coupon_impact"] = self.plot_coupon_impact(rfm)
        paths["calibration_holdout"] = self.plot_calibration_holdout(
            transactions, calibration_end_week
        )

        logger.info(f"\n  All {len(paths)} plots saved to {self.figures_dir}")
        return paths
