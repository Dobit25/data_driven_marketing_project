"""
mba_builder.py — Market Basket Analysis Module
================================================
Automated Apriori-based association rule mining for Dunnhumby transactions.

Produces association rules with Support, Confidence, and Lift metrics,
exported to reports/market_basket_rules.csv.

Requires: mlxtend >= 0.23.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger


class MBABuilder:
    """Market Basket Analysis using Apriori algorithm.

    Builds a binary basket × product matrix from transactions,
    runs Apriori to find frequent itemsets, then generates
    association rules.

    Parameters
    ----------
    config : dict
        Configuration dictionary from config.yaml.
    min_support : float
        Minimum support threshold for Apriori (default: 0.01).
    min_confidence : float
        Minimum confidence threshold for rules (default: 0.2).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        min_support: float = 0.01,
        min_confidence: float = 0.2,
    ) -> None:
        self.config = config
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.reports_dir = Path(config["output"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def build_basket_matrix(
        self,
        transactions: pd.DataFrame,
        item_col: str = "DEPARTMENT",
        basket_col: str = "BASKET_ID",
    ) -> pd.DataFrame:
        """Create binary basket × item matrix.

        Parameters
        ----------
        transactions : pd.DataFrame
            Transaction-level data.
        item_col : str
            Column to use as items (default: DEPARTMENT for tractability).
        basket_col : str
            Column identifying unique baskets.

        Returns
        -------
        pd.DataFrame
            Binary matrix (rows=baskets, columns=items).
        """
        logger.info(f"Building basket matrix (item_col='{item_col}') ...")

        # Need product info — if DEPARTMENT not in transactions, we need to
        # merge. For this module, we assume it's already merged or we use
        # available columns.
        if item_col not in transactions.columns:
            logger.warning(
                f"  Column '{item_col}' not found. "
                f"Falling back to 'PRODUCT_ID' (may be slow for large datasets)."
            )
            item_col = "PRODUCT_ID"

        # Create basket-item pairs (deduplicate)
        basket_items = (
            transactions[[basket_col, item_col]]
            .drop_duplicates()
            .assign(purchased=1)
        )

        # Pivot to wide format
        basket_matrix = basket_items.pivot_table(
            index=basket_col,
            columns=item_col,
            values="purchased",
            fill_value=0,
            aggfunc="max",
        )

        # Convert to bool for mlxtend efficiency
        basket_matrix = basket_matrix.astype(bool)

        logger.info(
            f"  Basket matrix: {basket_matrix.shape[0]:,} baskets × "
            f"{basket_matrix.shape[1]} items"
        )
        return basket_matrix

    def run_apriori(self, basket_matrix: pd.DataFrame) -> pd.DataFrame:
        """Run Apriori algorithm to find frequent itemsets.

        Parameters
        ----------
        basket_matrix : pd.DataFrame
            Binary basket × item matrix from build_basket_matrix().

        Returns
        -------
        pd.DataFrame
            Frequent itemsets with support values.
        """
        from mlxtend.frequent_patterns import apriori

        logger.info(
            f"Running Apriori (min_support={self.min_support}) ..."
        )

        frequent_itemsets = apriori(
            basket_matrix,
            min_support=self.min_support,
            use_colnames=True,
        )

        logger.info(f"  Found {len(frequent_itemsets):,} frequent itemsets")
        return frequent_itemsets

    def generate_rules(
        self, frequent_itemsets: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate association rules from frequent itemsets.

        Parameters
        ----------
        frequent_itemsets : pd.DataFrame
            Output of run_apriori().

        Returns
        -------
        pd.DataFrame
            Association rules with antecedents, consequents,
            support, confidence, and lift.
        """
        from mlxtend.frequent_patterns import association_rules

        logger.info(
            f"Generating rules (min_confidence={self.min_confidence}) ..."
        )

        if len(frequent_itemsets) == 0:
            logger.warning("  No frequent itemsets found. Returning empty rules.")
            return pd.DataFrame(
                columns=["antecedents", "consequents", "support",
                         "confidence", "lift"]
            )

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
        )

        # Convert frozensets to strings for CSV compatibility
        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(str(i) for i in x))
        )
        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(str(i) for i in x))
        )

        # Sort by lift descending
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

        # Keep key columns
        rules = rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ]

        logger.info(f"  Generated {len(rules):,} rules")
        high_lift = (rules["lift"] > 1).sum()
        logger.info(f"  Rules with lift > 1: {high_lift:,}")

        return rules

    def save_rules(self, rules: pd.DataFrame) -> str:
        """Save association rules to CSV.

        Parameters
        ----------
        rules : pd.DataFrame
            Output of generate_rules().

        Returns
        -------
        str
            Path to saved CSV file.
        """
        output_path = self.reports_dir / "market_basket_rules.csv"
        rules.to_csv(output_path, index=False)
        logger.info(f"  Saved: {output_path} ({len(rules):,} rules)")
        return str(output_path)

    def run_all(
        self,
        transactions: pd.DataFrame,
        item_col: str = "DEPARTMENT",
    ) -> pd.DataFrame:
        """Execute full MBA pipeline.

        Parameters
        ----------
        transactions : pd.DataFrame
            Transaction-level data with BASKET_ID and item columns.
        item_col : str
            Column to use as item identifier.

        Returns
        -------
        pd.DataFrame
            Association rules DataFrame.
        """
        logger.info("=" * 60)
        logger.info("  Market Basket Analysis (Apriori)")
        logger.info("=" * 60)

        basket_matrix = self.build_basket_matrix(transactions, item_col=item_col)
        frequent_itemsets = self.run_apriori(basket_matrix)
        rules = self.generate_rules(frequent_itemsets)
        self.save_rules(rules)

        return rules
