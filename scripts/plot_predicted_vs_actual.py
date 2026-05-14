"""
Generate a 3-panel Predicted vs Actual scatter plot for all CLV model tiers.
Output: reports/figures/predicted_vs_actual.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CLV_FILE = ROOT / "data" / "processed" / "rfm_clv_final.csv"
HOLDOUT_FILE = ROOT / "data" / "interim" / "rfm_holdout.csv"
OUTPUT_FILE = ROOT / "reports" / "figures" / "predicted_vs_actual.png"

TIERS = [
    ("predicted_clv_baseline",   "Tier 1: Rate-Based Baseline", "#E15759"),
    ("predicted_clv_supervised",  "Tier 2: XGBoost",             "#4E79A7"),
    ("predicted_clv_6m",          "Tier 3: MBG/NBD + Gamma-Gamma","#59A14F"),
]

# ── Load & Merge ─────────────────────────────────────────────────────────
clv = pd.read_csv(CLV_FILE)
holdout = pd.read_csv(HOLDOUT_FILE)[["household_key", "Net_Sales"]].rename(
    columns={"Net_Sales": "actual_clv"}
)
merged = clv.merge(holdout, on="household_key", how="inner").dropna(
    subset=["actual_clv"]
)
print(f"Merged: {len(merged)} households")

# ── Metrics ──────────────────────────────────────────────────────────────
def calc_metrics(actual, predicted):
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    nonzero = actual != 0
    mape = np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100
    return mae, rmse, mape

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Predicted vs Actual Holdout-Period CLV (Weeks 76–102)",
             fontsize=15, fontweight="bold", y=1.02)

global_max = max(
    merged["actual_clv"].max(),
    max(merged[col].max() for col, _, _ in TIERS if col in merged.columns)
)

for ax, (col, title, color) in zip(axes, TIERS):
    if col not in merged.columns:
        ax.set_title(f"{title}\n(not available)")
        continue

    sub = merged[["actual_clv", col]].dropna()
    actual = sub["actual_clv"]
    predicted = sub[col]
    mae, rmse, mape = calc_metrics(actual, predicted)

    ax.scatter(actual, predicted, alpha=0.25, s=12, color=color,
               edgecolors="none")

    # 45° perfect prediction line
    line_max = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, line_max], [0, line_max], "k--", lw=1.5, alpha=0.6,
            label="Perfect prediction")

    # Metrics box
    text = f"MAE  = ${mae:,.2f}\nRMSE = ${rmse:,.2f}\nMAPE = {mape:.1f}%"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85))

    ax.set_xlabel("Actual Holdout CLV ($)", fontsize=11)
    ax.set_ylabel("Predicted CLV ($)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    # Same scale for Tier 1 & 2 for fair comparison
    if col != "predicted_clv_6m":
        ax.set_xlim(-50, global_max * 0.55)
        ax.set_ylim(-50, global_max * 0.55)

plt.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT_FILE}")
