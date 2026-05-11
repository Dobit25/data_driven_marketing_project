"""
=============================================================================
 SEGMENT-STRATIFIED QUASI-EXPERIMENT
 Campaign Targeting Impact on Customer Spending — By Customer Segment

 Design:
   - Segment customers into 4 groups using K-Means (K=4) on RFM
   - Within EACH segment:
       Treatment = Households targeted by >=1 campaign
       Control   = Households NOT targeted
       Outcome   = Holdout-period (Weeks 76-102) spending & behavior
   - Methods: Welch t-test, Mann-Whitney U, Cohen's d, Bootstrap CI
   - PSM applied to overall sample for robustness check

 Business Questions:
   Q1: Which segment benefits MOST from campaign targeting?
   Q2: Is campaign spending on Champions wasteful?
   Q3: Can campaigns reactivate Needs Attention customers?
   Q4: Where should marketing budget be allocated?

 Author: DDM Group 6
 Date: 2025-05-11
=============================================================================
"""

# %% --- IMPORTS ---
import sys, os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "font.family": "sans-serif"})

PROJECT  = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT / "data" / "raw"
DATA_INT = PROJECT / "data" / "interim"
FIG_DIR  = PROJECT / "reports" / "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Segment config (consistent with tung.py & business_proposal.py)
SEG_COLORS = {
    "Champions": "#2ecc71", "Loyal Customers": "#3498db",
    "Promising": "#f39c12", "Needs Attention": "#e74c3c"
}
SEG_ORDER = list(SEG_COLORS.keys())

# =====================================================================
# STEP 1: LOAD & PREPARE DATA
# =====================================================================
print("=" * 70)
print("  STEP 1: LOAD DATA")
print("=" * 70)

campaign_table = pd.read_csv(DATA_RAW / "campaign_table.csv")
transaction    = pd.read_csv(DATA_RAW / "transaction_data.csv")
rfm_cal        = pd.read_csv(DATA_INT / "rfm_calibration.csv")

print(f"  campaign_table: {len(campaign_table):,} rows")
print(f"  transaction:    {len(transaction):,} rows")
print(f"  rfm_cal:        {len(rfm_cal):,} households")

# =====================================================================
# STEP 2: K-MEANS SEGMENTATION (K=4)
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 2: K-MEANS SEGMENTATION (K=4)")
print("=" * 70)

features = ["Recency", "Frequency", "avg_monetary"]
X = rfm_cal[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_cal["Cluster"] = km.fit_predict(X_scaled)

# Assign labels using composite RFM score (same logic as tung.py)
centroids = rfm_cal.groupby("Cluster")[features].mean()
n = len(centroids)
composite = (n + 1 - centroids["Recency"].rank()) + \
            centroids["Frequency"].rank() + centroids["avg_monetary"].rank()
sorted_c = composite.sort_values(ascending=False).index.tolist()
label_map = {c: SEG_ORDER[i] for i, c in enumerate(sorted_c)}
rfm_cal["Segment"] = rfm_cal["Cluster"].map(label_map)

for seg in SEG_ORDER:
    g = rfm_cal[rfm_cal["Segment"] == seg]
    print(f"  {seg:20s}: {len(g):>5,} ({len(g)/len(rfm_cal)*100:5.1f}%)")

# =====================================================================
# STEP 3: COMPUTE HOLDOUT SPENDING (Weeks 76-102)
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 3: COMPUTE HOLDOUT-PERIOD METRICS")
print("=" * 70)

holdout_txn = transaction[transaction["WEEK_NO"] >= 76].copy()
holdout_txn["Net_Sales"] = (
    holdout_txn["SALES_VALUE"] + holdout_txn["RETAIL_DISC"]
    + holdout_txn["COUPON_DISC"] + holdout_txn["COUPON_MATCH_DISC"]
)

holdout = holdout_txn.groupby("household_key").agg(
    holdout_sales=("Net_Sales", "sum"),
    holdout_baskets=("BASKET_ID", "nunique"),
    holdout_weeks=("WEEK_NO", "nunique"),
).reset_index()

print(f"  Holdout HHs: {len(holdout):,}")

# =====================================================================
# STEP 4: ASSIGN TREATMENT/CONTROL + MERGE
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 4: ASSIGN TREATMENT/CONTROL BY SEGMENT")
print("=" * 70)

targeted_hh = set(campaign_table["household_key"].unique())

# Campaign count per HH
camp_count = campaign_table.groupby("household_key")["CAMPAIGN"].nunique() \
    .reset_index().rename(columns={"CAMPAIGN": "num_campaigns"})

# Merge everything
df = rfm_cal.merge(holdout, on="household_key", how="left")
df = df.merge(camp_count, on="household_key", how="left")
df["holdout_sales"]   = df["holdout_sales"].fillna(0)
df["holdout_baskets"]  = df["holdout_baskets"].fillna(0)
df["holdout_weeks"]    = df["holdout_weeks"].fillna(0)
df["num_campaigns"]    = df["num_campaigns"].fillna(0).astype(int)
df["treatment"]        = df["household_key"].isin(targeted_hh).astype(int)
df["has_holdout"]      = (df["holdout_sales"] > 0).astype(int)

# Per-segment summary
print(f"\n  {'Segment':<20s} {'Total':>6} {'Treat':>6} {'Ctrl':>6} {'Treat%':>7}")
print("  " + "-" * 50)
for seg in SEG_ORDER:
    g = df[df["Segment"] == seg]
    t = g["treatment"].sum()
    c = len(g) - t
    print(f"  {seg:<20s} {len(g):>6,} {t:>6,} {c:>6,} {t/len(g)*100:>6.1f}%")

# =====================================================================
# STEP 5: SEGMENT-STRATIFIED ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 5: SEGMENT-STRATIFIED EXPERIMENT RESULTS")
print("=" * 70)

results_list = []

for seg in SEG_ORDER:
    g = df[df["Segment"] == seg]
    treat = g[g["treatment"] == 1]["holdout_sales"]
    ctrl  = g[g["treatment"] == 0]["holdout_sales"]

    # Skip if either group too small
    if len(treat) < 5 or len(ctrl) < 5:
        print(f"\n  [{seg}] SKIPPED — insufficient sample")
        continue

    # Means
    t_mean, c_mean = treat.mean(), ctrl.mean()
    diff = t_mean - c_mean
    lift = diff / c_mean * 100 if c_mean > 0 else 0

    # Welch t-test
    t_stat, t_pval = stats.ttest_ind(treat, ctrl, equal_var=False)

    # Mann-Whitney U
    u_stat, u_pval = stats.mannwhitneyu(treat, ctrl, alternative="two-sided")

    # Cohen's d
    pooled = np.sqrt((treat.var()*(len(treat)-1) + ctrl.var()*(len(ctrl)-1))
                     / (len(treat) + len(ctrl) - 2))
    d = diff / pooled if pooled > 0 else 0

    # Bootstrap 95% CI
    np.random.seed(42)
    boot_diffs = []
    for _ in range(5000):
        bt = np.random.choice(treat.values, size=len(treat), replace=True)
        bc = np.random.choice(ctrl.values, size=len(ctrl), replace=True)
        boot_diffs.append(bt.mean() - bc.mean())
    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)

    # Retention metric: % with holdout purchases
    t_retain = g[g["treatment"] == 1]["has_holdout"].mean() * 100
    c_retain = g[g["treatment"] == 0]["has_holdout"].mean() * 100

    # Holdout frequency (baskets)
    t_freq = g[g["treatment"] == 1]["holdout_baskets"].mean()
    c_freq = g[g["treatment"] == 0]["holdout_baskets"].mean()

    sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else "n.s."

    results_list.append({
        "Segment": seg,
        "n_treat": len(treat), "n_ctrl": len(ctrl),
        "ARPU_treat": t_mean, "ARPU_ctrl": c_mean,
        "ATE": diff, "Lift_pct": lift,
        "t_stat": t_stat, "p_value": t_pval, "sig": sig,
        "u_stat": u_stat, "u_pval": u_pval,
        "cohens_d": d,
        "CI_lower": ci_lo, "CI_upper": ci_hi,
        "retention_treat": t_retain, "retention_ctrl": c_retain,
        "freq_treat": t_freq, "freq_ctrl": c_freq,
    })

    print(f"\n  === [{seg}] ===")
    print(f"  Treatment: n={len(treat):,}, ARPU=${t_mean:,.2f}, retention={t_retain:.1f}%, freq={t_freq:.1f}")
    print(f"  Control:   n={len(ctrl):,},  ARPU=${c_mean:,.2f}, retention={c_retain:.1f}%, freq={c_freq:.1f}")
    print(f"  ATE = ${diff:,.2f} ({lift:+.1f}%) | p={t_pval:.4f} {sig} | d={d:.3f}")
    print(f"  95% CI: [${ci_lo:,.2f}, ${ci_hi:,.2f}]")

results_df = pd.DataFrame(results_list)

# =====================================================================
# STEP 6: OVERALL PSM ANALYSIS (Robustness Check)
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 6: OVERALL PSM ANALYSIS (Robustness)")
print("=" * 70)

cov_cols = ["Recency", "Frequency", "avg_monetary", "Net_Sales",
            "total_baskets", "tenure_weeks", "coupon_usage_rate",
            "avg_basket_size", "distinct_stores", "T"]

df_m = df.dropna(subset=cov_cols).copy()
X_cov = StandardScaler().fit_transform(df_m[cov_cols].values)

lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_cov, df_m["treatment"].values)
df_m["pscore"] = lr.predict_proba(X_cov)[:, 1]

# 1:1 NN matching with caliper
treat_idx = df_m[df_m["treatment"] == 1].index
ctrl_pool = df_m[df_m["treatment"] == 0]

nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(ctrl_pool[["pscore"]].values)
dists, idxs = nn.kneighbors(df_m.loc[treat_idx, ["pscore"]].values)

used, ft, fc = set(), [], []
for ti, ci, d in zip(treat_idx, ctrl_pool.index[idxs.flatten()], dists.flatten()):
    if ci not in used and d < 0.1:
        ft.append(ti); fc.append(ci); used.add(ci)

mt = df_m.loc[ft, "holdout_sales"]
mc = df_m.loc[fc, "holdout_sales"]
psm_ate = mt.mean() - mc.mean()
psm_t, psm_p = stats.ttest_ind(mt.values, mc.values, equal_var=False)
psm_lift = psm_ate / mc.mean() * 100 if mc.mean() > 0 else 0

# Covariate balance
balance_data = []
for col in cov_cols:
    bef_smd = abs(df[df["treatment"]==1][col].mean() - df[df["treatment"]==0][col].mean()) / (df[col].std()+1e-9)
    aft_smd = abs(df_m.loc[ft, col].mean() - df_m.loc[fc, col].mean()) / (df_m[col].std()+1e-9)
    balance_data.append({"Covariate": col, "SMD_before": bef_smd, "SMD_after": aft_smd})
balance_df = pd.DataFrame(balance_data)

print(f"  Matched pairs: {len(ft):,}")
print(f"  PSM Treatment ARPU: ${mt.mean():,.2f}")
print(f"  PSM Control ARPU:   ${mc.mean():,.2f}")
print(f"  PSM ATE: ${psm_ate:,.2f} ({psm_lift:+.1f}%)")
print(f"  PSM p-value: {psm_p:.4f} {'(Significant)' if psm_p < 0.05 else '(Not significant)'}")
print(f"  Avg SMD before: {balance_df['SMD_before'].mean():.4f} -> after: {balance_df['SMD_after'].mean():.4f}")

# Sample size needed
sigma = mc.std()
mde = mc.mean() * 0.10
z_a, z_b = stats.norm.ppf(0.95), stats.norm.ppf(0.80)
n_req = 2 * sigma**2 * (z_a + z_b)**2 / mde**2

print(f"  Required n per arm (MDE=10%): {n_req:,.0f}")
print(f"  Actual n per arm: {len(ft):,} -> {'Sufficient' if len(ft) >= n_req else 'Underpowered'}")

# =====================================================================
# STEP 7: VISUALIZATIONS
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 7: GENERATING FIGURES")
print("=" * 70)

# --- Fig 1: ARPU by Segment (Treatment vs Control) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(results_df))
w = 0.35
colors_t = [SEG_COLORS[s] for s in results_df["Segment"]]
colors_c = ["#bdc3c7"] * len(results_df)

bars_t = axes[0].bar(x - w/2, results_df["ARPU_treat"], w, color=colors_t,
                      edgecolor="white", label="Treatment (Targeted)")
bars_c = axes[0].bar(x + w/2, results_df["ARPU_ctrl"], w, color=colors_c,
                      edgecolor="white", label="Control (Not targeted)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df["Segment"], rotation=15, fontsize=9)
axes[0].set_ylabel("ARPU ($)")
axes[0].set_title("Holdout ARPU: Treatment vs Control", fontweight="bold")
axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
axes[0].legend(fontsize=9)

for i, row in results_df.iterrows():
    axes[0].text(i - w/2, row["ARPU_treat"] + 20, f"${row['ARPU_treat']:,.0f}",
                 ha="center", fontsize=8, fontweight="bold")
    axes[0].text(i + w/2, row["ARPU_ctrl"] + 20, f"${row['ARPU_ctrl']:,.0f}",
                 ha="center", fontsize=8)

# ATE with CI
axes[1].barh(results_df["Segment"][::-1], results_df["ATE"][::-1],
             color=[SEG_COLORS[s] for s in results_df["Segment"][::-1]],
             edgecolor="white", height=0.5)
for i, (_, row) in enumerate(results_df[::-1].iterrows()):
    seg_i = len(results_df) - 1 - i
    axes[1].plot([row["CI_lower"], row["CI_upper"]],
                 [i, i], color="black", lw=2, zorder=5)
    axes[1].plot([row["CI_lower"], row["CI_upper"]],
                 [i, i], "|", color="black", ms=10, zorder=5)
    label = f"${row['ATE']:+,.0f} {row['sig']}"
    axes[1].text(max(row["CI_upper"], row["ATE"]) + 30, i, label,
                 va="center", fontsize=9, fontweight="bold")

axes[1].axvline(0, color="black", lw=1, ls=":")
axes[1].set_xlabel("ATE — Average Treatment Effect ($)")
axes[1].set_title("Campaign Effect by Segment (with 95% CI)", fontweight="bold")

plt.suptitle("Segment-Stratified Quasi-Experiment Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ab_segment_experiment.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ab_segment_experiment.png")

# --- Fig 2: Retention Rate by Segment ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
ax.bar(x - w/2, results_df["retention_treat"], w, color=colors_t,
       edgecolor="white", label="Treatment")
ax.bar(x + w/2, results_df["retention_ctrl"], w, color=colors_c,
       edgecolor="white", label="Control")
ax.set_xticks(x)
ax.set_xticklabels(results_df["Segment"], rotation=15)
ax.set_ylabel("Retention Rate (%)")
ax.set_title("Holdout Retention: Treatment vs Control by Segment", fontweight="bold")
ax.legend()
for i, row in results_df.iterrows():
    ax.text(i - w/2, row["retention_treat"] + 0.5, f"{row['retention_treat']:.1f}%",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, row["retention_ctrl"] + 0.5, f"{row['retention_ctrl']:.1f}%",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "ab_segment_retention.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ab_segment_retention.png")

# --- Fig 3: PSM Propensity Score + Covariate Balance ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_m.loc[df_m["treatment"]==1, "pscore"], bins=40, alpha=0.6,
             color="#2ecc71", edgecolor="white", label="Treatment", density=True)
axes[0].hist(df_m.loc[df_m["treatment"]==0, "pscore"], bins=40, alpha=0.6,
             color="#3498db", edgecolor="white", label="Control", density=True)
axes[0].set_title("Propensity Score Distribution", fontweight="bold")
axes[0].set_xlabel("Propensity Score")
axes[0].legend()

y_pos = range(len(balance_df))
axes[1].scatter(balance_df["SMD_before"], y_pos, marker="o", s=80, color="#e74c3c",
                label="Before PSM", zorder=5)
axes[1].scatter(balance_df["SMD_after"], y_pos, marker="s", s=80, color="#2ecc71",
                label="After PSM", zorder=5)
for i in y_pos:
    axes[1].plot([balance_df["SMD_before"].iloc[i], balance_df["SMD_after"].iloc[i]],
                 [i, i], color="gray", lw=0.8, alpha=0.5)
axes[1].axvline(0.1, color="#e67e22", ls="--", lw=1.5, alpha=0.7, label="Threshold")
axes[1].set_yticks(list(y_pos))
axes[1].set_yticklabels(balance_df["Covariate"], fontsize=9)
axes[1].set_xlabel("Standardized Mean Difference")
axes[1].set_title("Covariate Balance (Love Plot)", fontweight="bold")
axes[1].legend(loc="lower right", fontsize=8)
axes[1].invert_yaxis()

plt.suptitle("Propensity Score Matching — Robustness Check", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ab_psm_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ab_psm_diagnostics.png")

# =====================================================================
# STEP 8: EXPORT & SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 8: SUMMARY TABLE")
print("=" * 70)

summary = results_df[["Segment", "n_treat", "n_ctrl", "ARPU_treat", "ARPU_ctrl",
                       "ATE", "Lift_pct", "p_value", "sig", "cohens_d",
                       "CI_lower", "CI_upper",
                       "retention_treat", "retention_ctrl"]].copy()

# Pretty print
for _, row in summary.iterrows():
    print(f"\n  [{row['Segment']}]")
    print(f"    Sample:     Treatment={row['n_treat']:,} | Control={row['n_ctrl']:,}")
    print(f"    ARPU:       ${row['ARPU_treat']:,.2f} vs ${row['ARPU_ctrl']:,.2f}")
    print(f"    ATE:        ${row['ATE']:,.2f} ({row['Lift_pct']:+.1f}%) {row['sig']}")
    print(f"    p-value:    {row['p_value']:.4f} | Cohen's d: {row['cohens_d']:.3f}")
    print(f"    95% CI:     [${row['CI_lower']:,.2f}, ${row['CI_upper']:,.2f}]")
    print(f"    Retention:  {row['retention_treat']:.1f}% vs {row['retention_ctrl']:.1f}%")

print(f"\n  --- Overall PSM (Robustness) ---")
print(f"    Matched pairs: {len(ft):,}")
print(f"    PSM ATE: ${psm_ate:,.2f} ({psm_lift:+.1f}%), p={psm_p:.4f}")

# Export
results_df.to_csv(PROJECT / "reports" / "ab_segment_results.csv",
                   index=False, encoding="utf-8-sig")
balance_df.to_csv(PROJECT / "reports" / "ab_covariate_balance.csv",
                   index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("  DONE! Output files:")
print("    reports/ab_segment_results.csv")
print("    reports/ab_covariate_balance.csv")
print("    reports/figures/ab_segment_experiment.png")
print("    reports/figures/ab_segment_retention.png")
print("    reports/figures/ab_psm_diagnostics.png")
print("=" * 70)
