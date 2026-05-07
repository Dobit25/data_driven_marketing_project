"""Simulate K=4 clustering and analyze imputation impact."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rfm = pd.read_csv("data/interim/rfm_calibration.csv")

features = ["Recency", "Frequency", "avg_monetary"]
X = rfm[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = km4.fit_predict(X_scaled)

cluster_col = "Cluster"
sil = silhouette_score(X_scaled, rfm[cluster_col])
print(f"=== K=4 Cluster Analysis (Silhouette={sil:.4f}) ===")
print()

# Rank clusters by composite score for labeling
centroids = rfm.groupby(cluster_col)[features].mean()
n = len(centroids)
r_rank = centroids["Recency"].rank()
f_rank = centroids["Frequency"].rank()
m_rank = centroids["avg_monetary"].rank()
score = (n + 1 - r_rank) + f_rank + m_rank
sorted_clusters = score.sort_values(ascending=False).index.tolist()

labels = ["Champions", "Loyal Customers", "Promising", "Needs Attention"]
label_map = {c: labels[i] for i, c in enumerate(sorted_clusters)}

for c in sorted_clusters:
    g = rfm[rfm[cluster_col] == c]
    print(f"--- {label_map[c]} (Cluster {c}): {len(g)} customers ({len(g)/len(rfm)*100:.1f}%) ---")
    print(f"  Recency:       mean={g['Recency'].mean():.1f}, median={g['Recency'].median():.1f}")
    print(f"  Frequency:     mean={g['Frequency'].mean():.1f}, median={g['Frequency'].median():.1f}")
    print(f"  avg_monetary:  mean=${g['avg_monetary'].mean():.2f}, median=${g['avg_monetary'].median():.2f}")
    print(f"  Net_Sales:     mean=${g['Net_Sales'].mean():.2f}")
    print(f"  total_baskets: mean={g['total_baskets'].mean():.1f}")
    print(f"  tenure_weeks:  mean={g['tenure_weeks'].mean():.1f}")
    print(f"  coupon_rate:   mean={g['coupon_usage_rate'].mean():.3f}")
    print()

print()
print("=" * 60)
print("=== QUANTITY / STORE_ID overflow analysis ===")
print("=" * 60)

txn = pd.read_csv("data/raw/transaction_data.csv",
                   dtype={"QUANTITY": np.int32, "STORE_ID": np.int32},
                   usecols=["QUANTITY", "STORE_ID"])
print(f"Total rows: {len(txn):,}")
print()
neg_qty = (txn["QUANTITY"] < 0).sum()
neg_store = (txn["STORE_ID"] < 0).sum()
print(f"QUANTITY < 0:  {neg_qty:,} rows ({neg_qty/len(txn)*100:.3f}%)")
print(f"STORE_ID < 0:  {neg_store:,} rows ({neg_store/len(txn)*100:.3f}%)")
print()
print(f"QUANTITY range (int32): min={txn['QUANTITY'].min()}, max={txn['QUANTITY'].max()}")
print(f"STORE_ID range (int32): min={txn['STORE_ID'].min()}, max={txn['STORE_ID'].max()}")
print()

# Check if negative QUANTITY is returns (legitimate)
neg_qty_df = txn[txn["QUANTITY"] < 0]
if len(neg_qty_df) > 0:
    print(f"Negative QUANTITY distribution:")
    print(neg_qty_df["QUANTITY"].describe())
print()

# Demographics imputation impact analysis
print("=" * 60)
print("=== Demographics Imputation Impact ===")
print("=" * 60)
cal = pd.read_csv("data/processed/clv_features_calibration.csv")
print(f"Total rows: {len(cal)}")
if "has_demographics" in cal.columns:
    real = cal[cal["has_demographics"] == 1]
    imputed = cal[cal["has_demographics"] == 0]
    print(f"Real demographics: {len(real)} ({len(real)/len(cal)*100:.1f}%)")
    print(f"Imputed:           {len(imputed)} ({len(imputed)/len(cal)*100:.1f}%)")
    print()
    # Check diversity of imputed categorical columns
    for col in ["AGE_DESC", "INCOME_DESC", "MARITAL_STATUS_CODE", "HOMEOWNER_DESC", "HH_COMP_DESC"]:
        if col in cal.columns:
            real_unique = real[col].nunique()
            imp_unique = imputed[col].nunique()
            print(f"  {col}: real={real_unique} unique, imputed={imp_unique} unique")
            if imp_unique <= 2:
                top = imputed[col].value_counts(normalize=True).head(3)
                print(f"    Top values in imputed: {dict(top)}")
else:
    print("  Column 'has_demographics' not found.")
    print(f"  Columns available: {sorted(cal.columns.tolist())}")
