# %% [markdown]
# # Phân tích K-Means K=4 — Mô tả 4 nhóm khách hàng
# **Project:** Dunnhumby CLV Pipeline  
# **Date:** 2026-05-07  
# **Author:** Tung

# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "sans-serif"
sns.set_style("whitegrid")

# %% Load data
rfm = pd.read_csv("data/interim/rfm_calibration.csv")
print(f"Loaded {len(rfm):,} households × {rfm.shape[1]} features")
rfm.head()

# %% K-Means clustering with K=4
features = ["Recency", "Frequency", "avg_monetary"]

X = rfm[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = km.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, rfm["Cluster"])
print(f"Silhouette Score (K=4): {sil_score:.4f}")

# %% Assign business labels based on RFM composite score
centroids = rfm.groupby("Cluster")[features].mean()
n = len(centroids)
r_rank = centroids["Recency"].rank()       # low R = good
f_rank = centroids["Frequency"].rank()     # high F = good
m_rank = centroids["avg_monetary"].rank()  # high M = good
composite = (n + 1 - r_rank) + f_rank + m_rank
sorted_clusters = composite.sort_values(ascending=False).index.tolist()

label_pool = ["Champions", "Loyal Customers", "Promising", "Needs Attention"]
label_map = {c: label_pool[i] for i, c in enumerate(sorted_clusters)}
rfm["Segment"] = rfm["Cluster"].map(label_map)

# Define colors for each segment
segment_colors = {
    "Champions": "#2ecc71",
    "Loyal Customers": "#3498db",
    "Promising": "#f39c12",
    "Needs Attention": "#e74c3c",
}
segment_order = ["Champions", "Loyal Customers", "Promising", "Needs Attention"]

print("\n=== Segment Assignment ===")
for c in sorted_clusters:
    g = rfm[rfm["Cluster"] == c]
    print(f"  Cluster {c} -> {label_map[c]} ({len(g)} customers)")

# %% [markdown]
# ## 1. Tổng quan 4 Segment

# %% Summary statistics per segment
summary_cols = [
    "Recency", "Frequency", "avg_monetary", "Net_Sales",
    "total_baskets", "tenure_weeks", "coupon_usage_rate",
    "retail_disc_usage_rate", "T", "distinct_stores",
]

summary = rfm.groupby("Segment")[summary_cols].agg(["mean", "median", "std", "min", "max"])
# Reorder
summary = summary.loc[segment_order]
print("=== Summary Statistics per Segment ===")
for seg in segment_order:
    g = rfm[rfm["Segment"] == seg]
    print(f"\n--- {seg} ({len(g)} customers, {len(g)/len(rfm)*100:.1f}%) ---")
    for col in summary_cols:
        m = g[col].mean()
        md = g[col].median()
        prefix = "$" if "monetary" in col.lower() or "sales" in col.lower() else ""
        print(f"  {col:25s} mean={prefix}{m:>10.2f}   median={prefix}{md:>10.2f}")

# %% [markdown]
# ## 2. Biểu đồ phân bố số lượng khách hàng theo Segment

# %% Pie + Bar chart for segment sizes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
counts = rfm["Segment"].value_counts().loc[segment_order]
colors = [segment_colors[s] for s in segment_order]
wedges, texts, autotexts = axes[0].pie(
    counts, labels=None, autopct="%1.1f%%",
    colors=colors, startangle=90,
    wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11),
)
axes[0].set_title("Tỷ trọng khách hàng theo Segment", fontsize=13, fontweight="bold")
axes[0].legend(
    [f"{s} ({c:,})" for s, c in zip(segment_order, counts)],
    loc="lower left", fontsize=9,
)

# Bar chart
bars = axes[1].bar(segment_order, counts, color=colors, edgecolor="white", linewidth=1.5)
for bar, count in zip(bars, counts):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
        f"{count:,}", ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
axes[1].set_title("Số lượng khách hàng theo Segment", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Số khách hàng")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("reports/figures/k4_segment_sizes.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## 3. So sánh RFM trung bình giữa 4 Segment (Radar Chart)

# %% Radar chart
from math import pi

radar_cols = ["Recency", "Frequency", "avg_monetary", "Net_Sales", "coupon_usage_rate", "distinct_stores"]
radar_labels = ["Recency\n(thấp = tốt)", "Frequency", "Avg Monetary", "Net Sales", "Coupon Rate", "Distinct Stores"]

# Normalize to [0, 1] for radar
radar_data = rfm.groupby("Segment")[radar_cols].mean().loc[segment_order]
radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)
# Invert Recency (lower is better)
radar_norm["Recency"] = 1 - radar_norm["Recency"]

angles = [n / float(len(radar_cols)) * 2 * pi for n in range(len(radar_cols))]
angles += angles[:1]  # close the circle

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for seg in segment_order:
    values = radar_norm.loc[seg].tolist()
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=seg, color=segment_colors[seg])
    ax.fill(angles, values, alpha=0.1, color=segment_colors[seg])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title("RFM Profile theo Segment (Normalized)", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig("reports/figures/k4_radar_chart.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## 4. Boxplot so sánh phân phối từng chỉ số RFM

# %% Boxplots for key metrics
box_cols = ["Recency", "Frequency", "avg_monetary", "Net_Sales"]
box_titles = [
    "Recency (tuần kể từ lần mua cuối)",
    "Frequency (số lượt mua lặp lại)",
    "Avg Monetary ($/giỏ hàng)",
    "Net Sales ($, tổng chi tiêu)",
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, col, title in zip(axes.ravel(), box_cols, box_titles):
    data_to_plot = [rfm[rfm["Segment"] == seg][col].values for seg in segment_order]
    bp = ax.boxplot(
        data_to_plot,
        labels=segment_order,
        patch_artist=True,
        showfliers=False,  # hide outliers for clarity
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, seg in zip(bp["boxes"], segment_order):
        patch.set_facecolor(segment_colors[seg])
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    if "Sales" in col or "monetary" in col.lower():
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

plt.suptitle("Phân phối chỉ số RFM theo 4 Segment", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/figures/k4_boxplots.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## 5. Heatmap — Giá trị trung bình chuẩn hóa theo Segment

# %% Heatmap
heatmap_cols = [
    "Recency", "Frequency", "avg_monetary", "Net_Sales",
    "total_baskets", "tenure_weeks", "coupon_usage_rate",
    "retail_disc_usage_rate", "distinct_stores",
]
hm_data = rfm.groupby("Segment")[heatmap_cols].mean().loc[segment_order]

# TÍNH Z-SCORE CHUẨN: Dựa trên trung bình và độ lệch chuẩn của TOÀN BỘ tập khách hàng (Population)
# Cách cũ (tính z-score trên 4 dòng mean) là sai về mặt thống kê vì không xét đến trọng số khách hàng.
pop_mean = rfm[heatmap_cols].mean()
pop_std = rfm[heatmap_cols].std() + 1e-9

hm_norm = (hm_data - pop_mean) / pop_std

# Đảo ngược Recency: Recency thấp (âm z-score) = Tốt (xanh) -> Đổi dấu để đồng nhất màu
hm_norm["Recency"] = -hm_norm["Recency"]

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(
    hm_norm,
    annot=hm_data.round(1).values,  # show raw values
    fmt="",
    cmap="RdYlGn",  # Normal RdYlGn (Green = High Z-score = Good)
    center=0,
    linewidths=1,
    linecolor="white",
    ax=ax,
    yticklabels=segment_order,
    cbar_kws={"label": "Chỉ số kinh doanh (Xanh = Tốt, Đỏ = Kém)"},
)
ax.set_title(
    "Heatmap đánh giá mức độ Tốt/Kém của từng Segment\n(Màu = Độ tốt, Số = Giá trị thực)",
    fontsize=13, fontweight="bold",
)
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("reports/figures/k4_heatmap.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## 6. Scatter Plot — Frequency vs Monetary (màu theo Segment)

# %% Scatter plot
fig, ax = plt.subplots(figsize=(10, 7))

for seg in segment_order:
    g = rfm[rfm["Segment"] == seg]
    ax.scatter(
        g["Frequency"], g["avg_monetary"],
        c=segment_colors[seg], label=f"{seg} ({len(g)})",
        alpha=0.5, s=20, edgecolors="none",
    )

ax.set_xlabel("Frequency (số lượt mua lặp lại)", fontsize=12)
ax.set_ylabel("Avg Monetary ($/giỏ hàng)", fontsize=12)
ax.set_title("Frequency vs Monetary — phân bố 4 Segment", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

plt.tight_layout()
plt.savefig("reports/figures/k4_scatter_freq_monetary.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## 7. Silhouette Plot — Chất lượng phân cụm từng điểm

# %% Silhouette plot
sample_silhouette = silhouette_samples(X_scaled, rfm["Cluster"])
rfm["silhouette"] = sample_silhouette

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10

for seg in segment_order:
    mask = rfm["Segment"] == seg
    cluster_sils = rfm.loc[mask, "silhouette"].sort_values().values
    size = len(cluster_sils)
    y_upper = y_lower + size

    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0, cluster_sils,
        facecolor=segment_colors[seg], alpha=0.7,
        label=f"{seg} ({size})",
    )
    ax.text(-0.05, y_lower + 0.5 * size, seg, fontsize=9, va="center")
    y_lower = y_upper + 10

ax.axvline(x=sil_score, color="red", linestyle="--", lw=2, label=f"Avg = {sil_score:.3f}")
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Customers (sorted)", fontsize=12)
ax.set_title("Silhouette Plot — Chất lượng phân cụm K=4", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_yticks([])

plt.tight_layout()
plt.savefig("reports/figures/k4_silhouette_plot.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## 8. Bảng tổng kết chiến lược Marketing

# %% Marketing strategy table
print("=" * 90)
print(f"{'Segment':<20} {'Số KH':>7} {'%':>6} {'Chiến lược Marketing đề xuất'}")
print("=" * 90)

strategies = {
    "Champions": "Giữ chân bằng loyalty program, early access, personalized offers",
    "Loyal Customers": "Upselling cao cấp, cross-sell, tăng basket size",
    "Promising": "Tăng frequency qua email/push notification, seasonal campaigns",
    "Needs Attention": "Chống churn: win-back campaigns, discount codes, re-engagement",
}

for seg in segment_order:
    g = rfm[rfm["Segment"] == seg]
    pct = len(g) / len(rfm) * 100
    print(f"  {seg:<18} {len(g):>7,} {pct:>5.1f}%  {strategies[seg]}")

print("=" * 90)

# %%
print("\n✓ All charts saved to reports/figures/")
print("  - k4_segment_sizes.png")
print("  - k4_radar_chart.png")
print("  - k4_boxplots.png")
print("  - k4_heatmap.png")
print("  - k4_scatter_freq_monetary.png")
print("  - k4_silhouette_plot.png")
