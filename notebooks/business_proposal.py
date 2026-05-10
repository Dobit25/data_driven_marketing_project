# %% [markdown]
# # ĐỀ XUẤT KINH DOANH — Tối ưu doanh thu dựa trên phân nhóm khách hàng
# **Project:** Dunnhumby CLV | **Focus:** Champions Segment

# %% Imports
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(".")
sys.path.insert(0, str(PROJECT))
plt.rcParams["figure.dpi"] = 130
sns.set_style("whitegrid")
FIGS = PROJECT / "reports" / "figures" / "segmentation"
FIGS.mkdir(parents=True, exist_ok=True)

SEG_COLORS = {"Champions":"#2ecc71","Loyal Customers":"#3498db","Promising":"#f39c12","Needs Attention":"#e74c3c"}
SEG_ORDER = list(SEG_COLORS.keys())

# %% Load Data
print("=" * 60)
print("  STEP 1: Load pre-processed RFM data")
print("=" * 60)

rfm = pd.read_csv(PROJECT / "data/interim/rfm_calibration.csv")
print(f"  RFM table: {len(rfm):,} households")

# %% K-Means K=4
print("\n" + "=" * 60)
print("  STEP 2: K-Means Segmentation (K=4)")
print("=" * 60)

feats = ["Recency","Frequency","avg_monetary"]
X = rfm[feats].copy()
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = km.fit_predict(X_sc)
print(f"  Silhouette: {silhouette_score(X_sc, rfm['Cluster']):.4f}")

centroids = rfm.groupby("Cluster")[feats].mean()
n = len(centroids)
score = (n+1 - centroids["Recency"].rank()) + centroids["Frequency"].rank() + centroids["avg_monetary"].rank()
sorted_c = score.sort_values(ascending=False).index.tolist()
label_map = {c: SEG_ORDER[i] for i, c in enumerate(sorted_c)}
rfm["Segment"] = rfm["Cluster"].map(label_map)

for seg in SEG_ORDER:
    g = rfm[rfm["Segment"]==seg]
    print(f"  {seg:20s}: {len(g):>5,} customers ({len(g)/len(rfm)*100:5.1f}%)")

# %% STEP 3: Revenue Analysis
print("\n" + "=" * 60)
print("  STEP 3: Revenue Analysis by Segment")
print("=" * 60)

rev = rfm.groupby("Segment").agg(
    n_customers=("household_key","count"),
    total_revenue=("Net_Sales","sum"),
    avg_revenue=("Net_Sales","mean"),
    avg_frequency=("Frequency","mean"),
    avg_recency=("Recency","mean"),
    avg_monetary=("avg_monetary","mean"),
    avg_basket=("avg_basket_size","mean"),
    avg_stores=("distinct_stores","mean"),
    avg_coupon=("coupon_usage_rate","mean"),
    avg_retail_disc=("retail_disc_usage_rate","mean"),
).loc[SEG_ORDER]

rev["revenue_pct"] = rev["total_revenue"] / rev["total_revenue"].sum() * 100
rev["customer_pct"] = rev["n_customers"] / rev["n_customers"].sum() * 100

print("\n  === BẢNG PHÂN TÍCH DOANH THU ===")
for seg in SEG_ORDER:
    r = rev.loc[seg]
    print(f"\n  [{seg}]")
    print(f"    Số KH: {r['n_customers']:,.0f} ({r['customer_pct']:.1f}%)")
    print(f"    Doanh thu: ${r['total_revenue']:,.0f} ({r['revenue_pct']:.1f}%)")
    print(f"    DT/KH: ${r['avg_revenue']:,.0f} | Frequency: {r['avg_frequency']:.0f} | Recency: {r['avg_recency']:.0f}w")
    print(f"    Avg Monetary: ${r['avg_monetary']:,.1f}/basket | Stores: {r['avg_stores']:.1f} | Coupon: {r['avg_coupon']:.1%}")

# %% Chart 1: Segment Size vs Revenue Contribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Customer Distribution vs Revenue Contribution by Segment", fontsize=14, fontweight="bold")

x = np.arange(4)
w = 0.35
colors = [SEG_COLORS[s] for s in SEG_ORDER]
axes[0].bar(x-w/2, rev["customer_pct"], w, color=colors, alpha=0.7, label="% Customers", edgecolor="white")
axes[0].bar(x+w/2, rev["revenue_pct"], w, color=colors, alpha=1.0, label="% Revenue", edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(SEG_ORDER, rotation=15, fontsize=9)
axes[0].set_ylabel("%"); axes[0].legend(); axes[0].set_title("% Customers vs % Revenue")
for i in range(4):
    axes[0].text(x[i]-w/2, rev["customer_pct"].iloc[i]+0.5, f'{rev["customer_pct"].iloc[i]:.1f}%', ha="center", fontsize=8)
    axes[0].text(x[i]+w/2, rev["revenue_pct"].iloc[i]+0.5, f'{rev["revenue_pct"].iloc[i]:.1f}%', ha="center", fontsize=8, fontweight="bold")

axes[1].bar(SEG_ORDER, rev["avg_revenue"], color=colors, edgecolor="white")
axes[1].set_title("Average Revenue / Customer ($)")
axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
for i, v in enumerate(rev["avg_revenue"]):
    axes[1].text(i, v+20, f"${v:,.0f}", ha="center", fontsize=9, fontweight="bold")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(FIGS / "01_segment_vs_revenue.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 01_segment_vs_revenue.png")

# %% Chart 2: Radar Chart
from math import pi
radar_cols = ["Recency","Frequency","avg_monetary","Net_Sales","coupon_usage_rate","retail_disc_usage_rate","distinct_stores"]
radar_labels = ["Recency\n(Lower=Better)","Frequency","Avg Monetary","Net Sales","Coupon Rate","Retail Disc Rate","Stores"]
rd = rfm.groupby("Segment")[radar_cols].mean().loc[SEG_ORDER]
rn = (rd - rd.min()) / (rd.max() - rd.min() + 1e-9)
rn["Recency"] = 1 - rn["Recency"]

angles = [n/float(len(radar_cols))*2*pi for n in range(len(radar_cols))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for seg in SEG_ORDER:
    vals = rn.loc[seg].tolist() + [rn.loc[seg].tolist()[0]]
    ax.plot(angles, vals, "o-", lw=2, label=seg, color=SEG_COLORS[seg])
    ax.fill(angles, vals, alpha=0.1, color=SEG_COLORS[seg])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0,1.1); ax.set_title("RFM Profile by Segment", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1))
plt.tight_layout()
plt.savefig(FIGS / "02_radar_rfm.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 02_radar_rfm.png")

# %% Chart 3: Champions Deep-Dive
champ = rfm[rfm["Segment"]=="Champions"]
others = rfm[rfm["Segment"]!="Champions"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"CHAMPIONS DEEP-DIVE ({len(champ):,} Customers — {len(champ)/len(rfm)*100:.1f}%)", fontsize=15, fontweight="bold", color="#2ecc71")

# Net Sales distribution
axes[0,0].hist(champ["Net_Sales"].clip(upper=champ["Net_Sales"].quantile(0.99)), bins=40, color="#2ecc71", alpha=0.8, edgecolor="white", label="Champions")
axes[0,0].hist(others["Net_Sales"].clip(upper=others["Net_Sales"].quantile(0.99)), bins=40, color="#bdc3c7", alpha=0.5, edgecolor="white", label="Others")
axes[0,0].set_title("Net Sales Distribution"); axes[0,0].set_xlabel("$"); axes[0,0].legend()

# Frequency
axes[0,1].hist(champ["Frequency"], bins=40, color="#2ecc71", alpha=0.8, edgecolor="white", label="Champions")
axes[0,1].hist(others["Frequency"], bins=40, color="#bdc3c7", alpha=0.5, edgecolor="white", label="Others")
axes[0,1].set_title("Frequency Distribution"); axes[0,1].legend()

# Avg monetary
axes[1,0].hist(champ["avg_monetary"], bins=40, color="#2ecc71", alpha=0.8, edgecolor="white", label="Champions")
axes[1,0].hist(others["avg_monetary"], bins=40, color="#bdc3c7", alpha=0.5, edgecolor="white", label="Others")
axes[1,0].set_title("Avg Monetary ($/basket)"); axes[1,0].legend()

# Coupon usage
axes[1,1].hist(champ["coupon_usage_rate"], bins=30, color="#2ecc71", alpha=0.8, edgecolor="white", label="Champions")
axes[1,1].hist(others["coupon_usage_rate"], bins=30, color="#bdc3c7", alpha=0.5, edgecolor="white", label="Others")
axes[1,1].set_title("Coupon Usage Rate"); axes[1,1].legend()

plt.tight_layout()
plt.savefig(FIGS / "03_champions_deepdive.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 03_champions_deepdive.png")

# %% Chart 4: Upgrade Funnel — Revenue Opportunity
fig, ax = plt.subplots(figsize=(12, 6))
champ_avg = rev.loc["Champions","avg_revenue"]
opportunity = []
for seg in SEG_ORDER[1:]:
    gap = champ_avg - rev.loc[seg,"avg_revenue"]
    n = rev.loc[seg,"n_customers"]
    upgrade_rate = {"Loyal Customers":0.20, "Promising":0.10, "Needs Attention":0.05}[seg]
    potential = gap * n * upgrade_rate
    opportunity.append({"Segment":seg, "Gap_per_customer":gap, "N":n, "Upgrade_rate":upgrade_rate, "Potential":potential})
    
opp = pd.DataFrame(opportunity)
bars = ax.barh(opp["Segment"], opp["Potential"], color=[SEG_COLORS[s] for s in opp["Segment"]], edgecolor="white", height=0.5)
for bar, pot, rate in zip(bars, opp["Potential"], opp["Upgrade_rate"]):
    ax.text(bar.get_width()+1000, bar.get_y()+bar.get_height()/2, f"${pot:,.0f} (upgrade {rate:.0%})", va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Potential Revenue ($)")
ax.set_title("REVENUE OPPORTUNITY — Upgrading Customers to Champions", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
plt.tight_layout()
plt.savefig(FIGS / "04_revenue_opportunity.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 04_revenue_opportunity.png")

# %% Chart 5: Strategy Priority Matrix
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(10, 8))
for seg in SEG_ORDER:
    r = rev.loc[seg]
    s_size = r["n_customers"] * 0.8
    ax.scatter(r["avg_revenue"], r["avg_frequency"], s=s_size,
               c=SEG_COLORS[seg], alpha=0.7, edgecolors="black", linewidth=1.5, label=seg, zorder=5)
    # Fix Text Overlapping
    ax.annotate(f"{seg}\n({r['n_customers']:,.0f} Cust)", 
                (r["avg_revenue"], r["avg_frequency"]),
                xytext=(0, np.sqrt(s_size)/2 + 10), # Offset dynamically based on bubble radius
                textcoords="offset points",
                fontsize=10, ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                zorder=10)

ax.set_xlabel("Average Revenue / Customer ($)", fontsize=12)
ax.set_ylabel("Average Purchase Frequency", fontsize=12)
ax.set_title("STRATEGIC PRIORITY MATRIX\n(Bubble Size = Number of Customers)", fontsize=14, fontweight="bold")

# Fix Legend Overlap (fixed marker size for all items in legend)
legend_elements = [Line2D([0], [0], marker='o', color='w', label=seg,
                          markerfacecolor=SEG_COLORS[seg], markersize=12, markeredgecolor='black', alpha=0.7)
                   for seg in SEG_ORDER]
ax.legend(handles=legend_elements, loc="upper left", title="Segments")

# Fix Clipping (add margin to axes limits)
ax.margins(0.15)

ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
plt.tight_layout()
plt.savefig(FIGS / "05_strategy_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 05_strategy_matrix.png")

# %% STEP 4: Business Recommendations
total_rev = rfm["Net_Sales"].sum()
champ_rev = champ["Net_Sales"].sum()
total_opp = opp["Potential"].sum()

print("\n" + "=" * 70)
print("  ĐỀ XUẤT KINH DOANH — TỐI ƯU DOANH THU")
print("=" * 70)

print(f"""
  TỔNG QUAN:
  ─────────────────────────────────────────────────────
  Tổng doanh thu (Calibration):     ${total_rev:>12,.0f}
  Doanh thu Champions:              ${champ_rev:>12,.0f} ({champ_rev/total_rev*100:.1f}%)
  Số KH Champions:                  {len(champ):>12,} ({len(champ)/len(rfm)*100:.1f}%)
  Doanh thu tiềm năng (upgrade):    ${total_opp:>12,.0f}
  ─────────────────────────────────────────────────────

  CHIẾN LƯỢC 1: GIỮ CHÂN CHAMPIONS (Retention)
  ├─ Mục tiêu: Giảm tỷ lệ rời bỏ xuống <2%
  ├─ Hành động:
  │   • Loyalty Program VIP: điểm thưởng x2, ưu đãi riêng
  │   • Early Access: sản phẩm mới, Flash Sale riêng
  │   • Personalized offers dựa trên lịch sử mua
  │   • Birthday rewards + Anniversary gifts
  ├─ KPI: Retention rate, Repeat purchase rate
  └─ ROI ước tính: Giữ 95% Champions = giữ ${champ_rev*0.95:,.0f} doanh thu

  CHIẾN LƯỢC 2: NÂNG CẤP LOYAL → CHAMPIONS (Upgrade)
  ├─ Mục tiêu: Chuyển 20% Loyal lên Champions
  ├─ Hành động:
  │   • Bundle deals: mua combo tăng basket size
  │   • Upselling: gợi ý sản phẩm premium
  │   • Cross-sell: danh mục bổ sung (MBA rules)
  │   • Frequency program: mua X lần/tháng = quà
  ├─ KPI: Avg basket size, Purchase frequency
  └─ ROI ước tính: +${opp.iloc[0]["Potential"]:,.0f}

  CHIẾN LƯỢC 3: KÍCH HOẠT PROMISING (Activation)
  ├─ Mục tiêu: Tăng frequency 30%
  ├─ Hành động:
  │   • Email/Push notification cá nhân hóa
  │   • Seasonal campaigns: Black Friday, Tết
  │   • Gamification: streak rewards (mua liên tục = bonus)
  │   • Category expansion: giới thiệu ngành hàng mới
  ├─ KPI: Visit frequency, Category breadth
  └─ ROI ước tính: +${opp.iloc[1]["Potential"]:,.0f}

  CHIẾN LƯỢC 4: WIN-BACK NEEDS ATTENTION (Re-engagement)
  ├─ Mục tiêu: Kéo lại 5% khách hàng sắp rời bỏ
  ├─ Hành động:
  │   • Win-back email: "Chúng tôi nhớ bạn" + discount code
  │   • Deep discount: giảm 20-30% cho lần mua tiếp
  │   • Survey: tìm hiểu lý do không quay lại
  │   • Retargeting ads trên social media
  ├─ KPI: Reactivation rate, Time to next purchase
  └─ ROI ước tính: +${opp.iloc[2]["Potential"]:,.0f}

  ═══════════════════════════════════════════════════════
  TỔNG DOANH THU TIỀM NĂNG TỪ 4 CHIẾN LƯỢC:
  ${champ_rev*0.95 + total_opp:>12,.0f}
  (Giữ Champions + Upgrade 3 nhóm còn lại)
  ═══════════════════════════════════════════════════════
""")

# %% Save summary to CSV
summary_data = []
for seg in SEG_ORDER:
    r = rev.loc[seg]
    summary_data.append({
        "Segment": seg,
        "Số KH": int(r["n_customers"]),
        "% KH": f"{r['customer_pct']:.1f}%",
        "Doanh thu": f"${r['total_revenue']:,.0f}",
        "% Doanh thu": f"{r['revenue_pct']:.1f}%",
        "DT/KH": f"${r['avg_revenue']:,.0f}",
        "Frequency TB": f"{r['avg_frequency']:.0f}",
        "Recency TB": f"{r['avg_recency']:.1f}",
        "Monetary TB": f"${r['avg_monetary']:,.1f}",
        "Retail Disc %": f"{r['avg_retail_disc']*100:.1f}%",
    })
pd.DataFrame(summary_data).to_csv(FIGS / "segment_summary.csv", index=False, encoding="utf-8-sig")
print("  Saved: segment_summary.csv")
print("\n  ✓ Hoàn tất! Tất cả charts đã lưu vào reports/figures/segmentation/")
