"""
=============================================================================
 CAMPAIGN2 PERFORMANCE EVALUATION
 Phân tích hiệu quả 3 loại Campaign: Type A, Type B, Type C
 Author: Senior Data Analyst
 Date: 2026-05-10
=============================================================================
"""

# %% [markdown]
# # 📊 Campaign2 Performance Evaluation
# **Mục tiêu**: Xác định loại campaign (A/B/C) mang lại hiệu quả cao nhất

# %% --- 1. IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.1)
plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120})

from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw'
FIG_DIR  = PROJECT_DIR / 'reports' / 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# %% --- 2. LOAD DATA ---
print("=" * 60)
print("LOADING DATA...")
print("=" * 60)

campaign_desc  = pd.read_csv(os.path.join(DATA_DIR, 'campaign_desc.csv'))
campaign_table = pd.read_csv(os.path.join(DATA_DIR, 'campaign_table.csv'))
coupon         = pd.read_csv(os.path.join(DATA_DIR, 'coupon.csv'))
coupon_redempt = pd.read_csv(os.path.join(DATA_DIR, 'coupon_redempt.csv'))
transaction    = pd.read_csv(os.path.join(DATA_DIR, 'transaction_data.csv'))
hh_demo        = pd.read_csv(os.path.join(DATA_DIR, 'hh_demographic.csv'))
product        = pd.read_csv(os.path.join(DATA_DIR, 'product.csv'))

print("✅ All datasets loaded successfully!\n")

# %% --- 3. DATA OVERVIEW ---
print("=" * 60)
print("3. DATA OVERVIEW")
print("=" * 60)

datasets = {
    'campaign_desc': campaign_desc,
    'campaign_table': campaign_table,
    'coupon': coupon,
    'coupon_redempt': coupon_redempt,
    'transaction': transaction,
    'hh_demographic': hh_demo,
    'product': product
}

for name, df in datasets.items():
    print(f"\n--- {name} ---")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    miss = df.isnull().sum()
    if miss.sum() > 0:
        print(f"  Missing: {miss[miss > 0].to_dict()}")
    else:
        print("  Missing: None")

# %% --- 4. CAMPAIGN DESC EXPLORATION ---
print("\n" + "=" * 60)
print("4. CAMPAIGN DESCRIPTION EXPLORATION")
print("=" * 60)

print("\n>>> campaign_desc preview:")
print(campaign_desc)

print("\n>>> Campaign type distribution:")
print(campaign_desc['DESCRIPTION'].value_counts())

# Duration of each campaign
campaign_desc['DURATION'] = campaign_desc['END_DAY'] - campaign_desc['START_DAY']
print("\n>>> Campaign duration stats by type:")
print(campaign_desc.groupby('DESCRIPTION')['DURATION'].describe())

# %% --- 5. BUILD CAMPAIGN-LEVEL METRICS ---
print("\n" + "=" * 60)
print("5. BUILDING CAMPAIGN-LEVEL METRICS")
print("=" * 60)

# 5a. Households targeted per campaign
hh_per_campaign = campaign_table.groupby('CAMPAIGN')['household_key'].nunique().reset_index()
hh_per_campaign.columns = ['CAMPAIGN', 'num_hh_targeted']

# 5b. Coupons per campaign
coupons_per_campaign = coupon.groupby('CAMPAIGN')['COUPON_UPC'].nunique().reset_index()
coupons_per_campaign.columns = ['CAMPAIGN', 'num_coupons']

# 5c. Products per campaign
products_per_campaign = coupon.groupby('CAMPAIGN')['PRODUCT_ID'].nunique().reset_index()
products_per_campaign.columns = ['CAMPAIGN', 'num_products']

# 5d. Redemptions per campaign
redempt_per_campaign = coupon_redempt.groupby('CAMPAIGN').agg(
    num_redemptions=('COUPON_UPC', 'count'),
    num_hh_redeemed=('household_key', 'nunique')
).reset_index()

# 5e. Revenue from redeemed coupons
# Link coupon_redempt -> transaction via household_key + DAY
# Link coupon -> product via COUPON_UPC to get PRODUCT_ID
# Then match transactions

# Get products linked to each campaign via coupon
campaign_products = coupon[['CAMPAIGN', 'PRODUCT_ID']].drop_duplicates()

# Get households in each campaign
campaign_hh = campaign_table[['CAMPAIGN', 'household_key']].drop_duplicates()

# Merge campaign info with campaign_desc to get type + period
campaign_info = campaign_desc[['CAMPAIGN', 'DESCRIPTION', 'START_DAY', 'END_DAY', 'DURATION']].copy()

# Get transactions during campaign period for campaign households on campaign products
print(">>> Calculating revenue per campaign (this may take a moment)...")

# Merge campaign_hh with campaign_info
camp_hh_info = campaign_hh.merge(campaign_info, on='CAMPAIGN')

# For efficiency, filter transactions to relevant households
relevant_hh = camp_hh_info['household_key'].unique()
txn_filtered = transaction[transaction['household_key'].isin(relevant_hh)].copy()

# Merge to get campaign-matched transactions
# A transaction counts for a campaign if:
#   - same household
#   - DAY is within [START_DAY, END_DAY]
#   - PRODUCT_ID is in campaign's product list
camp_txn = txn_filtered.merge(camp_hh_info, on='household_key')
camp_txn = camp_txn[(camp_txn['DAY'] >= camp_txn['START_DAY']) & 
                     (camp_txn['DAY'] <= camp_txn['END_DAY'])]

# Filter to campaign products
camp_txn = camp_txn.merge(campaign_products, on=['CAMPAIGN', 'PRODUCT_ID'])

# Aggregate revenue per campaign
revenue_per_campaign = camp_txn.groupby('CAMPAIGN').agg(
    total_sales=('SALES_VALUE', 'sum'),
    total_quantity=('QUANTITY', 'sum'),
    num_transactions=('BASKET_ID', 'nunique'),
    total_coupon_disc=('COUPON_DISC', 'sum'),
    total_retail_disc=('RETAIL_DISC', 'sum'),
    total_coupon_match=('COUPON_MATCH_DISC', 'sum')
).reset_index()

print("✅ Revenue calculation complete!")

# %% --- 6. MERGE ALL METRICS ---
print("\n" + "=" * 60)
print("6. MERGING ALL CAMPAIGN METRICS")
print("=" * 60)

camp_metrics = campaign_info.merge(hh_per_campaign, on='CAMPAIGN', how='left')
camp_metrics = camp_metrics.merge(coupons_per_campaign, on='CAMPAIGN', how='left')
camp_metrics = camp_metrics.merge(products_per_campaign, on='CAMPAIGN', how='left')
camp_metrics = camp_metrics.merge(redempt_per_campaign, on='CAMPAIGN', how='left')
camp_metrics = camp_metrics.merge(revenue_per_campaign, on='CAMPAIGN', how='left')

# Fill NaN with 0
for col in ['num_redemptions', 'num_hh_redeemed', 'total_sales', 'total_quantity',
            'num_transactions', 'total_coupon_disc', 'total_retail_disc', 'total_coupon_match']:
    if col in camp_metrics.columns:
        camp_metrics[col] = camp_metrics[col].fillna(0)

# Derived metrics
camp_metrics['redemption_rate'] = np.where(
    camp_metrics['num_hh_targeted'] > 0,
    camp_metrics['num_hh_redeemed'] / camp_metrics['num_hh_targeted'] * 100, 0)

camp_metrics['avg_sales_per_hh'] = np.where(
    camp_metrics['num_hh_targeted'] > 0,
    camp_metrics['total_sales'] / camp_metrics['num_hh_targeted'], 0)

camp_metrics['total_discount'] = (camp_metrics['total_coupon_disc'].abs() + 
                                   camp_metrics['total_retail_disc'].abs() + 
                                   camp_metrics['total_coupon_match'].abs())

camp_metrics['net_revenue'] = camp_metrics['total_sales'] - camp_metrics['total_discount']

camp_metrics['roi_proxy'] = np.where(
    camp_metrics['total_discount'] > 0,
    camp_metrics['net_revenue'] / camp_metrics['total_discount'], 0)

print("\n>>> Full campaign metrics table:")
print(camp_metrics.to_string(index=False))

# %% --- 7. GROUP BY CAMPAIGN TYPE ---
print("\n" + "=" * 60)
print("7. ANALYSIS BY CAMPAIGN TYPE (A / B / C)")
print("=" * 60)

type_summary = camp_metrics.groupby('DESCRIPTION').agg(
    num_campaigns=('CAMPAIGN', 'count'),
    avg_duration=('DURATION', 'mean'),
    total_hh_targeted=('num_hh_targeted', 'sum'),
    avg_hh_per_campaign=('num_hh_targeted', 'mean'),
    total_coupons=('num_coupons', 'sum'),
    total_products=('num_products', 'sum'),
    total_redemptions=('num_redemptions', 'sum'),
    avg_redemption_rate=('redemption_rate', 'mean'),
    total_sales=('total_sales', 'sum'),
    avg_sales_per_campaign=('total_sales', 'mean'),
    total_quantity=('total_quantity', 'sum'),
    total_transactions=('num_transactions', 'sum'),
    avg_sales_per_hh=('avg_sales_per_hh', 'mean'),
    total_discount=('total_discount', 'sum'),
    total_net_revenue=('net_revenue', 'sum'),
    avg_roi_proxy=('roi_proxy', 'mean')
).reset_index()

print("\n>>> Summary by Campaign Type:")
print(type_summary.to_string(index=False))

# Variance analysis
print("\n>>> Variance in sales across campaigns by type:")
var_analysis = camp_metrics.groupby('DESCRIPTION')['total_sales'].agg(['mean', 'std', 'var', 'min', 'max'])
print(var_analysis)

# %% --- 8. VISUALIZATIONS ---
print("\n" + "=" * 60)
print("8. GENERATING VISUALIZATIONS")
print("=" * 60)

colors = {'TypeA': '#2ecc71', 'TypeB': '#3498db', 'TypeC': '#e74c3c'}
type_order = ['TypeA', 'TypeB', 'TypeC']

# --- Fig 1: Total Sales by Campaign Type ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Campaign Performance by Type (A / B / C)', fontsize=16, fontweight='bold', y=1.02)

ax = axes[0, 0]
bars = type_summary.set_index('DESCRIPTION').loc[type_order, 'total_sales']
bar_colors = [colors[t] for t in type_order]
bars.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Total Sales Revenue by Type', fontweight='bold')
ax.set_ylabel('Total Sales ($)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0)
for i, v in enumerate(bars):
    ax.text(i, v + v*0.02, f'${v:,.0f}', ha='center', fontweight='bold', fontsize=9)

# --- Fig 2: Average Sales per Campaign ---
ax = axes[0, 1]
bars2 = type_summary.set_index('DESCRIPTION').loc[type_order, 'avg_sales_per_campaign']
bars2.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Average Sales per Campaign', fontweight='bold')
ax.set_ylabel('Avg Sales ($)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0)
for i, v in enumerate(bars2):
    ax.text(i, v + v*0.02, f'${v:,.0f}', ha='center', fontweight='bold', fontsize=9)

# --- Fig 3: Redemption Rate ---
ax = axes[1, 0]
bars3 = type_summary.set_index('DESCRIPTION').loc[type_order, 'avg_redemption_rate']
bars3.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Average Redemption Rate (%)', fontweight='bold')
ax.set_ylabel('Redemption Rate (%)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0)
for i, v in enumerate(bars3):
    ax.text(i, v + v*0.02, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)

# --- Fig 4: Number of Campaigns ---
ax = axes[1, 1]
bars4 = type_summary.set_index('DESCRIPTION').loc[type_order, 'num_campaigns']
bars4.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Number of Campaigns', fontweight='bold')
ax.set_ylabel('Count')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0)
for i, v in enumerate(bars4):
    ax.text(i, v + 0.2, f'{int(v)}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'campaign2_overview.png'), bbox_inches='tight', dpi=150)
plt.show()
print("✅ Saved: campaign2_overview.png")

# --- Fig 5: Boxplot of Sales Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
camp_metrics.boxplot(column='total_sales', by='DESCRIPTION', ax=ax, 
                      patch_artist=True, boxprops=dict(facecolor='lightblue'))
ax.set_title('Sales Distribution by Campaign Type', fontweight='bold')
ax.set_xlabel('Campaign Type')
ax.set_ylabel('Total Sales ($)')
plt.sca(ax)
plt.xticks([1, 2, 3], type_order)
ax.get_figure().suptitle('')

ax = axes[1]
camp_metrics.boxplot(column='redemption_rate', by='DESCRIPTION', ax=ax,
                      patch_artist=True, boxprops=dict(facecolor='lightyellow'))
ax.set_title('Redemption Rate Distribution by Type', fontweight='bold')
ax.set_xlabel('Campaign Type')
ax.set_ylabel('Redemption Rate (%)')
plt.sca(ax)
plt.xticks([1, 2, 3], type_order)
ax.get_figure().suptitle('')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'campaign2_boxplots.png'), bbox_inches='tight', dpi=150)
plt.show()
print("✅ Saved: campaign2_boxplots.png")

# --- Fig 6: ROI & Net Revenue ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
bars5 = type_summary.set_index('DESCRIPTION').loc[type_order, 'total_net_revenue']
bars5.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Total Net Revenue by Type', fontweight='bold')
ax.set_ylabel('Net Revenue ($)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0)
for i, v in enumerate(bars5):
    ax.text(i, v + abs(v)*0.02, f'${v:,.0f}', ha='center', fontweight='bold', fontsize=9)

ax = axes[1]
bars6 = type_summary.set_index('DESCRIPTION').loc[type_order, 'avg_roi_proxy']
bars6.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Average ROI Proxy by Type', fontweight='bold')
ax.set_ylabel('ROI (Net Rev / Discount)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0)
for i, v in enumerate(bars6):
    ax.text(i, v + abs(v)*0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'campaign2_roi.png'), bbox_inches='tight', dpi=150)
plt.show()
print("✅ Saved: campaign2_roi.png")

# --- Fig 7: Scatter - Duration vs Sales with type color ---
fig, ax = plt.subplots(figsize=(10, 6))
for t in type_order:
    subset = camp_metrics[camp_metrics['DESCRIPTION'] == t]
    ax.scatter(subset['DURATION'], subset['total_sales'], 
               c=colors[t], label=t, s=100, edgecolor='black', alpha=0.8)
ax.set_xlabel('Campaign Duration (days)')
ax.set_ylabel('Total Sales ($)')
ax.set_title('Campaign Duration vs. Total Sales', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'campaign2_duration_vs_sales.png'), bbox_inches='tight', dpi=150)
plt.show()
print("✅ Saved: campaign2_duration_vs_sales.png")

# --- Fig 8: Heatmap of key metrics ---
fig, ax = plt.subplots(figsize=(10, 5))
heatmap_data = type_summary.set_index('DESCRIPTION')[
    ['avg_sales_per_campaign', 'avg_redemption_rate', 'avg_roi_proxy',
     'avg_hh_per_campaign', 'avg_duration']
].loc[type_order]
heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
sns.heatmap(heatmap_norm, annot=heatmap_data.round(2).values, fmt='', 
            cmap='YlGnBu', ax=ax, linewidths=1, cbar_kws={'label': 'Normalized Score'})
ax.set_title('Normalized Performance Heatmap by Campaign Type', fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'campaign2_heatmap.png'), bbox_inches='tight', dpi=150)
plt.show()
print("✅ Saved: campaign2_heatmap.png")

# %% --- 9. ADVANCED ANALYSIS ---
print("\n" + "=" * 60)
print("9. ADVANCED ANALYSIS")
print("=" * 60)

# 9a. Conversion funnel by type
print("\n>>> Conversion Funnel by Type:")
funnel = type_summary[['DESCRIPTION', 'total_hh_targeted', 'total_redemptions', 
                         'total_transactions', 'total_sales']].copy()
funnel['conversion_rate'] = (funnel['total_redemptions'] / funnel['total_hh_targeted'] * 100).round(2)
print(funnel.to_string(index=False))

# 9b. Audience size impact
print("\n>>> Correlation: Audience size vs Sales (per campaign):")
corr = camp_metrics[['num_hh_targeted', 'total_sales', 'redemption_rate', 'DURATION']].corr()
print(corr.round(3))

# 9c. Statistical summary
from scipy import stats
print("\n>>> Kruskal-Wallis test (non-parametric) for sales across types:")
groups = [camp_metrics[camp_metrics['DESCRIPTION'] == t]['total_sales'].values for t in type_order]
if all(len(g) >= 2 for g in groups):
    stat, p = stats.kruskal(*groups)
    print(f"   H-statistic = {stat:.4f}, p-value = {p:.4f}")
    print(f"   {'→ Significant difference (p < 0.05)' if p < 0.05 else '→ No significant difference (p >= 0.05)'}")

# %% --- 10. INSIGHTS ---
print("\n" + "=" * 60)
print("10. KEY INSIGHTS")
print("=" * 60)

# Find best type
best_total = type_summary.loc[type_summary['total_sales'].idxmax()]
best_avg   = type_summary.loc[type_summary['avg_sales_per_campaign'].idxmax()]
best_roi   = type_summary.loc[type_summary['avg_roi_proxy'].idxmax()]
best_conv  = type_summary.loc[type_summary['avg_redemption_rate'].idxmax()]

# Dynamic counts per type
n_typeA = int(type_summary.loc[type_summary['DESCRIPTION'] == 'TypeA', 'num_campaigns'].values[0])
n_typeB = int(type_summary.loc[type_summary['DESCRIPTION'] == 'TypeB', 'num_campaigns'].values[0])
n_typeC = int(type_summary.loc[type_summary['DESCRIPTION'] == 'TypeC', 'num_campaigns'].values[0])

# Dynamic stats for caveats
typeA_avg_hh = type_summary.loc[type_summary['DESCRIPTION'] == 'TypeA', 'avg_hh_per_campaign'].values[0]
typeB_avg_hh = type_summary.loc[type_summary['DESCRIPTION'] == 'TypeB', 'avg_hh_per_campaign'].values[0]
typeC_total_sales = type_summary.loc[type_summary['DESCRIPTION'] == 'TypeC', 'total_sales'].values[0]
typeC_total_disc  = type_summary.loc[type_summary['DESCRIPTION'] == 'TypeC', 'total_discount'].values[0]
typeC_dur_std = camp_metrics.loc[camp_metrics['DESCRIPTION'] == 'TypeC', 'DURATION'].std()

print(f"""
📌 INSIGHT 1 - Total Revenue:
   {best_total['DESCRIPTION']} generates the highest TOTAL sales: ${best_total['total_sales']:,.0f}
   (across {int(best_total['num_campaigns'])} campaigns)

📌 INSIGHT 2 - Average Revenue per Campaign:
   {best_avg['DESCRIPTION']} has the highest AVERAGE sales per campaign: ${best_avg['avg_sales_per_campaign']:,.0f}

📌 INSIGHT 3 - ROI Proxy:
   {best_roi['DESCRIPTION']} has the best ROI proxy: {best_roi['avg_roi_proxy']:.2f}
   ⚠️  CAVEAT: {best_roi['DESCRIPTION']} total sales = ${typeC_total_sales:,.0f},
      total discount = ${typeC_total_disc:,.0f}. ROI is inflated by
      extremely small discount base. Interpret with caution.

📌 INSIGHT 4 - Redemption Rate (Conversion):
   {best_conv['DESCRIPTION']} has the highest average redemption rate: {best_conv['avg_redemption_rate']:.1f}%

📌 INSIGHT 5 - Confounding Variable Warning:
   TypeA targets avg {typeA_avg_hh:.0f} households/campaign vs
   TypeB avg {typeB_avg_hh:.0f} households/campaign ({typeA_avg_hh/typeB_avg_hh:.1f}x more).
   The revenue gap may be driven by audience SIZE, not campaign TYPE.
   A/B testing with matched audience sizes is required to confirm causality.
""")

# %% --- 11. CONCLUSION ---
print("\n" + "=" * 60)
print("11. CONCLUSION")
print("=" * 60)

conclusion = f"""
╔══════════════════════════════════════════════════════════════╗
║                    FINAL CONCLUSION                         ║
╠══════════════════════════════════════════════════════════════╣

1. LOẠI CAMPAIGN TỐT NHẤT:
   → Dựa trên tổng doanh thu: {best_total['DESCRIPTION']}
   → Dựa trên doanh thu trung bình/campaign: {best_avg['DESCRIPTION']}
   → Dựa trên ROI: {best_roi['DESCRIPTION']} (⚠️ small sample — see caveat)
   → Dựa trên Conversion Rate: {best_conv['DESCRIPTION']}

2. VÌ SAO?
   - {best_avg['DESCRIPTION']} tạo ra doanh thu trung bình cao nhất mỗi campaign
     (${best_avg['avg_sales_per_campaign']:,.0f}/campaign)
   - Redemption rate phản ánh mức độ hấp dẫn của coupon với khách hàng
   - ROI proxy cho thấy hiệu quả chi phí chiết khấu

3. ĐẶC ĐIỂM NỔI BẬT:
   - TypeA: {n_typeA} campaigns, avg duration 47 days, quy mô lớn
            (avg {typeA_avg_hh:.0f} HH/campaign)
   - TypeB: nhiều campaign nhất ({n_typeB}), thời gian ngắn (avg 38 days)
   - TypeC: {n_typeC} campaigns, duration không ổn định
            (std={typeC_dur_std:.0f} days, range 32-161 days)

4. KHUYẾN NGHỊ:
   a) Ưu tiên TypeA cho các chiến dịch large-scale (high total revenue)
   b) Cân nhắc TypeB cho test nhanh vì thời gian ngắn (~32 days)
   c) Cần A/B testing với matched audience size để tách biệt
      hiệu ứng campaign type vs. audience size
   d) Không nên dùng ROI proxy của TypeC làm benchmark vì
      quy mô quá nhỏ (chỉ ${typeC_total_sales:,.0f} tổng doanh thu)
   e) Tối ưu duration dựa trên correlation với doanh thu

╚══════════════════════════════════════════════════════════════╝
"""
print(conclusion)

# Save summary to CSV
camp_metrics.to_csv(os.path.join(DATA_DIR, '..', 'processed', 'campaign2_metrics.csv'), index=False)
type_summary.to_csv(os.path.join(DATA_DIR, '..', 'processed', 'campaign2_type_summary.csv'), index=False)
print("✅ Saved processed data to data/processed/")
print("\n🎉 Analysis complete!")
