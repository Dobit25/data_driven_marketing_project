"""
modeling.py — RFM Segmentation & CLV Modeling
=============================================
Giải quyết Phần B của dự án Data Driven Marketing:
1. Chuẩn hóa các biến RFM.
2. Phân cụm K-Means và tìm K tối ưu bằng Silhouette Score.
3. Huấn luyện mô hình BG/NBD (và Gamma-Gamma) để dự báo CLV 6 tháng.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lifetimes import BetaGeoFitter, GammaGammaFitter
import warnings

warnings.filterwarnings("ignore")

# Cấu hình đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_OUTPUT = PROCESSED_DIR / "customer_labels.csv"

def load_rfm_data() -> pd.DataFrame:
    """Tải dữ liệu RFM đã được tạo từ Phần A."""
    file_path = INTERIM_DIR / "rfm_calibration.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {file_path}. Hãy chạy data_prep.py trước.")
    
    print(f"[*] Đang tải dữ liệu RFM từ: {file_path}")
    df = pd.read_csv(file_path)
    return df

def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Chuẩn hóa các biến R, F, M.
    2. Chạy K-Means, dùng Silhouette score để tìm K tối ưu (từ 2 đến 8).
    """
    print("\n" + "="*50)
    print(" BƯỚC 1 & 2: RFM SEGMENTATION (K-MEANS)")
    print("="*50)
    
    # Lấy các cột RFM (Recency, Frequency, Net_Sales cho Monetary)
    # Lưu ý: BG/NBD dùng Recency khác với K-Means truyền thống, nhưng ở đây
    # ta chuẩn hóa trực tiếp các giá trị đã tính.
    features = ['Recency', 'Frequency', 'Net_Sales']
    X = df[features].copy()
    
    # Xử lý các giá trị <= 0 cho Monetary và Frequency nếu muốn dùng Log transform (tuỳ chọn)
    # Ở đây áp dụng Standard Scaler trực tiếp theo yêu cầu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Tìm K tối ưu bằng Silhouette Score
    best_k = 2
    best_score = -1
    
    print("[*] Đang tìm K tối ưu bằng Silhouette Score...")
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"    K = {k} | Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\n[+] Đã chọn K tối ưu = {best_k} (Score cao nhất: {best_score:.4f})")
    
    # Chạy lại K-Means với K tối ưu
    optimal_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['Cluster'] = optimal_kmeans.fit_predict(X_scaled)
    
    # Lưu file customer_labels.csv
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labels_df = df[['household_key', 'Cluster']]
    labels_df.to_csv(LABELS_OUTPUT, index=False)
    print(f"[+] Đã lưu nhãn phân cụm tại: {LABELS_OUTPUT}")
    
    return df

def forecast_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    3. Dùng lifetimes library fit BG/NBD model để dự báo CLV 6 tháng (26 tuần).
    """
    print("\n" + "="*50)
    print(" BƯỚC 3: CLV MODELING (BG/NBD & GAMMA-GAMMA)")
    print("="*50)
    
    # Lọc dữ liệu: BG/NBD yêu cầu Frequency > 0 (khách hàng mua lặp lại)
    clv_df = df[df['Frequency'] > 0].copy()
    print(f"[*] Số lượng khách hàng mua lặp lại để huấn luyện: {len(clv_df)}/{len(df)}")
    
    # Fit mô hình BG/NBD (Dự báo số lượng giao dịch)
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(clv_df['Frequency'], clv_df['Recency'], clv_df['T'])
    print("[+] BG/NBD Model fitted successfully.")
    
    # Dự báo số lượng giao dịch trong 6 tháng (26 tuần)
    t = 26 
    clv_df['predicted_purchases_6_months'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, clv_df['Frequency'], clv_df['Recency'], clv_df['T']
    )
    
    # Fit mô hình Gamma-Gamma (Dự báo giá trị tiền tệ của mỗi giao dịch)
    # Cần lọc những khách hàng có avg_monetary > 0
    ggf_df = clv_df[clv_df['avg_monetary'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(ggf_df['Frequency'], ggf_df['avg_monetary'])
    print("[+] Gamma-Gamma Model fitted successfully.")
    
    # Tính toán CLV 6 tháng (Tính theo tuần, discount_rate = 0.01 theo mặc định)
    clv_df['predicted_clv_6_months'] = ggf.customer_lifetime_value(
        bgf,
        clv_df['Frequency'],
        clv_df['Recency'],
        clv_df['T'],
        clv_df['avg_monetary'],
        time=6, # tính theo tháng (nếu input là tháng) hoặc t theo tuần tuỳ cấu hình lifetimes
        freq='W', # Đơn vị thời gian của dữ liệu gốc là Tuần (Weeks)
        discount_rate=0.01
    )
    
    print("\n[Kết quả dự báo CLV cho 5 khách hàng đầu tiên]")
    cols_to_show = ['household_key', 'predicted_purchases_6_months', 'predicted_clv_6_months']
    print(clv_df[cols_to_show].head().to_string(index=False))
    
    return clv_df

if __name__ == "__main__":
    # Load data
    rfm_data = load_rfm_data()
    
    # Bước 1 & 2
    rfm_segmented = rfm_segmentation(rfm_data)
    
    # Bước 3
    rfm_clv = forecast_clv(rfm_segmented)
    
    # Lưu lại bộ dữ liệu hoàn chỉnh có cả Cluster và CLV
    final_output = PROCESSED_DIR / "rfm_clv_final.csv"
    rfm_clv.to_csv(final_output, index=False)
    print(f"\n[+] Hoàn tất! Dữ liệu tổng hợp lưu tại: {final_output}")