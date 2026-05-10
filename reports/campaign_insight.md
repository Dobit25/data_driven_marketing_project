# Phân Tích Chuyên Sâu: Hiệu Quả Chiến Dịch Coupon (Campaign Performance Evaluation)
*(Dữ liệu trích xuất từ `notebooks/campaign2_analysis.py` — 30 chiến dịch, 7,208 hộ gia đình được target)*

Hệ thống Dunnhumby triển khai 3 loại chiến dịch Marketing (TypeA, TypeB, TypeC) với mục tiêu phân phối Coupon đến khách hàng mục tiêu. Bản phân tích này đo lường hiệu quả thực tế của từng loại chiến dịch dựa trên dữ liệu giao dịch, quy đổi coupon, và doanh thu phát sinh.

> [!IMPORTANT]
> **Ghi chú kỹ thuật về phương pháp tính doanh thu:** Doanh thu của mỗi chiến dịch được tính bằng cách lọc các giao dịch thỏa mãn **đồng thời 3 điều kiện**: (1) Đúng hộ gia đình được target, (2) Giao dịch xảy ra trong khung thời gian chiến dịch [START_DAY, END_DAY], (3) Sản phẩm mua nằm trong danh sách sản phẩm của chiến dịch. Đây là phép đo **attribution chặt** — chỉ tính doanh thu có thể quy trực tiếp cho chiến dịch.

---

## 1. Giải Phẫu Các Chỉ Số Kỹ Thuật (Technical Metrics)

*   **Redemption Rate (Tỷ lệ quy đổi Coupon):** Tỷ lệ phần trăm hộ gia đình **thực sự đi quy đổi coupon** trên tổng số hộ gia đình được target. Ví dụ: Target 1,000 hộ, 142 hộ quay lại quy đổi → Redemption Rate = 14.2%. Đây là chỉ số đo lường **sức hấp dẫn** của coupon.
*   **ROI Proxy (Chỉ số hiệu quả đầu tư ước lượng):** Tính bằng `Net Revenue / Total Discount`. Ví dụ: Thu $457K doanh thu ròng trên $140K chi phí chiết khấu → ROI = 3.26. Chỉ số càng cao thì mỗi đồng chiết khấu sinh ra càng nhiều doanh thu ròng.
*   **Conversion Rate (Tỷ lệ chuyển đổi — Phễu):** Tỷ lệ phần trăm hộ gia đình quy đổi coupon trên tổng hộ gia đình targeted. Khác Redemption Rate ở chỗ Conversion đếm **số lượt quy đổi** (1 hộ có thể quy đổi nhiều coupon), còn Redemption đếm **số hộ duy nhất**.

---

## 2. Tổng Quan 3 Loại Chiến Dịch

| Chỉ số | TypeA | TypeB | TypeC |
|---|---|---|---|
| **Số chiến dịch** | 5 | 19 | 6 |
| **Avg Duration (ngày)** | 47.2 | 37.6 | 74.5 |
| **Avg HH targeted/campaign** | 796 | 140 | 96 |
| **Total Sales** | **$597,343** | $18,573 | $3,219 |
| **Avg Sales/Campaign** | **$119,469** | $978 | $537 |
| **Avg Redemption Rate** | **14.2%** | 7.8% | 9.1% |
| **Avg ROI Proxy** | 2.94 | 3.34 | **11.77** ⚠️ |
| **Total Net Revenue** | **$457,041** | $13,654 | $2,838 |

![Tổng quan hiệu suất theo loại chiến dịch](C:/Users/HOANG TUNG/.gemini/antigravity/brain/b5854316-0ad9-45d9-8a0f-3d2bd966a8b7/artifacts/campaign2_overview.png)

---

## 3. Phân Tích Chuyên Sâu Từng Loại Chiến Dịch

### 🏆 TypeA — "Chiến Dịch Voi Ma Mút" (Large-Scale Campaigns)
*   **Hồ sơ:** 5 chiến dịch, mỗi chiến dịch target trung bình **796 hộ gia đình** (gấp 5.7x TypeB). Thời gian chạy trung bình 47 ngày. Danh mục sản phẩm khuyến mãi khổng lồ (avg ~17,000-35,000 products/campaign).
*   **Kết quả:** Áp đảo hoàn toàn với **$597,343 tổng doanh thu** (chiếm 96.5% tổng doanh thu toàn bộ 30 chiến dịch). Redemption Rate cao nhất: 14.2%.
*   **Insight:** TypeA chiếm ưu thế tuyệt đối nhưng cần lưu ý **confounding variable**: Doanh thu cao có thể do quy mô target audience lớn gấp 5.7x chứ không hẳn do bản chất chiến dịch tốt hơn. Cần A/B testing với matched audience size để xác nhận.

### 📊 TypeB — "Chiến Dịch Sóng Ngắn" (High-Frequency, Small-Scale)
*   **Hồ sơ:** 19 chiến dịch (nhiều nhất), mỗi chiến dịch target trung bình chỉ **140 hộ gia đình**. Thời gian chạy ngắn nhất: avg 38 ngày, phần lớn đúng 32 ngày.
*   **Kết quả:** Doanh thu tổng thấp ($18,573) nhưng đây là kết quả tự nhiên của quy mô nhỏ. ROI Proxy = 3.34 — thực sự cao hơn TypeA (2.94), cho thấy hiệu quả chi phí tốt.
*   **Insight:** TypeB là loại chiến dịch "thử nghiệm nhanh" (rapid testing). Với duration ngắn và chi phí thấp, đây là format lý tưởng để **A/B test** các giả thuyết Marketing trước khi scale lên thành TypeA.

### ⚠️ TypeC — "Chiến Dịch Ma Trận" (Long-Duration, Unstable)
*   **Hồ sơ:** 6 chiến dịch, target trung bình chỉ **96 hộ gia đình**. Duration **không ổn định**: dao động từ 32 đến 161 ngày (std = 44 ngày).
*   **Kết quả:** Doanh thu tổng cực kỳ thấp ($3,219). ROI Proxy = 11.77 — **con số ẢO** vì total discount chỉ $382. Phép chia cho mẫu số quá nhỏ tạo ra kết quả bị thổi phồng.
*   **Insight:** TypeC không nên được dùng làm benchmark. Quy mô mẫu quá nhỏ khiến mọi kết luận thống kê đều không đáng tin cậy. Campaign #15 (duration 161 ngày, chỉ 17 HH targeted, doanh thu $419) là một outlier kéo lệch toàn bộ số liệu trung bình.

---

## 4. Phân Phối & Tương Quan (Distribution & Correlation)

### 4.1. Boxplot — Phân Phối Doanh Thu & Redemption Rate

![Phân phối doanh thu và tỷ lệ quy đổi theo loại chiến dịch](C:/Users/HOANG TUNG/.gemini/antigravity/brain/b5854316-0ad9-45d9-8a0f-3d2bd966a8b7/artifacts/campaign2_boxplots.png)

*   **TypeA Sales Boxplot:** Phương sai cực lớn (std = $109,134). 3 chiến dịch lõi (Campaign 8, 13, 18) tạo ra $141K - $246K, trong khi 2 chiến dịch nhỏ (26, 30) chỉ tạo ra $6K-$7K. Điều này cho thấy **không phải mọi TypeA đều thành công** — quy mô audience mới là yếu tố quyết định.
*   **Redemption Rate Boxplot:** TypeA có median ~15% và IQR từ ~10% đến ~18%, trong khi TypeB phân tán rộng hơn (2.5% đến 13%). TypeC có 1 outlier campaign #3 (16.7%) kéo lệch.

### 4.2. Duration vs Sales — Thời Gian Chạy Có Ảnh Hưởng Đến Doanh Thu?

![Mối quan hệ giữa thời gian chạy chiến dịch và doanh thu](C:/Users/HOANG TUNG/.gemini/antigravity/brain/b5854316-0ad9-45d9-8a0f-3d2bd966a8b7/artifacts/campaign2_duration_vs_sales.png)

*   **Kết luận:** Tương quan Duration vs Sales gần bằng 0 (r = 0.052). Việc kéo dài thời gian chạy chiến dịch **KHÔNG** tự động tăng doanh thu. Yếu tố quyết định là **quy mô audience** (r = 0.939 với Sales).

### 4.3. Net Revenue & ROI Proxy

![Doanh thu ròng và chỉ số ROI theo loại chiến dịch](C:/Users/HOANG TUNG/.gemini/antigravity/brain/b5854316-0ad9-45d9-8a0f-3d2bd966a8b7/artifacts/campaign2_roi.png)

*   TypeA tạo ra doanh thu ròng áp đảo ($457K) nhưng ROI Proxy thấp nhất (2.94) — do chi phí chiết khấu lớn ($140K).
*   TypeC ROI 11.77 là **ảo** vì mẫu số quá nhỏ ($382 tổng discount).

### 4.4. Heatmap Tổng Hợp

![Heatmap hiệu suất chuẩn hóa theo loại chiến dịch](C:/Users/HOANG TUNG/.gemini/antigravity/brain/b5854316-0ad9-45d9-8a0f-3d2bd966a8b7/artifacts/campaign2_heatmap.png)

*   Heatmap chuẩn hóa (0-1) cho thấy TypeA thống trị ở Sales và Audience Size, TypeC thống trị ở ROI và Duration — nhưng cả hai đều bị bias bởi quy mô mẫu.

---

## 5. Kiểm Định Thống Kê (Statistical Validation)

*   **Kruskal-Wallis Test** (kiểm định phi tham số — phù hợp vì sample size nhỏ, phân phối lệch):
    *   H-statistic = **13.23**, p-value = **0.0013**
    *   → **Có sự khác biệt có ý nghĩa thống kê** giữa doanh thu 3 loại chiến dịch (p < 0.05).
*   **Correlation Matrix:**
    *   `num_hh_targeted` ↔ `total_sales`: **r = 0.939** (tương quan cực mạnh)
    *   `DURATION` ↔ `total_sales`: **r = 0.052** (gần như không tương quan)
    *   → **Kết luận:** Quy mô audience quyết định doanh thu, không phải thời gian chạy.

---

## 6. Executive Business Solutions: Đề Xuất Chiến Lược

### Chiến lược 1: Scale TypeA Cho Các Đợt Campaign Lớn
*   TypeA với avg 796 HH/campaign tạo ra $119,469/campaign. Đây là format nên được sử dụng cho các đợt khuyến mãi mùa vụ (Black Friday, Tết, Back-to-School) khi cần tối đa hóa tổng doanh thu.
*   **Hành động:** Kết hợp TypeA với chiến lược **Loyalty Card** (từ `segment_insight.md`) — coupon TypeA nên được tích hợp trực tiếp vào Thẻ Thành Viên thay vì in tờ rơi, vì dữ liệu segment cho thấy tỷ lệ sử dụng Retail Discount tự động (>80%) vượt xa tỷ lệ sử dụng Coupon thủ công (<10%).

### Chiến lược 2: Dùng TypeB Làm "Phòng Thí Nghiệm"
*   Với duration ngắn (~32 ngày) và chi phí thấp, TypeB là format lý tưởng để **A/B test** các giả thuyết:
    *   Test 1: Coupon giảm giá 10% vs. Hoàn tiền vào thẻ $5 — cái nào có redemption rate cao hơn?
    *   Test 2: Target nhóm Champions vs. Promising — nhóm nào phản ứng tốt hơn với coupon?
*   **Hành động:** Chạy 2-3 chiến dịch TypeB song song với matched audience size (cùng 140 HH/campaign) để tách biệt hiệu ứng treatment.

### Chiến lược 3: Audit Lại TypeC Trước Khi Tiếp Tục
*   TypeC có duration không ổn định (32-161 ngày) và doanh thu không đáng kể ($537/campaign). Trước khi chạy thêm TypeC, cần trả lời:
    *   TypeC khác gì TypeA/B về mặt cơ chế? (Loại sản phẩm khác? Kênh phân phối khác?)
    *   Campaign #15 (161 ngày, $419 doanh thu) có phải là lỗi setup hay là do thiết kế?
*   **Hành động:** Tạm dừng TypeC cho đến khi có audit report rõ ràng.

---

## 7. Tổng Kết

TypeA là "Cỗ máy doanh thu" nhưng cần kiểm soát biến nhiễu (audience size). TypeB là "Phòng Lab" để thí nghiệm nhanh. TypeC cần được audit trước khi tiếp tục đầu tư. Mọi chiến dịch coupon trong tương lai nên được tích hợp vào hệ thống **Thẻ Thành Viên** để tận dụng tỷ lệ Retail Discount Usage >80% thay vì phát coupon giấy.
