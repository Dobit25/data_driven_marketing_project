# Đánh giá & Phân tích Insight: Quasi-Experiment & Propensity Score Matching (A/B Testing)
*(Báo cáo Giải mã luồng Causal Inference từ file `ab_experiment.py`)*

Đây là tài liệu đính kèm giúp toàn đội nắm vững 100% logic của file `ab_experiment.py` - "vũ khí hạng nặng" giúp nâng tầm bài báo cáo từ một đồ án Data thông thường thành một nghiên cứu Marketing khoa học đúng chuẩn.

---

## 1. Tại sao dự án của chúng ta BẮT BUỘC phải có file này?

Trước khi có file này, chúng ta đã có `segment_insight.md` (đưa ra các chiến lược lý thuyết) và `campaign_insight.md` (phân tích tương quan). Tuy nhiên, chúng ta gặp một **"lỗ hổng tử huyệt"**: Chúng ta không có bằng chứng nào chứng minh các chiến dịch Marketing trong quá khứ của Dunnhumby *thực sự có tác dụng*, hay do khách hàng tự bỏ tiền ra mua.

**Vấn đề ngụy biện nhân quả (Correlation vs Causation):**
Nếu chỉ nhìn sơ qua dữ liệu, ta thấy: Khách hàng nhận Campaign tiêu tiền GẤP 3 LẦN khách không nhận. Ai không hiểu biết sẽ vỗ tay ăn mừng: *"Campaign tăng 200% doanh thu!"*. 

File `ab_experiment.py` sinh ra để **chống lại sự ngụy biện đó**. Nó sử dụng thuật toán **Propensity Score Matching (PSM)** để chứng minh sự thật bẽ bàng: Khách hàng tiêu nhiều tiền là do bản thân họ vốn dĩ đã là "khách sộp" rồi, và hệ thống của Dunnhumby chỉ chăm chăm nhắm vào khách sộp để gửi khuyến mãi (gọi là **Selection Bias - Thiên vị chọn mẫu**).

---

## 2. Giải phẫu chi tiết các Biểu đồ & Kết quả (Deep-Dive Analysis)

File `ab_experiment.py` sinh ra 3 biểu đồ và 2 bảng dữ liệu trong thư mục `reports/`. Dưới đây là cách giải thích chúng:

### A. Biểu đồ 1: `ab_segment_experiment.png` (ARPU & ATE theo Segment)
*   **Trực quan:** 
    *   *Bên trái:* Cột doanh thu (ARPU) so sánh giữa người được nhận Campaign (Màu sắc) và người không nhận (Màu xám) trong từng cụm khách hàng.
    *   *Bên phải:* Mức độ ảnh hưởng trung bình (ATE - Average Treatment Effect) kèm thanh sai số 95% (Confidence Interval).
*   **Cách giải thích số liệu:**
    *   **Nhóm Loyal & Promising:** Cột màu cao hơn hẳn cột xám, $p$-value cực nhỏ ($<0.001$). Nhìn qua có vẻ Campaign cực kỳ hiệu quả với 2 nhóm này (Tăng +\$1,075 cho Loyal và +\$464 cho Promising).
    *   **Nhóm Champions:** Cột xám lại cao hơn cột màu, nhưng thanh sai số bên phải đâm xuyên qua trục 0 ($p=0.44$). Nghĩa là: Gửi hay không gửi Campaign cho nhóm này thì họ vẫn tiêu tiền như điên. Sự khác biệt là vô nghĩa về mặt thống kê.

### B. Biểu đồ 2: `ab_psm_diagnostics.png` (Kiểm định Chất lượng PSM)
*   **Trực quan:** 
    *   *Bên trái:* Biểu đồ Histogram so sánh Điểm Xu Hướng (Propensity Score) giữa 2 nhóm.
    *   *Bên phải (Love Plot):* Biểu đồ chấm bi so sánh các chỉ số của khách hàng Trước (Đỏ) và Sau (Xanh) khi dùng thuật toán ghép cặp.
*   **Cách giải thích số liệu:**
    *   *Love Plot:* Biểu đồ này dùng để "khè" hội đồng chấm thi. Chấm màu Đỏ nằm văng tít ra xa, chứng tỏ trước khi ghép cặp, người nhận Campaign và người không nhận là 2 thế giới khác hẳn nhau (người nhận toàn là đại gia, người không nhận toàn khách lèo tèo). Chấm màu Xanh bám sát vào đường đứt nét số 0, chứng tỏ thuật toán PSM đã hoạt động xuất sắc: Nó tìm ra được các cặp khách hàng **giống hệt nhau về mọi mặt (Recency, Frequency, Net Sales...)**, chỉ khác mỗi cái là 1 người có nhận Campaign, 1 người không.
    *   **Cú "Plot Twist":** Sau khi ghép cặp thành công (chấm xanh), thuật toán tính lại hiệu quả Campaign thì nhận ra: **ATE tụt xuống chỉ còn -\$27 (âm 27 đô), $p=0.73$ (Không có ý nghĩa)**. Vậy là lòi đuôi chuột: Cả hệ thống Campaign chả có tác dụng gì, sự chênh lệch ban đầu hoàn toàn là do Selection Bias!

### C. Biểu đồ 3: `ab_segment_retention.png` (Tỷ lệ Giữ chân theo Segment)
*   **Trực quan:** Biểu đồ cột thể hiện tỷ lệ % khách hàng có quay lại mua hàng.
*   **Cách giải thích số liệu:**
    *   Dù Campaign không làm tăng doanh thu đột biến (như đã phân tích ở trên), nhưng biểu đồ này vớt vát lại được một tia hy vọng: **Campaign giúp giữ chân khách hàng**.
    *   Đặc biệt ở nhóm **Needs Attention (Sắp rời bỏ)**: Nhóm nhận Campaign có tỷ lệ quay lại là **88.9%**, so với chỉ 77.9% ở nhóm không nhận. 

---

## 3. Tóm tắt Ý nghĩa Kinh doanh (Business Impact) cho Project

Kết quả từ file này không "đá nhau" với các file `.md` trước đó, mà nó làm cho các đề xuất trước đó trở nên **sắc bén và có minh chứng thép**:

1.  **Chứng minh "Bệnh thành tích" của hệ thống cũ:** Thuật toán phát hiện ra Dunnhumby target tới **97.8% nhóm Champions** nhưng lại bỏ xó nhóm **Needs Attention (chỉ target 8.7%)**. Đây là cách đánh Marketing rất an toàn, lười biếng và lãng phí (vì Champions đằng nào cũng mua, target họ chỉ tốn tiền in Coupon).
2.  **Bảo vệ Đề xuất Chiến lược "Win-back":** Phân tích ở trên (Biểu đồ 3) cho thấy tín hiệu cực tốt về Retention ở nhóm Needs Attention. Đây là cơ sở toán học vững chắc để khẳng định: *"Siêu thị phải chuyển ngay ngân sách từ việc gửi Campaign cho nhóm Champions sang nhóm Needs Attention để cứu vớt khách hàng"*.
3.  **Tôn vinh sự chặt chẽ của nhóm (Rigorous Analytics):** Thay vì bịa số liệu A/B Test (như bản nháp ban đầu của báo cáo), việc sử dụng Data có sẵn kết hợp Quasi-Experiment (PSM) cho thấy nhóm có năng lực Data Science cực kỳ sâu sắc, biết hoài nghi dữ liệu và không bị lừa bởi bề nổi của số liệu thống kê (Naive lift).

---

## 4. Lời khuyên khi Thuyết trình (Pitching to Stakeholders)

Khi slide đến phần "Promotional Experiment", bạn hãy nói theo kịch bản "Bóc hành" (Lật mặt vấn đề):

> *"Thưa Hội đồng, nếu chỉ nhìn bề nổi của dữ liệu, chúng ta sẽ lầm tưởng rằng các chiến dịch Marketing của siêu thị đang rất thành công khi giúp tăng hơn 200% doanh thu ở nhóm khách hàng trung thành. Tuy nhiên, bằng việc áp dụng thuật toán Propensity Score Matching (như trên biểu đồ Love Plot), chúng em phát hiện ra một sự thật: Đó chỉ là sự ngụy biện. Siêu thị đang thiên vị, chỉ gửi mã giảm giá cho những khách hàng 'nhà giàu', những người vốn dĩ đã tiêu nhiều tiền. Khi chúng em ghép cặp những khách hàng tương đương nhau để so sánh, hiệu ứng tăng doanh thu bằng KHÔNG. Từ kết quả 'đắng cay' này, chúng em mạnh dạn đề xuất thay đổi toàn bộ chiến lược phân bổ ngân sách: Cắt ngân sách của nhóm Champions (nhóm 97.8% bị lãng phí) và dồn lực vào chiến dịch Win-back cho nhóm Needs Attention (hiện đang bị bỏ rơi ở mức 8.7%)."*

---

## 5. Bản đồ Liên kết Insights (Kết nối các file `.md` trong thư mục)

Đừng đọc file này một cách rời rạc. Để thuyết trình một cách thuyết phục nhất, bạn hãy hình dung chuỗi logic kết nối toàn bộ các file insight trong thư mục `reports/` như sau:

1.  🎯 **[segment_insight.md](./segment_insight.md)** *(Chẩn đoán khách hàng)*
    *   Chia khách hàng thành 4 nhóm. Phát hiện ra nhóm **Needs Attention** đang gặp nguy hiểm và nhóm **Promising** có tiềm năng nâng cấp lớn nhất để kiếm thêm **$875K**.
2.  📢 **[campaign_insight.md](./campaign_insight.md)** *(Kiểm tra diện rộng)*
    *   Phát hiện ra quy mô tệp khách hàng (Audience Size) quyết định doanh thu, chứ không phải do loại Campaign. Đặt ra nghi vấn: *Liệu có phải Dunnhumby chỉ đang nhắm vào khách nhà giàu?*
3.  ⚖️ **[experiment_insight.md](./experiment_insight.md) (File này)** *(Kết án và Bẻ lái chiến lược)*
    *   **"Cú vả" thực tế:** Dùng thuật toán PSM chứng minh nghi vấn ở file số 2 là đúng. Phản biện lại thói quen rải coupon vô tội vạ cho nhóm Champions ở file số 1.
    *   **Chốt hạ:** Chuyển ngân sách cứu lấy nhóm Needs Attention (Win-back).
4.  🛒 **[market_basket_rules_insight.md](./market_basket_rules_insight.md)** *(Thực thi)*
    *   Bây giờ khi đã biết phải làm gì (Win-back, Upsell), chúng ta dùng kết quả từ file này (Quy luật kết hợp giỏ hàng) để biết chính xác nên **bán combo gì, giảm giá mặt hàng nào** cho từng đối tượng.

*Quy trình khép kín: Tìm ra nhóm mục tiêu $\rightarrow$ Nghi ngờ cách làm cũ $\rightarrow$ Chứng minh bằng thuật toán $\rightarrow$ Hành động bằng combo sản phẩm thực tế!*
