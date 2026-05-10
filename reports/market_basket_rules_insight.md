# Phân Tích Chuyên Sâu: Chiến Lược Bán Chéo Từ Market Basket Analysis (MBA)
*(Dữ liệu trích xuất từ `reports/market_basket_rules.csv` — 1,006 luật kết hợp)*

Sau khi tinh lọc hàng triệu giao dịch thông qua thuật toán Apriori, hệ thống đã phát hiện ra các quy luật mua sắm ngầm (Association Rules) của khách hàng. Không giống như việc phân chia khách hàng (Segmentation) tập trung vào "Ai mua", MBA tập trung vào **"Mua cái gì cùng nhau"**. 

Dưới đây là bản phân tích chuyên sâu (Deep-dive Insight) về các luật kết hợp mạnh nhất và cách biến chúng thành doanh thu thực tế.

> [!IMPORTANT]
> **Ghi chú kỹ thuật về đơn vị phân tích:** Toàn bộ các "items" xuất hiện trong file `market_basket_rules.csv` (như MEAT, DELI, PRODUCE...) đều là **tên Ngành hàng (Department)**, không phải mã sản phẩm cụ thể (SKU). Ví dụ: `MEAT` đại diện cho toàn bộ ngành Thịt tươi (gồm hàng trăm SKU: thịt bò, thịt heo, thịt gà...), còn `DRUG GM` (Drug & General Merchandise) bao gồm từ thuốc, mỹ phẩm, cho tới đồ dùng gia đình. Điều này ảnh hưởng trực tiếp đến cách diễn giải Insight: các quy luật phản ánh xu hướng mua sắm **liên ngành hàng**, chứ không phải liên sản phẩm đơn lẻ. MBA được chạy ở cấp Department thay vì cấp SKU là vì lý do khả thi tính toán (hàng triệu giao dịch × hàng chục nghìn SKU sẽ tạo ra ma trận quá lớn).

---

## 1. Giải phẫu các Chỉ số Kỹ thuật (Technical Metrics)

Trước khi đi vào chiến lược, Data Team và Business Team cần thống nhất góc nhìn về 3 chỉ số tạo nên một "Luật" trong `market_basket_rules.csv`:

*   **Support (Độ phủ):** Tỷ lệ phần trăm giỏ hàng chứa toàn bộ combo ngành hàng này trên tổng số toàn bộ giỏ hàng (baskets) của siêu thị. Ví dụ: Support = 0.015 nghĩa là cứ 1,000 giỏ hàng (baskets) thì có 15 giỏ chứa combo này. Dù có vẻ nhỏ, nhưng nhân với tổng lượng giao dịch khổng lồ của hệ thống, đây là một tần suất đáng kể.
*   **Confidence (Độ tự tin/Tỷ lệ chuyển đổi):** Xác suất giỏ hàng sẽ chứa nhóm ngành B, nếu đã chứa nhóm ngành A. Ví dụ: Nếu giỏ hàng đã có (DELI + GROCERY + MEAT), có tới 34.4% khả năng nó sẽ đồng thời chứa thêm (PRODUCE + MEAT-PCKGD + DRUG GM). Đây chính là "Tỷ lệ chốt sale" tự nhiên, không cần can thiệp Marketing.
*   **Lift (Độ nâng / Mức độ đột biến):** Đây là **chỉ số quan trọng nhất**. Lift = 1 nghĩa là việc mua A và B hoàn toàn ngẫu nhiên, không liên quan. Lift > 1 nghĩa là A "kích thích" mua B. **Lift > 4.0** (như trong dữ liệu của chúng ta) là một mức độ liên kết **cực kỳ mạnh**, chứng tỏ đây là một thói quen mua sắm có chủ đích (ví dụ: mua đồ về chuẩn bị bữa ăn gia đình).

---

## 2. Lựa chọn 3 Quy Luật Chiến Lược (Strategic Rule Selection)

Trong 1,006 luật kết hợp được phát hiện, có tới **34 luật có Lift vượt ngưỡng 4.0** (liên kết cực kỳ mạnh). Tuy nhiên, top 3 Lift cao nhất tuyệt đối (Lift 4.58, 4.55, 4.53) đều xoay quanh **cùng một cụm hành vi** (DELI + MEAT), chỉ khác nhau ở vị trí ngành GROCERY trong phương trình. Nếu trình bày cả 3 trong report, chúng ta chỉ đang mô tả 1 hiện tượng dưới 3 góc nhìn khác nhau, không tạo ra giá trị hành động mới.

Vì vậy, chúng tôi **chủ động chọn 3 luật đại diện cho 3 nhóm hành vi mua sắm hoàn toàn khác biệt** để tối đa hóa số chiến dịch Marketing có thể triển khai song song:

### 🏆 Quy luật 1: Bữa Tiệc Thịnh Soạn — Combo "DELI + MEAT" (Lift cao nhất trong nhóm)
*   **Nếu khách mua (Antecedents):** `DELI` (Đồ ăn sẵn/Đồ nguội) + `MEAT` (Ngành Thịt tươi)
*   **Chắc chắn mua thêm (Consequents):** `DRUG GM` (Đồ dùng chung) + `GROCERY` (Tạp hóa) + `MEAT-PCKGD` (Ngành Thịt đóng gói) + `PRODUCE` (Ngành Rau củ quả)
*   **Các chỉ số:** Lift = **4.547** | Confidence = **34.0%** | Support = 1.47%
*   **Vì sao chọn luật này thay vì luật Lift cao nhất (4.577)?** Luật Lift 4.577 yêu cầu khách đã mua 3 thứ (DELI + GROCERY + MEAT), tức GROCERY đã nằm trong giỏ rồi — không còn cơ hội bán chéo GROCERY nữa. Luật này chỉ cần detect **2 ngành hàng** (DELI + MEAT) trong giỏ nhưng gợi ý được **4 ngành hàng**, bao gồm cả GROCERY — tối đa hóa cơ hội bán chéo.
*   **Insight:** Đây là hành vi mua sắm chuẩn bị cho bữa ăn gia đình lớn cuối tuần. Khách hàng đã quyết định "ăn sang" (mua Thịt tươi + Đồ nguội) thì xác suất cực cao họ sẽ ghé qua quầy Rau củ (Produce), Tạp hóa (Grocery — gia vị, nước sốt), Thịt đóng gói (xúc xích, thịt nguội dự trữ), và đồ dùng gia đình (Drug GM — khăn giấy, túi đựng...).

### 🥈 Quy luật 2: Bữa Ăn Dinh Dưỡng Cao Cấp — Combo "MEAT + NUTRITION"
*   **Nếu khách mua:** `MEAT` (Ngành Thịt tươi) + `NUTRITION` (Ngành Đồ dinh dưỡng/Thực phẩm chức năng)
*   **Chắc chắn mua thêm:** `GROCERY` (Tạp hóa) + `MEAT-PCKGD` (Thịt đóng gói) + `PRODUCE` (Rau củ quả)
*   **Các chỉ số:** Lift = **4.181** | Confidence = **52.6% (Cao nhất trong 3 luật)** | Support = 1.05%
*   **Insight:** Đây là tệp khách hàng rất quan tâm đến sức khỏe (nhặt đồ Nutrition và Thịt tươi). Tỷ lệ Confidence **lên tới gần 53%** nghĩa là: Cứ 2 giỏ hàng chứa đồ Dinh dưỡng & Thịt, thì có hơn 1 giỏ sẽ **TỰ ĐỘNG** chứa thêm Rau củ, Tạp hóa và Thịt đóng gói. Đây là mức chuyển đổi tự nhiên cực kỳ cao — Marketing gần như không cần can thiệp gì thêm, chỉ cần đảm bảo các ngành hàng này nằm trên cùng tuyến đường di chuyển (Aisle) trong siêu thị.

### 🥉 Quy luật 3: Bữa Brunch / Dã Ngoại — Combo "MEAT + PASTRY"
*   **Nếu khách mua:** `MEAT` (Ngành Thịt tươi) + `PASTRY` (Ngành Bánh ngọt/Đồ nướng)
*   **Chắc chắn mua thêm:** `DELI` (Đồ ăn sẵn) + `GROCERY` (Tạp hóa) + `PRODUCE` (Rau củ quả)
*   **Các chỉ số:** Lift = **4.166** | Confidence = **30.5%** | Support = 1.05%
*   **Insight:** Giỏ hàng có sự pha trộn giữa Thịt tươi và Bánh ngọt thường là giỏ hàng chuẩn bị cho bữa sáng kiểu Mỹ (brunch) hoặc các dịp picnic/dã ngoại. Họ sẽ cần thêm đồ Deli (chả lụa, dăm bông để kẹp bánh mì), Tạp hóa (bơ, mứt, trứng) và Rau củ (salad đi kèm).

---

## 3. Executive Business Solutions: Hành Động Cấp Doanh Nghiệp

Thay vì áp dụng giảm giá vô tội vạ, hãy dùng chính các tỷ lệ Confidence tự nhiên này để vắt kiệt "Share of Wallet" (Ví tiền) của khách hàng. Đặc biệt là với nhóm **Loyal Customers** — nhóm mua giỏ hàng siêu bự $50.5/giỏ và dùng Retail Discount tới 89.9% (theo `segment_insight.md`).

### Chiến lược 1: Bán Chéo Thông Minh (Smart Bundling & Merchandising)
*   **Tối ưu Layout (Planogram):** Cả 3 quy luật đều cho thấy `PRODUCE` (Rau củ) luôn nằm phía "consequents" — tức nó là ngành hàng bị kéo theo chứ không phải ngành khởi phát. Hãy đặt quầy Rau củ quả trên cùng tuyến đường (Aisle) di chuyển từ quầy Thịt ra máy tính tiền. Đặt các kệ nhỏ (End-caps) bán gia vị, nước sốt (Grocery) ngay sát quầy MEAT và DELI.
*   **Dynamic POS Prompts (Gợi ý tại quầy thu ngân):** Khi hệ thống quét mã vạch nhận diện giỏ hàng chứa `MEAT` và `NUTRITION` nhưng chưa có `PRODUCE`, màn hình thu ngân lập tức hiện gợi ý: *"Mua thêm 1 bó rau sạch organic (Produce) để được giảm 5% cho toàn bộ bill dinh dưỡng này"*. Lợi dụng mức Confidence 52.6%, khách hàng gần như chắc chắn sẽ đồng ý vì đó vốn là thứ họ định mua.

### Chiến lược 2: Trọng Tâm Hóa Ngân Sách "Retail Discount" Vào Combo
Như đã phân tích trong `segment_insight.md`, hơn 80% khách hàng rất chịu khó dùng Retail Discount qua thẻ thành viên, nhưng gần như không ai chủ động cắt mã Coupon (chỉ 5-9%). 
*   **Thay vì giảm giá đồ lẻ (Single item markdown):** Hãy ngừng giảm giá 10% riêng rẽ cho một ngành hàng. 
*   **Chuyển sang Bundle Discount (Giảm giá theo combo):** Cài đặt vào thẻ thành viên luật: *"Hoàn tiền $5 vào thẻ nếu giỏ hàng chứa đủ combo (MEAT + DELI + PRODUCE)"*. Vì Lift của bộ 3 này lên tới trên 4.5 (tự nhiên họ đã muốn mua cùng nhau rồi), việc cho thêm một chút động lực (Incentive) bằng thẻ VIP sẽ ép cả những khách hàng bình thường chỉ định mua Thịt cũng phải vác thêm Rau và Deli để nhận được tiền hoàn.

### Chiến lược 3: Đồng bộ Chuỗi Cung Ứng (Supply Chain Sync)
*   **Dự báo tồn kho (Inventory Forecasting):** Tỷ lệ Lift khổng lồ giữa ngành `MEAT` và ngành `MEAT-PCKGD` (Thịt tươi và Thịt đóng gói) cảnh báo một điều: Khi siêu thị chạy Promotion kích cầu quầy Thịt Tươi cuối tuần, đội Vận hành PHẢI tăng lượng tồn kho của quầy Thịt Đóng Gói (xúc xích, thịt nguội) và Rau Củ (Produce) lên tương ứng. Nếu chỉ nhập thêm Thịt Tươi mà quên nhập Rau, hệ thống sẽ gặp hiện tượng "Out-of-stock" (cháy hàng) ở quầy Rau củ, làm thất thoát doanh thu từ hiệu ứng bán chéo.

---

## 4. Tổng kết

Bằng cách biến các chỉ số Lift > 4.0 thành các combo khuyến mãi (Bundles) cài đặt sẵn vào Thẻ Thành Viên, hệ thống của Dunnhumby có tiềm năng tăng quy mô giỏ hàng trung bình (Average Basket Size) mà không tốn thêm chi phí thu hút khách hàng mới. Hiệu quả cụ thể cần được đo lường qua các vòng A/B Testing trong giai đoạn triển khai thí điểm (Pilot), theo dõi các KPI:
*   **Basket Size lift** (So sánh giá trị giỏ hàng trung bình trước và sau khi áp dụng Bundle Discount)
*   **Cross-sell acceptance rate** (Tỷ lệ khách hàng thêm sản phẩm được gợi ý vào giỏ)
*   **Bundle redemption rate** (Tỷ lệ quy đổi combo trên thẻ thành viên)
