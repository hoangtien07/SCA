# PHẢN BIỆN: Kế hoạch nghiên cứu AI Mỹ phẩm Việt Nam

> Góc nhìn chuyên gia đối lập — Phản biện file `research-plan-beauty-vn.md`

---

## 1. CON SỐ DOANH THU: Lạc quan đến mức nguy hiểm

### Affiliate revenue bị thổi phồng nghiêm trọng

Kế hoạch ước tính: `50K MAU × 15% click × 250K avg × 4% commission = 187 triệu/tháng`. Nhưng:

- **15% affiliate click rate là phi thực tế.** Benchmark ngành affiliate toàn cầu là 1-3% CTR. Ngay cả Wirecutter (đỉnh cao affiliate content) cũng chỉ đạt ~5-8%. Một app mới, chưa có brand trust, ở VN — 15% là ảo tưởng.
- **4% commission từ Shopee Affiliate** — thực tế Shopee VN trả 2-3% cho beauty, và có xu hướng giảm commission theo thời gian khi platform đã đủ lớn.
- **250K avg order** — skincare VN thực tế nhiều đơn 100-150K (serum Hàn giá rẻ, kem chống nắng phổ thông). 250K là thiên về mid-range.
- Con số thực tế hơn: `50K × 3% click × 180K × 2.5% = ~6.75 triệu/tháng` — **thấp hơn 27 lần** so với dự kiến.

### Premium subscription 3% conversion — dựa trên gì?

- App skincare không phải SaaS B2B. Spotify mất **10 năm** để đạt ~45% premium conversion. Các app beauty/health thường đạt 0.5-1.5% ở năm đầu. 3% cho một app mới, chưa brand, ở thị trường VN nơi người dùng **cực kỳ nhạy cảm giá** — không có cơ sở.

### Gross margin 90% là sai lệch

- Chi phí vận hành 30-50 triệu/tháng bỏ qua hoàn toàn: chi phí LLM API (Claude/Gemini cho 50K MAU có thể là 50-200 triệu/tháng), chi phí acquire user (CAC), chi phí content marketing, lương nhân sự. Nếu tính đúng, margin thực tế có thể **âm** trong 12-18 tháng đầu.

---

## 2. "50,000 MAU SAU 12 THÁNG" — Từ đâu ra?

Đây là lỗ hổng lớn nhất: **không có chiến lược acquisition cụ thể.**

- Zalo Mini App reach 98% — nhưng reach ≠ install. Có **3,465 Mini Apps** đang hoạt động, người dùng trung bình dùng bao nhiêu? Việc "có trên Zalo" không tự động mang user đến.
- Không có budget marketing. "Content marketing TikTok, Instagram" — ai làm? Tốn bao nhiêu? Bao lâu mới có organic traction?
- Không có viral loop hoặc referral mechanism nào được thiết kế vào product.
- So sánh: **Think Dirty** (ingredient scanner, thị trường Mỹ/Canada, 7+ năm) mới đạt 2M downloads tổng cộng. Kế hoạch kỳ vọng 50K MAU ở VN sau 12 tháng — với budget <20 triệu?

---

## 3. GAP ANALYSIS: Tự vẽ khoảng trống cho mình

Ma trận "Personalized vs Generic" × "Unbiased vs Biased" có vấn đề:

- **"Unbiased" là một ảo tưởng** khi revenue model chính là affiliate. Recommend sản phẩm A vì nó tốt cho da user, hay vì Shopee trả commission cao hơn? Ngay khi có affiliate, bạn **đã biased** — chỉ là bias khác loại so với KOL.
- Google Search bị xếp vào "Generic" — nhưng thực tế Google + AI Overview đang ngày càng personalized. Đến lúc launch, Google có thể đã giải quyết phần lớn bài toán này.
- Hasaki bị xếp vào "Biased" — nhưng user **biết** Hasaki bán hàng. Trust model khác hoàn toàn với một app tuyên bố "unbiased" nhưng sống nhờ affiliate.

---

## 4. KẾ HOẠCH KỸ THUẬT: Quá tham vọng, thiếu focus

- **PhoBERT sentiment 92.74%** là trên phone reviews, **không phải beauty reviews.** Domain shift là vấn đề thực — ngôn ngữ review mỹ phẩm VN rất khác (nhiều tiếng lóng, emoji, Vietnglish). Accuracy thực tế có thể chỉ 75-80% khi chuyển domain.
- **OCR INCI từ ảnh chụp bao bì** — đây là bài toán **cực khó**. Bao bì mỹ phẩm có font nhỏ, cong, nhiều ngôn ngữ, ánh sáng không đều. Ngay cả Google Lens cũng chưa giỏi việc này. Kế hoạch dành 4 tuần cho feature này trong MVP — phi thực tế.
- **6 nguồn data scraping cùng lúc** (Hasaki, Shopee, TikTok, Open Beauty Facts, CosIng, Brand websites) — mỗi nguồn cần maintain riêng, handle breaking changes riêng. Với team nhỏ, nên tập trung 1-2 nguồn thôi.

---

## 5. PHÁP LÝ: Đánh giá quá nhẹ

- **Web scraping Hasaki, Shopee** — Shopee đã kiện và block nhiều scraper. Terms of Service cấm rõ ràng. "Rate limiting" không phải giải pháp pháp lý — nó chỉ tránh bị detect. Rủi ro bị cease-and-desist là thực.
- **Thu thập skin data + ảnh da** thuộc **dữ liệu sinh trắc học** theo Nghị định 13/2023. Yêu cầu không chỉ consent form mà còn đánh giá tác động xử lý dữ liệu (DPIA). Chi phí compliance thực tế có thể 50-100 triệu, không phải "2-5 triệu tham vấn luật sư".
- **Medical claims**: "Sản phẩm phù hợp da bạn" đã ranh giới với tư vấn y tế. Nếu user bị dị ứng theo recommendation của app, trách nhiệm pháp lý là rất lớn.

---

## 6. COMPETITIVE MOAT: Không có

Câu hỏi quan trọng nhất mà kế hoạch **không trả lời**: **Điều gì ngăn Hasaki, Shopee, hoặc TikTok tự build feature này?**

- Hasaki có data khách hàng thực, lịch sử mua hàng thực, inventory thực. Họ chỉ cần thuê 2-3 ML engineer là có thể build AI recommendation tốt hơn.
- Shopee đã có recommendation engine. Nếu beauty AI tư vấn có traction, Shopee sẽ copy trong 3 tháng — với data tốt hơn gấp 1000 lần.
- **Network effect**: không có. **Data moat**: không có (scrape data công khai). **Brand moat**: chưa có. **Switching cost**: gần zero.

---

## 7. ROADMAP: Cố gắng làm quá nhiều thứ

Trong 20 tuần, kế hoạch bao gồm: Routine Builder, Ingredient Scanner, Trend Radar, Dupe Finder, Price Comparison, Routine Tracker, B2B outreach, KOL matching, Male grooming vertical. Đó là **9 features/products** trong **5 tháng.**

Các startup thành công thường mất 6-12 tháng để làm **một feature đúng**. Kế hoạch đang plan như một công ty 50 người, nhưng budget cho thấy đây là team 1-3 người.

---

## 8. GỢI Ý NẾU THẬT SỰ MUỐN LÀM

Nếu phải cứu kế hoạch này:

1. **Chọn MỘT feature duy nhất** — Ingredient Scanner hoặc Dupe Finder (có viral potential cao nhất, dễ demo, dễ word-of-mouth).
2. **Bỏ affiliate khỏi MVP** — tập trung build trust trước, monetize sau. User sẽ bỏ ngay nếu ngửi thấy mùi bán hàng.
3. **Validation thực tế**: Tạo một Zalo OA hoặc Facebook Group, tư vấn skincare **thủ công** cho 100 người. Nếu không attract được 100 người free, thì app sẽ không attract 50,000.
4. **Tính lại unit economics** với số thực tế (1-2% CTR, 0.5% premium conversion) — nếu vẫn viable thì mới tiếp tục.
5. **Trả lời câu hỏi moat** trước khi viết một dòng code nào.

---

## KẾT LUẬN

Kế hoạch này có research thị trường tốt, nhưng bị mắc bẫy kinh điển — **dùng data thị trường lớn để justify một sản phẩm nhỏ, rồi assume mình sẽ capture được một phần thị trường đó mà không giải thích bằng cách nào.** Thị trường 3 tỷ USD không có nghĩa sẽ kiếm được 1 đồng nào từ đó.

Trước khi build bất cứ thứ gì, hãy trả lời được 3 câu:

1. **Tại sao user chọn app này thay vì hỏi Facebook Group miễn phí?**
2. **Tại sao Hasaki/Shopee không tự làm điều này?**
3. **Nếu chỉ có 5,000 MAU (thay vì 50,000), business có sống được không?**

Nếu không trả lời được — đây chưa phải lúc để code.
