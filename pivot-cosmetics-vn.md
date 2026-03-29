# PIVOT: AI Khuyến nghị Mỹ phẩm Việt Nam

## Tổng quan ý tưởng

Chuyển từ bài toán "medical skincare AI dựa trên nghiên cứu y khoa" sang "AI khuyến nghị mỹ phẩm cá nhân hóa cho thị trường Việt Nam", kết hợp:
- Profile cá nhân (da, ngân sách, thói quen, sở thích)
- Product database thực tế (giá, INCI, reviews, availability tại VN)
- Trend tracking real-time từ MXH (TikTok, Shopee, Facebook)
- Xác thực thông tin sản phẩm (ingredient analysis, fake review detection)

**Đánh giá khả thi tổng thể: 8/10** — khả quan hơn nhiều so với bài toán gốc (5.5/10).

---

## 1. Thị trường

**Thị trường mỹ phẩm Việt Nam (2025-2026):**
- Quy mô ~$3.5–4 tỷ USD, tăng trưởng 15–20%/năm
- 70%+ người mua tham khảo review trên MXH trước khi mua
- Kênh mua: Shopee, TikTok Shop, Lazada, Hasaki, Guardian, Sociolla
- K-beauty, J-beauty chiếm dominant, local brands đang lên (Cocoon, Thorakao mới, SVR Việt Nam)

**Pain point thực:**

| Pain point | Mức độ | Giải pháp hiện tại |
|-----------|--------|-------------------|
| "Không biết chọn gì phù hợp da mình" | Rất cao | Hỏi Facebook group, xem TikTok review — thiên lệch, không cá nhân hóa |
| "Sản phẩm trend có tốt thật không?" | Cao | Tự research — tốn thời gian, thiếu kiến thức |
| "Budget có hạn, ưu tiên cái gì?" | Cao | Không có tool nào giúp |
| "Mua rồi nhưng dùng sai thứ tự/kết hợp" | Trung bình | Google, YouTube — thông tin rời rạc |
| "Sản phẩm giả/nhái tràn lan" | Cao | Tự check — khó |

**Đối thủ hiện tại (Việt Nam):**
- **Hasaki app**: Quiz sơ bộ, recommend theo inventory (bán hàng), không thực sự cá nhân hóa
- **Sociolla Vietnam**: Review-driven, thiên về e-commerce
- **Facebook groups** (Skincare & Makeup VN, ~500K+ members): Community-driven, chất lượng không đồng đều
- **TikTok KOLs**: Influence lớn nhưng nhiều sponsored content không minh bạch

**=> Chưa có AI-powered, unbiased, personalized recommendation tool nào ở Việt Nam.** Khoảng trống thực sự.

---

## 2. Sản phẩm

### User flow

```
User input:
  - Loại da, vấn đề da, tuổi, giới tính
  - Ngân sách (200K / 500K / 1tr / tháng)
  - Thói quen (minimalist 3 bước / full routine 7+ bước)
  - Ưu tiên (K-beauty, natural, fragrance-free, vegan...)
  - Sản phẩm đang dùng (để tránh conflict)
        ↓
  AI Processing:
  - Match profile → ingredient needs (từ SCA engine hiện tại)
  - Filter product DB theo budget, availability, preference
  - Cross-check trend data (TikTok, Shopee reviews)
  - Verify claims vs actual ingredients (INCI analysis)
        ↓
  Output:
  - Routine cá nhân: Sữa rửa mặt → Toner → Serum → Kem dưỡng → SPF
  - Mỗi bước: 2-3 options theo tầm giá (budget / mid / premium)
  - Giải thích TẠI SAO sản phẩm này phù hợp (ingredient match)
  - Warning nếu combo sản phẩm conflict
  - Link mua trực tiếp (Shopee / Hasaki / Lazada)
  - Trend score: "Đang hot trên TikTok" vs "Ít người biết nhưng ingredient tốt"
```

### Tính năng chính

1. **Personalized Routine Builder** — Recommend routine theo profile + budget + preference
2. **Ingredient Checker** — Scan INCI list, phân tích thành phần, cảnh báo conflict
3. **Trend Radar** — Sản phẩm đang hot trên MXH, kèm đánh giá thực chất
4. **Price Tracker** — So giá giữa các platform, alert giảm giá
5. **Routine Conflict Checker** — Phát hiện sản phẩm trong routine hiện tại xung đột nhau

---

## 3. Phân tích kỹ thuật

### 3.1 Tái sử dụng từ SCA hiện tại (~40-50%)

| Component SCA hiện tại | Tái sử dụng | Ghi chú |
|------------------------|-------------|---------|
| `skin_conditions.yaml` (taxonomy) | 90% | Ingredient interactions, safety rules giữ nguyên |
| `safety_guard.py` | 80% | Pregnancy, allergen, conflict checks vẫn cần |
| `regimen_generator.py` | 60% | Rewrite prompt: "clinical regimen" → "product recommendation" |
| RAG pipeline (retriever, chunker, indexer) | 70% | Reuse cho product reviews + ingredient database |
| FastAPI backend | 90% | Giữ nguyên |
| Vision analyzer | 30% | Optional, ít cần hơn |
| Paper collectors | 0% | Thay bằng product/trend collectors |
| Scientific KB | 20% | Background knowledge, không phải primary source |

### 3.2 Cần xây mới

**A. Product Database — Phần quan trọng nhất**

```
Nguồn dữ liệu sản phẩm:
├── Shopee API / scraping     → Giá, reviews, ratings, sold count
├── Hasaki product pages      → INCI list, mô tả, giá
├── Lazada / TikTok Shop      → Giá, trend data
├── Open Beauty Facts          → INCI database (đã có collector!)
├── CosIng EU database         → Ingredient safety (đã có collector!)
└── Brand official sites       → Claims verification
```

Ước tính 2,000–5,000 SKU phổ biến ở VN là đủ cho MVP. Shopee có affiliate API, Hasaki có thể scrape.

**B. Social Media Trend Crawler**

```
Nguồn trend:
├── TikTok            → Hashtag tracking (#skincarevietnam, #reviewmypham)
│                       API hạn chế, cần unofficial scraping hoặc TikTok Shop API
├── Facebook Groups   → Post scraping (against ToS nhưng phổ biến)
│                       Alternative: RSS/webhook từ public pages
├── Shopee Reviews    → Sentiment analysis trên Vietnamese text
├── YouTube Vietnam   → Video transcript analysis
└── Instagram/Threads → Hashtag monitoring
```

**C. Vietnamese NLP Pipeline**

| Task | Tool | Accuracy ước tính |
|------|------|-------------------|
| Sentiment analysis (reviews) | PhoBERT fine-tuned | ~85-90% |
| Product name extraction | NER on PhoBERT | ~80% |
| Ingredient claim verification | Rule-based + LLM | ~90% |
| Trend detection | Time-series on mention frequency | ~95% |
| Slang/abbreviation normalization | Custom dictionary + LLM | ~85% |

### 3.3 Architecture đề xuất

```
                    ┌─────────────────────┐
                    │   User (Zalo Mini   │
                    │   App / Web App)    │
                    └─────────┬───────────┘
                              ↓
                    ┌─────────────────────┐
                    │   API Gateway       │
                    │   (FastAPI - giữ)   │
                    └─────────┬───────────┘
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
    ┌─────────────┐  ┌──────────────┐  ┌────────────┐
    │ Profile     │  │ Product      │  │ Trend      │
    │ Engine      │  │ Recommender  │  │ Tracker    │
    │ (từ SCA)    │  │ (MỚI)       │  │ (MỚI)     │
    └──────┬──────┘  └──────┬───────┘  └─────┬──────┘
           ↓                ↓                 ↓
    ┌─────────────┐  ┌──────────────┐  ┌────────────┐
    │ Ingredient  │  │ Product DB   │  │ Social     │
    │ Knowledge   │  │ (PostgreSQL) │  │ Media DB   │
    │ (từ SCA)    │  │              │  │ (MỚI)     │
    └─────────────┘  └──────────────┘  └────────────┘
```

---

## 4. Monetization

| Mô hình | Revenue/user | Khả thi | Ghi chú |
|---------|-------------|---------|---------|
| **Affiliate links** (Shopee/Lazada/Hasaki) | 3-8% commission | Cao | Shopee Affiliate Program có sẵn, user click mua = commission |
| **Freemium** | 49-99K/tháng premium | Trung bình | Free: 3 bước routine. Premium: full routine + trend alerts + ingredient checker |
| **B2B cho brands** | 5-20tr/tháng | Trung bình | Brand trả tiền để appear trong recommendations (phải minh bạch) |
| **Data insights** | Tùy deal | Thấp-TB | Trend reports cho brands/retailers |

**Ước tính conservative (tại 10,000 MAU):**
- Subscription: 10,000 × 5% conversion × 79K/tháng = ~39.5 triệu/tháng
- Affiliate: 10,000 × 20% click × 200K avg order × 5% commission = ~20 triệu/tháng
- **Tổng potential: ~50-60 triệu/tháng**
- Chi phí vận hành: ~$55/tháng ≈ 1.4 triệu → unit economics rất tốt

---

## 5. Rủi ro & Thách thức

| Rủi ro | Mức | Giải pháp |
|--------|-----|-----------|
| **Data freshness** — sản phẩm mới liên tục, giá thay đổi | Cao | Cron job crawl hàng ngày, priority cho top sellers |
| **Vietnamese NLP accuracy** — slang, viết tắt, Vietnglish | Trung bình | PhoBERT + custom dictionary + human review ban đầu |
| **Platform ToS** — scraping Shopee/TikTok | Cao | Dùng official API khi có (Shopee Affiliate), fallback semi-manual |
| **Bias** — recommend sản phẩm vì affiliate, không vì chất lượng | Cao | Tách rõ "AI pick" vs "Sponsored". Minh bạch = trust |
| **Fake reviews** — Shopee reviews nhiều fake | Trung bình | Review credibility scoring (verified purchase, review length, photo) |
| **Pháp lý** — quảng cáo mỹ phẩm tại VN | Thấp-TB | Nghị định 181/2013/NĐ-CP. Recommend ≠ quảng cáo nếu không nhận tiền từ brand cụ thể |

---

## 6. So sánh bài toán cũ vs mới

| Tiêu chí | SCA gốc (Medical Skincare AI) | SCA pivot (Mỹ phẩm VN) |
|----------|-------------------------------|------------------------|
| **Regulatory burden** | Rất cao (medical device) | Thấp (consumer app) |
| **Clinical validation** | Bắt buộc | Không cần |
| **Time to market** | 12-18 tháng | 2-3 tháng cho MVP |
| **Monetization** | Khó (ai trả tiền?) | Rõ ràng (affiliate + freemium) |
| **Competitive moat** | Thấp (replicable) | Trung bình (data + curation chất lượng) |
| **Market size (VN)** | Nhỏ (medical skincare consulting) | Lớn ($3.5B cosmetics market) |
| **User acquisition** | Khó (medical trust) | Dễ hơn (TikTok/Facebook viral potential) |
| **Technical reuse** | 100% | 40-50% |
| **Tổng điểm khả thi** | 5.5/10 | **8/10** |

---

## 7. Roadmap đề xuất

```
Tháng 1-2:  MVP
            - Product DB (top 500 SKU từ Hasaki + Shopee top sellers)
            - Profile engine (reuse từ SCA)
            - Basic recommendation (budget-aware routine builder)
            - Platform: Zalo Mini App hoặc Web app

Tháng 3:    Social + Affiliate
            - Trend tracker v1 (TikTok hashtag + Shopee review sentiment)
            - Affiliate integration (Shopee Affiliate Program)
            - Ingredient checker feature

Tháng 4-5:  Scale
            - Product DB lên 2,000+ SKU
            - Vietnamese NLP pipeline (PhoBERT)
            - User feedback loop (rate recommendations)
            - Price comparison across platforms

Tháng 6:    Premium + B2B
            - Premium tier (trend alerts, full routine, ingredient scanner)
            - B2B outreach (Korean/Japanese brands entering VN market)
            - Data insights package
```

---

## 8. Kết luận

**Pivot này khả quan hơn bài toán gốc trên mọi tiêu chí kinh doanh:**

1. **Giữ được lõi kỹ thuật mạnh nhất** của SCA: ingredient knowledge, safety checks, RAG pipeline
2. **Loại bỏ rào cản lớn nhất**: regulatory compliance cho medical device
3. **Thêm yếu tố viral**: trend tracking, social proof, budget optimization — dễ share, dễ hook user
4. **Monetization rõ ràng**: Affiliate là passive income ngay từ ngày 1
5. **Đúng thời điểm**: Thị trường mỹ phẩm VN đang boom, chưa có AI tool nào chiếm lĩnh

**Rủi ro lớn nhất** không phải kỹ thuật mà là **execution speed** — nếu ý tưởng này hiển nhiên, người khác cũng sẽ nghĩ tới. First-mover advantage quan trọng.
