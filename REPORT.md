# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Tiến Đạt — 2A202600595
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight ZIP)

---

## 1. Setup

| Thành phần | Giá trị |
|---|---|
| **Base model** | `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5-3B, quantize 4-bit NF4 — QLoRA) |
| **Dataset** | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train + 20 eval) |
| **Split** | 90% / 10%, `seed = 42` |
| **max_seq_length** | **1024** (phân tích token: p50=227, **p95=562**, p99=704 → làm tròn lên lũy thừa 2 = 1024, đúng bằng cap T4) |
| **GPU** | Tesla T4, 15.6 GB VRAM (≈14.56 GB khả dụng) |
| **Training cost** | **≈ $0.07** (tổng 12.8 phút cho 3 lần train @ $0.35/giờ) |

**Cấu hình LoRA / hyperparameters chung (cố định cho cả 3 rank):**
- `target_modules = ["q_proj", "v_proj"]` (theo lab spec)
- `lora_dropout = 0`, `bias = "none"`, gradient checkpointing (unsloth, −60% VRAM)
- 3 epochs · cosine LR schedule · `learning_rate = 2e-4` · `warmup_ratio = 0.10`
- Effective batch size = 8 (batch 1 × grad_accum 8) · optimizer `adamw_8bit` (paged AdamW)
- `packing = False` (tránh bug dim-mismatch trên transformers mới) · eval-during-training tắt (T4 thiếu VRAM)

> Chỉ **rank** và **alpha** thay đổi giữa 3 thí nghiệm — mọi yếu tố khác giữ nguyên để so sánh công bằng.

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % of model | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|-----------|-----------|-----------|-----------|------------|
| 8    | 16    | 1,843,200        | 0.059%    | 4.37 min  | 7.22 GB   | 1.5577    | **4.748**  |
| 16   | 32    | 3,686,400        | 0.119%    | 4.32 min  | 6.62 GB   | 1.5161    | **4.554**  |
| 64   | 128   | 14,745,600       | 0.476%    | 4.15 min  | 8.00 GB   | 1.4768    | **4.379**  |
| Base | —     | —                | —         | —         | —         | *(không đo — xem ghi chú)* | — |

**Quan sát số liệu:**
- **Perplexity giảm đều** khi rank tăng: 4.748 → 4.554 → 4.379. Rank lớn hơn = nhiều khả năng biểu diễn hơn = fit dữ liệu tốt hơn.
- **Số tham số tăng tuyến tính** với rank: r=64 có **4× tham số** so với r=16 (14.7M vs 3.7M).
- **Thời gian train gần như không đổi** (~4.2–4.4 phút cho cả 3): ở quy mô dataset nhỏ (180 mẫu), thời gian bị chi phối bởi forward/backward của base model bị đóng băng — phần adapter quá nhỏ để ảnh hưởng đáng kể.
- **Peak VRAM**: r=16 thấp nhất (6.62 GB); r=64 cao nhất (8.00 GB) do nhiều tham số adapter + optimizer states hơn.

> **Ghi chú về Base perplexity:** notebook T4 chỉ tính perplexity cho 3 adapter (để tiết kiệm VRAM), chưa đo perplexity của base model. Thay vào đó, chất lượng base được đánh giá trực tiếp qua **so sánh định tính** ở Mục 4 (cột BASE). Đây là điểm có thể bổ sung nếu muốn đủ 4 con số.

---

## 3. Loss Curve Analysis

![Loss Curve r=16](results/loss_curve.png)

- **Train loss (r=16)** giảm từ ~1.61 → ~1.39 qua 69 step (3 epochs), tuy có dao động (đặc thù batch nhỏ, dataset ít) nhưng xu hướng chung đi xuống → model **đang học thật**.
- **Eval loss = 1.52** (điểm đỏ cuối), cao hơn train loss cuối (~1.39) một khoảng **~0.13 nat** — đây là generalization gap **bình thường và lành mạnh**, không phải overfitting nghiêm trọng.
- Vì T4 tắt eval-during-training để tiết kiệm VRAM, ta chỉ có **một điểm eval cuối** nên không quan sát được quỹ đạo eval. Tuy nhiên, gap nhỏ ở 3 epochs cho thấy chưa overfit.
- **Dấu hiệu theo rank:** r=64 đẩy train loss xuống thấp hơn hẳn (~1.28) trong khi eval chỉ còn 1.477 → gap ~0.20, **rộng hơn** gap của r=16 (0.13). Đây là dấu hiệu sớm cho thấy rank cao bắt đầu "thuộc lòng" dữ liệu train nhiều hơn → rủi ro overfitting tăng theo rank.

---

## 4. Qualitative Comparison (5 examples)

So sánh **Base (Qwen2.5-3B gốc)** vs **Fine-tuned (r=16)**. Chọn lọc trung thực, có cả ca cải thiện lẫn ca thụt lùi (không cherry-pick).

### Example 1 — "Giải thích khái niệm machine learning cho người mới bắt đầu."
- **Base**: "Machine learning là một phân khúc của trí tuệ nhân tạo... máy tính học tập từ dữ liệu..."
- **Fine-tuned**: "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán... mà không có sự hướng dẫn trực tiếp từ người dùng..."
- **Nhận xét**: ~Ngang nhau. FT diễn đạt mạch lạc hơn chút, nhấn vào ý "tự học không cần hướng dẫn trực tiếp". **Cải thiện nhẹ về văn phong.**

### Example 2 — "Viết đoạn code Python tính số Fibonacci thứ n."
- **Base**: `def fibonacci(n): if n <= 0: return "N phải là một số dương"` → trả về **chuỗi** cho trường hợp lỗi (lẫn lộn kiểu dữ liệu, không chuẩn).
- **Fine-tuned**: `if n < 0: raise ValueError("Input phải là một số nguyên dương.")` + dùng vòng lặp `a, b = 0, 1` (iterative).
- **Nhận xét**: ✅ **FT thắng rõ** — dùng `raise ValueError` đúng chuẩn Python, code sạch và an toàn hơn.

### Example 3 — "Liệt kê 5 nguyên tắc thiết kế UI/UX."
- **Base**: viết đoạn văn dài dòng "1. Thân thiện với người dùng: Mục đích của thiết kế..."
- **Fine-tuned**: liệt kê ngắn gọn "1. Chuyển đổi 2. Thích ứng 3. Đơn giản 4. Tương thích..."
- **Nhận xét**: ✅ **FT thắng về format** — đúng kiểu danh sách súc tích (đúng phong cách Alpaca dataset). Tuy "Chuyển đổi" hơi tối nghĩa.

### Example 4 — "Tóm tắt sự khác biệt giữa LoRA và QLoRA."
- **Base**: "LoRA (**Low-Rank Adaptation**) và QLoRA (Quantized LoRA)..." → **đúng**.
- **Fine-tuned**: "LoRA (**Layer-wise Adaptive Regularization Optimization**)..." → **SAI** (bịa tên + gọi nhầm là "regularization").
- **Nhận xét**: ❌ **FT thụt lùi** — đây là ca **degraded** quan trọng: fine-tune trên 200 mẫu generic đã làm hỏng một kiến thức factual mà base vốn trả lời đúng.

### Example 5 — "Phân biệt prompt engineering, RAG, và fine-tuning."
- **Base**: "...ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật..."
- **Fine-tuned**: "...ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt)..."
- **Nhận xét**: ~Ngang nhau, cả hai đều đúng và đầy đủ. **Không khác biệt rõ.**

**Tổng kết định tính:** 2 ca cải thiện (format/code) · 2 ca ngang · **1 ca thụt lùi (factual)**. Với chỉ 200 mẫu generic + 3 epochs, fine-tuning chủ yếu **dịch chuyển văn phong/định dạng** (giống style Alpaca, list gọn, code chuẩn hơn) **chứ không bổ sung kiến thức** — thậm chí làm hỏng 1 fact. Điều này khớp chính xác với **"quy tắc vàng"** của bài học: *fine-tune cho style/format, dùng RAG cho knowledge*.

---

## 5. Conclusion về Rank Trade-off

Trên dataset 200 mẫu tiếng Việt này, **r=16 cho ROI tốt nhất** và là lựa chọn tôi sẽ deploy production.

Phân tích chi phí/lợi ích theo từng bước tăng rank cho thấy hiệu suất biên giảm dần (diminishing returns) rất rõ. Từ **r=8 → r=16**, ta gấp đôi số tham số (1.8M → 3.7M) và đổi lại perplexity giảm 0.194 (4.748 → 4.554) — một khoản đầu tư xứng đáng. Nhưng từ **r=16 → r=64**, ta phải bỏ ra **gấp 4 lần** tham số (3.7M → 14.7M) mà perplexity chỉ giảm thêm 0.175 (4.554 → 4.379) — tức **chi phí gấp 4 nhưng lợi ích tuyệt đối còn nhỏ hơn** bước trước. Diminishing returns bắt đầu lộ rõ **ngay sau r=16**. Ngoài ra, r=64 còn tốn thêm ~1.4 GB VRAM (8.0 vs 6.6 GB) và có gap train–eval rộng hơn (0.20 vs 0.13), tức rủi ro overfitting cao hơn trên tập dữ liệu nhỏ này.

Vì vậy, nếu deploy production tôi **chọn r=16**: nó đạt ~98% lợi ích của r=64 nhưng chỉ dùng 1/4 tham số, ít VRAM nhất, và adapter nhẹ hơn để serve multi-tenant. r=64 chỉ đáng cân nhắc khi dataset lớn hơn và phức tạp hơn nhiều — khi đó capacity dư của rank cao mới được tận dụng thay vì chỉ memorize.

---

## 6. What I Learned

- **Trade-off của rank là phi tuyến, có "sweet spot".** Tăng rank không giúp tuyến tính: bước 8→16 hiệu quả trên mỗi tham số cao hơn hẳn bước 16→64. Chọn rank là bài toán ROI, không phải "càng cao càng tốt".
- **QLoRA trên T4 free rẻ đến bất ngờ:** 3 lần train đầy đủ chỉ tốn ~13 phút và **≈$0.07**. Fine-tuning LLM giờ nằm trong tầm tay của bất kỳ sinh viên nào, không cần GPU đắt tiền.
- **Fine-tune đổi style, không đổi knowledge.** Ca LoRA/QLoRA bị trả lời sai (Example 4) là bằng chứng cụ thể cho quy tắc *"dùng RAG cho kiến thức, fine-tune cho văn phong/định dạng"* — train trên data generic nhỏ có thể vô tình làm hỏng kiến thức factual mà base model vốn nắm đúng.
