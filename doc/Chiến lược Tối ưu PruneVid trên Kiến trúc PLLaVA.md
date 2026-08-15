# **Chiến lược Tối ưu hóa PruneVid trên nền tảng Kiến trúc PLLaVA**

## **1\. Phân tích Xung đột và Cơ hội giữa PLLaVA và PruneVid**

Để cải tiến, trước tiên ta cần xem xét cách hai mô hình này tương tác:

* **PLLaVA (Pooling LLaVA):** Xử lý sự dư thừa của video bằng cách áp dụng Pooling (thường là Average Pooling) lên các đặc trưng không gian và thời gian từ Vision Encoder *trước khi* chiếu (project) vào LLM. Các token đầu vào của LLM trong PLLaVA đã là những **"siêu token" (dense tokens)** mang thông tin tổng hợp của nhiều khung hình.  
* **PruneVid:** Hoạt động dựa trên giả định rằng có rất nhiều token "nhiễu" hoặc "thừa". Tuy nhiên, khi áp dụng lên PLLaVA, lượng token thừa thực chất đã bị giảm đáng kể do Pooling. Nếu PruneVid tiếp tục "cắt tỉa cứng" (hard prune) quá mạnh tay, nó có thể loại bỏ nhầm các đặc trưng đã được nén kỹ lưỡng của PLLaVA.

**👉 Điểm đột phá (Opportunity):** Tinh chỉnh cơ chế Attention-guided Pruning của PruneVid sao cho phù hợp với bản chất "đậm đặc" của token trong PLLaVA, đồng thời tạo ra sự liên kết giữa cơ chế Pooling của PLLaVA và câu hỏi (Question).

## **2\. Các Phương án Chỉnh sửa Kiến trúc để Cải thiện Kết quả**

Dưới đây là 5 phương án kiến trúc cụ thể từ dễ đến khó để bạn can thiệp vào mã nguồn hiện tại nhằm tăng điểm đánh giá:

### **Phương án 1: Lựa chọn Attention Head chuyên biệt (Vision-Dominant Heads)**

* **Vấn đề hiện tại:** PruneVid tính điểm quan trọng của token bằng cách lấy trung bình (average) Attention Weights của *tất cả* các Attention Heads tại layer trung gian ![][image1]. Tuy nhiên, trong LLM, có những head chuyên xử lý ngữ pháp văn bản, có head chuyên xử lý logic, và chỉ một số ít head thực sự chú ý đến "Visual Tokens".  
* **Giải pháp chỉnh sửa:**  
  * Thực hiện một phân tích nhỏ (profiling) để xác định Top-20% Attention Heads trong PLLaVA nhạy cảm nhất với token thị giác (ví dụ: các head có trọng số lớn giữa Text Query và Visual Tokens).  
  * **Sửa code:** Thay vì mean(attention\_weights, dim=heads), hãy chỉ tính tổng trên các "Vision-Dominant Heads". Điều này giúp PruneVid không bị nhiễu bởi các head xử lý ngôn ngữ, chọn token chính xác hơn nhiều.

### **Phương án 2: Soft Pruning bằng Token Merging (Bảo toàn thông tin cho PLLaVA)**

* **Vấn đề hiện tại:** Token của PLLaVA là token đã được pooling. Việc vứt bỏ hoàn toàn (Hard Pruning) các token có điểm Attention thấp sẽ làm mất đi bối cảnh (background context) quan trọng cho các câu hỏi suy luận phức tạp.  
* **Giải pháp chỉnh sửa:** Thay vì xóa hoàn toàn Top\-![][image2] token thấp nhất, hãy áp dụng **ToMe (Token Merging)** ở layer ![][image1].  
  * **Sửa code:** Lấy các token bị loại (Pruned Tokens) tính toán độ tương đồng Cosine (Cosine Similarity) với các token được giữ lại (Kept Tokens).  
  * Cộng gộp (Weighted Sum) đặc trưng của token bị loại vào token được giữ lại gần nhất. Kết quả là số lượng token vẫn giảm như PruneVid yêu cầu (giúp giảm memory/FLOPs ở các layer sau), nhưng thông tin bối cảnh không bị mất mà được "hấp thụ" vào các token chính.

### **Phương án 3: Thay đổi Động Layer Cắt tỉa (Dynamic Intermediate Layer ![][image1])**

* **Vấn đề hiện tại:** PruneVid thiết lập một layer ![][image1] cố định (ví dụ layer thứ 15 trên 32\) để lấy Attention Map làm quyết định cắt tỉa. Đối với PLLaVA, đôi khi ở layer 15, LLM vẫn chưa hiểu được câu hỏi đòi hỏi thông tin gì từ các "siêu token" đã nén.  
* **Giải pháp chỉnh sửa:**  
  * Áp dụng **Cắt tỉa hai giai đoạn (Two-Stage Pruning)**.  
  * Thay vì tỉa 80% tại layer ![][image1], hãy chia làm 2: Tỉa 40% ở layer ![][image3] (ví dụ: layer 8 \- loại bỏ bối cảnh hoàn toàn rác), và tỉa tiếp 40% ở layer ![][image4] (ví dụ: layer 16 \- loại bỏ các vật thể không liên quan đến câu hỏi). Sự phân cấp này giúp mô hình PLLaVA có thời gian "làm quen" với token trước khi ra quyết định loại bỏ.

### **Phương án 4: Kiến trúc Hỗn hợp (Hybrid Multi-Granularity Input)**

* **Vấn đề hiện tại:** Pooling của PLLaVA làm mờ đi các vật thể nhỏ hoặc các chuyển động quá nhanh. Khi PruneVid hỏi về các vật thể này, Attention Map không thể tìm thấy token nào nổi bật.  
* **Giải pháp chỉnh sửa (Can thiệp sâu vào Projector):**  
  * Thay đổi đầu ra của PLLaVA Projector thành sự kết hợp của 2 luồng:  
    1. **Luồng Pooled (Background/Global):** Giữ nguyên kiến trúc PLLaVA (Token nén).  
    2. **Luồng Unpooled (Dynamic/Local):** Sử dụng Giai đoạn 1 của PruneVid (Static/Dynamic Decoupling) ở mức Pixel/Patch. Trích xuất khoảng 10-15% token gốc chưa qua pooling đại diện cho chủ thể đang chuyển động mạnh nhất, chiếu thẳng qua một MLP nhỏ.  
  * Nối (Concat) hai luồng này vào LLM. Nhờ đó, PruneVid ở bên trong LLM sẽ có cơ hội chú ý (attend) vào các chi tiết sắc nét của luồng Unpooled, cải thiện vượt bậc khả năng trả lời các câu hỏi về chi tiết nhỏ (fine-grained QA).

### **Phương án 5: Question-Guided Pooling cho PLLaVA (Đưa PruneVid ra ngoài LLM)**

* **Ý tưởng đột phá:** Thay vì để PLLaVA tự động pooling mù quáng (blind pooling) trước khi vào LLM, hãy trích xuất Token của Câu hỏi (Question Embeddings) và đưa nó xuống cơ chế Pooling của PLLaVA.  
* **Sửa code:** Chèn một khối Cross-Attention nhẹ trước PLLaVA Projector. Sử dụng Question Tokens làm Query, và Video Tokens (chưa pooling) làm Key/Value. Các frame/patch nào có điểm Cross-Attention cao sẽ được gán trọng số lớn trong quá trình Average Pooling. Điều này biến Pooling của PLLaVA thành một dạng "Soft Pruning" định hướng ngay từ cửa ngõ của LLM.

## **3\. Khuyến nghị Thực thi (Implementation Plan)**

Để bắt đầu cải thiện Evaluation cho mã nguồn hiện tại, bạn nên thực hiện theo thứ tự ưu tiên sau (từ chi phí thấp nhất đến cao nhất):

1. **Thử nghiệm Phương án 1 (Vision-Dominant Heads) và Phương án 3 (Two-Stage Pruning):** Hai phương án này **Training-Free 100%**. Bạn chỉ cần sửa code hàm forward của mô hình ngôn ngữ (Llama/Vicuna) trong file chứa logic của PruneVid để thay đổi cách tính toán attention\_weights và vị trí thực hiện hàm prune().  
2. **Thử nghiệm Phương án 2 (Token Merging):** Cũng là Training-Free, nhưng cần cài đặt thêm logic gom cụm (như bipartite matching). Tác động lên độ chính xác là rất lớn, đặc biệt cho các bộ benchmark đòi hỏi lý luận không gian (như MVBench).  
3. **Phương án 4 & 5:** Cần phải **Fine-tune lại Projector** của PLLaVA. Nếu bạn có tài nguyên GPU (1-2 thẻ A100), đây là con đường tạo ra một bài báo khoa học mới (Novelty cao).
