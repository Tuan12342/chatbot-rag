# Bộ kiểm thử 100 câu hỏi lịch sử Việt Nam

Bộ dữ liệu được tạo từ các PDF trong thư mục `papers`. Mỗi câu có đáp án tham chiếu và vị trí đối chiếu trong tài liệu nguồn.

## Các tệp

- `lich_su_viet_nam_100_qa.csv`: thuận tiện mở bằng Excel hoặc nhập vào công cụ đánh giá.
- `lich_su_viet_nam_100_qa.jsonl`: thuận tiện đọc bằng chương trình; mỗi dòng là một đối tượng JSON độc lập.
- `lich_su_viet_nam_100_qa.md`: bản dễ đọc để hỏi thủ công.

`source_page_pdf` là số thứ tự trang tính từ trang đầu của file PDF. Con số này có thể khác số trang được in trên sách.

## Cách chấm gợi ý

Với mỗi câu, chấm một trong ba mức:

- `1`: đúng ý chính và không có thông tin sai làm thay đổi đáp án.
- `0.5`: đúng một phần nhưng thiếu một thành phần quan trọng, chẳng hạn đúng năm nhưng sai ngày hoặc chỉ nêu một trong hai nhân vật.
- `0`: sai, không trả lời, hoặc trả lời mâu thuẫn với đáp án tham chiếu.

Công thức:

```text
độ chính xác (%) = tổng điểm / 100 × 100
```

Nên ghi thêm hai chỉ số riêng:

- `retrieval_hit`: chatbot có truy xuất đúng tài liệu/trang liên quan hay không.
- `citation_correct`: nguồn chatbot dẫn có thực sự hỗ trợ câu trả lời hay không.

Trường `expected_keywords` chỉ là gợi ý để lọc nhanh, không nên dùng như điều kiện chấm tuyệt đối vì chatbot có thể diễn đạt đúng bằng từ đồng nghĩa.

## Lưu ý về ngày tháng

Đáp án được chuẩn hóa theo chính các tài liệu nguồn trong bộ PDF. Ví dụ, `Lịch sử Việt Nam 15.pdf` ghi lễ kết nạp Việt Nam vào ASEAN ngày 27-7-1995 và ghi tuyên bố bình thường hóa quan hệ Việt Nam - Hoa Kỳ là ngày 12-7-1995 theo giờ Hà Nội. Khi đánh giá RAG trên đúng kho tài liệu này, nên dùng các mốc được ghi trong nguồn.
