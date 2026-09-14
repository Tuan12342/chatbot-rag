# Chatbot RAG

Dự án thử nghiệm RAG chạy local: đọc tài liệu PDF/Parquet, chia văn bản thành các chunk, tạo embedding bằng Ollama và lưu vector trong FAISS.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Cài Ollama, sau đó tải các model sử dụng trong dự án:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:0.6b
```

Đặt tài liệu PDF hoặc Parquet vào thư mục `papers`. File Parquet cần có các
cột `id`, `title`, `text`. Giao diện mặc định chỉ lập chỉ mục 100 dòng đầu
Parquet để chạy thử nhanh; đặt "Số dòng Parquet tối đa" thành `0` nếu muốn
đọc toàn bộ dữ liệu.

Chạy giao diện web local:

```bash
ollama serve
```

Mở terminal khác, kích hoạt môi trường và chạy:

```bash
source .venv/bin/activate
streamlit run app.py
```

Hoặc chạy phiên bản terminal:

```bash
python chatbot.py
```

## Đánh giá embedding model với MTEB

Cài dependency đánh giá và chạy `qwen3-embedding:0.6b` trên Banking77:

```bash
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python evaluate_ollama_mteb.py
```

Kết quả được lưu trong thư mục `results/`.
