# Chatbot RAG

Dự án thử nghiệm RAG: đọc tài liệu PDF, chia văn bản thành các chunk, tạo embedding bằng OpenAI và lưu vector trong FAISS.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Tạo file `.env` trong thư mục dự án:

```env
OPENAI_API_KEY=your_api_key_here
```

Đặt tài liệu PDF vào thư mục `papers`, sau đó chạy:

```powershell
python chatbot.py
```

