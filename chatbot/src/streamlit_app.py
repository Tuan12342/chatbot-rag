import streamlit as st

from pdfChat import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    build_messages,
    load_rag_system,
    stream_answer,
)


@st.cache_resource(show_spinner=False)
def cached_load_rag_system(index_version: tuple):
    return load_rag_system(index_version)


def show_sources(sources: list[dict]):
    if not sources:
        return

    with st.expander("Xem các đoạn tài liệu đã sử dụng"):
        for source in sources:
            st.markdown(
                f"**{source['label']} — {source['source']}, "
                f"{source['location']}**"
            )
            st.write(source["text"])
            st.divider()


st.set_page_config(page_title="PDF Chat", page_icon="📚", layout="wide")
st.title(" Chatbot pdf")

with st.sidebar:
    st.header("Cấu hình tìm kiếm")
    candidate_k = st.slider(
        "Số ứng viên mỗi nhánh",
        min_value=4,
        max_value=30,
        value=15,
    )
    final_k = st.slider(
        "Số đoạn đưa vào mô hình",
        min_value=1,
        max_value=8,
        value=4,
    )
    st.caption(f"Embedding: {EMBEDDING_MODEL}")

    if st.button("Nạp lại chỉ mục", use_container_width=True):
        cached_load_rag_system.clear()
        st.rerun()

    if st.button("Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

index_files = [
    INDEX_DIR / "index.faiss",
    INDEX_DIR / "documents.jsonl",
    INDEX_DIR / "config.json",
]
index_version = tuple(
    (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    for path in index_files
    if path.exists()
)

try:
    with st.spinner("Đang nạp FAISS và khởi tạo BM25..."):
        retriever, documents, llm, config = cached_load_rag_system(index_version)
except Exception:
    st.stop()

st.success(f"Sẵn sàng: {len(documents):,} đoạn · ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

if question := st.chat_input("Nhập câu hỏi về tài liệu..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    sources = []
    with st.chat_message("assistant"):
        try:
            with st.spinner("..."):
                candidates = retriever.search(question, candidate_k)
            selected_documents = candidates[:final_k]

            if not selected_documents:
                answer = "Không tìm thấy nội dung phù hợp trong tài liệu."
                st.markdown(answer)
            else:
                messages, sources = build_messages(question, selected_documents)
                answer = st.write_stream(stream_answer(llm, messages))
                if not isinstance(answer, str) or not answer.strip():
                    answer = "Tài liệu không có đủ thông tin để trả lời."
                    st.markdown(answer)
                show_sources(sources)
        except Exception:
            st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
