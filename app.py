import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import gradio as gr
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# In-memory document storage
INDEXED_RECORDS: List[Dict[str, str]] = []
VECTORIZER: TfidfVectorizer = None
DOC_VECTORS = None


# ==========================================
# 1. FILE PARSING ENGINES (CSV, PDF, TXT)
# ==========================================
def extract_text_from_file(file_path: str) -> List[Dict[str, str]]:
    """Extracts text content into small chunk records based on file extension."""
    path = Path(file_path)
    file_name = path.name
    extracted_records = []

    ext = path.suffix.lower()

    # Case A: CSV Files
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, nrows=3000)  # Safe limit for memory
            df = df.fillna("")
            for idx, row in df.iterrows():
                lines = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                extracted_records.append({
                    "context": "\n".join(lines),
                    "source": file_name,
                    "type": "CSV"
                })
        except Exception as e:
            print(f"Error parsing CSV {file_name}: {e}")

    # Case B: PDF Files
    elif ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                # Split large pages into ~500-word paragraphs
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                for p in paragraphs:
                    extracted_records.append({
                        "context": p,
                        "source": f"{file_name} (p. {page_num + 1})",
                        "type": "PDF"
                    })
        except Exception as e:
            print(f"Error parsing PDF {file_name}: {e}")

    # Case C: TXT / Markdown
    elif ext in [".txt", ".md", ".log"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
            for p in paragraphs:
                extracted_records.append({
                    "context": p,
                    "source": file_name,
                    "type": "TXT"
                })
        except Exception as e:
            print(f"Error parsing text file {file_name}: {e}")

    return extracted_records


# ==========================================
# 2. VECTOR INDEXING (LOW RAM TF-IDF)
# ==========================================
def process_and_index_files(uploaded_files) -> str:
    """Takes uploaded files, parses them, and updates the vector search matrix."""
    global INDEXED_RECORDS, VECTORIZER, DOC_VECTORS

    if not uploaded_files:
        return "⚠️ No files were uploaded."

    INDEXED_RECORDS = []
    total_docs = 0

    for file in uploaded_files:
        # file.name holds the local temporary path on disk
        parsed_chunks = extract_text_from_file(file.name)
        INDEXED_RECORDS.extend(parsed_chunks)
        total_docs += len(parsed_chunks)

    if not INDEXED_RECORDS:
        return "❌ Could not extract any readable text from the uploaded file(s)."

    # Fit TF-IDF Vectorizer
    corpus = [item["context"] for item in INDEXED_RECORDS]
    VECTORIZER = TfidfVectorizer(stop_words="english", max_features=12000)
    DOC_VECTORS = VECTORIZER.fit_transform(corpus)

    file_names = ", ".join([Path(f.name).name for f in uploaded_files])
    return f"✅ Successfully processed {len(uploaded_files)} file(s): **{file_names}** ({len(INDEXED_RECORDS)} total chunks indexed into memory)."


# ==========================================
# 3. SEARCH & RETRIEVAL LOGIC
# ==========================================
def search_index(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    global INDEXED_RECORDS, VECTORIZER, DOC_VECTORS

    if not INDEXED_RECORDS or VECTORIZER is None or DOC_VECTORS is None:
        return []

    # 1. Sparse vector similarity
    query_vec = VECTORIZER.transform([query])
    similarities = cosine_similarity(query_vec, DOC_VECTORS).flatten()

    query_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    scored_candidates = []

    for idx, score in enumerate(similarities):
        doc = INDEXED_RECORDS[idx]
        text_lower = doc["context"].lower()
        
        # Boost matches with exact keyword hits
        exact_matches = sum(1 for term in query_terms if term in text_lower)
        hybrid_score = (exact_matches * 3.0) + float(score)

        if hybrid_score > 0.02:
            scored_candidates.append({
                "context": doc["context"],
                "source": doc["source"],
                "type": doc["type"],
                "score": float(score),
                "hybrid_score": hybrid_score
            })

    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored_candidates[:top_k]


def answer_query(user_message: str, chat_history: list) -> Tuple[str, list]:
    if not user_message.strip():
        return "", chat_history

    if chat_history is None:
        chat_history = []

    if not INDEXED_RECORDS:
        bot_response = "⚠️ No files have been uploaded yet! Please upload a CSV, PDF, or TXT file using the sidebar first."
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": bot_response})
        return "", chat_history

    results = search_index(user_message, top_k=3)

    if not results:
        bot_response = f"No relevant information found in the uploaded documents for: *'{user_message}'*."
    else:
        output_parts = [f"### Results for: *'{user_message}'*\n"]
        for idx, item in enumerate(results, 1):
            content = item["context"].strip()
            source = item["source"]
            doc_type = item["type"]
            score = item["score"]

            output_parts.append(f"**Result #{idx}** (Source: `{source}` | Type: `{doc_type}` | Score: `{score:.3f}`)")

            if doc_type == "CSV" and ": " in content:
                for line in content.split("\n"):
                    if ": " in line:
                        k, v = line.split(": ", 1)
                        output_parts.append(f"- **{k.strip().title()}**: {v.strip()}")
            else:
                output_parts.append(f"> {content}")

            output_parts.append("\n---\n")

        bot_response = "\n".join(output_parts)

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_response})
    return "", chat_history


# ==========================================
# 4. GRADIO DUAL-COLUMN UI
# ==========================================
with gr.Blocks(title="Document Query Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Universal File RAG Assistant")
    gr.Markdown("Upload any **CSV, PDF, or TXT** file, let it index, and ask questions directly.")

    with gr.Row():
        # Left Column: File upload & Status
        with gr.Column(scale=1):
            file_uploader = gr.File(
                label="Upload File(s)",
                file_types=[".csv", ".pdf", ".txt", ".md"],
                file_count="multiple"
            )
            upload_btn = gr.Button("Index Uploaded Files", variant="primary")
            status_output = gr.Markdown("⏳ Waiting for file upload...")

            upload_btn.click(
                fn=process_and_index_files,
                inputs=[file_uploader],
                outputs=[status_output]
            )

        # Right Column: Chat Interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the uploaded document...",
                    show_label=False,
                    scale=8
                )
                submit_btn = gr.Button("Search", variant="primary", scale=1)

            clear_btn = gr.ClearButton([msg_input, chatbot])

            # Trigger on Enter or Click
            submit_btn.click(
                fn=answer_query,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            msg_input.submit(
                fn=answer_query,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)