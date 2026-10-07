import os
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ==========================================
# 1. LIGHTWEIGHT DATA LOADER (LOW RAM)
# ==========================================
def load_csv_data(file_paths: List[Path], max_rows_per_file: int = 2000):
    """
    Loads and slices CSV records directly using Pandas to stay well under 512MB RAM.
    """
    all_records = []
    
    for path in file_paths:
        if not path.exists():
            print(f"Skipping missing file: {path.name}")
            continue
            
        print(f"Loading {path.name} (capped at {max_rows_per_file} rows)...")
        # Load only the first N rows to guarantee staying under 512MB
        try:
            df = pd.read_csv(path, nrows=max_rows_per_file)
            # Fill NaNs
            df = df.fillna("")
            
            for _, row in df.iterrows():
                # Format into readable key-value context text
                content_lines = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                page_content = "\n".join(content_lines)
                
                all_records.append({
                    "context": page_content,
                    "source_file": path.name,
                })
        except Exception as e:
            print(f"Error loading {path.name}: {e}")

    print(f"Loaded {len(all_records)} total records into memory.")
    return all_records

target_files = [
    DATA_DIR / "Spotify Streaming Performance Dataset.csv",
    DATA_DIR / "women_clothing_50k.csv"
]

records = load_csv_data(target_files, max_rows_per_file=2000)

# ==========================================
# 2. AUTO DATASET ROUTER
# ==========================================
class DatasetRouter:
    def __init__(self, records: List[Dict[str, str]]):
        self.file_keywords = defaultdict(set)
        self._build_keyword_registry(records)

    def _build_keyword_registry(self, records):
        for rec in records:
            source = rec["source_file"]
            for line in rec["context"].split("\n"):
                if ": " in line:
                    key, val = line.split(": ", 1)
                    for word in re.findall(r"\w+", key.lower()):
                        if len(word) > 2:
                            self.file_keywords[source].add(word)

                    val_words = [w for w in re.findall(r"\w+", val.lower()) if 3 <= len(w) <= 15]
                    self.file_keywords[source].update(val_words[:4])

    def route_query(self, query: str) -> str:
        query_terms = set(re.findall(r"\w+", query.lower()))
        scores = {}
        for source_file, kws in self.file_keywords.items():
            scores[source_file] = len(query_terms.intersection(kws))
        if not scores:
            return None
        best_match = max(scores, key=scores.get)
        return best_match if scores[best_match] > 0 else None

# ==========================================
# 3. TF-IDF VECTOR RETRIEVER (< 50MB RAM)
# ==========================================
class MemorySafeRetriever:
    def __init__(self, records: List[Dict[str, str]], router: DatasetRouter):
        self.records = records
        self.router = router
        self.corpus = [r["context"] for r in records]
        
        # Fit vectorizer on vocabulary
        if self.corpus:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            self.doc_vectors = self.vectorizer.fit_transform(self.corpus)
        else:
            self.vectorizer = None
            self.doc_vectors = None

    def retrieve(self, query: str, top_k: int = 3):
        if not self.corpus or self.vectorizer is None:
            return []

        # 1. Routing
        target_file = self.router.route_query(query)
        
        # 2. Vector search via Cosine Similarity
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        scored_candidates = []

        for idx, score in enumerate(similarities):
            rec = self.records[idx]
            
            # Apply source filter if routed
            if target_file and rec["source_file"] != target_file:
                continue

            doc_text = rec["context"].lower()
            exact_hits = sum(1 for kw in query_words if kw in doc_text)
            
            # Boost exact keyword matches
            hybrid_score = (exact_hits * 5.0) + float(score)

            if hybrid_score > 0.05:
                scored_candidates.append({
                    "context": rec["context"],
                    "source_file": rec["source_file"],
                    "score": float(score),
                    "hybrid_score": hybrid_score
                })

        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored_candidates[:top_k]

router = DatasetRouter(records)
retriever = MemorySafeRetriever(records, router)

# ==========================================
# 4. CHAT LOGIC & GRADIO UI
# ==========================================
def answer_user_query(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a valid search query."

    if not records:
        return "No CSV records found. Please ensure the CSV files are placed inside the `data/` folder."

    results = retriever.retrieve(message, top_k=3)
    if not results:
        return f"No matching records found for: *'{message}'*."

    output_lines = [f"### Results for: *'{message}'*\n"]
    for idx, item in enumerate(results, 1):
        content = item["context"].strip()
        source = item["source_file"]
        score = item["score"]

        output_lines.append(f"**Result #{idx}** (Source: `{source}` | Similarity: `{score:.3f}`)")
        for line in content.split("\n"):
            if ": " in line:
                k, v = line.split(": ", 1)
                output_lines.append(f"- **{k.strip().title()}**: {v.strip()}")
        output_lines.append("\n---\n")

    return "\n".join(output_lines)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo = gr.ChatInterface(
        fn=answer_user_query,
        title="Lightweight CSV Assistant",
        description="Low-memory RAG Assistant optimized for Render Free Tier (512MB RAM)."
    )
    demo.launch(server_name="0.0.0.0", server_port=port)