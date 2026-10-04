import os
import re
import math
import uuid
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from langchain_community.document_loaders import CSVLoader
import chromadb

# Ensure data directory exists
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vector_store"

# ==========================================
# 1. LOAD CSV DOCUMENTS & BUILD VOCABULARY
# ==========================================
def process_specific_csvs(file_paths: List[Path]):
    all_documents = []
    csv_files = [f for f in file_paths if f.exists()]
    print(f"Found {len(csv_files)} valid CSV file(s).")

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        try:
            loader = CSVLoader(file_path=str(csv_file), encoding="utf-8")
            documents = loader.load()
            # For demonstration on low RAM, limit to top 1,000 rows if huge:
            # documents = documents[:1000]
            for doc in documents:
                doc.metadata['source_file'] = csv_file.name
                doc.metadata['file_type'] = 'csv'
            all_documents.extend(documents)
            print(f"  Loaded {len(documents)} rows from {csv_file.name}")
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

target_files = [
    DATA_DIR / "Spotify Streaming Performance Dataset.csv",
    DATA_DIR / "women_clothing_50k.csv"
]

chunks = process_specific_csvs(target_files)

if chunks:
    texts = [doc.page_content for doc in chunks]
    corpus = " ".join(texts[:500])  # Build character set from sample
    chars = sorted(list(set(corpus)))
else:
    # Fallback default characters if files aren't uploaded yet
    chars = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:-\n"))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = max(len(chars), 1)

# ==========================================
# 2. AUTO DATASET ROUTER
# ==========================================
class DatasetRouter:
    def __init__(self, documents):
        self.file_keywords = defaultdict(set)
        self._build_keyword_registry(documents)

    def _build_keyword_registry(self, documents):
        for doc in documents:
            source = doc.metadata.get('source_file')
            if not source:
                continue

            for line in doc.page_content.split('\n'):
                if ': ' in line:
                    key, val = line.split(': ', 1)
                    for word in re.findall(r'\w+', key.lower()):
                        if len(word) > 2:
                            self.file_keywords[source].add(word)

                    val_words = [w for w in re.findall(r'\w+', val.lower()) if 3 <= len(w) <= 15]
                    self.file_keywords[source].update(val_words[:5])

    def route_query(self, query: str) -> str:
        query_terms = set(re.findall(r'\w+', query.lower()))
        scores = {}

        for source_file, kws in self.file_keywords.items():
            scores[source_file] = len(query_terms.intersection(kws))

        if not scores:
            return None
        best_match = max(scores, key=scores.get)
        return best_match if scores[best_match] > 0 else None

# ==========================================
# 3. TRANSFORMER MODEL ARCHITECTURE
# ==========================================
class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, block_size, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, block_size, num_heads, head_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, block_size, head_size, dropout) for _ in range(num_heads)
        ])
        self.projection = nn.Linear(num_heads * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, block_size, num_heads, head_size, ff_hidden_dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, block_size, num_heads, head_size, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, head_size, ff_hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim, max_len=10000)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, block_size, num_heads, head_size, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.tok_emb(x)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    def get_pooled_embedding(self, idx):
        x = self.tok_emb(idx)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        pooled = torch.mean(x, dim=1)
        return F.normalize(pooled, p=2, dim=1)

# ==========================================
# 4. MODEL INITIALIZATION & EMBEDDING
# ==========================================
torch.manual_seed(1337)
block_size = 256
embed_dim = 128   # Reduced slightly for lower RAM usage on Render
num_heads = 4
head_size = embed_dim // num_heads
ff_hidden_dim = 4 * embed_dim
num_layers = 2    # Lower layer count for fast CPU inference
dropout = 0.2

device = torch.device('cpu')  # Force CPU on Render web services
model = GPTLanguageModel(vocab_size, embed_dim, block_size, num_heads, head_size, ff_hidden_dim, num_layers, dropout)

class CustomEmbeddingManager:
    def __init__(self, model, stoi, block_size=256, device='cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.stoi = stoi
        self.block_size = block_size

    def _encode_text(self, text: str) -> List[int]:
        tokens = [self.stoi[c] for c in text if c in self.stoi]
        if not tokens:
            tokens = [0]
        return tokens[:self.block_size]

    @torch.no_grad()
    def generate_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokenized_batch = [self._encode_text(t) for t in batch_texts]
            max_len = max(len(seq) for seq in tokenized_batch)
            padded_batch = [seq + [0] * (max_len - len(seq)) for seq in tokenized_batch]

            idx_tensor = torch.tensor(padded_batch, dtype=torch.long, device=self.device)
            batch_emb = self.model.get_pooled_embedding(idx_tensor)
            all_embeddings.append(batch_emb.cpu().numpy())

        return np.vstack(all_embeddings)

embedding_manager = CustomEmbeddingManager(model=model, stoi=stoi, block_size=block_size, device=device)

# ==========================================
# 5. VECTOR STORE
# ==========================================
class VectorStore:
    def __init__(self, collection_name: str = "csv_documents", persist_directory: str = str(VECTOR_DIR)):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "CSV document embeddings"}
        )

    def add_documents(self, documents: List[Any], embeddings: np.ndarray, batch_size: int = 250):
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]

            ids = [f"doc_{uuid.uuid4().hex[:8]}_{j}" for j in range(i, i + len(batch_docs))]
            metadatas = [
                {
                    **dict(doc.metadata),
                    "doc_index": i + idx,
                    "context_length": len(doc.page_content)
                }
                for idx, doc in enumerate(batch_docs)
            ]
            document_texts = [doc.page_content for doc in batch_docs]

            self.collection.add(
                ids=ids,
                embeddings=batch_embs.tolist(),
                metadatas=metadatas,
                documents=document_texts
            )

vector_store = VectorStore()

# Only index if documents exist and collection is empty
if chunks and vector_store.collection.count() == 0:
    print("Generating embeddings and writing to vector store...")
    texts = [doc.page_content for doc in chunks]
    embs = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embs)

# ==========================================
# 6. RETRIEVER & CHAT INTERFACE
# ==========================================
class RAGRetriever:
    def __init__(self, vector_store, embedding_manager, router):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.router = router

    def retrieve(self, query: str, top_k: int = 3):
        target_file = self.router.route_query(query)
        where_filter = {"source_file": target_file} if target_file else None

        query_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 10, 50),
                where=where_filter
            )

            if not results['documents'] or not results['documents'][0]:
                return []

            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            scored_candidates = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                doc_lower = doc.lower()
                exact_hits = sum(1 for kw in query_words if kw in doc_lower)
                base_sim = 1.0 - dist
                hybrid_score = (exact_hits * 10.0) + base_sim

                scored_candidates.append({
                    'context': doc,
                    'metadata': meta,
                    'similarity_score': base_sim,
                    'hybrid_score': hybrid_score
                })

            scored_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return scored_candidates[:top_k]
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

router = DatasetRouter(chunks)
rag_retriever = RAGRetriever(vector_store, embedding_manager, router)

def answer_user_query(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a valid search query."

    if vector_store.collection.count() == 0:
        return "No documents indexed. Please ensure the CSV files are present in the data folder."

    results = rag_retriever.retrieve(message, top_k=3)
    if not results:
        return f"No matching records found for: '{message}'."

    output_lines = [f"### Results for: '{message}'\n"]
    for idx, item in enumerate(results, 1):
        content = item["context"].strip()
        source = item["metadata"].get("source_file", "Unknown")
        score = item.get("similarity_score", 0.0)

        output_lines.append(f"**Result #{idx}** (Source: `{source}` | Match: `{score:.4f}`)")
        if ": " in content and "\n" in content:
            for line in content.split("\n"):
                if ": " in line:
                    k, v = line.split(": ", 1)
                    output_lines.append(f"- **{k.strip().title()}**: {v.strip()}")
        else:
            output_lines.append(f"> {content}")
        output_lines.append("\n---\n")

    return "\n".join(output_lines)

# Gradio Interface configured for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    demo = gr.ChatInterface(
        fn=answer_user_query,
        title="Custom RAG CSV Assistant",
        description="Query CSV datasets using a character-level transformer and hybrid retrieval."
    )
    demo.launch(server_name="0.0.0.0", server_port=port)