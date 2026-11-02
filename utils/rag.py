from __future__ import annotations

import os
import glob
import json
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from utils.openai_client import get_client


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str
    embedding: Optional[List[float]] = None


# ==========================================================
# Basic math
# ==========================================================
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ==========================================================
# Embeddings
# ==========================================================
def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> List[List[float]]:
    """
    Batch embeddings voor snelheid en stabiliteit.
    """
    if not texts:
        return []

    client = get_client()
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in resp.data)

    return all_embeddings


# ==========================================================
# File loading
# ==========================================================
def read_text_files(folder: str) -> List[Dict[str, str]]:
    paths: List[str] = []
    for pattern in ("*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(folder, pattern)))

    docs: List[Dict[str, str]] = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append(
                {
                    "source_file": os.path.basename(path),
                    "full_path": os.path.abspath(path),
                    "text": text,
                }
            )
    return docs


# ==========================================================
# Chunking
# ==========================================================
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Simpele en snelle character-based chunking.
    Iets grotere chunks = meestal minder chunks = sneller.
    """
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)

    return chunks


# ==========================================================
# Cache helpers
# ==========================================================
def _compute_rag_signature(
    rag_folder: str,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> str:
    """
    Signature van:
    - relevante bestanden
    - modified times
    - sizes
    - config
    """
    h = hashlib.sha256()
    h.update(os.path.abspath(rag_folder).encode("utf-8"))
    h.update(embedding_model.encode("utf-8"))
    h.update(str(chunk_size).encode("utf-8"))
    h.update(str(overlap).encode("utf-8"))

    paths: List[str] = []
    for pattern in ("*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(rag_folder, pattern)))

    for path in sorted(paths):
        st = os.stat(path)
        h.update(os.path.basename(path).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))

    return h.hexdigest()


def _default_cache_path(
    rag_folder: str,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> str:
    cache_dir = os.path.join(rag_folder, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    signature = _compute_rag_signature(
        rag_folder=rag_folder,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return os.path.join(cache_dir, f"chunk_index_{signature}.json")


def _save_chunk_index(cache_path: str, chunks: List[Chunk]) -> None:
    payload = {
        "chunks": [asdict(c) for c in chunks]
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_chunk_index(cache_path: str) -> List[Chunk]:
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: List[Chunk] = []
    for item in payload.get("chunks", []):
        out.append(
            Chunk(
                chunk_id=item["chunk_id"],
                source_file=item["source_file"],
                text=item["text"],
                embedding=item.get("embedding"),
            )
        )
    return out


# ==========================================================
# Index building
# ==========================================================
def build_chunk_index(
    rag_folder: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1200,
    overlap: int = 150,
    use_cache: bool = True,
    cache_path: Optional[str] = None,
) -> List[Chunk]:
    """
    Bouwt of laadt een chunk-index met embeddings.
    """
    if cache_path is None:
        cache_path = _default_cache_path(
            rag_folder=rag_folder,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if use_cache and os.path.exists(cache_path):
        return _load_chunk_index(cache_path)

    docs = read_text_files(rag_folder)
    raw_chunks: List[Chunk] = []

    for doc_idx, doc in enumerate(docs, start=1):
        pieces = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk_idx, piece in enumerate(pieces, start=1):
            raw_chunks.append(
                Chunk(
                    chunk_id=f"doc{doc_idx}_chunk{chunk_idx}",
                    source_file=doc["source_file"],
                    text=piece,
                )
            )

    if not raw_chunks:
        if use_cache:
            _save_chunk_index(cache_path, [])
        return []

    embeddings = embed_texts(
        [c.text for c in raw_chunks],
        model=embedding_model,
        batch_size=100,
    )

    for chunk, emb in zip(raw_chunks, embeddings):
        chunk.embedding = emb

    if use_cache:
        _save_chunk_index(cache_path, raw_chunks)

    return raw_chunks


# ==========================================================
# Retrieval
# ==========================================================
def retrieve_top_k(
    query: str,
    chunks: List[Chunk],
    top_k: int = 3,
    embedding_model: str = "text-embedding-3-small",
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []

    if not chunks:
        return []

    query_emb = embed_texts([query], model=embedding_model, batch_size=1)[0]

    scored: List[Dict[str, Any]] = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        score = cosine_similarity(query_emb, chunk.embedding)
        scored.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "text": chunk.text,
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ==========================================================
# Context builder
# ==========================================================
def build_augmented_context(
    case_context: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    rag_blocks: List[str] = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        rag_blocks.append(
            f"[Bron {i} | {ch['source_file']} | score={ch['score']:.4f}]\n{ch['text']}"
        )

    rag_text = "\n\n".join(rag_blocks) if rag_blocks else "Geen aanvullende broninformatie gevonden."

    return (
        "CASE_CONTEXT:\n"
        f"{case_context.strip()}\n\n"
        "AANVULLENDE_MENU_ENGINEERING_CONTEXT:\n"
        f"{rag_text}"
    )