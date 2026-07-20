# =============================================================
# hybrid_search.py — matched to your actual SQLite schema
# Table   : resume_chunks
# Columns : id, resume_id, candidate_name, personal_info,
#           chunk_content, chunk_type
# =============================================================

import re
import json
import os

import numpy as np
import faiss

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import DB_PATH
from db import get_connection, execute


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_PATH = "faiss_index.bin"
FAISS_MAP = "faiss_map.json"
EMBED_DIM = 384


# ─────────────────────────────────────────────
# SQLITE HELPERS
# ─────────────────────────────────────────────

def ensure_embedding_column(db_path: str = DB_PATH) -> None:
    """Add chunk_embeddings column if it doesn't exist yet."""

    with get_connection(db_path) as conn:
        cur = conn.cursor()

        cur.execute("PRAGMA table_info(resume_chunks)")
        cols = [r["name"] for r in cur.fetchall()]

        if "chunk_embeddings" not in cols:
            cur.execute(
                "ALTER TABLE resume_chunks ADD COLUMN chunk_embeddings TEXT"
            )
            conn.commit()

            print("[INFO] 'chunk_embeddings' column added to resume_chunks.")


def fetch_all_chunks(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Fetch all rows from resume_chunks.
    """

    with get_connection(db_path) as conn:

        cur = conn.cursor()

        cur.execute("""
            SELECT
                id,
                resume_id,
                candidate_name,
                personal_info,
                chunk_content,
                chunk_type,
                chunk_embeddings
            FROM resume_chunks
            ORDER BY resume_id, id
        """)

        return [dict(r) for r in cur.fetchall()]


def save_embedding(
    chunk_id: int,
    embedding: List[float],
    db_path: str = DB_PATH,
) -> None:

    execute(
        """
        UPDATE resume_chunks
        SET chunk_embeddings = ?
        WHERE id = ?
        """,
        (
            json.dumps(embedding),
            chunk_id,
        ),
        db_path,
    )


# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────

_model = None


def get_model() -> SentenceTransformer:
    global _model

    if _model is None:
        print("[INFO] Loading embedding model...")
        _model = SentenceTransformer(EMBED_MODEL)

    return _model


def embed_texts(texts: List[str]) -> np.ndarray:

    model = get_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype="float32")


# ─────────────────────────────────────────────
# FAISS INDEX
# ─────────────────────────────────────────────

def build_index(
    db_path: str = DB_PATH,
    force: bool = False,
) -> None:

    ensure_embedding_column(db_path)

    rows = fetch_all_chunks(db_path)

    if not rows:
        print("[WARN] No rows found in resume_chunks.")
        return

    if (
        not force
        and os.path.exists(FAISS_PATH)
        and os.path.exists(FAISS_MAP)
    ):
        print("[INFO] FAISS index already exists. Use force=True to rebuild.")
        return

    print(f"[INFO] Embedding {len(rows)} chunks...")

    texts = [row["chunk_content"] or "" for row in rows]

    embeddings = embed_texts(texts)

    for row, embedding in zip(rows, embeddings):
        save_embedding(
            row["id"],
            embedding.tolist(),
            db_path,
        )

    index = faiss.IndexFlatIP(EMBED_DIM)

    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)

    faiss_map = {
        str(i): rows[i]["id"]
        for i in range(len(rows))
    }

    with open(FAISS_MAP, "w") as f:
        json.dump(faiss_map, f)

    print(
        f"[OK] FAISS index built — "
        f"{len(rows)} chunks indexed → {FAISS_PATH}"
    )
    # ─────────────────────────────────────────────
# FAISS LOADER
# ─────────────────────────────────────────────

def load_index():

    index = faiss.read_index(FAISS_PATH)

    with open(FAISS_MAP, "r") as f:
        faiss_map = json.load(f)

    return index, faiss_map


# ─────────────────────────────────────────────
# BM25 SEARCH
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(
    query: str,
    top_k: int = 10,
    resume_id: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:

    rows = fetch_all_chunks(db_path)

    corpus = [
        tokenize(row["chunk_content"] or "")
        for row in rows
    ]

    bm25 = BM25Okapi(corpus)

    scores = bm25.get_scores(tokenize(query))

    results = [
        {
            **row,
            "bm25_score": float(score),
        }
        for row, score in zip(rows, scores)
    ]

    if resume_id:
        results = [
            row
            for row in results
            if row["resume_id"] == resume_id
        ]

    results.sort(
        key=lambda x: x["bm25_score"],
        reverse=True,
    )

    return results[:top_k]


# ─────────────────────────────────────────────
# VECTOR SEARCH
# ─────────────────────────────────────────────

def vector_search(
    query: str,
    top_k: int = 10,
    resume_id: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:

    index, faiss_map = load_index()

    all_chunks = {
        row["id"]: row
        for row in fetch_all_chunks(db_path)
    }

    query_embedding = embed_texts([query])

    scores, positions = index.search(
        query_embedding,
        k=min(top_k * 3, index.ntotal),
    )

    results = []

    for score, position in zip(scores[0], positions[0]):

        if position == -1:
            continue

        chunk_id = faiss_map[str(position)]

        row = all_chunks.get(chunk_id)

        if row is None:
            continue

        if resume_id and row["resume_id"] != resume_id:
            continue

        results.append(
            {
                **row,
                "vector_score": round(float(score), 6),
            }
        )

    return results[:top_k]


# ─────────────────────────────────────────────
# HYBRID SEARCH
# ─────────────────────────────────────────────

def _normalize(scores: List[float]) -> List[float]:

    if not scores:
        return []

    low = min(scores)
    high = max(scores)

    if low == high:
        return [1.0] * len(scores)

    return [
        (score - low) / (high - low)
        for score in scores
    ]


def hybrid_search(
    query: str,
    resume_id: Optional[str] = None,
    top_k: int = 10,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:

    fetch_k = max(top_k * 2, 20)

    bm25_hits = bm25_search(
        query=query,
        top_k=fetch_k,
        resume_id=resume_id,
        db_path=db_path,
    )

    vector_hits = vector_search(
        query=query,
        top_k=fetch_k,
        resume_id=resume_id,
        db_path=db_path,
    )

    merged: Dict[int, Dict[str, Any]] = {}

    for hit in bm25_hits:

        merged[hit["id"]] = {
            **hit,
            "bm25_score": hit["bm25_score"],
            "vector_score": 0.0,
        }

    for hit in vector_hits:

        chunk_id = hit["id"]

        if chunk_id in merged:

            merged[chunk_id]["vector_score"] = hit["vector_score"]

        else:

            merged[chunk_id] = {
                **hit,
                "bm25_score": 0.0,
                "vector_score": hit["vector_score"],
            }

    ids = list(merged.keys())

    bm25_norm = _normalize(
        [merged[i]["bm25_score"] for i in ids]
    )

    vector_norm = _normalize(
        [merged[i]["vector_score"] for i in ids]
    )

    for chunk_id, b, v in zip(
        ids,
        bm25_norm,
        vector_norm,
    ):

        merged[chunk_id]["bm25_norm"] = round(b, 6)
        merged[chunk_id]["vector_norm"] = round(v, 6)

        merged[chunk_id]["hybrid_score"] = round(
            bm25_weight * b + vector_weight * v,
            6,
        )

    ranked = sorted(
        merged.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    return ranked[:top_k]
