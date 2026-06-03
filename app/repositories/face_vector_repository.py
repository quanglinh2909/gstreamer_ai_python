import threading
from typing import List, Optional, Sequence

import numpy as np

from app.core.config import settings
from app.core.milvus import get_client


def _l2_normalize(embedding: Sequence[float]) -> List[float]:
    """Defensive L2-normalize. AdaFace nominally outputs unit-norm embeddings
    but precision can drift through FP16/JSON round-trips. Normalising on
    both insert and search guarantees consistent cosine scores in [-1, 1]."""
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec.tolist()
    return (vec / norm).tolist()


def _l2_normalize_np(embedding: Sequence[float]) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    return vec / norm


class FaceVectorRepository:
    # ---- In-memory mirror of the Milvus face collection -------------------
    # The ALPR/face pipeline matches on every frame; a Milvus gRPC round-trip
    # per frame overloaded the server (keepalive GOAWAYs) and was slow. The
    # collection uses a FLAT/COSINE index — i.e. brute force anyway — so we
    # keep a normalised embedding matrix in RAM and do the cosine search with
    # numpy. Milvus stays the source of truth (insert/delete still go there).
    _cache_lock = threading.RLock()
    _ids: List[int] = []                      # milvus primary keys, row-aligned
    _identity_ids: np.ndarray = np.empty((0,), dtype=np.int64)
    _matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)  # (N, dim), L2-normed

    @classmethod
    def load_all_to_cache(cls) -> None:
        """Pull every face vector from Milvus into the in-memory matrix. Call
        once at startup (after the collection is loaded)."""
        client = get_client()
        rows = client.query(
            collection_name=settings.MILVUS_FACE_COLLECTION,
            filter="id >= 0",
            output_fields=["id", "identity_id", "embedding"],
            limit=16384,
        )
        ids: List[int] = []
        identity_ids: List[int] = []
        vectors: List[np.ndarray] = []
        for r in rows:
            ids.append(int(r["id"]))
            identity_ids.append(int(r["identity_id"]))
            vectors.append(_l2_normalize_np(r["embedding"]))
        dim = settings.FACE_EMBEDDING_DIM
        matrix = (
            np.vstack(vectors).astype(np.float32)
            if vectors else np.empty((0, dim), dtype=np.float32)
        )
        with cls._cache_lock:
            cls._ids = ids
            cls._identity_ids = np.asarray(identity_ids, dtype=np.int64)
            cls._matrix = matrix
        print(f"[face vectors] loaded {len(ids)} embeddings into cache")

    @classmethod
    def _cache_add(cls, milvus_id: int, identity_id: int, embedding: np.ndarray) -> None:
        with cls._cache_lock:
            row = embedding.reshape(1, -1).astype(np.float32)
            if cls._matrix.size == 0:
                cls._matrix = row
            else:
                cls._matrix = np.vstack([cls._matrix, row])
            cls._ids.append(int(milvus_id))
            cls._identity_ids = np.append(cls._identity_ids, np.int64(identity_id))

    @classmethod
    def _cache_remove_identity(cls, identity_id: int) -> None:
        with cls._cache_lock:
            if not cls._ids:
                return
            keep = cls._identity_ids != np.int64(identity_id)
            cls._ids = [i for i, k in zip(cls._ids, keep) if k]
            cls._identity_ids = cls._identity_ids[keep]
            cls._matrix = cls._matrix[keep] if cls._matrix.size else cls._matrix

    @classmethod
    def search_cached(
        cls,
        embedding: Sequence[float],
        top_k: int = 5,
        identity_id: Optional[int] = None,
    ) -> List[dict]:
        """Brute-force cosine search against the in-memory matrix. Same shape of
        result as search() so callers are interchangeable."""
        q = _l2_normalize_np(embedding)
        with cls._cache_lock:
            matrix = cls._matrix
            ids = cls._ids
            iids = cls._identity_ids
            if not ids or matrix.size == 0 or q.shape[0] != matrix.shape[1]:
                return []
            # Both sides are L2-normalised, so the dot product is the cosine.
            sims = matrix @ q
            mask = (iids == np.int64(identity_id)) if identity_id is not None else None
            if mask is not None:
                idx_pool = np.nonzero(mask)[0]
                if idx_pool.size == 0:
                    return []
                order = idx_pool[np.argsort(-sims[idx_pool])][:top_k]
            else:
                order = np.argsort(-sims)[:top_k]
            return [
                {
                    "id": int(ids[i]),
                    "score": float(sims[i]),
                    "identity_id": int(iids[i]),
                }
                for i in order
            ]

    # ---- Milvus writes (source of truth) ----------------------------------
    @staticmethod
    def insert(embedding: Sequence[float], identity_id: int) -> int:
        client = get_client()
        normalized = _l2_normalize(embedding)
        result = client.insert(
            collection_name=settings.MILVUS_FACE_COLLECTION,
            data=[{
                "identity_id": int(identity_id),
                "embedding": normalized,
            }],
        )
        ids = result.get("ids") or []
        milvus_id = int(ids[0]) if ids else 0
        # Keep the in-memory cache in sync so the new face is matchable
        # immediately without reloading from Milvus.
        FaceVectorRepository._cache_add(
            milvus_id, int(identity_id), np.asarray(normalized, dtype=np.float32),
        )
        return milvus_id

    @staticmethod
    def search(
        embedding: Sequence[float],
        top_k: int = 5,
        identity_id: Optional[int] = None,
    ) -> List[dict]:
        client = get_client()
        filter_expr = f"identity_id == {int(identity_id)}" if identity_id is not None else ""
        results = client.search(
            collection_name=settings.MILVUS_FACE_COLLECTION,
            data=[_l2_normalize(embedding)],
            limit=top_k,
            filter=filter_expr,
            output_fields=["identity_id"],
        )
        hits = results[0] if results else []
        # pymilvus 3.x returns cosine *distance* (1 - similarity) in the
        # `distance` field for COSINE metric — opposite of the legacy 2.x
        # behaviour. Convert back to similarity so callers can compare
        # against a threshold the natural way ("higher = better match").
        return [
            {
                "id": int(h["id"]),
                "score": 1.0 - float(h["distance"]),
                "identity_id": h["entity"].get("identity_id"),
            }
            for h in hits
        ]

    @staticmethod
    def delete_by_identity(identity_id: int) -> None:
        client = get_client()
        client.delete(
            collection_name=settings.MILVUS_FACE_COLLECTION,
            filter=f"identity_id == {int(identity_id)}",
        )
        FaceVectorRepository._cache_remove_identity(int(identity_id))
