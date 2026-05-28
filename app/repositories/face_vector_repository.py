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


class FaceVectorRepository:
    @staticmethod
    def insert(embedding: Sequence[float], identity_id: int) -> int:
        client = get_client()
        result = client.insert(
            collection_name=settings.MILVUS_FACE_COLLECTION,
            data=[{
                "identity_id": int(identity_id),
                "embedding": _l2_normalize(embedding),
            }],
        )
        ids = result.get("ids") or []
        return int(ids[0]) if ids else 0

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
