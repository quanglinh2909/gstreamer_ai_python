from typing import List, Optional, Sequence

from app.core.config import settings
from app.core.milvus import get_client


class FaceVectorRepository:
    @staticmethod
    def insert(embedding: Sequence[float], identity_id: int) -> int:
        client = get_client()
        result = client.insert(
            collection_name=settings.MILVUS_FACE_COLLECTION,
            data=[{
                "identity_id": int(identity_id),
                "embedding": list(embedding),
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
            data=[list(embedding)],
            limit=top_k,
            filter=filter_expr,
            output_fields=["identity_id"],
            search_params={"metric_type": "COSINE"},
        )
        hits = results[0] if results else []
        return [
            {
                "id": int(h["id"]),
                "score": float(h["distance"]),
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
