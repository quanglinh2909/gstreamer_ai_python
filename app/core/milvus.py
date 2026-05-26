from pymilvus import DataType, MilvusClient

from app.core.config import settings

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.MILVUS_URI)
        _ensure_face_collection(_client)
    return _client


def _ensure_face_collection(client: MilvusClient) -> None:
    name = settings.MILVUS_FACE_COLLECTION
    if client.has_collection(name):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("identity_id", DataType.INT64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.FACE_EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
