import os

# gRPC keepalive must be set BEFORE pymilvus imports grpcio. Milvus servers
# (including Lite) reject pings sent more often than ~60s with GOAWAY
# "too_many_pings". pymilvus defaults to 10s — too aggressive — so the log
# fills with ENHANCE_YOUR_CALM warnings. Align with server tolerance.
os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "60000")
os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "10000")
os.environ.setdefault("GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", "0")

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
    if not client.has_collection(name):
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

    # Milvus Lite doesn't keep collections resident across process restarts;
    # search() returns code=101 unless we explicitly load on every startup.
    client.load_collection(name)


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
