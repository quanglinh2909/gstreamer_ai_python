from pymilvus import DataType, MilvusClient

from app.core.config import settings

# pymilvus hardcodes an aggressive gRPC keepalive on its channel
# (grpc.keepalive_time_ms=10000, permit_without_calls=True). Against a local
# Milvus Lite server that means a ping every 10s even on a fully idle
# connection, which the server answers with GOAWAY "too_many_pings"
# (ENHANCE_YOUR_CALM) and drops the connection — pure log noise over
# localhost, where keepalive buys nothing anyway.
#
# The old attempt at fixing this via GRPC_KEEPALIVE_* env vars was a no-op:
# grpcio does not read keepalive from the environment, only from channel
# args. The working lever is `grpc_options`, which pymilvus merges over its
# defaults when building the channel. Turning off idle keepalive removes the
# root cause; the interval bump is belt-and-suspenders for active calls.
_GRPC_OPTIONS = {
    "grpc.keepalive_permit_without_calls": 0,
    "grpc.keepalive_time_ms": 300000,
}

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.MILVUS_URI, grpc_options=_GRPC_OPTIONS)
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
