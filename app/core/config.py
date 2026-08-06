from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    AI_API_BASE_URL: Optional[str] = "http://localhost:8009"
    # Bluetooth "detect" service: notified with the matched identity's MAC so it
    # can correlate the BLE beacon with the parking-lot event.
    BLE_DETECT_URL: Optional[str] = None
    PORT: Optional[int] = 8010
    MILVUS_URI: Optional[str] = "./milvus_face.db"
    MILVUS_FACE_COLLECTION: Optional[str] = "face_embeddings"
    FACE_EMBEDDING_DIM: Optional[int] = 512

    # Detector confidence used when registering a face. Registration goes
    # through the C++ engine's /inference/run with FACE_SPEC — the same
    # models the live RTSP pipeline uses — so the model paths live on the
    # engine side (GET /ai-models), not here.
    FACE_DETECT_CONF: Optional[float] = 0.5
    IS_OPEN_DOOR_WHEN_FACE_MASK: Optional[bool] = True

    # --- MoQ (Media over QUIC): duong xem thu hai, song song voi WebRTC ---
    MOQ_ENABLED: Optional[bool] = True
    # Cong UDP cho QUIC. 4443 chu khong phai 443 vi tien trinh nay khong chay
    # bang root; muon dung 443 thi phai cap CAP_NET_BIND_SERVICE.
    MOQ_PORT: Optional[int] = 4443
    # "::" chu KHONG phai "0.0.0.0": tren may nay `localhost` phan giai ra ::1
    # (chi IPv6), nen socket chi nghe IPv4 se lam trinh duyet mo trang bang
    # http://localhost bao "Opening handshake failed" — QUIC khong co ai nghe o
    # dau ben kia. bindv6only=0 nen mot socket "::" nhan ca IPv4 lan IPv6.
    MOQ_HOST: Optional[str] = "::"
    # Dia chi trinh duyet phai NOI TOI. De trong = lay dung host cua trang dang
    # mo, dung cho LAN. Phai dat khi giao dien di qua proxy/ten mien ma QUIC
    # thi noi thang: vd trang o https://ai-test.oryza.io.vn (TCP 443 qua proxy)
    # con MoQ o UDP 4443 tren mot dia chi khac. Nho rang MoQ la UDP — proxy
    # HTTP khong chuyen tiep duoc, phai mo cong that tren router.
    MOQ_PUBLIC_HOST: Optional[str] = None
    MOQ_PUBLIC_PORT: Optional[int] = None
    # Unix socket engine C++ bom access unit vao. Phai khop
    # gstreamer.moqFeedSocket trong config.json cua engine.
    MOQ_FEED_SOCKET: Optional[str] = "/tmp/vms-moq-feed.sock"
    MOQ_CERT_DIR: Optional[str] = "certs"

    class Config:
        env_file = ".env"


settings = Settings()
