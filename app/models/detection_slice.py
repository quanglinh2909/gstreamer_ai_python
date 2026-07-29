from typing import Optional

from sqlalchemy import BigInteger, Index, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DetectionSlice(Base):
    """Quỹ đạo của MỘT track trong MỘT lát thời gian (mặc định 10 giây).

    Vì sao chọn đơn vị này (đã đo trên máy thật):
      - mỗi khung một dòng  -> ~1 TRIỆU dòng/ngày/job: bảng không sống nổi.
      - mỗi track một dòng  -> bbox-hợp của cả track phủ ~34% khung hình, nên
        vẽ một vùng nhỏ để tìm sự kiện thì cái gì cũng khớp -> vô dụng.
      - track × lát 10s     -> ~26 nghìn dòng/ngày, bbox lát còn ~12% khung.
        Lát nhỏ hơn (2s) làm số dòng gấp 4 mà bbox gần như không nhỏ thêm.

    Dung lượng ~26 MB/ngày, tức ~0,1% so với video (~24 GB/ngày) — nên tiêu chí
    thiết kế là TRUY XUẤT tốt, không phải tiết kiệm byte.
    """

    __tablename__ = "detection_slice"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(255), nullable=False)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False)
    ai_type: Mapped[str] = mapped_column(String(64), nullable=True)
    class_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    # Id theo dõi của tracker. Cùng tid + các lát liền nhau = một "sự kiện".
    tid: Mapped[Optional[int]] = mapped_column(nullable=True)

    # BigInteger BẮT BUỘC: epoch MILI giây (~1,78e12) vượt xa int32 mà
    # `Mapped[int]` mặc định ánh xạ sang. Các bảng event cũ dùng epoch GIÂY nên
    # int32 còn đủ tới 2038 — đừng chép kiểu từ đó sang đây.
    t_start: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t_end: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # bbox HỢP của mọi khung trong lát, chuẩn hoá [0,1]. Lọc theo vùng vẽ chỉ
    # cần 4 phép so sánh (giao nhau hình chữ nhật) — KHÔNG cần GiST/PostGIS:
    # truy vấn luôn bị chặn trước bởi (camera_id, t_start) nên tập quét đã nhỏ.
    bx1: Mapped[float] = mapped_column(nullable=False)
    by1: Mapped[float] = mapped_column(nullable=False)
    bx2: Mapped[float] = mapped_column(nullable=False)
    by2: Mapped[float] = mapped_column(nullable=False)

    # Lưới 16×16 = 256 bit = 32 byte: track chạm vào những ô nào trong lát.
    # Dùng để lọc TINH sau bbox mà KHÔNG phải giải mã `path` — bbox là bao lồi
    # nên người đi chéo màn hình sẽ khớp cả những vùng họ chưa từng bước vào.
    cells: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    best_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    n: Mapped[int] = mapped_column(nullable=False, default=0)

    # Quỹ đạo nén nhị phân. Định dạng (little-endian):
    #   [u8 version=1][u16 n] rồi n bản ghi:
    #   [u16 dt_ms kể từ mẫu trước][u16 x1][u16 y1][u16 x2][u16 y2][u8 score]
    #   x/y = giá trị chuẩn hoá × 65535; score = score × 255.
    #   -> 3 + 11×n byte (JSON tương đương tốn gấp ~16 lần).
    path: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    # Pose, chỉ có nếu model sinh keypoints. Định dạng:
    #   [u8 version=1][u8 K số điểm][u16 n] rồi n×K bản ghi [u16 x][u16 y][u8 s]
    #   -> 4 + 5×K×n byte.
    kps: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    __table_args__ = (
        # Mọi truy vấn đều là "camera này, khoảng thời gian này" — cả xem lại
        # lẫn tìm theo vùng. Đây là chỉ mục gánh toàn bộ.
        Index("ix_detection_slice_cam_time", "camera_id", "t_start"),
        Index("ix_detection_slice_cam_end", "camera_id", "t_end"),
    )
