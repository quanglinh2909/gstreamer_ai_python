import time

import cv2
import numpy as np


class PushEventMetadata:
    def __init__(self):
        self.event_queue = []
        self._last_id = 0

    def _next_id(self) -> int:
        """Id DUY NHẤT cho mỗi sự kiện khẩu trang.

        Khẩu trang là loại DUY NHẤT không có bảng DB, nên trước đây nó gửi
        thẳng `track_uuid` (id của tracker) làm `id`. Mọi bên tiêu thụ đều
        ngầm hiểu `id` là định danh của SỰ KIỆN và khử trùng theo nó — thế là
        mọi sự kiện sau của cùng một track bị vứt lặng lẽ. Cụ thể: bảng sự
        kiện ở trang Xem lại khử trùng bằng `${tab}-${ev.id}`, nên track 0 báo
        "có khẩu trang" trước thì mọi lần "không khẩu trang" sau đó của chính
        track 0 biến mất. Tường Live View thì không dính vì nó cộng thêm một
        bộ đếm vào khoá.

        Lấy mốc thời gian mili giây làm nền để id vẫn tăng và không đụng hàng
        sau khi khởi động lại; `max(...)+1` lo trường hợp hai sự kiện rơi vào
        cùng một mili giây.
        """
        self._last_id = max(self._last_id + 1, int(time.time() * 1000))
        return self._last_id

    def push_event(self, track_uuid, timestamp, mask_status, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   image, camera_id="", confidence=0.0):
        _mask_status = "unknown"
        _detection_class = "unknown"
        if mask_status == "face":
            _mask_status = "not_wearing_mask"
            _detection_class = "without_mask"
        elif mask_status == "face-mask":
            _mask_status = "wearing_mask"
            _detection_class = "with_mask"

        # Callers pass `image` as the raw full-frame JPEG bytes coming from the
        # C++ engine (full_jpeg). That is already what the device stream wants
        # for the full image, so it is forwarded as-is — decoding then
        # re-encoding it (the old code called imencode on the bytes, which
        # crashed because imencode wants a numpy array) would only burn CPU.
        # A numpy array is still accepted for robustness.
        if isinstance(image, (bytes, bytearray, memoryview)):
            img_bytes = bytes(image)
            if not img_bytes:
                return  # empty jpeg (no frame) — nothing to decode/crop
            decoded = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            ok, buf = cv2.imencode(".jpg", image)
            if not ok:
                return
            img_bytes = buf.tobytes()
            decoded = image
        else:
            return
        if decoded is None:
            return

        # Crop needs a decoded frame; clamp the bbox to the image so an
        # out-of-frame detection can't produce an empty slice.
        h, w = decoded.shape[:2]
        x1 = max(0, min(int(bbox_x1), w - 1))
        y1 = max(0, min(int(bbox_y1), h - 1))
        x2 = max(x1 + 1, min(int(bbox_x2), w))
        y2 = max(y1 + 1, min(int(bbox_y2), h))

        cropped = decoded[y1:y2, x1:x2]
        success_crop, cropped_encoded = cv2.imencode('.jpg', cropped)
        if not success_crop:
            return
        cropped_bytes = cropped_encoded.tobytes()

        # Bắn REALTIME lên WebSocket /ws/mask-events cho panel Xem trực tiếp.
        # Ảnh đi kèm dạng data URL base64 (mask event KHÔNG lưu vào /uploads như
        # face/plate/restricted), nên gói tự chứa. Import trong hàm để tránh
        # phụ thuộc vòng lúc nạp module tiện ích.
        try:
            import base64
            from app.ws.mask_event_ws import mask_event_broadcaster

            def _data_url(b):
                return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")

            # bbox CHUẨN HOÁ [0,1] theo khung FULL để frontend vẽ box đúng chỗ dù
            # ảnh hiển thị ở kích thước nào (x1,y1,x2,y2 đã được kẹp vào ảnh ở trên).
            box = {
                "x1": x1 / w, "y1": y1 / h,
                "x2": x2 / w, "y2": y2 / h,
            }
            mask_event_broadcaster.publish({
                # id của SỰ KIỆN (duy nhất). Id của tracker đi riêng ở
                # `track_id` để không mất thông tin.
                "id": self._next_id(),
                "track_id": track_uuid,
                "camera_id": camera_id,
                "confidence": float(confidence),
                "timestamp": int(timestamp),
                "mask_status": _mask_status,
                # Gửi ẢNH FULL (cả khung) + box: người dùng muốn xem toàn cảnh có
                # khung đánh dấu chỗ phát hiện, không phải chỉ vùng cắt. Kèm crop
                # để làm ảnh nhỏ dự phòng.
                "image_full": _data_url(img_bytes),
                "image_crop": _data_url(cropped_bytes),
                "box": box,
            })
        except Exception as exc:
            import sys
            print(f"mask ws publish error: {exc}", file=sys.stderr)

        event_data = {
            "id": track_uuid,  # Sử dụng track_uuid làm unique id
            "type": "AccessControl",
            "timestamp": timestamp,
            "mask_status": _mask_status,
            "detection_class": _detection_class,
            "direction": "UNKNOWN",
            "velocity_x": 0,
            "velocity_y": 0,
            "bbox_x1": bbox_x1,
            "bbox_y1": bbox_y1,
            "bbox_x2": bbox_x2,
            "bbox_y2": bbox_y2,
            "image_bytes": img_bytes,  # Full image
            "cropped_bytes": cropped_bytes,  # Cropped image (có thể None)
            "analysis": {}  # Kết quả phân tích (có thể None)
        }
        print("event_data", _mask_status)
        self.event_queue.append(event_data)
        # Giới hạn queue size (tránh memory leak)
        if len(self.event_queue) > 100:
            self.event_queue.pop(0)


push_event_metadata = PushEventMetadata()
