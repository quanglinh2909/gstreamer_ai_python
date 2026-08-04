import time

import cv2
import numpy as np


class PushEventMetadata:
    """Hàng đợi sự kiện khẩu trang cho luồng MJPEG của thiết bị ngoài.

    Chỉ còn phục vụ device_router — nó cần ảnh dạng BYTES thô để nhồi thẳng
    vào multipart. Sự kiện cho giao diện đã đi đường khác (bảng event_mask +
    WebSocket kèm URL), xem face_mask_service.

    `id` ở đây vẫn là track_uuid: người tiêu thụ là một thiết bị phần cứng đọc
    tuần tự từng gói, không khử trùng theo id như giao diện web.
    """

    def __init__(self):
        self.event_queue = []

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

        # KHÔNG bắn WebSocket ở đây nữa.
        #
        # Trước đây khẩu trang là loại duy nhất không có bảng DB, nên gói
        # realtime phải tự chứa ảnh dạng base64 và chỗ này là nơi duy nhất bắn
        # nó đi. Giờ sự kiện đã lưu vào event_mask qua
        # AIServiceBase.save_event, và chính hàm đó bắn WebSocket kèm ĐƯỜNG DẪN
        # /uploads. Giữ cả hai đường là mỗi sự kiện lên socket hai lần, một lần
        # base64 nặng gấp bội mà nội dung y hệt.
        #
        # Hàm này giờ chỉ còn một việc: nuôi hàng đợi của luồng MJPEG cho thiết
        # bị ngoài (device_router) — nơi cần BYTES thô, không dùng được URL.
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
