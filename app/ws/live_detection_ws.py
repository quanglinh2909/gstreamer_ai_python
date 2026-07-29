"""Đẩy KHUNG PHÁT HIỆN theo thời gian thực để vẽ đè lên video trực tiếp.

Khác 4 stream sự kiện (face/plate/restricted/mask) ở chỗ: các stream đó chỉ
bắn khi có SỰ KIỆN (nhận diện được người, đọc được biển số...), còn stream này
bắn MỖI KHUNG HÌNH mà AI xử lý — kể cả khung không có gì — để lớp phủ trên
video bám theo vật thể và biết lúc nào phải XOÁ khung cũ đi.

Gói tin KHÔNG kèm ảnh, chỉ toạ độ chuẩn hoá [0,1] nên rất nhẹ (~50 byte/khung
phát hiện). Vẫn dùng chung CameraFilteredBroadcaster nên client mở
`?camera_id=` chỉ nhận đúng camera đang xem — điều này quan trọng hơn hẳn so
với stream sự kiện: ở đây lưu lượng là liên tục, không lọc thì mỗi người xem
một camera vẫn phải nhận khung phát hiện của TẤT CẢ camera.
"""

from __future__ import annotations

from app.ws.base_event_ws import CameraFilteredBroadcaster

live_detection_broadcaster = CameraFilteredBroadcaster("live-detection")
