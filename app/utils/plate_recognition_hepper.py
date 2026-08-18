#!/usr/bin/env python3
import argparse
import math
import re

from itertools import combinations

RUNNING = True

# OCR label order used by the plate OCR model.
DEFAULT_OCR_LABELS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "U", "V", "X", "Y", "Z",
]

# DEFAULT_OCR_LABELS = ['OCR','1', '2', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', '3', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', '0', '4', '5', '6', '7', '8', '9', 'A']


CONFUSED_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}


def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b


def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)


def _intersection_ratio(box_a, box_b):
    x_left = max(box_a["x1"], box_b["x1"])
    y_top = max(box_a["y1"], box_b["y1"])
    x_right = min(box_a["x2"], box_b["x2"])
    y_bottom = min(box_a["y2"], box_b["y2"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = min(box_a["area"], box_b["area"])
    if min_box_area <= 0:
        return 0.0
    return intersection_area / min_box_area


def _is_same_character(box_a, box_b, overlap_threshold=0.5, x_distance_ratio=0.35):
    overlap_ratio = _intersection_ratio(box_a, box_b)
    avg_width = (box_a["width"] + box_b["width"]) / 2.0
    if avg_width <= 0:
        return False

    x_distance = abs(box_a["x_c"] - box_b["x_c"])
    return overlap_ratio >= overlap_threshold and x_distance <= avg_width * x_distance_ratio


def _deduplicate_overlapping_boxes(center_list):
    deduplicated = []
    for candidate in sorted(center_list, key=lambda item: item["x_c"]):
        matched_index = None
        for index, existing in enumerate(deduplicated):
            if _is_same_character(candidate, existing):
                matched_index = index
                break

        if matched_index is None:
            deduplicated.append(candidate)
        elif candidate["conf"] > deduplicated[matched_index]["conf"]:
            deduplicated[matched_index] = candidate

    return deduplicated


def _median_width(center_list):
    widths = sorted(item["width"] for item in center_list)
    if not widths:
        return 0.0

    middle = len(widths) // 2
    if len(widths) % 2 == 1:
        return widths[middle]
    return (widths[middle - 1] + widths[middle]) / 2.0


def _filter_edge_noise_boxes(center_list, image_shape, min_conf=0.5, narrow_ratio=0.6):
    if not image_shape or len(center_list) <= 1:
        return center_list

    image_height, image_width = image_shape
    median_width = _median_width(center_list)
    if median_width <= 0:
        return center_list

    filtered = []
    for item in center_list:
        touches_border = (
                item["x1"] <= 1
                or item["y1"] <= 1
                or item["x2"] >= image_width - 1
                or item["y2"] >= image_height - 1
        )
        is_narrow = item["width"] < median_width * narrow_ratio
        is_low_confidence = item["conf"] < min_conf

        if touches_border and is_narrow and is_low_confidence:
            continue
        filtered.append(item)

    return filtered if filtered else center_list


def _is_digit(label):
    return str(label).isdigit()


def _is_alpha(label):
    return str(label).isalpha()


def _is_valid_top_line(text):
    if len(text) != 4:
        return False
    if not text[:2].isdigit():
        return False
    return _is_alpha(text[2]) and (_is_alpha(text[3]) or _is_digit(text[3]))


def _is_valid_bottom_line(text):
    return len(text) == 5 and text.isdigit()


def _normalize_numeric_like_char(label):
    char = str(label)
    if char.isdigit():
        return char
    return CONFUSED_TO_DIGIT.get(char.upper(), char)


def _normalize_numeric_text(text):
    return "".join(_normalize_numeric_like_char(char) for char in str(text))


def _normalize_numeric_suffix(text, min_suffix_length=4):
    chars = list(str(text))
    suffix_start = len(chars)
    while suffix_start > 0:
        normalized = _normalize_numeric_like_char(chars[suffix_start - 1])
        if normalized.isdigit():
            suffix_start -= 1
            continue
        break

    suffix_length = len(chars) - suffix_start
    if suffix_length < min_suffix_length or suffix_start == 0:
        return str(text)

    normalized_suffix = _normalize_numeric_text("".join(chars[suffix_start:]))
    if not normalized_suffix.isdigit():
        return str(text)
    return "".join(chars[:suffix_start]) + normalized_suffix


def _box_noise_score(box, line_boxes, image_shape, median_width, median_height):
    score = max(0.0, 1.0 - box["conf"])

    if image_shape:
        image_height, image_width = image_shape
        if (
                box["x1"] <= 1
                or box["y1"] <= 1
                or box["x2"] >= image_width - 1
                or box["y2"] >= image_height - 1
        ):
            score += 0.35

    if median_width > 0 and box["width"] < median_width * 0.75:
        score += 0.3
    if median_height > 0 and box["height"] < median_height * 0.8:
        score += 0.2

    for other in line_boxes:
        if other is box:
            continue
        if _intersection_ratio(box, other) > 0.2:
            score += 0.25
            break

    return score


def _line_stats(line_boxes):
    if not line_boxes:
        return 0.0, 0.0
    widths = sorted(box["width"] for box in line_boxes)
    heights = sorted(box["height"] for box in line_boxes)
    middle = len(widths) // 2
    if len(widths) % 2 == 1:
        return widths[middle], heights[middle]
    return (
        (widths[middle - 1] + widths[middle]) / 2.0,
        (heights[middle - 1] + heights[middle]) / 2.0,
    )


def _sorted_line_text(line_boxes):
    return "".join(str(box["label"]) for box in sorted(line_boxes, key=lambda item: item["x_c"]))


def _normalized_line_text(line_boxes):
    return _normalize_numeric_text(_sorted_line_text(line_boxes))


def _choose_best_line_candidate(line_boxes, expected_length, validator, image_shape):
    sorted_boxes = sorted(line_boxes, key=lambda item: item["x_c"])
    if len(sorted_boxes) <= expected_length:
        return sorted_boxes

    median_width, median_height = _line_stats(sorted_boxes)
    removable_count = len(sorted_boxes) - expected_length
    best_candidate = sorted_boxes
    best_score = None

    for indexes_to_remove in combinations(range(len(sorted_boxes)), removable_count):
        candidate = [box for index, box in enumerate(sorted_boxes) if index not in indexes_to_remove]
        candidate_text = _normalized_line_text(candidate)
        if not validator(candidate_text):
            continue

        removed_boxes = [sorted_boxes[index] for index in indexes_to_remove]
        kept_confidence = sum(box["conf"] for box in candidate)
        removed_noise = sum(
            _box_noise_score(box, sorted_boxes, image_shape, median_width, median_height)
            for box in removed_boxes
        )
        kept_noise = sum(
            _box_noise_score(box, sorted_boxes, image_shape, median_width, median_height)
            for box in candidate
        )
        score = (removed_noise * 3.0) + kept_confidence - kept_noise

        if best_score is None or score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate


def _split_lines(center_list):
    """Split characters into the top (line_1) and bottom (line_2) rows.

    Rows are separated by each centre's position relative to a least-squares
    baseline fitted through all character centres (``row_pos = y_c - slope*x_c``)
    instead of a flat horizontal cut at ``y_mean``. On a tilted two-line plate a
    flat cut misassigns characters near the boundary — the higher end of the top
    row can dip below the lower end of the bottom row — so de-tilting first keeps
    each row intact. For a level plate ``slope`` is ~0 and this reduces to the
    old horizontal split.
    """
    n = len(center_list)
    if n == 0:
        return [], []

    xs = [c["x_c"] for c in center_list]
    ys = [c["y_c"] for c in center_list]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if denom > 1e-6:
        slope = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / denom

    # Vertical position after removing the plate's tilt; split at the midpoint.
    row_positions = [ys[i] - slope * xs[i] for i in range(n)]
    split = (max(row_positions) + min(row_positions)) / 2.0

    line_1 = []
    line_2 = []
    for item, pos in zip(center_list, row_positions):
        if pos > split:
            line_2.append(item)
        else:
            line_1.append(item)
    return line_1, line_2


def _build_plate_text(center_list, lp_type, image_shape):
    if lp_type != "2":
        return _normalize_numeric_suffix(_sorted_line_text(center_list))

    line_1, line_2 = _split_lines(center_list)
    line_1 = _choose_best_line_candidate(line_1, 4, _is_valid_top_line, image_shape)
    line_2 = _choose_best_line_candidate(line_2, 5, _is_valid_bottom_line, image_shape)
    return f"{_sorted_line_text(line_1)}-{_normalized_line_text(line_2)}"


def class_id_to_label(class_id):
    if 0 <= class_id < len(DEFAULT_OCR_LABELS):
        return DEFAULT_OCR_LABELS[class_id]
    return str(class_id)


def build_secondary_boxes(det, min_secondary_conf):
    center_list = []
    image_shape = None
    if det.secondary_model_height > 0 and det.secondary_model_width > 0:
        image_shape = (det.secondary_model_height, det.secondary_model_width)

    for idx in range(min(det.secondary_det_count, 16)):
        sdet = det.secondary_dets[idx]
        if sdet.score < min_secondary_conf:
            continue
        x1 = float(sdet.x1)
        y1 = float(sdet.y1)
        x2 = float(sdet.x2)
        y2 = float(sdet.y2)
        center_list.append(
            {
                "x_c": (x1 + x2) / 2.0,
                "y_c": (y1 + y2) / 2.0,
                "label": class_id_to_label(int(sdet.class_id)),
                "conf": float(sdet.score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
            }
        )
    return center_list, image_shape


def _classify_lp_type(center_list):
    """Decide 1-line ("1") vs 2-line ("2"), robust to tilt, to per-character
    wobble, and to stray high/low noise boxes.

    De-tilt the centres (row_pos = y_c - slope*x_c from a least-squares fit),
    then look for a single clear horizontal GAP that splits them into two rows.
    A real two-line plate has a gap of roughly one character height with several
    characters on BOTH sides. A single-line plate has no such balanced gap —
    even a stray box sitting high or low only puts ONE character across the gap,
    so requiring >=2 characters on each side stops one outlier from turning a
    single line into a (wrong) two-line read. (The earlier spread/median test
    misfired here: one outlier inflated the spread and flipped it to 2-line.)
    """
    n = len(center_list)
    if n < 4:
        return "1"  # VN two-line plates carry >=3 (top) + 5 (bottom) characters

    xs = [b["x_c"] for b in center_list]
    ys = [b["y_c"] for b in center_list]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if denom > 1e-6:
        slope = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / denom

    heights = sorted(b["height"] for b in center_list)
    mid = len(heights) // 2
    median_h = heights[mid] if len(heights) % 2 else (heights[mid - 1] + heights[mid]) / 2.0
    if median_h <= 0:
        return "1"

    row_pos = sorted(ys[i] - slope * xs[i] for i in range(n))
    best_gap = 0.0
    best_index = 0
    for i in range(n - 1):
        gap = row_pos[i + 1] - row_pos[i]
        if gap > best_gap:
            best_gap = gap
            best_index = i

    top_count = best_index + 1
    bottom_count = n - top_count
    if best_gap > 0.6 * median_h and top_count >= 2 and bottom_count >= 2:
        return "2"
    return "1"


def _detect_plate_from_center_list(center_list, image_shape):
    if not center_list:
        return ""

    center_list = _filter_edge_noise_boxes(center_list, image_shape)
    center_list = _deduplicate_overlapping_boxes(center_list)
    if not center_list:
        return ""

    lp_type = _classify_lp_type(center_list)

    return _build_plate_text(center_list, lp_type, image_shape)


def detect_plate_from_secondary(det, min_secondary_conf):
    center_list, image_shape = build_secondary_boxes(det, min_secondary_conf)
    return _detect_plate_from_center_list(center_list, image_shape)


def flatten_char_boxes(detection, limit=32):
    """Mọi ô KÝ TỰ nằm dưới một detection, sâu mấy tầng cũng lấy được.

    Hai cách đọc biển cho ra hai hình dạng cây khác nhau:
      * OCR một tầng   — con của biển đã LÀ từng ký tự.
      * PP-OCR hai tầng — con là DÒNG chữ, ký tự nằm ở cháu.
    Chỗ gọi không nên phải biết mình đang chạy cách nào, nên gom về một danh
    sách phẳng ở đây.

    TOẠ ĐỘ giữ nguyên khi mọi ký tự CÙNG một không gian (con trực tiếp của
    biển đã nắn phẳng) — đó là dạng mà thuật toán ghép biển bên dưới vốn được
    chỉnh cho, đừng đụng vào. Chỉ khi ký tự nằm rải ở nhiều DÒNG khác nhau mới
    phải quy về khung gốc (fx1..fy2 engine gắn cho mọi tầng con): x1..y2 của
    một ký tự là toạ độ trong ảnh cắt của DÒNG chứa nó, nên ký tự ở hai dòng
    khác nhau không so sánh được với nhau."""
    kids = (detection or {}).get("children") or []
    if not kids:
        return []

    # Con đã là ký tự (OCR một tầng): dùng thẳng, không đổi gì.
    if not any(kid.get("children") for kid in kids):
        return kids[:limit]

    out = []
    for line in kids:
        for char in line.get("children") or []:
            box = dict(char)
            if char.get("fx1") is not None:
                box["x1"] = char["fx1"]
                box["y1"] = char["fy1"]
                box["x2"] = char["fx2"]
                box["y2"] = char["fy2"]
            out.append(box)
            if len(out) >= limit:
                return out
    return out


# Biển số Việt Nam có cấu trúc CỐ ĐỊNH: hai chữ số mã tỉnh, rồi sê-ri một hoặc
# hai ký tự (chữ, hoặc chữ + số như "B2", "K1", "X3"), rồi 4-5 chữ số.
#   50H-380.66 -> 50 H  38066     93A-350.45 -> 93 A  35045
#   61B2-474.59-> 61 B2 47459     47K1-937.40-> 47 K1 93740
#
# Nhờ vậy chặn được đúng thứ đang làm phiền: ảnh mờ hoặc biển bị cắt mất một
# góc vẫn ra chuỗi, nhưng chuỗi đó không bao giờ có dạng này. Model OCR khi
# gặp nhiễu hay nhả ra cùng một mẫu ('ZG1AH', 'ZG12AH', 'ZG92AH') — không mẫu
# nào lọt qua được.
_VN_PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,2}\d?\d{4,5}$")


def looks_like_vn_plate(text_plate: str) -> bool:
    """Chuỗi đọc được có ĐÚNG DẠNG một biển số Việt Nam không.

    Chỉ xét cấu trúc, không xét nội dung — nó bắt được ảnh mờ / biển cụt / hộp
    bắt nhầm, chứ không bắt được lỗi đọc nhầm một chữ số thành chữ số khác.
    Sai dạng thì coi như CHƯA đọc được và thử lại ở khung sau, chứ đừng ghi
    một dòng rác vào lịch sử."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text_plate or "").upper()
    return bool(_VN_PLATE_RE.match(cleaned))


def plate_text_from_detection(detection, min_secondary_conf=0.3, image_shape=None):
    """Chuỗi biển số đọc từ cây con của MỘT detection biển.

    Hai cách đọc cho ra hai dạng cây, và phải xử lý khác nhau:

    * PP-OCR (con là DÒNG chữ, cháu là ký tự) — dòng đã được model đọc thành
      chuỗi có THỨ TỰ rồi, cứ ghép các dòng từ trên xuống. TUYỆT ĐỐI đừng tách
      ra rồi xếp lại theo toạ độ: toạ độ khung gốc của từng ký tự là kết quả
      quy đổi qua hai lần cắt nên chỉ gần đúng, đủ để hai dòng cài răng lược
      vào nhau. Đo trên biển thật: model đọc `59X2 | 72685` (đúng) mà xếp lại
      theo hình học ra `5972X2685`.

    * OCR một tầng (con đã LÀ từng ký tự, không có chữ sẵn) — mỗi ký tự là một
      hộp rời rạc, phải tự tách dòng và sắp thứ tự: đó là việc của
      detect_plate_from_children.
    """
    children = (detection or {}).get("children") or []
    lines = [c for c in children if c.get("children")]
    if not lines:
        return detect_plate_from_children(children, min_secondary_conf, image_shape)

    parts = []
    for line in sorted(lines, key=lambda l: l.get("fy1", l.get("y1", 0))):
        chars = sorted(line["children"], key=lambda c: c.get("fx1", c.get("x1", 0)))
        text = "".join(
            (c.get("text") or "")
            for c in chars
            if float(c.get("score", 0.0)) >= min_secondary_conf
        )
        if text:
            parts.append(text)
    # Biển hai dòng viết liền nhau bằng dấu gạch, cùng dạng mà nhánh OCR một
    # tầng vẫn sinh ra ('53-79622') để phía sau không phải phân biệt.
    return "-".join(parts)


def build_secondary_boxes_from_children(children, min_secondary_conf):
    center_list = []
    for child in (children or [])[:32]:
        score = float(child.get("score", 0.0))
        if score < min_secondary_conf:
            continue
        x1 = float(child["x1"])
        y1 = float(child["y1"])
        x2 = float(child["x2"])
        y2 = float(child["y2"])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        center_list.append(
            {
                "x_c": (x1 + x2) / 2.0,
                "y_c": (y1 + y2) / 2.0,
                # Model mang sẵn bảng nhãn (PP-OCR rec) thì ký tự nằm ở `text`;
                # model chỉ trả classId (ocr.rknn) thì tra bảng như cũ.
                "label": (child.get("text") or "").strip()
                or class_id_to_label(int(child.get("classId", -1))),
                "conf": score,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "area": width * height,
            }
        )
    return center_list


def detect_plate_from_children(children, min_secondary_conf=0.3, image_shape=None):
    center_list = build_secondary_boxes_from_children(children, min_secondary_conf)
    return _detect_plate_from_center_list(center_list, image_shape)
