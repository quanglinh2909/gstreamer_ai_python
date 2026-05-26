#!/usr/bin/env python3
import argparse
import math

from itertools import combinations

RUNNING = True

# OCR label order used by the plate OCR model.
DEFAULT_OCR_LABELS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "U", "V", "X", "Y", "Z",
]

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
    y_mean = int(sum(c["y_c"] for c in center_list) / len(center_list))
    line_1 = []
    line_2 = []
    for item in center_list:
        if int(item["y_c"]) > y_mean:
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


def _detect_plate_from_center_list(center_list, image_shape):
    if not center_list:
        return ""

    center_list = _filter_edge_noise_boxes(center_list, image_shape)
    center_list = _deduplicate_overlapping_boxes(center_list)
    if not center_list:
        return ""

    lp_type = "1"
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        if cp["x_c"] < l_point["x_c"]:
            l_point = cp
        if cp["x_c"] > r_point["x_c"]:
            r_point = cp

    if l_point["x_c"] != r_point["x_c"]:
        for ct in center_list:
            if not check_point_linear(
                    ct["x_c"], ct["y_c"],
                    l_point["x_c"], l_point["y_c"],
                    r_point["x_c"], r_point["y_c"],
            ):
                lp_type = "2"
                break

    return _build_plate_text(center_list, lp_type, image_shape)


def detect_plate_from_secondary(det, min_secondary_conf):
    center_list, image_shape = build_secondary_boxes(det, min_secondary_conf)
    return _detect_plate_from_center_list(center_list, image_shape)


def build_secondary_boxes_from_children(children, min_secondary_conf):
    center_list = []
    for child in (children or [])[:16]:
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
                "label": class_id_to_label(int(child.get("classId", -1))),
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
