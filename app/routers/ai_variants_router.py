# -*- coding: utf-8 -*-
"""Các CÁCH LÀM (biến thể) của từng loại AI.

Giao diện gọi endpoint này trước khi vẽ form cấu hình: trả về một phần tử thì
giấu ô chọn đi (không có gì để chọn), từ hai trở lên mới hiện. Nhờ vậy thêm
cách làm mới cho một loại AI là việc của backend — thêm một AIVariant vào
`VARIANTS` của service — giao diện tự mọc ra ô chọn mà không phải sửa gì.
"""

from fastapi import APIRouter, HTTPException

from app.enum.config_ai_enum import TypeConfigAiEnum
from app.utils.process_ai_hepper import ProcessAiHepper

router = APIRouter()
prefix = "/ai-variants"
tags = ["AI Variants"]


@router.get("")
async def list_all_variants():
    """Biến thể của MỌI loại AI, khoá theo loại — giao diện nạp một lần."""
    return {
        t.value: _variants_of(t.value) for t in TypeConfigAiEnum
    }


@router.get("/{ai_type}")
async def list_variants(ai_type: str):
    service = ProcessAiHepper.get_service_ai(ai_type)
    if service is None:
        raise HTTPException(status_code=404, detail=f"Loại AI không tồn tại: {ai_type}")
    return _variants_of(ai_type)


def _variants_of(ai_type: str) -> list:
    service = ProcessAiHepper.get_service_ai(ai_type)
    if service is None:
        return []
    options = getattr(service, "variant_options", None)
    return options() if callable(options) else []
