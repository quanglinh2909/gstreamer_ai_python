from typing import Generic, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PageResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    pages: int

    @classmethod
    def build(cls, items: List[T], total: int, page: int, size: int) -> "PageResponse[T]":
        pages = (total + size - 1) // size if size else 0
        return cls(items=items, total=total, page=page, size=size, pages=pages)
