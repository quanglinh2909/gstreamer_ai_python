from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException, status

from app.core.config import settings


class HTTPXClient:
    DEFAULT_TIMEOUT = 30

    @staticmethod
    async def _request(
        method: str,
        url: str,
        raw: bool = False,
        **kwargs,
    ) -> Any:
        try:
            async with httpx.AsyncClient(
                base_url=settings.AI_API_BASE_URL,
                timeout=HTTPXClient.DEFAULT_TIMEOUT,
            ) as client:
                response = await client.request(method, url, **kwargs)

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                if raw:
                    return response.content, content_type or "application/octet-stream"

                if "application/json" in content_type:
                    return response.json()

                return response.text

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail={
                    "message": "External API error",
                    "response": e.response.text,
                },
            )

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "Cannot connect to external service",
                    "error": str(e),
                },
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "message": "Unexpected error",
                    "error": str(e),
                },
            )

    @staticmethod
    async def get(
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        raw: bool = False,
    ):
        return await HTTPXClient._request(
            "GET",
            url,
            raw=raw,
            params=params,
            headers=headers,
        )

    @staticmethod
    async def post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        return await HTTPXClient._request(
            "POST",
            url,
            data=data,
            json=json,
            headers=headers,
        )

    @staticmethod
    async def put(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        return await HTTPXClient._request(
            "PUT",
            url,
            data=data,
            json=json,
            headers=headers,
        )

    @staticmethod
    async def patch(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        return await HTTPXClient._request(
            "PATCH",
            url,
            data=data,
            json=json,
            headers=headers,
        )

    @staticmethod
    async def delete(
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        return await HTTPXClient._request(
            "DELETE",
            url,
            headers=headers,
        )
