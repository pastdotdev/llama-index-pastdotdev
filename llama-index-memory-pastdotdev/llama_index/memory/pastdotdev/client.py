"""A small HTTP client for the past dev Memory API.

This file is the same in llama-index-memory-pastdotdev and
llama-index-tools-pastdotdev, so that each package installs without the other.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_API_HOST = "https://api.past.dev"
API_KEY_ENV = "PAST_API_KEY"
API_HOST_ENV = "PAST_API_HOST"
RECALL_LEVELS = ("low", "medium", "high", "extra-high")
RECALL_SORTS = ("relevance", "chronological")

# A 429 is retried once, after the Retry-After delay, when that delay is short.
MAX_RETRY_AFTER_SECONDS = 30.0


class PastDevError(Exception):
    """An error response from the past dev Memory API."""

    def __init__(
        self,
        code: str,
        status: int,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        data: Any = None,
    ) -> None:
        self.code = code
        self.status = status
        self.message = message
        self.request_id = request_id
        self.data = data
        text = f"past dev error {code} (HTTP {status})"
        if message:
            text += f": {message}"
        if request_id:
            text += f" [request id {request_id}]"
        super().__init__(text)


def resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(
            "past dev needs a project API key: pass api_key or set the "
            f"{API_KEY_ENV} environment variable."
        )
    return key


def resolve_api_host(api_host: Optional[str]) -> str:
    host = api_host or os.environ.get(API_HOST_ENV) or DEFAULT_API_HOST
    return host.rstrip("/")


def to_iso(value: Any) -> Optional[str]:
    """Return an ISO 8601 string for a datetime or a string, else None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_recall_body(
    query: str,
    identity: str,
    limit: Optional[int] = None,
    level: Optional[str] = None,
    max_tokens: Optional[int] = None,
    query_timestamp: Optional[str] = None,
    occurred_from: Optional[str] = None,
    occurred_to: Optional[str] = None,
    sort: Optional[str] = None,
) -> Dict[str, Any]:
    if level is not None and max_tokens is not None:
        raise ValueError("Send level or max_tokens to recall, never both.")
    if level is not None and level not in RECALL_LEVELS:
        raise ValueError(f"level must be one of {', '.join(RECALL_LEVELS)}.")
    if sort is not None and sort not in RECALL_SORTS:
        raise ValueError(f"sort must be one of {', '.join(RECALL_SORTS)}.")
    body: Dict[str, Any] = {"query": query, "identity": identity}
    optional = {
        "limit": limit,
        "level": level,
        "maxTokens": max_tokens,
        "queryTimestamp": query_timestamp,
        "occurredFrom": occurred_from,
        "occurredTo": occurred_to,
        "sort": sort,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return body


def build_ingest_body(
    content: str,
    identity: str,
    timestamp: Optional[str] = None,
    source_id: Optional[str] = None,
    audience: Optional[str] = None,
    label: Optional[str] = None,
    metadata: Any = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    if not content or not content.strip():
        raise ValueError("past dev refuses empty content: send some text to ingest.")
    body: Dict[str, Any] = {"content": content, "identity": identity}
    optional = {
        "timestamp": timestamp,
        "id": source_id,
        "audience": audience,
        "label": label,
        "metadata": metadata,
        "idempotencyKey": idempotency_key,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return body


def format_recall(response: Dict[str, Any]) -> str:
    """Render a recall response as plain text for a model to read."""
    results = response.get("results") or []
    return "\n\n".join(format_recall_results(results))


def format_recall_results(results: List[Dict[str, Any]]) -> List[str]:
    rendered = []
    for result in results:
        content = (result.get("content") or "").strip()
        if not content:
            continue
        artifact = result.get("artifact") or {}
        kind = artifact.get("kind") or "memory"
        head = (
            f"[{result.get('rank', len(rendered) + 1)}] {_date(result.get('occurredAt'))} · {kind}"
        )
        superseded = artifact.get("supersededAt")
        if superseded:
            head += f" · superseded {_date(superseded)}"
        lines = [head, content]
        for source in result.get("sources") or []:
            where = source.get("label") or source.get("sourceId") or "source"
            when = _date(source.get("occurredAt"))
            for excerpt in source.get("excerpts") or []:
                lines.append(f'  evidence ({where}, {when}): "{excerpt}"')
        rendered.append("\n".join(lines))
    return rendered


def _date(value: Optional[str]) -> str:
    return value[:10] if isinstance(value, str) else "undated"


class _Base:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_host: Optional[str] = None,
        timeout: float = 30.0,
        user_agent: str = "llama-index-pastdotdev",
    ) -> None:
        self.api_host = resolve_api_host(api_host)
        self._headers = {
            "Authorization": f"Bearer {resolve_api_key(api_key)}",
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.api_host}{path}"

    @staticmethod
    def _retry_delay(response: httpx.Response) -> Optional[float]:
        if response.status_code != 429:
            return None
        header = response.headers.get("Retry-After", "1")
        try:
            delay = float(header)
        except ValueError:
            return None
        return delay if 0 <= delay <= MAX_RETRY_AFTER_SECONDS else None

    @staticmethod
    def _result(response: httpx.Response) -> Any:
        if response.is_success:
            return response.json() if response.content else None
        request_id = response.headers.get("X-Request-Id")
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("code"):
            raise PastDevError(
                code=str(body["code"]),
                status=int(body.get("status") or response.status_code),
                message=body.get("debugMessage"),
                request_id=request_id,
                data=body.get("data"),
            )
        raise PastDevError(
            code="http-error",
            status=response.status_code,
            message=response.text[:200] or None,
            request_id=request_id,
        )


class PastDevClient(_Base):
    """Synchronous client."""

    def __init__(
        self, *args: Any, http_client: Optional[httpx.Client] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._http = http_client or httpx.Client(timeout=self.timeout)

    def request(self, method: str, path: str, json: Any = None, params: Any = None) -> Any:
        for attempt in range(2):
            response = self._http.request(
                method, self._url(path), json=json, params=params, headers=self._headers
            )
            delay = self._retry_delay(response)
            if delay is None or attempt == 1:
                return self._result(response)
            time.sleep(delay)
        return None  # pragma: no cover

    def ingest(self, **kwargs: Any) -> Dict[str, Any]:
        return self.request("POST", "/api/v1/ingest", json=build_ingest_body(**kwargs))

    def recall(self, **kwargs: Any) -> Dict[str, Any]:
        return self.request("POST", "/api/v1/recall", json=build_recall_body(**kwargs))

    def delete_data_point(self, source_id: str) -> Dict[str, Any]:
        return self.request("DELETE", f"/api/v1/data-points/{_path(source_id)}")

    def list_audiences(self) -> Dict[str, Any]:
        """List the project's audiences. It costs no credits, so it also tests a key."""
        return self.request("GET", "/api/v1/audiences")

    def list_identities(self, after: Optional[str] = None) -> Dict[str, Any]:
        return self.request("GET", "/api/v1/identities", params={"after": after} if after else None)

    def close(self) -> None:
        self._http.close()


class AsyncPastDevClient(_Base):
    """Asynchronous client."""

    def __init__(
        self, *args: Any, http_client: Optional[httpx.AsyncClient] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout)

    async def request(self, method: str, path: str, json: Any = None, params: Any = None) -> Any:
        for attempt in range(2):
            response = await self._http.request(
                method, self._url(path), json=json, params=params, headers=self._headers
            )
            delay = self._retry_delay(response)
            if delay is None or attempt == 1:
                return self._result(response)
            await asyncio.sleep(delay)
        return None  # pragma: no cover

    async def ingest(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.request("POST", "/api/v1/ingest", json=build_ingest_body(**kwargs))

    async def recall(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.request("POST", "/api/v1/recall", json=build_recall_body(**kwargs))

    async def delete_data_point(self, source_id: str) -> Dict[str, Any]:
        return await self.request("DELETE", f"/api/v1/data-points/{_path(source_id)}")

    async def list_audiences(self) -> Dict[str, Any]:
        """List the project's audiences. It costs no credits, so it also tests a key."""
        return await self.request("GET", "/api/v1/audiences")

    async def list_identities(self, after: Optional[str] = None) -> Dict[str, Any]:
        return await self.request(
            "GET", "/api/v1/identities", params={"after": after} if after else None
        )

    async def aclose(self) -> None:
        await self._http.aclose()


def _path(segment: str) -> str:
    from urllib.parse import quote

    return quote(segment, safe="")
