import json
from typing import List

import httpx
import pytest

from llama_index.memory.pastdotdev.client import (
    AsyncPastDevClient,
    PastDevClient,
    build_recall_body,
)


def recorder(requests: List[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(202, json={"operationId": "op-1", "ids": ["chat/1"]})
        if request.url.path == "/api/v1/identities":
            return httpx.Response(200, json={"identities": [], "hasMore": False, "endCursor": None})
        return httpx.Response(200, json={"audiences": []})

    return httpx.MockTransport(handler)


def test_sync_client_paths() -> None:
    requests: List[httpx.Request] = []
    client = PastDevClient(
        api_key="past_sk_test", http_client=httpx.Client(transport=recorder(requests))
    )
    assert client.delete_data_point("chat/1")["ids"] == ["chat/1"]
    assert client.list_audiences() == {"audiences": []}
    assert client.list_identities(after="cursor-1")["hasMore"] is False
    client.close()
    assert [(r.method, r.url.raw_path.decode()) for r in requests] == [
        ("DELETE", "/api/v1/data-points/chat%2F1"),
        ("GET", "/api/v1/audiences"),
        ("GET", "/api/v1/identities?after=cursor-1"),
    ]


async def test_async_client_paths() -> None:
    requests: List[httpx.Request] = []
    client = AsyncPastDevClient(
        api_key="past_sk_test", http_client=httpx.AsyncClient(transport=recorder(requests))
    )
    await client.delete_data_point("chat/1")
    await client.list_audiences()
    await client.list_identities()
    await client.aclose()
    assert [(r.method, r.url.raw_path.decode()) for r in requests] == [
        ("DELETE", "/api/v1/data-points/chat%2F1"),
        ("GET", "/api/v1/audiences"),
        ("GET", "/api/v1/identities"),
    ]


def test_recall_body_uses_api_keys() -> None:
    body = build_recall_body(
        query="q",
        identity="u",
        max_tokens=500,
        query_timestamp="2026-10-01T00:00:00Z",
        occurred_to="2026-09-01T00:00:00Z",
        sort="chronological",
    )
    assert json.loads(json.dumps(body)) == {
        "query": "q",
        "identity": "u",
        "maxTokens": 500,
        "queryTimestamp": "2026-10-01T00:00:00Z",
        "occurredTo": "2026-09-01T00:00:00Z",
        "sort": "chronological",
    }
    with pytest.raises(ValueError):
        build_recall_body(query="q", identity="u", level="huge")
    with pytest.raises(ValueError):
        build_recall_body(query="q", identity="u", sort="random")
