import json
from typing import Any, Callable, Dict, List

import httpx
import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.pastdotdev import PastDevError, PastDevToolSpec

KEY = "past_sk_test"

RECALL_RESPONSE = {
    "asOf": "2026-10-16T12:00:00Z",
    "usedEvidenceTokens": 42,
    "results": [
        {
            "id": "7f1c3a4e-0000-4000-8000-000000000001",
            "rank": 1,
            "occurredAt": "2026-10-01T09:00:00Z",
            "content": "Ada prefers email to phone calls.",
            "artifact": {
                "id": "7f1c3a4e-0000-4000-8000-000000000001",
                "kind": "observation",
                "occurredAt": "2026-10-01T09:00:00Z",
            },
            "sources": [
                {
                    "sourceId": "chat-1",
                    "occurredAt": "2026-10-01T09:00:00Z",
                    "excerpts": ["please email me, never call"],
                }
            ],
        }
    ],
}

INGEST_RESPONSE = {
    "projectId": 1,
    "ingestionId": "7f1c3a4e-0000-4000-8000-0000000000aa",
    "sourceId": "abc",
    "status": "queued",
}


class FakeApi:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def bodies(self, path: str) -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


def default_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/v1/recall":
        return httpx.Response(200, json=RECALL_RESPONSE)
    if request.url.path == "/api/v1/ingest":
        return httpx.Response(202, json=INGEST_RESPONSE)
    return httpx.Response(404, json={"code": "not-found", "status": 404})


def make_spec(api: FakeApi, **kwargs: Any) -> PastDevToolSpec:
    kwargs.setdefault("api_key", KEY)
    kwargs.setdefault("identity", "user-ada")
    return PastDevToolSpec(http_client=httpx.Client(transport=httpx.MockTransport(api)), **kwargs)


def test_is_a_tool_spec() -> None:
    names = [b.__name__ for b in PastDevToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names
    assert PastDevToolSpec.spec_functions == ["recall", "ingest"]


def test_requires_identity() -> None:
    with pytest.raises(ValueError):
        make_spec(FakeApi(default_handler), identity="")


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PAST_API_KEY"):
        PastDevToolSpec(identity="user-ada")


def test_refuses_level_and_max_tokens_together() -> None:
    with pytest.raises(ValueError):
        make_spec(FakeApi(default_handler), level="low", max_tokens=100)


def test_model_cannot_choose_identity_or_audience() -> None:
    tools = make_spec(FakeApi(default_handler)).to_tool_list()
    assert [tool.metadata.name for tool in tools] == ["recall", "ingest"]
    for tool in tools:
        properties = tool.metadata.get_parameters_dict()["properties"]
        assert "identity" not in properties
        assert "audience" not in properties
    recall, ingest = tools
    assert set(recall.metadata.get_parameters_dict()["properties"]) == {
        "query",
        "occurredFrom",
        "occurredTo",
        "sort",
    }
    assert recall.metadata.get_parameters_dict()["required"] == ["query"]
    assert set(ingest.metadata.get_parameters_dict()["properties"]) == {"content", "timestamp"}


def test_recall_tool_sends_fixed_identity_and_renders_results() -> None:
    api = FakeApi(default_handler)
    spec = make_spec(api, identity="user-ada", level="high", limit=10)
    recall = spec.to_tool_list()[0]
    output = recall(query="How should I contact Ada?", occurredFrom="2026-01-01T00:00:00Z")
    assert api.bodies("/api/v1/recall") == [
        {
            "query": "How should I contact Ada?",
            "identity": "user-ada",
            "limit": 10,
            "level": "high",
            "occurredFrom": "2026-01-01T00:00:00Z",
        }
    ]
    assert api.requests[0].headers["Authorization"] == f"Bearer {KEY}"
    assert "Ada prefers email to phone calls." in output.content
    assert '"please email me, never call"' in output.content


def test_recall_with_no_results_says_so() -> None:
    api = FakeApi(
        lambda request: httpx.Response(
            200, json={"asOf": "2026-10-16T12:00:00Z", "usedEvidenceTokens": 0, "results": []}
        )
    )
    assert "no memory" in make_spec(api).recall(query="Anything?")


def test_ingest_tool_sends_content_with_fixed_identity_and_audience() -> None:
    api = FakeApi(default_handler)
    spec = make_spec(api, audience="support-team", api_host="https://memory.example.com/")
    result = spec.ingest(content="Ada prefers email.", timestamp="2026-10-01T09:00:00Z")
    assert api.requests[0].url.host == "memory.example.com"
    assert api.bodies("/api/v1/ingest") == [
        {
            "content": "Ada prefers email.",
            "identity": "user-ada",
            "timestamp": "2026-10-01T09:00:00Z",
            "audience": "support-team",
            "label": "Saved by a LlamaIndex agent",
            "metadata": {"source": "llama-index"},
        }
    ]
    assert "source id abc" in result


def test_ingest_unchanged_says_so() -> None:
    api = FakeApi(lambda request: httpx.Response(200, json={**INGEST_RESPONSE, "unchanged": True}))
    assert "already had" in make_spec(api).ingest(content="Ada prefers email.")


def test_ingest_refuses_empty_content() -> None:
    api = FakeApi(default_handler)
    with pytest.raises(ValueError):
        make_spec(api).ingest(content="   ")
    assert api.requests == []


def test_bad_request_is_raised_and_not_retried() -> None:
    api = FakeApi(
        lambda request: httpx.Response(
            400,
            json={
                "code": "recall-range-invalid",
                "status": 400,
                "debugMessage": "Start after end.",
            },
            headers={"X-Request-Id": "req-9"},
        )
    )
    with pytest.raises(PastDevError) as caught:
        make_spec(api).recall(query="x", occurredFrom="2026-02-01", occurredTo="2026-01-01")
    assert caught.value.code == "recall-range-invalid"
    assert "req-9" in str(caught.value)
    assert len(api.requests) == 1


def test_rate_limit_is_retried_once() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, json={"code": "rate-limited", "status": 429}, headers={"Retry-After": "0"}
            )
        return httpx.Response(202, json=INGEST_RESPONSE)

    api = FakeApi(handler)
    assert "accepted" in make_spec(api).ingest(content="Ada prefers email.")
    assert len(api.requests) == 2


async def test_tools_work_async() -> None:
    api = FakeApi(default_handler)
    recall = make_spec(api).to_tool_list()[0]
    output = await recall.acall(query="How should I contact Ada?")
    assert "Ada prefers email" in output.content
