import json
from typing import Any, Callable, Dict, List

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import Memory
from llama_index.core.memory.memory import BaseMemoryBlock

from llama_index.memory.pastdotdev import PastDevError, PastDevMemoryBlock

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
                    "label": "Support chat",
                    "excerpts": ["please email me, never call"],
                }
            ],
        },
        {
            "id": "7f1c3a4e-0000-4000-8000-000000000002",
            "rank": 2,
            "occurredAt": "2026-09-01T09:00:00Z",
            "content": "Ada lives in Lyon.",
            "artifact": {
                "id": "7f1c3a4e-0000-4000-8000-000000000002",
                "kind": "claim",
                "occurredAt": "2026-09-01T09:00:00Z",
                "supersededAt": "2026-10-02T09:00:00Z",
            },
            "sources": [],
        },
    ],
}

INGEST_RESPONSE = {
    "projectId": 1,
    "ingestionId": "7f1c3a4e-0000-4000-8000-0000000000aa",
    "sourceId": "abc",
    "status": "queued",
}


class FakeApi:
    """Records each request and answers with a handler."""

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


def make_block(api: FakeApi, **kwargs: Any) -> PastDevMemoryBlock:
    kwargs.setdefault("api_key", KEY)
    kwargs.setdefault("identity", "user-ada")
    return PastDevMemoryBlock(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)), **kwargs
    )


def test_is_a_memory_block() -> None:
    names = [b.__name__ for b in PastDevMemoryBlock.__mro__]
    assert BaseMemoryBlock.__name__ in names


def test_requires_identity() -> None:
    with pytest.raises(ValueError):
        make_block(FakeApi(default_handler), identity="  ")


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PAST_API_KEY"):
        PastDevMemoryBlock(identity="user-ada")


def test_reads_key_and_host_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAST_API_KEY", "past_sk_from_env")
    monkeypatch.setenv("PAST_API_HOST", "https://memory.example.com/")
    api = FakeApi(default_handler)
    block = PastDevMemoryBlock(
        identity="user-ada", http_client=httpx.AsyncClient(transport=httpx.MockTransport(api))
    )
    assert block._client.api_host == "https://memory.example.com"


def test_refuses_level_and_max_tokens_together() -> None:
    with pytest.raises(ValueError):
        make_block(FakeApi(default_handler), level="high", max_tokens=1000)


def test_api_key_is_not_serialized() -> None:
    block = make_block(FakeApi(default_handler))
    assert KEY not in json.dumps(block.model_dump())
    assert KEY not in repr(block)


async def test_get_recalls_with_latest_user_message() -> None:
    api = FakeApi(default_handler)
    block = make_block(api, level="medium", limit=5)
    content = await block.aget(
        [
            ChatMessage(role=MessageRole.USER, content="Where does Ada live?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Let me check."),
            ChatMessage(role=MessageRole.USER, content="How should I contact Ada?"),
        ]
    )
    assert api.bodies("/api/v1/recall") == [
        {
            "query": "How should I contact Ada?",
            "identity": "user-ada",
            "level": "medium",
            "limit": 5,
        }
    ]
    request = api.requests[0]
    assert request.method == "POST"
    assert request.url.host == "api.past.dev"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert "Ada prefers email to phone calls." in content
    assert '"please email me, never call"' in content
    assert "Support chat" in content
    assert "superseded 2026-10-02" in content


async def test_get_without_user_message_calls_nothing() -> None:
    api = FakeApi(default_handler)
    block = make_block(api)
    assert await block.aget([ChatMessage(role=MessageRole.SYSTEM, content="Be kind.")]) == ""
    assert await block.aget(None) == ""
    assert api.requests == []


async def test_put_ingests_conversation_text() -> None:
    api = FakeApi(default_handler)
    block = make_block(api, audience="support-team", label="Support chat")
    await block.aput(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful agent."),
            ChatMessage(
                role=MessageRole.USER,
                content="Please email me, never call.",
                additional_kwargs={"timestamp": "2026-10-01T09:00:00Z"},
            ),
            ChatMessage(role=MessageRole.TOOL, content='{"ok": true}'),
            ChatMessage(role=MessageRole.ASSISTANT, content="Noted: email only."),
        ],
        session_id="session-1",
    )
    assert api.bodies("/api/v1/ingest") == [
        {
            "content": "user: Please email me, never call.\n\nassistant: Noted: email only.",
            "identity": "user-ada",
            "timestamp": "2026-10-01T09:00:00Z",
            "audience": "support-team",
            "label": "Support chat",
            "metadata": {"source": "llama-index", "sessionId": "session-1"},
        }
    ]


async def test_put_defaults_timestamp_to_now() -> None:
    api = FakeApi(default_handler)
    await make_block(api).aput([ChatMessage(role=MessageRole.USER, content="Hello")])
    body = api.bodies("/api/v1/ingest")[0]
    assert body["timestamp"].startswith("20")
    assert body["label"] == "LlamaIndex chat"
    assert "audience" not in body


async def test_put_with_no_text_calls_nothing() -> None:
    api = FakeApi(default_handler)
    await make_block(api).aput([ChatMessage(role=MessageRole.TOOL, content="result")])
    assert api.requests == []


async def test_out_of_credits_is_raised_and_not_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={"code": "out-of-credits", "status": 402, "debugMessage": "No credits left."},
            headers={"X-Request-Id": "req-123"},
        )

    api = FakeApi(handler)
    with pytest.raises(PastDevError) as caught:
        await make_block(api).aget([ChatMessage(role=MessageRole.USER, content="Hi")])
    assert caught.value.code == "out-of-credits"
    assert caught.value.status == 402
    assert caught.value.request_id == "req-123"
    assert "No credits left." in str(caught.value)
    assert len(api.requests) == 1


async def test_rate_limit_is_retried_once_after_retry_after() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, json={"code": "rate-limited", "status": 429}, headers={"Retry-After": "0"}
            )
        return httpx.Response(200, json=RECALL_RESPONSE)

    api = FakeApi(handler)
    content = await make_block(api).aget([ChatMessage(role=MessageRole.USER, content="Hi")])
    assert len(api.requests) == 2
    assert "Ada" in content


async def test_long_retry_after_is_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"code": "rate-limited", "status": 429}, headers={"Retry-After": "3600"}
        )

    api = FakeApi(handler)
    with pytest.raises(PastDevError, match="rate-limited"):
        await make_block(api).aget([ChatMessage(role=MessageRole.USER, content="Hi")])
    assert len(api.requests) == 1


async def test_errors_can_be_logged_instead_of_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"code": "unavailable", "status": 503})

    api = FakeApi(handler)
    block = make_block(api, raise_on_error=False)
    assert await block.aget([ChatMessage(role=MessageRole.USER, content="Hi")]) == ""
    await block.aput([ChatMessage(role=MessageRole.USER, content="Hi")])
    assert len(api.requests) == 2


async def test_non_json_error_is_reported() -> None:
    api = FakeApi(lambda request: httpx.Response(502, text="Bad gateway"))
    with pytest.raises(PastDevError) as caught:
        await make_block(api).aget([ChatMessage(role=MessageRole.USER, content="Hi")])
    assert caught.value.code == "http-error"
    assert caught.value.status == 502


async def test_truncate_drops_lowest_ranked_entries() -> None:
    block = make_block(FakeApi(default_handler))
    content = "[1] first\n\n[2] second\n\n[3] third"
    assert await block.atruncate(content, 2) == "[1] first\n\n[2] second"
    assert await block.atruncate(content, 1000) is None


async def test_works_inside_llama_index_memory() -> None:
    api = FakeApi(default_handler)
    memory = Memory.from_defaults(
        session_id="session-1",
        token_limit=60,
        chat_history_token_ratio=0.5,
        token_flush_size=20,
        memory_blocks=[make_block(api)],
    )
    for turn in range(4):
        await memory.aput(ChatMessage(role=MessageRole.USER, content=f"Question {turn} about Ada?"))
        await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content=f"Answer {turn}."))

    ingested = api.bodies("/api/v1/ingest")
    assert ingested, "the memory flushed no messages to the block"
    assert ingested[0]["identity"] == "user-ada"
    assert ingested[0]["metadata"]["sessionId"] == "session-1"
    assert ingested[0]["content"].startswith("user: Question 0 about Ada?")

    messages = await memory.aget()
    assert "Ada prefers email to phone calls." in str(messages[0].content)
    assert api.bodies("/api/v1/recall")[-1]["query"] == "Question 3 about Ada?"
