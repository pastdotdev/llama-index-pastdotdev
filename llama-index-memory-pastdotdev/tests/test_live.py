"""One smoke test against the real past dev API.

It is skipped unless PAST_API_KEY is set. It spends a few credits: one ingest and one recall.
"""

import os

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from llama_index.memory.pastdotdev import PastDevMemoryBlock

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("PAST_API_KEY"), reason="PAST_API_KEY is not set"),
]


async def test_live_put_and_get() -> None:
    block = PastDevMemoryBlock(identity="llama-index-smoke-test", label="LlamaIndex smoke test")
    # The connection test: listing audiences costs no credits.
    audiences = await block._client.list_audiences()
    assert isinstance(audiences["audiences"], list)

    await block.aput(
        [
            ChatMessage(role=MessageRole.USER, content="My favourite colour is teal."),
            ChatMessage(role=MessageRole.ASSISTANT, content="Noted: teal."),
        ]
    )
    content = await block.aget(
        [ChatMessage(role=MessageRole.USER, content="What is my favourite colour?")]
    )
    assert isinstance(content, str)
