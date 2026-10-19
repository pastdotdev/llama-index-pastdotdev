"""One smoke test against the real past dev API.

It is skipped unless PAST_API_KEY is set. It spends a few credits: one recall.
"""

import os

import pytest

from llama_index.tools.pastdotdev import PastDevToolSpec

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("PAST_API_KEY"), reason="PAST_API_KEY is not set"),
]


def test_live_recall_tool() -> None:
    spec = PastDevToolSpec(identity="llama-index-smoke-test")
    # The connection test: listing audiences costs no credits.
    assert isinstance(spec._client.list_audiences()["audiences"], list)

    tools = {tool.metadata.name: tool for tool in spec.to_tool_list()}
    output = tools["recall"](query="What is my favourite colour?")
    assert isinstance(output.content, str) and output.content
