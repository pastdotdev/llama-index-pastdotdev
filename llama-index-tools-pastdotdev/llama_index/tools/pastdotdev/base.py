"""past dev tool spec: Recall and Ingest as agent tools."""

from typing import Literal, Optional

import httpx
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.pastdotdev.client import (
    PastDevClient,
    format_recall,
    now_iso,
)

__version__ = "0.1.0"


class PastDevToolSpec(BaseToolSpec):
    """
    past dev tools: recall and ingest.

    The identity, the audience and the recall budget are fixed when you make
    the spec, from your own user record. The model never sees them and cannot
    change them: the tools take only the query or the content.

    Args:
        identity: Your id for the end user. The tools read and write as this identity.
        api_key: A past dev project API key (past_sk_...). Defaults to PAST_API_KEY.
        api_host: The API base URL. Defaults to PAST_API_HOST, else https://api.past.dev.
        audience: The audience slug for ingested memory. Omit it for project-visible memory.
        level: The recall evidence budget: low, medium, high or extra-high.
        max_tokens: An explicit recall evidence budget in tokens. Send level or max_tokens, not both.
        limit: The maximum number of recalled documents.
        label: The label of each data point the model ingests.
        timeout: The HTTP timeout in seconds.
        http_client: An httpx.Client to use, for example with a proxy.

    """

    spec_functions = ["recall", "ingest"]

    def __init__(
        self,
        identity: str,
        api_key: Optional[str] = None,
        api_host: Optional[str] = None,
        audience: Optional[str] = None,
        level: Optional[Literal["low", "medium", "high", "extra-high"]] = None,
        max_tokens: Optional[int] = None,
        limit: Optional[int] = None,
        label: Optional[str] = "Saved by a LlamaIndex agent",
        timeout: float = 30.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not identity or not identity.strip():
            raise ValueError("identity is required: set it from your user record.")
        if level is not None and max_tokens is not None:
            raise ValueError("Set level or max_tokens, never both.")
        self.identity = identity
        self.audience = audience
        self.level = level
        self.max_tokens = max_tokens
        self.limit = limit
        self.label = label
        self._client = PastDevClient(
            api_key=api_key,
            api_host=api_host,
            timeout=timeout,
            user_agent=f"llama-index-tools-pastdotdev/{__version__}",
            http_client=http_client,
        )

    def recall(
        self,
        query: str,
        occurredFrom: Optional[str] = None,
        occurredTo: Optional[str] = None,
        sort: Optional[Literal["relevance", "chronological"]] = None,
    ) -> str:
        """
        Recall what past dev remembers about the current user: earlier conversations, facts, preferences and decisions.

        Use it before you answer a question that may depend on what happened before.
        Each result has a rank, a date and a kind; evidence lines quote the original source.

        Args:
            query (str): A natural-language question, for example "What did the user decide about the launch date?".
            occurredFrom (Optional[str]): Only memory that occurred at or after this ISO 8601 date-time.
            occurredTo (Optional[str]): Only memory that occurred at or before this ISO 8601 date-time.
            sort (Optional[str]): "relevance" (the default) or "chronological" (oldest first).

        """
        response = self._client.recall(
            query=query,
            identity=self.identity,
            limit=self.limit,
            level=self.level,
            max_tokens=self.max_tokens,
            occurred_from=occurredFrom,
            occurred_to=occurredTo,
            sort=sort,
        )
        return format_recall(response) or "past dev has no memory that matches this query."

    def ingest(self, content: str, timestamp: Optional[str] = None) -> str:
        """
        Ingest one piece of information into past dev, so that a later recall can find it.

        Use it when the user tells you something worth keeping: a fact, a preference or a decision.
        Write the content as a complete, self-contained statement.

        Args:
            content (str): The text to keep, for example "The user prefers email to phone calls.".
            timestamp (Optional[str]): When it occurred, as an ISO 8601 date-time. Defaults to now.

        """
        result = self._client.ingest(
            content=content,
            identity=self.identity,
            timestamp=timestamp or now_iso(),
            audience=self.audience,
            label=self.label,
            metadata={"source": "llama-index"},
        )
        result = result or {}
        source_id = result.get("sourceId")
        if result.get("unchanged"):
            return f"past dev already had this data point (source id {source_id})."
        return f"past dev accepted the data point (source id {source_id}, {result.get('status', 'queued')})."
