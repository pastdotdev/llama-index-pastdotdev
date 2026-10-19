"""past dev memory block for the LlamaIndex Memory class."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import httpx
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, PrivateAttr, model_validator
from llama_index.core.memory.memory import BaseMemoryBlock

from llama_index.memory.pastdotdev.client import (
    DEFAULT_API_HOST,
    AsyncPastDevClient,
    PastDevError,
    format_recall_results,
    now_iso,
    resolve_api_key,
    to_iso,
)

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

DEFAULT_NAME = "past_dev_memory"
DEFAULT_DESCRIPTION = (
    "Long-term memory recalled from past dev for the current user. "
    "Each entry has a rank, a date and a kind; evidence lines quote the source."
)
DEFAULT_LABEL = "LlamaIndex chat"
INGESTED_ROLES = (MessageRole.USER, MessageRole.ASSISTANT)


class PastDevMemoryBlock(BaseMemoryBlock[str]):
    """
    A memory block that keeps long-term memory in past dev.

    On get, it recalls with the latest user message as the query. On put (the
    messages that the Memory flushes from its short-term queue), it ingests the
    user and assistant text as one data point. The identity is fixed when you
    make the block: the model never chooses it.

    Args:
        identity: Your id for the end user. Recall reads, and ingest writes, as this identity.
        api_key: A past dev project API key (past_sk_...). Defaults to PAST_API_KEY.
        api_host: The API base URL. Defaults to PAST_API_HOST, else https://api.past.dev.
        audience: The audience slug for ingested memory. Omit it for project-visible memory.
        level: The recall evidence budget: low, medium, high or extra-high.
        max_tokens: An explicit recall evidence budget in tokens. Send level or max_tokens, not both.
        limit: The maximum number of recalled documents.
        label: The label of each ingested data point.
        raise_on_error: When False, an API error is logged and the block returns nothing.

    """

    name: str = Field(default=DEFAULT_NAME, description="The name of the memory block.")
    description: Optional[str] = Field(default=DEFAULT_DESCRIPTION)
    identity: str = Field(
        description="The end user's id. Fixed by the builder, never by the model."
    )
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    api_host: Optional[str] = Field(
        default=None, description=f"Defaults to PAST_API_HOST, else {DEFAULT_API_HOST}."
    )
    audience: Optional[str] = Field(default=None)
    level: Optional[Literal["low", "medium", "high", "extra-high"]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    limit: Optional[int] = Field(default=None, ge=1, le=200)
    label: str = Field(default=DEFAULT_LABEL)
    timeout: float = Field(default=30.0, gt=0)
    raise_on_error: bool = Field(default=True)

    _client: AsyncPastDevClient = PrivateAttr()

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None, **data: Any) -> None:
        super().__init__(**data)
        self._client = AsyncPastDevClient(
            api_key=resolve_api_key(self.api_key),
            api_host=self.api_host,
            timeout=self.timeout,
            user_agent=f"llama-index-memory-pastdotdev/{__version__}",
            http_client=http_client,
        )

    @model_validator(mode="after")
    def _check(self) -> "PastDevMemoryBlock":
        if not self.identity or not self.identity.strip():
            raise ValueError("identity is required: set it from your user record.")
        if self.level is not None and self.max_tokens is not None:
            raise ValueError("Set level or max_tokens, never both.")
        return self

    @classmethod
    def class_name(cls) -> str:
        return "PastDevMemoryBlock"

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        query = _latest_user_text(messages or [])
        if not query:
            return ""
        try:
            response = await self._client.recall(
                query=query,
                identity=self.identity,
                limit=self.limit,
                level=self.level,
                max_tokens=self.max_tokens,
            )
        except (PastDevError, httpx.HTTPError) as error:
            if self.raise_on_error:
                raise
            logger.warning("past dev recall failed: %s", error)
            return ""
        return "\n\n".join(format_recall_results(response.get("results") or []))

    async def _aput(self, messages: List[ChatMessage]) -> None:
        content = _transcript(messages)
        if not content:
            return
        try:
            await self._client.ingest(
                content=content,
                identity=self.identity,
                timestamp=_first_timestamp(messages) or now_iso(),
                audience=self.audience,
                label=self.label,
                metadata=_metadata(messages),
            )
        except (PastDevError, httpx.HTTPError) as error:
            if self.raise_on_error:
                raise
            logger.warning("past dev ingest failed: %s", error)

    async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
        """Drop the lowest-ranked entries until about tokens_to_truncate tokens are gone."""
        entries = content.split("\n\n") if content else []
        to_remove = tokens_to_truncate * 4  # about four characters a token
        while entries and to_remove > 0:
            to_remove -= len(entries.pop()) + 2
        return "\n\n".join(entries) or None


def _text(message: ChatMessage) -> str:
    return (message.content or "").strip()


def _latest_user_text(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == MessageRole.USER and _text(message):
            return _text(message)
    return ""


def _transcript(messages: List[ChatMessage]) -> str:
    lines = [
        f"{message.role.value}: {_text(message)}"
        for message in messages
        if message.role in INGESTED_ROLES and _text(message)
    ]
    return "\n\n".join(lines)


def _first_timestamp(messages: List[ChatMessage]) -> Optional[str]:
    """Use a timestamp that the application put on a message, if there is one."""
    for message in messages:
        value = message.additional_kwargs.get("timestamp")
        if isinstance(value, (datetime, str)):
            return to_iso(value)
    return None


def _metadata(messages: List[ChatMessage]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"source": "llama-index"}
    for message in messages:
        session_id = message.additional_kwargs.get("session_id")
        if isinstance(session_id, str):
            metadata["sessionId"] = session_id
            break
    return metadata
