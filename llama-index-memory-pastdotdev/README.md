# LlamaIndex Memory Integration: past dev

past dev is a memory API for AI agents. This package gives a LlamaIndex agent
long-term memory in [past dev](https://past.dev), as a memory block for the
LlamaIndex `Memory` class.

- **Get:** the block does a past dev **Recall**. The query is the latest user
  message. The recalled memory goes into the system message, with the date,
  the kind and the quoted evidence of each entry.
- **Put:** when the `Memory` flushes old messages from its short-term queue,
  the block does a past dev **Ingest** of the user and assistant text, as one
  data point.

## Install

```bash
pip install llama-index-memory-pastdotdev
```

## Get a key

1. Sign up at [past.dev](https://past.dev) and open a project.
2. Make a project API key. It starts with `past_sk_`.
3. Set it in the environment as `PAST_API_KEY`, or give it to the block as `api_key`.

## Use

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.llms.openai import OpenAI
from llama_index.memory.pastdotdev import PastDevMemoryBlock

memory = Memory.from_defaults(
    session_id="session-42",
    token_limit=30000,
    memory_blocks=[
        PastDevMemoryBlock(
            identity=current_user.id,  # your id for the end user, from your own user record
        )
    ],
)

agent = FunctionAgent(tools=[], llm=OpenAI(model="gpt-4.1-mini"))
response = await agent.run("How should you contact me?", memory=memory)
```

### The identity

past dev keeps memory per identity: your id for the end user. Recall needs an
identity, and the first ingest for an identity creates it.

Set `identity` in your code, from your trigger or your user record. Do not let
the model choose it. The block has no tool and no prompt that can change the
identity, so each user reads and writes only their own memory.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `identity` | (required) | Your id for the end user. |
| `api_key` | `PAST_API_KEY` | A past dev project API key (`past_sk_...`). |
| `api_host` | `PAST_API_HOST`, else `https://api.past.dev` | The API base URL. Change it only if you run past dev in your own cloud. |
| `audience` | none | The audience slug for ingested memory. Without it, the memory is visible to the project. |
| `level` | `low` (on the server) | The recall evidence budget: `low`, `medium`, `high` or `extra-high`. |
| `max_tokens` | none | An explicit recall evidence budget in tokens. Use `level` or `max_tokens`, not both. |
| `limit` | 100 (on the server) | The maximum number of recalled documents. |
| `label` | `LlamaIndex chat` | The label of each ingested data point. |
| `name` | `past_dev_memory` | The name of the block in the memory template. |
| `priority` | `0` | The `BaseMemoryBlock` truncation priority. `0` means never truncate. |
| `raise_on_error` | `True` | When `False`, the block logs an API error and continues without memory. |

## What the block sends

- **Recall** (`POST /api/v1/recall`): `query` (the latest user message),
  `identity`, and `level`, `maxTokens` or `limit` if you set them.
- **Ingest** (`POST /api/v1/ingest`): `content` (the flushed user and assistant
  messages as `role: text` lines; system and tool messages are not sent),
  `identity`, `timestamp`, `label`, `audience` if you set it, and `metadata`
  with `{"source": "llama-index", "sessionId": ...}`.

The `timestamp` is the `timestamp` of the first flushed message, if your
application puts one in `message.additional_kwargs["timestamp"]`. Otherwise it
is the time of the flush.

## Credits

Each recall and each ingest spends credits on your past dev plan. The block
recalls once each time the agent reads its memory, and ingests once each time
the `Memory` flushes messages.

## Errors

An API error raises `PastDevError`, with the error `code`, the HTTP `status`,
the message and the `X-Request-Id` of the request. The block retries a `429`
once, after its `Retry-After` delay. It does not retry other errors, for
example `402 out-of-credits` or a `400`.

## Links

- [Quickstart](https://past.dev/docs/memory-api/quickstart)
- [API reference](https://past.dev/docs/memory-api/api-reference)
- [LlamaIndex memory guide](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/)
- Tools for the same API: [`llama-index-tools-pastdotdev`](https://pypi.org/project/llama-index-tools-pastdotdev/)

## License

MIT. See [LICENSE](LICENSE).
