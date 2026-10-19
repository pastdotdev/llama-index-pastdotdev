# LlamaIndex Tools Integration: past dev

past dev is a memory API for AI agents. This package gives a LlamaIndex agent
two tools over [past dev](https://past.dev):

- **`recall`**: a past dev **Recall**. The agent asks a question and gets the
  ranked memory of the current user, with the date, the kind and the quoted
  evidence of each entry.
- **`ingest`**: a past dev **Ingest**. The agent keeps one piece of information
  as a data point, so that a later recall can find it.

## Install

```bash
pip install llama-index-tools-pastdotdev
```

## Get a key

1. Sign up at [past.dev](https://past.dev) and open a project.
2. Make a project API key. It starts with `past_sk_`.
3. Set it in the environment as `PAST_API_KEY`, or give it to the spec as `api_key`.

## Use

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.pastdotdev import PastDevToolSpec

past_dev = PastDevToolSpec(
    identity=current_user.id,  # your id for the end user, from your own user record
)

agent = FunctionAgent(
    tools=past_dev.to_tool_list(),
    llm=OpenAI(model="gpt-4.1-mini"),
    system_prompt="Recall what you know about the user before you answer.",
)
response = await agent.run("What did I decide about the launch date?")
```

### The identity

past dev keeps memory per identity: your id for the end user. Recall needs an
identity, and the first ingest for an identity creates it.

Set `identity` in the `PastDevToolSpec` constructor, from your trigger or your
user record. It is not a tool argument: the model cannot see it or change it.
The same applies to `audience` and to the recall budget. Make one spec for
each user.

## The tools

| Tool | Arguments the model gives | API call |
| --- | --- | --- |
| `recall` | `query` (required), `occurredFrom`, `occurredTo`, `sort` (`relevance` or `chronological`) | `POST /api/v1/recall` |
| `ingest` | `content` (required), `timestamp` | `POST /api/v1/ingest` |

The argument names are the API's own field names.

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
| `label` | `Saved by a LlamaIndex agent` | The label of each data point that the model ingests. |
| `timeout` | `30` | The HTTP timeout in seconds. |
| `http_client` | none | An `httpx.Client` to use, for example with a proxy. |

## Credits

Each recall and each ingest spends credits on your past dev plan. The agent
decides when to call the tools, so tell it in the system prompt when to use them.

## Errors

An API error raises `PastDevError`, with the error `code`, the HTTP `status`,
the message and the `X-Request-Id` of the request. The LlamaIndex agent gives
the error to the model as the tool output. The spec retries a `429` once, after
its `Retry-After` delay. It does not retry other errors, for example
`402 out-of-credits` or a `400`.

## Links

- [Quickstart](https://past.dev/docs/memory-api/quickstart)
- [API reference](https://past.dev/docs/memory-api/api-reference)
- [LlamaIndex tools guide](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools/)
- A memory block for the same API: [`llama-index-memory-pastdotdev`](https://pypi.org/project/llama-index-memory-pastdotdev/)

## License

MIT. See [LICENSE](LICENSE).
