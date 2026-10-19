# past dev for LlamaIndex

past dev is a memory API for AI agents. This repository holds two LlamaIndex
integrations for [past dev](https://past.dev):

| Package | What it gives | Import |
| --- | --- | --- |
| [`llama-index-memory-pastdotdev`](llama-index-memory-pastdotdev/README.md) | `PastDevMemoryBlock`, a memory block for the LlamaIndex `Memory` class. It recalls before the agent answers and ingests the messages that the memory flushes. | `from llama_index.memory.pastdotdev import PastDevMemoryBlock` |
| [`llama-index-tools-pastdotdev`](llama-index-tools-pastdotdev/README.md) | `PastDevToolSpec`, with the `recall` and `ingest` tools for an agent. | `from llama_index.tools.pastdotdev import PastDevToolSpec` |

Use the memory block when memory must work without the model asking for it.
Use the tools when the model decides when to recall and what to keep. You can
use both together.

## Quick start

```bash
pip install llama-index-memory-pastdotdev llama-index-tools-pastdotdev
export PAST_API_KEY=past_sk_...
```

```python
from llama_index.core.memory import Memory
from llama_index.memory.pastdotdev import PastDevMemoryBlock
from llama_index.tools.pastdotdev import PastDevToolSpec

memory = Memory.from_defaults(
    session_id="session-42",
    memory_blocks=[PastDevMemoryBlock(identity=current_user.id)],
)
tools = PastDevToolSpec(identity=current_user.id).to_tool_list()
```

## The identity

past dev keeps memory per identity: your id for the end user. Recall needs an
identity, and the first ingest for an identity creates it. In both packages,
you set the identity in your code, from your trigger or your user record. The
model never chooses it: it is not a tool argument and no prompt can change it.

## Self-hosted past dev

Set `api_host` (or the `PAST_API_HOST` environment variable) to the address of
your own past dev. The default is `https://api.past.dev`.

## Development

Each package has its own `pyproject.toml`, tests and lock file, and uses `uv`:

```bash
cd llama-index-memory-pastdotdev
uv run -- pytest     # tests with a fake HTTP transport; no network
uv run ruff check .  # lint
uv build             # sdist and wheel
```

The live smoke tests call the real API and spend a few credits. They run only
when `PAST_API_KEY` is set. The two packages have the same `client.py`: if you
change one, copy it to the other.

The release steps are in [PUBLISHING.md](PUBLISHING.md).

## Links

- [past.dev](https://past.dev)
- [Quickstart](https://past.dev/docs/memory-api/quickstart)
- [API reference](https://past.dev/docs/memory-api/api-reference)
- [Privacy](https://past.dev/privacy) and [terms](https://past.dev/terms)

## License

MIT. See [LICENSE](LICENSE).
