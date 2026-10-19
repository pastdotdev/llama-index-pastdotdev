# Publishing

This file gives the release steps for the two packages and the steps for the
listing in the LlamaIndex documentation.

## How LlamaIndex takes a new integration

The LlamaIndex repository does not accept new integration packages. Its
`CONTRIBUTING.md` says:

> we are no longer accepting new integration packages in this repository.
> New integrations should be maintained in their own repositories and
> published to PyPI independently. PRs that add a new `pyproject.toml` will be
> automatically closed.

LlamaHub (`llamahub.ai`) now sends visitors to the community integrations page
of the LlamaIndex documentation:
<https://developers.llamaindex.ai/python/framework/community/integrations>.

Thus the release has three parts:

1. A public repository for this folder.
2. The two packages on PyPI.
3. A documentation-only pull request to `run-llama/llama_index` that adds
   links to the two packages on the community integrations page. It adds no
   `pyproject.toml`, so the rule above does not close it automatically. The
   maintainers decide if they merge it.

## 1. The public repository

1. Create the repository `pastdotdev/llama-index-pastdotdev`.
   The `Repository` URL in both `pyproject.toml` files points there. If you
   use a different name, change the two URLs, and the two PyPI trusted
   publishers, before the first release.
2. Copy the contents of this folder to the root of the new repository, as a
   plain copy (no history from another repository). The workflow
   `.github/workflows/publish.yml` must be at the root of the repository.
3. Push the first commit to `main`.
4. Make the repository public.

## 2. PyPI, with Trusted Publishing

The workflow `.github/workflows/publish.yml` publishes both packages when a
tag `v*` is pushed. It uses PyPI Trusted Publishing, so the release needs no
token and no secret.

### Once: the trusted publishers

The owner of the `pastdotdev` PyPI account adds two pending publishers at
<https://pypi.org/manage/account/publishing/>, one for each package:

| Field | Value |
| --- | --- |
| PyPI project name | `llama-index-memory-pastdotdev`, then `llama-index-tools-pastdotdev` |
| Owner | `pastdotdev` |
| Repository name | `llama-index-pastdotdev` |
| Workflow name | `publish.yml` |
| Environment name | (empty) |

### Each release

1. Make sure that the `version` in both `pyproject.toml` files and the
   `__version__` in both `base.py` files are the release version. The
   workflow stops if the tag and a `pyproject.toml` version are different.
2. Optional: run the live smoke tests. They spend a few credits (one ingest
   and two recalls).

   ```bash
   PAST_API_KEY=past_sk_... uv run -- pytest -m live
   ```

3. Push the tag:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. Watch the `publish` run in the repository's Actions tab. The `build` job
   runs the tests and builds both packages. The `publish` job uploads them.
5. Check the release in a clean environment:

   ```bash
   uv run --isolated --with llama-index-memory-pastdotdev --with llama-index-tools-pastdotdev \
     python -c "import llama_index.memory.pastdotdev, llama_index.tools.pastdotdev"
   ```

### Fallback: upload with a token

Use this only if Trusted Publishing is not available.

1. Make a PyPI API token at <https://pypi.org/manage/account/token/>. For the
   first upload, the scope must be "Entire account". After the first upload,
   replace it with one token for each project.
2. Test, build and upload each package. `uv publish` reads the token from
   `UV_PUBLISH_TOKEN`:

   ```bash
   cd llama-index-memory-pastdotdev
   uv run -- pytest && uv build && uvx twine check dist/*
   UV_PUBLISH_TOKEN=pypi-... uv publish
   cd ../llama-index-tools-pastdotdev
   uv run -- pytest && uv build && uvx twine check dist/*
   UV_PUBLISH_TOKEN=pypi-... uv publish
   ```

3. Tag the release: `git tag v0.1.0 && git push origin v0.1.0`. The workflow
   then runs and fails on the upload, because the files are already on PyPI.
   That failure is expected.

## 3. The listing pull request

Do this step after the packages are on PyPI.

1. Fork `run-llama/llama_index` and make the branch `docs/past-dev-integrations`.
2. Edit `docs/src/content/docs/framework/community/integrations.md`:

   - In the section `## Agent Tools`, after the line `- [MCP Toolbox](/python/examples/tools/mcp_toolbox)`, add:

     ```markdown
     - [past dev](https://pypi.org/project/llama-index-tools-pastdotdev/): recall and ingest tools for long-term agent memory (external package)
     ```

   - After the `## Agent Tools` section, add this section:

     ```markdown
     ## Memory

     Memory blocks for the [`Memory` class](/python/framework/module_guides/deploying/agents/memory):

     - [past dev](https://pypi.org/project/llama-index-memory-pastdotdev/): a memory block that recalls and ingests long-term memory with the past dev Memory API (external package)
     ```

3. Commit with the message `docs: list the past dev memory block and tools`.
4. Open the pull request. Use this title and body.

**Title:** `docs: list the past dev memory block and tools on the integrations page`

**Body:**

```markdown
# Description

This PR adds two links to the community integrations page. It changes documentation only.

past dev is a memory API for AI agents. We publish two external packages, in their own repository, as CONTRIBUTING.md asks for new integrations:

- `llama-index-memory-pastdotdev`: `PastDevMemoryBlock`, a `BaseMemoryBlock` for the `Memory` class. On get, it recalls with the latest user message as the query. On put, it ingests the messages that the memory flushes from its short-term queue.
- `llama-index-tools-pastdotdev`: `PastDevToolSpec`, a `BaseToolSpec` with the `recall` and `ingest` tools.

The end-user identity is fixed by the developer in the constructor. It is never a tool argument, so the model cannot read or write the memory of a different user.

The page has no Memory section at this time, so the PR adds one, next to Agent Tools.

- Source: https://github.com/pastdotdev/llama-index-pastdotdev
- PyPI: https://pypi.org/project/llama-index-memory-pastdotdev/ and https://pypi.org/project/llama-index-tools-pastdotdev/
- API documentation: https://past.dev/docs/memory-api/quickstart

## New Package?

- [ ] Yes
- [x] No

This PR adds no package to this repository. The packages are external. Their `pyproject.toml` files have a `tool.llamahub` section and each package has a README.

## Version Bump?

- [ ] Yes
- [x] No

Documentation only.

## Type of Change

- [x] This change requires a documentation update

## How Has This Been Tested?

- [x] I believe this change is already covered by existing unit tests

Documentation only. The external packages have their own pytest suites (fake HTTP with `httpx.MockTransport`, and a live smoke test that runs when an API key is set).

## Suggested Checklist:

- [x] I have performed a self-review of my own code
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
```
