help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Format the code (ruff).
	uv run ruff format .
	uv run ruff check --fix .

lint:	## Run the linter (ruff).
	uv run ruff check .
	uv run ruff format --check .

test:	## Run the tests. The live tests run only when PAST_API_KEY is set.
	uv run -- pytest

build:	## Build the sdist and the wheel into dist/.
	uv build
