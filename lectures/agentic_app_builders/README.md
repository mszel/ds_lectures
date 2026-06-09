# Agentic App Builders

A lesson demo for comparing agentic-AI frameworks side by side, so we can see *how* an
agentic app is actually wired together.

## The exercise

Each notebook builds the **same** small agentic app in a different framework, so you can compare
how they express identical ideas. You give the agent either a **list of URLs** *or* a **local file
path**, plus a **question**, and the agent **decides on its own** how to fetch, parse, and summarize
— returning a summary focused on your question.

The plumbing is shared and lives in `src/` (imported unchanged by every notebook):

- `src/mcp_servers/scraper_server.py` — a web **scraper exposed as an MCP server** (a separate
  process the agent talks to over the Model Context Protocol); run it standalone with
  `python -m src.mcp_servers.scraper_server`.
- `src/tools/parsers/` — `html_to_text` and `pdf_to_text` parsing tools.
- `src/tools/summarizer.py` — `summarize_for_question`, a question-focused summary (works for both
  the OpenAI and the local Ollama provider).

Only the framework-specific wiring (building the model, registering tools, consuming the MCP server,
memory) differs between notebooks — that difference is the point of the lesson:

| Notebook | Framework | MCP client | Memory |
|---|---|---|---|
| `01_pydantic_ai_summarizer.ipynb` | **Pydantic AI** | `pydantic_ai.mcp` (toolset) | conversation history |
| `02_langgraph_summarizer.ipynb` | **LangGraph** | `langchain-mcp-adapters` | `checkpointer` + `thread_id` |
| `03_openai_agents_sdk_summarizer.ipynb` | **OpenAI Agents SDK** | native `mcp_servers=[...]` | `SQLiteSession` (on disk) |
| `04_google_adk_summarizer.ipynb` | **Google ADK** | `MCPToolset` | session state (`{routing_hint}`) |

> **Note:** CrewAI was intentionally left out because it cannot run on Python 3.14 (its `chromadb` /
> pydantic-v1 dependency requires Python `<3.13`). To add it back, recreate the venv on Python 3.12.

Each notebook flips between OpenAI (`gpt-5-mini`) and a local Ollama model (`qwen3.5:4b`) with one
env var, `LLM_PROVIDER` (see below and `.env.example`).

### Installing the Environment

For installing the environment, the easiest is to use [uv](https://github.com/astral-sh/uv).
You can install uv as described in its README:
 - Linux/Mac: `curl -LsSf https://astral.sh/uv/install.sh | sh`
 - Mac (alternative): `brew install uv`
 - Windows:
   * PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   * However, the best is to [set up a WSL environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment), as it's uncomfortable to use Windows directly.

Follow these steps to create an environment where you can use the API:
 - Create a new uv environment: `uv venv --python=$(which python3.13) .venv`
 - Switch to the new environment: `source .venv/bin/activate`
 - Install packages from the `pyproject.toml` file (with dependencies): `uv pip install -e ".[dev]"`

As a next step, create a `.env` file by copying the provided template (`.env.example`) and
filling it with the needed secrets, like:
 - `OPENAI_API_KEY=...`  # obtain one from Marton.

It is also worth installing the pre-commit hooks (run from the repo root): `pre-commit install`.
Before adding the files to git, you can run the `pre-commit run --all-files` command, so it
will correct the formatting issues of all `.py` files.

## Running Local Models

If you'd like to try a locally hosted model in your local dev environment, you can use ollama:
 - Install ollama: https://ollama.com/download
 - Run the Ollama app on Windows / Mac, or run the following command (Linux or any OS): `ollama serve`
 - In a new terminal, run: `ollama run qwen3:8b` (you can use any model [from here](https://ollama.com/search) — e.g. `gemma3`, not just `qwen3`)

Ollama exposes an OpenAI-compatible endpoint, so you can point any OpenAI client at it:
 - Base URL: `http://localhost:11434/v1`
 - API key: any non-empty string (e.g. `ollama`) — it is ignored locally
 - Model: whatever you pulled, e.g. `qwen3:8b`

Example with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Explain agentic AI in one sentence."}],
)
print(response.choices[0].message.content)
```
