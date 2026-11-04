# Agentic Arithmetic

A small LangGraph agent that uses an LLM to decide when to call arithmetic
tools.

## Setup

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the project:

   ```powershell
   python -m pip install langchain dotenv langgraph langsmith langchain-openai pytest -e ".[dev]"
   ```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Run

```powershell
python agent.py
```

## Test

```powershell
python -m pytest
```
