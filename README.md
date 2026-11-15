# LangGraph Agentic Classical ML Tutorial

This project demonstrates two levels of an agentic classical ML application:

1. A deterministic LangGraph workflow trains and evaluates a digit classifier.
2. A FastAPI chat app uses an OpenAI tool-calling agent, LangGraph, and an MCP
   server to let users operate the ML workflow in natural language.

The dataset is `sklearn.datasets.load_digits`, a small local MNIST-like dataset
with 8x8 grayscale images. It is used instead of downloading full MNIST so the
demo is reproducible and fast.

## 1. Architecture

```text
Browser
  -> FastAPI /api/chat
    -> LangGraph chat_agent.py
      -> OpenAI chooses an MCP tool
        -> MCP stdio client
          -> mcp_server.py
            -> scikit-learn training/prediction
      <- tool result
    <- OpenAI writes the final response
  <- JSON response
```

The important boundary is the MCP server. The chat agent does not import the
ML implementation directly; it calls explicit tools:

- `train_mnist(target_accuracy)`: train and evaluate the classifier.
- `predict_digit(sample_index)`: classify a held-out sample.
- `model_status()`: return the latest persisted metrics.

The latest summary is written to `.mnist_status.json`. This file is ignored by
Git and allows status queries to work even though the stdio MCP server is
started as a short-lived process for each request.
The trained model and held-out test data are persisted in `.mnist_model.pkl`,
so a later prediction can load the existing model instead of retraining.

The MCP server also maintains a developer log at
`logs/mnist_mcp.log`. It appends timestamped records for dataset loading,
training start, per-model evaluation, model selection, prediction, status
checks, completion, and failures. The log is ignored by Git and is the primary
place to verify that an operation actually executed; it is not required to be
shown in the chat response.

## 2. Components

| File | Responsibility |
| --- | --- |
| [mnist_workflow.py] | LangGraph ML pipeline: load, train, evaluate, retry, predict |
| [mcp_server.py] | MCP tool server around the ML operations |
| [chat_agent.py] | Production OpenAI tool-calling LangGraph agent |
| [legacy_chat_agent.py] | Optional deterministic routing alternative; never imported by the app |
| [web_app.py] | FastAPI API and browser chat UI |
| [agent.py] | Separate arithmetic LangGraph example |
| [tests/] | Workflow and chat tests |

## 3. Installation

Use Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Set at least:

```text
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-5-nano
```

Never commit `.env` or expose API keys in chat, source code, or screenshots.

## 4. Run the classical ML workflow directly

```powershell
python mnist_workflow.py
```

This path does not need OpenAI. It produces deterministic metrics such as:

```text
Model: logistic_regression
Attempts: 1
Accuracy: 0.972
Predicted held-out digit: 5
```

The graph nodes are:

1. `load_data`
2. `train_model`
3. `evaluate_model`
4. `predict_digit`

If the target accuracy is not reached, the evaluation edge loops back to
training for another model attempt.

## 5. Run the web app

```powershell
python -m uvicorn web_app:app --reload
```

Open <http://127.0.0.1:8000>.

## 6. Use LangGraph Studio

The [langgraph.json]
configuration exposes the arithmetic agent, the OpenAI/MCP chat agent, and the
deterministic MNIST workflow to LangGraph Studio.

Install the LangGraph CLI and start the local development server:

```powershell
python -m pip install -U "langgraph-cli[inmem]"
# The CLI may install MCP 2.x; this project uses the MCP 1.x API.
python -m pip install "mcp>=1.0,<2"
langgraph dev --allow-blocking
```

Open the URL printed by the CLI and select a graph. The chat agent requires
`OPENAI_API_KEY` in `.env`; the arithmetic agent and MNIST workflow use the
configuration's existing environment loading. Restart the CLI after changing
the MCP version.

Try:

- `Train the model with a target accuracy of 0.95`
- `What is the current accuracy?`
- `Predict sample 12`
- `Which model is currently ready?`
- `Explain how the evaluation works`

### OpenAI agent flow

The production agent in [chat_agent.py]
does not match keywords. It:

1. Sends the user message and tool schemas to OpenAI.
2. Lets OpenAI select a tool and generate validated arguments.
3. Executes that tool through MCP.
4. Adds the MCP result to the LangGraph message state.
5. Calls OpenAI again to produce the final answer.
6. Repeats the tool loop if more tool calls are requested.

The system prompt instructs the model to use tools for all metrics and never
invent accuracy values.

## 7. Legacy deterministic alternative

[legacy_chat_agent.py]
is intentionally preserved for learning. It is not imported by
[web_app.py]
[chat_agent.py]

Run it directly:

```powershell
python -c "import asyncio; from legacy_chat_agent import run_legacy_chat; print(asyncio.run(run_legacy_chat('what is the current accuracy?')))"
```

| Topic | OpenAI agent | Legacy router |
| --- | --- | --- |
| Intent detection | Model understands varied language | Fixed keywords and regex |
| Tool choice | OpenAI tool calling | Hard-coded branches |
| Arguments | Model extracts arguments, MCP validates them | Regex/default values |
| Final response | Model-generated from tool output | Fixed string templates |
| Cost/latency | Requires API calls | No LLM cost |
| Flexibility | Handles paraphrases and follow-ups | Limited to coded phrases |
| Failure mode | API/model/tool errors | Unrecognized wording or wrong branch |
| Best use | Production-style conversational agent | Teaching, offline fallback, baseline |

The legacy file demonstrates routing, but it is not the application path.
Keeping it separate makes the architectural difference explicit.

## 8. LangSmith tracing

Set these values in `.env`:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=mnist-langgraph-demo
```

Run the app or workflow again, then open the project in LangSmith. The chat
trace should show:

- `mnist-chat-agent`
- `openai_chat_reasoning`
- `call_mnist_mcp_tool`
- the MCP server tool execution
- the final OpenAI response

The direct workflow uses the `mnist-agentic-classical-ml` run name and tags for
data, training, evaluation, and prediction.

If a run is stopped while MCP is active, LangSmith may retain an old trace as
`pending` because the completion event was never sent. Treat that as an
interrupted historical run; newer completed traces are the source of truth.

## 9. Testing

```powershell
python -m pytest
```

The tests cover the ML graph and use an injected fake chat model, so tests do
not consume OpenAI credits. The live web app requires `OPENAI_API_KEY`.
