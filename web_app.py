"""FastAPI web app for the LangGraph/MCP MNIST chat agent."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from chat_agent import run_chat

app = FastAPI(title="MNIST LangGraph MCP Agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNIST ML Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 3rem auto;
           padding: 0 1rem; background: #f5f7fb; color: #172033; }
    main { background: white; border-radius: 12px; padding: 1.5rem;
           box-shadow: 0 4px 20px #17203318; }
    #messages { min-height: 280px; display: flex; flex-direction: column; gap: .75rem; }
    .message { padding: .75rem 1rem; border-radius: 10px; white-space: pre-wrap; }
    .user { background: #e8f0ff; align-self: flex-end; }
    .agent { background: #eef7ee; align-self: flex-start; }
    form { display: flex; gap: .5rem; margin-top: 1rem; }
    input { flex: 1; padding: .8rem; border: 1px solid #ccd3e0; border-radius: 8px; }
    button { padding: .8rem 1.2rem; border: 0; border-radius: 8px;
             background: #2563eb; color: white; cursor: pointer; }
    small { color: #586174; }
  </style>
</head>
<body>
  <main>
    <h1>MNIST Classical ML Agent</h1>
    <small>LangGraph routes your request to tools exposed by an MCP server.</small>
    <div id="messages"></div>
    <form>
      <input id="message" autocomplete="off"
        placeholder="Try: train with accuracy 0.95">
      <button>Send</button>
    </form>
  </main>
  <script>
    const messages = document.querySelector("#messages");
    const form = document.querySelector("form");
    const input = document.querySelector("#message");
    function add(text, role) {
      const item = document.createElement("div");
      item.className = "message " + role;
      item.textContent = text;
      messages.appendChild(item);
      item.scrollIntoView();
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      add(message, "user");
      input.value = "";
      try {
        const reply = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({message})
        });
        const data = await reply.json();
        if (!reply.ok) throw new Error(data.detail || "Request failed");
        add(data.response, "agent");
      } catch (error) {
        add("Error: " + error.message, "agent");
      }
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(response=await run_chat(request.message))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Agent request failed") from error
