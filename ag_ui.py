import os
import json
import asyncio
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import the high-level MCP client
from client import FlightOpsMCPClient

app = FastAPI(title="FlightOps — AG-UI Adapter")

# CORS for your React front-end / AG-UI client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared FlightOpsMCPClient instance
mcp_client = FlightOpsMCPClient()


def sse_event(data: dict) -> str:
    """Format dict as a Server-Sent Event line."""
    payload = json.dumps(data, default=str)
    return f"data: {payload}\n\n"


async def ensure_mcp_connected():
    """Ensure MCP session/tools are ready."""
    if not getattr(mcp_client, "session", None):
        await mcp_client.connect()


@app.on_event("startup")
async def startup_event():
    try:
        await ensure_mcp_connected()
    except Exception as e:
        # Don't crash the API if MCP is down; client calls will surface errors anyway.
        logger = getattr(app, "logger", None)
        if logger:
            logger.warning(f"Could not preconnect to MCP: {e}")


@app.get("/")
async def root():
    return {"message": "FlightOps AG-UI Adapter is running", "status": "ok"}


@app.get("/health")
async def health():
    try:
        await ensure_mcp_connected()
        return {"status": "healthy", "mcp_connected": True}
    except Exception as e:
        return {"status": "unhealthy", "mcp_connected": False, "error": str(e)}


@app.post("/agent", response_class=StreamingResponse)
async def run_agent(request: Request):
    """
    AG-UI-compatible /agent endpoint.

    Expected JSON body:
      - thread_id (optional)
      - run_id (optional)
      - messages: list of AG-UI Message objects (we use last user message)
      - tools/context (optional)
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    thread_id = body.get("thread_id") or str(uuid.uuid4())
    run_id = body.get("run_id") or str(uuid.uuid4())
    messages = body.get("messages", [])
    tools = body.get("tools", [])  # not used yet, but reserved

    # Extract last user query
    user_query = ""
    if messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "user":
            user_query = last.get("content", "")
        elif isinstance(last, str):
            user_query = last

    if not user_query or not user_query.strip():
        raise HTTPException(status_code=400, detail="No user query in messages payload")

    async def event_stream() -> AsyncGenerator[str, None]:
        # 1) RUN_STARTED
        yield sse_event(
            {
                "type": "RUN_STARTED",
                "thread_id": thread_id,
                "run_id": run_id,
            }
        )

        # Ensure MCP is ready
        try:
            await ensure_mcp_connected()
        except Exception as e:
            yield sse_event({"type": "RUN_ERROR", "error": str(e)})
            return

        loop = asyncio.get_event_loop()

        # 2) PLAN
        yield sse_event(
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "content": "Generating tool plan...",
            }
        )

        plan_data = await loop.run_in_executor(None, mcp_client.run_chat, user_query)
        plan = plan_data.get("plan", [])

        # Emit plan snapshot
        yield sse_event(
            {
                "type": "STATE_SNAPSHOT",
                "snapshot": {"plan": plan},
            }
        )

        if not plan:
            yield sse_event(
                {
                    "type": "TEXT_MESSAGE_CONTENT",
                    "content": "LLM did not produce a valid plan.",
                }
            )
            yield sse_event({"type": "RUN_FINISHED", "run_id": run_id})
            return

        # 3) Execute each plan step
        results = []
        for step_index, step in enumerate(plan):
            tool_name = step.get("tool")
            args = step.get("arguments", {}) or {}
            tool_call_id = f"toolcall-{uuid.uuid4().hex[:8]}"

            # TOOL_CALL_START
            yield sse_event(
                {
                    "type": "TOOL_CALL_START",
                    "toolCallId": tool_call_id,
                    "toolCallName": tool_name,
                    "parentMessageId": None,
                }
            )

            # TOOL_CALL_ARGS
            args_json = json.dumps(args, default=str)
            yield sse_event(
                {
                    "type": "TOOL_CALL_ARGS",
                    "toolCallId": tool_call_id,
                    "delta": args_json,
                }
            )

            # TOOL_CALL_END (for args phase)
            yield sse_event(
                {
                    "type": "TOOL_CALL_END",
                    "toolCallId": tool_call_id,
                }
            )

            # Actual MCP tool invocation
            try:
                tool_result = await mcp_client.invoke_tool(tool_name, args)
            except Exception as exc:
                tool_result = {"error": str(exc)}

            # TOOL_CALL_RESULT
            tool_message = {
                "id": f"msg-{uuid.uuid4().hex[:8]}",
                "role": "tool",
                "content": json.dumps(tool_result, default=str),
                "tool_call_id": tool_call_id,
            }
            yield sse_event(
                {
                    "type": "TOOL_CALL_RESULT",
                    "message": tool_message,
                }
            )

            results.append({tool_name: tool_result})

            yield sse_event(
                {
                    "type": "STEP_FINISHED",
                    "step_index": step_index,
                    "tool": tool_name,
                }
            )

        # 4) Summarize
        yield sse_event(
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "content": "Summarizing results...",
            }
        )

        try:
            summary = await loop.run_in_executor(
                None, mcp_client.summarize_results, user_query, plan, results
            )
            assistant_text = (
                summary.get("summary", "") if isinstance(summary, dict) else str(summary)
            )
        except Exception as e:
            assistant_text = f"Failed to summarize results: {e}"

        # Final assistant message
        yield sse_event(
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "message": {
                    "id": f"msg-{uuid.uuid4().hex[:8]}",
                    "role": "assistant",
                    "content": assistant_text,
                },
            }
        )

        # Final snapshot + RUN_FINISHED
        yield sse_event(
            {
                "type": "STATE_SNAPSHOT",
                "snapshot": {"plan": plan, "results": results},
            }
        )
        yield sse_event({"type": "RUN_FINISHED", "run_id": run_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
