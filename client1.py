
import os
import json
import asyncio
from typing import Any, Dict, List, Optional

import openai
from mcp.client.async_client import AsyncClient
from mcp.client.transports.http import HTTPTransport

# === Config ===

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")

openai.api_key = OPENAI_API_KEY

# === Simple safety guard ===

BAD_KEYWORDS = [
    "hack", "exploit", "injection", "$where", "drop database",
    "weapon", "bomb", "kill", "passport", "credit card", "pnr",
    "ticket number", "password", "login details",
]

def cheap_safety_check(user_query: str) -> bool:
    lower = user_query.lower()
    return not any(kw in lower for kw in BAD_KEYWORDS)


class FlightOpsMCPClient:
    """
    High-level client that:
      • maintains a persistent MCP AsyncClient
      • can call MCP tools
      • can ask OpenAI to PLAN which tools to call
      • can ask OpenAI to SUMMARIZE tool results
    """

    def __init__(self) -> None:
        self.session: Optional[AsyncClient] = None
        self.tools_for_openai: List[Dict[str, Any]] = []

    # ----------------------------
    # MCP connection + tools
    # ----------------------------
    async def connect(self) -> None:
        """
        Create the MCP AsyncClient and cache tool schemas for OpenAI.
        Safe to call multiple times.
        """
        if self.session is not None:
            return

        transport = HTTPTransport(MCP_SERVER_URL)
        self.session = AsyncClient(transport=transport)

        tools_list = await self.session.list_tools()
        tools_for_openai: List[Dict[str, Any]] = []

        for t in tools_list.tools:
            schema = t.inputSchema or {"type": "object", "properties": {}}
            tools_for_openai.append(
                {
                    "name": t.name,
                    "description": t.description or "",
                    # OpenAI 0.x: use `parameters` for function schema
                    "parameters": schema,
                }
            )

        self.tools_for_openai = tools_for_openai

    async def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a MCP tool and return its content.
        """
        if self.session is None:
            raise RuntimeError("MCP client is not connected – call connect() first.")

        tool_resp = await self.session.call_tool(name=tool_name, arguments=arguments or {})

        # FastMCP(json_response=True) usually wraps the actual JSON in `.content`.
        # We just surface `content` – adapter will JSON-serialize with default=str.
        return getattr(tool_resp, "content", tool_resp)

    # ----------------------------
    # OpenAI helpers
    # ----------------------------
    def _openai_chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """
        Thin wrapper to call OpenAI ChatCompletion.
        """
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            **kwargs,
        )
        # Old OpenAI SDK response object acts like dict
        return resp

    # ----------------------------
    # Planner: returns {"plan": [...]}
    # ----------------------------
    def run_chat(self, user_query: str) -> Dict[str, Any]:
        """
        PLAN which MCP tools to call for the given user query.

        Returns:
            { "plan": [ { "tool": str, "arguments": dict, "reason": str }, ... ] }
        """
        if not cheap_safety_check(user_query):
            return {
                "plan": [],
                "safety_blocked": True,
                "message": "Query blocked by safety filter.",
            }

        # Describe available tools for the planner
        tools_description_lines = []
        for t in self.tools_for_openai:
            tools_description_lines.append(
                f"- {t['name']}: {t.get('description','')}"
            )
        tools_description = "\n".join(tools_description_lines)

        system_prompt = f"""
You are a flight operations planning assistant.

You have access to several MCP tools that can query a MongoDB flight database.
Available tools:
{tools_description}

Your job is ONLY to produce a JSON object with a key "plan".

"plan" is an array of steps. Each step is:
  {{
    "tool": "<tool_name>",
    "arguments": {{
       ... tool arguments as a JSON object ...
    }},
    "reason": "Why this tool and arguments help answer the user"
  }}

Rules:
- Use at most 3 steps unless absolutely necessary.
- Prefer specific tools over broad ones.
- If no tool is needed, return "plan": [] but still as valid JSON.
- Output VALID JSON only. Do not include explanations outside the JSON.
"""

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        resp = self._openai_chat(messages, temperature=0)
        msg = resp["choices"][0]["message"]
        content = msg.get("content", "") or ""

        # Try to parse JSON strictly; if it fails, fallback to empty plan.
        try:
            plan_data = json.loads(content)
            if not isinstance(plan_data, dict):
                raise ValueError("Planner output is not a JSON object")
        except Exception:
            plan_data = {"plan": []}

        if "plan" not in plan_data or not isinstance(plan_data["plan"], list):
            plan_data["plan"] = []

        return plan_data

    # ----------------------------
    # Summarizer: returns {"summary": "..."}
    # ----------------------------
    def summarize_results(
        self,
        user_query: str,
        plan: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Summarize final answer from user query + executed plan + MCP results.
        """
        plan_str = json.dumps(plan, indent=2, default=str)
        results_str = json.dumps(results, indent=2, default=str)

        system_prompt = """
You are a senior travel assistant for an airline.
Given:
  - the original user question,
  - the tool-call plan,
  - and the raw tool results,
produce the best possible answer.

Guidelines:
- Explain clearly and concisely.
- Use domain knowledge about flights (delays, airports, etc.).
- If data is missing or ambiguous, state assumptions explicitly.
- If you used any statistics (e.g., summarize_delays), interpret them in plain language.
"""

        user_context = f"""
User question:
{user_query}

Plan executed:
{plan_str}

Tool results:
{results_str}
"""

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]

        resp = self._openai_chat(messages, temperature=0.3)
        msg = resp["choices"][0]["message"]
        summary_text = msg.get("content", "") or ""

        return {"summary": summary_text}


# --------------------------------
# Optional: simple CLI for debugging
# --------------------------------

async def _cli_loop() -> None:
    client = FlightOpsMCPClient()
    await client.connect()
    print("> FlightOps debug CLI. Type 'quit' to exit.\n")

    loop = asyncio.get_event_loop()

    while True:
        user_query = input("you> ").strip()
        if user_query.lower() in ("quit", "exit"):
            print("bye.")
            break

        # Plan
        plan_data = await loop.run_in_executor(None, client.run_chat, user_query)
        plan = plan_data.get("plan", [])
        print("plan>", json.dumps(plan, indent=2, default=str))

        # Execute
        results: List[Dict[str, Any]] = []
        for step in plan:
            tool = step.get("tool")
            args = step.get("arguments", {}) or {}
            print(f"calling tool {tool} with args {args} …")
            try:
                res = await client.invoke_tool(tool, args)
            except Exception as e:
                res = {"error": str(e)}
            results.append({tool: res})

        # Summarize
        summary = await loop.run_in_executor(
            None, client.summarize_results, user_query, plan, results
        )
        print("bot>", summary.get("summary", ""))


if __name__ == "__main__":
    asyncio.run(_cli_loop())
