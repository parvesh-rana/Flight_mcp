import os
import json
import asyncio
from typing import Any, Dict, List

import openai
from mcp.client.async_client import AsyncClient  # from python-sdk
from mcp.client.transports.http import HTTPTransport

# Configuration
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-o")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")

openai.api_key = OPENAI_API_KEY

# Safety guard‐words
BAD_KEYWORDS = [
    "hack", "exploit", "injection", "$where", "drop database",
    "weapon", "bomb", "kill", "passport", "credit card", "pnr",
    "ticket number", "password", "login details"
]

def cheap_safety_check(user_query: str) -> bool:
    lower = user_query.lower()
    return not any(kw in lower for kw in BAD_KEYWORDS)

async def run_chat():
    async with AsyncClient(transport=HTTPTransport(MCP_SERVER_URL)) as session:
        tools_list = await session.list_tools()
        # build OpenAI tool definitions
        tools_for_openai = []
        for t in tools_list.tools:
            tools_for_openai.append({
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type":"object","properties":{}}
            })

        print("> You can ask about flights … (type quit to exit)")
        while True:
            user_query = input("you> ").strip()
            if user_query.lower() in ("quit","exit"):
                print("bye.")
                break
            if not cheap_safety_check(user_query):
                print("bot> No, I can’t help with that.")
                continue

            # First message with system and user
            system_prompt = (
                "You are a friendly travel assistant. Use tools when needed to look up flight data."
                "If you call a tool, then above your answer include the tool call JSON as allowed by MCP."
            )
            messages = [
                {"role":"system", "content": system_prompt},
                {"role":"user",   "content": user_query}
            ]

            # Call OpenAI with function‐calling support
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                functions=tools_for_openai,
                function_call="auto"
            )
            msg = response.choices[0].message

            if msg.get("function_call"):
                func_name = msg["function_call"]["name"]
                args = json.loads(msg["function_call"]["arguments"] or "{}")
                # call through MCP
                tool_resp = await session.call_tool(name=func_name, arguments=args)
                result_content = tool_resp.content
                # add tool result to messages
                messages.append({"role":"assistant","content":None,
                                 "tool_call":{"name":func_name,"arguments":args}})
                messages.append({"role":"tool","name":func_name,"content":json.dumps(result_content)})

                # second pass: let LLM craft answer
                second = openai.ChatCompletion.create(
                    model=OPENAI_API_KEY,
                    messages=messages
                )
                final = second.choices[0].message.content
                print("bot> " + final)
            else:
                # direct answer, no tool
                print("bot> " + (msg["content"] or ""))

if __name__ == "__main__":
    asyncio.run(run_chat())
