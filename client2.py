import os
import json
import logging
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tool_registry import TOOLS

# Load environment variables
load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")

# Azure OpenAI configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

if not AZURE_OPENAI_KEY:
    raise RuntimeError("❌ AZURE_OPENAI_KEY not set in environment")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("FlightOps.MCPClient")

# Initialize Azure OpenAI client
client_azure = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


def _build_tool_prompt() -> str:
    """Convert TOOLS dict into compact text to feed the LLM."""
    lines = []
    for name, meta in TOOLS.items():
        arg_str = ", ".join(meta["args"])
        lines.append(f"- {name}({arg_str}): {meta['desc']}")
    return "\n".join(lines)


SYSTEM_PROMPT_PLAN = f"""
You are an assistant that converts user questions into MCP tool calls.

Available tools:
{_build_tool_prompt()}

### Tool selection logic

1. **Use `run_aggregated_query`** when the user asks for:
   - counts, numbers, totals, sums, averages, minimums, or maximums
   - examples: "how many flights", "number of passengers", "average delay", "max flight time", "total fuel"
   - In such cases:
     - set `"query_type"` to one of ["count", "sum", "average", "min", "max"]
     - set `"field"` to the appropriate MongoDB path (e.g. "flightLegState.operation.fuel.inBlock")
     - if the user gives a condition (e.g. "where delay > 30"), include it as `"filter_json"`
     - optionally include `"start_date"` and `"end_date"` for time ranges

     Example:
     {{
       "plan": [
         {{
           "tool": "run_aggregated_query",
           "arguments": {{
             "query_type": "count",
             "field": "flightLegState.operation.fuel.inBlock",
             "filter_json": "{{ 'flightLegState.operation.fuel.inBlock': {{ '$gt': 100 }} }}"
           }}
         }}
       ]
     }}

2. **Use `raw_mongodb_query`** for:
   - retrieving lists of flights, filtered data, or detailed fields
   - when the question asks to "show", "list", "find", or "get" specific flight data
   - supports `"projection"` to reduce payload (LLM decides what to include)

     Example:
     {{
       "plan": [
         {{
           "tool": "raw_mongodb_query",
           "arguments": {{
             "query_json": "{{ 'flightLegState.startStation': 'DEL', 'flightLegState.endStation': 'BOM' }}",
             "projection": "{{ 'flightLegState.flightNumber': 1, 'flightLegState.startStation': 1, 'flightLegState.endStation': 1, '_id': 0 }}",
             "limit": 10
           }}
         }}
       ]
     }}
    
3. **Use existing tools** (like get_flight_basic_info, get_delay_summary, etc.) for single-flight queries (where a flight number and date are specified).

--- (omitted rest for brevity in the prompt block; unchanged)
"""

SYSTEM_PROMPT_SUMMARIZE = """
You are an assistant that summarizes tool outputs into a concise, readable answer.
Be factual, short, bullet points format and helpful. 
Give only bullet points and never use *(asteriks)
"""


class FlightOpsMCPClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or MCP_SERVER_URL).rstrip("/")
        self.session: ClientSession = None
        self._client_context = None

    
    async def connect(self):
        try:
            logger.info(f"Connecting to MCP server at {self.base_url}")
            self._client_context = streamablehttp_client(self.base_url)
            read_stream, write_stream, _ = await self._client_context.__aenter__()
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
            logger.info("✅ Connected to MCP server successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def disconnect(self):
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    # -------------------- AZURE OPENAI WRAPPER -------------------------
    def _call_azure_openai(self, messages: list, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        try:
            completion = client_azure.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            return json.dumps({"error": str(e)})

    # -------------------- MCP TOOL CALLS -------------------------
    async def list_tools(self) -> dict:
        try:
            if not self.session:
                await self.connect()
            tools_list = await self.session.list_tools()
            tools_dict = {tool.name: {"description": tool.description, "inputSchema": tool.inputSchema} for tool in tools_list.tools}
            return {"tools": tools_dict}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return {"error": str(e)}

    async def invoke_tool(self, tool_name: str, args: dict) -> dict:
        try:
            if not self.session:
                await self.connect()
            logger.info(f"Calling tool: {tool_name} with args: {args}")
            result = await self.session.call_tool(tool_name, args)  

            if result.content:
                content_items = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        try:
                            content_items.append(json.loads(item.text))
                        except json.JSONDecodeError:
                            content_items.append(item.text)
                if len(content_items) == 1:
                    return content_items[0]
                return {"results": content_items}

            return {"error": "No content in response"}
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            return {"error": str(e)}

    # -------------------- LLM PLANNING & SUMMARIZATION -------------------------
    def plan_tools(self, user_query: str) -> dict:
        """
        Ask the LLM to produce a valid JSON plan for which MCP tools to call.
        Cleans out Markdown-style fences (```json ... ```), which some models add.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PLAN},
            {"role": "user", "content": user_query},
        ]

        content = self._call_azure_openai(messages, temperature=0.1)
        if not content:
            logger.warning("⚠️ LLM returned empty response during plan generation.")
            return {"plan": []}
     
        cleaned = content.strip()
        if cleaned.startswith("```"):
            
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            
            cleaned = cleaned.replace("```", "").strip()

        
        if cleaned != content:
            logger.debug(f"🔍 Cleaned LLM plan output:\n{cleaned}")

       
        try:
            plan = json.loads(cleaned)
            if isinstance(plan, dict) and "plan" in plan:
                return plan
            else:
                logger.warning("⚠️ LLM output did not contain 'plan' key.")
                return {"plan": []}
        except json.JSONDecodeError:
            logger.warning(f"❌ Could not parse LLM plan output after cleaning:\n{cleaned}")
            return {"plan": []}


    def summarize_results(self, user_query: str, plan: list, results: list) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": f"Question:\n{user_query}"},
            {"role": "assistant", "content": f"Plan:\n{json.dumps(plan, indent=2)}"},
            {"role": "assistant", "content": f"Results:\n{json.dumps(results, indent=2)}"},
        ]
        summary = self._call_azure_openai(messages, temperature=0.3)
        return {"summary": summary}

   
    async def run_query(self, user_query: str) -> dict:
        """
        Full flow:
        1. LLM plans which tools to call (including possible MongoDB query).
        2. Execute tools sequentially via MCP.
        3. Summarize results using LLM.
        """
        try:
            logger.info(f"User query: {user_query}")
            plan_data = self.plan_tools(user_query)
            plan = plan_data.get("plan", [])
            if not plan:
                return {"error": "LLM did not produce a valid tool plan."}

            results = []
            for step in plan:
                tool = step.get("tool")
                args = step.get("arguments", {}) or {}

                # Clean up bad args
                args = {k: v for k, v in args.items() if v and str(v).strip().lower() != "unknown"}

                # Safety for MongoDB query
                if tool == "raw_mongodb_query":
                    query_json = args.get("query_json", "")
                    if not query_json:
                        results.append({"raw_mongodb_query": {"error": "Empty query_json"}})
                        continue
                    # Enforce safe default limit
                    args["limit"] = int(args.get("limit", 50))
                    logger.info(f"Executing raw MongoDB query: {query_json}")

                resp = await self.invoke_tool(tool, args)
                results.append({tool: resp})

            summary = self.summarize_results(user_query, plan, results)
            return {"plan": plan, "results": results, "summary": summary}
        except Exception as e:
            logger.error(f"Error in run_query: {e}")
            return {"error": str(e)}
