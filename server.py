import os
import logging
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flight-mcp-server")

# Mongo config
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "flights")
COLL_NAME = os.environ.get("MONGO_COLLECTION", "flights")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

# MCP server
mcp = FastMCP("FlightDataServer", json_response=True)

# Helper: sanitize PII
PII_KEYS = {"email","phone","pnr","ticketNumber","passport","ssn","aadhaar"}

def _sanitize_doc(doc: Dict[str,Any]) -> Dict[str,Any]:
    if not doc: 
        return {}
    out = {k:v for k,v in doc.items() if k not in PII_KEYS and k != "_id"}
    return out

def _sanitize_docs(docs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    return [_sanitize_doc(d) for d in docs]

# === Tools ===

@mcp.tool()
def search_flights(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    flight_number: Optional[str] = None,
    date_of_origin: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str,Any]]:
    """
    Search flights by origin, destination, flight_number, and/or date_of_origin.
    Returns up to 'limit' results.
    """
    try:
        query: Dict[str,Any] = {}
        if origin:
            query["flightLegState.startStation"] = origin.upper()
        if destination:
            query["flightLegState.endStation"] = destination.upper()
        if flight_number:
            query["flightLegState.flightNumber"] = flight_number.upper()
        if date_of_origin:
            query["dateOfOrigin"] = date_of_origin

        docs = list(coll.find(query).limit(limit))
        return _sanitize_docs(docs)
    except PyMongoError as e:
        logger.error(f"Mongo error in search_flights: {e}")
        raise RuntimeError("Internal database error")

@mcp.tool()
def get_flight_info(
    flight_number: str,
    date_of_origin: Optional[str] = None
) -> Dict[str,Any]:
    """
    Get a single flight's detailed info by flight_number (and optionally date_of_origin).
    """
    try:
        query: Dict[str,Any] = {"flightLegState.flightNumber": flight_number.upper()}
        if date_of_origin:
            query["dateOfOrigin"] = date_of_origin
        doc = coll.find_one(query)
        if not doc:
            return {}
        return _sanitize_doc(doc)
    except PyMongoError as e:
        logger.error(f"Mongo error in get_fflight_info: {e}")
        raise RuntimeError("Internal database error")

@mcp.tool()
def summarize_delays(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None
) -> Dict[str,Any]:
    """
    Return delay statistics for flights filtered by origin/destination and optional date range.
    Returns keys: count, avg_delay_minutes, max_delay_minutes, min_delay_minutes.
    """
    try:
        match: Dict[str,Any] = {}
        if origin:
            match["flightLegState.startStation"] = origin.upper()
        if destination:
            match["flightLegState.endStation"] = destination.upper()
        if since_date or until_date:
            date_range: Dict[str,Any] = {}
            if since_date:
                date_range["$gte"] = since_date
            if until_date:
                date_range["$lte"] = until_date
            match["dateOfOrigin"] = date_range

        pipeline = [
            {"$match": match},
            {"$addFields": {"delay": {"$ifNull": ["$flightLegState.delays.total", 0]}}},
            {"$match": {"delay": {"$ne": None}}},
            {
                "$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "avg_delay": {"$avg": "$delay"},
                    "max_delay": {"$max": "$delay"},
                    "min_delay": {"$min": "$delay"},
                }
            }
        ]
        res = list(coll.aggregate(pipeline))
        if not res:
            return {"count":0, "avg_delay_minutes":None, "max_delay_minutes":None, "min_delay_minutes":None}
        r = res[0]
        return {
            "count": int(r["count"]),
            "avg_delay_minutes": float(r["avg_delay"]),
            "max_delay_minutes": float(r["max_delay"]),
            "min_delay_minutes": float(r["min_delay"])
        }
    except PyMongoError as e:
        logger.error(f"Mongo error in summarize_delays: {e}")
        raise RuntimeError("Internal database error")

# === Resources (optional) ===

@mcp.resource("resource://flight-schema")
def flight_schema() -> str:
    """
    Returns a description of the flight DB schema for context.
    """
    return """
    flightLegState: { flightNumber: string, startStation: string, endStation: string, scheduledStartTime: ISOString, delays: { total: number } }
    dateOfOrigin: string (YYYY-MM-DD)
    origin: string, destination: string
    ...
    """

# === Prompts (optional) ===

@mcp.prompt()
def casual_instruction(style: str = "casual") -> str:
    """
    A prompt template for instructing the assistant in a given style.
    """
    if style == "casual":
        return "You are a friendly travel assistant. Speak casually and clearly."
    elif style == "formal":
        return "You are a travel assistant. Speak formally and clearly."
    else:
        return "You are a travel assistant."

# === Startup ===

if __name__ == "__main__":
    logger.info("Starting MCP FlightDataServer …")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.environ.get("PORT",8000)))
