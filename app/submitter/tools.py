import httpx
from typing import Optional, Any
from langchain.tools import tool
from pydantic import BaseModel, Field

from upstash_redis import Redis


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "http_request",
        "description": (
            "Make any HTTP request to the Celo Builders API (https://celobuilders.xyz). "
            "Use GET to fetch hackathon data, POST to start auth or ask questions, "
            "and PUT to save or update a submission. "
            "The bearer token is injected automatically — you do not need to pass it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "HTTP method: GET, POST, or PUT",
                },
                "url": {
                    "type": "string",
                    "description": "Full URL, e.g. https://celobuilders.xyz/hackathons",
                },
                "body": {
                    "type": "object",
                    "description": "JSON body for POST or PUT requests. Omit for GET.",
                },
            },
            "required": ["method", "url"],
        },
    },
    {
        "name": "kv_store",
        "description": (
            "Read or write any value to a private session-scoped key-value store. "
            "Use this to persist data across turns: bearer token (key: 'auth_token'), "
            "chosen hackathon (key: 'hackathon_id'), intake fields (key: 'intake.projectName'), etc. "
            "Actions: 'set' to write, 'get' to read, 'delete' to remove, 'list' to see all keys. "
            "Never expose auth_token in a chat message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "get", "delete", "list"],
                    "description": "Action to perform",
                },
                "key": {
                    "type": "string",
                    "description": "The key to read, write, or delete. Not needed for 'list'.",
                },
                "value": {
                    "description": "The value to store. Only needed for 'set'.",
                },
            },
            "required": ["action"],
        },
    },
]

# ---------------------------------------------------------------------------
# KV store
# ---------------------------------------------------------------------------

# Outer key: session_id. Inner key: any string the agent chooses.
_kv_store: Redis = Redis.from_env()


def _get_session_store(session_id: str) -> dict[str, Any]:
    store = _kv_store.get(session_id) or {}
    return store


def kv_get(session_id: str, key: str) -> Any:
    """Read a value from the session store. Used internally by http_request."""
    return _get_session_store(session_id).get(key)


def kv_set(session_id: str, key: str, value: Any) -> None:
    """Write a value to the session store. Used internally by http_request."""
    store = _get_session_store(session_id)
    store[key] = value
    _kv_store.set(session_id, store)

def kv_list(session_id: str) -> dict:
    """List all keys in the session store."""
    return {"keys": list(_get_session_store(session_id).keys())}

def kv_delete(session_id: str, key: str) -> None:
    """Clear all keys in the session store."""
    _kv_store.delete(session_id)

# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

@tool(args_schema=TOOLS[0]["parameters"], description=TOOLS[0]["description"], name_or_callable=TOOLS[0]["name"])
def http_request(
    method: str,
    url: str,
    body: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Make an HTTP request to the Celo Builders API.
    Automatically injects the bearer token from the KV store if present.
    """
    headers = {"Content-Type": "application/json"}

    if session_id:
        token = kv_get(session_id, "auth_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"

    try:
        response = httpx.request(
            method=method.upper(),
            url=url,
            json=body,
            headers=headers,
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        return {
            "error": True,
            "status_code": e.response.status_code,
            "detail": e.response.text,
        }
    except httpx.RequestError as e:
        return {
            "error": True,
            "detail": str(e),
        }


@tool(args_schema=TOOLS[1]["parameters"], description=TOOLS[1]["description"], name_or_callable=TOOLS[1]["name"])
def kv_store(
    action: str,
    key: Optional[str] = None,
    value: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Read or write any value to the session-scoped key-value store.
    The agent uses this to persist anything it needs across turns:
    bearer tokens, collected intake fields, the chosen hackathon ID, etc.
    """
    if action == "set":
        if key is None:
            return {"error": "key is required for set"}
        kv_set(session_id, key, value)
        return {"ok": True, "key": key}

    elif action == "get":
        if key is None:
            return {"error": "key is required for get"}
        return {"key": key, "value": kv_get(session_id, key)}

    elif action == "list":
        return kv_list(session_id)

    elif action == "delete":
        if key is None:
            return {"error": "key is required for delete"}
        kv_delete(session_id, key)
        return {"ok": True, "key": key}

    else:
        return {"error": f"Unknown action '{action}'. Use set, get, delete, or list."}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

tools = [http_request, kv_store]