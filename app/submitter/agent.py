from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from .prompt import SYSTEM_MESSAGE
from .tools import TOOLS, http_request, kv_store
from langchain.tools import tool
from app.logger import logger


# ---------------------------------------------------------------------------
# Session binding
# ---------------------------------------------------------------------------

def _bind_session(session_id: str) -> list:
    """
    Return session-bound versions of both tools.
    The agent never sees the session_id — it is baked in at the server level.
    """

    @tool(args_schema=TOOLS[0]["parameters"], description=TOOLS[0]["description"], name_or_callable=TOOLS[0]["name"])
    def http_request_with_session(method: str, url: str, body=None):
        return http_request(method=method, url=url, body=body, session_id=session_id)
    
    @tool(args_schema=TOOLS[1]["parameters"], description=TOOLS[1]["description"], name_or_callable=TOOLS[1]["name"])
    def kv_store_with_session(action: str, key=None, value=None):
        return kv_store(action=action, key=key, value=value, session_id=session_id)

    return [http_request_with_session, kv_store_with_session]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(session_id: str):
    """
    Build and return an AgentExecutor for the given session.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    tools = _bind_session(session_id)

    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_MESSAGE,
        checkpointer=InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# Execute Turn with History
# ---------------------------------------------------------------------------

def run_agent(session_id: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
    """
    Run the agent for one user turn, passing in a list of past messages.
    Returns a dictionary containing the text output and tool execution records.
    
    :param chat_history: A list of LangChain Message objects, e.g., [HumanMessage(...), AIMessage(...)]
    """
    executor = build_agent(session_id)
    
    # Execute synchronously and block until a final response is ready
    result = executor.invoke({
        "messages": chat_history  # Injected directly into your prompt layout
    })

    # Safely extract tool step outputs to see if an OAuth link was requested
    steps = result.get("intermediate_steps", [])
    open_url = None
    
    for agent_action, tool_output in steps:
        if agent_action.tool == "http_request" and isinstance(tool_output, dict):
            open_url = tool_output.get("signInUrl") or tool_output.get("url")
            if open_url:
                break # Grab the first sign-in URL discovered
    
    logger.info(f"Agent execution completed. Open URL: {open_url}, Total steps: {len(steps)}")
    logger.debug(f"Full agent result: {result}")
    return {
        "output": result.get("output", ""),
        "open_url": open_url,
        "intermediate_steps": steps
    }
