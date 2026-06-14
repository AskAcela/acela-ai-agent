from langchain_core import messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command

from app.agent import llm

# Router Tools

# ----- Router

router_preamble = """You are a query router. The vectorstore contains Celo (Ethereum L2) documentation
including platform contracts and developer guides.

Route to VECTORSTORE for questions about Celo, for example tokens, contracts, SDKs, dapps, or developer tooling.
Route to WEB SEARCH for real-time data, recent news, or non-Celo topics that need current info.
Route to LLM FALLBACK for general knowledge questions the LLM can answer without external sources
(e.g. explaining coding concepts, EVM basics, or general blockchain fundamentals).

Return only: "vectorstore", "web_search", or "llm_fallback"."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_preamble),
        ("placeholder", "{messages}"),
    ]
)

question_router = route_prompt | llm

from app.logger import logger

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    logger.info("Decision Node: Route Question")
    messages = state['messages']
    total_tokens = state["total_tokens"]
    question = messages[-1].content
    logger.info(f"Routing query: '{question}'")

    result = question_router.invoke({"messages": messages})
    datasource = StrOutputParser()(result)
    total_tokens += result.usage_metadata["total_tokens"]
    datasource = datasource.strip().lower()

    # Choose datasource
    if "web_search" in datasource:
        logger.info("Routing decision: Route to WEB SEARCH.")
        return Command(goto="web_search", update={
            "total_tokens": total_tokens
        })
    elif "vectorstore" in datasource:
        logger.info("Routing decision: Route to VECTORSTORE (RAG).")
        return Command(goto="vectorstore", update={
            "total_tokens": total_tokens
        })
    else:
        logger.info("Routing decision: Route to LLM FALLBACK.")
        return Command(goto="llm_fallback", update={
            "total_tokens": total_tokens
        })
