from langchain_core import messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent import llm

# Router Tools

# ----- Router

router_preamble = """You are a query router. The vectorstore contains Celo (Ethereum L2) documentation
including platform contracts and developer guides.

Route to VECTORSTORE for questions about Celo's architecture, contracts, or developer tooling.
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

question_router = route_prompt | llm | StrOutputParser()

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    messages = state['messages']
    datasource = question_router.invoke({"messages": messages})

    # Choose datasource
    if datasource == "web_search":
        return "web_search"
    elif datasource == "vectorstore":
        return "vectorstore"
    else:
        return "llm_fallback"
