from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from app.logger import logger

web_search_tool = TavilySearch(topic="finance", max_results=3, time_range="year")


def web_search(state):
    """
    Run a Tavily web search and append the results to any existing documents in state.
    Existing relevant documents from the vector store are preserved.
    """
    logger.info("Node: Web search")
    messages = state["messages"]
    question = messages[-1].content
    logger.info(f"Querying Tavily for: '{question}'")

    docs = web_search_tool.invoke({"query": question})
    results = docs["results"]
    logger.info(f"Tavily returned {len(results)} result(s).")

    web_content = "\n\n".join([d["content"] for d in results])
    web_doc = Document(page_content=web_content)

    # Merge with any relevant vectorstore docs already in state
    existing = state.get("documents", [])
    if not isinstance(existing, list):
        existing = [existing]

    return {"documents": existing + [web_doc], "messages": messages}
