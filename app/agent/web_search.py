from langchain_core.documents import Document
from langchain_tavily import TavilySearch


from app.logger import logger

# Web search

web_search_tool = TavilySearch(topic="finance", max_results=3, time_range="year")


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    logger.info("Node: Web search")
    messages = state['messages']
    question = messages[-1].content
    logger.info(f"Querying web search (Tavily) for: {question}")

    docs = web_search_tool.invoke({"query": question})
    results = docs["results"]
    logger.info(f"Retrieved {len(results)} search results from Tavily.")
    logger.debug(f"Tavily results: {results}")
    web_results = "\n".join([d["content"] for d in results])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "messages": messages}

