from langchain_core.documents import Document
from langchain_tavily import TavilySearch


# Web search

web_search_tool = TavilySearch()


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    messages = state['messages']
    question = messages[-1].content

    docs = web_search_tool.invoke({"query": question})
    results = docs["results"]
    print(f"web search documents============================================: {results}")
    web_results = "\n".join([d["content"] for d in results])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "messages": messages}
