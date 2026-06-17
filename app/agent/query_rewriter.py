from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agent import llm
from app.logger import logger

REWRITER_PREAMBLE = """You are a query rewriter specializing in the Celo blockchain ecosystem. \
Rephrase the user's question to make it clearer, more specific, and better suited for \
retrieval or web search. Expand vague terms, make implicit Celo context explicit, \
and sharpen the intent. Return ONLY the rewritten question — no explanation, no preamble."""

_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITER_PREAMBLE),
    ("human", "Original question: {question}\n\nRewrite it:"),
])


def query_rewriter(state):
    logger.info("Node: Query rewriter")
    messages = state["messages"]
    total_tokens = state.get("total_tokens", 0)

    question = messages[-1].content
    logger.info(f"Rewriting query: '{question}'")

    response = llm.invoke(_rewriter_prompt.format_messages(question=question))
    rewritten = response.content.strip()
    total_tokens += (response.usage_metadata or {}).get("total_tokens", 0)

    logger.info(f"Rewritten query: '{rewritten}'")
    return {
        "messages": [HumanMessage(content=rewritten)],
        "total_tokens": total_tokens,
    }


def route_after_rewrite(state):
    """Route to web_search or generate after the query has been rewritten."""
    decision = state["hallucination_grade"]
    if decision == "not useful":
        logger.info("Post-rewrite route: not useful → web_search")
        return "web_search"
    logger.info("Post-rewrite route: not supported → generate")
    return "generate"
