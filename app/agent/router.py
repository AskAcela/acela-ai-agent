from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agent import llm

# Router Tools

# ----- Router

router_preamble = """You are a query router.

Route to VECTORSTORE for any question that requires specific Celo knowledge — tokens, contracts, \
dApps, developer tooling, Celo protocols, general programming concepts, EVM/blockchain fundamentals that don't need Celo-specific context, or anything where the documentation would help.

Route to LLM FALLBACK for everything else: greetings, conversational messages, or simple questions the model can answer from general knowledge alone.

Return only one word: "vectorstore" or "llm_fallback"."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_preamble),
        ("placeholder", "{messages}"),
    ]
)

question_router_llm = route_prompt | llm

from app.logger import logger

def question_router(state):
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

    result = question_router_llm.invoke({"messages": messages})
    total_tokens += result.usage_metadata["total_tokens"]
    
    return {
        "grading_generation": result,
        "total_tokens": total_tokens,
    }


def route_question(state):
    datasource = StrOutputParser().invoke(state["grading_generation"])

    if "vectorstore" in datasource:
        logger.info("Routing decision: VECTORSTORE")
        return "vectorstore"
    else:
        logger.info("Routing decision: LLM FALLBACK")
        return "llm_fallback"
