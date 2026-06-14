from langchain_core.prompts import ChatPromptTemplate

from app.agent import llm

### LLM fallback

# Preamble
llm_fallback_preamble = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""


# Prompt
llm_fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", llm_fallback_preamble),
        ("placeholder", "{messages}"),
    ])


from app.logger import logger

# Chain
llm_chain = llm_fallback_prompt | llm

def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logger.info("Node: LLM Fallback (No Vector Store)")
    messages = state["messages"]
    total_tokens = state["total_tokens"]
    logger.info(f"Invoking LLM fallback chain for query: {messages[-1].content}")
    generation = llm_chain.invoke({"messages": messages})
    total_tokens += generation.usage_metadata["total_tokens"]
    logger.info("Fallback generation completed.")
    logger.debug(f"Fallback generation: {generation}")
    return {"messages": messages, "generation": generation, "total_tokens": total_tokens}

