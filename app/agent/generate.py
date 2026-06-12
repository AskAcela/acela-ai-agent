from app.agent import llm
from app.logger import logger

from langchain_core.prompts import ChatPromptTemplate


# Preamble 
generate_preamble = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""


# Prompt
generate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_preamble),
            ("placeholder", "{messages}"),
            ("human", "Context:\n{documents}"),
        ]
    )



# Chain
rag_chain = generate_prompt | llm

def generate(state):
    """
    Generate answer using the vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logger.info("Node: Generate answer using RAG")
    messages = state["messages"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]

    logger.info(f"Invoking RAG chain with {len(documents)} context document(s)...")
    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "messages": messages})
    logger.info("Answer generated successfully.")
    logger.debug(f"Generation output: {generation}")
    return {"documents": documents, "messages": messages, "generation": generation}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    logger.info("Decision Node: Decide to generate or web search")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info("Decision: No relevant documents found. Routing to web_search.")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        logger.info("Decision: Relevant documents found. Routing to generate.")
        return "generate"

