from langchain_core.utils.pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.agent import llm
from app.logger import logger


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


retrieval_grader_preamble = """You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# include_raw=True returns {"raw": AIMessage, "parsed": GradeDocuments} so we can track token usage
structured_llm_grader = llm.with_structured_output(GradeDocuments, include_raw=True)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_preamble),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


def grade_documents(state):
    """
    Grade each retrieved document for relevance. Sets web_search_needed=True if any
    document is irrelevant or if no relevant documents were found.
    """
    logger.info("Node: Grade document relevance")
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]
    total_tokens = state["total_tokens"]

    logger.info(f"Grading {len(documents)} document(s) for: '{question}'")

    filtered_docs = []
    has_irrelevant = False

    for i, d in enumerate(documents):
        result = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        total_tokens += (result["raw"].usage_metadata or {}).get("total_tokens", 0)

        if result["parsed"].binary_score == "yes":
            logger.info(f"Document {i + 1}: RELEVANT")
            filtered_docs.append(d)
        else:
            logger.info(f"Document {i + 1}: NOT RELEVANT")
            has_irrelevant = True

    web_search_needed = has_irrelevant or len(filtered_docs) == 0

    logger.info(
        f"Grading complete: {len(filtered_docs)}/{len(documents)} relevant. "
        f"Web search needed: {web_search_needed}"
    )
    return {
        "documents": filtered_docs,
        "messages": messages,
        "total_tokens": total_tokens,
        "web_search_needed": web_search_needed,
    }


def decide_to_web_search(state):
    """Route to web_search if grading flagged irrelevant docs, otherwise generate."""
    if state.get("web_search_needed", False):
        logger.info("Decision: Web search needed — routing to web_search.")
        return "web_search"
    logger.info("Decision: All documents relevant — routing to generate.")
    return "generate"
