from langchain_core.utils.pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.agent import llm

### Retrieval Grader


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Prompt
retrival_grader_preamble = """You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrival_grader_preamble),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


from app.logger import logger


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    logger.info("Node: Grade documents relevance to question")
    messages = state["messages"]
    question = messages[-1].content
    documents = state["documents"]
    total_tokens = state["total_tokens"]

    logger.info(f"Grading relevance of {len(documents)} retrieved document(s) for query: {question}")
    # Score each doc
    filtered_docs = []
    for i, d in enumerate(documents):
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        total_tokens += score.usage_metadata["total_tokens"]
        grade = score.binary_score
        if grade == "yes":
            logger.info(f"Document {i+1}: RELEVANT")
            filtered_docs.append(d)
        else:
            logger.info(f"Document {i+1}: NOT RELEVANT")
            continue
    logger.info(f"Filtering completed. {len(filtered_docs)}/{len(documents)} documents retained as relevant.")
    return {"documents": filtered_docs, "messages": messages, "total_tokens": total_tokens}

