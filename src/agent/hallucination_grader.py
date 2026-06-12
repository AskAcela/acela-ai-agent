
from langchain_core.utils.pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from agent import llm
# Hallucination grader
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Preamble
halucination_grader_preamble = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# LLM with function call
structured_hallucination_grader_llm = llm.with_structured_output(
    GradeHallucinations
)

# Prompt
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", halucination_grader_preamble),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_hallucination_grader_llm

# Answer Grader
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# Preamble
answer_grader_preamble = """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

# LLM with function call
structured_answer_grader_llm = llm.with_structured_output(GradeAnswer)

# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_preamble),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_answer_grader_llm


from logger import logger


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    logger.info("Decision Node: Grade generation vs documents and question")
    question = state["messages"][-1].content
    documents = state["documents"]
    generation = state["generation"]

    logger.info("Checking for hallucinations (groundedness)...")
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        logger.info("Decision: Generation is grounded in documents (no hallucination).")
        logger.info("Grading generation vs original question...")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            logger.info("Decision: Generation addresses the question. Route to END (useful).")
            return "useful"
        else:
            logger.info("Decision: Generation does not address the question. Route to web_search (not useful).")
            return "not useful"
    else:
        logger.info("Decision: Generation is NOT grounded in documents. Route to generate (not supported).")
        return "not supported"