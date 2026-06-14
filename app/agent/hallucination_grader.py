
from langchain_core.utils.pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agent import llm

# Hallucination grader
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Preamble
hallucination_grader_preamble = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# LLM with function call
structured_hallucination_grader_llm = llm.with_structured_output(
    GradeHallucinations
)

# Prompt
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_preamble),
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


from app.logger import logger


def grade_generation(state):
    logger.info("Node: Grade generation vs documents and question")
    question = state["messages"][-1].content
    documents = state["documents"]
    generation = state["generation"]
    total_tokens = state["total_tokens"]

    logger.info("Checking for hallucinations (groundedness)...")
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    total_tokens += score.usage_metadata["total_tokens"]
    grade = score.binary_score

    if grade == "yes":
        logger.info("Generation is grounded in documents. Checking answer quality...")
        score = answer_grader.invoke({"question": question, "generation": generation})
        total_tokens += score.usage_metadata["total_tokens"]
        decision = "useful" if score.binary_score == "yes" else "not useful"
    else:
        decision = "not supported"

    return {
        "hallucination_grade": decision,
        "total_tokens": total_tokens,
    }


def route_generation(state):
    decision = state["hallucination_grade"]
    if decision == "useful":
        logger.info("Routing decision: Generation addresses the question. Route to END.")
        return "useful"
    elif decision == "not useful":
        logger.info("Routing decision: Generation does not address the question. Route to web_search.")
        return "not useful"
    else:
        logger.info("Routing decision: Generation is NOT grounded in documents. Route to generate.")
        return "not supported"
