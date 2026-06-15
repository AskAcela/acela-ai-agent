from typing import Annotated

from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from app.vectorestore import retrieve
from app.agent.retrieval_grader import grade_documents, decide_to_web_search
from app.agent.web_search import web_search
from app.agent.generate import generate_ask, generate_idea, generate_explore
from app.agent.llm_fallback import llm_fallback
from app.agent.router import question_router, route_question
from app.agent.hallucination_grader import grade_generation, route_generation


class GraphState(TypedDict):
    """
    Shared state for all mode graphs.

    Attributes:
        messages: conversation history
        documents: retrieved / searched documents
        generation: final LLM response
        total_tokens: accumulated token usage
        grading_generation: raw router LLM output (used by route_question)
        hallucination_grade: grade decision from hallucination grader
        web_search_needed: set by grade_documents when any doc is irrelevant
    """

    messages: Annotated[list, add_messages]
    documents: list
    generation: str
    total_tokens: int
    grading_generation: str
    hallucination_grade: str
    web_search_needed: bool


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def _create_graph(generate_node, use_hallucination_grading: bool):
    """
    Build and compile a mode-specific StateGraph.

    All modes share the same topology up to the generate node:
      START → question_router → vectorstore path | llm_fallback

    Vectorstore path:
      retrieve → grade_documents → [web_search if any irrelevant] → generate_node

    Post-generate:
      ask / explore  → grade_generation → route_generation → END / web_search / generate
      idea           → END directly
    """
    workflow = StateGraph(GraphState)

    # --- Shared nodes ---
    workflow.add_node("question_router", question_router)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate_node)
    workflow.add_node("llm_fallback", llm_fallback)

    # --- Entry ---
    workflow.add_edge(START, "question_router")
    workflow.add_conditional_edges(
        "question_router",
        route_question,
        {
            "vectorstore": "retrieve",
            "llm_fallback": "llm_fallback",
        },
    )

    # --- Vectorstore path ---
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_web_search,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")

    # --- Fallback ---
    workflow.add_edge("llm_fallback", END)

    # --- Post-generate ---
    if use_hallucination_grading:
        workflow.add_node("grade_generation", grade_generation)
        workflow.add_edge("generate", "grade_generation")
        workflow.add_conditional_edges(
            "grade_generation",
            route_generation,
            {
                "useful": END,
                "not useful": "web_search",   # answer doesn't address question → web search
                "not supported": "generate",  # hallucination → regenerate
            },
        )
    else:
        workflow.add_edge("generate", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Compiled graphs (created at import time)
# ---------------------------------------------------------------------------

ask_graph = _create_graph(generate_ask, use_hallucination_grading=True)
idea_graph = _create_graph(generate_idea, use_hallucination_grading=False)
explore_graph = _create_graph(generate_explore, use_hallucination_grading=True)
