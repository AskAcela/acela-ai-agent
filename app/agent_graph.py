from typing import Annotated, Literal

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
from app.agent.query_rewriter import query_rewriter, route_after_rewrite


class GraphState(TypedDict):
    """
    Shared state for all mode graphs.

    Attributes:
        mode: "ask", "idea", or "explore"
        messages: conversation history
        documents: retrieved / searched documents
        generation: final LLM response
        total_tokens: accumulated token usage
        grading_generation: raw router LLM output (used by route_question)
        hallucination_grade: grade decision from hallucination grader
        web_search_needed: set by grade_documents when any doc is irrelevant
    """

    mode: Literal["ask", "idea", "explore"]
    messages: Annotated[list, add_messages]
    documents: list
    generation: str
    total_tokens: int
    grading_generation: str
    hallucination_grade: str
    web_search_needed: bool

def route_graph(state: GraphState):
    """
    Route to the appropriate graph based on the mode.

    Args:
        state (GraphState): The current graph state

    Returns:
        StateGraph: The next graph to invoke
    """
    mode = state["mode"]
    return mode

# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def _create_graph(mode: Literal["ask", "idea", "explore"]):
    generate_node = {"ask": generate_ask, "idea": generate_idea, "explore": generate_explore}[mode]

    workflow = StateGraph(GraphState)

    workflow.add_node("question_router", question_router)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "question_router")
    workflow.add_edge("retrieve", "grade_documents")

    if mode == "idea":
        # All queries use the idea prompt.
        # llm_fallback-classified messages skip retrieval and hit generate with empty context.
        # No web search, no hallucination grading, no retry loop.
        workflow.add_conditional_edges(
            "question_router", route_question,
            {"vectorstore": "generate", "llm_fallback": "generate"}, # skip retrieval and hit generate with empty context
        )
        workflow.add_edge("grade_documents", "generate") 
        workflow.add_edge("generate", END)

    else:
        # ask / explore: full pipeline — web search fallback, grading, rewrite-and-retry.
        workflow.add_node("llm_fallback", llm_fallback)
        workflow.add_node("web_search", web_search)
        workflow.add_node("grade_generation", grade_generation)
        workflow.add_node("query_rewriter", query_rewriter)

        workflow.add_conditional_edges(
            "question_router", route_question,
            {"vectorstore": "retrieve", "llm_fallback": "llm_fallback"},
        )
        workflow.add_conditional_edges(
            "grade_documents", decide_to_web_search,
            {"web_search": "web_search", "generate": "generate"},
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("llm_fallback", END)
        workflow.add_edge("generate", "grade_generation")
        workflow.add_conditional_edges(
            "grade_generation", route_generation,
            {"useful": END, "not useful": "query_rewriter", "not supported": "query_rewriter"},
        )
        workflow.add_conditional_edges(
            "query_rewriter", route_after_rewrite,
            {"web_search": "web_search", "generate": "generate"},
        )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Compiled graphs (created at import time)
# ---------------------------------------------------------------------------

ask_graph     = _create_graph("ask")
idea_graph    = _create_graph("idea")
explore_graph = _create_graph("explore")

graph_parent = StateGraph(GraphState)
graph_parent.add_node("ask", ask_graph)
graph_parent.add_node("idea", idea_graph)
graph_parent.add_node("explore", explore_graph)
graph_parent.add_conditional_edges(
    START,
    route_graph,
    {
        "ask": "ask",
        "idea": "idea",
        "explore": "explore",
    },
)

agent_graph = graph_parent.compile()
agent_graph.name = "Acela Agent Graph"