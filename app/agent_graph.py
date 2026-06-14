from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict

from app.vectorestore import retrieve
from app.agent.retrival_grader import grade_documents
from app.agent.web_search import web_search
from app.agent.generate import generate
from app.agent.llm_fallback import llm_fallback
from app.agent.router import question_router, route_question
from app.agent.hallucination_grader import grade_generation, route_generation
from app.agent.generate import decide_to_generate
from typing import Annotated
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        messages: conversation messages
        generation: LLM generation
        documents: list of documents
    """

    messages: Annotated[list, add_messages]
    documents: list
    generation: str
    total_tokens: int
    grading_generation: str
    hallucination_grade: str


def createAgentGraph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("generate", generate)  # rag
    workflow.add_node("grade_generation", grade_generation)  # hallucination + answer grading
    workflow.add_node("llm_fallback", llm_fallback)  # llm

    # Build graph
    workflow.add_conditional_edges(
        START,
        question_router,
        route_question,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
            "llm_fallback": "llm_fallback",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        route_generation,
        {
            "not supported": "generate",  # Hallucinations: re-generate
            "not useful": "web_search",  # Fails to answer question: fall-back to web-search
            "useful": END,
        },
    )
    workflow.add_edge("llm_fallback", END)

    # Compile
    app = workflow.compile()

    return app
