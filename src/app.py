from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict

from vectorestore import retrieve
from agent.retrival_grader import grade_documents
from agent.web_search import web_search
from agent.generate import generate
from agent.llm_fallback import llm_fallback
from agent.router import route_question
from agent.hallucination_grader import grade_generation_v_documents_and_question
from agent.generate import decide_to_generate
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


def createAgentGraph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("generate", generate)  # rag
    workflow.add_node("llm_fallback", llm_fallback)  # llm

    # Build graph
    workflow.add_conditional_edges(
        START,
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
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
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
