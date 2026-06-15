from app.logger import logger
from app.variables import validate_environment
from app.utils import convert_messages
from app.agent_graph import ask_graph, idea_graph, explore_graph
from app.agent.title import generate_title

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal

logger.info("Initializing system...")

validate_environment()

server = FastAPI()
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://askacela.xyz"
]
server.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI server created. Agent graphs loaded.")

ChatMode = Literal["ask", "idea", "explore"]

_graphs = {
    "ask": ask_graph,
    "idea": idea_graph,
    "explore": explore_graph,
}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class TitleRequest(BaseModel):
    message: str


@server.get("/")
def root():
    return {"message": "hello world!"}


@server.get("/health")
def health():
    return {"status": "ok"}


@server.post("/title")
def title(req: TitleRequest):
    logger.info("POST /title")
    return {"title": generate_title(req.message)}


@server.post("/chat")
def chat(
    req: ChatRequest,
    mode: ChatMode = Query(default="ask"),
):
    logger.info(f"POST /chat — mode={mode}, messages={len(req.messages)}")

    history = convert_messages(req.messages)
    graph = _graphs[mode]

    result = graph.invoke(
        {
            "messages": history,
            "documents": [],
            "generation": "",
            "total_tokens": 0,
            "web_search_needed": False,
        }
    )

    logger.info("Agent graph execution completed.")
    return {
        "message": result["generation"].content,
        "usage": {
            "total_tokens": result["total_tokens"],
        },
    }
