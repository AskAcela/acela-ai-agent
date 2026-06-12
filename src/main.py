from variables import validate_environment
from utils import convert_messages
from app import createAgentGraph

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Validate environment variables
validate_environment()

server = FastAPI()
agent_graph = createAgentGraph()

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@server.get("/")
def root():
    return {"message": "hello world"}


@server.get("/health")
def health():
    return {"status": "ok"}


@server.post("/chat")
def chat(req: ChatRequest):
    history = convert_messages(
        req.messages
    )

    result = agent_graph.invoke(
        {
            "messages": history,
            "documents": [],
            "generation": ""
        }
    )           
    
    return result