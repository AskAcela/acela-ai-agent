from logger import logger
from variables import validate_environment
from utils import convert_messages
from app import createAgentGraph

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

logger.info("Initializing system...")

# Validate environment variables
validate_environment()

server = FastAPI()
logger.info("FastAPI server created. Loading agent graph...")
agent_graph = createAgentGraph()
logger.info("Agent graph loaded successfully.")

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@server.get("/")
def root():
    logger.info("GET / endpoint called.")
    return {"message": "hello world"}


@server.get("/health")
def health():
    logger.info("GET /health endpoint called.")
    return {"status": "ok"}


@server.post("/chat")
def chat(req: ChatRequest):
    logger.info("POST /chat endpoint called.")
    logger.info(f"Received message history with {len(req.messages)} messages.")
    
    history = convert_messages(
        req.messages
    )

    logger.info("Invoking agent graph...")
    result = agent_graph.invoke(
        {
            "messages": history,
            "documents": [],
            "generation": ""
        }
    )           
    logger.info("Agent graph execution completed.")
    logger.debug(f"Response: {result}")
    
    return result["generation"]