from app.logger import logger
from app.variables import validate_environment
from app.utils import convert_messages
from app.agent_graph import createAgentGraph
from app.submitter.agent import run_agent

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


class ChatSubmitRequest(BaseModel):
    session_id: str
    messages: List[Message]

@server.get("/")
def root():
    logger.info("GET / endpoint called.")
    return {"message": "hello world!"}


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
            "generation": "",
            "total_tokens": 0
        }
    )           
    logger.info("Agent graph execution completed.")
    logger.debug(f"Response: {result}")
    
    return {
        "message": result["generation"].content,
        "usage": {
            "total_tokens": result["total_tokens"],
        }
    }

@server.post("/chat/submit")
def chat_submit(req: ChatSubmitRequest):
    logger.info("POST /chat/submit endpoint called.")
    logger.info(f"Received message history with {len(req.messages)} messages.")
    
    history = convert_messages(
        req.messages
    )

    logger.info("Submitting agent for asynchronous execution...")
    result = run_agent(req.session_id, history)
    logger.info("Agent submitted successfully.")
    
    return result
