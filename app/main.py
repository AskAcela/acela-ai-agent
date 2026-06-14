from app.logger import logger
from app.variables import validate_environment
from app.utils import convert_messages
from app.agent_graph import createAgentGraph

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

logger.info("Initializing system...")

# Validate environment variables
validate_environment()

server = FastAPI()
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://askacela.xyz"
]
server.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allowed domains
    allow_credentials=True,         # Allow cookies and auth headers
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all request headers
)

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
