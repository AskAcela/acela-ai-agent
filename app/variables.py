import os
from dotenv import load_dotenv

load_dotenv()

from logger import logger

def validate_environment():
    """Validates that all required environment variables are set and not empty."""
    logger.info("Validating environment variables...")
    
    required_vars = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_API_KEY",
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "PINECONE_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        val = os.environ.get(var)
        if not val or not val.strip():
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise ValueError(
            f"Missing or empty required environment variables: {', '.join(missing_vars)}. "
            f"Please ensure they are set in your .env file."
        )
    logger.info("Environment variables validation successful.")

