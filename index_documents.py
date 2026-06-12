from app.logger import logger
from app.vectorestore import index
from dotenv import load_dotenv

logger.info("Initializing document indexing process...")

load_dotenv()

index()

logger.info("Document indexing process finished.")
