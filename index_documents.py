from logger import logger
from vectorestore import index
from dotenv import load_dotenv

logger.info("Initializing document indexing process...")

load_dotenv()

index()

logger.info("Document indexing process finished.")