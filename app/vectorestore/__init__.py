import os
from pinecone import ServerlessSpec
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from app.logger import logger

vectorstore = None
retriever =     None
def init_vectore_store():
    global vectorstore, retriever
    if vectorstore:
        return vectorstore, retriever
    logger.info("Initializing Pinecone vector store...")
    google_embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=512)
    # pine cone vectore store
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "acela-test-index"
    if not pc.has_index(index_name):
        logger.info(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )   

    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=google_embedding)
    retriever = vectorstore.as_retriever()
    logger.info("Pinecone vector store initialized successfully.")
    return vectorstore, retriever

def index():
    logger.info("Starting indexing process...")
    vectorstore, _ = init_vectore_store()
    # Build Indexing

    # Load sources
    sources = [
        "https://github.com/celo-org/celopedia-skills/raw/refs/heads/main/skills/celopedia-skill/SKILL.md",
        "https://github.com/celo-org/celopedia-skills/raw/refs/heads/main/skills/celopedia-skill/references/contracts.md",
        "https://github.com/celo-org/celopedia-skills/raw/refs/heads/main/skills/celopedia-skill/references/dev-templates.md",
    ]

    logger.info(f"Loading documents from {len(sources)} sources")

    docs = [doc for source in sources for doc in WebBaseLoader(source).load()]
    logger.info(f"Loaded {len(docs)} documents. Splitting into chunks...")

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    logger.info(f"Split documents into {len(splits)} chunks. Adding to vector store...")
    
    vectorstore.add_documents(documents=splits)
    logger.info("Indexing completed successfully.")

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logger.info("Node: Retrieve documents from vector store")
    _, retriever = init_vectore_store()
    question = state["messages"][-1].content
    logger.info(f"Retrieving documents for query: {question}")

    # Retrieval
    documents = retriever.invoke(question, top_k=3)
    logger.info(f"Retrieved {len(documents)} documents.")
    return {"documents": documents, "messages": state["messages"]}
