import os
import hashlib
import requests
from pinecone import ServerlessSpec
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from app.logger import logger

# Repositories/directories to index. Each entry pulls every matching file under
# the listed directories of a GitHub repo. Set "directories" to [""] to index
# the whole repo. "extensions" filters by file suffix (None = all files).
INDEX_SOURCES = [
    {
        "repo": "celo-org/celopedia-skills",
        "branch": "main",
        "directories": ["skills"],
        "extensions": (".md", ".mdx", ".txt", ".rst"),
    },
]


def list_source_urls():
    """Resolve INDEX_SOURCES into a flat list of raw file URLs."""
    urls = []
    for cfg in INDEX_SOURCES:
        repo = cfg["repo"]
        branch = cfg.get("branch", "main")
        directories = cfg.get("directories") or [""]
        extensions = cfg.get("extensions")

        tree_url = (
            f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        )
        response = requests.get(tree_url, timeout=30)
        response.raise_for_status()
        tree = response.json().get("tree", [])

        for node in tree:
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            in_dir = any(
                path.startswith(f"{d.rstrip('/')}/") if d else True
                for d in directories
            )
            if not in_dir:
                continue
            if extensions and not path.endswith(tuple(extensions)):
                continue
            urls.append(
                f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
            )
    return urls


vectorstore = None
retriever = None
pinecone_index = None


def init_vectore_store():
    global vectorstore, retriever, pinecone_index
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

    pinecone_index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=google_embedding)
    retriever = vectorstore.as_retriever()
    logger.info("Pinecone vector store initialized successfully.")
    return vectorstore, retriever


def _source_prefix(source):
    """Stable ID prefix derived from a source URL."""
    return hashlib.sha256(source.encode()).hexdigest()[:16] + "#"


def _delete_existing(source):
    """Delete all vectors previously indexed for a source (clean update)."""
    prefix = _source_prefix(source)
    stale_ids = []
    for id_batch in pinecone_index.list(prefix=prefix):
        stale_ids.extend(id_batch)
    if stale_ids:
        pinecone_index.delete(ids=stale_ids)
        logger.info(f"Deleted {len(stale_ids)} stale chunks for {source}")


def index():
    logger.info("Starting indexing process...")
    init_vectore_store()

    sources = list_source_urls()
    logger.info(f"Indexing {len(sources)} sources")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    total_chunks = 0

    for source in sources:
        try:
            docs = WebBaseLoader(source).load()
        except Exception as exc:
            logger.error(f"Failed to load {source}: {exc}")
            continue

        splits = splitter.split_documents(docs)
        if not splits:
            logger.warning(f"No content found for {source}, skipping")
            continue

        prefix = _source_prefix(source)
        ids = [f"{prefix}{i}" for i in range(len(splits))]

        # Remove any previously indexed chunks for this source, then upsert.
        # Deterministic IDs make re-indexing unchanged files idempotent, and the
        # prefix delete clears orphaned chunks when a file shrinks or changes.
        _delete_existing(source)
        vectorstore.add_documents(documents=splits, ids=ids)
        total_chunks += len(splits)
        logger.info(f"Indexed {len(splits)} chunks from {source}")

    logger.info(f"Indexing completed successfully. {total_chunks} chunks total.")

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
