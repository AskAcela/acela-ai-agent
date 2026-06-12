import os
from pinecone import ServerlessSpec
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

vectorstore = None
retriever =     None
def init_vectore_store():
    global vectorstore, retriever
    if vectorstore:
        return vectorstore
    google_embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=512)
    # pine cone vectore store
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "acela-test-index"
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )   

    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=google_embedding)
    retriever = vectorstore.as_retriever()
    return vectorstore, retriever

def index():
    print("starting indexing============================================")
    vectorstore, _ = init_vectore_store()
    # Build Indexing

    # Load sources
    sources = [
        "https://github.com/celo-org/celopedia-skills/raw/refs/heads/main/skills/celopedia-skill/SKILL.md",
        "https://github.com/celo-org/celopedia-skills/raw/refs/heads/main/skills/celopedia-skill/references/contracts.md",
        "https://github.com/celo-org/celopedia-skills/raw/refs/heads/main/skills/celopedia-skill/references/dev-templates.md",
    ]

    print("loading documents from {len(sources)} sources")

    docs = [doc for source in sources for doc in WebBaseLoader(source).load()]
    print("splitting documents into chunks")

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    
    vectorstore.add_documents(documents=splits)
    print("Indexing completed")

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    _, retriever = init_vectore_store()
    question = state["messages"][-1].content

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "messages": state["messages"]}
